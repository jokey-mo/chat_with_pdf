from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from backend.core.config import settings
from backend.core.logging import get_logger
from backend.db.models import (
    Chunk as ChunkRow,
    Figure as FigureRow,
    IngestionRun,
    Notebook,
    Paper,
    PaperStatus,
)
from backend.db.session import session_scope
from backend.ingestion.dedup import filter_new
from backend.ingestion.download import download_pdf
from backend.ingestion.sources import get_sources
from backend.ingestion.sources.base import PaperRef
from backend.ingestion.sources.unpaywall import resolve_pdf
from backend.parsing import docling_parser
from backend.parsing.chunker import section_aware_chunks
from backend.parsing.figure_captioner import caption as caption_figure
from backend.rag import qdrant_store
from backend.rag.embeddings import embed_batch

log = get_logger(__name__)


def run_ingestion(notebook_id: int, enable_figures: bool = True) -> dict:
    return asyncio.run(_run_ingestion_async(notebook_id, enable_figures))


async def _run_ingestion_async(notebook_id: int, enable_figures: bool) -> dict:
    with session_scope() as db:
        nb = db.get(Notebook, notebook_id)
        if nb is None:
            raise ValueError(f"Notebook {notebook_id} not found")
        topic = nb.topic_query
        source_names = list(nb.sources or ["openalex"])
        since = nb.last_run_at
        run = IngestionRun(notebook_id=notebook_id)
        db.add(run)
        db.flush()
        run_id = run.id

    stats = {"found": 0, "new": 0, "embedded": 0, "failed": 0, "errors": []}
    try:
        sources = get_sources(source_names)
        all_refs: list[PaperRef] = []
        for src in sources:
            try:
                refs = await src.search(topic=topic, since=since, limit=50)
                all_refs.extend(refs)
            except Exception as e:
                log.exception("source %s failed", src.name)
                stats["errors"].append(f"{src.name}: {e}")

        stats["found"] = len(all_refs)

        with session_scope() as db:
            new_refs = filter_new(db, all_refs)
        stats["new"] = len(new_refs)
        log.info(
            "notebook=%s found=%d new=%d", notebook_id, stats["found"], stats["new"]
        )

        for ref in new_refs:
            try:
                await _ingest_one(notebook_id, ref, enable_figures)
                stats["embedded"] += 1
            except Exception as e:
                log.exception("failed to ingest %s/%s", ref.source, ref.external_id)
                stats["failed"] += 1
                stats["errors"].append(f"{ref.source}/{ref.external_id}: {e}")

    finally:
        with session_scope() as db:
            run = db.get(IngestionRun, run_id)
            if run:
                run.finished_at = datetime.now(timezone.utc)
                run.n_found = stats["found"]
                run.n_new = stats["new"]
                run.n_embedded = stats["embedded"]
                run.n_failed = stats["failed"]
                run.error_summary = "; ".join(stats["errors"])[:4000] or None
            nb = db.get(Notebook, notebook_id)
            if nb:
                nb.last_run_at = datetime.now(timezone.utc)

    return stats


async def _ingest_one(notebook_id: int, ref: PaperRef, enable_figures: bool) -> None:
    pdf_url = ref.pdf_url
    if not pdf_url and ref.doi:
        pdf_url = await resolve_pdf(ref.doi)
    if not pdf_url:
        raise RuntimeError("no PDF URL available")

    dl = await download_pdf(pdf_url)
    if dl is None:
        raise RuntimeError("download failed")
    pdf_path, sha1 = dl

    with session_scope() as db:
        paper = Paper(
            notebook_id=notebook_id,
            source=ref.source,
            external_id=ref.external_id,
            doi=ref.doi,
            title=ref.title,
            authors=ref.authors,
            abstract=ref.abstract,
            year=ref.year,
            pdf_url=pdf_url,
            pdf_path=str(pdf_path),
            checksum=sha1,
            status=PaperStatus.parsing,
        )
        db.add(paper)
        db.flush()
        paper_id = paper.id

    fig_dir = settings.data_dir / "figures" / str(paper_id)
    parsed = docling_parser.parse(pdf_path, figure_out_dir=fig_dir if enable_figures else None)

    figure_rows: list[FigureRow] = []
    if enable_figures:
        for fig in parsed.figures:
            caption_vlm = None
            try:
                caption_vlm = await caption_figure(fig.image_path, fig.caption)
            except Exception as e:
                log.warning("figure caption failed: %s", e)
            figure_rows.append(
                FigureRow(
                    paper_id=paper_id,
                    page=fig.page,
                    bbox=fig.bbox,
                    image_path=fig.image_path,
                    caption_original=fig.caption,
                    caption_vlm=caption_vlm,
                )
            )

    chunks = section_aware_chunks(parsed)
    for fig, row in zip(parsed.figures, figure_rows):
        cap = row.caption_vlm or row.caption_original
        if cap:
            from backend.parsing.chunker import Chunk as ParsedChunk, count_tokens

            chunks.append(
                ParsedChunk(
                    section=f"Figure (page {fig.page})" if fig.page else "Figure",
                    text=cap,
                    token_count=count_tokens(cap),
                    page_start=fig.page,
                    page_end=fig.page,
                )
            )

    if not chunks:
        raise RuntimeError("no text extracted")

    texts = [c.text for c in chunks]
    vectors = embed_batch(texts)

    points: list[qdrant_store.UpsertPoint] = []
    chunk_rows: list[ChunkRow] = []
    for c, vec in zip(chunks, vectors):
        pid = qdrant_store.new_point_id()
        chunk_rows.append(
            ChunkRow(
                paper_id=paper_id,
                section=c.section,
                page_start=c.page_start,
                page_end=c.page_end,
                text=c.text,
                token_count=c.token_count,
                qdrant_point_id=pid,
            )
        )
        points.append(
            qdrant_store.UpsertPoint(
                point_id=pid,
                vector=vec,
                payload={
                    "notebook_id": notebook_id,
                    "paper_id": paper_id,
                    "section": c.section,
                    "page": c.page_start,
                    "kind": "text",
                },
            )
        )

    qdrant_store.upsert(points)

    with session_scope() as db:
        for row in figure_rows:
            db.add(row)
        for row in chunk_rows:
            db.add(row)
        paper = db.get(Paper, paper_id)
        if paper:
            paper.status = PaperStatus.embedded
            paper.parsed_json_path = None

    log.info(
        "notebook=%s paper=%s embedded chunks=%d figures=%d",
        notebook_id,
        paper_id,
        len(chunk_rows),
        len(figure_rows),
    )
