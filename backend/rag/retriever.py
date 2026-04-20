from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.core.config import settings
from backend.db.models import Chunk, Paper
from backend.rag import qdrant_store
from backend.rag.embeddings import embed_query
from backend.rag.rerank import Hit, rerank


def retrieve(
    db: Session, notebook_id: int, query: str, k: int | None = None, top: int | None = None
) -> list[Hit]:
    k = k or settings.retrieve_k
    vec = embed_query(query)
    points = qdrant_store.search(vec, notebook_id=notebook_id, k=k)
    if not points:
        return []

    point_ids = [str(p.id) for p in points]
    rows = db.execute(
        select(Chunk, Paper).join(Paper, Chunk.paper_id == Paper.id).where(
            Chunk.qdrant_point_id.in_(point_ids)
        )
    ).all()
    by_pid = {c.qdrant_point_id: (c, p) for c, p in rows}

    hits: list[Hit] = []
    for pt in points:
        pair = by_pid.get(str(pt.id))
        if not pair:
            continue
        chunk, paper = pair
        payload = dict(pt.payload or {})
        payload.update(
            {
                "paper_id": paper.id,
                "paper_title": paper.title,
                "doi": paper.doi,
                "authors": paper.authors,
                "year": paper.year,
                "source": paper.source,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "section": chunk.section,
            }
        )
        hits.append(
            Hit(
                point_id=str(pt.id), score=float(pt.score), payload=payload, text=chunk.text
            )
        )

    return rerank(query, hits, top=top)
