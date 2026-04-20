from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from backend.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class ParsedFigure:
    page: int | None
    bbox: dict | None
    image_path: str
    caption: str | None


@dataclass
class ParsedSection:
    heading: str | None
    text: str
    page_start: int | None = None
    page_end: int | None = None


@dataclass
class ParsedDoc:
    markdown: str
    sections: list[ParsedSection] = field(default_factory=list)
    figures: list[ParsedFigure] = field(default_factory=list)
    source: str = "docling"


def parse(pdf_path: Path, figure_out_dir: Path | None = None) -> ParsedDoc:
    try:
        return _parse_docling(pdf_path, figure_out_dir)
    except Exception as e:
        log.warning("docling failed for %s: %s; falling back to pymupdf4llm", pdf_path, e)
        return _parse_pymupdf(pdf_path)


def _parse_docling(pdf_path: Path, figure_out_dir: Path | None) -> ParsedDoc:
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    doc = result.document
    markdown = doc.export_to_markdown()
    sections = _sections_from_markdown(markdown)

    figures: list[ParsedFigure] = []
    if figure_out_dir is not None:
        figure_out_dir.mkdir(parents=True, exist_ok=True)
        for idx, item in enumerate(getattr(doc, "pictures", []) or []):
            try:
                img = item.get_image(doc)
            except Exception:
                img = None
            if img is None:
                continue
            out = figure_out_dir / f"fig_{idx:03d}.png"
            img.save(out)
            caption = None
            cap_fn = getattr(item, "caption_text", None)
            if callable(cap_fn):
                try:
                    caption = cap_fn(doc)
                except Exception:
                    caption = None
            bbox = _bbox_of(item)
            page = _page_of(item)
            figures.append(ParsedFigure(page=page, bbox=bbox, image_path=str(out), caption=caption))

    return ParsedDoc(markdown=markdown, sections=sections, figures=figures, source="docling")


def _parse_pymupdf(pdf_path: Path) -> ParsedDoc:
    import pymupdf4llm

    md = pymupdf4llm.to_markdown(str(pdf_path))
    sections = _sections_from_markdown(md)
    return ParsedDoc(markdown=md, sections=sections, figures=[], source="pymupdf4llm")


def _sections_from_markdown(md: str) -> list[ParsedSection]:
    lines = md.splitlines()
    sections: list[ParsedSection] = []
    cur_heading: str | None = None
    cur_buf: list[str] = []

    def flush():
        if cur_buf and (cur_heading or "".join(cur_buf).strip()):
            sections.append(ParsedSection(heading=cur_heading, text="\n".join(cur_buf).strip()))

    for ln in lines:
        if ln.startswith("#"):
            flush()
            cur_heading = ln.lstrip("#").strip()
            cur_buf = []
        else:
            cur_buf.append(ln)
    flush()
    if not sections:
        sections = [ParsedSection(heading=None, text=md.strip())]
    return sections


def _bbox_of(item) -> dict | None:
    prov = getattr(item, "prov", None)
    if not prov:
        return None
    try:
        p = prov[0]
        b = p.bbox
        return {"l": b.l, "t": b.t, "r": b.r, "b": b.b}
    except Exception:
        return None


def _page_of(item) -> int | None:
    prov = getattr(item, "prov", None)
    if not prov:
        return None
    try:
        return prov[0].page_no
    except Exception:
        return None
