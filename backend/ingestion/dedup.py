from __future__ import annotations

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from backend.db.models import Paper
from backend.ingestion.sources.base import PaperRef


def filter_new(db: Session, refs: list[PaperRef]) -> list[PaperRef]:
    if not refs:
        return []
    dois = [r.doi for r in refs if r.doi]
    ext_keys = [(r.source, r.external_id) for r in refs]

    seen_dois: set[str] = set()
    seen_ext: set[tuple[str, str]] = set()

    if dois:
        for (doi,) in db.execute(select(Paper.doi).where(Paper.doi.in_(dois))):
            if doi:
                seen_dois.add(doi)

    if ext_keys:
        conds = [((Paper.source == s) & (Paper.external_id == e)) for s, e in ext_keys]
        for s, e in db.execute(
            select(Paper.source, Paper.external_id).where(or_(*conds))
        ):
            seen_ext.add((s, e))

    out: list[PaperRef] = []
    local_ext: set[tuple[str, str]] = set()
    local_doi: set[str] = set()
    for r in refs:
        if r.doi and r.doi in seen_dois:
            continue
        if (r.source, r.external_id) in seen_ext:
            continue
        if r.doi and r.doi in local_doi:
            continue
        if (r.source, r.external_id) in local_ext:
            continue
        out.append(r)
        if r.doi:
            local_doi.add(r.doi)
        local_ext.add((r.source, r.external_id))
    return out
