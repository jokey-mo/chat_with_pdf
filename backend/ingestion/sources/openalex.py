from __future__ import annotations

import asyncio
from datetime import datetime

import httpx

from backend.core.config import settings
from backend.core.logging import get_logger
from backend.ingestion.sources.base import PaperRef

log = get_logger(__name__)

OPENALEX_BASE = "https://api.openalex.org/works"


class OpenAlexSource:
    name = "openalex"

    async def search(
        self, topic: str, since: datetime | None = None, limit: int = 50
    ) -> list[PaperRef]:
        params: dict[str, str] = {
            "search": topic,
            "per-page": str(min(limit, 200)),
            "filter": "is_oa:true,has_fulltext:true,type:article",
        }
        if since:
            params["filter"] += f",from_publication_date:{since.date().isoformat()}"
        if settings.openalex_api_key:
            params["api_key"] = settings.openalex_api_key
        if settings.openalex_email:
            params["mailto"] = settings.openalex_email

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(OPENALEX_BASE, params=params)
            resp.raise_for_status()
            data = resp.json()

        refs: list[PaperRef] = []
        for w in data.get("results", []):
            ext_id = (w.get("id") or "").rsplit("/", 1)[-1]
            if not ext_id:
                continue
            refs.append(
                PaperRef(
                    source=self.name,
                    external_id=ext_id,
                    doi=_norm_doi(w.get("doi")),
                    title=w.get("title") or "(untitled)",
                    authors=[
                        a["author"]["display_name"]
                        for a in w.get("authorships", [])
                        if a.get("author")
                    ],
                    abstract=_inverted_to_text(w.get("abstract_inverted_index")),
                    year=w.get("publication_year"),
                    pdf_url=_pick_oa_pdf(w),
                    raw=w,
                )
            )
        log.info("openalex: topic=%r returned %d results", topic, len(refs))
        return refs[:limit]


def _norm_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    return doi.replace("https://doi.org/", "").lower()


def _pick_oa_pdf(work: dict) -> str | None:
    best = (work.get("best_oa_location") or {}).get("pdf_url")
    if best:
        return best
    primary = (work.get("primary_location") or {}).get("pdf_url")
    if primary:
        return primary
    for loc in work.get("locations", []) or []:
        if loc.get("pdf_url"):
            return loc["pdf_url"]
    return None


def _inverted_to_text(inverted: dict | None) -> str | None:
    if not inverted:
        return None
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted.items():
        for idx in idxs:
            positions.append((idx, word))
    positions.sort()
    return " ".join(w for _, w in positions) or None


if __name__ == "__main__":
    async def _demo():
        src = OpenAlexSource()
        refs = await src.search("hydroponics lettuce yield", limit=5)
        for r in refs:
            print(r.title, r.pdf_url)

    asyncio.run(_demo())
