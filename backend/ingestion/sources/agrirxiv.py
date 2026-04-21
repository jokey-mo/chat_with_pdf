from __future__ import annotations

from datetime import datetime

import httpx

from backend.core.logging import get_logger
from backend.ingestion.sources.base import PaperRef

log = get_logger(__name__)

OSF_BASE = "https://api.osf.io/v2/preprints/"
PROVIDER = "agrixiv"


class AgriRxivSource:
    name = "agrirxiv"

    async def search(
        self, topic: str, since: datetime | None = None, limit: int = 50
    ) -> list[PaperRef]:
        params = {
            "filter[provider]": PROVIDER,
            "filter[q]": topic,
            "page[size]": str(min(limit, 100)),
        }
        if since:
            params["filter[date_created][gte]"] = since.date().isoformat()

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(OSF_BASE, params=params)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as e:
            log.warning("agrirxiv: search failed: %s", e)
            return []

        refs: list[PaperRef] = []
        for item in data.get("data", []):
            attrs = item.get("attributes", {}) or {}
            links = item.get("links", {}) or {}
            pdf_url = links.get("preprint_doi_url") or None
            doi = (attrs.get("doi") or "").lower() or None
            refs.append(
                PaperRef(
                    source=self.name,
                    external_id=item.get("id") or "",
                    doi=doi,
                    title=attrs.get("title") or "(untitled)",
                    authors=[],
                    abstract=attrs.get("description"),
                    year=_year_from(attrs.get("date_published") or attrs.get("date_created")),
                    pdf_url=pdf_url,
                    raw=item,
                )
            )
        log.info("agrirxiv: topic=%r returned %d results", topic, len(refs))
        return refs[:limit]


def _year_from(date_str: str | None) -> int | None:
    if not date_str:
        return None
    try:
        return int(date_str[:4])
    except ValueError:
        return None
