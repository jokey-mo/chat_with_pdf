from __future__ import annotations

import httpx

from backend.core.config import settings
from backend.core.logging import get_logger

log = get_logger(__name__)

UNPAYWALL_BASE = "https://api.unpaywall.org/v2"


async def resolve_pdf(doi: str) -> str | None:
    if not doi or not settings.unpaywall_email:
        return None
    url = f"{UNPAYWALL_BASE}/{doi}"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, params={"email": settings.unpaywall_email})
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        log.warning("unpaywall: doi=%s error=%s", doi, e)
        return None

    loc = data.get("best_oa_location") or {}
    pdf = loc.get("url_for_pdf") or loc.get("url")
    if pdf:
        log.debug("unpaywall: doi=%s -> %s", doi, pdf)
    return pdf
