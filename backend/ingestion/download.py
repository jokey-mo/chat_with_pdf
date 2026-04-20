from __future__ import annotations

import hashlib
from pathlib import Path

import httpx

from backend.core.config import settings
from backend.core.logging import get_logger

log = get_logger(__name__)


async def download_pdf(url: str) -> tuple[Path, str] | None:
    try:
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            content = resp.content
    except httpx.HTTPError as e:
        log.warning("download failed url=%s error=%s", url, e)
        return None

    if not content.startswith(b"%PDF"):
        log.warning("not a pdf url=%s first_bytes=%r", url, content[:8])
        return None

    sha1 = hashlib.sha1(content).hexdigest()
    path = settings.data_dir / "pdfs" / f"{sha1}.pdf"
    if not path.exists():
        path.write_bytes(content)
    return path, sha1
