from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol


@dataclass
class PaperRef:
    source: str
    external_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    doi: str | None = None
    abstract: str | None = None
    year: int | None = None
    pdf_url: str | None = None
    raw: dict = field(default_factory=dict)


class Source(Protocol):
    name: str

    async def search(
        self, topic: str, since: datetime | None = None, limit: int = 50
    ) -> list[PaperRef]: ...
