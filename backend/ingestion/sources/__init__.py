from __future__ import annotations

from typing import Iterable

from backend.ingestion.sources.base import PaperRef, Source
from backend.ingestion.sources.openalex import OpenAlexSource
from backend.ingestion.sources.agrirxiv import AgriRxivSource


def get_source(name: str) -> Source:
    name = name.lower()
    if name == "openalex":
        return OpenAlexSource()
    if name == "agrirxiv":
        return AgriRxivSource()
    raise ValueError(f"Unknown source: {name}")


def get_sources(names: Iterable[str]) -> list[Source]:
    return [get_source(n) for n in names]


__all__ = ["PaperRef", "Source", "get_source", "get_sources"]
