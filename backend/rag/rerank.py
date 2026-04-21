from __future__ import annotations

import threading
from dataclasses import dataclass

from backend.core.config import settings
from backend.core.logging import get_logger

log = get_logger(__name__)

_reranker = None
_reranker_lock = threading.Lock()


@dataclass
class Hit:
    point_id: str
    score: float
    payload: dict
    text: str


def _get_reranker():
    global _reranker
    if _reranker is not None:
        return _reranker
    with _reranker_lock:
        if _reranker is not None:
            return _reranker
        try:
            from FlagEmbedding import FlagReranker

            _reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=False)
            return _reranker
        except Exception as e:
            log.warning("reranker unavailable: %s", e)
            return None


def rerank(query: str, hits: list[Hit], top: int | None = None) -> list[Hit]:
    if not settings.enable_reranker or not hits:
        return hits[: top or settings.rerank_top]
    rr = _get_reranker()
    if rr is None:
        return hits[: top or settings.rerank_top]
    pairs = [[query, h.text] for h in hits]
    scores = rr.compute_score(pairs, normalize=True)
    for h, s in zip(hits, scores):
        h.score = float(s)
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[: top or settings.rerank_top]
