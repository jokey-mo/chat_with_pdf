from __future__ import annotations

from redis import Redis
from rq import Queue

from backend.core.config import settings

_redis: Redis | None = None
_queue: Queue | None = None


def queue() -> Queue:
    global _redis, _queue
    if _queue is None:
        _redis = Redis.from_url(settings.redis_url)
        _queue = Queue("ingest", connection=_redis, default_timeout=60 * 60)
    return _queue


def enqueue_ingestion(notebook_id: int):
    return queue().enqueue(
        "backend.ingestion.pipeline.run_ingestion", notebook_id, job_timeout=60 * 60
    )
