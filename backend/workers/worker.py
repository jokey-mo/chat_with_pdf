from __future__ import annotations

from redis import Redis
from rq import Connection, Worker

from backend.core.config import settings
from backend.core.logging import setup_logging


def main() -> None:
    setup_logging()
    redis = Redis.from_url(settings.redis_url)
    with Connection(redis):
        Worker(["ingest"]).work(with_scheduler=False)


if __name__ == "__main__":
    main()
