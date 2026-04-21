from __future__ import annotations

import time

from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import select

from backend.core.config import settings
from backend.core.logging import get_logger, setup_logging
from backend.db.models import Notebook
from backend.db.session import SessionLocal, engine
from backend.scheduler.jobs import enqueue_ingestion

log = get_logger(__name__)


def _enqueue(notebook_id: int) -> None:
    log.info("scheduler: enqueueing ingestion for notebook=%s", notebook_id)
    enqueue_ingestion(notebook_id)


def _sync_jobs(scheduler: BackgroundScheduler) -> None:
    with SessionLocal() as db:
        notebooks = list(db.execute(select(Notebook)).scalars())

    existing = {j.id for j in scheduler.get_jobs()}
    desired: set[str] = set()
    for nb in notebooks:
        if not nb.schedule_cron:
            continue
        job_id = f"nb-{nb.id}"
        desired.add(job_id)
        try:
            trigger = CronTrigger.from_crontab(nb.schedule_cron)
        except ValueError:
            log.warning("invalid cron %r for notebook=%s", nb.schedule_cron, nb.id)
            continue
        scheduler.add_job(
            _enqueue,
            trigger=trigger,
            args=[nb.id],
            id=job_id,
            replace_existing=True,
            misfire_grace_time=60 * 30,
        )

    for stale in existing - desired:
        scheduler.remove_job(stale)


def main() -> None:
    setup_logging()
    scheduler = BackgroundScheduler(
        jobstores={"default": SQLAlchemyJobStore(engine=engine)}
    )
    scheduler.start()
    log.info("scheduler started; postgres=%s redis=%s", settings.postgres_dsn, settings.redis_url)

    try:
        while True:
            try:
                _sync_jobs(scheduler)
            except Exception:
                log.exception("scheduler sync failed")
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()


if __name__ == "__main__":
    main()
