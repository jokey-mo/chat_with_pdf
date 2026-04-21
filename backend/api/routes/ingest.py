from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from backend.api.schemas import IngestionRunOut
from backend.core.config import settings
from backend.db.models import IngestionRun, Notebook
from backend.db.session import get_db

router = APIRouter(tags=["ingest"])


@router.post("/notebooks/{notebook_id}/ingest", response_model=dict)
def trigger(notebook_id: int, db: Session = Depends(get_db)) -> dict:
    nb = db.get(Notebook, notebook_id)
    if nb is None:
        raise HTTPException(404, "notebook not found")
    if settings.run_ingestion_inline:
        from backend.ingestion.pipeline import run_ingestion

        import threading

        threading.Thread(
            target=run_ingestion, args=(notebook_id,), kwargs={"enable_figures": False}, daemon=True
        ).start()
        return {"notebook_id": notebook_id, "status": "started", "mode": "inline"}
    from backend.scheduler.jobs import enqueue_ingestion

    job = enqueue_ingestion(notebook_id)
    return {"job_id": job.id, "notebook_id": notebook_id, "status": "queued"}


@router.get("/notebooks/{notebook_id}/runs", response_model=list[IngestionRunOut])
def list_runs(notebook_id: int, db: Session = Depends(get_db)) -> list[IngestionRun]:
    return list(
        db.execute(
            select(IngestionRun)
            .where(IngestionRun.notebook_id == notebook_id)
            .order_by(desc(IngestionRun.started_at))
            .limit(50)
        ).scalars()
    )
