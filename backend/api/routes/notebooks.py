from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.api.schemas import NotebookCreate, NotebookOut
from backend.db.models import Notebook
from backend.db.session import get_db

router = APIRouter(prefix="/notebooks", tags=["notebooks"])


@router.post("", response_model=NotebookOut)
def create(req: NotebookCreate, db: Session = Depends(get_db)) -> Notebook:
    nb = Notebook(
        name=req.name,
        topic_query=req.topic_query,
        sources=req.sources,
        schedule_cron=req.schedule_cron,
        openrouter_model=req.openrouter_model,
    )
    db.add(nb)
    db.commit()
    db.refresh(nb)
    return nb


@router.get("", response_model=list[NotebookOut])
def list_all(db: Session = Depends(get_db)) -> list[Notebook]:
    return list(db.execute(select(Notebook).order_by(Notebook.id.desc())).scalars())


@router.get("/{notebook_id}", response_model=NotebookOut)
def get(notebook_id: int, db: Session = Depends(get_db)) -> Notebook:
    nb = db.get(Notebook, notebook_id)
    if nb is None:
        raise HTTPException(404, "not found")
    return nb


@router.delete("/{notebook_id}")
def delete(notebook_id: int, db: Session = Depends(get_db)) -> dict:
    nb = db.get(Notebook, notebook_id)
    if nb is None:
        raise HTTPException(404, "not found")
    db.delete(nb)
    db.commit()
    return {"deleted": notebook_id}
