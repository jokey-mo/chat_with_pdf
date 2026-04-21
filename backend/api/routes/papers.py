from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from backend.api.schemas import PaperOut
from backend.db.models import Figure, Notebook, Paper
from backend.db.session import get_db

router = APIRouter(tags=["papers"])


@router.get("/notebooks/{notebook_id}/papers", response_model=list[PaperOut])
def list_papers(
    notebook_id: int,
    db: Session = Depends(get_db),
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> list[Paper]:
    if db.get(Notebook, notebook_id) is None:
        raise HTTPException(404, "notebook not found")
    return list(
        db.execute(
            select(Paper)
            .where(Paper.notebook_id == notebook_id)
            .order_by(desc(Paper.ingested_at))
            .limit(limit)
            .offset(offset)
        ).scalars()
    )


@router.get("/papers/{paper_id}", response_model=PaperOut)
def get_paper(paper_id: int, db: Session = Depends(get_db)) -> Paper:
    p = db.get(Paper, paper_id)
    if p is None:
        raise HTTPException(404, "not found")
    return p


@router.get("/papers/{paper_id}/pdf")
def get_pdf(paper_id: int, db: Session = Depends(get_db)) -> FileResponse:
    p = db.get(Paper, paper_id)
    if p is None or not p.pdf_path:
        raise HTTPException(404, "not found")
    path = Path(p.pdf_path)
    if not path.exists():
        raise HTTPException(410, "pdf missing on disk")
    return FileResponse(path, media_type="application/pdf", filename=f"{paper_id}.pdf")


@router.get("/figures/{figure_id}")
def get_figure(figure_id: int, db: Session = Depends(get_db)) -> FileResponse:
    f = db.get(Figure, figure_id)
    if f is None:
        raise HTTPException(404, "not found")
    path = Path(f.image_path)
    if not path.exists():
        raise HTTPException(410, "image missing")
    return FileResponse(path, media_type="image/png")
