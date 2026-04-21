from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import chat, ingest, notebooks, papers
from backend.core.logging import get_logger, setup_logging
from backend.rag import qdrant_store

setup_logging()
log = get_logger(__name__)


app = FastAPI(title="Scientific Paper RAG", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(notebooks.router)
app.include_router(ingest.router)
app.include_router(papers.router)
app.include_router(chat.router)


@app.on_event("startup")
def _startup() -> None:
    try:
        qdrant_store.ensure_collection()
    except Exception:
        log.exception("Qdrant collection setup failed at startup")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
