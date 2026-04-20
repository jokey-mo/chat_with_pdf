from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class NotebookCreate(BaseModel):
    name: str
    topic_query: str
    sources: list[str] = Field(default_factory=lambda: ["openalex"])
    schedule_cron: str | None = None
    openrouter_model: str | None = None


class NotebookOut(BaseModel):
    id: int
    name: str
    topic_query: str
    sources: list[str]
    schedule_cron: str | None
    openrouter_model: str | None
    last_run_at: datetime | None
    created_at: datetime

    class Config:
        from_attributes = True


class PaperOut(BaseModel):
    id: int
    source: str
    external_id: str
    doi: str | None
    title: str
    authors: list[str]
    year: int | None
    pdf_url: str | None
    status: str
    ingested_at: datetime

    class Config:
        from_attributes = True


class IngestionRunOut(BaseModel):
    id: int
    notebook_id: int
    started_at: datetime
    finished_at: datetime | None
    n_found: int
    n_new: int
    n_embedded: int
    n_failed: int
    error_summary: str | None

    class Config:
        from_attributes = True


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    notebook_id: int
    messages: list[ChatMessage]
    model: str | None = None
