from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

VALID_SOURCES = {"openalex", "agrirxiv"}


class NotebookCreate(BaseModel):
    name: str = Field(min_length=1)
    topic_query: str = Field(min_length=1)
    sources: list[str] = Field(default_factory=lambda: ["openalex"])
    schedule_cron: str | None = None
    openrouter_model: str | None = None

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, v: list[str]) -> list[str]:
        unknown = [s for s in v if s not in VALID_SOURCES]
        if unknown:
            raise ValueError(f"Unknown source(s): {unknown}. Valid: {sorted(VALID_SOURCES)}")
        if not v:
            raise ValueError("At least one source is required")
        return v

    @field_validator("schedule_cron")
    @classmethod
    def validate_cron(cls, v: str | None) -> str | None:
        if v is None:
            return v
        parts = v.strip().split()
        if len(parts) != 5:
            raise ValueError("Cron must have 5 fields: minute hour day month weekday")
        return v


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
    role: str = Field(pattern=r"^(user|assistant|system)$")
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    notebook_id: int
    messages: list[ChatMessage] = Field(min_length=1)
    model: str | None = None
