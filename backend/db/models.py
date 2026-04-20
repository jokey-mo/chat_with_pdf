from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    JSON,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class PaperStatus(str, enum.Enum):
    queued = "queued"
    parsing = "parsing"
    embedded = "embedded"
    failed = "failed"


class Notebook(Base):
    __tablename__ = "notebooks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    topic_query: Mapped[str] = mapped_column(Text)
    sources: Mapped[list] = mapped_column(JSON, default=list)
    schedule_cron: Mapped[str | None] = mapped_column(String(64), nullable=True)
    openrouter_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    papers: Mapped[list[Paper]] = relationship(back_populates="notebook", cascade="all, delete-orphan")


class Paper(Base):
    __tablename__ = "papers"
    __table_args__ = (UniqueConstraint("source", "external_id", name="uq_source_extid"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    notebook_id: Mapped[int] = mapped_column(ForeignKey("notebooks.id", ondelete="CASCADE"))
    source: Mapped[str] = mapped_column(String(32))
    external_id: Mapped[str] = mapped_column(String(255))
    doi: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    title: Mapped[str] = mapped_column(Text)
    authors: Mapped[list] = mapped_column(JSON, default=list)
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pdf_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    pdf_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    parsed_json_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    checksum: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[PaperStatus] = mapped_column(
        Enum(PaperStatus, name="paper_status"), default=PaperStatus.queued
    )
    error_msg: Mapped[str | None] = mapped_column(Text, nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    notebook: Mapped[Notebook] = relationship(back_populates="papers")
    chunks: Mapped[list[Chunk]] = relationship(back_populates="paper", cascade="all, delete-orphan")
    figures: Mapped[list[Figure]] = relationship(back_populates="paper", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    paper_id: Mapped[int] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"))
    section: Mapped[str | None] = mapped_column(String(255), nullable=True)
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    qdrant_point_id: Mapped[str] = mapped_column(String(64))

    paper: Mapped[Paper] = relationship(back_populates="chunks")


class Figure(Base):
    __tablename__ = "figures"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    paper_id: Mapped[int] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"))
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bbox: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    image_path: Mapped[str] = mapped_column(Text)
    caption_original: Mapped[str | None] = mapped_column(Text, nullable=True)
    caption_vlm: Mapped[str | None] = mapped_column(Text, nullable=True)
    qdrant_point_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    paper: Mapped[Paper] = relationship(back_populates="figures")


class IngestionRun(Base):
    __tablename__ = "ingestion_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    notebook_id: Mapped[int] = mapped_column(ForeignKey("notebooks.id", ondelete="CASCADE"))
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    n_found: Mapped[int] = mapped_column(Integer, default=0)
    n_new: Mapped[int] = mapped_column(Integer, default=0)
    n_embedded: Mapped[int] = mapped_column(Integer, default=0)
    n_failed: Mapped[int] = mapped_column(Integer, default=0)
    error_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
