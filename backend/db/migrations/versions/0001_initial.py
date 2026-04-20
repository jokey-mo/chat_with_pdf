"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-04-20
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    paper_status = sa.Enum("queued", "parsing", "embedded", "failed", name="paper_status")
    paper_status.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "notebooks",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("topic_query", sa.Text, nullable=False),
        sa.Column("sources", sa.JSON, nullable=False, server_default="[]"),
        sa.Column("schedule_cron", sa.String(64), nullable=True),
        sa.Column("openrouter_model", sa.String(128), nullable=True),
        sa.Column("last_run_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "papers",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("notebook_id", sa.Integer, sa.ForeignKey("notebooks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("source", sa.String(32), nullable=False),
        sa.Column("external_id", sa.String(255), nullable=False),
        sa.Column("doi", sa.String(255), nullable=True),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("authors", sa.JSON, nullable=False, server_default="[]"),
        sa.Column("abstract", sa.Text, nullable=True),
        sa.Column("year", sa.Integer, nullable=True),
        sa.Column("pdf_url", sa.Text, nullable=True),
        sa.Column("pdf_path", sa.Text, nullable=True),
        sa.Column("parsed_json_path", sa.Text, nullable=True),
        sa.Column("checksum", sa.String(64), nullable=True),
        sa.Column("status", paper_status, nullable=False, server_default="queued"),
        sa.Column("error_msg", sa.Text, nullable=True),
        sa.Column("ingested_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("source", "external_id", name="uq_source_extid"),
    )
    op.create_index("ix_papers_doi", "papers", ["doi"])

    op.create_table(
        "chunks",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("paper_id", sa.Integer, sa.ForeignKey("papers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("section", sa.String(255), nullable=True),
        sa.Column("page_start", sa.Integer, nullable=True),
        sa.Column("page_end", sa.Integer, nullable=True),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("token_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("qdrant_point_id", sa.String(64), nullable=False),
    )

    op.create_table(
        "figures",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("paper_id", sa.Integer, sa.ForeignKey("papers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("page", sa.Integer, nullable=True),
        sa.Column("bbox", sa.JSON, nullable=True),
        sa.Column("image_path", sa.Text, nullable=False),
        sa.Column("caption_original", sa.Text, nullable=True),
        sa.Column("caption_vlm", sa.Text, nullable=True),
        sa.Column("qdrant_point_id", sa.String(64), nullable=True),
    )

    op.create_table(
        "ingestion_runs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("notebook_id", sa.Integer, sa.ForeignKey("notebooks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("n_found", sa.Integer, nullable=False, server_default="0"),
        sa.Column("n_new", sa.Integer, nullable=False, server_default="0"),
        sa.Column("n_embedded", sa.Integer, nullable=False, server_default="0"),
        sa.Column("n_failed", sa.Integer, nullable=False, server_default="0"),
        sa.Column("error_summary", sa.Text, nullable=True),
    )


def downgrade() -> None:
    op.drop_table("ingestion_runs")
    op.drop_table("figures")
    op.drop_table("chunks")
    op.drop_index("ix_papers_doi", table_name="papers")
    op.drop_table("papers")
    op.drop_table("notebooks")
    sa.Enum(name="paper_status").drop(op.get_bind(), checkfirst=True)
