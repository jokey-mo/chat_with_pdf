from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import AsyncIterator

from sqlalchemy.orm import Session

from backend.core.config import settings
from backend.core.openrouter import async_client
from backend.rag.rerank import Hit
from backend.rag.retriever import retrieve

SYSTEM_PROMPT = (
    "You answer questions about scientific papers on agriculture, soil science, "
    "hydroponics and greenhouses using the provided excerpts. "
    "ALWAYS cite the papers you use with inline markers like [P1 p.3] or "
    "[Fig 2 P1]. Do not invent facts not present in the excerpts. If the "
    "excerpts don't contain enough information, say so directly. Prefer "
    "concise, structured answers. Respond in the user's language."
)


@dataclass
class Citation:
    marker: str
    paper_id: int
    paper_title: str
    authors: list[str]
    year: int | None
    doi: str | None
    source: str
    section: str | None
    page_start: int | None


def _build_context(hits: list[Hit]) -> tuple[str, list[Citation]]:
    paper_to_marker: dict[int, str] = {}
    citations: list[Citation] = []
    parts: list[str] = []
    for h in hits:
        paper_id = h.payload.get("paper_id")
        if paper_id not in paper_to_marker:
            marker = f"P{len(paper_to_marker) + 1}"
            paper_to_marker[paper_id] = marker
            citations.append(
                Citation(
                    marker=marker,
                    paper_id=paper_id,
                    paper_title=h.payload.get("paper_title") or "",
                    authors=h.payload.get("authors") or [],
                    year=h.payload.get("year"),
                    doi=h.payload.get("doi"),
                    source=h.payload.get("source") or "",
                    section=h.payload.get("section"),
                    page_start=h.payload.get("page_start"),
                )
            )
        marker = paper_to_marker[paper_id]
        page = h.payload.get("page_start")
        section = h.payload.get("section") or ""
        header = f"[{marker}{(' p.' + str(page)) if page else ''}] {section}".strip()
        parts.append(f"{header}\n{h.text}")
    return "\n\n---\n\n".join(parts), citations


async def stream_chat(
    db: Session,
    notebook_id: int,
    messages: list[dict],
    model: str | None = None,
) -> AsyncIterator[dict]:
    query = messages[-1]["content"] if messages else ""
    hits = retrieve(db, notebook_id, query)
    context, citations = _build_context(hits)

    yield {"type": "citations", "citations": [asdict(c) for c in citations]}

    full_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Relevant excerpts:\n\n{context}\n\n---\nQuestion: {query}",
        },
    ]

    client = async_client()
    stream = await client.chat.completions.create(
        model=model or settings.openrouter_chat_model,
        messages=full_messages,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield {"type": "delta", "text": delta}

    yield {"type": "done"}
