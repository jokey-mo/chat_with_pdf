from __future__ import annotations

from dataclasses import dataclass

from backend.parsing.docling_parser import ParsedDoc

try:
    import tiktoken

    _enc = tiktoken.get_encoding("cl100k_base")
except Exception:  # offline / forbidden CDN — fall back to whitespace approx
    _enc = None


@dataclass
class Chunk:
    section: str | None
    text: str
    token_count: int
    page_start: int | None = None
    page_end: int | None = None


def count_tokens(text: str) -> int:
    if _enc is None:
        # crude approximation: ~1.3 tokens per word for English
        return max(1, int(len(text.split()) * 1.3))
    return len(_enc.encode(text))


def section_aware_chunks(
    parsed: ParsedDoc, max_tokens: int = 800, overlap: int = 100
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for sec in parsed.sections:
        if not sec.text.strip():
            continue
        for piece in _split_text(sec.text, max_tokens, overlap):
            chunks.append(
                Chunk(
                    section=sec.heading,
                    text=piece,
                    token_count=count_tokens(piece),
                    page_start=sec.page_start,
                    page_end=sec.page_end,
                )
            )
    for fig in parsed.figures:
        caption_parts = [p for p in [fig.caption] if p]
        if not caption_parts:
            continue
        text = " ".join(caption_parts)
        chunks.append(
            Chunk(
                section=f"Figure (page {fig.page})" if fig.page else "Figure",
                text=text,
                token_count=count_tokens(text),
                page_start=fig.page,
                page_end=fig.page,
            )
        )
    return chunks


def _split_text(text: str, max_tokens: int, overlap: int) -> list[str]:
    if _enc is not None:
        tokens = _enc.encode(text)
        if len(tokens) <= max_tokens:
            return [text]
        pieces: list[str] = []
        step = max_tokens - overlap
        for start in range(0, len(tokens), step):
            window = tokens[start : start + max_tokens]
            if not window:
                break
            pieces.append(_enc.decode(window))
            if start + max_tokens >= len(tokens):
                break
        return pieces
    # word-based fallback
    words = text.split()
    if count_tokens(text) <= max_tokens:
        return [text]
    words_per_chunk = max(1, int(max_tokens / 1.3))
    step = max(1, words_per_chunk - int(overlap / 1.3))
    pieces: list[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + words_per_chunk]
        if not window:
            break
        pieces.append(" ".join(window))
        if start + words_per_chunk >= len(words):
            break
    return pieces
