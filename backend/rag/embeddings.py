from __future__ import annotations

from openai import AsyncOpenAI, OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.core.config import settings
from backend.core.openrouter import async_client, sync_client


def _embed_sync_client() -> OpenAI:
    if settings.embed_base_url and settings.embed_api_key:
        return OpenAI(api_key=settings.embed_api_key, base_url=settings.embed_base_url)
    return sync_client()


def _embed_async_client() -> AsyncOpenAI:
    if settings.embed_base_url and settings.embed_api_key:
        return AsyncOpenAI(api_key=settings.embed_api_key, base_url=settings.embed_base_url)
    return async_client()


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=10))
def embed_batch(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    client = _embed_sync_client()
    resp = client.embeddings.create(
        model=settings.openrouter_embed_model, input=texts
    )
    return [d.embedding for d in resp.data]


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=10))
async def aembed_batch(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    client = _embed_async_client()
    resp = await client.embeddings.create(
        model=settings.openrouter_embed_model, input=texts
    )
    return [d.embedding for d in resp.data]


def embed_query(text: str) -> list[float]:
    return embed_batch([text])[0]
