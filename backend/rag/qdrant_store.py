from __future__ import annotations

import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from backend.core.config import settings
from backend.core.logging import get_logger

log = get_logger(__name__)

_client: QdrantClient | None = None


def client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.qdrant_url, timeout=30)
    return _client


def ensure_collection() -> None:
    c = client()
    collections = {col.name for col in c.get_collections().collections}
    if settings.qdrant_collection in collections:
        return
    c.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=qm.VectorParams(size=settings.embed_dim, distance=qm.Distance.COSINE),
    )
    c.create_payload_index(settings.qdrant_collection, field_name="notebook_id", field_schema="integer")
    c.create_payload_index(settings.qdrant_collection, field_name="paper_id", field_schema="integer")
    c.create_payload_index(settings.qdrant_collection, field_name="kind", field_schema="keyword")
    log.info("qdrant: created collection %s", settings.qdrant_collection)


@dataclass
class UpsertPoint:
    point_id: str
    vector: list[float]
    payload: dict


def upsert(points: list[UpsertPoint]) -> None:
    if not points:
        return
    ensure_collection()
    client().upsert(
        collection_name=settings.qdrant_collection,
        points=[
            qm.PointStruct(id=p.point_id, vector=p.vector, payload=p.payload)
            for p in points
        ],
    )


def search(
    vector: list[float], notebook_id: int, k: int = 40
) -> list[qm.ScoredPoint]:
    ensure_collection()
    return client().search(
        collection_name=settings.qdrant_collection,
        query_vector=vector,
        query_filter=qm.Filter(
            must=[qm.FieldCondition(key="notebook_id", match=qm.MatchValue(value=notebook_id))]
        ),
        limit=k,
        with_payload=True,
    )


def delete_paper(paper_id: int) -> None:
    ensure_collection()
    client().delete(
        collection_name=settings.qdrant_collection,
        points_selector=qm.FilterSelector(
            filter=qm.Filter(
                must=[qm.FieldCondition(key="paper_id", match=qm.MatchValue(value=paper_id))]
            )
        ),
    )


def new_point_id() -> str:
    return str(uuid.uuid4())
