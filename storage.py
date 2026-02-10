# storage.py
import os
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

class QdrantStore:
    """
    Single shared Qdrant collection; each point payload stores doc_id so we can filter per PDF.
    Local Qdrant => no API key needed.
    """

    def __init__(self, dim: int = 1536):
        self.dim = dim
        self.url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.collection = os.getenv("QDRANT_COLLECTION", "pdf_chunks")

        self.client = QdrantClient(url=self.url)
        self._ensure_collection()

    def _ensure_collection(self):
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )

    def add(self, *, doc_id: str, text: str, embedding: list[float], page: int | None = None):
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={"doc_id": doc_id, "text": text, "page": page},
        )
        self.client.upsert(collection_name=self.collection, points=[point])

    def search(self, *, doc_id: str, query_embedding: list[float], top_k: int = 3) -> list[str]:
        flt = Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        )

        # qdrant-client 1.16.x uses query_points()
        resp = self.client.query_points(
            collection_name=self.collection,
            query=query_embedding,
            query_filter=flt,
            limit=top_k,
            with_payload=True,
        )

        out: list[str] = []
        for p in (resp.points or []):
            payload = p.payload or {}
            t = payload.get("text")
            if t:
                out.append(t)
        return out

