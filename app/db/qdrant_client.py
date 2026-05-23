from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
)
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings

settings = get_settings()


class QdrantDB:
    """
    Manages the Qdrant connection and collection lifecycle.
    Handles: create collection, insert vectors, search, delete.

    Replaces Milvus with a single lightweight container.
    Same public interface: connect(), insert(), search(), count().
    """

    _client: QdrantClient | None = None

    def connect(self) -> None:
        logger.info(f"Connecting to Qdrant @ {settings.qdrant_host}:{settings.qdrant_port}")
        self._client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            timeout=30,
        )
        self._ensure_collection()
        logger.info("Qdrant connected ✓")

    def disconnect(self) -> None:
        if self._client:
            self._client.close()
            logger.info("Qdrant disconnected")

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self.connect()
        return self._client

    def _ensure_collection(self) -> None:
        name = settings.qdrant_collection
        existing = [c.name for c in self._client.get_collections().collections]

        if name not in existing:
            logger.info(f"Creating Qdrant collection: {name}")
            self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=settings.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Collection '{name}' created (dim={settings.embedding_dim}, metric=COSINE)")
        else:
            logger.info(f"Collection '{name}' already exists")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=4))
    def insert(self, paper_ids: list[str], embeddings: list[list[float]]) -> int:
        """
        Insert paper_id + embedding pairs into Qdrant.
        paper_id is stored as payload so we can retrieve it after search.
        Returns count inserted.
        """
        points = [
            PointStruct(
                id=abs(hash(paper_id)) % (2**63),  # Qdrant needs uint64 IDs
                vector=embedding,
                payload={"paper_id": paper_id},
            )
            for paper_id, embedding in zip(paper_ids, embeddings)
        ]

        self.client.upsert(
            collection_name=settings.qdrant_collection,
            points=points,
            wait=True,
        )
        logger.info(f"Inserted {len(points)} vectors into Qdrant")
        return len(points)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=4))
    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filter_: Filter | None = None,
    ) -> list[dict]:
        """
        Search for top_k nearest neighbors by cosine similarity.
        Returns list of {paper_id, score} dicts ordered by score desc.
        """
        results = self.client.search(
            collection_name=settings.qdrant_collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=filter_,
            with_payload=True,
        )

        hits = [
            {
                "paper_id": hit.payload.get("paper_id"),
                "score": round(hit.score, 4),
            }
            for hit in results
        ]
        return hits

    def count(self) -> int:
        """Return total number of vectors in the collection."""
        info = self.client.get_collection(settings.qdrant_collection)
        return info.vectors_count or 0

    def drop_collection(self) -> None:
        name = settings.qdrant_collection
        self.client.delete_collection(name)
        logger.warning(f"Dropped Qdrant collection: {name}")


# Module-level singleton
qdrant_db = QdrantDB()
