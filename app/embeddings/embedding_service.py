import hashlib

from loguru import logger
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings

settings = get_settings()


class EmbeddingService:
    """
    Wraps SentenceTransformers (all-mpnet-base-v2).
    Singleton pattern — model loads once at startup.
    """

    _instance: "EmbeddingService | None" = None
    _model: SentenceTransformer | None = None

    def __new__(cls) -> "EmbeddingService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self) -> None:
        if self._model is not None:
            return
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        self._model = SentenceTransformer(
            settings.embedding_model,
            device=settings.embedding_device,
        )
        logger.info(f"Model loaded — dim={settings.embedding_dim}, device={settings.embedding_device}")

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self.load()
        return self._model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=4))
    def embed(self, text: str) -> list[float]:
        """Embed a single text string. Returns a list[float] of length 768."""
        vec = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec.tolist()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=4))
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts efficiently.
        Returns list of embedding vectors.
        """
        logger.info(f"Embedding batch of {len(texts)} texts")
        vecs = self.model.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        return vecs.tolist()

    @staticmethod
    def cache_key(text: str) -> str:
        """Deterministic cache key for a given input text."""
        h = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f"embed:{settings.embedding_model.split('/')[-1]}:{h}"

    @staticmethod
    def text_for_paper(title: str, abstract: str) -> str:
        """
        Combine title + abstract into a single string for indexing.
        Title is prepended twice to boost its weight.
        """
        return f"{title}. {title}. {abstract}"


# Module-level singleton
embedding_service = EmbeddingService()
