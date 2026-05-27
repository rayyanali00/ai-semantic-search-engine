import hashlib
import time

from loguru import logger
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings

settings = get_settings()


class EmbeddingService:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self):
        if self._model is not None:
            return
        t0 = time.monotonic()
        logger.info(f"Loading model: {settings.embedding_model}")
        self._model = SentenceTransformer(
            settings.embedding_model,
            device="cpu",
        )
        logger.info(f"Model loaded ✓ ({round((time.monotonic() - t0) * 1000, 2)}ms)")

    @property
    def model(self):
        if self._model is None:
            self.load()
        return self._model

    def embed(self, text: str) -> list[float]:
        t0 = time.monotonic()
        vec = self.model.encode(text, normalize_embeddings=True).tolist()
        logger.info(f"Embedded 1 text ({round((time.monotonic() - t0) * 1000, 2)}ms)")
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        t0 = time.monotonic()
        logger.info(f"Embedding {len(texts)} texts")
        vecs = self.model.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).tolist()
        logger.info(f"Embedded {len(texts)} texts ({round((time.monotonic() - t0) * 1000, 2)}ms)")
        return vecs

    @staticmethod
    def cache_key(text: str) -> str:
        h = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f"embed:mpnet:{h}"

    @staticmethod
    def text_for_paper(title: str, abstract: str) -> str:
        return f"{title}. {title}. {abstract}"


embedding_service = EmbeddingService()


# import hashlib
# import time

# import httpx
# from loguru import logger
# from tenacity import retry, stop_after_attempt, wait_exponential

# from app.config import get_settings

# settings = get_settings()

# HF_API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{settings.embedding_model}"


# class EmbeddingService:
#     _instance = None
#     _client = None

#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super().__new__(cls)
#         return cls._instance

#     def load(self):
#         if self._client is not None:
#             return
#         logger.info(f"Connecting to HuggingFace API — {settings.embedding_model}")
#         self._client = httpx.Client(
#             headers={"Authorization": f"Bearer {settings.hf_api_token}"},
#             timeout=60.0,
#         )
#         logger.info("HuggingFace client ready ✓")

#     @property
#     def client(self):
#         if self._client is None:
#             self.load()
#         return self._client

#     @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=15))
#     def _raw_embed(self, texts: list[str]) -> list[list[float]]:
#         response = self.client.post(
#             HF_API_URL,
#             json={
#                 "inputs": texts,
#                 "options": {"wait_for_model": True, "use_cache": True},
#             },
#         )
#         if response.status_code == 503:
#             logger.warning("HF model loading — retrying...")
#             raise Exception("Model loading (503)")
#         response.raise_for_status()
#         result = response.json()
#         if isinstance(result[0], float):
#             return [result]
#         return result

#     def embed(self, text: str) -> list[float]:
#         return self._raw_embed([text])[0]

#     def embed_batch(self, texts: list[str]) -> list[list[float]]:
#         logger.info(f"Embedding {len(texts)} texts via HuggingFace API")
#         all_embeddings = []
#         for i in range(0, len(texts), 64):
#             chunk = texts[i : i + 64]
#             all_embeddings.extend(self._raw_embed(chunk))
#             if i + 64 < len(texts):
#                 time.sleep(0.2)
#         return all_embeddings

#     @staticmethod
#     def cache_key(text: str) -> str:
#         h = hashlib.sha256(text.encode()).hexdigest()[:16]
#         return f"embed:mpnet:{h}"

#     @staticmethod
#     def text_for_paper(title: str, abstract: str) -> str:
#         return f"{title}. {title}. {abstract}"


# embedding_service = EmbeddingService()
