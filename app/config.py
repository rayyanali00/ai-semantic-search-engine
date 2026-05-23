from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    app_name: str = "semantic-search-platform"
    app_env: str = "development"
    app_port: int = 8000
    debug: bool = True

    # Qdrant (replaces Milvus — single container, zero extra infra)
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "arxiv_papers"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    cache_ttl: int = 3600

    # MongoDB
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "semantic_search"
    mongo_collection: str = "papers"

    # Embedding
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_batch_size: int = 64
    embedding_dim: int = 768
    embedding_device: str = "cpu"

    # Search
    default_top_k: int = 10
    max_top_k: int = 50

    # Celery
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
