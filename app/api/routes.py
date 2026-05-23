from fastapi import APIRouter, HTTPException, status
from loguru import logger

from app.db.qdrant_client import qdrant_db
from app.db.mongo_client import mongo_db
from app.db.redis_client import redis_cache
from app.embeddings.embedding_service import embedding_service
from app.models.schemas import (
    BulkIndexRequest,
    BulkIndexResponse,
    EmbedRequest,
    EmbedResponse,
    HealthResponse,
    IndexRequest,
    IndexResponse,
    SearchRequest,
    SearchResponse,
)
from app.services.search_service import search_service
from app.config import get_settings

settings = get_settings()
router = APIRouter()


# ── /health ───────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check the status of all services."""
    redis_info = await redis_cache.info()
    try:
        mongo_count = await mongo_db.count()
        mongo_status = "ok"
    except Exception:
        mongo_status = "unreachable"
        mongo_count = 0

    try:
        qdrant_count = qdrant_db.count()
        qdrant_status = "ok"
    except Exception:
        qdrant_status = "unreachable"
        qdrant_count = 0

    return HealthResponse(
        status="ok",
        app=settings.app_name,
        services={
            "qdrant": f"{qdrant_status} ({qdrant_count} vectors)",
            "mongodb": f"{mongo_status} ({mongo_count} papers)",
            "redis": redis_info.get("status", "unknown"),
            "embedding_model": settings.embedding_model.split("/")[-1],
        },
    )


# ── /embed ────────────────────────────────────────────

@router.post("/embed", response_model=EmbedResponse, tags=["Embeddings"])
async def embed_text(req: EmbedRequest):
    """
    Embed a single text string.
    Results are cached in Redis by content hash.
    """
    cache_key = embedding_service.cache_key(req.text)
    cached = await redis_cache.get(cache_key)

    if cached:
        logger.info("Embed cache hit")
        vec = cached
    else:
        vec = embedding_service.embed(req.text)
        await redis_cache.set(cache_key, vec)

    return EmbedResponse(
        text=req.text,
        embedding=vec,
        dim=len(vec),
        model=settings.embedding_model,
    )


# ── /search ───────────────────────────────────────────

@router.post("/search", response_model=SearchResponse, tags=["Search"])
async def search(req: SearchRequest):
    """
    Semantic search over indexed ArXiv papers.
    Query is embedded → searched in Milvus → metadata fetched from MongoDB.
    """
    try:
        response = await search_service.search(query=req.query, top_k=req.top_k)
        return response
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# ── /index ────────────────────────────────────────────

@router.post("/index", response_model=IndexResponse, tags=["Indexing"])
async def index_paper(req: IndexRequest):
    """Index a single paper — embeds title+abstract and stores in Milvus + MongoDB."""
    try:
        return await search_service.index_paper(req)
    except Exception as e:
        logger.error(f"Index failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/index/bulk", response_model=BulkIndexResponse, tags=["Indexing"])
async def bulk_index(req: BulkIndexRequest):
    """
    Bulk index up to 1000 papers in a single request.
    Uses batched embedding for efficiency.
    """
    try:
        return await search_service.bulk_index(req)
    except Exception as e:
        logger.error(f"Bulk index failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
