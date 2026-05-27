import time

from loguru import logger

from app.db.mongo_client import mongo_db
from app.db.qdrant_client import qdrant_db
from app.db.redis_client import redis_cache
from app.embeddings.embedding_service import embedding_service
from app.models.schemas import (
    BulkIndexRequest,
    BulkIndexResponse,
    IndexRequest,
    IndexResponse,
    SearchResponse,
    SearchResult,
)


class SearchService:
    """
    Orchestrates the full semantic search pipeline:
      embed query -> cache check -> vector search -> MongoDB metadata hydration
    """

    async def search(self, query: str, top_k: int = 10) -> SearchResponse:
        t0 = time.monotonic()
        timings: dict[str, float] = {}

        redis_get_t0 = time.monotonic()
        cache_key = embedding_service.cache_key(query)
        query_vec = await redis_cache.get(cache_key)
        timings["redis_get_ms"] = round((time.monotonic() - redis_get_t0) * 1000, 2)
        cache_hit = query_vec is not None

        if not cache_hit:
            logger.info(f"Cache miss - embedding query: {query[:60]!r}")
            embed_t0 = time.monotonic()
            query_vec = embedding_service.embed(query)
            timings["embed_ms"] = round((time.monotonic() - embed_t0) * 1000, 2)

            redis_set_t0 = time.monotonic()
            await redis_cache.set(cache_key, query_vec)
            timings["redis_set_ms"] = round((time.monotonic() - redis_set_t0) * 1000, 2)
        else:
            logger.info(f"Cache hit for query: {query[:60]!r}")
            timings["embed_ms"] = 0.0

        qdrant_t0 = time.monotonic()
        hits = qdrant_db.search(query_vec, top_k=top_k)
        timings["qdrant_search_ms"] = round((time.monotonic() - qdrant_t0) * 1000, 2)
        if not hits:
            timings["total_ms"] = round((time.monotonic() - t0) * 1000, 2)
            logger.info(f"Search profile - {timings}")
            return SearchResponse(query=query, results=[], total=0, took_ms=0)

        paper_ids = [h["paper_id"] for h in hits]
        score_map = {h["paper_id"]: h["score"] for h in hits}

        mongo_t0 = time.monotonic()
        papers = await mongo_db.get_by_ids(paper_ids)
        timings["mongo_get_ms"] = round((time.monotonic() - mongo_t0) * 1000, 2)

        results = [
            SearchResult(
                paper_id=p["paper_id"],
                title=p["title"],
                abstract=p["abstract"],
                authors=p.get("authors", []),
                categories=p.get("categories", []),
                published=p.get("published"),
                score=score_map.get(p["paper_id"], 0.0),
            )
            for p in papers
        ]

        took_ms = round((time.monotonic() - t0) * 1000, 2)
        timings["total_ms"] = took_ms
        logger.info(f"Search complete - {len(results)} results in {took_ms}ms (cache={'hit' if cache_hit else 'miss'})")
        logger.info(f"Search profile - {timings}")

        return SearchResponse(
            query=query,
            results=results,
            total=len(results),
            took_ms=took_ms,
        )

    async def index_paper(self, req: IndexRequest) -> IndexResponse:
        """Embed and index a single paper."""
        text = embedding_service.text_for_paper(req.title, req.abstract)
        vec = embedding_service.embed(text)

        qdrant_db.insert([req.paper_id], [vec])

        await mongo_db.upsert_paper({
            "paper_id": req.paper_id,
            "title": req.title,
            "abstract": req.abstract,
            "authors": req.authors or [],
            "categories": req.categories or [],
            "published": req.published,
        })

        return IndexResponse(paper_id=req.paper_id, status="ok", message="Paper indexed successfully")

    async def bulk_index(self, req: BulkIndexRequest) -> BulkIndexResponse:
        """Embed and index a batch of papers efficiently."""
        papers = req.papers
        texts = [embedding_service.text_for_paper(p.title, p.abstract) for p in papers]

        logger.info(f"Bulk indexing {len(papers)} papers")
        vecs = embedding_service.embed_batch(texts)

        paper_ids = [p.paper_id for p in papers]
        qdrant_db.insert(paper_ids, vecs)

        mongo_docs = [
            {
                "paper_id": p.paper_id,
                "title": p.title,
                "abstract": p.abstract,
                "authors": p.authors or [],
                "categories": p.categories or [],
                "published": p.published,
            }
            for p in papers
        ]
        await mongo_db.upsert_many(mongo_docs)

        logger.info(f"Bulk index complete - {len(papers)} papers")
        return BulkIndexResponse(indexed=len(papers), failed=0, total=len(papers))


search_service = SearchService()
