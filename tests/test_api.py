"""
Tests for the Semantic Search Platform API.
Run with: pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

from main import app

client = TestClient(app)


# ── Fixtures ──────────────────────────────────────────

@pytest.fixture(autouse=True)
def mock_services():
    """Mock all external services so tests run without infrastructure."""
    mock_vec = [0.1] * 768

    with (
        patch("app.api.routes.redis_cache") as mock_redis,
        patch("app.api.routes.qdrant_db") as mock_qdrant,
        patch("app.api.routes.mongo_db") as mock_mongo,
        patch("app.api.routes.embedding_service") as mock_embed,
        patch("app.services.search_service.redis_cache") as mock_redis2,
        patch("app.services.search_service.qdrant_db") as mock_qdrant2,
        patch("app.services.search_service.mongo_db") as mock_mongo2,
        patch("app.services.search_service.embedding_service") as mock_embed2,
    ):
        # Redis
        for m in [mock_redis, mock_redis2]:
            m.get = AsyncMock(return_value=None)
            m.set = AsyncMock()
            m.info = AsyncMock(return_value={"status": "ok", "version": "7.2.0"})
            m.ping = AsyncMock(return_value=True)

        # Milvus
        for m in [mock_qdrant, mock_qdrant2]:
            m.search.return_value = [
                {"paper_id": "2301.00001", "score": 0.92},
                {"paper_id": "2301.00002", "score": 0.87},
            ]
            m.insert.return_value = 1
            m.count.return_value = 50000

        # MongoDB
        mock_mongo.count = AsyncMock(return_value=50000)
        mock_mongo2.count = AsyncMock(return_value=50000)
        mock_mongo2.get_by_ids = AsyncMock(return_value=[
            {
                "paper_id": "2301.00001",
                "title": "Attention Is All You Need",
                "abstract": "The dominant sequence transduction models...",
                "authors": ["Vaswani, A.", "Shazeer, N."],
                "categories": ["cs.CL", "cs.LG"],
                "published": "2017-06-12",
            },
            {
                "paper_id": "2301.00002",
                "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "abstract": "We introduce a new language representation...",
                "authors": ["Devlin, J.", "Chang, M."],
                "categories": ["cs.CL"],
                "published": "2018-10-11",
            },
        ])
        mock_mongo.upsert_paper = AsyncMock()
        mock_mongo2.upsert_paper = AsyncMock()
        mock_mongo2.upsert_many = AsyncMock(return_value=2)

        # Embedding
        for m in [mock_embed, mock_embed2]:
            m.embed.return_value = mock_vec
            m.embed_batch.return_value = [mock_vec, mock_vec]
            m.cache_key.return_value = "embed:all-mpnet-base-v2:abc123"
            m.text_for_paper.return_value = "Test title. Test title. Test abstract."

        yield


# ── Health ────────────────────────────────────────────

def test_health_check():
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "services" in data
    assert "qdrant" in data["services"]
    assert "redis" in data["services"]


# ── Embed ─────────────────────────────────────────────

def test_embed_text():
    resp = client.post("/api/v1/embed", json={"text": "transformer attention mechanism"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["dim"] == 768
    assert len(data["embedding"]) == 768
    assert data["text"] == "transformer attention mechanism"


def test_embed_empty_text_fails():
    resp = client.post("/api/v1/embed", json={"text": ""})
    assert resp.status_code == 422


def test_embed_too_long_fails():
    resp = client.post("/api/v1/embed", json={"text": "x" * 10001})
    assert resp.status_code == 422


# ── Search ────────────────────────────────────────────

def test_search_returns_results():
    resp = client.post("/api/v1/search", json={"query": "neural networks that explain their decisions", "top_k": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == "neural networks that explain their decisions"
    assert isinstance(data["results"], list)
    assert data["total"] >= 0


def test_search_top_k_validation():
    resp = client.post("/api/v1/search", json={"query": "test", "top_k": 100})
    assert resp.status_code == 422

    resp = client.post("/api/v1/search", json={"query": "test", "top_k": 0})
    assert resp.status_code == 422


def test_search_result_schema():
    resp = client.post("/api/v1/search", json={"query": "attention is all you need"})
    assert resp.status_code == 200
    results = resp.json()["results"]
    if results:
        r = results[0]
        assert "paper_id" in r
        assert "title" in r
        assert "abstract" in r
        assert "score" in r
        assert 0 <= r["score"] <= 1


# ── Index ─────────────────────────────────────────────

def test_index_paper():
    resp = client.post("/api/v1/index", json={
        "paper_id": "2301.99999",
        "title": "Test Paper Title",
        "abstract": "This is a test abstract with sufficient length to pass validation.",
        "authors": ["Test Author"],
        "categories": ["cs.LG"],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["paper_id"] == "2301.99999"


def test_bulk_index():
    papers = [
        {
            "paper_id": f"2301.{i:05d}",
            "title": f"Paper {i}",
            "abstract": "This is a test abstract for bulk indexing purposes with enough text.",
        }
        for i in range(5)
    ]
    resp = client.post("/api/v1/index/bulk", json={"papers": papers})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 5


def test_bulk_index_empty_fails():
    resp = client.post("/api/v1/index/bulk", json={"papers": []})
    assert resp.status_code == 422
