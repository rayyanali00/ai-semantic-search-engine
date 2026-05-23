from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ── Request Models ────────────────────────────────────

class EmbedRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to embed")

    model_config = {"json_schema_extra": {"example": {"text": "transformer attention mechanism"}}}


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results to return")
    filters: Optional[dict] = Field(default=None, description="Optional metadata filters")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "neural networks that explain their own decisions",
                "top_k": 10,
            }
        }
    }


class IndexRequest(BaseModel):
    paper_id: str = Field(..., description="ArXiv paper ID")
    title: str = Field(..., min_length=1, max_length=500)
    abstract: str = Field(..., min_length=1, max_length=5000)
    authors: Optional[list[str]] = Field(default=[])
    categories: Optional[list[str]] = Field(default=[])
    published: Optional[str] = Field(default=None)


class BulkIndexRequest(BaseModel):
    papers: list[IndexRequest] = Field(..., min_length=1, max_length=1000)


# ── Response Models ───────────────────────────────────

class EmbedResponse(BaseModel):
    text: str
    embedding: list[float]
    dim: int
    model: str


class SearchResult(BaseModel):
    paper_id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    published: Optional[str]
    score: float = Field(..., description="Cosine similarity score (0–1)")


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total: int
    took_ms: float


class IndexResponse(BaseModel):
    paper_id: str
    status: str
    message: str


class BulkIndexResponse(BaseModel):
    indexed: int
    failed: int
    total: int
    task_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str = "1.0.0"
    services: dict[str, str]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
