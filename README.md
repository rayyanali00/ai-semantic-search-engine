# Semantic Search Platform

AI-powered semantic search for ArXiv-style research papers.

Built with FastAPI, SentenceTransformers, Qdrant, MongoDB, Redis, Celery, and Streamlit.

![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![Docker](https://img.shields.io/badge/docker-compose-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## What It Does

Keyword search only matches exact words. This project searches by meaning.

Example:

- Query: `AI that explains itself`
- Relevant papers may say: `explainable AI`, `interpretability`, `LIME`, or `SHAP`
- Semantic search can still connect them because it compares embedding vectors, not just text tokens.

---

## Current Flow

```text
User query or paper text
        |
        v
FastAPI API (/api/v1)
        |
        v
SentenceTransformers embedding model
        |
        +--> Redis caches query embeddings
        |
        v
Qdrant stores and searches vectors by cosine similarity
        |
        v
MongoDB stores and returns paper metadata
        |
        v
FastAPI response -> Streamlit UI
```

### Indexing Flow

```text
scripts/ingest_arxiv.py
        |
        v
Hugging Face dataset stream
        |
        v
POST /api/v1/index/bulk
        |
        v
Embed title + abstract
        |
        +--> Qdrant: paper_id + vector
        |
        +--> MongoDB: title, abstract, authors, categories, published date
```

### Search Flow

```text
POST /api/v1/search
        |
        v
Check Redis for cached query embedding
        |
        v
Embed query on cache miss
        |
        v
Qdrant nearest-neighbor search
        |
        v
Fetch matching paper metadata from MongoDB
        |
        v
Return ranked results with similarity scores
```

---

## Features

- Semantic search over paper titles and abstracts
- Local embedding with `sentence-transformers/all-mpnet-base-v2`
- Qdrant vector storage using cosine similarity
- MongoDB metadata storage
- Redis embedding cache
- Bulk ingestion from Hugging Face datasets
- FastAPI JSON API with interactive docs
- Streamlit UI for searching and indexing single papers
- Docker Compose setup for API, UI, worker, Qdrant, Redis, and MongoDB

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Pydantic v2 |
| Embeddings | SentenceTransformers |
| Embedding model | `sentence-transformers/all-mpnet-base-v2` |
| Vector database | Qdrant |
| Metadata store | MongoDB + Motor |
| Cache / broker | Redis |
| Worker | Celery |
| Frontend | Streamlit + Plotly |
| Ingestion | Hugging Face `datasets` |
| Tests | pytest + pytest-asyncio |

---

## Services

| Service | URL / Port | Purpose |
|---|---:|---|
| API | http://localhost:8000 | FastAPI app |
| API docs | http://localhost:8000/docs | OpenAPI docs |
| Streamlit UI | http://localhost:8501 | Search interface |
| Qdrant | http://localhost:6333 | Vector database |
| Redis | localhost:6379 | Cache and Celery broker |
| MongoDB | localhost:27017 | Paper metadata |

---

## Quick Start

### 1. Configure Environment

```bash
cp .env.example .env
```

For Docker Compose, make sure `.env` uses Docker service hostnames:

```env
QDRANT_HOST=qdrant
REDIS_HOST=redis
MONGO_URI=mongodb://mongo:27017
API_BASE=http://api:8000/api/v1
```

For running the API directly on your host machine, use localhost values instead:

```env
QDRANT_HOST=localhost
REDIS_HOST=localhost
MONGO_URI=mongodb://localhost:27017
API_BASE=http://localhost:8000/api/v1
```

### 2. Start the Platform

```bash
docker-compose up -d
```

This starts:

```text
qdrant -> redis -> mongo -> api -> worker -> ui
```

### 3. Check Health

```bash
curl http://localhost:8000/api/v1/health
```

Expected shape:

```json
{
  "status": "ok",
  "app": "semantic-search-platform",
  "services": {
    "qdrant": "ok (... vectors)",
    "mongodb": "ok (... papers)",
    "redis": "ok",
    "embedding_model": "all-mpnet-base-v2"
  }
}
```

### 4. Ingest Papers

Run ingestion from your host machine:

```bash
python scripts/ingest_arxiv.py --limit 50000 --batch-size 64 --api-base http://localhost:8000/api/v1
```

Filter to selected categories:

```bash
python scripts/ingest_arxiv.py --limit 10000 --categories cs.AI cs.LG cs.CL --api-base http://localhost:8000/api/v1
```

The script streams `CShorten/ML-ArXiv-Papers` from Hugging Face, prepares paper records, and sends batches to `/api/v1/index/bulk`.

### 5. Search

Open the UI:

```text
http://localhost:8501
```

Or call the API directly:

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "transformers that run on mobile devices", "top_k": 5}'
```

---

## API Reference

### `GET /api/v1/health`

Checks Qdrant, MongoDB, Redis, and the embedding model configuration.

### `POST /api/v1/embed`

Embeds one text string and caches the result in Redis.

```json
{
  "text": "attention mechanism in transformers"
}
```

### `POST /api/v1/index`

Indexes one paper.

```json
{
  "paper_id": "2301.07041",
  "title": "Example Paper Title",
  "abstract": "Paper abstract text...",
  "authors": ["Author One", "Author Two"],
  "categories": ["cs.LG", "cs.AI"],
  "published": "2023-01-01"
}
```

### `POST /api/v1/index/bulk`

Indexes up to 1000 papers in one request.

```json
{
  "papers": [
    {
      "paper_id": "2301.07041",
      "title": "Example Paper Title",
      "abstract": "Paper abstract text...",
      "authors": [],
      "categories": ["cs.LG"],
      "published": ""
    }
  ]
}
```

### `POST /api/v1/search`

Searches indexed papers by semantic similarity.

```json
{
  "query": "neural networks that explain their decisions",
  "top_k": 10
}
```

---

## Development

Install dependencies:

```bash
pip install -r requirements.txt
```

Start only the backing services:

```bash
docker-compose up -d qdrant redis mongo
```

Run the API locally:

```bash
uvicorn main:app --reload
```

Run the UI locally:

```bash
streamlit run streamlit_app.py
```

Run tests:

```bash
pytest tests/ -v
```

---

## Notes

- The first embedding request may be slow while the SentenceTransformers model loads.
- Ingestion requires internet access to stream the Hugging Face dataset.
- Query embeddings are cached in Redis using a content hash.
- Qdrant stores vectors with `paper_id` in the payload.
- MongoDB is the source of truth for human-readable paper metadata.

---

## License

MIT
