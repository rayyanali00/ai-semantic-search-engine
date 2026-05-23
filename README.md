# Semantic Search Platform

> AI-powered semantic search over 2M+ ArXiv research papers.  
> Built with SentenceTransformers, Milvus, FastAPI, Redis, MongoDB, and Streamlit.

![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![Docker](https://img.shields.io/badge/docker-compose-blue)
![CI](https://img.shields.io/badge/CI-GitHub_Actions-black)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## The Problem with Keyword Search

Traditional search engines match exact words. Search for *"AI that explains itself"* and you get nothing — even though hundreds of papers on **explainable AI, LIME, and SHAP** exist. Keyword search has no understanding of *meaning*.

**The vocabulary mismatch problem:**
- You say: *"transformers running on mobile"*
- Papers say: *"efficient inference on edge devices"*
- Keyword search: zero results

---

## The Solution: Semantic Search

Semantic search understands **meaning**, not just words.

1. Every paper's title + abstract is converted to a **768-dimensional embedding vector** using `all-mpnet-base-v2`
2. Vectors are stored in **Milvus**, a purpose-built vector database
3. At query time, your query becomes a vector — and Milvus finds the most **geometrically similar** paper vectors (cosine similarity)
4. Results are ranked by semantic closeness, not keyword frequency

```
User query: "neural networks that explain their decisions"
     ↓
Embedding model (all-mpnet-base-v2)
     ↓
768-dim query vector
     ↓
Milvus ANN search (top-k cosine similarity)
     ↓
MongoDB metadata fetch (title, authors, abstract)
     ↓
FastAPI JSON response → Streamlit UI
```

---

## Features

- **Semantic retrieval** — finds papers by meaning, not keywords
- **2M+ paper index** — full ArXiv corpus via HuggingFace datasets
- **Sub-100ms search** — IVF_FLAT index with Redis embedding cache
- **Async FastAPI** — production-grade API with Pydantic validation
- **Batched indexing** — index 512 papers per request, ~1000 papers/min
- **Metadata filtering** — filter by ArXiv category (cs.AI, cs.LG, etc.)
- **Streamlit UI** — interactive search interface with score visualization
- **Fully dockerized** — one command to run everything

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embedding model | `sentence-transformers/all-mpnet-base-v2` |
| Vector database | Milvus 2.4 (IVF_FLAT, cosine similarity) |
| API framework | FastAPI + Pydantic v2 |
| Metadata store | MongoDB 7.0 (Motor async driver) |
| Caching | Redis 7.2 (embedding result cache) |
| Task queue | Celery + Redis broker |
| Frontend | Streamlit + Plotly |
| Containerization | Docker + docker-compose |
| CI | GitHub Actions |
| Logging | loguru |
| Testing | pytest + pytest-asyncio |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI :8501                    │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP
┌────────────────────────▼────────────────────────────────┐
│                  FastAPI API :8000                       │
│   POST /embed   POST /search   POST /index   GET /health │
└──────┬──────────────┬───────────────┬───────────────────┘
       │              │               │
  ┌────▼────┐   ┌─────▼──────┐  ┌────▼──────┐
  │  Redis  │   │   Milvus   │  │  MongoDB  │
  │  cache  │   │ vector DB  │  │ metadata  │
  │  :6379  │   │   :19530   │  │  :27017   │
  └─────────┘   └────────────┘  └───────────┘
```

---

## Quick Start

### Prerequisites
- Docker + docker-compose
- 8GB RAM (Milvus is memory-hungry)
- 10GB disk space

### 1. Clone and configure

```bash
git clone https://github.com/yourusername/semantic-search-platform
cd semantic-search-platform
cp .env.example .env
```

### 2. Start all services

```bash
docker-compose up -d
```

Services spin up in order: etcd → minio → milvus → redis → mongo → api → ui

### 3. Wait for readiness

```bash
curl http://localhost:8000/api/v1/health
```

Expected: `{"status": "ok", "services": {...}}`

### 4. Ingest ArXiv papers

```bash
# Index 50,000 papers (takes ~15 minutes)
python scripts/ingest_arxiv.py --limit 50000

# Index only ML/AI papers
python scripts/ingest_arxiv.py --limit 10000 --categories cs.AI cs.LG cs.CL
```

### 5. Search

Open http://localhost:8501 in your browser, or use the API directly:

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "transformers that run on mobile devices", "top_k": 5}'
```

---

## API Reference

### `POST /api/v1/search`
```json
{
  "query": "neural networks that explain their decisions",
  "top_k": 10
}
```

### `POST /api/v1/embed`
```json
{ "text": "attention mechanism in transformers" }
```

### `POST /api/v1/index`
```json
{
  "paper_id": "2301.07041",
  "title": "Scaling Laws for Neural Language Models",
  "abstract": "We study empirical scaling laws...",
  "authors": ["Kaplan, J.", "McCandlish, S."],
  "categories": ["cs.LG", "cs.AI"]
}
```

### `POST /api/v1/index/bulk`
```json
{ "papers": [...] }
```

### `GET /api/v1/health`
Returns service status for Milvus, MongoDB, Redis, and the embedding model.

Full interactive docs at http://localhost:8000/docs

---

## Development

```bash
# Install deps
pip install -r requirements.txt

# Start infrastructure only
docker-compose up -d milvus redis mongo

# Run API locally
uvicorn main:app --reload

# Run Streamlit locally
streamlit run streamlit_app.py

# Run tests
pytest tests/ -v

# Lint
ruff check app/ main.py
```

---

## License

MIT
