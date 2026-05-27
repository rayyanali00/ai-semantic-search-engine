import sys
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.routes import router
from app.config import get_settings
from app.db.qdrant_client import qdrant_db
from app.db.mongo_client import mongo_db
from app.db.redis_client import redis_cache

settings = get_settings()

# ── Logging ───────────────────────────────────────────
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>",
    level="DEBUG" if settings.debug else "INFO",
)
logger.add(
    "logs/app.log",
    rotation="10 MB",
    retention="7 days",
    level="INFO",
)

# ── App ───────────────────────────────────────────────
app = FastAPI(
    title="Semantic Search Platform",
    description="AI-powered semantic search over 2M+ ArXiv papers using SentenceTransformers + Milvus",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request timing middleware ──────────────────────────
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    t0 = time.monotonic()
    response = await call_next(request)
    duration_ms = round((time.monotonic() - t0) * 1000, 2)
    response.headers["X-Process-Time-Ms"] = str(duration_ms)
    logger.debug(f"{request.method} {request.url.path} → {response.status_code} ({duration_ms}ms)")
    return response


# ── Lifecycle ─────────────────────────────────────────
@app.on_event("startup")
async def startup():
    logger.info("Starting up Semantic Search Platform...")
    await redis_cache.connect()
    await mongo_db.connect()
    qdrant_db.connect()
    logger.info("All services ready ✓")


@app.on_event("shutdown")
async def shutdown():
    logger.info("Shutting down...")
    await redis_cache.disconnect()
    await mongo_db.disconnect()
    qdrant_db.disconnect()


# ── Routes ────────────────────────────────────────────
app.include_router(router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Semantic Search Platform", "docs": "/docs", "version": "1.0.0"}
