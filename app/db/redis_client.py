import json
from typing import Any

import redis.asyncio as aioredis
from loguru import logger

from app.config import get_settings

settings = get_settings()


class RedisCache:
    """
    Async Redis cache for embedding results.
    Avoids re-encoding identical queries (saves ~100ms per hit).
    """

    _client: aioredis.Redis | None = None

    async def connect(self) -> None:
        url = f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}"
        if settings.redis_password:
            url = f"redis://:{settings.redis_password}@{settings.redis_host}:{settings.redis_port}/{settings.redis_db}"
        self._client = aioredis.from_url(url, decode_responses=True)
        await self._client.ping()
        logger.info("Redis connected")

    async def disconnect(self) -> None:
        if self._client:
            await self._client.close()

    async def get(self, key: str) -> Any | None:
        try:
            val = await self._client.get(key)
            return json.loads(val) if val else None
        except Exception as e:
            logger.warning(f"Redis GET failed for key={key}: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        try:
            ttl = ttl or settings.cache_ttl
            await self._client.setex(key, ttl, json.dumps(value))
        except Exception as e:
            logger.warning(f"Redis SET failed for key={key}: {e}")

    async def delete(self, key: str) -> None:
        await self._client.delete(key)

    async def ping(self) -> bool:
        try:
            return await self._client.ping()
        except Exception:
            return False

    async def info(self) -> dict:
        try:
            i = await self._client.info("server")
            return {"version": i.get("redis_version", "unknown"), "status": "ok"}
        except Exception:
            return {"status": "unreachable"}


# Module-level singleton
redis_cache = RedisCache()
