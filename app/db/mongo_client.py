from loguru import logger
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from pymongo import ASCENDING, IndexModel

from app.config import get_settings

settings = get_settings()


class MongoDB:
    """
    Async MongoDB client using Motor.
    Stores full paper metadata (title, abstract, authors, categories).
    Milvus holds the vectors; MongoDB holds the human-readable data.
    """

    _client: AsyncIOMotorClient | None = None

    async def connect(self) -> None:
        logger.info(f"Connecting to MongoDB @ {settings.mongo_uri}")
        self._client = AsyncIOMotorClient(settings.mongo_uri)
        await self._ensure_indexes()
        logger.info("MongoDB connected")

    async def disconnect(self) -> None:
        if self._client:
            self._client.close()

    @property
    def collection(self) -> AsyncIOMotorCollection:
        return self._client[settings.mongo_db][settings.mongo_collection]

    async def _ensure_indexes(self) -> None:
        indexes = [
            IndexModel([("paper_id", ASCENDING)], unique=True),
            IndexModel([("categories", ASCENDING)]),
            IndexModel([("published", ASCENDING)]),
        ]
        await self.collection.create_indexes(indexes)
        logger.info("MongoDB indexes ensured")

    async def upsert_paper(self, paper: dict) -> None:
        """Insert or update a paper by paper_id."""
        await self.collection.update_one(
            {"paper_id": paper["paper_id"]},
            {"$set": paper},
            upsert=True,
        )

    async def upsert_many(self, papers: list[dict]) -> int:
        """Bulk upsert. Returns number of upserted documents."""
        from pymongo import UpdateOne

        ops = [
            UpdateOne({"paper_id": p["paper_id"]}, {"$set": p}, upsert=True)
            for p in papers
        ]
        result = await self.collection.bulk_write(ops, ordered=False)
        return result.upserted_count + result.modified_count

    async def get_by_ids(self, paper_ids: list[str]) -> list[dict]:
        """Fetch papers by paper_id list. Preserves input order."""
        cursor = self.collection.find(
            {"paper_id": {"$in": paper_ids}},
            {"_id": 0},
        )
        docs = await cursor.to_list(length=len(paper_ids))
        # Restore order to match Milvus ranking
        id_map = {d["paper_id"]: d for d in docs}
        return [id_map[pid] for pid in paper_ids if pid in id_map]

    async def count(self) -> int:
        return await self.collection.count_documents({})


# Module-level singleton
mongo_db = MongoDB()
