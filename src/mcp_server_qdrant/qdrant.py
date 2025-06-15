import logging
import uuid
from typing import Any

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models

from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.settings import METADATA_PATH

logger = logging.getLogger(__name__)

Metadata = dict[str, Any]
ArbitraryFilter = dict[str, Any]


class Entry(BaseModel):
    """
    A single entry in the Qdrant collection.
    """

    content: str
    metadata: Metadata | None = None


class QdrantConnector:
    """
    Encapsulates the connection to a Qdrant server and all the methods to interact with it.
    :param qdrant_url: The URL of the Qdrant server.
    :param qdrant_api_key: The API key to use for the Qdrant server.
    :param collection_name: The name of the default collection to use. If not provided, each tool will require
                            the collection name to be provided.
    :param embedding_provider: The embedding provider to use.
    :param qdrant_local_path: The path to the storage directory for the Qdrant client, if local mode is used.
    """

    def __init__(
        self,
        qdrant_url: str | None,
        qdrant_api_key: str | None,
        collection_name: str | None,
        embedding_provider: EmbeddingProvider,
        qdrant_local_path: str | None = None,
        field_indexes: dict[str, models.PayloadSchemaType] | None = None,
    ):
        self._qdrant_url = qdrant_url.rstrip("/") if qdrant_url else None
        self._qdrant_api_key = qdrant_api_key
        self._default_collection_name = collection_name
        self._embedding_provider = embedding_provider
        self._client = AsyncQdrantClient(
            location=qdrant_url, api_key=qdrant_api_key, path=qdrant_local_path
        )
        self._field_indexes = field_indexes

    async def get_collection_names(self) -> list[str]:
        """
        Get the names of all collections in the Qdrant server.
        :return: A list of collection names.
        """
        response = await self._client.get_collections()
        return [collection.name for collection in response.collections]

    async def store(self, entry: Entry, *, collection_name: str | None = None):
        """
        Store some information in the Qdrant collection, along with the specified metadata.
        :param entry: The entry to store in the Qdrant collection.
        :param collection_name: The name of the collection to store the information in, optional. If not provided,
                                the default collection is used.
        """
        collection_name = collection_name or self._default_collection_name
        if collection_name is None:
            # 提供智能預設值，避免 AI 忘記指定
            collection_name = "default"
            logger.warning(f"No collection name provided, using default collection: {collection_name}")
        await self._ensure_collection_exists(collection_name)

        # Embed the document
        # ToDo: instead of embedding text explicitly, use `models.Document`,
        # it should unlock usage of server-side inference.
        embeddings = await self._embedding_provider.embed_documents([entry.content])

        # Add to Qdrant
        vector_name = self._embedding_provider.get_vector_name()
        payload = {"document": entry.content, METADATA_PATH: entry.metadata}
        await self._client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=uuid.uuid4().hex,
                    vector={vector_name: embeddings[0]},
                    payload=payload,
                )
            ],
        )

    async def search(
        self,
        query: str,
        *,
        collection_name: str | None = None,
        limit: int = 10,
        query_filter: models.Filter | None = None,
    ) -> list[Entry]:
        """
        Find points in the Qdrant collection. If there are no entries found, an empty list is returned.
        :param query: The query to use for the search.
        :param collection_name: The name of the collection to search in, optional. If not provided,
                                the default collection is used.
        :param limit: The maximum number of entries to return.
        :param query_filter: The filter to apply to the query, if any.

        :return: A list of entries found.
        """
        collection_name = collection_name or self._default_collection_name
        if collection_name is None:
            # 提供智能預設值，避免 AI 忘記指定
            collection_name = "default"
            logger.warning(f"No collection name provided, using default collection: {collection_name}")
        
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return []

        # Embed the query
        # ToDo: instead of embedding text explicitly, use `models.Document`,
        # it should unlock usage of server-side inference.

        query_vector = await self._embedding_provider.embed_query(query)
        vector_name = self._embedding_provider.get_vector_name()

        # Search in Qdrant
        search_results = await self._client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=vector_name,
            limit=limit,
            query_filter=query_filter,
        )

        return [
            Entry(
                content=result.payload["document"],
                metadata=result.payload.get("metadata"),
            )
            for result in search_results.points
        ]

    async def _ensure_collection_exists(self, collection_name: str):
        """
        Ensure that the collection exists, creating it if necessary.
        :param collection_name: The name of the collection to ensure exists.
        """
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            # Create the collection with the appropriate vector size
            vector_size = self._embedding_provider.get_vector_size()

            # Use the vector name as defined in the embedding provider
            vector_name = self._embedding_provider.get_vector_name()
            await self._client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    vector_name: models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    )
                },
            )

            # Create payload indexes if configured

            if self._field_indexes:
                for field_name, field_type in self._field_indexes.items():
                    await self._client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema=field_type,
                    )

    async def delete_documents(
        self,
        query: str,
        *,
        collection_name: str | None = None,
        limit: int = 10,
        query_filter: models.Filter | None = None,
        confirm_delete: bool = False,
    ) -> dict[str, any]:
        """
        刪除符合搜尋條件的文檔
        :param query: 搜尋查詢，用於找到要刪除的文檔
        :param collection_name: collection 名稱
        :param limit: 最多刪除的文檔數量
        :param query_filter: 額外的過濾條件
        :param confirm_delete: 確認刪除，防止意外刪除
        :return: 刪除結果統計
        """
        collection_name = collection_name or self._default_collection_name
        if collection_name is None:
            collection_name = "default"
            logger.warning(f"No collection name provided, using default collection: {collection_name}")

        if not confirm_delete:
            raise ValueError("必須設定 confirm_delete=True 才能執行刪除操作，防止意外刪除")

        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return {"error": f"Collection '{collection_name}' 不存在", "deleted_count": 0}

        # 先搜尋要刪除的文檔
        query_vector = await self._embedding_provider.embed_query(query)
        vector_name = self._embedding_provider.get_vector_name()

        search_results = await self._client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=vector_name,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )

        if not search_results.points:
            return {"message": f"沒有找到符合條件 '{query}' 的文檔", "deleted_count": 0}

        # 收集要刪除的 point IDs
        point_ids = [point.id for point in search_results.points]
        
        # 執行刪除
        await self._client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(points=point_ids),
        )

        return {
            "message": f"成功刪除 {len(point_ids)} 個文檔",
            "deleted_count": len(point_ids),
            "collection_name": collection_name,
            "deleted_ids": point_ids,
        }

    async def delete_collection(
        self,
        collection_name: str,
        confirm_delete: bool = False,
    ) -> dict[str, any]:
        """
        刪除整個 collection
        :param collection_name: 要刪除的 collection 名稱
        :param confirm_delete: 確認刪除，防止意外刪除
        :return: 刪除結果
        """
        if not confirm_delete:
            raise ValueError("必須設定 confirm_delete=True 才能執行 collection 刪除操作，防止意外刪除")

        if not collection_name:
            raise ValueError("Collection name 不能為空")

        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return {"error": f"Collection '{collection_name}' 不存在", "deleted": False}

        # 執行刪除
        await self._client.delete_collection(collection_name)

        return {
            "message": f"成功刪除 collection '{collection_name}'",
            "deleted": True,
            "collection_name": collection_name,
        }

    async def list_collections(self) -> list[str]:
        """
        列出所有可用的 collections
        :return: collection 名稱列表
        """
        return await self.get_collection_names()

    async def move_documents(
        self,
        query: str,
        source_collection: str,
        target_collection: str,
        *,
        limit: int = 10,
        query_filter: models.Filter | None = None,
        confirm_move: bool = False,
    ) -> dict[str, any]:
        """
        搬移文檔從一個 collection 到另一個 collection
        :param query: 搜尋查詢，用於找到要搬移的文檔
        :param source_collection: 來源 collection
        :param target_collection: 目標 collection
        :param limit: 最多搬移的文檔數量
        :param query_filter: 額外的過濾條件
        :param confirm_move: 確認搬移，防止意外操作
        :return: 搬移結果統計
        """
        if not confirm_move:
            raise ValueError("必須設定 confirm_move=True 才能執行搬移操作，防止意外搬移")

        # 先從來源搜尋文檔
        entries = await self.search(
            query,
            collection_name=source_collection,
            limit=limit,
            query_filter=query_filter,
        )

        if not entries:
            return {"message": f"在 '{source_collection}' 中沒有找到符合條件 '{query}' 的文檔", "moved_count": 0}

        # 複製到目標 collection
        moved_count = 0
        for entry in entries:
            await self.store(entry, collection_name=target_collection)
            moved_count += 1

        # 從來源刪除
        delete_result = await self.delete_documents(
            query,
            collection_name=source_collection,
            limit=limit,
            query_filter=query_filter,
            confirm_delete=True,
        )

        return {
            "message": f"成功搬移 {moved_count} 個文檔從 '{source_collection}' 到 '{target_collection}'",
            "moved_count": moved_count,
            "source_collection": source_collection,
            "target_collection": target_collection,
            "deleted_from_source": delete_result.get("deleted_count", 0),
        }

    # 已修正 PointStruct 向量資料問題 - 加上 with_vectors=True 和向量驗證
    # 等待 MCP server 重啟後測試驗證
    async def update_metadata(
        self,
        query: str,
        new_metadata: dict[str, any],
        *,
        collection_name: str | None = None,
        limit: int = 10,
        query_filter: models.Filter | None = None,
        confirm_update: bool = False,
    ) -> dict[str, any]:
        """
        更新符合搜尋條件的文檔 metadata
        :param query: 搜尋查詢，用於找到要更新的文檔
        :param new_metadata: 新的 metadata，會與現有 metadata 合併
        :param collection_name: collection 名稱
        :param limit: 最多更新的文檔數量
        :param query_filter: 額外的過濾條件
        :param confirm_update: 確認更新，防止意外更新
        :return: 更新結果統計
        """
        collection_name = collection_name or self._default_collection_name
        if collection_name is None:
            collection_name = "default"
            logger.warning(f"No collection name provided, using default collection: {collection_name}")

        if not confirm_update:
            raise ValueError("必須設定 confirm_update=True 才能執行 metadata 更新操作，防止意外更新")

        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return {"error": f"Collection '{collection_name}' 不存在", "updated_count": 0}

        # 先搜尋要更新的文檔
        query_vector = await self._embedding_provider.embed_query(query)
        vector_name = self._embedding_provider.get_vector_name()

        search_results = await self._client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=vector_name,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=True,  # 修正：確保返回向量資料
        )

        if not search_results.points:
            return {"message": f"沒有找到符合條件 '{query}' 的文檔", "updated_count": 0}

        # 更新每個找到的文檔
        updated_count = 0
        updated_ids = []
        
        for point in search_results.points:
            # 合併現有 metadata 和新 metadata
            current_metadata = point.payload.get(METADATA_PATH, {}) or {}
            if isinstance(current_metadata, dict):
                updated_metadata = {**current_metadata, **new_metadata}
            else:
                updated_metadata = new_metadata

            # 更新文檔
            new_payload = {
                **point.payload,
                METADATA_PATH: updated_metadata
            }

            # 修正：檢查向量資料
            vector_data = point.vector
            if vector_data is None:
                logger.warning(f"Point {point.id} has no vector data, skipping update")
                continue

            await self._client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=point.id,
                        vector=vector_data,
                        payload=new_payload,
                    )
                ],
            )
            
            updated_count += 1
            updated_ids.append(point.id)

        return {
            "message": f"成功更新 {updated_count} 個文檔的 metadata",
            "updated_count": updated_count,
            "collection_name": collection_name,
            "updated_ids": updated_ids,
        }

    # 已修正 PointStruct 向量資料問題 (同 update_metadata)
    async def remove_metadata_keys(
        self,
        query: str,
        keys_to_remove: list[str],
        *,
        collection_name: str | None = None,
        limit: int = 10,
        query_filter: models.Filter | None = None,
        confirm_update: bool = False,
    ) -> dict[str, any]:
        """
        從符合搜尋條件的文檔 metadata 中移除指定的鍵
        :param query: 搜尋查詢，用於找到要更新的文檔
        :param keys_to_remove: 要移除的 metadata 鍵列表
        :param collection_name: collection 名稱
        :param limit: 最多更新的文檔數量
        :param query_filter: 額外的過濾條件
        :param confirm_update: 確認更新，防止意外更新
        :return: 更新結果統計
        """
        collection_name = collection_name or self._default_collection_name
        if collection_name is None:
            collection_name = "default"
            logger.warning(f"No collection name provided, using default collection: {collection_name}")

        if not confirm_update:
            raise ValueError("必須設定 confirm_update=True 才能執行 metadata 鍵移除操作，防止意外更新")

        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return {"error": f"Collection '{collection_name}' 不存在", "updated_count": 0}

        # 先搜尋要更新的文檔
        query_vector = await self._embedding_provider.embed_query(query)
        vector_name = self._embedding_provider.get_vector_name()

        search_results = await self._client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=vector_name,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=True,  # 修正：確保返回向量資料
        )

        if not search_results.points:
            return {"message": f"沒有找到符合條件 '{query}' 的文檔", "updated_count": 0}

        # 更新每個找到的文檔
        updated_count = 0
        updated_ids = []
        
        for point in search_results.points:
            # 從現有 metadata 中移除指定鍵
            current_metadata = point.payload.get(METADATA_PATH, {}) or {}
            if isinstance(current_metadata, dict):
                updated_metadata = {k: v for k, v in current_metadata.items() if k not in keys_to_remove}
            else:
                updated_metadata = {}

            # 更新文檔
            new_payload = {
                **point.payload,
                METADATA_PATH: updated_metadata
            }

            # 修正：檢查向量資料
            vector_data = point.vector
            if vector_data is None:
                logger.warning(f"Point {point.id} has no vector data, skipping update")
                continue

            await self._client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=point.id,
                        vector=vector_data,
                        payload=new_payload,
                    )
                ],
            )
            
            updated_count += 1
            updated_ids.append(point.id)

        return {
            "message": f"成功從 {updated_count} 個文檔的 metadata 中移除鍵: {', '.join(keys_to_remove)}",
            "updated_count": updated_count,
            "removed_keys": keys_to_remove,
            "collection_name": collection_name,
            "updated_ids": updated_ids,
        }
