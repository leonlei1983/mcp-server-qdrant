"""
Collection-Aware Qdrant Connector

支援不同 collection 使用不同 embedding 模型的 Qdrant 連接器
"""
import logging
from typing import Optional, Dict

from qdrant_client import AsyncQdrantClient, models

from mcp_server_qdrant.qdrant import Entry, Metadata, ArbitraryFilter
from mcp_server_qdrant.dynamic_embedding_manager import get_dynamic_embedding_manager
from mcp_server_qdrant.collection_config import CollectionConfig

logger = logging.getLogger(__name__)


class CollectionAwareQdrantConnector:
    """
    Collection-Aware Qdrant 連接器
    
    根據不同的 collection 自動使用對應的 embedding 模型
    """
    
    def __init__(
        self,
        qdrant_url: str | None,
        qdrant_api_key: str | None,
        default_collection_name: str | None = None,
        qdrant_local_path: str | None = None,
        field_indexes: dict[str, models.PayloadSchemaType] | None = None,
    ):
        self._qdrant_url = qdrant_url.rstrip("/") if qdrant_url else None
        self._qdrant_api_key = qdrant_api_key
        self._default_collection_name = default_collection_name
        self._client = AsyncQdrantClient(
            location=qdrant_url, api_key=qdrant_api_key, path=qdrant_local_path
        )
        self._field_indexes = field_indexes
        
        # 動態 embedding 管理器
        self._embedding_manager = get_dynamic_embedding_manager()
    
    async def get_collection_names(self) -> list[str]:
        """獲取所有 collection 名稱"""
        response = await self._client.get_collections()
        return [collection.name for collection in response.collections]
    
    async def store(self, entry: Entry, *, collection_name: str | None = None):
        """
        儲存資料到指定 collection
        
        自動使用該 collection 對應的 embedding 模型
        """
        collection_name = collection_name or self._default_collection_name
        if collection_name is None:
            collection_name = "default"
            logger.warning(f"No collection name provided, using default collection: {collection_name}")
        
        await self._ensure_collection_exists(collection_name)
        
        # 獲取對應的 embedding provider
        embedding_provider = self._embedding_manager.get_provider(collection_name)
        
        # 生成文檔向量
        document_vector = await embedding_provider.embed_documents([entry.content])
        vector_name = embedding_provider.get_vector_name()
        
        # 準備 payload
        payload = {"document": entry.content}
        if entry.metadata:
            payload["metadata"] = entry.metadata
        
        # 儲存到 Qdrant
        import uuid
        await self._client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={vector_name: document_vector[0]},
                    payload=payload,
                )
            ],
        )
        
        logger.info(f"Stored entry in collection '{collection_name}' using {embedding_provider.__class__.__name__}")
    
    async def search(
        self,
        query: str,
        *,
        collection_name: str | None = None,
        limit: int = 10,
        query_filter: models.Filter | None = None,
    ) -> list[Entry]:
        """
        在指定 collection 中搜尋
        
        自動使用該 collection 對應的 embedding 模型
        """
        collection_name = collection_name or self._default_collection_name
        if collection_name is None:
            collection_name = "default"
        
        # 檢查 collection 是否存在
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            logger.warning(f"Collection '{collection_name}' does not exist")
            return []
        
        # 獲取對應的 embedding provider
        embedding_provider = self._embedding_manager.get_provider(collection_name)
        
        # 生成查詢向量
        query_vector = await embedding_provider.embed_query(query)
        vector_name = embedding_provider.get_vector_name()
        
        # 在 Qdrant 中搜尋
        search_results = await self._client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=vector_name,
            limit=limit,
            query_filter=query_filter,
        )
        
        # 轉換結果
        results = [
            Entry(
                content=result.payload["document"],
                metadata=result.payload.get("metadata"),
            )
            for result in search_results.points
        ]
        
        logger.debug(f"Found {len(results)} results in collection '{collection_name}' "
                    f"using {embedding_provider.__class__.__name__}")
        
        return results
    
    async def _ensure_collection_exists(self, collection_name: str):
        """確保 collection 存在，如果不存在則創建"""
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            # 獲取配置信息
            vector_name, vector_size = self._embedding_manager.get_vector_info(collection_name)
            
            # 創建 collection
            await self._client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    vector_name: models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    )
                },
            )
            
            logger.info(f"Created collection '{collection_name}' with vector '{vector_name}' (size: {vector_size})")
    
    async def get_collection_info(self, collection_name: str) -> Optional[Dict]:
        """獲取 collection 的詳細信息"""
        try:
            collection_exists = await self._client.collection_exists(collection_name)
            if not collection_exists:
                return None
            
            # 獲取 Qdrant collection 信息
            collection_info = await self._client.get_collection(collection_name)
            
            # 獲取配置信息
            config = self._embedding_manager.config_manager.get_config(collection_name)
            
            # 組合信息
            result = {
                "name": collection_name,
                "points_count": collection_info.points_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "vectors_config": collection_info.config.params.vectors,
                "status": collection_info.status,
            }
            
            if config:
                result["embedding_config"] = {
                    "provider": config.embedding_provider.value,
                    "model": config.embedding_model,
                    "vector_name": config.vector_name,
                    "vector_size": config.vector_size,
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get collection info for '{collection_name}': {e}")
            return None
    
    async def validate_collection(self, collection_name: str) -> Dict:
        """驗證 collection 的配置和狀態"""
        return self._embedding_manager.validate_collection_compatibility(collection_name)
    
    def get_embedding_manager(self):
        """獲取 embedding 管理器（用於配置管理）"""
        return self._embedding_manager