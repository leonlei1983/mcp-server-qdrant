"""
Dynamic Embedding Provider Manager

根據 collection 配置動態管理不同的 embedding providers
"""
import logging
from typing import Dict, Optional

from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.embeddings.types import EmbeddingProviderType
from mcp_server_qdrant.collection_config import (
    CollectionConfig, 
    get_collection_config_manager
)

logger = logging.getLogger(__name__)


class DynamicEmbeddingManager:
    """
    動態 Embedding Provider 管理器
    
    根據不同的 collection 配置，動態創建和管理對應的 embedding providers
    """
    
    def __init__(self):
        self.providers: Dict[str, EmbeddingProvider] = {}
        self.config_manager = get_collection_config_manager()
    
    def get_provider(self, collection_name: str) -> EmbeddingProvider:
        """
        獲取指定 collection 的 embedding provider
        
        Args:
            collection_name: Collection 名稱
            
        Returns:
            對應的 EmbeddingProvider 實例
        """
        # 檢查快取
        if collection_name in self.providers:
            return self.providers[collection_name]
        
        # 獲取 collection 配置
        config = self.config_manager.get_config(collection_name)
        if config is None:
            # 如果沒有配置，使用全域默認設置創建
            logger.warning(f"No config found for collection '{collection_name}', using default")
            config = self._create_default_config(collection_name)
        
        # 創建 embedding provider
        provider = self._create_provider_from_config(config)
        
        # 快取 provider
        self.providers[collection_name] = provider
        
        logger.info(f"Created embedding provider for collection '{collection_name}': "
                   f"{config.embedding_provider.value} - {config.embedding_model}")
        
        return provider
    
    def _create_provider_from_config(self, config: CollectionConfig) -> EmbeddingProvider:
        """根據配置創建 embedding provider"""
        
        # 直接創建 embedding provider，繞過 Settings 類
        if config.embedding_provider == EmbeddingProviderType.FASTEMBED:
            from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider
            return FastEmbedProvider(config.embedding_model)
        
        elif config.embedding_provider == EmbeddingProviderType.OLLAMA:
            from mcp_server_qdrant.embeddings.ollama import OllamaProvider
            base_url = config.ollama_base_url or "http://localhost:11434"
            return OllamaProvider(config.embedding_model, base_url)
        
        else:
            raise ValueError(f"Unsupported embedding provider: {config.embedding_provider}")
    
    def _create_default_config(self, collection_name: str) -> CollectionConfig:
        """為未配置的 collection 創建默認配置"""
        return self.config_manager.get_or_create_default(collection_name)
    
    def get_vector_info(self, collection_name: str) -> tuple[str, int]:
        """
        獲取 collection 的向量信息
        
        Returns:
            (vector_name, vector_size) 元組
        """
        config = self.config_manager.get_config(collection_name)
        if config is None:
            config = self._create_default_config(collection_name)
        
        return config.vector_name, config.vector_size
    
    def list_collection_configs(self) -> Dict[str, CollectionConfig]:
        """列出所有 collection 配置"""
        return self.config_manager.list_collections()
    
    def add_collection_config(self, config: CollectionConfig):
        """添加新的 collection 配置"""
        self.config_manager.add_config(config)
        
        # 如果已經有對應的 provider，清除快取以便重新創建
        if config.name in self.providers:
            del self.providers[config.name]
    
    def remove_collection_config(self, collection_name: str) -> bool:
        """移除 collection 配置"""
        # 清除快取
        if collection_name in self.providers:
            del self.providers[collection_name]
        
        return self.config_manager.remove_config(collection_name)
    
    def save_configs(self):
        """保存配置到文件"""
        self.config_manager.save_configs()
    
    def reload_configs(self):
        """重新載入配置"""
        # 清空所有快取的 providers
        self.providers.clear()
        
        # 重新載入配置
        self.config_manager._load_configs()
        
        logger.info("Reloaded collection configurations")
    
    def validate_collection_compatibility(self, collection_name: str) -> Dict[str, any]:
        """
        驗證 collection 與配置的兼容性
        
        Returns:
            包含驗證結果的字典
        """
        result = {
            "collection_name": collection_name,
            "config_exists": False,
            "vector_name_match": False,
            "vector_size_match": False,
            "provider_available": False,
            "errors": [],
            "warnings": []
        }
        
        try:
            # 檢查配置是否存在
            config = self.config_manager.get_config(collection_name)
            if config is None:
                result["warnings"].append("No specific config found, will use default")
                config = self._create_default_config(collection_name)
            else:
                result["config_exists"] = True
            
            # 嘗試創建 provider
            try:
                provider = self.get_provider(collection_name)
                result["provider_available"] = True
                
                # 檢查向量信息
                actual_vector_name = provider.get_vector_name()
                actual_vector_size = provider.get_vector_size()
                
                result["actual_vector_name"] = actual_vector_name
                result["actual_vector_size"] = actual_vector_size
                result["expected_vector_name"] = config.vector_name
                result["expected_vector_size"] = config.vector_size
                
                result["vector_name_match"] = actual_vector_name == config.vector_name
                result["vector_size_match"] = actual_vector_size == config.vector_size
                
                if not result["vector_name_match"]:
                    result["warnings"].append(
                        f"Vector name mismatch: expected {config.vector_name}, "
                        f"got {actual_vector_name}"
                    )
                
                if not result["vector_size_match"]:
                    result["errors"].append(
                        f"Vector size mismatch: expected {config.vector_size}, "
                        f"got {actual_vector_size}"
                    )
                
            except Exception as e:
                result["errors"].append(f"Failed to create provider: {str(e)}")
            
        except Exception as e:
            result["errors"].append(f"Validation failed: {str(e)}")
        
        result["is_valid"] = len(result["errors"]) == 0
        return result


# 全域管理器實例
_embedding_manager: Optional[DynamicEmbeddingManager] = None


def get_dynamic_embedding_manager() -> DynamicEmbeddingManager:
    """獲取全域動態 embedding 管理器實例"""
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = DynamicEmbeddingManager()
    return _embedding_manager