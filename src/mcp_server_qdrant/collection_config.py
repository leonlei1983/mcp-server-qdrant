"""
Collection Configuration Management

管理不同 collection 與其對應的 embedding 模型配置
"""
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass

from mcp_server_qdrant.embeddings.types import EmbeddingProviderType

logger = logging.getLogger(__name__)


@dataclass
class CollectionConfig:
    """單個 collection 的配置"""
    name: str
    embedding_provider: EmbeddingProviderType
    embedding_model: str
    vector_name: str
    vector_size: int
    ollama_base_url: Optional[str] = None
    description: Optional[str] = None
    
    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "CollectionConfig":
        """從字典創建配置"""
        return cls(
            name=name,
            embedding_provider=EmbeddingProviderType(data["embedding_provider"]),
            embedding_model=data["embedding_model"],
            vector_name=data["vector_name"],
            vector_size=data["vector_size"],
            ollama_base_url=data.get("ollama_base_url"),
            description=data.get("description")
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        result = {
            "embedding_provider": self.embedding_provider.value,
            "embedding_model": self.embedding_model,
            "vector_name": self.vector_name,
            "vector_size": self.vector_size
        }
        if self.ollama_base_url:
            result["ollama_base_url"] = self.ollama_base_url
        if self.description:
            result["description"] = self.description
        return result


class CollectionConfigManager:
    """Collection 配置管理器"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，默認使用專案根目錄下的 collections.json
        """
        if config_path is None:
            # 使用專案根目錄
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "collections.json"
        
        self.config_path = config_path
        self.configs: Dict[str, CollectionConfig] = {}
        self._load_configs()
    
    def _load_configs(self):
        """載入配置文件"""
        if not self.config_path.exists():
            logger.info(f"Collection config file not found: {self.config_path}")
            logger.info("Using default configuration")
            self._create_default_config()
            return
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for collection_name, collection_data in data.get("collections", {}).items():
                self.configs[collection_name] = CollectionConfig.from_dict(
                    collection_name, collection_data
                )
            
            logger.info(f"Loaded {len(self.configs)} collection configurations")
            
        except Exception as e:
            logger.error(f"Failed to load collection config: {e}")
            self._create_default_config()
    
    def _create_default_config(self):
        """創建默認配置"""
        # 基於現有環境變數的默認配置
        default_config = CollectionConfig(
            name="default",
            embedding_provider=EmbeddingProviderType.FASTEMBED,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            vector_name="fast-all-minilm-l6-v2",
            vector_size=384,
            description="Default collection using FastEmbed"
        )
        
        self.configs["default"] = default_config
        
        # 創建示例配置文件
        self.save_configs()
    
    def get_config(self, collection_name: str) -> Optional[CollectionConfig]:
        """獲取指定 collection 的配置"""
        return self.configs.get(collection_name)
    
    def add_config(self, config: CollectionConfig):
        """添加新的 collection 配置"""
        self.configs[config.name] = config
    
    def remove_config(self, collection_name: str) -> bool:
        """移除 collection 配置"""
        if collection_name in self.configs:
            del self.configs[collection_name]
            return True
        return False
    
    def list_collections(self) -> Dict[str, CollectionConfig]:
        """列出所有 collection 配置"""
        return self.configs.copy()
    
    def save_configs(self):
        """保存配置到文件"""
        try:
            data = {
                "collections": {
                    name: config.to_dict() 
                    for name, config in self.configs.items()
                }
            }
            
            # 確保目錄存在
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved collection configs to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save collection config: {e}")
    
    def get_or_create_default(self, collection_name: str, 
                             default_provider: EmbeddingProviderType = EmbeddingProviderType.FASTEMBED,
                             default_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> CollectionConfig:
        """
        獲取配置，如果不存在則創建默認配置
        """
        config = self.get_config(collection_name)
        if config is not None:
            return config
        
        # 創建默認配置
        if default_provider == EmbeddingProviderType.FASTEMBED:
            # 從模型名稱推導 vector_name 和 vector_size
            model_name = default_model.split("/")[-1].lower()
            vector_name = f"fast-{model_name}"
            vector_size = 384  # all-MiniLM-L6-v2 的維度
        else:
            # 其他 provider 的默認設置
            vector_name = default_model.replace("/", "-").lower()
            vector_size = 768  # 常見維度
        
        config = CollectionConfig(
            name=collection_name,
            embedding_provider=default_provider,
            embedding_model=default_model,
            vector_name=vector_name,
            vector_size=vector_size,
            description=f"Auto-created config for {collection_name}"
        )
        
        self.add_config(config)
        self.save_configs()
        
        logger.info(f"Created default config for collection: {collection_name}")
        return config


# 全域配置管理器實例
_config_manager: Optional[CollectionConfigManager] = None


def get_collection_config_manager() -> CollectionConfigManager:
    """獲取全域配置管理器實例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = CollectionConfigManager()
    return _config_manager


def get_collection_config(collection_name: str) -> Optional[CollectionConfig]:
    """獲取指定 collection 的配置（便捷函數）"""
    manager = get_collection_config_manager()
    return manager.get_config(collection_name)