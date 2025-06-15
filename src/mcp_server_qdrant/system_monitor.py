"""
純 Qdrant API 監控模組 - 只使用 Qdrant 自身提供的功能
重構版本：完全基於 Qdrant API，無外部系統依賴
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

logger = logging.getLogger(__name__)


class QdrantOnlyMonitor:
    """純 Qdrant API 監控器 - 只使用 Qdrant 自身提供的功能"""
    
    def __init__(self, client: AsyncQdrantClient, qdrant_url: str):
        self.client = client
        self.qdrant_url = qdrant_url
        self.deployment_info = self._detect_deployment_type(qdrant_url)
        
    def _detect_deployment_type(self, qdrant_url: str) -> Dict[str, Any]:
        """檢測 Qdrant 的部署類型（僅基於 URL 分析）"""
        if not qdrant_url:
            return {"type": "unknown", "description": "未知部署類型"}
        
        parsed = urlparse(qdrant_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 6333
        
        # 基於 URL 的簡單檢測
        if any(cloud in host for cloud in ['qdrant.io', 'cloud.qdrant', 'qdrant.tech']):
            return {
                "type": "cloud",
                "description": "Qdrant Cloud 部署",
                "host": host,
                "port": port,
                "features": ["managed_service", "auto_scaling", "backup"]
            }
        elif host == "localhost" or host == "127.0.0.1":
            return {
                "type": "local",
                "description": "本地部署",
                "host": host,
                "port": port,
                "features": ["local_access", "development"]
            }
        else:
            return {
                "type": "remote",
                "description": "遠端部署",
                "host": host,
                "port": port,
                "features": ["remote_access", "production"]
            }
    
    async def get_cluster_info(self) -> Dict[str, Any]:
        """獲取集群資訊"""
        try:
            cluster_info = await self.client.get_cluster_info()
            return {
                "status": "healthy",
                "cluster_info": cluster_info.dict() if hasattr(cluster_info, 'dict') else str(cluster_info)
            }
        except Exception as e:
            logger.error(f"獲取集群資訊失敗: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def get_collections_info(self) -> Dict[str, Any]:
        """獲取所有 collection 的詳細資訊"""
        try:
            collections = await self.client.get_collections()
            collections_data = []
            total_points = 0
            total_vectors = 0
            
            for collection in collections.collections:
                try:
                    collection_info = await self.client.get_collection(collection.name)
                    
                    # 安全地獲取向量配置
                    vector_size = 0
                    try:
                        if hasattr(collection_info.config.params, 'vectors'):
                            vectors_config = collection_info.config.params.vectors
                            
                            # 處理不同的向量配置格式
                            if hasattr(vectors_config, 'size'):
                                # 簡單向量配置
                                vector_size = vectors_config.size
                            elif isinstance(vectors_config, dict):
                                # 字典格式的向量配置
                                if 'size' in vectors_config:
                                    vector_size = vectors_config['size']
                                else:
                                    # 命名向量配置，取第一個向量的維度
                                    for vector_name, vector_config in vectors_config.items():
                                        if isinstance(vector_config, dict) and 'size' in vector_config:
                                            vector_size = vector_config['size']
                                            break
                                        elif hasattr(vector_config, 'size'):
                                            vector_size = vector_config.size
                                            break
                            else:
                                logger.debug(f"未知的向量配置格式 {collection.name}: {type(vectors_config)}")
                    except Exception as e:
                        logger.debug(f"無法獲取向量維度 {collection.name}: {e}")
                        vector_size = 0
                    
                    points_count = collection_info.points_count or 0
                    vectors_count = collection_info.vectors_count or 0
                    
                    # 簡單的記憶體估算：向量數 × 向量維度 × 4 bytes (float32)
                    estimated_memory_mb = (vectors_count * vector_size * 4) / (1024 * 1024) if vector_size > 0 else 0
                    
                    collection_data = {
                        "name": collection.name,
                        "points_count": points_count,
                        "vectors_count": vectors_count,
                        "vector_size": vector_size,
                        "estimated_memory_mb": round(estimated_memory_mb, 2),
                        "status": collection_info.status.value if hasattr(collection_info, 'status') else "unknown",
                        "optimizer_status": collection_info.optimizer_status.dict() if hasattr(collection_info, 'optimizer_status') else {},
                        "indexed_vectors_count": collection_info.indexed_vectors_count or 0
                    }
                    
                    collections_data.append(collection_data)
                    total_points += points_count
                    total_vectors += vectors_count
                    
                except Exception as e:
                    logger.warning(f"獲取 collection {collection.name} 資訊失敗: {e}")
                    # 添加錯誤的 collection 資料
                    collections_data.append({
                        "name": collection.name,
                        "error": str(e),
                        "points_count": 0,
                        "vectors_count": 0,
                        "vector_size": 0,
                        "estimated_memory_mb": 0,
                        "status": "error",
                        "indexed_vectors_count": 0
                    })
            
            return {
                "collections": collections_data,
                "summary": {
                    "total_collections": len(collections_data),
                    "total_points": total_points,
                    "total_vectors": total_vectors,
                    "total_estimated_memory_mb": sum(c["estimated_memory_mb"] for c in collections_data)
                }
            }
        except Exception as e:
            logger.error(f"獲取 collection 資訊失敗: {e}")
            return {
                "error": str(e),
                "collections": [],
                "summary": {"total_collections": 0, "total_points": 0, "total_vectors": 0}
            }
    
    async def get_health_status(self) -> Dict[str, Any]:
        """獲取 Qdrant 健康狀態"""
        try:
            # 嘗試簡單的健康檢查
            collections = await self.client.get_collections()
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "collections_accessible": True,
                "collections_count": len(collections.collections)
            }
        except Exception as e:
            logger.error(f"健康檢查失敗: {e}")
            return {
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "collections_accessible": False
            }
    
    async def get_comprehensive_analysis(self) -> Dict[str, Any]:
        """獲取綜合分析報告（純 Qdrant API 版本）"""
        try:
            # 並行獲取各種資訊
            health_task = self.get_health_status()
            collections_task = self.get_collections_info()
            cluster_task = self.get_cluster_info()
            
            health_status, collections_info, cluster_info = await asyncio.gather(
                health_task, collections_task, cluster_task, return_exceptions=True
            )
            
            # 處理可能的異常
            if isinstance(health_status, Exception):
                health_status = {"status": "error", "error": str(health_status)}
            if isinstance(collections_info, Exception):
                collections_info = {"error": str(collections_info), "collections": []}
            if isinstance(cluster_info, Exception):
                cluster_info = {"status": "error", "error": str(cluster_info)}
            
            # 生成性能評估
            performance_analysis = self._analyze_performance(collections_info)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "deployment_info": self.deployment_info,
                "health_status": health_status,
                "collections_info": collections_info,
                "cluster_info": cluster_info,
                "performance_analysis": performance_analysis,
                "monitoring_scope": "qdrant_api_only",
                "limitations": [
                    "僅基於 Qdrant API 的資訊",
                    "無系統級資源監控",
                    "無容器級監控",
                    "記憶體估算為理論值"
                ]
            }
        except Exception as e:
            logger.error(f"綜合分析失敗: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": str(e),
                "monitoring_scope": "qdrant_api_only"
            }
    
    def _analyze_performance(self, collections_info: Dict[str, Any]) -> Dict[str, Any]:
        """分析性能（基於 Qdrant 資料）"""
        if "error" in collections_info:
            return {"status": "error", "error": collections_info["error"]}
        
        summary = collections_info.get("summary", {})
        collections = collections_info.get("collections", [])
        
        # 基於 Qdrant 資料的性能分析
        total_memory = summary.get("total_estimated_memory_mb", 0)
        total_vectors = summary.get("total_vectors", 0)
        total_collections = summary.get("total_collections", 0)
        
        # 性能評估
        performance_score = "excellent"
        recommendations = []
        
        if total_memory > 1000:  # 1GB
            performance_score = "good"
            recommendations.append("考慮監控記憶體使用趨勢")
        
        if total_memory > 5000:  # 5GB
            performance_score = "fair"
            recommendations.append("建議關注記憶體最佳化")
        
        if total_collections > 10:
            recommendations.append("多個 collection 可能影響查詢效能")
        
        # 分析每個 collection 的索引狀態
        indexing_issues = []
        for collection in collections:
            indexed_ratio = 0
            if collection.get("vectors_count", 0) > 0:
                indexed_ratio = collection.get("indexed_vectors_count", 0) / collection["vectors_count"]
            
            if indexed_ratio < 0.9:  # 索引率低於 90%
                indexing_issues.append({
                    "collection": collection["name"],
                    "indexed_ratio": round(indexed_ratio, 2),
                    "issue": "索引率偏低"
                })
        
        return {
            "overall_score": performance_score,
            "total_estimated_memory_mb": total_memory,
            "total_vectors": total_vectors,
            "recommendations": recommendations,
            "indexing_issues": indexing_issues,
            "collection_analysis": [
                {
                    "name": c["name"],
                    "efficiency": "good" if c.get("estimated_memory_mb", 0) < 500 else "fair",
                    "indexed_ratio": round(c.get("indexed_vectors_count", 0) / max(c.get("vectors_count", 1), 1), 2)
                }
                for c in collections
            ]
        }


# 為了向後相容，保留原來的類名但重新實現
class UniversalQdrantMonitor(QdrantOnlyMonitor):
    """向後相容的類名 - 重定向到純 Qdrant API 監控器"""
    pass
