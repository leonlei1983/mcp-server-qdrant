"""
資料遷移工具
安全地將舊的 Qdrant collections 遷移到新的 RAG Bridge 分片架構
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, Range

from mcp_server_qdrant.ragbridge.connector import RAGBridgeConnector
from mcp_server_qdrant.ragbridge.models import ContentType, RAGEntry, RAGMetadata
from mcp_server_qdrant.ragbridge.vocabulary import VocabularyManager, FragmentType
from mcp_server_qdrant.embeddings.factory import create_embedding_provider

logger = logging.getLogger(__name__)


@dataclass
class MigrationPlan:
    """遷移計劃"""
    source_collection: str
    target_content_type: ContentType
    estimated_records: int
    mapping_rules: Dict[str, str]  # 舊欄位 -> 新欄位的映射規則
    transformation_rules: Dict[str, Any]  # 資料轉換規則
    validation_rules: Dict[str, Any]  # 驗證規則
    notes: str = ""


@dataclass
class MigrationResult:
    """遷移結果"""
    plan: MigrationPlan
    total_records: int
    successful_records: int
    failed_records: int
    errors: List[str]
    start_time: datetime
    end_time: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        if self.total_records == 0:
            return 0.0
        return self.successful_records / self.total_records
    
    @property
    def duration_seconds(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()


class DataMigrationTool:
    """資料遷移工具"""
    
    def __init__(
        self,
        qdrant_client: QdrantClient,
        ragbridge_connector: RAGBridgeConnector,
        vocabulary_manager: VocabularyManager,
        backup_dir: str = "./migration_backups"
    ):
        self.qdrant_client = qdrant_client
        self.ragbridge_connector = ragbridge_connector
        self.vocabulary_manager = vocabulary_manager
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
    
    def analyze_collection_structure(self, collection_name: str) -> Dict[str, Any]:
        """分析舊 collection 的結構"""
        try:
            # 獲取 collection 資訊
            collection_info = self.qdrant_client.get_collection(collection_name)
            
            # 取樣一些點來分析資料結構
            sample_points = self.qdrant_client.scroll(
                collection_name=collection_name,
                limit=100,
                with_payload=True,
                with_vectors=False
            )[0]
            
            # 分析 payload 結構
            field_types = {}
            field_samples = {}
            
            for point in sample_points:
                if point.payload:
                    for field, value in point.payload.items():
                        value_type = type(value).__name__
                        if field not in field_types:
                            field_types[field] = set()
                            field_samples[field] = []
                        
                        field_types[field].add(value_type)
                        if len(field_samples[field]) < 5:
                            field_samples[field].append(value)
            
            # 轉換 set 為 list 以便 JSON 序列化
            field_types = {k: list(v) for k, v in field_types.items()}
            
            analysis = {
                "collection_name": collection_name,
                "total_points": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance.name,
                "field_types": field_types,
                "field_samples": field_samples,
                "analyzed_at": datetime.now().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze collection {collection_name}: {e}")
            raise
    
    def suggest_migration_plan(self, analysis: Dict[str, Any]) -> MigrationPlan:
        """基於分析結果建議遷移計劃"""
        collection_name = analysis["collection_name"]
        field_types = analysis["field_types"]
        
        # 基於欄位分析猜測內容類型
        suggested_content_type = self._guess_content_type(field_types)
        
        # 建立欄位映射規則
        mapping_rules = self._create_field_mapping(field_types)
        
        # 建立轉換規則
        transformation_rules = {
            "normalize_tags": True,
            "extract_keywords": True,
            "validate_content_length": True,
            "standardize_vocabulary": True
        }
        
        # 建立驗證規則
        validation_rules = {
            "min_content_length": 10,
            "max_content_length": 50000,
            "required_fields": ["content"],
            "valid_fragment_types": [ft.value for ft in FragmentType]
        }
        
        plan = MigrationPlan(
            source_collection=collection_name,
            target_content_type=suggested_content_type,
            estimated_records=analysis["total_points"],
            mapping_rules=mapping_rules,
            transformation_rules=transformation_rules,
            validation_rules=validation_rules,
            notes=f"Auto-generated plan based on collection analysis"
        )
        
        return plan
    
    def _guess_content_type(self, field_types: Dict[str, List[str]]) -> ContentType:
        """基於欄位猜測內容類型"""
        fields = set(field_types.keys())
        
        # 檢查是否包含流程相關欄位
        process_keywords = {"step", "workflow", "process", "procedure", "task"}
        if any(keyword in field.lower() for field in fields for keyword in process_keywords):
            return ContentType.PROCESS_WORKFLOW
        
        # 檢查是否包含經驗相關欄位
        experience_keywords = {"experience", "lesson", "practice", "tip", "solution"}
        if any(keyword in field.lower() for field in fields for keyword in experience_keywords):
            return ContentType.EXPERIENCE
        
        # 檢查是否包含知識庫相關欄位
        knowledge_keywords = {"knowledge", "reference", "documentation", "guide"}
        if any(keyword in field.lower() for field in fields for keyword in knowledge_keywords):
            return ContentType.KNOWLEDGE_BASE
        
        # 預設為經驗類型
        return ContentType.EXPERIENCE
    
    def _create_field_mapping(self, field_types: Dict[str, List[str]]) -> Dict[str, str]:
        """建立欄位映射規則"""
        mapping = {}
        
        # 常見欄位映射
        field_mappings = {
            "text": "content",
            "content": "content",
            "description": "content",
            "body": "content",
            "message": "content",
            "title": "title",
            "name": "title",
            "subject": "title",
            "tag": "tags",
            "tags": "tags",
            "label": "tags",
            "labels": "tags",
            "category": "tags",
            "categories": "tags",
            "source": "source",
            "author": "author",
            "creator": "author",
            "timestamp": "created_at",
            "created_at": "created_at",
            "date": "created_at",
            "id": "original_id"
        }
        
        for old_field in field_types.keys():
            old_field_lower = old_field.lower()
            for pattern, new_field in field_mappings.items():
                if pattern in old_field_lower:
                    mapping[old_field] = new_field
                    break
            else:
                # 如果沒有找到映射，保留原始欄位名
                mapping[old_field] = old_field
        
        return mapping
    
    async def create_backup(self, collection_name: str) -> str:
        """創建 collection 備份"""
        backup_filename = f"backup_{collection_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        backup_path = self.backup_dir / backup_filename
        
        try:
            # 獲取所有點
            all_points = []
            offset = None
            
            while True:
                points, next_offset = self.qdrant_client.scroll(
                    collection_name=collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True
                )
                
                if not points:
                    break
                
                # 轉換為可序列化格式
                for point in points:
                    point_data = {
                        "id": str(point.id),
                        "vector": point.vector,
                        "payload": point.payload
                    }
                    all_points.append(point_data)
                
                offset = next_offset
                if next_offset is None:
                    break
            
            # 寫入備份檔案
            backup_data = {
                "collection_name": collection_name,
                "backup_created_at": datetime.now().isoformat(),
                "total_points": len(all_points),
                "points": all_points
            }
            
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Created backup for {collection_name}: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Failed to create backup for {collection_name}: {e}")
            raise
    
    def validate_migration_plan(self, plan: MigrationPlan) -> List[str]:
        """驗證遷移計劃"""
        errors = []
        
        # 檢查來源 collection 是否存在
        try:
            self.qdrant_client.get_collection(plan.source_collection)
        except Exception:
            errors.append(f"Source collection '{plan.source_collection}' does not exist")
        
        # 檢查目標內容類型是否有效
        if plan.target_content_type not in ContentType:
            errors.append(f"Invalid target content type: {plan.target_content_type}")
        
        # 檢查映射規則
        if not plan.mapping_rules:
            errors.append("No field mapping rules defined")
        
        # 檢查是否有 content 欄位映射
        has_content_mapping = any(
            target == "content" for target in plan.mapping_rules.values()
        )
        if not has_content_mapping:
            errors.append("No mapping to 'content' field found - this is required")
        
        return errors
    
    async def execute_migration(
        self, 
        plan: MigrationPlan, 
        dry_run: bool = False,
        batch_size: int = 100
    ) -> MigrationResult:
        """執行遷移"""
        result = MigrationResult(
            plan=plan,
            total_records=0,
            successful_records=0,
            failed_records=0,
            errors=[],
            start_time=datetime.now()
        )
        
        try:
            # 驗證計劃
            validation_errors = self.validate_migration_plan(plan)
            if validation_errors:
                result.errors.extend(validation_errors)
                result.end_time = datetime.now()
                return result
            
            # 如果不是 dry run，創建備份
            if not dry_run:
                backup_path = await self.create_backup(plan.source_collection)
                logger.info(f"Backup created: {backup_path}")
            
            # 初始化 RAG Bridge connector 的目標 collection
            target_collection = f"ragbridge_{plan.target_content_type.value}"
            
            # 分批處理資料
            offset = None
            batch_count = 0
            
            while True:
                # 獲取一批資料
                points, next_offset = self.qdrant_client.scroll(
                    collection_name=plan.source_collection,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True
                )
                
                if not points:
                    break
                
                batch_count += 1
                logger.info(f"Processing batch {batch_count}, {len(points)} points")
                
                # 處理這批資料
                for point in points:
                    result.total_records += 1
                    
                    try:
                        # 轉換資料格式
                        rag_entry = await self._transform_point_to_rag_entry(
                            point, plan
                        )
                        
                        # 驗證轉換後的資料
                        if not self._validate_rag_entry(rag_entry, plan.validation_rules):
                            result.failed_records += 1
                            result.errors.append(f"Validation failed for point {point.id}")
                            continue
                        
                        # 如果不是 dry run，實際儲存資料
                        if not dry_run:
                            await self.ragbridge_connector.store_experience(rag_entry)
                        
                        result.successful_records += 1
                        
                    except Exception as e:
                        result.failed_records += 1
                        error_msg = f"Failed to process point {point.id}: {str(e)}"
                        result.errors.append(error_msg)
                        logger.warning(error_msg)
                
                offset = next_offset
                if next_offset is None:
                    break
            
            result.end_time = datetime.now()
            
            success_rate = result.success_rate
            logger.info(
                f"Migration {'(DRY RUN) ' if dry_run else ''}completed: "
                f"{result.successful_records}/{result.total_records} "
                f"({success_rate:.1%}) successful"
            )
            
            return result
            
        except Exception as e:
            result.end_time = datetime.now()
            result.errors.append(f"Migration failed: {str(e)}")
            logger.error(f"Migration execution failed: {e}")
            return result
    
    async def _transform_point_to_rag_entry(
        self, 
        point, 
        plan: MigrationPlan
    ) -> RAGEntry:
        """將舊格式的點轉換為 RAG Entry"""
        payload = point.payload or {}
        
        # 應用欄位映射
        mapped_data = {}
        for old_field, new_field in plan.mapping_rules.items():
            if old_field in payload:
                mapped_data[new_field] = payload[old_field]
        
        # 確保必要欄位存在
        content = mapped_data.get("content", "")
        if not content and "title" in mapped_data:
            content = mapped_data["title"]
        
        title = mapped_data.get("title", f"Migrated from {plan.source_collection}")
        
        # 處理標籤
        tags = mapped_data.get("tags", [])
        if isinstance(tags, str):
            tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
        elif not isinstance(tags, list):
            tags = [str(tags)]
        
        # 標準化詞彙（如果啟用）
        if plan.transformation_rules.get("standardize_vocabulary", False):
            try:
                standardized = await self.vocabulary_manager.standardize_content(
                    content, tags
                )
                if standardized.get("standardized_tags"):
                    tags = standardized["standardized_tags"]
            except Exception as e:
                logger.warning(f"Vocabulary standardization failed: {e}")
        
        # 建立元數據
        metadata = RAGMetadata(
            content_type=plan.target_content_type,
            title=title,
            tags=tags,
            source=mapped_data.get("source", plan.source_collection),
            author=mapped_data.get("author", "migrated"),
            created_at=mapped_data.get("created_at"),
            additional_metadata={
                "migration_source": plan.source_collection,
                "original_id": str(point.id),
                "migrated_at": datetime.now().isoformat(),
                **{k: v for k, v in mapped_data.items() 
                   if k not in ["content", "title", "tags", "source", "author", "created_at"]}
            }
        )
        
        # 建立 RAG Entry
        rag_entry = RAGEntry(
            content=content,
            metadata=metadata,
            embedding=point.vector  # 重用現有的向量
        )
        
        return rag_entry
    
    def _validate_rag_entry(
        self, 
        rag_entry: RAGEntry, 
        validation_rules: Dict[str, Any]
    ) -> bool:
        """驗證 RAG Entry"""
        try:
            # 檢查內容長度
            min_length = validation_rules.get("min_content_length", 0)
            max_length = validation_rules.get("max_content_length", 100000)
            
            if len(rag_entry.content) < min_length:
                return False
            if len(rag_entry.content) > max_length:
                return False
            
            # 檢查必要欄位
            required_fields = validation_rules.get("required_fields", [])
            for field in required_fields:
                if field == "content" and not rag_entry.content.strip():
                    return False
                elif field == "title" and not rag_entry.metadata.title.strip():
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Validation error: {e}")
            return False
    
    def generate_migration_report(self, result: MigrationResult) -> Dict[str, Any]:
        """生成遷移報告"""
        report = {
            "migration_summary": {
                "source_collection": result.plan.source_collection,
                "target_content_type": result.plan.target_content_type.value,
                "total_records": result.total_records,
                "successful_records": result.successful_records,
                "failed_records": result.failed_records,
                "success_rate": f"{result.success_rate:.1%}",
                "duration_seconds": result.duration_seconds,
                "start_time": result.start_time.isoformat(),
                "end_time": result.end_time.isoformat() if result.end_time else None
            },
            "migration_plan": asdict(result.plan),
            "errors": result.errors[:50],  # 只顯示前50個錯誤
            "total_errors": len(result.errors),
            "recommendations": self._generate_recommendations(result)
        }
        
        return report
    
    def _generate_recommendations(self, result: MigrationResult) -> List[str]:
        """基於遷移結果生成建議"""
        recommendations = []
        
        if result.success_rate < 0.8:
            recommendations.append("成功率低於80%，建議檢查映射規則和驗證邏輯")
        
        if result.failed_records > 0:
            recommendations.append("有失敗記錄，建議檢查錯誤日誌並調整轉換規則")
        
        if "content" in str(result.errors):
            recommendations.append("部分記錄缺少內容欄位，建議調整欄位映射規則")
        
        if result.success_rate == 1.0:
            recommendations.append("遷移成功！可以考慮移除原始 collection")
        
        return recommendations