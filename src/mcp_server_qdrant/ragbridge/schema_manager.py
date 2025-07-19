"""
動態 Metadata Schema 管理系統

此模組負責：
1. Schema 版本控制機制
2. 動態欄位擴展系統
3. Schema 演進 API
4. 自動檢測和建議功能
5. 數據遷移策略
"""
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class FieldType(str, Enum):
    """欄位類型"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    LIST = "list"
    DICT = "dict"
    JSON = "json"


class FieldValidation(BaseModel):
    """欄位驗證規則"""
    required: bool = False
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    default_value: Optional[Any] = None


class SchemaField(BaseModel):
    """Schema 欄位定義"""
    name: str = Field(..., description="欄位名稱")
    field_type: FieldType = Field(..., description="欄位類型")
    description: str = Field(default="", description="欄位描述")
    validation: FieldValidation = Field(default_factory=FieldValidation, description="驗證規則")
    is_core: bool = Field(default=False, description="是否為核心欄位")
    deprecated: bool = Field(default=False, description="是否已棄用")
    added_in_version: str = Field(..., description="新增於版本")
    deprecated_in_version: Optional[str] = Field(None, description="棄用於版本")
    migration_mapping: Optional[str] = Field(None, description="遷移映射規則")
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        """驗證欄位名稱"""
        import re
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', v):
            raise ValueError("欄位名稱必須是有效的識別符")
        return v


class SchemaVersion(BaseModel):
    """Schema 版本"""
    version: str = Field(..., description="版本號 (semantic versioning)")
    description: str = Field(default="", description="版本描述")
    created_at: datetime = Field(default_factory=datetime.now, description="建立時間")
    is_active: bool = Field(default=True, description="是否為活躍版本")
    backward_compatible: bool = Field(default=True, description="是否向後兼容")
    fields: Dict[str, SchemaField] = Field(default_factory=dict, description="欄位定義")
    migration_notes: str = Field(default="", description="遷移說明")
    
    @field_validator('version')
    @classmethod
    def validate_version(cls, v):
        """驗證版本號格式"""
        import re
        if not re.match(r'^\d+\.\d+\.\d+$', v):
            raise ValueError("版本號必須符合 semantic versioning 格式 (x.y.z)")
        return v


class SchemaMigration(BaseModel):
    """Schema 遷移規則"""
    from_version: str = Field(..., description="源版本")
    to_version: str = Field(..., description="目標版本")
    migration_type: str = Field(..., description="遷移類型: add_field, remove_field, modify_field, rename_field")
    field_name: str = Field(..., description="相關欄位名稱")
    migration_script: str = Field(default="", description="遷移腳本")
    rollback_script: str = Field(default="", description="回滾腳本")
    created_at: datetime = Field(default_factory=datetime.now, description="建立時間")


class SchemaEvolutionProposal(BaseModel):
    """Schema 演進提案"""
    proposal_id: str = Field(..., description="提案ID")
    title: str = Field(..., description="提案標題")
    description: str = Field(..., description="提案描述")
    proposed_changes: List[Dict[str, Any]] = Field(default_factory=list, description="提議的變更")
    impact_analysis: Dict[str, Any] = Field(default_factory=dict, description="影響分析")
    status: str = Field(default="pending", description="狀態: pending, approved, rejected")
    created_by: str = Field(default="system", description="提案者")
    created_at: datetime = Field(default_factory=datetime.now, description="建立時間")
    approved_at: Optional[datetime] = Field(None, description="核准時間")


class DynamicSchemaManager:
    """動態 Schema 管理器"""
    
    def __init__(self, schema_storage_path: Optional[str] = None):
        self.schema_storage_path = Path(schema_storage_path) if schema_storage_path else Path("./schema_storage")
        self.schema_storage_path.mkdir(exist_ok=True)
        
        # 內存緩存
        self.schemas: Dict[str, SchemaVersion] = {}
        self.migrations: List[SchemaMigration] = []
        self.proposals: Dict[str, SchemaEvolutionProposal] = {}
        
        # 核心欄位定義（不可刪除）
        self.core_fields = {
            "content_id": SchemaField(
                name="content_id",
                field_type=FieldType.STRING,
                description="內容唯一識別符",
                validation=FieldValidation(required=True),
                is_core=True,
                added_in_version="1.0.0"
            ),
            "title": SchemaField(
                name="title",
                field_type=FieldType.STRING,
                description="內容標題",
                validation=FieldValidation(required=True, max_length=200),
                is_core=True,
                added_in_version="1.0.0"
            ),
            "content_type": SchemaField(
                name="content_type",
                field_type=FieldType.STRING,
                description="內容類型",
                validation=FieldValidation(required=True),
                is_core=True,
                added_in_version="1.0.0"
            ),
            "created_at": SchemaField(
                name="created_at",
                field_type=FieldType.DATETIME,
                description="建立時間",
                validation=FieldValidation(required=True),
                is_core=True,
                added_in_version="1.0.0"
            ),
            "updated_at": SchemaField(
                name="updated_at",
                field_type=FieldType.DATETIME,
                description="更新時間",
                validation=FieldValidation(required=True),
                is_core=True,
                added_in_version="1.0.0"
            )
        }
        
        self._load_schemas()
        self._initialize_base_schema()
    
    def _load_schemas(self):
        """載入儲存的 Schema"""
        try:
            schema_file = self.schema_storage_path / "schemas.json"
            if schema_file.exists():
                with open(schema_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for version, schema_data in data.get("schemas", {}).items():
                        self.schemas[version] = SchemaVersion(**schema_data)
            
            migration_file = self.schema_storage_path / "migrations.json"
            if migration_file.exists():
                with open(migration_file, 'r', encoding='utf-8') as f:
                    migration_data = json.load(f)
                    self.migrations = [SchemaMigration(**m) for m in migration_data.get("migrations", [])]
            
            logger.info(f"載入 {len(self.schemas)} 個 schema 版本和 {len(self.migrations)} 個遷移規則")
        except Exception as e:
            logger.error(f"載入 schema 失敗: {e}")
    
    def _save_schemas(self):
        """儲存 Schema"""
        try:
            # 儲存 schemas
            schema_file = self.schema_storage_path / "schemas.json"
            with open(schema_file, 'w', encoding='utf-8') as f:
                schema_data = {
                    "schemas": {
                        version: schema.model_dump(mode='python')
                        for version, schema in self.schemas.items()
                    }
                }
                json.dump(schema_data, f, ensure_ascii=False, indent=2, default=str)
            
            # 儲存 migrations
            migration_file = self.schema_storage_path / "migrations.json"
            with open(migration_file, 'w', encoding='utf-8') as f:
                migration_data = {
                    "migrations": [
                        migration.model_dump(mode='python')
                        for migration in self.migrations
                    ]
                }
                json.dump(migration_data, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info("Schema 資料儲存成功")
        except Exception as e:
            logger.error(f"儲存 schema 失敗: {e}")
    
    def _initialize_base_schema(self):
        """初始化基礎 Schema"""
        if "1.0.0" not in self.schemas:
            base_schema = SchemaVersion(
                version="1.0.0",
                description="基礎 Schema 版本，包含核心欄位",
                fields=self.core_fields.copy()
            )
            self.schemas["1.0.0"] = base_schema
            self._save_schemas()
            logger.info("初始化基礎 Schema 1.0.0")
    
    def get_current_schema(self) -> SchemaVersion:
        """獲取當前活躍的 Schema 版本"""
        # 找到最新的活躍版本
        active_schemas = {v: s for v, s in self.schemas.items() if s.is_active}
        if not active_schemas:
            return self.schemas["1.0.0"]
        
        # 按版本號排序，取最新版本
        sorted_versions = sorted(active_schemas.keys(), key=self._parse_version, reverse=True)
        return active_schemas[sorted_versions[0]]
    
    def _parse_version(self, version: str) -> Tuple[int, int, int]:
        """解析版本號"""
        parts = version.split('.')
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    
    def add_field(self, field: SchemaField, target_version: Optional[str] = None) -> bool:
        """新增欄位到 Schema"""
        try:
            if target_version is None:
                # 創建新版本
                current = self.get_current_schema()
                new_version = self._increment_version(current.version, "minor")
                target_version = new_version
            
            if target_version not in self.schemas:
                # 創建新版本
                current = self.get_current_schema()
                new_schema = SchemaVersion(
                    version=target_version,
                    description=f"新增欄位 {field.name}",
                    fields=current.fields.copy()
                )
                self.schemas[target_version] = new_schema
            
            # 檢查欄位是否已存在
            if field.name in self.schemas[target_version].fields:
                logger.warning(f"欄位 {field.name} 已存在於版本 {target_version}")
                return False
            
            # 新增欄位
            field.added_in_version = target_version
            self.schemas[target_version].fields[field.name] = field
            
            # 創建遷移規則
            if target_version != "1.0.0":
                migration = SchemaMigration(
                    from_version=self.get_current_schema().version,
                    to_version=target_version,
                    migration_type="add_field",
                    field_name=field.name,
                    migration_script=f"ADD FIELD {field.name} {field.field_type.value}"
                )
                self.migrations.append(migration)
            
            self._save_schemas()
            logger.info(f"成功新增欄位 {field.name} 到版本 {target_version}")
            return True
            
        except Exception as e:
            logger.error(f"新增欄位失敗: {e}")
            return False
    
    def remove_field(self, field_name: str, target_version: Optional[str] = None) -> bool:
        """移除欄位（軟刪除，標記為棄用）"""
        try:
            # 檢查是否為核心欄位
            if field_name in self.core_fields:
                logger.error(f"無法移除核心欄位: {field_name}")
                return False
            
            if target_version is None:
                current = self.get_current_schema()
                target_version = self._increment_version(current.version, "minor")
            
            if target_version not in self.schemas:
                current = self.get_current_schema()
                new_schema = SchemaVersion(
                    version=target_version,
                    description=f"棄用欄位 {field_name}",
                    fields=current.fields.copy()
                )
                self.schemas[target_version] = new_schema
            
            # 檢查欄位是否存在
            if field_name not in self.schemas[target_version].fields:
                logger.warning(f"欄位 {field_name} 不存在")
                return False
            
            # 標記為棄用而非直接刪除
            self.schemas[target_version].fields[field_name].deprecated = True
            self.schemas[target_version].fields[field_name].deprecated_in_version = target_version
            
            # 創建遷移規則
            migration = SchemaMigration(
                from_version=self.get_current_schema().version,
                to_version=target_version,
                migration_type="remove_field",
                field_name=field_name,
                migration_script=f"DEPRECATE FIELD {field_name}"
            )
            self.migrations.append(migration)
            
            self._save_schemas()
            logger.info(f"成功棄用欄位 {field_name} 於版本 {target_version}")
            return True
            
        except Exception as e:
            logger.error(f"移除欄位失敗: {e}")
            return False
    
    def modify_field(self, field_name: str, new_field: SchemaField, target_version: Optional[str] = None) -> bool:
        """修改欄位定義"""
        try:
            if target_version is None:
                current = self.get_current_schema()
                target_version = self._increment_version(current.version, "patch")
            
            if target_version not in self.schemas:
                current = self.get_current_schema()
                new_schema = SchemaVersion(
                    version=target_version,
                    description=f"修改欄位 {field_name}",
                    fields=current.fields.copy()
                )
                self.schemas[target_version] = new_schema
            
            if field_name not in self.schemas[target_version].fields:
                logger.warning(f"欄位 {field_name} 不存在")
                return False
            
            # 保留原始的新增版本資訊
            original_field = self.schemas[target_version].fields[field_name]
            new_field.added_in_version = original_field.added_in_version
            new_field.name = field_name  # 確保名稱一致
            
            # 更新欄位
            self.schemas[target_version].fields[field_name] = new_field
            
            # 創建遷移規則
            migration = SchemaMigration(
                from_version=self.get_current_schema().version,
                to_version=target_version,
                migration_type="modify_field",
                field_name=field_name,
                migration_script=f"MODIFY FIELD {field_name}"
            )
            self.migrations.append(migration)
            
            self._save_schemas()
            logger.info(f"成功修改欄位 {field_name} 於版本 {target_version}")
            return True
            
        except Exception as e:
            logger.error(f"修改欄位失敗: {e}")
            return False
    
    def _increment_version(self, version: str, increment_type: str) -> str:
        """增加版本號"""
        major, minor, patch = self._parse_version(version)
        
        if increment_type == "major":
            return f"{major + 1}.0.0"
        elif increment_type == "minor":
            return f"{major}.{minor + 1}.0"
        elif increment_type == "patch":
            return f"{major}.{minor}.{patch + 1}"
        else:
            raise ValueError(f"無效的版本增加類型: {increment_type}")
    
    def validate_data(self, data: Dict[str, Any], schema_version: Optional[str] = None) -> Tuple[bool, List[str]]:
        """驗證數據是否符合 Schema"""
        if schema_version is None:
            schema = self.get_current_schema()
        else:
            schema = self.schemas.get(schema_version)
            if not schema:
                return False, [f"Schema 版本 {schema_version} 不存在"]
        
        errors = []
        
        # 檢查必填欄位
        for field_name, field in schema.fields.items():
            if field.deprecated:
                continue
                
            if field.validation.required and field_name not in data:
                errors.append(f"缺少必填欄位: {field_name}")
                continue
            
            if field_name in data:
                value = data[field_name]
                field_errors = self._validate_field_value(field, value)
                errors.extend(field_errors)
        
        # 檢查未知欄位
        for field_name in data:
            if field_name not in schema.fields:
                errors.append(f"未知欄位: {field_name}")
        
        return len(errors) == 0, errors
    
    def _validate_field_value(self, field: SchemaField, value: Any) -> List[str]:
        """驗證欄位值"""
        errors = []
        validation = field.validation
        
        # 類型檢查
        expected_type = self._get_python_type(field.field_type)
        if not isinstance(value, expected_type):
            errors.append(f"欄位 {field.name} 類型錯誤，期望 {field.field_type.value}，實際 {type(value).__name__}")
            return errors  # 類型錯誤就不繼續檢查其他規則
        
        # 字串驗證
        if field.field_type == FieldType.STRING and isinstance(value, str):
            if validation.min_length and len(value) < validation.min_length:
                errors.append(f"欄位 {field.name} 長度不足，最小 {validation.min_length}")
            if validation.max_length and len(value) > validation.max_length:
                errors.append(f"欄位 {field.name} 長度超限，最大 {validation.max_length}")
            if validation.pattern:
                import re
                if not re.match(validation.pattern, value):
                    errors.append(f"欄位 {field.name} 格式不符合規則: {validation.pattern}")
        
        # 數值驗證
        if field.field_type in [FieldType.INTEGER, FieldType.FLOAT] and isinstance(value, (int, float)):
            if validation.min_value is not None and value < validation.min_value:
                errors.append(f"欄位 {field.name} 值過小，最小 {validation.min_value}")
            if validation.max_value is not None and value > validation.max_value:
                errors.append(f"欄位 {field.name} 值過大，最大 {validation.max_value}")
        
        # 允許值檢查
        if validation.allowed_values and value not in validation.allowed_values:
            errors.append(f"欄位 {field.name} 值不在允許範圍內: {validation.allowed_values}")
        
        return errors
    
    def _get_python_type(self, field_type: FieldType):
        """獲取 Python 類型"""
        type_mapping = {
            FieldType.STRING: str,
            FieldType.INTEGER: int,
            FieldType.FLOAT: float,
            FieldType.BOOLEAN: bool,
            FieldType.DATETIME: datetime,
            FieldType.LIST: list,
            FieldType.DICT: dict,
            FieldType.JSON: (dict, list, str)
        }
        return type_mapping.get(field_type, str)
    
    def analyze_schema_usage(self, data_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析 Schema 使用情況"""
        if not data_samples:
            return {"error": "沒有數據樣本"}
        
        current_schema = self.get_current_schema()
        
        # 統計欄位使用率
        field_usage = {}
        missing_fields = {}
        unknown_fields = set()
        
        for sample in data_samples:
            for field_name in current_schema.fields:
                if field_name in sample:
                    field_usage[field_name] = field_usage.get(field_name, 0) + 1
                else:
                    missing_fields[field_name] = missing_fields.get(field_name, 0) + 1
            
            for field_name in sample:
                if field_name not in current_schema.fields:
                    unknown_fields.add(field_name)
        
        total_samples = len(data_samples)
        
        # 計算使用率
        usage_stats = {
            field_name: {
                "usage_count": field_usage.get(field_name, 0),
                "usage_rate": field_usage.get(field_name, 0) / total_samples,
                "missing_count": missing_fields.get(field_name, 0),
                "is_required": current_schema.fields[field_name].validation.required,
                "is_core": current_schema.fields[field_name].is_core
            }
            for field_name in current_schema.fields
        }
        
        return {
            "total_samples": total_samples,
            "current_schema_version": current_schema.version,
            "field_usage_stats": usage_stats,
            "unknown_fields": list(unknown_fields),
            "schema_compliance_rate": len([f for f in usage_stats.values() if f["usage_rate"] > 0.8]) / len(usage_stats) if usage_stats else 0
        }
    
    def suggest_schema_improvements(self, usage_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """根據使用情況建議 Schema 改進"""
        suggestions = []
        
        if "field_usage_stats" not in usage_analysis:
            return suggestions
        
        stats = usage_analysis["field_usage_stats"]
        unknown_fields = usage_analysis.get("unknown_fields", [])
        
        # 建議新增常用的未知欄位
        for field_name in unknown_fields:
            suggestions.append({
                "type": "add_field",
                "field_name": field_name,
                "reason": "發現未定義但常用的欄位",
                "priority": "medium"
            })
        
        # 建議移除很少使用的非核心欄位
        for field_name, stat in stats.items():
            if not stat["is_core"] and not stat["is_required"] and stat["usage_rate"] < 0.1:
                suggestions.append({
                    "type": "deprecate_field",
                    "field_name": field_name,
                    "reason": f"使用率過低 ({stat['usage_rate']:.1%})",
                    "priority": "low"
                })
        
        # 建議將高使用率的可選欄位設為必填
        for field_name, stat in stats.items():
            if not stat["is_required"] and stat["usage_rate"] > 0.9:
                suggestions.append({
                    "type": "make_required",
                    "field_name": field_name,
                    "reason": f"高使用率 ({stat['usage_rate']:.1%})，建議設為必填",
                    "priority": "medium"
                })
        
        return suggestions
    
    def get_schema_evolution_history(self) -> List[Dict[str, Any]]:
        """獲取 Schema 演進歷史"""
        history = []
        
        # 按版本排序
        sorted_versions = sorted(self.schemas.keys(), key=self._parse_version)
        
        for version in sorted_versions:
            schema = self.schemas[version]
            
            # 獲取該版本的遷移記錄
            version_migrations = [
                m for m in self.migrations 
                if m.to_version == version
            ]
            
            history.append({
                "version": version,
                "description": schema.description,
                "created_at": schema.created_at,
                "is_active": schema.is_active,
                "backward_compatible": schema.backward_compatible,
                "field_count": len(schema.fields),
                "migrations": [
                    {
                        "type": m.migration_type,
                        "field": m.field_name,
                        "from_version": m.from_version
                    }
                    for m in version_migrations
                ]
            })
        
        return history


# 全域 Schema 管理器實例
schema_manager = DynamicSchemaManager()