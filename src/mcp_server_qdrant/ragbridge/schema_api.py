"""
Schema 管理 API 介面

提供 Schema 管理的 HTTP API 和 MCP 工具介面，包括：
1. Schema 版本控制
2. 欄位管理（新增、修改、移除）
3. 數據驗證
4. Schema 演進分析和建議
5. 遷移管理
"""
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from mcp_server_qdrant.ragbridge.schema_manager import (
    DynamicSchemaManager,
    SchemaField,
    FieldType,
    FieldValidation,
    SchemaEvolutionProposal,
    schema_manager
)

logger = logging.getLogger(__name__)


class SchemaAPI:
    """Schema 管理 API"""
    
    def __init__(self):
        self.schema_manager = schema_manager
    
    async def get_current_schema(self) -> Dict[str, Any]:
        """獲取當前 Schema"""
        try:
            current_schema = self.schema_manager.get_current_schema()
            
            # 格式化欄位資訊
            fields_info = {}
            for field_name, field in current_schema.fields.items():
                fields_info[field_name] = {
                    "type": field.field_type.value,
                    "description": field.description,
                    "required": field.validation.required,
                    "is_core": field.is_core,
                    "deprecated": field.deprecated,
                    "added_in_version": field.added_in_version,
                    "validation_rules": {
                        "min_length": field.validation.min_length,
                        "max_length": field.validation.max_length,
                        "pattern": field.validation.pattern,
                        "min_value": field.validation.min_value,
                        "max_value": field.validation.max_value,
                        "allowed_values": field.validation.allowed_values,
                        "default_value": field.validation.default_value
                    }
                }
            
            return {
                "schema_version": current_schema.version,
                "description": current_schema.description,
                "created_at": current_schema.created_at.isoformat(),
                "is_active": current_schema.is_active,
                "backward_compatible": current_schema.backward_compatible,
                "total_fields": len(current_schema.fields),
                "core_fields_count": len([f for f in current_schema.fields.values() if f.is_core]),
                "deprecated_fields_count": len([f for f in current_schema.fields.values() if f.deprecated]),
                "fields": fields_info
            }
            
        except Exception as e:
            logger.error(f"獲取當前 Schema 失敗: {e}")
            return {
                "error": str(e),
                "schema_version": None
            }
    
    async def add_schema_field(
        self,
        field_name: str,
        field_type: str,
        description: str = "",
        required: bool = False,
        validation_rules: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """新增 Schema 欄位"""
        try:
            # 驗證欄位類型
            try:
                field_type_enum = FieldType(field_type)
            except ValueError:
                return {
                    "success": False,
                    "message": f"無效的欄位類型: {field_type}",
                    "valid_types": [t.value for t in FieldType]
                }
            
            # 建立驗證規則
            validation = FieldValidation(required=required)
            if validation_rules:
                for key, value in validation_rules.items():
                    if hasattr(validation, key) and value is not None:
                        setattr(validation, key, value)
            
            # 建立欄位
            field = SchemaField(
                name=field_name,
                field_type=field_type_enum,
                description=description,
                validation=validation,
                added_in_version="0.0.0"  # 將由 schema_manager 自動設定
            )
            
            # 新增到 Schema
            success = self.schema_manager.add_field(field)
            
            if success:
                new_schema = self.schema_manager.get_current_schema()
                return {
                    "success": True,
                    "message": f"成功新增欄位 '{field_name}'",
                    "field_name": field_name,
                    "new_schema_version": new_schema.version,
                    "field_info": {
                        "type": field.field_type.value,
                        "description": field.description,
                        "required": field.validation.required,
                        "added_in_version": new_schema.fields[field_name].added_in_version
                    }
                }
            else:
                return {
                    "success": False,
                    "message": f"新增欄位 '{field_name}' 失敗"
                }
                
        except Exception as e:
            logger.error(f"新增 Schema 欄位失敗: {e}")
            return {
                "success": False,
                "message": f"新增欄位失敗: {str(e)}"
            }
    
    async def remove_schema_field(self, field_name: str) -> Dict[str, Any]:
        """移除 Schema 欄位（標記為棄用）"""
        try:
            success = self.schema_manager.remove_field(field_name)
            
            if success:
                new_schema = self.schema_manager.get_current_schema()
                return {
                    "success": True,
                    "message": f"成功棄用欄位 '{field_name}'",
                    "field_name": field_name,
                    "new_schema_version": new_schema.version,
                    "note": "欄位已標記為棄用，而非直接刪除以保持向後兼容性"
                }
            else:
                return {
                    "success": False,
                    "message": f"棄用欄位 '{field_name}' 失敗"
                }
                
        except Exception as e:
            logger.error(f"移除 Schema 欄位失敗: {e}")
            return {
                "success": False,
                "message": f"移除欄位失敗: {str(e)}"
            }
    
    async def modify_schema_field(
        self,
        field_name: str,
        field_type: Optional[str] = None,
        description: Optional[str] = None,
        required: Optional[bool] = None,
        validation_rules: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """修改 Schema 欄位"""
        try:
            # 獲取現有欄位
            current_schema = self.schema_manager.get_current_schema()
            if field_name not in current_schema.fields:
                return {
                    "success": False,
                    "message": f"欄位 '{field_name}' 不存在"
                }
            
            existing_field = current_schema.fields[field_name]
            
            # 建立新的欄位定義
            new_field_type = FieldType(field_type) if field_type else existing_field.field_type
            new_description = description if description is not None else existing_field.description
            new_required = required if required is not None else existing_field.validation.required
            
            # 建立新的驗證規則
            new_validation = FieldValidation(
                required=new_required,
                min_length=existing_field.validation.min_length,
                max_length=existing_field.validation.max_length,
                pattern=existing_field.validation.pattern,
                min_value=existing_field.validation.min_value,
                max_value=existing_field.validation.max_value,
                allowed_values=existing_field.validation.allowed_values,
                default_value=existing_field.validation.default_value
            )
            
            if validation_rules:
                for key, value in validation_rules.items():
                    if hasattr(new_validation, key) and value is not None:
                        setattr(new_validation, key, value)
            
            # 建立新欄位
            new_field = SchemaField(
                name=field_name,
                field_type=new_field_type,
                description=new_description,
                validation=new_validation,
                is_core=existing_field.is_core,
                deprecated=existing_field.deprecated,
                added_in_version=existing_field.added_in_version
            )
            
            # 修改欄位
            success = self.schema_manager.modify_field(field_name, new_field)
            
            if success:
                updated_schema = self.schema_manager.get_current_schema()
                return {
                    "success": True,
                    "message": f"成功修改欄位 '{field_name}'",
                    "field_name": field_name,
                    "new_schema_version": updated_schema.version,
                    "changes": {
                        "type": new_field.field_type.value,
                        "description": new_field.description,
                        "required": new_field.validation.required
                    }
                }
            else:
                return {
                    "success": False,
                    "message": f"修改欄位 '{field_name}' 失敗"
                }
                
        except Exception as e:
            logger.error(f"修改 Schema 欄位失敗: {e}")
            return {
                "success": False,
                "message": f"修改欄位失敗: {str(e)}"
            }
    
    async def validate_data(
        self,
        data: Dict[str, Any],
        schema_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """驗證數據是否符合 Schema"""
        try:
            is_valid, errors = self.schema_manager.validate_data(data, schema_version)
            
            # 獲取使用的 Schema 版本
            if schema_version:
                used_schema = self.schema_manager.schemas.get(schema_version)
                used_version = schema_version
            else:
                used_schema = self.schema_manager.get_current_schema()
                used_version = used_schema.version
            
            result = {
                "is_valid": is_valid,
                "schema_version": used_version,
                "validation_errors": errors,
                "error_count": len(errors)
            }
            
            if is_valid:
                result["message"] = "數據驗證通過"
            else:
                result["message"] = f"數據驗證失敗，發現 {len(errors)} 個錯誤"
            
            return result
            
        except Exception as e:
            logger.error(f"數據驗證失敗: {e}")
            return {
                "is_valid": False,
                "error": str(e),
                "validation_errors": [f"驗證過程出錯: {str(e)}"]
            }
    
    async def analyze_schema_usage(
        self,
        data_samples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """分析 Schema 使用情況"""
        try:
            if not data_samples:
                return {
                    "error": "需要提供數據樣本進行分析",
                    "sample_count": 0
                }
            
            analysis = self.schema_manager.analyze_schema_usage(data_samples)
            
            # 格式化分析結果
            if "error" in analysis:
                return analysis
            
            # 計算額外統計資訊
            stats = analysis["field_usage_stats"]
            high_usage_fields = [name for name, stat in stats.items() if stat["usage_rate"] > 0.8]
            low_usage_fields = [name for name, stat in stats.items() if stat["usage_rate"] < 0.2 and not stat["is_core"]]
            
            return {
                **analysis,
                "summary": {
                    "high_usage_fields": high_usage_fields,
                    "low_usage_fields": low_usage_fields,
                    "compliance_level": "good" if analysis["schema_compliance_rate"] > 0.7 else "poor",
                    "suggestions_available": len(low_usage_fields) > 0 or len(analysis.get("unknown_fields", [])) > 0
                }
            }
            
        except Exception as e:
            logger.error(f"Schema 使用分析失敗: {e}")
            return {
                "error": str(e),
                "sample_count": len(data_samples) if data_samples else 0
            }
    
    async def get_schema_suggestions(
        self,
        data_samples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """獲取 Schema 改進建議"""
        try:
            if not data_samples:
                return {
                    "suggestions": [],
                    "message": "需要提供數據樣本以生成建議"
                }
            
            # 先進行使用分析
            usage_analysis = self.schema_manager.analyze_schema_usage(data_samples)
            if "error" in usage_analysis:
                return {
                    "error": usage_analysis["error"],
                    "suggestions": []
                }
            
            # 獲取改進建議
            suggestions = self.schema_manager.suggest_schema_improvements(usage_analysis)
            
            # 按優先級排序
            priority_order = {"high": 3, "medium": 2, "low": 1}
            suggestions.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 1), reverse=True)
            
            return {
                "analysis_summary": {
                    "total_samples": usage_analysis["total_samples"],
                    "schema_version": usage_analysis["current_schema_version"],
                    "compliance_rate": usage_analysis["schema_compliance_rate"]
                },
                "suggestion_count": len(suggestions),
                "suggestions": suggestions,
                "message": f"基於 {usage_analysis['total_samples']} 個樣本生成了 {len(suggestions)} 個建議"
            }
            
        except Exception as e:
            logger.error(f"獲取 Schema 建議失敗: {e}")
            return {
                "error": str(e),
                "suggestions": []
            }
    
    async def get_schema_evolution_history(self) -> Dict[str, Any]:
        """獲取 Schema 演進歷史"""
        try:
            history = self.schema_manager.get_schema_evolution_history()
            
            # 計算統計資訊
            total_versions = len(history)
            active_versions = len([v for v in history if v["is_active"]])
            total_migrations = sum(len(v["migrations"]) for v in history)
            
            return {
                "total_versions": total_versions,
                "active_versions": active_versions,
                "total_migrations": total_migrations,
                "evolution_history": history,
                "summary": {
                    "first_version": history[0]["version"] if history else None,
                    "latest_version": history[-1]["version"] if history else None,
                    "most_changes": max(history, key=lambda x: len(x["migrations"]))["version"] if history else None
                }
            }
            
        except Exception as e:
            logger.error(f"獲取 Schema 演進歷史失敗: {e}")
            return {
                "error": str(e),
                "evolution_history": []
            }
    
    async def create_schema_proposal(
        self,
        title: str,
        description: str,
        proposed_changes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """創建 Schema 演進提案"""
        try:
            # 生成提案 ID
            proposal_id = str(uuid.uuid4())
            
            # 進行影響分析
            impact_analysis = self._analyze_proposal_impact(proposed_changes)
            
            # 創建提案
            proposal = SchemaEvolutionProposal(
                proposal_id=proposal_id,
                title=title,
                description=description,
                proposed_changes=proposed_changes,
                impact_analysis=impact_analysis
            )
            
            # 儲存提案（這裡可以擴展到數據庫）
            self.schema_manager.proposals[proposal_id] = proposal
            
            return {
                "success": True,
                "proposal_id": proposal_id,
                "title": title,
                "status": proposal.status,
                "created_at": proposal.created_at.isoformat(),
                "impact_analysis": impact_analysis,
                "message": f"成功創建 Schema 演進提案 '{title}'"
            }
            
        except Exception as e:
            logger.error(f"創建 Schema 提案失敗: {e}")
            return {
                "success": False,
                "message": f"創建提案失敗: {str(e)}"
            }
    
    def _analyze_proposal_impact(self, proposed_changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析提案影響"""
        impact = {
            "breaking_changes": False,
            "migration_required": False,
            "affected_fields": [],
            "risk_level": "low",
            "estimated_effort": "minimal"
        }
        
        for change in proposed_changes:
            change_type = change.get("type", "")
            field_name = change.get("field_name", "")
            
            if change_type in ["remove_field", "modify_field_type"]:
                impact["breaking_changes"] = True
                impact["migration_required"] = True
                impact["risk_level"] = "high"
                impact["estimated_effort"] = "significant"
            elif change_type in ["add_required_field"]:
                impact["migration_required"] = True
                impact["risk_level"] = "medium"
                impact["estimated_effort"] = "moderate"
            
            if field_name:
                impact["affected_fields"].append(field_name)
        
        return impact


# 全域 Schema API 實例
schema_api = SchemaAPI()