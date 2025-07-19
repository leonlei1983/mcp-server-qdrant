"""
Schema 審查和核准機制

此模組負責：
1. Schema 變更提案的審查流程
2. 多級審查權限控制
3. 審查記錄和追蹤
4. 自動化安全檢查
"""
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set

from pydantic import BaseModel, Field

from mcp_server_qdrant.ragbridge.schema_manager import (
    DynamicSchemaManager,
    SchemaField,
    SchemaEvolutionProposal,
    SchemaMigration
)

logger = logging.getLogger(__name__)


class ApprovalLevel(str, Enum):
    """審查級別"""
    AUTOMATIC = "automatic"      # 自動核准（低風險變更）
    REVIEWER = "reviewer"        # 需要審查員核准
    ADMIN = "admin"             # 需要管理員核准
    COMMITTEE = "committee"      # 需要委員會核准（高風險變更）


class ChangeRiskLevel(str, Enum):
    """變更風險級別"""
    LOW = "low"                 # 低風險：新增可選欄位
    MEDIUM = "medium"           # 中風險：修改欄位、新增必填欄位
    HIGH = "high"              # 高風險：移除欄位、破壞性變更
    CRITICAL = "critical"       # 關鍵風險：核心欄位變更


class SchemaChangeRequest(BaseModel):
    """Schema 變更請求"""
    request_id: str = Field(..., description="請求ID")
    change_type: str = Field(..., description="變更類型")
    field_name: str = Field(..., description="相關欄位名稱")
    change_details: Dict = Field(..., description="變更詳情")
    risk_level: ChangeRiskLevel = Field(..., description="風險級別")
    required_approval_level: ApprovalLevel = Field(..., description="需要的審查級別")
    
    # 提案資訊
    proposed_by: str = Field(default="system", description="提案者")
    proposed_at: datetime = Field(default_factory=datetime.now, description="提案時間")
    justification: str = Field(default="", description="變更理由")
    
    # 審查狀態
    status: str = Field(default="pending", description="狀態: pending, reviewing, approved, rejected")
    reviewed_by: Optional[str] = Field(None, description="審查者")
    reviewed_at: Optional[datetime] = Field(None, description="審查時間")
    review_comments: str = Field(default="", description="審查意見")
    
    # 影響分析
    impact_analysis: Dict = Field(default_factory=dict, description="影響分析")
    affected_systems: List[str] = Field(default_factory=list, description="受影響的系統")


class SchemaApprovalManager:
    """Schema 審查管理器"""
    
    def __init__(self, schema_manager: DynamicSchemaManager):
        self.schema_manager = schema_manager
        self.pending_requests: Dict[str, SchemaChangeRequest] = {}
        self.approval_history: List[SchemaChangeRequest] = []
        
        # 審查權限設定
        self.reviewers: Set[str] = {"admin", "schema_reviewer", "lead_developer"}
        self.admins: Set[str] = {"admin", "system_admin"}
        
        # 自動核准規則
        self.auto_approval_rules = {
            "add_optional_field": ChangeRiskLevel.LOW,
            "update_field_description": ChangeRiskLevel.LOW,
        }
    
    def assess_change_risk(self, change_type: str, field_name: str, change_details: Dict) -> ChangeRiskLevel:
        """評估變更風險級別"""
        
        # 檢查是否為核心欄位
        if field_name in self.schema_manager.core_fields:
            return ChangeRiskLevel.CRITICAL
        
        # 根據變更類型評估風險
        if change_type == "add_field":
            required = change_details.get("required", False)
            return ChangeRiskLevel.MEDIUM if required else ChangeRiskLevel.LOW
        
        elif change_type == "remove_field":
            return ChangeRiskLevel.HIGH
        
        elif change_type == "modify_field":
            # 檢查是否改變了欄位類型或必填狀態
            if "field_type" in change_details or "required" in change_details:
                return ChangeRiskLevel.MEDIUM
            return ChangeRiskLevel.LOW
        
        elif change_type == "rename_field":
            return ChangeRiskLevel.HIGH
        
        return ChangeRiskLevel.MEDIUM
    
    def determine_approval_level(self, risk_level: ChangeRiskLevel) -> ApprovalLevel:
        """確定需要的審查級別"""
        mapping = {
            ChangeRiskLevel.LOW: ApprovalLevel.AUTOMATIC,
            ChangeRiskLevel.MEDIUM: ApprovalLevel.REVIEWER,
            ChangeRiskLevel.HIGH: ApprovalLevel.ADMIN,
            ChangeRiskLevel.CRITICAL: ApprovalLevel.COMMITTEE
        }
        return mapping.get(risk_level, ApprovalLevel.REVIEWER)
    
    def create_change_request(
        self,
        change_type: str,
        field_name: str,
        change_details: Dict,
        proposed_by: str = "system",
        justification: str = ""
    ) -> str:
        """創建變更請求"""
        
        request_id = str(uuid.uuid4())
        
        # 評估風險和審查級別
        risk_level = self.assess_change_risk(change_type, field_name, change_details)
        approval_level = self.determine_approval_level(risk_level)
        
        # 進行影響分析
        impact_analysis = self._analyze_change_impact(change_type, field_name, change_details)
        
        # 創建請求
        request = SchemaChangeRequest(
            request_id=request_id,
            change_type=change_type,
            field_name=field_name,
            change_details=change_details,
            risk_level=risk_level,
            required_approval_level=approval_level,
            proposed_by=proposed_by,
            justification=justification,
            impact_analysis=impact_analysis
        )
        
        # 檢查是否可以自動核准
        if approval_level == ApprovalLevel.AUTOMATIC:
            return self._auto_approve_request(request)
        
        # 儲存待審查請求
        self.pending_requests[request_id] = request
        
        logger.info(f"創建 Schema 變更請求: {request_id} (風險級別: {risk_level.value})")
        return request_id
    
    def _analyze_change_impact(self, change_type: str, field_name: str, change_details: Dict) -> Dict:
        """分析變更影響"""
        impact = {
            "breaking_change": False,
            "data_migration_required": False,
            "estimated_downtime": "0 minutes",
            "affected_queries": [],
            "rollback_complexity": "simple"
        }
        
        if change_type == "remove_field":
            impact["breaking_change"] = True
            impact["data_migration_required"] = True
            impact["rollback_complexity"] = "complex"
        
        elif change_type == "modify_field":
            if change_details.get("field_type"):
                impact["data_migration_required"] = True
                impact["rollback_complexity"] = "moderate"
        
        elif change_type == "add_field" and change_details.get("required"):
            impact["data_migration_required"] = True
            impact["estimated_downtime"] = "5-10 minutes"
        
        return impact
    
    def _auto_approve_request(self, request: SchemaChangeRequest) -> str:
        """自動核准請求"""
        request.status = "approved"
        request.reviewed_by = "system"
        request.reviewed_at = datetime.now()
        request.review_comments = "自動核准（低風險變更）"
        
        # 執行變更
        success = self._execute_change(request)
        
        if success:
            self.approval_history.append(request)
            logger.info(f"自動核准並執行變更請求: {request.request_id}")
            return request.request_id
        else:
            request.status = "rejected"
            request.review_comments = "執行失敗"
            return request.request_id
    
    def review_request(
        self,
        request_id: str,
        reviewer: str,
        action: str,  # "approve" 或 "reject"
        comments: str = ""
    ) -> bool:
        """審查請求"""
        
        if request_id not in self.pending_requests:
            logger.error(f"請求不存在: {request_id}")
            return False
        
        request = self.pending_requests[request_id]
        
        # 檢查審查權限
        if not self._check_review_permission(reviewer, request.required_approval_level):
            logger.error(f"用戶 {reviewer} 沒有審查權限")
            return False
        
        # 更新審查狀態
        request.reviewed_by = reviewer
        request.reviewed_at = datetime.now()
        request.review_comments = comments
        
        if action == "approve":
            request.status = "approved"
            
            # 執行變更
            success = self._execute_change(request)
            if not success:
                request.status = "rejected"
                request.review_comments += " (執行失敗)"
        
        elif action == "reject":
            request.status = "rejected"
        
        # 移動到歷史記錄
        self.approval_history.append(request)
        del self.pending_requests[request_id]
        
        logger.info(f"審查請求 {request_id}: {action} by {reviewer}")
        return True
    
    def _check_review_permission(self, reviewer: str, required_level: ApprovalLevel) -> bool:
        """檢查審查權限"""
        if required_level == ApprovalLevel.AUTOMATIC:
            return True
        elif required_level == ApprovalLevel.REVIEWER:
            return reviewer in self.reviewers
        elif required_level == ApprovalLevel.ADMIN:
            return reviewer in self.admins
        elif required_level == ApprovalLevel.COMMITTEE:
            # 委員會審查需要特殊處理
            return reviewer in self.admins  # 簡化處理
        
        return False
    
    def _execute_change(self, request: SchemaChangeRequest) -> bool:
        """執行 Schema 變更"""
        try:
            change_type = request.change_type
            field_name = request.field_name
            details = request.change_details
            
            if change_type == "add_field":
                # 處理驗證規則
                from mcp_server_qdrant.ragbridge.schema_manager import FieldValidation, FieldType
                validation_data = details.get("validation", {})
                if isinstance(validation_data, dict):
                    validation = FieldValidation(**validation_data)
                else:
                    validation = FieldValidation()
                
                # 設定必填狀態
                validation.required = details.get("required", False)
                
                field = SchemaField(
                    name=field_name,
                    field_type=FieldType(details["field_type"]),
                    description=details.get("description", ""),
                    validation=validation,
                    added_in_version="0.0.0"  # 將被自動設定
                )
                return self.schema_manager.add_field(field)
            
            elif change_type == "remove_field":
                return self.schema_manager.remove_field(field_name)
            
            elif change_type == "modify_field":
                current_schema = self.schema_manager.get_current_schema()
                if field_name not in current_schema.fields:
                    return False
                
                existing_field = current_schema.fields[field_name]
                
                # 處理驗證規則
                from mcp_server_qdrant.ragbridge.schema_manager import FieldValidation
                validation_data = details.get("validation", {})
                if isinstance(validation_data, dict):
                    validation = FieldValidation(**validation_data)
                else:
                    validation = existing_field.validation
                
                new_field = SchemaField(
                    name=field_name,
                    field_type=details.get("field_type", existing_field.field_type),
                    description=details.get("description", existing_field.description),
                    validation=validation,
                    is_core=existing_field.is_core,
                    added_in_version=existing_field.added_in_version
                )
                return self.schema_manager.modify_field(field_name, new_field)
            
            return False
            
        except Exception as e:
            logger.error(f"執行 Schema 變更失敗: {e}")
            return False
    
    def get_pending_requests(self, reviewer: Optional[str] = None) -> List[Dict]:
        """獲取待審查請求"""
        requests = []
        
        for request in self.pending_requests.values():
            # 權限過濾
            if reviewer and not self._check_review_permission(reviewer, request.required_approval_level):
                continue
            
            requests.append({
                "request_id": request.request_id,
                "change_type": request.change_type,
                "field_name": request.field_name,
                "risk_level": request.risk_level.value,
                "required_approval_level": request.required_approval_level.value,
                "proposed_by": request.proposed_by,
                "proposed_at": request.proposed_at.isoformat(),
                "justification": request.justification,
                "impact_analysis": request.impact_analysis
            })
        
        return requests
    
    def get_approval_history(self, limit: int = 50) -> List[Dict]:
        """獲取審查歷史"""
        history = []
        
        for request in self.approval_history[-limit:]:
            history.append({
                "request_id": request.request_id,
                "change_type": request.change_type,
                "field_name": request.field_name,
                "status": request.status,
                "risk_level": request.risk_level.value,
                "proposed_by": request.proposed_by,
                "proposed_at": request.proposed_at.isoformat(),
                "reviewed_by": request.reviewed_by,
                "reviewed_at": request.reviewed_at.isoformat() if request.reviewed_at else None,
                "review_comments": request.review_comments
            })
        
        return history


# 全域審查管理器實例
approval_manager = None

def get_approval_manager() -> SchemaApprovalManager:
    """獲取審查管理器實例"""
    global approval_manager
    if approval_manager is None:
        from mcp_server_qdrant.ragbridge.schema_manager import schema_manager
        approval_manager = SchemaApprovalManager(schema_manager)
    return approval_manager