"""
Permission Management System
分級權限管理系統，用於控制 MCP 工具的使用權限
"""
import logging
from enum import Enum
from typing import Set, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PermissionLevel(str, Enum):
    """權限級別"""
    USER = "user"              # 一般用戶：只能使用安全的讀取和RAG操作
    ADMIN = "admin"            # 管理員：可執行基本管理操作
    SUPER_ADMIN = "super_admin" # 超級管理員：可執行所有危險操作


class OperationType(str, Enum):
    """操作類型分類"""
    # 安全操作（所有用戶可用）
    READ_ONLY = "read_only"
    RAG_SEARCH = "rag_search"
    RAG_ANALYSIS = "rag_analysis"
    
    # 基本管理操作（需要 admin 權限）
    DATA_MODIFY = "data_modify"
    SCHEMA_MANAGE = "schema_manage" 
    COLLECTION_MANAGE = "collection_manage"
    
    # 危險操作（需要 super_admin 權限）
    DATA_DELETE = "data_delete"
    COLLECTION_DELETE = "collection_delete"
    SYSTEM_OPTIMIZE = "system_optimize"
    BULK_OPERATIONS = "bulk_operations"


class ToolPermission(BaseModel):
    """工具權限配置"""
    tool_name: str = Field(..., description="工具名稱")
    operation_type: OperationType = Field(..., description="操作類型")
    required_level: PermissionLevel = Field(..., description="需要的權限級別")
    description: str = Field(default="", description="權限說明")
    risk_level: str = Field(default="low", description="風險級別: low, medium, high, critical")


class PermissionManager:
    """權限管理器"""
    
    def __init__(self):
        self.user_permissions: Dict[str, PermissionLevel] = {}
        self.tool_permissions: Dict[str, ToolPermission] = {}
        self._initialize_default_permissions()
    
    def _initialize_default_permissions(self):
        """初始化預設權限配置"""
        
        # 安全操作 - USER 級別
        safe_tools = [
            # RAG Bridge 工具（安全操作）
            ("search-experience", OperationType.RAG_SEARCH, "搜尋個人經驗知識庫"),
            ("get-process-workflow", OperationType.RAG_SEARCH, "獲取流程工作流程"),
            ("suggest-similar", OperationType.RAG_ANALYSIS, "推薦相關經驗"),
            ("search-vocabulary", OperationType.RAG_SEARCH, "搜尋標準化詞彙"),
            ("standardize-content", OperationType.RAG_ANALYSIS, "標準化內容和標籤"),
            ("get-vocabulary-statistics", OperationType.READ_ONLY, "獲取詞彙統計"),
            ("manage-fragment-vocabulary", OperationType.RAG_ANALYSIS, "管理分片詞彙"),
            
            # Schema 查詢工具（安全操作）
            ("get-current-schema", OperationType.READ_ONLY, "獲取當前Schema"),
            ("validate-schema-data", OperationType.RAG_ANALYSIS, "驗證Schema數據"),
            ("analyze-schema-usage", OperationType.RAG_ANALYSIS, "分析Schema使用"),
            ("get-schema-suggestions", OperationType.RAG_ANALYSIS, "獲取Schema建議"),
            ("get-schema-evolution-history", OperationType.READ_ONLY, "查看Schema演進歷史"),
            
            # Qdrant 查詢工具（安全操作）
            ("qdrant-find", OperationType.READ_ONLY, "搜尋向量資料"),
            ("qdrant-list-collections", OperationType.READ_ONLY, "列出集合"),
            ("qdrant-system-status", OperationType.READ_ONLY, "查看系統狀態"),
            ("qdrant-performance-analysis", OperationType.READ_ONLY, "效能分析"),
            ("qdrant-analyze-storage", OperationType.READ_ONLY, "分析儲存使用"),
        ]
        
        for tool_name, op_type, desc in safe_tools:
            self.tool_permissions[tool_name] = ToolPermission(
                tool_name=tool_name,
                operation_type=op_type,
                required_level=PermissionLevel.USER,
                description=desc,
                risk_level="low"
            )
        
        # 管理操作 - ADMIN 級別
        admin_tools = [
            # Schema 管理（中風險）
            ("request-schema-field-addition", OperationType.SCHEMA_MANAGE, "請求新增Schema欄位"),
            ("request-schema-field-removal", OperationType.SCHEMA_MANAGE, "請求移除Schema欄位"),
            ("list-pending-schema-requests", OperationType.SCHEMA_MANAGE, "列出待審Schema請求"),
            ("review-schema-request", OperationType.SCHEMA_MANAGE, "審查Schema請求"),
            ("get-schema-approval-history", OperationType.SCHEMA_MANAGE, "查看Schema審查歷史"),
            
            # RAG 數據管理（中風險）
            ("update-experience", OperationType.DATA_MODIFY, "更新經驗反饋"),
            ("propose-vocabulary", OperationType.DATA_MODIFY, "提議新詞彙"),
            ("qdrant-store", OperationType.DATA_MODIFY, "儲存向量資料"),
            ("qdrant-update-metadata", OperationType.DATA_MODIFY, "更新元數據"),
            
            # 資料遷移分析（中風險）
            ("analyze-collection-for-migration", OperationType.DATA_MODIFY, "分析Collection遷移"),
            ("execute-migration-dry-run", OperationType.DATA_MODIFY, "執行遷移預演"),
        ]
        
        for tool_name, op_type, desc in admin_tools:
            self.tool_permissions[tool_name] = ToolPermission(
                tool_name=tool_name,
                operation_type=op_type,
                required_level=PermissionLevel.ADMIN,
                description=desc,
                risk_level="medium"
            )
        
        # 危險操作 - SUPER_ADMIN 級別
        dangerous_tools = [
            # 資料刪除（高風險）
            ("qdrant-delete-documents", OperationType.DATA_DELETE, "刪除文檔"),
            ("qdrant-delete-collection", OperationType.COLLECTION_DELETE, "刪除整個集合"),
            ("qdrant-remove-metadata-keys", OperationType.DATA_DELETE, "移除元數據鍵"),
            
            # 大量操作（高風險）
            ("qdrant-move-documents", OperationType.BULK_OPERATIONS, "搬移文檔"),
            ("qdrant-optimize-storage", OperationType.SYSTEM_OPTIMIZE, "儲存優化"),
            
            # 資料遷移（高風險）
            ("execute-data-migration", OperationType.BULK_OPERATIONS, "執行實際資料遷移"),
        ]
        
        for tool_name, op_type, desc in dangerous_tools:
            self.tool_permissions[tool_name] = ToolPermission(
                tool_name=tool_name,
                operation_type=op_type,
                required_level=PermissionLevel.SUPER_ADMIN,
                description=desc,
                risk_level="critical"
            )
    
    def set_user_permission(self, user_id: str, level: PermissionLevel):
        """設定用戶權限級別"""
        self.user_permissions[user_id] = level
        logger.info(f"Set user {user_id} permission to {level.value}")
    
    def get_user_permission(self, user_id: str) -> PermissionLevel:
        """獲取用戶權限級別"""
        return self.user_permissions.get(user_id, PermissionLevel.USER)
    
    def check_tool_permission(self, user_id: str, tool_name: str) -> bool:
        """檢查用戶是否有權限使用指定工具"""
        user_level = self.get_user_permission(user_id)
        tool_permission = self.tool_permissions.get(tool_name)
        
        if not tool_permission:
            # 未定義的工具預設拒絕存取
            logger.warning(f"Tool {tool_name} has no permission configuration")
            return False
        
        required_level = tool_permission.required_level
        
        # 權限層級檢查
        level_hierarchy = {
            PermissionLevel.USER: 0,
            PermissionLevel.ADMIN: 1,
            PermissionLevel.SUPER_ADMIN: 2
        }
        
        user_level_value = level_hierarchy.get(user_level, 0)
        required_level_value = level_hierarchy.get(required_level, 2)
        
        has_permission = user_level_value >= required_level_value
        
        if not has_permission:
            logger.warning(
                f"User {user_id} (level: {user_level.value}) denied access to tool {tool_name} "
                f"(requires: {required_level.value})"
            )
        
        return has_permission
    
    def get_available_tools(self, user_id: str) -> List[str]:
        """獲取用戶可用的工具列表"""
        available_tools = []
        for tool_name in self.tool_permissions.keys():
            if self.check_tool_permission(user_id, tool_name):
                available_tools.append(tool_name)
        return available_tools
    
    def get_permission_summary(self, user_id: str) -> Dict:
        """獲取用戶權限摘要"""
        user_level = self.get_user_permission(user_id)
        available_tools = self.get_available_tools(user_id)
        
        # 按風險級別分類工具
        tools_by_risk = {"low": [], "medium": [], "critical": []}
        for tool_name in available_tools:
            tool_perm = self.tool_permissions[tool_name]
            risk_level = tool_perm.risk_level
            if risk_level in tools_by_risk:
                tools_by_risk[risk_level].append(tool_name)
        
        return {
            "user_id": user_id,
            "permission_level": user_level.value,
            "total_available_tools": len(available_tools),
            "tools_by_risk": tools_by_risk,
            "available_operations": self._get_available_operations(user_level)
        }
    
    def _get_available_operations(self, user_level: PermissionLevel) -> List[str]:
        """獲取用戶可執行的操作類型"""
        if user_level == PermissionLevel.SUPER_ADMIN:
            return [op.value for op in OperationType]
        elif user_level == PermissionLevel.ADMIN:
            return [
                OperationType.READ_ONLY.value,
                OperationType.RAG_SEARCH.value,
                OperationType.RAG_ANALYSIS.value,
                OperationType.DATA_MODIFY.value,
                OperationType.SCHEMA_MANAGE.value,
                OperationType.COLLECTION_MANAGE.value
            ]
        else:  # USER
            return [
                OperationType.READ_ONLY.value,
                OperationType.RAG_SEARCH.value,
                OperationType.RAG_ANALYSIS.value
            ]
    
    def add_custom_tool_permission(
        self, 
        tool_name: str, 
        operation_type: OperationType,
        required_level: PermissionLevel,
        description: str = "",
        risk_level: str = "medium"
    ):
        """新增自定義工具權限"""
        self.tool_permissions[tool_name] = ToolPermission(
            tool_name=tool_name,
            operation_type=operation_type,
            required_level=required_level,
            description=description,
            risk_level=risk_level
        )
        logger.info(f"Added custom permission for tool {tool_name}")


# 全域權限管理器實例
permission_manager = None

def get_permission_manager() -> PermissionManager:
    """獲取權限管理器實例"""
    global permission_manager
    if permission_manager is None:
        permission_manager = PermissionManager()
    return permission_manager