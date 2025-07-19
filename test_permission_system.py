#!/usr/bin/env python3
"""
測試權限管理系統功能
"""
import asyncio
import os
from mcp_server_qdrant.mcp_server import QdrantMCPServer
from mcp_server_qdrant.settings import QdrantSettings, EmbeddingProviderSettings, ToolSettings

async def test_permission_system():
    """測試權限管理系統"""
    
    print("🧪 **開始測試權限管理系統**\n")
    
    # 載入環境變數
    from dotenv import load_dotenv
    load_dotenv()
    
    # 設定權限系統環境變數
    os.environ["QDRANT_ENABLE_PERMISSION_SYSTEM"] = "true"
    os.environ["QDRANT_DEFAULT_PERMISSION_LEVEL"] = "user"
    
    # 初始化設定
    tool_settings = ToolSettings()
    qdrant_settings = QdrantSettings()
    embedding_settings = EmbeddingProviderSettings()
    
    try:
        # 創建 MCP server
        server = QdrantMCPServer(
            tool_settings=tool_settings,
            qdrant_settings=qdrant_settings, 
            embedding_provider_settings=embedding_settings
        )
        print('✅ MCP Server 初始化成功 (權限系統啟用)')
        
        # 獲取所有工具
        tools = await server.get_tools()
        tool_names = list(tools.keys())
        
        print(f'\n📋 總工具數: {len(tool_names)}')
        
        # 測試權限管理器
        permission_manager = server.permission_manager
        
        # 1. 測試用戶權限查詢
        print("\n1️⃣ **測試權限查詢**")
        user_permission = permission_manager.get_user_permission("default_user")
        print(f"👤 預設用戶權限級別: {user_permission.value}")
        
        # 2. 測試工具權限檢查
        print("\n2️⃣ **測試工具權限檢查**")
        
        # 安全工具（user 級別應該可以使用）
        safe_tools = ["search-experience", "get-current-schema", "qdrant-find"]
        for tool in safe_tools:
            has_permission = permission_manager.check_tool_permission("default_user", tool)
            status = "✅ 允許" if has_permission else "❌ 拒絕"
            print(f"  {tool}: {status}")
        
        print()
        
        # 危險工具（user 級別應該被拒絕）
        dangerous_tools = ["qdrant-delete-documents", "qdrant-delete-collection", "execute-data-migration"]
        for tool in dangerous_tools:
            has_permission = permission_manager.check_tool_permission("default_user", tool)
            status = "✅ 允許" if has_permission else "❌ 拒絕"
            print(f"  {tool}: {status}")
        
        # 3. 測試權限級別統計
        print("\n3️⃣ **測試權限統計**")
        summary = permission_manager.get_permission_summary("default_user")
        print(f"📊 可用工具數量: {summary['total_available_tools']}")
        
        for risk_level, tools in summary['tools_by_risk'].items():
            if tools:
                print(f"🎯 {risk_level.upper()} 風險工具: {len(tools)} 個")
        
        # 4. 測試權限提升
        print("\n4️⃣ **測試權限提升**")
        
        # 提升到 admin 級別
        from mcp_server_qdrant.permission_manager import PermissionLevel
        permission_manager.set_user_permission("default_user", PermissionLevel.ADMIN)
        new_level = permission_manager.get_user_permission("default_user")
        print(f"🔝 權限提升後級別: {new_level.value}")
        
        # 重新檢查權限
        admin_summary = permission_manager.get_permission_summary("default_user")
        print(f"📈 提升後可用工具數量: {admin_summary['total_available_tools']}")
        
        # 檢查現在能否使用管理工具
        admin_tools = ["qdrant-store", "request-schema-field-addition", "analyze-collection-for-migration"]
        for tool in admin_tools:
            has_permission = permission_manager.check_tool_permission("default_user", tool)
            status = "✅ 允許" if has_permission else "❌ 拒絕"
            print(f"  {tool}: {status}")
        
        # 5. 測試超級管理員權限
        print("\n5️⃣ **測試超級管理員權限**")
        
        permission_manager.set_user_permission("default_user", PermissionLevel.SUPER_ADMIN)
        super_level = permission_manager.get_user_permission("default_user")
        print(f"🌟 超級管理員級別: {super_level.value}")
        
        super_summary = permission_manager.get_permission_summary("default_user")
        print(f"🚀 超級管理員可用工具數量: {super_summary['total_available_tools']}")
        
        # 檢查現在能否使用所有危險工具
        super_tools = ["qdrant-delete-collection", "execute-data-migration", "qdrant-optimize-storage"]
        for tool in super_tools:
            has_permission = permission_manager.check_tool_permission("default_user", tool)
            status = "✅ 允許" if has_permission else "❌ 拒絕"
            print(f"  {tool}: {status}")
        
        print("\n🎯 **權限系統測試結果: 成功**")
        return True
        
    except Exception as e:
        print(f'❌ 測試失敗: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_permission_system())
    print(f'\n🏁 最終測試結果: {"成功" if success else "失敗"}')