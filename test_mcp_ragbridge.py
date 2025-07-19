#!/usr/bin/env python3
"""
測試 MCP 工具的 ragbridge_default collection
直接呼叫 MCP 工具函數來測試寫入和讀取
"""
import asyncio
import sys
from pathlib import Path

# 添加專案路徑到 sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv

# 載入 .env 文件
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ 載入 .env 文件: {env_path}")

from mcp_server_qdrant.mcp_server import QdrantMCPServer
from mcp_server_qdrant.settings import (
    QdrantSettings, 
    EmbeddingProviderSettings, 
    ToolSettings
)
from fastmcp import Context


async def test_mcp_tools():
    """測試 MCP 工具功能"""
    print("\n🧪 **測試 MCP 工具的 ragbridge_default Collection**")
    print("=" * 60)
    
    # 創建 MCP Server 實例
    mcp_server = QdrantMCPServer(
        tool_settings=ToolSettings(),
        qdrant_settings=QdrantSettings(),
        embedding_provider_settings=EmbeddingProviderSettings(),
    )
    
    # 創建模擬的 Context
    class MockContext:
        async def debug(self, message: str):
            print(f"DEBUG: {message}")
    
    ctx = MockContext()
    
    # 測試 1: 檢查配置
    print("\n📋 **步驟 1: 檢查 Collection 配置**")
    try:
        # 這裡直接呼叫內部函數，因為 MCP 工具在 mcp_server 中是 async def
        # 我們需要找到對應的工具函數
        
        # 檢查 collection 配置
        result = []
        from mcp_server_qdrant.dynamic_embedding_manager import get_dynamic_embedding_manager
        
        manager = get_dynamic_embedding_manager()
        configs = manager.list_collection_configs()
        
        for name, config in configs.items():
            print(f"   📁 {name}")
            print(f"      🤖 Provider: {config.embedding_provider.value}")
            print(f"      📝 Model: {config.embedding_model}")
            print(f"      🏷️ Vector Name: {config.vector_name}")
            print(f"      📏 Vector Size: {config.vector_size}")
        
    except Exception as e:
        print(f"   ❌ 配置檢查失敗: {e}")
    
    # 測試 2: 驗證 ragbridge_default 配置
    print("\n🔍 **步驟 2: 驗證 ragbridge_default 配置**")
    try:
        from mcp_server_qdrant.dynamic_embedding_manager import get_dynamic_embedding_manager
        
        manager = get_dynamic_embedding_manager()
        validation = manager.validate_collection_compatibility("ragbridge_default")
        
        print(f"   📁 Collection: {validation['collection_name']}")
        print(f"   ⚙️ 配置存在: {'✅' if validation['config_exists'] else '❌'}")
        print(f"   🔌 Provider 可用: {'✅' if validation['provider_available'] else '❌'}")
        
        if validation.get('actual_vector_name'):
            print(f"   🏷️ Vector Name: {validation['actual_vector_name']} "
                  f"({'✅' if validation['vector_name_match'] else '❌'})")
            print(f"   📏 Vector Size: {validation['actual_vector_size']} "
                  f"({'✅' if validation['vector_size_match'] else '❌'})")
        
        if validation['warnings']:
            for warning in validation['warnings']:
                print(f"   ⚠️ Warning: {warning}")
        
        if validation['errors']:
            for error in validation['errors']:
                print(f"   ❌ Error: {error}")
        
        print(f"   📋 總結: {'✅ 配置有效' if validation['is_valid'] else '❌ 配置有問題'}")
        
    except Exception as e:
        print(f"   ❌ 驗證失敗: {e}")
        import traceback
        traceback.print_exc()
    
    # 測試 3: 寫入測試資料到 ragbridge_default
    print("\n💾 **步驟 3: 寫入測試資料到 ragbridge_default**")
    try:
        from mcp_server_qdrant.collection_aware_qdrant import CollectionAwareQdrantConnector
        from mcp_server_qdrant.qdrant import Entry
        
        connector = CollectionAwareQdrantConnector(
            qdrant_url="http://localhost:6333",
            qdrant_api_key="33c2f5c514ea6c31f6558ea5c237bd1a",
            default_collection_name="ragbridge_default"
        )
        
        # 測試資料
        test_entries = [
            Entry(
                content="這是一個使用 Ollama nomic-embed-text 模型的測試文檔。",
                metadata={"type": "test", "model": "nomic-embed-text", "test_id": "1"}
            ),
            Entry(
                content="Ollama embedding test document for ragbridge collection validation.",
                metadata={"type": "test", "model": "nomic-embed-text", "test_id": "2"}
            ),
            Entry(
                content="測試 ragbridge_default collection 的中英文混合內容支援能力。",
                metadata={"type": "test", "model": "nomic-embed-text", "test_id": "3"}
            )
        ]
        
        for i, entry in enumerate(test_entries, 1):
            await connector.store(entry, collection_name="ragbridge_default")
            print(f"   ✅ 成功寫入測試資料 {i}")
        
        print(f"   📝 總共寫入 {len(test_entries)} 筆測試資料")
        
    except Exception as e:
        print(f"   ❌ 寫入失敗: {e}")
        import traceback
        traceback.print_exc()
    
    # 測試 4: 讀取和搜尋測試
    print("\n🔍 **步驟 4: 讀取和搜尋測試**")
    try:
        from mcp_server_qdrant.collection_aware_qdrant import CollectionAwareQdrantConnector
        
        connector = CollectionAwareQdrantConnector(
            qdrant_url="http://localhost:6333",
            qdrant_api_key="33c2f5c514ea6c31f6558ea5c237bd1a"
        )
        
        # 測試不同的搜尋查詢
        test_queries = [
            "nomic-embed-text",
            "測試文檔",
            "Ollama embedding",
            "中英文混合",
            "ragbridge collection"
        ]
        
        for query in test_queries:
            print(f"\n   🔎 搜尋查詢: '{query}'")
            
            results = await connector.search(
                query=query,
                collection_name="ragbridge_default",
                limit=3
            )
            
            print(f"      📊 找到 {len(results)} 個結果")
            
            for j, result in enumerate(results):
                preview = result.content[:80] + "..." if len(result.content) > 80 else result.content
                print(f"         {j+1}. {preview}")
                if result.metadata:
                    print(f"            Metadata: {result.metadata}")
    
    except Exception as e:
        print(f"   ❌ 搜尋失敗: {e}")
        import traceback
        traceback.print_exc()
    
    # 測試 5: 檢查 Collection 詳細資訊
    print("\n📊 **步驟 5: 檢查 ragbridge_default Collection 詳細資訊**")
    try:
        from mcp_server_qdrant.collection_aware_qdrant import CollectionAwareQdrantConnector
        
        connector = CollectionAwareQdrantConnector(
            qdrant_url="http://localhost:6333",
            qdrant_api_key="33c2f5c514ea6c31f6558ea5c237bd1a"
        )
        
        info = await connector.get_collection_info("ragbridge_default")
        
        if info:
            print(f"   📁 Collection: ragbridge_default")
            print(f"   📄 Documents: {info['points_count']:,}")
            print(f"   🔍 Indexed Vectors: {info['indexed_vectors_count']:,}")
            print(f"   📊 Status: {info['status']}")
            
            # 向量配置
            for vector_name, vector_config in info['vectors_config'].items():
                print(f"   🎯 Vector: {vector_name} ({vector_config.size}維, {vector_config.distance})")
            
            # Embedding 配置
            if 'embedding_config' in info:
                config = info['embedding_config']
                print(f"   🤖 Provider: {config['provider']}")
                print(f"   📝 Model: {config['model']}")
                print(f"   🏷️ Vector Name: {config['vector_name']}")
                print(f"   📏 Vector Size: {config['vector_size']}")
        else:
            print("   ❌ Collection 不存在或無法獲取資訊")
    
    except Exception as e:
        print(f"   ❌ 獲取資訊失敗: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """主測試函數"""
    print("🧪 **MCP ragbridge_default Collection 測試**")
    
    try:
        await test_mcp_tools()
        
    except Exception as e:
        print(f"\n❌ **測試過程中發生錯誤**: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🏁 **測試完成**")


if __name__ == "__main__":
    asyncio.run(main())