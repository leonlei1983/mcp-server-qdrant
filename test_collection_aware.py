#!/usr/bin/env python3
"""
測試 Collection-Aware Qdrant 系統
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

from mcp_server_qdrant.collection_config import get_collection_config_manager
from mcp_server_qdrant.dynamic_embedding_manager import get_dynamic_embedding_manager
from mcp_server_qdrant.collection_aware_qdrant import CollectionAwareQdrantConnector
from mcp_server_qdrant.qdrant import Entry


async def test_collection_configs():
    """測試 collection 配置系統"""
    print("\n📋 **測試 Collection 配置系統**")
    
    manager = get_collection_config_manager()
    configs = manager.list_collections()
    
    for name, config in configs.items():
        print(f"   📁 {name}")
        print(f"      🤖 Provider: {config.embedding_provider.value}")
        print(f"      📝 Model: {config.embedding_model}")
        print(f"      🏷️ Vector Name: {config.vector_name}")
        print(f"      📏 Vector Size: {config.vector_size}")
        if config.ollama_base_url:
            print(f"      🌐 Ollama URL: {config.ollama_base_url}")


async def test_dynamic_embedding():
    """測試動態 embedding manager"""
    print("\n🤖 **測試動態 Embedding Manager**")
    
    manager = get_dynamic_embedding_manager()
    
    test_collections = ["default", "modern_docs"]
    
    for collection_name in test_collections:
        print(f"\n   🔧 測試 Collection: {collection_name}")
        
        try:
            # 獲取 provider
            provider = manager.get_provider(collection_name)
            print(f"      ✅ Provider: {provider.__class__.__name__}")
            
            # 獲取向量信息
            vector_name, vector_size = manager.get_vector_info(collection_name)
            print(f"      🏷️ Vector Name: {vector_name}")
            print(f"      📏 Vector Size: {vector_size}")
            
            # 驗證兼容性
            validation = manager.validate_collection_compatibility(collection_name)
            print(f"      🔍 Valid: {'✅' if validation['is_valid'] else '❌'}")
            
            if validation['warnings']:
                for warning in validation['warnings']:
                    print(f"      ⚠️ Warning: {warning}")
            
            if validation['errors']:
                for error in validation['errors']:
                    print(f"      ❌ Error: {error}")
            
        except Exception as e:
            print(f"      ❌ 失敗: {e}")


async def test_collection_aware_connector():
    """測試 collection-aware connector"""
    print("\n🔗 **測試 Collection-Aware Connector**")
    
    connector = CollectionAwareQdrantConnector(
        qdrant_url="http://localhost:6333",
        qdrant_api_key="33c2f5c514ea6c31f6558ea5c237bd1a",
        default_collection_name="default"
    )
    
    # 測試存儲到不同 collections
    test_data = [
        ("default", "這是一個使用 FastEmbed 的測試文檔"),
        ("modern_docs", "This is a test document using nomic-embed-text"),
    ]
    
    for collection_name, content in test_data:
        print(f"\n   📝 測試存儲到 {collection_name}")
        
        try:
            # 創建測試條目
            entry = Entry(
                content=content,
                metadata={"test": True, "collection": collection_name}
            )
            
            # 存儲（僅在 collection 存在時）
            collections = await connector.get_collection_names()
            if collection_name in collections or collection_name == "default":
                await connector.store(entry, collection_name=collection_name)
                print(f"      ✅ 成功存儲到 {collection_name}")
            else:
                print(f"      ⚠️ Collection {collection_name} 不存在，跳過存儲測試")
            
            # 測試搜尋
            results = await connector.search(
                "test document",
                collection_name=collection_name,
                limit=3
            )
            
            print(f"      🔍 搜尋結果: {len(results)} 個")
            if results:
                for i, result in enumerate(results[:2]):
                    preview = result.content[:50] + "..." if len(result.content) > 50 else result.content
                    print(f"         {i+1}. {preview}")
            
        except Exception as e:
            print(f"      ❌ 測試失敗: {e}")
            import traceback
            traceback.print_exc()


async def test_collection_info():
    """測試 collection 信息獲取"""
    print("\n📊 **測試 Collection 信息獲取**")
    
    connector = CollectionAwareQdrantConnector(
        qdrant_url="http://localhost:6333",
        qdrant_api_key="33c2f5c514ea6c31f6558ea5c237bd1a"
    )
    
    # 獲取所有 collections
    collections = await connector.get_collection_names()
    print(f"   📁 找到 {len(collections)} 個 collections: {collections}")
    
    # 測試每個 collection 的詳細信息
    for collection_name in collections[:3]:  # 只測試前3個
        print(f"\n   🔍 Collection: {collection_name}")
        
        try:
            info = await connector.get_collection_info(collection_name)
            if info:
                print(f"      📄 Documents: {info['points_count']:,}")
                print(f"      📊 Status: {info['status']}")
                
                # 向量配置
                for vector_name, vector_config in info['vectors_config'].items():
                    print(f"      🎯 {vector_name}: {vector_config.size}維")
                
                # Embedding 配置
                if 'embedding_config' in info:
                    config = info['embedding_config']
                    print(f"      🤖 Provider: {config['provider']}")
                    print(f"      📝 Model: {config['model']}")
            else:
                print(f"      ❌ 無法獲取信息")
                
        except Exception as e:
            print(f"      ❌ 獲取信息失敗: {e}")


async def main():
    """主測試函數"""
    print("🧪 **Collection-Aware Qdrant 系統測試**")
    print("=" * 60)
    
    try:
        await test_collection_configs()
        await test_dynamic_embedding()
        await test_collection_aware_connector()
        await test_collection_info()
        
    except Exception as e:
        print(f"\n❌ **測試過程中發生錯誤**: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🏁 **測試完成**")


if __name__ == "__main__":
    asyncio.run(main())