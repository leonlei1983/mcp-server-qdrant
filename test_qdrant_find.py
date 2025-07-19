#!/usr/bin/env python3
"""
測試 qdrant-find 功能的獨立程式
用來診斷為什麼 MCP 工具無法找到資料
"""
import asyncio
import os
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
else:
    print(f"❌ 未找到 .env 文件: {env_path}")

from mcp_server_qdrant.settings import QdrantSettings, EmbeddingProviderSettings
from mcp_server_qdrant.qdrant import QdrantConnector
from mcp_server_qdrant.embeddings.factory import create_embedding_provider


async def test_qdrant_connection():
    """測試 Qdrant 連接"""
    print("\n🔧 **測試 Qdrant 連接**")
    
    # 初始化設定
    qdrant_settings = QdrantSettings()
    embedding_settings = EmbeddingProviderSettings()
    
    print(f"   📍 Qdrant URL: {qdrant_settings.location}")
    print(f"   🔑 API Key: {'已設置' if qdrant_settings.api_key else '未設置'}")
    print(f"   📦 Collection: {qdrant_settings.collection_name}")
    print(f"   🔍 Search Limit: {qdrant_settings.search_limit}")
    
    # 建立 embedding provider
    embedding_provider = create_embedding_provider(embedding_settings)
    print(f"   🤖 Embedding Model: {embedding_settings.model_name}")
    print(f"   🏷️ Vector Name: {embedding_provider.get_vector_name()}")
    print(f"   📏 Vector Size: {embedding_provider.get_vector_size()}")
    
    # 建立 Qdrant connector
    qdrant_connector = QdrantConnector(
        qdrant_url=qdrant_settings.location,
        qdrant_api_key=qdrant_settings.api_key,
        collection_name=qdrant_settings.collection_name,
        embedding_provider=embedding_provider,
        qdrant_local_path=qdrant_settings.local_path,
    )
    
    return qdrant_connector, qdrant_settings


async def test_collection_info(connector):
    """測試 collection 資訊"""
    print("\n📊 **測試 Collection 資訊**")
    
    try:
        # 檢查多個 collections
        collections_to_check = ["default", "ragbridge_default"]
        
        for coll_name in collections_to_check:
            exists = await connector._client.collection_exists(coll_name)
            print(f"   📁 {coll_name} collection 存在: {exists}")
            
            if exists:
                # 取得 collection 資訊
                collection_info = await connector._client.get_collection(coll_name)
                print(f"      📈 Points count: {collection_info.points_count}")
                print(f"      🔧 Vectors config: {collection_info.config.params.vectors}")
        
        # 主要測試 ragbridge_default（MCP 預設使用的）
        return await connector._client.collection_exists("ragbridge_default")
    except Exception as e:
        print(f"   ❌ 檢查 collection 失敗: {e}")
        return False


async def test_direct_search(connector):
    """直接測試 Qdrant 搜尋"""
    print("\n🔍 **直接測試 Qdrant 搜尋**")
    
    try:
        from qdrant_client import models
        
        # 取得 embedding provider
        embedding_provider = connector._embedding_provider
        
        # 測試查詢
        test_queries = ["CoinAgent", "Vue", "SettingsIcon", "遊戲", "test"]
        
        for query in test_queries:
            print(f"\n   🔎 查詢: '{query}'")
            
            # 生成查詢向量
            query_vector = await embedding_provider.embed_query(query)
            vector_name = embedding_provider.get_vector_name()
            
            print(f"      🏷️ Vector name: {vector_name}")
            print(f"      📏 Vector length: {len(query_vector)}")
            print(f"      🔢 First few values: {query_vector[:3]}")
            
            # 執行搜尋 - 測試兩個 collections
            for test_collection in ["default", "ragbridge_default"]:
                collection_exists = await connector._client.collection_exists(test_collection)
                if not collection_exists:
                    print(f"      ⚠️ {test_collection} collection 不存在，跳過")
                    continue
                    
                print(f"      🔍 在 {test_collection} 中搜尋...")
                search_results = await connector._client.query_points(
                    collection_name=test_collection,
                    query=query_vector,
                    using=vector_name,
                    limit=5,
                )
                
                print(f"      📊 在 {test_collection} 找到結果數: {len(search_results.points)}")
                
                if search_results.points:
                    for i, point in enumerate(search_results.points):
                        score = point.score
                        content = point.payload.get("document", "")[:100]
                        print(f"         {i+1}. Score: {score:.4f} | Content: {content}...")
            else:
                print("      ❌ 無結果")
                
                # 嘗試查看是否有該向量名稱的問題
                print(f"      🔧 檢查向量名稱是否正確...")
                collection_info = await connector._client.get_collection("default")
                available_vectors = list(collection_info.config.params.vectors.keys())
                print(f"      📋 可用的向量名稱: {available_vectors}")
                
                if vector_name not in available_vectors and available_vectors:
                    print(f"      ⚠️ 向量名稱不匹配！嘗試使用: {available_vectors[0]}")
                    # 用正確的向量名稱重試
                    retry_results = await connector._client.query_points(
                        collection_name="default",
                        query=query_vector,
                        using=available_vectors[0],
                        limit=5,
                    )
                    print(f"      🔄 重試結果數: {len(retry_results.points)}")
            
    except Exception as e:
        print(f"   ❌ 搜尋測試失敗: {e}")
        import traceback
        traceback.print_exc()


async def test_connector_search(connector):
    """測試 QdrantConnector 的 search 方法"""
    print("\n🔧 **測試 QdrantConnector.search 方法**")
    
    test_queries = ["CoinAgent", "Vue", "SettingsIcon", "test"]
    
    for query in test_queries:
        print(f"\n   🔎 查詢: '{query}'")
        try:
            entries = await connector.search(
                query=query,
                collection_name="default",
                limit=5
            )
            
            print(f"      📊 找到 entries 數: {len(entries)}")
            
            if entries:
                for i, entry in enumerate(entries):
                    content = entry.content[:100] if entry.content else ""
                    print(f"      {i+1}. Content: {content}...")
            else:
                print("      ❌ 無結果")
                
        except Exception as e:
            print(f"      ❌ QdrantConnector.search 失敗: {e}")
            import traceback
            traceback.print_exc()


async def main():
    """主測試函數"""
    print("🧪 **Qdrant Find 功能測試程式**")
    print("=" * 50)
    
    try:
        # 測試連接
        connector, settings = await test_qdrant_connection()
        
        # 測試 collection 資訊
        collection_exists = await test_collection_info(connector)
        
        if collection_exists:
            # 直接測試搜尋
            await test_direct_search(connector)
            
            # 測試 connector 搜尋方法
            await test_connector_search(connector)
        else:
            print("❌ default collection 不存在，無法進行搜尋測試")
            
    except Exception as e:
        print(f"\n❌ **測試過程中發生錯誤**: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🏁 **測試完成**")


if __name__ == "__main__":
    asyncio.run(main())