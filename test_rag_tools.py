#!/usr/bin/env python3
"""
測試 RAG Bridge 工具是否正常註冊和運行
"""
import asyncio
import os
from mcp_server_qdrant.mcp_server import QdrantMCPServer
from mcp_server_qdrant.settings import QdrantSettings, EmbeddingProviderSettings, ToolSettings


async def test_rag_tools():
    """測試 RAG Bridge 工具"""
    
    # 載入環境變數
    from dotenv import load_dotenv
    load_dotenv()
    
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
        print('✅ MCP Server 初始化成功')
        
        # 獲取所有工具
        tools = await server.get_tools()
        
        # 提取工具名稱（tools 是字典格式）
        tool_names = list(tools.keys())
        
        print(f'\n📋 總工具數: {len(tool_names)}')
        print(f'🔧 所有工具:')
        for name in sorted(tool_names):
            print(f'   - {name}')
        
        # 檢查 RAG Bridge 工具
        expected_rag_tools = ['search-experience', 'get-process-workflow', 'suggest-similar', 'update-experience']
        expected_vocab_tools = ['search-vocabulary', 'standardize-content', 'get-vocabulary-statistics', 'manage-fragment-vocabulary', 'propose-vocabulary']
        expected_schema_tools = ['get-current-schema', 'validate-schema-data', 'analyze-schema-usage', 'get-schema-suggestions', 'get-schema-evolution-history', 'request-schema-field-addition', 'request-schema-field-removal', 'list-pending-schema-requests', 'review-schema-request', 'get-schema-approval-history']
        
        found_rag_tools = [name for name in tool_names if name in expected_rag_tools]
        found_vocab_tools = [name for name in tool_names if name in expected_vocab_tools]
        found_schema_tools = [name for name in tool_names if name in expected_schema_tools]
        
        if found_rag_tools:
            print(f'\n✅ RAG Bridge 工具已成功註冊:')
            for tool in found_rag_tools:
                print(f'   - {tool}')
        else:
            print('\n❌ RAG Bridge 工具未找到')
            
        if found_vocab_tools:
            print(f'\n✅ 詞彙管理工具已成功註冊:')
            for tool in found_vocab_tools:
                print(f'   - {tool}')
        else:
            print('\n❌ 詞彙管理工具未找到')
            
        if found_schema_tools:
            print(f'\n✅ Schema 管理工具已成功註冊:')
            for tool in found_schema_tools:
                print(f'   - {tool}')
        else:
            print('\n❌ Schema 管理工具未找到')
            
        # 檢查 ragbridge_connector 是否初始化
        if hasattr(server, 'ragbridge_connector'):
            print('\n✅ RAG Bridge 連接器已初始化')
        else:
            print('\n❌ RAG Bridge 連接器未初始化')
            
        rag_success = len(found_rag_tools) == len(expected_rag_tools)
        vocab_success = len(found_vocab_tools) == len(expected_vocab_tools)
        schema_success = len(found_schema_tools) == len(expected_schema_tools)
        
        return rag_success and vocab_success and schema_success
        
    except Exception as e:
        print(f'❌ 測試失敗: {e}')
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_rag_tools())
    print(f'\n🎯 測試結果: {"成功" if success else "失敗"}')