#!/usr/bin/env python3
"""
測試 Schema-Aware 存儲工具
驗證所有新的 store-* MCP tools 是否正常工作
"""

import asyncio
import uuid
from datetime import datetime
from pathlib import Path
import sys
import os

# 添加項目路徑到 Python 路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv

# 載入環境變數
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ 載入 .env 文件: {env_path}")
else:
    print(f"⚠️ 未找到 .env 文件: {env_path}")

from mcp_server_qdrant.collection_aware_qdrant import CollectionAwareQdrantConnector
from mcp_server_qdrant.dynamic_embedding_manager import DynamicEmbeddingManager
from mcp_server_qdrant.ragbridge.models import (
    RAGEntry, RAGMetadata, ExperienceContent, ProcessWorkflowContent,
    KnowledgeBaseContent, DecisionRecordContent, ContentType, ContentStatus
)


async def test_schema_aware_storage():
    """測試所有 Schema-Aware 存儲功能"""
    
    print("🧪 **測試 Schema-Aware 存儲工具**")
    print("=" * 60)
    
    # 初始化組件
    from mcp_server_qdrant.ragbridge.connector import RAGBridgeConnector
    from mcp_server_qdrant.embeddings.factory import create_embedding_provider
    from mcp_server_qdrant.settings import EmbeddingProviderSettings
    
    # 創建 Ollama embedding provider for testing
    os.environ["EMBEDDING_PROVIDER"] = "ollama"
    os.environ["EMBEDDING_MODEL"] = "nomic-embed-text"
    os.environ["OLLAMA_BASE_URL"] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    embedding_settings = EmbeddingProviderSettings()
    embedding_provider = create_embedding_provider(embedding_settings)
    
    # 使用 RAGBridge 連接器進行測試
    connector = RAGBridgeConnector(
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        embedding_provider=embedding_provider,
        qdrant_local_path=os.getenv("QDRANT_LOCAL_PATH"),
        default_collection_prefix="ragbridge"
    )
    
    test_results = []
    
    # 測試 1: store-experience 等效功能
    print("\n📝 **測試 1: Experience 存儲**")
    try:
        content_id = str(uuid.uuid4())  # 使用完整 UUID 作為 content_id
        now = datetime.now()
        
        metadata = RAGMetadata(
            content_type=ContentType.EXPERIENCE,
            content_id=content_id,
            title="測試經驗記錄：Collection-Aware 系統實作",
            tags=["test", "collection-aware", "embedding"],
            categories=["development", "system-design"],
            created_at=now,
            updated_at=now,
            status=ContentStatus.ACTIVE,
            custom_fields={"mandatory_tags": ["test", "collection-aware", "embedding"]}
        )
        
        structured_content = ExperienceContent(
            problem_description="需要實作支援多 embedding 模型的 Qdrant 系統",
            solution_approach="建立 Collection-Aware 架構，動態選擇 embedding provider",
            implementation_details="使用 DynamicEmbeddingManager 和 CollectionConfigManager",
            outcomes="成功實作多模型支援，ragbridge_default 使用 ollama-nomic-embed-text",
            lessons_learned="動態配置比靜態配置更靈活，但需要良好的錯誤處理",
            technologies_used=["Qdrant", "FastEmbed", "Ollama", "Python", "Pydantic"],
            difficulty_level="hard",
            time_invested="8 hours",
            confidence_level=0.9,
            reusability_score=0.8
        )
        
        entry = RAGEntry(
            content="實作 Collection-Aware Qdrant 系統的完整經驗記錄，包含問題分析、解決方案設計、實作細節和成果評估。",
            metadata=metadata,
            structured_content=structured_content,
            search_keywords=["collection-aware", "qdrant", "embedding", "ollama", "ragbridge"]
        )
        
        # 使用 Collection-Aware 連接器存儲
        result_id = await connector.store_rag_entry(entry)
        
        print(f"✅ Experience 存儲成功")
        print(f"   ID: {content_id}")
        print(f"   Collection: ragbridge_experience")
        print(f"   Qdrant ID: {result_id}")
        test_results.append(("Experience Storage", True, "成功"))
        
    except Exception as e:
        print(f"❌ Experience 存儲失敗: {e}")
        test_results.append(("Experience Storage", False, str(e)))
    
    # 測試 2: store-process-workflow 等效功能
    print("\n⚙️ **測試 2: Process Workflow 存儲**")
    try:
        content_id = str(uuid.uuid4())  # 使用完整 UUID
        now = datetime.now()
        
        metadata = RAGMetadata(
            content_type=ContentType.PROCESS_WORKFLOW,
            content_id=content_id,
            title="測試流程：Schema-Aware 工具實作流程",
            tags=["test", "workflow", "schema-aware"],
            categories=["development", "process"],
            created_at=now,
            updated_at=now,
            status=ContentStatus.ACTIVE,
            custom_fields={"mandatory_tags": ["test", "workflow", "schema-aware"]}
        )
        
        steps = [
            {"step": 1, "action": "分析需求", "description": "分析 Schema-Aware 存儲需求"},
            {"step": 2, "action": "設計架構", "description": "設計 RAG Bridge 模型和工具架構"},
            {"step": 3, "action": "實作工具", "description": "實作 store-* 系列 MCP tools"},
            {"step": 4, "action": "測試驗證", "description": "建立測試程式驗證功能"}
        ]
        
        structured_content = ProcessWorkflowContent(
            process_name="Schema-Aware 工具實作流程",
            process_description="從需求分析到測試驗證的完整開發流程",
            steps=steps,
            process_type="manual",
            estimated_duration="12 hours",
            prerequisites=["RAG Bridge 基礎架構", "Collection-Aware 系統"],
            success_criteria=["所有工具正常運作", "Schema 驗證通過", "測試覆蓋完整"],
            common_pitfalls=["Schema 不匹配", "權限設定錯誤", "Collection 配置問題"],
            validation_steps=["單元測試", "整合測試", "端到端測試"]
        )
        
        entry = RAGEntry(
            content="Schema-Aware 存儲工具的完整實作流程，包含需求分析、架構設計、實作和測試的標準化步驟。",
            metadata=metadata,
            structured_content=structured_content,
            search_keywords=["schema-aware", "workflow", "development", "mcp-tools"]
        )
        
        result_id = await connector.store_rag_entry(entry)
        
        print(f"✅ Process Workflow 存儲成功")
        print(f"   ID: {content_id}")
        print(f"   Collection: ragbridge_process_workflow")
        print(f"   步驟數: {len(steps)}")
        print(f"   Qdrant ID: {result_id}")
        test_results.append(("Process Workflow Storage", True, "成功"))
        
    except Exception as e:
        print(f"❌ Process Workflow 存儲失敗: {e}")
        test_results.append(("Process Workflow Storage", False, str(e)))
    
    # 測試 3: store-knowledge-base 等效功能
    print("\n📚 **測試 3: Knowledge Base 存儲**")
    try:
        content_id = str(uuid.uuid4())  # 使用完整 UUID
        now = datetime.now()
        
        metadata = RAGMetadata(
            content_type=ContentType.KNOWLEDGE_BASE,
            content_id=content_id,
            title="測試知識：Ollama Nomic-Embed-Text 模型特性",
            tags=["test", "ollama", "embedding", "knowledge"],
            categories=["ai-models", "embedding"],
            created_at=now,
            updated_at=now,
            status=ContentStatus.ACTIVE,
            custom_fields={"mandatory_tags": ["test", "ollama", "embedding"]}
        )
        
        structured_content = KnowledgeBaseContent(
            topic="Ollama Nomic-Embed-Text 模型",
            content="Nomic-Embed-Text 是一個開源的文本嵌入模型，支援多語言處理，向量維度為 768，適合語義搜索和 RAG 應用。",
            summary="開源多語言文本嵌入模型，768 維向量，適合 RAG 應用",
            knowledge_type="factual",
            complexity_level="intermediate",
            references=[
                "Nomic AI Official Documentation",
                "Ollama Model Library"
            ],
            external_links=[
                "https://ollama.ai/library/nomic-embed-text"
            ]
        )
        
        entry = RAGEntry(
            content="Ollama Nomic-Embed-Text 模型的詳細技術知識，包含模型特性、應用場景和技術規格。",
            metadata=metadata,
            structured_content=structured_content,
            search_keywords=["ollama", "nomic-embed-text", "embedding", "768", "multilingual"]
        )
        
        result_id = await connector.store_rag_entry(entry)
        
        print(f"✅ Knowledge Base 存儲成功")
        print(f"   ID: {content_id}")
        print(f"   Collection: ragbridge_knowledge_base")
        print(f"   主題: {structured_content.topic}")
        print(f"   複雜度: {structured_content.complexity_level}")
        print(f"   Qdrant ID: {result_id}")
        test_results.append(("Knowledge Base Storage", True, "成功"))
        
    except Exception as e:
        print(f"❌ Knowledge Base 存儲失敗: {e}")
        test_results.append(("Knowledge Base Storage", False, str(e)))
    
    # 測試 4: store-decision-record 等效功能
    print("\n⚖️ **測試 4: Decision Record 存儲**")
    try:
        content_id = str(uuid.uuid4())  # 使用完整 UUID
        now = datetime.now()
        
        metadata = RAGMetadata(
            content_type=ContentType.DECISION_RECORD,
            content_id=content_id,
            title="測試決策：選擇 Ollama Nomic-Embed-Text 作為新的 Embedding 模型",
            tags=["test", "decision", "embedding-model"],
            categories=["architecture", "ai-models"],
            created_at=now,
            updated_at=now,
            status=ContentStatus.ACTIVE,
            custom_fields={"mandatory_tags": ["test", "decision", "embedding-model"]}
        )
        
        structured_content = DecisionRecordContent(
            decision_title="選擇 Ollama Nomic-Embed-Text 作為新的 Embedding 模型",
            decision_description="為 RAG Bridge 系統選擇更先進的 embedding 模型以提升搜索效果",
            context="原有的 FastEmbed all-MiniLM-L6-v2 模型只有 384 維，無法滿足複雜語義搜索需求",
            alternatives_considered=[
                "繼續使用 FastEmbed all-MiniLM-L6-v2",
                "升級到 OpenAI text-embedding-ada-002",
                "使用 Sentence Transformers 大型模型"
            ],
            decision_rationale="Ollama Nomic-Embed-Text 提供 768 維向量，支援多語言，開源免費，適合本地部署",
            consequences="需要重建 collections，增加計算資源需求，但搜索精度顯著提升",
            decision_status="active",
            decision_date="2025-07-19",
            stakeholders=["開發團隊", "AI 工程師"],
            impact_level="high",
            affected_systems=["RAG Bridge", "Qdrant Collections", "MCP Server"],
            implementation_notes="需要實作 Collection-Aware 系統支援多模型並存",
            review_date="2025-10-19",
            success_metrics=[
                "搜索精度提升 > 20%",
                "多語言支援改善",
                "系統穩定性保持"
            ]
        )
        
        entry = RAGEntry(
            content="關於選擇 Ollama Nomic-Embed-Text 作為新 embedding 模型的完整決策記錄，包含背景、分析、決策和實施計劃。",
            metadata=metadata,
            structured_content=structured_content,
            search_keywords=["decision", "ollama", "nomic-embed-text", "embedding-model", "architecture"]
        )
        
        result_id = await connector.store_rag_entry(entry)
        
        print(f"✅ Decision Record 存儲成功")
        print(f"   ID: {content_id}")
        print(f"   Collection: ragbridge_decision_record")
        print(f"   決策狀態: {structured_content.decision_status}")
        print(f"   影響程度: {structured_content.impact_level}")
        print(f"   Qdrant ID: {result_id}")
        test_results.append(("Decision Record Storage", True, "成功"))
        
    except Exception as e:
        print(f"❌ Decision Record 存儲失敗: {e}")
        test_results.append(("Decision Record Storage", False, str(e)))
    
    # 測試摘要
    print("\n📊 **測試結果摘要**")
    print("=" * 60)
    
    success_count = sum(1 for _, success, _ in test_results if success)
    total_count = len(test_results)
    
    for test_name, success, message in test_results:
        status = "✅" if success else "❌"
        print(f"{status} {test_name}: {message}")
    
    print(f"\n🎯 **總體結果**: {success_count}/{total_count} 測試通過")
    
    if success_count == total_count:
        print("🎉 所有 Schema-Aware 存儲工具測試通過！")
        return True
    else:
        print("⚠️ 部分測試失敗，需要檢查問題。")
        return False


if __name__ == "__main__":
    result = asyncio.run(test_schema_aware_storage())
    sys.exit(0 if result else 1)