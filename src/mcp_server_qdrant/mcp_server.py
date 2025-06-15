import json
import logging
from datetime import datetime
from typing import Annotated, Any

from fastmcp import Context, FastMCP
from pydantic import Field
from qdrant_client import models

from mcp_server_qdrant.common.filters import make_indexes
from mcp_server_qdrant.common.func_tools import make_partial_function
from mcp_server_qdrant.common.wrap_filters import wrap_filters
from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.qdrant import ArbitraryFilter, Entry, Metadata, QdrantConnector
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    QdrantSettings,
    ToolSettings,
)
from mcp_server_qdrant.system_monitor import UniversalQdrantMonitor
from mcp_server_qdrant.storage_optimizer import QdrantStorageOptimizer

logger = logging.getLogger(__name__)


# FastMCP is an alternative interface for declaring the capabilities
# of the server. Its API is based on FastAPI.
class QdrantMCPServer(FastMCP):
    """
    A MCP server for Qdrant.
    """

    def __init__(
        self,
        tool_settings: ToolSettings,
        qdrant_settings: QdrantSettings,
        embedding_provider_settings: EmbeddingProviderSettings,
        name: str = "mcp-server-qdrant",
        instructions: str | None = None,
        **settings: Any,
    ):
        self.tool_settings = tool_settings
        self.qdrant_settings = qdrant_settings
        self.embedding_provider_settings = embedding_provider_settings

        self.embedding_provider = create_embedding_provider(embedding_provider_settings)
        self.qdrant_connector = QdrantConnector(
            qdrant_settings.location,
            qdrant_settings.api_key,
            qdrant_settings.collection_name,
            self.embedding_provider,
            qdrant_settings.local_path,
            make_indexes(qdrant_settings.filterable_fields_dict()),
        )
        
        # 初始化通用系統監控器
        self.system_monitor = UniversalQdrantMonitor(
            self.qdrant_connector._client,
            qdrant_settings.location
        )
        
        # 初始化儲存優化工具
        self.storage_optimizer = QdrantStorageOptimizer(
            self.qdrant_connector._client
        )

        super().__init__(name=name, instructions=instructions, **settings)

        self.setup_tools()

    def format_entry(self, entry: Entry) -> str:
        """
        Feel free to override this method in your subclass to customize the format of the entry.
        """
        entry_metadata = json.dumps(entry.metadata) if entry.metadata else ""
        return f"<entry><content>{entry.content}</content><metadata>{entry_metadata}</metadata></entry>"

    def setup_tools(self):
        """
        Register the tools in the server.
        """

        async def store(
            ctx: Context,
            information: Annotated[str, Field(description="Text to store")],
            collection_name: Annotated[
                str, Field(description="The collection to store the information in")
            ] = "default",
            # The `metadata` parameter is defined as non-optional, but it can be None.
            # If we set it to be optional, some of the MCP clients, like Cursor, cannot
            # handle the optional parameter correctly.
            metadata: Annotated[
                Metadata | None,
                Field(
                    description="Extra metadata stored along with memorised information. Any json is accepted."
                ),
            ] = None,
        ) -> str:
            """
            Store some information in Qdrant.
            :param ctx: The context for the request.
            :param information: The information to store.
            :param metadata: JSON metadata to store with the information, optional.
            :param collection_name: The name of the collection to store the information in, optional. If not provided,
                                    the default collection is used.
            :return: A message indicating that the information was stored.
            """
            await ctx.debug(f"Storing information {information} in Qdrant")

            entry = Entry(content=information, metadata=metadata)

            await self.qdrant_connector.store(entry, collection_name=collection_name)
            if collection_name:
                return f"Remembered: {information} in collection {collection_name}"
            return f"Remembered: {information}"

        async def find(
            ctx: Context,
            query: Annotated[str, Field(description="What to search for")],
            collection_name: Annotated[
                str, Field(description="The collection to search in")
            ] = "default",
            query_filter: ArbitraryFilter | None = None,
        ) -> list[str]:
            """
            Find memories in Qdrant.
            :param ctx: The context for the request.
            :param query: The query to use for the search.
            :param collection_name: The name of the collection to search in, optional. If not provided,
                                    the default collection is used.
            :param query_filter: The filter to apply to the query.
            :return: A list of entries found.
            """

            # Log query_filter
            await ctx.debug(f"Query filter: {query_filter}")

            query_filter = models.Filter(**query_filter) if query_filter else None

            await ctx.debug(f"Finding results for query {query}")

            entries = await self.qdrant_connector.search(
                query,
                collection_name=collection_name,
                limit=self.qdrant_settings.search_limit,
                query_filter=query_filter,
            )
            if not entries:
                return [f"No information found for the query '{query}'"]
            content = [
                f"Results for the query '{query}'",
            ]
            for entry in entries:
                content.append(self.format_entry(entry))
            return content

        find_foo = find
        store_foo = store

        filterable_conditions = (
            self.qdrant_settings.filterable_fields_dict_with_conditions()
        )

        if len(filterable_conditions) > 0:
            find_foo = wrap_filters(find_foo, filterable_conditions)
        elif not self.qdrant_settings.allow_arbitrary_filter:
            find_foo = make_partial_function(find_foo, {"query_filter": None})

        if self.qdrant_settings.collection_name:
            find_foo = make_partial_function(
                find_foo, {"collection_name": self.qdrant_settings.collection_name}
            )
            store_foo = make_partial_function(
                store_foo, {"collection_name": self.qdrant_settings.collection_name}
            )

        self.tool(
            find_foo,
            name="qdrant-find",
            description=self.tool_settings.tool_find_description,
        )

        if not self.qdrant_settings.read_only:
            # Those methods can modify the database
            self.tool(
                store_foo,
                name="qdrant-store",
                description=self.tool_settings.tool_store_description,
            )

            # 新增刪除文檔工具
            async def delete_documents(
                ctx: Context,
                query: Annotated[str, Field(description="搜尋要刪除的文檔")],
                collection_name: Annotated[
                    str, Field(description="要刪除文檔的 collection 名稱")
                ] = "default",
                confirm_delete: Annotated[
                    bool, Field(description="確認刪除操作，必須設為 True")
                ] = False,
                limit: Annotated[
                    int, Field(description="最多刪除的文檔數量")
                ] = 10,
            ) -> str:
                """
                刪除符合搜尋條件的文檔。需要明確確認才能執行。
                """
                await ctx.debug(f"Deleting documents matching '{query}' in collection '{collection_name}'")
                
                result = await self.qdrant_connector.delete_documents(
                    query=query,
                    collection_name=collection_name,
                    limit=limit,
                    confirm_delete=confirm_delete,
                )
                
                if "error" in result:
                    return f"錯誤: {result['error']}"
                
                return f"{result['message']} (在 collection '{result['collection_name']}')"

            # 新增刪除 collection 工具
            async def delete_collection(
                ctx: Context,
                collection_name: Annotated[
                    str, Field(description="要刪除的 collection 名稱")
                ] = "default",
                confirm_delete: Annotated[
                    bool, Field(description="確認刪除操作，必須設為 True")
                ] = False,
            ) -> str:
                """
                刪除整個 collection 及其所有文檔。需要明確確認才能執行。
                """
                await ctx.debug(f"Deleting collection '{collection_name}'")
                
                result = await self.qdrant_connector.delete_collection(
                    collection_name=collection_name,
                    confirm_delete=confirm_delete,
                )
                
                if "error" in result:
                    return f"錯誤: {result['error']}"
                
                return result["message"]

            # 新增搬移文檔工具
            async def move_documents(
                ctx: Context,
                query: Annotated[str, Field(description="搜尋要搬移的文檔")],
                source_collection: Annotated[
                    str, Field(description="來源 collection 名稱")
                ] = "default",
                target_collection: Annotated[
                    str, Field(description="目標 collection 名稱")
                ] = "default",
                confirm_move: Annotated[
                    bool, Field(description="確認搬移操作，必須設為 True")
                ] = False,
                limit: Annotated[
                    int, Field(description="最多搬移的文檔數量")
                ] = 10,
            ) -> str:
                """
                搬移文檔從一個 collection 到另一個 collection。需要明確確認才能執行。
                """
                await ctx.debug(f"Moving documents matching '{query}' from '{source_collection}' to '{target_collection}'")
                
                result = await self.qdrant_connector.move_documents(
                    query=query,
                    source_collection=source_collection,
                    target_collection=target_collection,
                    limit=limit,
                    confirm_move=confirm_move,
                )
                
                return f"{result['message']}"

            # 新增更新 metadata 工具
            async def update_metadata(
                ctx: Context,
                query: Annotated[str, Field(description="搜尋要更新的文檔")],
                new_metadata: Annotated[
                    dict, Field(description="新的 metadata，會與現有 metadata 合併")
                ],
                collection_name: Annotated[
                    str, Field(description="要更新文檔的 collection 名稱")
                ] = "default",
                confirm_update: Annotated[
                    bool, Field(description="確認更新操作，必須設為 True")
                ] = False,
                limit: Annotated[
                    int, Field(description="最多更新的文檔數量")
                ] = 10,
            ) -> str:
                """
                更新符合搜尋條件的文檔 metadata。需要明確確認才能執行。
                """
                await ctx.debug(f"Updating metadata for documents matching '{query}' in collection '{collection_name}'")
                
                result = await self.qdrant_connector.update_metadata(
                    query=query,
                    new_metadata=new_metadata,
                    collection_name=collection_name,
                    limit=limit,
                    confirm_update=confirm_update,
                )
                
                if "error" in result:
                    return f"錯誤: {result['error']}"
                
                return result["message"]

            # 新增移除 metadata 鍵工具
            async def remove_metadata_keys(
                ctx: Context,
                query: Annotated[str, Field(description="搜尋要處理的文檔")],
                keys_to_remove: Annotated[
                    list[str], Field(description="要移除的 metadata 鍵列表")
                ],
                collection_name: Annotated[
                    str, Field(description="要處理文檔的 collection 名稱")
                ] = "default",
                confirm_update: Annotated[
                    bool, Field(description="確認更新操作，必須設為 True")
                ] = False,
                limit: Annotated[
                    int, Field(description="最多處理的文檔數量")
                ] = 10,
            ) -> str:
                """
                從符合搜尋條件的文檔 metadata 中移除指定的鍵。需要明確確認才能執行。
                """
                await ctx.debug(f"Removing metadata keys {keys_to_remove} from documents matching '{query}' in collection '{collection_name}'")
                
                result = await self.qdrant_connector.remove_metadata_keys(
                    query=query,
                    keys_to_remove=keys_to_remove,
                    collection_name=collection_name,
                    limit=limit,
                    confirm_update=confirm_update,
                )
                
                if "error" in result:
                    return f"錯誤: {result['error']}"
                
                return result["message"]

            # 註冊新工具
            self.tool(
                delete_documents,
                name="qdrant-delete-documents",
                description="刪除符合搜尋條件的文檔。需要明確確認才能執行。",
            )
            
            self.tool(
                delete_collection,
                name="qdrant-delete-collection", 
                description="刪除整個 collection 及其所有文檔。需要明確確認才能執行。",
            )
            
            self.tool(
                move_documents,
                name="qdrant-move-documents",
                description="搬移文檔從一個 collection 到另一個 collection。需要明確確認才能執行。",
            )

            # 重新啟用，已修正 PointStruct 向量資料問題
            self.tool(
                update_metadata,
                name="qdrant-update-metadata",
                description="更新符合搜尋條件的文檔 metadata。需要明確確認才能執行。",
            )
            
            self.tool(
                remove_metadata_keys,
                name="qdrant-remove-metadata-keys",
                description="從符合搜尋條件的文檔 metadata 中移除指定的鍵。需要明確確認才能執行。",
            )

        # 新增列出 collections 工具（只讀操作）
        async def list_collections(ctx: Context) -> list[str]:
            """
            列出所有可用的 collections。
            """
            await ctx.debug("Listing all collections")
            collections = await self.qdrant_connector.list_collections()
            return collections

        self.tool(
            list_collections,
            name="qdrant-list-collections",
            description="列出所有可用的 collections。",
        )

        # 純 Qdrant API 監控工具
        async def get_qdrant_status(ctx: Context) -> list[str]:
            """
            獲取 Qdrant 的狀態資訊，僅基於 Qdrant 自身的 API。
            """
            await ctx.debug("Getting Qdrant status via pure API")
            try:
                report = await self.system_monitor.get_comprehensive_analysis()
                
                # 格式化報告為可讀格式
                result = ["🔍 **Qdrant 狀態分析 (純 API 版本)**"]
                result.append(f"📅 分析時間: {report['timestamp']}")
                result.append("")
                
                # 部署環境資訊
                deployment = report.get('deployment_info', {})
                result.append("🏗️ **部署環境**")
                result.append(f"   類型: {deployment.get('description', 'unknown')}")
                result.append(f"   主機: {deployment.get('host', 'unknown')}")
                result.append(f"   端口: {deployment.get('port', 'unknown')}")
                result.append(f"   可用功能: {', '.join(deployment.get('features', []))}")
                result.append("")
                
                # Qdrant 健康狀態
                health = report.get('health_status', {})
                result.append("🏥 **服務健康狀態**")
                status_emoji = "✅" if health.get('status') == 'healthy' else "❌"
                result.append(f"   狀態: {status_emoji} {health.get('status', 'unknown')}")
                if 'collections_count' in health:
                    result.append(f"   Collections: {health.get('collections_count', 0)} 個")
                if 'error' in health:
                    result.append(f"   錯誤: {health['error']}")
                result.append("")
                
                # Collections 分析
                collections_info = report.get('collections_info', {})
                if collections_info and 'collections' in collections_info:
                    collections = collections_info['collections']
                    summary = collections_info.get('summary', {})
                    
                    result.append("📊 **Collections 分析**")
                    for collection in collections:
                        name = collection.get('name', 'unknown')
                        points = collection.get('points_count', 0)
                        vectors = collection.get('vectors_count', 0)
                        memory_mb = collection.get('estimated_memory_mb', 0)
                        status = collection.get('status', 'unknown')
                        
                        # 效能評估
                        perf_emoji = "🟢" if memory_mb < 100 else "🟡" if memory_mb < 500 else "🔴"
                        result.append(f"   📁 {name}: {points:,} points, {vectors:,} vectors, ~{memory_mb:.1f}MB {perf_emoji}")
                        result.append(f"      狀態: {status}")
                        
                        # 索引狀態
                        indexed = collection.get('indexed_vectors_count', 0)
                        if vectors > 0:
                            index_ratio = indexed / vectors
                            if index_ratio < 0.9:
                                result.append(f"      ⚠️ 索引率: {index_ratio:.1%} (建議重新索引)")
                    
                    result.append("")
                    result.append("📈 **總計統計**")
                    result.append(f"   總 Collections: {summary.get('total_collections', 0)}")
                    result.append(f"   總 Points: {summary.get('total_points', 0):,}")
                    result.append(f"   總 Vectors: {summary.get('total_vectors', 0):,}")
                    result.append(f"   估計記憶體: {summary.get('total_estimated_memory_mb', 0):.1f}MB")
                    result.append("")
                
                # 性能分析
                performance = report.get('performance_analysis', {})
                if performance and 'error' not in performance:
                    result.append("⚡ **效能分析**")
                    score = performance.get('overall_score', 'unknown')
                    score_emoji = {"excellent": "🟢", "good": "🟡", "fair": "🟠", "poor": "🔴"}.get(score, "⚪")
                    result.append(f"   整體評分: {score_emoji} {score}")
                    
                    recommendations = performance.get('recommendations', [])
                    if recommendations:
                        result.append("   💡 建議:")
                        for rec in recommendations:
                            result.append(f"      • {rec}")
                    
                    indexing_issues = performance.get('indexing_issues', [])
                    if indexing_issues:
                        result.append("   ⚠️ 索引問題:")
                        for issue in indexing_issues:
                            result.append(f"      • {issue['collection']}: {issue['issue']} ({issue['indexed_ratio']:.1%})")
                    result.append("")
                
                # 集群資訊
                cluster = report.get('cluster_info', {})
                if cluster and 'error' not in cluster:
                    result.append("🌐 **集群資訊**")
                    if cluster.get('status') == 'healthy':
                        result.append("   狀態: ✅ 健康")
                    result.append("")
                
                # 監控範圍說明
                result.append("ℹ️ **監控範圍**")
                result.append(f"   監控類型: {report.get('monitoring_scope', 'unknown')}")
                limitations = report.get('limitations', [])
                if limitations:
                    result.append("   限制:")
                    for limitation in limitations:
                        result.append(f"      • {limitation}")
                
                return result
                
            except Exception as e:
                logger.error(f"獲取 Qdrant 狀態失敗: {e}")
                return [
                    "❌ **Qdrant 狀態分析失敗**",
                    f"錯誤: {str(e)}",
                    "",
                    "這可能是因為:",
                    "• Qdrant 服務未運行",
                    "• 連接配置錯誤", 
                    "• 網路問題"
                ]

        async def get_qdrant_performance(ctx: Context) -> list[str]:
            """
            獲取 Qdrant 的效能分析，僅基於 Qdrant API 資料。
            """
            await ctx.debug("Getting Qdrant performance analysis")
            try:
                collections_info = await self.system_monitor.get_collections_info()
                performance = self.system_monitor._analyze_performance(collections_info)
                
                result = ["📊 **Qdrant 效能分析**"]
                result.append("")
                
                if 'error' in performance:
                    result.append(f"❌ 分析失敗: {performance['error']}")
                    return result
                    
                # 整體效能評分
                score = performance.get('overall_score', 'unknown')
                score_emoji = {"excellent": "🟢", "good": "🟡", "fair": "🟠", "poor": "🔴"}.get(score, "⚪")
                result.append(f"⚡ **整體效能**: {score_emoji} {score}")
                result.append("")
                
                # 記憶體分析
                total_memory = performance.get('total_estimated_memory_mb', 0)
                total_vectors = performance.get('total_vectors', 0)
                result.append("💾 **記憶體分析**")
                result.append(f"   估計總記憶體: {total_memory:.1f} MB")
                result.append(f"   總向量數: {total_vectors:,}")
                if total_vectors > 0:
                    avg_memory_per_vector = (total_memory * 1024) / total_vectors  # KB per vector
                    result.append(f"   平均每向量: {avg_memory_per_vector:.2f} KB")
                result.append("")
                
                # Collection 效能分析
                collection_analysis = performance.get('collection_analysis', [])
                if collection_analysis:
                    result.append("� **Collection 效能**")
                    for analysis in collection_analysis:
                        name = analysis.get('name', 'unknown')
                        efficiency = analysis.get('efficiency', 'unknown')
                        indexed_ratio = analysis.get('indexed_ratio', 0)
                        
                        eff_emoji = {"good": "🟢", "fair": "🟡", "poor": "🔴"}.get(efficiency, "⚪")
                        result.append(f"   📂 {name}:")
                        result.append(f"      效率: {eff_emoji} {efficiency}")
                        result.append(f"      索引率: {indexed_ratio:.1%}")
                result.append("")
                
                # 建議
                recommendations = performance.get('recommendations', [])
                if recommendations:
                    result.append("💡 **最佳化建議**")
                    for i, rec in enumerate(recommendations, 1):
                        result.append(f"   {i}. {rec}")
                    result.append("")
                
                # 索引問題
                indexing_issues = performance.get('indexing_issues', [])
                if indexing_issues:
                    result.append("⚠️ **索引問題**")
                    for issue in indexing_issues:
                        result.append(f"   • {issue['collection']}: {issue['issue']} (索引率: {issue['indexed_ratio']:.1%})")
                    result.append("")
                
                result.append("ℹ️ **說明**: 此分析僅基於 Qdrant API 提供的資訊")
                
                return result
                
            except Exception as e:
                logger.error(f"獲取效能分析失敗: {e}")
                return [
                    "❌ **效能分析失敗**",
                    f"錯誤: {str(e)}"
                ]

        async def get_collections_detailed_analysis(ctx: Context) -> list[str]:
            """
            獲取 Collections 的詳細分析，包括效能評估和最佳化建議。
            """
            await ctx.debug("Getting detailed collections analysis")
            try:
                collections_info = await self.system_monitor.get_collections_info()
                
                if not collections_info or 'collections' not in collections_info:
                    return ["📋 沒有發現任何 Collections"]
                
                result = ["📊 **Collections 詳細分析**"]
                result.append("")
                
                for collection in collections_info['collections']:
                    if isinstance(collection, dict) and 'name' in collection:
                        if 'error' in collection:
                            result.append(f"❌ **{collection['name']}** - 分析失敗: {collection['error']}")
                            continue
                        
                        result.append(f"📁 **Collection: {collection['name']}**")
                        
                        # 基本統計
                        result.append("   📈 統計資訊:")
                        result.append(f"      Points: {collection.get('points_count', 0):,}")
                        result.append(f"      Vectors: {collection.get('vectors_count', 0):,}")
                        result.append(f"      已索引: {collection.get('indexed_vectors_count', 0):,}")
                        
                        # 計算索引率
                        vectors_count = collection.get('vectors_count', 0)
                        indexed_count = collection.get('indexed_vectors_count', 0)
                        indexing_ratio = indexed_count / vectors_count if vectors_count > 0 else 0
                        result.append(f"      索引率: {indexing_ratio:.1%}")
                        
                        # 向量配置
                        vector_size = collection.get('vector_size', 0)
                        if vector_size > 0:
                            result.append("   🔧 向量配置:")
                            result.append(f"      向量維度: {vector_size} 維")
                        
                        # 記憶體分析
                        memory_mb = collection.get('estimated_memory_mb', 0)
                        result.append("   🧠 記憶體分析:")
                        result.append(f"      總估算: {memory_mb:.2f} MB")
                        if collection.get('points_count', 0) > 0:
                            memory_per_point = memory_mb / collection['points_count']
                            result.append(f"      每點記憶體: {memory_per_point:.4f} MB")
                        
                        # 狀態資訊
                        status = collection.get('status', 'unknown')
                        result.append("   ⚙️ 狀態:")
                        result.append(f"      Collection: {status}")
                        
                        result.append("")
                
                return result
                
            except Exception as e:
                logger.error(f"Failed to get collections analysis: {e}")
                return [f"❌ 獲取 Collections 分析失敗: {str(e)}"]

        async def get_deployment_info(ctx: Context) -> list[str]:
            """
            獲取部署環境資訊和支援的功能。
            """
            await ctx.debug("Getting deployment information")
            try:
                deployment_info = self.system_monitor.deployment_info
                
                result = ["🏗️ **部署環境資訊**"]
                result.append("")
                
                result.append(f"� **基本資訊**")
                result.append(f"   類型: {deployment_info.get('type', 'unknown')}")
                result.append(f"   主機: {deployment_info.get('host', 'unknown')}")
                result.append(f"   端口: {deployment_info.get('port', 'unknown')}")
                result.append("")
                
                result.append(f"🔧 **環境特性**")
                result.append(f"   本地部署: {'✅' if deployment_info.get('is_local') else '❌'}")
                result.append(f"   雲端服務: {'✅' if deployment_info.get('is_cloud') else '❌'}")
                result.append(f"   Docker 容器: {'✅' if deployment_info.get('is_docker') else '❌'}")
                result.append("")
                
                # 可用功能
                features = deployment_info.get('features', [])
                result.append(f"✨ **可用功能** ({len(features)} 項)")
                feature_descriptions = {
                    'qdrant_api': '🔌 Qdrant API 查詢',
                    'health_check': '🏥 健康狀態檢查',
                    'collections_stats': '📊 Collections 統計',
                    'system_metrics': '💻 系統資源監控',
                    'docker_stats': '🐳 Docker 容器監控',
                    'container_logs': '📋 容器日誌查看'
                }
                
                for feature in features:
                    desc = feature_descriptions.get(feature, f'🔧 {feature}')
                    result.append(f"   {desc}")
                result.append("")
                
                # 限制
                limitations = deployment_info.get('limitations', [])
                if limitations:
                    result.append(f"⚠️ **功能限制** ({len(limitations)} 項)")
                    limitation_descriptions = {
                        'no_system_metrics': '💻 無法獲取系統資源資訊',
                        'no_container_access': '🐳 無法存取容器資訊',
                        'no_logs_access': '📋 無法獲取服務日誌',
                        'no_docker_stats': '📊 無法獲取 Docker 統計'
                    }
                    
                    for limitation in limitations:
                        desc = limitation_descriptions.get(limitation, f'⚠️ {limitation}')
                        result.append(f"   {desc}")
                    result.append("")
                
                # 部署建議
                deployment_type = deployment_info.get('type', 'unknown')
                result.append("💡 **部署建議**")
                
                if deployment_type == 'cloud':
                    result.append("   • 雲端部署：關注成本最佳化和資料傳輸")
                    result.append("   • 建議設定適當的查詢限制和快取策略")
                    result.append("   • 定期檢查 API 使用量和計費")
                elif deployment_type == 'docker_local':
                    result.append("   • Docker 部署：監控容器資源使用")
                    result.append("   • 建議設定記憶體限制和健康檢查")
                    result.append("   • 定期備份資料和配置")
                elif deployment_type == 'local_binary':
                    result.append("   • 本地部署：注意系統資源管理")
                    result.append("   • 建議設定適當的系統監控")
                    result.append("   • 確保資料備份和服務自動重啟")
                else:
                    result.append("   • 建議根據實際需求選擇合適的監控策略")
                
                return result
                
            except Exception as e:
                logger.error(f"Failed to get deployment info: {e}")
                return [f"❌ 獲取部署資訊失敗: {str(e)}"]

        async def get_docker_containers(ctx: Context) -> list[str]:
            """
            獲取 Qdrant 相關的 Docker 容器資訊。
            """
            await ctx.debug("Getting Docker containers info")
            try:
                from .system_monitor_backup import DockerSystemMonitor
                containers = DockerSystemMonitor.list_qdrant_containers()
                
                if not containers:
                    return ["📦 沒有發現 Qdrant 相關的 Docker 容器"]
                
                result = ["🐳 **Qdrant Docker 容器**"]
                result.append("")
                
                for container in containers:
                    result.append(f"📦 **容器: {container.get('name', 'unknown')}**")
                    result.append(f"   狀態: {container.get('status', 'unknown')}")
                    result.append(f"   映像: {container.get('image', 'unknown')}")
                    result.append(f"   端口: {container.get('ports', 'none')}")
                    result.append("")
                
                return result
                
            except Exception as e:
                logger.error(f"Failed to get Docker containers: {e}")
                return [f"❌ 獲取 Docker 容器資訊失敗: {str(e)}"]

        async def get_container_logs(
            ctx: Context,
            container_name: Annotated[str, Field(description="容器名稱")] = "qdrant",
            lines: Annotated[int, Field(description="日誌行數")] = 50
        ) -> list[str]:
            """
            獲取指定容器的日誌。
            """
            await ctx.debug(f"Getting container logs for {container_name}")
            try:
                from .system_monitor_backup import DockerSystemMonitor
                logs = DockerSystemMonitor.get_container_logs(container_name, lines)
                
                result = [f"📋 **容器日誌: {container_name} (最近 {lines} 行)**"]
                result.append("```")
                result.append(logs)
                result.append("```")
                
                return result
                
            except Exception as e:
                logger.error(f"Failed to get container logs: {e}")
                return [f"❌ 獲取容器日誌失敗: {str(e)}"]

        # 註冊通用系統監控工具
        async def get_qdrant_status(ctx: Context) -> list[str]:
            """
            獲取 Qdrant 的綜合狀態資訊，自動適應所有部署方式。
            支援 Cloud、Docker、Local、Remote 等多種部署環境。
            """
            await ctx.debug("Getting comprehensive Qdrant status")
            
            try:
                status = await self.system_monitor.get_comprehensive_analysis()
                
                result = ["🔍 **Qdrant 綜合狀態分析**"]
                result.append("")
                
                # 部署資訊
                deployment = status.get("deployment_info", {})
                result.append(f"📊 **部署類型**: {deployment.get('type', 'unknown')}")
                result.append(f"🌐 **端點**: {deployment.get('url', 'unknown')}")
                result.append(f"🏠 **主機**: {deployment.get('host', 'unknown')}:{deployment.get('port', 'unknown')}")
                result.append("")
                
                # 可用功能
                features = deployment.get("features", [])
                result.append("✅ **可用功能**:")
                for feature in features:
                    result.append(f"   • {feature}")
                result.append("")
                
                # 限制說明
                limitations = deployment.get("limitations", [])
                if limitations:
                    result.append("⚠️ **功能限制**:")
                    for limitation in limitations:
                        result.append(f"   • {limitation}")
                    result.append("")
                
                # Qdrant 健康狀態
                health = status.get("qdrant_health", {})
                result.append(f"💚 **Qdrant 狀態**: {health.get('status', 'unknown')}")
                result.append(f"📁 **Collections 數量**: {health.get('collections_count', 0)}")
                
                # 樣本 Collection 狀態
                if "sample_collection_status" in health:
                    sample = health["sample_collection_status"]
                    result.append(f"📂 **樣本 Collection**: {sample.get('name', 'unknown')}")
                    result.append(f"   • 文件數量: {sample.get('points_count', 0)}")
                    result.append(f"   • 狀態: {sample.get('status', 'unknown')}")
                result.append("")
                
                # 系統指標（如果可用）
                system_metrics = status.get("system_metrics", {})
                if not system_metrics.get("unavailable"):
                    result.append("💻 **系統資源**:")
                    if "cpu" in system_metrics:
                        cpu = system_metrics["cpu"]
                        result.append(f"   • CPU: {cpu.get('usage_percent', 0)}% ({cpu.get('core_count', 0)} 核心)")
                    if "memory" in system_metrics:
                        memory = system_metrics["memory"]
                        result.append(f"   • 記憶體: {memory.get('usage_percent', 0)}% ({memory.get('used_gb', 0):.1f}GB / {memory.get('total_gb', 0):.1f}GB)")
                    if "disk" in system_metrics:
                        disk = system_metrics["disk"]
                        result.append(f"   • 磁碟: {disk.get('usage_percent', 0)}% ({disk.get('used_gb', 0):.1f}GB / {disk.get('total_gb', 0):.1f}GB)")
                else:
                    result.append("💻 **系統資源**: 不可用")
                    result.append(f"   原因: {system_metrics.get('reason', 'unknown')}")
                    if "alternatives" in system_metrics:
                        result.append("   💡 **替代方案**:")
                        for alt in system_metrics["alternatives"]:
                            result.append(f"      • {alt}")
                result.append("")
                
                # Docker 指標（如果可用）
                docker_metrics = status.get("docker_metrics", {})
                if not docker_metrics.get("unavailable"):
                    result.append("🐳 **Docker 容器**:")
                    result.append(f"   • 容器: {docker_metrics.get('container_name', 'unknown')}")
                    result.append(f"   • CPU: {docker_metrics.get('cpu_percent', 'unknown')}")
                    result.append(f"   • 記憶體: {docker_metrics.get('memory_usage', 'unknown')} ({docker_metrics.get('memory_percent', 'unknown')})")
                    result.append(f"   • 網路 I/O: {docker_metrics.get('network_io', 'unknown')}")
                else:
                    result.append("🐳 **Docker 指標**: 不可用")
                    result.append(f"   原因: {docker_metrics.get('reason', 'unknown')}")
                result.append("")
                
                # 雲端資訊（如果是雲端部署）
                if "cloud_info" in status:
                    cloud = status["cloud_info"]
                    result.append("☁️ **雲端服務資訊**:")
                    result.append(f"   • 提供商: {cloud.get('provider', 'unknown')}")
                    if "dashboard_url" in cloud:
                        result.append(f"   • 控制台: {cloud['dashboard_url']}")
                    if "features" in cloud:
                        result.append("   • 雲端功能:")
                        for feature in cloud["features"]:
                            result.append(f"      • {feature}")
                    result.append("")
                
                # 遠端部署建議（如果是遠端部署）
                if "remote_monitoring_suggestions" in status:
                    result.append("🔗 **遠端監控建議**:")
                    for suggestion in status["remote_monitoring_suggestions"]:
                        result.append(f"   • {suggestion}")
                    result.append("")
                
                result.append(f"🕐 **更新時間**: {status.get('timestamp', 'unknown')}")
                
                return result
                
            except Exception as e:
                logger.error(f"Failed to get Qdrant status: {e}")
                return [f"❌ 獲取狀態失敗: {str(e)}"]

        async def get_collections_detailed_analysis(ctx: Context) -> list[str]:
            """
            獲取 Collections 的詳細效能分析，包括部署特定的最佳化建議。
            """
            await ctx.debug("Getting detailed collections analysis")
            
            try:
                analytics = await self.system_monitor.get_collections_info()
                
                result = ["📊 **Collections 詳細分析**"]
                result.append("")
                
                if not analytics or 'collections' not in analytics:
                    result.append("❌ 無法獲取 Collections 資訊")
                    return result
                
                for collection in analytics['collections']:
                    if "error" in collection:
                        result.append(f"❌ **{collection.get('name', 'unknown')}**: {collection['error']}")
                        continue
                    
                    result.append(f"📁 **Collection: {collection['name']}**")
                    
                    # 基本統計
                    result.append(f"   📈 **統計資料**:")
                    result.append(f"      • 文件數: {collection.get('points_count', 0):,}")
                    result.append(f"      • 向量數: {collection.get('vectors_count', 0):,}")
                    result.append(f"      • 已索引向量: {collection.get('indexed_vectors_count', 0):,}")
                    
                    # 計算索引率
                    vectors_count = collection.get('vectors_count', 0)
                    indexed_count = collection.get('indexed_vectors_count', 0)
                    indexing_ratio = indexed_count / vectors_count if vectors_count > 0 else 0
                    result.append(f"      • 索引率: {indexing_ratio:.1%}")
                    
                    # 記憶體估算
                    memory_mb = collection.get('estimated_memory_mb', 0)
                    result.append(f"      • 記憶體估算: {memory_mb:.1f}MB")
                    
                    # 向量配置
                    vector_size = collection.get('vector_size', 0)
                    if vector_size > 0:
                        result.append(f"   🎯 **向量配置**:")
                        result.append(f"      • 向量維度: {vector_size}")
                    
                    # 狀態資訊
                    status = collection.get('status', 'unknown')
                    result.append(f"   💚 **狀態**: {status}")
                    result.append("")
                    
                    # 記憶體分析
                    memory = collection.get("memory_analysis", {})
                    result.append(f"   💾 **記憶體分析**:")
                    result.append(f"      • 總計: {memory.get('total_estimate_mb', 0):.1f}MB")
                    result.append(f"      • 每文件: {memory.get('memory_per_point_mb', 0):.3f}MB")
                    result.append(f"      • 效能評分: {memory.get('performance_score', 'unknown')}")
                    
                    # 建議
                    recommendations = collection.get("recommendations", [])
                    if recommendations:
                        result.append(f"   💡 **最佳化建議**:")
                        for rec in recommendations:
                            result.append(f"      • {rec}")
                    
                    # 部署特定最佳化
                    deploy_opts = collection.get("deployment_optimizations", [])
                    if deploy_opts:
                        result.append(f"   🚀 **部署最佳化**:")
                        for opt in deploy_opts:
                            result.append(f"      • {opt}")
                    
                    result.append("")
                
                return result
                
            except Exception as e:
                logger.error(f"Failed to get collections analysis: {e}")
                return [f"❌ 獲取分析失敗: {str(e)}"]

        async def get_deployment_info(ctx: Context) -> list[str]:
            """
            獲取詳細的部署環境資訊和功能支援狀況。
            """
            await ctx.debug("Getting deployment information")
            
            try:
                deployment = self.system_monitor.deployment_info
                
                result = ["🏗️ **部署環境詳細資訊**"]
                result.append("")
                
                result.append(f"📍 **基本資訊**:")
                result.append(f"   • 部署類型: {deployment.get('type', 'unknown')}")
                result.append(f"   • 主機: {deployment.get('host', 'unknown')}")
                result.append(f"   • 端口: {deployment.get('port', 'unknown')}")
                result.append(f"   • URL: {deployment.get('url', 'unknown')}")
                result.append("")
                
                result.append(f"📊 **部署特性**:")
                result.append(f"   • 本地部署: {'✅' if deployment.get('is_local') else '❌'}")
                result.append(f"   • 雲端部署: {'✅' if deployment.get('is_cloud') else '❌'}")
                result.append(f"   • Docker 部署: {'✅' if deployment.get('is_docker') else '❌'}")
                result.append(f"   • 遠端部署: {'✅' if deployment.get('is_remote') else '❌'}")
                result.append("")
                
                features = deployment.get("features", [])
                result.append("✅ **支援功能**:")
                if features:
                    for feature in features:
                        feature_name = {
                            "qdrant_api": "Qdrant API 存取",
                            "health_check": "健康檢查",
                            "collections_stats": "Collections 統計",
                            "system_metrics": "系統資源監控",
                            "docker_stats": "Docker 容器監控",
                            "container_logs": "容器日誌存取",
                            "container_metrics": "容器指標",
                            "cloud_monitoring": "雲端監控功能"
                        }.get(feature, feature)
                        result.append(f"   • {feature_name}")
                else:
                    result.append("   • 無特殊功能")
                result.append("")
                
                limitations = deployment.get("limitations", [])
                if limitations:
                    result.append("⚠️ **功能限制**:")
                    for limitation in limitations:
                        limitation_name = {
                            "no_system_metrics": "無法取得系統資源指標",
                            "no_container_access": "無法存取容器資訊",
                            "no_logs_access": "無法存取日誌檔案",
                            "no_container_logs": "無法存取容器日誌"
                        }.get(limitation, limitation)
                        result.append(f"   • {limitation_name}")
                    result.append("")
                
                # 監控建議
                result.append("💡 **監控建議**:")
                if deployment.get('is_cloud'):
                    result.append("   • 使用雲端服務商的監控工具")
                    result.append("   • 設定 API 使用量告警")
                    result.append("   • 監控成本和配額")
                elif deployment.get('is_docker'):
                    result.append("   • 使用 Docker 健康檢查")
                    result.append("   • 監控容器資源使用")
                    result.append("   • 設定適當的重啟策略")
                elif deployment.get('is_local'):
                    result.append("   • 監控本地系統資源")
                    result.append("   • 設定服務管理和自動啟動")
                    result.append("   • 定期備份資料")
                else:
                    result.append("   • 監控網路連線品質")
                    result.append("   • 設定健康檢查端點")
                    result.append("   • 使用外部監控服務")
                
                return result
                
            except Exception as e:
                logger.error(f"Failed to get deployment info: {e}")
                return [f"❌ 獲取部署資訊失敗: {str(e)}"]

        # 註冊純 Qdrant API 監控工具
        self.tool(
            get_qdrant_status,
            name="qdrant-system-status",
            description="獲取 Qdrant 的狀態資訊，純基於 Qdrant API，適用所有部署方式。",
        )
        
        self.tool(
            get_qdrant_performance,
            name="qdrant-performance-analysis",
            description="獲取 Qdrant 的效能分析，基於 Qdrant API 資料提供記憶體使用和效能評估。",
        )

        # 新增儲存優化工具
        async def optimize_storage(
            ctx: Context,
            collection_name: Annotated[
                str, Field(description="要優化的 collection 名稱，設為 'all' 表示優化所有 collections")
            ] = "all",
            confirm_optimize: Annotated[
                bool, Field(description="確認優化操作，必須設為 True")
            ] = False,
        ) -> list[str]:
            """
            優化指定 collection 或所有 collections 的儲存配置，減少磁碟使用。需要明確確認才能執行。
            """
            await ctx.debug(f"Optimizing storage for collection '{collection_name}'")
            
            if not confirm_optimize:
                return [
                    "❌ **儲存優化被取消**",
                    "",
                    "請設定 confirm_optimize=True 來確認執行儲存優化。",
                    "",
                    "⚠️ **注意**: 優化後建議重啟 Qdrant container 以完全應用變更。"
                ]
            
            try:
                if collection_name == "all":
                    result = await self.storage_optimizer.optimize_all_collections()
                else:
                    result = await self.storage_optimizer.optimize_collection_storage(collection_name)
                
                if not result.get("success", False):
                    return [
                        f"❌ **儲存優化失敗**",
                        f"錯誤: {result.get('error', '未知錯誤')}"
                    ]
                
                # 格式化成功結果
                formatted_result = ["✅ **Qdrant 儲存優化完成**"]
                formatted_result.append("")
                
                if collection_name == "all":
                    # 批次優化結果
                    summary = result.get("summary", {})
                    formatted_result.append("📊 **優化總結**")
                    formatted_result.append(f"   Collections 優化: {result.get('collections_optimized', 0)}/{result.get('total_collections', 0)}")
                    formatted_result.append(f"   磁碟空間: {summary.get('disk_space_before_mb', 0):.1f}MB → {summary.get('disk_space_after_mb', 0):.1f}MB")
                    formatted_result.append(f"   節省磁碟: {summary.get('disk_space_saved_mb', 0):.1f}MB ({summary.get('disk_space_saved_percent', 0):.1f}%)")
                    formatted_result.append(f"   記憶體: {summary.get('ram_before_mb', 0):.1f}MB → {summary.get('ram_after_mb', 0):.1f}MB")
                    formatted_result.append(f"   節省記憶體: {summary.get('ram_saved_mb', 0):.1f}MB ({summary.get('ram_saved_percent', 0):.1f}%)")
                    formatted_result.append("")
                    
                    # 個別 collection 結果
                    if result.get("results"):
                        formatted_result.append("📁 **各 Collection 優化結果**")
                        for coll_result in result["results"]:
                            if coll_result.get("success"):
                                name = coll_result["collection"]
                                before_disk = coll_result["before"].get("disk_size", 0) / 1024 / 1024
                                after_disk = coll_result["after"].get("disk_size", 0) / 1024 / 1024
                                formatted_result.append(f"   ✅ {name}: {before_disk:.1f}MB → {after_disk:.1f}MB")
                            else:
                                formatted_result.append(f"   ❌ {coll_result['collection']}: {coll_result.get('error', '優化失敗')}")
                        formatted_result.append("")
                    
                    # 建議
                    recommendations = result.get("recommendations", [])
                    if recommendations:
                        formatted_result.append("💡 **建議**")
                        for rec in recommendations:
                            formatted_result.append(f"   • {rec}")
                        formatted_result.append("")
                
                else:
                    # 單一 collection 優化結果
                    before = result.get("before", {})
                    after = result.get("after", {})
                    optimizations = result.get("optimizations_applied", {})
                    
                    formatted_result.append(f"📁 **Collection: {collection_name}**")
                    formatted_result.append(f"   向量數: {before.get('vectors_count', 0):,} → {after.get('vectors_count', 0):,}")
                    formatted_result.append(f"   段數: {before.get('segments_count', 0)} → {after.get('segments_count', 0)}")
                    
                    before_disk = before.get("disk_size", 0) / 1024 / 1024
                    after_disk = after.get("disk_size", 0) / 1024 / 1024
                    formatted_result.append(f"   磁碟: {before_disk:.1f}MB → {after_disk:.1f}MB")
                    
                    before_ram = before.get("ram_size", 0) / 1024 / 1024
                    after_ram = after.get("ram_size", 0) / 1024 / 1024
                    formatted_result.append(f"   記憶體: {before_ram:.1f}MB → {after_ram:.1f}MB")
                    formatted_result.append("")
                    
                    if optimizations:
                        formatted_result.append("🔧 **應用的優化**")
                        for key, value in optimizations.items():
                            formatted_result.append(f"   • {key}: {value}")
                        formatted_result.append("")
                
                formatted_result.append("🔄 **後續步驟**")
                formatted_result.append("   建議重啟 Qdrant Docker container 以完全應用所有優化:")
                formatted_result.append("   ```bash")
                formatted_result.append("   docker restart <qdrant-container-name>")
                formatted_result.append("   ```")
                
                return formatted_result
                
            except Exception as e:
                logger.error(f"儲存優化失敗: {e}")
                return [
                    "❌ **儲存優化失敗**",
                    f"錯誤: {str(e)}",
                    "",
                    "可能原因:",
                    "• Qdrant 服務未運行",
                    "• 連接配置錯誤",
                    "• 權限不足"
                ]

        async def analyze_storage(ctx: Context) -> list[str]:
            """
            分析當前 Qdrant 的儲存使用情況，提供優化建議。
            """
            await ctx.debug("Analyzing Qdrant storage usage")
            
            try:
                analysis = await self.storage_optimizer.get_storage_analysis()
                
                if "error" in analysis:
                    return [
                        "❌ **儲存分析失敗**",
                        f"錯誤: {analysis['error']}"
                    ]
                
                result = ["📊 **Qdrant 儲存使用分析**"]
                result.append("")
                
                # 總結
                summary = analysis.get("summary", {})
                result.append("📈 **總計統計**")
                result.append(f"   Collections: {summary.get('total_collections', 0)} 個")
                result.append(f"   總向量數: {summary.get('total_vectors', 0):,}")
                result.append(f"   總磁碟使用: {summary.get('total_disk_mb', 0):.1f} MB")
                result.append(f"   總記憶體使用: {summary.get('total_ram_mb', 0):.1f} MB")
                result.append(f"   {summary.get('estimated_optimization_savings', '')}")
                result.append("")
                
                # 各 Collection 詳情
                collections = analysis.get("collections", [])
                if collections:
                    result.append("📁 **各 Collection 詳情**")
                    for coll in collections:
                        name = coll["collection"]
                        vectors = coll["vectors"]
                        segments = coll["segments"]
                        disk_mb = coll["disk_mb"]
                        ram_mb = coll["ram_mb"]
                        
                        # 效能評估 - 安全地處理 None 值
                        disk_mb = coll.get("disk_mb", 0) or 0
                        if disk_mb > 100:
                            perf_icon = "🔴 大"
                        elif disk_mb > 50:
                            perf_icon = "🟡 中"
                        else:
                            perf_icon = "🟢 小"
                        
                        result.append(f"   📂 {name}:")
                        result.append(f"      向量: {vectors:,}, 段: {segments}")
                        result.append(f"      磁碟: {disk_mb:.1f}MB, 記憶體: {ram_mb:.1f}MB {perf_icon}")
                        
                        # 配置檢查
                        config = coll.get("config", {})
                        optimizer = config.get("optimizer", {})
                        hnsw = config.get("hnsw", {})
                        
                        # 檢查是否需要優化 - 安全地處理 None 值
                        needs_optimization = []
                        max_segment_size = optimizer.get("max_segment_size", 100000) or 100000
                        if max_segment_size > 10000:
                            needs_optimization.append("段大小過大")
                        
                        memmap_threshold = optimizer.get("memmap_threshold", 10000) or 10000
                        if memmap_threshold > 2000:
                            needs_optimization.append("mmap 閾值過高")
                        
                        hnsw_m = hnsw.get("m", 16) or 16
                        if hnsw_m > 12:
                            needs_optimization.append("HNSW m 值過大")
                            
                        if not hnsw.get("on_disk", False):
                            needs_optimization.append("索引未存磁碟")
                        
                        if needs_optimization:
                            result.append(f"      ⚠️ 建議優化: {', '.join(needs_optimization)}")
                        else:
                            result.append(f"      ✅ 配置良好")
                    result.append("")
                
                # 優化建議
                total_disk = summary.get('total_disk_mb', 0) or 0
                if total_disk > 200:
                    result.append("💡 **優化建議**")
                    result.append(f"   當前磁碟使用 {total_disk:.1f}MB 偏高，建議執行儲存優化:")
                    result.append("   • 使用 qdrant-optimize-storage 工具優化所有 collections")
                    result.append("   • 預期可節省 60-70% 磁碟空間")
                    result.append("   • 優化後重啟 Qdrant container 以完全生效")
                elif total_disk > 100:
                    result.append("💡 **優化建議**")
                    result.append(f"   磁碟使用 {total_disk:.1f}MB 中等，可考慮優化:")
                    result.append("   • 針對大型 collections 執行優化")
                    result.append("   • 定期清理無用資料")
                else:
                    result.append("✅ **儲存狀態良好**")
                    result.append(f"   磁碟使用 {total_disk:.1f}MB 在合理範圍內")
                
                return result
                
            except Exception as e:
                logger.error(f"儲存分析失敗: {e}")
                return [
                    "❌ **儲存分析失敗**",
                    f"錯誤: {str(e)}"
                ]

        # 註冊儲存優化工具
        if not self.qdrant_settings.read_only:
            self.tool(
                optimize_storage,
                name="qdrant-optimize-storage",
                description="優化 Qdrant collections 的儲存配置，減少磁碟使用約 60-70%。需要明確確認才能執行。",
            )
        
        self.tool(
            analyze_storage,
            name="qdrant-analyze-storage", 
            description="分析 Qdrant 的儲存使用情況，提供優化建議。",
        )
