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
from mcp_server_qdrant.ragbridge.connector import RAGBridgeConnector
from mcp_server_qdrant.ragbridge.models import ContentType, SearchContext, RAGEntry, RAGMetadata
from mcp_server_qdrant.ragbridge.vocabulary_api import vocabulary_api
from mcp_server_qdrant.ragbridge.fragment_manager import fragment_manager
from mcp_server_qdrant.ragbridge.schema_api import schema_api
from mcp_server_qdrant.ragbridge.schema_approval import get_approval_manager
from mcp_server_qdrant.permission_manager import get_permission_manager, PermissionLevel
from mcp_server_qdrant.data_migration_tool import DataMigrationTool

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
        
        # Initialize RAG Bridge connector
        self.ragbridge_connector = RAGBridgeConnector(
            qdrant_url=qdrant_settings.location,
            qdrant_api_key=qdrant_settings.api_key,
            embedding_provider=self.embedding_provider,
            qdrant_local_path=qdrant_settings.local_path,
            default_collection_prefix="ragbridge",
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
        
        # 初始化權限管理系統
        self.permission_manager = get_permission_manager()
        if qdrant_settings.enable_permission_system:
            # 設定預設權限級別
            default_level = PermissionLevel(qdrant_settings.default_permission_level)
            self.permission_manager.set_user_permission("default_user", default_level)
            logger.info(f"Permission system enabled with default level: {default_level.value}")
        else:
            # 向後兼容：如果停用權限系統，設定為超級管理員
            self.permission_manager.set_user_permission("default_user", PermissionLevel.SUPER_ADMIN)
            logger.info("Permission system disabled - granting super_admin access")
        
        # 初始化資料遷移工具
        from mcp_server_qdrant.ragbridge.vocabulary import VocabularyManager
        vocabulary_manager = VocabularyManager()
        self.migration_tool = DataMigrationTool(
            qdrant_client=self.qdrant_connector._client,
            ragbridge_connector=self.ragbridge_connector,
            vocabulary_manager=vocabulary_manager
        )

        super().__init__(name=name, instructions=instructions, **settings)

        self.setup_tools()

    def format_entry(self, entry: Entry) -> str:
        """
        Feel free to override this method in your subclass to customize the format of the entry.
        """
        entry_metadata = json.dumps(entry.metadata) if entry.metadata else ""
        return f"<entry><content>{entry.content}</content><metadata>{entry_metadata}</metadata></entry>"
    
    def check_permission_wrapper(self, func, tool_name: str):
        """
        包裝工具函數以添加權限檢查
        """
        async def permission_checked_func(ctx: Context, *args, **kwargs):
            # 獲取用戶ID（在實際使用中可能來自 context 或認證系統）
            user_id = "default_user"  # 目前使用預設用戶
            
            # 檢查權限
            if not self.permission_manager.check_tool_permission(user_id, tool_name):
                permission_level = self.permission_manager.get_user_permission(user_id)
                required_permission = self.permission_manager.tool_permissions.get(tool_name)
                required_level = required_permission.required_level.value if required_permission else "unknown"
                
                await ctx.debug(f"Permission denied: User {user_id} (level: {permission_level.value}) trying to access {tool_name} (requires: {required_level})")
                return [
                    f"❌ **權限不足**",
                    f"🔐 工具名稱: {tool_name}",
                    f"👤 當前權限級別: {permission_level.value}",
                    f"⚠️ 需要權限級別: {required_level}",
                    "",
                    f"💡 **解決方案:**",
                    f"請聯絡管理員提升權限級別，或使用適合您權限的工具。",
                    f"使用 'get-user-permissions' 工具查看可用的工具列表。"
                ]
            
            # 權限檢查通過，執行原始函數
            return await func(ctx, *args, **kwargs)
        
        return permission_checked_func

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

        # RAG Bridge 工具集
        async def search_experience(
            ctx: Context,
            query: Annotated[str, Field(description="搜尋個人經驗和知識的查詢")],
            content_types: Annotated[
                list[str] | None, 
                Field(description="要搜尋的內容類型，可選: experience, process_workflow, knowledge_base, vocabulary, decision_record")
            ] = None,
            max_results: Annotated[int, Field(description="最多返回的結果數量")] = 5,
            min_similarity: Annotated[float, Field(description="最低相似度閾值")] = 0.7,
            include_experimental: Annotated[bool, Field(description="是否包含實驗性內容")] = False,
        ) -> list[str]:
            """
            搜尋個人經驗知識庫，支援多種內容類型和智能排序。
            """
            await ctx.debug(f"Searching experience for query: {query}")
            
            try:
                # 轉換內容類型
                parsed_content_types = []
                if content_types:
                    for ct in content_types:
                        try:
                            parsed_content_types.append(ContentType(ct))
                        except ValueError:
                            await ctx.debug(f"Invalid content type: {ct}")
                            continue
                
                # 建立搜尋上下文
                search_context = SearchContext(
                    query=query,
                    content_types=parsed_content_types if parsed_content_types else None,
                    max_results=max_results,
                    min_similarity=min_similarity,
                    include_experimental=include_experimental,
                )
                
                # 執行搜尋
                results = await self.ragbridge_connector.search_rag_entries(search_context)
                
                if not results:
                    return [f"沒有找到與查詢 '{query}' 相關的經驗知識"]
                
                # 格式化結果
                formatted_results = [f"🔍 搜尋結果 '{query}' ({len(results)} 個結果):"]
                formatted_results.append("")
                
                for idx, result in enumerate(results, 1):
                    entry = result.entry
                    metadata = entry.metadata
                    
                    formatted_results.append(f"**{idx}. {metadata.title}**")
                    formatted_results.append(f"   📝 類型: {metadata.content_type.value}")
                    formatted_results.append(f"   🎯 相似度: {result.similarity_score:.2f}")
                    formatted_results.append(f"   📊 品質: {metadata.quality_score:.2f}")
                    formatted_results.append(f"   📈 使用次數: {metadata.usage_count}")
                    formatted_results.append(f"   🏷️ 標籤: {', '.join(metadata.tags) if metadata.tags else '無'}")
                    
                    # 內容摘要
                    content_preview = entry.content[:200] + "..." if len(entry.content) > 200 else entry.content
                    formatted_results.append(f"   📄 內容: {content_preview}")
                    
                    # 匹配原因
                    if result.match_reasons:
                        formatted_results.append(f"   🎯 匹配原因: {', '.join(result.match_reasons)}")
                    
                    # 使用建議
                    formatted_results.append(f"   💡 建議: {result.usage_recommendation}")
                    formatted_results.append("")
                
                return formatted_results
                
            except Exception as e:
                logger.error(f"Search experience failed: {e}")
                return [f"❌ 搜尋經驗失敗: {str(e)}"]

        async def get_process_workflow(
            ctx: Context,
            workflow_name: Annotated[str, Field(description="工作流程名稱或相關關鍵字")],
            include_steps: Annotated[bool, Field(description="是否包含詳細步驟")] = True,
            include_checkpoints: Annotated[bool, Field(description="是否包含檢查點")] = True,
        ) -> list[str]:
            """
            獲取特定流程的工作流程步驟，支援結構化流程展示。
            """
            await ctx.debug(f"Getting process workflow for: {workflow_name}")
            
            try:
                # 建立搜尋上下文，專注於流程工作流
                search_context = SearchContext(
                    query=workflow_name,
                    content_types=[ContentType.PROCESS_WORKFLOW],
                    max_results=3,
                    min_similarity=0.6,
                    include_experimental=False,
                )
                
                # 執行搜尋
                results = await self.ragbridge_connector.search_rag_entries(search_context)
                
                if not results:
                    return [f"沒有找到 '{workflow_name}' 相關的工作流程"]
                
                # 格式化結果
                formatted_results = [f"🔄 工作流程: {workflow_name}"]
                formatted_results.append("")
                
                for idx, result in enumerate(results, 1):
                    entry = result.entry
                    metadata = entry.metadata
                    
                    formatted_results.append(f"**{idx}. {metadata.title}**")
                    formatted_results.append(f"   📊 品質評分: {metadata.quality_score:.2f}")
                    formatted_results.append(f"   ✅ 成功率: {metadata.success_rate:.2f}")
                    formatted_results.append(f"   🏷️ 標籤: {', '.join(metadata.tags) if metadata.tags else '無'}")
                    formatted_results.append("")
                    
                    # 顯示流程內容
                    formatted_results.append("📋 **流程內容:**")
                    formatted_results.append(entry.content)
                    formatted_results.append("")
                    
                    # 顯示結構化內容
                    if include_steps and entry.structured_content:
                        structured = entry.structured_content
                        
                        if "steps" in structured:
                            formatted_results.append("📝 **詳細步驟:**")
                            for step_idx, step in enumerate(structured["steps"], 1):
                                formatted_results.append(f"   {step_idx}. {step}")
                            formatted_results.append("")
                        
                        if include_checkpoints and "checkpoints" in structured:
                            formatted_results.append("🎯 **檢查點:**")
                            for checkpoint in structured["checkpoints"]:
                                formatted_results.append(f"   • {checkpoint}")
                            formatted_results.append("")
                        
                        if "prerequisites" in structured:
                            formatted_results.append("🔧 **前置需求:**")
                            for prereq in structured["prerequisites"]:
                                formatted_results.append(f"   • {prereq}")
                            formatted_results.append("")
                        
                        if "expected_outcomes" in structured:
                            formatted_results.append("🎯 **預期結果:**")
                            for outcome in structured["expected_outcomes"]:
                                formatted_results.append(f"   • {outcome}")
                            formatted_results.append("")
                    
                    # 語義塊 (如果有)
                    if entry.semantic_chunks:
                        formatted_results.append("🧩 **相關概念:**")
                        for chunk in entry.semantic_chunks[:3]:  # 只顯示前3個
                            formatted_results.append(f"   • {chunk}")
                        formatted_results.append("")
                    
                    formatted_results.append(f"   💡 使用建議: {result.usage_recommendation}")
                    formatted_results.append("")
                
                return formatted_results
                
            except Exception as e:
                logger.error(f"Get process workflow failed: {e}")
                return [f"❌ 獲取工作流程失敗: {str(e)}"]

        async def suggest_similar(
            ctx: Context,
            reference_content: Annotated[str, Field(description="參考內容或情境描述")],
            content_type: Annotated[str, Field(description="內容類型")] = "experience",
            similarity_threshold: Annotated[float, Field(description="相似度閾值")] = 0.6,
            max_suggestions: Annotated[int, Field(description="最多建議數量")] = 3,
        ) -> list[str]:
            """
            根據參考內容推薦相關的經驗和知識。
            """
            await ctx.debug(f"Getting similar suggestions for: {reference_content[:50]}...")
            
            try:
                # 轉換內容類型
                try:
                    parsed_content_type = ContentType(content_type)
                except ValueError:
                    parsed_content_type = ContentType.EXPERIENCE
                
                # 建立搜尋上下文
                search_context = SearchContext(
                    query=reference_content,
                    content_types=[parsed_content_type],
                    max_results=max_suggestions,
                    min_similarity=similarity_threshold,
                    include_experimental=False,
                )
                
                # 執行搜尋
                results = await self.ragbridge_connector.search_rag_entries(search_context)
                
                if not results:
                    return [f"沒有找到與參考內容相似的 {content_type} 內容"]
                
                # 格式化結果
                formatted_results = [f"🔗 相似內容推薦 ({len(results)} 個):"]
                formatted_results.append("")
                
                for idx, result in enumerate(results, 1):
                    entry = result.entry
                    metadata = entry.metadata
                    
                    formatted_results.append(f"**{idx}. {metadata.title}**")
                    formatted_results.append(f"   🎯 相似度: {result.similarity_score:.2f}")
                    formatted_results.append(f"   📊 品質: {metadata.quality_score:.2f}")
                    formatted_results.append(f"   📈 使用次數: {metadata.usage_count}")
                    formatted_results.append(f"   🏷️ 標籤: {', '.join(metadata.tags) if metadata.tags else '無'}")
                    
                    # 內容摘要
                    content_preview = entry.content[:150] + "..." if len(entry.content) > 150 else entry.content
                    formatted_results.append(f"   📄 摘要: {content_preview}")
                    
                    # 相似性原因
                    if result.match_reasons:
                        formatted_results.append(f"   🎯 相似原因: {', '.join(result.match_reasons)}")
                    
                    # 應用建議
                    formatted_results.append(f"   💡 如何應用: {result.usage_recommendation}")
                    formatted_results.append("")
                
                return formatted_results
                
            except Exception as e:
                logger.error(f"Suggest similar failed: {e}")
                return [f"❌ 推薦相似內容失敗: {str(e)}"]

        async def update_experience(
            ctx: Context,
            content_id: Annotated[str, Field(description="內容ID")],
            content_type: Annotated[str, Field(description="內容類型")] = "experience",
            feedback_type: Annotated[str, Field(description="反饋類型: success, failure, improvement")] = "success",
            feedback_notes: Annotated[str, Field(description="反饋詳細說明")] = "",
            quality_adjustment: Annotated[float, Field(description="品質調整 (-1.0 到 1.0)")] = 0.0,
        ) -> str:
            """
            更新經驗反饋，包括使用統計和品質評分。
            """
            await ctx.debug(f"Updating experience feedback for: {content_id}")
            
            try:
                # 轉換內容類型
                try:
                    parsed_content_type = ContentType(content_type)
                except ValueError:
                    return f"❌ 無效的內容類型: {content_type}"
                
                # 獲取現有內容
                existing_content = await self.ragbridge_connector.get_content_by_id(
                    content_id, parsed_content_type
                )
                
                if not existing_content:
                    return f"❌ 找不到內容 ID: {content_id}"
                
                # 準備更新資料
                current_metadata = existing_content.metadata
                updates = {
                    "updated_at": datetime.now().isoformat(),
                    "usage_count": current_metadata.usage_count + 1,
                }
                
                # 根據反饋類型更新統計
                if feedback_type == "success":
                    new_success_count = getattr(current_metadata, 'success_count', 0) + 1
                    total_usage = updates["usage_count"]
                    updates["success_rate"] = new_success_count / total_usage if total_usage > 0 else 0.0
                    updates["success_count"] = new_success_count
                elif feedback_type == "failure":
                    failure_count = getattr(current_metadata, 'failure_count', 0) + 1
                    updates["failure_count"] = failure_count
                    success_count = getattr(current_metadata, 'success_count', 0)
                    total_usage = updates["usage_count"]
                    updates["success_rate"] = success_count / total_usage if total_usage > 0 else 0.0
                
                # 調整品質分數
                if quality_adjustment != 0.0:
                    new_quality = max(0.0, min(1.0, current_metadata.quality_score + quality_adjustment))
                    updates["quality_score"] = new_quality
                
                # 添加反饋記錄
                if feedback_notes:
                    feedback_history = getattr(current_metadata, 'feedback_history', [])
                    feedback_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "type": feedback_type,
                        "notes": feedback_notes,
                        "quality_adjustment": quality_adjustment,
                    })
                    updates["feedback_history"] = feedback_history
                
                # 更新內容
                await self.ragbridge_connector.update_content_metadata(
                    content_id, parsed_content_type, updates
                )
                
                return f"✅ 已更新 {content_id} 的反饋資料 (類型: {feedback_type})"
                
            except Exception as e:
                logger.error(f"Update experience failed: {e}")
                return f"❌ 更新經驗反饋失敗: {str(e)}"

        # 註冊 RAG Bridge 工具
        self.tool(
            search_experience,
            name="search-experience",
            description="搜尋個人經驗知識庫，支援多種內容類型和智能排序",
        )
        
        self.tool(
            get_process_workflow,
            name="get-process-workflow",
            description="獲取特定流程的工作流程步驟，支援結構化流程展示",
        )
        
        self.tool(
            suggest_similar,
            name="suggest-similar",
            description="根據參考內容推薦相關的經驗和知識",
        )
        
        # 只在非唯讀模式下註冊更新工具
        if not self.qdrant_settings.read_only:
            self.tool(
                update_experience,
                name="update-experience",
                description="更新經驗反饋，包括使用統計和品質評分",
            )

        # 詞彙管理工具集 (Task 141)
        async def search_vocabulary(
            ctx: Context,
            query: Annotated[str, Field(description="搜尋詞彙的查詢字串")] = "",
            domain: Annotated[str | None, Field(description="詞彙領域過濾")] = None,
            status: Annotated[str | None, Field(description="詞彙狀態過濾")] = None,
            limit: Annotated[int, Field(description="最多返回結果數量")] = 10,
        ) -> list[str]:
            """
            搜尋和瀏覽標準化詞彙庫，支援領域和狀態過濾。
            """
            await ctx.debug(f"Searching vocabulary for: {query}")
            
            try:
                result = await vocabulary_api.search_vocabulary(
                    query=query,
                    domain=domain,
                    status=status,
                    limit=limit
                )
                
                if "error" in result:
                    return [f"❌ 搜尋詞彙失敗: {result['error']}"]
                
                formatted_result = [f"🔍 詞彙搜尋結果 '{query}' ({result['total_results']} 個結果):"]
                formatted_result.append("")
                
                for idx, vocab in enumerate(result['results'], 1):
                    formatted_result.append(f"**{idx}. {vocab['term']}** ({vocab['match_type']})")
                    formatted_result.append(f"   📂 領域: {vocab['domain']}")
                    formatted_result.append(f"   📊 狀態: {vocab['status']}")
                    formatted_result.append(f"   📈 使用次數: {vocab['usage_count']}")
                    
                    if vocab['synonyms']:
                        formatted_result.append(f"   🔗 同義詞: {', '.join(vocab['synonyms'])}")
                    
                    if vocab['definition']:
                        formatted_result.append(f"   📝 定義: {vocab['definition']}")
                    
                    formatted_result.append("")
                
                return formatted_result
                
            except Exception as e:
                logger.error(f"Search vocabulary failed: {e}")
                return [f"❌ 搜尋詞彙失敗: {str(e)}"]

        async def propose_vocabulary(
            ctx: Context,
            term: Annotated[str, Field(description="提議的新詞彙")],
            domain: Annotated[str, Field(description="詞彙所屬領域")],
            definition: Annotated[str, Field(description="詞彙定義")] = "",
            synonyms: Annotated[list[str] | None, Field(description="同義詞列表")] = None,
        ) -> str:
            """
            提議新的標準化詞彙項目，需要後續審核批准。
            """
            await ctx.debug(f"Proposing new vocabulary term: {term}")
            
            try:
                result = await vocabulary_api.propose_vocabulary(
                    term=term,
                    domain=domain,
                    definition=definition,
                    synonyms=synonyms or []
                )
                
                if result['success']:
                    return f"✅ {result['message']}"
                else:
                    return f"❌ {result['message']}"
                    
            except Exception as e:
                logger.error(f"Propose vocabulary failed: {e}")
                return f"❌ 提議詞彙失敗: {str(e)}"

        async def standardize_content(
            ctx: Context,
            content: Annotated[str, Field(description="要標準化的內容文本")],
            tags: Annotated[list[str] | None, Field(description="內容標籤列表")] = None,
        ) -> list[str]:
            """
            標準化內容和標籤，提供詞彙建議和優化。
            """
            await ctx.debug(f"Standardizing content: {content[:100]}...")
            
            try:
                result = await vocabulary_api.standardize_content(
                    content=content,
                    tags=tags or []
                )
                
                if "error" in result:
                    return [f"❌ 標準化失敗: {result['error']}"]
                
                formatted_result = ["🔧 內容標準化結果:"]
                formatted_result.append("")
                
                # 標籤標準化
                formatted_result.append("🏷️ **標籤標準化:**")
                if result['original_tags']:
                    formatted_result.append(f"   原始: {', '.join(result['original_tags'])}")
                formatted_result.append(f"   標準化: {', '.join(result['standardized_tags'])}")
                
                if result['suggested_additional_tags']:
                    formatted_result.append(f"   建議新增: {', '.join(result['suggested_additional_tags'])}")
                formatted_result.append("")
                
                # 詞彙建議
                if result['vocabulary_suggestions']:
                    formatted_result.append("💡 **詞彙標準化建議:**")
                    for suggestion in result['vocabulary_suggestions']:
                        formatted_result.append(f"   • '{suggestion['original']}' → '{suggestion['suggested']}' ({suggestion['reason']})")
                    formatted_result.append("")
                
                # 相關詞彙
                if result['related_terms']:
                    formatted_result.append("🔗 **相關詞彙:**")
                    formatted_result.append(f"   {', '.join(result['related_terms'])}")
                
                return formatted_result
                
            except Exception as e:
                logger.error(f"Standardize content failed: {e}")
                return [f"❌ 標準化內容失敗: {str(e)}"]

        async def get_vocabulary_statistics(ctx: Context) -> list[str]:
            """
            獲取詞彙管理系統的統計資訊和健康狀態。
            """
            await ctx.debug("Getting vocabulary statistics")
            
            try:
                result = await vocabulary_api.get_vocabulary_statistics()
                
                if "error" in result:
                    return [f"❌ 獲取統計失敗: {result['error']}"]
                
                vocab_stats = result['vocabulary_statistics']
                fragment_stats = result['fragment_statistics']
                health = result['system_health']
                
                formatted_result = ["📊 **詞彙管理系統統計**"]
                formatted_result.append("")
                
                # 詞彙統計
                formatted_result.append("📚 **詞彙庫統計:**")
                formatted_result.append(f"   總詞彙數: {vocab_stats['total_terms']}")
                formatted_result.append(f"   同義詞數: {vocab_stats['total_synonyms']}")
                formatted_result.append(f"   總使用次數: {vocab_stats['total_usage']}")
                formatted_result.append(f"   平均使用次數: {vocab_stats['average_usage']:.1f}")
                formatted_result.append("")
                
                # 領域分佈
                formatted_result.append("🗂️ **領域分佈:**")
                for domain, count in vocab_stats['domain_distribution'].items():
                    if count > 0:
                        formatted_result.append(f"   {domain}: {count} 個詞彙")
                formatted_result.append("")
                
                # 狀態分佈
                formatted_result.append("📈 **詞彙狀態:**")
                for status, count in vocab_stats['status_distribution'].items():
                    if count > 0:
                        formatted_result.append(f"   {status}: {count} 個詞彙")
                formatted_result.append("")
                
                # 最常用詞彙
                if vocab_stats['most_used_terms']:
                    formatted_result.append("🔥 **最常用詞彙:**")
                    for term in vocab_stats['most_used_terms'][:5]:
                        formatted_result.append(f"   • {term['term']} ({term['domain']}) - {term['usage_count']} 次")
                    formatted_result.append("")
                
                # 分片統計
                formatted_result.append("📄 **分片統計:**")
                formatted_result.append(f"   總分片數: {fragment_stats['total_fragments']}")
                formatted_result.append(f"   平均品質: {fragment_stats['average_quality']:.2f}")
                formatted_result.append(f"   總使用次數: {fragment_stats['total_usage']}")
                formatted_result.append("")
                
                # 系統健康
                formatted_result.append("💚 **系統健康:**")
                formatted_result.append(f"   詞彙覆蓋率: {health['vocabulary_coverage']:.1%}")
                formatted_result.append(f"   使用活躍度: {health['usage_activity']:.1%}")
                formatted_result.append(f"   領域多樣性: {health['domain_diversity']:.1%}")
                
                return formatted_result
                
            except Exception as e:
                logger.error(f"Get vocabulary statistics failed: {e}")
                return [f"❌ 獲取統計失敗: {str(e)}"]

        async def manage_fragment_vocabulary(
            ctx: Context,
            action: Annotated[str, Field(description="操作類型: search, create, analyze")],
            fragment_type: Annotated[str | None, Field(description="分片類型過濾")] = None,
            query: Annotated[str, Field(description="搜尋查詢")] = "",
            limit: Annotated[int, Field(description="結果限制")] = 5,
        ) -> list[str]:
            """
            管理分片詞彙，包括搜尋、創建和分析分片。
            """
            await ctx.debug(f"Managing fragment vocabulary: {action}")
            
            try:
                if action == "search":
                    # 搜尋分片
                    search_params = {"limit": limit}
                    if query:
                        search_params["query"] = query
                    if fragment_type:
                        # 這裡需要轉換字符串到枚舉
                        from mcp_server_qdrant.ragbridge.vocabulary import FragmentType
                        try:
                            ftype = FragmentType(fragment_type)
                            search_params["fragment_types"] = [ftype]
                        except ValueError:
                            return [f"❌ 無效的分片類型: {fragment_type}"]
                    
                    results = fragment_manager.search_fragments(**search_params)
                    
                    formatted_result = [f"🔍 分片搜尋結果 ({len(results)} 個):"]
                    formatted_result.append("")
                    
                    for idx, item in enumerate(results, 1):
                        fragment = item['fragment']
                        score = item['relevance_score']
                        
                        formatted_result.append(f"**{idx}. {fragment['title']}**")
                        formatted_result.append(f"   📝 類型: {fragment['fragment_type']}")
                        formatted_result.append(f"   🎯 相關性: {score:.2f}")
                        formatted_result.append(f"   📊 品質: {fragment['quality_score']:.2f}")
                        formatted_result.append(f"   🏷️ 標籤: {', '.join(fragment['tags'])}")
                        formatted_result.append(f"   📈 使用: {fragment['usage_count']} 次")
                        
                        # 相關分片
                        if item['related_fragments']:
                            related_info = [f"{r[1]}({r[2]:.1f})" for r in item['related_fragments'][:2]]
                            formatted_result.append(f"   🔗 相關: {', '.join(related_info)}")
                        
                        formatted_result.append("")
                
                elif action == "analyze":
                    # 分析分片統計
                    stats = fragment_manager.get_fragment_statistics()
                    
                    formatted_result = ["📊 **分片詞彙分析**"]
                    formatted_result.append("")
                    
                    formatted_result.append(f"📄 總分片數: {stats['total_fragments']}")
                    formatted_result.append(f"📈 總使用次數: {stats['total_usage']}")
                    formatted_result.append(f"📊 平均品質: {stats['average_quality']:.2f}")
                    formatted_result.append(f"🏷️ 總標籤數: {stats['total_tags']}")
                    formatted_result.append(f"🔑 總關鍵詞: {stats['total_keywords']}")
                    formatted_result.append("")
                    
                    formatted_result.append("📝 **類型分佈:**")
                    for ftype, count in stats['type_distribution'].items():
                        if count > 0:
                            formatted_result.append(f"   {ftype}: {count}")
                    formatted_result.append("")
                    
                    formatted_result.append("🗂️ **領域分佈:**")
                    for domain, count in stats['domain_distribution'].items():
                        if count > 0:
                            formatted_result.append(f"   {domain}: {count}")
                
                else:
                    return [f"❌ 不支援的操作: {action}"]
                
                return formatted_result
                
            except Exception as e:
                logger.error(f"Manage fragment vocabulary failed: {e}")
                return [f"❌ 管理分片詞彙失敗: {str(e)}"]

        # 註冊詞彙管理工具
        self.tool(
            search_vocabulary,
            name="search-vocabulary",
            description="搜尋和瀏覽標準化詞彙庫，支援領域和狀態過濾",
        )
        
        self.tool(
            standardize_content,
            name="standardize-content",
            description="標準化內容和標籤，提供詞彙建議和優化",
        )
        
        self.tool(
            get_vocabulary_statistics,
            name="get-vocabulary-statistics",
            description="獲取詞彙管理系統的統計資訊和健康狀態",
        )
        
        self.tool(
            manage_fragment_vocabulary,
            name="manage-fragment-vocabulary",
            description="管理分片詞彙，包括搜尋、創建和分析分片",
        )
        
        # 只在非唯讀模式下註冊編輯工具
        if not self.qdrant_settings.read_only:
            self.tool(
                propose_vocabulary,
                name="propose-vocabulary",
                description="提議新的標準化詞彙項目，需要後續審核批准",
            )

        # Schema 管理工具集 (Task 142)
        async def get_current_schema(
            ctx: Context,
        ) -> list[str]:
            """
            獲取當前活躍的 Schema 版本及其詳細資訊。
            """
            await ctx.debug("Getting current schema")
            
            try:
                result = await schema_api.get_current_schema()
                
                if "error" in result:
                    return [f"❌ 獲取 Schema 失敗: {result['error']}"]
                
                # 格式化輸出
                output = [
                    f"📋 **當前 Schema 版本: {result['schema_version']}**",
                    f"📝 描述: {result['description']}",
                    f"📅 建立時間: {result['created_at']}",
                    f"🔄 活躍狀態: {'是' if result['is_active'] else '否'}",
                    f"🔗 向後兼容: {'是' if result['backward_compatible'] else '否'}",
                    "",
                    f"📊 **統計資訊:**",
                    f"- 總欄位數: {result['total_fields']}",
                    f"- 核心欄位數: {result['core_fields_count']}",
                    f"- 已棄用欄位數: {result['deprecated_fields_count']}",
                    "",
                    f"🏗️ **欄位定義:**"
                ]
                
                for field_name, field_info in result['fields'].items():
                    status = "🔴 (已棄用)" if field_info['deprecated'] else "✅"
                    core = "🔒 (核心)" if field_info['is_core'] else ""
                    required = "⚠️ (必填)" if field_info['required'] else ""
                    
                    output.append(f"- **{field_name}** {status} {core} {required}")
                    output.append(f"  - 類型: {field_info['type']}")
                    if field_info['description']:
                        output.append(f"  - 描述: {field_info['description']}")
                    output.append(f"  - 新增於版本: {field_info['added_in_version']}")
                    output.append("")
                
                return output
                
            except Exception as e:
                logger.error(f"Get current schema failed: {e}")
                return [f"❌ 獲取當前 Schema 失敗: {str(e)}"]

        async def request_schema_field_addition(
            ctx: Context,
            field_name: Annotated[str, Field(description="欄位名稱")],
            field_type: Annotated[str, Field(description="欄位類型 (string, integer, float, boolean, datetime, list, dict, json)")],
            description: Annotated[str, Field(description="欄位描述")] = "",
            required: Annotated[bool, Field(description="是否為必填欄位")] = False,
            justification: Annotated[str, Field(description="變更理由說明")] = "",
            proposed_by: Annotated[str, Field(description="提案者身份")] = "mcp_user",
            min_length: Annotated[int | None, Field(description="最小長度限制")] = None,
            max_length: Annotated[int | None, Field(description="最大長度限制")] = None,
            pattern: Annotated[str | None, Field(description="正則表達式模式")] = None,
            min_value: Annotated[float | None, Field(description="最小值限制")] = None,
            max_value: Annotated[float | None, Field(description="最大值限制")] = None,
            allowed_values: Annotated[list[str] | None, Field(description="允許的值列表")] = None,
        ) -> list[str]:
            """
            請求新增 Schema 欄位，將創建審查請求而非直接執行變更。
            """
            await ctx.debug(f"Requesting schema field addition: {field_name}")
            
            try:
                # 組建變更詳情
                change_details = {
                    "field_type": field_type,
                    "description": description,
                    "required": required
                }
                
                # 組建驗證規則
                validation_rules = {}
                if min_length is not None:
                    validation_rules["min_length"] = min_length
                if max_length is not None:
                    validation_rules["max_length"] = max_length
                if pattern is not None:
                    validation_rules["pattern"] = pattern
                if min_value is not None:
                    validation_rules["min_value"] = min_value
                if max_value is not None:
                    validation_rules["max_value"] = max_value
                if allowed_values is not None:
                    validation_rules["allowed_values"] = allowed_values
                
                if validation_rules:
                    change_details["validation"] = validation_rules
                
                # 創建審查請求
                approval_manager = get_approval_manager()
                request_id = approval_manager.create_change_request(
                    change_type="add_field",
                    field_name=field_name,
                    change_details=change_details,
                    proposed_by=proposed_by,
                    justification=justification
                )
                
                # 檢查是否已自動核准
                if request_id in approval_manager.approval_history:
                    # 已自動執行
                    executed_request = next(
                        req for req in approval_manager.approval_history 
                        if req.request_id == request_id
                    )
                    
                    if executed_request.status == "approved":
                        return [
                            f"✅ 低風險變更已自動核准並執行",
                            f"📋 請求ID: {request_id}",
                            f"🏗️ 新增欄位: {field_name} ({field_type})",
                            f"📝 說明: {executed_request.review_comments}"
                        ]
                    else:
                        return [
                            f"❌ 自動執行失敗",
                            f"📋 請求ID: {request_id}",
                            f"💬 錯誤: {executed_request.review_comments}"
                        ]
                else:
                    # 等待審查
                    pending_request = approval_manager.pending_requests[request_id]
                    return [
                        f"📋 **Schema 變更請求已創建**",
                        f"🆔 請求ID: {request_id}",
                        f"🏗️ 變更類型: 新增欄位 '{field_name}'",
                        f"⚠️ 風險級別: {pending_request.risk_level.value}",
                        f"👥 需要審查級別: {pending_request.required_approval_level.value}",
                        f"📝 提案理由: {justification}",
                        "",
                        f"⏳ **狀態: 等待審查**",
                        f"💡 使用 'review-schema-request' 工具進行審查",
                        f"🔍 使用 'list-pending-schema-requests' 查看所有待審查請求"
                    ]
                
            except Exception as e:
                logger.error(f"Request schema field addition failed: {e}")
                return [f"❌ 創建 Schema 變更請求失敗: {str(e)}"]

        async def request_schema_field_removal(
            ctx: Context,
            field_name: Annotated[str, Field(description="要移除的欄位名稱")],
            justification: Annotated[str, Field(description="移除理由說明")] = "",
            proposed_by: Annotated[str, Field(description="提案者身份")] = "mcp_user",
        ) -> list[str]:
            """
            請求移除 Schema 欄位，將創建審查請求（高風險操作）。
            """
            await ctx.debug(f"Requesting schema field removal: {field_name}")
            
            try:
                # 創建審查請求
                approval_manager = get_approval_manager()
                request_id = approval_manager.create_change_request(
                    change_type="remove_field",
                    field_name=field_name,
                    change_details={"deprecated": True},
                    proposed_by=proposed_by,
                    justification=justification
                )
                
                # 移除欄位是高風險操作，不會自動核准
                pending_request = approval_manager.pending_requests[request_id]
                return [
                    f"⚠️ **高風險 Schema 變更請求已創建**",
                    f"🆔 請求ID: {request_id}",
                    f"🗑️ 變更類型: 移除欄位 '{field_name}'",
                    f"🔴 風險級別: {pending_request.risk_level.value}",
                    f"👥 需要審查級別: {pending_request.required_approval_level.value}",
                    f"📝 移除理由: {justification}",
                    "",
                    f"⏳ **狀態: 等待高級審查**",
                    f"💡 需要管理員權限進行審查",
                    f"🔍 使用 'review-schema-request' 工具進行審查"
                ]
                
            except Exception as e:
                logger.error(f"Request schema field removal failed: {e}")
                return [f"❌ 創建移除欄位請求失敗: {str(e)}"]

        async def validate_schema_data(
            ctx: Context,
            data: Annotated[dict, Field(description="要驗證的數據")],
            schema_version: Annotated[str | None, Field(description="指定的 Schema 版本")] = None,
        ) -> list[str]:
            """
            驗證數據是否符合 Schema 規範。
            """
            await ctx.debug(f"Validating data against schema version: {schema_version or 'current'}")
            
            try:
                result = await schema_api.validate_data(data, schema_version)
                
                output = [
                    f"🔍 **Schema 驗證結果**",
                    f"📋 使用 Schema 版本: {result['schema_version']}",
                    f"✅ 驗證結果: {'通過' if result['is_valid'] else '失敗'}",
                    f"❌ 錯誤數量: {result['error_count']}",
                    f"💬 {result['message']}"
                ]
                
                if result["validation_errors"]:
                    output.append("")
                    output.append("🚨 **驗證錯誤詳情:**")
                    for i, error in enumerate(result["validation_errors"], 1):
                        output.append(f"{i}. {error}")
                
                return output
                
            except Exception as e:
                logger.error(f"Validate schema data failed: {e}")
                return [f"❌ Schema 數據驗證失敗: {str(e)}"]

        async def analyze_schema_usage(
            ctx: Context,
            data_samples: Annotated[list[dict], Field(description="用於分析的數據樣本列表")],
        ) -> list[str]:
            """
            分析 Schema 使用情況，提供優化建議。
            """
            await ctx.debug(f"Analyzing schema usage with {len(data_samples)} samples")
            
            try:
                result = await schema_api.analyze_schema_usage(data_samples)
                
                if "error" in result:
                    return [f"❌ {result['error']}"]
                
                summary = result.get("summary", {})
                
                output = [
                    f"📊 **Schema 使用情況分析**",
                    f"📋 當前 Schema 版本: {result['current_schema_version']}",
                    f"📦 分析樣本數量: {result['total_samples']}",
                    f"✅ Schema 合規率: {result['schema_compliance_rate']:.1%}",
                    f"🎯 合規等級: {summary.get('compliance_level', 'unknown')}",
                    "",
                    f"📈 **高使用率欄位 (>80%):**"
                ]
                
                for field in summary.get("high_usage_fields", []):
                    output.append(f"  - {field}")
                
                output.append("")
                output.append(f"📉 **低使用率欄位 (<20%):**")
                for field in summary.get("low_usage_fields", []):
                    output.append(f"  - {field}")
                
                if result.get("unknown_fields"):
                    output.append("")
                    output.append(f"🔍 **未定義但常用的欄位:**")
                    for field in result["unknown_fields"]:
                        output.append(f"  - {field}")
                
                if summary.get("suggestions_available"):
                    output.append("")
                    output.append("💡 建議使用 get-schema-suggestions 工具獲取詳細改進建議")
                
                return output
                
            except Exception as e:
                logger.error(f"Analyze schema usage failed: {e}")
                return [f"❌ Schema 使用分析失敗: {str(e)}"]

        async def get_schema_suggestions(
            ctx: Context,
            data_samples: Annotated[list[dict], Field(description="用於生成建議的數據樣本列表")],
        ) -> list[str]:
            """
            基於數據使用情況獲取 Schema 改進建議。
            """
            await ctx.debug(f"Getting schema suggestions based on {len(data_samples)} samples")
            
            try:
                result = await schema_api.get_schema_suggestions(data_samples)
                
                if "error" in result:
                    return [f"❌ {result['error']}"]
                
                output = [
                    f"💡 **Schema 改進建議**",
                    f"📊 分析基礎: {result['analysis_summary']['total_samples']} 個樣本",
                    f"📋 Schema 版本: {result['analysis_summary']['schema_version']}",
                    f"✅ 合規率: {result['analysis_summary']['compliance_rate']:.1%}",
                    f"🔍 建議數量: {result['suggestion_count']}",
                    "",
                    f"📝 {result['message']}"
                ]
                
                if result["suggestions"]:
                    output.append("")
                    output.append("🎯 **具體建議:**")
                    
                    for i, suggestion in enumerate(result["suggestions"], 1):
                        priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                            suggestion.get("priority", "low"), "⚪"
                        )
                        
                        output.append(f"{i}. {priority_emoji} **{suggestion['type']}**: {suggestion['field_name']}")
                        output.append(f"   原因: {suggestion['reason']}")
                        output.append("")
                else:
                    output.append("")
                    output.append("🎉 目前 Schema 設計良好，無需調整！")
                
                return output
                
            except Exception as e:
                logger.error(f"Get schema suggestions failed: {e}")
                return [f"❌ 獲取 Schema 建議失敗: {str(e)}"]

        async def get_schema_evolution_history(
            ctx: Context,
        ) -> list[str]:
            """
            獲取 Schema 演進歷史記錄。
            """
            await ctx.debug("Getting schema evolution history")
            
            try:
                result = await schema_api.get_schema_evolution_history()
                
                if "error" in result:
                    return [f"❌ {result['error']}"]
                
                summary = result.get("summary", {})
                
                output = [
                    f"📚 **Schema 演進歷史**",
                    f"📊 總版本數: {result['total_versions']}",
                    f"✅ 活躍版本數: {result['active_versions']}",
                    f"🔄 總遷移數: {result['total_migrations']}",
                    "",
                    f"🏁 首個版本: {summary.get('first_version', 'N/A')}",
                    f"🚀 最新版本: {summary.get('latest_version', 'N/A')}",
                    f"🔥 變更最多版本: {summary.get('most_changes', 'N/A')}",
                    "",
                    f"📋 **版本詳情:**"
                ]
                
                for version_info in result["evolution_history"]:
                    status = "✅ 活躍" if version_info["is_active"] else "⏸️ 非活躍"
                    compat = "🔗 兼容" if version_info["backward_compatible"] else "⚠️ 破壞性"
                    
                    output.append(f"**版本 {version_info['version']}** {status} {compat}")
                    output.append(f"  - 描述: {version_info['description']}")
                    output.append(f"  - 建立時間: {version_info['created_at']}")
                    output.append(f"  - 欄位數量: {version_info['field_count']}")
                    
                    if version_info["migrations"]:
                        output.append(f"  - 遷移記錄:")
                        for migration in version_info["migrations"]:
                            output.append(f"    * {migration['type']}: {migration['field']} (來自 {migration['from_version']})")
                    
                    output.append("")
                
                return output
                
            except Exception as e:
                logger.error(f"Get schema evolution history failed: {e}")
                return [f"❌ 獲取 Schema 演進歷史失敗: {str(e)}"]

        # Schema 審查管理工具
        async def list_pending_schema_requests(
            ctx: Context,
            reviewer: Annotated[str, Field(description="審查者身份（用於權限過濾）")] = "admin",
        ) -> list[str]:
            """
            列出所有待審查的 Schema 變更請求。
            """
            await ctx.debug(f"Listing pending schema requests for reviewer: {reviewer}")
            
            try:
                approval_manager = get_approval_manager()
                pending_requests = approval_manager.get_pending_requests(reviewer)
                
                if not pending_requests:
                    return [
                        "✅ **目前沒有待審查的 Schema 變更請求**",
                        "🎉 所有變更請求都已處理完成"
                    ]
                
                output = [
                    f"📋 **待審查的 Schema 變更請求 ({len(pending_requests)} 個)**",
                    ""
                ]
                
                for i, request in enumerate(pending_requests, 1):
                    risk_emoji = {
                        "low": "🟢",
                        "medium": "🟡", 
                        "high": "🔴",
                        "critical": "🚨"
                    }.get(request["risk_level"], "⚪")
                    
                    output.extend([
                        f"**{i}. 請求 {request['request_id'][:8]}...** {risk_emoji}",
                        f"   🔧 變更類型: {request['change_type']}",
                        f"   🏗️ 欄位名稱: {request['field_name']}",
                        f"   ⚠️ 風險級別: {request['risk_level']}",
                        f"   👥 需要權限: {request['required_approval_level']}",
                        f"   👤 提案者: {request['proposed_by']}",
                        f"   📅 提案時間: {request['proposed_at'][:19]}",
                        f"   📝 理由: {request['justification'][:100]}{'...' if len(request['justification']) > 100 else ''}",
                        ""
                    ])
                
                output.extend([
                    "💡 **審查指令:**",
                    "✅ 核准: review-schema-request <request_id> approve <reviewer> [comments]",
                    "❌ 拒絕: review-schema-request <request_id> reject <reviewer> [comments]"
                ])
                
                return output
                
            except Exception as e:
                logger.error(f"List pending schema requests failed: {e}")
                return [f"❌ 列出待審查請求失敗: {str(e)}"]

        async def review_schema_request(
            ctx: Context,
            request_id: Annotated[str, Field(description="要審查的請求ID")],
            action: Annotated[str, Field(description="審查動作: approve 或 reject")],
            reviewer: Annotated[str, Field(description="審查者身份")],
            comments: Annotated[str, Field(description="審查意見")] = "",
        ) -> list[str]:
            """
            審查 Schema 變更請求，進行核准或拒絕。
            """
            await ctx.debug(f"Reviewing schema request {request_id}: {action} by {reviewer}")
            
            try:
                if action not in ["approve", "reject"]:
                    return [f"❌ 無效的審查動作: {action}，請使用 'approve' 或 'reject'"]
                
                approval_manager = get_approval_manager()
                
                # 檢查請求是否存在
                if request_id not in approval_manager.pending_requests:
                    return [
                        f"❌ 請求不存在: {request_id}",
                        "🔍 使用 'list-pending-schema-requests' 查看所有待審查請求"
                    ]
                
                # 執行審查
                success = approval_manager.review_request(
                    request_id=request_id,
                    reviewer=reviewer,
                    action=action,
                    comments=comments
                )
                
                if not success:
                    return [
                        f"❌ 審查失敗: 權限不足或請求不存在",
                        f"👤 審查者: {reviewer}",
                        f"📋 請求ID: {request_id}",
                        "💡 請檢查您的審查權限"
                    ]
                
                # 從歷史記錄中找到審查結果
                reviewed_request = next(
                    req for req in approval_manager.approval_history
                    if req.request_id == request_id
                )
                
                if action == "approve":
                    if reviewed_request.status == "approved":
                        return [
                            f"✅ **Schema 變更請求已核准並執行**",
                            f"📋 請求ID: {request_id}",
                            f"🏗️ 變更類型: {reviewed_request.change_type}",
                            f"🔧 欄位: {reviewed_request.field_name}",
                            f"👤 審查者: {reviewer}",
                            f"📝 審查意見: {comments}",
                            f"⏰ 審查時間: {reviewed_request.reviewed_at.strftime('%Y-%m-%d %H:%M:%S')}",
                            "",
                            f"🎉 Schema 變更已成功應用！"
                        ]
                    else:
                        return [
                            f"❌ **Schema 變更執行失敗**",
                            f"📋 請求ID: {request_id}",
                            f"👤 審查者: {reviewer}",
                            f"💬 錯誤: {reviewed_request.review_comments}",
                            "🔧 請檢查 Schema 定義是否正確"
                        ]
                else:  # reject
                    return [
                        f"❌ **Schema 變更請求已拒絕**",
                        f"📋 請求ID: {request_id}",
                        f"🏗️ 變更類型: {reviewed_request.change_type}",
                        f"🔧 欄位: {reviewed_request.field_name}",
                        f"👤 審查者: {reviewer}",
                        f"📝 拒絕理由: {comments}",
                        f"⏰ 審查時間: {reviewed_request.reviewed_at.strftime('%Y-%m-%d %H:%M:%S')}"
                    ]
                
            except Exception as e:
                logger.error(f"Review schema request failed: {e}")
                return [f"❌ 審查 Schema 請求失敗: {str(e)}"]

        async def get_schema_approval_history(
            ctx: Context,
            limit: Annotated[int, Field(description="返回的歷史記錄數量")] = 10,
        ) -> list[str]:
            """
            獲取 Schema 審查歷史記錄。
            """
            await ctx.debug(f"Getting schema approval history (limit: {limit})")
            
            try:
                approval_manager = get_approval_manager()
                history = approval_manager.get_approval_history(limit)
                
                if not history:
                    return [
                        "📋 **暫無 Schema 審查歷史記錄**",
                        "💡 當有 Schema 變更請求時，記錄會顯示在這裡"
                    ]
                
                output = [
                    f"📚 **Schema 審查歷史記錄 (最近 {len(history)} 個)**",
                    ""
                ]
                
                for i, record in enumerate(history, 1):
                    status_emoji = {"approved": "✅", "rejected": "❌"}.get(record["status"], "⏳")
                    risk_emoji = {
                        "low": "🟢",
                        "medium": "🟡",
                        "high": "🔴", 
                        "critical": "🚨"
                    }.get(record["risk_level"], "⚪")
                    
                    output.extend([
                        f"**{i}. 請求 {record['request_id'][:8]}...** {status_emoji} {risk_emoji}",
                        f"   🔧 變更: {record['change_type']} → {record['field_name']}",
                        f"   👤 提案者: {record['proposed_by']}",
                        f"   👥 審查者: {record['reviewed_by'] or 'N/A'}",
                        f"   📅 提案時間: {record['proposed_at'][:19]}",
                        f"   ⏰ 審查時間: {record['reviewed_at'][:19] if record['reviewed_at'] else 'N/A'}",
                        f"   📝 審查意見: {record['review_comments'][:80]}{'...' if len(record['review_comments']) > 80 else ''}",
                        ""
                    ])
                
                return output
                
            except Exception as e:
                logger.error(f"Get schema approval history failed: {e}")
                return [f"❌ 獲取審查歷史失敗: {str(e)}"]

        # 註冊 Schema 管理工具
        self.tool(
            get_current_schema,
            name="get-current-schema",
            description="獲取當前活躍的 Schema 版本及其詳細資訊",
        )
        
        self.tool(
            validate_schema_data,
            name="validate-schema-data", 
            description="驗證數據是否符合 Schema 規範",
        )
        
        self.tool(
            analyze_schema_usage,
            name="analyze-schema-usage",
            description="分析 Schema 使用情況，提供統計資訊",
        )
        
        self.tool(
            get_schema_suggestions,
            name="get-schema-suggestions",
            description="基於數據使用情況獲取 Schema 改進建議",
        )
        
        self.tool(
            get_schema_evolution_history,
            name="get-schema-evolution-history",
            description="獲取 Schema 演進歷史記錄",
        )
        
        self.tool(
            list_pending_schema_requests,
            name="list-pending-schema-requests",
            description="列出所有待審查的 Schema 變更請求",
        )
        
        self.tool(
            review_schema_request,
            name="review-schema-request",
            description="審查 Schema 變更請求，進行核准或拒絕",
        )
        
        self.tool(
            get_schema_approval_history,
            name="get-schema-approval-history",
            description="獲取 Schema 審查歷史記錄",
        )
        
        # 只在非唯讀模式下註冊變更請求工具
        if not self.qdrant_settings.read_only:
            self.tool(
                request_schema_field_addition,
                name="request-schema-field-addition",
                description="請求新增 Schema 欄位，將創建審查請求而非直接執行變更",
            )
            
            self.tool(
                request_schema_field_removal,
                name="request-schema-field-removal",
                description="請求移除 Schema 欄位，將創建審查請求（高風險操作）",
            )

        # 權限管理工具集 - 所有用戶都可以查看權限狀態
        async def get_user_permissions(ctx: Context) -> list[str]:
            """
            獲取當前用戶的權限摘要和可用工具列表。
            """
            await ctx.debug("Getting user permissions")
            
            try:
                user_id = "default_user"  # 目前使用預設用戶
                summary = self.permission_manager.get_permission_summary(user_id)
                
                output = [
                    f"👤 **用戶權限資訊**",
                    f"🆔 用戶ID: {summary['user_id']}",
                    f"🔐 權限級別: {summary['permission_level']}",
                    f"🛠️ 可用工具總數: {summary['total_available_tools']}",
                    "",
                    f"🎯 **可執行操作類型:**"
                ]
                
                for operation in summary['available_operations']:
                    output.append(f"  ✅ {operation}")
                
                output.append("")
                output.append(f"🛠️ **按風險級別分類的可用工具:**")
                
                for risk_level, tools in summary['tools_by_risk'].items():
                    if tools:
                        risk_emoji = {"low": "🟢", "medium": "🟡", "critical": "🔴"}.get(risk_level, "⚪")
                        output.append(f"  {risk_emoji} **{risk_level.upper()} 風險 ({len(tools)} 個):**")
                        for tool in sorted(tools):
                            output.append(f"    - {tool}")
                        output.append("")
                
                if summary['permission_level'] == 'user':
                    output.append("💡 **提升權限:**")
                    output.append("如需使用管理功能，請聯絡管理員提升權限級別至 admin 或 super_admin")
                
                return output
                
            except Exception as e:
                logger.error(f"Get user permissions failed: {e}")
                return [f"❌ 獲取權限資訊失敗: {str(e)}"]

        async def check_tool_permission(
            ctx: Context,
            tool_name: Annotated[str, Field(description="要檢查的工具名稱")],
        ) -> list[str]:
            """
            檢查特定工具的使用權限。
            """
            await ctx.debug(f"Checking permission for tool: {tool_name}")
            
            try:
                user_id = "default_user"
                has_permission = self.permission_manager.check_tool_permission(user_id, tool_name)
                user_level = self.permission_manager.get_user_permission(user_id)
                tool_permission = self.permission_manager.tool_permissions.get(tool_name)
                
                if not tool_permission:
                    return [
                        f"❓ **工具權限查詢**",
                        f"🔧 工具名稱: {tool_name}",
                        f"⚠️ 狀態: 未定義的工具",
                        f"💬 說明: 此工具可能不存在或未在權限系統中註冊"
                    ]
                
                permission_emoji = "✅" if has_permission else "❌"
                risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴", "critical": "🚨"}.get(
                    tool_permission.risk_level, "⚪"
                )
                
                output = [
                    f"🔍 **工具權限檢查結果**",
                    f"🔧 工具名稱: {tool_name}",
                    f"{permission_emoji} 使用權限: {'允許' if has_permission else '拒絕'}",
                    f"👤 當前權限級別: {user_level.value}",
                    f"⚠️ 需要權限級別: {tool_permission.required_level.value}",
                    f"{risk_emoji} 風險級別: {tool_permission.risk_level}",
                    f"📝 工具說明: {tool_permission.description}",
                ]
                
                if not has_permission:
                    output.append("")
                    output.append("💡 **解決方案:**")
                    if tool_permission.required_level == PermissionLevel.ADMIN:
                        output.append("需要 admin 權限，請聯絡管理員提升權限級別")
                    elif tool_permission.required_level == PermissionLevel.SUPER_ADMIN:
                        output.append("需要 super_admin 權限，此為高風險操作，需要最高管理員權限")
                
                return output
                
            except Exception as e:
                logger.error(f"Check tool permission failed: {e}")
                return [f"❌ 檢查工具權限失敗: {str(e)}"]

        # 註冊權限管理工具
        self.tool(
            get_user_permissions,
            name="get-user-permissions",
            description="獲取當前用戶的權限摘要和可用工具列表",
        )
        
        self.tool(
            check_tool_permission,
            name="check-tool-permission",
            description="檢查特定工具的使用權限",
        )

        # 只有 super_admin 可以管理權限
        async def set_user_permission_level(
            ctx: Context,
            user_id: Annotated[str, Field(description="要設定的用戶ID")],
            permission_level: Annotated[str, Field(description="權限級別: user, admin, super_admin")],
        ) -> str:
            """
            設定用戶的權限級別（僅限超級管理員）。
            """
            await ctx.debug(f"Setting user {user_id} permission to {permission_level}")
            
            try:
                # 檢查當前用戶是否有權限執行此操作
                current_user = "default_user"
                current_level = self.permission_manager.get_user_permission(current_user)
                
                if current_level != PermissionLevel.SUPER_ADMIN:
                    return f"❌ 權限不足：只有 super_admin 可以管理用戶權限（當前級別: {current_level.value}）"
                
                # 驗證權限級別
                try:
                    new_level = PermissionLevel(permission_level)
                except ValueError:
                    return f"❌ 無效的權限級別: {permission_level}，請使用: user, admin, super_admin"
                
                # 設定權限
                self.permission_manager.set_user_permission(user_id, new_level)
                
                return f"✅ 成功設定用戶 {user_id} 的權限級別為 {new_level.value}"
                
            except Exception as e:
                logger.error(f"Set user permission failed: {e}")
                return f"❌ 設定用戶權限失敗: {str(e)}"

        # 只在啟用權限系統且為超級管理員時註冊
        if (self.qdrant_settings.enable_permission_system and 
            self.permission_manager.get_user_permission("default_user") == PermissionLevel.SUPER_ADMIN):
            self.tool(
                set_user_permission_level,
                name="set-user-permission-level",
                description="設定用戶的權限級別（僅限超級管理員）",
            )

        # 資料遷移工具集 - 需要管理員權限
        async def analyze_collection_for_migration(
            ctx: Context,
            collection_name: Annotated[str, Field(description="要分析的 collection 名稱")],
        ) -> list[str]:
            """
            分析舊 collection 的結構，為遷移做準備。
            """
            await ctx.debug(f"Analyzing collection for migration: {collection_name}")
            
            try:
                analysis = self.migration_tool.analyze_collection_structure(collection_name)
                
                output = [
                    f"📊 **Collection 分析結果**",
                    f"🏗️ Collection: {analysis['collection_name']}",
                    f"📦 總點數: {analysis['total_points']:,}",
                    f"🔢 向量維度: {analysis['vector_size']}",
                    f"📏 距離度量: {analysis['distance_metric']}",
                    f"📅 分析時間: {analysis['analyzed_at']}",
                    "",
                    f"🗃️ **欄位結構分析:**"
                ]
                
                for field, types in analysis['field_types'].items():
                    types_str = ', '.join(types)
                    samples = analysis['field_samples'].get(field, [])
                    sample_preview = str(samples[0]) if samples else "N/A"
                    if len(sample_preview) > 50:
                        sample_preview = sample_preview[:47] + "..."
                    
                    output.append(f"  📝 **{field}** ({types_str})")
                    output.append(f"    範例: {sample_preview}")
                
                # 生成建議的遷移計劃
                suggested_plan = self.migration_tool.suggest_migration_plan(analysis)
                
                output.extend([
                    "",
                    f"💡 **建議的遷移計劃:**",
                    f"🎯 目標內容類型: {suggested_plan.target_content_type.value}",
                    f"📋 預估記錄數: {suggested_plan.estimated_records:,}",
                    "",
                    f"🔄 **欄位映射建議:**"
                ])
                
                for old_field, new_field in suggested_plan.mapping_rules.items():
                    output.append(f"  {old_field} → {new_field}")
                
                output.extend([
                    "",
                    f"⚙️ **轉換規則:**",
                    f"  標準化詞彙: {'✅' if suggested_plan.transformation_rules.get('standardize_vocabulary') else '❌'}",
                    f"  提取關鍵詞: {'✅' if suggested_plan.transformation_rules.get('extract_keywords') else '❌'}",
                    f"  正規化標籤: {'✅' if suggested_plan.transformation_rules.get('normalize_tags') else '❌'}",
                    "",
                    f"💡 **下一步:** 使用 'create-migration-plan' 工具創建正式的遷移計劃"
                ])
                
                return output
                
            except Exception as e:
                logger.error(f"Collection analysis failed: {e}")
                return [f"❌ Collection 分析失敗: {str(e)}"]

        async def execute_migration_dry_run(
            ctx: Context,
            source_collection: Annotated[str, Field(description="來源 collection 名稱")],
            target_content_type: Annotated[str, Field(description="目標內容類型: experience, process_workflow, knowledge_base, decision_record, vocabulary")],
            batch_size: Annotated[int, Field(description="批次大小")] = 100,
        ) -> list[str]:
            """
            執行遷移預演（不實際移動資料），檢查遷移可行性。
            """
            await ctx.debug(f"Running migration dry run: {source_collection} -> {target_content_type}")
            
            try:
                # 創建遷移計劃
                analysis = self.migration_tool.analyze_collection_structure(source_collection)
                plan = self.migration_tool.suggest_migration_plan(analysis)
                
                # 更新目標內容類型
                from mcp_server_qdrant.ragbridge.models import ContentType
                try:
                    plan.target_content_type = ContentType(target_content_type)
                except ValueError:
                    return [f"❌ 無效的內容類型: {target_content_type}"]
                
                # 執行 dry run
                result = await self.migration_tool.execute_migration(
                    plan=plan,
                    dry_run=True,
                    batch_size=batch_size
                )
                
                output = [
                    f"🧪 **遷移預演結果**",
                    f"📋 來源: {result.plan.source_collection}",
                    f"🎯 目標: {result.plan.target_content_type.value}",
                    f"⏱️ 執行時間: {result.duration_seconds:.1f} 秒",
                    "",
                    f"📊 **處理統計:**",
                    f"  總記錄數: {result.total_records:,}",
                    f"  成功處理: {result.successful_records:,}",
                    f"  處理失敗: {result.failed_records:,}",
                    f"  成功率: {result.success_rate:.1%}",
                ]
                
                if result.errors:
                    output.extend([
                        "",
                        f"⚠️ **發現的問題 (前10個):**"
                    ])
                    for error in result.errors[:10]:
                        output.append(f"  • {error}")
                    
                    if len(result.errors) > 10:
                        output.append(f"  ... 還有 {len(result.errors) - 10} 個錯誤")
                
                # 生成建議
                report = self.migration_tool.generate_migration_report(result)
                if report['recommendations']:
                    output.extend([
                        "",
                        f"💡 **建議:**"
                    ])
                    for rec in report['recommendations']:
                        output.append(f"  • {rec}")
                
                if result.success_rate >= 0.9:
                    output.extend([
                        "",
                        f"✅ **預演成功！** 可以使用 'execute-migration' 工具執行實際遷移"
                    ])
                else:
                    output.extend([
                        "",
                        f"⚠️ **預演發現問題！** 建議先修正問題再執行實際遷移"
                    ])
                
                return output
                
            except Exception as e:
                logger.error(f"Migration dry run failed: {e}")
                return [f"❌ 遷移預演失敗: {str(e)}"]

        async def execute_data_migration(
            ctx: Context,
            source_collection: Annotated[str, Field(description="來源 collection 名稱")],
            target_content_type: Annotated[str, Field(description="目標內容類型")],
            create_backup: Annotated[bool, Field(description="是否創建備份")] = True,
            batch_size: Annotated[int, Field(description="批次大小")] = 100,
        ) -> list[str]:
            """
            執行實際的資料遷移（高風險操作，需要 super_admin 權限）。
            """
            await ctx.debug(f"Executing data migration: {source_collection} -> {target_content_type}")
            
            try:
                # 創建遷移計劃
                analysis = self.migration_tool.analyze_collection_structure(source_collection)
                plan = self.migration_tool.suggest_migration_plan(analysis)
                
                # 更新目標內容類型
                from mcp_server_qdrant.ragbridge.models import ContentType
                try:
                    plan.target_content_type = ContentType(target_content_type)
                except ValueError:
                    return [f"❌ 無效的內容類型: {target_content_type}"]
                
                # 驗證計劃
                validation_errors = self.migration_tool.validate_migration_plan(plan)
                if validation_errors:
                    return [
                        f"❌ **遷移計劃驗證失敗:**",
                        *[f"  • {error}" for error in validation_errors]
                    ]
                
                # 執行遷移
                result = await self.migration_tool.execute_migration(
                    plan=plan,
                    dry_run=False,
                    batch_size=batch_size
                )
                
                output = [
                    f"🚀 **資料遷移執行結果**",
                    f"📋 來源: {result.plan.source_collection}",
                    f"🎯 目標: {result.plan.target_content_type.value}",
                    f"⏱️ 執行時間: {result.duration_seconds:.1f} 秒",
                    "",
                    f"📊 **遷移統計:**",
                    f"  總記錄數: {result.total_records:,}",
                    f"  成功遷移: {result.successful_records:,}",
                    f"  遷移失敗: {result.failed_records:,}",
                    f"  成功率: {result.success_rate:.1%}",
                ]
                
                if create_backup:
                    output.append(f"💾 備份已創建")
                
                if result.errors:
                    output.extend([
                        "",
                        f"⚠️ **遷移錯誤 (前10個):**"
                    ])
                    for error in result.errors[:10]:
                        output.append(f"  • {error}")
                
                # 生成最終建議
                if result.success_rate >= 0.95:
                    output.extend([
                        "",
                        f"✅ **遷移成功完成！**",
                        f"💡 建議使用 'qdrant-list-collections' 檢查新的 collection",
                        f"⚠️ 如果確認遷移成功，可以考慮移除原始 collection"
                    ])
                elif result.success_rate >= 0.8:
                    output.extend([
                        "",
                        f"⚠️ **遷移部分成功**",
                        f"💡 建議檢查失敗的記錄並考慮重新遷移"
                    ])
                else:
                    output.extend([
                        "",
                        f"❌ **遷移失敗率過高**",
                        f"💡 建議檢查錯誤原因並調整遷移策略"
                    ])
                
                return output
                
            except Exception as e:
                logger.error(f"Data migration failed: {e}")
                return [f"❌ 資料遷移失敗: {str(e)}"]

        # 註冊遷移工具（需要管理員權限）
        if not self.qdrant_settings.read_only:
            self.tool(
                analyze_collection_for_migration,
                name="analyze-collection-for-migration",
                description="分析舊 collection 的結構，為遷移做準備",
            )
            
            self.tool(
                execute_migration_dry_run,
                name="execute-migration-dry-run",
                description="執行遷移預演（不實際移動資料），檢查遷移可行性",
            )
            
            # 實際遷移工具只在非唯讀模式下提供
            self.tool(
                execute_data_migration,
                name="execute-data-migration",
                description="執行實際的資料遷移（高風險操作，需要管理員權限）",
            )

        # 環境變數檢查工具
        async def check_environment_config(ctx: Context) -> list[str]:
            """
            檢查系統的環境變數配置，用於調試和驗證設置。
            """
            import os
            from pathlib import Path
            
            result = ["🔧 **環境變數配置檢查**", ""]
            
            # 檢查 .env 文件路徑
            package_dir = Path(__file__).parent.parent.parent
            env_path = package_dir / ".env"
            result.append(f"📁 **專案根目錄**: {package_dir}")
            result.append(f"📄 **.env 文件路徑**: {env_path}")
            result.append(f"✅ **.env 文件存在**: {'是' if env_path.exists() else '否'}")
            result.append("")
            
            # 列出所有相關的環境變數
            env_vars = {
                "Qdrant 配置": [
                    "QDRANT_URL",
                    "QDRANT_API_KEY", 
                    "QDRANT_LOCAL_PATH",
                    "COLLECTION_NAME",
                    "QDRANT_SEARCH_LIMIT",
                    "QDRANT_READ_ONLY",
                    "QDRANT_ALLOW_ARBITRARY_FILTER"
                ],
                "權限系統": [
                    "QDRANT_ENABLE_PERMISSION_SYSTEM",
                    "QDRANT_DEFAULT_PERMISSION_LEVEL"
                ],
                "Embedding 配置": [
                    "EMBEDDING_PROVIDER",
                    "EMBEDDING_MODEL",
                    "OLLAMA_BASE_URL"
                ],
                "工具配置": [
                    "TOOL_STORE_DESCRIPTION",
                    "TOOL_FIND_DESCRIPTION"
                ]
            }
            
            for category, vars_list in env_vars.items():
                result.append(f"📋 **{category}**:")
                for var in vars_list:
                    value = os.getenv(var)
                    if value is not None:
                        # 對於敏感資訊（如 API KEY）進行遮罩
                        if "API_KEY" in var or "TOKEN" in var:
                            display_value = f"{value[:8]}..." if len(value) > 8 else value
                        else:
                            display_value = value
                        result.append(f"   ✅ {var} = {display_value}")
                    else:
                        result.append(f"   ❌ {var} = (未設置)")
                result.append("")
            
            # 檢查當前設置物件的實際值
            result.append("⚙️ **當前設定物件值**:")
            result.append(f"   📍 Qdrant URL: {self.qdrant_settings.location}")
            result.append(f"   🔑 API Key: {'已設置' if self.qdrant_settings.api_key else '未設置'}")
            result.append(f"   📦 Collection: {self.qdrant_settings.collection_name}")
            result.append(f"   🔍 Search Limit: {self.qdrant_settings.search_limit}")
            result.append(f"   📖 Read Only: {self.qdrant_settings.read_only}")
            result.append(f"   🎯 Allow Arbitrary Filter: {self.qdrant_settings.allow_arbitrary_filter}")
            result.append(f"   🔐 Permission System: {self.qdrant_settings.enable_permission_system}")
            result.append(f"   👤 Default Permission: {self.qdrant_settings.default_permission_level}")
            result.append("")
            result.append(f"   🤖 Embedding Provider: {self.embedding_provider_settings.provider_type}")
            result.append(f"   📝 Embedding Model: {self.embedding_provider_settings.model_name}")
            result.append(f"   🌐 Ollama Base URL: {self.embedding_provider_settings.base_url}")
            
            return result

        self.tool(
            check_environment_config,
            name="qdrant-check-environment",
            description="檢查系統的環境變數配置，用於調試和驗證 .env 文件是否正確載入",
        )

        # Collection 配置管理工具
        async def list_collection_configs(ctx: Context) -> list[str]:
            """
            列出所有 collection 的配置信息
            """
            from mcp_server_qdrant.dynamic_embedding_manager import get_dynamic_embedding_manager
            
            result = ["📋 **Collection 配置列表**", ""]
            
            try:
                manager = get_dynamic_embedding_manager()
                configs = manager.list_collection_configs()
                
                if not configs:
                    result.append("❌ 沒有找到任何 collection 配置")
                    return result
                
                for name, config in configs.items():
                    result.append(f"📁 **{name}**")
                    result.append(f"   🤖 Provider: {config.embedding_provider.value}")
                    result.append(f"   📝 Model: {config.embedding_model}")
                    result.append(f"   🏷️ Vector Name: {config.vector_name}")
                    result.append(f"   📏 Vector Size: {config.vector_size}")
                    if config.ollama_base_url:
                        result.append(f"   🌐 Ollama URL: {config.ollama_base_url}")
                    if config.description:
                        result.append(f"   📄 Description: {config.description}")
                    result.append("")
                
                return result
                
            except Exception as e:
                logger.error(f"Failed to list collection configs: {e}")
                return [f"❌ 獲取配置列表失敗: {str(e)}"]

        async def validate_collection_config(ctx: Context, collection_name: str) -> list[str]:
            """
            驗證指定 collection 的配置和兼容性
            """
            from mcp_server_qdrant.dynamic_embedding_manager import get_dynamic_embedding_manager
            
            result = [f"🔍 **Collection '{collection_name}' 驗證結果**", ""]
            
            try:
                manager = get_dynamic_embedding_manager()
                validation = manager.validate_collection_compatibility(collection_name)
                
                # 基本信息
                result.append("📊 **基本信息**")
                result.append(f"   📁 Collection: {validation['collection_name']}")
                result.append(f"   ⚙️ 配置存在: {'✅' if validation['config_exists'] else '❌'}")
                result.append(f"   🔌 Provider 可用: {'✅' if validation['provider_available'] else '❌'}")
                result.append("")
                
                # 向量信息
                if validation.get('actual_vector_name'):
                    result.append("🎯 **向量信息**")
                    result.append(f"   🏷️ Vector Name: {validation['actual_vector_name']} "
                                f"({'✅' if validation['vector_name_match'] else '❌'})")
                    result.append(f"   📏 Vector Size: {validation['actual_vector_size']} "
                                f"({'✅' if validation['vector_size_match'] else '❌'})")
                    result.append("")
                
                # 警告
                if validation['warnings']:
                    result.append("⚠️ **警告**")
                    for warning in validation['warnings']:
                        result.append(f"   • {warning}")
                    result.append("")
                
                # 錯誤
                if validation['errors']:
                    result.append("❌ **錯誤**")
                    for error in validation['errors']:
                        result.append(f"   • {error}")
                    result.append("")
                
                # 總結
                result.append(f"📋 **總結**: {'✅ 配置有效' if validation['is_valid'] else '❌ 配置有問題'}")
                
                return result
                
            except Exception as e:
                logger.error(f"Failed to validate collection config: {e}")
                return [f"❌ 驗證失敗: {str(e)}"]

        async def get_collection_detailed_info(ctx: Context, collection_name: str) -> list[str]:
            """
            獲取 collection 的詳細信息，包括 Qdrant 狀態和配置信息
            """
            from mcp_server_qdrant.collection_aware_qdrant import CollectionAwareQdrantConnector
            
            result = [f"📊 **Collection '{collection_name}' 詳細信息**", ""]
            
            try:
                # 創建 collection-aware connector
                connector = CollectionAwareQdrantConnector(
                    qdrant_url=self.qdrant_settings.location,
                    qdrant_api_key=self.qdrant_settings.api_key,
                    qdrant_local_path=self.qdrant_settings.local_path,
                )
                
                # 獲取詳細信息
                info = await connector.get_collection_info(collection_name)
                
                if info is None:
                    result.append(f"❌ Collection '{collection_name}' 不存在")
                    return result
                
                # Qdrant 統計
                result.append("📈 **Qdrant 統計**")
                result.append(f"   📄 Documents: {info['points_count']:,}")
                result.append(f"   🔍 Indexed Vectors: {info['indexed_vectors_count']:,}")
                result.append(f"   📊 Status: {info['status']}")
                result.append("")
                
                # 向量配置
                result.append("🎯 **向量配置**")
                for vector_name, vector_config in info['vectors_config'].items():
                    result.append(f"   🏷️ {vector_name}: {vector_config.size}維, {vector_config.distance}")
                result.append("")
                
                # Embedding 配置
                if 'embedding_config' in info:
                    config = info['embedding_config']
                    result.append("🤖 **Embedding 配置**")
                    result.append(f"   🔌 Provider: {config['provider']}")
                    result.append(f"   📝 Model: {config['model']}")
                    result.append(f"   🏷️ Vector Name: {config['vector_name']}")
                    result.append(f"   📏 Vector Size: {config['vector_size']}")
                    result.append("")
                
                return result
                
            except Exception as e:
                logger.error(f"Failed to get collection info: {e}")
                return [f"❌ 獲取信息失敗: {str(e)}"]

        # 註冊 collection 管理工具
        self.tool(
            list_collection_configs,
            name="qdrant-list-collection-configs",
            description="列出所有 collection 的 embedding 配置信息",
        )
        
        self.tool(
            validate_collection_config,
            name="qdrant-validate-collection",
            description="驗證指定 collection 的配置和兼容性",
        )
        
        self.tool(
            get_collection_detailed_info,
            name="qdrant-collection-info",
            description="獲取 collection 的詳細信息，包括 Qdrant 狀態和 embedding 配置",
        )
