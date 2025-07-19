"""
Enhanced RAG Bridge connector with advanced search capabilities.
"""
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from qdrant_client import models
from qdrant_client.http import exceptions as qdrant_exceptions

from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.qdrant import QdrantConnector
from mcp_server_qdrant.ragbridge.models import (
    ContentType,
    ContentStatus,
    RAGEntry,
    SearchContext,
    SearchResult,
)

logger = logging.getLogger(__name__)


class RAGBridgeConnector(QdrantConnector):
    """Enhanced Qdrant connector specifically for RAG Bridge."""
    
    def __init__(
        self,
        qdrant_url: str | None,
        qdrant_api_key: str | None,
        embedding_provider: EmbeddingProvider,
        qdrant_local_path: str | None = None,
        default_collection_prefix: str = "ragbridge",
    ):
        # Initialize base connector without a default collection
        super().__init__(
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            collection_name=None,  # No default collection
            embedding_provider=embedding_provider,
            qdrant_local_path=qdrant_local_path,
        )
        self.collection_prefix = default_collection_prefix
        
    async def store_rag_entry(self, entry: RAGEntry) -> str:
        """Store a RAG entry in the appropriate collection."""
        collection_name = f"{self.collection_prefix}_{entry.metadata.content_type.value}"
        
        # Ensure collection exists
        await self._ensure_collection_exists(collection_name)
        
        # Prepare the content for embedding
        search_text = entry.get_search_text()
        
        # Generate embeddings
        embeddings = await self._embedding_provider.embed_documents([search_text])
        
        # Prepare the payload
        payload = {
            "content": entry.content,
            "metadata": entry.metadata.model_dump(),
            "structured_content": entry.structured_content,
            "search_keywords": entry.search_keywords,
            "semantic_chunks": entry.semantic_chunks,
            "updated_at": datetime.now().isoformat(),
        }
        
        # Generate unique ID
        point_id = entry.metadata.content_id or str(uuid.uuid4())
        
        # Store in Qdrant
        vector_name = self._embedding_provider.get_vector_name()
        await self._client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector={vector_name: embeddings[0]},
                    payload=payload,
                )
            ],
        )
        
        # Update usage statistics
        await self._update_usage_stats(entry.metadata.content_type, "store")
        
        return point_id
    
    async def search_rag_entries(
        self,
        context: SearchContext,
    ) -> List[SearchResult]:
        """Enhanced search with context awareness and multi-collection support."""
        
        # Determine which collections to search
        collections_to_search = []
        if context.content_types:
            collections_to_search = [
                f"{self.collection_prefix}_{content_type.value}"
                for content_type in context.content_types
            ]
        else:
            # Search all available collections
            all_collections = await self.get_collection_names()
            collections_to_search = [
                col for col in all_collections 
                if col.startswith(self.collection_prefix)
            ]
        
        # Generate query embedding
        query_embeddings = await self._embedding_provider.embed_query(context.query)
        vector_name = self._embedding_provider.get_vector_name()
        
        all_results = []
        
        # Search each collection
        for collection_name in collections_to_search:
            try:
                # Build filters
                query_filter = self._build_search_filter(context)
                
                # Perform search
                search_results = await self._client.query_points(
                    collection_name=collection_name,
                    query=query_embeddings,
                    using=vector_name,
                    limit=context.max_results,
                    query_filter=query_filter,
                    score_threshold=context.min_similarity,
                )
                
                # Process results
                for idx, result in enumerate(search_results.points):
                    search_result = self._process_search_result(
                        result, idx, collection_name, context
                    )
                    all_results.append(search_result)
                    
            except qdrant_exceptions.UnexpectedResponse as e:
                logger.warning(f"Collection {collection_name} not found or empty: {e}")
                continue
        
        # Sort by relevance and apply final filtering
        sorted_results = self._rank_and_filter_results(all_results, context)
        
        # Update search statistics
        await self._update_search_stats(context.query, len(sorted_results))
        
        return sorted_results[:context.max_results]
    
    def _build_search_filter(self, context: SearchContext) -> Optional[models.Filter]:
        """Build Qdrant filter from search context."""
        conditions = []
        
        # Status filter
        if context.status_filter:
            status_values = [status.value for status in context.status_filter]
            conditions.append(
                models.FieldCondition(
                    key="metadata.status",
                    match=models.MatchAny(any=status_values)
                )
            )
        
        # Date range filter
        if context.date_range:
            if "start" in context.date_range:
                conditions.append(
                    models.FieldCondition(
                        key="metadata.created_at",
                        range=models.DatetimeRange(
                            gte=context.date_range["start"]
                        )
                    )
                )
            if "end" in context.date_range:
                conditions.append(
                    models.FieldCondition(
                        key="metadata.created_at",
                        range=models.DatetimeRange(
                            lte=context.date_range["end"]
                        )
                    )
                )
        
        # Quality filter
        if context.min_quality_score > 0:
            conditions.append(
                models.FieldCondition(
                    key="metadata.quality_score",
                    range=models.Range(gte=context.min_quality_score)
                )
            )
        
        # Experimental content filter
        if not context.include_experimental:
            conditions.append(
                models.FieldCondition(
                    key="metadata.status",
                    match=models.MatchExcept(**{"except": [ContentStatus.EXPERIMENTAL.value]})
                )
            )
        
        # Project context filter
        if context.current_project:
            conditions.append(
                models.FieldCondition(
                    key="metadata.tags",
                    match=models.MatchAny(any=[context.current_project])
                )
            )
        
        if conditions:
            return models.Filter(must=conditions)
        
        return None
    
    def _process_search_result(
        self, 
        result: Any, 
        rank: int, 
        collection_name: str, 
        context: SearchContext
    ) -> SearchResult:
        """Process a single search result."""
        
        # Extract data from result
        payload = result.payload
        similarity_score = result.score
        
        # Reconstruct RAG entry
        rag_entry = RAGEntry(
            content=payload["content"],
            metadata=payload["metadata"],
            structured_content=payload.get("structured_content", {}),
            search_keywords=payload.get("search_keywords", []),
            semantic_chunks=payload.get("semantic_chunks", []),
        )
        
        # Calculate relevance score
        relevance_score = self._calculate_relevance_score(
            rag_entry, similarity_score, context
        )
        
        # Generate match reasons
        match_reasons = self._generate_match_reasons(rag_entry, context)
        
        # Generate usage recommendation
        usage_recommendation = self._generate_usage_recommendation(rag_entry, context)
        
        return SearchResult(
            entry=rag_entry,
            similarity_score=similarity_score,
            relevance_score=relevance_score,
            rank=rank,
            match_reasons=match_reasons,
            usage_recommendation=usage_recommendation,
            confidence_level=min(similarity_score * relevance_score, 1.0),
        )
    
    def _calculate_relevance_score(
        self, 
        entry: RAGEntry, 
        similarity_score: float, 
        context: SearchContext
    ) -> float:
        """Calculate a relevance score based on multiple factors."""
        
        base_score = similarity_score
        
        # Quality boost
        quality_boost = entry.metadata.quality_score * 0.2
        
        # Recency boost (newer content gets slight boost)
        days_old = (datetime.now() - entry.metadata.updated_at).days
        recency_boost = max(0, (30 - days_old) / 30) * 0.1
        
        # Usage popularity boost
        usage_boost = min(entry.metadata.usage_count / 100, 0.1)
        
        # Success rate boost
        success_boost = entry.metadata.success_rate * 0.1
        
        # Status penalty for deprecated content
        status_penalty = 0.3 if entry.metadata.status == ContentStatus.DEPRECATED else 0
        
        total_score = base_score + quality_boost + recency_boost + usage_boost + success_boost - status_penalty
        
        return max(0, min(1.0, total_score))
    
    def _generate_match_reasons(self, entry: RAGEntry, context: SearchContext) -> List[str]:
        """Generate reasons why this entry matched the search."""
        reasons = []
        
        # Check for keyword matches
        query_words = set(context.query.lower().split())
        title_words = set(entry.metadata.title.lower().split())
        tag_words = set(tag.lower() for tag in entry.metadata.tags)
        
        if query_words & title_words:
            reasons.append("Title contains query keywords")
        
        if query_words & tag_words:
            reasons.append("Tags match query")
        
        # Check for high quality
        if entry.metadata.quality_score > 0.8:
            reasons.append("High quality content")
        
        # Check for recent updates
        if (datetime.now() - entry.metadata.updated_at).days < 30:
            reasons.append("Recently updated")
        
        # Check for popularity
        if entry.metadata.usage_count > 10:
            reasons.append("Popular content")
        
        return reasons
    
    def _generate_usage_recommendation(self, entry: RAGEntry, context: SearchContext) -> str:
        """Generate a recommendation on how to use this result."""
        
        content_type = entry.metadata.content_type
        
        recommendations = {
            ContentType.EXPERIENCE: "Review the solution approach and adapt to your specific context",
            ContentType.PROCESS_WORKFLOW: "Follow the process steps and validate each checkpoint",
            ContentType.KNOWLEDGE_BASE: "Use this as reference material for your current task",
            ContentType.VOCABULARY: "Ensure consistent terminology usage in your work",
            ContentType.DECISION_RECORD: "Consider the decision context and rationale",
        }
        
        base_recommendation = recommendations.get(
            content_type, 
            "Review the content and apply relevant parts to your situation"
        )
        
        # Add quality-based advice
        if entry.metadata.quality_score < 0.5:
            base_recommendation += " (Note: Verify accuracy as this content has lower quality rating)"
        
        return base_recommendation
    
    def _rank_and_filter_results(
        self, 
        results: List[SearchResult], 
        context: SearchContext
    ) -> List[SearchResult]:
        """Rank and filter results based on relevance."""
        
        # Sort by relevance score
        sorted_results = sorted(
            results, 
            key=lambda x: x.relevance_score, 
            reverse=True
        )
        
        # Update ranks
        for idx, result in enumerate(sorted_results):
            result.rank = idx + 1
        
        return sorted_results
    
    async def _update_usage_stats(self, content_type: ContentType, operation: str):
        """Update usage statistics."""
        # This would typically update a statistics collection
        # For now, we'll just log
        logger.info(f"Usage stats: {operation} operation on {content_type.value}")
    
    async def _update_search_stats(self, query: str, result_count: int):
        """Update search statistics."""
        # This would typically update search analytics
        # For now, we'll just log
        logger.info(f"Search stats: Query '{query}' returned {result_count} results")
    
    async def get_content_by_id(self, content_id: str, content_type: ContentType) -> Optional[RAGEntry]:
        """Get specific content by ID."""
        collection_name = f"{self.collection_prefix}_{content_type.value}"
        
        try:
            result = await self._client.retrieve(
                collection_name=collection_name,
                ids=[content_id],
                with_payload=True,
            )
            
            if result:
                payload = result[0].payload
                return RAGEntry(
                    content=payload["content"],
                    metadata=payload["metadata"],
                    structured_content=payload.get("structured_content", {}),
                    search_keywords=payload.get("search_keywords", []),
                    semantic_chunks=payload.get("semantic_chunks", []),
                )
            
        except Exception as e:
            logger.error(f"Error retrieving content {content_id}: {e}")
        
        return None
    
    async def update_content_metadata(self, content_id: str, content_type: ContentType, metadata_updates: Dict[str, Any]):
        """Update metadata for existing content."""
        collection_name = f"{self.collection_prefix}_{content_type.value}"
        
        try:
            # Get current content
            current_content = await self.get_content_by_id(content_id, content_type)
            if not current_content:
                raise ValueError(f"Content {content_id} not found")
            
            # Update metadata
            updated_metadata = current_content.metadata.model_dump()
            updated_metadata.update(metadata_updates)
            updated_metadata["updated_at"] = datetime.now().isoformat()
            
            # Update in Qdrant
            await self._client.set_payload(
                collection_name=collection_name,
                payload={"metadata": updated_metadata},
                points=[content_id],
            )
            
            logger.info(f"Updated metadata for content {content_id}")
            
        except Exception as e:
            logger.error(f"Error updating metadata for {content_id}: {e}")
            raise