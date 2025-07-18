"""
Advanced search engine for RAG Bridge with intelligent ranking and context awareness.
"""
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from mcp_server_qdrant.ragbridge.models import (
    ContentType,
    RAGEntry,
    SearchContext,
    SearchResult,
)

logger = logging.getLogger(__name__)


class IntelligentSearchEngine:
    """Advanced search engine with AI-powered ranking and context awareness."""
    
    def __init__(self):
        self.query_cache: Dict[str, List[SearchResult]] = {}
        self.user_preferences: Dict[str, Any] = {}
        self.search_history: List[Dict[str, Any]] = []
        
    def enhance_search_results(
        self, 
        results: List[SearchResult], 
        context: SearchContext
    ) -> List[SearchResult]:
        """Enhance search results with advanced ranking and filtering."""
        
        # Apply query expansion
        expanded_context = self._expand_query_context(context)
        
        # Re-rank results based on multiple factors
        enhanced_results = []
        for result in results:
            enhanced_result = self._enhance_single_result(result, expanded_context)
            enhanced_results.append(enhanced_result)
        
        # Apply intelligent filtering
        filtered_results = self._intelligent_filter(enhanced_results, expanded_context)
        
        # Final ranking
        final_results = self._apply_final_ranking(filtered_results, expanded_context)
        
        # Update search history
        self._update_search_history(context, final_results)
        
        return final_results
    
    def _expand_query_context(self, context: SearchContext) -> SearchContext:
        """Expand query context with synonyms and related terms."""
        
        expanded_query = context.query
        
        # Add synonyms and related terms
        synonyms = self._get_synonyms(context.query)
        if synonyms:
            expanded_query += " " + " ".join(synonyms)
        
        # Add contextual keywords based on project
        if context.current_project:
            project_keywords = self._get_project_keywords(context.current_project)
            expanded_query += " " + " ".join(project_keywords)
        
        # Create expanded context
        expanded_context = context.model_copy()
        expanded_context.query = expanded_query
        
        return expanded_context
    
    def _get_synonyms(self, query: str) -> List[str]:
        """Get synonyms for query terms."""
        # This would ideally use a proper thesaurus or word embeddings
        # For now, we'll use a simple mapping
        
        synonym_map = {
            "bug": ["issue", "problem", "error", "defect"],
            "fix": ["solve", "repair", "resolve", "correct"],
            "deploy": ["release", "publish", "deliver"],
            "test": ["verify", "validate", "check"],
            "optimize": ["improve", "enhance", "speed up"],
            "security": ["safety", "protection", "auth"],
            "performance": ["speed", "efficiency", "optimization"],
            "database": ["db", "storage", "data"],
            "api": ["interface", "service", "endpoint"],
            "frontend": ["ui", "interface", "client"],
            "backend": ["server", "service", "api"],
        }
        
        query_words = query.lower().split()
        synonyms = []
        
        for word in query_words:
            if word in synonym_map:
                synonyms.extend(synonym_map[word])
        
        return list(set(synonyms))  # Remove duplicates
    
    def _get_project_keywords(self, project: str) -> List[str]:
        """Get relevant keywords for a project context."""
        
        # This would ideally come from project metadata
        # For now, we'll use pattern matching
        
        project_keywords = {
            "web": ["html", "css", "javascript", "react", "vue", "angular"],
            "mobile": ["ios", "android", "flutter", "react native"],
            "ai": ["machine learning", "neural network", "tensorflow", "pytorch"],
            "data": ["sql", "database", "analytics", "etl", "pipeline"],
            "devops": ["docker", "kubernetes", "ci/cd", "deployment"],
            "security": ["authentication", "authorization", "encryption"],
        }
        
        project_lower = project.lower()
        keywords = []
        
        for category, terms in project_keywords.items():
            if category in project_lower:
                keywords.extend(terms)
        
        return keywords
    
    def _enhance_single_result(
        self, 
        result: SearchResult, 
        context: SearchContext
    ) -> SearchResult:
        """Enhance a single search result with additional scoring."""
        
        enhanced_result = result.model_copy()
        
        # Context relevance scoring
        context_score = self._calculate_context_relevance(result.entry, context)
        
        # Temporal relevance scoring
        temporal_score = self._calculate_temporal_relevance(result.entry, context)
        
        # Usage pattern scoring
        usage_score = self._calculate_usage_pattern_score(result.entry, context)
        
        # Combine scores
        enhanced_relevance = (
            result.relevance_score * 0.4 +
            context_score * 0.3 +
            temporal_score * 0.2 +
            usage_score * 0.1
        )
        
        enhanced_result.relevance_score = min(1.0, enhanced_relevance)
        
        # Update confidence based on multiple factors
        enhanced_result.confidence_level = self._calculate_confidence(
            result, context_score, temporal_score, usage_score
        )
        
        # Add enhanced match reasons
        enhanced_result.match_reasons.extend(
            self._generate_enhanced_match_reasons(result.entry, context)
        )
        
        return enhanced_result
    
    def _calculate_context_relevance(self, entry: RAGEntry, context: SearchContext) -> float:
        """Calculate how relevant the entry is to the search context."""
        
        score = 0.0
        
        # Project context matching
        if context.current_project:
            if context.current_project.lower() in entry.metadata.title.lower():
                score += 0.3
            if context.current_project.lower() in entry.metadata.tags:
                score += 0.2
            if context.current_project.lower() in " ".join(entry.metadata.categories).lower():
                score += 0.1
        
        # Content type preference
        if context.content_types:
            if entry.metadata.content_type in context.content_types:
                score += 0.2
        
        # Language matching
        if hasattr(context, 'preferred_language'):
            if entry.metadata.language == context.preferred_language:
                score += 0.1
        
        # Tag relevance
        query_words = set(context.query.lower().split())
        entry_tags = set(tag.lower() for tag in entry.metadata.tags)
        tag_overlap = len(query_words & entry_tags)
        if tag_overlap > 0:
            score += min(0.2, tag_overlap * 0.1)
        
        return min(1.0, score)
    
    def _calculate_temporal_relevance(self, entry: RAGEntry, context: SearchContext) -> float:
        """Calculate temporal relevance based on recency and update frequency."""
        
        now = datetime.now()
        
        # Recency scoring
        days_since_update = (now - entry.metadata.updated_at).days
        recency_score = max(0, (365 - days_since_update) / 365)  # Decay over a year
        
        # Update frequency scoring (more frequently updated content is more relevant)
        days_since_creation = (now - entry.metadata.created_at).days
        if days_since_creation > 0:
            update_frequency = (entry.metadata.updated_at - entry.metadata.created_at).days / days_since_creation
            frequency_score = min(1.0, update_frequency)
        else:
            frequency_score = 0.5  # New content
        
        # Time-based filtering from context
        temporal_bonus = 0.0
        if context.date_range:
            if "start" in context.date_range:
                if entry.metadata.created_at >= context.date_range["start"]:
                    temporal_bonus += 0.2
            if "end" in context.date_range:
                if entry.metadata.created_at <= context.date_range["end"]:
                    temporal_bonus += 0.2
        
        return min(1.0, recency_score * 0.6 + frequency_score * 0.4 + temporal_bonus)
    
    def _calculate_usage_pattern_score(self, entry: RAGEntry, context: SearchContext) -> float:
        """Calculate score based on usage patterns and success rates."""
        
        score = 0.0
        
        # Usage frequency scoring
        if entry.metadata.usage_count > 0:
            usage_score = min(1.0, entry.metadata.usage_count / 50)  # Normalize to 50 uses
            score += usage_score * 0.4
        
        # Success rate scoring
        if entry.metadata.success_rate > 0:
            score += entry.metadata.success_rate * 0.4
        
        # Quality score
        if entry.metadata.quality_score > 0:
            score += entry.metadata.quality_score * 0.2
        
        return min(1.0, score)
    
    def _calculate_confidence(
        self, 
        result: SearchResult, 
        context_score: float, 
        temporal_score: float, 
        usage_score: float
    ) -> float:
        """Calculate overall confidence in the result."""
        
        # Base confidence from similarity
        base_confidence = result.similarity_score
        
        # Boost from various factors
        context_boost = context_score * 0.2
        temporal_boost = temporal_score * 0.1
        usage_boost = usage_score * 0.1
        
        # Penalty for low-quality matches
        quality_penalty = 0.0
        if result.entry.metadata.quality_score < 0.3:
            quality_penalty = 0.2
        
        total_confidence = base_confidence + context_boost + temporal_boost + usage_boost - quality_penalty
        
        return max(0.0, min(1.0, total_confidence))
    
    def _generate_enhanced_match_reasons(self, entry: RAGEntry, context: SearchContext) -> List[str]:
        """Generate additional match reasons based on enhanced analysis."""
        
        reasons = []
        
        # Semantic similarity reasons
        if self._has_semantic_similarity(entry.content, context.query):
            reasons.append("Strong semantic similarity with query")
        
        # Context-specific reasons
        if context.current_project:
            if context.current_project.lower() in entry.metadata.title.lower():
                reasons.append(f"Related to current project: {context.current_project}")
        
        # Expertise level matching
        if hasattr(context, 'user_expertise_level'):
            content_complexity = self._assess_content_complexity(entry)
            if content_complexity == context.user_expertise_level:
                reasons.append("Matches your expertise level")
        
        # Trending content
        if self._is_trending_content(entry):
            reasons.append("Trending content in your domain")
        
        return reasons
    
    def _has_semantic_similarity(self, content: str, query: str) -> bool:
        """Check if content has semantic similarity with query."""
        
        # Simple semantic similarity check
        # In a real implementation, this would use more sophisticated NLP
        
        content_words = set(content.lower().split())
        query_words = set(query.lower().split())
        
        # Check for word overlap
        overlap = len(content_words & query_words)
        
        # Check for similar word patterns
        similar_patterns = 0
        for query_word in query_words:
            for content_word in content_words:
                if self._words_similar(query_word, content_word):
                    similar_patterns += 1
                    break
        
        return overlap >= 2 or similar_patterns >= 3
    
    def _words_similar(self, word1: str, word2: str) -> bool:
        """Check if two words are similar."""
        
        # Simple similarity check
        if len(word1) < 3 or len(word2) < 3:
            return False
        
        # Check for common prefix/suffix
        if word1[:3] == word2[:3] or word1[-3:] == word2[-3:]:
            return True
        
        # Check for character overlap
        overlap = len(set(word1) & set(word2))
        min_length = min(len(word1), len(word2))
        
        return overlap / min_length > 0.6
    
    def _assess_content_complexity(self, entry: RAGEntry) -> str:
        """Assess the complexity level of content."""
        
        # Simple complexity assessment
        content = entry.content
        
        # Technical terms indicator
        technical_terms = [
            "algorithm", "architecture", "implementation", "optimization",
            "configuration", "deployment", "integration", "framework"
        ]
        
        tech_count = sum(1 for term in technical_terms if term in content.lower())
        
        # Length indicator
        word_count = len(content.split())
        
        # Determine complexity
        if tech_count >= 5 or word_count > 500:
            return "advanced"
        elif tech_count >= 2 or word_count > 200:
            return "intermediate"
        else:
            return "beginner"
    
    def _is_trending_content(self, entry: RAGEntry) -> bool:
        """Check if content is trending based on recent usage."""
        
        # Simple trending check
        recent_usage = entry.metadata.usage_count
        recent_updates = (datetime.now() - entry.metadata.updated_at).days < 30
        
        return recent_usage > 10 and recent_updates
    
    def _intelligent_filter(
        self, 
        results: List[SearchResult], 
        context: SearchContext
    ) -> List[SearchResult]:
        """Apply intelligent filtering to remove less relevant results."""
        
        if not results:
            return results
        
        # Remove duplicates based on content similarity
        filtered_results = self._remove_duplicate_results(results)
        
        # Apply quality threshold
        quality_threshold = max(0.3, context.min_quality_score)
        filtered_results = [
            result for result in filtered_results
            if result.entry.metadata.quality_score >= quality_threshold
        ]
        
        # Apply confidence threshold
        confidence_threshold = 0.5
        filtered_results = [
            result for result in filtered_results
            if result.confidence_level >= confidence_threshold
        ]
        
        return filtered_results
    
    def _remove_duplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate or very similar results."""
        
        unique_results = []
        seen_titles = set()
        
        for result in results:
            title = result.entry.metadata.title.lower()
            
            # Check for exact title match
            if title in seen_titles:
                continue
            
            # Check for similar titles
            is_similar = False
            for seen_title in seen_titles:
                if self._titles_similar(title, seen_title):
                    is_similar = True
                    break
            
            if not is_similar:
                unique_results.append(result)
                seen_titles.add(title)
        
        return unique_results
    
    def _titles_similar(self, title1: str, title2: str) -> bool:
        """Check if two titles are similar."""
        
        words1 = set(title1.split())
        words2 = set(title2.split())
        
        if len(words1) == 0 or len(words2) == 0:
            return False
        
        overlap = len(words1 & words2)
        min_length = min(len(words1), len(words2))
        
        return overlap / min_length > 0.7
    
    def _apply_final_ranking(
        self, 
        results: List[SearchResult], 
        context: SearchContext
    ) -> List[SearchResult]:
        """Apply final ranking algorithm."""
        
        # Sort by relevance score
        sorted_results = sorted(
            results, 
            key=lambda x: (x.relevance_score, x.confidence_level), 
            reverse=True
        )
        
        # Update ranks
        for idx, result in enumerate(sorted_results):
            result.rank = idx + 1
        
        # Apply diversity promotion (ensure variety in content types)
        diverse_results = self._promote_diversity(sorted_results, context)
        
        return diverse_results
    
    def _promote_diversity(
        self, 
        results: List[SearchResult], 
        context: SearchContext
    ) -> List[SearchResult]:
        """Promote diversity in result types."""
        
        if len(results) <= 3:
            return results
        
        # Track content types in top results
        content_types_seen = set()
        diverse_results = []
        remaining_results = []
        
        for result in results:
            content_type = result.entry.metadata.content_type
            
            if len(diverse_results) < 3 or content_type not in content_types_seen:
                diverse_results.append(result)
                content_types_seen.add(content_type)
            else:
                remaining_results.append(result)
        
        # Add remaining results
        diverse_results.extend(remaining_results)
        
        # Update ranks
        for idx, result in enumerate(diverse_results):
            result.rank = idx + 1
        
        return diverse_results
    
    def _update_search_history(self, context: SearchContext, results: List[SearchResult]):
        """Update search history for learning purposes."""
        
        history_entry = {
            "timestamp": datetime.now(),
            "query": context.query,
            "content_types": [ct.value for ct in context.content_types],
            "result_count": len(results),
            "top_result_types": [r.entry.metadata.content_type.value for r in results[:3]],
        }
        
        self.search_history.append(history_entry)
        
        # Keep only recent history
        if len(self.search_history) > 1000:
            self.search_history = self.search_history[-1000:]
    
    def get_search_suggestions(self, partial_query: str) -> List[str]:
        """Get search suggestions based on history and content."""
        
        suggestions = []
        
        # Add suggestions from search history
        for entry in self.search_history[-50:]:  # Recent searches
            if partial_query.lower() in entry["query"].lower():
                suggestions.append(entry["query"])
        
        # Add common search patterns
        common_patterns = [
            f"how to {partial_query}",
            f"{partial_query} best practices",
            f"{partial_query} troubleshooting",
            f"{partial_query} examples",
            f"{partial_query} workflow",
        ]
        
        suggestions.extend(common_patterns)
        
        # Remove duplicates and return top suggestions
        unique_suggestions = list(set(suggestions))
        return unique_suggestions[:10]