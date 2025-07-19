"""
RAG 分片策略管理器

此模組負責：
1. 管理不同類型的 RAG 分片
2. 提供分片創建、更新、查詢功能
3. 實現分片間的關聯和依賴管理
4. 優化分片的搜尋和檢索效率
"""
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Union

from mcp_server_qdrant.ragbridge.models import ContentType, RAGEntry, RAGMetadata
from mcp_server_qdrant.ragbridge.vocabulary import (
    FragmentSchema, 
    FragmentType, 
    VocabularyDomain, 
    ExperienceType,
    normalize_fragment_content,
    get_fragment_keywords,
    vocabulary_manager
)

logger = logging.getLogger(__name__)


class FragmentRelation:
    """分片關聯關係"""
    
    def __init__(self, source_id: str, target_id: str, relation_type: str, strength: float = 1.0):
        self.source_id = source_id
        self.target_id = target_id
        self.relation_type = relation_type  # "depends_on", "related_to", "supersedes", "includes"
        self.strength = strength  # 關聯強度 0.0-1.0
        self.created_at = datetime.now()


class FragmentIndex:
    """分片索引系統"""
    
    def __init__(self):
        # 主索引
        self.fragments: Dict[str, FragmentSchema] = {}
        
        # 類型索引
        self.type_index: Dict[FragmentType, Set[str]] = {
            ftype: set() for ftype in FragmentType
        }
        
        # 領域索引
        self.domain_index: Dict[VocabularyDomain, Set[str]] = {
            domain: set() for domain in VocabularyDomain
        }
        
        # 標籤索引
        self.tag_index: Dict[str, Set[str]] = {}
        
        # 關聯索引
        self.relations: Dict[str, List[FragmentRelation]] = {}
        
        # 搜尋關鍵詞索引
        self.keyword_index: Dict[str, Set[str]] = {}
    
    def add_fragment(self, fragment: FragmentSchema, fragment_id: Optional[str] = None) -> str:
        """新增分片"""
        if fragment_id is None:
            fragment_id = str(uuid.uuid4())
        
        # 標準化分片內容
        fragment = normalize_fragment_content(fragment)
        
        # 儲存分片
        self.fragments[fragment_id] = fragment
        
        # 更新索引
        self._update_indexes(fragment_id, fragment)
        
        logger.info(f"新增分片: {fragment_id} - {fragment.title}")
        return fragment_id
    
    def update_fragment(self, fragment_id: str, fragment: FragmentSchema) -> bool:
        """更新分片"""
        if fragment_id not in self.fragments:
            logger.warning(f"分片不存在: {fragment_id}")
            return False
        
        # 清除舊索引
        self._remove_from_indexes(fragment_id)
        
        # 標準化並儲存新內容
        fragment = normalize_fragment_content(fragment)
        self.fragments[fragment_id] = fragment
        
        # 重建索引
        self._update_indexes(fragment_id, fragment)
        
        logger.info(f"更新分片: {fragment_id}")
        return True
    
    def remove_fragment(self, fragment_id: str) -> bool:
        """移除分片"""
        if fragment_id not in self.fragments:
            return False
        
        # 清除索引
        self._remove_from_indexes(fragment_id)
        
        # 移除分片
        del self.fragments[fragment_id]
        
        # 清除關聯
        if fragment_id in self.relations:
            del self.relations[fragment_id]
        
        # 清除其他分片對此分片的關聯
        for relations in self.relations.values():
            self.relations[fragment_id] = [
                rel for rel in relations 
                if rel.target_id != fragment_id
            ]
        
        logger.info(f"移除分片: {fragment_id}")
        return True
    
    def _update_indexes(self, fragment_id: str, fragment: FragmentSchema):
        """更新索引"""
        # 類型索引
        self.type_index[fragment.fragment_type].add(fragment_id)
        
        # 領域索引
        for domain in fragment.domains:
            self.domain_index[domain].add(fragment_id)
        
        # 標籤索引
        for tag in fragment.tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = set()
            self.tag_index[tag].add(fragment_id)
        
        # 關鍵詞索引
        keywords = get_fragment_keywords(fragment)
        for keyword in keywords:
            if keyword not in self.keyword_index:
                self.keyword_index[keyword] = set()
            self.keyword_index[keyword].add(fragment_id)
    
    def _remove_from_indexes(self, fragment_id: str):
        """從索引中移除分片"""
        if fragment_id not in self.fragments:
            return
        
        fragment = self.fragments[fragment_id]
        
        # 類型索引
        self.type_index[fragment.fragment_type].discard(fragment_id)
        
        # 領域索引
        for domain in fragment.domains:
            self.domain_index[domain].discard(fragment_id)
        
        # 標籤索引
        for tag in fragment.tags:
            if tag in self.tag_index:
                self.tag_index[tag].discard(fragment_id)
                if not self.tag_index[tag]:
                    del self.tag_index[tag]
        
        # 關鍵詞索引
        keywords = get_fragment_keywords(fragment)
        for keyword in keywords:
            if keyword in self.keyword_index:
                self.keyword_index[keyword].discard(fragment_id)
                if not self.keyword_index[keyword]:
                    del self.keyword_index[keyword]
    
    def search_fragments(
        self, 
        query: str = "",
        fragment_types: Optional[List[FragmentType]] = None,
        domains: Optional[List[VocabularyDomain]] = None,
        tags: Optional[List[str]] = None,
        min_quality: float = 0.0,
        limit: int = 10
    ) -> List[Tuple[str, FragmentSchema, float]]:
        """搜尋分片"""
        candidate_ids = set(self.fragments.keys())
        
        # 類型過濾
        if fragment_types:
            type_candidates = set()
            for ftype in fragment_types:
                type_candidates.update(self.type_index[ftype])
            candidate_ids = candidate_ids.intersection(type_candidates)
        
        # 領域過濾
        if domains:
            domain_candidates = set()
            for domain in domains:
                domain_candidates.update(self.domain_index[domain])
            candidate_ids = candidate_ids.intersection(domain_candidates)
        
        # 標籤過濾
        if tags:
            tag_candidates = set()
            for tag in tags:
                standard_tag = vocabulary_manager.get_standard_term(tag)
                search_tag = standard_tag if standard_tag else tag
                if search_tag in self.tag_index:
                    tag_candidates.update(self.tag_index[search_tag])
            candidate_ids = candidate_ids.intersection(tag_candidates)
        
        # 品質過濾
        if min_quality > 0:
            candidate_ids = {
                fid for fid in candidate_ids 
                if self.fragments[fid].quality_score >= min_quality
            }
        
        # 關鍵詞搜尋和評分
        results = []
        query_keywords = []
        if query:
            # 標準化查詢詞彙
            import re
            query_words = re.findall(r'\b\w+\b', query.lower())
            for word in query_words:
                standard_term = vocabulary_manager.get_standard_term(word)
                query_keywords.append(standard_term if standard_term else word)
        
        for fragment_id in candidate_ids:
            fragment = self.fragments[fragment_id]
            score = self._calculate_relevance_score(fragment, query_keywords)
            results.append((fragment_id, fragment, score))
        
        # 按分數排序
        results.sort(key=lambda x: x[2], reverse=True)
        
        return results[:limit]
    
    def _calculate_relevance_score(self, fragment: FragmentSchema, query_keywords: List[str]) -> float:
        """計算相關性分數"""
        if not query_keywords:
            return fragment.quality_score
        
        score = 0.0
        
        # 標題匹配
        title_words = set(fragment.title.lower().split())
        title_matches = sum(1 for kw in query_keywords if kw in title_words)
        score += title_matches * 0.3
        
        # 標籤匹配
        tag_matches = sum(1 for kw in query_keywords if kw in fragment.tags)
        score += tag_matches * 0.2
        
        # 內容匹配（簡單的關鍵詞出現次數）
        content_lower = fragment.content.lower()
        content_matches = sum(content_lower.count(kw) for kw in query_keywords)
        score += min(content_matches * 0.1, 0.3)  # 限制內容匹配的權重
        
        # 品質加權
        score *= fragment.quality_score
        
        # 使用統計加權
        usage_boost = min(fragment.usage_count / 10.0, 0.2)
        score += usage_boost
        
        return score
    
    def add_relation(self, relation: FragmentRelation):
        """新增分片關聯"""
        if relation.source_id not in self.relations:
            self.relations[relation.source_id] = []
        
        self.relations[relation.source_id].append(relation)
        logger.info(f"新增關聯: {relation.source_id} -> {relation.target_id} ({relation.relation_type})")
    
    def get_related_fragments(self, fragment_id: str, max_depth: int = 2) -> List[Tuple[str, str, float]]:
        """獲取相關分片"""
        related = []
        visited = set()
        
        def _traverse(current_id: str, depth: int, path_strength: float):
            if depth > max_depth or current_id in visited:
                return
            
            visited.add(current_id)
            
            if current_id in self.relations:
                for relation in self.relations[current_id]:
                    if relation.target_id in self.fragments:
                        strength = path_strength * relation.strength
                        related.append((relation.target_id, relation.relation_type, strength))
                        
                        if depth < max_depth:
                            _traverse(relation.target_id, depth + 1, strength)
        
        _traverse(fragment_id, 0, 1.0)
        
        # 按強度排序並去重
        unique_related = {}
        for target_id, rel_type, strength in related:
            if target_id not in unique_related or unique_related[target_id][1] < strength:
                unique_related[target_id] = (rel_type, strength)
        
        return [(tid, rel_type, strength) for tid, (rel_type, strength) in unique_related.items()]


class FragmentManager:
    """分片管理器"""
    
    def __init__(self):
        self.index = FragmentIndex()
        self._fragment_templates: Dict[FragmentType, Dict] = {}
        self._load_templates()
    
    def _load_templates(self):
        """載入分片模板"""
        self._fragment_templates = {
            FragmentType.PROCESS_WORKFLOW: {
                "required_fields": ["steps", "prerequisites", "expected_outcomes"],
                "default_domains": [VocabularyDomain.DEVELOPMENT],
                "default_experience_types": [ExperienceType.IMPLEMENTATION]
            },
            FragmentType.PROBLEM_SOLUTION: {
                "required_fields": ["content"],
                "default_domains": [VocabularyDomain.TECHNICAL],
                "default_experience_types": [ExperienceType.TROUBLESHOOTING]
            },
            FragmentType.CODE_SNIPPET: {
                "required_fields": ["content"],
                "default_domains": [VocabularyDomain.DEVELOPMENT],
                "default_experience_types": [ExperienceType.IMPLEMENTATION]
            },
            FragmentType.CONFIGURATION: {
                "required_fields": ["content", "prerequisites"],
                "default_domains": [VocabularyDomain.OPERATIONS],
                "default_experience_types": [ExperienceType.CONFIGURATION]
            },
            FragmentType.BEST_PRACTICE: {
                "required_fields": ["content", "expected_outcomes"],
                "default_domains": [VocabularyDomain.GENERAL],
                "default_experience_types": [ExperienceType.BEST_PRACTICE]
            }
        }
    
    def create_fragment_from_rag_entry(self, rag_entry: RAGEntry) -> str:
        """從 RAG Entry 創建分片"""
        # 判斷分片類型
        fragment_type = self._infer_fragment_type(rag_entry)
        
        # 創建分片
        fragment = FragmentSchema(
            fragment_type=fragment_type,
            title=rag_entry.metadata.title,
            content=rag_entry.content,
            tags=rag_entry.metadata.tags,
            quality_score=rag_entry.metadata.quality_score,
            usage_count=rag_entry.metadata.usage_count,
            success_rate=rag_entry.metadata.success_rate,
            version=rag_entry.metadata.version or "1.0"
        )
        
        # 從結構化內容提取欄位
        if rag_entry.structured_content:
            if "steps" in rag_entry.structured_content:
                fragment.steps = rag_entry.structured_content["steps"]
            if "prerequisites" in rag_entry.structured_content:
                fragment.prerequisites = rag_entry.structured_content["prerequisites"]
            if "expected_outcomes" in rag_entry.structured_content:
                fragment.expected_outcomes = rag_entry.structured_content["expected_outcomes"]
        
        # 推斷領域和經驗類型
        fragment.domains = self._infer_domains(rag_entry)
        fragment.experience_types = self._infer_experience_types(rag_entry)
        
        return self.index.add_fragment(fragment, rag_entry.metadata.content_id)
    
    def _infer_fragment_type(self, rag_entry: RAGEntry) -> FragmentType:
        """推斷分片類型"""
        content_type = rag_entry.metadata.content_type
        
        if content_type == ContentType.PROCESS_WORKFLOW:
            return FragmentType.PROCESS_WORKFLOW
        elif content_type == ContentType.EXPERIENCE:
            # 根據內容判斷具體類型
            content_lower = rag_entry.content.lower()
            if "step" in content_lower or "workflow" in content_lower:
                return FragmentType.PROCESS_WORKFLOW
            elif "problem" in content_lower or "solution" in content_lower:
                return FragmentType.PROBLEM_SOLUTION
            elif "best practice" in content_lower:
                return FragmentType.BEST_PRACTICE
            else:
                return FragmentType.LESSON_LEARNED
        elif content_type == ContentType.KNOWLEDGE_BASE:
            return FragmentType.KNOWLEDGE_BASE
        elif content_type == ContentType.DECISION_RECORD:
            return FragmentType.DECISION_RECORD
        else:
            return FragmentType.REFERENCE
    
    def _infer_domains(self, rag_entry: RAGEntry) -> List[VocabularyDomain]:
        """推斷相關領域"""
        domains = []
        content_and_tags = (rag_entry.content + " " + " ".join(rag_entry.metadata.tags)).lower()
        
        # 關鍵詞映射到領域
        domain_keywords = {
            VocabularyDomain.TECHNICAL: ["api", "database", "server", "code", "programming"],
            VocabularyDomain.DEVELOPMENT: ["development", "coding", "testing", "deployment"],
            VocabularyDomain.OPERATIONS: ["deployment", "monitoring", "infrastructure", "devops"],
            VocabularyDomain.SECURITY: ["security", "authentication", "encryption", "vulnerability"],
            VocabularyDomain.DATA: ["data", "analytics", "database", "query", "etl"],
            VocabularyDomain.AI_ML: ["ai", "ml", "machine learning", "model", "training"],
            VocabularyDomain.PROJECT: ["project", "planning", "milestone", "requirement"],
            VocabularyDomain.BUSINESS: ["business", "process", "workflow", "procedure"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in content_and_tags for keyword in keywords):
                domains.append(domain)
        
        # 如果沒有匹配，預設為 GENERAL
        if not domains:
            domains.append(VocabularyDomain.GENERAL)
        
        return domains
    
    def _infer_experience_types(self, rag_entry: RAGEntry) -> List[ExperienceType]:
        """推斷經驗類型"""
        experience_types = []
        content_lower = rag_entry.content.lower()
        
        # 關鍵詞映射到經驗類型
        type_keywords = {
            ExperienceType.TROUBLESHOOTING: ["problem", "issue", "fix", "debug", "error"],
            ExperienceType.IMPLEMENTATION: ["implement", "build", "create", "develop"],
            ExperienceType.CONFIGURATION: ["config", "setup", "configure", "install"],
            ExperienceType.OPTIMIZATION: ["optimize", "improve", "performance", "speed"],
            ExperienceType.TESTING: ["test", "testing", "qa", "verification"],
            ExperienceType.DEPLOYMENT: ["deploy", "deployment", "release", "rollout"],
            ExperienceType.DOCUMENTATION: ["document", "documentation", "guide", "manual"],
            ExperienceType.BEST_PRACTICE: ["best practice", "recommendation", "guideline"]
        }
        
        for exp_type, keywords in type_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                experience_types.append(exp_type)
        
        # 如果沒有匹配，根據內容長度和結構推斷
        if not experience_types:
            if len(rag_entry.content) > 500:
                experience_types.append(ExperienceType.DOCUMENTATION)
            else:
                experience_types.append(ExperienceType.IMPLEMENTATION)
        
        return experience_types
    
    def search_fragments(self, **kwargs) -> List[Dict]:
        """搜尋分片並返回詳細資訊"""
        results = self.index.search_fragments(**kwargs)
        
        formatted_results = []
        for fragment_id, fragment, score in results:
            # 獲取相關分片
            related = self.index.get_related_fragments(fragment_id, max_depth=1)
            
            formatted_results.append({
                "id": fragment_id,
                "fragment": fragment.dict(),
                "relevance_score": score,
                "related_fragments": related[:3]  # 限制相關分片數量
            })
        
        return formatted_results
    
    def get_fragment_statistics(self) -> Dict:
        """獲取分片統計資訊"""
        total_fragments = len(self.index.fragments)
        
        # 類型分佈
        type_distribution = {
            ftype.value: len(fragments) 
            for ftype, fragments in self.index.type_index.items()
        }
        
        # 領域分佈
        domain_distribution = {
            domain.value: len(fragments) 
            for domain, fragments in self.index.domain_index.items()
        }
        
        # 品質分佈
        quality_scores = [f.quality_score for f in self.index.fragments.values()]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # 使用統計
        usage_counts = [f.usage_count for f in self.index.fragments.values()]
        total_usage = sum(usage_counts)
        
        return {
            "total_fragments": total_fragments,
            "type_distribution": type_distribution,
            "domain_distribution": domain_distribution,
            "average_quality": avg_quality,
            "total_usage": total_usage,
            "total_tags": len(self.index.tag_index),
            "total_keywords": len(self.index.keyword_index)
        }


# 全域分片管理器實例
fragment_manager = FragmentManager()