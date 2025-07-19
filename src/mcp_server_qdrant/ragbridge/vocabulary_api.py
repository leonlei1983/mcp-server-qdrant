"""
詞彙管理 API 介面

提供詞彙管理的 HTTP API 和 MCP 工具介面，包括：
1. 詞彙查詢和建議
2. 詞彙新增和更新
3. 同義詞管理
4. 詞彙統計和分析
"""
import logging
from typing import Dict, List, Optional

from mcp_server_qdrant.ragbridge.vocabulary import (
    VocabularyTerm, 
    VocabularyDomain, 
    VocabularyStatus,
    vocabulary_manager
)
from mcp_server_qdrant.ragbridge.fragment_manager import fragment_manager

logger = logging.getLogger(__name__)


class VocabularyAPI:
    """詞彙管理 API"""
    
    def __init__(self):
        self.vocab_manager = vocabulary_manager
        self.fragment_manager = fragment_manager
    
    async def search_vocabulary(
        self, 
        query: str = "",
        domain: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 10
    ) -> Dict:
        """搜尋詞彙"""
        try:
            results = []
            
            if query:
                # 精確匹配和建議
                standard_term = self.vocab_manager.get_standard_term(query)
                if standard_term:
                    vocab_term = self.vocab_manager._vocabulary[standard_term]
                    results.append({
                        "term": vocab_term.term,
                        "domain": vocab_term.domain.value,
                        "status": vocab_term.status.value,
                        "synonyms": vocab_term.synonyms,
                        "definition": vocab_term.definition,
                        "usage_count": vocab_term.usage_count,
                        "match_type": "exact"
                    })
                
                # 建議相似詞彙
                suggestions = self.vocab_manager.suggest_terms(query, limit=limit-1)
                for term in suggestions:
                    if term != standard_term:  # 避免重複
                        if term in self.vocab_manager._vocabulary:
                            vocab_term = self.vocab_manager._vocabulary[term]
                            results.append({
                                "term": vocab_term.term,
                                "domain": vocab_term.domain.value,
                                "status": vocab_term.status.value,
                                "synonyms": vocab_term.synonyms,
                                "definition": vocab_term.definition,
                                "usage_count": vocab_term.usage_count,
                                "match_type": "suggestion"
                            })
            else:
                # 瀏覽模式 - 按領域或狀態過濾
                vocab_items = list(self.vocab_manager._vocabulary.values())
                
                if domain:
                    try:
                        domain_enum = VocabularyDomain(domain)
                        vocab_items = [v for v in vocab_items if v.domain == domain_enum]
                    except ValueError:
                        pass
                
                if status:
                    try:
                        status_enum = VocabularyStatus(status)
                        vocab_items = [v for v in vocab_items if v.status == status_enum]
                    except ValueError:
                        pass
                
                # 按使用次數排序
                vocab_items.sort(key=lambda x: x.usage_count, reverse=True)
                
                for vocab_term in vocab_items[:limit]:
                    results.append({
                        "term": vocab_term.term,
                        "domain": vocab_term.domain.value,
                        "status": vocab_term.status.value,
                        "synonyms": vocab_term.synonyms,
                        "definition": vocab_term.definition,
                        "usage_count": vocab_term.usage_count,
                        "match_type": "browse"
                    })
            
            return {
                "query": query,
                "total_results": len(results),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"搜尋詞彙失敗: {e}")
            return {
                "error": str(e),
                "query": query,
                "total_results": 0,
                "results": []
            }
    
    async def propose_vocabulary(
        self,
        term: str,
        domain: str,
        definition: str = "",
        synonyms: List[str] = None
    ) -> Dict:
        """提議新詞彙"""
        try:
            if synonyms is None:
                synonyms = []
            
            # 檢查是否已存在
            existing_term = self.vocab_manager.get_standard_term(term)
            if existing_term:
                return {
                    "success": False,
                    "message": f"詞彙 '{term}' 已存在，標準形式為 '{existing_term}'",
                    "existing_term": existing_term
                }
            
            # 驗證領域
            try:
                domain_enum = VocabularyDomain(domain)
            except ValueError:
                return {
                    "success": False,
                    "message": f"無效的領域: {domain}",
                    "valid_domains": [d.value for d in VocabularyDomain]
                }
            
            # 創建新詞彙項目
            new_term = VocabularyTerm(
                term=term,
                domain=domain_enum,
                status=VocabularyStatus.PROPOSED,
                synonyms=synonyms,
                definition=definition
            )
            
            # 新增到詞彙庫
            success = self.vocab_manager.add_term(new_term)
            
            if success:
                return {
                    "success": True,
                    "message": f"成功提議詞彙 '{new_term.term}'",
                    "term": {
                        "term": new_term.term,
                        "domain": new_term.domain.value,
                        "status": new_term.status.value,
                        "synonyms": new_term.synonyms,
                        "definition": new_term.definition
                    }
                }
            else:
                return {
                    "success": False,
                    "message": "新增詞彙失敗"
                }
                
        except Exception as e:
            logger.error(f"提議詞彙失敗: {e}")
            return {
                "success": False,
                "message": f"提議詞彙失敗: {str(e)}"
            }
    
    async def approve_vocabulary(self, term: str) -> Dict:
        """核准提議的詞彙"""
        try:
            standard_term = self.vocab_manager.get_standard_term(term)
            if not standard_term:
                return {
                    "success": False,
                    "message": f"詞彙 '{term}' 不存在"
                }
            
            vocab_term = self.vocab_manager._vocabulary[standard_term]
            if vocab_term.status != VocabularyStatus.PROPOSED:
                return {
                    "success": False,
                    "message": f"詞彙 '{standard_term}' 狀態為 {vocab_term.status.value}，無法核准"
                }
            
            # 更新狀態
            vocab_term.status = VocabularyStatus.APPROVED
            vocab_term.updated_at = vocab_term.updated_at.__class__.now()
            
            return {
                "success": True,
                "message": f"成功核准詞彙 '{standard_term}'",
                "term": standard_term
            }
            
        except Exception as e:
            logger.error(f"核准詞彙失敗: {e}")
            return {
                "success": False,
                "message": f"核准詞彙失敗: {str(e)}"
            }
    
    async def get_vocabulary_suggestions(self, text: str) -> Dict:
        """為文本內容提供詞彙標準化建議"""
        try:
            import re
            
            # 提取文本中的詞彙
            words = re.findall(r'\b\w+\b', text.lower())
            
            suggestions = []
            standardized_words = []
            
            for word in words:
                if len(word) > 2:  # 忽略太短的詞
                    standard_term = self.vocab_manager.get_standard_term(word)
                    if standard_term and standard_term != word:
                        # 找到標準化建議
                        suggestions.append({
                            "original": word,
                            "suggested": standard_term,
                            "reason": "standardization"
                        })
                        standardized_words.append(standard_term)
                    elif standard_term:
                        standardized_words.append(standard_term)
                    else:
                        # 找相似詞彙
                        similar_terms = self.vocab_manager.suggest_terms(word, limit=2)
                        if similar_terms:
                            suggestions.append({
                                "original": word,
                                "suggested": similar_terms[0],
                                "reason": "similarity",
                                "alternatives": similar_terms[1:] if len(similar_terms) > 1 else []
                            })
            
            # 找出遺漏的重要詞彙
            related_terms = []
            for word in standardized_words:
                related = self.vocab_manager.get_related_terms(word)
                related_terms.extend(related[:2])  # 限制數量
            
            return {
                "original_text": text,
                "suggestions": suggestions,
                "related_terms": list(set(related_terms))[:5],  # 去重並限制數量
                "standardized_count": len(standardized_words)
            }
            
        except Exception as e:
            logger.error(f"獲取詞彙建議失敗: {e}")
            return {
                "error": str(e),
                "original_text": text,
                "suggestions": [],
                "related_terms": []
            }
    
    async def get_vocabulary_statistics(self) -> Dict:
        """獲取詞彙統計資訊"""
        try:
            vocab_data = self.vocab_manager._vocabulary
            
            # 基本統計
            total_terms = len(vocab_data)
            total_synonyms = sum(len(term.synonyms) for term in vocab_data.values())
            
            # 領域分佈
            domain_distribution = {}
            for domain in VocabularyDomain:
                domain_terms = self.vocab_manager.get_terms_by_domain(domain)
                domain_distribution[domain.value] = len(domain_terms)
            
            # 狀態分佈
            status_distribution = {}
            for status in VocabularyStatus:
                count = sum(1 for term in vocab_data.values() if term.status == status)
                status_distribution[status.value] = count
            
            # 使用統計
            usage_counts = [term.usage_count for term in vocab_data.values()]
            total_usage = sum(usage_counts)
            avg_usage = total_usage / total_terms if total_terms > 0 else 0
            
            # 最常用詞彙
            top_terms = sorted(vocab_data.values(), key=lambda x: x.usage_count, reverse=True)[:10]
            most_used = [
                {
                    "term": term.term,
                    "domain": term.domain.value,
                    "usage_count": term.usage_count
                }
                for term in top_terms
            ]
            
            # 分片統計
            fragment_stats = self.fragment_manager.get_fragment_statistics()
            
            return {
                "vocabulary_statistics": {
                    "total_terms": total_terms,
                    "total_synonyms": total_synonyms,
                    "domain_distribution": domain_distribution,
                    "status_distribution": status_distribution,
                    "total_usage": total_usage,
                    "average_usage": avg_usage,
                    "most_used_terms": most_used
                },
                "fragment_statistics": fragment_stats,
                "system_health": {
                    "vocabulary_coverage": min(total_terms / 100, 1.0),  # 假設目標是100個詞彙
                    "usage_activity": min(total_usage / 1000, 1.0),     # 假設目標是1000次使用
                    "domain_diversity": len([d for d, c in domain_distribution.items() if c > 0]) / len(VocabularyDomain)
                }
            }
            
        except Exception as e:
            logger.error(f"獲取詞彙統計失敗: {e}")
            return {
                "error": str(e)
            }
    
    async def standardize_content(self, content: str, tags: List[str] = None) -> Dict:
        """標準化內容和標籤"""
        try:
            if tags is None:
                tags = []
            
            # 標準化標籤
            standardized_tags = self.vocab_manager.validate_and_normalize_tags(tags)
            
            # 為內容提供詞彙建議
            suggestions = await self.get_vocabulary_suggestions(content)
            
            # 建議額外的標籤
            import re
            content_words = re.findall(r'\b\w+\b', content.lower())
            suggested_additional_tags = []
            
            for word in set(content_words):
                if len(word) > 3:  # 忽略太短的詞
                    standard_term = self.vocab_manager.get_standard_term(word)
                    if standard_term and standard_term not in standardized_tags:
                        # 檢查詞彙的使用頻率，只建議常用詞彙
                        if standard_term in self.vocab_manager._vocabulary:
                            vocab_term = self.vocab_manager._vocabulary[standard_term]
                            if vocab_term.usage_count > 5:  # 使用次數閾值
                                suggested_additional_tags.append(standard_term)
            
            return {
                "original_tags": tags,
                "standardized_tags": standardized_tags,
                "suggested_additional_tags": suggested_additional_tags[:5],  # 限制建議數量
                "vocabulary_suggestions": suggestions["suggestions"],
                "related_terms": suggestions["related_terms"]
            }
            
        except Exception as e:
            logger.error(f"標準化內容失敗: {e}")
            return {
                "error": str(e),
                "original_tags": tags,
                "standardized_tags": tags  # 失敗時返回原始標籤
            }


# 全域 API 實例
vocabulary_api = VocabularyAPI()