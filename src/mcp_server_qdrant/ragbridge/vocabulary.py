"""
RAG Bridge 標準化詞彙管理系統

此模組負責：
1. 定義標準化詞彙類別和結構
2. 提供詞彙驗證和標準化功能
3. 管理同義詞映射和詞彙清理
4. 確保 embedding 搜尋的一致性
"""
import logging
import re
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class VocabularyDomain(str, Enum):
    """詞彙領域分類"""
    TECHNICAL = "technical"          # 技術相關
    BUSINESS = "business"            # 商業流程
    PROJECT = "project"              # 專案管理
    DEVELOPMENT = "development"      # 開發流程
    OPERATIONS = "operations"        # 營運維護
    SECURITY = "security"            # 資訊安全
    DATA = "data"                    # 資料處理
    AI_ML = "ai_ml"                  # AI/ML 相關
    GENERAL = "general"              # 一般通用


class VocabularyStatus(str, Enum):
    """詞彙狀態"""
    ACTIVE = "active"                # 活躍使用
    DEPRECATED = "deprecated"        # 已棄用
    PROPOSED = "proposed"            # 提議中
    APPROVED = "approved"            # 已核准
    REJECTED = "rejected"            # 已拒絕


class ExperienceType(str, Enum):
    """經驗類型標準化"""
    TROUBLESHOOTING = "troubleshooting"      # 問題排除
    IMPLEMENTATION = "implementation"        # 實作經驗
    CONFIGURATION = "configuration"         # 配置設定
    OPTIMIZATION = "optimization"           # 效能優化
    INTEGRATION = "integration"             # 系統整合
    TESTING = "testing"                     # 測試驗證
    DEPLOYMENT = "deployment"               # 部署發布
    MONITORING = "monitoring"               # 監控維護
    DOCUMENTATION = "documentation"         # 文件記錄
    BEST_PRACTICE = "best_practice"         # 最佳實踐


class VocabularyTerm(BaseModel):
    """標準化詞彙項目"""
    term: str = Field(..., description="詞彙項目")
    domain: VocabularyDomain = Field(..., description="所屬領域")
    status: VocabularyStatus = Field(default=VocabularyStatus.ACTIVE, description="詞彙狀態")
    synonyms: List[str] = Field(default_factory=list, description="同義詞列表")
    related_terms: List[str] = Field(default_factory=list, description="相關詞彙")
    definition: str = Field(default="", description="詞彙定義")
    usage_count: int = Field(default=0, description="使用次數")
    created_at: datetime = Field(default_factory=datetime.now, description="建立時間")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新時間")
    metadata: Dict[str, Union[str, int, float, bool]] = Field(default_factory=dict, description="額外元數據")

    @field_validator('term')
    @classmethod
    def normalize_term(cls, v):
        """標準化詞彙格式"""
        if not v:
            raise ValueError("詞彙不能為空")
        # 轉換為小寫並移除多餘空白
        normalized = re.sub(r'\s+', ' ', v.lower().strip())
        # 移除特殊字符（保留字母、數字、空格、連字符、底線）
        normalized = re.sub(r'[^\w\s\-_]', '', normalized)
        return normalized

    @field_validator('synonyms', mode='before')
    @classmethod
    def normalize_synonyms(cls, v):
        """標準化同義詞列表"""
        if not v:
            return []
        normalized = []
        for synonym in v:
            if isinstance(synonym, str):
                # 應用相同的標準化規則
                norm_synonym = re.sub(r'\s+', ' ', synonym.lower().strip())
                norm_synonym = re.sub(r'[^\w\s\-_]', '', norm_synonym)
                if norm_synonym:
                    normalized.append(norm_synonym)
        return list(set(normalized))  # 去重


class FragmentType(str, Enum):
    """RAG 分片類型"""
    PROCESS_WORKFLOW = "process_workflow"    # 流程工作流
    PROBLEM_SOLUTION = "problem_solution"    # 問題解決方案
    KNOWLEDGE_BASE = "knowledge_base"        # 知識庫文章
    CODE_SNIPPET = "code_snippet"           # 程式碼片段
    CONFIGURATION = "configuration"         # 配置模板
    CHECKLIST = "checklist"                 # 檢查清單
    DECISION_RECORD = "decision_record"      # 決策記錄
    LESSON_LEARNED = "lesson_learned"       # 經驗教訓
    BEST_PRACTICE = "best_practice"         # 最佳實踐
    REFERENCE = "reference"                 # 參考資料


class FragmentSchema(BaseModel):
    """RAG 分片標準化格式"""
    fragment_type: FragmentType = Field(..., description="分片類型")
    title: str = Field(..., description="分片標題")
    content: str = Field(..., description="主要內容")
    
    # 標準化詞彙欄位
    domains: List[VocabularyDomain] = Field(default_factory=list, description="相關領域")
    tags: List[str] = Field(default_factory=list, description="標籤（標準化詞彙）")
    experience_types: List[ExperienceType] = Field(default_factory=list, description="經驗類型")
    
    # 結構化欄位
    prerequisites: List[str] = Field(default_factory=list, description="前置條件")
    steps: List[str] = Field(default_factory=list, description="執行步驟")
    expected_outcomes: List[str] = Field(default_factory=list, description="預期結果")
    related_fragments: List[str] = Field(default_factory=list, description="相關分片ID")
    
    # 品質和使用統計
    quality_score: float = Field(default=0.5, ge=0.0, le=1.0, description="品質評分")
    difficulty_level: int = Field(default=1, ge=1, le=5, description="難度等級")
    estimated_time_minutes: int = Field(default=0, ge=0, description="預估執行時間（分鐘）")
    usage_count: int = Field(default=0, description="使用次數")
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="成功率")
    
    # 版本和時間
    version: str = Field(default="1.0", description="版本號")
    created_at: datetime = Field(default_factory=datetime.now, description="建立時間")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新時間")
    
    @field_validator('tags', mode='before')
    @classmethod
    def normalize_tags(cls, v):
        """標準化標籤"""
        if not v:
            return []
        normalized = []
        for tag in v:
            if isinstance(tag, str):
                # 標準化標籤格式
                norm_tag = re.sub(r'\s+', ' ', tag.lower().strip())
                norm_tag = re.sub(r'[^\w\s\-_]', '', norm_tag)
                if norm_tag:
                    normalized.append(norm_tag)
        return list(set(normalized))


class VocabularyManager:
    """詞彙管理器"""
    
    def __init__(self):
        self._vocabulary: Dict[str, VocabularyTerm] = {}
        self._synonym_mapping: Dict[str, str] = {}  # 同義詞 -> 標準詞彙映射
        self._domain_terms: Dict[VocabularyDomain, Set[str]] = {
            domain: set() for domain in VocabularyDomain
        }
        self._load_default_vocabulary()
    
    def _load_default_vocabulary(self):
        """載入預設詞彙庫"""
        default_terms = [
            # 技術詞彙
            VocabularyTerm(
                term="api", 
                domain=VocabularyDomain.TECHNICAL,
                synonyms=["interface", "endpoint", "service"],
                definition="應用程式介面，用於不同系統間的數據交換"
            ),
            VocabularyTerm(
                term="database", 
                domain=VocabularyDomain.TECHNICAL,
                synonyms=["db", "data store", "repository"],
                definition="結構化資料儲存系統"
            ),
            VocabularyTerm(
                term="docker", 
                domain=VocabularyDomain.TECHNICAL,
                synonyms=["container", "containerization"],
                definition="容器化平台技術"
            ),
            
            # 開發流程詞彙
            VocabularyTerm(
                term="deployment", 
                domain=VocabularyDomain.DEVELOPMENT,
                synonyms=["deploy", "release", "rollout"],
                definition="軟體部署發布流程"
            ),
            VocabularyTerm(
                term="testing", 
                domain=VocabularyDomain.DEVELOPMENT,
                synonyms=["test", "qa", "verification"],
                definition="軟體品質驗證過程"
            ),
            
            # 專案管理詞彙
            VocabularyTerm(
                term="milestone", 
                domain=VocabularyDomain.PROJECT,
                synonyms=["checkpoint", "target", "goal"],
                definition="專案重要節點或目標"
            ),
            VocabularyTerm(
                term="requirement", 
                domain=VocabularyDomain.PROJECT,
                synonyms=["spec", "specification", "criteria"],
                definition="專案需求規格"
            ),
        ]
        
        for term in default_terms:
            self.add_term(term)
    
    def add_term(self, term: VocabularyTerm) -> bool:
        """新增詞彙項目"""
        try:
            # 檢查是否已存在
            if term.term in self._vocabulary:
                logger.warning(f"詞彙 '{term.term}' 已存在")
                return False
            
            # 新增到主詞彙庫
            self._vocabulary[term.term] = term
            
            # 更新領域分類
            self._domain_terms[term.domain].add(term.term)
            
            # 更新同義詞映射
            for synonym in term.synonyms:
                self._synonym_mapping[synonym] = term.term
            
            logger.info(f"成功新增詞彙: {term.term}")
            return True
            
        except Exception as e:
            logger.error(f"新增詞彙失敗: {e}")
            return False
    
    def get_standard_term(self, input_term: str) -> Optional[str]:
        """獲取標準化詞彙"""
        normalized = re.sub(r'\s+', ' ', input_term.lower().strip())
        normalized = re.sub(r'[^\w\s\-_]', '', normalized)
        
        # 直接匹配
        if normalized in self._vocabulary:
            return normalized
        
        # 同義詞匹配
        if normalized in self._synonym_mapping:
            return self._synonym_mapping[normalized]
        
        return None
    
    def suggest_terms(self, input_term: str, limit: int = 5) -> List[str]:
        """建議相似詞彙"""
        normalized = re.sub(r'\s+', ' ', input_term.lower().strip())
        suggestions = []
        
        # 精確匹配
        standard_term = self.get_standard_term(input_term)
        if standard_term:
            suggestions.append(standard_term)
        
        # 部分匹配
        for term in self._vocabulary.keys():
            if normalized in term or term in normalized:
                if term not in suggestions:
                    suggestions.append(term)
        
        # 同義詞匹配
        for synonym, standard in self._synonym_mapping.items():
            if normalized in synonym or synonym in normalized:
                if standard not in suggestions:
                    suggestions.append(standard)
        
        return suggestions[:limit]
    
    def get_terms_by_domain(self, domain: VocabularyDomain) -> List[VocabularyTerm]:
        """獲取特定領域的詞彙"""
        return [
            self._vocabulary[term] 
            for term in self._domain_terms[domain]
            if term in self._vocabulary
        ]
    
    def validate_and_normalize_tags(self, tags: List[str]) -> List[str]:
        """驗證並標準化標籤"""
        normalized_tags = []
        
        for tag in tags:
            standard_term = self.get_standard_term(tag)
            if standard_term:
                normalized_tags.append(standard_term)
                # 更新使用統計
                if standard_term in self._vocabulary:
                    self._vocabulary[standard_term].usage_count += 1
            else:
                # 如果找不到標準詞彙，保留原始但標準化格式
                normalized = re.sub(r'\s+', ' ', tag.lower().strip())
                normalized = re.sub(r'[^\w\s\-_]', '', normalized)
                if normalized:
                    normalized_tags.append(normalized)
        
        return list(set(normalized_tags))  # 去重
    
    def get_related_terms(self, term: str) -> List[str]:
        """獲取相關詞彙"""
        standard_term = self.get_standard_term(term)
        if not standard_term or standard_term not in self._vocabulary:
            return []
        
        vocab_term = self._vocabulary[standard_term]
        related = set(vocab_term.related_terms + vocab_term.synonyms)
        
        # 從相同領域找相關詞彙
        domain_terms = self._domain_terms[vocab_term.domain]
        related.update(list(domain_terms)[:3])  # 限制數量
        
        return [t for t in related if t != standard_term]
    
    def export_vocabulary(self) -> Dict[str, Dict]:
        """匯出詞彙庫"""
        return {
            term: vocab.dict() 
            for term, vocab in self._vocabulary.items()
        }
    
    def import_vocabulary(self, vocabulary_data: Dict[str, Dict]) -> int:
        """匯入詞彙庫"""
        imported_count = 0
        
        for term_data in vocabulary_data.values():
            try:
                term = VocabularyTerm(**term_data)
                if self.add_term(term):
                    imported_count += 1
            except Exception as e:
                logger.error(f"匯入詞彙失敗: {e}")
        
        return imported_count


# 全域詞彙管理器實例
vocabulary_manager = VocabularyManager()


def normalize_fragment_content(fragment: FragmentSchema) -> FragmentSchema:
    """標準化分片內容"""
    # 標準化標籤
    fragment.tags = vocabulary_manager.validate_and_normalize_tags(fragment.tags)
    
    # 更新時間
    fragment.updated_at = datetime.now()
    
    return fragment


def get_fragment_keywords(fragment: FragmentSchema) -> List[str]:
    """提取分片關鍵詞用於搜尋"""
    keywords = []
    
    # 從標題提取
    title_words = re.findall(r'\b\w+\b', fragment.title.lower())
    keywords.extend(title_words)
    
    # 標準化標籤
    keywords.extend(fragment.tags)
    
    # 領域關鍵詞
    keywords.extend([domain.value for domain in fragment.domains])
    
    # 經驗類型關鍵詞
    keywords.extend([exp_type.value for exp_type in fragment.experience_types])
    
    # 分片類型
    keywords.append(fragment.fragment_type.value)
    
    # 去重並過濾空值
    return list(set([kw for kw in keywords if kw]))