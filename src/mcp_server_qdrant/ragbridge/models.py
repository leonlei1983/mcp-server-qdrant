"""
RAG Bridge specific data models and structures.
"""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ContentType(str, Enum):
    """Types of content in the RAG Bridge system."""
    EXPERIENCE = "experience"
    PROCESS_WORKFLOW = "process_workflow"
    KNOWLEDGE_BASE = "knowledge_base"
    VOCABULARY = "vocabulary"
    DECISION_RECORD = "decision_record"


class ContentStatus(str, Enum):
    """Status of content in the RAG Bridge system."""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"
    DRAFT = "draft"
    ARCHIVED = "archived"


class RAGMetadata(BaseModel):
    """Enhanced metadata structure for RAG Bridge."""
    
    # Core identification
    content_type: ContentType
    content_id: str = Field(description="Unique identifier for the content")
    title: str = Field(description="Human-readable title")
    
    # Version management
    version: str = Field(default="1.0", description="Version number")
    status: ContentStatus = Field(default=ContentStatus.ACTIVE)
    
    # Content classification
    tags: List[str] = Field(default_factory=list, description="Searchable tags")
    categories: List[str] = Field(default_factory=list, description="Content categories")
    
    # Source information
    source: Optional[str] = Field(None, description="Source of the content")
    author: Optional[str] = Field(None, description="Author or contributor")
    
    # Temporal information
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    # Usage and quality metrics
    usage_count: int = Field(default=0, description="Number of times accessed")
    success_rate: float = Field(default=0.0, description="Success rate when used")
    quality_score: float = Field(default=0.0, description="Quality assessment score")
    
    # Relationships
    related_content: List[str] = Field(default_factory=list, description="Related content IDs")
    dependencies: List[str] = Field(default_factory=list, description="Content dependencies")
    
    # Context information
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    
    # Language and localization
    language: str = Field(default="en", description="Content language")
    
    # Custom fields for extensibility
    custom_fields: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata fields")


class ExperienceContent(BaseModel):
    """Structure for experience-type content."""
    
    problem_description: str = Field(description="Description of the problem or situation")
    solution_approach: str = Field(description="Approach taken to solve the problem")
    implementation_details: str = Field(description="Technical implementation details")
    outcomes: str = Field(description="Results and outcomes achieved")
    lessons_learned: str = Field(description="Key insights and lessons")
    
    # Technical details
    technologies_used: List[str] = Field(default_factory=list)
    difficulty_level: str = Field(default="medium", description="easy, medium, hard")
    time_invested: Optional[str] = Field(None, description="Time spent on the task")
    
    # Quality metrics
    confidence_level: float = Field(default=0.8, description="Confidence in the solution")
    reusability_score: float = Field(default=0.7, description="How reusable is this solution")


class ProcessWorkflowContent(BaseModel):
    """Structure for process workflow content."""
    
    process_name: str = Field(description="Name of the process")
    process_description: str = Field(description="Description of what the process does")
    steps: List[Dict[str, Any]] = Field(description="Detailed steps of the process")
    
    # Process metadata
    process_type: str = Field(default="manual", description="manual, automated, hybrid")
    estimated_duration: Optional[str] = Field(None, description="Expected time to complete")
    prerequisites: List[str] = Field(default_factory=list)
    
    # Quality and validation
    success_criteria: List[str] = Field(default_factory=list)
    common_pitfalls: List[str] = Field(default_factory=list)
    validation_steps: List[str] = Field(default_factory=list)


class KnowledgeBaseContent(BaseModel):
    """Structure for knowledge base content."""
    
    topic: str = Field(description="Main topic of the knowledge")
    content: str = Field(description="The actual knowledge content")
    summary: str = Field(description="Brief summary of the content")
    
    # Knowledge classification
    knowledge_type: str = Field(default="factual", description="factual, procedural, conceptual")
    complexity_level: str = Field(default="intermediate", description="beginner, intermediate, advanced")
    
    # References and sources
    references: List[str] = Field(default_factory=list)
    external_links: List[str] = Field(default_factory=list)


class VocabularyContent(BaseModel):
    """Structure for vocabulary and terminology content."""
    
    term: str = Field(description="The term or concept")
    definition: str = Field(description="Definition of the term")
    synonyms: List[str] = Field(default_factory=list)
    related_terms: List[str] = Field(default_factory=list)
    
    # Context and usage
    domain: str = Field(description="Domain or field where this term is used")
    usage_examples: List[str] = Field(default_factory=list)
    
    # Standardization
    is_standard: bool = Field(default=False, description="Is this a standardized term")
    standard_source: Optional[str] = Field(None, description="Source of standardization")


class DecisionRecordContent(BaseModel):
    """Structure for decision record content (ADR style)."""
    
    decision_title: str = Field(description="Title of the decision")
    decision_description: str = Field(description="Detailed description of the decision")
    context: str = Field(description="Context and background that led to this decision")
    alternatives_considered: List[str] = Field(default_factory=list, description="Alternative options that were considered")
    decision_rationale: str = Field(description="Reasoning behind the decision")
    consequences: str = Field(description="Expected consequences and implications")
    
    # Decision metadata
    decision_status: str = Field(default="active", description="Status: proposed, active, superseded, deprecated")
    decision_date: Optional[str] = Field(None, description="Date when decision was made")
    stakeholders: List[str] = Field(default_factory=list, description="People involved in the decision")
    
    # Impact assessment
    impact_level: str = Field(default="medium", description="Impact level: low, medium, high, critical")
    affected_systems: List[str] = Field(default_factory=list, description="Systems or components affected")
    implementation_notes: str = Field(default="", description="Notes on implementation")
    
    # Review and validation
    review_date: Optional[str] = Field(None, description="Date for next review")
    success_metrics: List[str] = Field(default_factory=list, description="How to measure success of this decision")


class RAGEntry(BaseModel):
    """Complete RAG entry structure."""
    
    content: str = Field(description="The searchable text content")
    metadata: RAGMetadata = Field(description="Structured metadata")
    
    # Typed content based on content_type
    structured_content: Union[
        ExperienceContent,
        ProcessWorkflowContent, 
        KnowledgeBaseContent,
        VocabularyContent,
        DecisionRecordContent,
        Dict[str, Any]
    ] = Field(description="Structured content based on type")
    
    # Search and retrieval optimization
    search_keywords: List[str] = Field(default_factory=list)
    semantic_chunks: List[str] = Field(default_factory=list)
    
    def get_collection_name(self) -> str:
        """Get the appropriate collection name for this entry."""
        return f"ragbridge_{self.metadata.content_type.value}"
    
    def get_search_text(self) -> str:
        """Get the text that should be embedded for search."""
        # Combine main content with searchable metadata
        search_parts = [
            self.content,
            self.metadata.title,
            " ".join(self.metadata.tags),
            " ".join(self.metadata.categories),
            " ".join(self.search_keywords)
        ]
        return " ".join(part for part in search_parts if part)


class SearchContext(BaseModel):
    """Context for search operations."""
    
    query: str = Field(description="The search query")
    content_types: List[ContentType] = Field(default_factory=list, description="Filter by content types")
    status_filter: List[ContentStatus] = Field(default_factory=list, description="Filter by status")
    
    # Search preferences
    max_results: int = Field(default=10, description="Maximum number of results")
    min_similarity: float = Field(default=0.7, description="Minimum similarity threshold")
    
    # Context-aware search
    current_project: Optional[str] = Field(None, description="Current project context")
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    
    # Time-based filtering
    date_range: Optional[Dict[str, datetime]] = Field(None, description="Date range filter")
    
    # Quality filtering
    min_quality_score: float = Field(default=0.0, description="Minimum quality score")
    include_experimental: bool = Field(default=False, description="Include experimental content")


class SearchResult(BaseModel):
    """Search result structure."""
    
    entry: RAGEntry = Field(description="The found entry")
    similarity_score: float = Field(description="Similarity score (0-1)")
    relevance_score: float = Field(description="Computed relevance score")
    
    # Result metadata
    rank: int = Field(description="Rank in search results")
    match_reasons: List[str] = Field(default_factory=list, description="Why this was matched")
    
    # Usage context
    usage_recommendation: str = Field(description="How to use this result")
    confidence_level: float = Field(description="Confidence in this result")