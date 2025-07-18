"""
Test suite for RAG Bridge connector functionality.
"""
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_server_qdrant.ragbridge.models import (
    ContentType,
    ContentStatus,
    RAGMetadata,
    ExperienceContent,
    RAGEntry,
    SearchContext,
)
from mcp_server_qdrant.ragbridge.connector import RAGBridgeConnector
from mcp_server_qdrant.embeddings.base import EmbeddingProvider


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""
    
    async def embed_documents(self, documents):
        return [[0.1, 0.2, 0.3] for _ in documents]
    
    async def embed_query(self, query):
        return [0.1, 0.2, 0.3]
    
    def get_vector_name(self):
        return "test_vector"
    
    def get_vector_size(self):
        return 3


@pytest.fixture
def mock_embedding_provider():
    """Fixture for mock embedding provider."""
    return MockEmbeddingProvider()


@pytest.fixture
def mock_qdrant_client():
    """Fixture for mock Qdrant client."""
    mock_client = AsyncMock()
    mock_client.collection_exists.return_value = True
    mock_client.get_collections.return_value = MagicMock()
    mock_client.get_collections.return_value.collections = []
    return mock_client


@pytest.fixture
def rag_connector(mock_embedding_provider, mock_qdrant_client):
    """Fixture for RAG Bridge connector."""
    connector = RAGBridgeConnector(
        qdrant_url="http://localhost:6333",
        qdrant_api_key="test_key",
        embedding_provider=mock_embedding_provider,
    )
    connector._client = mock_qdrant_client
    return connector


@pytest.fixture
def sample_rag_entry():
    """Fixture for sample RAG entry."""
    metadata = RAGMetadata(
        content_type=ContentType.EXPERIENCE,
        content_id="test_123",
        title="Test Experience",
        tags=["testing", "development"],
        categories=["software", "qa"],
    )
    
    structured_content = ExperienceContent(
        problem_description="Need to test RAG functionality",
        solution_approach="Create comprehensive test suite",
        implementation_details="Use pytest and mock objects",
        outcomes="Successful test implementation",
        lessons_learned="Testing is crucial for reliability",
        technologies_used=["pytest", "python", "qdrant"],
    )
    
    return RAGEntry(
        content="This is a test experience about implementing RAG testing",
        metadata=metadata,
        structured_content=structured_content,
        search_keywords=["test", "rag", "experience"],
    )


@pytest.mark.asyncio
async def test_store_rag_entry(rag_connector, sample_rag_entry, mock_qdrant_client):
    """Test storing a RAG entry."""
    
    # Mock the upsert method
    mock_qdrant_client.upsert.return_value = MagicMock()
    
    # Store the entry
    content_id = await rag_connector.store_rag_entry(sample_rag_entry)
    
    # Verify the result
    assert content_id == "test_123"
    
    # Verify client was called correctly
    mock_qdrant_client.upsert.assert_called_once()
    call_args = mock_qdrant_client.upsert.call_args
    
    assert call_args[1]["collection_name"] == "ragbridge_experience"
    assert len(call_args[1]["points"]) == 1
    
    point = call_args[1]["points"][0]
    assert point.id == "test_123"
    assert "test_vector" in point.vector
    assert point.payload["content"] == sample_rag_entry.content


@pytest.mark.asyncio
async def test_search_rag_entries(rag_connector, mock_qdrant_client):
    """Test searching RAG entries."""
    
    # Mock search response
    mock_result = MagicMock()
    mock_result.points = [
        MagicMock(
            id="test_123",
            score=0.95,
            payload={
                "content": "Test content",
                "metadata": {
                    "content_type": "experience",
                    "content_id": "test_123",
                    "title": "Test Experience",
                    "tags": ["testing"],
                    "categories": ["software"],
                    "version": "1.0",
                    "status": "active",
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "usage_count": 5,
                    "success_rate": 0.8,
                    "quality_score": 0.9,
                    "related_content": [],
                    "dependencies": [],
                    "context": {},
                    "language": "en",
                    "custom_fields": {},
                },
                "structured_content": {
                    "problem_description": "Test problem",
                    "solution_approach": "Test solution",
                    "implementation_details": "Test implementation",
                    "outcomes": "Test outcomes",
                    "lessons_learned": "Test lessons",
                    "technologies_used": ["pytest"],
                    "difficulty_level": "medium",
                    "confidence_level": 0.8,
                    "reusability_score": 0.7,
                },
                "search_keywords": ["test"],
                "semantic_chunks": [],
            }
        )
    ]
    
    mock_qdrant_client.query_points.return_value = mock_result
    mock_qdrant_client.get_collections.return_value.collections = [
        MagicMock(name="ragbridge_experience")
    ]
    
    # Create search context
    search_context = SearchContext(
        query="test problem solving",
        content_types=[ContentType.EXPERIENCE],
        max_results=10,
    )
    
    # Perform search
    results = await rag_connector.search_rag_entries(search_context)
    
    # Verify results
    assert len(results) == 1
    assert results[0].entry.metadata.content_id == "test_123"
    assert results[0].similarity_score == 0.95
    assert results[0].rank == 1
    
    # Verify client was called correctly
    mock_qdrant_client.query_points.assert_called()


@pytest.mark.asyncio
async def test_get_content_by_id(rag_connector, mock_qdrant_client):
    """Test retrieving content by ID."""
    
    # Mock retrieve response
    mock_result = [
        MagicMock(
            payload={
                "content": "Test content",
                "metadata": {
                    "content_type": "experience",
                    "content_id": "test_123",
                    "title": "Test Experience",
                    "tags": ["testing"],
                    "categories": ["software"],
                    "version": "1.0",
                    "status": "active",
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "usage_count": 5,
                    "success_rate": 0.8,
                    "quality_score": 0.9,
                    "related_content": [],
                    "dependencies": [],
                    "context": {},
                    "language": "en",
                    "custom_fields": {},
                },
                "structured_content": {},
                "search_keywords": ["test"],
                "semantic_chunks": [],
            }
        )
    ]
    
    mock_qdrant_client.retrieve.return_value = mock_result
    
    # Retrieve content
    content = await rag_connector.get_content_by_id("test_123", ContentType.EXPERIENCE)
    
    # Verify result
    assert content is not None
    assert content.metadata.content_id == "test_123"
    assert content.content == "Test content"
    
    # Verify client was called correctly
    mock_qdrant_client.retrieve.assert_called_once_with(
        collection_name="ragbridge_experience",
        ids=["test_123"],
        with_payload=True,
    )


@pytest.mark.asyncio
async def test_update_content_metadata(rag_connector, mock_qdrant_client):
    """Test updating content metadata."""
    
    # Mock get_content_by_id
    with patch.object(rag_connector, 'get_content_by_id') as mock_get:
        mock_entry = MagicMock()
        mock_entry.metadata.model_dump.return_value = {
            "content_type": "experience",
            "content_id": "test_123",
            "title": "Test Experience",
            "quality_score": 0.8,
        }
        mock_get.return_value = mock_entry
        
        # Update metadata
        updates = {"quality_score": 0.9, "tags": ["updated"]}
        await rag_connector.update_content_metadata("test_123", ContentType.EXPERIENCE, updates)
        
        # Verify set_payload was called
        mock_qdrant_client.set_payload.assert_called_once()
        call_args = mock_qdrant_client.set_payload.call_args
        
        assert call_args[1]["collection_name"] == "ragbridge_experience"
        assert call_args[1]["points"] == ["test_123"]
        assert call_args[1]["payload"]["metadata"]["quality_score"] == 0.9
        assert call_args[1]["payload"]["metadata"]["tags"] == ["updated"]


def test_rag_entry_get_collection_name(sample_rag_entry):
    """Test collection name generation."""
    collection_name = sample_rag_entry.get_collection_name()
    assert collection_name == "ragbridge_experience"


def test_rag_entry_get_search_text(sample_rag_entry):
    """Test search text generation."""
    search_text = sample_rag_entry.get_search_text()
    
    # Should contain main content
    assert "test experience" in search_text.lower()
    
    # Should contain title
    assert "test experience" in search_text.lower()
    
    # Should contain tags
    assert "testing" in search_text.lower()
    assert "development" in search_text.lower()
    
    # Should contain categories
    assert "software" in search_text.lower()
    assert "qa" in search_text.lower()
    
    # Should contain search keywords
    assert "test" in search_text.lower()
    assert "rag" in search_text.lower()


@pytest.mark.asyncio
async def test_search_with_filters(rag_connector, mock_qdrant_client):
    """Test search with various filters."""
    
    # Mock empty search response
    mock_result = MagicMock()
    mock_result.points = []
    mock_qdrant_client.query_points.return_value = mock_result
    mock_qdrant_client.get_collections.return_value.collections = [
        MagicMock(name="ragbridge_experience")
    ]
    
    # Create search context with filters
    search_context = SearchContext(
        query="test query",
        content_types=[ContentType.EXPERIENCE],
        status_filter=[ContentStatus.ACTIVE],
        min_quality_score=0.5,
        include_experimental=False,
        current_project="test_project",
    )
    
    # Perform search
    results = await rag_connector.search_rag_entries(search_context)
    
    # Verify search was performed
    mock_qdrant_client.query_points.assert_called()
    call_args = mock_qdrant_client.query_points.call_args
    
    # Check that filters were applied
    query_filter = call_args[1]["query_filter"]
    assert query_filter is not None
    assert query_filter.must is not None
    assert len(query_filter.must) > 0


if __name__ == "__main__":
    pytest.main([__file__])