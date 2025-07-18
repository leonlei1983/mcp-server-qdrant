"""
Test suite for Ollama embedding provider.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import aiohttp
from aioresponses import aioresponses

from mcp_server_qdrant.embeddings.ollama import OllamaProvider


@pytest.fixture
def ollama_provider():
    """Fixture for Ollama provider."""
    return OllamaProvider(model_name="nomic-embed-text")


@pytest.mark.asyncio
async def test_embed_documents(ollama_provider):
    """Test embedding documents."""
    
    with aioresponses() as m:
        # Mock the HTTP response for multiple calls
        m.post(
            "http://localhost:11434/api/embeddings",
            payload={"embedding": [0.1, 0.2, 0.3]}
        )
        m.post(
            "http://localhost:11434/api/embeddings",
            payload={"embedding": [0.1, 0.2, 0.3]}
        )
        
        documents = ["test document 1", "test document 2"]
        embeddings = await ollama_provider.embed_documents(documents)
        
        # Verify results
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_embed_query(ollama_provider):
    """Test embedding a query."""
    
    with aioresponses() as m:
        # Mock the HTTP response
        m.post(
            "http://localhost:11434/api/embeddings",
            payload={"embedding": [0.4, 0.5, 0.6]}
        )
        
        query = "test query"
        embedding = await ollama_provider.embed_query(query)
        
        # Verify results
        assert embedding == [0.4, 0.5, 0.6]


@pytest.mark.asyncio
async def test_embed_error_handling(ollama_provider):
    """Test error handling in embedding."""
    
    with aioresponses() as m:
        # Mock error response
        m.post(
            "http://localhost:11434/api/embeddings",
            status=500,
            payload="Internal Server Error"
        )
        
        with pytest.raises(Exception) as exc_info:
            await ollama_provider.embed_query("test query")
        
        assert "Ollama API error" in str(exc_info.value)


def test_get_vector_name(ollama_provider):
    """Test vector name generation."""
    vector_name = ollama_provider.get_vector_name()
    assert vector_name == "ollama_nomic_embed_text"


def test_get_vector_size(ollama_provider):
    """Test vector size for different models."""
    
    # Test nomic-embed-text
    provider = OllamaProvider("nomic-embed-text")
    assert provider.get_vector_size() == 768
    
    # Test all-minilm
    provider = OllamaProvider("all-minilm")
    assert provider.get_vector_size() == 384
    
    # Test unknown model (should default to 768)
    provider = OllamaProvider("unknown-model")
    assert provider.get_vector_size() == 768


@pytest.mark.asyncio
async def test_test_connection_success(ollama_provider):
    """Test successful connection test."""
    
    with aioresponses() as m:
        # Mock successful response
        m.get(
            "http://localhost:11434/api/tags",
            payload={
                "models": [
                    {"name": "nomic-embed-text"},
                    {"name": "other-model"}
                ]
            }
        )
        
        result = await ollama_provider.test_connection()
        
        assert result is True


@pytest.mark.asyncio
async def test_test_connection_model_not_found(ollama_provider):
    """Test connection test when model is not found."""
    
    with aioresponses() as m:
        # Mock response without our model
        m.get(
            "http://localhost:11434/api/tags",
            payload={
                "models": [
                    {"name": "other-model-1"},
                    {"name": "other-model-2"}
                ]
            }
        )
        
        result = await ollama_provider.test_connection()
        
        assert result is False


@pytest.mark.asyncio
async def test_test_connection_server_error(ollama_provider):
    """Test connection test when server is not available."""
    
    with aioresponses() as m:
        # Mock server error
        m.get(
            "http://localhost:11434/api/tags",
            status=500
        )
        
        result = await ollama_provider.test_connection()
        
        assert result is False


@pytest.mark.asyncio
async def test_test_connection_network_error(ollama_provider):
    """Test connection test with network error."""
    
    with aioresponses() as m:
        # Mock network error
        m.get(
            "http://localhost:11434/api/tags",
            exception=aiohttp.ClientError("Network error")
        )
        
        result = await ollama_provider.test_connection()
        
        assert result is False


def test_custom_base_url():
    """Test custom base URL configuration."""
    provider = OllamaProvider(
        model_name="nomic-embed-text",
        base_url="http://custom-host:11434"
    )
    
    assert provider.base_url == "http://custom-host:11434"
    assert provider.model_name == "nomic-embed-text"


def test_base_url_trailing_slash_removal():
    """Test that trailing slashes are removed from base URL."""
    provider = OllamaProvider(
        model_name="nomic-embed-text",
        base_url="http://localhost:11434/"
    )
    
    assert provider.base_url == "http://localhost:11434"


if __name__ == "__main__":
    pytest.main([__file__])