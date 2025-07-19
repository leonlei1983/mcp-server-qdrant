import asyncio
import logging
from typing import Any

import aiohttp

from mcp_server_qdrant.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class OllamaProvider(EmbeddingProvider):
    """
    Ollama implementation of the embedding provider.
    Supports models like nomic-embed-text, all-minilm, etc.
    """

    def __init__(self, model_name: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self._vector_size = None
        
    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Embed a list of documents into vectors."""
        embeddings = []
        
        async with aiohttp.ClientSession() as session:
            for doc in documents:
                embedding = await self._get_embedding(session, doc)
                embeddings.append(embedding)
                
        return embeddings

    async def embed_query(self, query: str) -> list[float]:
        """Embed a query into a vector."""
        async with aiohttp.ClientSession() as session:
            return await self._get_embedding(session, query)

    async def _get_embedding(self, session: aiohttp.ClientSession, text: str) -> list[float]:
        """Get embedding for a single text using Ollama API."""
        payload = {
            "model": self.model_name,
            "prompt": text
        }
        
        try:
            async with session.post(
                f"{self.base_url}/api/embeddings",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["embedding"]
                else:
                    error_text = await response.text()
                    logger.error(f"Ollama API error {response.status}: {error_text}")
                    raise Exception(f"Ollama API error: {response.status}")
                    
        except Exception as e:
            logger.error(f"Failed to get embedding from Ollama: {e}")
            raise

    def get_vector_name(self) -> str:
        """Return the name of the vector for the Qdrant collection."""
        # Normalize model name for Qdrant vector name (consistent with fastembed style)
        model_name = self.model_name.replace("_", "-").replace(":", "-").lower()
        return f"ollama-{model_name}"

    def get_vector_size(self) -> int:
        """Get the size of the vector for the Qdrant collection."""
        if self._vector_size is None:
            # Common embedding dimensions for Ollama models
            model_dimensions = {
                "nomic-embed-text": 768,
                "all-minilm": 384,
                "all-minilm:l6-v2": 384,
                "bge-large-en-v1.5": 1024,
                "multilingual-e5-large": 1024,
            }
            
            self._vector_size = model_dimensions.get(self.model_name, 768)  # Default to 768
            logger.info(f"Using vector size {self._vector_size} for model {self.model_name}")
            
        return self._vector_size

    async def test_connection(self) -> bool:
        """Test if Ollama is available and the model is loaded."""
        try:
            async with aiohttp.ClientSession() as session:
                # First check if Ollama is running
                async with session.get(
                    f"{self.base_url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status != 200:
                        logger.error(f"Ollama not available at {self.base_url}")
                        return False
                    
                    # Check if model is available
                    models = await response.json()
                    model_names = [model["name"] for model in models.get("models", [])]
                    
                    if self.model_name not in model_names:
                        logger.warning(f"Model {self.model_name} not found in Ollama. Available models: {model_names}")
                        return False
                    
                    logger.info(f"Ollama connection successful. Model {self.model_name} is available.")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to test Ollama connection: {e}")
            return False