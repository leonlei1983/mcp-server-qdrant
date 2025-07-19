import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from the package directory
package_dir = Path(__file__).parent.parent.parent
env_path = package_dir / ".env"
if env_path.exists():
    load_dotenv(env_path)

from mcp_server_qdrant.mcp_server import QdrantMCPServer
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    QdrantSettings,
    ToolSettings,
)

mcp = QdrantMCPServer(
    tool_settings=ToolSettings(),
    qdrant_settings=QdrantSettings(),
    embedding_provider_settings=EmbeddingProviderSettings(),
)
