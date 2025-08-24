"""
MCP Server for KnowledgeBase Integration.

This module provides an MCP (Model Context Protocol) server for integrating
the KnowledgeBase with AI models and applications.
"""

import logging
import json
import asyncio
import time
from typing import List, Dict, Optional, Any, Union, AsyncGenerator
from pathlib import Path
import numpy as np
from dataclasses import dataclass, asdict
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import MCP components
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent

    MCP_AVAILABLE = True
except ImportError:
    logger.warning("MCP not available, MCP server functionality will be limited")
    MCP_AVAILABLE = False

# Import local modules
from ..knowledge_base import KnowledgeBase, KnowledgeBaseEntry, SearchResult


def _load_config():
    """Load configuration from pyproject.toml."""
    try:
        import tomllib

        pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
        if pyproject_path.exists():
            with open(pyproject_path, "rb") as f:
                config = tomllib.load(f)
                return config.get("tool", {}).get("knowledgebase", {})
    except Exception as e:
        logger.warning(f"Failed to load config from pyproject.toml: {e}")
    return {}


@dataclass
class MCPConfig:
    """Configuration for MCP server."""

    server_name: str = "knowledgebase"
    server_version: str = "0.2.0"
    max_results: int = 10
    max_context_length: int = 4000
    similarity_threshold: float = 0.1
    enable_caching: bool = True
    cache_ttl: int = 300  # 5 minutes


class KnowledgeBaseMCPServer:
    """
    MCP Server for KnowledgeBase operations.

    Features:
    - Vector search capabilities
    - Knowledge base management
    - Context-aware responses
    - Tool integration
    - Async operations
    """

    def __init__(
        self, knowledge_base: KnowledgeBase, config: Optional[MCPConfig] = None
    ):
        """
        Initialize the MCP server.

        Args:
            knowledge_base: KnowledgeBase instance
            config: MCP configuration
        """
        self.knowledge_base = knowledge_base
        self.config = config or MCPConfig()

        # Cache for search results
        self.search_cache = {}
        self.cache_lock = asyncio.Lock()

        # Initialize MCP server if available
        if MCP_AVAILABLE:
            self.mcp_server = Server(self.config.server_name)
            self._setup_mcp_tools()
        else:
            self.mcp_server = None

        logger.info(f"KnowledgeBase MCP server initialized")

    def _setup_mcp_tools(self):
        """Setup MCP tools and handlers."""
        if not MCP_AVAILABLE:
            return

        @self.mcp_server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available tools."""
            return [
                Tool(
                    name="search_knowledge_base",
                    description="Search the knowledge base for similar vectors",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query_vector": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "Query vector for similarity search",
                            },
                            "k": {
                                "type": "integer",
                                "description": "Number of results to return",
                                "default": 10,
                            },
                            "filters": {
                                "type": "object",
                                "description": "Optional metadata filters",
                            },
                            "similarity_threshold": {
                                "type": "number",
                                "description": "Minimum similarity threshold",
                                "default": 0.1,
                            },
                        },
                        "required": ["query_vector"],
                    },
                ),
                Tool(
                    name="add_vector",
                    description="Add a vector to the knowledge base",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "vector": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "Vector to add",
                            },
                            "text": {
                                "type": "string",
                                "description": "Text content associated with the vector",
                            },
                            "metadata": {
                                "type": "object",
                                "description": "Optional metadata",
                            },
                            "cluster_id": {
                                "type": "integer",
                                "description": "Optional cluster ID",
                            },
                        },
                        "required": ["vector", "text"],
                    },
                ),
                Tool(
                    name="get_vector",
                    description="Get a specific vector by ID",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "vector_id": {
                                "type": "string",
                                "description": "ID of the vector to retrieve",
                            }
                        },
                        "required": ["vector_id"],
                    },
                ),
                Tool(
                    name="delete_vector",
                    description="Delete a vector from the knowledge base",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "vector_id": {
                                "type": "string",
                                "description": "ID of the vector to delete",
                            }
                        },
                        "required": ["vector_id"],
                    },
                ),
                Tool(
                    name="get_statistics",
                    description="Get knowledge base statistics",
                    inputSchema={"type": "object", "properties": {}, "required": []},
                ),
                Tool(
                    name="clear_knowledge_base",
                    description="Clear all entries from the knowledge base",
                    inputSchema={"type": "object", "properties": {}, "required": []},
                ),
            ]

        @self.mcp_server.call_tool()
        async def handle_call_tool(
            name: str, arguments: Dict[str, Any]
        ) -> List[TextContent]:
            """Handle tool calls."""
            try:
                if name == "search_knowledge_base":
                    return await self._handle_search(arguments)
                elif name == "add_vector":
                    return await self._handle_add_vector(arguments)
                elif name == "get_vector":
                    return await self._handle_get_vector(arguments)
                elif name == "delete_vector":
                    return await self._handle_delete_vector(arguments)
                elif name == "get_statistics":
                    return await self._handle_get_statistics(arguments)
                elif name == "clear_knowledge_base":
                    return await self._handle_clear_knowledge_base(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")

            except Exception as e:
                logger.error(f"Tool call failed for {name}: {e}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]

    async def _handle_search(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle knowledge base search."""
        query_vector = arguments.get("query_vector", [])
        k = arguments.get("k", self.config.max_results)
        filters = arguments.get("filters", {})
        similarity_threshold = arguments.get(
            "similarity_threshold", self.config.similarity_threshold
        )

        # Validate input
        if not query_vector:
            raise ValueError("query_vector is required")

        if len(query_vector) != self.knowledge_base.embedding_dimension:
            raise ValueError(
                f"Query vector dimension mismatch: expected {self.knowledge_base.embedding_dimension}, "
                f"got {len(query_vector)}"
            )

        # Check cache
        cache_key = f"search_{hash(str(query_vector))}_{k}_{hash(str(filters))}"
        if self.config.enable_caching:
            async with self.cache_lock:
                cached_result = self.search_cache.get(cache_key)
                if (
                    cached_result
                    and (time.time() - cached_result["timestamp"])
                    < self.config.cache_ttl
                ):
                    logger.debug(f"Returning cached search result for {cache_key}")
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps(cached_result["data"], indent=2),
                        )
                    ]

        # Perform search
        query_embedding = np.array(query_vector, dtype=np.float32)
        results = self.knowledge_base.search(
            query_embedding=query_embedding,
            k=k,
            filters=filters,
        )

        # Filter by similarity threshold
        filtered_results = [
            result
            for result in results
            if result.similarity_score >= similarity_threshold
        ]

        # Format results
        formatted_results = []
        for result in filtered_results:
            formatted_results.append(
                {
                    "id": result.entry.id,
                    "text": result.entry.text,
                    "similarity_score": result.similarity_score,
                    "rank": result.rank,
                    "metadata": result.entry.metadata,
                    "cluster_id": result.entry.cluster_id,
                }
            )

        # Cache result
        if self.config.enable_caching:
            async with self.cache_lock:
                self.search_cache[cache_key] = {
                    "data": formatted_results,
                    "timestamp": time.time(),
                }

        return [TextContent(type="text", text=json.dumps(formatted_results, indent=2))]

    async def _handle_add_vector(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle adding a vector."""
        vector = arguments.get("vector", [])
        text = arguments.get("text", "")
        metadata = arguments.get("metadata", {})
        cluster_id = arguments.get("cluster_id")

        # Validate input
        if not vector:
            raise ValueError("vector is required")

        if not text:
            raise ValueError("text is required")

        if len(vector) != self.knowledge_base.embedding_dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.knowledge_base.embedding_dimension}, "
                f"got {len(vector)}"
            )

        # Add vector
        entry_id = self.knowledge_base.sync_manager.add_vector(
            vector=np.array(vector, dtype=np.float32),
            text=text,
            metadata=metadata,
            cluster_id=cluster_id,
        )

        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {"id": entry_id, "message": "Vector added successfully"}
                ),
            )
        ]

    async def _handle_get_vector(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle getting a vector."""
        vector_id = arguments.get("vector_id", "")

        if not vector_id:
            raise ValueError("vector_id is required")

        entry = self.knowledge_base.get_entry(vector_id)
        if not entry:
            raise ValueError(f"Vector {vector_id} not found")

        vector_data = {
            "id": entry.id,
            "text": entry.text,
            "embedding": entry.embedding.tolist(),
            "metadata": entry.metadata,
            "cluster_id": entry.cluster_id,
            "timestamp": entry.timestamp,
        }

        return [TextContent(type="text", text=json.dumps(vector_data, indent=2))]

    async def _handle_delete_vector(
        self, arguments: Dict[str, Any]
    ) -> List[TextContent]:
        """Handle deleting a vector."""
        vector_id = arguments.get("vector_id", "")

        if not vector_id:
            raise ValueError("vector_id is required")

        success = self.knowledge_base.sync_manager.delete_vector(vector_id)

        if success:
            return [
                TextContent(
                    type="text",
                    text=json.dumps({"message": "Vector deleted successfully"}),
                )
            ]
        else:
            raise ValueError(f"Failed to delete vector {vector_id}")

    async def _handle_get_statistics(
        self, arguments: Dict[str, Any]
    ) -> List[TextContent]:
        """Handle getting statistics."""
        stats = self.knowledge_base.get_statistics()

        return [TextContent(type="text", text=json.dumps(stats, indent=2))]

    async def _handle_clear_knowledge_base(
        self, arguments: Dict[str, Any]
    ) -> List[TextContent]:
        """Handle clearing the knowledge base."""
        self.knowledge_base.clear()

        return [
            TextContent(
                type="text",
                text=json.dumps({"message": "Knowledge base cleared successfully"}),
            )
        ]

    async def start_server(self, host: str = "localhost", port: int = 8001):
        """
        Start the MCP server.

        Args:
            host: Host to bind to
            port: Port to bind to
        """
        if not MCP_AVAILABLE:
            logger.error("MCP not available, cannot start server")
            return

        logger.info(f"Starting MCP server on {host}:{port}")

        # Start the server
        async with stdio_server() as (read_stream, write_stream):
            await self.mcp_server.run(
                read_stream,
                write_stream,
                self.mcp_server.create_initialization_options(),
            )

    def get_server_info(self) -> Dict[str, Any]:
        """Get server information."""
        return {
            "name": self.config.server_name,
            "version": self.config.server_version,
            "max_results": self.config.max_results,
            "max_context_length": self.config.max_context_length,
            "similarity_threshold": self.config.similarity_threshold,
            "enable_caching": self.config.enable_caching,
            "cache_ttl": self.config.cache_ttl,
            "mcp_available": MCP_AVAILABLE,
            "embedding_dimension": self.knowledge_base.embedding_dimension,
            "use_gpu": self.knowledge_base.use_gpu,
            "index_type": self.knowledge_base.index_type,
            "database_persist_directory": str(self.knowledge_base.persist_directory),
            "database_embedding_dimension": self.knowledge_base.embedding_dimension,
            "database_use_gpu": self.knowledge_base.use_gpu,
            "database_index_type": self.knowledge_base.index_type,
        }


def create_mcp_server(
    knowledge_base: KnowledgeBase, config: Optional[MCPConfig] = None
) -> KnowledgeBaseMCPServer:
    """
    Create an MCP server instance.

    Args:
        knowledge_base: KnowledgeBase instance
        config: MCP configuration

    Returns:
        KnowledgeBaseMCPServer instance
    """
    # Load configuration from pyproject.toml if not provided
    if config is None:
        config_data = _load_config()
        config = MCPConfig(
            server_name=config_data.get("mcp_server_name", "knowledgebase"),
            server_version=config_data.get("mcp_server_version", "0.2.0"),
            max_results=config_data.get("mcp_max_results", 10),
            max_context_length=config_data.get("mcp_max_context_length", 4000),
            similarity_threshold=config_data.get("mcp_similarity_threshold", 0.1),
            enable_caching=config_data.get("mcp_enable_caching", True),
            cache_ttl=config_data.get("mcp_cache_ttl", 300),
        )

    return KnowledgeBaseMCPServer(knowledge_base, config)


async def main():
    """Main entry point for MCP server."""
    import argparse

    # Load configuration from pyproject.toml
    config = _load_config()

    parser = argparse.ArgumentParser(description="KnowledgeBase MCP Server")
    parser.add_argument(
        "--persist-dir", help="Persistence directory (default from config)"
    )
    parser.add_argument(
        "--embedding-dim", type=int, help="Embedding dimension (default from config)"
    )
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for FAISS")
    parser.add_argument(
        "--index-type",
        choices=["flat", "ivf", "pq"],
        help="FAISS index type (default from config)",
    )
    parser.add_argument("--host", help="Host to bind to (default from config)")
    parser.add_argument(
        "--port", type=int, help="Port to bind to (default from config)"
    )

    args = parser.parse_args()

    # Load knowledge base configuration
    persist_dir = args.persist_dir or config.get(
        "database_persist_directory", "./data/knowledge_base"
    )
    embedding_dim = args.embedding_dim or config.get(
        "database_embedding_dimension", 1024
    )
    use_gpu = (
        args.use_gpu
        if args.use_gpu is not None
        else config.get("database_use_gpu", False)
    )
    index_type = args.index_type or config.get("database_index_type", "flat")

    # Create knowledge base
    knowledge_base = KnowledgeBase(
        persist_directory=persist_dir,
        embedding_dimension=embedding_dim,
        use_gpu=use_gpu,
        index_type=index_type,
    )

    # Create MCP server
    mcp_server = create_mcp_server(knowledge_base)

    # Load MCP server configuration
    mcp_host = args.host or config.get("mcp_host", "localhost")
    mcp_port = args.port or config.get("mcp_port", 8001)

    # Print server info
    print("KnowledgeBase MCP Server Info:")
    print(json.dumps(mcp_server.get_server_info(), indent=2))

    # Start server
    await mcp_server.start_server(host=mcp_host, port=mcp_port)


if __name__ == "__main__":
    asyncio.run(main())
