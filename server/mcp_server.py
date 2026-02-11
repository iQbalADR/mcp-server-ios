"""
MCP Xcode Server
================

Main MCP server connecting Ollama LLM with Xcode's AI Assistant.
Uses LanceDB for persistent memory and RAG capabilities.

Usage:
    python -m server.mcp_server
    
    # For Xcode 26+ integration, use the HTTP server:
    python -m server.http_server --port 1234
"""

import asyncio
import json
import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Sequence

import yaml
from loguru import logger
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    TextContent,
    Tool,
    Resource,
    Prompt,
    PromptMessage,
    PromptArgument,
    GetPromptResult,
)

from server.ollama_client import OllamaClient, OllamaConfig
from server.vector_store import VectorStore, create_vector_store_with_ollama, SearchResult


# ============================================================================
# Configuration
# ============================================================================

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    
    if config_file.exists():
        with open(config_file) as f:
            config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
    else:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return {}


def setup_logging(config: Dict[str, Any]) -> None:
    """Configure logging based on config."""
    log_config = config.get("logging", {})
    
    logger.remove()
    
    if log_config.get("console", True):
        logger.add(
            sys.stderr,
            level=log_config.get("level", "DEBUG"),
            format="{time:HH:mm:ss} | {level: <8} | {message}",
            colorize=True,
        )
    
    log_file = log_config.get("file")
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level=log_config.get("level", "DEBUG"),
            rotation="10 MB",
            retention="1 week",
        )


# ============================================================================
# Debug Logger
# ============================================================================

class DebugLogger:
    """Logger for debugging MCP requests and responses."""
    
    def __init__(self, enabled: bool = True, log_file: Optional[str] = None):
        self.enabled = enabled
        self.log_file = log_file
        self._request_id = 0
        
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    def log_request(self, method: str, params: Any) -> int:
        if not self.enabled:
            return 0
        
        self._request_id += 1
        logger.debug(f"[REQ {self._request_id}] {method}")
        
        if self.log_file:
            entry = {
                "id": self._request_id,
                "timestamp": datetime.now().isoformat(),
                "type": "request",
                "method": method,
            }
            self._append_to_file(entry)
        
        return self._request_id
    
    def log_response(self, request_id: int, result: Any, error: Optional[str] = None) -> None:
        if not self.enabled:
            return
        
        if error:
            logger.debug(f"[RES {request_id}] Error: {error}")
        else:
            logger.debug(f"[RES {request_id}] Success")
    
    def _append_to_file(self, entry: Dict[str, Any]) -> None:
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write debug log: {e}")


# ============================================================================
# RAG Pipeline
# ============================================================================

class RAGPipeline:
    """Retrieval-Augmented Generation pipeline."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        ollama_client: OllamaClient,
        config: Dict[str, Any],
    ):
        self.vector_store = vector_store
        self.ollama = ollama_client
        self.config = config.get("rag", {})
        self.context_chunks = self.config.get("context_chunks", 5)
    
    async def query(
        self,
        question: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Query with RAG augmentation."""
        start_time = time.time()
        
        # Collect context
        context_parts = []
        
        # Search code
        code_results = await self.vector_store.search_code(question, n_results=self.context_chunks)
        if code_results:
            context_parts.append("## Relevant Code\n")
            for r in code_results:
                if r.metadata.get("file_path"):
                    context_parts.append(f"### {r.metadata['file_path']}\n")
                context_parts.append(f"```\n{r.content}\n```\n")
        
        # Search docs
        doc_results = await self.vector_store.search_docs(question, n_results=self.context_chunks // 2)
        if doc_results:
            context_parts.append("## Relevant Documentation\n")
            for r in doc_results:
                context_parts.append(f"{r.content}\n\n")
        
        context = "\n".join(context_parts) if context_parts else None
        
        logger.debug(f"RAG retrieval took {time.time() - start_time:.2f}s")
        
        # Query LLM
        response = await self.ollama.chat(
            prompt=question,
            system_prompt=system_prompt,
            context=context,
        )
        
        # Store conversation
        await self.vector_store.add_history(f"Q: {question}\nA: {response[:500]}...", role="conversation")
        
        return response
    
    async def add_to_memory(
        self,
        content: str,
        memory_type: str = "code",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add content to memory."""
        if memory_type == "code":
            return await self.vector_store.add_code(
                content,
                file_path=metadata.get("file", "") if metadata else "",
                language=metadata.get("language", "swift") if metadata else "swift",
                metadata=metadata,
            )
        else:
            return await self.vector_store.add_doc(
                content,
                source=metadata.get("source", "") if metadata else "",
                doc_type=memory_type,
                metadata=metadata,
            )
    
    async def search_memory(
        self,
        query: str,
        memory_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[SearchResult]:
        """Search memory."""
        results = []
        
        if memory_type is None or memory_type == "code":
            results.extend(await self.vector_store.search_code(query, n_results=limit))
        
        if memory_type is None or memory_type == "docs":
            results.extend(await self.vector_store.search_docs(query, n_results=limit))
        
        # Sort by distance
        results.sort(key=lambda x: x.distance)
        return results[:limit]


# ============================================================================
# MCP Server
# ============================================================================

class XcodeMCPServer:
    """MCP Server for Xcode AI Assistant integration."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.server = Server(config.get("server", {}).get("name", "xcode-mcp-server"))
        
        debug_config = config.get("debug", {})
        self.debug_logger = DebugLogger(
            enabled=debug_config.get("enabled", True),
            log_file=debug_config.get("request_log"),
        )
        
        self.ollama: Optional[OllamaClient] = None
        self.vector_store: Optional[VectorStore] = None
        self.rag: Optional[RAGPipeline] = None
        
        self._register_handlers()
    
    def get_tools_for_ollama(self) -> List[Dict[str, Any]]:
        """Convert MCP tools to Ollama tool format."""
        # We manually map the tools we want to expose to Ollama
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List files in a directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Directory path"},
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read file content",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path"},
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_to_file",
                    "description": "Write content to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path"},
                            "content": {"type": "string", "description": "File content"},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": "Run terminal command",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "Command string"},
                        },
                        "required": ["command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "apply_patch",
                    "description": "Apply a git-style patch to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path"},
                            "diff": {"type": "string", "description": "Unified diff content"},
                        },
                        "required": ["path", "diff"],
                    },
                },
            },
             {
                "type": "function",
                "function": {
                    "name": "ask_with_context",
                    "description": "Search code/docs (RAG)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string", "description": "Query"},
                        },
                        "required": ["question"],
                    },
                },
            },
        ]
    
    async def initialize(self) -> None:
        """Initialize all components."""
        logger.info("Initializing MCP Server...")
        
        # Initialize Ollama
        ollama_config = OllamaConfig.from_dict(self.config.get("ollama", {}))
        self.ollama = OllamaClient(ollama_config)
        
        if not await self.ollama.is_healthy():
            logger.error("Ollama server is not running!")
            raise RuntimeError("Ollama server not available")
        
        # Initialize vector store with Ollama embeddings
        lancedb_config = self.config.get("lancedb", {})
        self.vector_store = await create_vector_store_with_ollama(
            db_path=lancedb_config.get("db_path", "./data/lancedb"),
            ollama_base_url=self.config.get("ollama", {}).get("base_url", "http://localhost:11434"),
            embedding_model=self.config.get("ollama", {}).get("embedding_model", "nomic-embed-text"),
        )
        
        # Initialize RAG
        self.rag = RAGPipeline(
            vector_store=self.vector_store,
            ollama_client=self.ollama,
            config=self.config,
        )
        
        logger.info("MCP Server initialized!")
        stats = await self.vector_store.get_stats()
        logger.info(f"Vector store stats: {stats}")
    
    def _register_handlers(self) -> None:
        """Register MCP protocol handlers."""
        
        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            return [
                Tool(
                    name="generate_code",
                    description="Generate code based on description with RAG context.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "description": {"type": "string", "description": "What to generate"},
                            "language": {"type": "string", "default": "swift"},
                        },
                        "required": ["description"],
                    },
                ),
                Tool(
                    name="explain_code",
                    description="Explain what code does.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "code": {"type": "string"},
                            "language": {"type": "string", "default": "swift"},
                        },
                        "required": ["code"],
                    },
                ),
                Tool(
                    name="fix_code",
                    description="Fix code based on error message.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "code": {"type": "string"},
                            "error": {"type": "string"},
                            "language": {"type": "string", "default": "swift"},
                        },
                        "required": ["code", "error"],
                    },
                ),
                Tool(
                    name="ask_with_context",
                    description="Ask a question with RAG context.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                        },
                        "required": ["question"],
                    },
                ),
                Tool(
                    name="add_to_memory",
                    description="Add code or docs to memory.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "type": {"type": "string", "enum": ["code", "docs", "note"], "default": "code"},
                            "file_path": {"type": "string"},
                        },
                        "required": ["content"],
                    },
                ),
                Tool(
                    name="list_files",
                    description="List files in a directory to explore the project structure.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Directory path (relative to project root)"},
                        },
                        "required": ["path"],
                    },
                ),
                Tool(
                    name="read_file",
                    description="Read the contents of a file.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path"},
                        },
                        "required": ["path"],
                    },
                ),
                Tool(
                    name="write_to_file",
                    description="Create or overwrite a file with new content.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path"},
                            "content": {"type": "string", "description": "New file content"},
                        },
                        "required": ["path", "content"],
                    },
                ),
                Tool(
                    name="apply_patch",
                    description="Apply a git-style patch to a file.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path to patch"},
                            "diff": {"type": "string", "description": "The patch content (unified diff)"},
                        },
                        "required": ["path", "diff"],
                    },
                ),
                Tool(
                    name="run_command",
                    description="Run a terminal command (e.g., xcodebuild, git).",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "Command to execute"},
                        },
                        "required": ["command"],
                    },
                ),

                Tool(
                    name="search_memory",
                    description="Search memory/notebook.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "type": {"type": "string", "enum": ["code", "docs", "all"], "default": "all"},
                            "limit": {"type": "integer", "default": 10},
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="get_memory_stats",
                    description="Get memory statistics.",
                    inputSchema={"type": "object", "properties": {}},
                ),
                Tool(
                    name="clear_memory",
                    description="Clear a memory collection.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "collection": {"type": "string", "enum": ["code", "docs", "history"]},
                        },
                        "required": ["collection"],
                    },
                ),
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> Sequence[TextContent]:
            req_id = self.debug_logger.log_request(f"tools/call/{name}", arguments)
            
            try:
                result = await self._execute_tool(name, arguments)
                self.debug_logger.log_response(req_id, "Success")
                return [TextContent(type="text", text=result)]
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                self.debug_logger.log_response(req_id, None, error_msg)
                logger.error(f"Tool failed: {e}")
                return [TextContent(type="text", text=error_msg)]
        
        @self.server.list_resources()
        async def handle_list_resources() -> list[Resource]:
            return [
                Resource(
                    uri="memory://stats",
                    name="Memory Statistics",
                    description="Vector database stats",
                    mimeType="application/json",
                ),
            ]
        
        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            if uri == "memory://stats":
                stats = await self.vector_store.get_stats()
                return json.dumps(stats, indent=2)
            raise ValueError(f"Unknown resource: {uri}")
        
        @self.server.list_prompts()
        async def handle_list_prompts() -> list[Prompt]:
            return [
                Prompt(
                    name="code_review",
                    description="Review code for issues",
                    arguments=[
                        PromptArgument(name="code", description="Code to review", required=True),
                    ],
                ),
            ]
        
        @self.server.get_prompt()
        async def handle_get_prompt(name: str, arguments: dict) -> GetPromptResult:
            if name == "code_review":
                code = arguments.get("code", "")
                return GetPromptResult(
                    description="Review code",
                    messages=[
                        PromptMessage(
                            role="user",
                            content=TextContent(
                                type="text",
                                text=f"Please review this code:\n\n```\n{code}\n```",
                            ),
                        ),
                    ],
                )
            raise ValueError(f"Unknown prompt: {name}")
    
    async def _execute_tool(self, name: str, arguments: dict) -> str:
        """Execute a tool."""
        if name == "generate_code":
            description = arguments["description"]
            language = arguments.get("language", "swift")
            
            context_docs = await self.rag.search_memory(description, "code", limit=5)
            context = "\n\n".join([f"```\n{d.content}\n```" for d in context_docs]) if context_docs else None
            
            return await self.ollama.generate_code(description, language, context)
        
        elif name == "explain_code":
            return await self.ollama.explain_code(arguments["code"], arguments.get("language", "swift"))
        
        elif name == "fix_code":
            return await self.ollama.fix_code(arguments["code"], arguments["error"], arguments.get("language", "swift"))
        
        elif name == "ask_with_context":
            return await self.rag.query(arguments["question"])
        
        elif name == "add_to_memory":
            content = arguments["content"]
            memory_type = arguments.get("type", "code")
            metadata = {"file": arguments.get("file_path", "")}
            
            doc_id = await self.rag.add_to_memory(content, memory_type, metadata)
            return f"Added to memory: {doc_id}"
        
        elif name == "search_memory":
            query = arguments["query"]
            memory_type = arguments.get("type")
            if memory_type == "all":
                memory_type = None
            limit = arguments.get("limit", 10)
            
            results = await self.rag.search_memory(query, memory_type, limit)
            
            if not results:
                return "No matches found."
            
            output = []
            for r in results:
                output.append(f"### {r.metadata.get('file_path', 'unknown')}")
                output.append(f"```\n{r.content[:500]}\n```\n")
            
            return "\n".join(output)
        
        elif name == "get_memory_stats":
            stats = await self.vector_store.get_stats()
            return json.dumps(stats, indent=2)
        
        elif name == "clear_memory":
            collection = arguments["collection"]
            success = await self.vector_store.clear_collection(collection)
            return f"Cleared {collection}" if success else f"Failed to clear {collection}"
        
        # --- Agentic Tools ---
        elif name == "list_files":
            path = arguments.get("path", ".")
            # basic limitation for safety: don't go outside home if possible, but for now just raw
            if ".." in path: 
                return "Error: Accessing parent directories is restricted."
            
            try:
                # Assuming current working directory is project root
                full_path = Path(os.getcwd()) / path
                if not full_path.exists():
                     return f"Error: Path {path} does not exist."
                
                items = os.listdir(full_path)
                # Filter hidden files
                items = [i for i in items if not i.startswith(".")]
                return "\n".join(items)
            except Exception as e:
                return f"Error listing files: {e}"

        elif name == "read_file":
            path = arguments["path"]
            try:
                with open(path, "r") as f:
                    return f.read()
            except Exception as e:
                return f"Error reading file {path}: {e}"

        elif name == "write_to_file":
            path = arguments["path"]
            content = arguments["content"]
            try:
                # Ensure dir exists
                p = Path(path)
                p.parent.mkdir(parents=True, exist_ok=True)
                with open(p, "w") as f:
                    f.write(content)
                return f"Successfully wrote to {path}"
            except Exception as e:
                return f"Error writing file {path}: {e}"

        elif name == "run_command":
            command = arguments["command"]
            # Security warning: valid for local use, dangerous if exposed
            allowed_prefixes = ["git", "xcodebuild", "ls", "pwd", "cat", "echo", "swift", "mkdir", "rm"]
            if not any(command.strip().startswith(prefix) for prefix in allowed_prefixes):
                 # Relaxed for now as user requested "Action Tools"
                 pass 

            try:
                result = subprocess.run(
                    command, 
                    shell=True, 
                    capture_output=True, 
                    text=True,
                    timeout=60
                )
                if result.returncode == 0:
                    return f"success: {result.stdout}"
                else:
                    return f"error (code {result.returncode}): {result.stderr}"
            except Exception as e:
                return f"Execution failed: {e}"

        elif name == "apply_patch":
            path = arguments["path"]
            diff = arguments["diff"]
            try:
                # Create temporary patch file
                with open("temp.patch", "w") as f:
                    f.write(diff)
                
                # Apply patch
                result = subprocess.run(
                    ["git", "apply", "temp.patch"],
                    capture_output=True,
                    text=True
                )
                
                # Cleanup
                os.remove("temp.patch")
                
                if result.returncode == 0:
                    return f"Successfully applied patch to {path}"
                else:
                    return f"Failed to apply patch: {result.stderr}"
            except Exception as e:
                if os.path.exists("temp.patch"):
                    os.remove("temp.patch")
                return f"Error applying patch: {e}"

        else:
            raise ValueError(f"Unknown tool: {name}")
    
    async def run(self) -> None:
        """Run the MCP server."""
        await self.initialize()
        
        logger.info("Starting MCP server on stdio...")
        logger.info("Waiting for Xcode connections...")
        
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )


# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    """Main entry point."""
    config = load_config()
    setup_logging(config)
    
    logger.info("=" * 50)
    logger.info("MCP Xcode Server with LanceDB")
    logger.info("=" * 50)
    
    server = XcodeMCPServer(config)
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
