"""
HTTP Server for Xcode Integration
==================================

This module provides an HTTP server wrapper for the MCP Xcode Server,
enabling integration with Xcode 26's "Locally Hosted" model provider.

Xcode 26+ expects an HTTP-based OpenAI-compatible API on a local port.

Usage:
    python -m server.http_server --port 1234
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

from aiohttp import web
from loguru import logger

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.mcp_server import XcodeMCPServer, load_config, setup_logging


class XcodeHTTPServer:
    """
    HTTP server providing OpenAI-compatible API for Xcode's locally hosted models.
    
    Xcode 26+ communicates with local model providers via an OpenAI-compatible API.
    This server wraps our MCP server to provide that interface.
    """
    
    def __init__(self, mcp_server: XcodeMCPServer, port: int = 1234):
        self.mcp_server = mcp_server
        self.port = port
        self.app = web.Application()
        self._setup_routes()
    
    def _setup_routes(self):
        """Set up HTTP routes."""
        self.app.router.add_get("/", self._handle_root)
        self.app.router.add_get("/health", self._handle_health)
        self.app.router.add_get("/v1/models", self._handle_models)
        self.app.router.add_post("/v1/chat/completions", self._handle_chat)
        self.app.router.add_post("/v1/completions", self._handle_completions)
        
        # MCP-specific endpoints
        self.app.router.add_get("/mcp/tools", self._handle_mcp_tools)
        self.app.router.add_post("/mcp/tools/{tool_name}", self._handle_mcp_tool_call)
        self.app.router.add_get("/mcp/stats", self._handle_mcp_stats)
        self.app.router.add_get("/mcp/labels", self._handle_mcp_labels)
    
    async def _handle_root(self, request: web.Request) -> web.Response:
        """Root endpoint with server info."""
        return web.json_response({
            "name": "MCP Xcode Server",
            "version": "1.0.0",
            "status": "running",
            "endpoints": {
                "chat": "/v1/chat/completions",
                "models": "/v1/models",
                "health": "/health",
                "mcp_tools": "/mcp/tools",
                "mcp_stats": "/mcp/stats",
                "mcp_labels": "/mcp/labels",
            }
        })
    
    async def _handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        ollama_healthy = await self.mcp_server.ollama.is_healthy()
        return web.json_response({
            "status": "healthy" if ollama_healthy else "degraded",
            "ollama": "connected" if ollama_healthy else "disconnected",
            "vector_store": "ready" if self.mcp_server.vector_store else "not initialized"
        })
    
    async def _handle_models(self, request: web.Request) -> web.Response:
        """List available models (OpenAI-compatible)."""
        models = await self.mcp_server.ollama.list_models()
        
        model_list = []
        for model in models:
            # Extract model name - Ollama returns dicts with 'name' key
            if isinstance(model, dict):
                model_name = model.get("name", model.get("model", str(model)))
            else:
                model_name = str(model)
            
            model_list.append({
                "id": model_name,
                "object": "model",
                "created": 0,
                "owned_by": "ollama"
            })
        
        return web.json_response({
            "object": "list",
            "data": model_list
        })
    
    async def _handle_chat(self, request: web.Request) -> web.Response:
        """Handle chat completions (OpenAI-compatible)."""
        try:
            data = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)
        
        messages = data.get("messages", [])
        model = data.get("model", self.mcp_server.ollama.config.chat_model)
        stream = data.get("stream", False)
        
        if not messages:
            return web.json_response({"error": "No messages provided"}, status=400)
        
        # Extract the last user message
        user_message = None
        system_message = None
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            # Xcode sends content as array: [{"text": "...", "type": "text"}]
            # Ollama expects a plain string
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        text_parts.append(part.get("text", str(part)))
                    else:
                        text_parts.append(str(part))
                content = "\n".join(text_parts)
            elif not isinstance(content, str):
                content = str(content)
            
            if role == "user":
                user_message = content
            elif role == "system":
                system_message = content
        
        if not user_message:
            return web.json_response({"error": "No user message found"}, status=400)
        
        # Get RAG context from our vector store (non-blocking)
        context = None
        if self.mcp_server.vector_store:
            try:
                code_results = await self.mcp_server.vector_store.search_code(user_message, n_results=5)
                if code_results:
                    context_parts = []
                    for result in code_results:
                        file_path = result.metadata.get("file_path", "unknown")
                        context_parts.append(f"# From {file_path}\n{result.content}")
                    context = "\n\n".join(context_parts)
            except Exception as e:
                logger.warning(f"RAG search failed (continuing without context): {e}")
        
        # Generate response with Ollama
        try:
            if stream:
                return await self._handle_streaming_chat(request, user_message, system_message, context, model)
            else:
                response = await self.mcp_server.ollama.chat(
                    prompt=user_message,
                    system_prompt=system_message,
                    context=context,
                )
                
                return web.json_response({
                    "id": f"chatcmpl-{id(response)}",
                    "object": "chat.completion",
                    "created": int(asyncio.get_event_loop().time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": response
                            },
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(user_message.split()),
                        "completion_tokens": len(response.split()),
                        "total_tokens": len(user_message.split()) + len(response.split())
                    }
                })
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def _handle_streaming_chat(
        self,
        request: web.Request,
        user_message: str,
        system_message: Optional[str],
        context: Optional[str],
        model: str
    ) -> web.StreamResponse:
        """Handle streaming chat response."""
        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={"Content-Type": "text/event-stream"}
        )
        await response.prepare(request)
        
        # For now, just return non-streaming wrapped as SSE
        full_response = await self.mcp_server.ollama.chat(
            prompt=user_message,
            system_prompt=system_message,
            context=context,
        )
        
        # Send as single chunk
        chunk_data = {
            "id": f"chatcmpl-{id(full_response)}",
            "object": "chat.completion.chunk",
            "created": int(asyncio.get_event_loop().time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": full_response},
                    "finish_reason": None
                }
            ]
        }
        
        await response.write(f"data: {json.dumps(chunk_data)}\n\n".encode())
        
        # Send done
        done_data = {
            "id": f"chatcmpl-{id(full_response)}",
            "object": "chat.completion.chunk",
            "created": int(asyncio.get_event_loop().time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }
            ]
        }
        await response.write(f"data: {json.dumps(done_data)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        
        return response
    
    async def _handle_completions(self, request: web.Request) -> web.Response:
        """Handle text completions (OpenAI-compatible)."""
        try:
            data = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)
        
        prompt = data.get("prompt", "")
        model = data.get("model", self.mcp_server.ollama.config.chat_model)
        
        if not prompt:
            return web.json_response({"error": "No prompt provided"}, status=400)
        
        try:
            response = await self.mcp_server.ollama.generate(prompt)
            
            return web.json_response({
                "id": f"cmpl-{id(response)}",
                "object": "text_completion",
                "created": int(asyncio.get_event_loop().time()),
                "model": model,
                "choices": [
                    {
                        "text": response,
                        "index": 0,
                        "finish_reason": "stop"
                    }
                ]
            })
        except Exception as e:
            logger.error(f"Completion error: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def _handle_mcp_tools(self, request: web.Request) -> web.Response:
        """List available MCP tools."""
        return web.json_response({
            "tools": [
                {
                    "name": "generate_code",
                    "description": "Generate Swift/iOS code from a description",
                    "parameters": {"description": "string", "language": "string (optional)"}
                },
                {
                    "name": "explain_code",
                    "description": "Explain what code does",
                    "parameters": {"code": "string"}
                },
                {
                    "name": "fix_code",
                    "description": "Fix code based on error message",
                    "parameters": {"code": "string", "error": "string"}
                },
                {
                    "name": "ask_with_context",
                    "description": "Ask a question with codebase context",
                    "parameters": {"question": "string"}
                },
                {
                    "name": "search_memory",
                    "description": "Search the code memory/notebook",
                    "parameters": {"query": "string", "collection": "string (optional)"}
                }
            ]
        })
    
    async def _handle_mcp_tool_call(self, request: web.Request) -> web.Response:
        """Execute an MCP tool."""
        tool_name = request.match_info.get("tool_name")
        
        try:
            data = await request.json()
        except json.JSONDecodeError:
            data = {}
        
        try:
            result = await self.mcp_server._execute_tool(tool_name, data)
            return web.json_response({"result": result})
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def _handle_mcp_stats(self, request: web.Request) -> web.Response:
        """Get memory/vector store statistics."""
        if self.mcp_server.vector_store:
            stats = await self.mcp_server.vector_store.get_stats()
            return web.json_response(stats)
        return web.json_response({"error": "Vector store not initialized"}, status=500)
    
    async def _handle_mcp_labels(self, request: web.Request) -> web.Response:
        """Get all labels with counts."""
        if self.mcp_server.vector_store:
            labels = await self.mcp_server.vector_store.get_labels()
            return web.json_response({
                "labels": labels,
                "total_labels": len(labels),
                "total_labeled_entries": sum(labels.values()) if labels else 0,
            })
        return web.json_response({"error": "Vector store not initialized"}, status=500)
    
    async def start(self):
        """Start the HTTP server."""
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, "127.0.0.1", self.port)
        await site.start()
        
        logger.info(f"🚀 MCP Xcode HTTP Server running on http://127.0.0.1:{self.port}")
        logger.info(f"   Add to Xcode: Settings → Intelligence → Add Provider → Locally Hosted")
        logger.info(f"   Port: {self.port}")
        
        # Keep running
        while True:
            await asyncio.sleep(3600)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MCP Xcode HTTP Server")
    parser.add_argument("--port", type=int, default=1234, help="Port to listen on")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    setup_logging(config)
    
    logger.info("Initializing MCP Xcode HTTP Server...")
    
    # Create MCP server
    mcp_server = XcodeMCPServer(config)
    await mcp_server.initialize()
    
    # Create and start HTTP server
    http_server = XcodeHTTPServer(mcp_server, port=args.port)
    await http_server.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped")
