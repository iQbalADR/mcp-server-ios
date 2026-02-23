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
import logging
import os
import sys
import time
import re
import signal
from pathlib import Path
from typing import Optional, Dict, Any, List

from aiohttp import web
from loguru import logger

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.mcp_server import XcodeMCPServer, load_config, setup_logging
from scripts.ingest_project import FileWatcher, IngestionEngine, ProjectScanner, AutoLabeler


class XcodeHTTPServer:
    """
    HTTP server providing OpenAI-compatible API for Xcode's locally hosted models.
    
    Xcode 26+ communicates with local model providers via an OpenAI-compatible API.
    This server wraps our MCP server to provide that interface.
    """
    
    def __init__(self, mcp_server: XcodeMCPServer, port: int = 1234):
        self.mcp_server = mcp_server
        self.port = port
        self.default_system_prompt = mcp_server.config.get("system_prompt", "")
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
            logger.info(f"📨 Chat Request: stream={data.get('stream')}, model={data.get('model')}")
            # logger.debug(f"Payload: {json.dumps(data, indent=2)}")
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)
        
        messages = data.get("messages", [])
        model = data.get("model", self.mcp_server.ollama.config.chat_model)
        stream = data.get("stream", False)
        
        if not messages:
            return web.json_response({"error": "No messages provided"}, status=400)
        
        # Extract system message and user messages
        system_message_content = None
        current_messages = []
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            # Handle list content (Xcode)
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
            
            if role == "system":
                system_message_content = content
            else:
                # Keep other messages (user/assistant)
                current_messages.append({"role": role, "content": content})
        
        # Determine the "query" for RAG (last user message)
        last_user_message = ""
        for m in reversed(current_messages):
            if m["role"] == "user":
                last_user_message = m["content"]
                break
                
        if not last_user_message:
             # Fallback if no user message found in history
             return web.json_response({"error": "No user message found"}, status=400)
        
        logger.info(f"🗣️ User Query: {last_user_message[:200]}..." if len(last_user_message) > 200 else f"🗣️ User Query: {last_user_message}")

        # Prepare messages list for Agent loop
        agent_messages = []
        
        # 1. System Prompt — merge default (from config.yaml) + any Xcode-provided system message
        combined_system = self.default_system_prompt
        if system_message_content:
            combined_system = combined_system + "\n\n" + system_message_content if combined_system else system_message_content
        
        if combined_system:
            agent_messages.append({"role": "system", "content": combined_system})
        
        # RAG context will be added inside the agent loop now

        
        # 3. Append valid conversation history
        agent_messages.extend(current_messages)

        # Get tools
        tools = self.mcp_server.get_tools_for_ollama()

        # --- Agent Loop ---
        try:
            if stream:
                return await self._handle_streaming_agent(request, agent_messages, tools, model, last_user_message)
            else:
                # Non-streaming: collect all chunks and return as one message
                full_response = ""
                async for chunk in self._run_agent_loop(agent_messages, tools, last_user_message):
                    full_response += chunk
                
                return web.json_response({
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": full_response},
                        "finish_reason": "stop"
                    }],
                    "usage": {"total_tokens": len(full_response)}
                })

        except Exception as e:
            logger.error(f"Agent Loop error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_streaming_agent(self, request, agent_messages, tools, model, user_query) -> web.StreamResponse:
        """Stream the agent loop outputs as they happen."""
        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
        await response.prepare(request)
        
        chunk_id = f"chatcmpl-{int(time.time())}"
        
        async for text_chunk in self._run_agent_loop(agent_messages, tools, user_query):
            if not text_chunk:
                continue
                
            data = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {"content": text_chunk}, "finish_reason": None}]
            }
            await response.write(f"data: {json.dumps(data)}\n\n".encode())
            # Yield control to event loop to ensure flush
            await asyncio.sleep(0.01)
            
        # Send Done
        done_data = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        }
        await response.write(f"data: {json.dumps(done_data)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        return response

    async def _run_agent_loop(self, agent_messages: List[Dict], tools: List[Dict], user_query: str):
        """
        Async generator that runs the Agent Loop and yields text chunks (logs + final answer).
        """
        
        # Immediate indicator — first byte to client
        yield "> ⏳ *Processing your request...*\n\n"
        
        # 1. RAG Search (Streaming status)
        if self.mcp_server.vector_store:
            yield "> _🔍 Searching codebase for context..._\n\n"
            try:
                code_results = await self.mcp_server.vector_store.search_code(user_query, n_results=3)
                if code_results:
                    context_parts = []
                    for result in code_results:
                        file_path = result.metadata.get("file_path", "unknown")
                        context_parts.append(f"# From {file_path}\n{result.content}")
                    context_str = "\n\n".join(context_parts)
                    agent_messages.insert(1, { # Insert after system prompt (index 0)
                        "role": "system", 
                        "content": f"Relevant Code Context:\n\n{context_str}"
                    })
                    yield f"> _✅ Found {len(code_results)} relevant code snippets._\n\n"
                else:
                    yield "> _ℹ️ No relevant context found._\n\n"
            except Exception as e:
                logger.warning(f"RAG search failed: {e}")
                yield f"> _⚠️ Context search failed: {e}_\n\n"

        # Initial thought
        yield "> _🧠 Thinking..._\n\n"

        
        for turn in range(5):
            logger.info(f"🔄 Agent Turn {turn+1}")
            
            # Call Model
            response_payload = await self.mcp_server.ollama.chat_completion(
                messages=agent_messages,
                tools=tools,
                stream=False
            )
            
            # 1. Tool Call
            if isinstance(response_payload, dict) and response_payload.get("tool_calls"):
                tool_calls = response_payload["tool_calls"]
                agent_messages.append(response_payload)
                
                for tool in tool_calls:
                    fn_name = tool["function"]["name"]
                    args = tool["function"]["arguments"]
                    
                    # Log tool usage to stream
                    yield f"> 🛠️ **Executing:** `{fn_name}`\n"
                    
                    try:
                        if isinstance(args, str):
                            args = json.loads(args)
                        result = await self.mcp_server._execute_tool(fn_name, args)
                    except Exception as e:
                        result = f"Error: {e}"
                        
                    # Log result (truncated)
                    result_str = str(result)
                    preview = result_str[:100].replace("\n", " ") + ("..." if len(result_str)>100 else "")
                    yield f"> ✅ **Result:** `{preview}`\n\n"
                    
                    # Add to history
                    agent_messages.append({
                        "role": "tool",
                        "content": result_str,
                        "name": fn_name
                    })
                
                # Yield thinking again for next turn
                yield "> _🧠 Thinking..._\n\n"
                continue
            
            # 2. Text Response (or Heuristic Tool)
            elif isinstance(response_payload, str):
                # Heuristic check for tools code blocks
                tool_match = re.search(r"```(?:\w+)?\s*(\{.*?\})\s*```", response_payload, re.DOTALL)
                if tool_match:
                    try:
                        tool_json = json.loads(tool_match.group(1))
                        if "name" in tool_json and "arguments" in tool_json:
                            fn_name = tool_json["name"]
                            args = tool_json["arguments"]
                            
                            yield f"> 🛠️ **Executing:** `{fn_name}`\n"
                            result = await self.mcp_server._execute_tool(fn_name, args)
                            
                            result_str = str(result)
                            preview = result_str[:100].replace("\n", " ")
                            yield f"> ✅ **Result:** `{preview}`\n\n"
                            
                            agent_messages.append({"role": "user", "content": f"Tool Output: {result}"})
                            yield "> _🧠 Thinking..._\n\n"
                            continue
                    except:
                        pass # Fall through to text
                
                # Is it a Chain of Thought? (DeepSeek style <think>)
                # If so, we might want to collapse it or format it?
                # For now, just yield the text.
                
                yield response_payload
                return
            
            else:
                 # Wtf case
                 content = response_payload.get("content", "") if isinstance(response_payload, dict) else str(response_payload)
                 yield content
                 return

    
    async def _handle_streaming_text(self, request, text: str, model: str) -> web.StreamResponse:
        """Stream a static text/string as if it were being generated."""
        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={"Content-Type": "text/event-stream"}
        )
        await response.prepare(request)
        
        # Chunk the text
        chunk_size = 20
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            data = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}]
            }
            await response.write(f"data: {json.dumps(data)}\n\n".encode())
            await asyncio.sleep(0.01) # fast stream
            
        # Done
        done_data = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
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
                    "name": "list_files",
                    "description": "List files in a directory",
                    "parameters": {"path": "string"}
                },
                {
                    "name": "read_file",
                    "description": "Read file content",
                    "parameters": {"path": "string"}
                },
                {
                    "name": "write_to_file",
                    "description": "Write content to a file",
                    "parameters": {"path": "string", "content": "string"}
                },
                {
                    "name": "apply_patch",
                    "description": "Apply a git-style patch to a file",
                    "parameters": {"path": "string", "diff": "string"}
                },
                {
                    "name": "run_command",
                    "description": "Run a terminal command",
                    "parameters": {"command": "string"}
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


# ============================================================================
# Main
# ============================================================================

async def run_watcher(mcp_server: XcodeMCPServer, watch_dir: str):
    """Run the file watcher in the background."""
    # Wait for vector store to be ready
    while not mcp_server.vector_store:
        logger.info("Watcher waiting for Vector Store to be initialized...")
        await asyncio.sleep(1)
    
    logger.info("Vector Store ready. Starting Watcher...")
    
    scanner = ProjectScanner()
    engine = IngestionEngine(vector_store=mcp_server.vector_store, scanner=scanner)
    watcher = FileWatcher(engine, scanner)
    
    # Run watch loop
    try:
         await watcher.watch(watch_dir)
    except Exception as e:
        logger.error(f"Watcher failed: {e}")

async def start_server(port: int = 1234, config_path: str = "config.yaml", watch_dir: str = None):
    """Start the HTTP server."""
    
    # Load config
    config = load_config(config_path)
    setup_logging(config)
    
    logger.info(f"Starting Xcode MCP Server on port {port}...")
    
    # Initialize MCP Server (which initializes Vector Store)
    mcp_server = XcodeMCPServer(config)
    await mcp_server.initialize() # Ensure async initialization is called
    
    # Create and start HTTP server
    http_server = XcodeHTTPServer(mcp_server, port=port)
    
    # If watch mode enabled
    if watch_dir:
        logger.info(f"👀 Watch mode enabled for: {watch_dir}")
        # Creating a background task for the watcher
        asyncio.create_task(run_watcher(mcp_server, watch_dir))
        
    # Start the HTTP server
    runner = web.AppRunner(http_server.app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    
    logger.info(f"🚀 MCP Xcode HTTP Server running on http://127.0.0.1:{port}")
    logger.info(f"   Add to Xcode: Settings → Intelligence → Add Provider → Locally Hosted")
    logger.info(f"   Port: {port}")
    
    # Keep running
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCP Xcode HTTP Server")
    parser.add_argument("--port", type=int, default=1234, help="Port to listen on")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--watch", help="Directory to watch for changes (enables auto-sync)")
    args = parser.parse_args()
    
    try:
        asyncio.run(start_server(args.port, args.config, args.watch))
    except (KeyboardInterrupt, SystemExit):
        logger.info("Server stopped")
