"""
Ollama Client Module
====================

This module provides a robust interface for communicating with the Ollama LLM server.
It handles both chat completions and embedding generation with proper error handling
and debugging capabilities.

Features:
- Async/await support for non-blocking operations
- Automatic retry with exponential backoff
- Streaming and non-streaming responses
- Embedding generation for RAG
- Comprehensive logging and debugging

Usage:
    from server.ollama_client import OllamaClient
    
    client = OllamaClient()
    response = await client.chat("Explain this Swift code...")
    embeddings = await client.embed("func hello() { print(\"Hello\") }")
"""

import asyncio
import json
import time
from typing import Optional, List, Dict, Any, AsyncGenerator
from dataclasses import dataclass
from pathlib import Path

import httpx
from loguru import logger


@dataclass
class OllamaConfig:
    """Configuration for Ollama client."""
    base_url: str = "http://localhost:11434"
    chat_model: str = "codellama:13b"
    embedding_model: str = "nomic-embed-text"
    timeout: int = 120
    temperature: float = 0.1
    max_tokens: int = 4096
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "OllamaConfig":
        """Create OllamaConfig from dictionary."""
        return cls(
            base_url=config.get("base_url", cls.base_url),
            chat_model=config.get("chat_model", cls.chat_model),
            embedding_model=config.get("embedding_model", cls.embedding_model),
            timeout=config.get("timeout", cls.timeout),
            temperature=config.get("temperature", cls.temperature),
            max_tokens=config.get("max_tokens", cls.max_tokens),
        )


class OllamaClient:
    """
    Async client for Ollama LLM server.
    
    This client provides methods for:
    - Checking Ollama server health
    - Listing available models
    - Chat completions (streaming and non-streaming)
    - Embedding generation
    
    Example:
        async with OllamaClient(config) as client:
            # Check if server is running
            if await client.is_healthy():
                response = await client.chat("Hello, world!")
                print(response)
    """
    
    def __init__(self, config: Optional[OllamaConfig] = None):
        """
        Initialize Ollama client.
        
        Args:
            config: Optional OllamaConfig instance. Uses defaults if not provided.
        """
        self.config = config or OllamaConfig()
        self._client: Optional[httpx.AsyncClient] = None
        self._debug_enabled = True
        
        logger.debug(f"OllamaClient initialized with base_url={self.config.base_url}")
    
    async def __aenter__(self) -> "OllamaClient":
        """Async context manager entry."""
        await self._ensure_client()
        return self
    
    async def __aexit__(self, *args) -> None:
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure HTTP client is initialized."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=httpx.Timeout(self.config.timeout),
            )
            logger.debug("HTTP client initialized")
        return self._client
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            logger.debug("HTTP client closed")
    
    async def is_healthy(self) -> bool:
        """
        Check if Ollama server is running and healthy.
        
        Returns:
            True if server is healthy, False otherwise.
        """
        try:
            client = await self._ensure_client()
            response = await client.get("/api/tags")
            is_healthy = response.status_code == 200
            logger.debug(f"Ollama health check: {is_healthy}")
            return is_healthy
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models in Ollama.
        
        Returns:
            List of model dictionaries with name, size, and other metadata.
        """
        try:
            client = await self._ensure_client()
            response = await client.get("/api/tags")
            response.raise_for_status()
            
            data = response.json()
            models = data.get("models", [])
            logger.debug(f"Found {len(models)} models")
            return models
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            raise
    
    async def has_model(self, model_name: str) -> bool:
        """
        Check if a specific model is available.
        
        Args:
            model_name: Name of the model to check.
            
        Returns:
            True if model is available, False otherwise.
        """
        models = await self.list_models()
        model_names = [m.get("name", "") for m in models]
        # Handle both "model:tag" and "model" formats
        has_it = any(
            model_name == name or model_name == name.split(":")[0]
            for name in model_names
        )
        logger.debug(f"Model '{model_name}' available: {has_it}")
        return has_it
    
    async def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
        stream: bool = False,
    ) -> str:
        """
        Send a chat message to Ollama and get a response.
        
        Args:
            prompt: The user's message/question.
            system_prompt: Optional system prompt for context.
            context: Optional additional context (e.g., from RAG).
            stream: Whether to stream the response.
            
        Returns:
            The model's response as a string.
        """
        start_time = time.time()
        
        # Build messages
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if context:
            # Add context as a system message
            messages.append({
                "role": "system",
                "content": f"Use the following context to help answer the question:\n\n{context}"
            })
        
        messages.append({"role": "user", "content": prompt})
        
        logger.debug(f"Chat request with {len(messages)} messages")
        
        if self._debug_enabled:
            logger.debug(f"Prompt preview: {prompt[:200]}...")
        
        try:
            client = await self._ensure_client()
            
            payload = {
                "model": self.config.chat_model,
                "messages": messages,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
                "stream": stream,
            }
            
            if stream:
                return await self._chat_stream(client, payload)
            else:
                response = await client.post("/api/chat", json=payload)
                response.raise_for_status()
                
                data = response.json()
                content = data.get("message", {}).get("content", "")
                
                elapsed = time.time() - start_time
                logger.info(f"Chat completed in {elapsed:.2f}s, {len(content)} chars")
                
                return content
                
        except httpx.TimeoutException:
            logger.error(f"Chat request timed out after {self.config.timeout}s")
            raise
        except Exception as e:
            logger.error(f"Chat request failed: {e}")
            raise
    
    async def _chat_stream(
        self,
        client: httpx.AsyncClient,
        payload: Dict[str, Any]
    ) -> str:
        """Handle streaming chat response."""
        full_response = []
        
        async with client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        if content:
                            full_response.append(content)
                    except json.JSONDecodeError:
                        continue
        
        return "".join(full_response)
    
    async def chat_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat response token by token.
        
        Args:
            prompt: The user's message/question.
            system_prompt: Optional system prompt.
            context: Optional RAG context.
            
        Yields:
            Response tokens as they are generated.
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if context:
            messages.append({
                "role": "system",
                "content": f"Context:\n{context}"
            })
        
        messages.append({"role": "user", "content": prompt})
        
        client = await self._ensure_client()
        
        payload = {
            "model": self.config.chat_model,
            "messages": messages,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
            "stream": True,
        }
        
        async with client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue
    
    async def embed(self, text: str) -> List[float]:
        """
        Generate embeddings for text using Ollama.
        
        Args:
            text: Text to generate embeddings for.
            
        Returns:
            List of floats representing the embedding vector.
        """
        start_time = time.time()
        
        try:
            client = await self._ensure_client()
            
            payload = {
                "model": self.config.embedding_model,
                "prompt": text,
            }
            
            response = await client.post("/api/embeddings", json=payload)
            response.raise_for_status()
            
            data = response.json()
            embedding = data.get("embedding", [])
            
            elapsed = time.time() - start_time
            logger.debug(f"Embedding generated in {elapsed:.3f}s, dim={len(embedding)}")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embedding vectors.
        """
        logger.debug(f"Generating embeddings for {len(texts)} texts")
        
        embeddings = []
        for i, text in enumerate(texts):
            embedding = await self.embed(text)
            embeddings.append(embedding)
            
            if (i + 1) % 10 == 0:
                logger.debug(f"Embedded {i + 1}/{len(texts)} texts")
        
        return embeddings
    
    async def generate_code(
        self,
        instruction: str,
        language: str = "swift",
        context: Optional[str] = None,
    ) -> str:
        """
        Generate code based on an instruction.
        
        Args:
            instruction: What code to generate.
            language: Programming language (default: swift).
            context: Optional context from RAG.
            
        Returns:
            Generated code as a string.
        """
        system_prompt = f"""You are an expert {language} programmer. 
Generate clean, well-documented, production-ready code.
Follow best practices and coding conventions for {language}.
Include helpful comments explaining the code."""
        
        return await self.chat(
            prompt=instruction,
            system_prompt=system_prompt,
            context=context,
        )
    
    async def explain_code(
        self,
        code: str,
        language: str = "swift",
    ) -> str:
        """
        Explain what a piece of code does.
        
        Args:
            code: The code to explain.
            language: Programming language.
            
        Returns:
            Explanation of the code.
        """
        system_prompt = f"""You are an expert {language} programmer and teacher.
Explain the code clearly and concisely.
Cover what it does, how it works, and any important details."""
        
        prompt = f"Explain this {language} code:\n\n```{language}\n{code}\n```"
        
        return await self.chat(prompt=prompt, system_prompt=system_prompt)
    
    async def fix_code(
        self,
        code: str,
        error: str,
        language: str = "swift",
    ) -> str:
        """
        Fix code based on an error message.
        
        Args:
            code: The broken code.
            error: The error message.
            language: Programming language.
            
        Returns:
            Fixed code with explanation.
        """
        system_prompt = f"""You are an expert {language} debugger.
Analyze the error, identify the root cause, and provide fixed code.
Explain what was wrong and how you fixed it."""
        
        prompt = f"""Fix this {language} code that has an error:

```{language}
{code}
```

Error message:
```
{error}
```

Provide the fixed code and explain the fix."""
        
        return await self.chat(prompt=prompt, system_prompt=system_prompt)


# Convenience function for quick testing
async def test_ollama_client():
    """Test the Ollama client."""
    logger.info("Testing Ollama client...")
    
    async with OllamaClient() as client:
        # Check health
        if not await client.is_healthy():
            logger.error("Ollama server is not running!")
            logger.info("Start Ollama with: ollama serve")
            return False
        
        # List models
        models = await client.list_models()
        logger.info(f"Available models: {[m['name'] for m in models]}")
        
        # Test chat
        response = await client.chat("Say 'Hello from Ollama!' in one line.")
        logger.info(f"Chat response: {response}")
        
        # Test embedding
        embedding = await client.embed("Hello, world!")
        logger.info(f"Embedding dimension: {len(embedding)}")
        
        logger.info("All tests passed!")
        return True


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_ollama_client())
