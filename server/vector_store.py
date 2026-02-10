"""
Vector Store Module using LanceDB
=================================

This module provides a vector database wrapper using LanceDB for:
- Storing code snippets with embeddings
- Storing documentation and notes
- Storing conversation history
- Semantic search across all collections

LanceDB is chosen for:
- Python 3.14+ compatibility
- Serverless operation (no separate server needed)
- Simple file-based persistence
- Fast vector similarity search
"""

import os
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass
import hashlib

import lancedb
import pyarrow as pa
from loguru import logger


@dataclass
class SearchResult:
    """Represents a search result from the vector store."""
    content: str
    metadata: Dict[str, Any]
    distance: float
    id: str


class VectorStore:
    """
    Vector database wrapper using LanceDB for persistent memory.
    
    Manages three collections:
    - code: Source code snippets and functions
    - docs: Documentation and comments
    - history: Conversation history for context
    """
    
    # Schema for code collection
    CODE_SCHEMA = pa.schema([
        pa.field("id", pa.string()),
        pa.field("content", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 384)),  # Embedding dimension
        pa.field("file_path", pa.string()),
        pa.field("language", pa.string()),
        pa.field("chunk_type", pa.string()),
        pa.field("created_at", pa.string()),
        pa.field("metadata_json", pa.string()),
    ])
    
    # Schema for docs collection
    DOCS_SCHEMA = pa.schema([
        pa.field("id", pa.string()),
        pa.field("content", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 384)),
        pa.field("source", pa.string()),
        pa.field("doc_type", pa.string()),
        pa.field("created_at", pa.string()),
        pa.field("metadata_json", pa.string()),
    ])
    
    # Schema for history collection
    HISTORY_SCHEMA = pa.schema([
        pa.field("id", pa.string()),
        pa.field("content", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 384)),
        pa.field("role", pa.string()),
        pa.field("created_at", pa.string()),
        pa.field("metadata_json", pa.string()),
    ])
    
    def __init__(
        self,
        db_path: str = "./data/lancedb",
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
        embedding_dim: int = 384,
    ):
        """
        Initialize the vector store.
        
        Args:
            db_path: Path to LanceDB database directory
            embedding_fn: Function to generate embeddings from text
            embedding_dim: Dimension of embedding vectors
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.embedding_fn = embedding_fn
        self.embedding_dim = embedding_dim
        
        # Connect to LanceDB
        self.db = lancedb.connect(str(self.db_path))
        
        # Initialize tables
        self._init_tables()
        
        logger.info(f"VectorStore initialized at {self.db_path}")
    
    def _init_tables(self):
        """Initialize or open the database tables."""
        existing_tables = self.db.table_names()
        
        # Code table
        if "code" in existing_tables:
            self._code_table = self.db.open_table("code")
        else:
            self._code_table = None
            
        # Docs table
        if "docs" in existing_tables:
            self._docs_table = self.db.open_table("docs")
        else:
            self._docs_table = None
            
        # History table
        if "history" in existing_tables:
            self._history_table = self.db.open_table("history")
        else:
            self._history_table = None
    
    def _generate_id(self, content: str, prefix: str = "") -> str:
        """Generate a unique ID for content."""
        hash_val = hashlib.md5(content.encode()).hexdigest()[:12]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{prefix}{timestamp}_{hash_val}"
    
    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using the configured embedding function."""
        if self.embedding_fn is None:
            # Return zero vector if no embedding function
            logger.warning("No embedding function configured, using zero vector")
            return [0.0] * self.embedding_dim
            
        # Run in thread pool if sync function
        if asyncio.iscoroutinefunction(self.embedding_fn):
            return await self.embedding_fn(text)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.embedding_fn, text)
    
    # =========================================================================
    # Code Collection Methods
    # =========================================================================
    
    async def add_code(
        self,
        content: str,
        file_path: str = "",
        language: str = "swift",
        chunk_type: str = "code",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a code snippet to the code collection.
        
        Args:
            content: The code content
            file_path: Path to the source file
            language: Programming language
            chunk_type: Type of chunk (function, class, etc.)
            metadata: Additional metadata
            
        Returns:
            The ID of the added document
        """
        doc_id = self._generate_id(content, "code_")
        embedding = await self._get_embedding(content)
        
        data = [{
            "id": doc_id,
            "content": content,
            "vector": embedding,
            "file_path": file_path,
            "language": language,
            "chunk_type": chunk_type,
            "created_at": datetime.now().isoformat(),
            "metadata_json": json.dumps(metadata or {}),
        }]
        
        if self._code_table is None:
            self._code_table = self.db.create_table("code", data)
        else:
            self._code_table.add(data)
        
        logger.debug(f"Added code: {doc_id} from {file_path}")
        return doc_id
    
    async def add_code_batch(
        self,
        items: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Add multiple code snippets in batch.
        
        Args:
            items: List of dicts with content, file_path, language, etc.
            
        Returns:
            List of added document IDs
        """
        if not items:
            return []
            
        data = []
        doc_ids = []
        
        for item in items:
            content = item.get("content", "")
            doc_id = self._generate_id(content, "code_")
            embedding = await self._get_embedding(content)
            
            data.append({
                "id": doc_id,
                "content": content,
                "vector": embedding,
                "file_path": item.get("file_path", ""),
                "language": item.get("language", "swift"),
                "chunk_type": item.get("chunk_type", "code"),
                "created_at": datetime.now().isoformat(),
                "metadata_json": json.dumps(item.get("metadata", {})),
            })
            doc_ids.append(doc_id)
        
        if self._code_table is None:
            self._code_table = self.db.create_table("code", data)
        else:
            self._code_table.add(data)
        
        logger.info(f"Added {len(data)} code items in batch")
        return doc_ids
    
    async def search_code(
        self,
        query: str,
        n_results: int = 5,
        language: Optional[str] = None,
        file_path_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Search for similar code snippets.
        
        Args:
            query: Search query
            n_results: Number of results to return
            language: Filter by language
            file_path_filter: Filter by file path prefix
            
        Returns:
            List of SearchResult objects
        """
        if self._code_table is None:
            return []
            
        query_embedding = await self._get_embedding(query)
        
        # Build search
        search = self._code_table.search(query_embedding).limit(n_results)
        
        # Apply filters if provided
        if language:
            search = search.where(f"language = '{language}'")
        if file_path_filter:
            search = search.where(f"file_path LIKE '{file_path_filter}%'")
        
        results = search.to_list()
        
        return [
            SearchResult(
                content=r["content"],
                metadata={
                    "file_path": r["file_path"],
                    "language": r["language"],
                    "chunk_type": r["chunk_type"],
                    **json.loads(r.get("metadata_json", "{}")),
                },
                distance=r.get("_distance", 0.0),
                id=r["id"],
            )
            for r in results
        ]
    
    # =========================================================================
    # Documentation Collection Methods
    # =========================================================================
    
    async def add_doc(
        self,
        content: str,
        source: str = "",
        doc_type: str = "note",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add documentation or notes to the docs collection."""
        doc_id = self._generate_id(content, "doc_")
        embedding = await self._get_embedding(content)
        
        data = [{
            "id": doc_id,
            "content": content,
            "vector": embedding,
            "source": source,
            "doc_type": doc_type,
            "created_at": datetime.now().isoformat(),
            "metadata_json": json.dumps(metadata or {}),
        }]
        
        if self._docs_table is None:
            self._docs_table = self.db.create_table("docs", data)
        else:
            self._docs_table.add(data)
        
        logger.debug(f"Added doc: {doc_id}")
        return doc_id
    
    async def search_docs(
        self,
        query: str,
        n_results: int = 5,
        doc_type: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search for similar documentation."""
        if self._docs_table is None:
            return []
            
        query_embedding = await self._get_embedding(query)
        
        search = self._docs_table.search(query_embedding).limit(n_results)
        
        if doc_type:
            search = search.where(f"doc_type = '{doc_type}'")
        
        results = search.to_list()
        
        return [
            SearchResult(
                content=r["content"],
                metadata={
                    "source": r["source"],
                    "doc_type": r["doc_type"],
                    **json.loads(r.get("metadata_json", "{}")),
                },
                distance=r.get("_distance", 0.0),
                id=r["id"],
            )
            for r in results
        ]
    
    # =========================================================================
    # History Collection Methods
    # =========================================================================
    
    async def add_history(
        self,
        content: str,
        role: str = "user",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a conversation entry to history."""
        doc_id = self._generate_id(content, "hist_")
        embedding = await self._get_embedding(content)
        
        data = [{
            "id": doc_id,
            "content": content,
            "vector": embedding,
            "role": role,
            "created_at": datetime.now().isoformat(),
            "metadata_json": json.dumps(metadata or {}),
        }]
        
        if self._history_table is None:
            self._history_table = self.db.create_table("history", data)
        else:
            self._history_table.add(data)
        
        logger.debug(f"Added history: {doc_id} ({role})")
        return doc_id
    
    async def search_history(
        self,
        query: str,
        n_results: int = 5,
    ) -> List[SearchResult]:
        """Search conversation history for relevant context."""
        if self._history_table is None:
            return []
            
        query_embedding = await self._get_embedding(query)
        
        results = (
            self._history_table
            .search(query_embedding)
            .limit(n_results)
            .to_list()
        )
        
        return [
            SearchResult(
                content=r["content"],
                metadata={
                    "role": r["role"],
                    **json.loads(r.get("metadata_json", "{}")),
                },
                distance=r.get("_distance", 0.0),
                id=r["id"],
            )
            for r in results
        ]
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    async def search_all(
        self,
        query: str,
        n_results: int = 5,
    ) -> Dict[str, List[SearchResult]]:
        """Search across all collections."""
        code_results = await self.search_code(query, n_results)
        docs_results = await self.search_docs(query, n_results)
        history_results = await self.search_history(query, n_results)
        
        return {
            "code": code_results,
            "docs": docs_results,
            "history": history_results,
        }
    
    async def get_stats(self) -> Dict[str, int]:
        """Get statistics about the vector store."""
        stats = {
            "code_count": 0,
            "docs_count": 0,
            "history_count": 0,
        }
        
        if self._code_table is not None:
            stats["code_count"] = self._code_table.count_rows()
        if self._docs_table is not None:
            stats["docs_count"] = self._docs_table.count_rows()
        if self._history_table is not None:
            stats["history_count"] = self._history_table.count_rows()
            
        return stats
    
    async def clear_collection(self, collection: str) -> bool:
        """Clear a specific collection."""
        try:
            if collection == "code" and self._code_table is not None:
                self.db.drop_table("code")
                self._code_table = None
            elif collection == "docs" and self._docs_table is not None:
                self.db.drop_table("docs")
                self._docs_table = None
            elif collection == "history" and self._history_table is not None:
                self.db.drop_table("history")
                self._history_table = None
            else:
                return False
                
            logger.info(f"Cleared collection: {collection}")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection {collection}: {e}")
            return False
    
    async def export_to_json(self, filepath: str) -> bool:
        """Export all data to a JSON file."""
        try:
            export_data = {
                "exported_at": datetime.now().isoformat(),
                "code": [],
                "docs": [],
                "history": [],
            }
            
            if self._code_table is not None:
                export_data["code"] = self._code_table.to_pandas().to_dict("records")
            if self._docs_table is not None:
                export_data["docs"] = self._docs_table.to_pandas().to_dict("records")
            if self._history_table is not None:
                export_data["history"] = self._history_table.to_pandas().to_dict("records")
            
            # Remove vectors from export (too large)
            for collection in ["code", "docs", "history"]:
                for item in export_data[collection]:
                    if "vector" in item:
                        del item["vector"]
            
            with open(filepath, "w") as f:
                json.dump(export_data, f, indent=2, default=str)
                
            logger.info(f"Exported data to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return False


async def create_vector_store_with_ollama(
    db_path: str = "./data/lancedb",
    ollama_base_url: str = "http://localhost:11434",
    embedding_model: str = "nomic-embed-text",
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> VectorStore:
    """
    Create a VectorStore with Ollama embeddings.
    
    Args:
        db_path: Path to LanceDB database
        ollama_base_url: Ollama server URL
        embedding_model: Ollama embedding model name
        max_retries: Number of retries for failed requests
        retry_delay: Delay between retries in seconds
        
    Returns:
        Configured VectorStore instance
    """
    import httpx
    import asyncio
    
    # Rate limiting - simple semaphore to limit concurrent requests
    _semaphore = asyncio.Semaphore(3)  # Max 3 concurrent requests
    
    async def ollama_embed(text: str) -> List[float]:
        """Generate embeddings using Ollama with retry logic."""
        async with _semaphore:
            for attempt in range(max_retries):
                try:
                    async with httpx.AsyncClient(timeout=120.0) as client:
                        response = await client.post(
                            f"{ollama_base_url}/api/embeddings",
                            json={
                                "model": embedding_model,
                                "prompt": text[:8000],  # Limit text length
                            },
                        )
                        response.raise_for_status()
                        embedding = response.json().get("embedding", [])
                        return embedding
                except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError) as e:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1)
                        logger.warning(f"Embedding request failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Embedding request failed after {max_retries} attempts: {e}")
                        # Return zero vector as fallback
                        return [0.0] * 768
    
    # Get embedding dimension by testing
    try:
        test_embedding = await ollama_embed("test")
        embedding_dim = len(test_embedding)
        logger.info(f"Detected embedding dimension: {embedding_dim}")
    except Exception as e:
        logger.warning(f"Could not detect embedding dimension: {e}, using 768")
        embedding_dim = 768
    
    return VectorStore(
        db_path=db_path,
        embedding_fn=ollama_embed,
        embedding_dim=embedding_dim,
    )

