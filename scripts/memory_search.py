"""
Memory Search Script
====================

Interactive script for searching and managing the vector database memory.
Use this to explore what's stored in your code memory/notebook.

Usage:
    # Interactive mode
    python scripts/memory_search.py
    
    # Direct search
    python scripts/memory_search.py --query "authentication function"
    
    # Show stats
    python scripts/memory_search.py --stats
    
    # Export memory
    python scripts/memory_search.py --export backup.json

Features:
- Interactive search mode
- Direct command-line queries
- Memory statistics
- Export/backup functionality
- Collection management
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import yaml
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.vector_store import VectorStore, VectorStoreConfig
from server.ollama_client import OllamaClient, OllamaConfig


# ============================================================================
# Memory Manager
# ============================================================================

class MemoryManager:
    """Interactive memory management for the vector store."""
    
    def __init__(self, store: VectorStore):
        self.store = store
    
    async def search(
        self,
        query: str,
        collection: str = "all",
        limit: int = 10,
    ) -> None:
        """Search memory and display results."""
        print(f"\n🔍 Searching for: '{query}'")
        print("-" * 50)
        
        results = []
        
        if collection in ["all", "code"]:
            code_results = await self.store.search_code(query, n_results=limit)
            for r in code_results:
                r.metadata["_collection"] = "code"
            results.extend(code_results)
        
        if collection in ["all", "docs"]:
            doc_results = await self.store.search_documentation(query, n_results=limit)
            for r in doc_results:
                r.metadata["_collection"] = "docs"
            results.extend(doc_results)
        
        if collection in ["all", "history"]:
            hist_results = await self.store.search_history(query, n_results=limit)
            for r in hist_results:
                r.metadata["_collection"] = "history"
            results.extend(hist_results)
        
        # Sort by score
        results.sort(key=lambda x: x.score or 0, reverse=True)
        results = results[:limit]
        
        if not results:
            print("No matching documents found.")
            return
        
        for i, doc in enumerate(results, 1):
            score = doc.score or 0
            collection_name = doc.metadata.pop("_collection", "unknown")
            file_path = doc.metadata.get("file", "N/A")
            
            print(f"\n[{i}] Score: {score:.3f} | Collection: {collection_name}")
            print(f"    File: {file_path}")
            
            # Show truncated content
            content = doc.content.strip()
            if len(content) > 300:
                content = content[:300] + "..."
            
            # Indent content
            for line in content.split("\n")[:10]:
                print(f"    │ {line}")
            
            if len(doc.content.split("\n")) > 10:
                print(f"    │ ... ({len(doc.content.split(chr(10)))} total lines)")
    
    async def show_stats(self) -> None:
        """Display memory statistics."""
        stats = await self.store.get_stats()
        
        print("\n📊 Memory Statistics")
        print("-" * 50)
        print(f"  Code documents:    {stats['code_documents']:,}")
        print(f"  Documentation:     {stats['documentation_documents']:,}")
        print(f"  Conversations:     {stats['conversation_entries']:,}")
        print(f"  Storage path:      {stats['persist_directory']}")
        
        total = (
            stats['code_documents'] +
            stats['documentation_documents'] +
            stats['conversation_entries']
        )
        print(f"\n  Total documents:   {total:,}")
    
    async def export_memory(self, output_path: str) -> None:
        """Export memory to JSON file."""
        print(f"\n📦 Exporting memory to: {output_path}")
        success = await self.store.export_to_json(output_path)
        
        if success:
            print(f"✓ Export complete!")
            # Show file size
            size = Path(output_path).stat().st_size
            print(f"  File size: {size:,} bytes")
        else:
            print("✗ Export failed")
    
    async def clear_collection(self, collection: str) -> None:
        """Clear a collection."""
        print(f"\n⚠️  Clearing collection: {collection}")
        confirm = input("Are you sure? (yes/no): ")
        
        if confirm.lower() == "yes":
            success = await self.store.clear_collection(collection)
            if success:
                print(f"✓ Collection '{collection}' cleared")
            else:
                print(f"✗ Failed to clear collection")
        else:
            print("Cancelled")
    
    async def interactive_mode(self) -> None:
        """Run interactive search mode."""
        print("\n🧠 Memory Search - Interactive Mode")
        print("=" * 50)
        print("Commands:")
        print("  search <query>    - Search all collections")
        print("  code <query>      - Search code only")
        print("  docs <query>      - Search documentation only")
        print("  history <query>   - Search conversation history")
        print("  stats             - Show statistics")
        print("  export <path>     - Export to JSON")
        print("  clear <coll>      - Clear collection (code/docs/history)")
        print("  quit              - Exit")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if not user_input:
                    continue
                
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                if command == "quit" or command == "exit":
                    print("Goodbye!")
                    break
                elif command == "search":
                    if args:
                        await self.search(args, "all")
                    else:
                        print("Usage: search <query>")
                elif command == "code":
                    if args:
                        await self.search(args, "code")
                    else:
                        print("Usage: code <query>")
                elif command == "docs":
                    if args:
                        await self.search(args, "docs")
                    else:
                        print("Usage: docs <query>")
                elif command == "history":
                    if args:
                        await self.search(args, "history")
                    else:
                        print("Usage: history <query>")
                elif command == "stats":
                    await self.show_stats()
                elif command == "export":
                    if args:
                        await self.export_memory(args)
                    else:
                        print("Usage: export <path>")
                elif command == "clear":
                    if args in ["code", "docs", "history"]:
                        await self.clear_collection(args)
                    else:
                        print("Usage: clear <code|docs|history>")
                else:
                    # Treat as search query
                    await self.search(user_input, "all")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


# ============================================================================
# Main
# ============================================================================

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Search and manage vector database memory"
    )
    parser.add_argument(
        "--query", "-q",
        help="Direct search query",
    )
    parser.add_argument(
        "--collection", "-c",
        choices=["all", "code", "docs", "history"],
        default="all",
        help="Collection to search",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=10,
        help="Maximum results",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics",
    )
    parser.add_argument(
        "--export",
        help="Export memory to JSON file",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Config file path",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger.remove()
    logger.add(
        sys.stderr,
        level="WARNING",
        format="{message}",
    )
    
    # Load config
    config_path = Path(args.config)
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    
    # Initialize Ollama client
    ollama_config = OllamaConfig.from_dict(config.get("ollama", {}))
    ollama = OllamaClient(ollama_config)
    
    # Check Ollama
    if not await ollama.is_healthy():
        print("❌ Ollama server is not running!")
        print("   Start with: ollama serve")
        sys.exit(1)
    
    # Initialize vector store
    vs_config = VectorStoreConfig.from_dict(config.get("chromadb", {}))
    store = VectorStore(
        config=vs_config,
        embedding_func=ollama.embed,
    )
    await store.initialize()
    
    # Create manager
    manager = MemoryManager(store)
    
    # Handle commands
    if args.stats:
        await manager.show_stats()
    elif args.export:
        await manager.export_memory(args.export)
    elif args.query:
        await manager.search(args.query, args.collection, args.limit)
    else:
        # Interactive mode
        await manager.interactive_mode()
    
    # Cleanup
    await ollama.close()


if __name__ == "__main__":
    asyncio.run(main())
