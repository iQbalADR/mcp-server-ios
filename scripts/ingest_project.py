"""
Project File Ingestion Script
=============================

Ingest Xcode project files into LanceDB for RAG-powered code assistance.

Usage:
    python scripts/ingest_project.py .
    python scripts/ingest_project.py /path/to/project --watch
    python scripts/ingest_project.py . --extensions swift m h
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import yaml
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.vector_store import VectorStore, create_vector_store_with_ollama


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_EXTENSIONS = [
    ".swift", ".m", ".h", ".mm", ".cpp", ".c",
    ".py", ".js", ".ts", ".json", ".yaml", ".yml",
    ".md", ".txt"
]

DEFAULT_EXCLUDE_DIRS = [
    ".git", ".build", "build", "DerivedData", "Pods",
    "node_modules", ".swiftpm", "__pycache__", ".venv", "venv",
    ".idea", ".vscode", "xcuserdata"
]


# ============================================================================
# Code Chunking
# ============================================================================

class CodeChunker:
    """Intelligent code chunking for semantic search."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_file(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Chunk a file into meaningful pieces."""
        extension = Path(file_path).suffix.lower()
        language = self._detect_language(extension)
        
        chunks = []
        lines = content.split('\n')
        current_chunk = []
        current_start = 0
        
        for i, line in enumerate(lines):
            current_chunk.append(line)
            chunk_text = '\n'.join(current_chunk)
            
            if len(chunk_text) >= self.chunk_size:
                # Find a good break point
                if line.strip() == '' or line.strip().startswith('//'):
                    chunks.append({
                        "content": chunk_text,
                        "file_path": str(file_path),
                        "language": language,
                        "chunk_type": "code",
                        "metadata": {
                            "start_line": current_start + 1,
                            "end_line": i + 1,
                        }
                    })
                    # Keep some overlap
                    overlap_lines = max(0, len(current_chunk) - 5)
                    current_chunk = current_chunk[overlap_lines:]
                    current_start = i - len(current_chunk) + 1
        
        # Add remaining content
        if current_chunk:
            chunks.append({
                "content": '\n'.join(current_chunk),
                "file_path": str(file_path),
                "language": language,
                "chunk_type": "code",
                "metadata": {
                    "start_line": current_start + 1,
                    "end_line": len(lines),
                }
            })
        
        return chunks if chunks else [{
            "content": content,
            "file_path": str(file_path),
            "language": language,
            "chunk_type": "code",
            "metadata": {}
        }]
    
    def _detect_language(self, extension: str) -> str:
        lang_map = {
            ".swift": "swift",
            ".m": "objc",
            ".h": "objc",
            ".mm": "objc",
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".md": "markdown",
        }
        return lang_map.get(extension, "text")


# ============================================================================
# Auto-Labeling
# ============================================================================

class AutoLabeler:
    """Automatically detect labels for code based on file paths and content."""
    
    # Path-based rules: if path contains key, add these labels
    PATH_RULES = {
        "Views/": ["ui", "views"],
        "View/": ["ui", "views"],
        "UI/": ["ui"],
        "Screen/": ["ui", "screens"],
        "Screens/": ["ui", "screens"],
        "Network/": ["networking"],
        "Networking/": ["networking"],
        "API/": ["networking", "api"],
        "Service/": ["services"],
        "Services/": ["services"],
        "Model/": ["models", "data"],
        "Models/": ["models", "data"],
        "ViewModel/": ["viewmodel"],
        "ViewModels/": ["viewmodel"],
        "Test/": ["tests"],
        "Tests/": ["tests"],
        "Util/": ["utilities"],
        "Utils/": ["utilities"],
        "Utilities/": ["utilities"],
        "Helper/": ["utilities"],
        "Helpers/": ["utilities"],
        "Extension/": ["extensions"],
        "Extensions/": ["extensions"],
        "Auth/": ["auth"],
        "Authentication/": ["auth"],
        "Core/": ["core"],
        "Common/": ["core"],
        "Config/": ["config"],
        "Configuration/": ["config"],
    }
    
    # Content-based rules: if content contains key, add these labels
    CONTENT_RULES = {
        "import SwiftUI": ["swiftui", "ui"],
        "import UIKit": ["uikit", "ui"],
        "import Combine": ["combine", "async"],
        "import Foundation": ["foundation"],
        "URLSession": ["networking"],
        "URLRequest": ["networking"],
        "Codable": ["models", "data"],
        "CoreData": ["coredata", "persistence"],
        "@Observable": ["swiftui", "state"],
        "@State": ["swiftui", "state"],
        "@Published": ["combine", "state"],
        "XCTest": ["tests"],
        "@testable": ["tests"],
        "async func": ["async"],
        "await ": ["async"],
    }
    
    def __init__(self, extra_path_rules: Dict[str, List[str]] = None,
                 extra_content_rules: Dict[str, List[str]] = None):
        self.path_rules = {**self.PATH_RULES, **(extra_path_rules or {})}
        self.content_rules = {**self.CONTENT_RULES, **(extra_content_rules or {})}
    
    def detect_labels(self, file_path: str, content: str = "") -> List[str]:
        """Detect labels from file path and content."""
        labels = set()
        
        # Extension-based labels
        ext = Path(file_path).suffix.lower()
        ext_map = {
            ".swift": "swift", ".m": "objc", ".h": "objc",
            ".py": "python", ".js": "javascript", ".ts": "typescript",
            ".json": "json", ".yaml": "yaml", ".yml": "yaml",
            ".md": "docs",
        }
        if ext in ext_map:
            labels.add(ext_map[ext])
        
        # Path-based labels
        for pattern, pattern_labels in self.path_rules.items():
            if pattern in file_path:
                labels.update(pattern_labels)
        
        # Content-based labels
        for pattern, pattern_labels in self.content_rules.items():
            if pattern in content:
                labels.update(pattern_labels)
        
        return sorted(labels)


# ============================================================================
# Project Scanner
# ============================================================================

class ProjectScanner:
    """Scans project directories for files to ingest."""
    
    def __init__(
        self,
        extensions: List[str] = None,
        exclude_dirs: List[str] = None,
        max_file_size: int = 1024 * 1024,
    ):
        self.extensions = extensions or DEFAULT_EXTENSIONS
        self.exclude_dirs = set(exclude_dirs or DEFAULT_EXCLUDE_DIRS)
        self.max_file_size = max_file_size
        
        # Ensure extensions have dots
        self.extensions = [
            ext if ext.startswith('.') else f'.{ext}'
            for ext in self.extensions
        ]
    
    def scan(self, root_path: str) -> List[Path]:
        """Scan directory for files to ingest."""
        root = Path(root_path).resolve()
        files = []
        
        logger.info(f"Scanning: {root}")
        
        for path in root.rglob('*'):
            if not path.is_file():
                continue
            
            if self._is_excluded(path):
                continue
            
            if path.suffix.lower() not in self.extensions:
                continue
            
            if path.stat().st_size > self.max_file_size:
                logger.warning(f"Skipping large file: {path}")
                continue
            
            files.append(path)
        
        logger.info(f"Found {len(files)} files to ingest")
        return files
    
    def _is_excluded(self, path: Path) -> bool:
        """Check if path should be excluded."""
        for part in path.parts:
            if part in self.exclude_dirs:
                return True
        return False


# ============================================================================
# Ingestion Engine
# ============================================================================

class IngestionEngine:
    """Main ingestion engine with auto-labeling support."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        chunker: CodeChunker = None,
        scanner: ProjectScanner = None,
        labeler: AutoLabeler = None,
        manual_labels: List[str] = None,
    ):
        self.vector_store = vector_store
        self.chunker = chunker or CodeChunker()
        self.scanner = scanner or ProjectScanner()
        self.labeler = labeler
        self.manual_labels = manual_labels or []
        
        self._processed_files = 0
        self._processed_chunks = 0
    
    async def ingest_directory(
        self,
        directory: str,
        batch_size: int = 10,
    ) -> Dict[str, Any]:
        """Ingest all files in a directory."""
        files = self.scanner.scan(directory)
        
        if not files:
            logger.warning("No files found to ingest")
            return {"files": 0, "chunks": 0}
        
        logger.info(f"Ingesting {len(files)} files...")
        
        all_chunks = []
        
        for i, file_path in enumerate(files):
            try:
                chunks = self._process_file(file_path)
                all_chunks.extend(chunks)
                
                # Add to vector store in batches
                if len(all_chunks) >= batch_size * 5:
                    await self.vector_store.add_code_batch(all_chunks)
                    logger.info(f"Added {len(all_chunks)} chunks to vector store")
                    all_chunks = []
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(files)} files")
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        # Add remaining chunks
        if all_chunks:
            await self.vector_store.add_code_batch(all_chunks)
            logger.info(f"Added final {len(all_chunks)} chunks to vector store")
        
        return {
            "files": len(files),
            "chunks": self._processed_chunks,
        }
    
    async def ingest_file(self, file_path: str) -> int:
        """Ingest a single file."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        chunks = self._process_file(path)
        
        if chunks:
            await self.vector_store.add_code_batch(chunks)
        
        return len(chunks)
    
    def _process_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process a single file and return chunks."""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            return []
        
        if not content.strip():
            return []
        
        chunks = self.chunker.chunk_file(content, str(file_path))
        
        # Auto-detect labels if labeler is configured
        if self.labeler:
            file_labels = self.labeler.detect_labels(str(file_path), content)
            # Merge with any manual labels
            all_labels = sorted(set(file_labels + self.manual_labels))
            for chunk in chunks:
                chunk["labels"] = all_labels
        elif self.manual_labels:
            for chunk in chunks:
                chunk["labels"] = self.manual_labels
        
        self._processed_files += 1
        self._processed_chunks += len(chunks)
        
        return chunks


# ============================================================================
# File Watcher
# ============================================================================

class FileWatcher:
    """Watches for file changes."""
    
    def __init__(self, engine: IngestionEngine, scanner: ProjectScanner):
        self.engine = engine
        self.scanner = scanner
        self._watching = False
    
    async def watch(self, directory: str) -> None:
        """Watch directory for changes."""
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent
        except ImportError:
            logger.error("watchdog not installed. Install with: pip install watchdog")
            return
        
        class Handler(FileSystemEventHandler):
            def __init__(self, engine, scanner, loop):
                self.engine = engine
                self.scanner = scanner
                self.loop = loop
            
            def on_created(self, event):
                if isinstance(event, FileCreatedEvent):
                    self._handle_file(event.src_path)
            
            def on_modified(self, event):
                if isinstance(event, FileModifiedEvent):
                    self._handle_file(event.src_path)
            
            def _handle_file(self, path):
                file_path = Path(path)
                
                if file_path.suffix.lower() not in self.scanner.extensions:
                    return
                
                if self.scanner._is_excluded(file_path):
                    return
                
                logger.info(f"File changed: {path}")
                
                asyncio.run_coroutine_threadsafe(
                    self.engine.ingest_file(path),
                    self.loop,
                )
        
        logger.info(f"Starting file watcher on: {directory}")
        
        observer = Observer()
        handler = Handler(self.engine, self.scanner, asyncio.get_event_loop())
        observer.schedule(handler, directory, recursive=True)
        observer.start()
        
        self._watching = True
        
        try:
            while self._watching:
                await asyncio.sleep(1)
        finally:
            observer.stop()
            observer.join()
    
    def stop(self):
        """Stop watching."""
        self._watching = False


# ============================================================================
# Main
# ============================================================================

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Ingest project files into LanceDB")
    parser.add_argument("directory", help="Directory to ingest", default=".", nargs="?")
    parser.add_argument("--extensions", nargs="+", help="File extensions to process")
    parser.add_argument("--exclude-dirs", nargs="+", help="Directories to exclude")
    parser.add_argument("--watch", action="store_true", help="Watch for file changes")
    parser.add_argument("--config", help="Path to config file", default="config.yaml")
    parser.add_argument("--clear", action="store_true", help="Clear existing code collection")
    parser.add_argument("--labels", nargs="+", help="Manual labels to apply to all ingested code")
    parser.add_argument("--no-auto-labels", action="store_true", help="Disable auto-labeling")
    
    args = parser.parse_args()
    
    # Setup logging
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )
    
    # Load config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Look for config relative to script location
        script_dir = Path(__file__).parent.parent
        config_path = script_dir / args.config
    
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        logger.warning(f"Config not found at {config_path}, using defaults")
    
    # Initialize vector store with Ollama embeddings
    lancedb_config = config.get("lancedb", {})
    ollama_config = config.get("ollama", {})
    
    # Get absolute path for db
    db_path = lancedb_config.get("db_path", "./data/lancedb")
    if not Path(db_path).is_absolute():
        script_dir = Path(__file__).parent.parent
        db_path = str(script_dir / db_path)
    
    logger.info("Connecting to Ollama for embeddings...")
    
    try:
        store = await create_vector_store_with_ollama(
            db_path=db_path,
            ollama_base_url=ollama_config.get("base_url", "http://localhost:11434"),
            embedding_model=ollama_config.get("embedding_model", "nomic-embed-text"),
        )
    except Exception as e:
        logger.error(f"Failed to connect to Ollama: {e}")
        logger.error("Make sure Ollama is running: ollama serve")
        sys.exit(1)
    
    # Clear if requested
    if args.clear:
        logger.info("Clearing existing code collection...")
        await store.clear_collection("code")
    
    # Create scanner, labeler, and engine
    scanner = ProjectScanner(
        extensions=args.extensions,
        exclude_dirs=args.exclude_dirs,
    )
    
    # Setup auto-labeling
    labeler = None
    if not args.no_auto_labels:
        label_config = config.get("labels", {})
        extra_path_rules = {}
        extra_content_rules = {}
        for rule in label_config.get("path_rules", []):
            extra_path_rules[rule["pattern"]] = rule["labels"]
        for rule in label_config.get("content_rules", []):
            extra_content_rules[rule["pattern"]] = rule["labels"]
        labeler = AutoLabeler(extra_path_rules, extra_content_rules)
        logger.info("Auto-labeling enabled")
    
    engine = IngestionEngine(
        vector_store=store,
        scanner=scanner,
        labeler=labeler,
        manual_labels=args.labels or [],
    )
    
    # Ingest directory
    directory = Path(args.directory).resolve()
    logger.info(f"Ingesting directory: {directory}")
    stats = await engine.ingest_directory(str(directory))
    logger.info(f"✓ Ingestion complete: {stats['files']} files, {stats['chunks']} chunks")
    
    # Watch mode
    if args.watch:
        watcher = FileWatcher(engine, scanner)
        try:
            await watcher.watch(str(directory))
        except KeyboardInterrupt:
            logger.info("Stopping watcher...")
            watcher.stop()


if __name__ == "__main__":
    asyncio.run(main())
