# MCP Xcode Server with Ollama + Vector Database

<div align="center">

🧠 **Local AI Coding Assistant for Xcode**

Connect your local Ollama LLM to Xcode's AI Assistant with persistent memory using LanceDB.

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Setup Guide](#detailed-setup-guide)
- [Usage](#usage)
- [Configuration](#configuration)
- [Tools Reference](#tools-reference)
- [Memory/Notebook Feature](#memorynotebook-feature)
- [Debugging](#debugging)
- [Troubleshooting](#troubleshooting)

---

## Overview

This project creates a **Model Context Protocol (MCP) server** that bridges:

| Component | Description |
|-----------|-------------|
| **Ollama** | Local LLM for code generation and understanding |
| **Xcode 26.x** | Apple's IDE with AI Assistant support |
| **LanceDB** | Vector database for persistent memory (RAG) |

### Key Features

- ✅ **Fully Local** - No cloud dependencies, complete privacy
- ✅ **Persistent Memory** - Your code "notebook" survives sessions
- ✅ **RAG-Powered** - Context-aware responses using your codebase
- ✅ **Debug Mode** - Comprehensive logging for troubleshooting
- ✅ **Easy Setup** - Automated scripts for quick installation

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Your Mac                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────────┐                                       │
│   │    Xcode 26.x       │                                       │
│   │  ┌───────────────┐  │                                       │
│   │  │ AI Assistant  │  │                                       │
│   │  └───────┬───────┘  │                                       │
│   └──────────┼──────────┘                                       │
│              │ OpenAI-compatible API                              │
│              ▼                                                   │
│   ┌─────────────────────────────────────────────┐               │
│   │         MCP Xcode Server (Python)           │               │
│   │  ┌───────────────────────────────────────┐  │               │
│   │  │            MCP Protocol               │  │               │
│   │  ├───────────────────────────────────────┤  │               │
│   │  │              Tools                    │  │               │
│   │  │  • generate_code    • explain_code    │  │               │
│   │  │  • fix_code         • ask_with_context│  │               │
│   │  │  • add_to_memory    • search_memory   │  │               │
│   │  ├───────────────────────────────────────┤  │               │
│   │  │           RAG Pipeline                │  │               │
│   │  │  • Retrieval  • Augmentation          │  │               │
│   │  └───────────────────────────────────────┘  │               │
│   └──────────┬──────────────────┬───────────────┘               │
│              │                  │                                │
│              ▼                  ▼                                │
│   ┌──────────────────┐  ┌──────────────────┐                    │
│   │     Ollama       │  │     LanceDB      │                    │
│   │  (Local LLM)     │  │  (Vector DB)     │                    │
│   │                  │  │                  │                    │
│   │  llama3.1:8b     │  │  Code Memory     │                    │
│   │  nomic-embed     │  │  Docs Memory     │                    │
│   │                  │  │  History         │                    │
│   └──────────────────┘  └──────────────────┘                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### 1. Xcode 26.x or later

```bash
# Check Xcode version
xcodebuild -version
```

> **Note**: MCP support was introduced in Xcode 26.3 (2025). Ensure you have the latest version.

### 2. Ollama

Install from [ollama.ai](https://ollama.ai/download) or via Homebrew:

```bash
brew install ollama
```

Verify installation:

```bash
ollama --version
```

### 3. Python 3.10+

```bash
# Check Python version
python3 --version

# Install if needed
brew install python@3.12
```

---

## Quick Start

### Step 1: Clone and Setup

```bash
# Navigate to project directory
cd /path/to/xcode-mcp-server

# Run setup script
chmod +x scripts/setup.sh
./scripts/setup.sh
```

The setup script will:
- Create a Python virtual environment
- Install all dependencies
- Verify Ollama is running
- Create custom iOS models or pull required models
- Create necessary directories

### Step 2: Ingest Your Project

```bash
# Ingest an Xcode project
python scripts/ingest_project.py /path/to/your/XcodeProject
```

### Step 3: Start the Server

```bash
./scripts/start_server.sh
```

### Step 4: Connect to Xcode 26+

Start the HTTP server for Xcode's "Locally Hosted" provider:

```bash
./scripts/start_http_server.sh
```

Then in Xcode:

1. Open **Xcode → Settings → Intelligence**
2. Click **"Add Model Provider..."**
3. Select **"Locally Hosted"**
4. Set Port: `1234`
5. Set Description: `MCP Xcode Server`
6. Click **"Add"**

That's it! Xcode will now use your local Ollama models with RAG context from your ingested codebase.

### Alternative: MCP Inspector Testing

For testing without Xcode:

```bash
npx @anthropic-ai/mcp-inspector ./venv/bin/python -m server.mcp_server
```

---

## Detailed Setup Guide

### Installing Ollama

1. Download from [ollama.ai/download](https://ollama.ai/download)
2. Install the application
3. Start the Ollama service:

```bash
ollama serve
```

4. Pull required models:

```bash
# Code generation model (choose one)
ollama pull codellama:13b      # Recommended for code
# OR
ollama pull llama3.2:latest    # General purpose
# OR
ollama pull deepseek-coder     # Alternative code model

# Embedding model (required for RAG)
ollama pull nomic-embed-text   # Best for embeddings
```

5. Verify models are available:

```bash
ollama list
```

### Project Setup

```bash
# 1. Navigate to project
cd xcode-mcp-server

# 2. Create virtual environment
python3 -m venv venv

# 3. Activate it
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Create directories
mkdir -p data/lancedb logs
```

### Xcode Configuration

#### Connect Xcode 26+ to the Server

1. **Start the HTTP server**
   ```bash
   ./scripts/start_http_server.sh
   ```

2. **Open Xcode Settings**
   - Menu: `Xcode > Settings...` or press `⌘,`

3. **Navigate to Intelligence**
   - Click the **Intelligence** tab in the sidebar

4. **Add Model Provider**
   - Click **"Add Model Provider..."**
   - Select **"Locally Hosted"**
   - Set Port: `1234`
   - Set Description: `MCP Xcode Server`
   - Click **"Add"**

5. **Select Models**
   - Click into your new provider
   - ✅ Enable **`ios-swift-architect:latest`** (custom iOS model)
   - ✅ Optionally enable **`llama3.1:8b`** (general purpose)
   - ❌ Do NOT enable `nomic-embed-text` (embedding only, cannot chat)

6. **Enable Xcode Tools**
   - Also enable the **Xcode Tools** (binoculars icon) in Intelligence settings
   - This gives AI access to your project structure and build diagnostics

#### Verify Connection

```bash
# Server health
curl http://localhost:1234/health

# Available models
curl http://localhost:1234/v1/models

# Vector DB stats
curl http://localhost:1234/mcp/stats
```

In Xcode: press **⌥⌘/** to open AI Assistant and type a prompt. Watch your terminal for request logs.

---

## Usage

### Ingesting Your Codebase

Before using the AI assistant effectively, ingest your project:

```bash
# Ingest a directory
python scripts/ingest_project.py /path/to/project

# Ingest with specific extensions only
python scripts/ingest_project.py /path/to/project --extensions swift m h

# Watch mode (auto-ingest on file changes)
python scripts/ingest_project.py /path/to/project --watch

# Clear existing memory and re-ingest
python scripts/ingest_project.py /path/to/project --clear
```

#### Shell Alias for Quick Access

Add these aliases to your `~/.zshrc` to easily ingest projects from anywhere:

```bash
# Add to ~/.zshrc
export MCP_SERVER_PATH="/Users/user65419/Documents/Development/AI/xcode-mcp-server"

# Ingest current directory
alias mcp-ingest='$MCP_SERVER_PATH/venv/bin/python $MCP_SERVER_PATH/scripts/ingest_project.py .'

# Ingest with watch mode
alias mcp-watch='$MCP_SERVER_PATH/venv/bin/python $MCP_SERVER_PATH/scripts/ingest_project.py . --watch'

# Search memory
alias mcp-search='$MCP_SERVER_PATH/venv/bin/python $MCP_SERVER_PATH/scripts/memory_search.py'

# Start MCP server
alias mcp-server='$MCP_SERVER_PATH/scripts/start_server.sh'
```

After adding, reload your shell:

```bash
source ~/.zshrc
```

Now you can run these from any Xcode project directory:

```bash
cd ~/MyXcodeProject

# Ingest current project
mcp-ingest

# Watch for changes during development
mcp-watch

# Search your code memory
mcp-search --query "authentication"
```

### Searching Memory

Use the interactive memory search tool:

```bash
# Interactive mode
python scripts/memory_search.py

# Direct search
python scripts/memory_search.py --query "authentication"

# Show statistics
python scripts/memory_search.py --stats

# Export memory backup
python scripts/memory_search.py --export backup.json
```

### Using in Xcode

Once connected, use the AI Assistant in Xcode normally. The MCP server provides these enhanced capabilities:

1. **Ask Questions** - Get context-aware answers using your codebase
2. **Generate Code** - Create code based on your existing patterns
3. **Explain Code** - Understand complex code sections
4. **Fix Errors** - Get error fixes with context from your project
5. **Add to Memory** - Save important snippets for future reference

---

## Configuration

Edit `config.yaml` to customize the server:

### Ollama Settings

```yaml
ollama:
  base_url: "http://localhost:11434"
  chat_model: "ios-swift-architect"   # Custom iOS model
  embedding_model: "nomic-embed-text"
  temperature: 0.1                    # Lower = more deterministic
  max_tokens: 4096
```

### LanceDB Settings

```yaml
lancedb:
  db_path: "./data/lancedb"
  max_results: 10
```

### Debug Settings

```yaml
debug:
  enabled: true
  save_requests: true
  request_log: "./logs/requests.jsonl"
  verbose_mcp: true
```

### Logging

```yaml
logging:
  level: "DEBUG"                       # DEBUG, INFO, WARNING, ERROR
  file: "./logs/mcp_server.log"
  console: true
```

---

## Tools Reference

The MCP server provides these tools to Xcode:

### `generate_code`
Generate code from natural language description using RAG context.

```
Input:
  - description: What to generate
  - language: Programming language (default: swift)

Example: "Create a function that fetches user data from an API"
```

### `explain_code`
Explain what code does in plain English.

```
Input:
  - code: The code to explain
  - language: Programming language

Example: Pass a complex Swift function for explanation
```

### `fix_code`
Fix broken code based on an error message.

```
Input:
  - code: The broken code
  - error: The error message
  - language: Programming language

Example: Pass code and compiler error for a fix
```

### `ask_with_context`
Ask any question with RAG context from your codebase.

```
Input:
  - question: Your question

Example: "How does authentication work in this project?"
```

### `add_to_memory`
Add code or notes to persistent memory.

```
Input:
  - content: Content to remember
  - type: "code", "docs", or "note"
  - file_path: Optional file reference
  - description: Optional description
```

### `search_memory`
Search through your code memory/notebook.

```
Input:
  - query: Search query
  - type: "code", "docs", or "all"
  - limit: Maximum results
```

### `get_memory_stats`
Get statistics about the vector database.

### `clear_memory`
Clear a memory collection (use with caution).

```
Input:
  - collection: "code", "docs", or "history"
```

---

## Memory/Notebook Feature

The "notebook" feature uses LanceDB to maintain persistent memory:

### How It Works

1. **Code Memory**: Stores code snippets from your projects
2. **Documentation Memory**: Stores comments and documentation
3. **Conversation History**: Stores past Q&A for context

### Memory Flow

```
┌─────────────┐     ┌───────────────┐     ┌─────────────┐
│ Your Code   │────▶│ Embeddings    │────▶│  LanceDB    │
│ Files       │     │ (Ollama)      │     │ Vector DB   │
└─────────────┘     └───────────────┘     └─────────────┘
                                                 │
                                                 │ Similarity
                                                 │ Search
                                                 ▼
┌─────────────┐     ┌───────────────┐     ┌─────────────┐
│ AI Response │◀────│ LLM + Context │◀────│ Relevant    │
│             │     │ (Ollama)      │     │ Chunks      │
└─────────────┘     └───────────────┘     └─────────────┘
```

### Best Practices

1. **Ingest your project** before asking questions
2. **Re-ingest periodically** to keep memory fresh
3. **Use watch mode** during development
4. **Add important snippets** manually for quick reference

---

## Debugging

### Enable Debug Mode

```bash
./scripts/start_server.sh --debug
```

### View Logs

```bash
# Real-time log viewing
tail -f logs/mcp_server.log

# View request/response log
cat logs/requests.jsonl | jq .
```

### Debug Request Log Format

Each line in `requests.jsonl` is a JSON object:

```json
{
  "id": 1,
  "timestamp": "2025-02-08T10:30:00",
  "type": "request",
  "method": "tools/call/generate_code",
  "params": {"description": "..."}
}
```

### Test Ollama Connection

```bash
# Test Ollama health
curl http://localhost:11434/api/tags

# Test chat
curl http://localhost:11434/api/chat -d '{
  "model": "codellama:13b",
  "messages": [{"role": "user", "content": "Hello!"}],
  "stream": false
}'
```

### Test Vector Store

```python
# In Python REPL
from server.vector_store import VectorStore, create_vector_store_with_ollama
import asyncio

async def test():
    store = await create_vector_store_with_ollama()
    stats = await store.get_stats()
    print(stats)

asyncio.run(test())
```

---

## Troubleshooting

### Ollama Not Running

**Error**: `Ollama server is not running!`

**Solution**:
```bash
ollama serve
```

### Model Not Found

**Error**: `Model 'codellama:13b' not found`

**Solution**:
```bash
ollama pull codellama:13b
```

### Python Import Errors

**Error**: `ModuleNotFoundError: No module named 'mcp'`

**Solution**:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### LanceDB Permission Error

**Error**: `Permission denied: ./data/lancedb`

**Solution**:
```bash
chmod -R 755 data/
```

### Xcode Not Connecting

1. Ensure the HTTP server is running: `./scripts/start_http_server.sh`
2. Verify the port matches: default is `1234`
3. Check Xcode > Settings > Intelligence shows your provider
4. Test the endpoint: `curl http://localhost:1234/health`

### Slow Responses

1. Use a smaller model: `ollama pull codellama:7b`
2. Reduce `max_results` in config.yaml
3. Check CPU/GPU usage with `htop`

### Memory Usage High

1. Limit ingested file size in config.yaml
2. Clear history: `python scripts/memory_search.py` then `clear history`
3. Reduce chunk size in RAG config

---

## File Structure

```
xcode-mcp-server/
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
├── README.md               # This file
│
├── server/                 # Main server package
│   ├── __init__.py
│   ├── mcp_server.py       # MCP protocol server
│   ├── ollama_client.py    # Ollama LLM client
│   └── vector_store.py     # LanceDB wrapper
│
├── scripts/                # Helper scripts
│   ├── setup.sh            # Initial setup
│   ├── start_server.sh     # Start server
│   ├── create_models.sh    # Create custom Ollama models
│   ├── ingest_project.py   # Ingest code files
│   └── memory_search.py    # Search memory
│
├── models/                 # Custom Ollama Modelfiles
│   ├── Modelfile.ios-architect
│   ├── Modelfile.swiftui
│   └── Modelfile.reviewer
│
├── data/                   # Data storage
│   └── lancedb/            # Vector database
│
└── logs/                   # Log files
    ├── mcp_server.log
    └── requests.jsonl
```

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## License

MIT License - feel free to use and modify.

---

## Acknowledgments

- [Model Context Protocol](https://modelcontextprotocol.io/) by Anthropic
- [Ollama](https://ollama.ai/) for local LLM serving
- [LanceDB](https://lancedb.github.io/lancedb/) for vector storage
- Apple for Xcode MCP integration
