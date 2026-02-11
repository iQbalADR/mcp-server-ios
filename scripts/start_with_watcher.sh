#!/bin/bash
# Start both the MCP Server (HTTP) and the Project Watcher (Auto-Sync)
# Usage: ./scripts/start_with_watcher.sh /path/to/project

PROJECT_DIR="$1"

if [ -z "$PROJECT_DIR" ]; then
    echo "Error: Project directory required."
    echo "Usage: ./scripts/start_with_watcher.sh /path/to/project"
    exit 1
fi

# Trap Ctrl+C to kill background processes
trap "kill 0" EXIT

echo "🚀 Starting Unified Xcode AI Server..."
echo "1. Starting Auto-Sync Watcher for: $PROJECT_DIR"

# Start watcher in background
# We pipe output to a log file to keep the console clean, or we can let it interleave.
# Let's keep it visible but prefixed? No, simple background.
./scripts/watch_project.sh "$PROJECT_DIR" &

# Wait a moment for watcher to initialize
sleep 2

echo "2. Starting HTTP Server..."
echo "---------------------------------------------------"

# Start HTTP server (this will block until stopped)
./scripts/start_http_server.sh
