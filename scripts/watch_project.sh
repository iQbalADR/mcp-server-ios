#!/bin/bash

# Default to current directory if no argument provided
PROJECT_DIR="${1:-.}"

# Ensure we're in the right directory (where this script is located's parent)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

# Source virtual environment or fail
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment not found at ./venv"
    exit 1
fi

echo "👀 Starting Automatic Project Sync (RAG Watch Mode)"
echo "Watching directory: $PROJECT_DIR"
echo "Changes will be automatically indexed into vector database."
echo "Press Ctrl+C to stop."

python scripts/ingest_project.py "$PROJECT_DIR" --watch
