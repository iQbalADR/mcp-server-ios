#!/bin/bash
# Start MCP Xcode HTTP Server for Xcode 26+ Integration
#
# This starts the HTTP server that Xcode can connect to as a "Locally Hosted" model provider.
#
# Usage:
#              # Default port 1234
#   ./scripts/start_http_server.sh 8080      # Custom port

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Defaults
PORT=1234
WATCH_DIR=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --port) PORT="$2"; shift ;;
        --watch) WATCH_DIR="$2"; shift ;;
        *) 
          # Fallback for positional args (legacy support)
          if [[ -z "$WATCH_DIR" && "$1" =~ ^[0-9]+$ ]]; then
            PORT="$1"
          elif [[ -d "$1" ]]; then
            WATCH_DIR="$1"
          fi
          ;;
    esac
    shift
done

cd "$PROJECT_DIR"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo -e "${YELLOW}Virtual environment not found. Run ./scripts/setup.sh first.${NC}"
    exit 1
fi

# Kill any existing process on the port
if lsof -ti:$PORT > /dev/null 2>&1; then
    echo -e "${YELLOW}Killing existing process on port ${PORT}...${NC}"
    lsof -ti:$PORT | xargs kill -9 2>/dev/null
    sleep 1
fi

# Check Ollama
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠ Ollama is not running. Starting...${NC}"
    ollama serve &
    sleep 3
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}                    🧠 MCP Xcode HTTP Server                               ${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${GREEN}Starting HTTP server on port ${PORT}...${NC}"
echo ""
echo -e "  ${YELLOW}To add to Xcode:${NC}"
echo "    1. Open Xcode → Settings → Intelligence"
echo "    2. Click 'Add Model Provider...'"
echo "    3. Select 'Locally Hosted'"
echo -e "    4. Set Port: ${GREEN}${PORT}${NC}"
echo "    5. Set Description: MCP Xcode Server"
echo "    6. Click 'Add'"
echo ""
if [ ! -z "$WATCH_DIR" ]; then
    echo -e "${YELLOW}👀 Auto-Sync Enabled for: ${GREEN}${WATCH_DIR}${NC}"
fi
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Start server
if [ -z "$WATCH_DIR" ]; then
    python -m server.http_server --port "$PORT"
else
    python -m server.http_server --port "$PORT" --watch "$WATCH_DIR"
fi
