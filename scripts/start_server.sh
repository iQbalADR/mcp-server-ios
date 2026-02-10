#!/bin/bash
# ============================================================================
# MCP Xcode Server Start Script
# ============================================================================
#
# This script starts the MCP server with proper environment setup.
#
# Usage:
#   ./scripts/start_server.sh           # Normal mode
#   ./scripts/start_server.sh --debug   # Debug mode with verbose logging
#   ./scripts/start_server.sh --help    # Show help
#
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Parse arguments
DEBUG_MODE=false
HELP_MODE=false

for arg in "$@"; do
    case $arg in
        --debug|-d)
            DEBUG_MODE=true
            shift
            ;;
        --help|-h)
            HELP_MODE=true
            shift
            ;;
    esac
done

if [ "$HELP_MODE" = true ]; then
    echo "MCP Xcode Server"
    echo ""
    echo "Usage: ./scripts/start_server.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --debug, -d    Enable debug mode with verbose logging"
    echo "  --help, -h     Show this help message"
    echo ""
    echo "Xcode 26+ Integration:"
    echo "  For Xcode integration, use the HTTP server instead:"
    echo "  ./scripts/start_http_server.sh"
    echo ""
    echo "  Then in Xcode:"
    echo "  1. Open Xcode > Settings > Intelligence"
    echo "  2. Click 'Add Model Provider...'"
    echo "  3. Select 'Locally Hosted'"
    echo "  4. Set Port: 1234"
    echo "  5. Click 'Add'"
    exit 0
fi

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  MCP Xcode Server${NC}"
echo -e "${BLUE}============================================${NC}"

# Check virtual environment
if [ ! -d "venv" ]; then
    echo -e "${RED}✗ Virtual environment not found${NC}"
    echo -e "Run ./scripts/setup.sh first"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Check Ollama
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${YELLOW}! Ollama server not running, starting...${NC}"
    ollama serve &> /dev/null &
    sleep 3
    
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${RED}✗ Failed to start Ollama${NC}"
        exit 1
    fi
fi
echo -e "${GREEN}✓ Ollama is running${NC}"

# Set debug environment
if [ "$DEBUG_MODE" = true ]; then
    export MCP_DEBUG=1
    export LOGURU_LEVEL=DEBUG
    echo -e "${YELLOW}Debug mode enabled${NC}"
fi

echo ""
echo -e "${GREEN}Starting MCP server (stdio mode)...${NC}"
echo -e "Press Ctrl+C to stop"
echo ""
echo -e "${BLUE}Tip: For Xcode 26+ integration, use the HTTP server instead:${NC}"
echo -e "  ./scripts/start_http_server.sh"
echo ""
echo -e "${BLUE}============================================${NC}"
echo ""

# Start server
python -m server.mcp_server
