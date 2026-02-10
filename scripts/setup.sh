#!/bin/bash
# ============================================================================
# MCP Xcode Server Setup Script
# ============================================================================
#
# This script sets up the MCP Xcode Server environment:
# 1. Creates Python virtual environment
# 2. Installs dependencies
# 3. Verifies Ollama is running
# 4. Pulls required models
# 5. Initializes the vector database
#
# Usage:
#   chmod +x scripts/setup.sh
#   ./scripts/setup.sh
#
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  MCP Xcode Server Setup${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

cd "$PROJECT_ROOT"

# ============================================================================
# Step 1: Check Python Version
# ============================================================================
echo -e "${YELLOW}Step 1: Checking Python version...${NC}"

if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 10 ]; then
        echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"
    else
        echo -e "${RED}✗ Python 3.10+ required, found $PYTHON_VERSION${NC}"
        echo -e "Install with: brew install python@3.12"
        exit 1
    fi
else
    echo -e "${RED}✗ Python 3 not found${NC}"
    echo -e "Install with: brew install python@3.12"
    exit 1
fi

# ============================================================================
# Step 2: Create Virtual Environment
# ============================================================================
echo ""
echo -e "${YELLOW}Step 2: Setting up virtual environment...${NC}"

if [ -d "venv" ]; then
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
else
    python3 -m venv venv
    echo -e "${GREEN}✓ Created virtual environment${NC}"
fi

# Activate virtual environment
source venv/bin/activate
echo -e "${GREEN}✓ Activated virtual environment${NC}"

# ============================================================================
# Step 3: Install Dependencies
# ============================================================================
echo ""
echo -e "${YELLOW}Step 3: Installing dependencies...${NC}"

pip install --upgrade pip -q
pip install -r requirements.txt -q

echo -e "${GREEN}✓ Dependencies installed${NC}"

# ============================================================================
# Step 4: Check Ollama
# ============================================================================
echo ""
echo -e "${YELLOW}Step 4: Checking Ollama...${NC}"

if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✓ Ollama is installed${NC}"
else
    echo -e "${RED}✗ Ollama not found${NC}"
    echo -e "Install from: https://ollama.ai/download"
    exit 1
fi

# Check if Ollama is running
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Ollama server is running${NC}"
else
    echo -e "${YELLOW}! Ollama server is not running${NC}"
    echo -e "  Starting Ollama server in background..."
    ollama serve &> /dev/null &
    sleep 3
    
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Ollama server started${NC}"
    else
        echo -e "${RED}✗ Failed to start Ollama server${NC}"
        echo -e "  Try running manually: ollama serve"
        exit 1
    fi
fi

# ============================================================================
# Step 5: Setup Required Models
# ============================================================================
echo ""
echo -e "${YELLOW}Step 5: Setting up required models...${NC}"

# Read model names from config
if [ -f "config.yaml" ]; then
    CHAT_MODEL=$(grep "chat_model:" config.yaml | head -1 | awk -F'"' '{print $2}')
    EMBED_MODEL=$(grep "embedding_model:" config.yaml | head -1 | awk -F'"' '{print $2}')
else
    CHAT_MODEL="ios-swift-architect"
    EMBED_MODEL="nomic-embed-text"
fi

# List of custom models that need to be created (not pulled)
CUSTOM_MODELS=("ios-swift-architect" "swiftui-specialist" "ios-code-reviewer")

# Check if chat model is a custom model
is_custom_model() {
    local model="$1"
    for custom in "${CUSTOM_MODELS[@]}"; do
        if [[ "$model" == "$custom"* ]]; then
            return 0
        fi
    done
    return 1
}

# Handle chat model
echo -e "  Checking $CHAT_MODEL..."
if ollama list 2>/dev/null | grep -q "$CHAT_MODEL"; then
    echo -e "${GREEN}  ✓ $CHAT_MODEL is available${NC}"
else
    if is_custom_model "$CHAT_MODEL"; then
        # Custom model - needs to be created
        echo -e "${YELLOW}  ! Creating $CHAT_MODEL (custom model)...${NC}"
        
        # Determine which Modelfile to use
        if [[ "$CHAT_MODEL" == "ios-swift-architect"* ]]; then
            MODELFILE="models/Modelfile.ios-architect"
        elif [[ "$CHAT_MODEL" == "swiftui-specialist"* ]]; then
            MODELFILE="models/Modelfile.swiftui"
        elif [[ "$CHAT_MODEL" == "ios-code-reviewer"* ]]; then
            MODELFILE="models/Modelfile.reviewer"
        fi
        
        if [ -f "$MODELFILE" ]; then
            # Check base model first
            BASE_MODEL=$(grep "^FROM" "$MODELFILE" | awk '{print $2}')
            if ! ollama list 2>/dev/null | grep -q "$BASE_MODEL"; then
                echo -e "${YELLOW}    Pulling base model $BASE_MODEL...${NC}"
                ollama pull "$BASE_MODEL"
            fi
            
            ollama create "$CHAT_MODEL" -f "$MODELFILE"
            echo -e "${GREEN}  ✓ $CHAT_MODEL created${NC}"
        else
            echo -e "${RED}  ✗ Modelfile not found: $MODELFILE${NC}"
            echo -e "    Run: ./scripts/create_models.sh"
        fi
    else
        # Standard model - can be pulled
        echo -e "${YELLOW}  ! Pulling $CHAT_MODEL (this may take a while)...${NC}"
        ollama pull "$CHAT_MODEL"
        echo -e "${GREEN}  ✓ $CHAT_MODEL pulled${NC}"
    fi
fi

# Check and pull embedding model (these are always standard models)
echo -e "  Checking $EMBED_MODEL..."
if ollama list 2>/dev/null | grep -q "$EMBED_MODEL"; then
    echo -e "${GREEN}  ✓ $EMBED_MODEL is available${NC}"
else
    echo -e "${YELLOW}  ! Pulling $EMBED_MODEL...${NC}"
    ollama pull "$EMBED_MODEL"
    echo -e "${GREEN}  ✓ $EMBED_MODEL pulled${NC}"
fi

# ============================================================================
# Step 6: Create Required Directories
# ============================================================================
echo ""
echo -e "${YELLOW}Step 6: Creating directories...${NC}"

mkdir -p data/lancedb
mkdir -p logs

echo -e "${GREEN}✓ Directories created${NC}"

# ============================================================================
# Step 7: Verify Installation
# ============================================================================
echo ""
echo -e "${YELLOW}Step 7: Verifying installation...${NC}"

# Test Python imports
python3 -c "
import mcp
import lancedb
import ollama
import yaml
from loguru import logger
print('All imports successful!')
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ All dependencies verified${NC}"
else
    echo -e "${RED}✗ Dependency verification failed${NC}"
    exit 1
fi

# ============================================================================
# Done!
# ============================================================================
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo -e "Next steps:"
echo -e ""
echo -e "1. ${BLUE}Ingest your project:${NC}"
echo -e "   python scripts/ingest_project.py /path/to/your/xcode/project"
echo -e ""
echo -e "2. ${BLUE}Start the HTTP server for Xcode:${NC}"
echo -e "   ./scripts/start_http_server.sh"
echo -e ""
echo -e "3. ${BLUE}Connect Xcode 26+:${NC}"
echo -e "   - Open Xcode > Settings > Intelligence"
echo -e "   - Click 'Add Model Provider...'"
echo -e "   - Select 'Locally Hosted'"
echo -e "   - Set Port: 1234"
echo -e "   - Set Description: MCP Xcode Server"
echo -e "   - Click 'Add'"
echo ""

