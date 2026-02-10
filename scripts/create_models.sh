#!/bin/bash
# ============================================================================
# Create Custom Ollama Models for iOS Development
# ============================================================================
#
# This script creates specialized Ollama models for iOS/Swift development.
# Each model has a custom system prompt that specializes it for specific tasks.
#
# Usage:
#   chmod +x scripts/create_models.sh
#   ./scripts/create_models.sh
#
# Available models after creation:
#   - ios-qwen-coder: Qwen2.5-Coder-7B iOS architect (RECOMMENDED)
#   - ios-swift-architect: General iOS architecture and Swift coding
#   - swiftui-specialist: SwiftUI-focused development
#   - ios-code-reviewer: Code review and improvement suggestions
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
MODELS_DIR="$PROJECT_ROOT/models"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Creating Custom iOS Ollama Models${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo -e "${RED}✗ Ollama not found${NC}"
    echo -e "Install from: https://ollama.ai/download"
    exit 1
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${YELLOW}! Starting Ollama server...${NC}"
    ollama serve &> /dev/null &
    sleep 3
fi

# Check base model exists
echo -e "${YELLOW}Checking base model...${NC}"
if ! ollama list 2>/dev/null | grep -q "llama3.1:8b"; then
    echo -e "${YELLOW}! Base model llama3.1:8b not found, pulling...${NC}"
    ollama pull llama3.1:8b
fi
echo -e "${GREEN}✓ Base model llama3.1:8b ready${NC}"

# Check/pull Qwen2.5-Coder base model
echo -e "${YELLOW}Checking Qwen2.5-Coder base model...${NC}"
if ! ollama list 2>/dev/null | grep -q "qwen2.5-coder:7b"; then
    echo -e "${YELLOW}! Qwen2.5-Coder not found, pulling (~4.7GB)...${NC}"
    ollama pull qwen2.5-coder:7b
fi
echo -e "${GREEN}✓ Base model qwen2.5-coder:7b ready${NC}"

# Create iOS Qwen Coder model (RECOMMENDED)
echo ""
echo -e "${YELLOW}Creating ios-qwen-coder model (recommended)...${NC}"
if [ -f "$MODELS_DIR/Modelfile.qwen-coder" ]; then
    ollama create ios-qwen-coder -f "$MODELS_DIR/Modelfile.qwen-coder"
    echo -e "${GREEN}✓ ios-qwen-coder created${NC}"
else
    echo -e "${RED}✗ Modelfile not found: $MODELS_DIR/Modelfile.qwen-coder${NC}"
fi

# Create iOS Swift Architect model
echo ""
echo -e "${YELLOW}Creating ios-swift-architect model...${NC}"
if [ -f "$MODELS_DIR/Modelfile.ios-architect" ]; then
    ollama create ios-swift-architect -f "$MODELS_DIR/Modelfile.ios-architect"
    echo -e "${GREEN}✓ ios-swift-architect created${NC}"
else
    echo -e "${RED}✗ Modelfile not found: $MODELS_DIR/Modelfile.ios-architect${NC}"
fi

# Create SwiftUI Specialist model
echo ""
echo -e "${YELLOW}Creating swiftui-specialist model...${NC}"
if [ -f "$MODELS_DIR/Modelfile.swiftui" ]; then
    ollama create swiftui-specialist -f "$MODELS_DIR/Modelfile.swiftui"
    echo -e "${GREEN}✓ swiftui-specialist created${NC}"
else
    echo -e "${RED}✗ Modelfile not found: $MODELS_DIR/Modelfile.swiftui${NC}"
fi

# Create iOS Code Reviewer model
echo ""
echo -e "${YELLOW}Creating ios-code-reviewer model...${NC}"
if [ -f "$MODELS_DIR/Modelfile.reviewer" ]; then
    ollama create ios-code-reviewer -f "$MODELS_DIR/Modelfile.reviewer"
    echo -e "${GREEN}✓ ios-code-reviewer created${NC}"
else
    echo -e "${RED}✗ Modelfile not found: $MODELS_DIR/Modelfile.reviewer${NC}"
fi

# List created models
echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}  Custom Models Created!${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo "Available models:"
ollama list | grep -E "(ios-qwen-coder|ios-swift-architect|swiftui-specialist|ios-code-reviewer)" || true

echo ""
echo -e "To use a model with MCP server, update ${BLUE}config.yaml${NC}:"
echo ""
echo -e "  ollama:"
echo -e "    chat_model: \"${GREEN}ios-qwen-coder${NC}\"  # Recommended"
echo ""
echo -e "Or test a model directly:"
echo -e "  ollama run ios-swift-architect"
echo ""
