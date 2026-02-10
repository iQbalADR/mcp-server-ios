# Ollama Model Files for iOS Development

This directory contains Modelfiles to create specialized Ollama models for iOS/Swift development.

## Quick Start

```bash
# Create all custom models at once
./scripts/create_models.sh

# Or create individually
ollama create ios-swift-architect -f models/Modelfile.ios-architect
```

## Available Models

### 1. iOS Qwen Coder (`ios-qwen-coder`) ⭐ Recommended

**Best for**: All iOS/Swift development — code-optimized with 16K context

```bash
ollama create ios-qwen-coder -f models/Modelfile.qwen-coder
```

**Base model**: `qwen2.5-coder:7b` (~4.7 GB, Q4_K_M quantization)

**Why Qwen2.5-Coder?**
- Trained specifically on code, outperforms general-purpose LLMs
- 128K max context window (we use 16K for speed/quality balance)
- Better code completion, generation, and understanding

### 2. iOS Swift Architect (`ios-swift-architect`)

**Best for**: General iOS development, architecture decisions, production code

```bash
ollama create ios-swift-architect -f models/Modelfile.ios-architect
```

**Expertise**:
- Swift 5.9+ features (async/await, actors, macros)
- Architecture patterns (MVVM, VIPER, Clean Architecture, TCA)
- Protocol-Oriented Programming
- Memory management and performance
- Full iOS SDK knowledge

### 2. SwiftUI Specialist (`swiftui-specialist`)

**Best for**: SwiftUI development, declarative UI, modern iOS interfaces

```bash
ollama create swiftui-specialist -f models/Modelfile.swiftui
```

**Expertise**:
- SwiftUI views and modifiers
- @State, @Binding, @Observable
- NavigationStack and modern navigation
- Animations and transitions
- Accessibility

### 3. iOS Code Reviewer (`ios-code-reviewer`)

**Best for**: Code reviews, finding bugs, suggesting improvements

```bash
ollama create ios-code-reviewer -f models/Modelfile.reviewer
```

**Expertise**:
- Memory safety (retain cycles, leaks)
- Thread safety (data races, main thread)
- Swift best practices
- Performance optimization
- Security issues

## Using with MCP Server

Update `config.yaml` to use your custom model:

```yaml
ollama:
  chat_model: "ios-qwen-coder"    # Recommended (Qwen2.5-Coder-7B)
  # chat_model: "ios-swift-architect"  # Alternative (llama3.1:8b based)
  embedding_model: "nomic-embed-text:v1.5"
```

## Testing Models

```bash
# Interactive chat
ollama run ios-swift-architect

# Quick test
ollama run ios-swift-architect "Create a Swift struct for a User with name and email"

# Compare models
ollama run swiftui-specialist "Create a SwiftUI view for a login form"
ollama run ios-swift-architect "Create a SwiftUI view for a login form"
```

## Customizing Models

Edit the Modelfiles to adjust:

### System Prompt
The `SYSTEM` section defines the model's personality and expertise:

```
SYSTEM """Your custom instructions here..."""
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 0.1 | Lower = more deterministic, Higher = more creative |
| `repeat_penalty` | 1.1 | Reduces repetition in output |
| `num_ctx` | 16384 | Context window size (tokens). Qwen supports up to 128K |
| `top_p` | 0.9 | Nucleus sampling threshold |
| `num_predict` | 4096 | Max tokens to generate |

### Base Model

Change `FROM` to use a different base:

```
FROM qwen2.5-coder:7b   # Recommended for code
FROM llama3.1:8b        # Good general purpose
FROM codellama:7b       # Smaller, faster
FROM codellama:13b      # Larger, better quality
FROM deepseek-coder     # Alternative code model
```

## Creating Your Own Model

1. Create a new Modelfile:

```bash
touch models/Modelfile.my-model
```

2. Add content:

```
FROM codellama:13b

SYSTEM """Your expertise description here..."""

PARAMETER temperature 0.1
PARAMETER num_ctx 8192
```

3. Create the model:

```bash
ollama create my-custom-model -f models/Modelfile.my-model
```

4. Test it:

```bash
ollama run my-custom-model
```

## Model File Reference

| File | Model Name | Purpose |
|------|------------|---------|
| `Modelfile.qwen-coder` | ios-qwen-coder | ⭐ Recommended iOS/Swift (Qwen2.5-Coder) |
| `Modelfile.ios-architect` | ios-swift-architect | General iOS/Swift (llama3.1:8b) |
| `Modelfile.swiftui` | swiftui-specialist | SwiftUI development |
| `Modelfile.reviewer` | ios-code-reviewer | Code review |
| `Modelfile.embeddings` | (documentation) | Embedding config notes |
