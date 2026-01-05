llm-plugin, uzu, inference, apple-silicon, local-models

## Overview
- ADI plugin for local LLM inference using Uzu engine
- Apple Silicon optimized (M1/M2/M3)
- Distributed as pre-built binary via plugin registry
- License: MIT

## Architecture
- **Plugin Type**: cdylib with ABI-stable interface
- **Services Provided**:
  - `adi.llm.uzu.cli` - CLI commands for model management
  - `adi.llm.inference` - Inference service for programmatic access
- **Model Management**: HashMap of loaded models (lazy loading)
- **Thread Safety**: Mutex-protected model storage

## CLI Commands
```bash
adi llm-uzu load <model-path>          # Load model
adi llm-uzu generate <path> <prompt>   # Generate text
adi llm-uzu list                        # List loaded models
adi llm-uzu info <path>                 # Show model info
adi llm-uzu unload <path>               # Unload model
```

## Service Interface
```json
{
  "method": "generate",
  "args": {
    "model_path": "models/llama-3.2-1b.gguf",
    "prompt": "Tell me about Rust",
    "max_tokens": 256,
    "temperature": 0.7
  }
}
```

## Build Requirements
- macOS with Apple Silicon
- Xcode Command Line Tools
- Metal Toolchain: `xcodebuild -downloadComponent MetalToolchain`
- Rust 2021 edition

## Distribution
- Pre-built binaries published to plugin registry
- Users install via: `adi plugin install adi.llm.uzu`
- No build tools required for end users
- Binary includes compiled Metal shaders

## Performance
- Apple M2: ~35 tokens/sec (Llama-3.2-1B)
- Lazy model loading (only when needed)
- Models stay loaded until explicitly unloaded
- Supports multiple concurrent models

## Integration Points
- Can be called from agent-loop for local inference
- Alternative to OpenAI/Anthropic for privacy-focused workflows
- Zero-cost local development and testing
- Works offline (no API required)

## Plugin Registry Entry
```toml
[adi.llm.uzu]
name = "ADI Uzu LLM"
version = "0.1.0"
type = "extension"
platforms = ["macos-aarch64"]
binary_url = "https://registry.adi.the-ihor.com/plugins/adi.llm.uzu/0.1.0/adi_llm_uzu_plugin.dylib"
```

## Development
```bash
# Build (requires Metal Toolchain)
cargo build --release

# Test binary name
ls -la target/release/*.dylib

# Install locally
adi plugin install --local target/release/libadi_llm_uzu_plugin.dylib
```
