# adi-llm-uzu-plugin

ADI plugin for local LLM inference on Apple Silicon using the Uzu engine.

## Features

- 🚀 **Apple Silicon Optimized**: ~35 tokens/sec on M2 (Llama-3.2-1B)
- 🔒 **100% Local**: No network, fully offline inference
- 📦 **Pre-built Binaries**: No build tools required for users
- ⚡ **Fast Installation**: `adi plugin install adi.llm.uzu`
- 🎯 **Simple API**: CLI and programmatic access

## Installation

**For Users** (recommended):
```bash
# Install pre-built binary from plugin registry
adi plugin install adi.llm.uzu
```

**For Developers**:
```bash
# Requirements: Metal Toolchain
xcodebuild -downloadComponent MetalToolchain

# Build plugin
cargo build --release

# Install locally
adi plugin install --local target/release/libadi_llm_uzu_plugin.dylib
```

## Usage

### Load a Model
```bash
adi llm-uzu load models/llama-3.2-1b.gguf
```

### Generate Text
```bash
adi llm-uzu generate models/llama-3.2-1b.gguf "Explain Rust ownership"
```

### List Loaded Models
```bash
adi llm-uzu list
```

### Model Information
```bash
adi llm-uzu info models/llama-3.2-1b.gguf
```

### Unload Model
```bash
adi llm-uzu unload models/llama-3.2-1b.gguf
```

## Programmatic Access

Use the inference service from other plugins or applications:

```rust
// Register service dependency in plugin.toml
[[requires]]
id = "adi.llm.inference"
version = "^1.0.0"

// Call from your code
let args = json!({
    "model_path": "models/llama-3.2-1b.gguf",
    "prompt": "Hello, world!",
    "max_tokens": 128,
    "temperature": 0.7
});

let result = service.invoke("generate", &args)?;
```

## Supported Models

Download GGUF models from:
- [Hugging Face](https://huggingface.co/models?library=gguf)
- [TheBloke](https://huggingface.co/TheBloke)

Recommended models:
- **Llama 3.2 1B/3B** - Fast, general purpose
- **Qwen 2.5 1B/3B** - Multilingual
- **Gemma 2B** - Efficient, high quality

## Requirements

- macOS with Apple Silicon (M1/M2/M3+)
- Model files in GGUF format

## Performance

| Model | Apple M2 (tokens/sec) |
|-------|----------------------|
| Llama-3.2-1B | ~35 |
| Qwen-2.5-1B | ~33 |
| Gemma-2B | ~28 |

## Why Use This Plugin?

**vs OpenAI/Anthropic:**
- ✅ Free (no API costs)
- ✅ Private (100% local)
- ✅ Fast (no network latency)
- ❌ Smaller models (less capable)

**vs lib-client-ollama:**
- ✅ Faster on Apple Silicon
- ✅ Lower overhead (no server)
- ❌ macOS only
- ❌ Fewer features

## Troubleshooting

### "Plugin not found"
Install from registry:
```bash
adi plugin install adi.llm.uzu
```

### "Model not found"
Check model file exists:
```bash
ls -lh models/llama-3.2-1b.gguf
```

### "Failed to load model"
Ensure:
- You're on Apple Silicon (M1/M2/M3)
- Model is GGUF format
- Model fits in memory

## License

MIT

## Contributing

Contributions welcome! Open an issue or PR on GitHub.

## Related Projects

- [Uzu](https://github.com/trymirai/uzu) - Inference engine
- [lib-client-uzu](https://github.com/adi-family/lib-client-uzu) - Rust client
- [ADI](https://adi.the-ihor.com) - Agent development infrastructure
