# Neurax Agent Rust Inference

Rust inference engine for the fine-tuned Neurax agent using [Candle](https://github.com/huggingface/candle).

## Prerequisites

1. **Train the model** (Python):
   ```bash
   cd /home/christian/fine-tune-GLM
   pip install -r requirements.txt
   ./run_training.sh
   ```

2. **Merge LoRA adapter** (Python - required for Rust):
   ```bash
   python merge_lora.py --adapter ./neurax-lora --output ./merged_model
   ```
   
   > Candle doesn't support LoRA merging directly, so we merge in Python first.

3. **Install Rust** (if not already):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

## Build

```bash
cd neurax-inference-rust

# CPU only (default)
cargo build --release

# With CUDA support
cargo build --release --features cuda
```

## Usage

### Interactive Mode
```bash
cargo run --release -- --model ../merged_model --interactive
```

### Single Prompt
```bash
cargo run --release -- --model ../merged_model --prompt "Build a CNN for image classification"
```

### From HuggingFace Hub
```bash
# If you pushed the merged model to Hub
cargo run --release -- --model your-username/neurax-qwen-merged --interactive
```

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `Qwen/Qwen2.5-3B-Instruct` | Model name or local path |
| `--adapter` | `./neurax-lora` | LoRA adapter path (requires merge) |
| `--prompt` | None | Single prompt for inference |
| `--max-tokens` | 512 | Maximum tokens to generate |
| `--temperature` | 0.7 | Sampling temperature |
| `--top-p` | 0.9 | Nucleus sampling threshold |
| `--cpu` | true | Use CPU (disable CUDA) |
| `--interactive` | true | Start interactive REPL |

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Training (Python)                │
│  PyTorch + PEFT/LoRA → neurax-lora adapter          │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│                    Merge (Python)                   │
│  merge_lora.py → merged_model (safetensors)         │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│                   Inference (Rust)                  │
│  Candle + safetensors → Fast, native inference      │
└─────────────────────────────────────────────────────┘
```

## Performance

| Platform | Model | Memory | Speed |
|----------|-------|--------|-------|
| CPU | Qwen 3B | ~6GB | ~5-10 tok/s |
| CUDA | Qwen 3B | ~4GB VRAM | ~50-100 tok/s |
| CPU | Qwen 1.5B | ~3GB | ~15-20 tok/s |

## Troubleshooting

### "Failed to load model"
- Ensure model is in safetensors format (use `merge_lora.py`)
- Check all shards are downloaded

### "CUDA out of memory"
- Use smaller model (Qwen 1.5B)
- Reduce `--max-tokens`
- Use CPU mode

### "Tokenizer not found"
- Ensure `tokenizer.json` exists in model directory
- Download from HuggingFace Hub

## Dependencies

- `candle-core` - Tensor operations
- `candle-nn` - Neural network layers
- `candle-transformers` - Pre-built model architectures
- `tokenizers` - HuggingFace tokenizers
- `safetensors` - Safe tensor format
- `hf-hub` - HuggingFace Hub integration
