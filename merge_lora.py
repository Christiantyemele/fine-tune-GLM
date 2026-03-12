#!/usr/bin/env python3
"""
Merge LoRA adapter with base model for Rust inference.

Last updated: 2026-03-12

Since Candle doesn't support LoRA merging directly, use this script
to merge the adapter weights into the base model, then load the merged
model in Rust.

Usage:
  python merge_lora.py --adapter ./neurax-lora --output ./merged_model
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument("--adapter", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--output", type=str, required=True, help="Output directory for merged model")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Base model name")
    parser.add_argument("--push-to-hub", type=str, default=None, help="Push to HuggingFace Hub with this repo name")
    args = parser.parse_args()
    
    print(f"Loading base model: {args.model}")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_fast=True,
    )
    
    print(f"Loading LoRA adapter: {args.adapter}")
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, args.adapter)
    
    print("Merging adapter weights into base model...")
    
    # Merge and unload
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to: {args.output}")
    
    # Save merged model in safetensors format (required for Candle)
    merged_model.save_pretrained(
        args.output,
        safe_serialization=True,  # Use safetensors format
    )
    tokenizer.save_pretrained(args.output)
    
    print("Done! Merged model saved in safetensors format.")
    print(f"\nTo use with Rust inference:")
    print(f"  cd neurax-inference-rust")
    print(f"  cargo run --release -- --model {args.output} --interactive")
    
    # Optionally push to HuggingFace Hub
    if args.push_to_hub:
        print(f"\nPushing to HuggingFace Hub: {args.push_to_hub}")
        merged_model.push_to_hub(args.push_to_hub, safe_serialization=True)
        tokenizer.push_to_hub(args.push_to_hub)
        print("Done!")


if __name__ == "__main__":
    main()
