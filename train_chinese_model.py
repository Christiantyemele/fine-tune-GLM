#!/usr/bin/env python3
"""
Fine-tune Chinese reasoning model (Qwen/GLM) on CPU with LoRA.

Uses 4-bit quantization for memory efficiency.

Last updated: 2026-03-12

Usage:
  python train_chinese_model.py \
    --train ./dataset/neurax_train.jsonl \
    --valid ./dataset/neurax_valid.jsonl \
    --output ./neurax-qwen-lora \
    --model Qwen/Qwen2.5-3B-Instruct \
    --epochs 3
"""

import argparse
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


def load_dataset(filepath: str) -> Dataset:
    """Load JSONL dataset."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return Dataset.from_list(data)


def format_conversation(example: dict, tokenizer) -> str:
    """Format conversation for Chinese model training."""
    conversations = example.get("conversations", [])
    
    # Build message list for chat template
    messages = []
    for turn in conversations:
        role = turn.get("from", "")
        content = turn.get("value", "")
        
        if role == "system":
            messages.append({"role": "system", "content": content})
        elif role == "user":
            messages.append({"role": "user", "content": content})
        elif role == "assistant":
            messages.append({"role": "assistant", "content": content})
    
    # Apply chat template
    if hasattr(tokenizer, 'apply_chat_template'):
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
    else:
        # Fallback for models without chat template
        text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                text += f"<|system|>\n{content}<|end|>\n"
            elif role == "user":
                text += f"<|user|>\n{content}<|end|>\n"
            elif role == "assistant":
                text += f"<|assistant|>\n{content}<|end|>\n"
    
    return text


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Chinese reasoning model on CPU")
    parser.add_argument("--train", type=str, required=True, help="Training data JSONL")
    parser.add_argument("--valid", type=str, default=None, help="Validation data JSONL")
    parser.add_argument("--output", type=str, default="./output", help="Output directory")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct", 
                        help="Model name or path")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=8, help="Gradient accumulation")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--use-4bit", action="store_true", default=False, help="Use 4-bit quantization (GPU only)")
    parser.add_argument("--use-cpu", action="store_true", default=True, help="Force CPU usage")
    parser.add_argument("--use-fp16", action="store_true", default=True, help="Use float16 on CPU to save memory")
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    print(f"Device: {'CPU' if args.use_cpu else 'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Quantization config for CPU
    if args.use_4bit and not args.use_cpu:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        device_map = "auto"
    else:
        quantization_config = None
        device_map = {"": "cpu"} if args.use_cpu else "auto"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_fast=True,
    )
    
    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"
    
    # Load model with memory-efficient settings
    # Use float16 on CPU to cut memory in half
    dtype = torch.float16 if (args.use_cpu and args.use_fp16) else (torch.float32 if args.use_cpu else torch.float16)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    
    # Prepare for training
    if quantization_config:
        model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load datasets
    print(f"Loading training data: {args.train}")
    train_dataset = load_dataset(args.train)
    valid_dataset = load_dataset(args.valid) if args.valid else None
    
    print(f"Training samples: {len(train_dataset)}")
    if valid_dataset:
        print(f"Validation samples: {len(valid_dataset)}")
    
    # Format function
    def formatting_func(example):
        return {"text": format_conversation(example, tokenizer)}
    
    train_dataset = train_dataset.map(formatting_func)
    if valid_dataset:
        valid_dataset = valid_dataset.map(formatting_func)
    
    # Training arguments - Optimized for CPU speed
    # Key optimizations:
    # - No gradient checkpointing (faster, uses more RAM)
    # - No eval during training (saves time)
    # - Fewer save steps (less I/O)
    training_args = SFTConfig(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=1,  # Log every step for progress visibility
        save_steps=500,  # Save less frequently
        save_total_limit=1,  # Keep only 1 checkpoint
        eval_strategy="no",  # Skip eval during training for speed
        eval_steps=None,
        load_best_model_at_end=False,
        metric_for_best_model=None,
        greater_is_better=False,
        report_to="none",
        use_cpu=args.use_cpu,
        dataloader_num_workers=0,  # CPU training
        fp16=not args.use_cpu,
        bf16=False,
        gradient_checkpointing=False,  # DISABLE for speed (uses more RAM but 2-3x faster)
        optim="adamw_torch",  # CPU-compatible optimizer
        push_to_hub=False,
        dataset_text_field="text",
        max_length=args.max_seq_length,
    )
    
    # Create trainer with SFTConfig
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        args=training_args,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print(f"Saving model to {args.output}")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    
    # Save training config
    config = {
        "model": args.model,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "use_4bit": args.use_4bit,
        "use_cpu": args.use_cpu,
    }
    with open(Path(args.output) / "training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Training complete!")


if __name__ == "__main__":
    main()
