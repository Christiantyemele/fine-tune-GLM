#!/usr/bin/env python3
"""
Simple LoRA training without SFTTrainer - uses basic PyTorch loop.
"""

import argparse
import json
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

class SimpleDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    # Format conversation
                    text = self.format_conversation(item)
                    self.data.append(text)
    
    def format_conversation(self, item):
        text = ""
        for conv in item.get("conversations", []):
            role = conv.get("from", "")
            value = conv.get("value", "")
            if role == "system":
                text += f"<|im_start|>system\n{value}<|im_end|>\n"
            elif role == "user":
                text += f"<|im_start|>user\n{value}<|im_end|>\n"
            elif role == "assistant":
                text += f"<|im_start|>assistant\n{value}<|im_end|>\n"
        return text
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": enc["input_ids"].squeeze().clone()
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--output", type=str, default="./output")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    print(f"Loading dataset: {args.train}")
    dataset = SimpleDataset(args.train, tokenizer, args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=10, num_training_steps=len(dataloader) * args.epochs
    )
    
    # Training loop
    model.train()
    global_step = 0
    
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
        
        for batch in tqdm(dataloader, desc="Training"):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            
            # Forward
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Backward
            loss.backward()
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            global_step += 1
            
            if global_step % 10 == 0:
                print(f"  Step {global_step}: loss = {loss.item():.4f}")
        
        # Save checkpoint after each epoch
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir / f"checkpoint-{epoch+1}")
        tokenizer.save_pretrained(output_dir / f"checkpoint-{epoch+1}")
    
    # Save final model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n✓ Training complete! Model saved to {args.output}")


if __name__ == "__main__":
    main()
