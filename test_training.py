#!/usr/bin/env python3
"""Minimal test to diagnose training issues."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

print("1. Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype=torch.float16,
    device_map="cpu",
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

print("2. Adding LoRA...")
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("3. Creating dummy input...")
# Simple test input
test_text = "Hello, this is a test."
inputs = tokenizer(test_text, return_tensors="pt", max_length=64, truncation=True)

print("4. Testing forward pass...")
with torch.no_grad():
    outputs = model(**inputs)
    print(f"   Output shape: {outputs.logits.shape}")

print("5. Testing forward pass with gradients...")
model.train()
outputs = model(**inputs)
loss = outputs.logits.mean()  # Dummy loss
print(f"   Loss: {loss.item()}")

print("6. Testing backward pass...")
loss.backward()
print("   Backward pass completed!")

print("\n✓ All tests passed! Model is working correctly.")
print("The issue might be with the SFTTrainer or dataset format.")
