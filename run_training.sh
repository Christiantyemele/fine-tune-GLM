#!/bin/bash
# Launch script for Neurax agent fine-tuning
# Last updated: 2026-03-12
# Usage: ./run_training.sh [model_name]

set -e

# Default model - Qwen 3B (ALREADY CACHED - 5.8GB complete)
# Do NOT change to 1.5B - it's not cached and downloads are slow
MODEL="${1:-Qwen/Qwen2.5-3B-Instruct}"

# Alternative models (may need download):
# Qwen/Qwen2.5-1.5B-Instruct - Smaller (~3GB RAM with fp16)
# THUDM/glm-4-9b-chat        - Large (~9GB RAM with fp16)

echo "=========================================="
echo "Neurax Agent Fine-tuning"
echo "=========================================="
echo "Model: $MODEL"
echo "Training data: dataset/neurax_train.jsonl (800 samples)"
echo "Validation data: dataset/neurax_valid.jsonl (200 samples)"
echo "=========================================="

python train_chinese_model.py \
    --train ./dataset/neurax_train.jsonl \
    --valid ./dataset/neurax_valid.jsonl \
    --output ./neurax-lora \
    --model "$MODEL" \
    --epochs 3 \
    --batch-size 1 \
    --gradient-accumulation 8 \
    --learning-rate 1e-5 \
    --lora-r 8 \
    --lora-alpha 16 \
    --max-seq-length 1024 \
    --use-cpu

echo "=========================================="
echo "Training complete! LoRA adapter saved to ./neurax-lora"
echo "Test with: python inference.py --adapter ./neurax-lora"
echo "=========================================="
