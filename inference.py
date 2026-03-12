#!/usr/bin/env python3
"""
Inference script for fine-tuned Neurax agent.

Last updated: 2026-03-12

Usage:
  python inference.py --adapter ./neurax-lora
  python inference.py --adapter ./neurax-lora --model Qwen/Qwen2.5-3B-Instruct
"""

import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


SYSTEM_PROMPT = """You are Neurax AI controller. You build neural network architectures using tools.

TOOLS: set_family, set_hw_config, add_node, move_node, connect, set_node_params, done

OUTPUT FORMAT: Return ONLY valid JSON:
{"assistant": "<your reasoning for this step>", "tool": {"name": "<tool_name>", "args": {...}}}

REASONING GUIDELINES:
- Analyze the user's request to understand the task, constraints, and desired properties
- Consider what blocks are available in the catalogue for the selected family
- Think about the data flow: input → processing → output
- Explain WHY you're choosing specific blocks and parameters
- Check the current graph state before making connections
- Handle fan-in restrictions: most nodes accept only ONE input

RULES:
1. INFER FAMILY FIRST: Determine best architecture family based on the task description
2. CONNECT SEQUENTIALLY: After adding input AND first layer, immediately connect input → layer
3. ADD OUTPUT: Always create an output node and connect final layer to it
4. CHECK STATE: Review existing connections before proposing new ones
5. USE CATALOGUE: Only use blocks from the provided catalogue for the selected family"""


def load_model(base_model: str, adapter_path: str, use_cpu: bool = True):
    """Load base model with LoRA adapter."""
    print(f"Loading base model: {base_model}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        use_fast=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.float32 if use_cpu else torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": "cpu"} if use_cpu else "auto",
    )
    
    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    return model, tokenizer


def generate_response(model, tokenizer, user_input: str, max_new_tokens: int = 512) -> str:
    """Generate response for a user request."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant response
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1].strip()
    elif "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()
    
    return response


def interactive_mode(model, tokenizer):
    """Interactive chat mode."""
    print("\n" + "=" * 50)
    print("Neurax Agent Interactive Mode")
    print("Type 'quit' to exit, 'test' for test prompts")
    print("=" * 50 + "\n")
    
    test_prompts = [
        "Build a CNN for image classification with 10 classes",
        "Build a transformer for text generation with 12 layers",
        "Build a MoE model with 8 experts for language modeling",
        "Build a diffusion model for image generation",
        "Build an SSM for long-range sequence modeling",
    ]
    
    while True:
        try:
            user_input = input("\nUser: ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            
            if user_input.lower() == 'test':
                print("\nTest prompts available:")
                for i, prompt in enumerate(test_prompts, 1):
                    print(f"  {i}. {prompt}")
                
                choice = input("\nSelect test (1-5): ").strip()
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(test_prompts):
                        user_input = test_prompts[idx]
                        print(f"\nRunning: {user_input}")
                    else:
                        print("Invalid selection")
                        continue
                except ValueError:
                    print("Invalid input")
                    continue
            
            if not user_input:
                continue
            
            print("\nAssistant: ", end="", flush=True)
            response = generate_response(model, tokenizer, user_input)
            print(response)
            
            # Try to parse as JSON for validation
            try:
                parsed = json.loads(response)
                if "tool" in parsed:
                    print(f"\n[Tool call: {parsed['tool']['name']}]")
            except json.JSONDecodeError:
                pass  # Not JSON, that's okay for multi-turn
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break


def batch_eval(model, tokenizer, prompts: list) -> list:
    """Evaluate model on a batch of prompts."""
    results = []
    
    for prompt in prompts:
        response = generate_response(model, tokenizer, prompt)
        results.append({
            "prompt": prompt,
            "response": response,
        })
        print(f"\nPrompt: {prompt}")
        print(f"Response: {response[:200]}...")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Inference for fine-tuned Neurax agent")
    parser.add_argument("--adapter", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="Base model name")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt to test")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max new tokens")
    parser.add_argument("--use-cpu", action="store_true", default=True, help="Use CPU")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()
    
    model, tokenizer = load_model(args.model, args.adapter, args.use_cpu)
    
    if args.interactive or not args.prompt:
        interactive_mode(model, tokenizer)
    else:
        response = generate_response(model, tokenizer, args.prompt, args.max_tokens)
        print(f"User: {args.prompt}")
        print(f"Assistant: {response}")


if __name__ == "__main__":
    main()
