#!/usr/bin/env python3
"""
Interactive script to play NYT Connections with the fine-tuned model.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path


def load_model(model_path: str, base_model: str = "Qwen/Qwen2.5-4B-Instruct"):
    """Load fine-tuned model."""
    from peft import PeftModel
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    base_model_obj = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA weights if they exist
    if Path(model_path).exists():
        model = PeftModel.from_pretrained(base_model_obj, model_path)
        model = model.merge_and_unload()
    else:
        model = base_model_obj
    
    model.eval()
    return model, tokenizer


def solve_puzzle(model, tokenizer, words: List[str]):
    """Solve a Connections puzzle."""
    words_str = ", ".join(words)
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that solves NYT Connections puzzles."
        },
        {
            "role": "user",
            "content": (
                "You are playing NYT Connections. You are given 16 words. "
                "Your task is to group them into 4 categories of 4 words each. "
                "Each group shares a common theme or connection.\n\n"
                f"Words: {words_str}"
            )
        }
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Play NYT Connections with fine-tuned model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to fine-tuned model checkpoint"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model name"
    )
    
    args = parser.parse_args()
    
    print("Loading model...")
    model, tokenizer = load_model(args.model_path, args.base_model)
    print("Model loaded! Enter 16 words separated by commas, or 'quit' to exit.\n")
    
    while True:
        user_input = input("Enter 16 words (comma-separated): ").strip()
        
        if user_input.lower() in ["quit", "exit", "q"]:
            break
        
        words = [w.strip().upper() for w in user_input.split(",")]
        
        if len(words) != 16:
            print(f"Error: Expected 16 words, got {len(words)}")
            continue
        
        print("\nSolving puzzle...")
        solution = solve_puzzle(model, tokenizer, words)
        print(f"\nSolution:\n{solution}\n")
        print("-" * 50 + "\n")


if __name__ == "__main__":
    from typing import List
    main()

