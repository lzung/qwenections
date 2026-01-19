#!/usr/bin/env python3
"""
Interactive script to play NYT Connections with the fine-tuned model.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from typing import List

from scripts.prompt_utils import (
    load_prompt_templates,
    get_prompt_style,
    create_chat_messages
)


def load_model(model_path: str, base_model: str = "Qwen/Qwen2.5-3B-Instruct"):
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


def solve_puzzle_all_at_once(
    model, 
    tokenizer, 
    words: List[str],
    prompt_templates: dict
):
    """Solve a Connections puzzle using all-at-once approach."""
    messages = create_chat_messages(
        words,
        prompt_templates["system_message"],
        prompt_templates["user_message_template"]
    )
    
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


def solve_puzzle_iterative(
    model,
    tokenizer,
    words: List[str],
    prompt_templates: dict
):
    """Solve a Connections puzzle using iterative approach (one group at a time)."""
    remaining_words = words.copy()
    found_groups = []
    all_responses = []
    
    # Find 3 groups iteratively
    for step in range(3):
        remaining_count = len(remaining_words)
        
        # Create messages for this step
        messages = create_chat_messages(
            remaining_words,
            prompt_templates["system_message"],
            prompt_templates["user_message_template"],
            remaining_count=remaining_count
        )
        
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )
        
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        all_responses.append(f"Step {step + 1}: {response.strip()}")
        
        # In a real implementation, you would parse the response to extract the found group
        # For now, we'll just show the responses
        # TODO: Parse response to extract group and remove those words from remaining_words
        
        print(f"\nStep {step + 1} response: {response.strip()}")
        if step < 2:  # Not the last step
            print(f"Remaining words: {', '.join(remaining_words)}")
    
    # The 4th group is automatically the remaining words
    if len(remaining_words) == 4:
        all_responses.append(f"Step 4 (automatic): Remaining words form the final group: {', '.join(remaining_words)}")
    
    return "\n".join(all_responses)


def solve_puzzle(
    model, 
    tokenizer, 
    words: List[str],
    prompt_templates: dict = None,
    approach: str = "all_at_once"
):
    """Solve a Connections puzzle using specified approach."""
    if prompt_templates is None:
        # Fallback to default
        prompt_templates = {
            "system_message": "You are a helpful assistant that solves NYT Connections puzzles.",
            "user_message_template": (
                "You are playing NYT Connections. You are given 16 words. "
                "Your task is to group them into 4 categories of 4 words each. "
                "Each group shares a common theme or connection.\n\n"
                "Words: {words}"
            )
        }
    
    if approach == "iterative":
        return solve_puzzle_iterative(model, tokenizer, words, prompt_templates)
    else:
        return solve_puzzle_all_at_once(model, tokenizer, words, prompt_templates)


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
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Base model name"
    )
    parser.add_argument(
        "--prompt-template-file",
        type=str,
        default="./prompt_templates.yaml",
        help="Path to prompt template YAML file (default: ./prompt_templates.yaml)"
    )
    parser.add_argument(
        "--prompt-style",
        type=str,
        default="default",
        help="Prompt style to use: default, concise, detailed, iterative, all_at_once (default: default)"
    )
    parser.add_argument(
        "--approach",
        type=str,
        choices=["all_at_once", "iterative"],
        default="all_at_once",
        help="Solving approach: 'all_at_once' (find all 4 groups) or 'iterative' (find one at a time) (default: all_at_once)"
    )
    
    args = parser.parse_args()
    
    # Load prompt templates
    try:
        templates = load_prompt_templates(args.prompt_template_file)
        # Use approach-specific style if available, otherwise use specified style
        style_to_use = args.approach if args.approach in ["iterative", "all_at_once"] else args.prompt_style
        prompt_templates = get_prompt_style(templates, style_to_use)
        print(f"Loaded prompt style: {style_to_use}")
    except Exception as e:
        print(f"Warning: Could not load prompt templates: {e}")
        print("Using default prompts...")
        prompt_templates = None
    
    print("Loading model...")
    model, tokenizer = load_model(args.model_path, args.base_model)
    print(f"Model loaded! Using {args.approach} approach.")
    print("Enter 16 words separated by commas, or 'quit' to exit.\n")
    
    while True:
        user_input = input("Enter 16 words (comma-separated): ").strip()
        
        if user_input.lower() in ["quit", "exit", "q"]:
            break
        
        words = [w.strip().upper() for w in user_input.split(",")]
        
        if len(words) != 16:
            print(f"Error: Expected 16 words, got {len(words)}")
            continue
        
        print(f"\nSolving puzzle using {args.approach} approach...")
        solution = solve_puzzle(model, tokenizer, words, prompt_templates, args.approach)
        print(f"\nSolution:\n{solution}\n")
        print("-" * 50 + "\n")


if __name__ == "__main__":
    main()

