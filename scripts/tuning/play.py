#!/usr/bin/env python3
"""
Interactive script to play NYT Connections with the fine-tuned model.
"""

import torch
import json
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from typing import List, Tuple

from scripts.prompt_utils import (
    load_prompt_templates,
    get_prompt_style,
    create_chat_messages
)


def load_latest_puzzle(connections_file: str = "./connections.json") -> Tuple[List[str], dict]:
    """Load the most recent puzzle from connections.json and extract words.
    
    Returns:
        Tuple of (scrambled_words, puzzle_data)
    """
    with open(connections_file, "r") as f:
        puzzles = json.load(f)
    
    # Get the most recent puzzle (highest id)
    latest_puzzle = max(puzzles, key=lambda x: x["id"])
    
    # Extract all words from the puzzle
    words = []
    for group in latest_puzzle["answers"]:
        words.extend(group["members"])
    
    # Scramble the words
    scrambled = words.copy()
    random.shuffle(scrambled)
    
    return scrambled, latest_puzzle


def load_puzzles_by_ids(puzzle_ids: List[int], connections_file: str = "./connections.json") -> List[Tuple[List[str], dict]]:
    """Load specific puzzles by ID.
    
    Returns:
        List of tuples (scrambled_words, puzzle_data)
    """
    with open(connections_file, "r") as f:
        puzzles = json.load(f)
    
    puzzle_map = {p["id"]: p for p in puzzles}
    results = []
    
    for puzzle_id in puzzle_ids:
        if puzzle_id in puzzle_map:
            puzzle = puzzle_map[puzzle_id]
            words = []
            for group in puzzle["answers"]:
                words.extend(group["members"])
            scrambled = words.copy()
            random.shuffle(scrambled)
            results.append((scrambled, puzzle))
        else:
            print(f"Warning: Puzzle ID {puzzle_id} not found")
    
    return results


def load_n_recent_puzzles(n: int, connections_file: str = "./connections.json") -> List[Tuple[List[str], dict]]:
    """Load the N most recent puzzles.
    
    Returns:
        List of tuples (scrambled_words, puzzle_data)
    """
    with open(connections_file, "r") as f:
        puzzles = json.load(f)
    
    # Sort by ID descending and take first N
    sorted_puzzles = sorted(puzzles, key=lambda x: x["id"], reverse=True)[:n]
    results = []
    
    for puzzle in sorted_puzzles:
        words = []
        for group in puzzle["answers"]:
            words.extend(group["members"])
        scrambled = words.copy()
        random.shuffle(scrambled)
        results.append((scrambled, puzzle))
    
    return results


def load_model(model_path: str, base_model: str = "Qwen/Qwen2.5-3B-Instruct"):
    """Load fine-tuned model."""
    from peft import PeftModel
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    
    # Detect device and set appropriate dtype and device_map
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch_dtype = torch.float32 if device.type == "cpu" else torch.float16
    device_map = None if device.type == "cpu" else "auto"  # No device_map offloading on CPU
    
    base_model_obj = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        device_map=device_map,
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
        "--approach",
        type=str,
        choices=["all_at_once", "iterative"],
        default="all_at_once",
        help="Solving approach: 'all_at_once' (find all 4 groups) or 'iterative' (find one at a time) (default: all_at_once)"
    )
    parser.add_argument(
        "--from-latest",
        action="store_true",
        help="Load words from the most recent puzzle in connections.json and scramble them"
    )
    parser.add_argument(
        "--connections-file",
        type=str,
        default="./connections.json",
        help="Path to connections.json file (default: ./connections.json)"
    )
    parser.add_argument(
        "--num-puzzles",
        type=int,
        default=1,
        help="Number of recent puzzles to solve (default: 1)"
    )
    parser.add_argument(
        "--puzzle-ids",
        type=int,
        nargs="+",
        help="Specific puzzle IDs to solve (e.g., --puzzle-ids 1 5 10)"
    )
    
    args = parser.parse_args()
    
    # Load prompt templates based on approach
    try:
        templates = load_prompt_templates("./prompt_templates.yaml")
        prompt_templates = get_prompt_style(templates, args.approach)
        print(f"Loaded {args.approach} approach templates")
    except Exception as e:
        print(f"Warning: Could not load prompt templates: {e}")
        print("Using default prompts...")
        prompt_templates = None
    
    print("Loading model...")
    model, tokenizer = load_model(args.model_path, args.base_model)
    print(f"Model loaded! Using {args.approach} approach.")
    
    if args.from_latest:
        # Load and solve the latest puzzle automatically
        print(f"Loading latest puzzle from {args.connections_file}...")
        words, puzzle_data = load_latest_puzzle(args.connections_file)
        print(f"Loaded puzzle ID {puzzle_data['id']} (date: {puzzle_data['date']})")
        print(f"Words (scrambled): {', '.join(words)}\n")
        print(f"Solving puzzle using {args.approach} approach...")
        solution = solve_puzzle(model, tokenizer, words, prompt_templates, args.approach)
        print(f"\nSolution:\n{solution}\n")
        
        # Print expected answers for comparison
        print("Expected answers:")
        for group in puzzle_data["answers"]:
            print(f"  {group['group']} ({group['level']}): {', '.join(group['members'])}")
    
    elif args.puzzle_ids:
        # Solve specific puzzles by ID
        print(f"Loading puzzles with IDs: {args.puzzle_ids}")
        puzzles_to_solve = load_puzzles_by_ids(args.puzzle_ids, args.connections_file)
        
        for i, (words, puzzle_data) in enumerate(puzzles_to_solve, 1):
            print(f"\n{'='*60}")
            print(f"Puzzle {i}/{len(puzzles_to_solve)} - ID {puzzle_data['id']} (date: {puzzle_data['date']})")
            print(f"{'='*60}")
            print(f"Words (scrambled): {', '.join(words)}\n")
            print(f"Solving puzzle using {args.approach} approach...")
            solution = solve_puzzle(model, tokenizer, words, prompt_templates, args.approach)
            print(f"\nSolution:\n{solution}\n")
            
            print("Expected answers:")
            for group in puzzle_data["answers"]:
                print(f"  {group['group']} ({group['level']}): {', '.join(group['members'])}")
    
    elif args.num_puzzles > 1:
        # Solve N most recent puzzles
        print(f"Loading {args.num_puzzles} most recent puzzles...")
        puzzles_to_solve = load_n_recent_puzzles(args.num_puzzles, args.connections_file)
        
        for i, (words, puzzle_data) in enumerate(puzzles_to_solve, 1):
            print(f"\n{'='*60}")
            print(f"Puzzle {i}/{len(puzzles_to_solve)} - ID {puzzle_data['id']} (date: {puzzle_data['date']})")
            print(f"{'='*60}")
            print(f"Words (scrambled): {', '.join(words)}\n")
            print(f"Solving puzzle using {args.approach} approach...")
            solution = solve_puzzle(model, tokenizer, words, prompt_templates, args.approach)
            print(f"\nSolution:\n{solution}\n")
            
            print("Expected answers:")
            for group in puzzle_data["answers"]:
                print(f"  {group['group']} ({group['level']}): {', '.join(group['members'])}")
    
    else:
        # Single manual puzzle entry
        print("Enter 16 words separated by commas, or 'quit' to exit.\n")
        
        user_input = input("Enter 16 words (comma-separated): ").strip()
        
        if user_input.lower() not in ["quit", "exit", "q"]:
            words = [w.strip().upper() for w in user_input.split(",")]
            
            if len(words) != 16:
                print(f"Error: Expected 16 words, got {len(words)}")
            else:
                print(f"\nSolving puzzle using {args.approach} approach...")
                solution = solve_puzzle(model, tokenizer, words, prompt_templates, args.approach)
                print(f"\nSolution:\n{solution}\n")


if __name__ == "__main__":
    main()

