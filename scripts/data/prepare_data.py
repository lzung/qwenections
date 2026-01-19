#!/usr/bin/env python3
"""
Prepare collected puzzle data for fine-tuning.

Converts puzzle data into instruction-following format suitable for Qwen.
"""

import json
import random
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from scripts.prompt_utils import (
    load_prompt_templates,
    get_prompt_style,
    format_instruction,
    format_output_group
)


def convert_nyt_format_to_puzzle(nyt_puzzle: Dict) -> Dict:
    """
    Convert NYT Connections API format to internal puzzle format.
    
    NYT format: {id, date, answers: [{level, group, members}]}
    Internal format: {words: [], groups: [{words: [], category: str, difficulty: str}]}
    """
    # Map level to difficulty (0=yellow, 1=green, 2=blue, 3=purple)
    level_to_difficulty = {0: "yellow", 1: "green", 2: "blue", 3: "purple", -1: "unknown"}
    
    # Collect all words and groups
    all_words = []
    groups = []
    
    for answer in nyt_puzzle["answers"]:
        members = answer["members"]
        if not members or len(members) != 4:
            # Skip invalid groups
            continue
            
        all_words.extend(members)
        
        level = answer.get("level", -1)
        difficulty = level_to_difficulty.get(level, "unknown")
        
        groups.append({
            "words": members,
            "category": answer.get("group", "Unknown Category"),
            "difficulty": difficulty
        })
    
    # Sort groups by difficulty level for consistency
    difficulty_order = {"yellow": 0, "green": 1, "blue": 2, "purple": 3, "unknown": 4}
    groups.sort(key=lambda g: difficulty_order.get(g["difficulty"], 4))
    
    return {
        "words": all_words,
        "groups": groups,
        "date": nyt_puzzle.get("date", "unknown"),
        "id": nyt_puzzle.get("id", None)
    }


def format_puzzle_as_instruction(puzzle: Dict, prompt_templates: Dict) -> Dict:
    """
    Format a puzzle into instruction-following format using prompt templates.
    
    Training data always uses the same format - the model always sees all 16 words
    and is expected to identify all 4 groups. The iterative vs all-at-once distinction
    is only for inference-time prompting and parsing.
    
    Args:
        puzzle: Puzzle dictionary
        prompt_templates: Prompt template dictionary
    
    Returns:
        Dictionary with instruction, input, and output
    """
    words = puzzle["words"]
    groups = puzzle["groups"]
    
    # Shuffle words to avoid position bias
    shuffled_words = words.copy()
    random.shuffle(shuffled_words)
    
    return format_all_at_once_puzzle(puzzle, prompt_templates, shuffled_words)


def format_all_at_once_puzzle(puzzle: Dict, prompt_templates: Dict, shuffled_words: List[str]) -> Dict:
    """Format puzzle for all-at-once approach (all 4 groups at once)."""
    words = puzzle["words"]
    groups = puzzle["groups"]
    
    # Format instruction using template
    instruction_template = prompt_templates["instruction_template"]
    instruction = format_instruction(instruction_template, shuffled_words)
    
    # Format output using template
    output_template = prompt_templates["output_group_template"]
    group_descriptions = []
    for group in groups:
        words_in_group = group["words"]
        category = group["category"]
        difficulty = group.get("difficulty", "unknown")
        group_desc = format_output_group(
            output_template,
            words_in_group,
            category,
            difficulty
        )
        group_descriptions.append(group_desc)
    
    output = "\n".join(group_descriptions)
    
    return {
        "instruction": instruction,
        "input": ", ".join(shuffled_words),
        "output": output,
        "metadata": {
            "date": puzzle.get("date", "unknown"),
            "id": puzzle.get("id", None),
            "original_words": words,
            "groups": groups
        }
    }


# Note: format_iterative_puzzle removed - training data always uses consistent format
# Iterative approach is only used at inference time in play.py and evaluate.py


def create_chat_format(example: Dict, prompt_templates: Dict) -> Dict:
    """
    Convert to Qwen chat format using prompt templates.
    Qwen2.5 uses a specific chat template.
    
    Training data always uses consistent format - combine instruction and input.
    """
    # Combine instruction and input
    user_content = f"{example['instruction']}\n\nWords: {example['input']}"
    
    messages = [
        {
            "role": "system",
            "content": prompt_templates["system_message"]
        },
        {
            "role": "user",
            "content": user_content
        },
        {
            "role": "assistant",
            "content": example["output"]
        }
    ]
    
    return {
        "messages": messages,
        "metadata": example.get("metadata", {})
    }


def load_puzzles_from_json(json_file: str) -> List[Dict]:
    """Load puzzles from connections.json file."""
    with open(json_file, "r") as f:
        nyt_puzzles = json.load(f)
    
    # Convert NYT format to internal format
    puzzles = []
    for nyt_puzzle in nyt_puzzles:
        puzzle = convert_nyt_format_to_puzzle(nyt_puzzle)
        puzzles.append(puzzle)
    
    return puzzles


def load_puzzles(data_dir: Path) -> List[Dict]:
    """Load all puzzle JSON files from a directory (legacy support)."""
    puzzles = []
    for json_file in data_dir.glob("*.json"):
        # Skip connections.json if it's in the directory
        if json_file.name == "connections.json":
            continue
        with open(json_file, "r") as f:
            puzzle = json.load(f)
            puzzles.append(puzzle)
    return puzzles


def prepare_training_data(
    connections_file: str = "./connections.json",
    output_dir: str = "./data/processed",
    train_split: float = 0.9,
    prompt_template_file: str = "./prompt_templates.yaml",
    prompt_style: str = "default"
):
    """
    Prepare training data from connections.json
    
    Model always sees all 16 words and is expected to identify all 4 groups during training.
    
    Args:
        connections_file: Path to connections.json file (primary source)
        output_dir: Directory to save processed data
        train_split: Fraction of data to use for training (rest for eval)
        prompt_template_file: Path to prompt template YAML file
        prompt_style: Prompt style to use (default, concise, detailed, etc.)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    puzzles = []
    
    # Load from connections.json (primary source)
    connections_path = Path(connections_file)
    if connections_path.exists():
        print(f"Loading puzzles from {connections_file}...")
        puzzles = load_puzzles_from_json(str(connections_path))
        print(f"Loaded {len(puzzles)} puzzles from {connections_file}")
    else:
        print(f"Warning: {connections_file} not found. Skipping...")
    
    if len(puzzles) == 0:
        print("Warning: No puzzles found. Please ensure connections.json exists or collect data first.")
        return
    
    # Load prompt templates
    print(f"Loading prompt templates from {prompt_template_file} (style: {prompt_style})...")
    try:
        templates = load_prompt_templates(prompt_template_file)
        prompt_templates = get_prompt_style(templates, prompt_style)
    except Exception as e:
        print(f"Warning: Could not load prompt templates: {e}")
        print("Using default prompts...")
        # Fallback to default templates
        prompt_templates = {
            "system_message": "You are a helpful assistant that solves NYT Connections puzzles.",
            "instruction_template": "You are playing NYT Connections. You are given 16 words.\nYour task is to group them into 4 categories of 4 words each.\nEach group shares a common theme or connection.\n\nWords: {words}",
            "user_message_template": "You are playing NYT Connections. You are given 16 words.\nYour task is to group them into 4 categories of 4 words each.\nEach group shares a common theme or connection.\n\nWords: {words}",
            "output_group_template": "Group ({difficulty}): {group_words} - Category: {category}"
        }
    
    # Convert to instruction format - always use consistent format for training
    formatted_examples = []
    for puzzle in puzzles:
        formatted = format_puzzle_as_instruction(puzzle, prompt_templates)
        chat_format = create_chat_format(formatted, prompt_templates)
        formatted_examples.append(chat_format)
    
    # Shuffle and split
    random.shuffle(formatted_examples)
    split_idx = int(len(formatted_examples) * train_split)
    train_data = formatted_examples[:split_idx]
    eval_data = formatted_examples[split_idx:]
    
    # Save as JSONL
    train_file = output_path / "train.jsonl"
    eval_file = output_path / "eval.jsonl"
    
    with open(train_file, "w") as f:
        for example in train_data:
            f.write(json.dumps(example) + "\n")
    
    with open(eval_file, "w") as f:
        for example in eval_data:
            f.write(json.dumps(example) + "\n")
    
    print(f"\nPrepared {len(train_data)} training examples")
    print(f"Prepared {len(eval_data)} evaluation examples")
    print(f"Saved to {train_file} and {eval_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare puzzle data for fine-tuning")
    parser.add_argument(
        "--connections-file",
        type=str,
        default="./connections.json",
        help="Path to connections.json file (default: ./connections.json)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/processed",
        help="Directory to save processed data"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.9,
        help="Fraction of data for training (default: 0.9)"
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
        help="Prompt style to use: default, concise, detailed, etc. (default: default)"
    )
    
    args = parser.parse_args()
    prepare_training_data(
        args.connections_file,
        args.output_dir,
        args.train_split,
        args.prompt_template_file,
        args.prompt_style
    )

