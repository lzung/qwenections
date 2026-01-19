#!/usr/bin/env python3
"""
Prepare collected puzzle data for fine-tuning.

Converts puzzle data into instruction-following format suitable for Qwen.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple


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


def format_puzzle_as_instruction(puzzle: Dict) -> Dict:
    """
    Format a puzzle into instruction-following format.
    
    Format:
    - Instruction: Describe the task
    - Input: The 16 words
    - Output: The 4 groups with their categories
    """
    words = puzzle["words"]
    groups = puzzle["groups"]
    
    # Shuffle words to avoid position bias
    shuffled_words = words.copy()
    random.shuffle(shuffled_words)
    
    # Format input
    words_str = ", ".join(shuffled_words)
    instruction = (
        "You are playing NYT Connections. You are given 16 words. "
        "Your task is to group them into 4 categories of 4 words each. "
        "Each group shares a common theme or connection."
    )
    
    # Format output
    group_descriptions = []
    for group in groups:
        words_in_group = ", ".join(group["words"])
        category = group["category"]
        difficulty = group.get("difficulty", "unknown")
        group_descriptions.append(
            f"Group ({difficulty}): {words_in_group} - Category: {category}"
        )
    
    output = "\n".join(group_descriptions)
    
    return {
        "instruction": instruction,
        "input": words_str,
        "output": output,
        "metadata": {
            "date": puzzle.get("date", "unknown"),
            "id": puzzle.get("id", None),
            "original_words": words,
            "groups": groups
        }
    }


def create_chat_format(example: Dict) -> Dict:
    """
    Convert to Qwen chat format.
    Qwen2.5 uses a specific chat template.
    """
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that solves NYT Connections puzzles."
        },
        {
            "role": "user",
            "content": f"{example['instruction']}\n\nWords: {example['input']}"
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
    raw_data_dir: str = None,
    output_dir: str = "./data/processed",
    train_split: float = 0.9
):
    """
    Prepare training data from connections.json or raw puzzle files.
    
    Args:
        connections_file: Path to connections.json file (primary source)
        raw_data_dir: Optional directory containing additional raw puzzle JSON files
        output_dir: Directory to save processed data
        train_split: Fraction of data to use for training (rest for eval)
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
    
    # Optionally load additional puzzles from raw_data_dir
    if raw_data_dir:
        raw_path = Path(raw_data_dir)
        if raw_path.exists():
            additional_puzzles = load_puzzles(raw_path)
            print(f"Loaded {len(additional_puzzles)} additional puzzles from {raw_data_dir}")
            puzzles.extend(additional_puzzles)
    
    if len(puzzles) == 0:
        print("Warning: No puzzles found. Please ensure connections.json exists or collect data first.")
        return
    
    # Convert to instruction format
    formatted_examples = []
    for puzzle in puzzles:
        formatted = format_puzzle_as_instruction(puzzle)
        chat_format = create_chat_format(formatted)
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
        "--raw-data-dir",
        type=str,
        default=None,
        help="Optional directory containing additional raw puzzle JSON files"
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
    
    args = parser.parse_args()
    prepare_training_data(
        args.connections_file,
        args.raw_data_dir,
        args.output_dir,
        args.train_split
    )

