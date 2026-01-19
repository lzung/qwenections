#!/usr/bin/env python3
"""
Evaluate fine-tuned Qwen model on NYT Connections puzzles.
"""

import json
import torch
from pathlib import Path
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.prompt_utils import (
    load_prompt_templates,
    get_prompt_style,
    create_chat_messages
)


def load_model(model_path: str, base_model: str = "Qwen/Qwen2.5-3B-Instruct"):
    """Load fine-tuned model."""
    from peft import PeftModel
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA weights if they exist
    if Path(model_path).exists():
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()  # Merge LoRA weights
    else:
        model = base_model
    
    model.eval()
    return model, tokenizer


def solve_puzzle(
    model, 
    tokenizer, 
    words: List[str], 
    max_new_tokens: int = 512,
    prompt_templates: dict = None
):
    """Solve a Connections puzzle using prompt templates."""
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
    
    messages = create_chat_messages(
        words,
        prompt_templates["system_message"],
        prompt_templates["user_message_template"]
    )
    
    # Format with chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )
    
    # Decode
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def evaluate_on_puzzle(model, tokenizer, puzzle: Dict, prompt_templates: dict = None) -> Dict:
    """Evaluate model on a single puzzle."""
    words = puzzle["words"]
    expected_groups = puzzle["groups"]
    
    # Get model prediction
    prediction = solve_puzzle(model, tokenizer, words, prompt_templates=prompt_templates)
    
    # Parse prediction (simplified - you may want more sophisticated parsing)
    # For now, just return the raw prediction
    return {
        "date": puzzle.get("date", "unknown"),
        "words": words,
        "expected_groups": expected_groups,
        "prediction": prediction,
    }


def evaluate_model(
    model_path: str,
    test_data_file: str = "./data/processed/eval.jsonl",
    base_model: str = "Qwen/Qwen2.5-3B-Instruct",
    prompt_template_file: str = "./prompt_templates.yaml",
    prompt_style: str = "default"
):
    """Evaluate model on test dataset."""
    print(f"Loading model from {model_path}...")
    model, tokenizer = load_model(model_path, base_model)
    
    # Load prompt templates
    try:
        templates = load_prompt_templates(prompt_template_file)
        prompt_templates = get_prompt_style(templates, prompt_style)
        print(f"Loaded prompt style: {prompt_style}")
    except Exception as e:
        print(f"Warning: Could not load prompt templates: {e}")
        print("Using default prompts...")
        prompt_templates = None
    
    # Load test data
    print(f"Loading test data from {test_data_file}...")
    test_puzzles = []
    with open(test_data_file, "r") as f:
        for line in f:
            example = json.loads(line)
            # Reconstruct puzzle from metadata
            if "metadata" in example:
                metadata = example["metadata"]
                puzzle = {
                    "date": metadata.get("date", "unknown"),
                    "words": metadata.get("original_words", []),
                    "groups": metadata.get("groups", [])
                }
                test_puzzles.append(puzzle)
    
    if len(test_puzzles) == 0:
        print("Warning: No test puzzles found. Using eval.jsonl metadata.")
        # Try to extract from messages if metadata not available
        with open(test_data_file, "r") as f:
            for line in f:
                example = json.loads(line)
                # This is a fallback - you may need to adjust based on your data format
                pass
    
    print(f"Evaluating on {len(test_puzzles)} puzzles...")
    
    results = []
    for i, puzzle in enumerate(test_puzzles):
        print(f"\nEvaluating puzzle {i+1}/{len(test_puzzles)}...")
        result = evaluate_on_puzzle(model, tokenizer, puzzle, prompt_templates)
        results.append(result)
        
        print(f"Date: {result['date']}")
        print(f"Prediction:\n{result['prediction']}")
        print("-" * 50)
    
    # Save results
    output_file = Path(model_path).parent / "evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation complete! Results saved to {output_file}")
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Qwen model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to fine-tuned model checkpoint"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="./data/processed/eval.jsonl",
        help="Path to test data file"
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
        help="Prompt style to use: default, concise, detailed, etc. (default: default)"
    )
    
    args = parser.parse_args()
    evaluate_model(
        args.model_path, 
        args.test_data, 
        args.base_model,
        args.prompt_template_file,
        args.prompt_style
    )

