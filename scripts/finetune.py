#!/usr/bin/env python3
"""
Fine-tune Qwen model for NYT Connections using LoRA.
"""

import os
import yaml
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from accelerate import Accelerator


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "Qwen/Qwen2.5-4B-Instruct"
    max_length: int = 2048
    trust_remote_code: bool = True


@dataclass
class LoRAConfig:
    """LoRA configuration."""
    enabled: bool = True
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])


def load_config(config_path: str = "./config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model_and_tokenizer(config: dict):
    """Load model and tokenizer."""
    model_config = ModelConfig(**config["model"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.name,
        trust_remote_code=model_config.trust_remote_code
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_config.name,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # Apply LoRA if enabled
    if config.get("lora", {}).get("enabled", True):
        lora_config_dict = config.get("lora", {})
        lora_config = LoRAConfig(**lora_config_dict)
        
        peft_config = LoraConfig(
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            target_modules=lora_config.target_modules,
            lora_dropout=lora_config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Prepare model for training (works with both quantized and float16 models)
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


def format_prompts(examples, tokenizer):
    """Format examples for training."""
    # Apply chat template
    # When batched=True, examples["messages"] is a list of message lists
    formatted_texts = []
    for messages in examples["messages"]:
        # Qwen2.5 uses apply_chat_template
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        formatted_texts.append(formatted)
    
    return {"text": formatted_texts}


def tokenize_function(examples, tokenizer, max_length):
    """Tokenize the formatted texts."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune Qwen for NYT Connections")
    parser.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="Path to config YAML file"
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Load dataset
    print("Loading dataset...")
    data_config = config["data"]
    train_dataset = load_dataset(
        "json",
        data_files=data_config["train_file"],
        split="train"
    )
    eval_dataset = load_dataset(
        "json",
        data_files=data_config["eval_file"],
        split="train"
    )
    
    # Limit samples if specified
    if data_config.get("max_samples"):
        train_dataset = train_dataset.select(range(min(
            data_config["max_samples"],
            len(train_dataset)
        )))
    
    # Format prompts
    print("Formatting prompts...")
    train_dataset = train_dataset.map(
        lambda examples: format_prompts(examples, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    eval_dataset = eval_dataset.map(
        lambda examples: format_prompts(examples, tokenizer),
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    
    # Tokenize
    print("Tokenizing...")
    max_length = config["model"]["max_length"]
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=["text"]
    )
    eval_dataset = eval_dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=["text"]
    )
    
    # Set up training arguments
    training_config = config["training"]
    training_args = TrainingArguments(
        output_dir=training_config["output_dir"],
        num_train_epochs=training_config["num_train_epochs"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        warmup_steps=training_config["warmup_steps"],
        logging_steps=training_config["logging_steps"],
        save_steps=training_config["save_steps"],
        eval_steps=training_config["eval_steps"],
        save_total_limit=training_config["save_total_limit"],
        fp16=training_config.get("fp16", True),
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        optim=training_config.get("optim", "adamw_torch"),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        report_to="none",  # Can change to "wandb" or "tensorboard"
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print("Saving model...")
    final_output_dir = Path(training_config["output_dir"]) / "final"
    trainer.save_model(str(final_output_dir))
    tokenizer.save_pretrained(str(final_output_dir))
    
    print(f"Training complete! Model saved to {final_output_dir}")


if __name__ == "__main__":
    main()

