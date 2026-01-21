#!/usr/bin/env python3
"""
Fine-tune Qwen model for NYT Connections using LoRA.
"""

import os
import yaml
import torch
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timedelta

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "Qwen/Qwen2.5-3B-Instruct"
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
    
    # Load model - detect device for CPU vs GPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch_dtype = torch.float32 if device.type == "cpu" else torch.float16
    
    # For CPU training, disable device_map offloading to avoid device mismatch
    device_map = None if device.type == "cpu" else "auto"
    

    model = AutoModelForCausalLM.from_pretrained(
        model_config.name,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch_dtype,
        device_map=device_map
    )
    
    # Disable use_cache to enable gradient computation
    model.config.use_cache = False
    
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
        
        # Apply LoRA and enable gradients
        model = get_peft_model(model, peft_config)
        # Ensure model parameters require gradients
        for param in model.parameters():
            param.requires_grad = True
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


class TrainingStatsCallback(TrainerCallback):
    """Callback to display interactive training statistics."""
    
    def __init__(self):
        self.start_time = None
        self.epoch_start_time = None
        self.last_log_time = None
        self.total_steps = None
        self.current_epoch = 0
        self.step_times = []
        self.logger = logging.getLogger(f"{__name__}.TrainingStats")
        
    def on_train_begin(self, args, state, control, **kwargs):
        """Called when training begins."""
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.logger.info("="*80)
        self.logger.info("TRAINING STARTED")
        self.logger.info("="*80)
        self.logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Total epochs: {args.num_train_epochs}")
        self.logger.info(f"Total steps: {state.max_steps if state.max_steps else 'N/A'}")
        self.logger.info("="*80)
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each epoch."""
        self.current_epoch = int(state.epoch) + 1
        self.epoch_start_time = time.time()
        self.logger.info("-"*80)
        self.logger.info(f"EPOCH {self.current_epoch}/{int(args.num_train_epochs)}")
        self.logger.info("-"*80)
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logs are written."""
        if logs is None:
            return
            
        current_time = time.time()
        elapsed_total = current_time - self.start_time
        
        # Calculate step time
        if state.global_step > 0:
            if self.last_log_time:
                step_time = current_time - self.last_log_time
                self.step_times.append(step_time)
                # Keep only last 10 step times for average
                if len(self.step_times) > 10:
                    self.step_times.pop(0)
            self.last_log_time = current_time
        
        # Calculate epoch progress
        if hasattr(state, 'max_steps') and state.max_steps and int(args.num_train_epochs) > 0:
            steps_per_epoch = state.max_steps // int(args.num_train_epochs)
            if steps_per_epoch > 0:
                epoch_progress = (state.global_step % steps_per_epoch) / steps_per_epoch * 100
            else:
                epoch_progress = 0
        else:
            # Fallback: estimate based on dataset size
            if hasattr(kwargs.get('train_dataloader'), '__len__'):
                try:
                    total_batches = len(kwargs['train_dataloader'])
                    current_batch = state.global_step % total_batches if total_batches > 0 else 0
                    epoch_progress = (current_batch / total_batches * 100) if total_batches > 0 else 0
                except:
                    epoch_progress = 0
            else:
                epoch_progress = 0
        
        # Calculate ETA
        eta_str = "N/A"
        if self.step_times and state.global_step > 0:
            avg_step_time = sum(self.step_times) / len(self.step_times)
            if hasattr(state, 'max_steps') and state.max_steps:
                remaining_steps = state.max_steps - state.global_step
                eta_seconds = remaining_steps * avg_step_time
                eta_str = str(timedelta(seconds=int(eta_seconds)))
            elif self.current_epoch > 0:
                # Estimate based on epoch time
                if self.epoch_start_time:
                    epoch_elapsed = current_time - self.epoch_start_time
                    remaining_epochs = int(args.num_train_epochs) - self.current_epoch
                    eta_seconds = (epoch_elapsed / epoch_progress * 100) * remaining_epochs if epoch_progress > 0 else 0
                    eta_str = str(timedelta(seconds=int(eta_seconds)))
        
        # Format elapsed time
        elapsed_str = str(timedelta(seconds=int(elapsed_total)))
        
        # Get loss values
        train_loss = logs.get('loss', 'N/A')
        eval_loss = logs.get('eval_loss', 'N/A')
        learning_rate = logs.get('learning_rate', 'N/A')
        
        # Log stats
        self.logger.info(f"[Step {state.global_step}] Epoch: {self.current_epoch}/{int(args.num_train_epochs)} ({epoch_progress:.1f}%)")
        self.logger.info(f"  Elapsed: {elapsed_str} | ETA: {eta_str}")
        
        if isinstance(train_loss, (int, float)):
            self.logger.info(f"  Train Loss: {train_loss:.4f}")
        else:
            self.logger.info(f"  Train Loss: {train_loss}")
            
        if isinstance(eval_loss, (int, float)):
            self.logger.info(f"  Eval Loss: {eval_loss:.4f}")
            
        if isinstance(learning_rate, (int, float)):
            self.logger.info(f"  Learning Rate: {learning_rate:.2e}")
        
        # Log step time if available
        if self.step_times:
            avg_step_time = sum(self.step_times) / len(self.step_times)
            self.logger.debug(f"  Avg Step Time: {avg_step_time:.2f}s")
        
    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch."""
        if self.epoch_start_time:
            epoch_duration = time.time() - self.epoch_start_time
            epoch_str = str(timedelta(seconds=int(epoch_duration)))
            self.logger.info(f"Epoch {self.current_epoch} completed in {epoch_str}")
            self.logger.info("-"*80)
        
    def on_train_end(self, args, state, control, **kwargs):
        """Called when training ends."""
        total_time = time.time() - self.start_time
        total_str = str(timedelta(seconds=int(total_time)))
        self.logger.info("="*80)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info("="*80)
        self.logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Total training time: {total_str}")
        self.logger.info(f"Total epochs: {self.current_epoch}")
        self.logger.info(f"Total steps: {state.global_step}")
        self.logger.info("="*80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune Qwen for NYT Connections")
    parser.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level (default: INFO)"
    )
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Load configuration
    config = load_config(args.config)
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Load dataset
    logger.info("Loading dataset...")
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
    logger.info("Formatting prompts...")
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
    logger.info("Tokenizing...")
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
        eval_steps=training_config.get("eval_steps", 500),
        save_total_limit=training_config["save_total_limit"],
        fp16=training_config.get("fp16", True),
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        optim=training_config.get("optim", "adamw_torch"),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        eval_strategy="no",  # Disable evaluation during training for speed
        report_to="none",  # Can change to "wandb" or "tensorboard"
        dataloader_pin_memory=False,  # Disable pin_memory for CPU training
        dataloader_num_workers=0,  # No multiprocessing on CPU to save memory
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create trainer with stats callback
    stats_callback = TrainingStatsCallback()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[stats_callback],
    )
    
    # Calculate steps per epoch
    batch_size = training_config['per_device_train_batch_size'] * training_config['gradient_accumulation_steps']
    steps_per_epoch = len(train_dataset) // batch_size
    total_steps = steps_per_epoch * training_config['num_train_epochs']
    
    # Log dataset info
    logger.info("Dataset Info:")
    logger.info(f"  Training examples: {len(train_dataset):,}")
    logger.info(f"  Evaluation examples: {len(eval_dataset):,}")
    logger.info(f"  Batch size (effective): {batch_size}")
    logger.info(f"  Steps per epoch: {steps_per_epoch:,}")
    logger.info(f"  Total steps: {total_steps:,}")
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info("Saving model...")
    final_output_dir = Path(training_config["output_dir"]) / "final"
    trainer.save_model(str(final_output_dir))
    tokenizer.save_pretrained(str(final_output_dir))
    
    logger.info(f"Training complete! Model saved to {final_output_dir}")


if __name__ == "__main__":
    main()

