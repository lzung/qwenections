# qwenections

Fine-tune Qwen models to play NYT Connections puzzles.

## Overview

This project fine-tunes Qwen language models (specifically Qwen2.5) to solve NYT Connections puzzles. The model learns to group 16 words into 4 categories of 4 words each, where each group shares a common theme.

## Setup

### 1. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate qwenections
```

### 2. Install Dependencies

```bash
uv pip install -e .
```

## Usage

### Step 1: Collect Training Data

The project automatically collects NYT Connections puzzles via GitHub Actions, which runs daily to fetch new puzzles and update `connections.json`. 

**Manual Update:**
If you want to manually update the puzzle list:

```bash
python -m scripts.data.collect_data
```

### Step 2: Prepare Training Data

Convert puzzles from `connections.json` into instruction-following format:

```bash
python -m scripts.data.prepare_data --connections-file ./connections.json --output-dir ./data/processed
```

This creates:
- `./data/processed/train.jsonl` - Training examples
- `./data/processed/eval.jsonl` - Evaluation examples

### Step 3: Fine-tune the Model

Start fine-tuning with LoRA:

```bash
python -m scripts.tuning.finetune --config config.yaml
```

The training configuration can be adjusted in `config.yaml`:
- Model selection (default: Qwen/Qwen2.5-3B-Instruct)
- Training hyperparameters
- LoRA settings
- Data paths

### Step 4: Evaluate the Model

Evaluate the fine-tuned model on test data:

```bash
python -m scripts.tuning.evaluate --model-path ./checkpoints/final --test-data ./data/processed/eval.jsonl
```

### Step 5: Play with the Model

Interactively solve puzzles. Choose your solving approach:

**All-at-once approach** (finds all 4 groups at once):
```bash
python -m scripts.tuning.play --model-path ./checkpoints/final --approach all_at_once
```

**Iterative approach** (finds one group at a time):
```bash
python -m scripts.tuning.play --model-path ./checkpoints/final --approach iterative
```

Enter 16 words separated by commas when prompted.

**Difficulty levels:**
- `level: 0` = Yellow (easiest)
- `level: 1` = Green
- `level: 2` = Blue
- `level: 3` = Purple (hardest)
