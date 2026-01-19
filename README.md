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
python scripts/collect_data.py
```

### Step 2: Prepare Training Data

Convert puzzles from `connections.json` into instruction-following format:

```bash
python scripts/prepare_data.py --connections-file ./connections.json --output-dir ./data/processed
```

You can also include additional puzzles from a directory:

```bash
python scripts/prepare_data.py --connections-file ./connections.json --raw-data-dir ./data/raw --output-dir ./data/processed
```

This creates:
- `./data/processed/train.jsonl` - Training examples
- `./data/processed/eval.jsonl` - Evaluation examples

### Step 3: Fine-tune the Model

Start fine-tuning with LoRA:

```bash
python scripts/finetune.py --config config.yaml
```

The training configuration can be adjusted in `config.yaml`:
- Model selection (default: Qwen/Qwen2.5-7B-Instruct)
- Training hyperparameters
- LoRA settings
- Data paths

### Step 4: Evaluate the Model

Evaluate the fine-tuned model on test data:

```bash
python scripts/evaluate.py --model-path ./checkpoints/final --test-data ./data/processed/eval.jsonl
```

### Step 5: Play with the Model

Interactively solve puzzles:

```bash
python scripts/play.py --model-path ./checkpoints/final
```

Enter 16 words separated by commas when prompted.

## Project Structure

```
qwenections/
├── connections.json         # NYT Connections puzzle data (auto-updated daily)
├── config.yaml              # Fine-tuning configuration
├── environment.yml          # Conda environment setup
├── pyproject.toml           # Python dependencies
├── .github/
│   └── workflows/
│       └── actions.yml      # GitHub Actions workflow for daily updates
├── scripts/
│   ├── collect_data.py     # Fetch puzzles from NYT Connections API
│   ├── prepare_data.py     # Prepare data for training
│   ├── finetune.py         # Fine-tuning script
│   ├── evaluate.py         # Model evaluation
│   └── play.py             # Interactive puzzle solving
└── data/
    ├── raw/                 # Optional: Additional raw puzzle JSON files
    └── processed/           # Processed training data
```

## Configuration

Edit `config.yaml` to customize:

- **Model**: Change the base model (e.g., Qwen2.5-1.5B, Qwen2.5-14B)
- **Training**: Adjust epochs, batch size, learning rate, etc.
- **LoRA**: Modify rank, alpha, dropout, and target modules
- **Data**: Set paths to training and evaluation files

## Data Format

### connections.json Format

The `connections.json` file (auto-updated by GitHub Actions) uses the NYT Connections API format:

```json
[
  {
    "id": 1,
    "date": "2023-06-12",
    "answers": [
      {
        "level": 0,
        "group": "WET WEATHER",
        "members": ["HAIL", "RAIN", "SLEET", "SNOW"]
      },
      {
        "level": 1,
        "group": "NBA TEAMS",
        "members": ["BUCKS", "HEAT", "JAZZ", "NETS"]
      },
      {
        "level": 2,
        "group": "KEYBOARD KEYS",
        "members": ["OPTION", "RETURN", "SHIFT", "TAB"]
      },
      {
        "level": 3,
        "group": "PALINDROMES",
        "members": ["KAYAK", "LEVEL", "MOM", "RACECAR"]
      }
    ]
  }
]
```

**Difficulty levels:**
- `level: 0` = Yellow (easiest)
- `level: 1` = Green
- `level: 2` = Blue
- `level: 3` = Purple (hardest)

### Optional: Additional Raw Puzzle Format

If you add additional puzzles to `./data/raw/`, they should use this format:

```json
{
  "date": "2024-01-01",
  "words": ["WORD1", "WORD2", ..., "WORD16"],
  "groups": [
    {
      "words": ["WORD1", "WORD2", "WORD3", "WORD4"],
      "category": "Category Name",
      "difficulty": "yellow"
    },
    ...
  ]
}
```

## Requirements

- Python 3.11+
- CUDA-capable GPU (recommended for training)
- Sufficient VRAM (7B model needs ~14GB with LoRA)

## Notes

- **Automatic Updates**: The GitHub Actions workflow runs daily at 1:00 AM UTC to fetch new puzzles. Make sure to pull the latest changes regularly to get the most up-to-date training data.
- **Real Puzzle Data**: The project uses real NYT Connections puzzles from the official API, providing high-quality training data.
- Training time depends on your hardware and dataset size.
- LoRA allows efficient fine-tuning with minimal memory usage compared to full fine-tuning.
- The `connections.json` file contains all historical puzzles and grows over time as new puzzles are added.

## License

See LICENSE file for details.
