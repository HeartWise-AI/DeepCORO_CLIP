# DeepCORO_CLIP

A deep learning model for echocardiography video interpretation using contrastive learning and multiple instance learning.

## Overview

DeepCORO_CLIP is trained on over 12 million echocardiography videos paired with text reports from 275,442 studies. It uses a Multiscale Vision Transformer (mVIT) for video encoding and BioMedBERT for text encoding, combined with anatomical attention mechanisms for comprehensive study-level interpretation.

## Project Structure

- `data/`: Raw and processed datasets
- `models/`: Model architectures and checkpoints
- `training/`: Training scripts and logs
- `inference/`: Inference pipeline
- `evaluation/`: Model evaluation tools
- `utils/`: Utility functions
- `scripts/`: High-level execution scripts
- `experiments/`: Experimental configurations
- `logs/`: Training and evaluation logs
- `reports/`: Documentation and results

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare data:
```bash
python scripts/preprocess_data.py
```

3. Train model:
```bash
python scripts/train_model.py
```

## Model Architecture

- Video Encoder: mVIT (512-dim embeddings)
- Text Encoder: Modified BioMedBERT (512-dim embeddings)
- Multiple Instance Learning with anatomical attention
- Retrieval-augmented interpretation system

## Training Process

- Contrastive learning with 32 video-report pairs per batch
- 60 epochs pretraining
- 20 epochs fine-tuning
- Augmentations: RandAugment, RandomErasing

## Evaluation

Evaluated through cross-modal retrieval tasks:
- Video-to-text retrieval
- Text-to-video retrieval 

## Environment Setup

### Prerequisites

- CUDA-capable GPU
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer
- Python 3.11+

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DeepCORO_CLIP.git
cd DeepCORO_CLIP
```

2. Install uv if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Create and activate a new virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows
```

4. Install dependencies using uv:
```bash
# Install base dependencies
uv pip install -e .

# Install development dependencies (optional)
uv pip install -e ".[dev]"
```

5. Install PyTorch with CUDA support:
```bash
uv pip install torch==2.4.0 torchvision==0.19.0 torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Alternative Installation with Mamba/Conda

If you prefer using Mamba or Conda:

1. Create and activate environment:
```bash
# Using mamba (recommended)
mamba create -n deepcoro_clip python=3.11
mamba activate deepcoro_clip

# Or using conda
conda create -n deepcoro_clip python=3.11
conda activate deepcoro_clip
```

2. Install PyTorch with CUDA:
```bash
mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

3. Install remaining dependencies:
```bash
uv pip install -e ".[dev]"
```

### Development Setup

We use pre-commit hooks to ensure code quality and consistency. The hooks include:
- Black (code formatting)
- Ruff (linting)
- MyPy (type checking)
- Various file checks (YAML, TOML, trailing whitespace, etc.)
- Jupyter notebook formatting and cleaning

1. Install pre-commit hooks:
```bash
# Install pre-commit if you haven't already
uv pip install pre-commit

# Install the git hooks
pre-commit install
```

2. (Optional) Run hooks manually:
```bash
# Run on all files
pre-commit run --all-files

# Run on staged files
pre-commit run
```

The hooks will automatically run on `git commit`. You can temporarily skip them with `git commit --no-verify`.

### Code Style

This project follows:
- Black code style
- Ruff for linting
- Type hints for all functions

Code formatting is automatically handled by pre-commit hooks. Configuration can be found in:
- `pyproject.toml` for Black and Ruff settings
- `.pre-commit-config.yaml` for git hooks

## Pre-commit Configuration

repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files
    -   id: check-toml
    -   id: check-merge-conflict
    -   id: debug-statements

-   repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
    -   id: black
        language_version: python3.11

-   repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.11
    hooks:
    -   id: ruff
        args: [--fix, --exit-non-zero-on-fix]

-   repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
    -   id: mypy
        additional_dependencies: [types-all]

-   repo: https://github.com/nbQA-dev/nbQA
    rev: 1.7.1
    hooks:
    -   id: nbqa-black
        additional_dependencies: [black==23.12.1]
    -   id: nbqa-ruff
        additional_dependencies: [ruff==0.1.11]

-   repo: https://github.com/kynan/nbstripout
    rev: 0.6.1
    hooks:
    -   id: nbstripout