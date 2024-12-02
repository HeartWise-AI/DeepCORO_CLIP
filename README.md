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

1. Install pre-commit hooks:
```bash
pre-commit install
```

2. Format code:
```bash
# Format with black
black .

# Check with ruff
ruff check .
```