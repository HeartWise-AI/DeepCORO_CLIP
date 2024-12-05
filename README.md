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

## Build System

This project uses modern Python packaging tools:
- `pyproject.toml` for build configuration and package metadata
- `requirements.txt` for dependency management
- Hatch as the build backend


## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Login to Weights & Biases:

```bash
# Login to your wandb account
wandb login

# If you don't have an account, sign up at https://wandb.ai
# After signing up, get your API key from https://wandb.ai/settings
```

3. Train model:

Single GPU training:

```bash
python scripts/train_model.py
```

Multi-GPU training (2 GPUs):

```torchrun --nproc_per_node=2 scripts/train_model.py --batch-size 16
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

## Training Options

### Using Configuration Files

The training script supports YAML configuration files for easier experiment management. You can specify all training parameters in a YAML file instead of using command-line arguments.

1. **Basic Usage with Config File**:

```bash
python scripts/train_model.py --config config/default_config.yaml
```

2. **Override Config with Command Line**:
   You can override config file values with command-line arguments:

```bash
python scripts/train_model.py --config config/default_config.yaml --batch-size 16 --lr 0.0001
```

3. **Example Config File Structure** (`config/default_config.yaml`):

```yaml
# Training parameters
epochs: 20
batch_size: 8
num_workers: 8
lr: 0.000025
debug: false
temp: 0.1

# Data parameters
data_filename: data/reports/reports_sampled_1000.csv
root: "."
target_label: Report
datapoint_loc_label: FileName
frames: 16

# Model parameters
model_name: mvit
pretrained: true

# Logging parameters
project: your_project_name
entity: your_wandb_entity
tag: experiment_tag
```

4. **Multi-GPU Training with Config**:

```bash
torchrun --nproc_per_node=2 scripts/train_model.py --config config/default_config.yaml
```

### Single GPU vs Multi-GPU Training

The model can be trained in two modes:

1. **Single GPU Training** (Recommended for most users)

   - Uses one GPU for training
   - Simpler setup and debugging
   - Lower memory requirements
   - Slower training but more stable

1. **Multi-GPU Training** (For large-scale training)

   - Uses multiple GPUs with Distributed Data Parallel (DDP)
   - Faster training through data parallelism
   - Higher memory requirements
   - More complex setup and potential synchronization issues

### How to Run Training

#### Force Single GPU Training (Recommended)

```bash
# Method 1: Set CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=0 python scripts/train_model.py

# Method 2: Use specific GPU index
python scripts/train_model.py --gpu 0
```

#### Multi-GPU Training (Advanced)

```bash
# Use all available GPUs
python scripts/train_model.py

# Use specific GPUs
CUDA_VISIBLE_DEVICES=0,1 python scripts/train_model.py
```

### Memory Requirements

- Single GPU: Minimum 12GB VRAM
- Multi-GPU: Minimum 8GB VRAM per GPU

### Performance Comparison

- Single GPU:

  - Batch size: 32
  - Training time: ~2 days
  - Memory usage: ~10GB VRAM

- Multi-GPU (2 GPUs):

  - Batch size: 32 per GPU (64 total)
  - Training time: ~1 day
  - Memory usage: ~8GB VRAM per GPU

### Tips for GPU Selection

1. For research and development:

   - Use single GPU for easier debugging
   - Better reproducibility
   - More stable training

1. For production training:

   - Use multi-GPU when you need faster training
   - Ensure all GPUs are identical models
   - Monitor GPU memory usage

1. For limited GPU memory:

   - Force single GPU mode
   - Reduce batch size
   - Use gradient accumulation

### Common Issues

1. Out of Memory (OOM):

   - Reduce batch size
   - Use gradient accumulation
   - Force single GPU mode

1. GPU Selection:

   - Use `CUDA_VISIBLE_DEVICES` to select specific GPUs
   - Monitor GPU usage with `nvidia-smi`

1. Training Speed:

   - Multi-GPU isn't always faster due to overhead
   - Start with single GPU and scale up if needed

### Command-Line Arguments

The training script supports various command-line arguments to customize the training process:

```bash
python scripts/train_model.py [OPTIONS]

Options:
  --gpu INTEGER        GPU index to use (forces single GPU training)
  --batch-size INTEGER Default: 32. Batch size per GPU
  --num-workers INTEGER Default: 4. Number of data loading workers
  --epochs INTEGER     Default: 50. Number of epochs to train
  --lr FLOAT          Default: 1e-4. Learning rate
```

#### Examples

1. Basic training with defaults:

```bash
python scripts/train_model.py
```

2. Training with specific GPU and batch size:

```bash
python scripts/train_model.py --gpu 0 --batch-size 16
```

3. Full configuration:

```bash
python scripts/train_model.py \
    --gpu 0 \
    --batch-size 16 \
    --epochs 50 \
    --lr 1e-4 \
    --num-workers 4
```

#### Recommended Batch Sizes by GPU Memory

| GPU Memory | Recommended Batch Size | Command           |
| ---------- | ---------------------- | ----------------- |
| 8GB        | 4-8                    | `--batch-size 8`  |
| 12GB       | 8-16                   | `--batch-size 16` |
| 16GB       | 16-24                  | `--batch-size 24` |
| 24GB+      | 24-32                  | `--batch-size 32` |

#### Tips for Training Configuration

1. **Batch Size Selection**:

   - Start with a smaller batch size and increase if memory allows
   - Larger batch sizes generally allow faster training but require more memory
   - If you get OOM (Out of Memory) errors, reduce the batch size

1. **Number of Workers**:

   - Rule of thumb: use `num_workers = 4 * num_gpus`
   - Reduce if you get memory or file handle errors
   - Example: `--num-workers 2` for slower storage systems

1. **Learning Rate**:

   - Default (1e-4) works well for most cases
   - For larger batch sizes, consider scaling up: `lr = 1e-4 * (batch_size/32)`
   - Example: `--lr 2e-4` for batch size 64

1. **Number of Epochs**:

   - Default (50) is good for most cases
   - Increase for better performance: `--epochs 100`
   - Decrease for quick experiments: `--epochs 10`

#### Example Configurations

1. **Quick Testing**:

```bash
python scripts/train_model.py --gpu 0 --batch-size 8 --epochs 5 --num-workers 2
```

2. **Standard Training**:

```bash
python scripts/train_model.py --gpu 0 --batch-size 16 --epochs 50 --num-workers 4
```

3. **Production Training** (large GPU):

```bash
python scripts/train_model.py --gpu 0 --batch-size 32 --epochs 100 --num-workers 8
```

4. **Memory-Limited Setup**:

```bash
python scripts/train_model.py --gpu 0 --batch-size 4 --epochs 50 --num-workers 2
```

#### Monitoring Training

1. **GPU Memory Usage**:

```bash
nvidia-smi -l 1  # Monitor GPU usage every second
```

2. **Training Progress**:

- Progress bar shows current epoch and batch
- Loss values are printed every 10 batches
- Checkpoints are saved every 5 epochs

3. **WandB Logging**:

- Training metrics are logged to Weights & Biases
- Includes loss, learning rate, batch size
- Access via WandB dashboard
