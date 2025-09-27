# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepCORO_CLIP is a deep learning model for echocardiography video interpretation using contrastive learning. It implements three training paradigms:
1. **CLIP-style contrastive learning** - video-text alignment
2. **Multitask learning** - combines contrastive, captioning, and masked video modeling (SigLIP 2-inspired)
3. **Linear probing** - task-specific fine-tuning

## Essential Commands

### Environment Setup
```bash
# ALWAYS activate virtual environment first
source .venv/bin/activate

# Install dependencies (if needed)
uv sync
```

### Running Training

#### Standard Training (scripts/runner.sh)
```bash
# Single GPU training
bash scripts/runner.sh --base_config config/clip/base_config.yaml --selected_gpus 0 --use_wandb false --run_mode train

# Multi-GPU training with W&B logging
bash scripts/runner.sh --base_config config/clip/multitask_config.yaml --selected_gpus 0,1,2 --use_wandb true --run_mode train

# Inference mode (single GPU only)
bash scripts/runner.sh --base_config config/clip/base_config.yaml --selected_gpus 0 --run_mode inference --use_wandb false
```

#### Direct Multi-GPU Training (torchrun)
```bash
source .venv/bin/activate && \
export MASTER_PORT=29505 && \
export NCCL_P2P_LEVEL=NVL && \
export NCCL_ALGO=Tree && \
export NCCL_MIN_NCHANNELS=4 && \
export NCCL_SHM_DISABLE=0 && \
export NCCL_NET_GDR_LEVEL=PHB && \
torchrun --nproc_per_node=3 scripts/main.py --base_config config/clip/multitask_config.yaml
```

### Hyperparameter Sweeps

```bash
# Always activate environment first
source .venv/bin/activate

# Single GPU sweep
bash scripts/run_sweep.sh --base_config config/clip/multitask_config.yaml --sweep_config config/clip/sweep_config_multitask.yaml --selected_gpus 0 --count 10

# Multi-GPU sweep (NCCL variables set automatically)
bash scripts/run_sweep.sh --base_config config/clip/multitask_config.yaml --sweep_config config/clip/sweep_config_multitask.yaml --selected_gpus 0,1,2,3 --count 10
```

### Testing
```bash
# Run all tests
pytest

# Test multitask setup specifically
python test_multitask_setup.py

# Run linter (if available)
ruff check .

# Format code
black .
```

## High-Level Architecture

### Core Components

1. **Video Encoder** (`models/video_encoder.py`)
   - Uses Multiscale Vision Transformer (mVIT) backbone
   - Configurable output modes via `aggregate` and `per_video_pool` flags:
     - Study-level: `aggregate=True` → `[B, D]`
     - Video-level: `aggregate=False, per_video_pool=True` → `[B, N, D]`
     - Patch-level: `aggregate=False, per_video_pool=False` → `[B, N×L, D]`

2. **Text Encoder** (`models/text_encoder.py`)
   - BioMedBERT for medical text encoding
   - Configurable freezing ratio for fine-tuning

3. **Multitask Components** (when using multitask_config.yaml)
   - **Captioning Decoder** (`models/captioning_decoder.py`): LocCa-style transformer for report generation
   - **Masked Video Modeling** (`models/masked_video_modeling.py`): Self-supervised learning with 75% masking
   - **Multitask Loss** (`utils/loss/multitask_loss.py`): Weighted combination of three objectives

### Training Projects

- **ContrastivePretrainingProject** (`projects/contrastive_pretraining_project.py`): Standard CLIP training
- **MultitaskPretrainingProject** (`projects/multitask_pretraining_project.py`): SigLIP 2-inspired multitask training
- **LinearProbingProject** (`projects/linear_probing_project.py`): Task-specific fine-tuning

### Configuration Structure

```
config/
├── clip/
│   ├── base_config.yaml              # Standard CLIP training
│   ├── multitask_config.yaml         # Multitask training
│   └── sweep_config_*.yaml           # Hyperparameter sweep configs
└── linear_probing/
    ├── base_config.yaml               # Linear probing base
    └── sweep_config.yaml              # Linear probing sweeps
```

## Key Configuration Parameters

### Multitask Training (config/clip/multitask_config.yaml)
- `loss_weights`: Balance between contrastive, captioning, and masked_modeling
- `lr`, `text_lr`, `captioning_lr`, `mvm_lr`: Component-specific learning rates
- `mask_ratio`: Fraction of patches to mask (default 0.75)
- `decoder_layers`, `decoder_heads`: Captioning decoder architecture
- `batch_size`: Reduce if OOM (16 → 12 → 8)
- `gradient_accumulation_steps`: Increase if using smaller batch sizes

### Memory Management
- Reduce batch_size if OOM errors
- Use gradient_accumulation_steps to simulate larger batches
- Monitor with `nvidia-smi -l 1`

## Important Notes

1. **NCCL Environment Variables**: Required for optimal multi-GPU performance. The sweep script sets these automatically, but manual runs need explicit export.

2. **Python Dataclasses**: Fields with default values must come after fields without defaults.

3. **W&B Integration**: Set `use_wandb=true` for experiment tracking. Login required: `wandb login`

4. **GPU Memory**:
   - Single video: ~8GB per GPU
   - Multi-video: ~12GB per GPU
   - Multitask: ~16-24GB per GPU

5. **Run Modes**:
   - `train`: Training mode
   - `val`: Validation (linear probing only)
   - `test`: Testing (linear probing only)
   - `inference`: Process data where Split=='inference'

6. **Common Issues**:
   - OOM: Reduce batch_size or use fewer GPUs
   - NCCL timeout: Check NCCL environment variables
   - Import errors: Ensure `.venv` is activated