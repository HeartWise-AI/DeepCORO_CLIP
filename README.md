# DeepCORO_CLIP

DeepCORO_CLIP is a deep learning model for echocardiography video interpretation using contrastive learning. It leverages a Multiscale Vision Transformer (mVIT) for video encoding and BioMedBERT for text encoding, trained on millions of video-report pairs.

## Environment Setup

### Prerequisites

- **CUDA-capable GPU**
- **Python 3.11+**
- **[uv](https://github.com/ashttps://github.com/astral-sh/uvtral-sh/uv)** (optional) or `pip` for installing dependencies

### Steps

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/DeepCORO_CLIP.git
   cd DeepCORO_CLIP
   ```

1. **Install Dependencies**:
   First install uv (Mandatory):

   ```bash
   # On macOS and Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   Then install dependencies:

   ```bash
   # Using uv
   uv sync
   ```

1. **Activate Virtual Environment** (do this every time you start):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Linux/Mac
   # or
   .venv\Scripts\activate     # On Windows
   ```

1. **Log into Weights & Biases (optional)**:

   ```bash
   wandb login
   ```

## Configuration Files

Training parameters (batch size, epochs, model name, data paths) are defined in YAML config files under `config/`.

**Example: `config/default_config.yaml`**:

````yaml
# Training parameters
epochs: 50
batch_size: 32
num_workers: 4
lr: 5e-5
debug: false
temp: 0.1

# Data parameters
data_filename: processed/reports/reports_sampled_1000.csv
root: data/
target_label: Report
datapoint_loc_label: FileName
frames: 16

# Model parameters
model_name: mvit
pretrained: true

# Multi-video parameters
multi_video: true
n_video: 4
scorn: StudyInstanceUID
aggregate_function: mean

# Logging parameters
project: deepcoro_clip
entity: your_wandb_entity
tag: experiment_tag
output_dir: outputs



## Training

### Use Config Files

- **Run with default config**:
    ```bash
    python scripts/train_model.py --config config/default_config.yaml
    ```

- **Override config values with CLI arguments**:
    ```bash
    python scripts/train_model.py --config config/default_config.yaml --batch-size 16 --lr 0.0001 --n_video 4 --scorn StudyInstanceUID --aggregate_function mean
    ```

### Single GPU Training

Ideal for development and debugging:
```bash
python scripts/train_model.py --config config/default_config.yaml --gpu 0
```

### Multi-GPU Training

```bash
torchrun --nproc_per_node=2 scripts/train_model_multi_gpu.py --config config/default_config.yaml
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

### Performance Comparison

- Single GPU:

  - Batch size: 32
  - Training time: ~2 days
  - Memory usage: ~10GB VRAM

- Multi-GPU (2 GPUs):

  - Batch size: 32 per GPU (64 total)
  - Training time: ~1 day
  - Memory usage: ~8GB VRAM per GPU

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

## Multi-Video Parameters

- **multi_video**: Enables the use of multiple videos per study for training and evaluation.
- **n_video**: Specifies the number of videos to be used per study.
- **scorn**: The column name used as a unique identifier for grouping videos by study.
- **aggregate_function**: The function used to aggregate embeddings from multiple videos. Options include `mean`, `max`, and `median`.

## Default Configuration Parameters

Here are the default configuration parameters used in `config/default_config.yaml`:

- **Training parameters**
  - **epochs**: Number of training epochs. Example: `50`
  - **batch_size**: Number of samples per batch. Example: `8`
  - **num_workers**: Number of subprocesses for data loading. Example: `16`
  - **learning_rate**: Learning rate for the optimizer. Example: `0.0001`
  - **temperature**: Temperature for contrastive loss. Example: `0.07`

- **Data parameters**
  - **data_filename**: Path to the CSV file containing data. Example: `data/reports/reports_sampled_300_study_ids.csv`
  - **root**: Root directory for data. Example: `.`
  - **target_label**: Column name for target text. Example: `Report`
  - **datapoint_loc_label**: Column name for file paths. Example: `FileName`
  - **frames**: Number of frames to sample from each video. Example: `16`
  - **stride**: Frame sampling stride. Example: `2`

- **Model parameters**
  - **model_name**: Name of the video backbone model. Example: `mvit`
  - **pretrained**: Whether to use a pretrained model. Example: `true`

- **Optimization parameters**
  - **optimizer**: Type of optimizer to use. Example: `RAdam`
  - **weight_decay**: Weight decay for the optimizer. Example: `0.000001`
  - **scheduler_type**: Type of learning rate scheduler. Example: `step`
  - **lr_step_period**: Period for learning rate step. Example: `15`
  - **factor**: Factor for learning rate scheduler. Example: `0.3`

- **Distributed training**
  - **gpu**: GPU index to use. Example: `2`
  - **local_rank**: Local rank for distributed training. Example: `-1`

- **Logging parameters**
  - **project**: Name of the W&B project. Example: `deepCORO_CLIP`
  - **entity**: W&B entity name. Example: `mhi_ai`
  - **tag**: Tag for the W&B run. Example: `DeepCORO_Clip_Sweep_Learnable_Temp_Full`
  - **multi_video**: Enable multi-video per study. Example: `true`
  - **n_video**: Number of videos per study. Example: `4`
  - **scorn**: Column name for study ID. Example: `StudyInstanceUID`
  - **aggregate_function**: Function to aggregate video embeddings. Example: `mean`

- **Additional parameters**
  - **output_dir**: Directory to save outputs. Example: `outputs`
  - **seed**: Random seed for reproducibility. Example: `0`
  - **use_amp**: Use automatic mixed precision. Example: `true`
  - **device**: Device to use for training. Example: `cuda`
  - **period**: Period for logging. Example: `1`

- **Metrics control**
  - **optimization_threshold**: Metric for optimization threshold. Example: `g_mean`
  - **plot_prediction_distribution**: Plot prediction distribution. Example: `true`
  - **plot_metrics_moving_threshold**: Plot metrics with moving threshold. Example: `true`
  - **num_bootstrap**: Number of bootstrap samples. Example: `100`
  - **quantile_bootstrap**: Quantile for bootstrap. Example: `0.05`

- **Data augmentation**
  - **random_augment**: Enable random augmentations. Example: `true`
  - **resize**: Resize dimension for images. Example: `224`
  - **apply_mask**: Apply mask to images. Example: `false`
  - **view_count**: Number of views. Example: `null`

- **Checkpointing**
  - **save_best**: Criterion to save the best model. Example: `loss`
  - **resume**: Resume training from checkpoint. Example: `false`

## Training

### Use Config Files

- **Run with default config**:
    ```bash
    python scripts/train_model.py --config config/default_config.yaml
    ```