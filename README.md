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
   https://github.com/HeartWise-AI/DeepCORO_CLIP.git
   cd DeepCORO_CLIP
   ```

2. **Set up Virtual Environment**:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Linux/Mac
   pip install --upgrade pip
   pip install uv
   ```

2. **Install Dependencies**:
   Install dependencies:

   ```bash
   # Using uv
   uv sync
   ```

3. **Install yq required for sweep**:

   ```bash
   wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/bin/yq && \
   chmod +x /usr/bin/yq
   ```

4. **Log into Weights & Biases (optional)**:

   ```bash
   wandb login
   ```

5. If CV2 error:

```CV2 error:  libGL.so.1: cannot open shared object file

```

```bash
sudo apt update
sudo apt install libgl1-mesa-glx
sudo apt-get update
sudo apt-get install libglib2.0-0
sudo ldconfig
```

6. Make sure you have h264 installed: `ffmpeg -encoders | grep -w libx264` should return a list of encoders like this ``` V....D libx264              libx264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 (codec h264)
 V....D libx264rgb           libx264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 RGB (codec h264)```

 If not you must install C compiler and then get FFMPEG v.60 with H264 and then compile it yourself.
 a. Get c compiler:
 ```apt-get update
apt-get install -y gcc g++ make git pkg-config autoconf automake libtool \
                   bzip2 cmake libfreetype6-dev zlib1g-dev yasm nasm```
	1.	Download the x264 source:
                   ```cd /usr/local/src
# Download a release (e.g., FFmpeg 6.0) or the latest snapshot:
wget https://ffmpeg.org/releases/ffmpeg-6.0.tar.bz2
tar xjf ffmpeg-6.0.tar.bz2
cd ffmpeg-6.0```

	2.	Configure x264:
  ``./configure --prefix=/usr/local --enable-gpl --enable-libx264```

3. 	Compile and install FFmpeg:
```make -j$(nproc)
make install```
4. 	Update the dynamic linker run-time bindings:
```export PATH="/usr/local/bin:$PATH"```
  ```ldconfig```

7. 	Verify the installation:
```ffmpeg -encoders | grep -w libx264```




## Configuration Files

The project uses two main types of configuration files located in the `config/` directory:

### Base Configuration (`base_config.yaml`)

Contains the default settings for training and model parameters:

```yaml
# Training parameters
epochs: 10
num_workers: 12
debug: false
use_amp: true
mode: train  

# Dataset parameters
data_filename: data/reports/reports_with_splits_subset.csv
frames: 16
stride: 2
multi_video: true
num_videos: 5

# Model parameters
model_name: mvit
pretrained: true

# Data augmentation
rand_augment: false
resize: 224
apply_mask: false
```

### Sweep Configuration (`sweep_config.yaml`)

Used for hyperparameter optimization with Weights & Biases:

```yaml
method: bayes
metric:
  name: val/loss
  goal: minimize

parameters:
  lr:
    distribution: log_uniform_values
    min: 1e-5
    max: 1e-4
  optimizer:
    values: ["AdamW", "RAdam"]
  scheduler_type:
    values: ["cosine", "step"]
  batch_size:
    values: [10, 12]
  temperature:
    min: 0.05
    max: 0.15
    distribution: uniform
  dropout:
    distribution: uniform
    min: 0.0
    max: 0.5
  video_freeze_ratio:
    values: [0.0, 0.5, 0.8, 1.0]
  text_freeze_ratio:
    values: [0.0, 0.5, 1.0]
```

### Running Hyperparameter Sweeps

The project provides a convenient script `scripts/run_sweep.sh` for running hyperparameter sweeps:

```bash
# Basic usage
./scripts/run_sweep.sh --selected_gpus 0,1 --sweep_config config/sweep_config.yaml --count 5

# Arguments:
#   --selected_gpus    Comma-separated list of GPU IDs to use (default: 1,3)
#   --sweep_config     Path to the sweep configuration file (default: config/sweep_config.yaml)
#   --count           Number of runs to execute (default: 5)
```

The script will:
1. Automatically update the sweep config with the correct number of GPUs
2. Initialize a W&B sweep and generate a sweep ID
3. Start the sweep agent with the specified number of runs
4. Log all outputs to `logs/sweep_<SWEEP_ID>_<timestamp>.log`

You can monitor your sweep progress at: `https://wandb.ai/<entity>/<project>/sweeps/<SWEEP_ID>`

#### Manual Sweep Setup

Alternatively, you can run sweeps manually:

```bash
# Initialize the sweep
wandb sweep config/sweep_config.yaml

# Start an agent (replace SWEEP_ID with the ID from the previous command)
wandb agent your_entity/your_project/SWEEP_ID
```

The sweep uses Bayesian optimization to find the best hyperparameters, with early termination using the Hyperband algorithm to stop underperforming runs.

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
torchrun --nproc_per_node=1 scripts/train_model_multi_gpu.py --base_config config/base_config.yaml
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
## Parameters

### Training Parameters
- `epochs`: Number of training epochs (default: 50)
- `batch_size`: Samples per batch (default: 12) 
- `num_workers`: Subprocesses for data loading (default: 16)
- `learning_rate`: Optimizer learning rate (default: 0.0001)
- `temperature`: Contrastive loss temperature (default: 0.07)

### Data Parameters
- `data_filename`: CSV file containing data (default: `data/reports/reports_sampled_no_conclusion.csv`)
- `root`: Root data directory (default: ".")
- `target_label`: Target text column name (default: `Report`)
- `datapoint_loc_label`: File paths column name (default: `FileName`)
- `frames`: Frames to sample per video (default: 16)
- `stride`: Frame sampling stride (default: 2)

### Model Parameters
- `model_name`: Video backbone model (default: `mvit`)
- `pretrained`: Use pretrained model (default: `true`)

### Checkpointing
- `resume`: Resume from checkpoint (default: `false`)
- `resume_checkpoint`: Path to checkpoint file for resuming training
- `save_best`: Best model save criterion (default: `loss`)

### Optimization Parameters
- `optimizer`: Optimizer type (default: `RAdam`)
- `weight_decay`: Optimizer weight decay (default: 0.000001)
- `scheduler_type`: Learning rate scheduler (default: `step`)
- `lr_step_period`: Learning rate step period (default: 15)
- `factor`: Scheduler factor (default: 0.3)

### Distributed Training
- `gpu`: GPU index (default: 1)
- `local_rank`: Local rank for distributed training (default: -1)

### Logging Parameters
- `project`: W&B project name (default: `deepCORO_CLIP`)
- `entity`: W&B entity name (default: `mhi_ai`)
- `tag`: W&B run tag (default: `DeepCORO_Clip_Sweep_Learnable_Temp_Full`)

### Multi-Video Parameters
- `multi_video`: Enable multiple videos per study (default: `true`)
- `max_num_videos`: Maximum videos per study (default: 5)
- `groupby_column`: Column for grouping videos by study (default: `StudyInstanceUID`)
- `shuffle_videos`: Randomly sample videos within groups (default: `true`)
- `seed`: Random seed for reproducibility (default: 42)

### Additional Parameters
- `output_dir`: Output directory (default: `outputs`)
- `use_amp`: Use automatic mixed precision (default: `true`)
- `device`: Training device (default: `cuda`)


### Data Augmentation
- `random_augment`: Enable random augmentations (default: `true`)
- `resize`: Image resize dimension (default: 224)
- `apply_mask`: Apply image masking (default: `false`)
- `period`: Num of frames of sampling stride (default: 1)


### Use Config Files

- **Run with default config**:
    ```bash
    python scripts/train_model.py --config config/default_config.yaml
    ```