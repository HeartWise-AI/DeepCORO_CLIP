# DeepCORO_CLIP

DeepCORO_CLIP is a deep learning model for echocardiography video interpretation using contrastive learning. It leverages a Multiscale Vision Transformer (mVIT) for video encoding and BioMedBERT for text encoding, trained on millions of video-report pairs.

## 🚀 Features

- **Contrastive Learning**: Train on video-report pairs using CLIP-style contrastive learning
  - Single video mode: Process one video per study
  - Multi-video mode: Process multiple videos per study with aggregation
- **Multitask Learning (SigLIP 2-Inspired)**: Advanced training setup combining three complementary objectives
  - **LocCa-style Captioning Decoder**: Generate structured angiographic reports from video embeddings using transformer decoder with cross-attention
  - **Masked Video Modeling (MVM)**: Self-supervised learning through 75% masked patch reconstruction for enhanced spatial understanding
  - **Contrastive Learning**: Simultaneous video-text alignment with sigmoid or softmax loss
  - **Configurable Loss Weights**: Flexible weighting and optional dynamic scheduling for each task
  - **Component-Specific Learning Rates**: Independent optimization for encoder, decoder, and MVM components
  - **Comprehensive W&B Logging**: Real-time monitoring of all loss components, gradients, and metrics
- **Linear Probing**: Fine-tune the model for specific tasks using linear probing
- **Multi-GPU Training**: Support for distributed training across multiple GPUs
- **Hyperparameter Optimization**: Built-in support for Weights & Biases sweeps
- **Automatic Mixed Precision**: Optimized training with AMP
- **Distributed Data Parallel**: Efficient multi-GPU training
- **Patch- vs. Video-level Reasoning**: Expose *all* patch tokens, a single
  token per video, or a single token per study with two simple flags
  (`aggregate` and `per_video_pool`) in the `VideoEncoder`.

## 🛠️ Environment Setup

### Prerequisites

- **CUDA-capable GPU**
- **Python 3.11+**

### Steps

1. 📥 **Clone the Repository**:

   ```bash
   https://github.com/HeartWise-AI/DeepCORO_CLIP.git
   cd DeepCORO_CLIP
   ```

2. **Set up Virtual Environment**:

   ```bash
   pip install uv
   uv sync
   ```

3. **Activate Virtual Environment**:

   ```bash
   source .venv/bin/activate
   ```
   
4. **Install yq required to run scripts/run_sweep.sh**:

   ```bash
   wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/bin/yq && \
   chmod +x /usr/bin/yq
   ```

5. **Log into Weights & Biases required for sweep**:

   ```bash
   wandb login
   ```

6. **Make sure you have FFMPEG 4.4.x is installed - required for sweep**:
   ```bash
   which ffmpeg
   conda remove ffmpeg # remove if /opt/conda/bin/ffmpeg exists
   sudo apt update
   sudo apt install ffmpeg
   sudo apt install libavcodec-extra
   ffmpeg -version
   ```

## 📄 Configuration Files

The project uses configuration files located in the `config/` directory:

### Base Configurations

1. **CLIP Training** (`config/clip/base_config.yaml`):
   - Training parameters (epochs, batch size, learning rate)
   - Model architecture settings
   - Data loading parameters
   - Optimization settings
   - Video mode settings (single/multi)
   - Video aggregation parameters

2. **Multitask Training** (`config/clip/multitask_config.yaml`):
   - Advanced training combining contrastive, captioning, and masked modeling
   - Captioning decoder parameters (layers, heads, vocabulary)
   - Masked video modeling parameters (mask ratio, decoder architecture)
   - Configurable loss weights and learning rates for each component
   - Biomedical tokenizer integration

3. **Linear Probing** (`config/linear_probing/base_config.yaml`):
   - Task-specific parameters
   - Head structure configuration
   - Loss function settings
   - Backbone freezing options

### Sweep Configurations

1. **CLIP Training** (`config/clip/sweep_config_*.yaml`):
   - Hyperparameter search space for CLIP training
   - Supports both single and multi-video training

2. **Linear Probing** (`config/linear_probing/sweep_config.yaml`):
   - Hyperparameter optimization for linear probing tasks
   - Task-specific parameter ranges

## 🔬 Hyperparameter Optimization with W&B Sweeps

The project includes comprehensive support for hyperparameter optimization using Weights & Biases sweeps. This allows systematic exploration of the hyperparameter space to find optimal configurations.

### How to Run Sweeps

```bash
bash scripts/run_sweep.sh --base_config BASE_CONFIG --sweep_config SWEEP_CONFIG --selected_gpus GPU_IDS --count NUM_RUNS
```

**Arguments:**
- `--base_config`: Path to the base configuration file (required)
- `--sweep_config`: Path to the W&B sweep configuration file (required)
- `--selected_gpus`: Comma-separated list of GPU IDs to use (required)
- `--count`: Number of sweep agents to run (required)

**Notes:**
- The script automatically sets `run_mode=train` and `use_wandb=true`
- Each agent will try one hyperparameter configuration
- Results are tracked in real-time on the W&B dashboard
- Best hyperparameters are highlighted in the sweep summary

### Available Sweep Configurations

1. **Single Video CLIP**: `config/clip/sweep_config_single_video.yaml`
2. **Multi-Video CLIP**: `config/clip/sweep_config_multi_video.yaml`
3. **Multitask Learning**: `config/clip/sweep_config_multitask.yaml`
4. **Linear Probing**: `config/linear_probing/sweep_config.yaml`

## 💻 Run Modes

### 1. Contrastive Learning (CLIP)

#### Train the model on video-report pairs using contrastive learning

Process multiple videos per study with aggregation:
```bash
# Single GPU training without logging results to wandb (see scripts/runner.sh)
bash scripts/runner.sh --base_config config/clip/base_config.yaml --selected_gpus 0 --use_wandb false --run_mode train

# Multi-GPU training with results logging on wandb (see scripts/runner.sh)
bash scripts/runner.sh --base_config config/clip/base_config.yaml --selected_gpus 0,1 --use_wandb true --run_mode train

# Hyperparameter optimization with W&B sweeps (see scripts/run_sweep.sh for details)
# RunMode and UseWandB are automatically set to 'train' and 'true' respectively

# Example 1: Run 5 sweep agents on GPU 3 with single video config
bash scripts/run_sweep.sh --base_config config/clip/base_config.yaml --sweep_config config/clip/sweep_config_single_video.yaml --selected_gpus 3 --count 5

# Example 2: Run 10 sweep agents on GPUs 0,1,2,3 with multi-video config  
bash scripts/run_sweep.sh --base_config config/clip/base_config.yaml --sweep_config config/clip/sweep_config_multi_video.yaml --selected_gpus 0,1,2,3 --count 10
```

### 2. Multitask Learning (SigLIP 2-Inspired Architecture)

#### Train the model with advanced multitask setup combining three complementary objectives:

```bash
# Test the multitask setup before training
python test_multitask_setup.py

# Single GPU training with multitask learning
bash scripts/runner.sh --base_config config/clip/multitask_config.yaml --selected_gpus 0 --use_wandb false --run_mode train

# Multi-GPU training with multitask learning (recommended)
bash scripts/runner.sh --base_config config/clip/multitask_config.yaml --selected_gpus 0,1 --use_wandb true --run_mode train

# Training with custom batch size (reduce if OOM)
bash scripts/runner.sh --base_config config/clip/multitask_config.yaml --selected_gpus 0,1,2,3 --use_wandb true --run_mode train --batch_size 8

# Hyperparameter optimization for multitask learning
bash scripts/run_sweep.sh --base_config config/clip/multitask_config.yaml --sweep_config config/clip/sweep_config_multitask.yaml --selected_gpus 0,1,2,3 --count 5
```

**Key Components:**

##### 1. **Contrastive Learning** (Video-Text Alignment)
- Aligns video and text representations in shared embedding space
- Supports both sigmoid (SigLIP-style) and softmax (CLIP-style) loss
- Configurable temperature parameter for similarity scaling

##### 2. **LocCa-style Captioning Decoder** (Report Generation)
- Generates structured angiographic reports from video embeddings
- Transformer decoder with cross-attention to video tokens
- Biomedical tokenizer (PubMedBERT) for medical terminology
- Autoregressive generation with beam search support

##### 3. **Masked Video Modeling** (Self-Supervised Learning)
- Randomly masks 75% of video patches during training
- Lightweight decoder reconstructs masked tokens
- Improves spatial understanding without additional annotations

**Configuration Options** (`config/clip/multitask_config.yaml`):

```yaml
# Loss weights for each task
loss_weights:
  contrastive: 1.0      # Video-text alignment
  captioning: 1.0       # Report generation
  masked_modeling: 0.1  # Self-supervised learning

# Component-specific learning rates
lr: 0.00006                    # Base learning rate
text_lr: 0.00002              # Text encoder learning rate
captioning_lr: 0.00006        # Captioning decoder learning rate  
mvm_lr: 0.000006              # Masked modeling decoder learning rate

# Captioning decoder parameters
decoder_layers: 6             # Number of transformer layers
decoder_heads: 8              # Number of attention heads
max_generation_length: 128   # Maximum caption length

# Masked video modeling parameters
mask_ratio: 0.75             # Fraction of patches to mask
mvm_decoder_layers: 2        # Decoder depth
mvm_decoder_hidden_size: 256 # Decoder hidden size
```

**Monitoring Training (W&B Metrics):**
- **Step-wise Logging**: Individual loss components logged per batch
- **Gradient Norms**: Monitor training stability
- **Learning Rates**: Track per-component learning rates
- **Validation Metrics**: Recall@K, NDCG, caption quality
- **Loss Weights**: Dynamic weight scheduling if enabled

**Recommended Settings:**
- **Batch Size**: 12 (reduce to 8 for GPUs with <24GB memory)
- **GPUs**: 2-4 GPUs for optimal training speed
- **Mixed Precision**: Enabled by default (`use_amp: true`)
- **Gradient Accumulation**: Set to 2-4 if using smaller batch sizes

### Run validation
**Not supported**

#### Run inference
Process validation data from input CSV (rows where **Split == 'inference'**) - **working on single GPU only**
```bash
bash scripts/runner.sh --selected_gpus 0 --base_config config/clip/base_config.yaml --run_mode inference --use_wandb false
```

### Run test
**Not supported**

### 3. Linear Probing

Fine-tune the model for specific tasks using linear probing - couple of combination examples:

```bash
# Single GPU training without logging results to wandb (see script/runner.sh)
bash scripts/runner.sh --base_config config/linear_probing/base_config.yaml --selected_gpus 0 --use_wandb false --run_mode train

# Multi-GPU training with results logging on wandb (see script/runner.sh)
bash scripts/runner.sh --base_config config/linear_probing/base_config.yaml --selected_gpus 0,1 --use_wandb true --run_mode train

# Multi-GPU hyperparameters fine-tuning - RunMode and UseWandb are forced to train and true respectively (see scripts/run_sweep.sh)
bash scripts/run_sweep.sh --base_config config/linear_probing/base_config.yaml --sweep_config config/linear_probing/sweep_config.yaml --selected_gpus 0,1 --count 5

```

#### Run validation
Process validation data from input CSV (rows where **Split == 'val'**) and compute CI for each head
``` bash
bash scripts/runner.sh --use_wandb false --base_config config/linear_probing/stenosis/base_config_stenosis_2vue.yaml --run_mode val --selected_gpus 1,2,3
```

#### Run test 
Process validation data from input CSV (rows where **Split == 'test'**) and compute CI for each head
``` bash
bash scripts/runner.sh --use_wandb false --base_config config/linear_probing/stenosis/base_config_stenosis_2vue.yaml --run_mode test --selected_gpus 1,2,3
```

### Run inference
Process validation data from input CSV (rows where **Split == 'inference'**)
``` bash
bash scripts/runner.sh --use_wandb false --base_config config/linear_probing/stenosis/base_config_stenosis_2vue.yaml --run_mode inference --selected_gpus 1,2,3
```

## 🐳 Docker Setup
Optionally, you can build a Docker container to run training, validation, and inference pipelines. 
For the **validation pipeline**, please set up your huggingface API key in `api_key.json` as weights will be publicly available only upon publication.

### Build Docker Image
``` bash
docker build -t deepcoro_clip-docker .
```

### Run Docker
**Requirements:**
* Make sure your CSV file is in the data folder : `$(pwd)/data` can be replaced by the absolute path to that folder
* Create a folder results : `$(pwd)/results` can be replaced by the absolute path to that folder
* Make sure you have a column `FileName` with root path fined as `/app/videos` : `$(pwd)/videos` can be replaced by the absolute path to your base video folder path
``` bash
docker run -it --gpus all --shm-size=32g --memory=64g --ipc=host --network=host -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results -v $(pwd)/videos:/app/videos deepcoro_clip-docker
```
**Inside the container:**
Once connected to the docker container:
1. **For validation and inference:** Follow step 3. only from the `Environment Setup` section above
2. **For training:** Follow step 3. 5. and 6. from the `Environment Setup` section above
3. **Download pretrained weights:**
``` bash
python utils/download_pretrained_weights.py
```
The pretrained weights will be in the folder `/app/pretrained_models`.

4. **Run your pipeline:** Select the appropriate command from the `Run Modes` section above


## Model Architecture

### Video Encoder
- Multiscale Vision Transformer (mVIT) backbone
- Configurable number of heads and layers
- Support for pretrained weights
- Optional backbone freezing
- New flags for fine-grained control over the output:
  * `aggregate=True` (default) – returns **one** study-level vector `[B, D]`.
  * `aggregate=False, per_video_pool=True` – returns one token **per video**
    `[B, N, D]`, ready for MIL / linear probing heads.
  * `aggregate=False, per_video_pool=False` – returns **all patch tokens - ONLY Setting that preeservs all the tokens**
    `[B, N·L, D]` for the most detailed downstream reasoning.

### Captioning Decoder (Multitask)
- LocCa-style transformer decoder for structured angiographic report generation
- Causal attention for autoregressive generation
- Cross-attention to video tokens
- Biomedical tokenizer integration (PubMedBERT)
- Configurable architecture (layers, heads, hidden size)

### Masked Video Modeling (Multitask)
- Self-supervised masked patch modeling
- Random token masking with configurable ratio
- Lightweight decoder for reconstruction
- MSE loss on masked tokens only
- Improves video representation learning
    


Example (video-level MIL):

```python
from models.video_encoder import VideoEncoder
from models.multi_instance_linear_probing import MultiInstanceLinearProbing

encoder = VideoEncoder(
    backbone="mvit",
    aggregate=False,        # skip internal aggregator
    aggregate_videos_tokens=True,    # one token per video
)

probe = MultiInstanceLinearProbing(
    embedding_dim=encoder.embedding_dim,
    head_structure={"severity": 4},
    pooling_mode="attention",
)

video_batch = ...                  # [B, N, T, H, W, C]
feats = encoder(video_batch)       # [B, N, D]
logits = probe(feats)              # dict with head output
```

### Text Encoder
- BioMedBERT for medical text encoding
- Configurable freezing ratio
- Contrastive learning with video features

### Linear Probing Heads
- Task-specific classification heads
- Configurable dropout and architecture
- Support for multiple output classes per head

## Development Setup

We use pre-commit hooks to ensure code quality and consistency:

```bash
# Install pre-commit
uv pip install pre-commit
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

### Performance Guidelines

#### Recommended Batch Sizes by GPU Memory

| GPU Memory | Recommended Batch Size | Command           |
| ---------- | ---------------------- | ----------------- |
| 8GB        | 4-8                    | `--batch-size 8`  |
| 12GB       | 8-16                   | `--batch-size 16` |
| 16GB       | 16-24                  | `--batch-size 24` |
| 24GB+      | 24-32                  | `--batch-size 32` |

#### Training Tips

1. **Batch Size Selection**:
   - Start with smaller batch sizes and increase if memory allows
   - Larger batch sizes generally allow faster training
   - Reduce if you get OOM errors

2. **Number of Workers**:
   - Rule of thumb: `num_workers = 4 * num_gpus`
   - Reduce if you get memory or file handle errors
   - Example: `--num-workers 2` for slower storage systems

3. **Learning Rate**:
   - Default (1e-4) works well for most cases
   - For larger batch sizes: `lr = 1e-4 * (batch_size/32)`
   - Example: `--lr 2e-4` for batch size 64

4. **Number of Epochs**:
   - Default (50) is good for most cases
   - Increase for better performance: `--epochs 100`
   - Decrease for quick experiments: `--epochs 10`

### Common Issues

1. **Out of Memory (OOM)**:
   - Reduce batch size (e.g., `--batch_size 8` or `--batch_size 4`)
   - Use gradient accumulation (`gradient_accumulation_steps: 2` in config)
   - Force single GPU mode (`--selected_gpus 0`)
   - For multitask: Reduce decoder layers or hidden sizes

2. **GPU Selection**:
   - Use `CUDA_VISIBLE_DEVICES` to select specific GPUs
   - Monitor GPU usage with `nvidia-smi`
   - Ensure GPUs are not already in use

3. **Training Speed**:
   - Multi-GPU isn't always faster due to overhead
   - Start with single GPU and scale up if needed
   - For multitask: Consider reducing mask_ratio to speed up MVM

4. **Multitask Training Issues**:
   - **High captioning loss**: Normal in early epochs, should decrease over time
   - **Unstable gradients**: Reduce learning rates or increase gradient clipping
   - **Imbalanced losses**: Adjust loss_weights in config
   - **Slow convergence**: Increase batch size or use gradient accumulation

5. **Configuration Errors**:
   - Ensure `pipeline_project: DeepCORO_multitask` is set in multitask config
   - Check that all required parameters are present in the YAML file
   - Verify dataset paths and file existence

## Monitoring Training

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

## Project Structure

```
heartwise-ai-deepcoro_clip/
├── config/                        # Configuration files
│   ├── clip/                     # CLIP training configs
│   │   ├── base_config.yaml      # Base CLIP training
│   │   └── multitask_config.yaml # Multitask training
│   └── linear_probing/           # Linear probing configs
├── dataloaders/                  # Data loading modules
├── dataset_creation/             # How MHI dataset was built
├── docs/                         # Documentation
│   └── MULTITASK_SETUP.md       # Multitask setup documentation
├── models/                       # Neural network models
│   ├── captioning_decoder.py     # LocCa-style captioning decoder
│   ├── masked_video_modeling.py # Self-supervised masked modeling
│   └── ...                      # Other models
├── projects/                     # Project implementations
│   ├── multitask_pretraining_project.py # Multitask training project
│   └── ...                      # Other projects
├── runners/                      # Training runners
│   ├── multitask_runner.py       # Multitask training runner
│   └── ...                      # Other runners
├── scripts/                      # Training scripts
├── utils/                        # Utility functions
│   └── loss/
│       └── multitask_loss.py    # Multitask loss function
├── tests/                        # Unit test pipeline
└── test_multitask_setup.py      # Multitask setup test script
```

## 🤝 Contributing

Contributions to DeepECG_Docker repository are welcome! Please follow these steps to contribute:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes and commit them with clear, descriptive messages
4. Push your changes to your fork
5. Submit a pull request to the main repository

## 📚 Citation

If you find this repository useful, please cite our work:

```
@article{,
  title={},
  author={},
  journal={},
  year={},
  publisher={}
}
