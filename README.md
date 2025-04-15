# DeepCORO_CLIP

DeepCORO_CLIP is a deep learning model for echocardiography video interpretation using contrastive learning. It leverages a Multiscale Vision Transformer (mVIT) for video encoding and BioMedBERT for text encoding, trained on millions of video-report pairs.

## 🚀 Features

- **Contrastive Learning**: Train on video-report pairs using CLIP-style contrastive learning
  - Single video mode: Process one video per study
  - Multi-video mode: Process multiple videos per study with aggregation
- **Linear Probing**: Fine-tune the model for specific tasks using linear probing
- **Multi-GPU Training**: Support for distributed training across multiple GPUs
- **Hyperparameter Optimization**: Built-in support for Weights & Biases sweeps
- **Automatic Mixed Precision**: Optimized training with AMP
- **Distributed Data Parallel**: Efficient multi-GPU training

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

2. **Linear Probing** (`config/linear_probing/base_config.yaml`):
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

## 💻 Training Modes

### 1. Contrastive Learning (CLIP)

Train the model on video-report pairs using contrastive learning:

Process multiple videos per study with aggregation:
```bash
# Single GPU training without logging results to wandb (see scripts/runner.sh)
bash scripts/runner.sh --base_config config/clip/base_config.yaml --selected_gpus 0 --use_wandb false --run_mode train

# Multi-GPU training with results logging on wandb (see scripts/runner.sh)
bash scripts/runner.sh --base_config config/clip/base_config.yaml --selected_gpus 0,1 --use_wandb true --run_mode train

# Multi-GPU hyperparameters fine-tuning - RunMode and UseWandb are forced to train and true respectively (see scripts/run_sweep.sh)
bash scripts/run_sweep.sh --base_config config/clip/base_config.yaml --sweep_config config/clip/sweep_config_single_video.yaml --selected_gpus 0,1 --count 5
```

### 2. Linear Probing

Fine-tune the model for specific tasks using linear probing - couple of combination examples:

```bash
# Single GPU training without logging results to wandb (see script/runner.sh)
bash scripts/runner.sh --base_config config/linear_probing/base_config.yaml --selected_gpus 0 --use_wandb false --run_mode train

# Multi-GPU training with results logging on wandb (see script/runner.sh)
bash scripts/runner.sh --base_config config/linear_probing/base_config.yaml --selected_gpus 0,1 --use_wandb true --run_mode train

# Multi-GPU hyperparameters fine-tuning - RunMode and UseWandb are forced to train and true respectively (see scripts/run_sweep.sh)
bash scripts/run_sweep.sh --base_config config/linear_probing/base_config.yaml --sweep_config config/linear_probing/sweep_config.yaml --selected_gpus 0,1 --count 5
```

## Model Architecture

### Video Encoder
- Multiscale Vision Transformer (mVIT) backbone
- Configurable number of heads and layers
- Support for pretrained weights
- Optional backbone freezing

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
   - Reduce batch size
   - Use gradient accumulation
   - Force single GPU mode

2. **GPU Selection**:
   - Use `CUDA_VISIBLE_DEVICES` to select specific GPUs
   - Monitor GPU usage with `nvidia-smi`

3. **Training Speed**:
   - Multi-GPU isn't always faster due to overhead
   - Start with single GPU and scale up if needed

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
│   └── linear_probing/           # Linear probing configs
├── dataloaders/                  # Data loading modules
├── models/                       # Neural network models
├── projects/                     # Project implementations
├── runners/                      # Training runners
├── scripts/                      # Training scripts
└── utils/                        # Utility functions
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
