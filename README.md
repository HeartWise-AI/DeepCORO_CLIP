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

## 💻 Run Modes

### 1. Contrastive Learning (CLIP)

#### Train the model on video-report pairs using contrastive learning

Process multiple videos per study with aggregation:
```bash
# Single GPU training without logging results to wandb (see scripts/runner.sh)
bash scripts/runner.sh --base_config config/clip/base_config.yaml --selected_gpus 0 --use_wandb false --run_mode train

# Multi-GPU training with results logging on wandb (see scripts/runner.sh)
bash scripts/runner.sh --base_config config/clip/base_config.yaml --selected_gpus 0,1 --use_wandb true --run_mode train

# Multi-GPU hyperparameters fine-tuning - RunMode and UseWandb are forced to train and true respectively (see scripts/run_sweep.sh)
bash scripts/run_sweep.sh --base_config config/clip/base_config.yaml --sweep_config config/clip/sweep_config_single_video.yaml --selected_gpus 3 --count 5
```

### Run validation
**Not supported**

#### Run inference
Process validation data from input CSV (rows where **Split == 'inference'**) - **working on single GPU only**
```bash
bash scripts/runner.sh --selected_gpus 0 --base_config config/clip/base_config.yaml --run_mode inference --use_wandb false
```

### Run test
**Not supported**

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

#### Run validation
Process validation data from input CSV (rows where **Split == 'val'**)
``` bash
bash scripts/runner.sh --use_wandb false --base_config config/linear_probing/stenosis/base_config_stenosis_2vue.yaml --run_mode val --selected_gpus 1,2,3
```

#### Run test 
Process validation data from input CSV (rows where **Split == 'test'**)
``` bash
bash scripts/runner.sh --use_wandb false --base_config config/linear_probing/stenosis/base_config_stenosis_2vue.yaml --run_mode test --selected_gpus 1,2,3
```

### Run inference
**Not supported**

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
``` bash
docker run -it --gpus all -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results deepcoro_clip-docker
```
** Inside the container: **
Once connected to the docker container:
1. **For validation and inference:** Follow step 3. only from the `Environment Setup` section above
2. **For training:** Follow step 3. 4. 5. and 6. from the `Environment Setup` section above
3. **Download pretrained weights:**
``` bash
python utils/download_pretrained_weights.py
```
The pretrained weights will be in the folder `/app/pretrained_models`
4. ** Run your pipeline: ** Select the appropriate command from the `Run Modes` section above


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
├── dataset_creation/             # How MHI dataset was built
├── docs/                         # Documentation on CLS-Token implementation
├── models/                       # Neural network models
├── projects/                     # Project implementations
├── runners/                      # Training runners
├── scripts/                      # Training scripts
├── utils/                        # Utility functions
└── tests/                        # Unit test pipeline
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
