# Claude Development Guide

## Environment Setup

### Activating Virtual Environment
Always activate the virtual environment before running any Python scripts:
```bash
source .venv/bin/activate
```

## Running Training

### Direct Multi-GPU Training (without sweep)
```bash
# Activate environment and set NCCL variables for optimal multi-GPU performance
source .venv/bin/activate && \
export MASTER_PORT=29505 && \
export NCCL_P2P_LEVEL=NVL && \
export NCCL_ALGO=Tree && \
export NCCL_MIN_NCHANNELS=4 && \
export NCCL_SHM_DISABLE=0 && \
export NCCL_NET_GDR_LEVEL=PHB && \
torchrun --nproc_per_node=3 scripts/main.py --base_config config/clip/multitask_config.yaml
```

## Running Sweeps

### Single GPU Training
```bash
# Activate environment first
source .venv/bin/activate

# Run sweep (NCCL variables are set automatically in the script)
bash scripts/run_sweep.sh --base_config config/clip/multitask_config.yaml --sweep_config config/clip/sweep_config_multitask.yaml --selected_gpus 0 --count 10
```

### Multi-GPU Training with Sweep
```bash
# Activate environment first
source .venv/bin/activate

# Run sweep on multiple GPUs (NCCL variables are set automatically in the script)
bash scripts/run_sweep.sh --base_config config/clip/multitask_config.yaml --sweep_config config/clip/sweep_config_multitask.yaml --selected_gpus 0,1,2 --count 10

# Alternative: Explicitly set NCCL variables before sweep
source .venv/bin/activate && \
export MASTER_PORT=29505 && \
export NCCL_P2P_LEVEL=NVL && \
export NCCL_ALGO=Tree && \
export NCCL_MIN_NCHANNELS=4 && \
export NCCL_SHM_DISABLE=0 && \
export NCCL_NET_GDR_LEVEL=PHB && \
bash scripts/run_sweep.sh --base_config config/clip/multitask_config.yaml --sweep_config config/clip/sweep_config_multitask.yaml --selected_gpus 0,1,2 --count 10
```

### Sweep Arguments
- `--base_config`: Path to base configuration file (required)
- `--sweep_config`: Path to sweep configuration file (required) 
- `--selected_gpus`: Comma-separated GPU IDs to use (required)
- `--count`: Number of sweep runs to execute (required)

### Example Sweep Commands
```bash
# Run 5 agents on GPU 3 with single video config
bash scripts/run_sweep.sh --base_config config/clip/base_config.yaml --sweep_config config/clip/sweep_config_single_video.yaml --selected_gpus 3 --count 5

# Run 10 agents on GPUs 0,1,2,3 with multitask config
bash scripts/run_sweep.sh --base_config config/clip/multitask_config.yaml --sweep_config config/clip/sweep_config_multitask.yaml --selected_gpus 0,1,2,3 --count 10
```

## Important Commands

### Linting and Type Checking
```bash
# Run linter
npm run lint

# Run type checker
npm run typecheck
```

### Testing
```bash
# Run tests
pytest
```

## Notes
- Always ensure CUDA devices are available before running GPU training
- The sweep script automatically handles distributed training setup and NCCL environment variables
- Monitor GPU memory usage to avoid OOM errors
- Always use `source .venv/bin/activate` before running any scripts
- In Python dataclasses, fields with default values must come after fields without defaults

### NCCL Environment Variables
The following NCCL variables are configured for optimal multi-GPU training performance:
- `MASTER_PORT=29505`: Port for distributed training communication
- `NCCL_P2P_LEVEL=NVL`: Peer-to-peer communication level for GPUs
- `NCCL_ALGO=Tree`: Communication algorithm for collective operations
- `NCCL_MIN_NCHANNELS=4`: Minimum number of channels for communication
- `NCCL_SHM_DISABLE=0`: Enables shared memory (faster intra-node communication)
- `NCCL_NET_GDR_LEVEL=PHB`: GPU Direct RDMA level for network communication