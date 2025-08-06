# Claude Development Guide

## Environment Setup

### Activating Virtual Environment
Always activate the virtual environment before running any Python scripts:
```bash
source .venv/bin/activate
```

## Running Sweeps

### Single GPU Training
```bash
# Activate environment first
source .venv/bin/activate 

#and then
bash scripts/run_sweep.sh --base_config config/clip/multitask_config.yaml --sweep_config config/clip/sweep_config_multitask.yaml --selected_gpus 0,1 --count 10

```

### Multi-GPU Training with Sweep
```bash
# Activate environment first
source .venv/bin/activate

# Run sweep on specific GPUs
bash scripts/run_sweep.sh --base_config config/clip/multitask_config.yaml --sweep_config config/clip/sweep_config_multitask.yaml --selected_gpus 0,1 --count 10
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
- The sweep script automatically handles distributed training setup
- Monitor GPU memory usage to avoid OOM errors