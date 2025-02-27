#!/bin/bash

# Define color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print usage
print_usage() {
    echo "Usage: $0 --selected_gpus GPU_IDS --base_config CONFIG_PATH"
    echo "Example:"
    echo "  $0 --selected_gpus 0,1,2,3 --base_config configs/default_config.yaml"
    echo "  $0 --selected_gpus 1,3 --base_config configs/my_config.yaml"
    echo ""
    echo "Arguments:"
    echo "  --selected_gpus  Comma-separated list of GPU IDs to use"
    echo "  --base_config    Path to the base configuration file"
    echo "  --run_mode       Run mode to use (supported: train, inference)"
    echo "  --use_wandb      Use W&B for logging (true/false)"
    echo "  --help, -h       Display this help message"
    exit 1
}

# Default values
SELECTED_GPUS="0"
CONFIG_PATH="config/clip/base_config.yaml"
RUN_MODE="train"
USE_WANDB="false"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --selected_gpus)
            SELECTED_GPUS="$2"
            shift 2
            ;;
        --base_config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --run_mode)
            RUN_MODE="$2"
            shift 2
            ;;
        --use_wandb)
            USE_WANDB="$2"
            shift 2
            ;;
        --help | -h)
            print_usage
            ;;
        *)
            echo "Unknown parameter: $1"
            print_usage
            ;;
    esac
done

# Check if required arguments are provided
if [ -z "$SELECTED_GPUS" ] || [ -z "$CONFIG_PATH" ]; then
    echo -e "${RED}Error: Missing required arguments${NC}"
    print_usage
fi

# Check if config file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo -e "${RED}Error: Config file not found at $CONFIG_PATH${NC}"
    exit 1
fi

# Check if run_mode is supported
if [ "$RUN_MODE" != "train" ] && [ "$RUN_MODE" != "inference" ]; then
    echo -e "${RED}Error: Unsupported run mode: $RUN_MODE${NC}"
    print_usage
fi

# Check if use_wandb is supported
if [ "$USE_WANDB" != "true" ] && [ "$USE_WANDB" != "false" ]; then
    echo -e "${RED}Error: Unsupported use_wandb: $USE_WANDB${NC}"
    print_usage
fi

# Check if yq is installed
if ! command -v yq &> /dev/null
then
    echo -e "${RED}Error: yq is not installed. Please install yq to proceed.${NC}"
    echo -e "${YELLOW}Installation instructions: https://github.com/mikefarah/yq/#install${NC}"
    exit 1
fi

echo -e "${BLUE}========================================${NC}"

# Setup run_mode in base config
echo -e "${GREEN}Setting run_mode in $CONFIG_PATH to $RUN_MODE${NC}"
yq eval -i ".run_mode = \"$RUN_MODE\"" "$CONFIG_PATH"

# Setup use_wandb in base config
echo -e "${GREEN}Setting use_wandb in $CONFIG_PATH to $USE_WANDB${NC}"
yq eval -i ".use_wandb = $USE_WANDB" "$CONFIG_PATH"

echo -e "${BLUE}========================================\n${NC}"

# Print training configuration
echo -e "${BLUE}Starting training with:${NC}"
echo "Selected GPUs: $SELECTED_GPUS (Total: $NUM_GPUS GPUs)"
echo "Config path: $CONFIG_PATH"

# Calculate number of GPUs from the comma-separated list
NUM_GPUS=$(echo $SELECTED_GPUS | tr ',' '\n' | wc -l)

# Environment variables for better DDP performance
export NCCL_DEBUG=WARNING
export CUDA_VISIBLE_DEVICES=$SELECTED_GPUS
export OMP_NUM_THREADS=1

# Run the training
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    --nnodes=1 \
    --node_rank=0 \
    scripts/main.py \
    --base_config $CONFIG_PATH