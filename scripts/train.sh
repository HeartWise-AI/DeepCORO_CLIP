#!/bin/bash

# Function to print usage
print_usage() {
    echo "Usage: $0 --selected_gpus GPU_IDS --config_path CONFIG_PATH"
    echo "Example:"
    echo "  $0 --selected_gpus 0,1,2,3 --config_path configs/default_config.yaml"
    echo "  $0 --selected_gpus 1,3 --config_path configs/my_config.yaml"
    echo ""
    echo "Arguments:"
    echo "  --selected_gpus  Comma-separated list of GPU IDs to use"
    echo "  --config_path    Path to the configuration file"
    exit 1
}

# Default values
SELECTED_GPUS="0"
CONFIG_PATH="config/base_config.yaml"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --selected_gpus)
            SELECTED_GPUS="$2"
            shift 2
            ;;
        --config_path)
            CONFIG_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            print_usage
            ;;
    esac
done

# Check if required arguments are provided
if [ -z "$SELECTED_GPUS" ] || [ -z "$CONFIG_PATH" ]; then
    echo "Error: Missing required arguments"
    print_usage
fi

# Check if config file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found at $CONFIG_PATH"
    exit 1
fi

# Calculate number of GPUs from the comma-separated list
NUM_GPUS=$(echo $SELECTED_GPUS | tr ',' '\n' | wc -l)

# Print training configuration
echo "Starting training with:"
echo "Selected GPUs: $SELECTED_GPUS (Total: $NUM_GPUS GPUs)"
echo "Config path: $CONFIG_PATH"

# Environment variables for better DDP performance
export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=$SELECTED_GPUS
export OMP_NUM_THREADS=1

# Run the training
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    --nnodes=1 \
    --node_rank=0 \
    scripts/train_model_multi_gpu.py \
    --base_config $CONFIG_PATH