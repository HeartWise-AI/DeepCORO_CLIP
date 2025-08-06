#!/bin/bash

# Optimized training script with memory management settings

# Set CUDA memory allocator options to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256

# Optional: Set to use less memory for CUDA graphs (if using)
export CUDA_LAUNCH_BLOCKING=0

# Run training
echo "Starting optimized training with memory management..."
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"

# Run the main training script
python scripts/main.py \
    --base_config config/clip/multitask_config.yaml \
    "$@"  # Pass any additional arguments

echo "Training completed."