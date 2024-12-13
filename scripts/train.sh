#!/bin/bash

# Usage: ./train.sh <num_gpus>
NUM_GPUS=${1:-1}  # Default to 1 if not specified

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train_model_multi_gpu.py \
    --config configs/training_config.yaml 