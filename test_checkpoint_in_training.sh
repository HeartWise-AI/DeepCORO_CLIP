#!/bin/bash
# Test script to verify checkpoint loading in training

echo "Testing checkpoint loading in training..."
echo "========================================="

# Activate environment
source .venv/bin/activate

# Run training for just 1 step to see initialization messages
timeout 60 bash scripts/runner.sh \
    --base_config config/clip/multitask_config.yaml \
    --selected_gpus 0 \
    --use_wandb false \
    --run_mode train 2>&1 | \
    grep -E "mvit_rope|RoPE|CHECKPOINT|Checkpoint|Loading|LOADED|✅" | \
    head -20

echo ""
echo "========================================="
echo "Check the output above for:"
echo "1. [VideoEncoder] Will load checkpoint from: ..."
echo "2. [VideoEncoder] ========== CHECKPOINT LOADING =========="
echo "3. [VideoEncoder] ✅ CHECKPOINT LOADED SUCCESSFULLY!"
echo ""