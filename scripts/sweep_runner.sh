#!/bin/bash
# Wrapper for wandb sweep runs that picks a random port to avoid conflicts
# Usage: Called by wandb agent via sweep config command
# Args: --nproc_per_node=1 --master_port=IGNORED scripts/main.py --base_config ...

# Replace the fixed master_port with a random one
RANDOM_PORT=$((29000 + RANDOM % 1000))
ARGS=()
for arg in "$@"; do
    if [[ "$arg" == --master_port=* ]]; then
        ARGS+=("--master_port=$RANDOM_PORT")
    else
        ARGS+=("$arg")
    fi
done

echo "Using random master_port=$RANDOM_PORT"
torchrun "${ARGS[@]}"
EXIT_CODE=$?

# Wait for GPU memory to be freed after the process ends
sleep 15

exit $EXIT_CODE
