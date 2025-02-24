#!/bin/bash

# runner.sh
# Activate environment
python3 -m venv .venv
source .venv/bin/activate  # On Linux/Mac

# Execute torchrun with single GPU/node
torchrun --standalone --nnodes=1 --nproc_per_node=1 "$1" --config "$2" "${@:3}"