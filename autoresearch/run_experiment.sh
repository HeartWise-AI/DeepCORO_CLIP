#!/bin/bash
set -euo pipefail

cd /volume/DeepCORO_CLIP

export CUDA_VISIBLE_DEVICES=3

timeout 5400 torchrun --nproc_per_node=1 --master_port=29503 \
    scripts/main.py --base_config autoresearch/experiment.yaml
