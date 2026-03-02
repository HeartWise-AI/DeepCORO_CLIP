#!/bin/bash
# Reproducibility tests: run inference on 100-study samples with different batch sizes
# Usage: bash scripts/run_reproducibility_tests.sh <model> <gpu>
# model: zcb8cu0l or sarra
# gpu: GPU device number

set -e

MODEL=$1
GPU=$2
BASE_PORT=29520

if [ "$MODEL" == "zcb8cu0l" ]; then
    BASE_CONFIG="config/linear_probing/stenosis/zcb8cu0l_inference_config.yaml"
    SAMPLE_CSV="/volume/DeepCORO_CLIP/outputs/sample_100_zcb8cu0l.csv"
    OUTPUT_BASE="outputs/reproducibility/zcb8cu0l"
elif [ "$MODEL" == "sarra" ]; then
    BASE_CONFIG="config/linear_probing/stenosis/sarra_inference_config.yaml"
    SAMPLE_CSV="/volume/DeepCORO_CLIP/outputs/sample_100_sarra.csv"
    OUTPUT_BASE="outputs/reproducibility/sarra"
else
    echo "Usage: $0 <zcb8cu0l|sarra> <gpu>"
    exit 1
fi

cd /volume/DeepCORO_CLIP

for BS in 1 2 4 12; do
    echo "============================================"
    echo "Running $MODEL with batch_size=$BS on GPU $GPU"
    echo "============================================"

    PORT=$((BASE_PORT + BS))
    OUTPUT_DIR="${OUTPUT_BASE}/bs${BS}"

    PYTHONPATH=/volume/DeepCORO_CLIP \
    LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 \
    MASTER_ADDR=localhost MASTER_PORT=$PORT \
    CUDA_VISIBLE_DEVICES=$GPU \
    python scripts/main.py \
        --base_config "$BASE_CONFIG" \
        --data_filename "$SAMPLE_CSV" \
        --batch_size $BS \
        --output_dir "$OUTPUT_DIR"

    echo "Completed batch_size=$BS -> $OUTPUT_DIR"
    echo ""
done

echo "All reproducibility tests completed for $MODEL"
