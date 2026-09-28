#!/bin/bash
# =============================================================================
# EchoJEPA -> DeepCORO_CLIP two-stage GPU benchmark
#
#   Stage 1  CLIP-style video/report contrastive pretraining with the EchoJEPA
#            ViT-L coronary encoder as the video backbone (config/clip/echojepa_*).
#   Stage 2  Linear probing (18-target stenosis regression, multi-video MIL) on
#            the Stage-1 encoder (config/linear_probing/stenosis/echojepa_*).
#
# Both stages run under torchrun on the selected GPUs through scripts/runner.sh.
# nvidia-smi is sampled every 5 s into gpu_log.csv for the whole run and a
# per-stage utilisation / memory / throughput summary is printed at the end.
#
# Usage:
#   bash scripts/run_echojepa_benchmark.sh [--selected_gpus 0,1,2,3]
#        [--clip_config PATH] [--lp_config PATH] [--skip_clip CKPT] [--skip_lp]
# =============================================================================
set -uo pipefail

SELECTED_GPUS="0,1,2,3"
CLIP_CONFIG="config/clip/echojepa_vitl_clip_benchmark.yaml"
LP_CONFIG="config/linear_probing/stenosis/echojepa_vitl_stenosis_benchmark.yaml"
SKIP_CLIP_CKPT=""
SKIP_LP="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        --selected_gpus) SELECTED_GPUS="$2"; shift 2 ;;
        --clip_config)   CLIP_CONFIG="$2"; shift 2 ;;
        --lp_config)     LP_CONFIG="$2"; shift 2 ;;
        --skip_clip)     SKIP_CLIP_CKPT="$2"; shift 2 ;;   # reuse an existing CLIP checkpoint
        --skip_lp)       SKIP_LP="true"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

cd "$(dirname "$0")/.."
REPO=$(pwd)
TS=$(date +%Y%m%d-%H%M%S)
LOG_DIR="$REPO/logs/echojepa_benchmark_$TS"
mkdir -p "$LOG_DIR"
echo "Logs: $LOG_DIR"
echo "$SELECTED_GPUS" > "$LOG_DIR/gpus.txt"

# ---- GPU sampler --------------------------------------------------------------
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv -l 5 > "$LOG_DIR/gpu_log.csv" 2>/dev/null &
SMI_PID=$!
cleanup() { kill "$SMI_PID" 2>/dev/null || true; }
trap cleanup EXIT

stage_marker() { date +%s > "$LOG_DIR/$1"; }

# ---- Stage 1: CLIP contrastive pretraining ------------------------------------
CLIP_CKPT="$SKIP_CLIP_CKPT"
if [ -z "$CLIP_CKPT" ]; then
    echo "=============== Stage 1: CLIP pretraining ($CLIP_CONFIG) ==============="
    stage_marker clip_start
    bash scripts/runner.sh --selected_gpus "$SELECTED_GPUS" --base_config "$CLIP_CONFIG" \
        --run_mode train --use_wandb false 2>&1 | tee "$LOG_DIR/clip.log"
    CLIP_STATUS=${PIPESTATUS[0]}
    stage_marker clip_end
    if [ "$CLIP_STATUS" -ne 0 ]; then
        echo "Stage 1 failed (exit $CLIP_STATUS); see $LOG_DIR/clip.log"; exit "$CLIP_STATUS"
    fi
    # The run prints "Output directory: <dir>"; the checkpoint lives under <dir>/checkpoints.
    CLIP_OUT=$(grep -m1 -oP "Output directory: \K.*" "$LOG_DIR/clip.log" | tr -d '\r')
    CLIP_CKPT=$(ls -t "$CLIP_OUT"/checkpoints/best_model_epoch_*.pt 2>/dev/null | head -1)
    [ -z "$CLIP_CKPT" ] && CLIP_CKPT="$CLIP_OUT/checkpoints/checkpoint.pt"
    echo "Stage 1 checkpoint: $CLIP_CKPT"
    echo "$CLIP_CKPT" > "$LOG_DIR/clip_checkpoint.txt"
fi
if [ ! -f "$CLIP_CKPT" ]; then
    echo "No CLIP checkpoint found at '$CLIP_CKPT'"; exit 1
fi

# ---- Stage 2: linear probing --------------------------------------------------
if [ "$SKIP_LP" = "false" ]; then
    echo "=============== Stage 2: linear probing ($LP_CONFIG) ==============="
    python scripts/set_yaml_key.py "$LP_CONFIG" video_encoder_checkpoint_path "!!str $CLIP_CKPT"
    stage_marker lp_start
    bash scripts/runner.sh --selected_gpus "$SELECTED_GPUS" --base_config "$LP_CONFIG" \
        --run_mode train --use_wandb false 2>&1 | tee "$LOG_DIR/lp.log"
    LP_STATUS=${PIPESTATUS[0]}
    stage_marker lp_end
    if [ "$LP_STATUS" -ne 0 ]; then
        echo "Stage 2 failed (exit $LP_STATUS); see $LOG_DIR/lp.log"; exit "$LP_STATUS"
    fi
fi

cleanup
python scripts/summarize_gpu_benchmark.py "$LOG_DIR" | tee "$LOG_DIR/summary.txt"
echo "Done. Summary: $LOG_DIR/summary.txt"
