#!/bin/bash

set -euo pipefail  # Enhance script safety

# Define color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_usage() {
    echo "Usage: $0 --base_config BASE_CONFIG_PATH --sweep_config SWEEP_CONFIG_PATH --selected_gpus GPU_IDS --count COUNT"
    echo ""
    echo "Examples:"
    echo "  # Run 5 sweep agents on GPU 3 with single video config"
    echo "  $0 --base_config config/clip/base_config.yaml --sweep_config config/clip/sweep_config_single_video.yaml --selected_gpus 3 --count 5"
    echo ""
    echo "  # Run 10 sweep agents on GPUs 0,1,2,3 with multitask config"
    echo "  $0 --base_config config/clip/multitask_config.yaml --sweep_config config/clip/sweep_config_multitask.yaml --selected_gpus 0,1,2,3 --count 10"
    echo ""
    echo "Arguments:"
    echo "  --base_config      Path to the base configuration file (required)"
    echo "  --sweep_config     Path to the sweep configuration file (required)"
    echo "  --selected_gpus    Comma-separated list of GPU IDs to use (required)"
    echo "  --count           Number of runs to execute (required)"
    echo "  --help, -h         Display this help message"
    exit 1
}

# Default values - all are required
SELECTED_GPUS=""
BASE_CONFIG_PATH=""
SWEEP_CONFIG_PATH=""
COUNT=""

# Activate virtual environment
source .venv/bin/activate

# Set NCCL environment variables for better stability with H200s
export NCCL_DEBUG=INFO
export NCCL_IB_TIMEOUT=1800  # Reduced to 30 minutes
export NCCL_SOCKET_TIMEOUT=1800  # Reduced to 30 minutes
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5
export NCCL_IB_TC=106
export NCCL_IB_SL=0
export NCCL_IB_AR_THRESHOLD=4096  # Reduced from 8192
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_TIMEOUT_MS=1800000  # 30 minutes in milliseconds
export NCCL_SOCKET_NTHREADS=2  # Reduced from 4
export NCCL_NSOCKS_PERTHREAD=2  # Reduced from 4
export NCCL_BUFFSIZE=1048576  # Reduced to 1MB
export NCCL_IB_RETRY_CNT=5  # Reduced from 7
export NCCL_IB_QUEUE_LEN=2048  # Reduced from 4096

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --selected_gpus)
            SELECTED_GPUS="$2"
            shift 2
            ;;
        --base_config)
            BASE_CONFIG_PATH="$2"
            shift 2
            ;;
        --sweep_config)
            SWEEP_CONFIG_PATH="$2"
            shift 2
            ;;
        --count)
            COUNT="$2"
            shift 2
            ;;
        --help|-h)
            print_usage
            ;;
        *)
            echo "Unknown parameter: $1"
            print_usage
            ;;
    esac
done

# Check if required arguments are provided
if [ -z "${SELECTED_GPUS}" ] || [ -z "${SWEEP_CONFIG_PATH}" ] || [ -z "${BASE_CONFIG_PATH}" ] || [ -z "${COUNT}" ]; then
    echo -e "${RED}Error: Missing required arguments${NC}"
    echo ""
    print_usage
fi

# Check if sweep config exists
if [ ! -f "${SWEEP_CONFIG_PATH}" ]; then
    echo "Error: Sweep config file not found at ${SWEEP_CONFIG_PATH}"
    exit 1
fi

# Check if base config exists
if [ ! -f "${BASE_CONFIG_PATH}" ]; then
    echo "Error: Base config file not found at ${BASE_CONFIG_PATH}"
    exit 1
fi

# Check if yq is installed
if ! command -v yq &> /dev/null
then
    echo "Error: yq is not installed. Please install yq to proceed."
    echo "Installation instructions: https://github.com/mikefarah/yq/#install"
    exit 1
fi

# Setup base config
# Check if run_mode is set to train
echo -e "${BLUE}Loading base configuration from ${BASE_CONFIG_PATH}...${NC}"
MODE=$(yq e '.run_mode' "${BASE_CONFIG_PATH}")
USE_WANDB=$(yq e '.use_wandb' "${BASE_CONFIG_PATH}")
if [ "${MODE}" != "train" ]; then
    echo -e "${YELLOW}Warning: Base config must be set to TRAIN mode${NC} (current run_mode: ${MODE})"
    yq eval -i '.run_mode = "train"' "${BASE_CONFIG_PATH}"
    MODE=$(yq e '.run_mode' "${BASE_CONFIG_PATH}")
    if [ "${MODE}" != "train" ]; then
        echo -e "${RED}Error: Failed to set run_mode to train${NC}"
        exit 1
    else
        echo -e "${GREEN}Successfully set run_mode to train${NC}"
    fi
fi

# Check if use_wandb is set to true
if [ "${USE_WANDB}" != "true" ]; then
    echo -e "${YELLOW}Warning: Base config must be set to use W&B${NC} (current use_wandb: ${USE_WANDB})"
    yq eval -i '.use_wandb = true' "${BASE_CONFIG_PATH}"
    USE_WANDB=$(yq e '.use_wandb' "${BASE_CONFIG_PATH}")
    if [ "${USE_WANDB}" != "true" ]; then
        echo -e "${RED}Error: Failed to set use_wandb to true${NC}"
        exit 1
    else
        echo -e "${GREEN}Successfully set use_wandb to true${NC}"
    fi
fi

# Calculate number of GPUs
NUM_GPUS=$(echo "${SELECTED_GPUS}" | tr ',' '\n' | wc -l)

# Update the sweep config with the correct number of GPUs
# This assumes your sweep config is in YAML format
echo -e "${BLUE}Updating --nproc_per_node in ${SWEEP_CONFIG_PATH}...${NC}"
if sed -i "s/--nproc_per_node=[0-9]*/--nproc_per_node=$NUM_GPUS/" "${SWEEP_CONFIG_PATH}"; then
    echo -e "${GREEN}Updated --nproc_per_node to $NUM_GPUS in ${SWEEP_CONFIG_PATH}${NC}"
    echo ""
else
    echo -e "${RED}Failed to update --nproc_per_node in ${SWEEP_CONFIG_PATH}${NC}"
    exit 1
fi

# Update the sweep config to use the base_config path given via script args.
echo -e "${BLUE}Updating base_config path in ${SWEEP_CONFIG_PATH}...${NC}"
if sed -i '/--base_config/{n;s|.*|  - "'"$BASE_CONFIG_PATH"'"|;}' "${SWEEP_CONFIG_PATH}"; then
    echo -e "${GREEN}Updated base_config path to $BASE_CONFIG_PATH in ${SWEEP_CONFIG_PATH}${NC}"
else
    echo -e "${RED}Failed to update base_config path in ${SWEEP_CONFIG_PATH}${NC}"
    exit 1
fi

# Extract configuration fields using yq
mapfile -t COMMANDS < <(yq e '.command[]' "${SWEEP_CONFIG_PATH}")
NAME=$(yq e '.name' "${SWEEP_CONFIG_PATH}" | tr -d "'")
PROJECT=$(yq e '.project' "${SWEEP_CONFIG_PATH}" | tr -d "'")
ENTITY=$(yq e '.entity' "${SWEEP_CONFIG_PATH}" | tr -d "'")

# Validate extracted values
if [ -z "${NAME}" ] || [ -z "${PROJECT}" ] || [ -z "${ENTITY}" ]; then
    echo -e "${RED}Error: Failed to extract required values from sweep config${NC}"
    echo -e "${YELLOW}Please ensure name, project, and entity are properly defined in ${SWEEP_CONFIG_PATH}${NC}"
    exit 1
fi

# Join commands into a single string
COMMAND_STR=$(IFS=' '; echo "${COMMANDS[*]}")

# Print extracted configuration with colors
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Extracted Configuration:${NC}"
echo -e "${GREEN}Name: ${NC}$NAME"
echo -e "${GREEN}Project: ${NC}$PROJECT"
echo -e "${GREEN}Entity: ${NC}$ENTITY"
echo -e "${YELLOW}Commands:${NC}"
for i in "${!COMMANDS[@]}"; do
    echo -e "  Command $((i + 1)): ${COMMANDS[i]}"
done
echo -e "${BLUE}========================================${NC}"
echo ""

# Print training configuration
echo -e "${BLUE}Starting training with:${NC}"
echo "Selected GPUs: ${SELECTED_GPUS} (Total: ${NUM_GPUS} GPUs)"
echo "Sweep config path: ${SWEEP_CONFIG_PATH}"
echo "Count: ${COUNT}"
echo ""

# Environment variables for better DDP performance
export NCCL_DEBUG=WARNING
export CUDA_VISIBLE_DEVICES="${SELECTED_GPUS}"
export OMP_NUM_THREADS=1

# Run the sweep and extract the SWEEP_ID while displaying logs
echo -e "${BLUE}Initializing W&B Sweep...${NC}"
if ! SWEEP_OUTPUT=$(wandb sweep "${SWEEP_CONFIG_PATH}" 2>&1); then
    echo -e "${RED}Error: wandb sweep command failed${NC}"
    echo "${SWEEP_OUTPUT}"
    exit 1
fi

# Extract the Sweep ID using Perl-compatible regex for robustness
SWEEP_ID=$(echo "${SWEEP_OUTPUT}" | grep -oP '([a-z0-9]+)$' | tail -n1)

# Verify that the SWEEP_ID was extracted
if [ -n "${SWEEP_ID}" ]; then
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}Sweep successfully initialized with ID: ${SWEEP_ID}${NC}"
    echo -e "${YELLOW}You can view the sweep at: https://wandb.ai/${ENTITY}/${PROJECT}/sweeps/${SWEEP_ID}${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""

    # Create logs directory if it doesn't exist
    mkdir -p logs

    # Define log file
    LOG_FILE="logs/sweep_${SWEEP_ID}_$(date +%Y%m%d_%H%M%S).log"
    echo "Logs can be found in ${LOG_FILE}"
    echo ""
    # Starting a sweep agent
    echo -e "${BLUE}Starting Sweep Agent...${NC}"
    echo ""
    if [ -n "${COUNT}" ]; then
        wandb agent --count "${COUNT}" "${ENTITY}/${PROJECT}/${SWEEP_ID}" | tee -a "${LOG_FILE}"
    else
        wandb agent "${ENTITY}/${PROJECT}/${SWEEP_ID}" | tee -a "${LOG_FILE}"
    fi

else
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${RED}Failed to extract Sweep ID.${NC}"
    echo -e "${YELLOW}Check the logs at ${LOG_FILE} for more details.${NC}"
    echo -e "${BLUE}========================================${NC}"
    exit 1
fi