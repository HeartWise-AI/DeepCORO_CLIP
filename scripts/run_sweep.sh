#!/bin/bash

set -euo pipefail  # Enhance script safety

# Define color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_usage() {
    echo "Usage: $0 [--selected_gpus GPU_IDS] [--sweep_config SWEEP_CONFIG_PATH] [--count COUNT]"
    echo ""
    echo "Example:"
    echo "  $0 --selected_gpus 0,1,2,3 --sweep_config config/sweep_config.yaml --count 5"
    echo "  $0 --selected_gpus 1,3 --sweep_config config/sweep_config.yaml"
    echo ""
    echo "Arguments:"
    echo "  --selected_gpus    Comma-separated list of GPU IDs to use (default: 1,3)"
    echo "  --sweep_config     Path to the sweep configuration file (default: config/sweep_config.yaml)"
    echo "  --count           Number of runs to execute (default: 5)"
    echo "  --help, -h         Display this help message"
    exit 1
}

# Default values
SELECTED_GPUS="1,3" # Comma-separated list of GPU IDs to use
SWEEP_CONFIG_PATH="config/sweep_config_3.yaml"
COUNT="5" # Number of runs to execute

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --selected_gpus)
            SELECTED_GPUS="$2"
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
if [ -z "${SELECTED_GPUS}" ] || [ -z "${SWEEP_CONFIG_PATH}" ]; then
    echo "Error: Missing required arguments"
    print_usage
fi

# Check if sweep config exists
if [ ! -f "${SWEEP_CONFIG_PATH}" ]; then
    echo "Error: Sweep config file not found at ${SWEEP_CONFIG_PATH}"
    exit 1
fi

# Check if yq is installed
if ! command -v yq &> /dev/null
then
    echo "Error: yq is not installed. Please install yq to proceed."
    echo "Installation instructions: https://github.com/mikefarah/yq/#install"
    exit 1
fi

# Backup the original sweep config
cp "${SWEEP_CONFIG_PATH}" "${SWEEP_CONFIG_PATH}.bak"

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

# Extract configuration fields using yq
mapfile -t COMMANDS < <(yq e '.command[]' "${SWEEP_CONFIG_PATH}")
NAME=$(yq e '.parameters.name.values[]' "${SWEEP_CONFIG_PATH}" | tr -d "'")
PROJECT=$(yq e '.parameters.project.values[]' "${SWEEP_CONFIG_PATH}" | tr -d "'")
ENTITY=$(yq e '.parameters.entity.values[]' "${SWEEP_CONFIG_PATH}" | tr -d "'")

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
export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES="${SELECTED_GPUS}"
export OMP_NUM_THREADS=1

# Run the sweep and extract the SWEEP_ID while displaying logs
echo -e "${BLUE}Initializing W&B Sweep...${NC}"
SWEEP_OUTPUT=$(wandb sweep "${SWEEP_CONFIG_PATH}" 2>&1)

# Extract the Sweep ID using Perl-compatible regex for robustness
SWEEP_ID=$(echo "${SWEEP_OUTPUT}" | grep -oP '(?<=Creating sweep with ID: )\w+')

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
