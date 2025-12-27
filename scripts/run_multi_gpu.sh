#!/bin/bash
# Multi-GPU training with DeepSpeed on H100 cluster
# Usage: ./scripts/run_multi_gpu.sh [num_gpus] [config]

set -e

# Default values
NUM_GPUS=${1:-8}
CONFIG=${2:-"pokemon_ai/configs/h100_8gpu.yaml"}

# Activate environment if needed
# source /path/to/venv/bin/activate

echo "=================================="
echo "Pokemon AI - Multi-GPU Training"
echo "=================================="
echo "GPUs: $NUM_GPUS"
echo "Config: $CONFIG"
echo ""

# Run with DeepSpeed
deepspeed --num_gpus=$NUM_GPUS \
    scripts/train.py \
    --config "$CONFIG" \
    "${@:3}"
