#!/bin/bash
# Multi-GPU training with torchrun (alternative to DeepSpeed launcher)
# Usage: ./scripts/run_torchrun.sh [num_gpus] [config]

set -e

# Default values
NUM_GPUS=${1:-8}
CONFIG=${2:-"pokemon_ai/configs/h100_8gpu.yaml"}

echo "=================================="
echo "Pokemon AI - Torchrun Training"
echo "=================================="
echo "GPUs: $NUM_GPUS"
echo "Config: $CONFIG"
echo ""

# Run with torchrun
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    scripts/train.py \
    --config "$CONFIG" \
    "${@:3}"
