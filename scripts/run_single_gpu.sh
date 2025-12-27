#!/bin/bash
# Single GPU training on H100
# Usage: ./scripts/run_single_gpu.sh [config_override]

set -e

# Default config
CONFIG=${1:-"pokemon_ai/configs/h100_1gpu.yaml"}

# Activate environment if needed
# source /path/to/venv/bin/activate

echo "=================================="
echo "Pokemon AI - Single GPU Training"
echo "=================================="
echo "Config: $CONFIG"
echo ""

# Run training
python scripts/train.py \
    --config "$CONFIG" \
    "$@"
