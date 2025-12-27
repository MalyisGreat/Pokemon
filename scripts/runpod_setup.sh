#!/bin/bash
# RunPod H100 Setup Script - Fast Training
# Run this after starting your pod

set -e  # Exit on error

echo "=============================================="
echo "Pokemon AI - RunPod H100 Setup"
echo "=============================================="

# 1. Clone repo (if not already done)
if [ ! -d "Pokemon" ]; then
    echo "Cloning repository..."
    git clone https://github.com/MalyisGreat/Pokemon.git
fi
cd Pokemon

# 2. Install dependencies
echo ""
echo "Installing dependencies..."
pip install -e . --quiet
pip install huggingface_hub datasets lz4 flash-attn --no-build-isolation --quiet

# 3. Download Metamon dataset (3.5M trajectories)
echo ""
echo "=============================================="
echo "Downloading Metamon Dataset (3.5M trajectories)"
echo "=============================================="
python scripts/download_hf_data.py --dataset metamon

# 4. Convert to training format
echo ""
echo "Converting to training format..."
python scripts/convert_metamon_data.py --workers 12

# 5. Check data
echo ""
echo "Data ready:"
ls -lh data/replays/

# 6. Start training
echo ""
echo "=============================================="
echo "Starting Training"
echo "=============================================="
echo "Model: 200M params (base)"
echo "Data: 3.5M trajectories"
echo "Expected time: 6-12 hours"
echo ""

python scripts/train.py --config pokemon_ai/configs/h100_1gpu.yaml

echo ""
echo "=============================================="
echo "Training Complete!"
echo "=============================================="
echo "Checkpoint saved to: checkpoints/h100_base/"
