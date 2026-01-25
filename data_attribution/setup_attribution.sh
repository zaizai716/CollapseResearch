#!/bin/bash
# Setup script for data attribution experiments
# Run this after cloning the repo to apply patches and install dependencies

set -e

echo "=== Setting up Data Attribution Experiments ==="

# 1. Install Python dependencies
echo "Installing Python dependencies..."
pip install transformers torch accelerate datasets peft bitsandbytes tqdm scipy -q

# 2. Apply engine.py patch for s_test caching
echo "Applying s_test cache patch to RapidIn..."
cp patches/engine_with_s_test_cache.py RapidIn/RapidIn/engine.py

# 3. Create necessary directories
echo "Creating directories..."
mkdir -p attribution_work/s_test_cache
mkdir -p attribution_work/gradients/qwen/gen_0
mkdir -p attribution_work/gradients/qwen/gen_1
mkdir -p attribution_work/gradients/qwen/gen_2
mkdir -p attribution_work/gradients/qwen/gen_3
mkdir -p attribution_work/gradients/qwen/gen_4
mkdir -p attribution_results/qwen/gen_0
mkdir -p attribution_results/qwen/gen_1
mkdir -p attribution_results/qwen/gen_2
mkdir -p attribution_results/qwen/gen_3
mkdir -p attribution_results/qwen/gen_4
mkdir -p logs

# 4. Verify setup
echo ""
echo "=== Setup Complete ==="
echo "To run attribution experiments:"
echo "  python RapidIn/MP_main.py --config_path attribution_work/configs/qwen_gen_0_config.json"
echo ""
echo "To monitor progress:"
echo "  bash monitor_attribution.sh"
