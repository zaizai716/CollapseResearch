#!/bin/bash
# Setup script for data attribution experiments
# Run this after cloning the repo to apply patches and install dependencies

set -e

echo "=== Setting up Data Attribution Experiments ==="

# 1. Install Python dependencies
echo "Installing Python dependencies..."
pip install transformers torch accelerate datasets peft bitsandbytes tqdm scipy huggingface_hub -q

# 2. Clone RapidIn if submodule is empty
if [ ! -f "RapidIn/MP_main.py" ]; then
    echo "Cloning RapidIn..."
    rm -rf RapidIn
    git clone https://github.com/huawei-lin/RapidIn.git RapidIn
fi

# 3. Apply patches to RapidIn
echo "Applying patches to RapidIn..."
cp patches/engine_with_s_test_cache.py RapidIn/RapidIn/engine.py
cp patches/data_loader_with_qwen_support.py RapidIn/RapidIn/data_loader.py
echo "  - engine.py: s_test caching support"
echo "  - data_loader.py: Qwen model and LM format support"

# 4. Create necessary directories
echo "Creating directories..."
mkdir -p attribution_work/s_test_cache
mkdir -p attribution_work/gradients/qwen/gen_{0,1,2,3,4}
mkdir -p attribution_work/data/qwen/gen_{0,1,2,3,4}
mkdir -p attribution_work/checkpoints/qwen/gen_{0,1,2,3,4}
mkdir -p attribution_results/qwen/gen_{0,1,2,3,4}
mkdir -p analysis/{figures,tables,reports}
mkdir -p logs

# 5. Verify setup
echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Download data from HuggingFace (zaizaiiiii/model-collapse-experiment)"
echo "2. Run: python RapidIn/MP_main.py --config_path attribution_work/configs/qwen_gen_0_config.json"
echo ""
echo "To monitor progress:"
echo "  bash monitor_attribution.sh"
