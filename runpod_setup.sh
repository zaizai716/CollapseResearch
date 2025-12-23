#!/bin/bash
# RunPod Setup Script for Nature Model Collapse Experiment
# Run this on a RunPod instance with RTX 4090 or A100 GPU

echo "=========================================="
echo "Nature Model Collapse - RunPod Setup"
echo "=========================================="

# Check GPU availability first
echo ""
echo "1. Checking GPU availability..."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
if [ $? -ne 0 ]; then
    echo "ERROR: No GPU detected! Make sure you selected a GPU instance on RunPod."
    exit 1
fi

# Clone repository if not already present
echo ""
echo "2. Setting up repository..."
if [ ! -d "Nature_Model_Collapse" ]; then
    git clone https://github.com/zaizai716/CollapseResearch.git Nature_Model_Collapse
    cd Nature_Model_Collapse
else
    cd Nature_Model_Collapse
    git pull
fi

# Install Python dependencies
echo ""
echo "3. Installing dependencies..."
pip install --upgrade pip

# Install PyTorch with CUDA support
# RunPod templates come with PyTorch pre-installed, but we'll ensure the right version
# Check if PyTorch is already installed with CUDA
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "PyTorch with CUDA already installed and working!"
else
    echo "Installing PyTorch with CUDA 12.4 support..."
    pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
fi

# Install other required packages
pip install pytorch-lightning transformers datasets numpy tqdm tensorboard tensorboardX hf_transfer accelerate

# Verify GPU is working with PyTorch
echo ""
echo "4. Verifying GPU setup..."
python3 -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU Name:', torch.cuda.get_device_name(0))
    print('GPU Memory:', torch.cuda.get_device_properties(0).total_memory / 1024**3, 'GB')
else:
    print('ERROR: CUDA not available in PyTorch!')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "ERROR: GPU verification failed!"
    exit 1
fi

# Create necessary directories
echo ""
echo "5. Creating directories..."
mkdir -p nature_exact_experiment
mkdir -p graphs
mkdir -p data_cache_dir

# Display instructions
echo ""
echo "=========================================="
echo "Setup Complete! ✓"
echo "=========================================="
echo ""
echo "To run the experiment:"
echo ""
echo "Option 1: Run in screen (recommended for long experiments):"
echo "  screen -S experiment"
echo "  python run_nature_experiment.py"
echo "  # Press Ctrl+A then D to detach"
echo "  # Use 'screen -r experiment' to reattach"
echo ""
echo "Option 2: Run directly with monitoring:"
echo "  python run_nature_experiment.py 2>&1 | tee experiment_log.txt"
echo ""
echo "Option 3: Run a single generation test first:"
echo "  python Zakahler-curse_recurse-b48c90a/main.py \\"
echo "    --model_tag facebook/opt-125m \\"
echo "    --batch-size 128 \\"
echo "    --learning-rate 2e-5 \\"
echo "    --max-epochs 5 \\"
echo "    --save-name test_gen/ \\"
echo "    --pretrained"
echo ""
echo "Monitor GPU usage in another terminal:"
echo "  watch -n 1 nvidia-smi"
echo ""