#!/bin/bash
# Emergency PyTorch 2.6+ upgrade script for CVE-2025-32434 fix

echo "========================================"
echo "FORCE UPGRADING PyTorch to 2.6+"
echo "========================================"

# Check current version
echo ""
echo "Current PyTorch version:"
python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "Not installed"

# Completely remove old PyTorch
echo ""
echo "1. Removing old PyTorch installation..."
pip uninstall torch torchvision torchaudio -y

# Clear pip cache to ensure fresh install
echo ""
echo "2. Clearing pip cache..."
pip cache purge

# Install PyTorch 2.6.0 specifically (known working version)
echo ""
echo "3. Installing PyTorch 2.6.0..."
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Verify installation
echo ""
echo "4. Verifying installation..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
assert torch.__version__ >= '2.6.0', f'ERROR: PyTorch {torch.__version__} is still too old!'
assert torch.cuda.is_available(), 'ERROR: CUDA not available!'
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print('✓ PyTorch 2.6.0 successfully installed with CUDA support!')
"

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Installation failed! Trying alternative approach..."
    echo ""
    echo "Installing latest PyTorch..."
    pip install torch torchvision torchaudio --upgrade --index-url https://download.pytorch.org/whl/cu124
    
    python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
fi

echo ""
echo "========================================"
echo "Done! You can now run the experiment."
echo "========================================"