#!/bin/bash
# RunPod GPU Setup Script for Model Collapse + Data Attribution Experiments
# Run this after cloning the repo on RunPod

echo "========================================================"
echo "RUNPOD GPU SETUP FOR MODEL COLLAPSE RESEARCH"
echo "========================================================"

# Update system
echo "1. Updating system packages..."
apt-get update && apt-get install -y git wget curl vim htop

# Check CUDA
echo "2. Checking CUDA installation..."
nvidia-smi
nvcc --version

# Setup Python environment
echo "3. Setting up Python environment..."
conda create -n collapse_research python=3.10 -y
source activate collapse_research

# Install PyTorch with CUDA support
echo "4. Installing PyTorch with CUDA..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies for model collapse experiments
echo "5. Installing model collapse dependencies..."
pip install transformers==4.36.0 datasets accelerate pytorch-lightning huggingface-hub
pip install numpy pandas matplotlib seaborn tqdm

# Install dependencies for RapidIn (data attribution)
echo "6. Installing RapidIn dependencies..."
cd data_attribution/RapidIn
pip install -r requirements.txt
cd ../..

# Install additional tools
echo "7. Installing additional tools..."
pip install deepspeed wandb tensorboard

# Setup HuggingFace
echo "8. Setting up HuggingFace..."
echo "Please set your HuggingFace token:"
echo "export HF_TOKEN='your-token-here'"

# Create necessary directories
echo "9. Creating directory structure..."
mkdir -p model_collapse_experiments/checkpoints
mkdir -p model_collapse_experiments/results
mkdir -p data_attribution/datasets
mkdir -p data_attribution/gradients_cache
mkdir -p data_attribution/attribution_results

# Download WikiText dataset
echo "10. Downloading WikiText-2 dataset..."
cd model_collapse_experiments
python -c "from datasets import load_dataset; dataset = load_dataset('wikitext', 'wikitext-2-raw-v1'); print('WikiText-2 downloaded')"
cd ..

# Create DeepSpeed config for efficient training
echo "11. Creating DeepSpeed configuration..."
cat > data_attribution/deepspeed_config.json << EOF
{
  "train_batch_size": "auto",
  "gradient_accumulation_steps": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": true
  },
  "gradient_clipping": 1.0,
  "wall_clock_breakdown": false
}
EOF

# Create run scripts
echo "12. Creating run scripts..."

# Script for model collapse experiment
cat > run_collapse_experiment.sh << 'EOF'
#!/bin/bash
source activate collapse_research
cd model_collapse_experiments

echo "Running Model Collapse Experiment..."
echo "Generation: $1"

if [ -z "$1" ]; then
  echo "Usage: ./run_collapse_experiment.sh <generation_number>"
  exit 1
fi

python run_nature_experiment.py --generation $1
EOF
chmod +x run_collapse_experiment.sh

# Script for data attribution
cat > run_attribution.sh << 'EOF'
#!/bin/bash
source activate collapse_research
cd data_attribution

echo "Running Data Attribution Analysis..."
python run_attribution_experiment.py
EOF
chmod +x run_attribution.sh

# Create README with instructions
cat > README_RUNPOD.md << 'EOF'
# RunPod GPU Setup Complete!

## Environment
- Conda environment: `collapse_research`
- CUDA-enabled PyTorch installed
- All dependencies installed

## Directory Structure
```
├── model_collapse_experiments/  # Collapse experiments
│   ├── checkpoints/
│   ├── results/
│   └── Zakahler-curse_recurse-b48c90a/
├── data_attribution/            # Attribution analysis
│   ├── RapidIn/
│   ├── datasets/
│   └── attribution_results/
```

## Running Experiments

### 1. Model Collapse Experiment
```bash
# Set HuggingFace token
export HF_TOKEN='your-token-here'

# Run specific generation
./run_collapse_experiment.sh 0  # For generation 0

# Or run complete experiment
cd model_collapse_experiments
python run_nature_experiment.py
```

### 2. Data Attribution Analysis
```bash
# Prepare datasets
cd data_attribution
python prepare_collapse_data.py

# Run attribution analysis
./run_attribution.sh
```

### 3. Monitor GPU Usage
```bash
watch -n 1 nvidia-smi
```

### 4. Check Results
- Collapse results: `model_collapse_experiments/results/`
- Attribution results: `data_attribution/attribution_results/`

## Notes
- Ensure GPU memory is sufficient (min 24GB recommended)
- Use DeepSpeed for larger models if needed
- Results auto-upload to HuggingFace if token is set
EOF

echo ""
echo "========================================================"
echo "SETUP COMPLETE!"
echo "========================================================"
echo ""
echo "Next steps:"
echo "1. Set your HuggingFace token: export HF_TOKEN='your-token'"
echo "2. Activate environment: source activate collapse_research"
echo "3. Run collapse experiment: ./run_collapse_experiment.sh 0"
echo "4. Run attribution analysis: ./run_attribution.sh"
echo ""
echo "See README_RUNPOD.md for detailed instructions"