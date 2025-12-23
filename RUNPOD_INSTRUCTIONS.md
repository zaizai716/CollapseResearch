# RunPod Setup Instructions for Model Collapse Experiment

## ⚠️ IMPORTANT: GPU Requirements

**DO NOT USE RTX 5090** - It's not supported by PyTorch yet (as of Dec 2024)

**Recommended GPUs on RunPod:**
- RTX 4090 (best price/performance)
- A100 (40GB or 80GB)
- H100
- RTX 3090

## Quick Start

### 1. Create RunPod Instance
1. Go to [RunPod.io](https://runpod.io)
2. Select **RTX 4090** GPU (NOT RTX 5090!)
3. Choose a template with PyTorch 2.0+ or use base Ubuntu
4. Launch the instance

### 2. Connect to Instance
```bash
ssh root@[your-instance-ip] -i [your-ssh-key]
# Or use RunPod's web terminal
```

### 3. Run Setup Script
```bash
# Clone the repository
git clone https://github.com/zaizai716/CollapseResearch.git Nature_Model_Collapse
cd Nature_Model_Collapse

# Run the setup script
bash runpod_setup.sh
```

### 4. Test GPU Setup
```bash
# Verify GPU is working properly
python test_gpu.py
```

You should see:
- ✓ CUDA is available
- ✓ GPU detected (RTX 4090 or similar)
- ✓ Model loads successfully
- ✓ Inference works

### 5. Run the Experiment

**Option A: Full Experiment (5+ hours)**
```bash
# Use screen to prevent disconnection
screen -S experiment
python run_nature_experiment.py
# Press Ctrl+A then D to detach
# Use 'screen -r experiment' to reattach later
```

**Option B: Quick Test (30 minutes)**
```bash
# Run just Generation 0
python Zakahler-curse_recurse-b48c90a/main.py \
  --model_tag facebook/opt-125m \
  --batch-size 128 \
  --learning-rate 2e-5 \
  --max-epochs 5 \
  --save-name test_gen/ \
  --pretrained
```

### 6. Monitor Progress
In a separate terminal:
```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Monitor logs
tail -f nature_exact_experiment/gen_0/logs.txt
```

## Troubleshooting

### If GPU not detected:
1. Check you selected GPU instance (not CPU)
2. Reinstall PyTorch:
```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```
3. Run test again: `python test_gpu.py`

### If "no kernel image" error:
- You're likely on RTX 5090 (unsupported)
- Terminate instance and select RTX 4090 instead

### If out of memory:
- Reduce batch size in run_nature_experiment.py
- Change line: `--batch-size 64` (instead of 128)

## Expected Results

Each generation takes ~30-60 minutes on RTX 4090:
- Generation 0: Perplexity ~27-30
- Generation 1: Perplexity ~35-40  
- Generation 5: Perplexity ~50-60

Total experiment time: 5-10 hours for 10 generations

## Files Created

- `nature_exact_experiment/gen_*/`: Checkpoints and models
- `graphs/`: Visualization plots
- `experiment_log.txt`: Full experiment output

## Cost Estimate

- RTX 4090 on RunPod: ~$0.40-0.60/hour
- Full 10-generation experiment: ~$3-6
- Quick test (1 generation): ~$0.50