# Context for Claude - Model Collapse Experiment Status

## Current Situation (Dec 25, 2024)
User is running the Nature paper's "Curse of Recursion" model collapse experiment on RunPod GPU instance.

## Directory Structure
- **Working Directory**: `/workspace/CollapseResearch` (on 184TB network storage)
- **Symlink**: `/CollapseResearch` -> `/workspace/CollapseResearch` (for compatibility)
- **NOT using**: Root filesystem `/` (only 20GB, gets full quickly)

## Key Problems Solved

### 1. PyTorch Version Issue (CVE-2025-32434)
- **Problem**: Transformers requires PyTorch 2.6+ due to security vulnerability
- **Solution**: Upgraded to PyTorch 2.6.0 in `runpod_setup.sh`

### 2. Batch Size Mismatch (8192 vs 128)
- **Problem**: Generated data had wrong shapes (input_ids was [1] instead of [64])
- **Root Cause**: Line 347 in main.py used `outputs[:, 64:]` instead of `outputs[:, -64:]`
- **Solution**: Fixed to take last 64 tokens: `preds = outputs[:, -64:]`
- **Also Fixed**: Padding token for OPT models: `tokenizer.pad_token = tokenizer.eos_token`

### 3. CUDA Multiprocessing Error
- **Problem**: DataLoader workers causing CUDA fork issues
- **Solution**: Set `num_workers=0` in main.py and run_nature_experiment.py

### 4. Disk Space Issues
- **Problem**: Root filesystem only 20GB, kept filling up
- **Solution**: Must work in `/workspace/` directory (184TB available)
- **Trap**: Had nested `/workspace/CollapseResearch/CollapseResearch/` causing confusion

## Current Code Status

### Files Modified from Original Nature Paper
1. `Zakahler-curse_recurse-b48c90a/main.py`:
   - Added padding token fix (line ~210)
   - Fixed generation slicing (line ~347)
   - Set num_workers=0 default (line ~119)
   - Force GPU usage when available (lines ~254-273)

2. `run_nature_experiment.py`:
   - Added `--num_workers 0` to command
   - Wrapper script to run Nature's code with correct parameters

### Helper Scripts Created
- `final_check.py` - Verifies all fixes are in place
- `simple_test.py` - Tests batch processing
- `test_gpu.py` - GPU verification
- `fix_pytorch.sh` - Force upgrades PyTorch to 2.6+

## Experiment Parameters (Nature Paper)
- **Model**: facebook/opt-125m
- **Dataset**: WikiText-2
- **Batch Size**: 128
- **Learning Rate**: 2e-5
- **Epochs**: 5 per generation
- **Generations**: 10 total
- **Block Size**: 64 tokens
- **Generation Method**: Beam search (5 beams, repetition penalty 3.0)

## How to Run
```bash
cd /workspace/CollapseResearch
python3 final_check.py  # Verify setup
python3 run_nature_experiment.py  # Run experiment
```

## Common Issues
1. **"No space left on device"** - Not using /workspace, using root filesystem
2. **"Expected batch_size 8192 vs 128"** - Generation creating wrong shaped data
3. **"No module named X"** - Wrong directory or symlink issues
4. **"Too many symbolic links"** - Recursive symlink created by accident

## Current Status
- Generation 0: Should complete successfully
- Generation 1+: Will work if all fixes are applied
- Must run from `/workspace/CollapseResearch` to avoid space issues
- All fixes have been pushed to GitHub repo: https://github.com/zaizai716/CollapseResearch