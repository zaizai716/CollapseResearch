#!/usr/bin/env python3
"""
Final verification script to ensure everything will work
"""
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("="*60)
print("FINAL PRE-FLIGHT CHECK FOR NATURE EXPERIMENT")
print("="*60)

all_good = True

# 1. Check PyTorch version
print("\n1. Checking PyTorch version...")
pytorch_version = torch.__version__
if pytorch_version >= "2.6.0":
    print(f"✓ PyTorch {pytorch_version} (>= 2.6.0 required)")
else:
    print(f"✗ PyTorch {pytorch_version} is too old! Need >= 2.6.0")
    all_good = False

# 2. Check CUDA availability
print("\n2. Checking GPU...")
if torch.cuda.is_available():
    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ WARNING: No GPU detected, will run on CPU (very slow)")

# 3. Check padding token fix is in place
print("\n3. Checking padding token fix in main.py...")
main_py_path = "Zakahler-curse_recurse-b48c90a/main.py"
if os.path.exists(main_py_path):
    with open(main_py_path, 'r') as f:
        content = f.read()
    if "tokenizer.pad_token = tokenizer.eos_token" in content:
        print("✓ Padding token fix is in place")
    else:
        print("✗ Padding token fix missing!")
        all_good = False
else:
    print("✗ main.py not found!")
    all_good = False

# 4. Check generation fix is in place
print("\n4. Checking generation fix...")
if os.path.exists(main_py_path):
    with open(main_py_path, 'r') as f:
        content = f.read()
    if "preds = outputs[:, -64:]" in content:
        print("✓ Generation fix is in place (takes last 64 tokens)")
    else:
        print("✗ Generation fix missing! Still using old [:, 64:] slicing")
        all_good = False

# 5. Check num_workers is set to 0
print("\n5. Checking num_workers setting...")
if os.path.exists(main_py_path):
    with open(main_py_path, 'r') as f:
        for line in f:
            if "'default': 0,  # Set to 0 to avoid CUDA" in line:
                print("✓ num_workers default is 0")
                break
        else:
            print("⚠️ num_workers might not be 0 (could cause CUDA issues)")

# 6. Test basic model functionality
print("\n6. Testing model with correct shapes...")
try:
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
    
    # Test with batch of 128, sequence length 64
    texts = ["Test sequence"] * 128
    inputs = tokenizer(texts, return_tensors="pt", padding="max_length", max_length=64, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
    
    print(f"✓ Model forward pass works! Shape: {inputs['input_ids'].shape}, Loss: {outputs.loss.item():.4f}")
except Exception as e:
    print(f"✗ Model test failed: {e}")
    all_good = False

# 7. Check experiment structure
print("\n7. Checking experiment structure...")
if os.path.exists("nature_exact_experiment/gen_0/best.ckpt"):
    print("✓ Generation 0 checkpoint exists")
else:
    print("⚠️ Generation 0 checkpoint not found (will train from scratch)")

if os.path.exists("nature_exact_experiment/gen_1"):
    print("⚠️ Generation 1 directory exists (might have bad data)")
    print("   Recommendation: rm -rf nature_exact_experiment/gen_1")
else:
    print("✓ Generation 1 will be created fresh")

# 8. Check run script
print("\n8. Checking run_nature_experiment.py...")
if os.path.exists("run_nature_experiment.py"):
    with open("run_nature_experiment.py", 'r') as f:
        content = f.read()
    if '"--num_workers", "0"' in content:
        print("✓ run_nature_experiment.py sets num_workers to 0")
    else:
        print("⚠️ run_nature_experiment.py might not set num_workers to 0")
else:
    print("✗ run_nature_experiment.py not found!")
    all_good = False

# Final verdict
print("\n" + "="*60)
if all_good:
    print("✅ ALL CHECKS PASSED! Ready to run:")
    print("   python3 run_nature_experiment.py")
else:
    print("❌ SOME CHECKS FAILED! Fix issues above before running.")
    
print("="*60)

# Show next steps
print("\nRecommended steps:")
print("1. If gen_1 exists with bad data: rm -rf nature_exact_experiment/gen_1")
print("2. Run: python3 run_nature_experiment.py")
print("3. Monitor with: watch -n 1 nvidia-smi")
print("="*60)