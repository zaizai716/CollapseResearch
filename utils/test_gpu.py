#!/usr/bin/env python3
"""
GPU Test Script for RunPod
Tests PyTorch GPU functionality and model loading
"""

import torch
import sys
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_cuda():
    """Test CUDA availability and basic operations"""
    print("="*60)
    print("CUDA/GPU Test")
    print("="*60)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if not cuda_available:
        print("\n⚠️ ERROR: CUDA is not available!")
        print("Make sure you:")
        print("1. Selected a GPU instance on RunPod (RTX 4090 recommended)")
        print("2. Selected a PyTorch template with CUDA support")
        print("3. CUDA version matches template requirements")
        return False
    
    # Print GPU information
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        
        # Check if compute capability is supported
        if props.major < 5 or (props.major == 12 and props.minor == 0):
            print(f"  ⚠️ WARNING: Compute capability {props.major}.{props.minor} may not be fully supported!")
            if props.major == 12:
                print("  RTX 5090 is NOT supported by PyTorch yet. Use RTX 4090 instead.")
    
    # Test basic GPU operations
    print("\nTesting GPU operations...")
    try:
        # Create tensors on GPU
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        
        # Perform computation
        start = time.time()
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(f"✓ Matrix multiplication (1000x1000) completed in {elapsed:.3f} seconds")
        
        # Check memory usage
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"✓ GPU Memory - Allocated: {allocated:.1f} MB, Reserved: {reserved:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"✗ GPU operation failed: {e}")
        return False

def test_model_loading():
    """Test loading OPT-125M model on GPU"""
    print("\n" + "="*60)
    print("Model Loading Test")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("Skipping model test - CUDA not available")
        return False
    
    try:
        print("Loading OPT-125M model...")
        model_name = "facebook/opt-125m"
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✓ Tokenizer loaded")
        
        # Load model directly on GPU
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use FP16 for memory efficiency
            device_map="cuda:0"
        )
        print("✓ Model loaded on GPU")
        
        # Test inference
        print("\nTesting model inference...")
        text = "The weather today is"
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            start = time.time()
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=20,
                do_sample=False
            )
            elapsed = time.time() - start
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✓ Inference successful in {elapsed:.3f} seconds")
        print(f"  Input: '{text}'")
        print(f"  Output: '{generated}'")
        
        # Check GPU memory after model loading
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"✓ Model GPU memory usage: {allocated:.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*60)
    print("RunPod GPU Test Suite")
    print("="*60)
    
    # Run tests
    cuda_ok = test_cuda()
    model_ok = test_model_loading() if cuda_ok else False
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    if cuda_ok and model_ok:
        print("✅ All tests passed! GPU is ready for experiments.")
        print("\nYou can now run:")
        print("  python run_nature_experiment.py")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please check the errors above.")
        if not cuda_ok:
            print("\nTroubleshooting:")
            print("1. Verify you selected a GPU instance on RunPod")
            print("2. Reinstall PyTorch with CUDA support:")
            print("   pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124")
        sys.exit(1)

if __name__ == "__main__":
    main()