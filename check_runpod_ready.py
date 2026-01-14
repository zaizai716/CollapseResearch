#!/usr/bin/env python3
"""
RunPod Readiness Check Script
Verifies everything is set up correctly before running experiments
"""

import os
import sys
import subprocess
from pathlib import Path
import torch
import json

def colored_print(text, color='green'):
    """Print with colors"""
    colors = {
        'green': '\033[92m',
        'red': '\033[91m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'end': '\033[0m'
    }
    print(f"{colors.get(color, '')}{text}{colors['end']}")

def check_gpu():
    """Check GPU availability and CUDA"""
    print("\n" + "="*60)
    print("1. GPU CHECK")
    print("="*60)
    
    try:
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            colored_print(f"✓ CUDA available", 'green')
            colored_print(f"✓ GPU Count: {gpu_count}", 'green')
            colored_print(f"✓ GPU Name: {gpu_name}", 'green')
            colored_print(f"✓ GPU Memory: {gpu_memory:.1f} GB", 'green')
            
            # Check if GPU has enough memory
            if gpu_memory < 16:
                colored_print(f"⚠ Warning: GPU has <16GB. May need batch size adjustment", 'yellow')
            
            return True
        else:
            colored_print("✗ No GPU detected!", 'red')
            return False
            
    except Exception as e:
        colored_print(f"✗ GPU check failed: {e}", 'red')
        return False

def check_environment():
    """Check Python environment and packages"""
    print("\n" + "="*60)
    print("2. ENVIRONMENT CHECK")
    print("="*60)
    
    all_good = True
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major == 3 and python_version.minor >= 9:
        colored_print(f"✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}", 'green')
    else:
        colored_print(f"✗ Python {python_version.major}.{python_version.minor} (need 3.9+)", 'red')
        all_good = False
    
    # Check required packages
    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'datasets': 'Datasets',
        'accelerate': 'Accelerate',
        'numpy': 'NumPy',
        'huggingface_hub': 'HuggingFace Hub'
    }
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            colored_print(f"✓ {name} installed", 'green')
        except ImportError:
            colored_print(f"✗ {name} not installed", 'red')
            all_good = False
    
    return all_good

def check_directory_structure():
    """Check if all directories are properly organized"""
    print("\n" + "="*60)
    print("3. DIRECTORY STRUCTURE CHECK")
    print("="*60)
    
    required_dirs = {
        'model_collapse_experiments': 'Collapse experiments directory',
        'model_collapse_experiments/Zakahler-curse_recurse-b48c90a': 'Nature paper code',
        'data_attribution': 'Attribution analysis directory',
        'data_attribution/RapidIn': 'RapidIn repository'
    }
    
    all_good = True
    for dir_path, description in required_dirs.items():
        if Path(dir_path).exists():
            colored_print(f"✓ {description}: {dir_path}", 'green')
        else:
            colored_print(f"✗ Missing: {dir_path}", 'red')
            all_good = False
    
    # Check for important files
    important_files = {
        'runpod_setup.sh': 'Setup script',
        'CLAUDE.md': 'Project documentation',
        'data_attribution/download_and_attribute.py': 'Attribution script'
    }
    
    for file_path, description in important_files.items():
        if Path(file_path).exists():
            colored_print(f"✓ {description}: {file_path}", 'green')
        else:
            colored_print(f"✗ Missing: {file_path}", 'red')
            all_good = False
    
    return all_good

def check_huggingface():
    """Check HuggingFace token and access"""
    print("\n" + "="*60)
    print("4. HUGGINGFACE CHECK")
    print("="*60)
    
    hf_token = os.environ.get('HF_TOKEN')
    
    if hf_token:
        colored_print(f"✓ HF_TOKEN is set ({len(hf_token)} chars)", 'green')
        
        # Try to access the repository
        try:
            from huggingface_hub import list_repo_files
            files = list_repo_files("zaizaiiiii/model-collapse-experiment")
            colored_print(f"✓ Can access HuggingFace repository", 'green')
            colored_print(f"  Found {len(files)} files in repo", 'blue')
            return True
        except Exception as e:
            colored_print(f"⚠ Cannot access repo: {e}", 'yellow')
            return True  # Token is set, might just be permission issue
    else:
        colored_print("✗ HF_TOKEN not set!", 'red')
        colored_print("  Run: export HF_TOKEN='your-token-here'", 'yellow')
        return False

def check_disk_space():
    """Check available disk space"""
    print("\n" + "="*60)
    print("5. DISK SPACE CHECK")
    print("="*60)
    
    try:
        # Get disk usage
        result = subprocess.run(['df', '-h', '.'], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:
            parts = lines[1].split()
            if len(parts) >= 4:
                available = parts[3]
                colored_print(f"✓ Available disk space: {available}", 'green')
                
                # Parse available space
                if 'G' in available:
                    gb_available = float(available.replace('G', ''))
                    if gb_available < 50:
                        colored_print(f"⚠ Warning: Only {gb_available}GB available. Recommend >50GB", 'yellow')
                
                return True
    except:
        colored_print("⚠ Could not check disk space", 'yellow')
        return True
    
    return False

def check_rapidin_setup():
    """Check if RapidIn is properly set up"""
    print("\n" + "="*60)
    print("6. RAPIDIN SETUP CHECK")
    print("="*60)
    
    rapidin_path = Path("data_attribution/RapidIn")
    
    if not rapidin_path.exists():
        colored_print("✗ RapidIn not found", 'red')
        return False
    
    # Check for key RapidIn files
    key_files = ['MP_main.py', 'requirements.txt', 'README.md']
    all_good = True
    
    for file in key_files:
        if (rapidin_path / file).exists():
            colored_print(f"✓ RapidIn/{file} found", 'green')
        else:
            colored_print(f"✗ RapidIn/{file} missing", 'red')
            all_good = False
    
    return all_good

def estimate_runtime():
    """Estimate experiment runtime based on GPU"""
    print("\n" + "="*60)
    print("7. RUNTIME ESTIMATES")
    print("="*60)
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        
        estimates = {
            "4090": {"opt": "10-15 hours", "qwen": "2-3 days", "attribution": "5-7 hours"},
            "A100": {"opt": "8-12 hours", "qwen": "1.5-2 days", "attribution": "4-6 hours"},
            "H100": {"opt": "6-8 hours", "qwen": "1-1.5 days", "attribution": "3-4 hours"},
            "V100": {"opt": "15-20 hours", "qwen": "3-4 days", "attribution": "8-10 hours"}
        }
        
        # Find matching GPU
        gpu_match = None
        for gpu_type in estimates:
            if gpu_type in gpu_name:
                gpu_match = gpu_type
                break
        
        if gpu_match:
            est = estimates[gpu_match]
            colored_print(f"Runtime estimates for {gpu_name}:", 'blue')
            print(f"  • OPT-125M (5 generations): {est['opt']}")
            print(f"  • Qwen3-1.7B (5 generations): {est['qwen']}")
            print(f"  • Token Attribution Analysis: {est['attribution']}")
        else:
            colored_print(f"No specific estimates for {gpu_name}", 'yellow')
            print("  Typical estimates:")
            print("  • OPT-125M: 10-20 hours")
            print("  • Qwen3-1.7B: 2-4 days")
            print("  • Attribution: 5-10 hours")
    else:
        colored_print("No GPU detected - experiments will be very slow!", 'red')

def generate_quick_start():
    """Generate quick start commands"""
    print("\n" + "="*60)
    print("8. QUICK START COMMANDS")
    print("="*60)
    
    print("\nTo run experiments:")
    print("\n1. Model Collapse Experiment:")
    colored_print("   cd model_collapse_experiments", 'blue')
    colored_print("   python run_nature_experiment.py", 'blue')
    
    print("\n2. Token Attribution Analysis:")
    colored_print("   cd data_attribution", 'blue')
    colored_print("   python download_and_attribute.py", 'blue')
    
    print("\n3. Monitor GPU usage:")
    colored_print("   watch -n 1 nvidia-smi", 'blue')
    
    print("\n4. Check results:")
    colored_print("   ls model_collapse_experiments/graphs*/", 'blue')
    colored_print("   ls data_attribution/token_attribution_results/", 'blue')

def main():
    """Main readiness check"""
    print("="*60)
    print("RUNPOD READINESS CHECK")
    print("="*60)
    
    all_checks = []
    
    # Run all checks
    all_checks.append(("GPU", check_gpu()))
    all_checks.append(("Environment", check_environment()))
    all_checks.append(("Directory Structure", check_directory_structure()))
    all_checks.append(("HuggingFace", check_huggingface()))
    all_checks.append(("Disk Space", check_disk_space()))
    all_checks.append(("RapidIn", check_rapidin_setup()))
    
    # Runtime estimates
    estimate_runtime()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, status in all_checks if status)
    total = len(all_checks)
    
    if passed == total:
        colored_print(f"\n✓ ALL CHECKS PASSED ({passed}/{total})", 'green')
        colored_print("🚀 Ready to run experiments on RunPod!", 'green')
        generate_quick_start()
    else:
        colored_print(f"\n⚠ Some checks failed ({passed}/{total} passed)", 'yellow')
        print("\nFailed checks:")
        for name, status in all_checks:
            if not status:
                colored_print(f"  ✗ {name}", 'red')
        print("\nRun ./runpod_setup.sh to fix issues")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)