#!/usr/bin/env python3
"""
Run the exact Nature paper experiment with their settings.
Uses OPT-125M, WikiText2, beam search, and all their parameters.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

# Disable HF_TRANSFER to avoid download issues
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

# Add the Nature paper code directory to path
sys.path.append('Zakahler-curse_recurse-b48c90a')

def run_generation_experiment(num_generations=5):
    """
    Run the recursive training experiment with Nature paper settings.
    """
    
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║     NATURE PAPER EXACT REPLICATION EXPERIMENT         ║
    ╠══════════════════════════════════════════════════════╣
    ║ Model: OPT-125M (facebook/opt-125m)                   ║
    ║ Dataset: WikiText2 (64-token chunks)                  ║
    ║ Generation: 5-way beam search, repetition penalty 3.0 ║
    ║ Training: 5 epochs, batch size 128, LR 2e-5           ║
    ║ Generations: 5 recursive training cycles              ║
    ╚══════════════════════════════════════════════════════╝
    """)
    
    base_dir = Path("nature_exact_experiment")
    base_dir.mkdir(exist_ok=True)
    
    # Track metrics for each generation
    metrics_history = []
    
    for gen in range(num_generations):
        print(f"\n{'='*60}")
        print(f"GENERATION {gen}")
        print(f"{'='*60}\n")
        
        gen_dir = base_dir / f"gen_{gen}"
        gen_dir.mkdir(exist_ok=True)
        
        # Build command with Nature paper's exact settings
        cmd = [
            "python3", "Zakahler-curse_recurse-b48c90a/main.py",
            "--model_tag", "facebook/opt-125m",  # Their default model
            "--batch-size", "128",                # Their batch size
            "--learning-rate", "2e-5",            # Their LR
            "--max-epochs", "5",                  # Their epochs
            "--save-name", str(gen_dir) + "/",
            "--accelerator", "auto",              # Auto-detect GPU/CPU
            "--num_devices", "1",                 # Single GPU
        ]
        
        if gen == 0:
            # Generation 0: Train on original WikiText2
            print("📚 Training on original WikiText2 dataset...")
            # Add --pretrained flag for generation 0 to load from HuggingFace
            cmd.append("--pretrained")
        else:
            # Later generations: Load previous model and generated data
            prev_gen_dir = base_dir / f"gen_{gen-1}"
            
            # First, generate synthetic data from previous model
            print(f"🔄 Generating synthetic data from Generation {gen-1}...")
            
            gen_cmd = [
                "python3", "Zakahler-curse_recurse-b48c90a/main.py",
                "--model_tag", "facebook/opt-125m",
                "--load-name", str(prev_gen_dir / "best.ckpt"),
                "--generate", str(gen_dir / f"generated_data_gen{gen}"),
                # Use their exact generation settings from main.py line 307
                # num_beams=5, max_new_tokens=64, min_new_tokens=64, repetition_penalty=3.0
            ]
            
            print(f"  Command: {' '.join(gen_cmd)}")
            result = subprocess.run(gen_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Generation failed: {result.stderr}")
                continue
            
            # Now train on the generated data
            print(f"📚 Training Generation {gen} on synthetic data...")
            cmd.extend([
                "--load-generate", str(gen_dir / f"generated_data_gen{gen}.pkl"),
            ])
        
        # Run training
        print(f"  Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Training failed: {result.stderr}")
            continue
        
        # Parse output for metrics
        output_lines = result.stdout.split('\n')
        perplexity = None
        for line in output_lines:
            if "perplexity" in line.lower():
                try:
                    # Extract perplexity value from output
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "perplexity" in part.lower() and i+1 < len(parts):
                            perplexity = float(parts[i+1].replace(',', '').replace(':', ''))
                            break
                except:
                    pass
        
        metrics = {
            "generation": gen,
            "perplexity": perplexity,
            "model_path": str(gen_dir / "best.ckpt")
        }
        metrics_history.append(metrics)
        
        print(f"\n📊 Generation {gen} Metrics:")
        print(f"   Perplexity: {perplexity if perplexity else 'N/A'}")
        
        # Save intermediate results
        with open(base_dir / "metrics_history.json", "w") as f:
            json.dump(metrics_history, f, indent=2)
    
    print(f"""
    ╔══════════════════════════════════════════════════════╗
    ║            NATURE REPLICATION COMPLETE!               ║
    ╠══════════════════════════════════════════════════════╣
    ║ Results saved to: nature_exact_experiment/            ║
    ║ Metrics: nature_exact_experiment/metrics_history.json ║
    ╚══════════════════════════════════════════════════════╝
    """)
    
    # Print summary
    if len(metrics_history) > 1:
        print("\n📈 Model Collapse Progression:")
        for m in metrics_history:
            print(f"   Generation {m['generation']}: Perplexity = {m.get('perplexity', 'N/A')}")
        
        if metrics_history[0].get('perplexity') and metrics_history[-1].get('perplexity'):
            initial_ppl = metrics_history[0]['perplexity']
            final_ppl = metrics_history[-1]['perplexity']
            degradation = (final_ppl / initial_ppl - 1) * 100
            print(f"\n   Total Degradation: {degradation:.1f}% increase in perplexity")

if __name__ == "__main__":
    # First, check if we have the required dependencies
    print("🔍 Checking dependencies...")
    
    # Check if the Nature codebase is present
    if not Path("Zakahler-curse_recurse-b48c90a/main.py").exists():
        print("❌ Nature paper codebase not found!")
        print("   Please ensure Zakahler-curse_recurse-b48c90a/ directory exists")
        sys.exit(1)
    
    # Install required packages if needed
    print("📦 Installing required packages...")
    subprocess.run([
        "pip3", "install", "-q",
        "torch", "transformers", "datasets", 
        "pytorch-lightning", "numpy", "tqdm"
    ])
    
    print("\n✅ Dependencies ready!")
    
    # Run the experiment
    run_generation_experiment(num_generations=5)