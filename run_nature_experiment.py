#!/usr/bin/env python3
"""
Run the exact Nature paper experiment with comprehensive metrics.
Tracks both Nature paper metrics (perplexity, loss) and additional diversity metrics.
"""

import os
import sys
import subprocess
import json
import torch
import numpy as np
from pathlib import Path
from collections import Counter

# Disable HF_TRANSFER to avoid download issues
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

# Add the Nature paper code directory to path
sys.path.append('Zakahler-curse_recurse-b48c90a')

def calculate_diversity_metrics(text_samples):
    """Calculate comprehensive diversity and quality metrics."""
    metrics = {}
    
    # Combine all samples
    all_text = " ".join(text_samples) if text_samples else ""
    words = all_text.split()
    
    if len(words) > 0:
        # 1. Vocabulary diversity (unique words / total words)
        unique_words = len(set(words))
        total_words = len(words)
        metrics['vocab_diversity'] = unique_words / total_words
        
        # 2. N-gram diversity (for n=1,2,3,4)
        for n in range(1, 5):
            ngrams = []
            for sample in text_samples:
                sample_words = sample.split()
                ngrams.extend([tuple(sample_words[i:i+n]) for i in range(len(sample_words)-n+1)])
            unique_ngrams = len(set(ngrams))
            total_ngrams = len(ngrams)
            metrics[f'{n}gram_diversity'] = unique_ngrams / total_ngrams if total_ngrams > 0 else 0
        
        # 3. Word entropy
        word_counts = Counter(words)
        total = sum(word_counts.values())
        probs = [count/total for count in word_counts.values()]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        metrics['word_entropy'] = entropy
        
        # 4. Repetition rate (duplicate lines)
        lines = [s.strip() for s in all_text.split('.') if s.strip()]
        repetition_rate = 1 - (len(set(lines)) / len(lines)) if lines else 0
        metrics['repetition_rate'] = repetition_rate
    else:
        metrics = {
            'vocab_diversity': 0,
            '1gram_diversity': 0,
            '2gram_diversity': 0,
            '3gram_diversity': 0,
            '4gram_diversity': 0,
            'word_entropy': 0,
            'repetition_rate': 0
        }
    
    return metrics

def generate_samples(model_path, num_samples=20):
    """Generate text samples from a trained model."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model from checkpoint
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location='cpu')
            state_dict = checkpoint['state_dict']
            
            # Remove 'model.' prefix from keys
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    new_state_dict[k[6:]] = v
                else:
                    new_state_dict[k] = v
            
            model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
            model.load_state_dict(new_state_dict, strict=False)
        else:
            return []
        
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        samples = []
        prompts = ["The weather today", "In recent news", "Scientists have discovered", 
                  "The economy is", "Technology has"]
        
        with torch.no_grad():
            for _ in range(num_samples // len(prompts)):
                for prompt in prompts:
                    inputs = tokenizer(prompt, return_tensors="pt").to(device)
                    
                    # Generate with beam search (matching Nature paper settings)
                    outputs = model.generate(
                        inputs['input_ids'],
                        num_beams=5,
                        max_new_tokens=64,
                        min_new_tokens=64,
                        repetition_penalty=3.0,
                        pad_token_id=tokenizer.pad_token_id,
                        do_sample=False
                    )
                    
                    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    samples.append(text)
        
        return samples
    except Exception as e:
        print(f"  Warning: Could not generate samples: {e}")
        return []

def run_generation_experiment(num_generations=5, collect_extra_metrics=True):
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
    ╠══════════════════════════════════════════════════════╣
    ║ NATURE PAPER METRICS:                                 ║
    ║ - Perplexity (primary metric)                         ║
    ║ - Training/Validation Loss                            ║
    ║ - Sample Generated Texts                              ║""")
    
    if collect_extra_metrics:
        print("""    ╠══════════════════════════════════════════════════════╣
    ║ ADDITIONAL METRICS (our analysis):                    ║
    ║ - Vocabulary Diversity                                 ║
    ║ - N-gram Diversity (1,2,3,4-grams)                   ║
    ║ - Word Entropy                                        ║
    ║ - Repetition Rate                                     ║""")
    
    print("""    ╚══════════════════════════════════════════════════════╝
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
        
        # Set environment variable to disable HF_TRANSFER for this subprocess
        env = os.environ.copy()
        env['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
        
        # Build command with Nature paper's exact settings
        cmd = [
            "python3", "Zakahler-curse_recurse-b48c90a/main.py",
            "--model_tag", "facebook/opt-125m",  # Their default model
            "--batch-size", "128",                # Their batch size
            "--learning-rate", "2e-5",            # Their LR
            "--max-epochs", "5",                  # Their epochs
            "--save-name", str(gen_dir) + "/",
            "--accelerator", "cuda",              # Force GPU usage
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
            
            print(f"  Command: {' '.join(gen_cmd[:5])}...")
            result = subprocess.run(gen_cmd, capture_output=True, text=True, env=env)
            
            if result.returncode != 0:
                print(f"❌ Generation failed: {result.stderr}")
                continue
            
            # Now train on the generated data
            print(f"📚 Training Generation {gen} on synthetic data...")
            cmd.extend([
                "--load-generate", str(gen_dir / f"generated_data_gen{gen}.pkl"),
            ])
        
        # Run training with HF_TRANSFER disabled
        print(f"  Command: {' '.join(cmd[:5])}...")
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        if result.returncode != 0:
            print(f"❌ Training failed: {result.stderr}")
            continue
        
        # Parse Nature paper metrics from output
        perplexity = None
        train_loss = None
        val_loss = None
        
        for line in result.stdout.split('\n'):
            if "perplexity" in line.lower():
                try:
                    import re
                    match = re.search(r'perplexity[:\s]+([0-9.]+)', line.lower())
                    if match:
                        perplexity = float(match.group(1))
                except:
                    pass
            if "train_loss" in line.lower() or "training loss" in line.lower():
                try:
                    match = re.search(r'loss[:\s]+([0-9.]+)', line.lower())
                    if match:
                        train_loss = float(match.group(1))
                except:
                    pass
            if "val_loss" in line.lower() or "validation loss" in line.lower():
                try:
                    match = re.search(r'loss[:\s]+([0-9.]+)', line.lower())
                    if match:
                        val_loss = float(match.group(1))
                except:
                    pass
        
        # Compile metrics
        metrics = {
            "generation": gen,
            "nature_metrics": {
                "perplexity": perplexity,
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            "model_path": str(gen_dir / "best.ckpt")
        }
        
        # Generate samples and calculate additional metrics if requested
        if collect_extra_metrics:
            print(f"📊 Calculating additional metrics for Generation {gen}...")
            model_path = gen_dir / "best.ckpt"
            
            if model_path.exists():
                samples = generate_samples(model_path, num_samples=20)
                diversity_metrics = calculate_diversity_metrics(samples)
                
                # Save sample texts
                with open(gen_dir / "sample_texts.json", "w") as f:
                    json.dump(samples[:10], f, indent=2)
            else:
                diversity_metrics = {
                    'vocab_diversity': 0,
                    '1gram_diversity': 0,
                    '2gram_diversity': 0,
                    '3gram_diversity': 0,
                    '4gram_diversity': 0,
                    'word_entropy': 0,
                    'repetition_rate': 0
                }
                samples = []
            
            metrics["additional_metrics"] = diversity_metrics
            metrics["num_samples"] = len(samples)
        
        metrics_history.append(metrics)
        
        # Print current generation metrics
        print(f"\n📊 Generation {gen} Results:")
        print(f"  NATURE PAPER METRICS:")
        print(f"   Perplexity: {perplexity:.2f}" if perplexity else "   Perplexity: N/A")
        if train_loss:
            print(f"   Train Loss: {train_loss:.3f}")
        if val_loss:
            print(f"   Val Loss: {val_loss:.3f}")
        
        if collect_extra_metrics and "additional_metrics" in metrics:
            print(f"  ADDITIONAL METRICS:")
            dm = metrics["additional_metrics"]
            print(f"   Vocabulary Diversity: {dm['vocab_diversity']:.3f}")
            print(f"   2-gram Diversity: {dm['2gram_diversity']:.3f}")
            print(f"   Word Entropy: {dm['word_entropy']:.3f}")
            print(f"   Repetition Rate: {dm['repetition_rate']:.3f}")
        
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
        
        # Nature paper metrics
        print("\n  NATURE PAPER METRICS:")
        print("   Generation | Perplexity")
        print("   " + "-"*30)
        for m in metrics_history:
            nm = m['nature_metrics']
            ppl_str = f"{nm['perplexity']:.2f}" if nm['perplexity'] else "N/A"
            print(f"   {m['generation']:^10} | {ppl_str:^10}")
        
        # Additional metrics if collected
        if collect_extra_metrics and "additional_metrics" in metrics_history[0]:
            print("\n  ADDITIONAL METRICS:")
            print("   Generation | Vocab Div | 2-gram Div | Entropy")
            print("   " + "-"*50)
            for m in metrics_history:
                if "additional_metrics" in m:
                    am = m['additional_metrics']
                    print(f"   {m['generation']:^10} | {am['vocab_diversity']:^9.3f} | "
                          f"{am['2gram_diversity']:^10.3f} | {am['word_entropy']:^7.2f}")
        
        # Calculate degradation
        if metrics_history[0]['nature_metrics'].get('perplexity') and metrics_history[-1]['nature_metrics'].get('perplexity'):
            initial_ppl = metrics_history[0]['nature_metrics']['perplexity']
            final_ppl = metrics_history[-1]['nature_metrics']['perplexity']
            degradation = (final_ppl / initial_ppl - 1) * 100
            print(f"\n   📊 NATURE PAPER RESULT: {degradation:.1f}% perplexity increase")
            
        if collect_extra_metrics and "additional_metrics" in metrics_history[0] and "additional_metrics" in metrics_history[-1]:
            initial_div = metrics_history[0]['additional_metrics']['vocab_diversity']
            final_div = metrics_history[-1]['additional_metrics']['vocab_diversity']
            if initial_div > 0:
                div_loss = (1 - final_div / initial_div) * 100
                print(f"   📊 ADDITIONAL FINDING: {div_loss:.1f}% diversity decrease")

if __name__ == "__main__":
    # First, check if we have the required dependencies
    print("🔍 Checking dependencies...")
    
    # Check if the Nature codebase is present
    if not Path("Zakahler-curse_recurse-b48c90a/main.py").exists():
        print("❌ Nature paper codebase not found!")
        print("   Please ensure Zakahler-curse_recurse-b48c90a/ directory exists")
        sys.exit(1)
    
    # Install required packages if needed, including hf_transfer to fix the error
    print("📦 Installing required packages...")
    subprocess.run([
        "pip3", "install", "-q",
        "torch", "transformers", "datasets", 
        "pytorch-lightning", "numpy", "tqdm",
        "tensorboard", "tensorboardX", "hf_transfer"
    ])
    
    print("\n✅ Dependencies ready!")
    
    # Run the experiment with full metrics
    run_generation_experiment(num_generations=5, collect_extra_metrics=True)