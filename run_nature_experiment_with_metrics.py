#!/usr/bin/env python3
"""
Enhanced Nature paper experiment with comprehensive metrics collection.
Tracks perplexity, diversity metrics, and generates sample texts.
"""

import os
import sys
import subprocess
import json
import torch
import numpy as np
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM

# Disable HF_TRANSFER to avoid download issues
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

# Add the Nature paper code directory to path
sys.path.append('Zakahler-curse_recurse-b48c90a')

def calculate_metrics(text_samples, tokenizer):
    """Calculate comprehensive diversity and quality metrics."""
    metrics = {}
    
    # Combine all samples
    all_text = " ".join(text_samples)
    words = all_text.split()
    
    # 1. Vocabulary diversity (unique words / total words)
    unique_words = len(set(words))
    total_words = len(words)
    metrics['vocab_diversity'] = unique_words / total_words if total_words > 0 else 0
    
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
    
    # 4. Average sentence length
    sentences = all_text.split('.')
    avg_sent_len = np.mean([len(s.split()) for s in sentences if s.strip()])
    metrics['avg_sentence_length'] = avg_sent_len
    
    # 5. Repetition rate (duplicate lines)
    lines = [s.strip() for s in all_text.split('.') if s.strip()]
    repetition_rate = 1 - (len(set(lines)) / len(lines)) if lines else 0
    metrics['repetition_rate'] = repetition_rate
    
    return metrics

def generate_samples(model_path, num_samples=50):
    """Generate text samples from a trained model."""
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    
    # Load model from checkpoint
    if model_path.exists():
        # Load the PyTorch Lightning checkpoint
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
        print(f"Warning: Model not found at {model_path}, using base model")
        model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
    
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    
    samples = []
    prompts = [
        "The weather today",
        "In recent news",
        "Scientists have discovered",
        "The economy is",
        "Technology has"
    ]
    
    with torch.no_grad():
        for _ in range(num_samples // len(prompts)):
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # Generate with beam search (matching Nature paper settings)
                outputs = model.generate(
                    inputs['input_ids'],
                    num_beams=5,
                    max_new_tokens=64,
                    min_new_tokens=64,
                    repetition_penalty=3.0,
                    pad_token_id=tokenizer.pad_token_id
                )
                
                text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                samples.append(text)
    
    return samples

def run_generation_experiment_with_metrics(num_generations=5):
    """
    Run the recursive training experiment with comprehensive metrics.
    """
    
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║   NATURE PAPER REPLICATION WITH FULL METRICS          ║
    ╠══════════════════════════════════════════════════════╣
    ║ Model: OPT-125M (facebook/opt-125m)                   ║
    ║ Dataset: WikiText2 (64-token chunks)                  ║
    ║ Generation: 5-way beam search, repetition penalty 3.0 ║
    ║ Training: 5 epochs, batch size 128, LR 2e-5           ║
    ╠══════════════════════════════════════════════════════╣
    ║ NATURE PAPER METRICS:                                 ║
    ║ - Perplexity (primary metric)                         ║
    ║ - Training/Validation Loss                            ║
    ║ - Sample Generated Texts                              ║
    ╠══════════════════════════════════════════════════════╣
    ║ ADDITIONAL METRICS (our analysis):                    ║
    ║ - Vocabulary Diversity                                 ║
    ║ - N-gram Diversity (1,2,3,4-grams)                   ║
    ║ - Word Entropy                                        ║
    ║ - Repetition Rate                                     ║
    ╚══════════════════════════════════════════════════════╝
    """)
    
    base_dir = Path("nature_exact_experiment_metrics")
    base_dir.mkdir(exist_ok=True)
    
    # Initialize tokenizer once
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Track all metrics across generations
    all_metrics = []
    
    for gen in range(num_generations):
        print(f"\n{'='*60}")
        print(f"GENERATION {gen}")
        print(f"{'='*60}\n")
        
        gen_dir = base_dir / f"gen_{gen}"
        gen_dir.mkdir(exist_ok=True)
        
        # Build command with Nature paper's exact settings
        cmd = [
            "python3", "Zakahler-curse_recurse-b48c90a/main.py",
            "--model_tag", "facebook/opt-125m",
            "--batch-size", "128",
            "--learning-rate", "2e-5",
            "--max-epochs", "5",
            "--save-name", str(gen_dir) + "/",
            "--accelerator", "auto",
            "--num_devices", "1",
        ]
        
        if gen == 0:
            print("📚 Training on original WikiText2 dataset...")
            cmd.append("--pretrained")
        else:
            prev_gen_dir = base_dir / f"gen_{gen-1}"
            
            print(f"🔄 Generating synthetic data from Generation {gen-1}...")
            gen_cmd = [
                "python3", "Zakahler-curse_recurse-b48c90a/main.py",
                "--model_tag", "facebook/opt-125m",
                "--load-name", str(prev_gen_dir / "best.ckpt"),
                "--generate", str(gen_dir / f"generated_data_gen{gen}")
            ]
            
            result = subprocess.run(gen_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Generation failed: {result.stderr}")
                continue
            
            print(f"📚 Training Generation {gen} on synthetic data...")
            cmd.extend([
                "--load-generate", str(gen_dir / f"generated_data_gen{gen}.pkl"),
            ])
        
        # Run training
        print(f"  Command: {' '.join(cmd[:5])}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
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
        
        # Generate text samples and calculate diversity metrics
        print(f"📊 Calculating comprehensive metrics for Generation {gen}...")
        model_path = gen_dir / "best.ckpt"
        
        if model_path.exists():
            samples = generate_samples(model_path, num_samples=50)
            diversity_metrics = calculate_metrics(samples, tokenizer)
            
            # Save sample texts
            with open(gen_dir / "sample_texts.json", "w") as f:
                json.dump(samples[:10], f, indent=2)  # Save first 10 samples
        else:
            print(f"  Warning: No model checkpoint found, using default metrics")
            diversity_metrics = {
                'vocab_diversity': 0,
                '1gram_diversity': 0,
                '2gram_diversity': 0,
                '3gram_diversity': 0,
                '4gram_diversity': 0,
                'word_entropy': 0,
                'avg_sentence_length': 0,
                'repetition_rate': 0
            }
            samples = []
        
        # Compile all metrics
        metrics = {
            "generation": gen,
            # Nature paper metrics
            "nature_metrics": {
                "perplexity": perplexity,
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            # Additional analysis metrics
            "additional_metrics": diversity_metrics,
            "model_path": str(model_path),
            "num_samples": len(samples)
        }
        
        all_metrics.append(metrics)
        
        # Print current generation metrics
        print(f"\n📊 Generation {gen} Results:")
        print(f"  NATURE PAPER METRICS:")
        print(f"   Perplexity: {perplexity:.2f}" if perplexity else "   Perplexity: N/A")
        print(f"   Train Loss: {train_loss:.3f}" if train_loss else "   Train Loss: N/A")
        print(f"   Val Loss: {val_loss:.3f}" if val_loss else "   Val Loss: N/A")
        print(f"  ADDITIONAL METRICS:")
        print(f"   Vocabulary Diversity: {diversity_metrics['vocab_diversity']:.3f}")
        print(f"   2-gram Diversity: {diversity_metrics['2gram_diversity']:.3f}")
        print(f"   Word Entropy: {diversity_metrics['word_entropy']:.3f}")
        print(f"   Repetition Rate: {diversity_metrics['repetition_rate']:.3f}")
        
        # Save intermediate results
        with open(base_dir / "metrics_history.json", "w") as f:
            json.dump(all_metrics, f, indent=2)
    
    # Print final summary
    print(f"""
    ╔══════════════════════════════════════════════════════╗
    ║            EXPERIMENT COMPLETE!                       ║
    ╠══════════════════════════════════════════════════════╣
    ║ Results saved to: {str(base_dir):<36} ║
    ║ Metrics: metrics_history.json                         ║
    ║ Samples: gen_*/sample_texts.json                      ║
    ╚══════════════════════════════════════════════════════╝
    """)
    
    if len(all_metrics) > 1:
        print("\n📈 Model Collapse Progression:")
        print("\n  NATURE PAPER METRICS:")
        print("   Generation | Perplexity | Train Loss | Val Loss")
        print("   " + "-"*50)
        for m in all_metrics:
            nm = m['nature_metrics']
            print(f"   {m['generation']:^10} | {nm.get('perplexity', 0):^10.2f} | "
                  f"{nm.get('train_loss', 0):^10.3f} | {nm.get('val_loss', 0):^8.3f}")
        
        print("\n  ADDITIONAL METRICS:")
        print("   Generation | Vocab Div | 2-gram Div | Entropy | Repetition")
        print("   " + "-"*60)
        for m in all_metrics:
            am = m['additional_metrics']
            print(f"   {m['generation']:^10} | {am['vocab_diversity']:^9.3f} | "
                  f"{am['2gram_diversity']:^10.3f} | {am['word_entropy']:^7.2f} | "
                  f"{am['repetition_rate']:^10.3f}")
        
        if all_metrics[0]['nature_metrics'].get('perplexity') and all_metrics[-1]['nature_metrics'].get('perplexity'):
            initial_ppl = all_metrics[0]['nature_metrics']['perplexity']
            final_ppl = all_metrics[-1]['nature_metrics']['perplexity']
            degradation = (final_ppl / initial_ppl - 1) * 100
            print(f"\n   📊 NATURE PAPER RESULT: {degradation:.1f}% perplexity increase")
            
            initial_div = all_metrics[0]['additional_metrics']['vocab_diversity']
            final_div = all_metrics[-1]['additional_metrics']['vocab_diversity']
            div_loss = (1 - final_div / initial_div) * 100
            print(f"   📊 ADDITIONAL FINDING: {div_loss:.1f}% diversity decrease")

if __name__ == "__main__":
    # Install required packages if needed
    print("🔍 Checking dependencies...")
    subprocess.run(["pip3", "install", "-q", "torch", "transformers", "datasets", 
                   "pytorch-lightning", "numpy", "tqdm", "tensorboard", "tensorboardX"])
    
    print("\n✅ Dependencies ready!")
    run_generation_experiment_with_metrics(num_generations=5)