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

def calculate_nature_paper_metrics(model_path, tokenizer_name="facebook/opt-125m"):
    """
    Calculate Nature paper-specific metrics:
    1. Distribution analysis - probability mass in tails
    2. Token diversity analysis
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    metrics = {}
    
    try:
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
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
            
            model = AutoModelForCausalLM.from_pretrained(tokenizer_name)
            model.load_state_dict(new_state_dict, strict=False)
        else:
            return metrics
        
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        # 1. DISTRIBUTION ANALYSIS - Probability mass in tails
        # Generate many samples and analyze token probability distributions
        test_prompts = ["The", "In", "A", "This", "Today"]
        all_token_probs = []
        
        with torch.no_grad():
            for prompt in test_prompts:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                outputs = model(**inputs)
                logits = outputs.logits[0, -1, :]  # Last token's logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_token_probs.append(probs)
        
        # Analyze probability mass distribution
        avg_probs = np.mean(all_token_probs, axis=0)
        sorted_probs = np.sort(avg_probs)[::-1]  # Sort descending
        
        # Calculate probability mass in top tokens vs tail
        top_10_mass = np.sum(sorted_probs[:10])
        top_100_mass = np.sum(sorted_probs[:100])
        top_1000_mass = np.sum(sorted_probs[:1000])
        tail_mass = 1.0 - top_1000_mass  # Mass in tokens beyond top 1000
        
        metrics['prob_mass_top_10'] = float(top_10_mass)
        metrics['prob_mass_top_100'] = float(top_100_mass)
        metrics['prob_mass_top_1000'] = float(top_1000_mass)
        metrics['prob_mass_tail'] = float(tail_mass)
        
        # Calculate effective vocab size (tokens with non-negligible probability)
        threshold = 1e-6
        effective_vocab = np.sum(avg_probs > threshold)
        metrics['effective_vocab_size'] = int(effective_vocab)
        
        # 2. TOKEN DIVERSITY ANALYSIS
        # Generate longer sequences and analyze token usage patterns
        num_sequences = 20
        generated_tokens = []
        
        for _ in range(num_sequences):
            prompt = np.random.choice(test_prompts)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs['input_ids'],
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            # Extract generated tokens (excluding prompt)
            generated = outputs[0][len(inputs['input_ids'][0]):]
            generated_tokens.extend(generated.cpu().tolist())
        
        # Calculate token-level diversity
        unique_tokens = len(set(generated_tokens))
        total_tokens = len(generated_tokens)
        token_diversity = unique_tokens / total_tokens if total_tokens > 0 else 0
        
        metrics['token_diversity'] = float(token_diversity)
        metrics['unique_tokens_generated'] = unique_tokens
        metrics['total_tokens_generated'] = total_tokens
        
        # Calculate token frequency distribution (to detect collapse to common tokens)
        token_counts = Counter(generated_tokens)
        top_10_tokens_freq = sum(count for _, count in token_counts.most_common(10))
        top_10_tokens_ratio = top_10_tokens_freq / total_tokens if total_tokens > 0 else 0
        
        metrics['top_10_tokens_frequency_ratio'] = float(top_10_tokens_ratio)
        
    except Exception as e:
        print(f"  Warning: Could not calculate Nature paper metrics: {e}")
    
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
    
    print("\n=== Nature Collapse Experiment ===")
    print("Model: OPT-125M")
    print("Dataset: WikiText2")
    print("Generations: 5 recursive cycles")
    print("Batch size: 128, LR: 2e-5, Epochs: 5")
    print("\nMetrics tracked:")
    print("- Perplexity (main)")
    print("- Train/val loss")
    print("- Generated samples")
    print("- Distribution analysis")
    print("- Token diversity")
    
    if collect_extra_metrics:
        print("\nAdditional metrics:")
        print("- Vocab diversity")
        print("- N-gram diversity")
        print("- Word entropy")
        print("- Repetition rate")
    
    print("\n" + "="*40 + "\n")
    
    base_dir = Path("nature_exact_experiment")
    base_dir.mkdir(exist_ok=True)
    
    # Track metrics for each generation
    metrics_history = []
    
    for gen in range(num_generations):
        print(f"\n--- Generation {gen} ---")
        
        gen_dir = base_dir / f"gen_{gen}"
        gen_dir.mkdir(exist_ok=True)
        
        # Check if this generation is already trained
        checkpoint_path = gen_dir / "best.ckpt"
        if checkpoint_path.exists():
            print(f"Gen {gen} already done, skipping")
            
            # Still calculate metrics for this generation
            metrics = {
                "generation": gen,
                "status": "already_completed",
                "model_path": str(checkpoint_path)
            }
            
            # calc metrics if model exists
            if collect_extra_metrics:
                print(f"calculating metrics for gen {gen}...")
                nature_extra_metrics = calculate_nature_paper_metrics(checkpoint_path)
                metrics["nature_distribution_metrics"] = nature_extra_metrics
                
                samples = generate_samples(checkpoint_path, num_samples=20)
                diversity_metrics = calculate_diversity_metrics(samples)
                metrics["additional_metrics"] = diversity_metrics
                metrics["num_samples"] = len(samples)
            
            metrics_history.append(metrics)
            continue  # Skip to next generation
        
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
            "--accelerator", "gpu",               # Force GPU usage
            "--num_devices", "1",                 # Single GPU
        ]
        
        if gen == 0:
            # gen 0: train on original wikitext2
            print("training on original wikitext2...")
            # add pretrained flag for gen 0 to load from HF
            cmd.append("--pretrained")
        else:
            # later gens: load prev model and generated data
            prev_gen_dir = base_dir / f"gen_{gen-1}"
            
            # first generate synthetic data from prev model
            print(f"generating synthetic data from gen {gen-1}...")
            
            gen_cmd = [
                "python3", "Zakahler-curse_recurse-b48c90a/main.py",
                "--model_tag", "facebook/opt-125m",
                "--load-name", str(prev_gen_dir / "best.ckpt"),
                "--generate", str(gen_dir / f"generated_data_gen{gen}"),
                # exact generation settings from main.py line 307
                # num_beams=5, max_new_tokens=64, min_new_tokens=64, repetition_penalty=3.0
            ]
            
            print(f"  cmd: {' '.join(gen_cmd[:5])}...")
            result = subprocess.run(gen_cmd, capture_output=True, text=True, env=env)
            
            if result.returncode != 0:
                print(f"generation failed: {result.stderr}")
                continue
            
            # now train on the generated data
            print(f"training gen {gen} on synthetic data...")
            cmd.extend([
                "--load-generate", str(gen_dir / f"generated_data_gen{gen}.pkl"),
            ])
        
        # run training with HF_TRANSFER disabled
        print(f"  running: {' '.join(cmd[:5])}...")
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        if result.returncode != 0:
            print(f"training failed: {result.stderr}")
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
        
        # generate samples and calc additional metrics if requested
        if collect_extra_metrics:
            print(f"calculating metrics for gen {gen}...")
            model_path = gen_dir / "best.ckpt"
            
            # calc nature paper metrics (distribution analysis & token diversity)
            print(f"  calculating nature paper metrics...")
            nature_extra_metrics = calculate_nature_paper_metrics(model_path)
            metrics["nature_distribution_metrics"] = nature_extra_metrics
            
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
        
        # print current generation metrics
        print(f"\ngen {gen} results:")
        print(f"  nature paper metrics:")
        print(f"   perplexity: {perplexity:.2f}" if perplexity else "   perplexity: n/a")
        if train_loss:
            print(f"   Train Loss: {train_loss:.3f}")
        if val_loss:
            print(f"   Val Loss: {val_loss:.3f}")
        
        # Print Nature paper distribution metrics
        if "nature_distribution_metrics" in metrics:
            nm = metrics["nature_distribution_metrics"]
            if nm:
                print(f"   Distribution Analysis:")
                print(f"    - Prob mass in top 10 tokens: {nm.get('prob_mass_top_10', 0):.3f}")
                print(f"    - Prob mass in top 100 tokens: {nm.get('prob_mass_top_100', 0):.3f}")
                print(f"    - Prob mass in tail (>1000): {nm.get('prob_mass_tail', 0):.3f}")
                print(f"   Token Diversity:")
                print(f"    - Token diversity: {nm.get('token_diversity', 0):.3f}")
                print(f"    - Effective vocab size: {nm.get('effective_vocab_size', 0)}")
        
        if collect_extra_metrics and "additional_metrics" in metrics:
            print(f"  ADDITIONAL METRICS (our analysis):")
            dm = metrics["additional_metrics"]
            print(f"   Vocabulary Diversity: {dm['vocab_diversity']:.3f}")
            print(f"   2-gram Diversity: {dm['2gram_diversity']:.3f}")
            print(f"   Word Entropy: {dm['word_entropy']:.3f}")
            print(f"   Repetition Rate: {dm['repetition_rate']:.3f}")
        
        # Save intermediate results
        with open(base_dir / "metrics_history.json", "w") as f:
            json.dump(metrics_history, f, indent=2)
    
    print("\n--- experiment complete ---")
    print(f"results saved to: nature_exact_experiment/")
    print(f"metrics: nature_exact_experiment/metrics_history.json")
    
    # print summary
    if len(metrics_history) > 1:
        print("\nmodel collapse progression:")
        
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