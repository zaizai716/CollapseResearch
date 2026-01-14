#!/usr/bin/env python3
import os
import sys
import subprocess
import json
import torch
import numpy as np
from pathlib import Path
from collections import Counter
import shutil

os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

sys.path.append('Zakahler-curse_recurse-b48c90a')

# Hugging Face configuration
HF_REPO_ID = "zaizaiiiii/model-collapse-experiment"  # Your HF username
HF_TOKEN = os.environ.get('HF_TOKEN', None)  # Set with: export HF_TOKEN="your-token"

def setup_huggingface():
    """Setup Hugging Face for model uploads"""
    try:
        from huggingface_hub import HfApi, create_repo
        api = HfApi()
        
        # Create repo if it doesn't exist
        try:
            create_repo(HF_REPO_ID, repo_type="model", token=HF_TOKEN, exist_ok=True)
            print(f"✓ Hugging Face repo ready: {HF_REPO_ID}")
            return api
        except Exception as e:
            print(f"⚠️ Could not create HF repo (may already exist): {e}")
            return api
    except ImportError:
        print("Installing huggingface-hub...")
        subprocess.run(["pip", "install", "-q", "huggingface-hub"])
        return setup_huggingface()

def upload_to_huggingface(file_path, repo_path, api):
    """Upload a file to Hugging Face and delete local copy"""
    try:
        from huggingface_hub import upload_file
        
        print(f"  Uploading {file_path} to Hugging Face...")
        upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=repo_path,
            repo_id=HF_REPO_ID,
            token=HF_TOKEN
        )
        print(f"  ✓ Uploaded to HF: {repo_path}")
        return True
    except Exception as e:
        print(f"  ✗ Upload failed: {e}")
        print(f"  ⚠️ Keeping local file due to upload failure")
        return False

def download_from_huggingface(repo_path, local_path, api):
    """Download a file from Hugging Face when needed"""
    try:
        from huggingface_hub import hf_hub_download
        
        print(f"  Downloading {repo_path} from Hugging Face...")
        downloaded_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=repo_path,
            local_dir=os.path.dirname(local_path),
            token=HF_TOKEN
        )
        
        # Move to exact location if needed
        if downloaded_path != local_path:
            shutil.move(downloaded_path, local_path)
        
        print(f"  ✓ Downloaded to: {local_path}")
        return True
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        return False

def cleanup_unnecessary_files(gen_dir):
    """Delete unnecessary files to save space"""
    # Delete last.ckpt files (we only need best.ckpt)
    last_ckpt = gen_dir / "last.ckpt"
    if last_ckpt.exists():
        os.remove(last_ckpt)
        print(f"  ✓ Deleted last.ckpt to save space")
    
    # Delete lightning_logs
    lightning_logs = gen_dir / "lightning_logs"
    if lightning_logs.exists():
        shutil.rmtree(lightning_logs)
        print(f"  ✓ Deleted lightning_logs to save space")
    
    # Delete __pycache__
    for pycache in gen_dir.glob("**/__pycache__"):
        shutil.rmtree(pycache)

def manage_disk_space(current_gen, base_dir, hf_api):
    """Intelligent disk space management"""
    print(f"\n=== Managing Disk Space ===")
    
    # Clean current generation
    cleanup_unnecessary_files(base_dir / f"gen_{current_gen}")
    
    # Upload current generation to HF and delete old ones
    if current_gen >= 0:
        current_checkpoint = base_dir / f"gen_{current_gen}" / "best.ckpt"
        if current_checkpoint.exists():
            # Upload to Hugging Face
            if hf_api and HF_TOKEN:
                success = upload_to_huggingface(
                    current_checkpoint,
                    f"gen_{current_gen}/best.ckpt",
                    hf_api
                )
                
                # Delete older generations (keep only current and previous)
                if success and current_gen >= 2:
                    old_gen = current_gen - 2
                    old_gen_dir = base_dir / f"gen_{old_gen}"
                    if old_gen_dir.exists():
                        shutil.rmtree(old_gen_dir)
                        print(f"  ✓ Deleted gen_{old_gen} directory (backed up to HF)")
    
    # Delete generated data from previous generation once current is trained
    if current_gen > 0:
        prev_data = base_dir / f"gen_{current_gen}" / f"generated_data_gen{current_gen}.pkl"
        if prev_data.exists() and (base_dir / f"gen_{current_gen}" / "best.ckpt").exists():
            os.remove(prev_data)
            print(f"  ✓ Deleted generated_data_gen{current_gen}.pkl (model already trained)")
    
    # Show disk usage
    try:
        result = subprocess.run(["df", "-h", "/"], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if '/' in line and 'Filesystem' not in line:
                print(f"  Disk usage: {line}")
                break
    except:
        pass

def calculate_diversity_metrics(text_samples):
    """Calculate vocabulary and n-gram diversity metrics."""
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
        
    else:
        metrics = {
            'vocab_diversity': 0,
            '1gram_diversity': 0,
            '2gram_diversity': 0,
            '3gram_diversity': 0,
            '4gram_diversity': 0
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
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model from checkpoint
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location='cpu')
            state_dict = checkpoint['state_dict']
            
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
        
        # 1. Distribution analysis - Probability mass in tails
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
        
        # 2. Token diversity
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
    Now with Hugging Face backup and intelligent space management.
    """
    
    print("\n=== Nature Collapse Experiment ===")
    print("Model: OPT-125M")
    print("Dataset: WikiText2")
    print("Generations: 5 recursive cycles")
    print("Batch size: 128, LR: 2e-5, Epochs: 5")
    
    # Setup Hugging Face
    hf_api = None
    if HF_TOKEN:
        hf_api = setup_huggingface()
    else:
        print("⚠️ No HF_TOKEN set. Skipping Hugging Face uploads.")
        print("  Set with: export HF_TOKEN='your-token-here'")
    
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
        
        # Try to download from HF if not local
        if not checkpoint_path.exists() and hf_api:
            print(f"Checking Hugging Face for gen_{gen}...")
            download_from_huggingface(f"gen_{gen}/best.ckpt", checkpoint_path, hf_api)
        
        if checkpoint_path.exists():
            print(f"Gen {gen} already done, skipping training")
            
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
            
            # Space management
            manage_disk_space(gen, base_dir, hf_api)
            continue  # Skip to next generation
        
        # Set environment variable to disable HF_TRANSFER for this subprocess
        env = os.environ.copy()
        env['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
        
        # Force GPU usage
        if torch.cuda.is_available():
            env['CUDA_VISIBLE_DEVICES'] = '0'
            accelerator_arg = 'gpu'
        else:
            accelerator_arg = 'auto'
        
        # Build command with Nature paper's exact settings
        cmd = [
            "python3", "Zakahler-curse_recurse-b48c90a/main.py",
            "--model_tag", "facebook/opt-125m", 
            "--batch-size", "128",           
            "--learning-rate", "2e-5",           
            "--max-epochs", "5",                 
            "--save-name", str(gen_dir) + "/",
            "--num_workers", "0",  # Set to 0 to avoid CUDA multiprocessing issues
            "--accelerator", accelerator_arg,  # Force GPU usage               
        ]
        
        if gen == 0:
            # gen 0: train on original wikitext2
            print("training on original wikitext2...")
            cmd.append("--pretrained")
        else:
            # later gens: need previous model
            prev_gen = gen - 1
            prev_checkpoint = base_dir / f"gen_{prev_gen}" / "best.ckpt"
            
            # Download previous gen from HF if needed
            if not prev_checkpoint.exists() and hf_api:
                print(f"Downloading gen_{prev_gen} from Hugging Face...")
                download_from_huggingface(f"gen_{prev_gen}/best.ckpt", prev_checkpoint, hf_api)
            
            if not prev_checkpoint.exists():
                print(f"ERROR: Previous generation {prev_gen} model not found!")
                print(f"  Looking for: {prev_checkpoint}")
                break
            
            # first generate synthetic data from prev model
            print(f"generating synthetic data from gen {prev_gen}...")
            
            gen_cmd = [
                "python3", "Zakahler-curse_recurse-b48c90a/main.py",
                "--model_tag", "facebook/opt-125m",
                "--load-name", str(prev_checkpoint),
                "--generate", str(gen_dir / f"generated_data_gen{gen}"),
                "--num_workers", "0",
                "--accelerator", accelerator_arg,
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
        
        print(f"  running: {' '.join(cmd[:5])}...")
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        if result.returncode != 0:
            print(f"training failed: {result.stderr}")
            continue
        
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
        
        # generate samples and calc additional metrics
        if collect_extra_metrics:
            print(f"calculating metrics for gen {gen}...")
            model_path = gen_dir / "best.ckpt"
            
            if model_path.exists():
                # calc nature paper metrics (distribution analysis & token diversity)
                print(f"  calculating nature paper metrics...")
                nature_extra_metrics = calculate_nature_paper_metrics(model_path)
                metrics["nature_distribution_metrics"] = nature_extra_metrics
                
                samples = generate_samples(model_path, num_samples=20)
                diversity_metrics = calculate_diversity_metrics(samples)
                
                # Save sample texts
                with open(gen_dir / "sample_texts.json", "w") as f:
                    json.dump(samples[:10], f, indent=2)
            else:
                nature_extra_metrics = {}
                diversity_metrics = {
                    'vocab_diversity': 0,
                    '1gram_diversity': 0,
                    '2gram_diversity': 0,
                    '3gram_diversity': 0,
                    '4gram_diversity': 0
                }
                samples = []
            
            metrics["nature_distribution_metrics"] = nature_extra_metrics
            metrics["additional_metrics"] = diversity_metrics
            metrics["num_samples"] = len(samples)
        
        metrics_history.append(metrics)
        
        # Space management after each generation
        manage_disk_space(gen, base_dir, hf_api)
        
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
        
        # Save intermediate results
        with open(base_dir / "metrics_history.json", "w") as f:
            json.dump(metrics_history, f, indent=2)
    
    print("\n--- experiment complete ---")
    print(f"results saved to: nature_exact_experiment/")
    print(f"metrics: nature_exact_experiment/metrics_history.json")
    
    # Upload final metrics to HF
    if hf_api and (base_dir / "metrics_history.json").exists():
        upload_to_huggingface(base_dir / "metrics_history.json", "metrics_history.json", hf_api)
    
    # print summary
    if len(metrics_history) > 1:
        print("\nmodel collapse progression:")
        
        # Nature paper metrics
        print("\n  NATURE PAPER METRICS:")
        print("   Generation | Perplexity")
        print("   " + "-"*30)
        for m in metrics_history:
            if 'nature_metrics' in m:
                nm = m['nature_metrics']
                ppl_str = f"{nm['perplexity']:.2f}" if nm.get('perplexity') else "N/A"
                print(f"   {m['generation']:^10} | {ppl_str:^10}")
        
        # Additional metrics if collected
        if collect_extra_metrics and "additional_metrics" in metrics_history[0]:
            print("\n  ADDITIONAL METRICS:")
            print("   Generation | Vocab Div | 2-gram Div")
            print("   " + "-"*40)
            for m in metrics_history:
                if "additional_metrics" in m:
                    am = m['additional_metrics']
                    print(f"   {m['generation']:^10} | {am['vocab_diversity']:^9.3f} | "
                          f"{am['2gram_diversity']:^10.3f}")
        
        # Calculate degradation
        first_with_ppl = next((m for m in metrics_history if m.get('nature_metrics', {}).get('perplexity')), None)
        last_with_ppl = next((m for m in reversed(metrics_history) if m.get('nature_metrics', {}).get('perplexity')), None)
        
        if first_with_ppl and last_with_ppl:
            initial_ppl = first_with_ppl['nature_metrics']['perplexity']
            final_ppl = last_with_ppl['nature_metrics']['perplexity']
            degradation = (final_ppl / initial_ppl - 1) * 100
            print(f"\n   Nature Paper Result: {degradation:.1f}% perplexity increase")
            
        if collect_extra_metrics and "additional_metrics" in metrics_history[0] and "additional_metrics" in metrics_history[-1]:
            initial_div = metrics_history[0]['additional_metrics']['vocab_diversity']
            final_div = metrics_history[-1]['additional_metrics']['vocab_diversity']
            if initial_div > 0:
                div_loss = (1 - final_div / initial_div) * 100
                print(f"   Additional Finding: {div_loss:.1f}% diversity decrease")

if __name__ == "__main__":
    # First, check if we have the required dependencies
    print("Checking dependencies...")
    
    # Check if the Nature codebase is present
    if not Path("Zakahler-curse_recurse-b48c90a/main.py").exists():
        print("Nature paper codebase not found!")
        print("   Please ensure Zakahler-curse_recurse-b48c90a/ directory exists")
        sys.exit(1)
    
    # Install required packages if needed
    print("Installing required packages...")
    subprocess.run([
        "pip3", "install", "-q",
        "torch", "transformers", "datasets", 
        "pytorch-lightning", "numpy", "tqdm",
        "huggingface-hub"  # Added for uploads
    ])
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ No GPU detected, will run on CPU (very slow)")
    
    print("\nDependencies ready")
    
    # Check for HF token
    if not os.environ.get('HF_TOKEN'):
        print("\n" + "="*60)
        print("⚠️ IMPORTANT: Set your Hugging Face token for automatic backups:")
        print("   export HF_TOKEN='your-token-here'")
        print("   Get token from: https://huggingface.co/settings/tokens")
        print("="*60 + "\n")
    
    run_generation_experiment(num_generations=5, collect_extra_metrics=True)