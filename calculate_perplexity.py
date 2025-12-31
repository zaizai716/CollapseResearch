#!/usr/bin/env python3
"""
Calculate perplexity for all trained models
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
import sys
sys.path.append('Zakahler-curse_recurse-b48c90a')
from dataset import prepare_data, preprocess_datasets, WikiText2Dataset
import numpy as np
from pathlib import Path

def calculate_perplexity(model_checkpoint, dataset_type='test'):
    """Calculate perplexity on test set for a model checkpoint"""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model from checkpoint
    checkpoint = torch.load(model_checkpoint, map_location='cpu')
    state_dict = checkpoint['state_dict']
    
    # Clean up state dict (remove 'model.' prefix if present)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
    
    # Create model and load weights
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load dataset
    raw_dataset = prepare_data()
    dataset = preprocess_datasets(raw_dataset, tokenizer)
    test_dataset = WikiText2Dataset(dataset=dataset, partition=dataset_type, tokenizer=tokenizer)
    
    # Create dataloader
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Calculate perplexity
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i > 50:  # Limit to first 50 batches for speed
                break
                
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Ensure 2D tensors
            if len(input_ids.shape) == 1:
                input_ids = input_ids.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)
                labels = labels.unsqueeze(0)
            
            try:
                outputs = model(input_ids=input_ids, 
                              attention_mask=attention_mask, 
                              labels=labels)
                
                loss = outputs.loss
                
                # Skip if loss is NaN or Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"  Warning: NaN/Inf loss detected, skipping batch {i}")
                    continue
                
                # Count actual tokens (not padding)
                num_tokens = attention_mask.sum().item()
                
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
                
            except Exception as e:
                print(f"  Error in batch {i}: {e}")
                continue
    
    if total_tokens == 0:
        return float('inf')
    
    # Calculate average loss and perplexity
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity

def main():
    print("="*60)
    print("CALCULATING PERPLEXITY FOR ALL GENERATIONS")
    print("="*60)
    
    results = []
    
    for gen in range(5):
        checkpoint_path = Path(f"nature_exact_experiment/gen_{gen}/best.ckpt")
        
        if checkpoint_path.exists():
            print(f"\nGeneration {gen}:")
            print(f"  Loading from: {checkpoint_path}")
            
            try:
                perplexity = calculate_perplexity(checkpoint_path)
                print(f"  ✓ Perplexity: {perplexity:.2f}")
                results.append((gen, perplexity))
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                results.append((gen, None))
        else:
            print(f"\nGeneration {gen}: ✗ No checkpoint found")
            results.append((gen, None))
    
    print("\n" + "="*60)
    print("SUMMARY - Model Collapse Progression:")
    print("-"*40)
    print("Generation | Perplexity | % Increase")
    print("-"*40)
    
    baseline = None
    for gen, ppl in results:
        if ppl is not None:
            if baseline is None:
                baseline = ppl
                increase = 0
            else:
                increase = ((ppl / baseline) - 1) * 100
            print(f"    {gen:^6} | {ppl:^10.2f} | {increase:+^10.1f}%")
        else:
            print(f"    {gen:^6} |     N/A    |     N/A")
    
    print("="*60)
    
    # Compare to Nature paper expectations
    print("\nNature Paper Expected:")
    print("  Gen 0: ~27-30 perplexity")
    print("  Gen 5: ~50-60 perplexity")
    print("  Total: 60-100% increase")

if __name__ == "__main__":
    main()