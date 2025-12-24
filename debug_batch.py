#!/usr/bin/env python3
"""
Debug script to check batch dimensions
"""

import torch
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
from Zakahler-curse_recurse-b48c90a.dataset import WikiText2Dataset, MyDataLoader

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load dataset
with open("nature_exact_experiment/gen_1/generated_data_gen1.pkl", "rb") as f:
    train_dataset = pickle.load(f)

print(f"Dataset type: {type(train_dataset)}")
print(f"Dataset length: {len(train_dataset)}")

# Create dataloader
data_loader = MyDataLoader(
    train_dataset,
    train_dataset,  # Using same for val
    train_dataset,  # Using same for test
    128,  # batch size
    0  # num_workers
)

# Get one batch
for batch in data_loader.train_dataloader:
    print("\nBatch shapes:")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  attention_mask: {batch['attention_mask'].shape}")
    print(f"  labels: {batch['labels'].shape}")
    
    # Check actual values
    print(f"\nFirst sequence length: {batch['input_ids'][0].shape}")
    print(f"Batch size from data: {batch['input_ids'].shape[0]}")
    print(f"Sequence length: {batch['input_ids'].shape[1] if len(batch['input_ids'].shape) > 1 else 'N/A'}")
    
    # Try model forward pass
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
    model.eval()
    
    with torch.no_grad():
        try:
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            print(f"\n✓ Model forward pass successful!")
            print(f"  Loss: {outputs.loss.item()}")
            print(f"  Logits shape: {outputs.logits.shape}")
        except Exception as e:
            print(f"\n✗ Model forward pass failed: {e}")
            print(f"  This is the error we need to fix!")
    
    break  # Just check first batch