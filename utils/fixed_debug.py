#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Testing OPT-125M batch processing...")

# Load tokenizer with fix
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("✓ Set pad_token to eos_token")

# Load model
model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
print("✓ Model loaded")

# Create batch of 128 sequences, each 64 tokens
batch_size = 128
seq_length = 64
texts = ["Hello world this is a test"] * batch_size
inputs = tokenizer(texts, return_tensors="pt", padding="max_length", max_length=seq_length, truncation=True)

print(f"\nBatch shapes:")
print(f"  input_ids: {inputs['input_ids'].shape}")
print(f"  Should be: [{batch_size}, {seq_length}]")

# Test forward pass
try:
    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['input_ids'])
    print(f"\n✓ Forward pass works! Loss: {outputs.loss.item():.4f}")
except Exception as e:
    print(f"\n✗ Error: {e}")
    print("This is the batch size mismatch issue!")

print(f"\nNote: {batch_size} * {seq_length} = {batch_size * seq_length} (this is where 8192 comes from)")