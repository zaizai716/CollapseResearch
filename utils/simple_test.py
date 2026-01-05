import torch
import pickle
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

print("="*60)
print("COMPREHENSIVE BATCH SIZE DEBUG TEST")
print("="*60)

# Test 1: Basic model test
print("\n1. Testing basic model with clean data...")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

texts = ["test"] * 128
inputs = tokenizer(texts, return_tensors="pt", padding="max_length", max_length=64, truncation=True)

print(f"Shape: {inputs['input_ids'].shape}")

try:
    outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['input_ids'])
    print(f"✓ Basic test works! Loss: {outputs.loss.item()}")
except Exception as e:
    print(f"✗ Basic test failed: {e}")

# Test 2: Check generated data format
print("\n2. Checking generated data format...")
if os.path.exists("nature_exact_experiment/gen_1/generated_data_gen1.pkl"):
    with open("nature_exact_experiment/gen_1/generated_data_gen1.pkl", "rb") as f:
        dataset = pickle.load(f)
    
    print(f"Dataset type: {type(dataset)}")
    print(f"Dataset length: {len(dataset)}")
    
    if len(dataset) > 0:
        first = dataset[0]
        print(f"First item type: {type(first)}")
        
        if hasattr(first, 'keys'):
            print(f"Keys: {list(first.keys())}")
            for key in first.keys():
                if hasattr(first[key], 'shape'):
                    print(f"  {key} shape: {first[key].shape}")
                elif hasattr(first[key], '__len__'):
                    print(f"  {key} length: {len(first[key])}")
                else:
                    print(f"  {key}: {type(first[key])}")
        
        # Test with actual data
        print("\n3. Testing with actual generated data...")
        if hasattr(first, '__getitem__') and 'input_ids' in first:
            # Create a batch manually
            batch_items = dataset[:128] if len(dataset) >= 128 else dataset[:len(dataset)]
            
            # Check shapes
            print(f"Creating batch from {len(batch_items)} items...")
            
            # Try to stack them
            try:
                input_ids = torch.stack([torch.tensor(item['input_ids']) if not isinstance(item['input_ids'], torch.Tensor) else item['input_ids'] for item in batch_items])
                print(f"Stacked input_ids shape: {input_ids.shape}")
                
                # Check if this causes the 8192 issue
                if input_ids.numel() == 8192:
                    print(f"⚠️ WARNING: Total elements = 8192! This is the problem!")
                    print(f"  Shape: {input_ids.shape}")
                    print(f"  Expected: [128, 64] but got {list(input_ids.shape)}")
            except Exception as e:
                print(f"Cannot stack batch: {e}")
else:
    print("Generated data file not found")

# Test 3: Import and test DataLoader
print("\n4. Testing DataLoader...")
try:
    import sys
    sys.path.append('Zakahler-curse_recurse-b48c90a')
    from dataset import MyDataLoader
    
    if os.path.exists("nature_exact_experiment/gen_1/generated_data_gen1.pkl"):
        with open("nature_exact_experiment/gen_1/generated_data_gen1.pkl", "rb") as f:
            train_dataset = pickle.load(f)
        
        data_loader = MyDataLoader(
            train_dataset,
            train_dataset,  # Using same for val
            train_dataset,  # Using same for test
            128,  # batch size
            0  # num_workers
        )
        
        # Get one batch
        for batch in data_loader.train_dataloader:
            print(f"\nDataLoader batch shapes:")
            print(f"  input_ids: {batch['input_ids'].shape}")
            print(f"  attention_mask: {batch['attention_mask'].shape}")
            print(f"  labels: {batch['labels'].shape}")
            
            # Check if shapes match
            if batch['input_ids'].shape[0] * batch['input_ids'].shape[1] == 8192:
                print(f"⚠️ FOUND THE PROBLEM! Batch is {batch['input_ids'].shape} = {batch['input_ids'].numel()} elements")
            
            # Try forward pass
            try:
                outputs = model(**batch)
                print(f"✓ DataLoader batch works! Loss: {outputs.loss.item()}")
            except Exception as e:
                print(f"✗ DataLoader batch failed: {e}")
                print("THIS IS THE EXACT ERROR WE'RE SEEING!")
            
            break  # Just test first batch
except Exception as e:
    print(f"DataLoader test failed: {e}")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)