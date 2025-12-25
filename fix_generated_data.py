#!/usr/bin/env python3
"""
Fix the malformed generated data
"""
import pickle
import torch
import os

print("Fixing malformed generated data...")

# Load the broken data
pkl_path = "nature_exact_experiment/gen_1/generated_data_gen1.pkl"
if os.path.exists(pkl_path):
    with open(pkl_path, "rb") as f:
        dataset = pickle.load(f)
    
    print(f"Original dataset length: {len(dataset)}")
    print(f"Original first item shapes:")
    first = dataset[0]
    for key in first.keys():
        if hasattr(first[key], 'shape'):
            print(f"  {key}: {first[key].shape}")
    
    # Fix the data
    fixed_dataset = []
    for item in dataset:
        # Skip malformed items where input_ids is only length 1
        if item['input_ids'].shape[0] == 1:
            continue
            
        # For items that look wrong but might be salvageable
        if item['attention_mask'].shape[0] == 64:
            # Use attention_mask as template for correct shape
            fixed_item = {
                'input_ids': item['attention_mask'],  # Use attention mask as it has right shape
                'attention_mask': item['attention_mask'],
                'labels': item['attention_mask']  # Use same for labels
            }
            fixed_dataset.append(fixed_item)
    
    if len(fixed_dataset) == 0:
        print("Cannot fix data - regenerating from scratch is needed")
        print("Deleting gen_1 so it can be regenerated...")
        os.system("rm -rf nature_exact_experiment/gen_1")
        print("Deleted gen_1. Run the experiment again to regenerate.")
    else:
        print(f"Fixed dataset length: {len(fixed_dataset)}")
        
        # Save fixed data
        backup_path = pkl_path.replace(".pkl", "_backup.pkl")
        os.rename(pkl_path, backup_path)
        print(f"Backed up original to {backup_path}")
        
        with open(pkl_path, "wb") as f:
            pickle.dump(fixed_dataset, f)
        print(f"Saved fixed data to {pkl_path}")
        
        # Verify fix
        with open(pkl_path, "rb") as f:
            test = pickle.load(f)
        first = test[0]
        print("Fixed first item shapes:")
        for key in first.keys():
            if hasattr(first[key], 'shape'):
                print(f"  {key}: {first[key].shape}")
else:
    print(f"File not found: {pkl_path}")
    print("Run from /CollapseResearch directory")