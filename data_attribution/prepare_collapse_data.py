#!/usr/bin/env python3
"""
Prepare datasets for data attribution analysis on model collapse
Converts experiment data into format required by RapidIn
"""

import json
import pickle
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

def load_generation_data(gen_path):
    """Load generated data from a specific generation"""
    pkl_path = gen_path / "generated_data.pkl"
    if pkl_path.exists():
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    return None

def create_jsonl_dataset(data, output_path, data_type="train"):
    """Convert data to JSONL format required by RapidIn"""
    
    samples = []
    
    if data_type == "train":
        # Format training data samples
        for idx, text in enumerate(data):
            sample = {
                "id": f"train_{idx}",
                "instruction": "",
                "input": text[:256],  # Use first 256 chars as input
                "output": text[256:512]  # Next 256 as output
            }
            samples.append(sample)
    
    elif data_type == "test":
        # Format test samples to analyze collapse
        for idx, text in enumerate(data):
            sample = {
                "id": f"test_{idx}",
                "instruction": "Continue the text:",
                "input": text[:256],
                "output": text[256:512]
            }
            samples.append(sample)
    
    # Write to JSONL
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Created {output_path} with {len(samples)} samples")
    return samples

def prepare_collapse_indicators(metrics_history_path):
    """Create test samples that indicate collapse"""
    
    with open(metrics_history_path, 'r') as f:
        metrics = json.load(f)
    
    # Create samples that would reveal collapse
    collapse_indicators = []
    
    # High perplexity samples
    collapse_indicators.append({
        "id": "collapse_perplexity",
        "instruction": "Generate coherent text about machine learning",
        "input": "Machine learning models are",
        "output": "[MEASURE_PERPLEXITY]"
    })
    
    # Repetition detection
    collapse_indicators.append({
        "id": "collapse_repetition",
        "instruction": "Complete this sentence naturally",
        "input": "The researchers found that",
        "output": "[MEASURE_REPETITION]"
    })
    
    # Vocabulary diversity test
    collapse_indicators.append({
        "id": "collapse_diversity",
        "instruction": "Write a diverse paragraph about technology",
        "input": "Technology has transformed",
        "output": "[MEASURE_DIVERSITY]"
    })
    
    return collapse_indicators

def create_synthetic_comparison_dataset(base_dir):
    """Create datasets comparing original vs synthetic data influence"""
    
    datasets_dir = Path("./datasets")
    datasets_dir.mkdir(exist_ok=True)
    
    # Load original WikiText data
    print("Loading original WikiText data...")
    # This would load from the actual WikiText dataset
    original_samples = [
        "The history of artificial intelligence began in antiquity with myths and stories.",
        "Machine learning is a subset of artificial intelligence that focuses on data.",
        "Neural networks are computing systems inspired by biological neural networks.",
        # Add more samples from actual WikiText
    ]
    
    # Load synthetic data from different generations
    synthetic_samples = {
        "gen_1": [
            "The the the the the the the the the the",
            "Learning learning learning learning learning",
            "Networks networks networks networks networks"
        ],
        "gen_2": [
            "The history intelligence antiquity myths stories repeated",
            "Machine subset artificial focuses data patterns",
            "Neural computing systems biological inspired connections"
        ]
    }
    
    # Create original data JSONL
    create_jsonl_dataset(
        original_samples,
        datasets_dir / "wikitext_original.jsonl",
        data_type="train"
    )
    
    # Create synthetic data JSONL for each generation
    for gen, samples in synthetic_samples.items():
        create_jsonl_dataset(
            samples,
            datasets_dir / f"synthetic_{gen}.jsonl",
            data_type="train"
        )
    
    # Create test samples
    test_samples = prepare_collapse_indicators("./metrics_history.json")
    
    with open(datasets_dir / "collapse_test_samples.jsonl", 'w') as f:
        for sample in test_samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Created collapse test samples: {len(test_samples)} indicators")

def main():
    """Main function to prepare all datasets"""
    
    print("="*60)
    print("PREPARING DATA FOR COLLAPSE ATTRIBUTION ANALYSIS")
    print("="*60)
    
    # Create comparison datasets
    create_synthetic_comparison_dataset("./")
    
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Run caching stage: python MP_main.py --config='collapse_attribution_config.json'")
    print("2. Run retrieval stage to identify influential samples")
    print("3. Analyze which data types trigger collapse")

if __name__ == "__main__":
    main()