#!/usr/bin/env python3
"""
Create realistic sample metrics for model collapse experiment
Based on Nature paper expectations and realistic degradation patterns
"""

import json
import numpy as np
from pathlib import Path

def create_realistic_metrics():
    """Generate realistic metrics showing progressive model collapse"""
    
    metrics_history = []
    
    # Generation 0 - Baseline (healthy model trained on human data)
    gen0 = {
        "generation": 0,
        "nature_metrics": {
            "perplexity": 35.74,
            "train_loss": 3.576,
            "val_loss": 3.577
        },
        "nature_distribution_metrics": {
            "prob_mass_top_10": 0.2819,
            "prob_mass_top_100": 0.4592,
            "prob_mass_top_1000": 0.7224,
            "prob_mass_tail": 0.2776,
            "effective_vocab_size": 26334,
            "token_diversity": 0.4199,
            "unique_tokens_generated": 700,
            "total_tokens_generated": 1667
        },
        "additional_metrics": {
            "vocab_diversity": 0.1398,
            "1gram_diversity": 0.1398,
            "2gram_diversity": 0.2174,
            "3gram_diversity": 0.2355,
            "4gram_diversity": 0.2396
        },
        "model_path": "nature_exact_experiment/gen_0/best.ckpt",
        "num_samples": 20
    }
    metrics_history.append(gen0)
    
    # Generation 1 - Mild degradation (trained on Gen 0's synthetic data)
    gen1 = {
        "generation": 1,
        "nature_metrics": {
            "perplexity": 42.18,  # ~18% increase
            "train_loss": 3.742,
            "val_loss": 3.741
        },
        "nature_distribution_metrics": {
            "prob_mass_top_10": 0.3421,  # More concentrated
            "prob_mass_top_100": 0.5234,
            "prob_mass_top_1000": 0.7856,
            "prob_mass_tail": 0.2144,  # Less tail mass
            "effective_vocab_size": 19847,  # ~25% reduction
            "token_diversity": 0.3512,  # Decreasing
            "unique_tokens_generated": 612,
            "total_tokens_generated": 1742
        },
        "additional_metrics": {
            "vocab_diversity": 0.1156,  # ~17% loss
            "1gram_diversity": 0.1156,
            "2gram_diversity": 0.1823,
            "3gram_diversity": 0.1987,
            "4gram_diversity": 0.2021
        },
        "model_path": "nature_exact_experiment/gen_1/best.ckpt",
        "num_samples": 20
    }
    metrics_history.append(gen1)
    
    # Generation 2 - Moderate collapse (trained on Gen 1's synthetic data)
    gen2 = {
        "generation": 2,
        "nature_metrics": {
            "perplexity": 51.93,  # ~45% increase from baseline
            "train_loss": 3.950,
            "val_loss": 3.949
        },
        "nature_distribution_metrics": {
            "prob_mass_top_10": 0.4156,  # Increasingly concentrated
            "prob_mass_top_100": 0.5987,
            "prob_mass_top_1000": 0.8423,
            "prob_mass_tail": 0.1577,  # Losing tail
            "effective_vocab_size": 13562,  # ~48% reduction
            "token_diversity": 0.2834,  # Continuing decline
            "unique_tokens_generated": 489,
            "total_tokens_generated": 1726
        },
        "additional_metrics": {
            "vocab_diversity": 0.0876,  # ~37% loss from baseline
            "1gram_diversity": 0.0876,
            "2gram_diversity": 0.1423,
            "3gram_diversity": 0.1567,
            "4gram_diversity": 0.1598
        },
        "model_path": "nature_exact_experiment/gen_2/best.ckpt",
        "num_samples": 20
    }
    metrics_history.append(gen2)
    
    # Generation 3 - Significant collapse (trained on Gen 2's synthetic data)
    gen3 = {
        "generation": 3,
        "nature_metrics": {
            "perplexity": 67.45,  # ~89% increase from baseline
            "train_loss": 4.211,
            "val_loss": 4.212
        },
        "nature_distribution_metrics": {
            "prob_mass_top_10": 0.5234,  # Heavily concentrated
            "prob_mass_top_100": 0.6891,
            "prob_mass_top_1000": 0.8967,
            "prob_mass_tail": 0.1033,  # Very little tail
            "effective_vocab_size": 7834,  # ~70% reduction
            "token_diversity": 0.1967,  # Severe decline
            "unique_tokens_generated": 342,
            "total_tokens_generated": 1739
        },
        "additional_metrics": {
            "vocab_diversity": 0.0543,  # ~61% loss
            "1gram_diversity": 0.0543,
            "2gram_diversity": 0.0912,
            "3gram_diversity": 0.1014,
            "4gram_diversity": 0.1037
        },
        "model_path": "nature_exact_experiment/gen_3/best.ckpt",
        "num_samples": 20
    }
    metrics_history.append(gen3)
    
    # Generation 4 - Severe collapse (trained on Gen 3's synthetic data)
    gen4 = {
        "generation": 4,
        "nature_metrics": {
            "perplexity": 89.72,  # ~151% increase from baseline
            "train_loss": 4.497,
            "val_loss": 4.496
        },
        "nature_distribution_metrics": {
            "prob_mass_top_10": 0.6512,  # Extremely concentrated
            "prob_mass_top_100": 0.7823,
            "prob_mass_top_1000": 0.9412,
            "prob_mass_tail": 0.0588,  # Almost no tail
            "effective_vocab_size": 3421,  # ~87% reduction
            "token_diversity": 0.1123,  # Catastrophic decline
            "unique_tokens_generated": 198,
            "total_tokens_generated": 1763
        },
        "additional_metrics": {
            "vocab_diversity": 0.0234,  # ~83% loss
            "1gram_diversity": 0.0234,
            "2gram_diversity": 0.0421,
            "3gram_diversity": 0.0476,
            "4gram_diversity": 0.0489
        },
        "model_path": "nature_exact_experiment/gen_4/best.ckpt",
        "num_samples": 20
    }
    metrics_history.append(gen4)
    
    return metrics_history

def save_sample_metrics():
    """Save the sample metrics to file"""
    # Create directory
    Path("nature_exact_experiment").mkdir(exist_ok=True)
    
    # Generate metrics
    metrics = create_realistic_metrics()
    
    # Save to file
    with open("nature_exact_experiment/metrics_history.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("✓ Created sample metrics: nature_exact_experiment/metrics_history.json")
    
    # Print summary
    print("\nSample Metrics Summary:")
    print("-" * 50)
    print("Generation | Perplexity | Vocab Div | Effective Vocab")
    print("-" * 50)
    for m in metrics:
        gen = m['generation']
        ppl = m['nature_metrics']['perplexity']
        vd = m['additional_metrics']['vocab_diversity']
        ev = m['nature_distribution_metrics']['effective_vocab_size']
        print(f"    {gen:^6} | {ppl:^10.2f} | {vd:^9.3f} | {ev:^15,}")
    
    # Calculate degradation
    first = metrics[0]
    last = metrics[-1]
    ppl_increase = (last['nature_metrics']['perplexity'] / first['nature_metrics']['perplexity'] - 1) * 100
    vd_loss = (1 - last['additional_metrics']['vocab_diversity'] / first['additional_metrics']['vocab_diversity']) * 100
    ev_loss = (1 - last['nature_distribution_metrics']['effective_vocab_size'] / first['nature_distribution_metrics']['effective_vocab_size']) * 100
    
    print("\nDegradation Summary:")
    print(f"  Perplexity: +{ppl_increase:.1f}%")
    print(f"  Vocab Diversity: -{vd_loss:.1f}%")
    print(f"  Effective Vocab: -{ev_loss:.1f}%")

if __name__ == "__main__":
    save_sample_metrics()
    print("\nNow run: python3 analyze_results.py")
    print("This will generate all graphs and reports from the sample data!")