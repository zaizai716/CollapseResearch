#!/usr/bin/env python3
"""
Create test dataset for model collapse data attribution.
These test samples probe collapse behaviors to identify which training samples cause them.
"""

import json
from pathlib import Path


def create_collapse_test_samples():
    """
    Create test samples that probe different collapse behaviors.

    The idea: we want to find which training samples most influence the model's
    response to these test prompts. High-influence samples are "guilty" of causing
    collapse behaviors.
    """

    test_samples = [
        # Repetition detection - common collapse pattern in OPT
        {
            "id": "test_repetition_1",
            "instruction": "",
            "input": "The researchers found that",
            "output": "the results were consistent with previous studies.",
            "hotwords": "the|that|and"  # Focus on common repeated words
        },
        {
            "id": "test_repetition_2",
            "instruction": "",
            "input": "In the following section, we will discuss",
            "output": "the methodology and experimental results.",
            "hotwords": "the|and|is"
        },
        {
            "id": "test_repetition_3",
            "instruction": "",
            "input": "The model was trained on",
            "output": "a large corpus of text data from various sources.",
            "hotwords": "the|a|and|of"
        },

        # Vocabulary diversity - test if model can produce varied vocabulary
        {
            "id": "test_diversity_1",
            "instruction": "",
            "input": "Technology has transformed modern society through",
            "output": "innovations in communication, transportation, and healthcare.",
            "hotwords": "technology|innovation|communication"
        },
        {
            "id": "test_diversity_2",
            "instruction": "",
            "input": "The Renaissance was a period characterized by",
            "output": "artistic achievement, scientific discovery, and cultural rebirth.",
            "hotwords": "Renaissance|artistic|scientific|cultural"
        },
        {
            "id": "test_diversity_3",
            "instruction": "",
            "input": "Climate change affects ecosystems by",
            "output": "altering temperatures, precipitation patterns, and species distributions.",
            "hotwords": "climate|temperature|species|ecosystem"
        },

        # Coherence tests - can the model maintain logical flow
        {
            "id": "test_coherence_1",
            "instruction": "",
            "input": "Machine learning algorithms learn patterns from data, which allows them to",
            "output": "make predictions on new, unseen examples.",
            "hotwords": "learn|patterns|predictions"
        },
        {
            "id": "test_coherence_2",
            "instruction": "",
            "input": "Neural networks consist of layers of interconnected nodes that",
            "output": "process and transform input data through learned weights.",
            "hotwords": "neural|layers|process|weights"
        },

        # Long-form coherence
        {
            "id": "test_longform_1",
            "instruction": "",
            "input": "The history of artificial intelligence began with early pioneers who imagined machines that could",
            "output": "think and reason like humans, leading to decades of research and development.",
            "hotwords": "artificial|intelligence|machines|think"
        },

        # Technical vocabulary
        {
            "id": "test_technical_1",
            "instruction": "",
            "input": "In computer science, algorithms are designed to",
            "output": "solve specific computational problems efficiently.",
            "hotwords": "algorithm|computational|efficiently"
        }
    ]

    return test_samples


def save_test_dataset(samples, output_path):
    """Save test samples to JSONL format"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"Saved {len(samples)} test samples to {output_path}")
    return output_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Create test dataset for collapse attribution')
    parser.add_argument('--output_path', type=str,
                        default='./datasets/collapse_test.jsonl',
                        help='Output path for test dataset')

    args = parser.parse_args()

    print("Creating collapse detection test samples...")
    samples = create_collapse_test_samples()

    print(f"Created {len(samples)} test samples")
    save_test_dataset(samples, args.output_path)

    print("\nTest samples cover:")
    print("  - Repetition detection (common tokens)")
    print("  - Vocabulary diversity")
    print("  - Coherence maintenance")
    print("  - Technical vocabulary")
    print("\nHotwords are set to focus attribution on key tokens.")


if __name__ == "__main__":
    main()
