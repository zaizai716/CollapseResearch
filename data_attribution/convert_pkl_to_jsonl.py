#!/usr/bin/env python3
"""
Convert synthetic_data.pkl files to JSONL format for RapidIn data attribution.
The PKL files contain tokenized data that needs to be decoded back to text.
"""

import pickle
import json
from pathlib import Path
from transformers import AutoTokenizer
import torch


def load_pkl_data(pkl_path):
    """Load tokenized data from PKL file"""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data


def decode_to_text(data, tokenizer):
    """Decode tokenized data back to text"""
    decoded_samples = []

    for i, sample in enumerate(data):
        input_ids = sample['input_ids']

        # Convert to list if tensor
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()

        # Decode tokens to text
        text = tokenizer.decode(input_ids, skip_special_tokens=True)

        # Clean up the text
        text = text.strip()

        if len(text) > 0:
            decoded_samples.append({
                'index': i,
                'text': text,
                'num_tokens': len(input_ids)
            })

    return decoded_samples


def convert_to_rapidin_jsonl(decoded_samples, output_path, split_ratio=0.5):
    """
    Convert decoded samples to RapidIn JSONL format.

    RapidIn expects:
    {"instruction": "", "input": "<context>", "output": "<continuation>"}

    We split each text sample into input (first half) and output (second half)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for i, sample in enumerate(decoded_samples):
            text = sample['text']

            # Split text into input and output portions
            # Use character-level split to maintain readability
            split_point = int(len(text) * split_ratio)

            # Try to find a space near the split point to avoid cutting words
            search_start = max(0, split_point - 50)
            search_end = min(len(text), split_point + 50)

            best_split = split_point
            for j in range(search_start, search_end):
                if text[j] == ' ':
                    best_split = j
                    if j >= split_point:
                        break

            input_text = text[:best_split].strip()
            output_text = text[best_split:].strip()

            # Skip if either part is empty
            if not input_text or not output_text:
                continue

            jsonl_entry = {
                "id": f"sample_{i}",
                "instruction": "",
                "input": input_text,
                "output": output_text
            }

            f.write(json.dumps(jsonl_entry, ensure_ascii=False) + '\n')

    print(f"Wrote {len(decoded_samples)} samples to {output_path}")
    return output_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Convert synthetic_data.pkl to JSONL')
    parser.add_argument('--pkl_path', type=str, required=True, help='Path to synthetic_data.pkl')
    parser.add_argument('--output_path', type=str, required=True, help='Output JSONL path')
    parser.add_argument('--model_name', type=str, default='facebook/opt-125m',
                        help='Model name for tokenizer')
    parser.add_argument('--split_ratio', type=float, default=0.5,
                        help='Ratio for splitting text into input/output')

    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print(f"Loading PKL data from: {args.pkl_path}")
    data = load_pkl_data(args.pkl_path)
    print(f"Loaded {len(data)} samples")

    print("Decoding tokens to text...")
    decoded = decode_to_text(data, tokenizer)
    print(f"Successfully decoded {len(decoded)} samples")

    print(f"Converting to JSONL format...")
    convert_to_rapidin_jsonl(decoded, args.output_path, args.split_ratio)

    print("Done!")


if __name__ == "__main__":
    main()
