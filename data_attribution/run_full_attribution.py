#!/usr/bin/env python3
"""
Run full data attribution experiment across all generations.
For each generation:
1. Download data and checkpoint from HuggingFace
2. Convert PKL to JSONL if needed
3. Run RapidIn attribution
4. Upload results to HuggingFace
5. Clean up local files to save space
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi, upload_file

# Configuration
HF_REPO = "zaizaiiiii/model-collapse-experiment"
BASE_MODEL = "facebook/opt-125m"
NUM_GENERATIONS = 5
WORK_DIR = Path("./attribution_work")
RESULTS_DIR = Path("./attribution_results")


def setup_directories():
    """Create working directories"""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (WORK_DIR / "data").mkdir(exist_ok=True)
    (WORK_DIR / "checkpoints").mkdir(exist_ok=True)
    (WORK_DIR / "configs").mkdir(exist_ok=True)


def download_generation_data(gen: int):
    """Download data for a specific generation"""
    print(f"\n{'='*60}")
    print(f"Downloading data for Generation {gen}")
    print('='*60)

    data_dir = WORK_DIR / "data" / f"gen_{gen}"
    data_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = WORK_DIR / "checkpoints"

    # Download checkpoint for this generation
    ckpt_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=f"opt-125m/gen_{gen}/best.ckpt",
        local_dir=str(ckpt_dir)
    )
    print(f"Downloaded checkpoint: {ckpt_path}")

    # Download training data
    if gen == 0:
        # Gen 0 uses WikiText-2 (training_data.jsonl)
        train_path = hf_hub_download(
            repo_id=HF_REPO,
            filename="opt-125m/gen_0/training_data.jsonl",
            local_dir=str(data_dir)
        )
        print(f"Downloaded training data: {train_path}")
        return {
            "checkpoint": ckpt_path,
            "training_data": train_path,
            "needs_conversion": False
        }
    else:
        # Gen 1-4 use synthetic data from previous generation
        pkl_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=f"opt-125m/gen_{gen-1}/synthetic_data.pkl",
            local_dir=str(data_dir)
        )
        print(f"Downloaded synthetic data: {pkl_path}")
        return {
            "checkpoint": ckpt_path,
            "training_data": pkl_path,
            "needs_conversion": True
        }


def convert_pkl_to_jsonl(pkl_path: str, output_path: str):
    """Convert PKL file to JSONL format"""
    print(f"Converting {pkl_path} to JSONL...")

    cmd = [
        sys.executable,
        "convert_pkl_to_jsonl.py",
        "--pkl_path", pkl_path,
        "--output_path", output_path,
        "--model_name", BASE_MODEL
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).parent))
    if result.returncode != 0:
        print(f"Error converting PKL: {result.stderr}")
        raise RuntimeError(f"PKL conversion failed: {result.stderr}")

    print(f"Converted to: {output_path}")
    return output_path


def create_test_dataset():
    """Create test dataset if it doesn't exist"""
    test_path = WORK_DIR / "data" / "collapse_test.jsonl"
    if not test_path.exists():
        print("Creating test dataset...")
        cmd = [
            sys.executable,
            "create_test_data.py",
            "--output_path", str(test_path)
        ]
        subprocess.run(cmd, cwd=str(Path(__file__).parent))
    return str(test_path)


def create_config(gen: int, train_data_path: str, test_data_path: str, checkpoint_path: str):
    """Create RapidIn config for a generation"""
    config = {
        "data": {
            "train_data_path": train_data_path,
            "test_data_path": test_data_path,
            "use_lm_format": True  # Use simple LM format for OPT
        },
        "influence": {
            "outdir": str(RESULTS_DIR / f"gen_{gen}"),
            "seed": 42,
            "cal_words_infl": True,  # Enable token-wise attribution
            "grads_path": str(WORK_DIR / f"gradients/gen_{gen}"),
            "save_to_grads_path": True,
            "load_from_grads_path": False,
            "n_threads": 1,
            "top_k": 100,
            "RapidGrad": {
                "enable": True,
                "RapidGrad_K": 65536,
                "shuffle_lambda": 20
            }
        },
        "model": {
            "model_path": checkpoint_path,
            "base_model_name": BASE_MODEL,  # For Lightning checkpoint loading
            "max_length": 512,
            "load_in_4bit": False
        }
    }

    config_path = WORK_DIR / "configs" / f"gen_{gen}_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Created config: {config_path}")
    return str(config_path)


def run_rapidin(config_path: str, stage: str = "caching"):
    """Run RapidIn attribution"""
    print(f"\nRunning RapidIn {stage} stage...")

    rapidin_dir = Path(__file__).parent / "RapidIn"
    cmd = [
        sys.executable,
        str(rapidin_dir / "MP_main.py"),
        f"--config={config_path}"
    ]

    # Set environment for proper CUDA handling
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        print(f"RapidIn {stage} error:")
        print(result.stderr)
        raise RuntimeError(f"RapidIn {stage} failed")

    print(f"RapidIn {stage} complete")
    return result.stdout


def upload_results_to_hf(gen: int):
    """Upload attribution results to HuggingFace"""
    print(f"\nUploading Gen {gen} results to HuggingFace...")

    api = HfApi()
    results_dir = RESULTS_DIR / f"gen_{gen}"

    if not results_dir.exists():
        print(f"No results found for Gen {gen}")
        return

    # Find result files
    for result_file in results_dir.glob("*.json"):
        hf_path = f"opt-125m/gen_{gen}/attribution_results/{result_file.name}"

        try:
            api.upload_file(
                path_or_fileobj=str(result_file),
                path_in_repo=hf_path,
                repo_id=HF_REPO,
                repo_type="model"
            )
            print(f"Uploaded: {hf_path}")
        except Exception as e:
            print(f"Failed to upload {result_file.name}: {e}")


def cleanup_generation(gen: int):
    """Clean up local files for a generation to save space"""
    print(f"\nCleaning up Gen {gen} local files...")

    # Remove gradients (largest files)
    grad_dir = WORK_DIR / f"gradients/gen_{gen}"
    if grad_dir.exists():
        shutil.rmtree(grad_dir)
        print(f"Removed gradient cache: {grad_dir}")

    # Remove checkpoint
    ckpt_dir = WORK_DIR / "checkpoints" / "opt-125m" / f"gen_{gen}"
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
        print(f"Removed checkpoint: {ckpt_dir}")

    # Remove training data
    data_dir = WORK_DIR / "data" / f"gen_{gen}"
    if data_dir.exists():
        shutil.rmtree(data_dir)
        print(f"Removed data: {data_dir}")


def run_generation_attribution(gen: int):
    """Run complete attribution pipeline for one generation"""
    print(f"\n{'='*60}")
    print(f"RUNNING ATTRIBUTION FOR GENERATION {gen}")
    print('='*60)

    try:
        # Step 1: Download data
        data_info = download_generation_data(gen)

        # Step 2: Convert PKL to JSONL if needed
        if data_info["needs_conversion"]:
            jsonl_path = str(WORK_DIR / "data" / f"gen_{gen}" / "training_data.jsonl")
            train_data_path = convert_pkl_to_jsonl(data_info["training_data"], jsonl_path)
        else:
            train_data_path = data_info["training_data"]

        # Step 3: Create test dataset
        test_data_path = create_test_dataset()

        # Step 4: Create config
        config_path = create_config(
            gen=gen,
            train_data_path=train_data_path,
            test_data_path=test_data_path,
            checkpoint_path=data_info["checkpoint"]
        )

        # Step 5: Run RapidIn attribution
        run_rapidin(config_path)

        # Step 6: Upload results to HuggingFace
        upload_results_to_hf(gen)

        # Step 7: Cleanup to save space
        cleanup_generation(gen)

        print(f"\nGeneration {gen} attribution COMPLETE!")
        return True

    except Exception as e:
        print(f"\nGeneration {gen} attribution FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run full data attribution experiment')
    parser.add_argument('--start_gen', type=int, default=0, help='Starting generation')
    parser.add_argument('--end_gen', type=int, default=4, help='Ending generation (inclusive)')
    parser.add_argument('--skip_upload', action='store_true', help='Skip HuggingFace upload')
    parser.add_argument('--skip_cleanup', action='store_true', help='Skip cleanup after each gen')

    args = parser.parse_args()

    print("="*60)
    print("MODEL COLLAPSE DATA ATTRIBUTION EXPERIMENT")
    print("="*60)
    print(f"Generations: {args.start_gen} to {args.end_gen}")
    print(f"HuggingFace repo: {HF_REPO}")
    print(f"Base model: {BASE_MODEL}")

    # Setup
    setup_directories()

    # Run attribution for each generation
    results = {}
    for gen in range(args.start_gen, args.end_gen + 1):
        success = run_generation_attribution(gen)
        results[gen] = "SUCCESS" if success else "FAILED"

    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print("\nResults summary:")
    for gen, status in results.items():
        print(f"  Generation {gen}: {status}")


if __name__ == "__main__":
    main()
