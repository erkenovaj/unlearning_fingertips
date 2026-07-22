#!/usr/bin/env python3
"""Regenerate samples for all models with concise prompting.

Usage:
    python regen_samples.py                    # all models
    python regen_samples.py --model Qwen2.5-1.5B-Instruct  # single model
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from train_pipeline import (
    MODELS, FORGET_SPLIT, RETAIN_SPLIT, SEED, NUM_SAMPLES,
    SAMPLES_DIR, HF_USER, generate_samples,
    load_manifest, mark_done, mark_failed, is_done,
)

UNLEARN_METHODS = ["GradAscent", "GradDiff", "NPO", "RMU", "SimNPO"]


def collect_model_paths(filter_model=None):
    manifest = load_manifest()
    paths = {}

    for model_name in MODELS:
        if filter_model and model_name != filter_model:
            continue

        paths[f"base_{model_name}"] = MODELS[model_name]

        key_full = f"finetune_{model_name}_full"
        if is_done(manifest, key_full):
            hf_id = manifest["steps"][key_full].get("hf_id")
            if hf_id:
                paths[f"finetune_{model_name}_full"] = hf_id

        key_retain = f"finetune_{model_name}_retain90"
        if is_done(manifest, key_retain):
            hf_id = manifest["steps"][key_retain].get("hf_id")
            if hf_id:
                paths[f"finetune_{model_name}_retain90"] = hf_id

        for method in UNLEARN_METHODS:
            key = f"unlearn_{model_name}_{method}"
            if is_done(manifest, key):
                hf_id = manifest["steps"][key].get("hf_id")
                if hf_id:
                    paths[f"unlearn_{model_name}_{method}"] = hf_id

    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=str(SAMPLES_DIR / "concise"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_paths = collect_model_paths(args.model)
    print(f"Found {len(model_paths)} models to regenerate:\n")
    for name in sorted(model_paths):
        print(f"  {name:<50} → {model_paths[name]}")
    print()

    for name in sorted(model_paths):
        hf_id = model_paths[name]
        out_path = output_dir / f"{name}.json"
        if out_path.exists():
            print(f"  [skip] {name} (already exists)")
            continue
        print(f"  [run]  {name} ...")
        t0 = time.time()
        try:
            n = generate_samples(hf_id, out_path)
            elapsed = time.time() - t0
            print(f"  [done] {name}: {n} samples in {elapsed:.0f}s")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [FAIL] {name}: {e} ({elapsed:.0f}s)")
            import traceback
            traceback.print_exc()

    print(f"\nSamples saved to {output_dir}")


if __name__ == "__main__":
    main()
