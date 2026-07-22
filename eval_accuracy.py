#!/usr/bin/env python3
"""Evaluate accuracy of model responses against TOFU reference answers.

Reads generated samples from train_output/samples/ and computes ROUGE-L
and token-level exact-match accuracy against the TOFU dataset.

Usage:
    python eval_accuracy.py                          # all available samples
    python eval_accuracy.py --samples_dir train_output/samples/
    python eval_accuracy.py --filter Qwen2.5-1.5B    # substring match on filename
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from rouge_score import rouge_scorer


ROOT = Path(__file__).resolve().parent
DEFAULT_SAMPLES_DIR = ROOT / "train_output" / "samples"

SPLITS = {"forget": "forget10", "retain": "retain90"}
SEED = 42
NUM_SAMPLES = 50


def load_tofu_references(seed=SEED):
    import random
    rng = random.Random(seed)
    refs = {}
    for domain, split_name in SPLITS.items():
        ds = load_dataset("locuslab/TOFU", split_name)["train"]
        idx = rng.sample(range(len(ds)), min(NUM_SAMPLES, len(ds)))
        refs[domain] = {ds[i]["question"]: ds[i]["answer"] for i in idx}
    return refs


def normalize_text(text):
    import re
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text


def compute_token_f1(prediction, reference):
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not ref_tokens and not pred_tokens:
        return 1.0
    if not ref_tokens or not pred_tokens:
        return 0.0
    common = sum(1 for t in pred_tokens if t in ref_tokens)
    precision = common / len(pred_tokens) if pred_tokens else 0
    recall = common / len(ref_tokens) if ref_tokens else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_samples(samples_path, references, scorer):
    with open(samples_path) as f:
        data = json.load(f)

    results = {}
    for domain in ["forget", "retain"]:
        domain_samples = [s for s in data["samples"] if s["domain"] == domain]
        if not domain_samples:
            continue

        rouge_scores = []
        token_f1_scores = []
        exact_matches = 0

        for sample in domain_samples:
            q = sample["question"]
            pred = sample["response"]
            ref = references.get(domain, {}).get(q, None)
            if ref is None:
                continue

            rs = scorer.score(ref, pred)
            rouge_l = rs["rougeL"].fmeasure
            rouge_scores.append(rouge_l)

            f1 = compute_token_f1(pred, ref)
            token_f1_scores.append(f1)

            if normalize_text(pred) == normalize_text(ref):
                exact_matches += 1

        n = len(rouge_scores)
        if n == 0:
            continue

        results[domain] = {
            "n": n,
            "rouge_l_mean": float(np.mean(rouge_scores)),
            "rouge_l_std": float(np.std(rouge_scores)),
            "token_f1_mean": float(np.mean(token_f1_scores)),
            "exact_match_pct": exact_matches / n * 100,
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_dir", type=str, default=str(DEFAULT_SAMPLES_DIR))
    parser.add_argument("--filter", type=str, default=None, help="Substring filter on sample filenames")
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    if not samples_dir.exists():
        print(f"Samples dir not found: {samples_dir}")
        sys.exit(1)

    print("Loading TOFU references ...")
    references = load_tofu_references()
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    sample_files = sorted(f for f in samples_dir.glob("*.json") if f.name != "accuracy_summary.json")
    if args.filter:
        sample_files = [f for f in sample_files if args.filter in f.name]

    if not sample_files:
        print("No sample files found.")
        sys.exit(1)

    all_results = {}
    for sf in sample_files:
        name = sf.stem
        res = evaluate_samples(sf, references, scorer)
        if res:
            all_results[name] = res

    print()
    header = f"{'Model':<50} {'Domain':<10} {'ROUGE-L':>10} {'TokenF1':>10} {'ExactMatch':>12} {'N':>5}"
    print(header)
    print("-" * len(header))

    for name, domains in sorted(all_results.items()):
        for domain in ["forget", "retain"]:
            if domain not in domains:
                continue
            r = domains[domain]
            print(
                f"{name:<50} {domain:<10} "
                f"{r['rouge_l_mean']:>9.4f} "
                f"{r['token_f1_mean']:>9.4f} "
                f"{r['exact_match_pct']:>11.1f}% "
                f"{r['n']:>5}"
            )

    out_path = samples_dir / "accuracy_summary.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()
