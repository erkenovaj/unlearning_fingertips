#!/usr/bin/env python3
"""Evaluate accuracy of model responses against TOFU reference answers.

Reads generated samples from train_output/samples/ and computes:
- ROUGE-L (response vs reference)
- TokenF1 (response vs reference)  
- Containment (does the response contain the reference answer?)
- OU eval metrics (from TOFU_EVAL.json if available)

Usage:
    python eval_accuracy.py                          # all available samples
    python eval_accuracy.py --filter Qwen2.5-1.5B    # substring match on filename
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from rouge_score import rouge_scorer


ROOT = Path(__file__).resolve().parent
DEFAULT_SAMPLES_DIR = ROOT / "train_output" / "samples"
OU_EVAL_DIR = ROOT / "open-unlearning" / "saves" / "eval"

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


def compute_containment(prediction, reference):
    pred_norm = normalize_text(prediction)
    ref_norm = normalize_text(reference)
    ref_words = ref_norm.split()
    if not ref_words:
        return 0.0
    contained = all(w in pred_norm for w in ref_words)
    partial = sum(1 for w in ref_words if w in pred_norm) / len(ref_words)
    return 1.0 if contained else partial


def load_ou_eval_results():
    results = {}
    if not OU_EVAL_DIR.exists():
        return results
    for eval_dir in sorted(OU_EVAL_DIR.iterdir()):
        if not eval_dir.is_dir():
            continue
        eval_json = eval_dir / "TOFU_EVAL.json"
        if not eval_json.exists():
            continue
        try:
            data = json.loads(eval_json.read_text())
            name = eval_dir.name
            metrics = {}
            for key in ["model_utility", "forget_Q_A_Prob", "forget_Q_A_ROUGE",
                         "retain_Q_A_Prob", "retain_Q_A_ROUGE",
                         "forget_truth_ratio", "retain_Truth_Ratio",
                         "extraction_strength", "privleak"]:
                if key in data and isinstance(data[key], dict) and "agg_value" in data[key]:
                    metrics[key] = data[key]["agg_value"]
            if metrics:
                results[name] = metrics
        except Exception:
            continue
    return results


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
        containment_scores = []
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

            c = compute_containment(pred, ref)
            containment_scores.append(c)

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
            "containment_mean": float(np.mean(containment_scores)),
            "exact_match_pct": exact_matches / n * 100,
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_dir", type=str, default=str(DEFAULT_SAMPLES_DIR))
    parser.add_argument("--filter", type=str, default=None)
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

    ou_results = load_ou_eval_results()

    print()
    print("=" * 120)
    print("SAMPLE-BASED METRICS (ROUGE-L / TokenF1 / Containment)")
    print("=" * 120)
    header = f"{'Model':<48} {'Domain':<8} {'ROUGE-L':>8} {'TokF1':>7} {'Contain':>8} {'EM%':>6} {'N':>4}"
    print(header)
    print("-" * len(header))

    for name, domains in sorted(all_results.items()):
        for domain in ["forget", "retain"]:
            if domain not in domains:
                continue
            r = domains[domain]
            print(
                f"{name:<48} {domain:<8} "
                f"{r['rouge_l_mean']:>7.4f} "
                f"{r['token_f1_mean']:>6.4f} "
                f"{r['containment_mean']:>7.4f} "
                f"{r['exact_match_pct']:>5.1f} "
                f"{r['n']:>4}"
            )
        print()

    if ou_results:
        print("=" * 120)
        print("OU EVAL METRICS (from TOFU_EVAL.json — probabilities, not text matching)")
        print("=" * 120)
        ou_header = f"{'Eval Dir':<58} {'ModUtil':>8} {'F_QProb':>8} {'F_ROUGE':>8} {'R_QProb':>8} {'R_ROUGE':>8} {'F_TR':>8} {'ES':>8}"
        print(ou_header)
        print("-" * len(ou_header))
        for name, m in sorted(ou_results.items()):
            print(
                f"{name:<58} "
                f"{m.get('model_utility', float('nan')):>8.4f} "
                f"{m.get('forget_Q_A_Prob', float('nan')):>8.4f} "
                f"{m.get('forget_Q_A_ROUGE', float('nan')):>8.4f} "
                f"{m.get('retain_Q_A_Prob', float('nan')):>8.4f} "
                f"{m.get('retain_Q_A_ROUGE', float('nan')):>8.4f} "
                f"{m.get('forget_truth_ratio', float('nan')):>8.4f} "
                f"{m.get('extraction_strength', float('nan')):>8.4f}"
            )
        print()

    out_path = samples_dir / "accuracy_summary.json"
    with open(out_path, "w") as f:
        json.dump({"sample_metrics": all_results, "ou_eval_metrics": ou_results}, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
