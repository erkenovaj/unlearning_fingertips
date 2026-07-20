#!/usr/bin/env python3
"""Resumable training pipeline for TOFU unlearning experiments.

Clones open-unlearning, finetunes base models on TOFU, unlearns with 5
methods, uploads checkpoints to HuggingFace, and generates samples.

Usage:
    python train_pipeline.py                     # run all phases
    python train_pipeline.py --phase 0           # base model samples only
    python train_pipeline.py --phase 3 --model Llama-3.2-1B-Instruct
    python train_pipeline.py --phase 3 --model Llama-3.2-1B-Instruct --method GradAscent
    python train_pipeline.py --status            # show manifest summary
    python train_pipeline.py --reset STEP_KEY    # mark a step as pending
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
OU_DIR = REPO_ROOT / "open-unlearning"
MANIFEST = REPO_ROOT / "train_manifest.json"
LOG_FILE = REPO_ROOT / "train_log.jsonl"
SAMPLES_DIR = REPO_ROOT / "train_output" / "samples"

# ── Experiment constants ───────────────────────────────────────────────────────
HF_USER = "erkenovaj"

MODELS = {
    "Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "Phi-3.5-mini-instruct": "microsoft/Phi-3.5-mini-instruct",
    "Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct",
}

METHODS = ["GradAscent", "GradDiff", "NPO", "RMU", "SimNPO"]

FORGET_SPLIT = "forget10"
RETAIN_SPLIT = "retain90"
NUM_SAMPLES = 50
FINETUNE_EPOCHS = 5
UNLEARN_EPOCHS = 10
SEED = 42

RMU_LAYER = {
    "Qwen2.5-1.5B-Instruct": r"model\.layers\.14",
    "Phi-3.5-mini-instruct": r"model\.layers\.16",
    "Qwen2.5-3B-Instruct": r"model\.layers\.18",
}


# ── Manifest ───────────────────────────────────────────────────────────────────

def load_manifest():
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {"steps": {}}


def save_manifest(manifest):
    MANIFEST.write_text(json.dumps(manifest, indent=2))


def mark_done(manifest, key, **info):
    manifest["steps"][key] = {
        "status": "done",
        "timestamp": datetime.now().isoformat(),
        **info,
    }
    save_manifest(manifest)


def mark_failed(manifest, key, error):
    manifest["steps"][key] = {
        "status": "failed",
        "timestamp": datetime.now().isoformat(),
        "error": str(error)[-2000:],
    }
    save_manifest(manifest)


def is_done(manifest, key):
    step = manifest["steps"].get(key)
    return step is not None and step["status"] == "done"


# ── Logging ────────────────────────────────────────────────────────────────────

def log_step(step_type, model, method=None, **info):
    entry = {
        "step": step_type,
        "model": model,
        "method": method,
        "timestamp": datetime.now().isoformat(),
        **info,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Shell helper ───────────────────────────────────────────────────────────────

def run_cmd(cmd, cwd=None, timeout=None):
    """Run command, stream output live, return (success, combined_output)."""
    proc = subprocess.Popen(
        cmd, shell=True, cwd=str(cwd or OU_DIR),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    lines = []
    for line in proc.stdout:
        print(line, end="")
        lines.append(line)
    proc.wait()
    output = "".join(lines)
    return proc.returncode == 0, output


# ── OpenUnlearning commands ────────────────────────────────────────────────────

def finetune(model_name, split="full"):
    config_name = model_name
    task_name = f"tofu_{model_name}_{split}"

    data_override = ""
    if split == "retain90":
        data_override = (
            " data/datasets@data.train=TOFU_QA_retain"
        )

    cmd = (
        f"python src/train.py --config-name=train.yaml"
        f" experiment=finetune/tofu/default"
        f" model={config_name}"
        f"{data_override}"
        f" trainer.args.num_train_epochs={FINETUNE_EPOCHS}"
        f" trainer.args.seed={SEED}"
        f" task_name={task_name}"
    )
    return run_cmd(cmd)


def eval_model(model_name, checkpoint_path, retain_logs_path=None, task_name=None):
    config_name = model_name
    if task_name is None:
        task_name = f"eval_{Path(checkpoint_path).name}"

    retain_arg = ""
    if retain_logs_path:
        retain_arg = f" retain_logs_path={retain_logs_path}"

    cmd = (
        f"python src/eval.py --config-name=eval.yaml"
        f" experiment=eval/tofu/default"
        f" model={config_name}"
        f" model.model_args.pretrained_model_name_or_path={checkpoint_path}"
        f"{retain_arg}"
        f" task_name={task_name}"
    )
    return run_cmd(cmd)


def unlearn(model_name, method):
    config_name = model_name
    source_checkpoint = f"saves/finetune/tofu_{model_name}_full"
    retain_logs = f"saves/eval/eval_retain_{model_name}/TOFU_EVAL.json"
    task_name = f"{model_name}_{method}_{FORGET_SPLIT}"

    rmu_override = ""
    if method == "RMU":
        layer = RMU_LAYER.get(model_name, r"model\.layers\.7")
        rmu_override = f" trainer.method_args.module_regex={layer}"

    cmd = (
        f"python src/train.py --config-name=unlearn.yaml"
        f" experiment=unlearn/tofu/default"
        f" model={config_name}"
        f" model.model_args.pretrained_model_name_or_path={source_checkpoint}"
        f" trainer={method}"
        f" forget_split={FORGET_SPLIT}"
        f" retain_split={RETAIN_SPLIT}"
        f" retain_logs_path={retain_logs}"
        f" trainer.args.num_train_epochs={UNLEARN_EPOCHS}"
        f" trainer.args.seed={SEED}"
        f"{rmu_override}"
        f" task_name={task_name}"
    )
    return run_cmd(cmd)


# ── HF upload ──────────────────────────────────────────────────────────────────

def upload_checkpoint(local_path, hf_repo_id):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(hf_repo_id, exist_ok=True, private=False)
    api.upload_folder(
        folder_path=str(local_path),
        repo_id=hf_repo_id,
        repo_type="model",
    )
    return f"https://huggingface.co/{hf_repo_id}"


# ── Sample generation ──────────────────────────────────────────────────────────

def generate_samples(model_path_or_hf_id, output_path, num_samples=NUM_SAMPLES):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"    Loading model {model_path_or_hf_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_hf_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path_or_hf_id, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()

    forget_ds = load_dataset("locuslab/TOFU", FORGET_SPLIT)["train"]
    retain_ds = load_dataset("locuslab/TOFU", RETAIN_SPLIT)["train"]

    rng = random.Random(SEED)
    f_idx = rng.sample(range(len(forget_ds)), min(num_samples, len(forget_ds)))
    r_idx = rng.sample(range(len(retain_ds)), min(num_samples, len(retain_ds)))

    forget_qs = [forget_ds[i]["question"] for i in f_idx]
    retain_qs = [retain_ds[i]["question"] for i in r_idx]

    def format_prompts(questions):
        out = []
        for q in questions:
            msgs = [{"role": "user", "content": q}]
            out.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        return out

    def gen_batch(prompts, questions, domain):
        results = []
        for i, (prompt, q) in enumerate(zip(prompts, questions)):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.9)
            resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            results.append({"question": q, "response": resp, "domain": domain})
            if (i + 1) % 10 == 0:
                print(f"      {domain} {i+1}/{len(prompts)}")
        return results

    samples = []
    samples.extend(gen_batch(format_prompts(forget_qs), forget_qs, "forget"))
    samples.extend(gen_batch(format_prompts(retain_qs), retain_qs, "retain"))

    out = {
        "model": model_path_or_hf_id,
        "forget_indices": f_idx,
        "retain_indices": r_idx,
        "num_samples": len(samples),
        "samples": samples,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(out, indent=2))

    n = len(samples)
    del model
    torch.cuda.empty_cache()
    return n


# ── Phases ─────────────────────────────────────────────────────────────────────

def phase0_base_samples(manifest):
    print("\n" + "=" * 60)
    print("Phase 0: Base model samples")
    print("=" * 60)
    for model_name, hf_id in MODELS.items():
        key = f"base_{model_name}"
        if is_done(manifest, key):
            print(f"  [skip] {model_name}")
            continue
        print(f"  [run]  {model_name} ...")
        try:
            out_path = SAMPLES_DIR / f"base_{model_name}.json"
            n = generate_samples(hf_id, out_path)
            mark_done(manifest, key, hf_id=hf_id, samples=n)
            log_step("base_samples", model_name, hf_id=hf_id, num_samples=n)
            print(f"  [done] {model_name}: {n} samples")
        except Exception as e:
            mark_failed(manifest, key, e)
            print(f"  [FAIL] {model_name}: {e}")
            traceback.print_exc()


def phase1_finetune_originals(manifest):
    print("\n" + "=" * 60)
    print("Phase 1: Finetune originals (TOFU full)")
    print("=" * 60)
    for model_name in MODELS:
        key = f"finetune_{model_name}_full"
        if is_done(manifest, key):
            print(f"  [skip] {model_name}")
            continue
        print(f"  [run]  {model_name} finetune full ...")
        t0 = time.time()
        try:
            ok, output = finetune(model_name, split="full")
            elapsed = time.time() - t0
            if not ok:
                raise RuntimeError(f"finetune failed (last 2000 chars):\n{output[-2000:]}")

            checkpoint = OU_DIR / "saves" / "finetune" / f"tofu_{model_name}_full"
            hf_id = f"{HF_USER}/tofu_{model_name}_full"
            upload_checkpoint(checkpoint, hf_id)

            samples_path = SAMPLES_DIR / f"finetune_{model_name}_full.json"
            n = generate_samples(str(checkpoint), samples_path)

            mark_done(manifest, key, hf_id=hf_id, elapsed=elapsed, samples=n)
            log_step("finetune", model_name, split="full", elapsed=elapsed, hf_id=hf_id)
            print(f"  [done] {model_name}: {elapsed:.0f}s → {hf_id}")
        except Exception as e:
            mark_failed(manifest, key, e)
            print(f"  [FAIL] {model_name}: {e}")
            traceback.print_exc()


def phase2_finetune_retains(manifest):
    print("\n" + "=" * 60)
    print("Phase 2: Finetune retains (TOFU retain90)")
    print("=" * 60)
    for model_name in MODELS:
        key = f"finetune_{model_name}_retain90"
        if is_done(manifest, key):
            print(f"  [skip] {model_name}")
            continue
        print(f"  [run]  {model_name} finetune retain90 ...")
        t0 = time.time()
        try:
            ok, output = finetune(model_name, split="retain90")
            elapsed = time.time() - t0
            if not ok:
                raise RuntimeError(f"finetune failed (last 2000 chars):\n{output[-2000:]}")

            checkpoint = OU_DIR / "saves" / "finetune" / f"tofu_{model_name}_retain90"
            hf_id = f"{HF_USER}/tofu_{model_name}_retain90"
            upload_checkpoint(checkpoint, hf_id)

            samples_path = SAMPLES_DIR / f"finetune_{model_name}_retain90.json"
            n = generate_samples(str(checkpoint), samples_path)

            mark_done(manifest, key, hf_id=hf_id, elapsed=elapsed, samples=n)
            log_step("finetune", model_name, split="retain90", elapsed=elapsed, hf_id=hf_id)
            print(f"  [done] {model_name}: {elapsed:.0f}s → {hf_id}")
        except Exception as e:
            mark_failed(manifest, key, e)
            print(f"  [FAIL] {model_name}: {e}")
            traceback.print_exc()


def phase2b_eval_retains(manifest):
    print("\n" + "=" * 60)
    print("Phase 2b: Eval retain models (for unlearning metrics)")
    print("=" * 60)
    for model_name in MODELS:
        key = f"eval_retain_{model_name}"
        if is_done(manifest, key):
            print(f"  [skip] {model_name}")
            continue
        print(f"  [run]  {model_name} eval retain ...")
        t0 = time.time()
        try:
            checkpoint = f"saves/finetune/tofu_{model_name}_retain90"
            task_name = f"eval_retain_{model_name}"
            ok, output = eval_model(model_name, checkpoint, task_name=task_name)
            elapsed = time.time() - t0
            if not ok:
                raise RuntimeError(f"eval failed (last 2000 chars):\n{output[-2000:]}")

            eval_json = OU_DIR / "saves" / "eval" / task_name / "TOFU_EVAL.json"
            if not eval_json.exists():
                raise FileNotFoundError(f"Expected eval output not found: {eval_json}")

            mark_done(manifest, key, elapsed=elapsed, eval_json=str(eval_json))
            log_step("eval_retain", model_name, elapsed=elapsed)
            print(f"  [done] {model_name}: {elapsed:.0f}s")
        except Exception as e:
            mark_failed(manifest, key, e)
            print(f"  [FAIL] {model_name}: {e}")
            traceback.print_exc()


def phase3_unlearn(manifest, filter_model=None, filter_method=None):
    print("\n" + "=" * 60)
    print("Phase 3: Unlearn")
    print("=" * 60)
    for model_name in MODELS:
        if filter_model and model_name != filter_model:
            continue
        for method in METHODS:
            if filter_method and method != filter_method:
                continue
            key = f"unlearn_{model_name}_{method}"
            if is_done(manifest, key):
                print(f"  [skip] {model_name} × {method}")
                continue
            print(f"  [run]  {model_name} × {method} ...")
            t0 = time.time()
            try:
                ok, output = unlearn(model_name, method)
                elapsed = time.time() - t0
                if not ok:
                    raise RuntimeError(f"unlearn failed (last 2000 chars):\n{output[-2000:]}")

                task_name = f"{model_name}_{method}_{FORGET_SPLIT}"
                checkpoint = OU_DIR / "saves" / "unlearn" / task_name
                hf_id = f"{HF_USER}/unlearn_tofu_{model_name}_{FORGET_SPLIT}_{method}"
                upload_checkpoint(checkpoint, hf_id)

                samples_path = SAMPLES_DIR / f"unlearn_{model_name}_{method}.json"
                n = generate_samples(str(checkpoint), samples_path)

                mark_done(manifest, key, hf_id=hf_id, elapsed=elapsed, samples=n)
                log_step("unlearn", model_name, method=method, elapsed=elapsed, hf_id=hf_id)
                print(f"  [done] {model_name} × {method}: {elapsed:.0f}s → {hf_id}")
            except Exception as e:
                mark_failed(manifest, key, e)
                print(f"  [FAIL] {model_name} × {method}: {e}")
                traceback.print_exc()


# ── Status / reset ─────────────────────────────────────────────────────────────

def show_status(manifest):
    steps = manifest["steps"]
    if not steps:
        print("No steps recorded yet.")
        return

    done = sum(1 for s in steps.values() if s["status"] == "done")
    failed = sum(1 for s in steps.values() if s["status"] == "failed")
    print(f"Manifest: {done} done, {failed} failed, {len(steps)} total\n")

    for key in sorted(steps):
        s = steps[key]
        status = s["status"]
        tag = "✓" if status == "done" else "✗" if status == "failed" else "?"
        ts = s.get("timestamp", "")
        extra = ""
        if status == "done" and "hf_id" in s:
            extra = f" → {s['hf_id']}"
        elif status == "failed" and "error" in s:
            extra = f" — {s['error'][:120]}"
        print(f"  {tag} {key}  [{ts}]{extra}")


def reset_step(manifest, key):
    if key in manifest["steps"]:
        del manifest["steps"][key]
        save_manifest(manifest)
        print(f"Reset '{key}' — will re-run on next invocation.")
    else:
        print(f"Step '{key}' not found in manifest.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TOFU training pipeline")
    parser.add_argument("--phase", type=int, choices=[0, 1, 2, 23, 3],
                        help="Run a specific phase (0=base, 1=finetune full, 2=finetune retain, 23=eval retains, 3=unlearn)")
    parser.add_argument("--model", type=str, help="Filter to a single model")
    parser.add_argument("--method", type=str, help="Filter to a single method (phase 3 only)")
    parser.add_argument("--status", action="store_true", help="Show manifest summary")
    parser.add_argument("--reset", type=str, help="Reset a step key to pending")
    args = parser.parse_args()

    manifest = load_manifest()

    if args.status:
        show_status(manifest)
        return

    if args.reset:
        reset_step(manifest, args.reset)
        return

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    phases = [args.phase] if args.phase is not None else [0, 1, 2, 23, 3]

    for p in phases:
        if p == 0:
            phase0_base_samples(manifest)
        elif p == 1:
            phase1_finetune_originals(manifest)
        elif p == 2:
            phase2_finetune_retains(manifest)
        elif p == 23:
            phase2b_eval_retains(manifest)
        elif p == 3:
            phase3_unlearn(manifest, filter_model=args.model, filter_method=args.method)

    print("\n" + "=" * 60)
    show_status(manifest)


if __name__ == "__main__":
    main()
