#!/usr/bin/env python3
"""TOFU unlearning-trace detection for XPU (Intel GPUs).

Compares pre-logit activations from original vs unlearned Phi-1.5 checkpoints
on the TOFU forget10 split, plus a within-model sanity check.

Fixes the acc=1 problem by:
  - Using a small probe (hidden_dim=16) to prevent overfitting
  - Always normalizing features
  - Comparing forget-set detection vs within-model (forget vs retain) detection

Usage:
    python reproduction_tofu_xpu.py \
        --num_samples 100 --act_new_tokens 32 \
        --log_dir ./logs
"""

import argparse
import io
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "xpu" if torch.xpu.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

TOFU_REPO = "locuslab/phi_grad_ascent_1e-05_forget10"
ORIGINAL_REVISION = "checkpoint-625"
DTYPE_ALIASES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def list_checkpoint_branches(repo):
    from huggingface_hub import list_repo_refs
    refs = list_repo_refs(repo)
    names = [b.name for b in refs.branches if b.name.startswith("checkpoint-")]
    names.sort(key=lambda n: int(n.split("-")[-1]))
    return names


def _model_device(model):
    return next(model.parameters()).device


def build_tofu_prompts(split, num_samples, seed=42):
    from datasets import load_dataset
    ds = load_dataset("locuslab/TOFU", split)["train"]
    rng = random.Random(seed)
    idx = rng.sample(range(len(ds)), min(num_samples, len(ds)))
    return [f"Question: {ds[i]['question']}\nAnswer:" for i in idx]


@torch.no_grad()
def get_pre_logit_activations(model, tokenizer, prompt, max_new_tokens=32):
    inputs = tokenizer(prompt, return_tensors="pt").to(_model_device(model))
    activations = []

    def hook_fn(module, inp, out):
        activations.append(inp[0][:, -1, :].detach().float().cpu())

    handle = model.lm_head.register_forward_hook(hook_fn)
    try:
        model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    finally:
        handle.remove()
    return torch.cat(activations, dim=0).mean(dim=0)


def build_activation_features(repo, revision, prompts, max_new_tokens=32, dtype=torch.bfloat16):
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        repo, revision=revision, torch_dtype=dtype,
    ).to(DEVICE)
    model.eval()
    feats = [get_pre_logit_activations(model, tokenizer, p, max_new_tokens) for p in prompts]
    del model
    if DEVICE == "xpu":
        torch.xpu.empty_cache()
    elif DEVICE == "cuda":
        torch.cuda.empty_cache()
    return torch.stack(feats, dim=0).numpy().astype(np.float32)


@torch.no_grad()
def log_sample_outputs(repo, revisions, prompts, max_new_tokens=64, dtype=torch.bfloat16,
                       log_dir=None, n=5):
    """Generate text samples for each revision and save to a JSON file."""
    if not log_dir or n < 1:
        return
    samples = []
    for rev_label, rev in revisions:
        tokenizer = AutoTokenizer.from_pretrained(repo, revision=rev)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            repo, revision=rev, torch_dtype=dtype,
        ).to(DEVICE)
        model.eval()
        for i, p in enumerate(prompts[:n]):
            inp = tokenizer(p, return_tensors="pt").to(DEVICE)
            gen = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            out = tokenizer.decode(gen[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
            samples.append({"revision": rev_label, "prompt_index": i, "prompt": p, "generation": out})
        del model
        if DEVICE == "xpu":
            torch.xpu.empty_cache()
        elif DEVICE == "cuda":
            torch.cuda.empty_cache()
    path = os.path.join(log_dir, "sample_outputs.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)
    print(f"  [samples] -> {path}  ({len(samples)} generations)")


def normalize_features(X):
    mu, std = X.mean(axis=0, keepdims=True), X.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    return (X - mu) / std, (mu, std)


class BinaryClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _save_dataset(X_raw, y, norm_params, prompts_list, metadata, log_dir, tag):
    """Save the raw features, labels, normalization params, prompts, and metadata."""
    os.makedirs(log_dir, exist_ok=True)
    base = os.path.join(log_dir, f"dataset_{tag}")
    mu, std = norm_params
    np.savez_compressed(
        f"{base}.npz",
        X_raw=X_raw.astype(np.float32),
        y=y.astype(np.float32),
        norm_mu=mu.astype(np.float32),
        norm_std=std.astype(np.float32),
    )
    with open(f"{base}.json", "w", encoding="utf-8") as f:
        json.dump({**metadata, "tag": tag, "prompts": prompts_list}, f, indent=2, default=str)
    print(f"  [dataset] -> {base}.*  ({X_raw.shape[0]} samples, {X_raw.shape[1]} dims)")


def _save_splits(tr_idx, val_idx, te_idx, log_dir, tag):
    """Append train/val/test split indices to an existing dataset metadata file."""
    base = os.path.join(log_dir, f"dataset_{tag}")
    path = f"{base}.json"
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["train_indices"] = [int(i) for i in tr_idx]
    data["val_indices"]   = [int(i) for i in val_idx]
    data["test_indices"]  = [int(i) for i in te_idx]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

def _make_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32),
                       torch.as_tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _save_training_plot(metrics, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    epochs = [m["epoch"] for m in metrics]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, [m["train_loss"] for m in metrics], label="train")
    ax1.plot(epochs, [m["val_loss"] for m in metrics],   label="val")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.legend(); ax1.set_title("Loss")
    ax2.plot(epochs, [m["train_acc"]  for m in metrics], label="train")
    ax2.plot(epochs, [m["val_acc"]    for m in metrics], label="val")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy"); ax2.legend(); ax2.set_title("Accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [plot] -> {path}")


def train_mlp(X, y, epochs=200, lr=3e-4, batch_size=32, weight_decay=1e-4, device=DEVICE,
              log_dir=None, tag=""):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    indices = np.arange(len(X))
    tr_idx, tmp_idx, y_tr, y_tmp = train_test_split(indices, y, test_size=0.3, random_state=42, stratify=y)
    val_idx, te_idx = train_test_split(tmp_idx, test_size=0.5, random_state=42, stratify=y_tmp)
    X_tr, X_val, X_te = X[tr_idx], X[val_idx], X[te_idx]
    y_tr_, y_val_, y_te_ = y[tr_idx], y[val_idx], y[te_idx]
    train_loader = _make_loader(X_tr, y_tr_, batch_size, shuffle=True)
    val_loader = _make_loader(X_val, y_val_, batch_size, shuffle=False)
    test_loader = _make_loader(X_te, y_te_, batch_size, shuffle=False)

    model = BinaryClassifier(X.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def _run_loader(loader, train=False):
        if train:
            model.train()
        else:
            model.eval()
        total_loss, total = 0.0, 0
        preds, gts = [], []
        for xb, yb in loader:
            if train:
                optimizer.zero_grad()
            logits = model(xb.to(device))
            loss = criterion(logits, yb.to(device))
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(yb)
            total += len(yb)
            logit = logits.detach() if train else logits
            preds.append((torch.sigmoid(logit) > 0.5).long().cpu())
            gts.append(yb.long())
        return total_loss / total, accuracy_score(torch.cat(gts), torch.cat(preds))

    metrics = []
    best_val, best_state = -1.0, None
    for epoch in range(epochs):
        tr_loss, tr_acc = _run_loader(train_loader, train=True)
        val_loss, val_acc = _run_loader(val_loader)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        metrics.append({
            "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })
    if best_state is not None:
        model.load_state_dict(best_state)
    test_acc = _run_loader(test_loader)[1]
    print(f"  best val acc: {best_val:.4f} | test acc: {test_acc:.4f}")

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        prefix = f"mlp_{tag}{len(X)}samples"
        json_path = os.path.join(log_dir, f"{prefix}_metrics.json")
        with io.open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"  [log] -> {json_path}")
        _save_training_plot(metrics, os.path.join(log_dir, f"{prefix}_curves.png"))

    return test_acc, tr_idx, val_idx, te_idx


def run_comparison(repo, orig_revision, unlearn_revision, prompts, holdout_prompts,
                   max_new_tokens, dtype, log_dir, tag):
    """Run two comparisons:
       1.  Original vs unlearned on forget prompts.
       2.  Unlearned model on forget prompts vs unlearned on holdout prompts
           (within-model sanity check).
    """
    print(f"\n[{tag}] building original activations on forget prompts...")
    Xo = build_activation_features(repo, orig_revision, prompts,
                                   max_new_tokens=max_new_tokens, dtype=dtype)
    print(f"[{tag}] building unlearned activations on forget prompts...")
    Xu = build_activation_features(repo, unlearn_revision, prompts,
                                   max_new_tokens=max_new_tokens, dtype=dtype)

    # ---- Comparison 1: original vs unlearned on forget set ----
    X1_raw = np.concatenate([Xo, Xu], axis=0)
    y1 = np.array([0] * len(Xo) + [1] * len(Xu))
    X1, (mu1, std1) = normalize_features(X1_raw)
    print(f"\n[{tag}] original vs unlearned (forget set):")
    if log_dir:
        _save_dataset(X1_raw, y1, (mu1, std1),
                      prompts + prompts,  # same prompts, two model outputs
                      {"comparison": "original_vs_unlearned", "repo": repo,
                       "revision_0": orig_revision, "revision_1": unlearn_revision,
                       "label_0": "original", "label_1": "unlearned",
                       "n_original": len(Xo), "n_unlearned": len(Xu)},
                      log_dir, f"{tag}_forget")
    forget_acc, *forget_splits = train_mlp(X1, y1, log_dir=log_dir, tag=f"{tag}_forget_")
    if log_dir:
        _save_splits(*forget_splits, log_dir, f"{tag}_forget")

    # ---- Comparison 2: unlearned on forget vs unlearned on holdout ----
    print(f"[{tag}] building unlearned activations on holdout prompts...")
    Xh = build_activation_features(repo, unlearn_revision, holdout_prompts,
                                   max_new_tokens=max_new_tokens, dtype=dtype)
    X2_raw = np.concatenate([Xu, Xh], axis=0)
    y2 = np.array([0] * len(Xu) + [1] * len(Xh))
    X2, (mu2, std2) = normalize_features(X2_raw)
    print(f"\n[{tag}] unlearned model: forget vs holdout (within-model):")
    if log_dir:
        _save_dataset(X2_raw, y2, (mu2, std2),
                      prompts + holdout_prompts,
                      {"comparison": "within_model", "repo": repo,
                       "revision": unlearn_revision,
                       "label_0": "unlearned_forget", "label_1": "unlearned_holdout",
                       "n_forget": len(Xu), "n_holdout": len(Xh)},
                      log_dir, f"{tag}_within")
    within_acc, *within_splits = train_mlp(X2, y2, log_dir=log_dir, tag=f"{tag}_within_")
    if log_dir:
        _save_splits(*within_splits, log_dir, f"{tag}_within")

    return forget_acc, within_acc


class _Tee:
    def __init__(self, file_path):
        self.file = io.open(file_path, "a", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def main():
    p = argparse.ArgumentParser(
        description="TOFU unlearning-trace detection (XPU).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--act_new_tokens", type=int, default=32)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--results_file", default="./results_tofu.json")
    p.add_argument("--log_dir", default=None)
    args = p.parse_args()

    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        sys.stdout = _Tee(os.path.join(args.log_dir, f"run_{ts}.log"))

    if DEVICE == "xpu":
        vram_gb = torch.xpu.get_device_properties(0).total_memory / 1e9
        print(f"[device] XPU: {torch.xpu.get_device_name(0)} | VRAM {vram_gb:.1f} GB")
    elif DEVICE == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[device] CUDA: {torch.cuda.get_device_name(0)} | VRAM {vram_gb:.1f} GB")
    else:
        print("[device] CPU (no GPU available)")
    dtype = DTYPE_ALIASES.get(args.dtype)

    print("\n[config] discovering TOFU checkpoint branches...")
    branches = list_checkpoint_branches(TOFU_REPO)
    print(f"  branches: {branches}")
    orig_rev = ORIGINAL_REVISION
    if orig_rev not in branches:
        orig_rev = branches[0]
    candidates = [b for b in branches if b != orig_rev]
    unlearn_rev = candidates[-1] if candidates else branches[-1]
    print(f"  original  -> {TOFU_REPO} @ {orig_rev}")
    print(f"  unlearned -> {TOFU_REPO} @ {unlearn_rev}")

    print("\n[config] building prompts...")
    forget_prompts = build_tofu_prompts("forget10", args.num_samples)
    holdout_prompts = build_tofu_prompts("forget10", args.num_samples, seed=99)
    print(f"  forget prompts:  {len(forget_prompts)}")
    print(f"  holdout prompts: {len(holdout_prompts)}")

    log_sample_outputs(TOFU_REPO, [("original", orig_rev), ("unlearned", unlearn_rev)],
                       forget_prompts, max_new_tokens=args.act_new_tokens, dtype=dtype,
                       log_dir=args.log_dir, n=5)

    forget_acc, within_acc = run_comparison(
        TOFU_REPO, orig_rev, unlearn_rev,
        forget_prompts, holdout_prompts,
        args.act_new_tokens, dtype, args.log_dir, "forget10",
    )

    print(f"\n{'='*50}")
    print(f"RESULTS:")
    print(f"  forget10 (orig vs unlearn):   {forget_acc:.4f}")
    print(f"  within-model (forget vs hold): {within_acc:.4f}")
    print(f"{'='*50}")
    print(f"Interpretation:")
    print(f"  If forget10 >> within-model → probe detects unlearning, not checkpoint noise.")
    print(f"  If both ≈ 0.5              → no detectable trace.")
    print(f"  If both ≈ 1.0              → probe is memorizing (still overfitting).")

    results = {
        "dataset": "TOFU_forget10",
        "num_samples": args.num_samples,
        "forget_accuracy": float(forget_acc),
        "within_accuracy": float(within_acc),
    }
    os.makedirs(os.path.dirname(args.results_file) or ".", exist_ok=True)
    with open(args.results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] results -> {args.results_file}")


if __name__ == "__main__":
    main()
