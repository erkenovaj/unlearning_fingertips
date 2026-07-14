"""Shared utilities for unlearning detection experiments."""

import json
import os
import random
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Model registry: OpenUnlearning TOFU checkpoints (Llama-3.2-1B-Instruct)
# 1B params, bf16 — fits on any GPU with >= 4GB VRAM.
# Unlearned checkpoints: https://huggingface.co/collections/open-unlearning/tofu-unlearned-models-6860f6cf3fe35d0223d92e88
# Base checkpoints:      https://huggingface.co/collections/open-unlearning/tofu-new-models
# ---------------------------------------------------------------------------
MODELS = {
    "original": "open-unlearning/tofu_Llama-3.2-1B-Instruct_full",
    "retain": "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90",
    "rmu": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr1e-05_layer5_scoeff100_epoch10",
    "npo": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr5e-05_beta0.1_alpha2_epoch10",
    "graddiff": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha5_epoch10",
    "altpo": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr5e-05_beta0.1_alpha1_epoch10",
    "undial": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_UNDIAL_lr0.0001_beta10_alpha1_epoch10",
    "idknll": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr4e-05_alpha5_epoch10",
}

UNLEARNED_METHODS = ["rmu", "npo", "graddiff", "altpo", "undial", "idknll"]
REFERENCE_MODELS = ["original", "retain"]

DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def load_tofu_prompts(forget_fraction=10, num_samples=200, seed=42):
    """Load TOFU forget and retain question prompts."""
    from datasets import load_dataset

    forget_cfg = f"forget{forget_fraction:02d}"
    retain_cfg = f"retain{100 - forget_fraction:02d}"

    forget_ds = load_dataset("locuslab/TOFU", forget_cfg)["train"]
    retain_ds = load_dataset("locuslab/TOFU", retain_cfg)["train"]

    rng = random.Random(seed)
    f_idx = rng.sample(range(len(forget_ds)), min(num_samples, len(forget_ds)))
    r_idx = rng.sample(range(len(retain_ds)), min(num_samples, len(retain_ds)))

    forget_prompts = [f"Question: {forget_ds[i]['question']}\nAnswer:" for i in f_idx]
    retain_prompts = [f"Question: {retain_ds[i]['question']}\nAnswer:" for i in r_idx]
    return forget_prompts, retain_prompts


def load_model(model_path, dtype=torch.bfloat16, device="cuda"):
    """Load a causal LM and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True,
    )
    model.eval()
    model.to(device)
    return model, tokenizer


@torch.no_grad()
def collect_hidden_states(model, tokenizer, prompts, device, batch_size=8, max_length=128):
    """Collect last-token hidden states for all layers via single forward pass per batch.

    Returns: (n_layers+1, n_prompts, hidden_dim) float32 numpy array.
    """
    all_hidden = []

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        outputs = model(**enc, output_hidden_states=True)

        for i in range(len(batch)):
            last_pos = enc["attention_mask"][i].sum().item() - 1
            hidden = [
                hs[i, last_pos, :].detach().float().cpu().numpy()
                for hs in outputs.hidden_states
            ]
            all_hidden.append(hidden)

    arr = np.transpose(np.array(all_hidden, dtype=np.float32), (1, 0, 2))
    return arr


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def effective_rank(H):
    """Effective rank = exp(entropy of normalized eigenvalues).

    Uses Gram matrix (n x n) for efficiency when n << d.
    Returns (erank, entropy, participation_ratio).
    """
    H = H.astype(np.float64)
    H = H - H.mean(axis=0, keepdims=True)
    n = H.shape[0]
    G = H @ H.T / n
    eigvals = np.linalg.eigvalsh(G)
    eigvals = np.maximum(eigvals, 0)
    total = eigvals.sum()
    if total < 1e-12:
        return 1.0, 0.0, 1.0
    p = eigvals / total
    p = p[p > 1e-12]
    entropy = -np.sum(p * np.log(p))
    erank = float(np.exp(entropy))
    pr = float(1.0 / np.sum(p ** 2))
    return erank, float(entropy), pr


def mean_cosine(H):
    """Mean pairwise cosine similarity (excluding diagonal)."""
    H = H.astype(np.float64)
    norms = np.linalg.norm(H, axis=1, keepdims=True)
    H_norm = H / (norms + 1e-12)
    cos_sim = H_norm @ H_norm.T
    n = H.shape[0]
    upper = cos_sim[np.triu_indices(n, k=1)]
    return float(upper.mean())


def stable_rank(W):
    """Stable rank: ||W||_F^2 / ||W||_2^2."""
    W = W.astype(np.float64)
    frob_sq = np.sum(W ** 2)
    spectral = np.linalg.norm(W, ord=2)
    return float(frob_sq / (spectral ** 2 + 1e-12))


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def save_json(data, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
