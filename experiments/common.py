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
    "idknll": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr3e-05_alpha5_epoch10",
}

UNLEARNED_METHODS = ["rmu", "npo", "graddiff", "altpo", "undial", "idknll"]
REFERENCE_MODELS = ["original", "retain"]

PHI_MODELS = {
    "original": "locuslab/tofu_ft_phi-1.5",
    "retain": "locuslab/tofu_ft_retain90_phi-1.5",
    "grad_ascent": "locuslab/phi_grad_ascent_1e-05_forget10",
    "grad_diff": "locuslab/phi_grad_diff_1e-05_forget10",
    "kl": "locuslab/phi_KL_1e-05_forget10",
    "idk": "locuslab/phi_idk_1e-05_forget10",
}

PHI_REVISIONS = {
    "grad_ascent": "checkpoint-60",
    "grad_diff": "checkpoint-60",
    "kl": "checkpoint-60",
    "idk": "checkpoint-48",
}

PHI_TOKENIZER_PATH = "microsoft/phi-1.5"

PHI_UNLEARNED_METHODS = ["grad_ascent", "grad_diff", "kl", "idk"]
PHI_REFERENCE_MODELS = ["original", "retain"]

STRONG_MODELS = {
    "original": "erkenovaj/strong_tofu_Qwen2.5-1.5B-Instruct_full",
    "retain": "erkenovaj/strong_tofu_Qwen2.5-1.5B-Instruct_retain90",
    "rmu": "erkenovaj/unlearn_strong_tofu_Qwen2.5-1.5B-Instruct_forget10_RMU",
    "npo": "erkenovaj/unlearn_strong_tofu_Qwen2.5-1.5B-Instruct_forget10_NPO",
    "graddiff": "erkenovaj/unlearn_strong_tofu_Qwen2.5-1.5B-Instruct_forget10_GradDiff",
    "altpo": "erkenovaj/unlearn_strong_tofu_Qwen2.5-1.5B-Instruct_forget10_SimNPO",
    "undial": "erkenovaj/unlearn_strong_tofu_Qwen2.5-1.5B-Instruct_forget10_GradAscent",
    "idknll": "erkenovaj/unlearn_strong_tofu_Qwen2.5-1.5B-Instruct_forget10_NPO",
}

STRONG_UNLEARNED_METHODS = ["rmu", "npo", "graddiff", "altpo", "undial", "idknll"]
STRONG_REFERENCE_MODELS = ["original", "retain"]

REGISTRIES = {
    "llama": (MODELS, UNLEARNED_METHODS, REFERENCE_MODELS),
    "phi": (PHI_MODELS, PHI_UNLEARNED_METHODS, PHI_REFERENCE_MODELS),
    "strong": (STRONG_MODELS, STRONG_UNLEARNED_METHODS, STRONG_REFERENCE_MODELS),
}

DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def load_tofu_prompts(forget_fraction=10, num_samples=200, seed=42):
    """Load TOFU forget and retain question prompts as raw question strings."""
    from datasets import load_dataset

    forget_cfg = f"forget{forget_fraction:02d}"
    retain_cfg = f"retain{100 - forget_fraction:02d}"

    forget_ds = load_dataset("locuslab/TOFU", forget_cfg)["train"]
    retain_ds = load_dataset("locuslab/TOFU", retain_cfg)["train"]

    rng = random.Random(seed)
    f_idx = rng.sample(range(len(forget_ds)), min(num_samples, len(forget_ds)))
    r_idx = rng.sample(range(len(retain_ds)), min(num_samples, len(retain_ds)))

    forget_questions = [forget_ds[i]["question"] for i in f_idx]
    retain_questions = [retain_ds[i]["question"] for i in r_idx]
    return forget_questions, retain_questions


def format_prompts(questions, tokenizer, raw=False):
    """Apply the model's chat template to a list of raw questions.

    Llama-3.2-1B-Instruct checkpoints (and the OpenUnlearning checkpoints
    derived from them) were trained/unlearned with the tokenizer's chat
    template; sending raw "Question: ... \\nAnswer:" prompts produces
    off-distribution activations dominated by template mismatch rather than
    memorization signal. Use the chat template by default.

    Set raw=True to format as "Question: {q}\\nAnswer:" (legacy/ablation).
    """
    if raw:
        return [f"Question: {q}\nAnswer:" for q in questions]

    prompts = []
    for q in questions:
        msgs = [{"role": "user", "content": q}]
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)
    return prompts


def load_model(model_path, dtype=torch.bfloat16, device="cuda", revision=None, tokenizer_path=None):
    """Load a causal LM and tokenizer."""
    tok_path = tokenizer_path or model_path
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tok_path, trust_remote_code=True, padding_side="left",
        )
    except (OSError, ImportError):
        tokenizer = AutoTokenizer.from_pretrained(
            tok_path, trust_remote_code=False, padding_side="left",
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True,
            revision=revision,
        )
    except (OSError, ImportError):
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=False,
            revision=revision,
        )
    model.eval()
    model.to(device)
    return model, tokenizer


@torch.no_grad()
def collect_hidden_states(model, tokenizer, prompts, device, batch_size=8, max_length=512):
    """Collect last-token hidden states for all layers via single forward pass per batch.

    Left-padding (set on tokenizer in load_model) keeps each sequence's final
    real token at the right edge of the batch, so the last-token hidden state
    for every prompt is at position -1 regardless of individual lengths.

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

        last_pos = enc["input_ids"].shape[1] - 1
        for i in range(len(batch)):
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


def per_prompt_cosine(H):
    """Cosine of each prompt to the centroid of all prompts.

    Returns (n_prompts,) array: cosine(prompt_i, mean(H)) for each i.
    This is the per-prompt contribution to the global mean cosine and
    enables paired significance testing across forget/retain sets.
    """
    H = H.astype(np.float64)
    centroid = H.mean(axis=0, keepdims=True)
    centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-12)
    norms = np.linalg.norm(H, axis=1, keepdims=True)
    H_norm = H / (norms + 1e-12)
    return (H_norm @ centroid_norm.T).ravel()


def stable_rank(W):
    """Stable rank: ||W||_F^2 / ||W||_2^2."""
    W = W.astype(np.float64)
    frob_sq = np.sum(W ** 2)
    spectral = np.linalg.norm(W, ord=2)
    return float(frob_sq / (spectral ** 2 + 1e-12))


def spectral_metrics(W):
    """Stable rank + (sigma_max, sigma_min) deviation from rank ratio.

    Stable rank is scale-invariant — unlearning FT perturbs the leading
    singular direction much more than the rank ratio, so sigma_max captures
    the trace that stable rank misses.
    """
    W = W.astype(np.float64)
    frob_sq = np.sum(W ** 2)
    sv = np.linalg.svd(W, compute_uv=False)
    sigma_max = float(sv[0])
    sigma_min = float(sv[-1])
    sr = float(frob_sq / (sv[0] ** 2 + 1e-12))
    return sr, sigma_max, sigma_min


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


# ---------------------------------------------------------------------------
# MIA / MINT-style metrics (token-level likelihood)
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_token_scores(model, tokenizer, prompts, device, batch_size=8, max_length=512):
    """Compute per-prompt token-level scores for MIA metrics.

    For each prompt, returns:
      - loss: negative cross-entropy (higher = model predicts this better)
      - min_k_plus: mean standardized log-prob of lowest 20% tokens
      - logrank: mean log-rank of true tokens across positions
      - entropy: mean next-token entropy

    Returns dict of (n_prompts,) arrays.
    """
    all_loss = []
    all_min_k = []
    all_logrank = []
    all_entropy = []

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100

        outputs = model(**enc, labels=labels)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].float()
        shift_labels = enc["input_ids"][:, 1:]
        shift_mask = enc["attention_mask"][:, 1:]

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

        for i in range(len(batch)):
            mask = shift_mask[i].bool()
            tlps = token_log_probs[i][mask].cpu().numpy()
            if len(tlps) == 0:
                all_loss.append(0.0)
                all_min_k.append(0.0)
                all_logrank.append(0.0)
                all_entropy.append(0.0)
                continue

            all_loss.append(float(-tlps.mean()))

            probs_i = torch.nn.functional.softmax(shift_logits[i][mask], dim=-1)
            mu = (probs_i * log_probs[i][mask]).sum(dim=-1)
            sigma_sq = (probs_i * log_probs[i][mask] ** 2).sum(dim=-1) - mu ** 2
            sigma = sigma_sq.clamp(min=1e-12).sqrt()
            standardized = (token_log_probs[i][mask].cpu() - mu.cpu()) / sigma.cpu()
            k = max(1, int(len(standardized) * 0.2))
            topk = torch.sort(standardized)[0][:k]
            all_min_k.append(float(topk.mean()))

            ranks = (shift_logits[i][mask].argsort(dim=-1, descending=True) == shift_labels[i][mask].unsqueeze(1)).float().argmax(dim=-1)
            all_logrank.append(float(-torch.log(ranks.float() + 1).mean()))

            ent = -(probs_i * log_probs[i][mask]).sum(dim=-1)
            all_entropy.append(float(ent.mean()))

    return {
        "loss": np.array(all_loss),
        "min_k_plus": np.array(all_min_k),
        "logrank": np.array(all_logrank),
        "entropy": np.array(all_entropy),
    }


@torch.no_grad()
def generate_samples(model, tokenizer, prompts, device, max_new_tokens=128, temperature=0.7, top_p=0.9):
    """Generate text completions for prompts.

    Returns list of dicts with 'prompt' and 'completion' keys.
    """
    samples = []
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, do_sample=True, pad_token_id=tokenizer.pad_token_id,
        )
        generated = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        samples.append({
            "prompt": prompt,
            "completion": generated,
        })
    return samples
