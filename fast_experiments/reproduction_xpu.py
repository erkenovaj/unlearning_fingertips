#!/usr/bin/env python3
"""XPU-adapted version of reproduction.py for Intel GPUs (Data Center GPU / Arc).

Usage (same CLI shape as reproduction.py, minus --load_in_4bit):
    python reproduction_xpu.py \
        --model Zephyr-7b --unlearn rmu --pretrained \
        --dataset MMLU --num_samples 50 \
        --feature activation --act_new_tokens 32

Requires:
  pip install torch intel-extension-for-pytorch transformers datasets scikit-learn
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

# cais/Zephyr_RMU has a corrupted tokenizer.model on the Hub; use the base model's.
_TOKENIZER_OVERRIDE = {"cais/Zephyr_RMU": "HuggingFaceH4/zephyr-7b-beta"}

MODEL_TO_HF = {
    "TinyLlama-1.1B": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B",
    "Qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "Zephyr-7b": "HuggingFaceH4/zephyr-7b-beta",
    "Llama3.1-8b": "meta-llama/Meta-Llama-3.1-8B",
    "Yi-34B-Chat": "01-ai/Yi-34B-Chat",
    "Qwen2.5-14b": "Qwen/Qwen2.5-14B",
}

UNLEARN_PATH = {
    "TinyLlama-1.1B": "./tinyllama_{method}_model",
    "Qwen2.5-1.5B": "./qwen1.5b-{method}-model",
    "Qwen2.5-7b": "./qwen7b-{method}-model",
    "Zephyr-7b": "./zephyr_{method}_model",
    "Yi-34B-Chat": "./yi-{method}-model",
    "Llama3.1-8b": "./llama8b-{method}-model",
    "Qwen2.5-14b": "./qwen7b-{method}-model",
}

INSTRUCT_MODELS = {"TinyLlama-1.1B", "Yi-34B-Chat"}

PRETRAINED_UNLEARN = {
    ("Zephyr-7b", "rmu"): "cais/Zephyr_RMU",
}

DTYPE_ALIASES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def _tok_name(model_name):
    return _TOKENIZER_OVERRIDE.get(model_name, model_name)


def _model_device(model):
    return next(model.parameters()).device


def resolve_paths(model, method):
    if model not in MODEL_TO_HF:
        raise ValueError(f"Unknown --model {model!r}. Known: {list(MODEL_TO_HF)}")
    orig = MODEL_TO_HF[model]
    if method == "none":
        return orig, None, model in INSTRUCT_MODELS
    unlearn = UNLEARN_PATH[model].format(method=method.lower())
    return orig, unlearn, model in INSTRUCT_MODELS


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #
def build_prompts(dataset, num_samples, wmdp_json_path=None, wmdp_subset=None, seed=42):
    rng = random.Random(seed)
    from datasets import load_dataset
    if dataset == "WMDP":
        if wmdp_json_path and os.path.exists(wmdp_json_path):
            with open(wmdp_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            cfg = f"wmdp-{wmdp_subset}" if wmdp_subset else "wmdp-cyber"
            data = load_dataset("cais/wmdp", cfg)["test"]
        idx = rng.sample(range(len(data)), min(num_samples, len(data)))
        return [f"Question: {data[i]['question']}\nAnswer:" for i in idx]
    if dataset == "MMLU":
        ds = load_dataset("cais/mmlu", name="all")["test"]
        idx = rng.sample(range(len(ds)), min(num_samples, len(ds)))
        return [
            f"{ds[i]['question'].strip()}\n{ds[i]['choices']}\n\n"
            f"Please provide your analysis, then give the final answer.\n\nAnalysis:"
            for i in idx
        ]
    if dataset == "UltraChat":
        ds = load_dataset("HuggingFaceH4/ultrachat_200k")["train_sft"]
        idx = rng.sample(range(len(ds)), min(num_samples, len(ds)))
        return [ds[i]["prompt"] for i in idx if isinstance(ds[i]["prompt"], str) and ds[i]["prompt"]]
    raise ValueError(f"Unknown --dataset {dataset!r}")


# --------------------------------------------------------------------------- #
# Response generation — XPU: no device_map, no bitsandbytes
# --------------------------------------------------------------------------- #
def load_causal_lm(model_name, dtype=torch.bfloat16):
    if DEVICE == "cpu":
        raise RuntimeError("Generation requires a GPU.")
    tokenizer = AutoTokenizer.from_pretrained(_tok_name(model_name), trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True,
    )
    model = model.to(DEVICE)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_responses(model, tokenizer, prompts, instruct, temperature=0.0,
                       max_new_tokens=256, batch_size=16, repetition_penalty=1.1):
    device = _model_device(model)
    dialogs = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        if instruct and tokenizer.chat_template:
            texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
                )
                for p in batch
            ]
        else:
            texts = []
            for p in batch:
                toks = tokenizer.encode(p, padding=False)[:128]
                texts.append(tokenizer.decode(toks, skip_special_tokens=True))
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                        max_length=1024).to(device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            repetition_penalty=repetition_penalty if not instruct else 1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = gen[:, enc["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for prompt, response in zip(batch, decoded):
            dialogs.append([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ])
    return dialogs


def save_responses(dialogs, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with io.open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(dialogs, ensure_ascii=False, indent=2))
    print(f"  saved {len(dialogs)} responses -> {output_path}")


def load_responses(response_paths, max_per_label=None):
    texts, labels = [], []
    for label, path in enumerate(response_paths):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if max_per_label is not None:
            data = data[:max_per_label]
        for dialog in data:
            texts.append(dialog[-1]["content"])
            labels.append(label)
    print(f"  loaded {len(texts)} responses across {len(response_paths)} labels")
    return texts, labels


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def build_text_features(texts, batch_size=16):
    try:
        from llm2vec import LLM2Vec
    except ImportError as e:
        raise RuntimeError("Text features need llm2vec (pip install llm2vec==0.2.3).") from e
    l2v = LLM2Vec.from_pretrained(
        "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
        device_map=DEVICE, torch_dtype=torch.bfloat16,
        pooling_mode="mean", max_length=512,
    )
    embeddings = l2v.encode(texts, batch_size=batch_size)
    return np.asarray(embeddings, dtype=np.float32)


@torch.no_grad()
def get_pre_logit_activations(model, tokenizer, prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(_model_device(model))
    activations = []

    def hook_fn(module, inp, out):
        hidden = inp[0]
        activations.append(hidden[:, -1, :].detach().float().cpu())

    handle = model.lm_head.register_forward_hook(hook_fn)
    try:
        model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    finally:
        handle.remove()
    seq = torch.cat(activations, dim=0)
    return seq.mean(dim=0)


def build_activation_features(model_name, prompts, max_new_tokens=50, dtype=None):
    dtype = dtype or DTYPE_ALIASES.get("bf16")
    tokenizer = AutoTokenizer.from_pretrained(_tok_name(model_name), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True,
    )
    model = model.to(DEVICE)
    model.eval()
    feats = [get_pre_logit_activations(model, tokenizer, p, max_new_tokens) for p in prompts]
    del model
    if DEVICE == "xpu":
        torch.xpu.empty_cache()
    elif DEVICE == "cuda":
        torch.cuda.empty_cache()
    return torch.stack(feats, dim=0).numpy().astype(np.float32)


def normalize_features(X):
    mu, std = X.mean(axis=0, keepdims=True), X.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    return (X - mu) / std, (mu, std)


# --------------------------------------------------------------------------- #
# MLP probe
# --------------------------------------------------------------------------- #
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
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
    ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32))
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
    print(f"  [plot] training curves -> {path}")


def train_mlp(X, y, epochs=200, lr=3e-4, batch_size=32, weight_decay=1e-4, device=DEVICE,
              log_dir=None):
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
        prefix = f"mlp_{len(X)}samples_{epochs}epochs"
        json_path = os.path.join(log_dir, f"{prefix}_metrics.json")
        with io.open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"  [log] training metrics -> {json_path}")
        _save_training_plot(metrics, os.path.join(log_dir, f"{prefix}_curves.png"))

    return test_acc, tr_idx, val_idx, te_idx


# --------------------------------------------------------------------------- #
# Optional unlearning (RMU / NPO)
# --------------------------------------------------------------------------- #
def _get_decoder_layers(model):
    base = getattr(model, "model", model)
    return base.layers


def _hidden_at_layer(model, input_ids, attention_mask, layer_id):
    out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    return out.hidden_states[layer_id + 1]


def _load_forget_texts(forget_corpus_dir, max_items):
    texts = []
    if not forget_corpus_dir or not os.path.isdir(forget_corpus_dir):
        raise FileNotFoundError(
            f"--forget_corpus_dir {forget_corpus_dir!r} not found. "
            "Place WMDP .jsonl files there or use --unlearn none with an existing checkpoint."
        )
    for fn in sorted(os.listdir(forget_corpus_dir)):
        if not fn.endswith(".jsonl"):
            continue
        with open(os.path.join(forget_corpus_dir, fn), "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = obj.get("text") or obj.get("content") or ""
                if t:
                    texts.append(t)
                if len(texts) >= max_items:
                    return texts
    return texts


def _load_retain_texts(max_items):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    texts = []
    for ex in ds:
        t = ex["text"].strip()
        if len(t) > 50:
            texts.append(t)
        if len(texts) >= max_items:
            break
    return texts


def rmu_unlearn(model_name, forget_texts, retain_texts, layer_id=7, steering_coeff=20.0,
                alpha=1200.0, lr=5e-5, max_batches=80, batch_size=4, max_len=512, output_dir=None,
                dtype=torch.bfloat16):
    if DEVICE == "cpu":
        raise RuntimeError("rmu_unlearn requires a GPU.")
    tokenizer = AutoTokenizer.from_pretrained(_tok_name(model_name), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    updated = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True,
    ).to(DEVICE)
    frozen = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True,
    ).to(DEVICE)
    frozen.eval()
    for p in frozen.parameters():
        p.requires_grad_(False)

    trainable_ids = {layer_id - 2, layer_id - 1, layer_id}
    layers = _get_decoder_layers(updated)
    params = []
    for p in updated.parameters():
        p.requires_grad_(False)
    for i in trainable_ids:
        for p in layers[i].parameters():
            p.requires_grad_(True)
            params.append(p)
    optimizer = torch.optim.AdamW(params, lr=lr)

    hidden_size = updated.config.hidden_size
    u = torch.rand(hidden_size, device=_model_device(updated), dtype=dtype)
    control = steering_coeff * (u / u.norm())

    def batches(texts_):
        for s in range(0, len(texts_), batch_size):
            enc = tokenizer(texts_[s:s + batch_size], return_tensors="pt", padding=True,
                            truncation=True, max_length=max_len)
            yield {k: v.to(_model_device(updated)) for k, v in enc.items()}

    updated.train()
    f_iter, r_iter = batches(forget_texts), batches(retain_texts)
    for step in range(max_batches):
        try:
            f_batch = next(f_iter)
            r_batch = next(r_iter)
        except StopIteration:
            break
        optimizer.zero_grad()
        h_f = _hidden_at_layer(updated, f_batch["input_ids"], f_batch["attention_mask"], layer_id)
        forget_loss = torch.nn.functional.mse_loss(h_f, control.expand_as(h_f).to(h_f.dtype))
        h_r = _hidden_at_layer(updated, r_batch["input_ids"], r_batch["attention_mask"], layer_id)
        with torch.no_grad():
            h_r_frozen = _hidden_at_layer(frozen, r_batch["input_ids"], r_batch["attention_mask"], layer_id)
        retain_loss = torch.nn.functional.mse_loss(h_r, h_r_frozen)
        loss = forget_loss + alpha * retain_loss
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            print(f"  [RMU] step {step} | loss {loss.item():.3f} | "
                  f"forget {forget_loss.item():.3f} | retain {retain_loss.item():.4f}")

    if output_dir:
        updated.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"  saved RMU-unlearned model -> {output_dir}")
    return output_dir


def _seq_logprob(model, input_ids, attention_mask):
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    logp = torch.log_softmax(logits.float(), dim=-1)
    token_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    mask = attention_mask[:, 1:].float()
    return (token_logp * mask).sum(dim=-1)


def npo_unlearn(model_name, forget_texts, retain_texts, beta=0.1, gamma=1.0, lr=1e-5,
                max_batches=80, batch_size=2, max_len=512, output_dir=None,
                dtype=torch.bfloat16):
    if DEVICE == "cpu":
        raise RuntimeError("npo_unlearn requires a GPU.")
    tokenizer = AutoTokenizer.from_pretrained(_tok_name(model_name), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True,
    ).to(DEVICE)
    reference = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True,
    ).to(DEVICE)
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)

    def batches(texts_):
        for s in range(0, len(texts_), batch_size):
            enc = tokenizer(texts_[s:s + batch_size], return_tensors="pt", padding=True,
                            truncation=True, max_length=max_len)
            yield {k: v.to(_model_device(policy)) for k, v in enc.items()}

    policy.train()
    f_iter, r_iter = batches(forget_texts), batches(retain_texts)
    for step in range(max_batches):
        try:
            f_batch = next(f_iter)
            r_batch = next(r_iter)
        except StopIteration:
            break
        optimizer.zero_grad()
        lp_policy = _seq_logprob(policy, f_batch["input_ids"], f_batch["attention_mask"])
        with torch.no_grad():
            lp_ref = _seq_logprob(reference, f_batch["input_ids"], f_batch["attention_mask"])
        log_ratio = lp_policy - lp_ref
        forget_loss = (2.0 / beta) * torch.nn.functional.softplus(beta * log_ratio).mean()
        retain_out = policy(input_ids=r_batch["input_ids"], attention_mask=r_batch["attention_mask"],
                            labels=r_batch["input_ids"])
        loss = forget_loss + gamma * retain_out.loss
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            print(f"  [NPO] step {step} | loss {loss.item():.3f} | "
                  f"forget {forget_loss.item():.3f} | retain {retain_out.loss.item():.3f}")

    if output_dir:
        policy.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"  saved NPO-unlearned model -> {output_dir}")
    return output_dir


# --------------------------------------------------------------------------- #
# Pipeline stages
# --------------------------------------------------------------------------- #
def stage_unlearn(args, orig_model_path):
    dtype = DTYPE_ALIASES.get(args.dtype)
    if args.unlearn not in ("none",) and not args.pretrained:
        if DEVICE == "xpu":
            vram_gb = torch.xpu.get_device_properties(0).total_memory / 1e9
            model_gb = {"bf16": 2, "fp16": 1}.get(args.dtype, 2) * 7
            if vram_gb < model_gb * 2.5:
                print(f"  [warn] VRAM {vram_gb:.1f}GB may be too low for unlearning "
                      f"(needs ~{model_gb * 2.5:.0f}GB for two 7B models). Try --dtype fp16.")
        elif DEVICE == "cuda":
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            model_gb = {"bf16": 2, "fp16": 1}.get(args.dtype, 2) * 7
            if vram_gb < model_gb * 2.5:
                print(f"  [warn] VRAM {vram_gb:.1f}GB may be too low for unlearning "
                      f"(needs ~{model_gb * 2.5:.0f}GB for two 7B models). Try --dtype fp16.")
    if args.pretrained:
        key = (args.model, args.unlearn)
        if args.unlearn == "none":
            raise ValueError("--pretrained requires --unlearn rmu|npo (not none).")
        if key not in PRETRAINED_UNLEARN:
            raise ValueError(f"No published checkpoint for {args.model} + {args.unlearn}. "
                             f"Available: {list(PRETRAINED_UNLEARN)}.")
        repo = PRETRAINED_UNLEARN[key]
        print(f"[unlearn] using published checkpoint: {repo}")
        return repo
    if args.unlearn == "none":
        if not args.unlearn_path:
            raise ValueError("--unlearn none requires --unlearn_path <existing checkpoint dir> "
                             "(or use --pretrained for a published HF checkpoint).")
        if not (os.path.isdir(args.unlearn_path) or "/" in args.unlearn_path):
            raise FileNotFoundError(f"Unlearned checkpoint not found: {args.unlearn_path}")
        print(f"[unlearn] using existing checkpoint: {args.unlearn_path}")
        return args.unlearn_path
    out = args.unlearn_path or UNLEARN_PATH[args.model].format(method=args.unlearn.lower())
    if os.path.isdir(out) and not args.force:
        print(f"[unlearn] checkpoint exists, reuse ({out}); pass --force to retrain.")
        return out
    print(f"[unlearn] training {args.unlearn.upper()} on {orig_model_path} -> {out}")
    forget = _load_forget_texts(args.forget_corpus_dir, args.unlearn_max_batches * args.unlearn_batch_size)
    retain = _load_retain_texts(args.unlearn_max_batches * args.unlearn_batch_size)
    print(f"  forget texts: {len(forget)} | retain texts: {len(retain)}")
    dtype = DTYPE_ALIASES.get(args.dtype)
    if args.unlearn == "rmu":
        return rmu_unlearn(orig_model_path, forget, retain,
                           layer_id=args.rmu_layer, steering_coeff=args.rmu_coeff,
                           alpha=args.rmu_alpha, lr=args.unlearn_lr,
                           max_batches=args.unlearn_max_batches, batch_size=args.unlearn_batch_size,
                           output_dir=out, dtype=dtype)
    if args.unlearn == "npo":
        return npo_unlearn(orig_model_path, forget, retain,
                           beta=args.npo_beta, gamma=args.npo_gamma, lr=args.unlearn_lr,
                           max_batches=args.unlearn_max_batches, batch_size=args.unlearn_batch_size,
                           output_dir=out, dtype=dtype)
    raise ValueError(f"Unknown --unlearn {args.unlearn!r}")


def stage_generate(args, orig_model_path, unlearn_model_path, prompts):
    tag = args.model
    method_tag = f"-{args.unlearn}" if args.unlearn != "none" else "-unlearned"
    out_dir = os.path.join(args.responses_dir, args.dataset)
    orig_path = args.orig_response or os.path.join(out_dir, f"{tag}.json")
    unlearn_path = args.unlearn_response or os.path.join(out_dir, f"{tag}{method_tag}.json")
    paths = [("original", orig_model_path, orig_path)]
    if unlearn_model_path:
        paths.append(("unlearned", unlearn_model_path, unlearn_path))

    dtype = DTYPE_ALIASES.get(args.dtype)
    for label, model_path, resp_path in paths:
        if os.path.exists(resp_path) and not args.force and not args.regenerate:
            print(f"[generate] {label}: {resp_path} exists, reuse (use --regenerate to overwrite).")
            continue
        print(f"[generate] {label}: {model_path} -> {resp_path}")
        model, tokenizer = load_causal_lm(model_path, dtype=dtype)
        dialogs = generate_responses(
            model, tokenizer, prompts, instruct=args.instruct,
            temperature=args.temperature, max_new_tokens=args.max_new_tokens,
            batch_size=args.gen_batch_size,
        )
        save_responses(dialogs, resp_path)
        del model
        if DEVICE == "xpu":
            torch.xpu.empty_cache()
        elif DEVICE == "cuda":
            torch.cuda.empty_cache()
    return orig_path, unlearn_path if unlearn_model_path else None


def stage_classify(args, orig_resp, unlearn_resp, orig_model_path, unlearn_model_path, prompts):
    print(f"[detect] feature={args.feature} | probe=MLP | normalize={args.normalize} | mix_train={args.mix_train}")
    X_raw = None
    prompts_list = []
    tag = ""
    if args.feature == "text":
        if not unlearn_resp:
            raise ValueError("Text path needs both --orig_response and --unlearn_response.")
        texts, labels = load_responses([orig_resp, unlearn_resp], max_per_label=args.max_per_label)
        X_raw = build_text_features(texts, batch_size=args.encode_batch_size)
        prompts_list = texts
        tag = f"{args.dataset}_text"
    elif args.feature == "activation":
        if not unlearn_model_path:
            raise ValueError("Activation path needs an unlearned model "
                             "(use --unlearn or --unlearn_path).")
        retain_prompts = []
        if args.mix_train:
            print("  mixing MMLU retain-domain prompts into training...")
            tmp = build_prompts("MMLU", min(args.num_samples, 200))
            np.random.seed(42)
            retain_prompts = list(np.random.choice(tmp, min(len(tmp), 100), replace=False))

        dtype = DTYPE_ALIASES.get(args.dtype)
        print("  building original activations...")
        Xo_all = build_activation_features(orig_model_path, prompts + retain_prompts,
                                           max_new_tokens=args.act_new_tokens, dtype=dtype)
        print("  building unlearned activations...")
        Xu_all = build_activation_features(unlearn_model_path, prompts + retain_prompts,
                                           max_new_tokens=args.act_new_tokens, dtype=dtype)

        n = len(prompts)
        if retain_prompts:
            Xo, Xr_o = Xo_all[:n], Xo_all[n:]
            Xu, Xr_u = Xu_all[:n], Xu_all[n:]
            X_raw = np.concatenate([Xo, Xu, Xr_o, Xr_u], axis=0)
            labels = np.array([0] * n + [1] * n + [0] * len(Xr_o) + [0] * len(Xr_u))
            prompts_list = prompts + prompts + retain_prompts + retain_prompts
        else:
            X_raw = np.concatenate([Xo_all, Xu_all], axis=0)
            labels = np.array([0] * n + [1] * n)
            prompts_list = prompts + prompts
        tag = f"{args.dataset}_{args.model}_{args.unlearn}_activation"
    else:
        raise ValueError(f"Unknown --feature {args.feature!r}")

    norm_params = None
    if args.normalize and X_raw is not None:
        print(f"  normalizing features (z-score, {X_raw.shape[1]} dims)...")
        X, norm_params = normalize_features(X_raw)
    else:
        X = X_raw

    if args.log_dir and X_raw is not None:
        if norm_params is None:
            mu = np.zeros((1, X_raw.shape[1]), dtype=np.float32)
            std = np.ones((1, X_raw.shape[1]), dtype=np.float32)
            norm_params = (mu, std)
        _save_dataset(X_raw, labels, norm_params, prompts_list,
                      {"feature": args.feature, "dataset": args.dataset,
                       "model": args.model, "unlearn": args.unlearn,
                       "normalize": args.normalize, "n_samples": len(X_raw)},
                      args.log_dir, tag)

    test_acc, *splits = train_mlp(X, labels, epochs=args.epochs, lr=args.lr,
                                  batch_size=args.batch_size, log_dir=args.log_dir)
    if args.log_dir:
        _save_splits(*splits, args.log_dir, tag)
    return test_acc


# --------------------------------------------------------------------------- #
@torch.no_grad()
def print_samples_generations(model_pairs, prompts, n=3, dtype=torch.bfloat16, log_dir=None):
    """Generate sample text for each model and print/save to file."""
    all_samples = []
    for label, model_name in model_pairs:
        tokenizer = AutoTokenizer.from_pretrained(_tok_name(model_name), trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True,
        ).to(DEVICE)
        model.eval()
        device = _model_device(model)
        print(f"\n[sanity] sample generations from {label} ({model_name.split('/')[-1]}):")
        for i, p in enumerate(prompts[:n]):
            enc = tokenizer(p, return_tensors="pt", truncation=True, max_length=256).to(device)
            gen = model.generate(**enc, max_new_tokens=64, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
            out = tokenizer.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            print(f"  [prompt {i}] {p[:100]}...")
            print(f"  [gen]     {out[:200]}\n")
            all_samples.append({"model": label, "prompt_index": i, "prompt": p, "generation": out})
        del model
        if DEVICE == "xpu":
            torch.xpu.empty_cache()
        elif DEVICE == "cuda":
            torch.cuda.empty_cache()
    if log_dir and all_samples:
        path = os.path.join(log_dir, "sample_outputs.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(all_samples, f, indent=2)
        print(f"  [samples] -> {path}  ({len(all_samples)} generations)")


def parse_args():
    p = argparse.ArgumentParser(
        description="End-to-end unlearning-trace detection runner (XPU edition).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="Zephyr-7b", choices=list(MODEL_TO_HF))
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"],
                   help="Model dtype: bf16 (better quality) or fp16 (smaller memory).")
    p.add_argument("--unlearn", default="none", choices=["none", "rmu", "npo"])
    p.add_argument("--unlearn_path", default=None, help="Path to unlearned checkpoint (skip training).")
    p.add_argument("--dataset", default="WMDP", choices=["WMDP", "MMLU", "UltraChat"])
    p.add_argument("--wmdp_json_path", default=None, help="Optional local WMDP MCQ JSON (overrides HF).")
    p.add_argument("--wmdp_subset", default="cyber", choices=["bio", "cyber", "chem"],
                   help="WMDP subset loaded from cais/wmdp when no local JSON is given.")
    p.add_argument("--num_samples", type=int, default=400)
    p.add_argument("--feature", default="activation", choices=["text", "activation"])
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--gen_batch_size", type=int, default=16)
    p.add_argument("--encode_batch_size", type=int, default=16)
    p.add_argument("--act_new_tokens", type=int, default=50)
    p.add_argument("--responses_dir", default="./responses")
    p.add_argument("--orig_response", default=None, help="Reuse this original response JSON.")
    p.add_argument("--unlearn_response", default=None, help="Reuse this unlearned response JSON.")
    p.add_argument("--skip_unlearn", action="store_true")
    p.add_argument("--skip_generate", action="store_true")
    p.add_argument("--regenerate", action="store_true", help="Overwrite existing response JSONs.")
    p.add_argument("--force", action="store_true", help="Retrain unlearned checkpoint even if it exists.")
    p.add_argument("--pretrained", action="store_true",
                   help="Use a published HF unlearned checkpoint (see PRETRAINED_UNLEARN) instead of training.")
    p.add_argument("--max_per_label", type=int, default=None)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--no-normalize", action="store_false", dest="normalize", default=True,
                   help="Disable feature z-score normalization.")
    p.add_argument("--no-mix-train", action="store_false", dest="mix_train", default=True,
                   help="Disable MMLU retain mixing (probe sees only WMDP samples).")
    p.add_argument("--samples-sanity", action="store_true", dest="samples_sanity", default=False,
                   help="Print 3 sample generations from each model (loads full model, off by default).")
    p.add_argument("--results_file", default="./results.json")
    p.add_argument("--log_dir", default=None, help="Directory for a run log file (all terminal output teed to a timestamped file).")
    # unlearning hyperparameters
    p.add_argument("--forget_corpus_dir", default="./data", help="Dir with WMDP forget .jsonl files.")
    p.add_argument("--unlearn_lr", type=float, default=5e-5)
    p.add_argument("--unlearn_max_batches", type=int, default=80)
    p.add_argument("--unlearn_batch_size", type=int, default=4)
    p.add_argument("--rmu_layer", type=int, default=7)
    p.add_argument("--rmu_coeff", type=float, default=20.0)
    p.add_argument("--rmu_alpha", type=float, default=1200.0)
    p.add_argument("--npo_beta", type=float, default=0.1)
    p.add_argument("--npo_gamma", type=float, default=1.0)
    args = p.parse_args()
    args.instruct = args.model in INSTRUCT_MODELS
    return args


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
    args = parse_args()

    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(args.log_dir, f"run_{ts}.log")
        sys.stdout = _Tee(log_path)
        print(f"[log] tee -> {log_path}")

    if DEVICE == "xpu":
        vram_gb = torch.xpu.get_device_properties(0).total_memory / 1e9
        print(f"[device] XPU: {torch.xpu.get_device_name(0)} | VRAM {vram_gb:.1f} GB")
        if vram_gb < 20 and "7b" in args.model.lower():
            print(f"[device] WARNING: {vram_gb:.1f}GB is tight for {args.model} in {args.dtype}. "
                  f"Consider --model TinyLlama-1.1B or --dtype fp16.")
    elif DEVICE == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[device] CUDA: {torch.cuda.get_device_name(0)} | VRAM {vram_gb:.1f} GB")
        if vram_gb < 20 and "7b" in args.model.lower():
            print(f"[device] WARNING: {vram_gb:.1f}GB is tight for {args.model} in {args.dtype}. "
                  f"Consider --model TinyLlama-1.1B or --dtype fp16.")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    orig_model_path, unlearn_model_path, _ = resolve_paths(args.model, args.unlearn)
    print(f"[config] model={args.model} unlearn={args.unlearn} dataset={args.dataset} "
          f"feature={args.feature} num_samples={args.num_samples}")
    print(f"[config] original: {orig_model_path}")
    if unlearn_model_path:
        print(f"[config] unlearned: {unlearn_model_path}")

    if not args.skip_unlearn:
        unlearn_model_path = stage_unlearn(args, orig_model_path)
    elif unlearn_model_path is None and args.unlearn_path:
        unlearn_model_path = args.unlearn_path
    elif args.pretrained:
        unlearn_model_path = stage_unlearn(args, orig_model_path)

    prompts = build_prompts(args.dataset, args.num_samples,
                            wmdp_json_path=args.wmdp_json_path,
                            wmdp_subset=args.wmdp_subset)
    print(f"[prompts] built {len(prompts)} prompts from {args.dataset}")

    dtype = DTYPE_ALIASES.get(args.dtype)
    if args.samples_sanity and unlearn_model_path and not args.skip_generate:
        print("\n━━━ [sanity] sample generations ━━━")
        model_pairs = [("original", orig_model_path), ("unlearned", unlearn_model_path)]
        print_samples_generations(model_pairs, prompts, n=3, dtype=dtype, log_dir=args.log_dir)

    orig_resp, unlearn_resp = None, None
    if args.feature == "text" and not args.skip_generate:
        orig_resp, unlearn_resp = stage_generate(args, orig_model_path, unlearn_model_path, prompts)
    else:
        if args.orig_response and os.path.exists(args.orig_response):
            orig_resp = args.orig_response
        if args.unlearn_response and os.path.exists(args.unlearn_response):
            unlearn_resp = args.unlearn_response

    test_acc = stage_classify(args, orig_resp, unlearn_resp, orig_model_path,
                              unlearn_model_path, prompts)

    results = {
        "model": args.model, "unlearn": args.unlearn, "dataset": args.dataset,
        "feature": args.feature, "num_samples": args.num_samples,
        "normalize": args.normalize, "mix_train": args.mix_train,
        "test_accuracy": float(test_acc),
    }
    os.makedirs(os.path.dirname(args.results_file) or ".", exist_ok=True)
    with open(args.results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] test accuracy: {test_acc:.4f}  (saved -> {args.results_file})")

if __name__ == "__main__":
    main()
