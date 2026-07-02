import argparse
import io
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HAS_GPU = torch.cuda.is_available()

MODEL_TO_HF = {
    "Zephyr-7b": "HuggingFaceH4/zephyr-7b-beta",
    "Yi-34B-Chat": "01-ai/Yi-34B-Chat",
    "Llama3.1-8b": "meta-llama/Meta-Llama-3.1-8B",
    "Qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "Qwen2.5-14b": "Qwen/Qwen2.5-14B",
}

# Local-directory convention used by generate_response.py for unlearned checkpoints.
UNLEARN_PATH = {
    "Zephyr-7b": "./zephyr_{method}_model",
    "Yi-34B-Chat": "./yi-{method}-model",
    "Llama3.1-8b": "./llama8b-{method}-model",
    "Qwen2.5-7b": "./qwen7b-{method}-model",
    "Qwen2.5-14b": "./qwen7b-{method}-model",
}

INSTRUCT_MODELS = {"Yi-34B-Chat"}

# Published unlearned checkpoints on HF Hub (skips RMU/NPO training).
# Keys: (model, method) -> HF repo. Add more as they become available.
PRETRAINED_UNLEARN = {
    ("Zephyr-7b", "rmu"): "cais/Zephyr_RMU",
}

# T4 (Turing) has no native bf16; fall back to fp16 to avoid slow emulation / dtype errors.
DEFAULT_DTYPE = torch.bfloat16 if (HAS_GPU and torch.cuda.is_bf16_supported()) else torch.float16


def _quant_config(load_4bit):
    if not load_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=DEFAULT_DTYPE,
        bnb_4bit_use_double_quant=True,
    )


def _model_load_kwargs(load_4bit):
    """Kwargs for AutoModelForCausalLM.from_pretrained that are safe with bitsandbytes.

    With bnb we must NOT pass torch_dtype (it triggers a model.to(dtype) cast that
    bnb rejects) and must use device_map="auto" (HF's dispatcher special-cases bnb
    modules and skips .to() on them). A bare "cuda" or a dict like {"": 0} makes the
    dispatcher call model.to(device) post-load to enforce the map, which bnb rejects.
    """
    if load_4bit:
        return dict(quantization_config=_quant_config(True),
                    device_map="auto", low_cpu_mem_usage=True, trust_remote_code=True)
    return dict(torch_dtype=DEFAULT_DTYPE, device_map="auto", trust_remote_code=True)


def _model_device(model):
    """Device of a (possibly dispatched/quantized) model's first parameter."""
    return next(model.parameters()).device


def _patch_accelerate_for_bnb():
    """Make dispatch_model place bitsandbytes-quantized models without calling model.to().

    Older accelerate uses `model.to(device)` for single-GPU device maps, which bnb
    4-bit models reject (weights are already placed during quantized loading). But we
    can't just skip dispatch: the NON-quantized modules (embeddings, norms, lm_head)
    are still on CPU and must be moved, or forward hits a device mismatch. So for
    quantized models we move only the non-bnb params/buffers to the target GPU and
    leave the 4-bit weights (which reject .to()) alone. transformers imports
    dispatch_model into its own namespace, so we patch it there too.
    """
    def _is_quantized(model):
        cfg = getattr(model, "config", None)
        return bool(
            getattr(model, "is_quantized", False)
            or getattr(model, "is_loaded_in_4bit", False)
            or getattr(model, "is_loaded_in_8bit", False)
            or getattr(model, "hf_quantizer", None) is not None
            or getattr(cfg, "quantization_config", None) is not None
        )

    def _target_device(device_map):
        if isinstance(device_map, dict):
            for d in device_map.values():
                if d not in ("cpu", "disk", None):
                    return torch.device(f"cuda:{d}" if isinstance(d, int) else d)
        return torch.device("cuda:0")

    def _place_non_bnb(model, target):
        bnb_types = ("Params4bit", "Int8Params")
        for module in model.modules():
            for name, p in list(module._parameters.items()):
                if p is None or p.__class__.__name__ in bnb_types:
                    continue
                if p.device.type != "cuda":
                    module._parameters[name] = torch.nn.Parameter(
                        p.data.to(target), requires_grad=p.requires_grad)
            for name, b in list(module._buffers.items()):
                if b is not None and b.device.type != "cuda":
                    module._buffers[name] = b.to(target)

    def _make_safe(original):
        def _safe_dispatch(model, device_map=None, **kwargs):
            if _is_quantized(model):
                target = _target_device(device_map)
                _place_non_bnb(model, target)
                if device_map is not None:
                    model.hf_device_map = device_map
                return model
            return original(model, device_map=device_map, **kwargs)
        return _safe_dispatch

    for mod_name in ("transformers.modeling_utils", "accelerate.big_modeling", "accelerate"):
        try:
            import importlib
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        fn = getattr(mod, "dispatch_model", None)
        if fn is None or getattr(fn, "_bnb_patched", False):
            continue
        safe = _make_safe(fn)
        safe._bnb_patched = True
        mod.dispatch_model = safe


_patch_accelerate_for_bnb()


def resolve_paths(model, method):
    """Return (original_model_path, unlearned_model_path, is_instruct)."""
    if model not in MODEL_TO_HF:
        raise ValueError(f"Unknown --model {model!r}. Known: {list(MODEL_TO_HF)}")
    orig = MODEL_TO_HF[model]
    if method == "none":
        return orig, None, model in INSTRUCT_MODELS
    unlearn = UNLEARN_PATH[model].format(method=method.lower())
    return orig, unlearn, model in INSTRUCT_MODELS


# --------------------------------------------------------------------------- #
# Prompt construction (mirrors generate_response.py / generate_wmdp_response.py)
# --------------------------------------------------------------------------- #
def build_prompts(dataset, num_samples, wmdp_json_path=None, wmdp_subset=None, seed=42):
    rng = random.Random(seed)
    from datasets import load_dataset
    if dataset == "WMDP":
        # cais/wmdp has configs wmdp-bio / wmdp-cyber / wmdp-chem with
        # {question, choices, answer}. A local *_questions.json (see docs/Data.md)
        # can override via wmdp_json_path for parity with generate_wmdp_response.py.
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
# Response generation (HuggingFace transformers; same format as the repo)
# --------------------------------------------------------------------------- #
def load_causal_lm(model_name, dtype=None, load_4bit=False):
    if not HAS_GPU:
        raise RuntimeError("Generation requires a CUDA GPU for 7B+ models.")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, **_model_load_kwargs(load_4bit))
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_responses(model, tokenizer, prompts, instruct, temperature=0.0,
                       max_new_tokens=256, batch_size=16, repetition_penalty=1.1):
    """Generate dialogs in the classification.py JSON format (last turn = assistant)."""
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
        device_map="cuda", torch_dtype=torch.bfloat16,
        pooling_mode="mean", max_length=512,
    )
    embeddings = l2v.encode(texts, batch_size=batch_size)
    return np.asarray(embeddings, dtype=np.float32)


@torch.no_grad()
def get_pre_logit_activations(model, tokenizer, prompt, max_new_tokens=50):
    """Mean-pooled pre-logit (lm_head input) activation for the generated continuation."""
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


def build_activation_features(model_name, prompts, max_new_tokens=50, load_4bit=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, **_model_load_kwargs(load_4bit))
    model.eval()
    feats = [get_pre_logit_activations(model, tokenizer, p, max_new_tokens) for p in prompts]
    del model
    torch.cuda.empty_cache()
    return torch.stack(feats, dim=0).numpy().astype(np.float32)


# --------------------------------------------------------------------------- #
# MLP probe + training (mini-batch, train/val/test split)
# --------------------------------------------------------------------------- #
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _make_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_mlp(X, y, epochs=50, lr=1e-3, batch_size=64, weight_decay=1e-4, device=DEVICE):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
    train_loader = _make_loader(X_tr, y_tr, batch_size, shuffle=True)
    val_loader = _make_loader(X_val, y_val, batch_size, shuffle=False)
    test_loader = _make_loader(X_te, y_te, batch_size, shuffle=False)

    model = BinaryClassifier(X.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def evaluate(loader):
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for xb, yb in loader:
                logits = model(xb.to(device))
                preds.append((torch.sigmoid(logits) > 0.5).long().cpu())
                gts.append(yb.long())
        return accuracy_score(torch.cat(gts), torch.cat(preds))

    best_val, best_state = -1.0, None
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb.to(device)), yb.to(device))
            loss.backward()
            optimizer.step()
        val_acc = evaluate(val_loader)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    test_acc = evaluate(test_loader)
    print(f"  best val acc: {best_val:.4f} | test acc: {test_acc:.4f}")
    return model, test_acc


# --------------------------------------------------------------------------- #
# Optional unlearning (RMU / NPO) — reference implementations from the notebook
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
            f"--forget_corpus_dir {forget_corpus_dir!r} not found. Place WMDP .jsonl files there "
            "(see docs/Data.md) or use --unlearn none with an existing checkpoint."
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
                alpha=1200.0, lr=5e-5, max_batches=80, batch_size=4, max_len=512, output_dir=None):
    if not HAS_GPU:
        raise RuntimeError("rmu_unlearn requires a CUDA GPU.")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    updated = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=DEFAULT_DTYPE,
                                                   device_map="auto", trust_remote_code=True)
    frozen = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=DEFAULT_DTYPE,
                                                  device_map="auto", trust_remote_code=True)
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
    u = torch.rand(hidden_size, device=_model_device(updated), dtype=DEFAULT_DTYPE)
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
                max_batches=80, batch_size=2, max_len=512, output_dir=None):
    if not HAS_GPU:
        raise RuntimeError("npo_unlearn requires a CUDA GPU.")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=DEFAULT_DTYPE,
                                                  device_map="auto", trust_remote_code=True)
    reference = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=DEFAULT_DTYPE,
                                                     device_map="auto", trust_remote_code=True)
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
    if args.pretrained:
        key = (args.model, args.unlearn)
        if args.unlearn == "none":
            raise ValueError("--pretrained requires --unlearn rmu|npo (not none).")
        if key not in PRETRAINED_UNLEARN:
            raise ValueError(
                f"No published checkpoint for {args.model} + {args.unlearn}. "
                f"Available: {list(PRETRAINED_UNLEARN)}. Drop --pretrained to train locally."
            )
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
    if args.unlearn == "rmu":
        return rmu_unlearn(orig_model_path, forget, retain,
                           layer_id=args.rmu_layer, steering_coeff=args.rmu_coeff,
                           alpha=args.rmu_alpha, lr=args.unlearn_lr,
                           max_batches=args.unlearn_max_batches, batch_size=args.unlearn_batch_size,
                           output_dir=out)
    if args.unlearn == "npo":
        return npo_unlearn(orig_model_path, forget, retain,
                           beta=args.npo_beta, gamma=args.npo_gamma, lr=args.unlearn_lr,
                           max_batches=args.unlearn_max_batches, batch_size=args.unlearn_batch_size,
                           output_dir=out)
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

    for label, model_path, resp_path in paths:
        if os.path.exists(resp_path) and not args.force and not args.regenerate:
            print(f"[generate] {label}: {resp_path} exists, reuse (use --regenerate to overwrite).")
            continue
        print(f"[generate] {label}: {model_path} -> {resp_path}")
        model, tokenizer = load_causal_lm(model_path, load_4bit=args.load_in_4bit)
        bs = args.gen_batch_size if not args.load_in_4bit else min(args.gen_batch_size, 4)
        dialogs = generate_responses(
            model, tokenizer, prompts, instruct=args.instruct,
            temperature=args.temperature, max_new_tokens=args.max_new_tokens,
            batch_size=bs,
        )
        save_responses(dialogs, resp_path)
        del model
        torch.cuda.empty_cache()
    return orig_path, unlearn_path if unlearn_model_path else None


def stage_classify(args, orig_resp, unlearn_resp, orig_model_path, unlearn_model_path, prompts):
    print(f"[detect] feature={args.feature} | probe=MLP")
    if args.feature == "text":
        if not unlearn_resp:
            raise ValueError("Text path needs both --orig_response and --unlearn_response.")
        texts, labels = load_responses([orig_resp, unlearn_resp], max_per_label=args.max_per_label)
        X = build_text_features(texts, batch_size=args.encode_batch_size)
    elif args.feature == "activation":
        if not unlearn_model_path:
            raise ValueError("Activation path needs an unlearned model (use --unlearn or --unlearn_path).")
        print("  building original activations...")
        Xo = build_activation_features(orig_model_path, prompts,
                                       max_new_tokens=args.act_new_tokens, load_4bit=args.load_in_4bit)
        print("  building unlearned activations...")
        Xu = build_activation_features(unlearn_model_path, prompts,
                                       max_new_tokens=args.act_new_tokens, load_4bit=args.load_in_4bit)
        X = np.concatenate([Xo, Xu], axis=0)
        labels = np.array([0] * len(Xo) + [1] * len(Xu))
    else:
        raise ValueError(f"Unknown --feature {args.feature!r}")
    _, test_acc = train_mlp(X, labels, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
    return test_acc


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="End-to-end unlearning-trace detection runner.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", default="Zephyr-7b", choices=list(MODEL_TO_HF))
    p.add_argument("--unlearn", default="none", choices=["none", "rmu", "npo"])
    p.add_argument("--unlearn_path", default=None, help="Path to unlearned checkpoint (skip training).")
    p.add_argument("--dataset", default="MMLU", choices=["WMDP", "MMLU", "UltraChat"])
    p.add_argument("--wmdp_json_path", default=None, help="Optional local WMDP MCQ JSON (overrides HF).")
    p.add_argument("--wmdp_subset", default="cyber", choices=["bio", "cyber", "chem"],
                   help="WMDP subset loaded from cais/wmdp when no local JSON is given.")
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--feature", default="text", choices=["text", "activation"])
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
    p.add_argument("--load_in_4bit", action="store_true",
                   help="Load 7B models in 4-bit (bitsandbytes) so they fit on a 16GB T4.")
    p.add_argument("--max_per_label", type=int, default=None)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--results_file", default="./results.json")
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


def main():
    args = parse_args()
    if not HAS_GPU:
        raise SystemExit("No CUDA GPU detected. All heavy stages require CUDA.")

    if HAS_GPU:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[gpu] {torch.cuda.get_device_name(0)} | VRAM {vram_gb:.1f} GB")
        if vram_gb < 20 and not args.load_in_4bit and args.model in ("Zephyr-7b", "Llama3.1-8b", "Qwen2.5-7b"):
            print(f"[gpu] WARNING: {vram_gb:.1f}GB is tight for a 7B in bf16. "
                  f"Add --load_in_4bit (needs `pip install bitsandbytes`).")

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
        "test_accuracy": float(test_acc),
    }
    os.makedirs(os.path.dirname(args.results_file) or ".", exist_ok=True)
    with open(args.results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] test accuracy: {test_acc:.4f}  (saved -> {args.results_file})")


if __name__ == "__main__":
    main()
