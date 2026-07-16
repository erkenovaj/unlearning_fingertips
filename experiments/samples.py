"""Save model text completions for qualitative inspection.

Generates responses from each model on forget and retain prompts and saves
them to JSON for manual review. Useful for checking whether unlearned models
actually refuse to answer forget questions or just produce lower-quality
responses.

Usage:
  python samples.py --model original --output_dir results/samples/
  python samples.py --model rmu    --output_dir results/samples/
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    MODELS, DTYPE_MAP, get_device, load_tofu_prompts, format_prompts, load_model,
    generate_samples, save_json,
)


def main():
    p = argparse.ArgumentParser(description="Save model outputs for qualitative inspection.")
    p.add_argument("--model", default=None, help="Key from MODELS registry in common.py")
    p.add_argument("--model_path", default=None, help="Direct HF model path")
    p.add_argument("--model_tag", default=None, help="Tag for output files")
    p.add_argument("--output_dir", default="results/samples")
    p.add_argument("--num_samples", type=int, default=50)
    p.add_argument("--forget_fraction", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--device", default=None)
    p.add_argument("--raw_prompts", action="store_true")
    args = p.parse_args()

    if args.model_path:
        model_path = args.model_path
        tag = args.model_tag or args.model_path.split("/")[-1]
    elif args.model:
        model_path = MODELS[args.model]
        tag = args.model_tag or args.model
    else:
        p.error("Specify --model (registry key) or --model_path (HF path)")

    device = args.device or get_device()
    dtype = DTYPE_MAP[args.dtype]

    print(f"[samples] model={tag}  path={model_path}")
    print(f"[samples] device={device}  dtype={args.dtype}  samples={args.num_samples}")

    forget_qs, retain_qs = load_tofu_prompts(args.forget_fraction, args.num_samples, seed=args.seed)
    print(f"[samples] forget={len(forget_qs)}  retain={len(retain_qs)}  seed={args.seed}")

    model, tokenizer = load_model(model_path, dtype=dtype, device=device)

    forget_prompts = format_prompts(forget_qs, tokenizer, raw=args.raw_prompts)
    retain_prompts = format_prompts(retain_qs, tokenizer, raw=args.raw_prompts)

    print("[samples] generating forget completions...")
    forget_samples = generate_samples(
        model, tokenizer, forget_prompts, device,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
    )
    for i, s in enumerate(forget_samples):
        s["question"] = forget_qs[i]
        s["set"] = "forget"

    print("[samples] generating retain completions...")
    retain_samples = generate_samples(
        model, tokenizer, retain_prompts, device,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
    )
    for i, s in enumerate(retain_samples):
        s["question"] = retain_qs[i]
        s["set"] = "retain"

    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()

    results = {
        "model_tag": tag,
        "model_path": model_path,
        "num_samples": args.num_samples,
        "forget_fraction": args.forget_fraction,
        "seed": args.seed,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "forget_samples": forget_samples,
        "retain_samples": retain_samples,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, f"{tag}_samples.json")
    save_json(results, json_path)
    print(f"[samples] saved -> {json_path}")

    print(f"\n[samples] preview for {tag}:")
    print(f"  --- FORGET sample (Q: {forget_qs[0][:60]}...) ---")
    print(f"  {forget_samples[0]['completion'][:200]}")
    print(f"  --- RETAIN sample (Q: {retain_qs[0][:60]}...) ---")
    print(f"  {retain_samples[0]['completion'][:200]}")


if __name__ == "__main__":
    main()
