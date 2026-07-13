#!/usr/bin/env python3
"""Deterministic base-versus-LoRA evaluation on valid.jsonl."""

from __future__ import annotations

import argparse
import platform
import re
import sys
import time
from pathlib import Path

from common import device_name, set_seed, write_json
from data import load_jsonl
from reward import reward


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma3-270m-it")
    parser.add_argument("--adapter", default="adapter")
    parser.add_argument("--valid", default="valid.jsonl")
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def clean_svg(text: str) -> str:
    match = re.search(r"<svg\b.*?</svg\s*>", text, re.I | re.S)
    return match.group(0) if match else text.strip()


def get_eos_token_ids(tokenizer) -> list[int]:
    ids = [tokenizer.eos_token_id]
    end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if isinstance(end_of_turn_id, int) and end_of_turn_id >= 0 and end_of_turn_id not in ids:
        ids.append(end_of_turn_id)
    return ids


def generate(model, tokenizer, rows, device, max_new_tokens):
    import torch

    outputs = []
    model.eval()
    eos_token_ids = get_eos_token_ids(tokenizer)
    for index, row in enumerate(rows):
        prompt_messages = row["messages"][:-1]
        input_ids = tokenizer.apply_chat_template(
            prompt_messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_ids,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        continuation = generated[0, input_ids.shape[1] :]
        raw_text = tokenizer.decode(continuation, skip_special_tokens=True).strip()
        outputs.append({"raw_text": raw_text, "svg": clean_svg(raw_text)})
        if device == "mps":
            torch.mps.empty_cache()
        print(f"generated {index + 1}/{len(rows)}", file=sys.stderr, flush=True)
    return outputs


def summarize(samples, key):
    import statistics

    summary = {}
    for metric in ("total", "validity", "fidelity"):
        values = [sample[key]["reward"][metric] for sample in samples]
        summary[metric] = {
            "mean": sum(values) / len(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
    summary["fatal_rate"] = sum(sample[key]["reward"]["metadata"]["fatal"] for sample in samples) / len(samples)
    return summary


def main() -> None:
    args = parse_args()
    import peft
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_seed(args.seed)
    device = device_name()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    rows = load_jsonl(args.valid)
    if args.limit:
        rows = rows[: args.limit]
    dtype = torch.bfloat16 if device in {"mps", "cuda"} else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    started = time.time()
    base_outputs = generate(base_model, tokenizer, rows, device, args.max_new_tokens)
    adapter_path = Path(args.adapter)
    if not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(f"adapter not found: {adapter_path}")
    tuned_model = PeftModel.from_pretrained(base_model, adapter_path).to(device)
    tuned_outputs = generate(tuned_model, tokenizer, rows, device, args.max_new_tokens)

    samples = []
    for index, (row, base_output, tuned_output) in enumerate(zip(rows, base_outputs, tuned_outputs)):
        prompt = row["messages"][1]["content"]
        samples.append({
            "id": index,
            "prompt": prompt,
            "reference_svg": row["messages"][2]["content"],
            "base": {**base_output, "reward": reward(prompt, base_output["raw_text"])},
            "tuned": {**tuned_output, "reward": reward(prompt, tuned_output["raw_text"])},
        })
    result = {
        "schema_version": 1,
        "environment": {
            "device": device,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
        "model": {"base_path": args.model, "adapter_path": args.adapter},
        "decoding": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "eos_token_ids": get_eos_token_ids(tokenizer),
        },
        "counts": {"validation_samples": len(samples)},
        "summary": {"base": summarize(samples, "base"), "tuned": summarize(samples, "tuned")},
        "samples": samples,
        "elapsed_seconds": time.time() - started,
    }
    write_json(args.output, result)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
