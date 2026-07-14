#!/usr/bin/env python3
"""Materialize the predeclared seed-42/123 full-data configs after dev selection."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from common import read_json


def config(learning_rate: float, seed: int, output_dir: str) -> dict:
    return {
        "model_path": "gemma3-270m-it",
        "train_file": "train.jsonl",
        "output_dir": output_dir,
        "init_adapter_path": "adapter_curriculum_stage1",
        "seed": seed,
        "dev_size": 0,
        "max_train_samples": None,
        "drop_uninformative_prompts": True,
        "simplify_targets": False,
        "require_no_truncation": True,
        "max_length": 3584,
        "eval_max_length": 3584,
        "precision": "auto",
        "auto_precision_fallback": True,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "gradient_checkpointing": True,
        "loss_chunk_size": 256,
        "num_train_epochs": 1,
        "learning_rate": learning_rate,
        "weight_decay": 0.01,
        "warmup_ratio": 0.05,
        "lr_scheduler_type": "cosine",
        "max_grad_norm": 1.0,
        "logging_steps": 5,
        "eval_steps": 25,
        "save_steps": 25,
        "early_stopping": False,
        "early_stopping_patience": 1,
        "lora_r": 4,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }


def write_yaml(path: str | Path, value: dict) -> None:
    Path(path).write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", default="artifacts/model_selection.json")
    args = parser.parse_args()
    learning_rate = float(read_json(args.selection)["selected_learning_rate"])
    e13 = config(learning_rate, 42, "runs/e13_full217_seed42")
    e14 = config(learning_rate, 123, "runs/e14_full217_seed123")
    write_yaml("configs/e13_full217_seed42.yaml", e13)
    write_yaml("configs/e14_full217_seed123.yaml", e14)
    write_yaml("train_config.yaml", e13)
    print(f"wrote E13/E14 configs with learning_rate={learning_rate:g}")


if __name__ == "__main__":
    main()
