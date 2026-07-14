#!/usr/bin/env python3
"""Run the reproducible short-screening experiment matrix."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

from common import read_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="experiment_matrix.yaml")
    parser.add_argument("--config", default="train_config.yaml")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    matrix = read_yaml(args.matrix)
    defaults = matrix["screening_defaults"]
    env = os.environ.copy()
    env.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.0")
    env.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.8")
    for experiment in matrix["experiments"]:
        name = experiment["name"]
        if args.only and name not in args.only:
            continue
        summary = Path("runs") / name / "training_summary.json"
        if summary.exists():
            print(f"skip completed {name}", flush=True)
            continue
        if experiment.get("config"):
            resolved = read_yaml(experiment["config"])
        else:
            resolved = read_yaml(args.config)
            resolved.update(defaults)
            resolved.update({
                key: value for key, value in experiment.items()
                if key not in {"name", "role", "config"}
            })
            resolved["num_train_epochs"] = resolved.pop("epochs", resolved["num_train_epochs"])
        resolved["output_dir"] = str(Path("runs") / name)
        experiment_dir = Path("runs") / name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        config_path = experiment_dir / "resolved_config.yaml"
        config_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
        command = [sys.executable, "student_kit/train_peft.py", "--config", str(config_path)]
        print("running", name, flush=True)
        subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
