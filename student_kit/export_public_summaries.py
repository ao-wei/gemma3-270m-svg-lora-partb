#!/usr/bin/env python3
"""Export selected run summaries while removing user-specific absolute paths."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import read_json, write_json


def sanitize(value, workspace: Path):
    if isinstance(value, dict):
        return {key: sanitize(item, workspace) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item, workspace) for item in value]
    if isinstance(value, str):
        prefix = str(workspace) + "/"
        return value.replace(prefix, "")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/experiment_summaries")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    workspace = Path.cwd().resolve()
    paths = {
        "e00_default": Path("runs/e00_default/training_summary.json"),
        "e01_rank4": Path("runs/e01_rank4/training_summary.json"),
        "e02_rank16": Path("runs/e02_rank16/training_summary.json"),
        "e03_lr1e4": Path("runs/e03_lr1e4/training_summary.json"),
        "e04_lr5e4": Path("runs/e04_lr5e4/training_summary.json"),
        "e05_length3072": Path("runs/e05_length3072/training_summary.json"),
        "e06_full_early_final": Path("runs/e06_full_early_final/training_summary.json"),
        "e07_seed123_final": Path("runs/e07_seed123_final/training_summary.json"),
        "e08_short_curriculum": Path("runs/e08_short_curriculum/training_summary.json"),
        "e09_simplified_targets": Path("runs/e09_simplified_targets/training_summary.json"),
        "e10_stage1": Path("runs/e10_three_elements/training_summary.json"),
        "e11_fullsvg_lr5e5": Path("runs/e11_fullsvg_lr5e5/training_summary.json"),
        "e12_fullsvg_lr1e4": Path("runs/e12_fullsvg_lr1e4/training_summary.json"),
        "e13_full217_seed42": Path("runs/e13_full217_seed42/training_summary.json"),
        "e14_full217_seed123": Path("runs/e14_full217_seed123/training_summary.json"),
    }
    exported = []
    for name, source in paths.items():
        if source.exists():
            target = output / f"{name}.json"
            write_json(target, sanitize(read_json(source), workspace))
            exported.append(str(target))
    write_json(output / "manifest.json", {"files": exported, "absolute_paths_removed": True})
    print(f"exported {len(exported)} summaries")


if __name__ == "__main__":
    main()
