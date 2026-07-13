#!/usr/bin/env python3
"""Export the deterministic internal development split as a derived artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data import filter_uninformative_prompts, load_jsonl, split_train_dev


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train.jsonl")
    parser.add_argument("--output", default="runs/internal_dev_seed42.jsonl")
    parser.add_argument("--dev-size", type=int, default=22)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows, _ = filter_uninformative_prompts(load_jsonl(args.train))
    _, dev = split_train_dev(rows, args.dev_size, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in dev), encoding="utf-8")
    print(f"wrote {len(dev)} rows to {output}")


if __name__ == "__main__":
    main()
