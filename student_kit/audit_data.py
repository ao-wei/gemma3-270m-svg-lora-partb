#!/usr/bin/env python3
"""Audit assignment data without modifying source JSONL files."""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from data import filter_uninformative_prompts, load_jsonl


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    train = load_jsonl("train.jsonl")
    valid = load_jsonl("valid.jsonl")
    train_prompts = [row["messages"][1]["content"] for row in train]
    valid_prompts = [row["messages"][1]["content"] for row in valid]
    filtered_train, rejected = filter_uninformative_prompts(train)
    invalid_xml = []
    viewboxes = set()
    for split, rows in (("train", train), ("valid", valid)):
        for index, row in enumerate(rows):
            try:
                root = ET.fromstring(row["messages"][2]["content"])
                viewboxes.add(root.attrib.get("viewBox"))
            except ET.ParseError as error:
                invalid_xml.append({"split": split, "index": index, "error": str(error)})
    result = {
        "train_rows": len(train),
        "valid_rows": len(valid),
        "unique_train_prompts": len(set(train_prompts)),
        "unique_valid_prompts": len(set(valid_prompts)),
        "prompt_overlap": len(set(train_prompts) & set(valid_prompts)),
        "uninformative_train_rows": rejected,
        "usable_train_rows": len(filtered_train),
        "invalid_target_xml": invalid_xml,
        "viewboxes": sorted(str(value) for value in viewboxes),
        "sha256": {"train.jsonl": sha256("train.jsonl"), "valid.jsonl": sha256("valid.jsonl")},
    }
    Path("data_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if invalid_xml or result["prompt_overlap"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
