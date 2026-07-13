"""Dataset preparation shared by training and tests."""

from __future__ import annotations

import json
import random
import copy
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


VISIBLE_SVG_TAGS = {"path", "circle", "ellipse", "rect", "polygon", "polyline", "line"}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    for index, row in enumerate(rows):
        roles = [message.get("role") for message in row.get("messages", [])]
        if roles != ["system", "user", "assistant"]:
            raise ValueError(f"{path}:{index + 1}: expected system/user/assistant, got {roles}")
    return rows


def filter_uninformative_prompts(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[int]]:
    """Remove known non-task rows while preserving the source JSONL unchanged."""
    rejected = []
    kept = []
    for index, row in enumerate(rows):
        prompt = row["messages"][1]["content"].strip().lower()
        if prompt in {"placeholder", "todo", "n/a"}:
            rejected.append(index)
        else:
            kept.append(row)
    return kept, rejected


def simplify_svg_targets(rows: list[dict[str, Any]], max_visible_elements: int = 6) -> list[dict[str, Any]]:
    """Create short, complete SVG targets from source targets without modifying them."""
    if max_visible_elements < 1:
        raise ValueError("max_visible_elements must be positive")
    namespace = "http://www.w3.org/2000/svg"
    ET.register_namespace("", namespace)
    simplified = []
    for row in rows:
        cloned = copy.deepcopy(row)
        source = ET.fromstring(row["messages"][2]["content"])
        output = ET.Element(f"{{{namespace}}}svg", {"viewBox": source.attrib.get("viewBox", "0 0 256 256")})
        selected = []
        for element in source.iter():
            tag = element.tag.rsplit("}", 1)[-1].lower()
            if tag not in VISIBLE_SVG_TAGS:
                continue
            item = copy.deepcopy(element)
            item.tail = None
            for key, value in list(item.attrib.items()):
                if "url(" in str(value).lower():
                    if key.rsplit("}", 1)[-1] in {"fill", "stroke"}:
                        item.attrib[key] = "#666666"
                    else:
                        del item.attrib[key]
            selected.append(item)
            if len(selected) == max_visible_elements:
                break
        if not selected:
            raise ValueError("source SVG has no supported visible elements")
        output.extend(selected)
        cloned["messages"][2]["content"] = ET.tostring(output, encoding="unicode", short_empty_elements=True)
        simplified.append(cloned)
    return simplified


def split_train_dev(rows: list[dict[str, Any]], dev_size: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < dev_size < len(rows):
        raise ValueError("dev_size must be between 1 and len(rows)-1")
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    dev_indices = set(indices[:dev_size])
    train = [row for index, row in enumerate(rows) if index not in dev_indices]
    dev = [row for index, row in enumerate(rows) if index in dev_indices]
    return train, dev


def tokenize_example(row: dict[str, Any], tokenizer: Any, max_length: int) -> dict[str, list[int]]:
    messages = row["messages"]
    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True
    )
    full_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("chat template prompt is not a prefix of the full conversation")
    input_ids = full_ids[:max_length]
    prompt_length = min(len(prompt_ids), len(input_ids))
    labels = [-100] * prompt_length + input_ids[prompt_length:]
    if not any(label != -100 for label in labels):
        raise ValueError("max_length truncates the entire assistant response")
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


class CompletionOnlyCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]):
        import torch

        max_length = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * padding)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
            batch["labels"].append(feature["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}
