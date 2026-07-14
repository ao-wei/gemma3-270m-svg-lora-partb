#!/usr/bin/env python3
"""Apply the predeclared E11/E12 learning-rate selection rule on internal dev only."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import read_json, write_json


def metrics(result: dict, training: dict) -> dict:
    tuned = result["summary"]["tuned"]
    return {
        "total": tuned["total"]["mean"],
        "fidelity": tuned["fidelity"]["mean"],
        "fatal_rate": tuned["fatal_rate"],
        "valid_pass_rate": tuned["valid_pass_rate"],
        "quality_pass_rate": tuned["quality_pass_rate"],
        "eval_loss": training.get("eval_metrics", {}).get("eval_loss"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-result", required=True)
    parser.add_argument("--e11-result", required=True)
    parser.add_argument("--e12-result", required=True)
    parser.add_argument("--e11-training", default="runs/e11_fullsvg_lr5e5/training_summary.json")
    parser.add_argument("--e12-training", default="runs/e12_fullsvg_lr1e4/training_summary.json")
    parser.add_argument("--output", default="artifacts/model_selection.json")
    args = parser.parse_args()

    stage1 = read_json(args.stage1_result)["summary"]["tuned"]
    candidates = {
        "E11": {
            "learning_rate": 5e-5,
            **metrics(read_json(args.e11_result), read_json(args.e11_training)),
        },
        "E12": {
            "learning_rate": 1e-4,
            **metrics(read_json(args.e12_result), read_json(args.e12_training)),
        },
    }
    for candidate in candidates.values():
        candidate["excluded"] = bool(
            candidate["fatal_rate"] > stage1["fatal_rate"] + 0.10
            and candidate["fidelity"] <= stage1["fidelity"]["mean"]
        )
    eligible = [(name, item) for name, item in candidates.items() if not item["excluded"]]
    if eligible:
        def key(pair):
            item = pair[1]
            eval_loss = item["eval_loss"] if item["eval_loss"] is not None else float("inf")
            return (item["total"], item["fidelity"], -eval_loss)

        selected, selected_metrics = max(eligible, key=key)
        reason = "eligible candidates ranked by total, fidelity, then lower eval loss"
    else:
        selected, selected_metrics = "E11", candidates["E11"]
        reason = "all candidates failed the gate; conservative 5e-5 fallback"

    write_json(args.output, {
        "selection_scope": "22-row internal development split only",
        "validation_set_used": False,
        "rule": (
            "Exclude fatal_rate > stage1 + 0.10 when fidelity does not improve; "
            "rank remaining by total, fidelity, then lower eval_loss; fallback to 5e-5."
        ),
        "stage1": {
            "fatal_rate": stage1["fatal_rate"],
            "fidelity": stage1["fidelity"]["mean"],
        },
        "candidates": candidates,
        "selected_experiment": selected,
        "selected_learning_rate": selected_metrics["learning_rate"],
        "reason": reason,
        "inputs": {key: str(Path(value)) for key, value in vars(args).items() if key != "output"},
    })
    print(f"selected {selected} ({selected_metrics['learning_rate']}) -> {args.output}")


if __name__ == "__main__":
    main()
