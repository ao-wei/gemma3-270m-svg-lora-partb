#!/usr/bin/env python3
"""Add paired deltas and bootstrap confidence intervals to results.json."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from reward import reward


def percentile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def paired_bootstrap(deltas, seed=42, iterations=10_000):
    rng = random.Random(seed)
    means = [sum(rng.choice(deltas) for _ in deltas) / len(deltas) for _ in range(iterations)]
    return [percentile(means, 0.025), percentile(means, 0.975)]


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="results.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=10_000)
    args = parser.parse_args()
    path = Path(args.results)
    data = json.loads(path.read_text(encoding="utf-8"))
    for sample in data["samples"]:
        prompt = sample["prompt"]
        sample["base"]["reward"] = reward(prompt, sample["base"]["raw_text"])
        sample["tuned"]["reward"] = reward(prompt, sample["tuned"]["raw_text"])
    data["summary"] = {
        "base": summarize(data["samples"], "base"),
        "tuned": summarize(data["samples"], "tuned"),
    }
    comparison = {}
    for metric in ("total", "validity", "fidelity"):
        deltas = [sample["tuned"]["reward"][metric] - sample["base"]["reward"][metric] for sample in data["samples"]]
        comparison[metric] = {
            "paired_mean_delta": sum(deltas) / len(deltas),
            "bootstrap_95_ci": paired_bootstrap(deltas, args.seed, args.iterations),
            "improved": sum(delta > 1e-12 for delta in deltas),
            "unchanged": sum(abs(delta) <= 1e-12 for delta in deltas),
            "worsened": sum(delta < -1e-12 for delta in deltas),
        }
    data["comparison"] = comparison
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
