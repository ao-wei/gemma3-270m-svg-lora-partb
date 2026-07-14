"""Shared results-v2 pass rules and aggregation."""

from __future__ import annotations

import statistics


DEGENERACY_VIOLATIONS = {"background_only_or_blank", "no_on_canvas_elements"}


def add_passes(scored_output: dict) -> dict:
    assessment = scored_output["reward"]
    valid = not assessment["metadata"]["fatal"] and assessment["validity"] >= 0.8
    quality = (
        valid
        and assessment["total"] >= 0.5
        and assessment["fidelity"] >= 0.3
        and not (DEGENERACY_VIOLATIONS & set(assessment["violations"]))
    )
    scored_output["passes"] = {"valid": valid, "quality": quality}
    return scored_output


def summarize(samples: list[dict], key: str) -> dict:
    summary = {}
    for metric in ("total", "validity", "fidelity"):
        values = [sample[key]["reward"][metric] for sample in samples]
        summary[metric] = {
            "mean": sum(values) / len(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
    valid_rate = sum(sample[key]["passes"]["valid"] for sample in samples) / len(samples)
    quality_rate = sum(sample[key]["passes"]["quality"] for sample in samples) / len(samples)
    summary.update({
        "fatal_rate": sum(sample[key]["reward"]["metadata"]["fatal"] for sample in samples) / len(samples),
        "pass_rate": quality_rate,
        "valid_pass_rate": valid_rate,
        "quality_pass_rate": quality_rate,
    })
    return summary
