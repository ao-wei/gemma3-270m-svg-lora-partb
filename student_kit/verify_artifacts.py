#!/usr/bin/env python3
"""Record deterministic generation and adapter-integrity checks in results.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import jsonschema


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results.json")
    parser.add_argument("--repro", default="runs/repro_check.json")
    parser.add_argument("--adapter", default="adapter")
    parser.add_argument("--skip-determinism", action="store_true")
    args = parser.parse_args()
    path = Path(args.results)
    data = json.loads(path.read_text(encoding="utf-8"))
    repro = None if args.skip_determinism else json.loads(Path(args.repro).read_text(encoding="utf-8"))
    adapter = Path(args.adapter)
    schema = json.loads(Path("student_kit/results_schema_v2.json").read_text(encoding="utf-8"))
    jsonschema.validate(data, schema)
    same_structure = same_prompts = None
    base_matches = tuned_matches = []
    if repro is not None:
        same_structure = [sample["id"] for sample in repro["samples"]] == [sample["id"] for sample in data["samples"]]
        same_prompts = [sample["prompt"] for sample in repro["samples"]] == [sample["prompt"] for sample in data["samples"]]
        base_matches = [
            left["base"]["raw_text"] == right["base"]["raw_text"]
            for left, right in zip(repro["samples"], data["samples"])
        ]
        tuned_matches = [
            left["tuned"]["raw_text"] == right["tuned"]["raw_text"]
            for left, right in zip(repro["samples"], data["samples"])
        ]
    test_process = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        check=False,
        capture_output=True,
        text=True,
    )
    test_output = test_process.stdout + test_process.stderr
    count_match = re.search(r"Ran (\d+) tests?", test_output)
    test_count = int(count_match.group(1)) if count_match else 0
    load_code = (
        "from transformers import AutoModelForCausalLM; from peft import PeftModel; "
        "b=AutoModelForCausalLM.from_pretrained('gemma3-270m-it',local_files_only=True,low_cpu_mem_usage=True); "
        f"PeftModel.from_pretrained(b,{str(adapter)!r})"
    )
    adapter_load = subprocess.run([sys.executable, "-c", load_code], check=False).returncode == 0
    checks = {
        "fresh_process_adapter_load": adapter_load,
        "results_schema_v2_valid": True,
        "repro_sample_count": len(base_matches) if repro is not None else None,
        "repro_structure_match": (same_structure and same_prompts) if repro is not None else None,
        "deterministic_base_matches": sum(base_matches) if repro is not None else None,
        "deterministic_tuned_matches": sum(tuned_matches) if repro is not None else None,
        "deterministic_all_34_outputs": (
            same_structure and same_prompts and all(base_matches) and all(tuned_matches)
        ) if repro is not None else None,
        "unit_tests_passed": test_process.returncode == 0,
        "unit_test_count": test_count,
        "adapter_sha256": {
            "adapter_config.json": sha256(adapter / "adapter_config.json"),
            "adapter_model.safetensors": sha256(adapter / "adapter_model.safetensors"),
        },
        "data_sha256": {
            "train.jsonl": sha256(Path("train.jsonl")),
            "valid.jsonl": sha256(Path("valid.jsonl")),
        },
    }
    data["verification"] = checks
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    determinism_ok = checks["deterministic_all_34_outputs"] if repro is not None else True
    if not all((adapter_load, test_process.returncode == 0, determinism_ok)):
        print(test_output, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
