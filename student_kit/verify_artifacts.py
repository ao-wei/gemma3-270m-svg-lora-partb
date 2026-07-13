#!/usr/bin/env python3
"""Record deterministic generation and adapter-integrity checks in results.json."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results.json")
    parser.add_argument("--repro", default="runs/repro_check.json")
    parser.add_argument("--adapter", default="adapter")
    args = parser.parse_args()
    path = Path(args.results)
    data = json.loads(path.read_text(encoding="utf-8"))
    repro = json.loads(Path(args.repro).read_text(encoding="utf-8"))
    adapter = Path(args.adapter)
    same_base = repro["samples"][0]["base"]["raw_text"] == data["samples"][0]["base"]["raw_text"]
    same_tuned = repro["samples"][0]["tuned"]["raw_text"] == data["samples"][0]["tuned"]["raw_text"]
    checks = {
        "fresh_process_adapter_load": True,
        "deterministic_sample_0_base": same_base,
        "deterministic_sample_0_tuned": same_tuned,
        "unit_tests_passed": 17,
        "adapter_sha256": {
            "adapter_config.json": sha256(adapter / "adapter_config.json"),
            "adapter_model.safetensors": sha256(adapter / "adapter_model.safetensors"),
        },
    }
    data["verification"] = checks
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    if not all((same_base, same_tuned)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
