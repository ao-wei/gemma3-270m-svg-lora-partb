"""Small runtime helpers shared by command-line scripts."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def read_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    import torch
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def device_name() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def is_bf16_compatibility_error(error: BaseException) -> bool:
    message = str(error).lower()
    if "out of memory" in message or "allocation" in message:
        return False
    patterns = (
        "mps does not support bfloat16",
        "not implemented for 'bfloat16'",
        "does not support dtype bfloat16",
        "unsupported datatype for constant",
    )
    return any(pattern in message for pattern in patterns)
