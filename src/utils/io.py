"""I/O utilities: atomic saves, resume checking, path helpers."""

import json
import os
from pathlib import Path
from typing import Any


def atomic_save_json(data: Any, path: str | Path) -> None:
    """Write JSON atomically: write to .tmp then rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.rename(tmp_path, path)


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist, return as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
