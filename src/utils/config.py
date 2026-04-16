"""Configuration management: loading, validation, device detection, provenance."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from src.data.bbq_loader import ALL_CATEGORIES
from src.utils.io import atomic_save_json
from src.utils.logging import log


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def detect_device(preferred: str | None = None) -> str:
    """Auto-detect the best available compute device.

    Priority: explicit preference → CUDA → MPS → CPU.  Validates that the
    requested device is actually available.
    """
    if preferred and preferred != "auto":
        # Validate explicit preference.
        if preferred.startswith("cuda"):
            if not torch.cuda.is_available():
                log(f"WARNING: requested device '{preferred}' but CUDA unavailable, "
                    f"falling back to auto-detect")
            else:
                return preferred
        elif preferred == "mps":
            if not torch.backends.mps.is_available():
                log(f"WARNING: requested device 'mps' but MPS unavailable, "
                    f"falling back to auto-detect")
            else:
                return "mps"
        elif preferred == "cpu":
            return "cpu"

    # Auto-detect.
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        log(f"Detected CUDA: {device_name}")
        return "cuda"
    if torch.backends.mps.is_available():
        log("Detected MPS (Apple Silicon)")
        return "mps"
    log("No GPU detected, using CPU")
    return "cpu"


def device_info() -> dict[str, Any]:
    """Collect device/platform info for provenance."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
        info["cuda_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_mem / (1024**3), 1,
        )
    return info


# ---------------------------------------------------------------------------
# Config loading and validation
# ---------------------------------------------------------------------------

REQUIRED_FIELDS_A1 = {"bbq_data_dir"}
REQUIRED_FIELDS_A2 = {"model_path"}
REQUIRED_FIELDS_A3 = {"sae_source", "sae_expansion"}


def load_config(run_dir: Path) -> dict[str, Any]:
    """Load and return config.json from a run directory."""
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found at {config_path}")
    with open(config_path) as f:
        return json.load(f)


def validate_config(
    config: dict[str, Any],
    stages: list[str],
    project_root: Path,
) -> list[str]:
    """Validate config has all fields needed for the given stages.

    Returns a list of error messages (empty = valid).
    """
    errors: list[str] = []

    required: set[str] = set()
    if "A1" in stages:
        required |= REQUIRED_FIELDS_A1
    if "A2" in stages:
        required |= REQUIRED_FIELDS_A2
    if "A3" in stages:
        required |= REQUIRED_FIELDS_A3

    for field in sorted(required):
        if field not in config or not config[field]:
            errors.append(f"Missing required config field: '{field}'")

    # Validate paths exist.
    if "A1" in stages and "bbq_data_dir" in config:
        bbq_dir = Path(config["bbq_data_dir"])
        if not bbq_dir.is_absolute():
            bbq_dir = project_root / bbq_dir
        if not bbq_dir.is_dir():
            errors.append(f"bbq_data_dir not found: {bbq_dir}")

    if "A2" in stages and "model_path" in config:
        mp = config["model_path"]
        mp_path = Path(mp)
        # Accept either a local path or a HuggingFace repo ID (contains "/").
        is_local = mp_path.exists()
        is_hf_id = "/" in mp and not mp.startswith("/")
        if not is_local and not is_hf_id:
            errors.append(f"model_path not found locally and doesn't look like "
                          f"a HuggingFace repo ID: {mp}")

    # Validate categories.
    cats = config.get("categories", ALL_CATEGORIES)
    invalid = [c for c in cats if c not in ALL_CATEGORIES]
    if invalid:
        errors.append(f"Unknown categories: {invalid}. Valid: {ALL_CATEGORIES}")

    return errors


def save_config(config: dict[str, Any], run_dir: Path) -> None:
    """Atomically save config.json to the run directory."""
    atomic_save_json(config, run_dir / "config.json")


# ---------------------------------------------------------------------------
# Run directory setup
# ---------------------------------------------------------------------------

def setup_run_dir(
    run_dir: str | Path,
    config: dict[str, Any],
) -> Path:
    """Ensure run directory exists and config.json is written."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.json"
    if not config_path.exists():
        save_config(config, run_dir)
        log(f"Created config.json at {config_path}")
    else:
        log(f"Using existing config.json at {config_path}")

    return run_dir


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def git_hash() -> str | None:
    """Return the current git commit hash, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def build_provenance(
    config: dict[str, Any],
    device: str,
    stages: list[str],
) -> dict[str, Any]:
    """Build a provenance record for this pipeline run."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stages": stages,
        "device": device,
        "device_info": device_info(),
        "git_hash": git_hash(),
        "config_snapshot": {k: v for k, v in config.items()
                           if k not in ("model_path",)},
    }
