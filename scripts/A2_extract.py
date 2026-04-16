"""A2: Extract Activations — run BBQ items through the model and capture hidden states.

Produces one .npz file per item containing unit-normalized last-token hidden
states at every layer, plus behavioral metadata (model answer, logits, margin).

Usage:
    python scripts/A2_extract.py --run_dir runs/llama-3.1-8b_2026-04-15/

    # Quick test (20 items per category)
    python scripts/A2_extract.py --run_dir runs/llama-3.1-8b_2026-04-15/ --max_items 20

    # Single category
    python scripts/A2_extract.py --run_dir runs/llama-3.1-8b_2026-04-15/ --categories so

    # Override device
    python scripts/A2_extract.py --run_dir runs/llama-3.1-8b_2026-04-15/ --device cuda
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.bbq_loader import ALL_CATEGORIES
from src.extraction.activations import (
    build_and_save_extraction_summary,
    extract_category,
    load_model,
)
from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A2: Extract activations for a run.",
    )
    parser.add_argument(
        "--run_dir", type=str, required=True,
        help="Path to the run directory (must contain config.json and A1 stimuli).",
    )
    parser.add_argument(
        "--categories", type=str, default=None,
        help="Comma-separated category short names (default: from config.json).",
    )
    parser.add_argument(
        "--max_items", type=int, default=None,
        help="Max items per category (for quick testing).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Override device (cuda, mps, cpu).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)

    # ── Load config ──────────────────────────────────────────────────
    config_path = run_dir / "config.json"
    if not config_path.exists():
        log(f"ERROR: config.json not found at {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    device = args.device or config.get("device", "cpu")
    model_path = config["model_path"]

    # Resolve categories.
    if args.categories is not None:
        categories = [c.strip() for c in args.categories.split(",")]
    elif "categories" in config:
        categories = config["categories"]
    else:
        categories = list(ALL_CATEGORIES)

    log(f"A2: Extracting activations")
    log(f"  run_dir: {run_dir}")
    log(f"  model_path: {model_path}")
    log(f"  device: {device}")
    log(f"  categories: {categories}")
    if args.max_items:
        log(f"  max_items: {args.max_items}")

    # ── Load model once ──────────────────────────────────────────────
    model, tokenizer, n_layers, hidden_dim, get_layer_fn = load_model(
        model_path, device,
    )

    # Update config with architecture info (first run only).
    if config.get("n_layers") is None:
        config["n_layers"] = n_layers
        config["hidden_dim"] = hidden_dim
        atomic_save_json(config, config_path)
        log(f"  Updated config.json with n_layers={n_layers}, hidden_dim={hidden_dim}")

    # ── Process each category ────────────────────────────────────────
    for cat in categories:
        stimuli_path = run_dir / "A_extraction" / "stimuli" / f"{cat}.json"
        if not stimuli_path.exists():
            log(f"No stimuli for {cat}, skipping. Run A1 first.")
            continue

        output_dir = ensure_dir(
            run_dir / "A_extraction" / "activations" / cat,
        )

        with open(stimuli_path) as f:
            items = json.load(f)

        if args.max_items:
            items = items[: args.max_items]

        extract_category(
            items, model, tokenizer, get_layer_fn,
            n_layers, hidden_dim, device, output_dir, cat,
        )

    # ── Save extraction summary ──────────────────────────────────────
    build_and_save_extraction_summary(run_dir, categories, config)


if __name__ == "__main__":
    main()
