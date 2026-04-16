"""A3: SAE Encoding — pass hidden states through SAE encoders at every layer.

Produces sparse feature activations as parquet files.  Pure matrix computation
— no model forward passes needed.

Usage:
    python scripts/A3_sae_encode.py --run_dir runs/llama-3.1-8b_2026-04-15/

    # Specific layers
    python scripts/A3_sae_encode.py --run_dir runs/... --layers 12,14,16

    # Quick test
    python scripts/A3_sae_encode.py --run_dir runs/... --max_items 20 --categories so

    # Single category
    python scripts/A3_sae_encode.py --run_dir runs/... --categories so
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.bbq_loader import ALL_CATEGORIES
from src.extraction.sae_encoding import (
    build_encoding_summary,
    build_subgroup_lookup,
    encode_layer,
    load_all_stimuli,
    load_metadata,
    select_encode_device,
    validate_sae_source,
)
from src.utils.logging import log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A3: Encode hidden states with SAEs.",
    )
    parser.add_argument(
        "--run_dir", type=str, required=True,
        help="Path to the run directory.",
    )
    parser.add_argument(
        "--layers", type=str, default=None,
        help="Comma-separated layer indices (default: all layers).",
    )
    parser.add_argument(
        "--categories", type=str, default=None,
        help="Comma-separated category short names.",
    )
    parser.add_argument(
        "--max_items", type=int, default=None,
        help="Max items per category (for quick testing).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Override encoding device (cuda, mps, cpu).",
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

    n_layers = config.get("n_layers")
    if n_layers is None:
        log("ERROR: config.json missing n_layers. Run A2 first.")
        sys.exit(1)

    encode_device = args.device or select_encode_device(config.get("device", "cpu"))

    # ── Resolve layers ───────────────────────────────────────────────
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        layers = list(range(n_layers))

    # ── Resolve categories ───────────────────────────────────────────
    if args.categories is not None:
        categories = [c.strip() for c in args.categories.split(",")]
    elif "categories" in config:
        categories = config["categories"]
    else:
        categories = list(ALL_CATEGORIES)

    log(f"A3: SAE Encoding")
    log(f"  run_dir: {run_dir}")
    log(f"  sae_source: {config['sae_source']}")
    log(f"  sae_expansion: {config['sae_expansion']}x")
    log(f"  encode_device: {encode_device}")
    log(f"  layers: {layers}")
    log(f"  categories: {categories}")
    if args.max_items:
        log(f"  max_items: {args.max_items}")

    # ── Startup validation ───────────────────────────────────────────
    validate_sae_source(config["sae_source"], layers[0], config["sae_expansion"])

    # ── Load metadata and subgroup lookup ────────────────────────────
    load_metadata(run_dir)  # ensure metadata.parquet exists
    stimuli_by_cat = load_all_stimuli(run_dir, categories)
    subgroup_lookup = build_subgroup_lookup(stimuli_by_cat)

    # ── Process each layer ───────────────────────────────────────────
    failed_layers: list[int] = []
    for layer in layers:
        success = encode_layer(
            layer, run_dir, config, categories, encode_device,
            subgroup_lookup, args.max_items,
        )
        if not success:
            failed_layers.append(layer)

    # ── Build global summary ─────────────────────────────────────────
    build_encoding_summary(
        run_dir, config, layers, failed_layers, categories, encode_device,
    )


if __name__ == "__main__":
    main()
