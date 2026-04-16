"""B1: Differential Feature Analysis — identify bias-associated SAE features.

For each subgroup at each layer, tests which SAE features activate differently
when the model produces a stereotyped response vs. when it doesn't.

Usage:
    python scripts/B1_differential.py --run_dir runs/llama-3.1-8b_2026-04-15/

    # Quick test
    python scripts/B1_differential.py --run_dir runs/... --max_items 100 --layers 14

    # Single category
    python scripts/B1_differential.py --run_dir runs/... --categories so

    # Override min sample size
    python scripts/B1_differential.py --run_dir runs/... --min_n_per_group 15
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.differential import (
    build_differential_summary,
    build_subgroup_catalog,
    load_layer_summary,
    load_metadata,
    process_layer,
    save_layer_parquet,
    save_layer_summary,
)
from src.data.bbq_loader import ALL_CATEGORIES
from src.utils.logging import log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="B1: Differential feature analysis.",
    )
    parser.add_argument(
        "--run_dir", type=str, required=True,
        help="Path to the run directory.",
    )
    parser.add_argument(
        "--categories", type=str, default=None,
        help="Comma-separated category short names.",
    )
    parser.add_argument(
        "--layers", type=str, default=None,
        help="Comma-separated layer indices (default: all layers).",
    )
    parser.add_argument(
        "--max_items", type=int, default=None,
        help="Max ambig items per category (for quick testing).",
    )
    parser.add_argument(
        "--min_n_per_group", type=int, default=10,
        help="Minimum items per comparison group (default: 10).",
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

    # ── Resolve parameters ───────────────────────────────────────────
    if args.categories is not None:
        categories = [c.strip() for c in args.categories.split(",")]
    elif "categories" in config:
        categories = config["categories"]
    else:
        categories = list(ALL_CATEGORIES)

    if args.layers is not None:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        layers = list(range(n_layers))

    min_n = args.min_n_per_group

    log(f"B1: Differential Feature Analysis")
    log(f"  run_dir: {run_dir}")
    log(f"  categories: {categories}")
    log(f"  layers: {layers}")
    log(f"  min_n_per_group: {min_n}")
    if args.max_items:
        log(f"  max_items: {args.max_items}")

    # ── Step 0: Load metadata ────────────────────────────────────────
    meta_df = load_metadata(run_dir)

    # ── Step 0b: Subgroup catalog ────────────────────────────────────
    subgroup_catalog = build_subgroup_catalog(meta_df, categories, min_n)

    n_analyzable = sum(1 for v in subgroup_catalog.values() if v["analyzable"])
    log(f"Subgroups to analyze: {n_analyzable} across {len(categories)} categories")
    for cat in categories:
        subs = [
            s for s, v in subgroup_catalog.items()
            if v["category"] == cat and v["analyzable"]
        ]
        log(f"  {cat}: {len(subs)} subgroups")

    # ── Process each layer ───────────────────────────────────────────
    all_summaries: dict[int, dict] = {}

    for layer in layers:
        out_path = run_dir / "B_differential" / f"layer_{layer:02d}.parquet"

        if out_path.exists():
            log(f"Layer {layer:02d}: already processed, skipping")
            all_summaries[layer] = load_layer_summary(run_dir, layer)
            continue

        layer_results, layer_summary = process_layer(
            layer=layer,
            run_dir=run_dir,
            meta_df=meta_df,
            categories=categories,
            subgroup_catalog=subgroup_catalog,
            min_n=min_n,
            max_items=args.max_items,
        )

        save_layer_parquet(run_dir, layer, layer_results)
        save_layer_summary(run_dir, layer, layer_summary)
        all_summaries[layer] = layer_summary

    # ── Build global summary ─────────────────────────────────────────
    build_differential_summary(
        run_dir, layers, subgroup_catalog, all_summaries, min_n,
    )


if __name__ == "__main__":
    main()
