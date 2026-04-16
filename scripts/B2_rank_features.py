"""B2: Feature Ranking Per Subgroup — rank, injection layers, overlap analysis.

Collects FDR-significant features from B1 across all layers, ranks by effect
size, determines injection layers, and computes cross-subgroup feature overlap.

Usage:
    python scripts/B2_rank_features.py --run_dir runs/llama-3.1-8b_2026-04-15/

    # Custom k values and structural threshold
    python scripts/B2_rank_features.py --run_dir runs/... --overlap_ks 5,10,20,50

    # Skip figures
    python scripts/B2_rank_features.py --run_dir runs/... --skip_figures

    # Force rerun
    python scripts/B2_rank_features.py --run_dir runs/... --force
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.differential import load_metadata
from src.analysis.ranking import (
    K_VALUES_DEFAULT,
    build_injection_layers,
    build_ranking_summary,
    compute_all_overlaps,
    compute_item_overlap,
    deduplicate_defensive,
    enumerate_subgroups,
    load_all_significant,
    rank_features_all,
)
from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="B2: Feature ranking per subgroup.",
    )
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument(
        "--overlap_ks", type=str, default=None,
        help=f"Comma-separated k values (default: {K_VALUES_DEFAULT}).",
    )
    parser.add_argument(
        "--structural_overlap_threshold", type=float, default=0.5,
        help="Item-overlap fraction to flag structural pairs (default: 0.5).",
    )
    parser.add_argument("--skip_figures", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Rerun even if output exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    t0 = time.time()

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

    # ── Resume check ─────────────────────────────────────────────────
    out_dir = ensure_dir(run_dir / "B_feature_ranking")
    summary_path = out_dir / "ranking_summary.json"
    if summary_path.exists() and not args.force:
        log(f"B2 output already exists at {summary_path}. Use --force to rerun.")
        sys.exit(0)

    k_values = (
        [int(x) for x in args.overlap_ks.split(",")]
        if args.overlap_ks else K_VALUES_DEFAULT
    )
    structural_threshold = args.structural_overlap_threshold

    log(f"B2: Feature Ranking Per Subgroup")
    log(f"  run_dir: {run_dir}")
    log(f"  k_values: {k_values}")
    log(f"  structural_threshold: {structural_threshold}")

    # ── Step 1: Load all significant features ────────────────────────
    combined = load_all_significant(run_dir, n_layers)

    # ── Step 2: Defensive dedup ──────────────────────────────────────
    combined = deduplicate_defensive(combined)

    # ── Step 3: Enumerate subgroups ──────────────────────────────────
    b1_summary_path = run_dir / "B_differential" / "differential_summary.json"
    b1_summary: dict = {}
    if b1_summary_path.exists():
        with open(b1_summary_path) as f:
            b1_summary = json.load(f)

    subgroups, enum_report = enumerate_subgroups(combined, b1_summary)
    log(f"Subgroups to rank: {len(subgroups)}")

    # ── Step 4: Rank features ────────────────────────────────────────
    ranked_df = rank_features_all(combined, subgroups)
    log(f"Ranked features: {len(ranked_df)} rows")

    # Save ranked features parquet.
    ranked_path = out_dir / "ranked_features.parquet"
    tmp_path = ranked_path.with_suffix(".parquet.tmp")
    ranked_df.to_parquet(tmp_path, index=False, compression="snappy")
    tmp_path.rename(ranked_path)
    log(f"  Saved → {ranked_path.name}")

    # ── Step 5: Injection layers ─────────────────────────────────────
    injection_layers = build_injection_layers(ranked_df, subgroups)
    atomic_save_json(injection_layers, out_dir / "injection_layers.json")
    log(f"  Saved injection_layers.json ({len(injection_layers)} subgroups)")

    # ── Step 6: Feature overlap ──────────────────────────────────────
    overlap_data = compute_all_overlaps(ranked_df, subgroups, k_values)
    atomic_save_json(overlap_data, out_dir / "feature_overlap.json")
    n_cats_overlap = len(overlap_data)
    log(f"  Saved feature_overlap.json ({n_cats_overlap} categories)")

    # ── Step 7: Item overlap ─────────────────────────────────────────
    meta_df = load_metadata(run_dir)
    item_overlap = compute_item_overlap(
        meta_df, subgroups, structural_threshold,
    )
    atomic_save_json(item_overlap, out_dir / "item_overlap_report.json")
    log(f"  Saved item_overlap_report.json")

    # ── Step 8: Summary ──────────────────────────────────────────────
    elapsed = time.time() - t0
    summary = build_ranking_summary(
        n_layers, subgroups, ranked_df, injection_layers,
        enum_report, k_values, structural_threshold, elapsed,
    )
    atomic_save_json(summary, summary_path)
    log(f"  Saved ranking_summary.json")

    # ── Step 9: Figures ──────────────────────────────────────────────
    if not args.skip_figures:
        from src.visualization.ranking_figures import generate_all_b2_figures
        fig_dir = ensure_dir(out_dir / "figures")
        generate_all_b2_figures(
            ranked_df, subgroups, overlap_data, injection_layers,
            n_layers, fig_dir,
        )

    log(f"\nB2 complete ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
