"""B3: Subgroup Direction Geometry — DIM directions, cosines, alignment.

Computes Difference-in-Means (DIM) directions for each subgroup under two
normalization regimes (raw-based, normed-based), plus pairwise cosine
matrices, differentiation metrics, and bias-identity alignment scores.

Independent of the SAE — reads raw A2 activations directly.

Usage:
    python scripts/B3_geometry.py --run_dir runs/llama-3.1-8b_2026-04-15/

    # Quick test
    python scripts/B3_geometry.py --run_dir runs/... --max_items 100

    # Single category
    python scripts/B3_geometry.py --run_dir runs/... --categories so

    # Override minimum sample size
    python scripts/B3_geometry.py --run_dir runs/... --min_n_per_group 15
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.differential import load_metadata
from src.analysis.geometry import (
    b3_complete,
    build_summary,
    compute_alignment,
    compute_all_cosines,
    compute_differentiation_metrics,
    process_category,
    save_cosines,
    save_directions,
)
from src.data.bbq_loader import ALL_CATEGORIES
from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log
from src.visualization.geometry_figures import generate_all_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="B3: Subgroup direction geometry (DIM directions).",
    )
    parser.add_argument(
        "--run_dir", type=str, required=True,
        help="Path to the run directory.",
    )
    parser.add_argument(
        "--categories", type=str, default=None,
        help="Comma-separated category short names (default: all from config).",
    )
    parser.add_argument(
        "--max_items", type=int, default=None,
        help="Max items per category (for quick testing).",
    )
    parser.add_argument(
        "--min_n_per_group", type=int, default=10,
        help="Minimum items per comparison group (default: 10).",
    )
    parser.add_argument(
        "--skip_figures", action="store_true",
        help="Skip figure generation.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rerun even if output already exists.",
    )
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
    hidden_dim = config.get("hidden_dim")
    if n_layers is None or hidden_dim is None:
        log("ERROR: config.json missing n_layers or hidden_dim. Run A2 first.")
        sys.exit(1)

    # ── Resume check ─────────────────────────────────────────────────
    if b3_complete(run_dir) and not args.force:
        log("B3 output already exists. Use --force to rerun.")
        sys.exit(0)

    # ── Resolve parameters ───────────────────────────────────────────
    if args.categories is not None:
        categories = [c.strip() for c in args.categories.split(",")]
    elif "categories" in config:
        categories = config["categories"]
    else:
        categories = list(ALL_CATEGORIES)

    min_n = args.min_n_per_group

    log("B3: Subgroup Direction Geometry")
    log(f"  run_dir: {run_dir}")
    log(f"  categories: {categories}")
    log(f"  n_layers: {n_layers}, hidden_dim: {hidden_dim}")
    log(f"  min_n_per_group: {min_n}")
    if args.max_items:
        log(f"  max_items: {args.max_items}")

    # ── Step 1: Load metadata ────────────────────────────────────────
    meta_df = load_metadata(run_dir)

    # ── Step 2: Process each category ────────────────────────────────
    directions_arrays: dict[str, "np.ndarray"] = {}
    directions_norms: dict[str, "np.ndarray"] = {}
    subgroup_info: dict[tuple[str, str], dict] = {}

    for cat in categories:
        process_category(
            cat=cat,
            run_dir=run_dir,
            meta_df=meta_df,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            min_n=min_n,
            max_items=args.max_items,
            directions_arrays=directions_arrays,
            directions_norms=directions_norms,
            subgroup_info=subgroup_info,
        )

    if not directions_arrays:
        log("ERROR: No directions computed across any category. Check activations.")
        sys.exit(1)

    # ── Step 3: Save directions ──────────────────────────────────────
    save_directions(run_dir, directions_arrays, directions_norms)

    # ── Step 4: Pairwise cosines ─────────────────────────────────────
    log("\nComputing pairwise cosines...")
    cosine_df = compute_all_cosines(directions_arrays, categories, n_layers)
    save_cosines(run_dir, cosine_df)

    # ── Step 5: Differentiation metrics ──────────────────────────────
    log("Computing differentiation metrics...")
    differentiation = compute_differentiation_metrics(cosine_df, categories, n_layers)
    out_dir = ensure_dir(run_dir / "B_geometry")
    atomic_save_json(differentiation, out_dir / "differentiation_metrics.json")
    log(f"Saved differentiation_metrics.json")

    # Log peak layers
    for cat in categories:
        for dtype_label in ["identity_normed", "bias_normed"]:
            d = differentiation.get(cat, {}).get(dtype_label)
            if d:
                mp = d.get("most_anti_aligned_pair_at_peak")
                mp_str = ""
                if mp:
                    mp_str = f", most anti-aligned: {mp['subgroup_A']}--{mp['subgroup_B']} cos={mp['cosine']:.3f}"
                log(f"  {cat} {dtype_label}: peak L={d['peak_layer']}, "
                    f"var={d['peak_variance']:.4f}{mp_str}")

    # ── Step 6: Bias-identity alignment ──────────────────────────────
    log("\nComputing bias-identity alignment...")
    alignment = compute_alignment(directions_arrays, categories, n_layers)
    atomic_save_json(alignment, out_dir / "bias_identity_alignment.json")
    log("Saved bias_identity_alignment.json")

    # ── Step 7: Summary ──────────────────────────────────────────────
    summary = build_summary(subgroup_info, directions_norms, categories, min_n)
    atomic_save_json(summary, out_dir / "subgroup_directions_summary.json")
    log(f"Saved subgroup_directions_summary.json "
        f"({summary['n_subgroups_total']} subgroups, "
        f"{summary['n_subgroups_with_bias']} with bias, "
        f"{summary['n_subgroups_with_identity']} with identity)")

    # ── Step 8: Figures ──────────────────────────────────────────────
    if not args.skip_figures:
        generate_all_figures(
            run_dir=run_dir,
            directions_norms=directions_norms,
            cosine_df=cosine_df,
            differentiation=differentiation,
            alignment=alignment,
            categories=categories,
            n_layers=n_layers,
        )

    elapsed = time.time() - t0
    log(f"\nB3 complete in {elapsed:.1f}s")
    log(f"  Directions: {len(directions_arrays)} arrays")
    log(f"  Cosine pairs: {len(cosine_df)} rows")
    log(f"  Output: {out_dir}")


if __name__ == "__main__":
    main()
