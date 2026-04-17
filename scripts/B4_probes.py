"""B4: Probe Training + Controls — linear probes on hidden states and SAE features.

Trains PCA-50 → L2-regularized logistic regression probes to test what is
linearly decodable at each layer.  Includes permutation baselines, structural
controls (context condition, template ID), cross-category and cross-subgroup
generalization, and SAE-feature probes.

Usage:
    python scripts/B4_probes.py --run_dir runs/llama-3.1-8b_2026-04-15/

    # Subset of layers
    python scripts/B4_probes.py --run_dir runs/... --layers 0,4,8,12,16,20,24,28,31

    # Override permutation trials
    python scripts/B4_probes.py --run_dir runs/... --n_permutations 20

    # Skip SAE-based probes
    python scripts/B4_probes.py --run_dir runs/... --skip_sae_probes

    # Skip figures
    python scripts/B4_probes.py --run_dir runs/... --skip_figures
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
from src.analysis.probes import (
    SEED,
    N_COMPONENTS,
    N_FOLDS,
    LR_C,
    SAE_TOP_K,
    b4_complete,
    build_probes_summary,
    build_sae_feature_matrix,
    enumerate_subgroups,
    get_groups_for_items,
    load_category_hidden_states_by_layer,
    load_question_index_map,
    probe_binary_subgroup,
    probe_context_condition,
    probe_cross_category,
    probe_multiclass_subgroup,
    probe_sae_binary_subgroup,
    probe_stereotyped_response,
    probe_template_id,
    probe_within_cat_cross_subgroup,
    save_cross_cat_results,
    save_probe_results,
    save_within_cat_results,
)
from src.data.bbq_loader import ALL_CATEGORIES
from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log, progress_bar
from src.visualization.probe_figures import generate_all_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="B4: Probe training + controls.",
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
        "--layers", type=str, default=None,
        help="Comma-separated layer indices (default: all layers).",
    )
    parser.add_argument(
        "--n_permutations", type=int, default=10,
        help="Number of permutation baseline trials (default: 10).",
    )
    parser.add_argument(
        "--min_n_per_class", type=int, default=20,
        help="Minimum items per class for binary probes (default: 20).",
    )
    parser.add_argument(
        "--skip_sae_probes", action="store_true",
        help="Skip SAE-based probes (raw-only).",
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
    if b4_complete(run_dir) and not args.force:
        log("B4 output already exists. Use --force to rerun.")
        sys.exit(0)

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

    n_permutations = args.n_permutations
    min_n = args.min_n_per_class

    log("B4: Probe Training + Controls")
    log(f"  run_dir: {run_dir}")
    log(f"  categories: {categories}")
    log(f"  layers: {len(layers)} layers")
    log(f"  n_permutations: {n_permutations}")
    log(f"  min_n_per_class: {min_n}")
    log(f"  skip_sae_probes: {args.skip_sae_probes}")

    # ── Load metadata + question_index ───────────────────────────────
    meta_df = load_metadata(run_dir)
    qi_map = load_question_index_map(run_dir, categories)
    log(f"Loaded question_index for {len(qi_map)} items")

    # ── Load ranked features for SAE probes ──────────────────────────
    ranked_df = None
    if not args.skip_sae_probes:
        ranked_path = run_dir / "B_feature_ranking" / "ranked_features.parquet"
        if ranked_path.exists():
            ranked_df = pd.read_parquet(ranked_path)
            log(f"Loaded ranked_features.parquet: {len(ranked_df)} rows")
        else:
            log("WARNING: ranked_features.parquet not found, skipping SAE probes.")

    # ── Main probe loop ──────────────────────────────────────────────
    all_results: list[dict] = []
    within_cat_records: list[dict] = []

    for cat in categories:
        log(f"\n{'=' * 60}")
        log(f"Probing category: {cat}")
        log(f"{'=' * 60}")

        # Filter metadata for this category
        cat_meta = meta_df[meta_df["category"] == cat].copy()

        # Ensure stereotyped_groups is a list
        cat_meta["stereotyped_groups"] = cat_meta["stereotyped_groups"].apply(
            lambda x: x if isinstance(x, list) else json.loads(x)
        )

        if len(cat_meta) == 0:
            log(f"  No items for category '{cat}', skipping.")
            continue

        # Load all hidden states for this category
        hs_by_layer, loaded_idxs = load_category_hidden_states_by_layer(
            run_dir, cat, cat_meta["item_idx"].tolist(), n_layers, hidden_dim,
        )

        if not loaded_idxs:
            log(f"  No activations found for '{cat}', skipping.")
            continue

        log(f"  Loaded {len(loaded_idxs)} items across {len(hs_by_layer)} layers")

        # Re-index cat_meta to match loaded order with positional index
        cat_meta = cat_meta.set_index("item_idx").loc[loaded_idxs].reset_index()

        # Build groups array (question_index for group-aware CV)
        groups = get_groups_for_items(cat_meta["item_idx"].values, qi_map)

        # Pre-filter metadata views
        ambig_mask = cat_meta["context_condition"] == "ambig"
        cat_meta_ambig = cat_meta[ambig_mask]

        # Enumerate subgroups
        subgroups = enumerate_subgroups(meta_df, cat, min_n, ambig_only=True)
        log(f"  Subgroups with ≥{min_n} items: {subgroups}")

        # ── Per-layer probes ─────────────────────────────────────────
        for layer in progress_bar(layers, desc=f"  {cat} layers"):
            if layer not in hs_by_layer:
                continue
            X_layer = hs_by_layer[layer]

            # Probe 1: multiclass subgroup
            result = probe_multiclass_subgroup(
                cat, layer, X_layer, cat_meta_ambig, groups,
                n_permutations, min_n,
            )
            if result:
                all_results.append(result)

            # Probe 2: binary subgroup detection (per subgroup)
            for sub in subgroups:
                result = probe_binary_subgroup(
                    cat, sub, layer, X_layer, cat_meta_ambig, groups,
                    n_permutations, min_n,
                )
                if result:
                    all_results.append(result)

            # Probe 3: stereotyped response binary
            result = probe_stereotyped_response(
                cat, layer, X_layer, cat_meta_ambig, groups,
                n_permutations, min_n,
            )
            if result:
                all_results.append(result)

            # Control B1: context condition (uses ALL items)
            result = probe_context_condition(
                cat, layer, X_layer, cat_meta, groups,
                n_permutations, min_n,
            )
            if result:
                all_results.append(result)

            # Control B2: template ID (stratified CV)
            result = probe_template_id(
                cat, layer, X_layer, cat_meta, groups,
            )
            if result:
                all_results.append(result)

            # Control D2: within-category cross-subgroup
            records = probe_within_cat_cross_subgroup(
                cat, layer, X_layer, cat_meta, min_n,
            )
            within_cat_records.extend(records)

        # SAE probes (layer-independent — run once per subgroup per category)
        if ranked_df is not None and not args.skip_sae_probes:
            log(f"  Running SAE probes for {cat}...")
            for sub in subgroups:
                result = probe_sae_binary_subgroup(
                    cat, sub, run_dir, cat_meta_ambig, groups,
                    ranked_df, n_permutations, min_n,
                )
                if result:
                    all_results.append(result)

        # Incremental save after each category
        save_probe_results(run_dir, all_results)
        log(f"  Incremental save: {len(all_results)} probe results so far")

        # Free memory for this category
        del hs_by_layer

    # ── Control D1: cross-category generalization ────────────────────
    log(f"\n{'=' * 60}")
    log("Cross-category generalization (D1)")
    log(f"{'=' * 60}")

    cross_cat_records: list[dict] = []
    for layer in progress_bar(layers, desc="  Cross-cat layers"):
        records = probe_cross_category(
            layer, categories, meta_df, qi_map, run_dir, hidden_dim, min_n,
        )
        cross_cat_records.extend(records)

    # ── Save final outputs ───────────────────────────────────────────
    save_probe_results(run_dir, all_results)
    save_cross_cat_results(run_dir, cross_cat_records)
    save_within_cat_results(run_dir, within_cat_records)

    elapsed = time.time() - t0

    config_info = {
        "n_components": N_COMPONENTS,
        "n_folds": N_FOLDS,
        "cv_method": "GroupKFold(question_index)",
        "template_probe_cv_method": "StratifiedKFold (non-group-aware)",
        "n_permutations": n_permutations,
        "min_n_per_class": min_n,
        "lr_C": LR_C,
        "random_seed": SEED,
        "sae_top_k": SAE_TOP_K,
        "layers_used": layers,
        "skip_sae_probes": args.skip_sae_probes,
    }

    summary = build_probes_summary(
        all_results, cross_cat_records, within_cat_records,
        categories, config_info, elapsed,
    )
    out_dir = ensure_dir(run_dir / "B_probes")
    atomic_save_json(summary, out_dir / "probes_summary.json")
    log(f"\nSaved probes_summary.json")

    # Log highlights
    for cat, info in summary.get("peak_selectivity_per_category", {}).items():
        log(f"  {cat}: peak selectivity={info['peak_selectivity']:.3f} "
            f"at L{info['peak_layer']}")

    # ── Figures ──────────────────────────────────────────────────────
    if not args.skip_figures:
        probe_df = pd.DataFrame(all_results)
        cross_df = pd.DataFrame(cross_cat_records)
        within_df = pd.DataFrame(within_cat_records)

        # Try to load B3 differentiation for within-cat peak layers
        differentiation = None
        diff_path = run_dir / "B_geometry" / "differentiation_metrics.json"
        if diff_path.exists():
            with open(diff_path) as f:
                differentiation = json.load(f)

        generate_all_figures(
            run_dir=run_dir,
            probe_df=probe_df,
            cross_df=cross_df,
            within_df=within_df,
            categories=categories,
            differentiation=differentiation,
        )

    log(f"\nB4 complete in {elapsed:.1f}s")
    log(f"  Probe results: {len(all_results)}")
    log(f"  Cross-category: {len(cross_cat_records)}")
    log(f"  Within-category: {len(within_cat_records)}")
    log(f"  Output: {run_dir / 'B_probes'}")


if __name__ == "__main__":
    main()
