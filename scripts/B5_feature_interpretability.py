"""B5: Feature Interpretability (Item-Level) — characterise top-ranked SAE features.

For each top-K feature from B2, computes activation distributions, matched-pairs
ambig/disambig comparisons, subgroup and category specificity, template-artifact
detection, cross-subgroup activation matrices, and feature co-occurrence.

Produces artifact_flags.json consumed by C1 to exclude template features from
steering candidates.

Usage:
    python scripts/B5_feature_interpretability.py --run_dir runs/llama-3.1-8b_2026-04-15/

    # Specific categories
    python scripts/B5_feature_interpretability.py --run_dir runs/... --categories so,race

    # Override top-K
    python scripts/B5_feature_interpretability.py --run_dir runs/... --top_k 30

    # Skip figures
    python scripts/B5_feature_interpretability.py --run_dir runs/... --skip_figures
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
from src.analysis.interpretability import (
    DEFAULT_TOP_K,
    HIGH_FIRING_RATE_THRESHOLD,
    LENGTH_CORRELATION_THRESHOLD,
    LOW_CATEGORY_SPECIFICITY_THRESHOLD,
    LayerCache,
    b5_complete,
    build_cross_subgroup_matrix,
    build_interpretability_summary,
    compute_activation_distribution,
    compute_category_specificity_ratio,
    compute_feature_cooccurrence,
    compute_matched_pairs_comparison,
    compute_subgroup_specificity,
    detect_template_artifacts,
    get_top_activating_items,
    load_characterization_features,
)
from src.analysis.probes import load_question_index_map
from src.data.bbq_loader import ALL_CATEGORIES
from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log, progress_bar
from src.visualization.interpretability_figures import generate_all_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="B5: Feature interpretability (item-level).",
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
        "--top_k", type=int, default=DEFAULT_TOP_K,
        help=f"Top-K features per (category, subgroup, direction) (default: {DEFAULT_TOP_K}).",
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

    # ── Resume check ─────────────────────────────────────────────────
    if b5_complete(run_dir) and not args.force:
        log("B5 output already exists. Use --force to rerun.")
        sys.exit(0)

    # ── Resolve parameters ───────────────────────────────────────────
    if args.categories is not None:
        categories = [c.strip() for c in args.categories.split(",")]
    elif "categories" in config:
        categories = config["categories"]
    else:
        categories = list(ALL_CATEGORIES)

    top_k = args.top_k

    log("B5: Feature Interpretability (Item-Level)")
    log(f"  run_dir: {run_dir}")
    log(f"  categories: {categories}")
    log(f"  top_k: {top_k}")

    # ── Load inputs ──────────────────────────────────────────────────
    metadata_df = load_metadata(run_dir)

    # Ensure stereotyped_groups is deserialized
    if len(metadata_df) > 0:
        first = metadata_df["stereotyped_groups"].iloc[0]
        if isinstance(first, str):
            metadata_df["stereotyped_groups"] = metadata_df["stereotyped_groups"].apply(
                json.loads
            )

    top_features = load_characterization_features(run_dir, top_k=top_k)
    top_features = top_features[top_features["category"].isin(categories)].copy()

    log(f"Characterising {len(top_features)} feature entries across {len(categories)} categories")

    # Load question_index mapping (for matched-pairs analysis)
    qi_map = load_question_index_map(run_dir, categories)

    # Load stimuli per category
    stimuli_by_cat: dict[str, list[dict]] = {}
    for cat in categories:
        stim_path = run_dir / "A_extraction" / "stimuli" / f"{cat}.json"
        if stim_path.exists():
            with open(stim_path) as f:
                stimuli_by_cat[cat] = json.load(f)
        else:
            log(f"  WARNING: stimuli not found for {cat}")
            stimuli_by_cat[cat] = []

    # Layer cache for SAE parquets
    layer_cache = LayerCache(run_dir)

    # ── Per-feature characterisation ─────────────────────────────────
    stats_records: list[dict] = []
    top_items_dict: dict[str, list] = {}
    artifact_list: list[dict] = []

    for feat_i, (_, feat_row) in enumerate(
        progress_bar(top_features.iterrows(), total=len(top_features),
                     desc="Characterising features")
    ):
        cat = feat_row["category"]
        sub = feat_row["subgroup"]
        direction = feat_row["direction"]
        rank = int(feat_row["rank"])
        fidx = int(feat_row["feature_idx"])
        layer = int(feat_row["layer"])

        # Feature activations across all items
        feature_activations = layer_cache.feature_activations(fidx, layer)

        # Category items from metadata
        cat_meta = metadata_df[metadata_df["category"] == cat]

        # Analysis A: activation distribution
        dist = compute_activation_distribution(feature_activations, cat_meta, cat)

        # Analysis B: matched-pairs comparison
        matched = compute_matched_pairs_comparison(
            feature_activations, cat_meta, cat, qi_map,
        )

        # Analysis D: subgroup specificity
        spec = compute_subgroup_specificity(
            feature_activations, cat_meta, cat, sub,
        )

        # Analysis F: category specificity ratio
        cat_spec = compute_category_specificity_ratio(
            feature_activations, cat, categories, metadata_df,
        )

        # Analysis G: artifact detection
        artifact = detect_template_artifacts(
            feature_activations, cat,
            cat_spec["category_specificity_ratio"],
            cat_meta, stimuli_by_cat.get(cat, []),
        )

        # Assemble flat record
        stats_records.append({
            "category": cat,
            "subgroup": sub,
            "direction": direction,
            "rank": rank,
            "feature_idx": fidx,
            "layer": layer,
            "cohens_d": float(feat_row["cohens_d"]),
            "p_value_fdr": float(feat_row["p_value_fdr"]),
            "mean_all": dist["mean_all"],
            "std_all": dist["std_all"],
            "median_all": dist["median_all"],
            "max_activation": dist["max_activation"],
            "fraction_nonzero": dist["fraction_nonzero"],
            "mean_ambig_stereo_response": dist["ambig"]["mean_stereo_response"],
            "mean_ambig_non_stereo_response": dist["ambig"]["mean_non_stereo_response"],
            "mean_disambig": dist["disambig"]["mean_all"],
            "matched_n_pairs": matched["n_pairs"],
            "matched_mean_delta": matched.get("mean_delta"),
            "matched_correlation": matched.get("pearson_correlation"),
            "subgroup_specificity": spec["subgroup_specificity"],
            "target_mean": spec.get("target_mean"),
            "category_mean": spec.get("category_mean"),
            "within_category_mean": cat_spec["within_category_mean"],
            "cross_category_mean": cat_spec["cross_category_mean"],
            "category_specificity_ratio": cat_spec["category_specificity_ratio"],
            "length_correlation": artifact["length_correlation"],
            "firing_rate_source_category": artifact["firing_rate_source_category"],
            "is_artifact_flagged": artifact["is_artifact_flagged"],
            "artifact_flags": ",".join(artifact["artifact_flags"]),
        })

        # Analysis C: top-activating items
        top_items = get_top_activating_items(
            feature_activations, cat_meta,
            stimuli_by_cat.get(cat, []), cat, top_n=20,
        )
        key = f"{cat}/{sub}/{direction}/rank{rank:03d}/L{layer}_F{fidx}"
        top_items_dict[key] = top_items

        # Track artifacts
        if artifact["is_artifact_flagged"]:
            artifact_list.append({
                "category": cat,
                "subgroup": sub,
                "direction": direction,
                "rank": rank,
                "feature_idx": fidx,
                "layer": layer,
                "flags": artifact["artifact_flags"],
                "category_specificity_ratio": cat_spec["category_specificity_ratio"],
                "length_correlation": artifact["length_correlation"],
                "firing_rate": artifact["firing_rate_source_category"],
            })

    # ── Analysis E: cross-subgroup matrices per category ─────────────
    log("\nBuilding cross-subgroup activation matrices...")
    cross_subgroup_matrices: dict[str, dict] = {}
    for cat in categories:
        result = build_cross_subgroup_matrix(
            cat, top_features, layer_cache, metadata_df,
        )
        if result is not None:
            cross_subgroup_matrices[cat] = result
            log(f"  {cat}: ARI={result['adjusted_rand_index']:.2f}, "
                f"BDS={result['block_diagonal_strength']:.1f}")

    # ── Analysis H: co-occurrence per subgroup ───────────────────────
    log("Computing feature co-occurrence...")
    cooccurrence: dict[str, dict] = {}
    for cat in categories:
        cat_subs = top_features[top_features["category"] == cat]["subgroup"].unique()
        for sub in sorted(cat_subs):
            cooccur = compute_feature_cooccurrence(
                cat, sub, top_features, layer_cache, metadata_df,
            )
            if cooccur.get("matrix") is not None:
                cooccurrence[f"{cat}/{sub}"] = cooccur

    # ── Save outputs ─────────────────────────────────────────────────
    out_dir = ensure_dir(run_dir / "B_feature_interpretability")

    # feature_stats.parquet
    stats_df = pd.DataFrame(stats_records)
    tmp = (out_dir / "feature_stats.parquet").with_suffix(".parquet.tmp")
    stats_df.to_parquet(tmp, index=False, compression="snappy")
    tmp.rename(out_dir / "feature_stats.parquet")
    log(f"Saved feature_stats.parquet: {len(stats_df)} rows")

    # top_activating_items.json
    atomic_save_json(top_items_dict, out_dir / "top_activating_items.json")
    log(f"Saved top_activating_items.json: {len(top_items_dict)} features")

    # cross_subgroup_activation_matrices.json
    atomic_save_json(cross_subgroup_matrices,
                     out_dir / "cross_subgroup_activation_matrices.json")

    # feature_cooccurrence.json
    atomic_save_json(cooccurrence, out_dir / "feature_cooccurrence.json")

    # artifact_flags.json
    artifact_output = {
        "n_flagged": len(artifact_list),
        "flagging_criteria": {
            "low_category_specificity_threshold": LOW_CATEGORY_SPECIFICITY_THRESHOLD,
            "length_correlation_threshold": LENGTH_CORRELATION_THRESHOLD,
            "high_firing_rate_threshold": HIGH_FIRING_RATE_THRESHOLD,
        },
        "flagged_features": artifact_list,
    }
    atomic_save_json(artifact_output, out_dir / "artifact_flags.json")
    log(f"Saved artifact_flags.json: {len(artifact_list)} flagged features")

    # interpretability_summary.json
    elapsed = time.time() - t0
    summary = build_interpretability_summary(
        stats_records, cross_subgroup_matrices, artifact_list,
        categories, top_k, elapsed,
    )
    atomic_save_json(summary, out_dir / "interpretability_summary.json")
    log("Saved interpretability_summary.json")

    # ── Figures ──────────────────────────────────────────────────────
    if not args.skip_figures:
        generate_all_figures(
            run_dir=run_dir,
            stats_df=stats_df,
            cross_matrices=cross_subgroup_matrices,
            cooccurrence=cooccurrence,
            categories=categories,
        )

    log(f"\nB5 complete in {elapsed:.1f}s")
    log(f"  Features characterised: {len(stats_records)}")
    log(f"  Artifact-flagged: {len(artifact_list)}")
    log(f"  Output: {out_dir}")


if __name__ == "__main__":
    main()
