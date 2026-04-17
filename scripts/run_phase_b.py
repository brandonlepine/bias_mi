"""Phase B unified runner: B1 → B2 → B3 → B4 → B5 sequentially.

Orchestrates the full analysis pipeline.  Each stage checks for existing
output and skips unless --force is passed.

Dependency graph:
    A1 → A2 → A3 ──┬── B1 → B2 ──┬── B4 (SAE probes need B2)
                    │              └── B5 (needs B2 ranked features)
                    └── B3 (reads A2 activations directly, no SAE)

All stages are CPU/numpy/sklearn — no GPU required.

Usage:
    # Full pipeline (all five stages)
    python scripts/run_phase_b.py --run_dir runs/llama-3.1-8b_2026-04-15/

    # Specific stages
    python scripts/run_phase_b.py --run_dir runs/... --stages B1,B2
    python scripts/run_phase_b.py --run_dir runs/... --stages B3
    python scripts/run_phase_b.py --run_dir runs/... --stages B4,B5

    # Quick test (subset of categories and layers)
    python scripts/run_phase_b.py --run_dir runs/... --categories so --layers 0,14,31

    # Force rerun of completed stages
    python scripts/run_phase_b.py --run_dir runs/... --force

    # Skip figures across all stages
    python scripts/run_phase_b.py --run_dir runs/... --skip_figures

    # B4-specific: reduce permutation trials, skip SAE probes
    python scripts/run_phase_b.py --run_dir runs/... --stages B4 --n_permutations 5 --skip_sae_probes
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.bbq_loader import ALL_CATEGORIES
from src.utils.config import build_provenance, load_config, save_config
from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log


# =========================================================================
# Argument parsing
# =========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase B: unified analysis pipeline (B1 → B2 → B3 → B4 → B5).",
    )
    parser.add_argument(
        "--run_dir", type=str, required=True,
        help="Path to the run directory (must contain config.json from Phase A).",
    )
    parser.add_argument(
        "--stages", type=str, default="B1,B2,B3,B4,B5",
        help="Comma-separated stages to run (default: B1,B2,B3,B4,B5).",
    )
    parser.add_argument(
        "--categories", type=str, default=None,
        help="Comma-separated category short names (default: all from config).",
    )
    parser.add_argument(
        "--layers", type=str, default=None,
        help="Comma-separated layer indices (default: all). Applies to B1, B4.",
    )
    parser.add_argument(
        "--max_items", type=int, default=None,
        help="Max items per category for B1 (quick testing).",
    )
    parser.add_argument(
        "--min_n_per_group", type=int, default=10,
        help="Minimum items per comparison group for B1/B3 (default: 10).",
    )

    # B2 options
    parser.add_argument(
        "--overlap_ks", type=str, default=None,
        help="Comma-separated k values for B2 overlap (default: 5,10,20,50,100,200).",
    )

    # B4 options
    parser.add_argument(
        "--n_permutations", type=int, default=10,
        help="Permutation baseline trials for B4 (default: 10).",
    )
    parser.add_argument(
        "--min_n_per_class", type=int, default=20,
        help="Minimum items per class for B4 probes (default: 20).",
    )
    parser.add_argument(
        "--skip_sae_probes", action="store_true",
        help="Skip SAE-based probes in B4.",
    )

    # B5 options
    parser.add_argument(
        "--top_k", type=int, default=20,
        help="Top-K features per subgroup for B5 characterisation (default: 20).",
    )

    # Global options
    parser.add_argument(
        "--skip_figures", action="store_true",
        help="Skip figure generation across all stages.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rerun stages even if output already exists.",
    )

    return parser.parse_args()


# =========================================================================
# Stage B1: Differential Feature Analysis
# =========================================================================

def run_b1(
    run_dir: Path,
    config: dict,
    categories: list[str],
    layers: list[int],
    min_n: int,
    max_items: int | None,
    force: bool,
) -> None:
    """B1: Identify bias-associated SAE features per subgroup per layer."""
    log(f"\n{'=' * 70}")
    log(f"  STAGE B1: Differential Feature Analysis")
    log(f"{'=' * 70}")
    t0 = time.time()

    from src.analysis.differential import (
        build_differential_summary,
        build_subgroup_catalog,
        load_layer_summary,
        load_metadata,
        process_layer,
        save_layer_parquet,
        save_layer_summary,
    )

    # Check if already complete (coarse: summary exists)
    summary_path = run_dir / "B_differential" / "differential_summary.json"
    if summary_path.exists() and not force:
        log("B1 output already exists. Use --force to rerun.")
        return

    meta_df = load_metadata(run_dir)
    subgroup_catalog = build_subgroup_catalog(meta_df, categories, min_n)

    n_analyzable = sum(1 for v in subgroup_catalog.values() if v["analyzable"])
    log(f"Subgroups to analyse: {n_analyzable} across {len(categories)} categories")

    all_summaries: dict[int, dict] = {}

    for layer in layers:
        out_path = run_dir / "B_differential" / f"layer_{layer:02d}.parquet"

        if out_path.exists() and not force:
            log(f"  Layer {layer:02d}: already processed, skipping")
            all_summaries[layer] = load_layer_summary(run_dir, layer)
            continue

        layer_results, layer_summary = process_layer(
            layer=layer,
            run_dir=run_dir,
            meta_df=meta_df,
            categories=categories,
            subgroup_catalog=subgroup_catalog,
            min_n=min_n,
            max_items=max_items,
        )

        save_layer_parquet(run_dir, layer, layer_results)
        save_layer_summary(run_dir, layer, layer_summary)
        all_summaries[layer] = layer_summary

    build_differential_summary(
        run_dir, layers, subgroup_catalog, all_summaries, min_n,
    )

    elapsed = time.time() - t0
    log(f"B1 complete ({elapsed:.1f}s)")


# =========================================================================
# Stage B2: Feature Ranking
# =========================================================================

def run_b2(
    run_dir: Path,
    config: dict,
    n_layers: int,
    overlap_ks: list[int] | None,
    skip_figures: bool,
    force: bool,
) -> None:
    """B2: Rank features, compute injection layers and overlap."""
    log(f"\n{'=' * 70}")
    log(f"  STAGE B2: Feature Ranking Per Subgroup")
    log(f"{'=' * 70}")
    t0 = time.time()

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

    out_dir = ensure_dir(run_dir / "B_feature_ranking")
    summary_path = out_dir / "ranking_summary.json"
    if summary_path.exists() and not force:
        log("B2 output already exists. Use --force to rerun.")
        return

    k_values = overlap_ks if overlap_ks else K_VALUES_DEFAULT

    combined = load_all_significant(run_dir, n_layers)
    combined = deduplicate_defensive(combined)

    b1_summary_path = run_dir / "B_differential" / "differential_summary.json"
    b1_summary: dict = {}
    if b1_summary_path.exists():
        with open(b1_summary_path) as f:
            b1_summary = json.load(f)

    subgroups, enum_report = enumerate_subgroups(combined, b1_summary)
    log(f"Subgroups to rank: {len(subgroups)}")

    ranked_df = rank_features_all(combined, subgroups)
    log(f"Ranked features: {len(ranked_df)} rows")

    # Save ranked features
    ranked_path = out_dir / "ranked_features.parquet"
    tmp_path = ranked_path.with_suffix(".parquet.tmp")
    ranked_df.to_parquet(tmp_path, index=False, compression="snappy")
    tmp_path.rename(ranked_path)

    injection_layers = build_injection_layers(ranked_df, subgroups)
    atomic_save_json(injection_layers, out_dir / "injection_layers.json")

    overlap_data = compute_all_overlaps(ranked_df, subgroups, k_values)
    atomic_save_json(overlap_data, out_dir / "feature_overlap.json")

    meta_df = load_metadata(run_dir)
    item_overlap = compute_item_overlap(meta_df, subgroups, 0.5)
    atomic_save_json(item_overlap, out_dir / "item_overlap_report.json")

    elapsed = time.time() - t0
    summary = build_ranking_summary(
        n_layers, subgroups, ranked_df, injection_layers,
        enum_report, k_values, 0.5, elapsed,
    )
    atomic_save_json(summary, summary_path)

    if not skip_figures:
        from src.visualization.ranking_figures import generate_all_b2_figures
        fig_dir = ensure_dir(out_dir / "figures")
        generate_all_b2_figures(
            ranked_df, subgroups, overlap_data, injection_layers,
            n_layers, fig_dir,
        )

    log(f"B2 complete ({elapsed:.1f}s)")


# =========================================================================
# Stage B3: Subgroup Direction Geometry
# =========================================================================

def run_b3(
    run_dir: Path,
    config: dict,
    categories: list[str],
    n_layers: int,
    hidden_dim: int,
    min_n: int,
    max_items: int | None,
    skip_figures: bool,
    force: bool,
) -> None:
    """B3: DIM directions, pairwise cosines, differentiation, alignment."""
    log(f"\n{'=' * 70}")
    log(f"  STAGE B3: Subgroup Direction Geometry")
    log(f"{'=' * 70}")
    t0 = time.time()

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
    from src.visualization.geometry_figures import generate_all_figures

    if b3_complete(run_dir) and not force:
        log("B3 output already exists. Use --force to rerun.")
        return

    meta_df = load_metadata(run_dir)

    directions_arrays: dict[str, "np.ndarray"] = {}
    directions_norms: dict[str, "np.ndarray"] = {}
    subgroup_info: dict[tuple[str, str], dict] = {}

    for cat in categories:
        process_category(
            cat=cat, run_dir=run_dir, meta_df=meta_df,
            n_layers=n_layers, hidden_dim=hidden_dim,
            min_n=min_n, max_items=max_items,
            directions_arrays=directions_arrays,
            directions_norms=directions_norms,
            subgroup_info=subgroup_info,
        )

    if not directions_arrays:
        log("WARNING: No directions computed. Skipping B3 outputs.")
        return

    save_directions(run_dir, directions_arrays, directions_norms)

    cosine_df = compute_all_cosines(directions_arrays, categories, n_layers)
    save_cosines(run_dir, cosine_df)

    differentiation = compute_differentiation_metrics(cosine_df, categories, n_layers)
    out_dir = ensure_dir(run_dir / "B_geometry")
    atomic_save_json(differentiation, out_dir / "differentiation_metrics.json")

    alignment = compute_alignment(directions_arrays, categories, n_layers)
    atomic_save_json(alignment, out_dir / "bias_identity_alignment.json")

    summary = build_summary(subgroup_info, directions_norms, categories, min_n)
    atomic_save_json(summary, out_dir / "subgroup_directions_summary.json")

    if not skip_figures:
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
    log(f"B3 complete ({elapsed:.1f}s)")


# =========================================================================
# Stage B4: Probe Training + Controls
# =========================================================================

def run_b4(
    run_dir: Path,
    config: dict,
    categories: list[str],
    layers: list[int],
    n_layers: int,
    hidden_dim: int,
    n_permutations: int,
    min_n_per_class: int,
    skip_sae_probes: bool,
    skip_figures: bool,
    force: bool,
) -> None:
    """B4: Linear probes on hidden states and SAE features."""
    log(f"\n{'=' * 70}")
    log(f"  STAGE B4: Probe Training + Controls")
    log(f"{'=' * 70}")
    t0 = time.time()

    import pandas as pd

    from src.analysis.differential import load_metadata
    from src.analysis.probes import (
        SEED,
        N_COMPONENTS,
        N_FOLDS,
        LR_C,
        SAE_TOP_K,
        b4_complete,
        build_probes_summary,
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
    from src.utils.logging import progress_bar
    from src.visualization.probe_figures import generate_all_figures

    if b4_complete(run_dir) and not force:
        log("B4 output already exists. Use --force to rerun.")
        return

    meta_df = load_metadata(run_dir)
    qi_map = load_question_index_map(run_dir, categories)
    log(f"Loaded question_index for {len(qi_map)} items")

    ranked_df = None
    if not skip_sae_probes:
        ranked_path = run_dir / "B_feature_ranking" / "ranked_features.parquet"
        if ranked_path.exists():
            ranked_df = pd.read_parquet(ranked_path)
            log(f"Loaded ranked_features.parquet: {len(ranked_df)} rows")
        else:
            log("WARNING: ranked_features.parquet not found, skipping SAE probes.")

    all_results: list[dict] = []
    within_cat_records: list[dict] = []

    for cat in categories:
        log(f"\n  Probing category: {cat}")

        cat_meta = meta_df[meta_df["category"] == cat].copy()
        cat_meta["stereotyped_groups"] = cat_meta["stereotyped_groups"].apply(
            lambda x: x if isinstance(x, list) else json.loads(x)
        )
        if len(cat_meta) == 0:
            continue

        hs_by_layer, loaded_idxs = load_category_hidden_states_by_layer(
            run_dir, cat, cat_meta["item_idx"].tolist(), n_layers, hidden_dim,
        )
        if not loaded_idxs:
            continue

        cat_meta = cat_meta.set_index("item_idx").loc[loaded_idxs].reset_index()
        groups = get_groups_for_items(cat_meta["item_idx"].values, qi_map)

        ambig_mask = cat_meta["context_condition"] == "ambig"
        cat_meta_ambig = cat_meta[ambig_mask]

        subgroups = enumerate_subgroups(meta_df, cat, min_n_per_class, ambig_only=True)

        for layer in progress_bar(layers, desc=f"    {cat} layers"):
            if layer not in hs_by_layer:
                continue
            X_layer = hs_by_layer[layer]

            result = probe_multiclass_subgroup(
                cat, layer, X_layer, cat_meta_ambig, groups,
                n_permutations, min_n_per_class,
            )
            if result:
                all_results.append(result)

            for sub in subgroups:
                result = probe_binary_subgroup(
                    cat, sub, layer, X_layer, cat_meta_ambig, groups,
                    n_permutations, min_n_per_class,
                )
                if result:
                    all_results.append(result)

            result = probe_stereotyped_response(
                cat, layer, X_layer, cat_meta_ambig, groups,
                n_permutations, min_n_per_class,
            )
            if result:
                all_results.append(result)

            result = probe_context_condition(
                cat, layer, X_layer, cat_meta, groups,
                n_permutations, min_n_per_class,
            )
            if result:
                all_results.append(result)

            result = probe_template_id(cat, layer, X_layer, cat_meta, groups)
            if result:
                all_results.append(result)

            records = probe_within_cat_cross_subgroup(
                cat, layer, X_layer, cat_meta, min_n_per_class,
            )
            within_cat_records.extend(records)

        if ranked_df is not None and not skip_sae_probes:
            for sub in subgroups:
                result = probe_sae_binary_subgroup(
                    cat, sub, run_dir, cat_meta_ambig, groups,
                    ranked_df, n_permutations, min_n_per_class,
                )
                if result:
                    all_results.append(result)

        save_probe_results(run_dir, all_results)
        del hs_by_layer

    # Cross-category generalization
    log("\n  Cross-category generalization (D1)")
    cross_cat_records: list[dict] = []
    for layer in progress_bar(layers, desc="    Cross-cat layers"):
        records = probe_cross_category(
            layer, categories, meta_df, qi_map, run_dir, hidden_dim, min_n_per_class,
        )
        cross_cat_records.extend(records)

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
        "min_n_per_class": min_n_per_class,
        "lr_C": LR_C,
        "random_seed": SEED,
        "sae_top_k": SAE_TOP_K,
        "layers_used": layers,
        "skip_sae_probes": skip_sae_probes,
    }

    summary = build_probes_summary(
        all_results, cross_cat_records, within_cat_records,
        categories, config_info, elapsed,
    )
    probe_dir = ensure_dir(run_dir / "B_probes")
    atomic_save_json(summary, probe_dir / "probes_summary.json")

    if not skip_figures:
        probe_df = pd.DataFrame(all_results)
        cross_df = pd.DataFrame(cross_cat_records)
        within_df = pd.DataFrame(within_cat_records)

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

    log(f"B4 complete ({elapsed:.1f}s)")


# =========================================================================
# Stage B5: Feature Interpretability
# =========================================================================

def run_b5(
    run_dir: Path,
    config: dict,
    categories: list[str],
    top_k: int,
    skip_figures: bool,
    force: bool,
) -> None:
    """B5: Item-level feature characterisation and artifact detection."""
    log(f"\n{'=' * 70}")
    log(f"  STAGE B5: Feature Interpretability (Item-Level)")
    log(f"{'=' * 70}")
    t0 = time.time()

    import pandas as pd

    from src.analysis.differential import load_metadata
    from src.analysis.interpretability import (
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
    from src.utils.logging import progress_bar
    from src.visualization.interpretability_figures import generate_all_figures

    if b5_complete(run_dir) and not force:
        log("B5 output already exists. Use --force to rerun.")
        return

    metadata_df = load_metadata(run_dir)

    # Ensure stereotyped_groups deserialized
    if metadata_df["stereotyped_groups"].dtype == object:
        first = metadata_df["stereotyped_groups"].iloc[0] if len(metadata_df) else "[]"
        if isinstance(first, str):
            metadata_df["stereotyped_groups"] = metadata_df["stereotyped_groups"].apply(
                json.loads
            )

    top_features = load_characterization_features(run_dir, top_k=top_k)
    top_features = top_features[top_features["category"].isin(categories)].copy()
    log(f"Characterising {len(top_features)} feature entries")

    qi_map = load_question_index_map(run_dir, categories)

    stimuli_by_cat: dict[str, list[dict]] = {}
    for cat in categories:
        stim_path = run_dir / "A_extraction" / "stimuli" / f"{cat}.json"
        if stim_path.exists():
            with open(stim_path) as f:
                stimuli_by_cat[cat] = json.load(f)
        else:
            stimuli_by_cat[cat] = []

    layer_cache = LayerCache(run_dir)

    stats_records: list[dict] = []
    top_items_dict: dict[str, list] = {}
    artifact_list: list[dict] = []

    for _, feat_row in progress_bar(
        top_features.iterrows(), total=len(top_features),
        desc="  Characterising features",
    ):
        cat = feat_row["category"]
        sub = feat_row["subgroup"]
        direction = feat_row["direction"]
        rank = int(feat_row["rank"])
        fidx = int(feat_row["feature_idx"])
        layer = int(feat_row["layer"])

        feature_activations = layer_cache.feature_activations(fidx, layer)
        cat_meta = metadata_df[metadata_df["category"] == cat]

        dist = compute_activation_distribution(feature_activations, cat_meta, cat)
        matched = compute_matched_pairs_comparison(
            feature_activations, cat_meta, cat, qi_map,
        )
        spec = compute_subgroup_specificity(feature_activations, cat_meta, cat, sub)
        cat_spec = compute_category_specificity_ratio(
            feature_activations, cat, categories, metadata_df,
        )
        artifact = detect_template_artifacts(
            feature_activations, cat,
            cat_spec["category_specificity_ratio"],
            cat_meta, stimuli_by_cat.get(cat, []),
        )

        stats_records.append({
            "category": cat, "subgroup": sub, "direction": direction,
            "rank": rank, "feature_idx": fidx, "layer": layer,
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

        top_items = get_top_activating_items(
            feature_activations, cat_meta,
            stimuli_by_cat.get(cat, []), cat, top_n=20,
        )
        key = f"{cat}/{sub}/{direction}/rank{rank:03d}/L{layer}_F{fidx}"
        top_items_dict[key] = top_items

        if artifact["is_artifact_flagged"]:
            artifact_list.append({
                "category": cat, "subgroup": sub, "direction": direction,
                "rank": rank, "feature_idx": fidx, "layer": layer,
                "flags": artifact["artifact_flags"],
                "category_specificity_ratio": cat_spec["category_specificity_ratio"],
                "length_correlation": artifact["length_correlation"],
                "firing_rate": artifact["firing_rate_source_category"],
            })

    log("\n  Building cross-subgroup matrices...")
    cross_subgroup_matrices: dict[str, dict] = {}
    for cat in categories:
        result = build_cross_subgroup_matrix(
            cat, top_features, layer_cache, metadata_df,
        )
        if result is not None:
            cross_subgroup_matrices[cat] = result

    log("  Computing feature co-occurrence...")
    cooccurrence: dict[str, dict] = {}
    for cat in categories:
        cat_subs = top_features[top_features["category"] == cat]["subgroup"].unique()
        for sub in sorted(cat_subs):
            cooccur = compute_feature_cooccurrence(
                cat, sub, top_features, layer_cache, metadata_df,
            )
            if cooccur.get("matrix") is not None:
                cooccurrence[f"{cat}/{sub}"] = cooccur

    out_dir = ensure_dir(run_dir / "B_feature_interpretability")

    stats_df = pd.DataFrame(stats_records)
    tmp = (out_dir / "feature_stats.parquet").with_suffix(".parquet.tmp")
    stats_df.to_parquet(tmp, index=False, compression="snappy")
    tmp.rename(out_dir / "feature_stats.parquet")

    atomic_save_json(top_items_dict, out_dir / "top_activating_items.json")
    atomic_save_json(cross_subgroup_matrices,
                     out_dir / "cross_subgroup_activation_matrices.json")
    atomic_save_json(cooccurrence, out_dir / "feature_cooccurrence.json")

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

    elapsed = time.time() - t0
    summary = build_interpretability_summary(
        stats_records, cross_subgroup_matrices, artifact_list,
        categories, top_k, elapsed,
    )
    atomic_save_json(summary, out_dir / "interpretability_summary.json")

    if not skip_figures:
        generate_all_figures(
            run_dir=run_dir,
            stats_df=stats_df,
            cross_matrices=cross_subgroup_matrices,
            cooccurrence=cooccurrence,
            categories=categories,
        )

    log(f"B5 complete ({elapsed:.1f}s)")


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    args = parse_args()
    stages = [s.strip().upper() for s in args.stages.split(",")]
    valid_stages = {"B1", "B2", "B3", "B4", "B5"}
    invalid = set(stages) - valid_stages
    if invalid:
        log(f"ERROR: invalid stages: {invalid}. Valid: {sorted(valid_stages)}")
        sys.exit(1)

    run_dir = Path(args.run_dir)

    # ── Load config ──────────────────────────────────────────────────
    config_path = run_dir / "config.json"
    if not config_path.exists():
        log(f"ERROR: config.json not found at {config_path}")
        log("Run Phase A first to create the run directory.")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    n_layers = config.get("n_layers")
    hidden_dim = config.get("hidden_dim")
    if n_layers is None or hidden_dim is None:
        log("ERROR: config.json missing n_layers or hidden_dim. Run A2 first.")
        sys.exit(1)

    # ── Resolve categories ───────────────────────────────────────────
    if args.categories is not None:
        categories = [c.strip() for c in args.categories.split(",")]
    elif "categories" in config:
        categories = config["categories"]
    else:
        categories = list(ALL_CATEGORIES)

    # ── Resolve layers ───────────────────────────────────────────────
    if args.layers is not None:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        layers = list(range(n_layers))

    # ── Dependency checks ────────────────────────────────────────────
    if "B2" in stages and "B1" not in stages:
        b1_summary = run_dir / "B_differential" / "differential_summary.json"
        if not b1_summary.exists():
            log("ERROR: B2 requires B1 output. Run B1 first or include B1 in --stages.")
            sys.exit(1)

    if "B4" in stages and not args.skip_sae_probes:
        ranked_path = run_dir / "B_feature_ranking" / "ranked_features.parquet"
        if not ranked_path.exists() and "B2" not in stages:
            log("WARNING: B4 SAE probes need B2 output. Will skip SAE probes.")

    if "B5" in stages:
        ranked_path = run_dir / "B_feature_ranking" / "ranked_features.parquet"
        if not ranked_path.exists() and "B2" not in stages:
            log("ERROR: B5 requires B2 output. Run B2 first or include B2 in --stages.")
            sys.exit(1)

    # ── Provenance ───────────────────────────────────────────────────
    provenance = build_provenance(config, "cpu", stages)
    provenance_path = run_dir / "provenance.json"
    existing_provenance: list = []
    if provenance_path.exists():
        try:
            with open(provenance_path) as f:
                existing_provenance = json.load(f)
        except (json.JSONDecodeError, TypeError):
            existing_provenance = []
    existing_provenance.append(provenance)
    atomic_save_json(existing_provenance, provenance_path)

    # ── Banner ───────────────────────────────────────────────────────
    log(f"\n{'=' * 70}")
    log(f"  Phase B Pipeline")
    log(f"{'=' * 70}")
    log(f"  Run dir:    {run_dir}")
    log(f"  Stages:     {' → '.join(stages)}")
    log(f"  Categories: {categories}")
    log(f"  Layers:     {len(layers)} layers")
    if args.max_items:
        log(f"  Max items:  {args.max_items}")
    log(f"{'=' * 70}")

    pipeline_t0 = time.time()

    # ── Execute stages ───────────────────────────────────────────────
    if "B1" in stages:
        run_b1(
            run_dir, config, categories, layers,
            args.min_n_per_group, args.max_items, args.force,
        )

    if "B2" in stages:
        overlap_ks = (
            [int(x) for x in args.overlap_ks.split(",")]
            if args.overlap_ks else None
        )
        run_b2(
            run_dir, config, n_layers,
            overlap_ks, args.skip_figures, args.force,
        )

    if "B3" in stages:
        run_b3(
            run_dir, config, categories, n_layers, hidden_dim,
            args.min_n_per_group, args.max_items,
            args.skip_figures, args.force,
        )

    if "B4" in stages:
        run_b4(
            run_dir, config, categories, layers, n_layers, hidden_dim,
            args.n_permutations, args.min_n_per_class,
            args.skip_sae_probes, args.skip_figures, args.force,
        )

    if "B5" in stages:
        run_b5(
            run_dir, config, categories, args.top_k,
            args.skip_figures, args.force,
        )

    total_elapsed = time.time() - pipeline_t0
    log(f"\n{'=' * 70}")
    log(f"  Phase B complete — {total_elapsed:.1f}s total")
    log(f"{'=' * 70}")


if __name__ == "__main__":
    main()
