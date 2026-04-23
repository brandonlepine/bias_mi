"""Phase C unified runner: C1 → C2 → C3 → C4 sequentially.

Orchestrates the full steering & evaluation pipeline.  Each stage checks for
existing output and skips unless --force is passed.  Model and SAEs are loaded
once and shared across stages that need them.

Dependency graph:
    B2 (ranked features) ──→ C1 (steering optimisation)
    B3 (cosine geometry) ──→ C2 (universal backfire — needs C1 vectors + B3 cosines)
    C1 (steering vectors) ──→ C3 (generalisation)
    B2 + C1 ────────────────→ C4 (token-level — needs optimal features + model)

C1 must run first.  C2, C3, C4 can run in any order after C1 (all need the
model but read different Phase B outputs).

Usage:
    # Full pipeline
    python scripts/run_phase_c.py --run_dir runs/llama-3.1-8b_2026-04-15/

    # Specific stages
    python scripts/run_phase_c.py --run_dir runs/... --stages C1,C2
    python scripts/run_phase_c.py --run_dir runs/... --stages C3,C4

    # Quick test
    python scripts/run_phase_c.py --run_dir runs/... --max_items 20

    # Full run with custom norms, skip exacerbation
    python scripts/run_phase_c.py --run_dir runs/... --target_norms -0.5,-1,-2,-5,-10,-20,-40 --skip_exacerbation

    # Force rerun of completed stages
    python scripts/run_phase_c.py --run_dir runs/... --force

    # Skip figures across all stages
    python scripts/run_phase_c.py --run_dir runs/... --skip_figures
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import build_provenance, load_config
from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log


# =========================================================================
# Argument parsing
# =========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase C: unified steering & evaluation pipeline "
                    "(C1 → C2 → C3 → C4).",
    )
    p.add_argument(
        "--run_dir", type=str, required=True,
        help="Path to the run directory (must contain config.json and Phase B outputs).",
    )
    p.add_argument(
        "--stages", type=str, default="C1,C2,C3,C4",
        help="Comma-separated stages to run (default: C1,C2,C3,C4).",
    )
    p.add_argument(
        "--categories", type=str, default=None,
        help="Comma-separated category short names (default: all from config).",
    )

    # ── Global test / control ────────────────────────────────────────
    p.add_argument(
        "--max_items", type=int, default=None,
        help="Max items per subgroup/target (quick test). Applies to C1, C2, C4.",
    )
    p.add_argument(
        "--skip_exacerbation", action="store_true",
        help="Skip exacerbation tests in C1 and C3.",
    )
    p.add_argument(
        "--skip_figures", action="store_true",
        help="Skip figure generation across all stages.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Rerun stages even if output already exists.",
    )

    # ── C1-specific ──────────────────────────────────────────────────
    p.add_argument(
        "--subgroups", type=str, default=None,
        help="Comma-separated subgroup filter for C1 (format: cat/sub).",
    )
    p.add_argument(
        "--min_n_per_group", type=int, default=10,
        help="C1: minimum items per stereotype/non-stereotype group (default: 10).",
    )
    p.add_argument(
        "--target_norms", type=str, default=None,
        help="C1: comma-separated target norm values (e.g. -0.5,-1,-2,-5,-10,-20,-40,-80).",
    )
    p.add_argument(
        "--injection_layer_min", type=int, default=10,
        help="C1: minimum layer for steering features (inclusive, default: 10).",
    )
    p.add_argument(
        "--injection_layer_max", type=int, default=24,
        help="C1: maximum layer for steering features (inclusive, default: 24).",
    )
    p.add_argument(
        "--min_n_eligible", type=int, default=10,
        help="C1: minimum items with margin >= tau for RCR to be meaningful.",
    )
    p.add_argument(
        "--max_degeneration_rate", type=float, default=0.05,
        help="C1: maximum degeneration rate for a config to be safe.",
    )
    p.add_argument(
        "--max_corruption_rate", type=float, default=0.15,
        help="C1: maximum corruption rate (relaxed tier).",
    )
    p.add_argument(
        "--max_corruption_rate_strict", type=float, default=0.05,
        help="C1: maximum corruption rate (strict tier).",
    )
    p.add_argument(
        "--optimizer_metric", type=str, default="mwcs_1.0",
        choices=["mwcs_0.5", "mwcs_1.0", "mwcs_2.0",
                 "rcr_0.5", "rcr_1.0", "rcr_2.0"],
        help="C1: metric the optimizer maximises (numerator of eta).",
    )
    p.add_argument(
        "--mwcs_floor", type=float, default=0.05,
        help="C1: minimum metric value for Phase 1 viability.",
    )

    # ── C2-specific ──────────────────────────────────────────────────
    p.add_argument(
        "--min_n_per_target", type=int, default=10,
        help="C2: minimum items per target subgroup (default: 10).",
    )
    p.add_argument(
        "--primary_only", action="store_true",
        help="C2: skip stable-range sensitivity analysis.",
    )

    # ── C3-specific ──────────────────────────────────────────────────
    p.add_argument(
        "--max_medqa_items", type=int, default=None,
        help="C3: max MedQA items.",
    )
    p.add_argument(
        "--max_mmlu_items", type=int, default=None,
        help="C3: max MMLU items.",
    )
    p.add_argument(
        "--datasets", type=str, default=None,
        help="C3: comma-separated dataset filter (medqa,mmlu).",
    )
    p.add_argument(
        "--c3_subgroups", type=str, default=None,
        help="C3: comma-separated subgroup filter for generalization vectors.",
    )

    # ── C4-specific ──────────────────────────────────────────────────
    p.add_argument(
        "--max_features_per_subgroup", type=int, default=10,
        help="C4: max features to analyse per subgroup (default: 10).",
    )
    p.add_argument(
        "--save_full_per_token", action="store_true",
        help="C4: save full per-item per-token activation parquet (LARGE).",
    )

    return p.parse_args()


# =========================================================================
# Completion checks
# =========================================================================

def c1_complete(run_dir: Path) -> bool:
    return (run_dir / "C_steering" / "steering_manifests.json").exists()


def c2_complete(run_dir: Path) -> bool:
    return (run_dir / "C_transfer" / "c2_summary.json").exists()


def c3_complete(run_dir: Path) -> bool:
    return (run_dir / "C_generalization" / "manifests_with_generalization.json").exists()


def c4_complete(run_dir: Path) -> bool:
    return (run_dir / "C_token_features" / "feature_interpretability.parquet").exists()


# =========================================================================
# Shared resource loading
# =========================================================================

def load_model_and_saes(
    config: dict,
    run_dir: Path,
    stages: list[str],
) -> tuple:
    """Load model and SAE cache once, shared across all stages.

    Returns (wrapper, sae_cache).  SAE layers are loaded lazily per-stage,
    but the model is loaded once here.
    """
    import torch
    from src.models.wrapper import ModelWrapper

    device = torch.device(config["device"])
    log("\nLoading model...")
    wrapper = ModelWrapper.from_pretrained(config["model_path"], device=str(device))

    # SAE cache starts empty — stages populate it as needed.
    # Once a layer is loaded it persists for subsequent stages.
    sae_cache: dict[int, object] = {}

    return wrapper, sae_cache


def ensure_sae_layer(
    sae_cache: dict,
    layer: int,
    config: dict,
) -> None:
    """Load an SAE layer into the cache if not already present."""
    if layer in sae_cache:
        return
    from src.sae.wrapper import SAEWrapper
    log(f"  Loading SAE for layer {layer}...")
    sae_cache[layer] = SAEWrapper(
        config["sae_source"],
        layer=layer,
        expansion=config.get("sae_expansion", 32),
        device=config["device"],
    )


# =========================================================================
# Stage C1: Subgroup-Specific Steering Optimisation
# =========================================================================

def run_c1(
    run_dir: Path,
    config: dict,
    wrapper: object,
    sae_cache: dict,
    args: argparse.Namespace,
) -> None:
    """C1: Joint (k, target_norm) steering optimisation per subgroup."""
    log(f"\n{'=' * 70}")
    log(f"  STAGE C1: Subgroup-Specific Steering Optimisation")
    log(f"{'=' * 70}")

    if c1_complete(run_dir) and not args.force:
        log("C1 output already exists. Use --force to rerun.")
        return

    t0 = time.time()

    import numpy as np
    import pandas as pd
    import torch

    from scripts.C1_steering import (
        DEFAULT_TARGET_NORMS,
        build_k_steps,
        build_subgroup_steering_vector,
        compute_baselines,
        compute_marginal_analysis,
        determine_subgroups,
        filter_flagged_features,
        get_ranked_s_marking_features,
        identify_needed_layers,
        identify_viable_target_norms,
        load_artifact_flags,
        load_injection_layers,
        load_metadata,
        load_ranked_features,
        load_stimuli,
        partition_items,
        process_subgroup,
        save_top_level_outputs,
        select_optimal_tiered,
    )
    from src.sae_localization.steering import SAESteerer

    # Parse target norms
    if args.target_norms:
        target_norms = [float(x) for x in args.target_norms.split(",")]
    else:
        target_norms = DEFAULT_TARGET_NORMS

    log(f"  target_norms: {target_norms}")
    if args.max_items:
        log(f"  max_items: {args.max_items}")

    # Load B2 outputs
    ranked_df = load_ranked_features(run_dir)
    injection_layers = load_injection_layers(run_dir)

    # Load B5 artifact flags
    flagged_set = load_artifact_flags(run_dir)
    ranked_df = filter_flagged_features(ranked_df, flagged_set)

    # Load metadata
    metadata_df = load_metadata(run_dir)

    # Determine subgroups
    subgroups_to_process = determine_subgroups(
        ranked_df, injection_layers,
        filter_categories=args.categories,
        filter_subgroups=args.subgroups,
    )

    # Pre-load SAE layers for the full injection layer range
    layer_min = args.injection_layer_min
    layer_max = args.injection_layer_max
    for layer in range(layer_min, layer_max + 1):
        ensure_sae_layer(sae_cache, layer, config)

    # Output directories
    output_dir = run_dir / "C_steering"
    ensure_dir(output_dir / "checkpoints")
    ensure_dir(output_dir / "vectors")
    ensure_dir(output_dir / "per_item")
    ensure_dir(output_dir / "figures")

    # Process each subgroup
    all_manifests: list[dict] = []
    all_phase1_results: dict = {}
    all_grid_records: list[dict] = []

    for cat, sub in subgroups_to_process:
        manifest = process_subgroup(
            cat=cat, sub=sub,
            ranked_df=ranked_df,
            injection_layers=injection_layers,
            metadata_df=metadata_df,
            wrapper=wrapper,
            sae_cache=sae_cache,
            run_dir=run_dir,
            output_dir=output_dir,
            config=config,
            args=args,
            target_norms=target_norms,
        )

        all_manifests.append(manifest)
        if "phase1_results" in manifest:
            all_phase1_results[f"{cat}/{sub}"] = manifest.pop("phase1_results")
        if "phase2_grid" in manifest:
            all_grid_records.extend(manifest.pop("phase2_grid"))

    # Save outputs
    runtime = time.time() - t0
    save_top_level_outputs(
        output_dir, all_manifests, all_phase1_results,
        all_grid_records, runtime,
    )

    # Figures
    if not args.skip_figures:
        try:
            from src.visualization.steering_figures import generate_c1_figures
            generate_c1_figures(output_dir, all_manifests, all_grid_records)
        except Exception as e:
            log(f"WARNING: C1 figure generation failed: {e}")

    log(f"C1 complete ({runtime:.1f}s)")


# =========================================================================
# Stage C2: Cross-Subgroup Transfer & Universal Backfire
# =========================================================================

def run_c2(
    run_dir: Path,
    config: dict,
    wrapper: object,
    sae_cache: dict,
    args: argparse.Namespace,
) -> None:
    """C2: Cross-subgroup transfer evaluation and backfire analysis."""
    log(f"\n{'=' * 70}")
    log(f"  STAGE C2: Cross-Subgroup Transfer & Universal Backfire")
    log(f"{'=' * 70}")

    if c2_complete(run_dir) and not args.force:
        log("C2 output already exists. Use --force to rerun.")
        return

    t0 = time.time()

    import numpy as np
    import pandas as pd
    import torch

    from scripts.C2_transfer import (
        build_scatter_data,
        compare_sae_vs_dim_cosines,
        get_primary_cosines,
        get_sae_cosines,
        load_metadata,
        load_viable_manifests,
        run_regression_analyses,
        run_transfer_evaluation,
        save_all_outputs,
        stable_range_sensitivity,
    )

    device = torch.device(config["device"])
    dtype = getattr(torch, config.get("dtype", "float16"))

    # Load viable manifests
    viable_manifests = load_viable_manifests(run_dir, args.categories)

    if not viable_manifests:
        log("No viable subgroups found; skipping C2")
        return

    # Pre-load SAE layers
    for m in viable_manifests:
        vec_path = run_dir / "C_steering" / "vectors" / f"{m['category']}_{m['subgroup']}.npz"
        if vec_path.exists():
            data = np.load(vec_path)
            ensure_sae_layer(sae_cache, int(data["injection_layer"]), config)

    # Output directories
    output_dir = run_dir / "C_transfer"
    ensure_dir(output_dir / "per_pair_checkpoints")
    ensure_dir(output_dir / "per_pair")
    ensure_dir(output_dir / "figures")

    metadata_df = load_metadata(run_dir)

    # Build C2 args namespace to pass through
    c2_args = argparse.Namespace(
        max_items=args.max_items,
        min_n_per_target=args.min_n_per_target,
        primary_only=args.primary_only,
    )

    # Steps 2-3: Cosines
    log("\n  Computing cosines...")
    primary_cosines = get_primary_cosines(run_dir, viable_manifests)
    sae_cosines = get_sae_cosines(run_dir, viable_manifests, device, dtype)

    # Step 4: Transfer evaluation
    log("\n  Running transfer evaluation...")
    transfer_df = run_transfer_evaluation(
        viable_manifests, metadata_df, run_dir, output_dir,
        wrapper, sae_cache, device, dtype, c2_args,
    )

    if transfer_df.empty:
        log("WARNING: No transfer pairs evaluated")
        runtime = time.time() - t0
        save_all_outputs(
            output_dir, transfer_df, primary_cosines, sae_cosines,
            pd.DataFrame(), {}, {}, {}, viable_manifests, runtime,
        )
        return

    # Steps 5-8: Analysis
    scatter_df = build_scatter_data(transfer_df, primary_cosines, sae_cosines)
    regression_results = run_regression_analyses(scatter_df)
    sae_dim_comparison = compare_sae_vs_dim_cosines(scatter_df)

    stable_range_results = {}
    if not args.primary_only:
        stable_range_results = stable_range_sensitivity(run_dir, transfer_df)

    runtime = time.time() - t0
    save_all_outputs(
        output_dir, transfer_df, primary_cosines, sae_cosines,
        scatter_df, regression_results, sae_dim_comparison,
        stable_range_results, viable_manifests, runtime,
    )

    if not args.skip_figures:
        try:
            from src.visualization.transfer_figures import generate_c2_figures
            generate_c2_figures(
                output_dir, scatter_df, regression_results,
                sae_dim_comparison, stable_range_results,
                transfer_df, viable_manifests,
            )
        except Exception as e:
            log(f"WARNING: C2 figure generation failed: {e}")

    log(f"C2 complete ({runtime:.1f}s)")


# =========================================================================
# Stage C3: Generalisation Evaluation
# =========================================================================

def run_c3(
    run_dir: Path,
    config: dict,
    wrapper: object,
    sae_cache: dict,
    args: argparse.Namespace,
) -> None:
    """C3: MedQA + MMLU generalisation evaluation."""
    log(f"\n{'=' * 70}")
    log(f"  STAGE C3: Generalisation Evaluation (MedQA + MMLU)")
    log(f"{'=' * 70}")

    if c3_complete(run_dir) and not args.force:
        log("C3 output already exists. Use --force to rerun.")
        return

    t0 = time.time()

    import torch
    import pandas as pd

    from scripts.C3_generalization import (
        aggregate_medqa,
        aggregate_mmlu,
        annotate_mmlu_supercategories,
        classify_medqa_demographics,
        compute_or_load_baselines,
        identify_top_impact_items,
        load_viable_vectors,
        run_dataset_evaluation,
        update_manifests,
    )
    from src.data.medqa_loader import load_medqa_items
    from src.data.mmlu_loader import load_mmlu_items

    device = torch.device(config["device"])
    dtype = getattr(torch, config.get("dtype", "float16"))

    # Load vectors
    subgroup_filter = args.c3_subgroups or args.subgroups
    viable_vectors = load_viable_vectors(run_dir, device, dtype, subgroup_filter)

    if not viable_vectors:
        log("No viable vectors; skipping C3")
        return

    # Pre-load SAE layers
    for v in viable_vectors:
        ensure_sae_layer(sae_cache, v["injection_layer"], config)

    output_dir = run_dir / "C_generalization"
    ensure_dir(output_dir / "baselines")
    ensure_dir(output_dir / "medqa" / "per_vector_checkpoints")
    ensure_dir(output_dir / "mmlu" / "per_vector_checkpoints")
    ensure_dir(output_dir / "figures")

    datasets_to_run = args.datasets.split(",") if args.datasets else ["medqa", "mmlu"]

    # Load datasets
    medqa_items = None
    mmlu_items = None

    if "medqa" in datasets_to_run:
        try:
            medqa_path = config.get("medqa_path", "datasets/medqa")
            medqa_items = load_medqa_items(medqa_path)
            if args.max_medqa_items:
                medqa_items = medqa_items[:args.max_medqa_items]
            log(f"Loaded {len(medqa_items)} MedQA items")
            medqa_items = classify_medqa_demographics(medqa_items, output_dir)
        except Exception as e:
            log(f"WARNING: MedQA loading failed: {e}")

    if "mmlu" in datasets_to_run:
        try:
            mmlu_path = config.get("mmlu_path", "datasets/mmlu")
            mmlu_items = load_mmlu_items(mmlu_path)
            if args.max_mmlu_items:
                mmlu_items = mmlu_items[:args.max_mmlu_items]
            log(f"Loaded {len(mmlu_items)} MMLU items")
            mmlu_items = annotate_mmlu_supercategories(mmlu_items)
        except Exception as e:
            log(f"WARNING: MMLU loading failed: {e}")

    medqa_letters = ("A", "B", "C", "D", "E")
    mmlu_letters = ("A", "B", "C", "D")

    # Baselines
    medqa_baselines_df = None
    mmlu_baselines_df = None

    if medqa_items:
        log("\n  Computing MedQA baselines...")
        medqa_baselines_df = compute_or_load_baselines(
            medqa_items, wrapper, sae_cache, medqa_letters,
            output_dir / "baselines" / "medqa_baselines.parquet", "medqa",
        )
    if mmlu_items:
        log("\n  Computing MMLU baselines...")
        mmlu_baselines_df = compute_or_load_baselines(
            mmlu_items, wrapper, sae_cache, mmlu_letters,
            output_dir / "baselines" / "mmlu_baselines.parquet", "mmlu",
        )

    # Evaluate
    medqa_records: list[dict] = []
    mmlu_records: list[dict] = []

    log("\n  Evaluating steering vectors...")
    for vi, vec_info in enumerate(viable_vectors):
        log(f"\n  Vector {vi + 1}/{len(viable_vectors)}: "
            f"{vec_info['category']}/{vec_info['subgroup']}")

        if medqa_items is not None and medqa_baselines_df is not None:
            records = run_dataset_evaluation(
                "medqa", medqa_items, medqa_baselines_df,
                vec_info, sae_cache, wrapper, output_dir,
                medqa_letters, args.skip_exacerbation,
            )
            medqa_records.extend(records)

        if mmlu_items is not None and mmlu_baselines_df is not None:
            records = run_dataset_evaluation(
                "mmlu", mmlu_items, mmlu_baselines_df,
                vec_info, sae_cache, wrapper, output_dir,
                mmlu_letters, args.skip_exacerbation,
            )
            mmlu_records.extend(records)

    # Save per-item parquets
    if medqa_records:
        pd.DataFrame(medqa_records).to_parquet(
            output_dir / "medqa" / "per_item.parquet",
            index=False, compression="snappy",
        )
    if mmlu_records:
        pd.DataFrame(mmlu_records).to_parquet(
            output_dir / "mmlu" / "per_item.parquet",
            index=False, compression="snappy",
        )

    # Aggregate
    if medqa_records:
        aggregate_medqa(pd.DataFrame(medqa_records), output_dir)
    if mmlu_records:
        aggregate_mmlu(pd.DataFrame(mmlu_records), output_dir)

    update_manifests(run_dir, output_dir)
    identify_top_impact_items(output_dir)

    if not args.skip_figures:
        try:
            from src.visualization.generalization_figures import generate_c3_figures
            generate_c3_figures(output_dir, viable_vectors)
        except Exception as e:
            log(f"WARNING: C3 figure generation failed: {e}")

    runtime = time.time() - t0
    log(f"C3 complete ({runtime:.1f}s)")


# =========================================================================
# Stage C4: Token-Level Feature Interpretability
# =========================================================================

def run_c4(
    run_dir: Path,
    config: dict,
    wrapper: object,
    sae_cache: dict,
    args: argparse.Namespace,
) -> None:
    """C4: Token-level feature interpretability."""
    log(f"\n{'=' * 70}")
    log(f"  STAGE C4: Token-Level Feature Interpretability")
    log(f"{'=' * 70}")

    if c4_complete(run_dir) and not args.force:
        log("C4 output already exists. Use --force to rerun.")
        return

    t0 = time.time()

    import json as json_mod
    import numpy as np
    import pandas as pd
    import torch

    from scripts.C4_token_features import (
        _fkey,
        aggregate_per_template,
        aggregate_token_rankings,
        collect_needed_features,
        compute_activation_density,
        compute_logit_effects,
        compute_position_template_positions,
        compute_string_template_tokens,
        extract_token_activations,
        select_top_activating_examples,
    )

    # Load manifests
    with open(run_dir / "C_steering" / "steering_manifests.json") as f:
        manifests = json_mod.load(f)
    viable = [m for m in manifests if m.get("steering_viable")]
    log(f"Viable subgroups: {len(viable)}")

    cat_filter = set(args.categories.split(",")) if args.categories else None

    # Collect features
    vectors_dir = run_dir / "C_steering" / "vectors"
    needed_by_layer, feature_manifest = collect_needed_features(
        viable, vectors_dir, args.max_features_per_subgroup, cat_filter,
    )

    if not feature_manifest:
        log("No features to analyse; skipping C4")
        return

    # Pre-load SAE layers
    for layer in sorted(needed_by_layer.keys()):
        ensure_sae_layer(sae_cache, layer, config)

    # Output directories
    output_dir = run_dir / "C_token_features"
    ensure_dir(output_dir / "token_rankings")
    ensure_dir(output_dir / "per_template_rankings")
    ensure_dir(output_dir / "top_activating_examples")
    ensure_dir(output_dir / "figures")

    atomic_save_json(feature_manifest, output_dir / "feature_manifest.json")

    # Step 1: Logit effects
    log("\n  Step 1: Logit effect decomposition...")
    logit_effects_df = compute_logit_effects(wrapper, sae_cache, feature_manifest)
    logit_effects_df.to_parquet(
        output_dir / "logit_effects.parquet",
        index=False, compression="snappy",
    )

    # Load metadata
    metadata_df = pd.read_parquet(run_dir / "A_extraction" / "metadata.parquet")

    categories = sorted(set(e["category"] for e in feature_manifest))
    if cat_filter:
        categories = [c for c in categories if c in cat_filter]

    # Step 2: Token activation extraction
    log("\n  Step 2: Token activation extraction...")
    from collections import defaultdict

    all_accumulator: dict = {}
    template_filters_data: dict = {
        "string_level_threshold": 0.90,
        "position_level_threshold": 0.80,
        "per_category": {},
    }
    all_string_templates: dict = {}
    all_stimuli_by_cat: dict = {}

    tokenizer = wrapper.tokenizer

    for category in categories:
        log(f"\n    Category: {category}")
        stim_path = run_dir / "A_extraction" / "stimuli" / f"{category}.json"
        with open(stim_path) as f:
            stimuli = json_mod.load(f)

        stimuli_by_idx = {s["item_idx"]: s for s in stimuli}
        all_stimuli_by_cat[category] = stimuli_by_idx

        string_templates = compute_string_template_tokens(stimuli, tokenizer)
        position_templates = compute_position_template_positions(
            stimuli, metadata_df, tokenizer,
        )
        all_string_templates[category] = string_templates

        template_filters_data["per_category"][category] = {
            "string_level_template_tokens": sorted(string_templates),
            "n_template_tokens": len(string_templates),
        }

        cat_accum = extract_token_activations(
            category, needed_by_layer, feature_manifest,
            wrapper, sae_cache, metadata_df, stimuli,
            max_items=args.max_items,
        )
        all_accumulator.update(cat_accum)

    atomic_save_json(template_filters_data, output_dir / "template_filters.json")

    # Steps 3-4: Aggregate
    log("\n  Steps 3-4: Per-feature aggregation...")
    activation_densities: dict = {}
    interpretability_rows: list[dict] = []
    full_per_token_rows: list[dict] = []

    feat_to_subs: dict = defaultdict(list)
    for entry in feature_manifest:
        feat_to_subs[(entry["layer"], entry["feature_idx"])].append(entry)

    for (layer, feat_idx), records in all_accumulator.items():
        fk = _fkey(layer, feat_idx)
        entries = feat_to_subs.get((layer, feat_idx), [])
        cats_for_feat = sorted(set(e["category"] for e in entries))
        subs_for_feat = sorted(set(
            f"{e['category']}/{e['subgroup']}" for e in entries
        ))

        cat_for_filter = cats_for_feat[0] if cats_for_feat else ""
        string_tpl = all_string_templates.get(cat_for_filter, set())

        token_df = aggregate_token_rankings(records, string_tpl)
        if not token_df.empty:
            token_df.to_parquet(
                output_dir / "token_rankings" / f"{fk}.parquet",
                index=False, compression="snappy",
            )

        template_df = aggregate_per_template(records)
        if not template_df.empty:
            template_df.to_parquet(
                output_dir / "per_template_rankings" / f"{fk}.parquet",
                index=False, compression="snappy",
            )

        density = compute_activation_density(records)
        activation_densities[fk] = density

        all_stim = {}
        for cat in cats_for_feat:
            all_stim.update(all_stimuli_by_cat.get(cat, {}))
        top_examples = select_top_activating_examples(records, all_stim)
        atomic_save_json(
            top_examples,
            output_dir / "top_activating_examples" / f"{fk}.json",
        )

        # Summary row
        feat_logit = logit_effects_df[
            (logit_effects_df["layer"] == layer)
            & (logit_effects_df["feature_idx"] == feat_idx)
        ]
        top_pos_token = ""
        top_neg_token = ""
        if not feat_logit.empty:
            pos = feat_logit[(feat_logit["direction"] == "positive") & (feat_logit["rank"] == 1)]
            neg = feat_logit[(feat_logit["direction"] == "negative") & (feat_logit["rank"] == 1)]
            if not pos.empty:
                top_pos_token = str(pos["token_str"].iloc[0])
            if not neg.empty:
                top_neg_token = str(neg["token_str"].iloc[0])

        top_tok_unfiltered = ""
        top_tok_filtered = ""
        if not token_df.empty:
            top_tok_unfiltered = str(token_df.iloc[0]["token"])
            filtered = token_df[~token_df["is_template_string"]]
            if not filtered.empty:
                top_tok_filtered = str(filtered.iloc[0]["token"])

        interpretability_rows.append({
            "layer": layer, "feature_idx": feat_idx,
            "categories": ",".join(cats_for_feat),
            "subgroups": ",".join(subs_for_feat),
            "n_items_processed": len(records),
            "density": density["density"],
            "mean_activation_nonzero": density["mean_activation_nonzero"],
            "median_activation_nonzero": density["median_activation_nonzero"],
            "max_activation": density["max_activation"],
            "top_activating_token": top_tok_unfiltered,
            "top_activating_token_filtered": top_tok_filtered,
            "top_positive_logit_token": top_pos_token,
            "top_negative_logit_token": top_neg_token,
        })

        if args.save_full_per_token:
            for rec in records:
                for pos, (tok, act) in enumerate(
                    zip(rec["tokens"], rec["activations"]),
                ):
                    full_per_token_rows.append({
                        "layer": layer, "feature_idx": feat_idx,
                        "item_idx": rec["item_idx"],
                        "position": pos, "token": tok, "activation": act,
                    })

    # Save outputs
    if interpretability_rows:
        pd.DataFrame(interpretability_rows).to_parquet(
            output_dir / "feature_interpretability.parquet",
            index=False, compression="snappy",
        )
    atomic_save_json(activation_densities, output_dir / "activation_densities.json")

    if full_per_token_rows:
        pd.DataFrame(full_per_token_rows).to_parquet(
            output_dir / "per_item_per_token_activations.parquet",
            index=False, compression="snappy",
        )

    if not args.skip_figures:
        try:
            from src.visualization.token_feature_figures import generate_c4_figures
            generate_c4_figures(
                output_dir, feature_manifest, logit_effects_df,
                activation_densities, all_accumulator,
                all_string_templates, viable,
            )
        except Exception as e:
            log(f"WARNING: C4 figure generation failed: {e}")

    runtime = time.time() - t0
    log(f"C4 complete ({runtime:.1f}s)")


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    args = parse_args()
    stages = [s.strip().upper() for s in args.stages.split(",")]
    valid_stages = {"C1", "C2", "C3", "C4"}
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

    config = load_config(run_dir)

    # ── Dependency checks ────────────────────────────────────────────
    if "C1" in stages:
        ranked_path = run_dir / "B_feature_ranking" / "ranked_features.parquet"
        if not ranked_path.exists():
            log("ERROR: C1 requires B2 output (ranked_features.parquet). "
                "Run Phase B first.")
            sys.exit(1)

    if "C2" in stages and "C1" not in stages:
        if not c1_complete(run_dir):
            log("ERROR: C2 requires C1 output (steering_manifests.json). "
                "Run C1 first or include C1 in --stages.")
            sys.exit(1)
        b3_dir = run_dir / "B_geometry"
        if not (b3_dir / "cosine_pairs.parquet").exists():
            log("WARNING: C2 needs B3 cosine outputs for the primary analysis. "
                "Proceeding, but primary cosines will be unavailable.")

    if "C3" in stages and "C1" not in stages:
        if not c1_complete(run_dir):
            log("ERROR: C3 requires C1 output. Run C1 first or include C1.")
            sys.exit(1)

    if "C4" in stages and "C1" not in stages:
        if not c1_complete(run_dir):
            log("ERROR: C4 requires C1 output. Run C1 first or include C1.")
            sys.exit(1)

    # ── Provenance ───────────────────────────────────────────────────
    provenance = build_provenance(config, config.get("device", "cpu"), stages)
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
    log(f"  Phase C Pipeline: Steering & Evaluation")
    log(f"{'=' * 70}")
    log(f"  Run dir:    {run_dir}")
    log(f"  Stages:     {' → '.join(stages)}")
    log(f"  Device:     {config.get('device', 'cpu')}")
    if args.categories:
        log(f"  Categories: {args.categories}")
    if args.max_items:
        log(f"  Max items:  {args.max_items}")
    if args.skip_exacerbation:
        log(f"  Exacerbation: SKIPPED")
    log(f"{'=' * 70}")

    pipeline_t0 = time.time()

    # ── Load model (shared across all stages) ────────────────────────
    wrapper, sae_cache = load_model_and_saes(config, run_dir, stages)

    # ── Execute stages ───────────────────────────────────────────────
    if "C1" in stages:
        run_c1(run_dir, config, wrapper, sae_cache, args)

    if "C2" in stages:
        run_c2(run_dir, config, wrapper, sae_cache, args)

    if "C3" in stages:
        run_c3(run_dir, config, wrapper, sae_cache, args)

    if "C4" in stages:
        run_c4(run_dir, config, wrapper, sae_cache, args)

    total_elapsed = time.time() - pipeline_t0
    log(f"\n{'=' * 70}")
    log(f"  Phase C complete — {total_elapsed:.1f}s total")
    log(f"{'=' * 70}")


if __name__ == "__main__":
    main()
