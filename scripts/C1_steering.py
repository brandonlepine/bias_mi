"""C1: Subgroup-specific steering optimisation.

For each subgroup with significant pro-bias features from B1/B2, finds the
optimal single-hook steering configuration (k features, target vector norm τ)
that maximises debiasing efficiency η = RCR₁.₀ / ‖v‖₂.

Produces per-subgroup steering vectors consumed by C2, C3, and C4.  Also runs
exacerbation tests (positive-direction steering) to characterise the asymmetry
between debiasing and bias-amplification.

Usage:
    python scripts/C1_steering.py --run_dir runs/llama-3.1-8b_2026-04-15/

    # Quick test
    python scripts/C1_steering.py --run_dir runs/llama-3.1-8b_2026-04-15/ --max_items 20

    # Specific categories / subgroups
    python scripts/C1_steering.py --run_dir runs/llama-3.1-8b_2026-04-15/ --categories so,disability
    python scripts/C1_steering.py --run_dir runs/llama-3.1-8b_2026-04-15/ --subgroups so/gay,race/black

    # Skip exacerbation, override target norms
    python scripts/C1_steering.py --run_dir runs/llama-3.1-8b_2026-04-15/ --skip_exacerbation
    python scripts/C1_steering.py --run_dir runs/llama-3.1-8b_2026-04-15/ --target_norms -0.5,-1,-2,-5,-10,-20,-40
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.metrics.bias_metrics import (
    build_result_dict,
    compute_all_metrics,
)
from src.models.wrapper import ModelWrapper
from src.sae.wrapper import SAEWrapper
from src.sae_localization.steering import SAESteerer
from src.utils.config import load_config
from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log, progress_bar


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TARGET_NORMS = [-0.5, -1.0, -2.0, -5.0, -10.0, -20.0, -40.0, -80.0]
BASE_K_STEPS = [1, 2, 3, 5, 8, 13, 21, 34, 55]
DEFAULT_MIN_N_PER_GROUP = 10


def extract_metric_value(metrics: dict[str, Any], metric_key: str) -> float:
    """Extract the scalar metric value from a compute_all_metrics result.

    Handles both MWCS keys (``metrics["mwcs_1.0"]["mwcs"]``) and RCR keys
    (``metrics["rcr_1.0"]["rcr"]``).
    """
    entry = metrics.get(metric_key, {})
    if "mwcs" in entry:
        return float(entry["mwcs"])
    if "rcr" in entry:
        return float(entry["rcr"])
    return 0.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="C1: Subgroup-specific steering optimisation")
    p.add_argument("--run_dir", required=True, type=str,
                   help="Path to run directory (e.g. runs/llama-3.1-8b_2026-04-15/)")
    p.add_argument("--categories", type=str, default=None,
                   help="Comma-separated category filter (e.g. so,disability)")
    p.add_argument("--subgroups", type=str, default=None,
                   help="Comma-separated subgroup filter (format: cat/sub)")
    p.add_argument("--max_items", type=int, default=None,
                   help="Max items per subgroup group (for quick tests)")
    p.add_argument("--min_n_per_group", type=int, default=DEFAULT_MIN_N_PER_GROUP,
                   help="Minimum items per stereotype/non-stereotype group")
    p.add_argument("--skip_exacerbation", action="store_true",
                   help="Skip exacerbation tests")
    p.add_argument("--skip_figures", action="store_true",
                   help="Skip figure generation")
    p.add_argument("--target_norms", type=str, default=None,
                   help="Comma-separated target norm values (negative for debiasing)")
    p.add_argument("--injection_layer_min", type=int, default=10,
                   help="Minimum layer for steering features (inclusive, default: 10)")
    p.add_argument("--injection_layer_max", type=int, default=24,
                   help="Maximum layer for steering features (inclusive, default: 24)")
    p.add_argument("--min_n_eligible", type=int, default=10,
                   help="Minimum items with margin >= tau for RCR to be meaningful")
    p.add_argument("--max_degeneration_rate", type=float, default=0.05,
                   help="Maximum degeneration rate for a config to be safe")
    p.add_argument("--max_corruption_rate", type=float, default=0.15,
                   help="Maximum corruption rate (relaxed tier)")
    p.add_argument("--max_corruption_rate_strict", type=float, default=0.05,
                   help="Maximum corruption rate (strict tier)")
    p.add_argument("--optimizer_metric", type=str, default="mwcs_1.0",
                   choices=["mwcs_0.5", "mwcs_1.0", "mwcs_2.0",
                            "rcr_0.5", "rcr_1.0", "rcr_2.0"],
                   help="Metric the optimizer maximises (numerator of eta)")
    p.add_argument("--mwcs_floor", type=float, default=0.05,
                   help="Minimum metric value for Phase 1 viability")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Vector construction
# ---------------------------------------------------------------------------

def build_subgroup_steering_vector(
    top_k_features: list[dict[str, Any]],
    sae_cache: dict[int, SAEWrapper],
    alpha: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, float]:
    """Construct steering vector as α × mean of unit-normalised decoder columns.

    Features may come from different layers; their decoder columns are all in
    the same ambient space (R^hidden_dim) so averaging is well-defined.

    Returns
    -------
    vec : Tensor
        Shape ``(hidden_dim,)``.
    mean_dir_norm : float
        ``||mean_of_unit_directions||`` — needed to convert target_norm → alpha.
    """
    directions = []
    for f in top_k_features:
        sae = sae_cache[f["layer"]]
        d = sae.get_feature_direction(f["feature_idx"])  # unit-normalised
        directions.append(torch.from_numpy(d).float())

    stacked = torch.stack(directions, dim=0)  # (k, hidden_dim)
    mean_dir = stacked.mean(dim=0)             # (hidden_dim,)
    mean_dir_norm = float(mean_dir.norm().item())

    vec = (alpha * mean_dir).to(dtype=dtype, device=device)
    return vec, mean_dir_norm


def alpha_for_target_norm(target_norm: float, mean_dir_norm: float) -> float:
    """Compute alpha so that ||alpha * mean_dir|| == |target_norm|, sign preserved."""
    if mean_dir_norm < 1e-8:
        return 0.0
    return target_norm / mean_dir_norm


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_ranked_features(run_dir: Path) -> pd.DataFrame:
    """Load B2 ranked features parquet."""
    path = run_dir / "B_feature_ranking" / "ranked_features.parquet"
    df = pd.read_parquet(path)
    log(f"Loaded {len(df)} ranked features from {path}")
    return df


def load_injection_layers(run_dir: Path) -> dict[str, Any]:
    """Load B2 injection layers JSON."""
    path = run_dir / "B_feature_ranking" / "injection_layers.json"
    with open(path) as f:
        data = json.load(f)
    log(f"Loaded injection layers for {len(data)} subgroups")
    return data


def load_artifact_flags(run_dir: Path) -> set[tuple[int, int]]:
    """Load B5 artifact flags, returning set of (feature_idx, layer) pairs."""
    path = run_dir / "B_feature_interpretability" / "artifact_flags.json"
    if not path.exists():
        log("No artifact_flags.json found; no features excluded")
        return set()
    with open(path) as f:
        data = json.load(f)
    flagged = set()
    for entry in data.get("flagged_features", []):
        flagged.add((entry["feature_idx"], entry["layer"]))
    log(f"Loaded {len(flagged)} flagged feature (idx, layer) pairs for exclusion")
    return flagged


def filter_flagged_features(
    df: pd.DataFrame, flagged_set: set[tuple[int, int]],
) -> pd.DataFrame:
    """Remove flagged features from the ranked features dataframe."""
    if not flagged_set:
        return df
    before = len(df)
    mask = df.apply(
        lambda row: (row["feature_idx"], row["layer"]) not in flagged_set,
        axis=1,
    )
    df = df[mask].copy()
    removed = before - len(df)
    if removed > 0:
        log(f"  Removed {removed} flagged features from ranking")
    return df


def load_metadata(run_dir: Path) -> pd.DataFrame:
    """Load A2 metadata parquet."""
    path = run_dir / "A_extraction" / "metadata.parquet"
    df = pd.read_parquet(path)
    log(f"Loaded metadata: {len(df)} items")
    return df


def load_stimuli(run_dir: Path, category: str) -> list[dict[str, Any]]:
    """Load stimuli JSON for a category."""
    path = run_dir / "A_extraction" / "stimuli" / f"{category}.json"
    with open(path) as f:
        data = json.load(f)
    return data


def get_ranked_s_marking_features(
    df: pd.DataFrame,
    category: str,
    subgroup: str,
    layer_min: int | None = None,
    layer_max: int | None = None,
) -> list[dict[str, Any]]:
    """Get ranked s_marking features for a subgroup, optionally filtered to a
    layer range.  Re-ranks by |cohens_d| within the filtered set."""
    mask = (
        (df["category"] == category)
        & (df["subgroup"] == subgroup)
        & (df["direction"] == "s_marking")
    )
    if layer_min is not None:
        mask = mask & (df["layer"] >= layer_min)
    if layer_max is not None:
        mask = mask & (df["layer"] <= layer_max)

    sub_df = df[mask].copy()
    if sub_df.empty:
        return []

    sub_df["abs_d"] = sub_df["cohens_d"].abs()
    sub_df = sub_df.sort_values("abs_d", ascending=False).reset_index(drop=True)

    features: list[dict[str, Any]] = []
    for _, row in sub_df.iterrows():
        features.append({
            "feature_idx": int(row["feature_idx"]),
            "layer": int(row["layer"]),
            "cohens_d": float(row["cohens_d"]),
        })
    return features


def compute_injection_layer_from_features(
    features: list[dict[str, Any]],
) -> int | None:
    """Effect-weighted injection layer from a list of features.

    Mirrors B2's logic: sum |cohens_d| per layer, pick max, break ties
    by preferring deeper layers.
    """
    if not features:
        return None
    layer_scores: dict[int, float] = {}
    for f in features:
        layer_scores.setdefault(f["layer"], 0.0)
        layer_scores[f["layer"]] += abs(f["cohens_d"])
    max_score = max(layer_scores.values())
    candidates = [l for l, s in layer_scores.items() if s == max_score]
    return max(candidates)


# ---------------------------------------------------------------------------
# Subgroup enumeration
# ---------------------------------------------------------------------------

def determine_subgroups(
    ranked_df: pd.DataFrame,
    injection_layers: dict[str, Any],
    filter_categories: str | None,
    filter_subgroups: str | None,
) -> list[tuple[str, str]]:
    """Determine which (category, subgroup) pairs to process.

    Uses ranked_df to find all subgroups with s_marking features (does NOT
    filter by injection layer range here — that happens in process_subgroup).
    """
    s_marking = ranked_df[ranked_df["direction"] == "s_marking"]
    all_pairs = set()
    for _, row in s_marking[["category", "subgroup"]].drop_duplicates().iterrows():
        all_pairs.add((row["category"], row["subgroup"]))

    # Apply category filter
    if filter_categories:
        cats = set(filter_categories.split(","))
        all_pairs = {(c, s) for c, s in all_pairs if c in cats}

    # Apply subgroup filter
    if filter_subgroups:
        subs = set(filter_subgroups.split(","))
        all_pairs = {(c, s) for c, s in all_pairs if f"{c}/{s}" in subs}

    result = sorted(all_pairs)
    log(f"Will process {len(result)} subgroups")
    return result


def identify_needed_layers(
    ranked_df: pd.DataFrame,
    subgroups: list[tuple[str, str]],
    max_k: int = 55,
) -> set[int]:
    """Identify all SAE layers needed across all subgroups."""
    needed = set()
    for cat, sub in subgroups:
        sub_df = ranked_df[
            (ranked_df["category"] == cat)
            & (ranked_df["subgroup"] == sub)
            & (ranked_df["direction"] == "s_marking")
        ].sort_values("rank").head(max_k)
        for _, row in sub_df.iterrows():
            needed.add(int(row["layer"]))
    return needed


# ---------------------------------------------------------------------------
# Item partitioning
# ---------------------------------------------------------------------------

def partition_items(
    metadata_df: pd.DataFrame,
    stimuli: list[dict[str, Any]],
    category: str,
    subgroup: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition items into stereotyped-response and non-stereotyped-response.

    Only ambiguous items targeting the subgroup are included.

    Returns
    -------
    stereo_items : list[dict]
        Items where the model gave a stereotyped response.
    non_stereo_items : list[dict]
        Items where the model gave a non-stereotyped response.
    """
    # Build a lookup from item_idx to metadata row
    cat_meta = metadata_df[metadata_df["category"] == category]
    meta_by_idx: dict[int, dict[str, Any]] = {}
    for _, row in cat_meta.iterrows():
        meta_by_idx[int(row["item_idx"])] = row.to_dict()

    # Build a lookup from item_idx to stimulus
    stim_by_idx = {s["item_idx"]: s for s in stimuli}

    stereo_items: list[dict[str, Any]] = []
    non_stereo_items: list[dict[str, Any]] = []

    for item_idx, stim in stim_by_idx.items():
        # Only ambiguous items
        if stim.get("context_condition") != "ambig":
            continue

        # Only items targeting this subgroup
        stereo_groups = stim.get("stereotyped_groups", [])
        if subgroup not in stereo_groups:
            continue

        meta = meta_by_idx.get(item_idx)
        if meta is None:
            continue

        model_answer_role = meta.get("model_answer_role", "")

        item_record = {
            "item_idx": item_idx,
            "prompt": stim["prompt"],
            "stereotyped_option": stim.get("stereotyped_option"),
            "model_answer_role": model_answer_role,
        }

        if model_answer_role == "stereotyped_target":
            stereo_items.append(item_record)
        else:
            non_stereo_items.append(item_record)

    return stereo_items, non_stereo_items


# ---------------------------------------------------------------------------
# Baseline computation
# ---------------------------------------------------------------------------

def compute_baselines(
    steerer: SAESteerer,
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compute baseline (no-steering) forward passes for a list of items.

    Returns list of baseline result dicts (one per item, in order).
    """
    baselines = []
    for item in items:
        bl = steerer.evaluate_baseline(item["prompt"])
        baselines.append(bl)
    return baselines


# ---------------------------------------------------------------------------
# Checkpoint naming
# ---------------------------------------------------------------------------

def config_ckpt_name(cat: str, sub: str, k: int, target_norm: float) -> str:
    """Build deterministic checkpoint filename.

    target_norm = -1.25 → "norm-0125"
    target_norm = -10.0 → "norm-1000"
    target_norm = +5.0  → "norm+0500"
    """
    norm_int = int(round(target_norm * 100))
    return f"{cat}_{sub}_k{k:03d}_norm{norm_int:+05d}.json"


# ---------------------------------------------------------------------------
# Phase 1: Target-norm viability at k=1
# ---------------------------------------------------------------------------

def run_phase1(
    s_marking_features: list[dict[str, Any]],
    sae_cache: dict[int, SAEWrapper],
    baselines: list[dict[str, Any]],
    stereo_items: list[dict[str, Any]],
    steerer: SAESteerer,
    target_norms: list[float],
    dtype: torch.dtype,
    output_dir: Path,
    cat: str,
    sub: str,
) -> dict[str, Any]:
    """Sweep target_norms at k=1 to identify viable magnitudes.

    A target_norm is viable if:
        - RCR_1.0 > 0 (some correction occurs)
        - degeneration_rate < 0.05
    """
    ckpt_dir = output_dir / "checkpoints"
    phase1_ckpt = ckpt_dir / f"{cat}_{sub}_phase1.json"

    if phase1_ckpt.exists():
        with open(phase1_ckpt) as f:
            loaded = json.load(f)
        log(f"  Phase 1: LOADED from checkpoint")
        return loaded

    # Compute mean_dir_norm at k=1
    top_1 = s_marking_features[:1]
    _, mean_norm_k1 = build_subgroup_steering_vector(
        top_1, sae_cache, alpha=1.0,
        device=torch.device(steerer.device), dtype=dtype,
    )

    results: dict[str, Any] = {}

    for tn in target_norms:
        alpha = alpha_for_target_norm(tn, mean_norm_k1)
        vec, _ = build_subgroup_steering_vector(
            top_1, sae_cache, alpha,
            device=torch.device(steerer.device), dtype=dtype,
        )
        vec_norm = float(vec.norm().item())

        per_item: list[dict[str, Any]] = []
        for item, baseline in zip(stereo_items, baselines):
            steered = steerer.steer_and_evaluate(item["prompt"], vec)
            per_item.append(build_result_dict(item, baseline, steered, vec))

        metrics = compute_all_metrics(per_item)
        degen = sum(1 for r in per_item if r["degenerated"]) / len(per_item)
        corrupt = sum(1 for r in per_item if r["corrupted"]) / len(per_item)

        results[str(tn)] = {
            "target_norm": tn,
            "alpha": alpha,
            "vector_norm": vec_norm,
            "metrics": metrics,
            "degeneration_rate": degen,
            "corruption_rate": corrupt,
            "n_items": len(per_item),
        }

        mwcs_val = extract_metric_value(metrics, "mwcs_1.0")
        rcr_val = metrics["rcr_1.0"]["rcr"]
        n_elig = metrics["rcr_1.0"]["n_eligible"]
        log(f"    τ={tn:+.1f} α={alpha:+.2f} ||v||={vec_norm:.2f}: "
            f"MWCS₁.₀={mwcs_val:.3f} RCR₁.₀={rcr_val:.3f}(n_elig={n_elig}) "
            f"degen={degen:.3f}")

    atomic_save_json(results, phase1_ckpt)
    return results


def identify_viable_target_norms(
    phase1_results: dict[str, Any],
    optimizer_metric: str = "mwcs_1.0",
    metric_floor: float = 0.05,
    max_degen: float = 0.05,
) -> list[float]:
    """Select target_norms passing viability criteria.

    A target_norm is viable if:
    - The optimizer metric value > metric_floor
    - degeneration_rate < max_degen
    """
    viable = []
    for tn, r in phase1_results.items():
        metrics = r.get("metrics", {})
        metric_val = extract_metric_value(metrics, optimizer_metric)
        if metric_val > metric_floor and r["degeneration_rate"] < max_degen:
            viable.append(float(tn))

    if len(viable) < 2:
        # Relax degeneration to 0.10
        viable = []
        for tn, r in phase1_results.items():
            metrics = r.get("metrics", {})
            metric_val = extract_metric_value(metrics, optimizer_metric)
            if metric_val > metric_floor and r["degeneration_rate"] < 0.10:
                viable.append(float(tn))

    return sorted(viable, reverse=True)


# ---------------------------------------------------------------------------
# Phase 2: Joint (k, target_norm) sweep
# ---------------------------------------------------------------------------

def build_k_steps(n_features: int) -> list[int]:
    """K values from base list that don't exceed n_features."""
    return [k for k in BASE_K_STEPS if k <= n_features]


def run_phase2(
    s_marking_features: list[dict[str, Any]],
    sae_cache: dict[int, SAEWrapper],
    baselines: list[dict[str, Any]],
    stereo_items: list[dict[str, Any]],
    steerer: SAESteerer,
    k_steps: list[int],
    viable_target_norms: list[float],
    dtype: torch.dtype,
    output_dir: Path,
    cat: str,
    sub: str,
    injection_layer: int,
    optimizer_metric: str = "mwcs_1.0",
) -> list[dict[str, Any]]:
    """Full (k, target_norm) sweep with per-config checkpointing."""
    ckpt_dir = output_dir / "checkpoints"
    grid: list[dict[str, Any]] = []
    device = torch.device(steerer.device)

    for k in k_steps:
        top_k = s_marking_features[:k]
        # Compute mean_dir_norm for this k
        _, mean_norm_k = build_subgroup_steering_vector(
            top_k, sae_cache, alpha=1.0, device=device, dtype=dtype,
        )

        for tn in viable_target_norms:
            ckpt_name = config_ckpt_name(cat, sub, k, tn)
            ckpt_path = ckpt_dir / ckpt_name

            if ckpt_path.exists():
                with open(ckpt_path) as f:
                    record = json.load(f)
                grid.append(record)
                log(f"    k={k:03d} τ={tn:+.1f}: LOADED from checkpoint")
                continue

            alpha = alpha_for_target_norm(tn, mean_norm_k)
            vec, _ = build_subgroup_steering_vector(
                top_k, sae_cache, alpha, device=device, dtype=dtype,
            )
            vec_norm = float(vec.norm().item())

            per_item: list[dict[str, Any]] = []
            for item, baseline in zip(stereo_items, baselines):
                steered = steerer.steer_and_evaluate(item["prompt"], vec)
                per_item.append(build_result_dict(item, baseline, steered, vec))

            metrics = compute_all_metrics(per_item)
            degen = sum(1 for r in per_item if r["degenerated"]) / len(per_item)
            corrupt = sum(1 for r in per_item if r["corrupted"]) / len(per_item)

            # Use the chosen optimizer metric for eta
            optimizer_metric_value = extract_metric_value(metrics, optimizer_metric)
            eta = optimizer_metric_value / max(vec_norm, 1e-8)

            n_elig = metrics["rcr_1.0"]["n_eligible"]

            record = {
                "category": cat,
                "subgroup": sub,
                "k": k,
                "target_norm": tn,
                "alpha": alpha,
                "injection_layer": injection_layer,
                "vector_norm": vec_norm,
                "eta": eta,
                "optimizer_metric_value": optimizer_metric_value,
                "optimizer_metric_name": optimizer_metric,
                "metrics": metrics,
                "degeneration_rate": degen,
                "corruption_rate": corrupt,
                "n_items": len(per_item),
                "per_item_results": per_item,
            }

            atomic_save_json(record, ckpt_path)
            grid.append(record)

            log(f"    k={k:03d} τ={tn:+.1f} α={alpha:+.2f}: "
                f"MWCS={optimizer_metric_value:.3f} RCR₁.₀(n_elig={n_elig}) "
                f"η={eta:.3f} ||v||={vec_norm:.2f} "
                f"degen={degen:.3f} corrupt={corrupt:.3f}")

    return grid


# ---------------------------------------------------------------------------
# Phase 3: Optimal selection
# ---------------------------------------------------------------------------

def select_optimal_tiered(
    grid: list[dict[str, Any]],
    max_degen: float,
    max_corrupt_relaxed: float,
    max_corrupt_strict: float,
    optimizer_metric: str = "mwcs_1.0",
    metric_floor: float = 0.05,
) -> dict[str, Any]:
    """Select optimal config under both strict and relaxed safety tiers.

    A config is eligible if:
    - optimizer metric value > metric_floor
    - ``degeneration_rate < max_degen``
    - ``corruption_rate < corruption_threshold`` (varies by tier)

    Tie-breaking within 1% of best eta: smaller ||v||, then larger eta.

    Returns ``{"relaxed": config_or_None, "strict": config_or_None}``.
    """
    def _select(max_corrupt: float) -> dict[str, Any] | None:
        eligible = [
            r for r in grid
            if extract_metric_value(r["metrics"], optimizer_metric) > metric_floor
            and r["degeneration_rate"] < max_degen
            and r["corruption_rate"] < max_corrupt
        ]
        if not eligible:
            return None
        best_eta = max(r["eta"] for r in eligible)
        candidates = [r for r in eligible if r["eta"] >= best_eta * 0.99]
        candidates.sort(key=lambda r: (r["vector_norm"], -r["eta"]))
        return candidates[0]

    return {
        "relaxed": _select(max_corrupt_relaxed),
        "strict": _select(max_corrupt_strict),
    }


# ---------------------------------------------------------------------------
# Phase 4: Marginal analysis
# ---------------------------------------------------------------------------

def compute_marginal_analysis(
    grid: list[dict[str, Any]], optimal: dict[str, Any],
) -> list[dict[str, Any]]:
    """At optimal target_norm, show how RCR and ||v|| evolve with k."""
    optimal_tn = optimal["target_norm"]
    relevant = [r for r in grid if r["target_norm"] == optimal_tn]
    relevant.sort(key=lambda r: r["k"])

    marginal: list[dict[str, Any]] = []
    for i, r in enumerate(relevant):
        entry: dict[str, Any] = {
            "k": r["k"],
            "rcr_1.0": r["metrics"]["rcr_1.0"]["rcr"],
            "vector_norm": r["vector_norm"],
            "eta": r["eta"],
        }
        if i > 0:
            prev = relevant[i - 1]
            entry["marginal_rcr_gain"] = (
                r["metrics"]["rcr_1.0"]["rcr"]
                - prev["metrics"]["rcr_1.0"]["rcr"]
            )
            entry["marginal_norm_cost"] = r["vector_norm"] - prev["vector_norm"]
            denom = max(abs(entry["marginal_norm_cost"]), 1e-8)
            entry["marginal_efficiency"] = entry["marginal_rcr_gain"] / denom
        marginal.append(entry)

    return marginal


# ---------------------------------------------------------------------------
# Exacerbation test
# ---------------------------------------------------------------------------

def run_exacerbation_test(
    optimal: dict[str, Any],
    s_marking_features: list[dict[str, Any]],
    sae_cache: dict[int, SAEWrapper],
    steerer: SAESteerer,
    stereo_items: list[dict[str, Any]],
    non_stereo_items: list[dict[str, Any]],
    baselines_stereo: list[dict[str, Any]],
    dtype: torch.dtype,
    output_dir: Path,
    cat: str,
    sub: str,
) -> dict[str, Any]:
    """At +|optimal_target_norm| (exacerbation direction):
    1. Test corruption on non-stereotyped items
    2. Test amplification on stereotyped items
    """
    ckpt_path = output_dir / "checkpoints" / f"{cat}_{sub}_exac.json"
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            loaded = json.load(f)
        log(f"  Exacerbation: LOADED from checkpoint")
        return loaded

    exac_target_norm = abs(optimal["target_norm"])
    k = optimal["k"]
    device = torch.device(steerer.device)

    top_k = s_marking_features[:k]
    _, mean_norm = build_subgroup_steering_vector(
        top_k, sae_cache, alpha=1.0, device=device, dtype=dtype,
    )
    exac_alpha = alpha_for_target_norm(exac_target_norm, mean_norm)
    vec, _ = build_subgroup_steering_vector(
        top_k, sae_cache, exac_alpha, device=device, dtype=dtype,
    )
    vec_norm = float(vec.norm().item())

    # Baselines for non_stereo_items
    baselines_non_stereo = compute_baselines(steerer, non_stereo_items)

    # Test on non_stereo items (corruption check)
    per_item_non_stereo: list[dict[str, Any]] = []
    for item, baseline in zip(non_stereo_items, baselines_non_stereo):
        steered = steerer.steer_and_evaluate(item["prompt"], vec)
        per_item_non_stereo.append(build_result_dict(item, baseline, steered, vec))

    metrics_non_stereo = compute_all_metrics(per_item_non_stereo)
    corruption_rate = (
        sum(1 for r in per_item_non_stereo if r["corrupted"])
        / max(len(per_item_non_stereo), 1)
    )

    # Test on stereo items (amplification check)
    per_item_stereo: list[dict[str, Any]] = []
    for item, baseline in zip(stereo_items, baselines_stereo):
        steered = steerer.steer_and_evaluate(item["prompt"], vec)
        per_item_stereo.append(build_result_dict(item, baseline, steered, vec))

    # Amplification: did logit for stereotyped_option INCREASE?
    stereo_logit_shifts: list[float] = []
    for r in per_item_stereo:
        opt = r.get("stereotyped_option")
        if opt and opt in r["logit_baseline"] and opt in r["logit_steered"]:
            stereo_logit_shifts.append(
                r["logit_steered"][opt] - r["logit_baseline"][opt]
            )

    mean_amplification = (
        float(np.mean(stereo_logit_shifts)) if stereo_logit_shifts else None
    )
    frac_amplified = (
        float(sum(1 for s in stereo_logit_shifts if s > 0) / len(stereo_logit_shifts))
        if stereo_logit_shifts else None
    )

    result = {
        "exac_target_norm": exac_target_norm,
        "exac_alpha": exac_alpha,
        "vector_norm": vec_norm,
        "n_non_stereo_items": len(non_stereo_items),
        "n_stereo_items": len(stereo_items),
        "corruption_rate_non_stereo": corruption_rate,
        "metrics_non_stereo": metrics_non_stereo,
        "mean_logit_amplification_stereo": mean_amplification,
        "fraction_amplified_stereo": frac_amplified,
        "per_item_non_stereo": per_item_non_stereo,
        "per_item_stereo": per_item_stereo,
    }

    # Save checkpoint WITHOUT per-item lists (too large for JSON checkpoints)
    ckpt_data = {k: v for k, v in result.items()
                 if k not in ("per_item_non_stereo", "per_item_stereo")}
    atomic_save_json(ckpt_data, ckpt_path)

    log(f"  Exacerbation: corruption={corruption_rate:.3f}, "
        f"mean_amplification={mean_amplification if mean_amplification is not None else 'n/a'}")

    return result


# ---------------------------------------------------------------------------
# Per-item parquet saving
# ---------------------------------------------------------------------------

def save_per_item_parquet(
    output_dir: Path,
    cat: str,
    sub: str,
    optimal: dict[str, Any],
    exac_result: dict[str, Any] | None,
) -> None:
    """Save consolidated per-item results at optimal config."""
    rows: list[dict[str, Any]] = []

    # Debiasing at optimal
    per_item_optimal = optimal.get("per_item_results", [])
    for r in per_item_optimal:
        row = _flatten_result(r, cat, sub, optimal, condition="debiasing_optimal")
        rows.append(row)

    # Exacerbation results on stereo items
    if exac_result and "per_item_stereo" in exac_result:
        exac_config = {
            "k": optimal.get("k"),
            "target_norm": exac_result.get("exac_target_norm"),
            "alpha": exac_result.get("exac_alpha"),
            "injection_layer": optimal.get("injection_layer"),
            "vector_norm": exac_result.get("vector_norm"),
        }
        for r in exac_result["per_item_stereo"]:
            row = _flatten_result(r, cat, sub, exac_config,
                                  condition="exacerbation_optimal")
            rows.append(row)

    # Exacerbation results on non-stereo items
    if exac_result and "per_item_non_stereo" in exac_result:
        exac_config = {
            "k": optimal.get("k"),
            "target_norm": exac_result.get("exac_target_norm"),
            "alpha": exac_result.get("exac_alpha"),
            "injection_layer": optimal.get("injection_layer"),
            "vector_norm": exac_result.get("vector_norm"),
        }
        for r in exac_result["per_item_non_stereo"]:
            row = _flatten_result(r, cat, sub, exac_config,
                                  condition="exacerbation_on_non_stereo")
            rows.append(row)

    if not rows:
        return

    df = pd.DataFrame(rows)
    path = output_dir / "per_item" / f"{cat}_{sub}.parquet"
    df.to_parquet(path, index=False, compression="snappy")


def _flatten_result(
    r: dict[str, Any],
    cat: str,
    sub: str,
    config: dict[str, Any],
    condition: str,
) -> dict[str, Any]:
    """Flatten a per-item result dict for parquet storage."""
    row: dict[str, Any] = {
        "item_idx": r["item_idx"],
        "category": cat,
        "subgroup": sub,
        "condition": condition,
        "k": config.get("k"),
        "target_norm": config.get("target_norm"),
        "alpha": config.get("alpha"),
        "injection_layer": config.get("injection_layer"),
        "vector_norm": r.get("vector_norm", config.get("vector_norm")),
        "baseline_answer": r["baseline_answer"],
        "steered_answer": r["steered_answer"],
        "baseline_role": r["baseline_role"],
        "steered_role": r["steered_role"],
        "corrected": r["corrected"],
        "corrupted": r["corrupted"],
        "degenerated": r["degenerated"],
        "margin": r["margin"],
        "margin_bin": r["margin_bin"],
        "stereotyped_option": r["stereotyped_option"],
    }
    # Individual logit columns
    for letter in ("A", "B", "C"):
        row[f"logit_baseline_{letter}"] = r["logit_baseline"].get(letter)
        row[f"logit_steered_{letter}"] = r["logit_steered"].get(letter)

    return row


# ---------------------------------------------------------------------------
# Steering vector saving
# ---------------------------------------------------------------------------

def save_steering_vector(
    output_dir: Path,
    cat: str,
    sub: str,
    vec: torch.Tensor,
    optimal: dict[str, Any],
    features: list[dict[str, Any]],
    injection_layer: int,
) -> None:
    """Save optimal steering vector as .npz for C2/C3 consumption."""
    k = optimal["k"]
    optimal_features = features[:k]

    np.savez(
        output_dir / "vectors" / f"{cat}_{sub}.npz",
        vector=vec.float().cpu().numpy(),
        injection_layer=np.int32(injection_layer),
        target_norm=np.float32(optimal["target_norm"]),
        alpha=np.float32(optimal["alpha"]),
        k=np.int32(k),
        vector_norm=np.float32(optimal["vector_norm"]),
        eta=np.float32(optimal["eta"]),
        rcr_at_optimal=np.float32(optimal["metrics"]["rcr_1.0"]["rcr"]),
        feature_idxs=np.array(
            [f["feature_idx"] for f in optimal_features], dtype=np.int32,
        ),
        feature_layers=np.array(
            [f["layer"] for f in optimal_features], dtype=np.int32,
        ),
        category=cat,
        subgroup=sub,
    )
    log(f"  Saved steering vector: vectors/{cat}_{sub}.npz")


# ---------------------------------------------------------------------------
# Per-subgroup processing
# ---------------------------------------------------------------------------

def process_subgroup(
    cat: str,
    sub: str,
    ranked_df: pd.DataFrame,
    injection_layers: dict[str, Any],  # kept for compatibility; not used for layer selection
    metadata_df: pd.DataFrame,
    wrapper: ModelWrapper,
    sae_cache: dict[int, SAEWrapper],
    run_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
    args: argparse.Namespace,
    target_norms: list[float],
) -> dict[str, Any]:
    """Run full C1 pipeline for one subgroup.  Returns manifest dict."""
    sub_key = f"{cat}/{sub}"
    log(f"\n{'=' * 60}")
    log(f"Subgroup: {sub_key}")
    log(f"{'=' * 60}")

    device = torch.device(config["device"])
    dtype = getattr(torch, config.get("dtype", "float16"))

    # Initialise manifest
    manifest: dict[str, Any] = {
        "subgroup": sub,
        "category": cat,
        "steering_viable": False,
        "steering_skip_reason": None,
    }

    # Step 0: Get s_marking features filtered to allowed layer range
    layer_min = getattr(args, "injection_layer_min", 10)
    layer_max = getattr(args, "injection_layer_max", 24)

    s_marking_features = get_ranked_s_marking_features(
        ranked_df, cat, sub,
        layer_min=layer_min, layer_max=layer_max,
    )

    # Count global features for comparison
    n_global = int(
        ((ranked_df["category"] == cat)
         & (ranked_df["subgroup"] == sub)
         & (ranked_df["direction"] == "s_marking")).sum()
    )
    manifest["n_s_marking_features_global"] = n_global

    if len(s_marking_features) == 0:
        manifest["steering_skip_reason"] = (
            f"no_s_marking_features_in_range[{layer_min},{layer_max}]"
        )
        log(f"  SKIP: no s_marking features in layer range "
            f"[{layer_min}, {layer_max}] (global: {n_global})")
        return manifest

    # Compute injection layer from filtered features (effect-weighted)
    injection_layer = compute_injection_layer_from_features(s_marking_features)
    manifest["injection_layer"] = injection_layer
    manifest["injection_layer_range"] = [layer_min, layer_max]
    manifest["n_s_marking_features_in_range"] = len(s_marking_features)

    # Log layer distribution of top-50 features
    layer_dist = Counter(f["layer"] for f in s_marking_features[:50])
    log(f"  s_marking features in range [{layer_min}, {layer_max}]: "
        f"{len(s_marking_features)} (global: {n_global})")
    log(f"  Injection layer: {injection_layer} (effect-weighted within range)")
    log(f"  Top-50 layer distribution: {dict(sorted(layer_dist.items()))}")

    # Step 1: Partition items
    stimuli = load_stimuli(run_dir, cat)
    stereo_items, non_stereo_items = partition_items(
        metadata_df, stimuli, cat, sub,
    )

    if args.max_items:
        stereo_items = stereo_items[:args.max_items]
        non_stereo_items = non_stereo_items[:args.max_items]

    n_stereo = len(stereo_items)
    n_non_stereo = len(non_stereo_items)

    manifest["n_stereo_items"] = n_stereo
    manifest["n_non_stereo_items"] = n_non_stereo

    if n_stereo < args.min_n_per_group or n_non_stereo < args.min_n_per_group:
        manifest["steering_skip_reason"] = (
            f"insufficient_items (stereo={n_stereo}, non_stereo={n_non_stereo})"
        )
        log(f"  SKIP: insufficient items (need ≥{args.min_n_per_group} per group)")
        return manifest

    log(f"  n_stereo={n_stereo}, n_non_stereo={n_non_stereo}")

    # Ensure SAE for injection layer is loaded
    if injection_layer not in sae_cache:
        log(f"  Loading SAE for injection layer {injection_layer}...")
        sae_cache[injection_layer] = SAEWrapper(
            config["sae_source"],
            layer=injection_layer,
            expansion=config.get("sae_expansion", 32),
            device=config["device"],
        )

    steerer = SAESteerer(wrapper, sae_cache[injection_layer], injection_layer)

    # Compute baselines for stereo_items (reused across Phase 1/2)
    log(f"  Computing baselines for {n_stereo} stereo items...")
    baselines_stereo = compute_baselines(steerer, stereo_items)

    # Step 2: Phase 1 — target_norm viability at k=1
    log(f"  Phase 1: target_norm sweep at k=1...")
    phase1_results = run_phase1(
        s_marking_features, sae_cache, baselines_stereo, stereo_items,
        steerer, target_norms, dtype, output_dir, cat, sub,
    )
    manifest["phase1_results"] = phase1_results

    viable_target_norms = identify_viable_target_norms(
        phase1_results,
        optimizer_metric=args.optimizer_metric,
        metric_floor=args.mwcs_floor,
        max_degen=args.max_degeneration_rate,
    )

    if len(viable_target_norms) < 2:
        manifest["steering_skip_reason"] = "no_viable_target_norms_in_phase1"
        log(f"  SKIP: <2 viable target_norms after Phase 1")
        return manifest

    log(f"  Phase 1: {len(viable_target_norms)}/{len(target_norms)} viable: "
        f"{viable_target_norms}")

    # Step 3: Phase 2 — joint (k, target_norm) sweep
    log(f"  Phase 2: joint (k, target_norm) sweep...")
    k_steps = build_k_steps(len(s_marking_features))

    phase2_grid = run_phase2(
        s_marking_features, sae_cache, baselines_stereo, stereo_items,
        steerer, k_steps, viable_target_norms, dtype,
        output_dir, cat, sub, injection_layer,
        optimizer_metric=args.optimizer_metric,
    )
    manifest["phase2_grid"] = phase2_grid

    # Step 4: Phase 3 — tiered optimal selection
    optimal_tiered = select_optimal_tiered(
        phase2_grid,
        max_degen=args.max_degeneration_rate,
        max_corrupt_relaxed=args.max_corruption_rate,
        max_corrupt_strict=args.max_corruption_rate_strict,
        optimizer_metric=args.optimizer_metric,
        metric_floor=args.mwcs_floor,
    )

    optimal = optimal_tiered["relaxed"]

    if optimal is None:
        manifest["steering_viable"] = False

        # Diagnose the failure mode
        any_n_elig = any(
            r["metrics"]["rcr_1.0"]["n_eligible"] >= args.min_n_eligible
            for r in phase2_grid
        )
        any_rcr = any(r["metrics"]["rcr_1.0"]["rcr"] > 0 for r in phase2_grid)
        any_low_degen = any(
            r["degeneration_rate"] < args.max_degeneration_rate
            for r in phase2_grid
        )
        any_low_corrupt = any(
            r["corruption_rate"] < args.max_corruption_rate
            for r in phase2_grid
        )

        if not any_n_elig:
            reason = f"no_config_with_n_eligible>={args.min_n_eligible}"
        elif not any_rcr:
            reason = "rcr_zero_everywhere"
        elif not any_low_degen:
            reason = f"all_configs_degenerate>={args.max_degeneration_rate}"
        elif not any_low_corrupt:
            reason = f"all_configs_corrupt>={args.max_corruption_rate}"
        else:
            reason = "no_config_passing_all_constraints"

        manifest["steering_skip_reason"] = reason
        manifest["optimal_relaxed"] = None
        manifest["optimal_strict"] = None
        log(f"  SKIP: {reason}")
        return manifest

    manifest["steering_viable"] = True
    manifest["optimal_config_tier"] = "relaxed"
    manifest["optimal_k"] = optimal["k"]
    manifest["optimal_target_norm"] = optimal["target_norm"]
    manifest["optimal_alpha"] = optimal["alpha"]
    manifest["optimal_vector_norm"] = optimal["vector_norm"]
    manifest["optimal_eta"] = optimal["eta"]
    manifest["optimal_rcr_1.0"] = optimal["metrics"]["rcr_1.0"]["rcr"]
    manifest["optimal_rcr_1.0_n_eligible"] = optimal["metrics"]["rcr_1.0"]["n_eligible"]
    manifest["optimal_rcr_0.5"] = optimal["metrics"]["rcr_0.5"]["rcr"]
    manifest["optimal_rcr_2.0"] = optimal["metrics"]["rcr_2.0"]["rcr"]
    manifest["optimal_mwcs_0.5"] = optimal["metrics"]["mwcs_0.5"]["mwcs"]
    manifest["optimal_mwcs_1.0"] = optimal["metrics"]["mwcs_1.0"]["mwcs"]
    manifest["optimal_mwcs_2.0"] = optimal["metrics"]["mwcs_2.0"]["mwcs"]
    manifest["optimizer_metric"] = args.optimizer_metric
    manifest["optimal_logit_shift"] = optimal["metrics"]["logit_shift"]
    manifest["optimal_degeneration_rate"] = optimal["degeneration_rate"]
    manifest["optimal_corruption_rate"] = optimal["corruption_rate"]
    manifest["optimal_features"] = s_marking_features[:optimal["k"]]

    # Report strict tier for comparison
    if optimal_tiered["strict"] is not None:
        strict = optimal_tiered["strict"]
        manifest["optimal_strict"] = {
            "k": strict["k"], "target_norm": strict["target_norm"],
            "alpha": strict["alpha"], "vector_norm": strict["vector_norm"],
            "eta": strict["eta"],
            "rcr_1.0": strict["metrics"]["rcr_1.0"]["rcr"],
            "rcr_1.0_n_eligible": strict["metrics"]["rcr_1.0"]["n_eligible"],
            "degeneration_rate": strict["degeneration_rate"],
            "corruption_rate": strict["corruption_rate"],
            "matches_relaxed": (
                strict["k"] == optimal["k"]
                and strict["target_norm"] == optimal["target_norm"]
            ),
        }
    else:
        manifest["optimal_strict"] = None

    log(f"  OPTIMAL (relaxed, corrupt<{args.max_corruption_rate}): "
        f"k={optimal['k']} τ={optimal['target_norm']} α={optimal['alpha']:.2f}")
    log(f"    η={optimal['eta']:.3f} RCR₁.₀={optimal['metrics']['rcr_1.0']['rcr']:.3f} "
        f"(n_eligible={optimal['metrics']['rcr_1.0']['n_eligible']})")
    log(f"    ||v||={optimal['vector_norm']:.3f} "
        f"degen={optimal['degeneration_rate']:.3f} "
        f"corrupt={optimal['corruption_rate']:.3f}")
    if optimal_tiered["strict"] is not None:
        s = optimal_tiered["strict"]
        if s["k"] == optimal["k"] and s["target_norm"] == optimal["target_norm"]:
            log(f"    (strict tier matches relaxed)")
        else:
            log(f"  OPTIMAL (strict, corrupt<{args.max_corruption_rate_strict}): "
                f"k={s['k']} τ={s['target_norm']} "
                f"RCR₁.₀={s['metrics']['rcr_1.0']['rcr']:.3f}")
    else:
        log(f"  No config passes strict corrupt<{args.max_corruption_rate_strict}")

    # Step 5: Marginal analysis
    manifest["marginal_analysis"] = compute_marginal_analysis(
        phase2_grid, optimal,
    )

    # Step 6: Save optimal steering vector
    vec, _ = build_subgroup_steering_vector(
        s_marking_features[:optimal["k"]],
        sae_cache, optimal["alpha"],
        device=device, dtype=dtype,
    )
    save_steering_vector(
        output_dir, cat, sub, vec, optimal,
        s_marking_features, injection_layer,
    )

    # Step 7: Exacerbation test
    exac_result = None
    if not args.skip_exacerbation:
        log(f"  Step 7: exacerbation test at "
            f"+|optimal_target_norm|={abs(optimal['target_norm'])}...")
        exac_result = run_exacerbation_test(
            optimal, s_marking_features, sae_cache,
            steerer, stereo_items, non_stereo_items, baselines_stereo,
            dtype, output_dir, cat, sub,
        )
        manifest["exacerbation"] = exac_result

    # Step 8: Save per-item parquet
    save_per_item_parquet(output_dir, cat, sub, optimal, exac_result)

    # Placeholders for C3 (filled in later)
    for key in [
        "medqa_matched_delta", "medqa_within_cat_mismatched_delta",
        "medqa_cross_cat_mismatched_delta", "medqa_nodemo_delta",
        "medqa_exacerbation_matched_delta",
        "mmlu_delta", "mmlu_worst_subject", "mmlu_worst_subject_delta",
    ]:
        manifest[key] = None

    return manifest


# ---------------------------------------------------------------------------
# Top-level output saving
# ---------------------------------------------------------------------------

def save_top_level_outputs(
    output_dir: Path,
    all_manifests: list[dict[str, Any]],
    all_phase1_results: dict[str, Any],
    all_grid_records: list[dict[str, Any]],
    runtime_seconds: float,
) -> None:
    """Save all top-level C1 output files."""
    # steering_manifests.json
    atomic_save_json(all_manifests, output_dir / "steering_manifests.json")
    log(f"\nSaved steering_manifests.json ({len(all_manifests)} subgroups)")

    # phase1_results.json
    atomic_save_json(all_phase1_results, output_dir / "phase1_results.json")

    # optimal_configs.json — nested {cat: {sub: config}}
    optimal_configs: dict[str, dict[str, Any]] = {}
    for m in all_manifests:
        if m.get("steering_viable"):
            cat = m["category"]
            sub = m["subgroup"]
            optimal_configs.setdefault(cat, {})[sub] = {
                "k": m["optimal_k"],
                "target_norm": m["optimal_target_norm"],
                "alpha": m["optimal_alpha"],
                "injection_layer": m.get("injection_layer"),
                "eta": m["optimal_eta"],
                "vector_norm": m["optimal_vector_norm"],
                "rcr_1.0": m["optimal_rcr_1.0"],
            }
    atomic_save_json(optimal_configs, output_dir / "optimal_configs.json")

    # marginal_analysis.json
    marginal_by_sub: dict[str, Any] = {}
    for m in all_manifests:
        if m.get("marginal_analysis"):
            marginal_by_sub[f"{m['category']}/{m['subgroup']}"] = m["marginal_analysis"]
    atomic_save_json(marginal_by_sub, output_dir / "marginal_analysis.json")

    # phase2_grid.parquet — flat grid of all Phase 2 configurations
    if all_grid_records:
        flat_rows = []
        for r in all_grid_records:
            flat_rows.append({
                "category": r["category"],
                "subgroup": r["subgroup"],
                "k": r["k"],
                "target_norm": r["target_norm"],
                "alpha": r["alpha"],
                "injection_layer": r["injection_layer"],
                "vector_norm": r["vector_norm"],
                "eta": r["eta"],
                "rcr_0.5": r["metrics"]["rcr_0.5"]["rcr"],
                "rcr_0.5_n_eligible": r["metrics"]["rcr_0.5"]["n_eligible"],
                "rcr_1.0": r["metrics"]["rcr_1.0"]["rcr"],
                "rcr_1.0_n_eligible": r["metrics"]["rcr_1.0"]["n_eligible"],
                "rcr_2.0": r["metrics"]["rcr_2.0"]["rcr"],
                "rcr_2.0_n_eligible": r["metrics"]["rcr_2.0"]["n_eligible"],
                "mwcs_1.0": r["metrics"]["mwcs_1.0"]["mwcs"],
                "mean_logit_shift": r["metrics"]["logit_shift"]["mean_shift"],
                "degeneration_rate": r["degeneration_rate"],
                "corruption_rate": r["corruption_rate"],
                "n_items": r["n_items"],
            })
        pd.DataFrame(flat_rows).to_parquet(
            output_dir / "phase2_grid.parquet", index=False, compression="snappy",
        )

    # c1_summary.json
    viable = [m for m in all_manifests if m.get("steering_viable")]
    skip_reasons: dict[str, int] = Counter()
    for m in all_manifests:
        reason = m.get("steering_skip_reason")
        if reason:
            skip_reasons[reason] += 1

    k_dist: dict[str, int] = Counter()
    for m in viable:
        k_dist[str(m["optimal_k"])] += 1

    summary = {
        "n_subgroups_processed": len(all_manifests),
        "n_subgroups_viable": len(viable),
        "n_subgroups_skipped": len(all_manifests) - len(viable),
        "skip_reasons": dict(skip_reasons),
        "optimal_k_distribution": dict(k_dist),
        "median_optimal_k": (
            float(np.median([m["optimal_k"] for m in viable]))
            if viable else None
        ),
        "median_optimal_eta": (
            round(float(np.median([m["optimal_eta"] for m in viable])), 4)
            if viable else None
        ),
        "median_optimal_rcr_1.0": (
            round(float(np.median([m["optimal_rcr_1.0"] for m in viable])), 4)
            if viable else None
        ),
        "runtime_seconds": round(runtime_seconds, 1),
    }
    atomic_save_json(summary, output_dir / "c1_summary.json")
    log(f"Saved c1_summary.json")

    # Print summary
    log(f"\n{'=' * 60}")
    log(f"C1 Summary")
    log(f"{'=' * 60}")
    log(f"  Viable: {len(viable)}/{len(all_manifests)}")
    log(f"  Skip reasons: {dict(skip_reasons)}")
    if viable:
        log(f"  Median optimal k: {summary['median_optimal_k']}")
        log(f"  Median η: {summary['median_optimal_eta']}")
        log(f"  Median RCR₁.₀: {summary['median_optimal_rcr_1.0']}")
    log(f"  Runtime: {runtime_seconds:.0f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    t0 = time.time()

    run_dir = Path(args.run_dir)
    config = load_config(run_dir)

    device = torch.device(config["device"])
    dtype = getattr(torch, config.get("dtype", "float16"))

    # Parse target norms
    if args.target_norms:
        target_norms = [float(x) for x in args.target_norms.split(",")]
    else:
        target_norms = DEFAULT_TARGET_NORMS

    log(f"C1 Steering Optimisation")
    log(f"  run_dir: {run_dir}")
    log(f"  device: {device}")
    log(f"  dtype: {dtype}")
    log(f"  target_norms: {target_norms}")
    if args.max_items:
        log(f"  max_items: {args.max_items}")

    # Load model
    log("\nLoading model...")
    wrapper = ModelWrapper.from_pretrained(config["model_path"], device=str(device))

    # Load B2 outputs
    ranked_df = load_ranked_features(run_dir)
    injection_layers = load_injection_layers(run_dir)

    # Load B5 artifact flags
    flagged_set = load_artifact_flags(run_dir)
    ranked_df = filter_flagged_features(ranked_df, flagged_set)

    # Load metadata
    metadata_df = load_metadata(run_dir)

    # Determine subgroups to process
    subgroups_to_process = determine_subgroups(
        ranked_df, injection_layers,
        filter_categories=args.categories,
        filter_subgroups=args.subgroups,
    )

    # Pre-load SAE layers for the full injection layer range
    sae_cache: dict[int, SAEWrapper] = {}
    for layer in range(args.injection_layer_min, args.injection_layer_max + 1):
        log(f"  Loading SAE for layer {layer}...")
        sae_cache[layer] = SAEWrapper(
            config["sae_source"],
            layer=layer,
            expansion=config.get("sae_expansion", 32),
            device=str(device),
        )

    # Output directory structure
    output_dir = run_dir / "C_steering"
    ensure_dir(output_dir / "checkpoints")
    ensure_dir(output_dir / "vectors")
    ensure_dir(output_dir / "per_item")
    ensure_dir(output_dir / "figures")

    # Process each subgroup
    all_manifests: list[dict[str, Any]] = []
    all_phase1_results: dict[str, Any] = {}
    all_grid_records: list[dict[str, Any]] = []

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

    # Save top-level outputs
    runtime = time.time() - t0
    save_top_level_outputs(
        output_dir, all_manifests, all_phase1_results,
        all_grid_records, runtime,
    )

    # Generate figures
    if not args.skip_figures:
        try:
            from src.visualization.steering_figures import generate_c1_figures
            generate_c1_figures(output_dir, all_manifests, all_grid_records)
        except ImportError:
            log("WARNING: steering_figures module not available; skipping figures")
        except Exception as e:
            log(f"WARNING: figure generation failed: {e}")

    log(f"\nC1 complete in {runtime:.1f}s")
    log(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
