"""C2: Cross-subgroup transfer & universal backfire prediction.

Tests whether subgroup fragmentation is causally operative: when you apply
subgroup A's steering vector to items targeting subgroup B, does the bias
change depend on the cosine similarity between A's and B's identity
representations?  The core product is the **universal backfire scatter** — the
headline figure of the paper.

Usage:
    python scripts/C2_transfer.py --run_dir runs/llama-3.1-8b_2026-04-15/

    # Specific categories
    python scripts/C2_transfer.py --run_dir runs/llama-3.1-8b_2026-04-15/ --categories so,race,religion

    # Quick test
    python scripts/C2_transfer.py --run_dir runs/llama-3.1-8b_2026-04-15/ --max_items 20

    # Skip sensitivity analyses
    python scripts/C2_transfer.py --run_dir runs/llama-3.1-8b_2026-04-15/ --primary_only
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import linregress, spearmanr

from src.metrics.bias_metrics import build_result_dict, compute_all_metrics
from src.models.wrapper import ModelWrapper
from src.sae.wrapper import SAEWrapper
from src.sae_localization.steering import SAESteerer
from src.utils.config import load_config
from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MIN_N_PER_TARGET = 10


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="C2: Cross-subgroup transfer & universal backfire prediction",
    )
    p.add_argument("--run_dir", required=True, type=str)
    p.add_argument("--categories", type=str, default=None,
                   help="Comma-separated category filter")
    p.add_argument("--max_items", type=int, default=None,
                   help="Max items per target subgroup (for quick tests)")
    p.add_argument("--min_n_per_target", type=int, default=DEFAULT_MIN_N_PER_TARGET,
                   help="Minimum items per target subgroup")
    p.add_argument("--primary_only", action="store_true",
                   help="Skip stable-range sensitivity analysis")
    p.add_argument("--skip_figures", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_viable_manifests(
    run_dir: Path,
    category_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Load C1 steering manifests, filter to viable subgroups."""
    with open(run_dir / "C_steering" / "steering_manifests.json") as f:
        manifests = json.load(f)
    viable = [m for m in manifests if m.get("steering_viable")]
    if category_filter:
        cats = set(category_filter.split(","))
        viable = [m for m in viable if m["category"] in cats]
    log(f"Loaded {len(viable)} viable subgroups (of {len(manifests)} total)")
    return viable


def load_steering_vector(
    run_dir: Path, cat: str, sub: str, device: torch.device, dtype: torch.dtype,
) -> tuple[torch.Tensor, int]:
    """Load pre-computed steering vector from C1."""
    data = np.load(run_dir / "C_steering" / "vectors" / f"{cat}_{sub}.npz")
    vec = torch.from_numpy(data["vector"]).to(device=device, dtype=dtype)
    injection_layer = int(data["injection_layer"])
    return vec, injection_layer


def load_metadata(run_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(run_dir / "A_extraction" / "metadata.parquet")


def load_stimuli(run_dir: Path, category: str) -> list[dict[str, Any]]:
    with open(run_dir / "A_extraction" / "stimuli" / f"{category}.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Step 2: Primary cosines (DIM identity_normed)
# ---------------------------------------------------------------------------

def get_primary_cosines(
    run_dir: Path,
    viable_manifests: list[dict[str, Any]],
) -> pd.DataFrame:
    """Build primary cosine table from B3 output.

    Uses DIM identity_normed direction cosines at each category's peak
    differentiation layer.
    """
    cosine_path = run_dir / "B_geometry" / "cosine_pairs.parquet"
    diff_path = run_dir / "B_geometry" / "differentiation_metrics.json"

    if not cosine_path.exists() or not diff_path.exists():
        log("WARNING: B3 geometry outputs not found; primary cosines unavailable")
        return pd.DataFrame()

    cosine_df = pd.read_parquet(cosine_path)
    with open(diff_path) as f:
        diff_metrics = json.load(f)

    rows: list[dict[str, Any]] = []
    for cat, cat_metrics in diff_metrics.items():
        if "identity_normed" not in cat_metrics:
            continue
        peak_layer = cat_metrics["identity_normed"]["peak_layer"]

        cat_cosines = cosine_df[
            (cosine_df["category"] == cat)
            & (cosine_df["direction_type"] == "identity_normed")
            & (cosine_df["layer"] == peak_layer)
        ]

        for _, row in cat_cosines.iterrows():
            rows.append({
                "category": cat,
                "subgroup_A": row["subgroup_A"],
                "subgroup_B": row["subgroup_B"],
                "peak_layer": int(peak_layer),
                "cosine_dim_identity_normed": float(row["cosine"]),
            })

    df = pd.DataFrame(rows)
    log(f"Loaded {len(df)} primary DIM cosine pairs")
    return df


# ---------------------------------------------------------------------------
# Step 3: Secondary cosines (SAE steering vectors)
# ---------------------------------------------------------------------------

def get_sae_cosines(
    run_dir: Path,
    viable_manifests: list[dict[str, Any]],
    device: torch.device,
    dtype: torch.dtype,
) -> pd.DataFrame:
    """Compute pairwise cosines between SAE steering vector directions."""
    by_cat: dict[str, list[str]] = {}
    for m in viable_manifests:
        by_cat.setdefault(m["category"], []).append(m["subgroup"])

    sae_directions: dict[tuple[str, str], np.ndarray] = {}
    for m in viable_manifests:
        vec, _ = load_steering_vector(run_dir, m["category"], m["subgroup"],
                                      device, dtype)
        vec_np = vec.float().cpu().numpy()
        norm = np.linalg.norm(vec_np)
        if norm > 1e-8:
            sae_directions[(m["category"], m["subgroup"])] = vec_np / norm

    rows: list[dict[str, Any]] = []
    for cat, subs in by_cat.items():
        subs_sorted = sorted(subs)
        for i, sub_a in enumerate(subs_sorted):
            for sub_b in subs_sorted[i + 1:]:
                key_a = (cat, sub_a)
                key_b = (cat, sub_b)
                if key_a not in sae_directions or key_b not in sae_directions:
                    continue
                cos = float(np.dot(sae_directions[key_a], sae_directions[key_b]))
                rows.append({
                    "category": cat,
                    "subgroup_A": sub_a,
                    "subgroup_B": sub_b,
                    "cosine_sae_steering": round(cos, 6),
                })

    df = pd.DataFrame(rows)
    log(f"Computed {len(df)} SAE cosine pairs")
    return df


# ---------------------------------------------------------------------------
# Step 4: Cross-subgroup steering transfer
# ---------------------------------------------------------------------------

def run_transfer_evaluation(
    viable_manifests: list[dict[str, Any]],
    metadata_df: pd.DataFrame,
    run_dir: Path,
    output_dir: Path,
    wrapper: ModelWrapper,
    sae_cache: dict[int, SAEWrapper],
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """For each (source, target) pair within a category, run cross-subgroup
    steering and measure bias change.  Resume-safe with per-pair checkpoints.
    """
    transfer_dir = ensure_dir(output_dir / "per_pair_checkpoints")
    per_pair_parquet_dir = ensure_dir(output_dir / "per_pair")

    by_cat: dict[str, list[dict[str, Any]]] = {}
    for m in viable_manifests:
        by_cat.setdefault(m["category"], []).append(m)

    all_records: list[dict[str, Any]] = []

    for cat, cat_manifests in by_cat.items():
        log(f"\n{'=' * 60}")
        log(f"Category: {cat} ({len(cat_manifests)} viable subgroups)")
        log(f"{'=' * 60}")

        subs_in_cat = sorted(m["subgroup"] for m in cat_manifests)
        if len(subs_in_cat) < 2:
            log(f"  SKIP: fewer than 2 viable subgroups")
            continue

        # Load stimuli and metadata for this category
        cat_stimuli = load_stimuli(run_dir, cat)
        stim_by_idx = {s["item_idx"]: s for s in cat_stimuli}
        cat_meta = metadata_df[metadata_df["category"] == cat]

        for source_m in cat_manifests:
            source_sub = source_m["subgroup"]

            source_vec, source_inj_layer = load_steering_vector(
                run_dir, cat, source_sub, device, dtype,
            )

            # Ensure SAE is loaded for this layer
            if source_inj_layer not in sae_cache:
                sae_cache[source_inj_layer] = SAEWrapper(
                    load_config(run_dir)["sae_source"],
                    layer=source_inj_layer,
                    expansion=load_config(run_dir).get("sae_expansion", 32),
                    device=str(device),
                )

            source_steerer = SAESteerer(
                wrapper, sae_cache[source_inj_layer], source_inj_layer,
            )

            for target_sub in subs_in_cat:
                is_self = source_sub == target_sub

                ckpt_path = transfer_dir / f"{cat}_{source_sub}_to_{target_sub}.json"
                if ckpt_path.exists():
                    with open(ckpt_path) as f:
                        record = json.load(f)
                    all_records.append(record)
                    log(f"  {source_sub}→{target_sub}: LOADED from checkpoint")
                    continue

                # Get ALL ambig items targeting target_sub
                target_ambig_idxs = []
                for _, meta_row in cat_meta.iterrows():
                    if meta_row.get("context_condition") != "ambig":
                        continue
                    stereo_groups = meta_row.get("stereotyped_groups", "[]")
                    if isinstance(stereo_groups, str):
                        stereo_groups = json.loads(stereo_groups)
                    if target_sub in stereo_groups:
                        target_ambig_idxs.append(int(meta_row["item_idx"]))

                if args.max_items:
                    target_ambig_idxs = target_ambig_idxs[:args.max_items]

                n_items = len(target_ambig_idxs)
                if n_items < args.min_n_per_target:
                    log(f"  {source_sub}→{target_sub}: SKIP "
                        f"(n={n_items} < {args.min_n_per_target})")
                    continue

                log(f"  {source_sub}→{target_sub}: {n_items} items, "
                    f"inj_layer={source_inj_layer}")

                per_item_results: list[dict[str, Any]] = []
                for item_idx in target_ambig_idxs:
                    stim = stim_by_idx.get(item_idx)
                    if stim is None:
                        continue

                    baseline = source_steerer.evaluate_baseline(stim["prompt"])
                    steered = source_steerer.steer_and_evaluate(
                        stim["prompt"], source_vec,
                    )

                    item_for_result = {
                        "item_idx": item_idx,
                        "prompt": stim["prompt"],
                        "stereotyped_option": stim.get("stereotyped_option"),
                    }
                    r = build_result_dict(
                        item_for_result, baseline, steered, source_vec,
                    )
                    r["target_subgroup"] = target_sub
                    r["source_subgroup"] = source_sub
                    r["category"] = cat

                    # Was the baseline response stereotyped?
                    meta_match = cat_meta[cat_meta["item_idx"] == item_idx]
                    if len(meta_match) > 0:
                        r["baseline_is_stereotyped_response"] = bool(
                            meta_match.iloc[0]["model_answer_role"]
                            == "stereotyped_target"
                        )
                    else:
                        r["baseline_is_stereotyped_response"] = False

                    per_item_results.append(r)

                # Compute bias_change: change in stereotyped_response rate
                n_baseline_stereo = sum(
                    1 for r in per_item_results
                    if r["baseline_role"] == "stereotyped_target"
                )
                n_steered_stereo = sum(
                    1 for r in per_item_results
                    if r["steered_role"] == "stereotyped_target"
                )
                n_total = max(len(per_item_results), 1)
                baseline_stereo_rate = n_baseline_stereo / n_total
                steered_stereo_rate = n_steered_stereo / n_total
                bias_change = steered_stereo_rate - baseline_stereo_rate

                # Continuous metric: mean logit shift of stereotyped option
                logit_shifts: list[float] = []
                for r in per_item_results:
                    opt = r.get("stereotyped_option")
                    if (opt and opt in r.get("logit_baseline", {})
                            and opt in r.get("logit_steered", {})):
                        logit_shifts.append(
                            r["logit_steered"][opt] - r["logit_baseline"][opt]
                        )
                mean_logit_shift = (
                    float(np.mean(logit_shifts)) if logit_shifts else None
                )

                # RCR on stereo-baseline subset
                stereo_subset = [
                    r for r in per_item_results
                    if r["baseline_role"] == "stereotyped_target"
                ]
                metrics = (
                    compute_all_metrics(stereo_subset) if stereo_subset else None
                )

                n_degen = sum(
                    1 for r in per_item_results if r.get("degenerated", False)
                )
                degen_rate = n_degen / n_total

                record: dict[str, Any] = {
                    "category": cat,
                    "source_subgroup": source_sub,
                    "target_subgroup": target_sub,
                    "is_self": is_self,
                    "n_items": len(per_item_results),
                    "source_injection_layer": source_inj_layer,
                    "baseline_stereotyped_rate": round(baseline_stereo_rate, 4),
                    "steered_stereotyped_rate": round(steered_stereo_rate, 4),
                    "bias_change": round(bias_change, 4),
                    "mean_logit_shift_stereotyped_option": (
                        round(mean_logit_shift, 4)
                        if mean_logit_shift is not None else None
                    ),
                    "degeneration_rate": round(degen_rate, 4),
                    "rcr_1.0": (
                        round(metrics["rcr_1.0"]["rcr"], 4) if metrics else None
                    ),
                    "rcr_1.0_n_eligible": (
                        metrics["rcr_1.0"]["n_eligible"] if metrics else 0
                    ),
                }

                # Save checkpoint (without per-item results for disk)
                atomic_save_json(record, ckpt_path)
                all_records.append(record)

                # Save per-pair per-item parquet
                _save_per_pair_parquet(
                    per_item_results, per_pair_parquet_dir,
                    cat, source_sub, target_sub,
                )

                bc_str = f"{bias_change:+.3f}"
                ls_str = (
                    f"{mean_logit_shift:+.3f}"
                    if mean_logit_shift is not None else "n/a"
                )
                rcr_str = (
                    f"{metrics['rcr_1.0']['rcr']:.3f}" if metrics else "n/a"
                )
                log(f"    bias_change={bc_str}, logit_shift={ls_str}, "
                    f"RCR₁.₀={rcr_str}, degen={degen_rate:.3f}")

    return pd.DataFrame(all_records)


def _save_per_pair_parquet(
    per_item_results: list[dict[str, Any]],
    output_dir: Path,
    cat: str,
    source_sub: str,
    target_sub: str,
) -> None:
    """Save per-pair per-item results as parquet."""
    if not per_item_results:
        return

    rows = []
    for r in per_item_results:
        row: dict[str, Any] = {
            "item_idx": r["item_idx"],
            "category": cat,
            "source_subgroup": source_sub,
            "target_subgroup": target_sub,
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
            "baseline_is_stereotyped_response": r.get(
                "baseline_is_stereotyped_response", False,
            ),
        }
        for letter in ("A", "B", "C"):
            row[f"logit_baseline_{letter}"] = r["logit_baseline"].get(letter)
            row[f"logit_steered_{letter}"] = r["logit_steered"].get(letter)
        rows.append(row)

    df = pd.DataFrame(rows)
    path = output_dir / f"{cat}_{source_sub}_to_{target_sub}.parquet"
    df.to_parquet(path, index=False, compression="snappy")


# ---------------------------------------------------------------------------
# Step 5: Assemble scatter data
# ---------------------------------------------------------------------------

def build_scatter_data(
    transfer_df: pd.DataFrame,
    primary_cosines: pd.DataFrame,
    sae_cosines: pd.DataFrame,
) -> pd.DataFrame:
    """Merge transfer results with cosines to produce scatter plot data."""
    rows: list[dict[str, Any]] = []

    for _, t in transfer_df.iterrows():
        cat = t["category"]
        src = t["source_subgroup"]
        tgt = t["target_subgroup"]

        # Look up DIM cosine (pairs are stored alphabetically)
        cos_dim = _lookup_cosine(primary_cosines, cat, src, tgt,
                                  "cosine_dim_identity_normed")
        peak_layer = _lookup_cosine(primary_cosines, cat, src, tgt,
                                     "peak_layer")

        # Self-pair: cosine = 1.0 by definition
        if t["is_self"] and cos_dim is None:
            cos_dim = 1.0

        # SAE cosine
        cos_sae = _lookup_cosine(sae_cosines, cat, src, tgt,
                                  "cosine_sae_steering")
        if t["is_self"] and cos_sae is None:
            cos_sae = 1.0

        rows.append({
            "category": cat,
            "source_subgroup": src,
            "target_subgroup": tgt,
            "is_self": t["is_self"],
            "cosine_dim_identity_normed": cos_dim,
            "cosine_dim_peak_layer": (
                int(peak_layer) if peak_layer is not None else None
            ),
            "cosine_sae_steering": cos_sae,
            "bias_change": t["bias_change"],
            "baseline_stereotyped_rate": t["baseline_stereotyped_rate"],
            "steered_stereotyped_rate": t["steered_stereotyped_rate"],
            "mean_logit_shift": t["mean_logit_shift_stereotyped_option"],
            "degeneration_rate": t["degeneration_rate"],
            "rcr_1.0": t["rcr_1.0"],
            "n_items": t["n_items"],
        })

    return pd.DataFrame(rows)


def _lookup_cosine(
    cosine_df: pd.DataFrame,
    cat: str,
    sub_a: str,
    sub_b: str,
    col: str,
) -> float | None:
    """Look up a cosine value regardless of A/B ordering."""
    if cosine_df.empty:
        return None

    fwd = cosine_df[
        (cosine_df["category"] == cat)
        & (cosine_df["subgroup_A"] == sub_a)
        & (cosine_df["subgroup_B"] == sub_b)
    ]
    if len(fwd) > 0:
        return float(fwd[col].iloc[0])

    rev = cosine_df[
        (cosine_df["category"] == cat)
        & (cosine_df["subgroup_A"] == sub_b)
        & (cosine_df["subgroup_B"] == sub_a)
    ]
    if len(rev) > 0:
        return float(rev[col].iloc[0])

    return None


# ---------------------------------------------------------------------------
# Step 6: Regression analysis
# ---------------------------------------------------------------------------

def do_regression(
    x: np.ndarray, y: np.ndarray, label: str = "",
) -> dict[str, Any]:
    """Linear regression with bootstrap CI."""
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    rng = np.random.default_rng(42)
    boot_slopes: list[float] = []
    boot_intercepts: list[float] = []
    n = len(x)
    for _ in range(1000):
        idx = rng.choice(n, size=n, replace=True)
        x_boot, y_boot = x[idx], y[idx]
        # Skip degenerate resamples where all x values are identical
        if np.ptp(x_boot) < 1e-12:
            continue
        s, i, _, _, _ = linregress(x_boot, y_boot)
        boot_slopes.append(s)
        boot_intercepts.append(i)

    # Fall back if too many resamples were degenerate
    if not boot_slopes:
        boot_slopes = [slope]
        boot_intercepts = [intercept]

    return {
        "label": label,
        "n": int(n),
        "slope": float(slope),
        "intercept": float(intercept),
        "r_value": float(r_value),
        "r_squared": float(r_value ** 2),
        "p_value": float(p_value),
        "std_err": float(std_err),
        "slope_ci_95": [
            float(np.percentile(boot_slopes, 2.5)),
            float(np.percentile(boot_slopes, 97.5)),
        ],
        "intercept_ci_95": [
            float(np.percentile(boot_intercepts, 2.5)),
            float(np.percentile(boot_intercepts, 97.5)),
        ],
    }


def run_regression_analyses(scatter_df: pd.DataFrame) -> dict[str, Any]:
    """Run multiple regression analyses."""
    non_self = scatter_df[~scatter_df["is_self"]].copy()
    non_self = non_self.dropna(subset=["cosine_dim_identity_normed", "bias_change"])

    results: dict[str, Any] = {
        "n_total_non_self_pairs": int(len(non_self)),
    }

    # Primary: DIM cosine, all categories
    if len(non_self) >= 5:
        results["primary_dim_all"] = do_regression(
            non_self["cosine_dim_identity_normed"].values,
            non_self["bias_change"].values,
            label="DIM cosine, all categories",
        )

    # Sensitivity: exclude disability
    no_dis = non_self[non_self["category"] != "disability"]
    if len(no_dis) >= 5:
        results["sensitivity_dim_no_disability"] = do_regression(
            no_dis["cosine_dim_identity_normed"].values,
            no_dis["bias_change"].values,
            label="DIM cosine, excluding disability",
        )

    # Secondary: SAE cosine
    has_sae = non_self.dropna(subset=["cosine_sae_steering"])
    if len(has_sae) >= 5:
        results["secondary_sae_all"] = do_regression(
            has_sae["cosine_sae_steering"].values,
            has_sae["bias_change"].values,
            label="SAE cosine, all categories",
        )
        has_sae_no_dis = has_sae[has_sae["category"] != "disability"]
        if len(has_sae_no_dis) >= 5:
            results["secondary_sae_no_disability"] = do_regression(
                has_sae_no_dis["cosine_sae_steering"].values,
                has_sae_no_dis["bias_change"].values,
                label="SAE cosine, excluding disability",
            )

    # Per-source-subgroup regressions
    per_source: dict[str, Any] = {}
    for (cat, src), grp in non_self.groupby(["category", "source_subgroup"]):
        key = f"{cat}/{src}"
        if len(grp) < 3:
            per_source[key] = {"n_pairs": len(grp), "status": "insufficient_data"}
            continue
        reg = do_regression(
            grp["cosine_dim_identity_normed"].values,
            grp["bias_change"].values,
            label=key,
        )
        reg["n_pairs"] = len(grp)
        per_source[key] = reg
    results["per_source_subgroup"] = per_source

    # Per-category regressions
    per_category: dict[str, Any] = {}
    for cat, grp in non_self.groupby("category"):
        if len(grp) < 4:
            per_category[cat] = {
                "n_pairs": len(grp), "status": "insufficient_data",
            }
            continue
        reg = do_regression(
            grp["cosine_dim_identity_normed"].values,
            grp["bias_change"].values,
            label=f"category={cat}",
        )
        reg["n_pairs"] = len(grp)
        per_category[cat] = reg
    results["per_category"] = per_category

    return results


# ---------------------------------------------------------------------------
# Step 7: SAE vs DIM cosine comparison
# ---------------------------------------------------------------------------

def compare_sae_vs_dim_cosines(scatter_df: pd.DataFrame) -> dict[str, Any]:
    """Correlate SAE cosines with DIM cosines across all pairs."""
    paired = scatter_df[~scatter_df["is_self"]].dropna(
        subset=["cosine_dim_identity_normed", "cosine_sae_steering"],
    )

    if len(paired) < 5:
        return {"status": "insufficient_data", "n_pairs": len(paired)}

    x = paired["cosine_dim_identity_normed"].values
    y = paired["cosine_sae_steering"].values

    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman_rho, spearman_p = spearmanr(x, y)

    return {
        "n_pairs": int(len(paired)),
        "pearson_r": round(pearson, 4),
        "spearman_rho": round(float(spearman_rho), 4),
        "spearman_p": float(spearman_p),
        "mean_dim_cosine": round(float(x.mean()), 4),
        "mean_sae_cosine": round(float(y.mean()), 4),
    }


# ---------------------------------------------------------------------------
# Step 8: Stable-range sensitivity analysis
# ---------------------------------------------------------------------------

def stable_range_sensitivity(
    run_dir: Path,
    transfer_df: pd.DataFrame,
) -> dict[str, Any]:
    """Re-run primary regression at each layer in B3's stable range."""
    cosine_path = run_dir / "B_geometry" / "cosine_pairs.parquet"
    diff_path = run_dir / "B_geometry" / "differentiation_metrics.json"

    if not cosine_path.exists() or not diff_path.exists():
        return {"status": "b3_outputs_missing"}

    cosine_df = pd.read_parquet(cosine_path)
    with open(diff_path) as f:
        diff_metrics = json.load(f)

    results: dict[str, Any] = {}

    for cat, cat_metrics in diff_metrics.items():
        if "identity_normed" not in cat_metrics:
            continue
        stable_range = cat_metrics["identity_normed"]["stable_range"]
        stable_start, stable_end = stable_range

        cat_transfers = transfer_df[
            (transfer_df["category"] == cat) & (~transfer_df["is_self"])
        ]
        if len(cat_transfers) < 4:
            continue

        per_layer: dict[str, Any] = {}
        for layer in range(stable_start, stable_end + 1):
            layer_cos = cosine_df[
                (cosine_df["category"] == cat)
                & (cosine_df["direction_type"] == "identity_normed")
                & (cosine_df["layer"] == layer)
            ]

            merged_rows: list[tuple[float, float]] = []
            for _, t in cat_transfers.iterrows():
                src, tgt = t["source_subgroup"], t["target_subgroup"]
                cos_fwd = layer_cos[
                    (layer_cos["subgroup_A"] == src)
                    & (layer_cos["subgroup_B"] == tgt)
                ]
                cos_rev = layer_cos[
                    (layer_cos["subgroup_A"] == tgt)
                    & (layer_cos["subgroup_B"] == src)
                ]
                if len(cos_fwd) > 0:
                    cos_val = float(cos_fwd["cosine"].iloc[0])
                elif len(cos_rev) > 0:
                    cos_val = float(cos_rev["cosine"].iloc[0])
                else:
                    continue
                merged_rows.append((cos_val, float(t["bias_change"])))

            if len(merged_rows) < 4:
                continue

            x = np.array([m[0] for m in merged_rows])
            y = np.array([m[1] for m in merged_rows])
            reg = do_regression(x, y, label=f"{cat} L{layer}")
            per_layer[str(layer)] = {
                "r_squared": reg["r_squared"],
                "slope": reg["slope"],
                "p_value": reg["p_value"],
                "n": reg["n"],
            }

        if per_layer:
            r2_values = [v["r_squared"] for v in per_layer.values()]
            results[cat] = {
                "stable_range": stable_range,
                "per_layer": per_layer,
                "r_squared_min": round(min(r2_values), 4),
                "r_squared_max": round(max(r2_values), 4),
                "r_squared_mean": round(float(np.mean(r2_values)), 4),
            }

    return results


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------

def save_all_outputs(
    output_dir: Path,
    transfer_df: pd.DataFrame,
    primary_cosines: pd.DataFrame,
    sae_cosines: pd.DataFrame,
    scatter_df: pd.DataFrame,
    regression_results: dict[str, Any],
    sae_dim_comparison: dict[str, Any],
    stable_range: dict[str, Any],
    viable_manifests: list[dict[str, Any]],
    runtime: float,
) -> None:
    """Save all C2 output files."""
    # transfer_pairs.parquet
    transfer_df.to_parquet(
        output_dir / "transfer_pairs.parquet", index=False, compression="snappy",
    )

    # cosines.parquet — merge DIM and SAE cosines
    if not primary_cosines.empty or not sae_cosines.empty:
        if not primary_cosines.empty and not sae_cosines.empty:
            cosines_merged = pd.merge(
                primary_cosines, sae_cosines,
                on=["category", "subgroup_A", "subgroup_B"],
                how="outer",
            )
        elif not primary_cosines.empty:
            cosines_merged = primary_cosines
        else:
            cosines_merged = sae_cosines
        cosines_merged.to_parquet(
            output_dir / "cosines.parquet", index=False, compression="snappy",
        )

    # scatter_data.parquet
    scatter_df.to_parquet(
        output_dir / "scatter_data.parquet", index=False, compression="snappy",
    )

    # regression_results.json
    atomic_save_json(regression_results, output_dir / "regression_results.json")

    # sae_vs_dim_comparison.json
    atomic_save_json(sae_dim_comparison, output_dir / "sae_vs_dim_comparison.json")

    # stable_range_sensitivity.json
    atomic_save_json(stable_range, output_dir / "stable_range_sensitivity.json")

    # c2_summary.json
    non_self_pairs = int(
        transfer_df[~transfer_df["is_self"]].shape[0]
        if not transfer_df.empty else 0
    )
    primary = regression_results.get("primary_dim_all", {})

    # Per-source r² median
    per_source = regression_results.get("per_source_subgroup", {})
    source_r2s = [
        v["r_squared"] for v in per_source.values()
        if isinstance(v, dict) and v.get("r_squared") is not None
    ]
    per_source_r2_median = (
        round(float(np.median(source_r2s)), 4) if source_r2s else None
    )

    # The regression's n_total_non_self_pairs counts pairs WITH valid
    # cosine+bias_change; the transfer_df non-self count is the total
    # evaluated.  Difference = pairs that lack cosine data.
    n_regression_pairs = regression_results.get("n_total_non_self_pairs", 0)

    summary = {
        "n_viable_subgroups_used": len(viable_manifests),
        "n_total_non_self_pairs": non_self_pairs,
        "n_pairs_skipped_insufficient_items": max(
            non_self_pairs - n_regression_pairs, 0,
        ),
        "primary_finding": {
            "r_squared": primary.get("r_squared"),
            "slope": primary.get("slope"),
            "p_value": primary.get("p_value"),
            "n": primary.get("n"),
        } if primary else None,
        "sae_vs_dim_pearson_r": sae_dim_comparison.get("pearson_r"),
        "per_source_subgroup_regression_r_squared_median": per_source_r2_median,
        "runtime_seconds": round(runtime, 1),
    }
    atomic_save_json(summary, output_dir / "c2_summary.json")

    log(f"\nSaved all C2 outputs to {output_dir}")
    if primary:
        log(f"  Primary regression: r²={primary['r_squared']:.3f}, "
            f"slope={primary['slope']:+.3f}, "
            f"p={primary['p_value']:.2e}, n={primary['n']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    t0 = time.time()

    run_dir = Path(args.run_dir)
    config = load_config(run_dir)

    device = torch.device(config["device"])
    dtype = getattr(torch, config["dtype"])

    log("C2 Cross-Subgroup Transfer & Universal Backfire Prediction")
    log(f"  run_dir: {run_dir}")
    log(f"  device: {device}")

    # Load model
    log("\nLoading model...")
    wrapper = ModelWrapper.from_pretrained(config["model_path"], device=str(device))

    # Load viable manifests
    viable_manifests = load_viable_manifests(run_dir, args.categories)

    # Identify needed SAE layers and load
    needed_layers = set()
    for m in viable_manifests:
        vec_path = run_dir / "C_steering" / "vectors" / f"{m['category']}_{m['subgroup']}.npz"
        if vec_path.exists():
            data = np.load(vec_path)
            needed_layers.add(int(data["injection_layer"]))

    sae_cache: dict[int, SAEWrapper] = {}
    for layer in sorted(needed_layers):
        log(f"  Loading SAE for layer {layer}...")
        sae_cache[layer] = SAEWrapper(
            config["sae_source"],
            layer=layer,
            expansion=config.get("sae_expansion", 32),
            device=str(device),
        )

    # Output directory
    output_dir = run_dir / "C_transfer"
    ensure_dir(output_dir / "per_pair_checkpoints")
    ensure_dir(output_dir / "per_pair")
    ensure_dir(output_dir / "figures")

    # Load metadata
    metadata_df = load_metadata(run_dir)

    # Step 2: Primary cosines (DIM)
    log("\nStep 2: Computing primary DIM cosines...")
    primary_cosines = get_primary_cosines(run_dir, viable_manifests)

    # Step 3: Secondary cosines (SAE)
    log("Step 3: Computing SAE steering vector cosines...")
    sae_cosines = get_sae_cosines(run_dir, viable_manifests, device, dtype)

    # Step 4: Transfer evaluation
    log("\nStep 4: Cross-subgroup transfer evaluation...")
    transfer_df = run_transfer_evaluation(
        viable_manifests, metadata_df, run_dir, output_dir,
        wrapper, sae_cache, device, dtype, args,
    )

    if transfer_df.empty:
        log("WARNING: No transfer pairs evaluated; exiting")
        runtime = time.time() - t0
        save_all_outputs(
            output_dir, transfer_df, primary_cosines, sae_cosines,
            pd.DataFrame(), {}, {}, {}, viable_manifests, runtime,
        )
        return

    # Step 5: Build scatter data
    log("\nStep 5: Assembling scatter data...")
    scatter_df = build_scatter_data(transfer_df, primary_cosines, sae_cosines)

    # Step 6: Regression analyses
    log("Step 6: Regression analyses...")
    regression_results = run_regression_analyses(scatter_df)

    # Step 7: SAE vs DIM comparison
    log("Step 7: SAE vs DIM cosine comparison...")
    sae_dim_comparison = compare_sae_vs_dim_cosines(scatter_df)

    # Step 8: Stable-range sensitivity
    stable_range_results: dict[str, Any] = {}
    if not args.primary_only:
        log("Step 8: Stable-range sensitivity analysis...")
        stable_range_results = stable_range_sensitivity(run_dir, transfer_df)
    else:
        log("Step 8: SKIPPED (--primary_only)")

    # Save outputs
    runtime = time.time() - t0
    save_all_outputs(
        output_dir, transfer_df, primary_cosines, sae_cosines,
        scatter_df, regression_results, sae_dim_comparison,
        stable_range_results, viable_manifests, runtime,
    )

    # Figures
    if not args.skip_figures:
        try:
            from src.visualization.transfer_figures import generate_c2_figures
            generate_c2_figures(
                output_dir, scatter_df, regression_results,
                sae_dim_comparison, stable_range_results,
                transfer_df, viable_manifests,
            )
        except ImportError:
            log("WARNING: transfer_figures module not available; skipping figures")
        except Exception as e:
            log(f"WARNING: figure generation failed: {e}")

    log(f"\nC2 complete in {runtime:.1f}s")
    log(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
