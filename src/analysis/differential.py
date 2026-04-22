"""B1 core: differential SAE feature analysis per subgroup per layer.

Identifies SAE features that activate differently when the model produces
a stereotyped response vs. when it doesn't.  Uses Mann-Whitney U tests with
BH FDR correction, vectorized across features via sparse matrices.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log


# ---------------------------------------------------------------------------
# Step 0: Load metadata
# ---------------------------------------------------------------------------

def load_metadata(run_dir: Path) -> pd.DataFrame:
    """Load item metadata.  Prefer parquet; fall back to .npz scanning."""
    meta_path = run_dir / "A_extraction" / "metadata.parquet"

    if meta_path.exists():
        df = pd.read_parquet(meta_path)
        log(f"Loaded metadata: {len(df)} items from metadata.parquet")
    else:
        log("metadata.parquet not found; scanning .npz files (slow)...")
        df = _build_metadata_from_npz(run_dir)

    # Deserialize stereotyped_groups from JSON string to list.
    # Parquet may store strings as dtype 'object' (older pandas) or
    # 'string'/'str' (newer pandas/pyarrow).  Check the first value.
    if len(df) > 0:
        first = df["stereotyped_groups"].iloc[0]
        if isinstance(first, str):
            df["stereotyped_groups"] = df["stereotyped_groups"].apply(json.loads)

    return df


def _build_metadata_from_npz(run_dir: Path) -> pd.DataFrame:
    """Fall back: scan .npz files to build metadata DataFrame."""
    records: list[dict[str, Any]] = []
    act_dir = run_dir / "A_extraction" / "activations"
    for cat_dir in sorted(act_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        for npz_path in sorted(cat_dir.glob("item_*.npz")):
            try:
                data = np.load(npz_path, allow_pickle=True)
                raw = data["metadata_json"]
                meta_str = raw.item() if raw.shape == () else str(raw)
                meta = json.loads(meta_str)
                meta["stereotyped_groups"] = json.dumps(meta["stereotyped_groups"])
                records.append(meta)
            except Exception:
                continue
    df = pd.DataFrame(records)
    meta_path = run_dir / "A_extraction" / "metadata.parquet"
    df.to_parquet(meta_path, index=False, compression="snappy")
    log(f"  Built metadata.parquet: {len(df)} items")
    return df


# ---------------------------------------------------------------------------
# Step 0b: Subgroup catalog
# ---------------------------------------------------------------------------

def build_subgroup_catalog(
    meta_df: pd.DataFrame,
    categories: list[str],
    min_n: int,
) -> dict[str, dict[str, Any]]:
    """Build catalog of analyzable subgroups.

    A subgroup is analyzable if it has ``>= min_n`` items in **both** the
    stereotyped-response group and the non-stereotyped-response group
    (ambig items only).
    """
    catalog: dict[str, dict[str, Any]] = {}

    for cat in categories:
        cat_df = meta_df[meta_df["category"] == cat]
        ambig = cat_df[cat_df["context_condition"] == "ambig"]

        all_subs: set[str] = set()
        for gs in ambig["stereotyped_groups"]:
            all_subs.update(gs)

        for sub in sorted(all_subs):
            targeting = ambig[ambig["stereotyped_groups"].apply(lambda gs: sub in gs)]
            n_stereo = int(
                (targeting["model_answer_role"] == "stereotyped_target").sum()
            )
            n_non_stereo = int(
                (targeting["model_answer_role"] != "stereotyped_target").sum()
            )
            n_unknown = int(
                (targeting["model_answer_role"] == "unknown").sum()
            )
            n_non_stereo_strict = int(
                (targeting["model_answer_role"] == "non_stereotyped").sum()
            )

            catalog[sub] = {
                "category": cat,
                "n_stereo": n_stereo,
                "n_non_stereo": n_non_stereo,
                "n_non_stereo_strict": n_non_stereo_strict,
                "n_unknown_in_non_stereo_group": n_unknown,
                "total_ambig": int(len(targeting)),
                "analyzable": n_stereo >= min_n and n_non_stereo >= min_n,
            }

    # Cross-category name uniqueness check.
    name_to_cats: dict[str, list[str]] = defaultdict(list)
    for sub, entry in catalog.items():
        name_to_cats[sub].append(entry["category"])
    for name, cats in name_to_cats.items():
        if len(cats) > 1:
            log(f"WARNING: subgroup '{name}' appears in multiple categories: {cats}")

    # Log skipped subgroups.
    n_analyzable = sum(1 for v in catalog.values() if v["analyzable"])
    n_skipped = len(catalog) - n_analyzable
    if n_skipped > 0:
        log(f"Subgroups with <{min_n} items in either group (skipped):")
        for sub, entry in sorted(catalog.items()):
            if not entry["analyzable"]:
                log(f"  {entry['category']}/{sub}: "
                    f"n_stereo={entry['n_stereo']}, "
                    f"n_non_stereo={entry['n_non_stereo']}")

    return catalog


# ---------------------------------------------------------------------------
# Step 1: Comparison groups
# ---------------------------------------------------------------------------

def get_comparison_groups(
    meta_df: pd.DataFrame,
    category: str,
    subgroup: str,
) -> tuple[list[int], list[int]]:
    """Return ``(stereo_item_idxs, non_stereo_item_idxs)`` for one subgroup.

    Both lists contain ``item_idx`` values for ambig items targeting the
    subgroup.  The non-stereotyped group includes both "unknown" and
    "non_stereotyped" model answers (everything that isn't the biased answer).
    """
    cat_df = meta_df[meta_df["category"] == category]
    targeting = cat_df[
        cat_df["stereotyped_groups"].apply(lambda gs: subgroup in gs)
        & (cat_df["context_condition"] == "ambig")
    ]
    stereo_mask = targeting["model_answer_role"] == "stereotyped_target"
    stereo_idxs = targeting.loc[stereo_mask, "item_idx"].tolist()
    non_stereo_idxs = targeting.loc[~stereo_mask, "item_idx"].tolist()
    return stereo_idxs, non_stereo_idxs


# ---------------------------------------------------------------------------
# Step 2: Sparse matrix construction
# ---------------------------------------------------------------------------

def build_sparse_matrix(
    cat_df: pd.DataFrame,
    all_item_idxs: list[int],
) -> tuple[csr_matrix, dict[int, int], np.ndarray]:
    """Build a sparse (n_items, n_active_features) matrix for one (category, layer).

    Returns ``(matrix, item_idx_to_row, active_features)``.
    """
    item_idx_to_row = {idx: i for i, idx in enumerate(all_item_idxs)}

    active_features = np.sort(cat_df["feature_idx"].unique())
    feature_idx_to_col = {int(fidx): i for i, fidx in enumerate(active_features)}

    rows = cat_df["item_idx"].map(item_idx_to_row).values
    cols = cat_df["feature_idx"].map(feature_idx_to_col).values
    vals = cat_df["activation_value"].values.astype(np.float32)

    valid_mask = ~(pd.isna(rows) | pd.isna(cols))

    matrix = csr_matrix(
        (
            vals[valid_mask],
            (rows[valid_mask].astype(int), cols[valid_mask].astype(int)),
        ),
        shape=(len(all_item_idxs), len(active_features)),
        dtype=np.float32,
    )
    return matrix, item_idx_to_row, active_features


# ---------------------------------------------------------------------------
# Step 3: Vectorized statistical tests
# ---------------------------------------------------------------------------

def test_subgroup_vectorized(
    matrix: csr_matrix,
    item_idx_to_row: dict[int, int],
    active_features: np.ndarray,
    stereo_idxs: list[int],
    non_stereo_idxs: list[int],
) -> pd.DataFrame | None:
    """Run vectorized differential tests for all active features for one subgroup."""
    stereo_rows = [item_idx_to_row[i] for i in stereo_idxs if i in item_idx_to_row]
    non_stereo_rows = [item_idx_to_row[i] for i in non_stereo_idxs if i in item_idx_to_row]

    if len(stereo_rows) == 0 or len(non_stereo_rows) == 0:
        return None

    stereo_mat = matrix[stereo_rows, :].toarray()
    non_stereo_mat = matrix[non_stereo_rows, :].toarray()

    n_s = len(stereo_rows)
    n_ns = len(non_stereo_rows)

    # Vectorized Cohen's d.
    mean_s = stereo_mat.mean(axis=0)
    mean_ns = non_stereo_mat.mean(axis=0)
    var_s = stereo_mat.var(axis=0, ddof=1) if n_s > 1 else np.zeros_like(mean_s)
    var_ns = non_stereo_mat.var(axis=0, ddof=1) if n_ns > 1 else np.zeros_like(mean_ns)

    pooled_var = (
        (n_s - 1) * var_s + (n_ns - 1) * var_ns
    ) / max(n_s + n_ns - 2, 1)
    pooled_std = np.sqrt(np.maximum(pooled_var, 1e-12))
    cohens_d = (mean_s - mean_ns) / pooled_std

    # Vectorized firing rates.
    firing_s = (stereo_mat > 0).mean(axis=0)
    firing_ns = (non_stereo_mat > 0).mean(axis=0)

    # Firing rate filter.
    max_firing = np.maximum(firing_s, firing_ns)
    combined_nonzero = (stereo_mat > 0).sum(axis=0) + (non_stereo_mat > 0).sum(axis=0)
    passes_filter = (max_firing >= 0.05) | (combined_nonzero >= 10)

    # Vectorized Mann-Whitney U.
    pvalues = np.ones(len(active_features), dtype=np.float64)

    if passes_filter.any():
        filtered_s = stereo_mat[:, passes_filter]
        filtered_ns = non_stereo_mat[:, passes_filter]
        try:
            _, pvals_filtered = mannwhitneyu(
                filtered_s, filtered_ns,
                alternative="two-sided", axis=0, nan_policy="omit",
            )
            pvalues[passes_filter] = pvals_filtered
        except (ValueError, TypeError) as e:
            log(f"    Vectorized Mann-Whitney failed ({e}), falling back to loop")
            idx_passing = np.where(passes_filter)[0]
            for j in idx_passing:
                try:
                    _, p = mannwhitneyu(
                        stereo_mat[:, j], non_stereo_mat[:, j],
                        alternative="two-sided",
                    )
                    pvalues[j] = p
                except Exception:
                    pvalues[j] = 1.0

    result = pd.DataFrame({
        "feature_idx": active_features.astype(np.int32),
        "cohens_d": cohens_d.astype(np.float32),
        "p_value_raw": pvalues,
        "firing_rate_stereo": firing_s.astype(np.float32),
        "firing_rate_non_stereo": firing_ns.astype(np.float32),
        "mean_activation_stereo": mean_s.astype(np.float32),
        "mean_activation_non_stereo": mean_ns.astype(np.float32),
        "passes_firing_filter": passes_filter,
    })
    result["n_stereo"] = np.int32(n_s)
    result["n_non_stereo"] = np.int32(n_ns)

    return result


# ---------------------------------------------------------------------------
# Step 4: FDR correction
# ---------------------------------------------------------------------------

def apply_fdr(result_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Apply Benjamini-Hochberg FDR correction to features passing firing filter."""
    result_df = result_df.copy()
    result_df["p_value_fdr"] = 1.0
    result_df["is_significant"] = False

    mask = result_df["passes_firing_filter"].values
    if mask.any():
        raw_pvals = result_df.loc[mask, "p_value_raw"].values
        rejected, p_adj, _, _ = multipletests(
            raw_pvals, alpha=alpha, method="fdr_bh",
        )
        result_df.loc[mask, "p_value_fdr"] = p_adj
        result_df.loc[mask, "is_significant"] = rejected

    result_df["direction"] = np.where(
        result_df["cohens_d"] > 0, "pro_bias", "anti_bias",
    )
    return result_df


# ---------------------------------------------------------------------------
# Step 5: Per-layer processing
# ---------------------------------------------------------------------------

def process_layer(
    layer: int,
    run_dir: Path,
    meta_df: pd.DataFrame,
    categories: list[str],
    subgroup_catalog: dict[str, dict[str, Any]],
    min_n: int,
    max_items: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Process one layer: test all subgroups in all categories.

    Returns ``(combined_df, layer_summary)``.
    """
    t0 = time.time()
    log(f"\n{'=' * 60}")
    log(f"Layer {layer:02d}")
    log(f"{'=' * 60}")

    parquet_path = (
        run_dir / "A_extraction" / "sae_encoding" / f"layer_{layer:02d}.parquet"
    )
    if not parquet_path.exists():
        log(f"  ERROR: layer parquet not found: {parquet_path}")
        return pd.DataFrame(), {}

    layer_df = pd.read_parquet(parquet_path)
    log(f"  Loaded parquet: {len(layer_df)} rows")

    all_results: list[pd.DataFrame] = []
    layer_summary: dict[str, Any] = {"per_subgroup": {}}

    for cat in categories:
        cat_df = layer_df[layer_df["category"] == cat]
        if cat_df.empty:
            continue

        # Get all ambig item_idxs for this category.
        cat_ambig_meta = meta_df[
            (meta_df["category"] == cat)
            & (meta_df["context_condition"] == "ambig")
        ]
        all_item_idxs = cat_ambig_meta["item_idx"].tolist()
        if max_items:
            all_item_idxs = all_item_idxs[:max_items]
        if not all_item_idxs:
            continue

        cat_df = cat_df[cat_df["item_idx"].isin(all_item_idxs)]
        if cat_df.empty:
            continue

        matrix, item_idx_to_row, active_features = build_sparse_matrix(
            cat_df, all_item_idxs,
        )
        log(f"  {cat}: sparse matrix {matrix.shape}, "
            f"{len(active_features)} active features")

        for sub, entry in sorted(subgroup_catalog.items()):
            if entry["category"] != cat or not entry["analyzable"]:
                continue

            stereo_idxs, non_stereo_idxs = get_comparison_groups(
                meta_df, cat, sub,
            )
            stereo_idxs = [i for i in stereo_idxs if i in item_idx_to_row]
            non_stereo_idxs = [i for i in non_stereo_idxs if i in item_idx_to_row]

            if len(stereo_idxs) < min_n or len(non_stereo_idxs) < min_n:
                continue

            result = test_subgroup_vectorized(
                matrix, item_idx_to_row, active_features,
                stereo_idxs, non_stereo_idxs,
            )
            if result is None:
                continue

            result = result[result["passes_firing_filter"]].copy()
            if result.empty:
                continue

            result = apply_fdr(result, alpha=0.05)

            result["layer"] = np.int32(layer)
            result["subgroup"] = sub
            result["category"] = cat

            n_sig = int(result["is_significant"].sum())
            n_pro = int(
                (result["is_significant"] & (result["direction"] == "pro_bias")).sum()
            )
            n_anti = int(
                (result["is_significant"] & (result["direction"] == "anti_bias")).sum()
            )

            layer_summary["per_subgroup"][sub] = {
                "category": cat,
                "n_stereo": int(result["n_stereo"].iloc[0]),
                "n_non_stereo": int(result["n_non_stereo"].iloc[0]),
                "n_features_tested": len(result),
                "n_significant": n_sig,
                "n_significant_pro_bias": n_pro,
                "n_significant_anti_bias": n_anti,
            }

            log(f"    {sub}: n_stereo={result['n_stereo'].iloc[0]}, "
                f"n_non_stereo={result['n_non_stereo'].iloc[0]}, "
                f"n_tested={len(result)}, "
                f"n_sig={n_sig} (pro={n_pro}, anti={n_anti})")

            all_results.append(result)

    combined = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

    elapsed = time.time() - t0
    log(f"  Layer {layer:02d} complete in {elapsed:.1f}s")

    layer_summary["layer"] = layer
    layer_summary["elapsed_seconds"] = round(elapsed, 1)
    layer_summary["total_rows"] = len(combined)

    return combined, layer_summary


# ---------------------------------------------------------------------------
# Step 6: Output I/O
# ---------------------------------------------------------------------------

def save_layer_parquet(run_dir: Path, layer: int, df: pd.DataFrame) -> None:
    """Atomic parquet write for one layer's results."""
    out_dir = ensure_dir(run_dir / "B_differential")
    out_path = out_dir / f"layer_{layer:02d}.parquet"

    if df.empty:
        log(f"  WARNING: empty results for layer {layer}, skipping write")
        return

    # Drop internal helper column before saving.
    save_cols = [
        "feature_idx", "layer", "subgroup", "category",
        "cohens_d", "p_value_raw", "p_value_fdr", "is_significant", "direction",
        "firing_rate_stereo", "firing_rate_non_stereo",
        "mean_activation_stereo", "mean_activation_non_stereo",
        "n_stereo", "n_non_stereo",
    ]
    save_df = df[[c for c in save_cols if c in df.columns]]

    tmp_path = out_path.with_suffix(".parquet.tmp")
    save_df.to_parquet(tmp_path, index=False, compression="snappy")
    tmp_path.rename(out_path)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    log(f"  Saved {len(save_df)} rows ({size_mb:.1f} MB) → {out_path.name}")


def save_layer_summary(
    run_dir: Path, layer: int, layer_summary: dict[str, Any],
) -> None:
    """Write per-layer summary JSON."""
    out_dir = ensure_dir(run_dir / "B_differential")
    atomic_save_json(layer_summary, out_dir / f"layer_{layer:02d}_summary.json")


def load_layer_summary(run_dir: Path, layer: int) -> dict[str, Any]:
    """Load an existing per-layer summary JSON."""
    path = run_dir / "B_differential" / f"layer_{layer:02d}_summary.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Global summary
# ---------------------------------------------------------------------------

def build_differential_summary(
    run_dir: Path,
    layers: list[int],
    subgroup_catalog: dict[str, dict[str, Any]],
    all_summaries: dict[int, dict[str, Any]],
    min_n: int = 10,
) -> None:
    """Build differential_summary.json from per-layer results."""
    # Per-subgroup aggregation across layers.
    per_subgroup_results: dict[str, dict[str, Any]] = {}
    total_significant = 0
    total_time = 0.0

    for sub, entry in sorted(subgroup_catalog.items()):
        if not entry["analyzable"]:
            continue
        n_sig_by_layer: dict[str, int] = {}
        total_pro = 0
        total_anti = 0
        total_sig = 0

        for layer in layers:
            ls = all_summaries.get(layer, {})
            ps = ls.get("per_subgroup", {}).get(sub, {})
            n_sig = ps.get("n_significant", 0)
            n_sig_by_layer[str(layer)] = n_sig
            total_pro += ps.get("n_significant_pro_bias", 0)
            total_anti += ps.get("n_significant_anti_bias", 0)
            total_sig += n_sig

        peak_layer = max(n_sig_by_layer, key=n_sig_by_layer.get) if n_sig_by_layer else "0"
        per_subgroup_results[sub] = {
            "category": entry["category"],
            "n_significant_by_layer": n_sig_by_layer,
            "peak_layer": int(peak_layer),
            "peak_n_significant": n_sig_by_layer.get(peak_layer, 0),
            "total_significant_pro_bias_all_layers": total_pro,
            "total_significant_anti_bias_all_layers": total_anti,
            "total_significant_all_layers": total_sig,
        }
        total_significant += total_sig

    for layer_summary in all_summaries.values():
        total_time += layer_summary.get("elapsed_seconds", 0)

    # Skipped subgroups.
    skipped: dict[str, dict[str, str]] = {}
    for sub, entry in sorted(subgroup_catalog.items()):
        if not entry["analyzable"]:
            reason_parts = []
            if entry["n_stereo"] < min_n:
                reason_parts.append(f"n_stereo={entry['n_stereo']} < min_n={min_n}")
            if entry["n_non_stereo"] < min_n:
                reason_parts.append(
                    f"n_non_stereo={entry['n_non_stereo']} < min_n={min_n}"
                )
            skipped[sub] = {
                "reason": ", ".join(reason_parts),
                "category": entry["category"],
            }

    summary = {
        "layers_analyzed": sorted(layers),
        "fdr_threshold": 0.05,
        "min_firing_rate_threshold": 0.05,
        "min_nonzero_count_threshold": 10,
        "min_n_per_group": min_n,
        "test": "mann_whitney_u",
        "alternative": "two-sided",
        "context_condition_filter": "ambig",
        "comparison_type": "question_A_stereotyped_vs_all_other",
        "subgroup_catalog": subgroup_catalog,
        "per_subgroup_results": per_subgroup_results,
        "skipped_subgroups": skipped,
        "total_significant_features": total_significant,
        "total_runtime_seconds": round(total_time, 1),
    }

    out_dir = ensure_dir(run_dir / "B_differential")
    atomic_save_json(summary, out_dir / "differential_summary.json")

    log(f"\nDifferential summary → {out_dir / 'differential_summary.json'}")
    log(f"Analyzable subgroups: {len(per_subgroup_results)}")
    log(f"Skipped subgroups: {len(skipped)}")
    log(f"Total significant features: {total_significant}")
    log(f"Total runtime: {total_time:.0f}s")
