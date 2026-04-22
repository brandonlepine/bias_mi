"""B5 core: item-level feature interpretability for top-ranked SAE features.

For each top-K feature from B2, characterises what it responds to:
activation distributions, matched-pairs ambig/disambig comparison,
subgroup specificity, cross-category specificity, template-artifact
detection, cross-subgroup activation matrices, and feature co-occurrence.

Produces artifact_flags consumed by C1 to exclude template features
from steering candidates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import adjusted_rand_score

from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log, progress_bar


# ---------------------------------------------------------------------------
# Constants / thresholds
# ---------------------------------------------------------------------------

DEFAULT_TOP_K = 20
TOP_N_ITEMS = 20
TOP_N_COOCCUR = 10

LOW_CATEGORY_SPECIFICITY_THRESHOLD = 2.0
LENGTH_CORRELATION_THRESHOLD = 0.5
HIGH_FIRING_RATE_THRESHOLD = 0.8


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------

def load_characterization_features(run_dir: Path, top_k: int = DEFAULT_TOP_K) -> pd.DataFrame:
    """Load the top-K features per (category, subgroup, direction) from B2."""
    ranked = pd.read_parquet(run_dir / "B_feature_ranking" / "ranked_features.parquet")
    return ranked[ranked["rank"] <= top_k].copy()


# ---------------------------------------------------------------------------
# Layer-cached parquet loading
# ---------------------------------------------------------------------------

class LayerCache:
    """Cache SAE-encoding parquets keyed by layer to avoid redundant reads."""

    def __init__(self, run_dir: Path) -> None:
        self._run_dir = run_dir
        self._cache: dict[int, pd.DataFrame] = {}

    def get(self, layer: int) -> pd.DataFrame:
        if layer not in self._cache:
            path = (
                self._run_dir / "A_extraction" / "sae_encoding"
                / f"layer_{layer:02d}.parquet"
            )
            self._cache[layer] = pd.read_parquet(path)
        return self._cache[layer]

    def feature_activations(self, feature_idx: int, layer: int) -> pd.Series:
        """Return activations for one feature indexed by (category, item_idx)."""
        layer_df = self.get(layer)
        feat_df = layer_df[layer_df["feature_idx"] == feature_idx]
        return feat_df.set_index(["category", "item_idx"])["activation_value"]

    def clear(self) -> None:
        self._cache.clear()


# ---------------------------------------------------------------------------
# Analysis A: Activation distribution per feature
# ---------------------------------------------------------------------------

def compute_activation_distribution(
    feature_activations: pd.Series,
    category_items: pd.DataFrame,
    category: str,
) -> dict[str, Any]:
    """Compute activation distribution stats within a feature's source category."""
    # Build item_idx → activation lookup
    cat_acts: dict[int, float] = {}
    for idx in category_items["item_idx"]:
        cat_acts[idx] = float(feature_activations.get((category, idx), 0.0))

    acts_array = np.array(list(cat_acts.values()))

    if len(acts_array) == 0:
        return {
            "mean_all": 0.0, "std_all": 0.0, "median_all": 0.0,
            "max_activation": 0.0, "fraction_nonzero": 0.0, "n_items": 0,
            "ambig": {"mean_all": None, "mean_stereo_response": None,
                      "mean_non_stereo_response": None, "n_stereo": 0, "n_non_stereo": 0},
            "disambig": {"mean_all": None, "n_items": 0},
        }

    stats: dict[str, Any] = {
        "mean_all": float(np.mean(acts_array)),
        "std_all": float(np.std(acts_array)),
        "median_all": float(np.median(acts_array)),
        "max_activation": float(np.max(acts_array)),
        "fraction_nonzero": float(np.mean(acts_array > 0)),
        "n_items": len(acts_array),
    }

    # Ambig-specific (primary, matches B1)
    ambig = category_items[category_items["context_condition"] == "ambig"]
    ambig_acts = np.array([cat_acts[i] for i in ambig["item_idx"]])

    ambig_stereo = ambig[ambig["is_stereotyped_response"] == True]
    ambig_non_stereo = ambig[ambig["is_stereotyped_response"] == False]
    stereo_acts = np.array([cat_acts[i] for i in ambig_stereo["item_idx"]])
    non_stereo_acts = np.array([cat_acts[i] for i in ambig_non_stereo["item_idx"]])

    stats["ambig"] = {
        "mean_all": float(ambig_acts.mean()) if len(ambig_acts) else None,
        "mean_stereo_response": float(stereo_acts.mean()) if len(stereo_acts) else None,
        "mean_non_stereo_response": float(non_stereo_acts.mean()) if len(non_stereo_acts) else None,
        "n_stereo": int(len(stereo_acts)),
        "n_non_stereo": int(len(non_stereo_acts)),
    }

    # Disambig (secondary)
    disambig = category_items[category_items["context_condition"] == "disambig"]
    disambig_acts = np.array([cat_acts[i] for i in disambig["item_idx"]])
    stats["disambig"] = {
        "mean_all": float(disambig_acts.mean()) if len(disambig_acts) else None,
        "n_items": int(len(disambig_acts)),
    }

    return stats


# ---------------------------------------------------------------------------
# Analysis B: Matched-pairs ambig/disambig comparison
# ---------------------------------------------------------------------------

def compute_matched_pairs_comparison(
    feature_activations: pd.Series,
    category_items: pd.DataFrame,
    category: str,
    qi_map: dict[int, int],
) -> dict[str, Any]:
    """Compare feature activation on matched ambig/disambig item pairs.

    Pairs items by (question_index, question_polarity, stereotyped_groups).
    """
    # Attach question_index to items
    cat_items = category_items.copy()
    cat_items["question_index"] = cat_items["item_idx"].map(qi_map)

    # Drop items without question_index mapping
    cat_items = cat_items[cat_items["question_index"].notna()]

    matched_pairs: list[dict[str, Any]] = []

    for (qidx, polarity), group in cat_items.groupby(["question_index", "question_polarity"]):
        ambig_rows = group[group["context_condition"] == "ambig"]
        disambig_rows = group[group["context_condition"] == "disambig"]

        if len(ambig_rows) == 0 or len(disambig_rows) == 0:
            continue

        for _, a_row in ambig_rows.iterrows():
            a_groups = sorted(
                a_row["stereotyped_groups"]
                if isinstance(a_row["stereotyped_groups"], list)
                else json.loads(a_row["stereotyped_groups"])
            )
            for _, d_row in disambig_rows.iterrows():
                d_groups = sorted(
                    d_row["stereotyped_groups"]
                    if isinstance(d_row["stereotyped_groups"], list)
                    else json.loads(d_row["stereotyped_groups"])
                )
                if a_groups == d_groups:
                    a_act = float(feature_activations.get(
                        (category, a_row["item_idx"]), 0.0))
                    d_act = float(feature_activations.get(
                        (category, d_row["item_idx"]), 0.0))
                    matched_pairs.append({
                        "ambig_activation": a_act,
                        "disambig_activation": d_act,
                        "delta": a_act - d_act,
                    })
                    break  # one-to-one matching

    if not matched_pairs:
        return {"n_pairs": 0, "matched_pairs_available": False}

    deltas = np.array([p["delta"] for p in matched_pairs])
    ambig_acts = np.array([p["ambig_activation"] for p in matched_pairs])
    disambig_acts = np.array([p["disambig_activation"] for p in matched_pairs])

    correlation = None
    if (len(matched_pairs) >= 2
            and np.std(ambig_acts) > 0 and np.std(disambig_acts) > 0):
        correlation = float(np.corrcoef(ambig_acts, disambig_acts)[0, 1])

    return {
        "matched_pairs_available": True,
        "n_pairs": len(matched_pairs),
        "mean_ambig_activation": float(ambig_acts.mean()),
        "mean_disambig_activation": float(disambig_acts.mean()),
        "mean_delta": float(deltas.mean()),
        "std_delta": float(deltas.std()),
        "fraction_ambig_higher": float((deltas > 0).mean()),
        "pearson_correlation": correlation,
    }


# ---------------------------------------------------------------------------
# Analysis C: Top-activating items
# ---------------------------------------------------------------------------

def get_top_activating_items(
    feature_activations: pd.Series,
    category_items: pd.DataFrame,
    stimuli: list[dict[str, Any]],
    category: str,
    top_n: int = TOP_N_ITEMS,
) -> list[dict[str, Any]]:
    """Return the top-N items by activation value with prompt previews."""
    item_acts: list[tuple[float, int, pd.Series]] = []
    for _, row in category_items.iterrows():
        act = float(feature_activations.get((category, row["item_idx"]), 0.0))
        item_acts.append((act, row["item_idx"], row))

    item_acts.sort(key=lambda x: -x[0])

    stimuli_by_idx = {s["item_idx"]: s for s in stimuli}

    top_items: list[dict[str, Any]] = []
    for act, item_idx, meta_row in item_acts[:top_n]:
        stim = stimuli_by_idx.get(item_idx, {})
        sg = meta_row["stereotyped_groups"]
        if isinstance(sg, str):
            sg = json.loads(sg)
        top_items.append({
            "item_idx": int(item_idx),
            "activation": round(act, 4),
            "prompt_preview": stim.get("prompt", "")[:150],
            "model_answer_role": meta_row["model_answer_role"],
            "model_answer": meta_row["model_answer"],
            "stereotyped_groups": sg,
            "n_target_groups": int(meta_row["n_target_groups"]),
            "context_condition": meta_row["context_condition"],
            "question_polarity": meta_row["question_polarity"],
            "is_stereotyped_response": bool(meta_row["is_stereotyped_response"]),
        })

    return top_items


# ---------------------------------------------------------------------------
# Analysis D: Subgroup specificity (within category)
# ---------------------------------------------------------------------------

def compute_subgroup_specificity(
    feature_activations: pd.Series,
    category_items: pd.DataFrame,
    category: str,
    target_subgroup: str,
) -> dict[str, Any]:
    """Compute target subgroup mean / category mean on ambig items."""
    ambig = category_items[category_items["context_condition"] == "ambig"]

    all_subgroups: set[str] = set()
    for gs in ambig["stereotyped_groups"]:
        gs_list = gs if isinstance(gs, list) else json.loads(gs)
        all_subgroups.update(gs_list)

    per_subgroup: dict[str, dict[str, Any]] = {}
    for sub in sorted(all_subgroups):
        sub_items = ambig[ambig["stereotyped_groups"].apply(
            lambda gs: sub in (gs if isinstance(gs, list) else json.loads(gs))
        )]
        if len(sub_items) == 0:
            continue
        acts = np.array([
            float(feature_activations.get((category, idx), 0.0))
            for idx in sub_items["item_idx"]
        ])
        per_subgroup[sub] = {
            "n_items": int(len(acts)),
            "mean_activation": float(acts.mean()),
            "fraction_nonzero": float((acts > 0).mean()),
        }

    if target_subgroup not in per_subgroup:
        return {"per_subgroup_activations": per_subgroup, "subgroup_specificity": None,
                "target_mean": None, "category_mean": None}

    target_mean = per_subgroup[target_subgroup]["mean_activation"]
    category_mean = float(np.mean([v["mean_activation"] for v in per_subgroup.values()]))
    specificity = target_mean / max(category_mean, 1e-8)

    return {
        "per_subgroup_activations": per_subgroup,
        "target_mean": float(target_mean),
        "category_mean": float(category_mean),
        "subgroup_specificity": round(float(specificity), 4),
    }


# ---------------------------------------------------------------------------
# Analysis E: Cross-subgroup activation matrix
# ---------------------------------------------------------------------------

def build_cross_subgroup_matrix(
    category: str,
    top_features: pd.DataFrame,
    layer_cache: LayerCache,
    metadata_df: pd.DataFrame,
) -> dict[str, Any] | None:
    """Build feature × subgroup activation matrix, cluster, compute ARI + BDS."""
    cat_features = top_features[
        (top_features["category"] == category)
        & (top_features["direction"] == "s_marking")
    ].copy()

    if cat_features.empty:
        return None

    cat_meta = metadata_df[metadata_df["category"] == category]
    ambig = cat_meta[cat_meta["context_condition"] == "ambig"]

    all_subgroups: set[str] = set()
    for gs in ambig["stereotyped_groups"]:
        gs_list = gs if isinstance(gs, list) else json.loads(gs)
        all_subgroups.update(gs_list)
    subgroups_list = sorted(all_subgroups)

    if len(subgroups_list) < 2:
        return None

    n_features = len(cat_features)
    n_subgroups = len(subgroups_list)
    matrix = np.zeros((n_features, n_subgroups), dtype=np.float32)
    feature_labels: list[str] = []
    source_subgroups: list[str] = []

    for i, (_, feat_row) in enumerate(cat_features.iterrows()):
        fidx = int(feat_row["feature_idx"])
        layer = int(feat_row["layer"])
        src_sub = feat_row["subgroup"]

        feature_labels.append(f"{src_sub}:L{layer}_F{fidx}")
        source_subgroups.append(src_sub)

        layer_df = layer_cache.get(layer)
        feat_acts = layer_df[
            (layer_df["feature_idx"] == fidx)
            & (layer_df["category"] == category)
        ].set_index("item_idx")["activation_value"]

        for j, tgt_sub in enumerate(subgroups_list):
            tgt_items = ambig[ambig["stereotyped_groups"].apply(
                lambda gs, s=tgt_sub: s in (gs if isinstance(gs, list) else json.loads(gs))
            )]
            if len(tgt_items) == 0:
                continue
            vals = np.array([
                float(feat_acts.get(idx, 0.0))
                for idx in tgt_items["item_idx"]
            ])
            matrix[i, j] = float(vals.mean())

    # Cluster rows by normalised activation profiles
    row_maxes = np.maximum(matrix.max(axis=1, keepdims=True), 1e-8)
    normalised = matrix / row_maxes

    try:
        linkage_matrix = linkage(normalised, method="ward")
        cluster_assignments = fcluster(
            linkage_matrix, t=n_subgroups, criterion="maxclust",
        )
    except Exception:
        cluster_assignments = np.arange(n_features)

    ari = float(adjusted_rand_score(source_subgroups, cluster_assignments))

    # Block-diagonal strength
    diag_vals: list[float] = []
    off_diag_vals: list[float] = []
    for i, src in enumerate(source_subgroups):
        for j, tgt in enumerate(subgroups_list):
            if src == tgt:
                diag_vals.append(matrix[i, j])
            else:
                off_diag_vals.append(matrix[i, j])

    diag_mean = float(np.mean(diag_vals)) if diag_vals else 0.0
    off_diag_mean = float(np.mean(off_diag_vals)) if off_diag_vals else 1e-8
    bds = diag_mean / max(off_diag_mean, 1e-8)

    return {
        "feature_labels": feature_labels,
        "source_subgroups": source_subgroups,
        "target_subgroups": subgroups_list,
        "matrix": matrix.tolist(),
        "cluster_assignments": cluster_assignments.tolist(),
        "adjusted_rand_index": round(ari, 4),
        "block_diagonal_strength": round(bds, 4),
        "diagonal_mean": round(diag_mean, 4),
        "off_diagonal_mean": round(off_diag_mean, 4),
    }


# ---------------------------------------------------------------------------
# Analysis F: Cross-category baseline (category specificity ratio)
# ---------------------------------------------------------------------------

def compute_category_specificity_ratio(
    feature_activations: pd.Series,
    source_category: str,
    all_categories: list[str],
    metadata_df: pd.DataFrame,
) -> dict[str, Any]:
    """Ratio of within-category (ambig) mean to cross-category (all items) mean."""
    src_ambig = metadata_df[
        (metadata_df["category"] == source_category)
        & (metadata_df["context_condition"] == "ambig")
    ]
    src_acts = np.array([
        float(feature_activations.get((source_category, idx), 0.0))
        for idx in src_ambig["item_idx"]
    ])
    within_mean = float(src_acts.mean()) if len(src_acts) else 0.0

    other_means: list[float] = []
    for cat in all_categories:
        if cat == source_category:
            continue
        other_items = metadata_df[metadata_df["category"] == cat]
        if len(other_items) == 0:
            continue
        other_acts = np.array([
            float(feature_activations.get((cat, idx), 0.0))
            for idx in other_items["item_idx"]
        ])
        other_means.append(float(other_acts.mean()))

    cross_mean = float(np.mean(other_means)) if other_means else 1e-8
    ratio = within_mean / max(cross_mean, 1e-8)

    return {
        "within_category_mean": round(within_mean, 4),
        "cross_category_mean": round(cross_mean, 4),
        "category_specificity_ratio": round(ratio, 4),
    }


# ---------------------------------------------------------------------------
# Analysis G: Template/artifact detection
# ---------------------------------------------------------------------------

def detect_template_artifacts(
    feature_activations: pd.Series,
    source_category: str,
    category_specificity_ratio: float,
    category_items: pd.DataFrame,
    stimuli: list[dict[str, Any]],
) -> dict[str, Any]:
    """Flag features that resemble template/artifact responses.

    Three heuristics (flag on ANY):
      1. category_specificity_ratio < 2.0
      2. |corr(activation, prompt_length)| > 0.5
      3. firing_rate > 0.8 within source category
    """
    flags: list[str] = []

    if category_specificity_ratio < LOW_CATEGORY_SPECIFICITY_THRESHOLD:
        flags.append("low_category_specificity")

    stimuli_by_idx = {s["item_idx"]: s for s in stimuli}
    activations: list[float] = []
    lengths: list[int] = []
    for _, row in category_items.iterrows():
        idx = row["item_idx"]
        act = float(feature_activations.get((source_category, idx), 0.0))
        prompt = stimuli_by_idx.get(idx, {}).get("prompt", "")
        activations.append(act)
        lengths.append(len(prompt))

    acts_arr = np.array(activations)
    lens_arr = np.array(lengths)

    length_correlation: float | None = None
    if len(acts_arr) >= 10 and np.std(acts_arr) > 0 and np.std(lens_arr) > 0:
        length_correlation = float(np.corrcoef(acts_arr, lens_arr)[0, 1])
        if abs(length_correlation) > LENGTH_CORRELATION_THRESHOLD:
            flags.append("length_correlation")

    firing_rate = float((acts_arr > 0).mean()) if len(acts_arr) else 0.0
    if firing_rate > HIGH_FIRING_RATE_THRESHOLD:
        flags.append("high_firing_rate")

    return {
        "length_correlation": (round(length_correlation, 4)
                               if length_correlation is not None else None),
        "firing_rate_source_category": round(firing_rate, 4),
        "artifact_flags": flags,
        "is_artifact_flagged": len(flags) > 0,
    }


# ---------------------------------------------------------------------------
# Analysis H: Feature co-occurrence
# ---------------------------------------------------------------------------

def compute_feature_cooccurrence(
    category: str,
    subgroup: str,
    top_features: pd.DataFrame,
    layer_cache: LayerCache,
    metadata_df: pd.DataFrame,
    top_n_cooccur: int = TOP_N_COOCCUR,
) -> dict[str, Any]:
    """Pairwise activation correlations among top-N pro-bias features for one subgroup."""
    sub_features = top_features[
        (top_features["category"] == category)
        & (top_features["subgroup"] == subgroup)
        & (top_features["direction"] == "s_marking")
        & (top_features["rank"] <= top_n_cooccur)
    ].sort_values("rank")

    if len(sub_features) < 2:
        return {"matrix": None, "feature_labels": [], "n_features": len(sub_features)}

    sub_items = metadata_df[
        (metadata_df["category"] == category)
        & (metadata_df["context_condition"] == "ambig")
        & (metadata_df["stereotyped_groups"].apply(
            lambda gs: subgroup in (gs if isinstance(gs, list) else json.loads(gs))
        ))
    ]
    item_idxs = sub_items["item_idx"].tolist()

    if len(item_idxs) < 10:
        return {"matrix": None, "feature_labels": [], "n_items": len(item_idxs)}

    n_feat = len(sub_features)
    n_items = len(item_idxs)
    activations = np.zeros((n_feat, n_items), dtype=np.float32)
    feature_labels: list[str] = []

    for i, (_, feat_row) in enumerate(sub_features.iterrows()):
        fidx = int(feat_row["feature_idx"])
        layer = int(feat_row["layer"])
        feature_labels.append(f"L{layer}_F{fidx}")

        layer_df = layer_cache.get(layer)
        feat_acts = layer_df[
            (layer_df["feature_idx"] == fidx)
            & (layer_df["category"] == category)
        ].set_index("item_idx")["activation_value"]

        for j, idx in enumerate(item_idxs):
            activations[i, j] = float(feat_acts.get(idx, 0.0))

    corr_matrix = np.corrcoef(activations)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    return {
        "feature_labels": feature_labels,
        "matrix": corr_matrix.tolist(),
        "n_features": int(n_feat),
        "n_items": int(n_items),
    }


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def build_interpretability_summary(
    stats_records: list[dict[str, Any]],
    cross_subgroup_matrices: dict[str, Any],
    artifact_list: list[dict[str, Any]],
    categories: list[str],
    top_k: int,
    runtime_seconds: float,
) -> dict[str, Any]:
    """Build interpretability_summary.json content."""
    stats_df = pd.DataFrame(stats_records) if stats_records else pd.DataFrame()

    artifact_rate_per_cat: dict[str, float] = {}
    if not stats_df.empty:
        for cat in categories:
            cat_rows = stats_df[stats_df["category"] == cat]
            if len(cat_rows) > 0:
                artifact_rate_per_cat[cat] = round(
                    float(cat_rows["is_artifact_flagged"].mean()), 4,
                )

    bds_per_cat: dict[str, float] = {}
    ari_per_cat: dict[str, float] = {}
    for cat, data in cross_subgroup_matrices.items():
        bds_per_cat[cat] = data["block_diagonal_strength"]
        ari_per_cat[cat] = data["adjusted_rand_index"]

    return {
        "config": {
            "top_k": top_k,
            "categories": categories,
            "length_correlation_threshold": LENGTH_CORRELATION_THRESHOLD,
            "high_firing_rate_threshold": HIGH_FIRING_RATE_THRESHOLD,
            "low_category_specificity_threshold": LOW_CATEGORY_SPECIFICITY_THRESHOLD,
        },
        "n_features_characterized": len(stats_records),
        "n_artifact_flagged": len(artifact_list),
        "artifact_flag_rate_per_category": artifact_rate_per_cat,
        "block_diagonal_strength_per_category": bds_per_cat,
        "adjusted_rand_index_per_category": ari_per_cat,
        "runtime_seconds": round(runtime_seconds, 1),
    }


# ---------------------------------------------------------------------------
# Resume check
# ---------------------------------------------------------------------------

def b5_complete(run_dir: Path) -> bool:
    """Check whether B5 outputs already exist."""
    return (run_dir / "B_feature_interpretability" / "interpretability_summary.json").exists()
