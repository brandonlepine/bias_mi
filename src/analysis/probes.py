"""B4 core: linear probe training with group-aware cross-validation and controls.

Trains PCA-50 → L2-regularized logistic regression probes on saved hidden
states.  Uses GroupKFold(question_index) for all probes except the template-ID
control which uses StratifiedKFold.

Independent of GPU — pure numpy/sklearn.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log, progress_bar

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42
N_COMPONENTS = 50
N_FOLDS = 5
LR_C = 1.0
LR_MAX_ITER = 1000
SAE_TOP_K = 50


# ---------------------------------------------------------------------------
# Group-aware cross-validation
# ---------------------------------------------------------------------------

def safe_cv_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = N_FOLDS,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    """Produce GroupKFold splits, skipping degenerate folds.

    Returns (splits, n_skipped) where each split is (train_idx, test_idx).
    """
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    skipped = 0

    # GroupKFold needs n_splits <= n_groups
    n_groups = len(np.unique(groups))
    effective_splits = min(n_splits, n_groups)
    if effective_splits < 2:
        return [], n_splits

    for train_idx, test_idx in GroupKFold(n_splits=effective_splits).split(X, y, groups):
        y_train = y[train_idx]
        y_test = y[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            skipped += 1
            continue
        splits.append((train_idx, test_idx))

    return splits, skipped


# ---------------------------------------------------------------------------
# Core probe trainer
# ---------------------------------------------------------------------------

def train_probe(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int = N_FOLDS,
    class_weight: str | None = "balanced",
    seed: int = SEED,
    skip_pca: bool = False,
) -> dict[str, Any]:
    """Run group-aware cross-validated probe (PCA-50 → LogisticRegression).

    Parameters
    ----------
    skip_pca : bool
        If True, skip PCA (e.g. for low-dimensional SAE feature matrices).
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    n_components = min(N_COMPONENTS, X.shape[0] - 1, X.shape[1])

    splits, n_skipped = safe_cv_splits(X, y_enc, groups, n_splits=n_folds)
    if not splits:
        return {
            "mean_accuracy": None,
            "std_accuracy": None,
            "mean_balanced_accuracy": None,
            "std_balanced_accuracy": None,
            "n_folds_used": 0,
            "n_folds_skipped": n_skipped,
            "n_items": len(y),
            "n_classes": len(le.classes_),
            "classes": [str(c) for c in le.classes_],
            "n_components": n_components,
            "status": "no_valid_folds",
        }

    accs: list[float] = []
    balanced_accs: list[float] = []

    for train_idx, test_idx in splits:
        if skip_pca or n_components >= X.shape[1]:
            X_train = X[train_idx]
            X_test = X[test_idx]
        else:
            pca = PCA(n_components=n_components, random_state=seed)
            X_train = pca.fit_transform(X[train_idx])
            X_test = pca.transform(X[test_idx])

        clf = LogisticRegression(
            C=LR_C,
            max_iter=LR_MAX_ITER,
            solver="lbfgs",
            class_weight=class_weight,
            random_state=seed,
        )
        clf.fit(X_train, y_enc[train_idx])
        y_pred = clf.predict(X_test)

        accs.append(float((y_pred == y_enc[test_idx]).mean()))
        balanced_accs.append(float(balanced_accuracy_score(y_enc[test_idx], y_pred)))

    return {
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_balanced_accuracy": float(np.mean(balanced_accs)),
        "std_balanced_accuracy": float(np.std(balanced_accs)),
        "per_fold_accuracy": accs,
        "per_fold_balanced_accuracy": balanced_accs,
        "n_folds_used": len(accs),
        "n_folds_skipped": n_skipped,
        "n_items": len(y),
        "n_classes": len(le.classes_),
        "classes": [str(c) for c in le.classes_],
        "n_components": n_components,
        "status": "ok",
    }


def train_probe_stratified(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = N_FOLDS,
    class_weight: str | None = None,
    seed: int = SEED,
) -> dict[str, Any]:
    """Probe with StratifiedKFold (non-group-aware). For template ID control."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    n_classes = len(le.classes_)
    # Need at least n_folds samples per class for stratified CV
    class_counts = np.bincount(y_enc)
    min_count = class_counts.min()
    effective_folds = min(n_folds, min_count)
    if effective_folds < 2:
        return {
            "mean_accuracy": None, "std_accuracy": None,
            "mean_balanced_accuracy": None, "std_balanced_accuracy": None,
            "n_folds_used": 0, "n_folds_skipped": 0,
            "n_items": len(y), "n_classes": n_classes,
            "classes": [str(c) for c in le.classes_],
            "n_components": min(N_COMPONENTS, X.shape[0] - 1, X.shape[1]),
            "status": "insufficient_per_class_for_stratified",
        }

    n_components = min(N_COMPONENTS, X.shape[0] - 1, X.shape[1])
    skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=seed)

    accs: list[float] = []
    balanced_accs: list[float] = []

    for train_idx, test_idx in skf.split(X, y_enc):
        if n_components >= X.shape[1]:
            X_train, X_test = X[train_idx], X[test_idx]
        else:
            pca = PCA(n_components=n_components, random_state=seed)
            X_train = pca.fit_transform(X[train_idx])
            X_test = pca.transform(X[test_idx])

        clf = LogisticRegression(
            C=LR_C, max_iter=LR_MAX_ITER, solver="lbfgs",
            class_weight=class_weight, random_state=seed,
        )
        clf.fit(X_train, y_enc[train_idx])
        y_pred = clf.predict(X_test)

        accs.append(float((y_pred == y_enc[test_idx]).mean()))
        balanced_accs.append(float(balanced_accuracy_score(y_enc[test_idx], y_pred)))

    return {
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_balanced_accuracy": float(np.mean(balanced_accs)),
        "std_balanced_accuracy": float(np.std(balanced_accs)),
        "per_fold_accuracy": accs,
        "per_fold_balanced_accuracy": balanced_accs,
        "n_folds_used": len(accs),
        "n_folds_skipped": 0,
        "n_items": len(y),
        "n_classes": n_classes,
        "classes": [str(c) for c in le.classes_],
        "n_components": n_components,
        "status": "ok",
    }


# ---------------------------------------------------------------------------
# Permutation baseline
# ---------------------------------------------------------------------------

def permutation_baseline(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_permutations: int = 10,
    class_weight: str | None = "balanced",
    seed: int = SEED,
    skip_pca: bool = False,
) -> dict[str, Any]:
    """Run probes with shuffled labels to establish chance baseline."""
    perm_accs: list[float] = []
    perm_balanced_accs: list[float] = []

    for trial in range(n_permutations):
        rng = np.random.default_rng(seed + 1000 + trial)
        y_perm = rng.permutation(y)

        result = train_probe(
            X, y_perm, groups, class_weight=class_weight,
            seed=seed + trial, skip_pca=skip_pca,
        )

        if result["status"] == "ok":
            perm_accs.append(result["mean_accuracy"])
            perm_balanced_accs.append(result["mean_balanced_accuracy"])

    if not perm_accs:
        return {
            "mean_accuracy": None, "std_accuracy": None,
            "mean_balanced_accuracy": None, "std_balanced_accuracy": None,
            "n_trials": 0,
        }

    return {
        "mean_accuracy": float(np.mean(perm_accs)),
        "std_accuracy": float(np.std(perm_accs)),
        "mean_balanced_accuracy": float(np.mean(perm_balanced_accs)),
        "std_balanced_accuracy": float(np.std(perm_balanced_accs)),
        "n_trials": len(perm_accs),
    }


# ---------------------------------------------------------------------------
# Hidden-state cache helpers
# ---------------------------------------------------------------------------

def load_category_hidden_states_by_layer(
    run_dir: Path,
    cat: str,
    item_idxs: list[int],
    n_layers: int,
    hidden_dim: int,
) -> tuple[dict[int, np.ndarray], list[int]]:
    """Load hidden states for a category, indexed by layer.

    Returns
    -------
    hs_by_layer : {layer: (n_items, hidden_dim) float32}
    loaded_idxs : list of item_idx in row order
    """
    cat_dir = run_dir / "A_extraction" / "activations" / cat
    per_item: list[np.ndarray] = []
    loaded_idxs: list[int] = []

    for idx in progress_bar(item_idxs, desc=f"  Loading {cat}"):
        npz_path = cat_dir / f"item_{idx:06d}.npz"
        if not npz_path.exists():
            continue
        data = np.load(npz_path, allow_pickle=True)
        per_item.append(data["hidden_states"].astype(np.float32))
        loaded_idxs.append(idx)

    if not per_item:
        return {}, []

    # (n_items, n_layers, hidden_dim)
    stacked = np.stack(per_item, axis=0)
    hs_by_layer = {layer: stacked[:, layer, :] for layer in range(n_layers)}
    return hs_by_layer, loaded_idxs


def load_single_layer_hidden_states(
    run_dir: Path,
    cat: str,
    item_idxs: list[int],
    layer: int,
    hidden_dim: int,
) -> tuple[np.ndarray, list[int]]:
    """Load hidden states for one layer only (memory-efficient for cross-cat)."""
    cat_dir = run_dir / "A_extraction" / "activations" / cat
    hs_list: list[np.ndarray] = []
    loaded_idxs: list[int] = []

    for idx in item_idxs:
        npz_path = cat_dir / f"item_{idx:06d}.npz"
        if not npz_path.exists():
            continue
        data = np.load(npz_path, allow_pickle=True)
        hs_list.append(data["hidden_states"][layer].astype(np.float32))
        loaded_idxs.append(idx)

    if not hs_list:
        return np.zeros((0, hidden_dim), dtype=np.float32), []

    return np.stack(hs_list, axis=0), loaded_idxs


# ---------------------------------------------------------------------------
# Stimuli/question_index lookup
# ---------------------------------------------------------------------------

def load_question_index_map(run_dir: Path, categories: list[str]) -> dict[int, int]:
    """Build item_idx → question_index lookup from stimuli JSONs."""
    mapping: dict[int, int] = {}
    for cat in categories:
        stim_path = run_dir / "A_extraction" / "stimuli" / f"{cat}.json"
        if not stim_path.exists():
            log(f"  WARNING: stimuli file not found: {stim_path}")
            continue
        with open(stim_path) as f:
            items = json.load(f)
        for item in items:
            mapping[int(item["item_idx"])] = int(item["question_index"])
    return mapping


def get_groups_for_items(
    item_idxs: list[int] | np.ndarray,
    qi_map: dict[int, int],
) -> np.ndarray:
    """Look up question_index group labels for a list of item_idxs."""
    return np.array([qi_map.get(int(idx), -1) for idx in item_idxs])


# ---------------------------------------------------------------------------
# Subgroup enumeration
# ---------------------------------------------------------------------------

def enumerate_subgroups(
    meta_df: pd.DataFrame,
    cat: str,
    min_n: int,
    ambig_only: bool = True,
) -> list[str]:
    """List subgroups in a category that have enough items for binary probes."""
    cat_df = meta_df[meta_df["category"] == cat]
    if ambig_only:
        cat_df = cat_df[cat_df["context_condition"] == "ambig"]

    all_subs: set[str] = set()
    for gs in cat_df["stereotyped_groups"]:
        gs_list = gs if isinstance(gs, list) else json.loads(gs)
        all_subs.update(gs_list)

    viable: list[str] = []
    for sub in sorted(all_subs):
        mask = cat_df["stereotyped_groups"].apply(lambda gs: sub in gs)
        n_pos = int(mask.sum())
        n_neg = int((~mask).sum())
        if n_pos >= min_n and n_neg >= min_n:
            viable.append(sub)

    return viable


# ---------------------------------------------------------------------------
# Probe 1: Multiclass subgroup classification
# ---------------------------------------------------------------------------

def probe_multiclass_subgroup(
    cat: str,
    layer: int,
    X_layer: np.ndarray,
    cat_meta_ambig: pd.DataFrame,
    groups: np.ndarray,
    n_permutations: int,
    min_n: int,
) -> dict[str, Any] | None:
    """Multi-class subgroup probe: single-group ambig items only."""
    single = cat_meta_ambig[cat_meta_ambig["n_target_groups"] == 1].copy()
    if len(single) == 0:
        return None

    y = np.array([gs[0] for gs in single["stereotyped_groups"]])
    unique_labels, counts = np.unique(y, return_counts=True)

    # Filter: at least 2 subgroups with ≥ min_n items
    valid_labels = unique_labels[counts >= min_n]
    if len(valid_labels) < 2:
        return None

    mask = np.isin(y, valid_labels)
    if mask.sum() < min_n * 2:
        return None

    # Map back to row positions in X_layer
    row_idxs = single.index.values[mask]
    X = X_layer[row_idxs]
    y_filtered = y[mask]
    g = groups[row_idxs]

    result = train_probe(X, y_filtered, g, class_weight=None)
    perm = permutation_baseline(X, y_filtered, g, n_permutations=n_permutations,
                                class_weight=None)

    selectivity = None
    if result["mean_balanced_accuracy"] is not None and perm["mean_balanced_accuracy"] is not None:
        selectivity = result["mean_balanced_accuracy"] - perm["mean_balanced_accuracy"]

    return _build_record(
        probe_type="subgroup_multiclass", category=cat, subgroup=None,
        layer=layer, result=result, perm=perm, selectivity=selectivity,
    )


# ---------------------------------------------------------------------------
# Probe 2: Binary subgroup detection
# ---------------------------------------------------------------------------

def probe_binary_subgroup(
    cat: str,
    sub: str,
    layer: int,
    X_layer: np.ndarray,
    cat_meta_ambig: pd.DataFrame,
    groups: np.ndarray,
    n_permutations: int,
    min_n: int,
) -> dict[str, Any] | None:
    """Binary one-vs-rest subgroup detection, ambig items."""
    y = np.array([
        int(sub in gs) for gs in cat_meta_ambig["stereotyped_groups"]
    ])

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos < min_n or n_neg < min_n:
        return None

    row_idxs = cat_meta_ambig.index.values
    X = X_layer[row_idxs]
    g = groups[row_idxs]

    result = train_probe(X, y, g, class_weight="balanced")
    perm = permutation_baseline(X, y, g, n_permutations=n_permutations,
                                class_weight="balanced")

    selectivity = None
    if result["mean_balanced_accuracy"] is not None and perm["mean_balanced_accuracy"] is not None:
        selectivity = result["mean_balanced_accuracy"] - perm["mean_balanced_accuracy"]

    return _build_record(
        probe_type="subgroup_binary", category=cat, subgroup=sub,
        layer=layer, result=result, perm=perm, selectivity=selectivity,
    )


# ---------------------------------------------------------------------------
# Probe 3: Stereotyped response binary
# ---------------------------------------------------------------------------

def probe_stereotyped_response(
    cat: str,
    layer: int,
    X_layer: np.ndarray,
    cat_meta_ambig: pd.DataFrame,
    groups: np.ndarray,
    n_permutations: int,
    min_n: int,
) -> dict[str, Any] | None:
    """Binary probe: predict is_stereotyped_response, ambig items."""
    y = cat_meta_ambig["is_stereotyped_response"].values.astype(int)

    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    if n_pos < min_n or n_neg < min_n:
        return None

    row_idxs = cat_meta_ambig.index.values
    X = X_layer[row_idxs]
    g = groups[row_idxs]

    result = train_probe(X, y, g, class_weight="balanced")
    perm = permutation_baseline(X, y, g, n_permutations=n_permutations,
                                class_weight="balanced")

    selectivity = None
    if result["mean_balanced_accuracy"] is not None and perm["mean_balanced_accuracy"] is not None:
        selectivity = result["mean_balanced_accuracy"] - perm["mean_balanced_accuracy"]

    return _build_record(
        probe_type="stereotyped_response_binary", category=cat, subgroup=None,
        layer=layer, result=result, perm=perm, selectivity=selectivity,
    )


# ---------------------------------------------------------------------------
# Control B1: Context condition
# ---------------------------------------------------------------------------

def probe_context_condition(
    cat: str,
    layer: int,
    X_layer: np.ndarray,
    cat_meta_all: pd.DataFrame,
    groups: np.ndarray,
    n_permutations: int,
    min_n: int,
) -> dict[str, Any] | None:
    """Binary probe: ambig vs disambig. Uses ALL items (both conditions)."""
    y = cat_meta_all["context_condition"].values

    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 2 or counts.min() < min_n:
        return None

    row_idxs = cat_meta_all.index.values
    X = X_layer[row_idxs]
    g = groups[row_idxs]

    result = train_probe(X, y, g, class_weight="balanced")
    perm = permutation_baseline(X, y, g, n_permutations=n_permutations,
                                class_weight="balanced")

    selectivity = None
    if result["mean_balanced_accuracy"] is not None and perm["mean_balanced_accuracy"] is not None:
        selectivity = result["mean_balanced_accuracy"] - perm["mean_balanced_accuracy"]

    return _build_record(
        probe_type="context_condition", category=cat, subgroup=None,
        layer=layer, result=result, perm=perm, selectivity=selectivity,
    )


# ---------------------------------------------------------------------------
# Control B2: Template ID (stratified CV, non-group-aware)
# ---------------------------------------------------------------------------

def probe_template_id(
    cat: str,
    layer: int,
    X_layer: np.ndarray,
    cat_meta_all: pd.DataFrame,
    groups: np.ndarray,
) -> dict[str, Any] | None:
    """Multi-class template-ID probe with StratifiedKFold (not group-aware).

    No permutation baseline — this is itself a control.
    """
    y = groups[cat_meta_all.index.values]

    # Need ≥2 templates and ≥2 samples per template for stratified CV
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 2 or counts.min() < 2:
        return None

    row_idxs = cat_meta_all.index.values
    X = X_layer[row_idxs]

    result = train_probe_stratified(X, y, class_weight=None)

    return _build_record(
        probe_type="template_id", category=cat, subgroup=None,
        layer=layer, result=result, perm=None, selectivity=None,
    )


# ---------------------------------------------------------------------------
# Probe 4: SAE-based binary subgroup detection
# ---------------------------------------------------------------------------

def build_sae_feature_matrix(
    item_idxs: list[int],
    top_features: pd.DataFrame,
    run_dir: Path,
) -> np.ndarray:
    """Build (n_items, n_features) matrix from SAE activations at each feature's layer.

    Parameters
    ----------
    top_features : DataFrame with columns feature_idx, layer (and a positional index
        that maps to the column in the output matrix).
    """
    n_items = len(item_idxs)
    n_features = len(top_features)
    X = np.zeros((n_items, n_features), dtype=np.float32)

    item_idx_to_pos = {idx: i for i, idx in enumerate(item_idxs)}
    item_idx_set = set(item_idxs)

    # Reset index so positional indexing is clean
    top_features = top_features.reset_index(drop=True)

    for layer_val, layer_group in top_features.groupby("layer"):
        parquet_path = (
            run_dir / "A_extraction" / "sae_encoding" / f"layer_{int(layer_val):02d}.parquet"
        )
        if not parquet_path.exists():
            continue

        sae_df = pd.read_parquet(parquet_path)
        needed_feats = set(layer_group["feature_idx"].values)

        # Filter to relevant features + items for efficiency
        sae_sub = sae_df[
            sae_df["feature_idx"].isin(needed_feats)
            & sae_df["item_idx"].isin(item_idx_set)
        ]

        for row_pos, feat_row in layer_group.iterrows():
            feat_idx = int(feat_row["feature_idx"])
            col_idx = int(row_pos)

            feat_acts = sae_sub[sae_sub["feature_idx"] == feat_idx]
            for _, act_row in feat_acts.iterrows():
                iidx = int(act_row["item_idx"])
                if iidx in item_idx_to_pos:
                    X[item_idx_to_pos[iidx], col_idx] = float(act_row["activation_value"])

    return X


def probe_sae_binary_subgroup(
    cat: str,
    sub: str,
    run_dir: Path,
    cat_meta_ambig: pd.DataFrame,
    groups: np.ndarray,
    ranked_df: pd.DataFrame,
    n_permutations: int,
    min_n: int,
) -> dict[str, Any] | None:
    """Binary subgroup detection using top-K SAE features from B2."""
    y = np.array([
        int(sub in gs) for gs in cat_meta_ambig["stereotyped_groups"]
    ])

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos < min_n or n_neg < min_n:
        return None

    # Select top-K features for this subgroup
    top_features = ranked_df[
        (ranked_df["category"] == cat)
        & (ranked_df["subgroup"] == sub)
        & (ranked_df["direction"] == "pro_bias")
        & (ranked_df["rank"] <= SAE_TOP_K)
    ].copy()

    if len(top_features) == 0:
        return None

    item_idxs = cat_meta_ambig["item_idx"].tolist()
    X = build_sae_feature_matrix(item_idxs, top_features, run_dir)

    # SAE features are already low-dim, skip PCA
    result = train_probe(X, y, groups[cat_meta_ambig.index.values],
                         class_weight="balanced", skip_pca=True)
    perm = permutation_baseline(X, y, groups[cat_meta_ambig.index.values],
                                n_permutations=n_permutations,
                                class_weight="balanced", skip_pca=True)

    selectivity = None
    if result["mean_balanced_accuracy"] is not None and perm["mean_balanced_accuracy"] is not None:
        selectivity = result["mean_balanced_accuracy"] - perm["mean_balanced_accuracy"]

    return _build_record(
        probe_type="sae_subgroup_binary", category=cat, subgroup=sub,
        layer=-1,  # SAE probes span multiple layers
        result=result, perm=perm, selectivity=selectivity,
    )


# ---------------------------------------------------------------------------
# Control D1: Cross-category generalization
# ---------------------------------------------------------------------------

def probe_cross_category(
    layer: int,
    categories: list[str],
    meta_df: pd.DataFrame,
    qi_map: dict[int, int],
    run_dir: Path,
    hidden_dim: int,
    min_n: int,
) -> list[dict[str, Any]]:
    """Train is_stereotyped_response on each category, test on every other."""
    records: list[dict[str, Any]] = []

    # Pre-load hidden states per category for this layer
    cat_data: dict[str, tuple[np.ndarray, pd.DataFrame, np.ndarray]] = {}

    for cat in categories:
        cat_ambig = meta_df[
            (meta_df["category"] == cat)
            & (meta_df["context_condition"] == "ambig")
        ]
        if len(cat_ambig) == 0:
            continue

        X, loaded_idxs = load_single_layer_hidden_states(
            run_dir, cat, cat_ambig["item_idx"].tolist(), layer, hidden_dim,
        )
        if len(loaded_idxs) == 0:
            continue

        # Align metadata to loaded items
        aligned = cat_ambig.set_index("item_idx").loc[loaded_idxs].reset_index()
        y = aligned["is_stereotyped_response"].values.astype(int)
        groups = get_groups_for_items(loaded_idxs, qi_map)

        if int(y.sum()) < min_n or int((y == 0).sum()) < min_n:
            continue

        cat_data[cat] = (X, aligned, groups, y)

    for train_cat in categories:
        if train_cat not in cat_data:
            continue
        X_train, _, groups_train, y_train = cat_data[train_cat]

        for test_cat in categories:
            if test_cat not in cat_data:
                continue
            X_test, _, _, y_test = cat_data[test_cat]

            if train_cat == test_cat:
                # Within-category: group-aware CV
                result = train_probe(X_train, y_train, groups_train,
                                     class_weight="balanced")
                ba = result["mean_balanced_accuracy"]
            else:
                # Cross-category: single train/test split
                n_comp = min(N_COMPONENTS, X_train.shape[0] - 1, X_train.shape[1])
                pca = PCA(n_components=n_comp, random_state=SEED)
                X_tr_pca = pca.fit_transform(X_train)
                X_te_pca = pca.transform(X_test)

                clf = LogisticRegression(
                    C=LR_C, max_iter=LR_MAX_ITER,
                    class_weight="balanced", random_state=SEED,
                )
                clf.fit(X_tr_pca, y_train)
                y_pred = clf.predict(X_te_pca)
                ba = float(balanced_accuracy_score(y_test, y_pred))

            records.append({
                "train_category": train_cat,
                "test_category": test_cat,
                "layer": layer,
                "balanced_accuracy": ba,
                "n_train": len(y_train),
                "n_test": len(y_test),
                "is_within_category": train_cat == test_cat,
            })

    return records


# ---------------------------------------------------------------------------
# Control D2: Within-category cross-subgroup generalization
# ---------------------------------------------------------------------------

def probe_within_cat_cross_subgroup(
    cat: str,
    layer: int,
    X_layer: np.ndarray,
    cat_meta: pd.DataFrame,
    min_n: int,
) -> list[dict[str, Any]]:
    """Train is_stereotyped on subgroup A items, test on subgroup B items."""
    # Single-group ambig items only
    single_ambig = cat_meta[
        (cat_meta["n_target_groups"] == 1)
        & (cat_meta["context_condition"] == "ambig")
    ]
    if len(single_ambig) == 0:
        return []

    # Identify subgroups
    sub_items: dict[str, pd.DataFrame] = {}
    all_subs: set[str] = set()
    for gs in single_ambig["stereotyped_groups"]:
        all_subs.update(gs)

    for sub in sorted(all_subs):
        mask = single_ambig["stereotyped_groups"].apply(lambda gs: gs[0] == sub)
        sub_df = single_ambig[mask]
        if len(sub_df) >= min_n:
            y = sub_df["is_stereotyped_response"].values.astype(int)
            if len(set(y)) >= 2:
                sub_items[sub] = sub_df

    records: list[dict[str, Any]] = []
    subs = sorted(sub_items.keys())

    for train_sub in subs:
        train_df = sub_items[train_sub]
        train_idxs = train_df.index.values
        X_train = X_layer[train_idxs]
        y_train = train_df["is_stereotyped_response"].values.astype(int)

        for test_sub in subs:
            test_df = sub_items[test_sub]
            test_idxs = test_df.index.values
            X_test = X_layer[test_idxs]
            y_test = test_df["is_stereotyped_response"].values.astype(int)

            n_comp = min(N_COMPONENTS, X_train.shape[0] - 1, X_train.shape[1])
            if n_comp < 1:
                continue

            pca = PCA(n_components=n_comp, random_state=SEED)
            X_tr_pca = pca.fit_transform(X_train)
            X_te_pca = pca.transform(X_test)

            clf = LogisticRegression(
                C=LR_C, max_iter=LR_MAX_ITER,
                class_weight="balanced", random_state=SEED,
            )
            clf.fit(X_tr_pca, y_train)
            y_pred = clf.predict(X_te_pca)
            ba = float(balanced_accuracy_score(y_test, y_pred))

            records.append({
                "category": cat,
                "train_subgroup": train_sub,
                "test_subgroup": test_sub,
                "layer": layer,
                "balanced_accuracy": ba,
                "n_train": len(y_train),
                "n_test": len(y_test),
                "is_same_subgroup": train_sub == test_sub,
            })

    return records


# ---------------------------------------------------------------------------
# Record builder helper
# ---------------------------------------------------------------------------

def _build_record(
    probe_type: str,
    category: str,
    subgroup: str | None,
    layer: int,
    result: dict[str, Any],
    perm: dict[str, Any] | None,
    selectivity: float | None,
) -> dict[str, Any]:
    """Build a flat record for probe_results.parquet."""
    rec: dict[str, Any] = {
        "probe_type": probe_type,
        "category": category,
        "subgroup": subgroup,
        "layer": layer,
        "n_items": result.get("n_items"),
        "n_classes": result.get("n_classes"),
        "n_folds_used": result.get("n_folds_used"),
        "mean_accuracy": result.get("mean_accuracy"),
        "std_accuracy": result.get("std_accuracy"),
        "mean_balanced_accuracy": result.get("mean_balanced_accuracy"),
        "std_balanced_accuracy": result.get("std_balanced_accuracy"),
        "status": result.get("status"),
    }

    if perm is not None:
        rec["permutation_mean_accuracy"] = perm.get("mean_accuracy")
        rec["permutation_std_accuracy"] = perm.get("std_accuracy")
        rec["permutation_mean_balanced_accuracy"] = perm.get("mean_balanced_accuracy")
    else:
        rec["permutation_mean_accuracy"] = None
        rec["permutation_std_accuracy"] = None
        rec["permutation_mean_balanced_accuracy"] = None

    rec["selectivity"] = selectivity

    return rec


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_probe_results(run_dir: Path, records: list[dict[str, Any]]) -> Path:
    """Save main probe results to parquet (atomic)."""
    out_dir = ensure_dir(run_dir / "B_probes")
    out_path = out_dir / "probe_results.parquet"
    df = pd.DataFrame(records)
    tmp = out_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False, compression="snappy")
    tmp.rename(out_path)
    log(f"Saved probe_results.parquet: {len(df)} rows")
    return out_path


def save_cross_cat_results(run_dir: Path, records: list[dict[str, Any]]) -> Path:
    """Save cross-category generalization to parquet (atomic)."""
    out_dir = ensure_dir(run_dir / "B_probes")
    out_path = out_dir / "cross_category_generalization.parquet"
    df = pd.DataFrame(records)
    tmp = out_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False, compression="snappy")
    tmp.rename(out_path)
    log(f"Saved cross_category_generalization.parquet: {len(df)} rows")
    return out_path


def save_within_cat_results(run_dir: Path, records: list[dict[str, Any]]) -> Path:
    """Save within-category cross-subgroup results to parquet (atomic)."""
    out_dir = ensure_dir(run_dir / "B_probes")
    out_path = out_dir / "within_category_cross_subgroup.parquet"
    df = pd.DataFrame(records)
    tmp = out_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False, compression="snappy")
    tmp.rename(out_path)
    log(f"Saved within_category_cross_subgroup.parquet: {len(df)} rows")
    return out_path


def build_probes_summary(
    probe_records: list[dict[str, Any]],
    cross_cat_records: list[dict[str, Any]],
    within_cat_records: list[dict[str, Any]],
    categories: list[str],
    config_info: dict[str, Any],
    runtime_seconds: float,
) -> dict[str, Any]:
    """Build probes_summary.json content."""
    probe_df = pd.DataFrame(probe_records)

    # Count probes by type
    probes_run: dict[str, int] = {}
    if not probe_df.empty:
        probes_run = probe_df.groupby("probe_type").size().to_dict()

    # Peak selectivity per category (subgroup_multiclass)
    peak_sel: dict[str, Any] = {}
    if not probe_df.empty:
        multi = probe_df[
            (probe_df["probe_type"] == "subgroup_multiclass")
            & probe_df["selectivity"].notna()
        ]
        for cat in categories:
            cat_multi = multi[multi["category"] == cat]
            if not cat_multi.empty:
                best = cat_multi.loc[cat_multi["selectivity"].idxmax()]
                peak_sel[cat] = {
                    "peak_layer": int(best["layer"]),
                    "peak_selectivity": round(float(best["selectivity"]), 4),
                    "peak_balanced_accuracy": round(float(best["mean_balanced_accuracy"]), 4),
                }

    # Peak binary subgroup per subgroup
    peak_binary: dict[str, Any] = {}
    if not probe_df.empty:
        binary = probe_df[
            (probe_df["probe_type"] == "subgroup_binary")
            & probe_df["mean_balanced_accuracy"].notna()
        ]
        for _, row in binary.iterrows():
            key = f"{row['category']}/{row['subgroup']}"
            ba = float(row["mean_balanced_accuracy"])
            if key not in peak_binary or ba > peak_binary[key]["peak_balanced_accuracy"]:
                peak_binary[key] = {
                    "peak_layer": int(row["layer"]),
                    "peak_balanced_accuracy": round(ba, 4),
                }

    return {
        "config": config_info,
        "probes_run": probes_run,
        "peak_selectivity_per_category": peak_sel,
        "peak_binary_subgroup_per_subgroup": peak_binary,
        "n_cross_category_records": len(cross_cat_records),
        "n_within_category_records": len(within_cat_records),
        "runtime_seconds": round(runtime_seconds, 1),
    }


# ---------------------------------------------------------------------------
# Resume check
# ---------------------------------------------------------------------------

def b4_complete(run_dir: Path) -> bool:
    """Check whether B4 outputs already exist."""
    return (run_dir / "B_probes" / "probes_summary.json").exists()
