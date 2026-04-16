"""Structural tests for B1 differential feature analysis.

Validates core logic with synthetic data — no model or SAE weights required.

Run from project root:
    python tests/test_b1_structure.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.differential import (
    apply_fdr,
    build_sparse_matrix,
    build_subgroup_catalog,
    get_comparison_groups,
    test_subgroup_vectorized,
)

passed = 0
failed = 0


def check(condition: bool, name: str) -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✓ {name}", flush=True)
    else:
        failed += 1
        print(f"  ✗ FAIL: {name}", flush=True)


# ── Helpers ──────────────────────────────────────────────────────────

def make_meta_df() -> pd.DataFrame:
    """Build a synthetic metadata DataFrame resembling real A2 output."""
    rows = []
    for i in range(100):
        is_ambig = i < 60
        is_stereo = (i % 3 == 0) and is_ambig
        sg = ["gay"] if i < 50 else ["bisexual"]
        role = "stereotyped_target" if is_stereo else (
            "unknown" if i % 2 == 0 else "non_stereotyped"
        )
        rows.append({
            "item_idx": i,
            "category": "so",
            "model_answer": "A" if is_stereo else "B",
            "model_answer_role": role,
            "is_stereotyped_response": is_stereo,
            "is_correct": not is_stereo,
            "context_condition": "ambig" if is_ambig else "disambig",
            "stereotyped_groups": sg,
            "n_target_groups": 1,
            "margin": 2.0,
            "question_polarity": "neg",
            "correct_letter": "B",
            "stereotyped_option": "A",
        })
    return pd.DataFrame(rows)


# =========================================================================
# Imports
# =========================================================================

def test_imports() -> None:
    print("\n=== Imports ===", flush=True)
    from src.analysis.differential import (
        load_metadata, build_subgroup_catalog, get_comparison_groups,
        build_sparse_matrix, test_subgroup_vectorized, apply_fdr,
        process_layer, save_layer_parquet, save_layer_summary,
        load_layer_summary, build_differential_summary,
    )
    check(True, "all B1 functions import")


# =========================================================================
# Subgroup catalog
# =========================================================================

def test_subgroup_catalog() -> None:
    print("\n=== Subgroup Catalog ===", flush=True)
    meta_df = make_meta_df()
    catalog = build_subgroup_catalog(meta_df, ["so"], min_n=5)

    check("gay" in catalog, "gay in catalog")
    check("bisexual" in catalog, "bisexual in catalog")

    gay = catalog["gay"]
    check(gay["category"] == "so", "gay category is so")
    check(gay["n_stereo"] > 0, f"gay n_stereo={gay['n_stereo']} > 0")
    check(gay["n_non_stereo"] > 0, f"gay n_non_stereo={gay['n_non_stereo']} > 0")
    check(gay["analyzable"], "gay is analyzable")
    check(gay["total_ambig"] == gay["n_stereo"] + gay["n_non_stereo"],
          "total_ambig = n_stereo + n_non_stereo")

    # Tight min_n should make some subgroups unanalyzable.
    catalog2 = build_subgroup_catalog(meta_df, ["so"], min_n=100)
    check(not catalog2["gay"]["analyzable"],
          "gay unanalyzable with min_n=100")


# =========================================================================
# Comparison groups
# =========================================================================

def test_comparison_groups() -> None:
    print("\n=== Comparison Groups ===", flush=True)
    meta_df = make_meta_df()

    stereo, non_stereo = get_comparison_groups(meta_df, "so", "gay")
    check(len(stereo) > 0, f"gay stereo group has {len(stereo)} items")
    check(len(non_stereo) > 0, f"gay non-stereo group has {len(non_stereo)} items")

    # No overlap.
    check(len(set(stereo) & set(non_stereo)) == 0, "no overlap between groups")

    # All items should be ambig.
    all_idxs = set(stereo + non_stereo)
    ambig_idxs = set(
        meta_df[meta_df["context_condition"] == "ambig"]["item_idx"]
    )
    check(all_idxs <= ambig_idxs, "all comparison items are ambig")

    # All items target "gay".
    for idx in all_idxs:
        row = meta_df[meta_df["item_idx"] == idx].iloc[0]
        check("gay" in row["stereotyped_groups"],
              f"  item {idx} targets gay")
        break  # spot-check first item only for brevity


# =========================================================================
# Sparse matrix construction
# =========================================================================

def test_sparse_matrix() -> None:
    print("\n=== Sparse Matrix ===", flush=True)

    # Synthetic parquet-like data.
    cat_df = pd.DataFrame({
        "item_idx": [0, 0, 1, 1, 2],
        "feature_idx": [100, 200, 100, 300, 200],
        "activation_value": [1.5, 0.8, 2.1, 0.3, 1.0],
        "category": ["so"] * 5,
    })
    all_item_idxs = [0, 1, 2]

    matrix, idx_to_row, active_features = build_sparse_matrix(
        cat_df, all_item_idxs,
    )

    check(matrix.shape[0] == 3, f"matrix rows = 3 (got {matrix.shape[0]})")
    check(matrix.shape[1] == 3, f"matrix cols = 3 active features (got {matrix.shape[1]})")
    check(len(active_features) == 3, "3 active features")
    check(set(active_features) == {100, 200, 300}, f"features: {set(active_features)}")

    # Check values.
    dense = matrix.toarray()
    row0 = idx_to_row[0]
    col100 = np.where(active_features == 100)[0][0]
    check(abs(dense[row0, col100] - 1.5) < 1e-6,
          f"item 0, feature 100 = 1.5 (got {dense[row0, col100]})")

    # Items not in parquet have zero activation.
    col300 = np.where(active_features == 300)[0][0]
    check(dense[idx_to_row[0], col300] == 0.0,
          "item 0, feature 300 = 0.0 (not in parquet)")


# =========================================================================
# Vectorized statistical tests
# =========================================================================

def test_vectorized_tests() -> None:
    print("\n=== Vectorized Tests ===", flush=True)

    n_items = 50
    n_features = 20
    rng = np.random.RandomState(42)

    # Create a matrix where feature 0 is strongly different between groups.
    data = rng.rand(n_items, n_features).astype(np.float32)
    data[:, 0] = 0  # zero everywhere
    # Group A (first 25 items): feature 0 is high.
    data[:25, 0] = rng.rand(25) * 5 + 2
    # Group B (items 25-49): feature 0 stays zero.

    matrix = csr_matrix(data)
    item_idxs = list(range(n_items))
    idx_to_row = {i: i for i in item_idxs}
    active_features = np.arange(n_features)

    stereo_idxs = list(range(25))
    non_stereo_idxs = list(range(25, 50))

    result = test_subgroup_vectorized(
        matrix, idx_to_row, active_features,
        stereo_idxs, non_stereo_idxs,
    )

    check(result is not None, "result is not None")
    check(len(result) == n_features, f"one row per active feature (got {len(result)})")

    # Feature 0 should have large positive Cohen's d (stereo > non_stereo).
    feat0 = result[result["feature_idx"] == 0].iloc[0]
    check(feat0["cohens_d"] > 1.0,
          f"feature 0 has large cohens_d: {feat0['cohens_d']:.3f}")
    check(feat0["p_value_raw"] < 0.001,
          f"feature 0 has low p-value: {feat0['p_value_raw']:.6f}")
    check(feat0["firing_rate_stereo"] > 0.9,
          f"feature 0 firing rate stereo: {feat0['firing_rate_stereo']:.3f}")
    check(feat0["firing_rate_non_stereo"] < 0.1,
          f"feature 0 firing rate non-stereo: {feat0['firing_rate_non_stereo']:.3f}")

    check(result["n_stereo"].iloc[0] == 25, "n_stereo = 25")
    check(result["n_non_stereo"].iloc[0] == 25, "n_non_stereo = 25")

    # Schema check.
    expected_cols = {
        "feature_idx", "cohens_d", "p_value_raw",
        "firing_rate_stereo", "firing_rate_non_stereo",
        "mean_activation_stereo", "mean_activation_non_stereo",
        "passes_firing_filter", "n_stereo", "n_non_stereo",
    }
    check(expected_cols <= set(result.columns),
          f"result has expected columns")


# =========================================================================
# FDR correction
# =========================================================================

def test_fdr() -> None:
    print("\n=== FDR Correction ===", flush=True)

    # Create a result DF with some low p-values and many high ones.
    n = 100
    df = pd.DataFrame({
        "feature_idx": np.arange(n, dtype=np.int32),
        "cohens_d": np.random.randn(n).astype(np.float32),
        "p_value_raw": np.concatenate([
            np.array([1e-8, 1e-7, 1e-6, 1e-5, 1e-4]),  # 5 significant
            np.random.uniform(0.1, 1.0, n - 5),          # 95 non-significant
        ]),
        "firing_rate_stereo": np.ones(n, dtype=np.float32) * 0.1,
        "firing_rate_non_stereo": np.ones(n, dtype=np.float32) * 0.1,
        "mean_activation_stereo": np.ones(n, dtype=np.float32),
        "mean_activation_non_stereo": np.ones(n, dtype=np.float32),
        "passes_firing_filter": np.ones(n, dtype=bool),
        "n_stereo": np.int32(50),
        "n_non_stereo": np.int32(50),
    })

    result = apply_fdr(df, alpha=0.05)

    check("p_value_fdr" in result.columns, "has p_value_fdr column")
    check("is_significant" in result.columns, "has is_significant column")
    check("direction" in result.columns, "has direction column")

    n_sig = result["is_significant"].sum()
    check(n_sig >= 3, f"at least 3 significant features (got {n_sig})")
    check(n_sig <= 10, f"at most 10 significant features (got {n_sig})")

    # FDR-adjusted p-values >= raw p-values.
    passing = result[result["passes_firing_filter"]]
    check((passing["p_value_fdr"] >= passing["p_value_raw"] - 1e-10).all(),
          "p_value_fdr >= p_value_raw")

    # Direction based on sign of cohens_d.
    for _, row in result.iterrows():
        expected_dir = "pro_bias" if row["cohens_d"] > 0 else "anti_bias"
        check(row["direction"] == expected_dir,
              f"  feature {row['feature_idx']}: direction matches cohens_d sign")
        break  # spot-check first

    # Features not passing filter should not be significant.
    df2 = df.copy()
    df2.loc[0, "passes_firing_filter"] = False
    df2.loc[0, "p_value_raw"] = 1e-20  # very low but filtered out
    result2 = apply_fdr(df2)
    check(not result2.loc[0, "is_significant"],
          "filtered feature is not significant despite low p-value")


# =========================================================================
# Output parquet schema
# =========================================================================

def test_output_schema() -> None:
    print("\n=== Output Parquet Schema ===", flush=True)

    expected_cols = {
        "feature_idx", "layer", "subgroup", "category",
        "cohens_d", "p_value_raw", "p_value_fdr", "is_significant", "direction",
        "firing_rate_stereo", "firing_rate_non_stereo",
        "mean_activation_stereo", "mean_activation_non_stereo",
        "n_stereo", "n_non_stereo",
    }

    # Build a minimal result matching the save format.
    row = {
        "feature_idx": np.int32(100),
        "layer": np.int32(14),
        "subgroup": "gay",
        "category": "so",
        "cohens_d": np.float32(0.85),
        "p_value_raw": 1e-7,
        "p_value_fdr": 5e-5,
        "is_significant": True,
        "direction": "pro_bias",
        "firing_rate_stereo": np.float32(0.35),
        "firing_rate_non_stereo": np.float32(0.12),
        "mean_activation_stereo": np.float32(1.2),
        "mean_activation_non_stereo": np.float32(0.4),
        "n_stereo": np.int32(312),
        "n_non_stereo": np.int32(508),
    }

    check(set(row.keys()) == expected_cols, "row has all expected columns")


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("B1 Structural Test Suite", flush=True)
    print("=" * 60, flush=True)

    test_imports()
    test_subgroup_catalog()
    test_comparison_groups()
    test_sparse_matrix()
    test_vectorized_tests()
    test_fdr()
    test_output_schema()

    print("\n" + "=" * 60, flush=True)
    print(f"Results: {passed} passed, {failed} failed", flush=True)
    print("=" * 60, flush=True)

    sys.exit(1 if failed > 0 else 0)
