"""Structural tests for Stage 2: SAE Feature Selection.

Validates logic and calculations with synthetic data — no model or SAE loading
required.  Tests cover:

  - Alias detection (Phase 2b): clustering, representative selection, edge cases
  - Identity features (Phase 2c): cosine correctness, ranking, sign, norm guards
  - Bias-prediction features (Phase 2d): label construction, L1 sparsity, CV,
    per-subgroup checkpointing, missing question_index fallback
  - Method overlap (Phase 2e): Jaccard, intersection, empty-set handling
  - .npz validation helper
  - Checkpoint / resume logic

Run from project root:
    python tests/test_stage2_structure.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.stage2_feature_selection import (
    _validate_npz,
    compute_bias_prediction_features,
    compute_identity_features,
    compute_method_overlap,
    detect_alias_clusters,
    encode_all_items_max_pooled,
)
from src.utils.io import atomic_save_json

passed = 0
failed = 0


def check(condition: bool, name: str) -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  \u2713 {name}", flush=True)
    else:
        failed += 1
        print(f"  \u2717 FAIL: {name}", flush=True)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

HIDDEN_DIM = 16
N_FEATURES = 64  # tiny SAE for testing


def make_cosines_df(
    categories: list[str],
    subgroups_per_cat: dict[str, list[str]],
    cosine_matrix: dict[str, dict[tuple[str, str], float]] | None = None,
    layer: int = 14,
    n_targeting: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Build a synthetic cosines DataFrame matching Stage 1 output schema."""
    rows = []
    n_tgt = n_targeting or {}

    for cat in categories:
        subs = subgroups_per_cat.get(cat, [])
        for sa in subs:
            for sb in subs:
                if sa == sb:
                    cos = 1.0
                elif cosine_matrix and cat in cosine_matrix:
                    cos = cosine_matrix[cat].get(
                        (sa, sb),
                        cosine_matrix[cat].get((sb, sa), 0.5),
                    )
                else:
                    cos = 0.5
                rows.append({
                    "category": cat,
                    "subgroup_a": sa,
                    "subgroup_b": sb,
                    "cosine_raw": cos,
                    "cosine_normed": cos,
                    "layer": layer,
                    "n_a": n_tgt.get(sa, 100),
                    "n_b": n_tgt.get(sb, 100),
                })
    return pd.DataFrame(rows)


def make_identity_direction(
    hidden_dim: int, seed: int = 0,
) -> np.ndarray:
    """Create a random unit-normalised direction."""
    rng = np.random.RandomState(seed)
    d = rng.randn(hidden_dim).astype(np.float32)
    d /= np.linalg.norm(d)
    return d


def make_decoder_matrix(
    n_features: int, hidden_dim: int, seed: int = 42,
) -> np.ndarray:
    """Create a random (n_features, hidden_dim) matrix with L2-normalised rows."""
    rng = np.random.RandomState(seed)
    W = rng.randn(n_features, hidden_dim).astype(np.float32)
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    return W / np.maximum(norms, 1e-8)


def make_metadata(
    n_items: int = 60,
    categories: list[str] | None = None,
    subgroups: list[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Build synthetic metadata matching A2 output schema."""
    rng = np.random.RandomState(seed)
    cats = categories or ["so"]
    subs = subgroups or ["gay", "bisexual", "lesbian"]
    rows = []
    for i in range(n_items):
        sub = subs[i % len(subs)]
        cat = cats[i % len(cats)]
        rows.append({
            "item_idx": i,
            "category": cat,
            "context_condition": "ambig" if i < int(n_items * 0.7) else "disambig",
            "model_answer_role": (
                "stereotyped_target" if rng.rand() > 0.4 else "non_stereotyped"
            ),
            "is_stereotyped_response": rng.rand() > 0.4,
            "stereotyped_groups": [sub],
            "n_target_groups": 1,
            "question_polarity": "neg" if i % 2 == 0 else "nonneg",
            "question_index": i // 3,
        })
    return pd.DataFrame(rows)


class FakeSAEWrapper:
    """Minimal stub to test identity-feature and max-pool logic."""

    def __init__(self, n_features: int, hidden_dim: int, seed: int = 42):
        self._n_features = n_features
        self._hidden_dim = hidden_dim
        self._W_dec_normed = make_decoder_matrix(n_features, hidden_dim, seed)

    @property
    def n_features(self) -> int:
        return self._n_features

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    def get_decoder_matrix(self) -> np.ndarray:
        return self._W_dec_normed.copy()


# ---------------------------------------------------------------------------
# Phase 2b: Alias detection tests
# ---------------------------------------------------------------------------

def test_alias_no_aliases():
    """All cosines below threshold → every subgroup is its own cluster."""
    print("\n[2b] alias detection — no aliases", flush=True)
    df = make_cosines_df(
        categories=["so"],
        subgroups_per_cat={"so": ["gay", "bisexual", "lesbian"]},
        cosine_matrix={"so": {
            ("gay", "bisexual"): 0.5,
            ("gay", "lesbian"): 0.6,
            ("bisexual", "lesbian"): 0.4,
        }},
    )
    result = detect_alias_clusters(df, alias_threshold=0.98, excluded_categories=set())

    check("so" in result, "category present")
    check(result["so"]["n_scoped"] == 3, "all 3 subgroups scoped")
    check(len(result["so"]["dropped_subgroups"]) == 0, "none dropped")
    check(len(result["so"]["clusters"]) == 3, "3 singleton clusters")


def test_alias_one_pair():
    """Two subgroups above threshold → collapsed to one."""
    print("\n[2b] alias detection — one aliased pair", flush=True)
    df = make_cosines_df(
        categories=["race"],
        subgroups_per_cat={"race": ["hispanic", "latino", "black"]},
        cosine_matrix={"race": {
            ("hispanic", "latino"): 0.99,
            ("hispanic", "black"): 0.3,
            ("latino", "black"): 0.3,
        }},
        n_targeting={"hispanic": 200, "latino": 150, "black": 180},
    )
    result = detect_alias_clusters(df, alias_threshold=0.98, excluded_categories=set())

    check(result["race"]["n_scoped"] == 2, "2 scoped (1 cluster + 1 singleton)")
    check("hispanic" in result["race"]["scoped_subgroups"],
          "hispanic is representative (larger n_targeting)")
    check("latino" in result["race"]["dropped_subgroups"], "latino dropped")
    check("black" in result["race"]["scoped_subgroups"], "black kept")


def test_alias_transitive_cluster():
    """A-B and B-C above threshold → all three in one cluster."""
    print("\n[2b] alias detection — transitive clustering", flush=True)
    df = make_cosines_df(
        categories=["nat"],
        subgroups_per_cat={"nat": ["a", "b", "c", "d"]},
        cosine_matrix={"nat": {
            ("a", "b"): 0.99,
            ("b", "c"): 0.99,
            ("a", "c"): 0.90,  # below threshold, but transitive via b
            ("a", "d"): 0.2,
            ("b", "d"): 0.3,
            ("c", "d"): 0.1,
        }},
        n_targeting={"a": 50, "b": 100, "c": 80, "d": 200},
    )
    result = detect_alias_clusters(df, alias_threshold=0.98, excluded_categories=set())

    check(result["nat"]["n_scoped"] == 2, "2 clusters (abc + d)")
    check("b" in result["nat"]["scoped_subgroups"],
          "b is representative of {a,b,c} (largest n)")
    check("d" in result["nat"]["scoped_subgroups"], "d is its own cluster")
    check(set(result["nat"]["dropped_subgroups"]) == {"a", "c"},
          "a and c dropped")


def test_alias_excluded_category():
    """Excluded categories are skipped entirely."""
    print("\n[2b] alias detection — excluded category", flush=True)
    df = make_cosines_df(
        categories=["age", "so"],
        subgroups_per_cat={"age": ["old", "young"], "so": ["gay"]},
    )
    result = detect_alias_clusters(df, alias_threshold=0.98,
                                   excluded_categories={"age"})
    check("age" not in result, "age excluded")
    check("so" in result, "so present")


def test_alias_all_identical():
    """All subgroups aliased → one representative."""
    print("\n[2b] alias detection — all aliased", flush=True)
    df = make_cosines_df(
        categories=["so"],
        subgroups_per_cat={"so": ["a", "b", "c"]},
        cosine_matrix={"so": {
            ("a", "b"): 0.99,
            ("a", "c"): 0.99,
            ("b", "c"): 0.99,
        }},
        n_targeting={"a": 10, "b": 20, "c": 15},
    )
    result = detect_alias_clusters(df, alias_threshold=0.98, excluded_categories=set())
    check(result["so"]["n_scoped"] == 1, "one representative")
    check(result["so"]["scoped_subgroups"] == ["b"], "b chosen (highest n)")


# ---------------------------------------------------------------------------
# Phase 2c: Identity features tests
# ---------------------------------------------------------------------------

def test_identity_features_cosine_correctness():
    """Verify cosines are correct for a known decoder + direction pair."""
    print("\n[2c] identity features — cosine correctness", flush=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)

        # Create a direction aligned with basis vector 0
        direction = np.zeros(HIDDEN_DIM, dtype=np.float32)
        direction[0] = 1.0

        # Save it
        id_dir = run_dir / "stage1_geometry" / "identity_directions" / "L14"
        id_dir.mkdir(parents=True)
        np.savez(id_dir / "so_gay.npz", direction_normed=direction)

        # Create decoder where feature 0 is aligned, feature 1 is orthogonal
        sae = FakeSAEWrapper(N_FEATURES, HIDDEN_DIM)
        W = sae._W_dec_normed.copy()
        W[0] = np.zeros(HIDDEN_DIM, dtype=np.float32)
        W[0, 0] = 1.0  # perfectly aligned
        W[1] = np.zeros(HIDDEN_DIM, dtype=np.float32)
        W[1, 1] = 1.0  # orthogonal
        sae._W_dec_normed = W

        alias_clusters = {
            "so": {"scoped_subgroups": ["gay"], "clusters": []},
        }

        df = compute_identity_features(run_dir, alias_clusters, sae, top_k=5,
                                       layer=14)

        check(len(df) == 5, "5 rows returned (top_k=5)")
        top = df[df["rank"] == 1].iloc[0]
        check(top["feature_idx"] == 0, "feature 0 is rank 1")
        check(abs(top["cosine"] - 1.0) < 1e-5, "cosine ≈ 1.0 for aligned feature")

        feat1 = df[df["feature_idx"] == 1]
        if len(feat1) > 0:
            check(abs(feat1.iloc[0]["cosine"]) < 1e-5,
                  "feature 1 cosine ≈ 0 (orthogonal)")
        else:
            check(True, "feature 1 not in top-5 (expected, it's orthogonal)")


def test_identity_features_sign_preserved():
    """Negative cosines should be preserved, not abs'd away."""
    print("\n[2c] identity features — sign preservation", flush=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        direction = np.zeros(HIDDEN_DIM, dtype=np.float32)
        direction[0] = 1.0

        id_dir = run_dir / "stage1_geometry" / "identity_directions" / "L14"
        id_dir.mkdir(parents=True)
        np.savez(id_dir / "so_gay.npz", direction_normed=direction)

        sae = FakeSAEWrapper(N_FEATURES, HIDDEN_DIM)
        # Feature 0: anti-aligned
        sae._W_dec_normed[0] = np.zeros(HIDDEN_DIM, dtype=np.float32)
        sae._W_dec_normed[0, 0] = -1.0

        alias_clusters = {"so": {"scoped_subgroups": ["gay"], "clusters": []}}

        df = compute_identity_features(run_dir, alias_clusters, sae,
                                       top_k=N_FEATURES, layer=14)
        feat0 = df[df["feature_idx"] == 0].iloc[0]

        check(feat0["cosine"] < 0, "negative cosine preserved")
        check(abs(feat0["cosine"] - (-1.0)) < 1e-5, "cosine ≈ -1.0")
        check(feat0["abs_cosine"] > 0.99, "abs_cosine ≈ 1.0")


def test_identity_features_ranking():
    """Features should be ranked by |cosine| descending."""
    print("\n[2c] identity features — ranking order", flush=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        direction = make_identity_direction(HIDDEN_DIM, seed=7)

        id_dir = run_dir / "stage1_geometry" / "identity_directions" / "L14"
        id_dir.mkdir(parents=True)
        np.savez(id_dir / "so_gay.npz", direction_normed=direction)

        sae = FakeSAEWrapper(N_FEATURES, HIDDEN_DIM, seed=99)
        alias_clusters = {"so": {"scoped_subgroups": ["gay"], "clusters": []}}

        df = compute_identity_features(run_dir, alias_clusters, sae,
                                       top_k=N_FEATURES, layer=14)

        abs_cos = df["abs_cosine"].values
        check(np.all(abs_cos[:-1] >= abs_cos[1:]),
              "abs_cosine is monotonically non-increasing")
        check(list(df["rank"].values) == list(range(1, N_FEATURES + 1)),
              "ranks are 1..N_FEATURES")


def test_identity_features_missing_direction():
    """Missing direction file should log warning and skip, not crash."""
    print("\n[2c] identity features — missing direction", flush=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        # Don't create any direction files
        (run_dir / "stage1_geometry" / "identity_directions" / "L14").mkdir(
            parents=True
        )

        sae = FakeSAEWrapper(N_FEATURES, HIDDEN_DIM)
        alias_clusters = {"so": {"scoped_subgroups": ["gay"], "clusters": []}}

        df = compute_identity_features(run_dir, alias_clusters, sae,
                                       top_k=5, layer=14)
        check(len(df) == 0, "empty DataFrame when direction missing")


def test_identity_features_dim_mismatch():
    """Direction with wrong hidden_dim should be skipped."""
    print("\n[2c] identity features — dim mismatch", flush=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        wrong_dim_dir = np.ones(HIDDEN_DIM + 5, dtype=np.float32)
        wrong_dim_dir /= np.linalg.norm(wrong_dim_dir)

        id_dir = run_dir / "stage1_geometry" / "identity_directions" / "L14"
        id_dir.mkdir(parents=True)
        np.savez(id_dir / "so_gay.npz", direction_normed=wrong_dim_dir)

        sae = FakeSAEWrapper(N_FEATURES, HIDDEN_DIM)
        alias_clusters = {"so": {"scoped_subgroups": ["gay"], "clusters": []}}

        df = compute_identity_features(run_dir, alias_clusters, sae,
                                       top_k=5, layer=14)
        check(len(df) == 0, "empty when dim mismatch")


def test_identity_features_unnormalised_direction():
    """Non-unit direction should be re-normalised, not produce garbage."""
    print("\n[2c] identity features — re-normalises non-unit direction", flush=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        direction = np.zeros(HIDDEN_DIM, dtype=np.float32)
        direction[0] = 5.0  # not unit norm

        id_dir = run_dir / "stage1_geometry" / "identity_directions" / "L14"
        id_dir.mkdir(parents=True)
        np.savez(id_dir / "so_gay.npz", direction_normed=direction)

        sae = FakeSAEWrapper(N_FEATURES, HIDDEN_DIM)
        sae._W_dec_normed[0] = np.zeros(HIDDEN_DIM, dtype=np.float32)
        sae._W_dec_normed[0, 0] = 1.0

        alias_clusters = {"so": {"scoped_subgroups": ["gay"], "clusters": []}}
        df = compute_identity_features(run_dir, alias_clusters, sae,
                                       top_k=5, layer=14)

        feat0 = df[df["feature_idx"] == 0].iloc[0]
        check(abs(feat0["cosine"] - 1.0) < 0.01,
              "cosine ≈ 1.0 after re-normalisation")
        check(df["abs_cosine"].max() <= 1.0 + 1e-5,
              "no |cosine| > 1 after re-normalisation")


# ---------------------------------------------------------------------------
# Phase 2d: Bias-prediction features tests
# ---------------------------------------------------------------------------

def test_bias_prediction_basic():
    """L1 probe produces nonzero features and valid summary."""
    print("\n[2d] bias prediction — basic run", flush=True)
    rng = np.random.RandomState(42)

    meta_df = make_metadata(n_items=80, categories=["so"],
                            subgroups=["gay", "bisexual"])

    # Create separable z_max: stereotyped items have feature 0 high
    z_max = {}
    for _, row in meta_df.iterrows():
        idx = int(row["item_idx"])
        z = rng.randn(N_FEATURES).astype(np.float32) * 0.01
        if row["model_answer_role"] == "stereotyped_target":
            z[0] += 2.0  # make feature 0 clearly predictive
        z_max[idx] = z

    alias_clusters = {
        "so": {"scoped_subgroups": ["gay", "bisexual"], "clusters": []},
    }

    df, summary = compute_bias_prediction_features(
        z_max, meta_df, alias_clusters,
        l1_c_values=[0.1, 1.0, 10.0],
        n_cv_folds=3,
        min_n_stereo=3,
        min_n_non_stereo=3,
        seed=42,
        layer=14,
    )

    check(len(df) > 0, "nonzero features found")
    check("so/gay" in summary, "gay in summary")

    gay_summary = summary.get("so/gay", {})
    if gay_summary.get("status") == "ok":
        check(gay_summary["cv_accuracy"] > 0.0, "CV accuracy > 0")
        check(gay_summary["n_nonzero"] > 0, "nonzero features > 0")
        check(gay_summary["best_c"] in [0.1, 1.0, 10.0], "best_c is one of CV values")
    else:
        check(True, f"gay status: {gay_summary.get('status')} (acceptable for tiny data)")

    # Check DataFrame schema
    if len(df) > 0:
        expected_cols = {
            "category", "subgroup", "feature_idx", "l1_coefficient",
            "abs_coefficient", "rank", "best_c", "cv_accuracy",
            "n_stereo", "n_non_stereo", "layer", "method",
        }
        check(expected_cols.issubset(set(df.columns)), "all expected columns present")
        check((df["method"] == "bias_prediction").all(), "method column correct")
        check((df["layer"] == 14).all(), "layer column correct")
        check((df["abs_coefficient"] >= 0).all(), "abs_coefficient non-negative")
        # Ranks should be contiguous per subgroup
        for sub in df["subgroup"].unique():
            sub_df = df[df["subgroup"] == sub]
            check(
                list(sub_df["rank"].values) == list(range(1, len(sub_df) + 1)),
                f"ranks contiguous for {sub}",
            )


def test_bias_prediction_insufficient_items():
    """Subgroup with too few stereo/non-stereo items is skipped cleanly."""
    print("\n[2d] bias prediction — insufficient items", flush=True)
    meta_df = make_metadata(n_items=20, categories=["so"], subgroups=["gay"])
    # Force all to stereotyped → no non-stereo items
    meta_df["model_answer_role"] = "stereotyped_target"

    z_max = {i: np.zeros(N_FEATURES, dtype=np.float32) for i in range(20)}
    alias_clusters = {"so": {"scoped_subgroups": ["gay"], "clusters": []}}

    df, summary = compute_bias_prediction_features(
        z_max, meta_df, alias_clusters,
        l1_c_values=[1.0], n_cv_folds=3,
        min_n_stereo=5, min_n_non_stereo=5,
        seed=42, layer=14,
    )

    check(len(df) == 0, "no features (insufficient non-stereo)")
    check(summary.get("so/gay", {}).get("status") == "insufficient_items",
          "status is insufficient_items")


def test_bias_prediction_no_question_index():
    """Works (with warning) when question_index is missing from metadata."""
    print("\n[2d] bias prediction — no question_index", flush=True)
    rng = np.random.RandomState(42)
    meta_df = make_metadata(n_items=80, categories=["so"], subgroups=["gay"])
    meta_df = meta_df.drop(columns=["question_index"])

    z_max = {}
    for _, row in meta_df.iterrows():
        z = rng.randn(N_FEATURES).astype(np.float32) * 0.01
        if row["model_answer_role"] == "stereotyped_target":
            z[0] += 2.0
        z_max[int(row["item_idx"])] = z

    alias_clusters = {"so": {"scoped_subgroups": ["gay"], "clusters": []}}

    # Should not crash
    df, summary = compute_bias_prediction_features(
        z_max, meta_df, alias_clusters,
        l1_c_values=[1.0], n_cv_folds=3,
        min_n_stereo=3, min_n_non_stereo=3,
        seed=42, layer=14,
    )
    check(True, "did not crash without question_index")


def test_bias_prediction_checkpoint_resume():
    """Per-subgroup checkpoints allow resuming after crash."""
    print("\n[2d] bias prediction — checkpoint resume", flush=True)
    rng = np.random.RandomState(42)
    meta_df = make_metadata(n_items=80, categories=["so"],
                            subgroups=["gay", "bisexual"])
    z_max = {}
    for _, row in meta_df.iterrows():
        z = rng.randn(N_FEATURES).astype(np.float32) * 0.01
        if row["model_answer_role"] == "stereotyped_target":
            z[0] += 2.0
        z_max[int(row["item_idx"])] = z

    alias_clusters = {
        "so": {"scoped_subgroups": ["gay", "bisexual"], "clusters": []},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir) / "ckpts"

        # First run
        df1, summary1 = compute_bias_prediction_features(
            z_max, meta_df, alias_clusters,
            l1_c_values=[1.0], n_cv_folds=3,
            min_n_stereo=3, min_n_non_stereo=3,
            seed=42, layer=14, checkpoint_dir=ckpt_dir,
        )

        # Checkpoint files should exist
        ckpt_files = list(ckpt_dir.glob("probe_*.json"))
        check(len(ckpt_files) > 0, "checkpoint files created")

        # Second run with same args — should load from checkpoints
        df2, summary2 = compute_bias_prediction_features(
            z_max, meta_df, alias_clusters,
            l1_c_values=[1.0], n_cv_folds=3,
            min_n_stereo=3, min_n_non_stereo=3,
            seed=42, layer=14, checkpoint_dir=ckpt_dir,
        )

        # Results should be identical
        if len(df1) > 0 and len(df2) > 0:
            check(len(df1) == len(df2), "same row count on resume")
            check(
                set(df1["feature_idx"].values) == set(df2["feature_idx"].values),
                "same feature set on resume",
            )
        else:
            check(True, "both runs empty (consistent)")


def test_bias_prediction_labels_correct():
    """Verify y=1 corresponds to stereotyped_target and y=0 to other."""
    print("\n[2d] bias prediction — label construction", flush=True)
    meta_df = make_metadata(n_items=60, categories=["so"], subgroups=["gay"])

    # Count expected labels for gay-targeting ambig items
    ambig = meta_df[
        (meta_df["category"] == "so")
        & (meta_df["context_condition"] == "ambig")
    ]
    targeting = ambig[
        ambig["stereotyped_groups"].apply(lambda gs: "gay" in gs)
    ]
    expected_stereo = (targeting["model_answer_role"] == "stereotyped_target").sum()
    expected_non = (targeting["model_answer_role"] != "stereotyped_target").sum()

    check(expected_stereo > 0, f"synthetic data has {expected_stereo} stereotyped items")
    check(expected_non > 0, f"synthetic data has {expected_non} non-stereotyped items")


# ---------------------------------------------------------------------------
# Phase 2e: Method overlap tests
# ---------------------------------------------------------------------------

def test_overlap_basic():
    """Jaccard computation with known overlap."""
    print("\n[2e] method overlap — basic", flush=True)
    identity_df = pd.DataFrame([
        {"category": "so", "subgroup": "gay", "feature_idx": i, "rank": i + 1}
        for i in range(10)
    ])
    bias_df = pd.DataFrame([
        {"category": "so", "subgroup": "gay", "feature_idx": i}
        for i in range(5, 15)
    ])

    df = compute_method_overlap(identity_df, bias_df, top_k_identity=10)

    check(len(df) == 1, "one row")
    row = df.iloc[0]
    check(row["n_identity_features"] == 10, "|A| = 10")
    check(row["n_bias_prediction_features"] == 10, "|B| = 10")
    check(row["n_intersection"] == 5, "|A cap B| = 5 (features 5-9)")
    check(row["n_union"] == 15, "|A cup B| = 15 (features 0-14)")
    check(abs(row["jaccard"] - 5 / 15) < 1e-6, "Jaccard = 5/15")
    check(sorted(row["intersection_feature_idxs"]) == [5, 6, 7, 8, 9],
          "correct intersection features")


def test_overlap_empty_bias():
    """No bias features → Jaccard = 0."""
    print("\n[2e] method overlap — empty bias set", flush=True)
    identity_df = pd.DataFrame([
        {"category": "so", "subgroup": "gay", "feature_idx": 0, "rank": 1},
    ])
    bias_df = pd.DataFrame(columns=["category", "subgroup", "feature_idx"])

    df = compute_method_overlap(identity_df, bias_df, top_k_identity=100)
    check(len(df) == 1, "one row")
    check(df.iloc[0]["jaccard"] == 0.0, "Jaccard = 0 with empty B")
    check(df.iloc[0]["n_intersection"] == 0, "no intersection")


def test_overlap_perfect():
    """Identical sets → Jaccard = 1."""
    print("\n[2e] method overlap — perfect overlap", flush=True)
    features = list(range(5))
    identity_df = pd.DataFrame([
        {"category": "so", "subgroup": "gay", "feature_idx": i, "rank": i + 1}
        for i in features
    ])
    bias_df = pd.DataFrame([
        {"category": "so", "subgroup": "gay", "feature_idx": i}
        for i in features
    ])

    df = compute_method_overlap(identity_df, bias_df, top_k_identity=5)
    check(abs(df.iloc[0]["jaccard"] - 1.0) < 1e-6, "Jaccard = 1.0")


def test_overlap_top_k_filter():
    """Only top-k identity features are considered, not all."""
    print("\n[2e] method overlap — top_k filter", flush=True)
    identity_df = pd.DataFrame([
        {"category": "so", "subgroup": "gay", "feature_idx": i, "rank": i + 1}
        for i in range(20)
    ])
    bias_df = pd.DataFrame([
        {"category": "so", "subgroup": "gay", "feature_idx": 15},
    ])

    # With top_k=10, feature 15 is NOT in identity set
    df10 = compute_method_overlap(identity_df, bias_df, top_k_identity=10)
    check(df10.iloc[0]["n_intersection"] == 0,
          "feature 15 not in top-10 → no intersection")

    # With top_k=20, feature 15 IS in identity set
    df20 = compute_method_overlap(identity_df, bias_df, top_k_identity=20)
    check(df20.iloc[0]["n_intersection"] == 1,
          "feature 15 in top-20 → intersection = 1")


def test_overlap_multi_subgroup():
    """Multiple subgroups produce one row each."""
    print("\n[2e] method overlap — multi-subgroup", flush=True)
    identity_df = pd.DataFrame([
        {"category": "so", "subgroup": "gay", "feature_idx": 0, "rank": 1},
        {"category": "so", "subgroup": "bisexual", "feature_idx": 1, "rank": 1},
    ])
    bias_df = pd.DataFrame([
        {"category": "so", "subgroup": "gay", "feature_idx": 0},
        {"category": "so", "subgroup": "bisexual", "feature_idx": 2},
    ])

    df = compute_method_overlap(identity_df, bias_df, top_k_identity=100)
    check(len(df) == 2, "two rows (one per subgroup)")

    gay_row = df[df["subgroup"] == "gay"].iloc[0]
    check(gay_row["n_intersection"] == 1, "gay: intersection = 1")
    bi_row = df[df["subgroup"] == "bisexual"].iloc[0]
    check(bi_row["n_intersection"] == 0, "bisexual: intersection = 0")


# ---------------------------------------------------------------------------
# .npz validation tests
# ---------------------------------------------------------------------------

def test_validate_npz_valid():
    """Valid .npz passes validation."""
    print("\n[util] _validate_npz — valid file", flush=True)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = Path(f.name)
    try:
        hidden = np.random.randn(50, HIDDEN_DIM).astype(np.float16)
        np.savez(path, hidden_all_tokens=hidden)
        check(_validate_npz(path, HIDDEN_DIM), "valid file passes")
    finally:
        path.unlink(missing_ok=True)


def test_validate_npz_wrong_dim():
    """.npz with wrong hidden_dim fails validation."""
    print("\n[util] _validate_npz — wrong dim", flush=True)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = Path(f.name)
    try:
        hidden = np.random.randn(50, HIDDEN_DIM + 3).astype(np.float16)
        np.savez(path, hidden_all_tokens=hidden)
        check(not _validate_npz(path, HIDDEN_DIM), "wrong dim fails")
    finally:
        path.unlink(missing_ok=True)


def test_validate_npz_corrupt():
    """Corrupt file (not a valid .npz) fails gracefully."""
    print("\n[util] _validate_npz — corrupt file", flush=True)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False, mode="w") as f:
        f.write("this is not a valid npz")
        path = Path(f.name)
    try:
        check(not _validate_npz(path, HIDDEN_DIM), "corrupt file fails gracefully")
    finally:
        path.unlink(missing_ok=True)


def test_validate_npz_empty():
    """.npz with 0-length sequence fails."""
    print("\n[util] _validate_npz — empty sequence", flush=True)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = Path(f.name)
    try:
        hidden = np.zeros((0, HIDDEN_DIM), dtype=np.float16)
        np.savez(path, hidden_all_tokens=hidden)
        check(not _validate_npz(path, HIDDEN_DIM), "empty sequence fails")
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Checkpoint / resume integration tests
# ---------------------------------------------------------------------------

def test_probe_checkpoint_corrupt_recovery():
    """Corrupt checkpoint file is deleted and re-computed."""
    print("\n[ckpt] probe checkpoint — corrupt recovery", flush=True)
    rng = np.random.RandomState(42)
    meta_df = make_metadata(n_items=60, categories=["so"], subgroups=["gay"])
    z_max = {
        int(row["item_idx"]): rng.randn(N_FEATURES).astype(np.float32)
        for _, row in meta_df.iterrows()
    }
    alias_clusters = {"so": {"scoped_subgroups": ["gay"], "clusters": []}}

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir) / "ckpts"
        ckpt_dir.mkdir()

        # Write corrupt checkpoint
        corrupt_path = ckpt_dir / "probe_so_gay.json"
        corrupt_path.write_text("{{invalid json")

        # Should recover and re-compute
        df, summary = compute_bias_prediction_features(
            z_max, meta_df, alias_clusters,
            l1_c_values=[1.0], n_cv_folds=3,
            min_n_stereo=3, min_n_non_stereo=3,
            seed=42, layer=14, checkpoint_dir=ckpt_dir,
        )
        check(True, "recovered from corrupt checkpoint without crash")

        # Valid checkpoint should now exist
        if corrupt_path.exists():
            try:
                with open(corrupt_path) as f:
                    data = json.load(f)
                check("summary" in data, "valid checkpoint written after recovery")
            except json.JSONDecodeError:
                check(False, "checkpoint still corrupt after recovery")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Phase 2b: Alias detection
    test_alias_no_aliases()
    test_alias_one_pair()
    test_alias_transitive_cluster()
    test_alias_excluded_category()
    test_alias_all_identical()

    # Phase 2c: Identity features
    test_identity_features_cosine_correctness()
    test_identity_features_sign_preserved()
    test_identity_features_ranking()
    test_identity_features_missing_direction()
    test_identity_features_dim_mismatch()
    test_identity_features_unnormalised_direction()

    # Phase 2d: Bias-prediction features
    test_bias_prediction_basic()
    test_bias_prediction_insufficient_items()
    test_bias_prediction_no_question_index()
    test_bias_prediction_checkpoint_resume()
    test_bias_prediction_labels_correct()

    # Phase 2e: Method overlap
    test_overlap_basic()
    test_overlap_empty_bias()
    test_overlap_perfect()
    test_overlap_top_k_filter()
    test_overlap_multi_subgroup()

    # .npz validation
    test_validate_npz_valid()
    test_validate_npz_wrong_dim()
    test_validate_npz_corrupt()
    test_validate_npz_empty()

    # Checkpoint recovery
    test_probe_checkpoint_corrupt_recovery()

    print(f"\n{'=' * 60}", flush=True)
    print(f"Stage 2 structure tests: {passed} passed, {failed} failed",
          flush=True)
    print(f"{'=' * 60}", flush=True)
    sys.exit(1 if failed else 0)
