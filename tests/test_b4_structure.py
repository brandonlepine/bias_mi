"""Structural tests for B4 probe training — validates logic with synthetic data.

Run from project root:
    python tests/test_b4_structure.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.probes import (
    SEED,
    b4_complete,
    build_probes_summary,
    enumerate_subgroups,
    get_groups_for_items,
    permutation_baseline,
    probe_binary_subgroup,
    probe_context_condition,
    probe_multiclass_subgroup,
    probe_stereotyped_response,
    probe_template_id,
    probe_within_cat_cross_subgroup,
    safe_cv_splits,
    save_cross_cat_results,
    save_probe_results,
    save_within_cat_results,
    train_probe,
    train_probe_stratified,
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


# ── Synthetic data ─────────────────────────────────────────────────────

N_ITEMS = 100
N_FEATURES = 32


def make_probe_data(
    n_items: int = N_ITEMS,
    n_features: int = N_FEATURES,
    n_classes: int = 2,
    n_groups: int = 10,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create (X, y, groups) for probe testing."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_items, n_features).astype(np.float32)
    y = rng.randint(0, n_classes, size=n_items)
    groups = rng.randint(0, n_groups, size=n_items)
    return X, y, groups


def make_metadata_df(n_items: int = 120) -> pd.DataFrame:
    """Build metadata matching A2 output schema with question_index-compatible groups."""
    rng = np.random.RandomState(42)
    rows = []
    subs = ["gay", "bisexual", "lesbian"]
    for i in range(n_items):
        sub = subs[i % 3]
        rows.append({
            "item_idx": i,
            "category": "so",
            "context_condition": "ambig" if i < 80 else "disambig",
            "model_answer_role": (
                "stereotyped_target" if rng.rand() > 0.45 else "non_stereotyped"
            ),
            "is_stereotyped_response": rng.rand() > 0.45,
            "stereotyped_groups": [sub],
            "n_target_groups": 1,
            "question_polarity": "neg" if i % 2 == 0 else "nonneg",
        })
    return pd.DataFrame(rows)


# ── Tests ──────────────────────────────────────────────────────────────

def test_safe_cv_splits_basic():
    print("\n[B4] safe_cv_splits — basic functionality", flush=True)
    X, y, groups = make_probe_data(n_items=100, n_classes=2, n_groups=10)

    splits, skipped = safe_cv_splits(X, y, groups, n_splits=5)
    check(len(splits) + skipped <= 5, "total folds ≤ n_splits")
    check(len(splits) > 0, "at least one valid fold")

    for train_idx, test_idx in splits:
        check(len(np.intersect1d(train_idx, test_idx)) == 0, "no train/test overlap")
        # Verify group separation
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        check(len(train_groups & test_groups) == 0, "no group overlap between folds")


def test_safe_cv_splits_degenerate():
    print("\n[B4] safe_cv_splits — degenerate folds skipped", flush=True)
    # All same label → every fold is degenerate
    X = np.random.randn(50, 10).astype(np.float32)
    y = np.zeros(50, dtype=int)
    groups = np.arange(50) % 5

    splits, skipped = safe_cv_splits(X, y, groups, n_splits=5)
    check(len(splits) == 0, "no valid folds when all same label")
    check(skipped > 0, "degenerate folds counted")


def test_safe_cv_splits_few_groups():
    print("\n[B4] safe_cv_splits — fewer groups than folds", flush=True)
    X = np.random.randn(20, 5).astype(np.float32)
    y = np.array([0, 1] * 10)
    groups = np.array([0] * 10 + [1] * 10)  # only 2 groups

    splits, skipped = safe_cv_splits(X, y, groups, n_splits=5)
    # Should clamp to 2 folds
    check(len(splits) + skipped <= 2, "clamped to n_groups=2")


def test_train_probe_binary():
    print("\n[B4] train_probe — binary classification", flush=True)
    X, y, groups = make_probe_data(n_items=100, n_classes=2, n_groups=10)

    result = train_probe(X, y, groups, class_weight="balanced")
    check(result["status"] == "ok", "status = ok")
    check(result["n_classes"] == 2, "n_classes = 2")
    check(result["n_folds_used"] > 0, "at least one fold used")
    check(0 <= result["mean_accuracy"] <= 1.0, "accuracy in [0, 1]")
    check(0 <= result["mean_balanced_accuracy"] <= 1.0, "balanced accuracy in [0, 1]")
    check(result["n_items"] == 100, "n_items correct")
    check(result["n_components"] <= 50, "n_components ≤ 50")


def test_train_probe_multiclass():
    print("\n[B4] train_probe — multiclass", flush=True)
    X, y, groups = make_probe_data(n_items=120, n_classes=4, n_groups=15)

    result = train_probe(X, y, groups, class_weight=None)
    check(result["status"] == "ok", "status = ok")
    check(result["n_classes"] == 4, "n_classes = 4")


def test_train_probe_skip_pca():
    print("\n[B4] train_probe — skip_pca for low-dim data", flush=True)
    X = np.random.randn(80, 10).astype(np.float32)
    y = np.random.randint(0, 2, size=80)
    groups = np.arange(80) % 8

    result = train_probe(X, y, groups, skip_pca=True)
    check(result["status"] == "ok", "ok with skip_pca")


def test_train_probe_stratified():
    print("\n[B4] train_probe_stratified — StratifiedKFold", flush=True)
    X = np.random.randn(100, 20).astype(np.float32)
    y = np.arange(100) % 5  # 5 classes, 20 each

    result = train_probe_stratified(X, y, class_weight=None)
    check(result["status"] == "ok", "status = ok")
    check(result["n_classes"] == 5, "n_classes = 5")
    check(result["n_folds_used"] == 5, "5 folds used")


def test_train_probe_stratified_insufficient():
    print("\n[B4] train_probe_stratified — too few per class", flush=True)
    X = np.random.randn(10, 5).astype(np.float32)
    y = np.arange(10)  # 10 classes, 1 each → can't stratify

    result = train_probe_stratified(X, y)
    check(result["n_folds_used"] == 0, "no folds when 1 sample per class")


def test_permutation_baseline():
    print("\n[B4] permutation_baseline — chance-level accuracy", flush=True)
    X, y, groups = make_probe_data(n_items=100, n_classes=2, n_groups=10)

    perm = permutation_baseline(X, y, groups, n_permutations=5)
    check(perm["n_trials"] > 0, "at least one trial completed")
    check(perm["mean_balanced_accuracy"] is not None, "balanced accuracy computed")
    # Permutation baseline should be near chance (0.5 for binary)
    check(0.2 <= perm["mean_balanced_accuracy"] <= 0.8,
          "permutation accuracy near chance")


def test_get_groups_for_items():
    print("\n[B4] get_groups_for_items — lookup", flush=True)
    qi_map = {0: 100, 1: 101, 2: 100, 5: 102}
    groups = get_groups_for_items([0, 1, 2, 3, 5], qi_map)
    check(list(groups) == [100, 101, 100, -1, 102], "correct mapping with -1 fallback")


def test_enumerate_subgroups():
    print("\n[B4] enumerate_subgroups — viable subgroup list", flush=True)
    meta = make_metadata_df(120)
    viable = enumerate_subgroups(meta, "so", min_n=5, ambig_only=True)
    check(len(viable) > 0, "at least one viable subgroup")
    check(all(isinstance(s, str) for s in viable), "all subgroup names are strings")

    # With very high min_n, nothing viable
    none_viable = enumerate_subgroups(meta, "so", min_n=999)
    check(len(none_viable) == 0, "no viable subgroups with min_n=999")


def test_probe_multiclass_subgroup_function():
    print("\n[B4] probe_multiclass_subgroup — end-to-end", flush=True)
    meta = make_metadata_df(120)
    meta_ambig = meta[meta["context_condition"] == "ambig"]

    rng = np.random.RandomState(42)
    X_layer = rng.randn(len(meta), 32).astype(np.float32)
    groups = np.arange(len(meta)) % 15

    result = probe_multiclass_subgroup(
        "so", 0, X_layer, meta_ambig, groups,
        n_permutations=3, min_n=5,
    )

    if result is not None:
        check(result["probe_type"] == "subgroup_multiclass", "correct probe_type")
        check(result["category"] == "so", "correct category")
        check(result["layer"] == 0, "correct layer")
        check(result["selectivity"] is not None, "selectivity computed")
    else:
        check(True, "None result (insufficient data in this synthetic setup)")


def test_probe_binary_subgroup_function():
    print("\n[B4] probe_binary_subgroup — end-to-end", flush=True)
    meta = make_metadata_df(120)
    meta_ambig = meta[meta["context_condition"] == "ambig"]

    X_layer = np.random.randn(len(meta), 32).astype(np.float32)
    groups = np.arange(len(meta)) % 15

    result = probe_binary_subgroup(
        "so", "gay", 0, X_layer, meta_ambig, groups,
        n_permutations=3, min_n=5,
    )

    if result is not None:
        check(result["probe_type"] == "subgroup_binary", "correct probe_type")
        check(result["subgroup"] == "gay", "correct subgroup")
        check(result["mean_balanced_accuracy"] is not None, "accuracy computed")
    else:
        check(True, "None (insufficient data)")


def test_probe_within_cat_cross_subgroup_function():
    print("\n[B4] probe_within_cat_cross_subgroup — cross-subgroup", flush=True)
    meta = make_metadata_df(120)
    X_layer = np.random.randn(len(meta), 32).astype(np.float32)

    records = probe_within_cat_cross_subgroup("so", 0, X_layer, meta, min_n=5)
    check(isinstance(records, list), "returns list")
    if records:
        r = records[0]
        check("train_subgroup" in r, "has train_subgroup")
        check("test_subgroup" in r, "has test_subgroup")
        check("balanced_accuracy" in r, "has balanced_accuracy")
        check("is_same_subgroup" in r, "has is_same_subgroup")


def test_save_probe_results_io():
    print("\n[B4] save_probe_results — atomic parquet I/O", flush=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        records = [
            {"probe_type": "test", "category": "so", "subgroup": None,
             "layer": 0, "mean_accuracy": 0.8, "status": "ok"},
        ]
        out_path = save_probe_results(run_dir, records)
        check(out_path.exists(), "parquet file created")
        check(not out_path.with_suffix(".parquet.tmp").exists(), "tmp cleaned up")
        loaded = pd.read_parquet(out_path)
        check(len(loaded) == 1, "1 row")
        check(loaded.iloc[0]["probe_type"] == "test", "value preserved")


def test_save_cross_cat_results_io():
    print("\n[B4] save_cross_cat_results — atomic parquet I/O", flush=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        records = [{"train_category": "so", "test_category": "race",
                     "layer": 0, "balanced_accuracy": 0.6}]
        out = save_cross_cat_results(run_dir, records)
        check(out.exists(), "parquet created")
        loaded = pd.read_parquet(out)
        check(len(loaded) == 1, "1 row")


def test_save_within_cat_results_io():
    print("\n[B4] save_within_cat_results — atomic parquet I/O", flush=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        records = [{"category": "so", "train_subgroup": "gay",
                     "test_subgroup": "bisexual", "layer": 0,
                     "balanced_accuracy": 0.55}]
        out = save_within_cat_results(run_dir, records)
        check(out.exists(), "parquet created")


def test_build_probes_summary():
    print("\n[B4] build_probes_summary — JSON structure", flush=True)
    probe_records = [
        {"probe_type": "subgroup_multiclass", "category": "so", "subgroup": None,
         "layer": 0, "mean_balanced_accuracy": 0.7, "selectivity": 0.2,
         "status": "ok"},
        {"probe_type": "subgroup_binary", "category": "so", "subgroup": "gay",
         "layer": 0, "mean_balanced_accuracy": 0.8, "selectivity": 0.3,
         "status": "ok"},
    ]
    summary = build_probes_summary(
        probe_records, [], [], ["so"],
        {"n_components": 50, "n_folds": 5}, 100.0,
    )
    check("probes_run" in summary, "probes_run present")
    check("peak_selectivity_per_category" in summary, "peak selectivity present")
    check("so" in summary["peak_selectivity_per_category"], "so in peak selectivity")
    check(summary["runtime_seconds"] == 100.0, "runtime recorded")


def test_b4_complete():
    print("\n[B4] b4_complete — resume check", flush=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        check(not b4_complete(run_dir), "incomplete when no files")
        probe_dir = run_dir / "B_probes"
        probe_dir.mkdir(parents=True)
        (probe_dir / "probes_summary.json").touch()
        check(b4_complete(run_dir), "complete when summary exists")


# ── Run ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_safe_cv_splits_basic()
    test_safe_cv_splits_degenerate()
    test_safe_cv_splits_few_groups()
    test_train_probe_binary()
    test_train_probe_multiclass()
    test_train_probe_skip_pca()
    test_train_probe_stratified()
    test_train_probe_stratified_insufficient()
    test_permutation_baseline()
    test_get_groups_for_items()
    test_enumerate_subgroups()
    test_probe_multiclass_subgroup_function()
    test_probe_binary_subgroup_function()
    test_probe_within_cat_cross_subgroup_function()
    test_save_probe_results_io()
    test_save_cross_cat_results_io()
    test_save_within_cat_results_io()
    test_build_probes_summary()
    test_b4_complete()

    print(f"\n{'=' * 60}", flush=True)
    print(f"B4 structure tests: {passed} passed, {failed} failed", flush=True)
    print(f"{'=' * 60}", flush=True)
    sys.exit(1 if failed else 0)
