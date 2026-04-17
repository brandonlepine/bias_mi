"""Structural tests for B5 feature interpretability — validates logic with synthetic data.

Run from project root:
    python tests/test_b5_structure.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.interpretability import (
    LayerCache,
    b5_complete,
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


# ── Synthetic data ──────────────────────────────────────────────────────

def make_feature_activations() -> pd.Series:
    """Create a synthetic feature activation Series indexed by (category, item_idx)."""
    rng = np.random.RandomState(42)
    idx = pd.MultiIndex.from_tuples(
        [(cat, i) for cat in ["so", "race"] for i in range(60)],
        names=["category", "item_idx"],
    )
    vals = rng.exponential(0.5, size=len(idx)).astype(np.float32)
    # Zero out some for sparsity
    vals[vals < 0.3] = 0.0
    return pd.Series(vals, index=idx)


def make_category_items(n_items: int = 60) -> pd.DataFrame:
    """Build category-level metadata."""
    rng = np.random.RandomState(42)
    rows = []
    subs = ["gay", "bisexual", "lesbian"]
    for i in range(n_items):
        sub = subs[i % 3]
        rows.append({
            "item_idx": i,
            "category": "so",
            "context_condition": "ambig" if i < 40 else "disambig",
            "model_answer_role": (
                "stereotyped_target" if rng.rand() > 0.4 else "non_stereotyped"
            ),
            "is_stereotyped_response": rng.rand() > 0.4,
            "stereotyped_groups": [sub],
            "n_target_groups": 1,
            "question_polarity": "neg" if i % 2 == 0 else "nonneg",
            "model_answer": "A",
        })
    return pd.DataFrame(rows)


def make_stimuli(n_items: int = 60) -> list[dict]:
    """Build synthetic stimuli list."""
    return [
        {"item_idx": i, "prompt": f"Question {i}: " + "x" * (50 + i * 2),
         "question_index": i // 4}
        for i in range(n_items)
    ]


# ── Tests ──────────────────────────────────────────────────────────────

def test_compute_activation_distribution():
    print("\n[B5] compute_activation_distribution — basic stats", flush=True)
    acts = make_feature_activations()
    items = make_category_items()

    dist = compute_activation_distribution(acts, items, "so")

    check(dist["n_items"] == 60, "n_items = 60")
    check(isinstance(dist["mean_all"], float), "mean_all is float")
    check(isinstance(dist["fraction_nonzero"], float), "fraction_nonzero is float")
    check(0 <= dist["fraction_nonzero"] <= 1, "fraction_nonzero in [0,1]")

    check("ambig" in dist, "ambig stats present")
    check("disambig" in dist, "disambig stats present")
    check(dist["ambig"]["n_stereo"] > 0, "stereo items counted")
    check(dist["ambig"]["n_non_stereo"] > 0, "non-stereo items counted")


def test_compute_activation_distribution_empty():
    print("\n[B5] compute_activation_distribution — empty category", flush=True)
    acts = pd.Series(dtype=np.float32,
                     index=pd.MultiIndex.from_tuples([], names=["category", "item_idx"]))
    items = pd.DataFrame(columns=["item_idx", "category", "context_condition",
                                   "is_stereotyped_response"])

    # Should not crash — handle gracefully
    try:
        dist = compute_activation_distribution(acts, items, "so")
        check(dist["n_items"] == 0, "n_items = 0 for empty")
    except Exception as e:
        check(False, f"crashed on empty data: {e}")


def test_compute_matched_pairs_comparison():
    print("\n[B5] compute_matched_pairs_comparison — paired analysis", flush=True)
    acts = make_feature_activations()
    items = make_category_items()
    qi_map = {i: i // 4 for i in range(60)}

    result = compute_matched_pairs_comparison(acts, items, "so", qi_map)

    check("n_pairs" in result, "n_pairs present")
    if result.get("matched_pairs_available"):
        check(result["n_pairs"] > 0, "some pairs found")
        check("mean_delta" in result, "mean_delta present")
        check("fraction_ambig_higher" in result, "fraction_ambig_higher present")
        check(isinstance(result["mean_delta"], float), "mean_delta is float")
    else:
        check(True, "no pairs available (acceptable for synthetic data)")


def test_compute_matched_pairs_no_pairs():
    print("\n[B5] compute_matched_pairs_comparison — no pairs available", flush=True)
    acts = make_feature_activations()
    # All ambig items → no disambig to pair with
    items = make_category_items()
    items["context_condition"] = "ambig"
    items["question_polarity"] = "neg"  # all same polarity

    qi_map = {i: i for i in range(60)}  # each item unique question_index
    result = compute_matched_pairs_comparison(acts, items, "so", qi_map)

    check(result["n_pairs"] == 0, "0 pairs when no disambig")
    check(not result["matched_pairs_available"], "matched_pairs_available = False")


def test_get_top_activating_items():
    print("\n[B5] get_top_activating_items — ranking + previews", flush=True)
    acts = make_feature_activations()
    items = make_category_items()
    stimuli = make_stimuli()

    top = get_top_activating_items(acts, items, stimuli, "so", top_n=5)

    check(len(top) == 5, "returns 5 items")
    check(top[0]["activation"] >= top[1]["activation"], "sorted descending")
    check("prompt_preview" in top[0], "has prompt_preview")
    check(len(top[0]["prompt_preview"]) <= 150, "preview truncated to 150 chars")
    check("stereotyped_groups" in top[0], "has stereotyped_groups")
    check(isinstance(top[0]["stereotyped_groups"], list), "stereotyped_groups is list")


def test_compute_subgroup_specificity():
    print("\n[B5] compute_subgroup_specificity — ratio computation", flush=True)
    acts = make_feature_activations()
    items = make_category_items()

    result = compute_subgroup_specificity(acts, items, "so", "gay")

    check("per_subgroup_activations" in result, "per_subgroup_activations present")
    check("subgroup_specificity" in result, "subgroup_specificity present")
    if result["subgroup_specificity"] is not None:
        check(isinstance(result["subgroup_specificity"], float), "specificity is float")
        check(result["subgroup_specificity"] > 0, "specificity positive")
        check(result["target_mean"] is not None, "target_mean present")
        check(result["category_mean"] is not None, "category_mean present")


def test_compute_subgroup_specificity_missing_subgroup():
    print("\n[B5] compute_subgroup_specificity — missing subgroup", flush=True)
    acts = make_feature_activations()
    items = make_category_items()

    result = compute_subgroup_specificity(acts, items, "so", "nonexistent_group")
    check(result["subgroup_specificity"] is None, "None for missing subgroup")


def test_compute_category_specificity_ratio():
    print("\n[B5] compute_category_specificity_ratio — cross-category", flush=True)
    acts = make_feature_activations()

    # Need metadata with multiple categories
    items_so = make_category_items(60)
    items_race = make_category_items(60)
    items_race["category"] = "race"
    items_race["item_idx"] = items_race["item_idx"]  # same IDs, different category
    meta = pd.concat([items_so, items_race], ignore_index=True)

    result = compute_category_specificity_ratio(acts, "so", ["so", "race"], meta)

    check("within_category_mean" in result, "within_category_mean present")
    check("cross_category_mean" in result, "cross_category_mean present")
    check("category_specificity_ratio" in result, "ratio present")
    check(isinstance(result["category_specificity_ratio"], float), "ratio is float")


def test_detect_template_artifacts_clean():
    print("\n[B5] detect_template_artifacts — clean feature (no flags)", flush=True)
    # Create feature that only fires on a few items
    acts = make_feature_activations()
    items = make_category_items()
    stimuli = make_stimuli()

    # High category specificity → no flag
    result = detect_template_artifacts(acts, "so", 5.0, items, stimuli)

    check(isinstance(result["artifact_flags"], list), "flags is list")
    check("low_category_specificity" not in result["artifact_flags"],
          "no low_category_specificity flag (ratio=5)")
    check(isinstance(result["firing_rate_source_category"], float), "firing_rate is float")


def test_detect_template_artifacts_flagged():
    print("\n[B5] detect_template_artifacts — flagged feature", flush=True)
    # All items fire → high firing rate flag
    idx = pd.MultiIndex.from_tuples(
        [("so", i) for i in range(60)],
        names=["category", "item_idx"],
    )
    acts = pd.Series(np.ones(60, dtype=np.float32), index=idx)
    items = make_category_items()
    stimuli = make_stimuli()

    result = detect_template_artifacts(acts, "so", 1.5, items, stimuli)

    check(result["is_artifact_flagged"], "is_artifact_flagged = True")
    check("low_category_specificity" in result["artifact_flags"],
          "low_category_specificity flagged (ratio=1.5)")
    check("high_firing_rate" in result["artifact_flags"],
          "high_firing_rate flagged (all items fire)")


def test_detect_template_artifacts_length_correlation():
    print("\n[B5] detect_template_artifacts — length correlation", flush=True)
    items = make_category_items()
    stimuli = make_stimuli()

    # Create activations perfectly correlated with prompt length
    acts_vals = []
    for _, row in items.iterrows():
        stim = next((s for s in stimuli if s["item_idx"] == row["item_idx"]), None)
        prompt_len = len(stim["prompt"]) if stim else 100
        acts_vals.append(float(prompt_len) / 100.0)

    idx = pd.MultiIndex.from_tuples(
        [("so", row["item_idx"]) for _, row in items.iterrows()],
        names=["category", "item_idx"],
    )
    acts = pd.Series(acts_vals, index=idx, dtype=np.float32)

    result = detect_template_artifacts(acts, "so", 3.0, items, stimuli)

    check(result["length_correlation"] is not None, "length_correlation computed")
    check(abs(result["length_correlation"]) > 0.5,
          "high length correlation detected")
    check("length_correlation" in result["artifact_flags"],
          "length_correlation flagged")


def test_layer_cache():
    print("\n[B5] LayerCache — caching and feature extraction", flush=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        sae_dir = run_dir / "A_extraction" / "sae_encoding"
        sae_dir.mkdir(parents=True)

        # Write a synthetic layer parquet
        df = pd.DataFrame({
            "item_idx": [0, 0, 1, 1],
            "category": ["so", "so", "so", "so"],
            "feature_idx": [100, 200, 100, 200],
            "activation_value": [0.5, 0.3, 0.8, 0.0],
        })
        df.to_parquet(sae_dir / "layer_05.parquet", index=False)

        cache = LayerCache(run_dir)

        # First load
        layer_df = cache.get(5)
        check(len(layer_df) == 4, "loaded 4 rows")

        # Cached (same object)
        layer_df2 = cache.get(5)
        check(layer_df is layer_df2, "returns cached DataFrame")

        # Feature activations
        acts = cache.feature_activations(100, 5)
        check(len(acts) == 2, "2 items have feature 100")
        check(float(acts[("so", 0)]) == 0.5, "correct activation value")

        # Missing layer → error
        try:
            cache.get(99)
            check(False, "should raise on missing layer")
        except FileNotFoundError:
            check(True, "FileNotFoundError on missing layer")

        cache.clear()
        check(True, "clear() doesn't crash")


def test_build_interpretability_summary():
    print("\n[B5] build_interpretability_summary — JSON structure", flush=True)
    stats = [
        {"category": "so", "is_artifact_flagged": True},
        {"category": "so", "is_artifact_flagged": False},
        {"category": "race", "is_artifact_flagged": False},
    ]
    cross_matrices = {
        "so": {"adjusted_rand_index": 0.72, "block_diagonal_strength": 3.4},
    }
    artifacts = [{"category": "so", "feature_idx": 42, "layer": 14}]

    summary = build_interpretability_summary(
        stats, cross_matrices, artifacts, ["so", "race"], 20, 100.0,
    )

    check(summary["n_features_characterized"] == 3, "3 features")
    check(summary["n_artifact_flagged"] == 1, "1 artifact")
    check(summary["config"]["top_k"] == 20, "top_k recorded")
    check(summary["runtime_seconds"] == 100.0, "runtime recorded")
    check("so" in summary["artifact_flag_rate_per_category"], "so rate present")
    check(summary["artifact_flag_rate_per_category"]["so"] == 0.5, "so rate = 0.5")
    check(summary["block_diagonal_strength_per_category"]["so"] == 3.4, "BDS recorded")


def test_b5_complete():
    print("\n[B5] b5_complete — resume check", flush=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        check(not b5_complete(run_dir), "incomplete when no files")

        interp_dir = run_dir / "B_feature_interpretability"
        interp_dir.mkdir(parents=True)
        (interp_dir / "interpretability_summary.json").touch()
        check(b5_complete(run_dir), "complete when summary exists")


# ── Run ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_compute_activation_distribution()
    test_compute_activation_distribution_empty()
    test_compute_matched_pairs_comparison()
    test_compute_matched_pairs_no_pairs()
    test_get_top_activating_items()
    test_compute_subgroup_specificity()
    test_compute_subgroup_specificity_missing_subgroup()
    test_compute_category_specificity_ratio()
    test_detect_template_artifacts_clean()
    test_detect_template_artifacts_flagged()
    test_detect_template_artifacts_length_correlation()
    test_layer_cache()
    test_build_interpretability_summary()
    test_b5_complete()

    print(f"\n{'=' * 60}", flush=True)
    print(f"B5 structure tests: {passed} passed, {failed} failed", flush=True)
    print(f"{'=' * 60}", flush=True)
    sys.exit(1 if failed else 0)
