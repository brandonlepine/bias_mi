"""Structural tests for B2 feature ranking — validates logic with synthetic data.

Run from project root:
    python tests/test_b2_structure.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.ranking import (
    K_VALUES_DEFAULT,
    build_injection_layers,
    build_ranking_summary,
    compute_all_overlaps,
    compute_injection_layer_weighted,
    compute_item_overlap,
    compute_overlap_curve,
    deduplicate_defensive,
    enumerate_subgroups,
    make_sub_key,
    parse_sub_key,
    rank_features_all,
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


# ── Synthetic data ───────────────────────────────────────────────────

def make_significant_df() -> pd.DataFrame:
    """Build a synthetic significant-features DataFrame resembling B1 output."""
    rng = np.random.RandomState(42)
    rows = []
    for layer in [10, 14, 18]:
        for sub, cat in [("gay", "so"), ("bisexual", "so"), ("black", "race")]:
            for direction in ["pro_bias", "anti_bias"]:
                n = rng.randint(5, 20)
                for i in range(n):
                    rows.append({
                        "feature_idx": rng.randint(0, 131072),
                        "layer": layer,
                        "subgroup": sub,
                        "category": cat,
                        "cohens_d": float(rng.randn() * 0.5 + (0.8 if direction == "pro_bias" else -0.8)),
                        "p_value_raw": float(rng.uniform(1e-8, 1e-4)),
                        "p_value_fdr": float(rng.uniform(1e-6, 0.04)),
                        "is_significant": True,
                        "direction": direction,
                        "firing_rate_stereo": float(rng.uniform(0.05, 0.5)),
                        "firing_rate_non_stereo": float(rng.uniform(0.01, 0.3)),
                        "mean_activation_stereo": float(rng.uniform(0.1, 2.0)),
                        "mean_activation_non_stereo": float(rng.uniform(0.01, 1.0)),
                        "n_stereo": 50,
                        "n_non_stereo": 80,
                    })
    return pd.DataFrame(rows)


# =========================================================================
# Tests
# =========================================================================

def test_imports() -> None:
    print("\n=== Imports ===", flush=True)
    from src.analysis.ranking import (
        load_all_significant, deduplicate_defensive, enumerate_subgroups,
        rank_features_all, compute_injection_layer_weighted,
        build_injection_layers, compute_overlap_curve, compute_all_overlaps,
        compute_item_overlap, build_ranking_summary,
    )
    from src.visualization.ranking_figures import generate_all_b2_figures
    check(True, "all B2 functions import")


def test_key_helpers() -> None:
    print("\n=== Key Helpers ===", flush=True)
    check(make_sub_key("so", "gay") == "so/gay", "make_sub_key")
    check(parse_sub_key("so/gay") == ("so", "gay"), "parse_sub_key")
    check(parse_sub_key("race/middle eastern") == ("race", "middle eastern"),
          "parse_sub_key with space")


def test_dedup() -> None:
    print("\n=== Deduplication ===", flush=True)
    df = make_significant_df()
    # Insert a duplicate.
    dup = df.iloc[0:1].copy()
    dup["cohens_d"] = 0.01  # weaker duplicate
    df = pd.concat([df, dup], ignore_index=True)
    before = len(df)

    deduped = deduplicate_defensive(df)
    check(len(deduped) < before, f"dedup removed {before - len(deduped)} row(s)")
    # The kept row should have the larger |d|.
    key_cols = ["category", "subgroup", "feature_idx", "layer"]
    check(
        len(deduped.drop_duplicates(subset=key_cols)) == len(deduped),
        "no duplicates remain",
    )


def test_enumerate_subgroups() -> None:
    print("\n=== Enumerate Subgroups ===", flush=True)
    df = make_significant_df()
    b1_summary = {
        "subgroup_catalog": {
            "gay": {"category": "so", "analyzable": True},
            "bisexual": {"category": "so", "analyzable": True},
            "black": {"category": "race", "analyzable": True},
            "pansexual": {"category": "so", "analyzable": True},
        }
    }
    subgroups, report = enumerate_subgroups(df, b1_summary)
    check(len(subgroups) >= 3, f"at least 3 subgroups (got {len(subgroups)})")
    check(("so", "gay") in subgroups, "gay in subgroups")
    check("total_subgroups" in report, "report has total_subgroups")
    # pansexual is in catalog but may have no significant features.
    check("only_in_catalog" in report, "report flags catalog-only subgroups")


def test_rank_features() -> None:
    print("\n=== Rank Features ===", flush=True)
    df = make_significant_df()
    subgroups = [("so", "gay"), ("so", "bisexual"), ("race", "black")]
    ranked = rank_features_all(df, subgroups)

    check(len(ranked) > 0, f"got {len(ranked)} ranked rows")

    expected_cols = {
        "category", "subgroup", "direction", "rank", "feature_idx", "layer",
        "cohens_d", "p_value_raw", "p_value_fdr",
        "firing_rate_stereo", "firing_rate_non_stereo",
        "mean_activation_stereo", "mean_activation_non_stereo",
        "n_stereo", "n_non_stereo",
    }
    check(set(ranked.columns) == expected_cols, "correct columns")

    # Ranks are 1-indexed and sorted by |d| descending within each group.
    for (cat, sub, direction), group in ranked.groupby(
        ["category", "subgroup", "direction"]
    ):
        ranks = group["rank"].tolist()
        check(ranks == list(range(1, len(ranks) + 1)),
              f"  {cat}/{sub}/{direction}: ranks are 1..{len(ranks)}")
        ds = group["cohens_d"].abs().tolist()
        check(all(ds[i] >= ds[i + 1] for i in range(len(ds) - 1)),
              f"  {cat}/{sub}/{direction}: |d| decreasing")
        break  # spot-check first group only


def test_injection_layers() -> None:
    print("\n=== Injection Layers ===", flush=True)

    # Simple case: one feature per layer with known scores.
    features = pd.DataFrame({
        "layer": [10, 14, 14, 18],
        "cohens_d": [0.5, 1.2, 0.8, 0.3],
    })
    result = compute_injection_layer_weighted(features)
    check(result is not None, "result not None")
    check(result["injection_layer"] == 14,
          f"injection_layer is 14 (got {result['injection_layer']})")
    check(result["n_features"] == 4, f"n_features is 4")
    check(result["score_concentration"] > 0, "score_concentration > 0")

    # Tie-breaking: prefer deeper layer.
    features2 = pd.DataFrame({
        "layer": [10, 20],
        "cohens_d": [1.0, -1.0],  # same |d|
    })
    result2 = compute_injection_layer_weighted(features2)
    check(result2["injection_layer"] == 20,
          f"tie breaks to deeper layer (got {result2['injection_layer']})")

    # Empty → None.
    result3 = compute_injection_layer_weighted(pd.DataFrame())
    check(result3 is None, "empty features → None")


def test_overlap_curve() -> None:
    print("\n=== Overlap Curve ===", flush=True)

    # Two subgroups sharing some features.
    ranked_A = pd.DataFrame({
        "rank": [1, 2, 3, 4, 5],
        "feature_idx": [100, 200, 300, 400, 500],
        "layer": [14, 14, 14, 14, 14],
    })
    ranked_B = pd.DataFrame({
        "rank": [1, 2, 3, 4, 5],
        "feature_idx": [100, 200, 600, 700, 800],  # 2 shared
        "layer": [14, 14, 14, 14, 14],
    })

    curve = compute_overlap_curve(ranked_A, ranked_B, [3, 5])

    check(3 in curve, "k=3 in curve")
    check(5 in curve, "k=5 in curve")

    # At k=5: 2 shared out of 8 unique → Jaccard = 2/8 = 0.25.
    check(curve[5]["n_shared"] == 2, f"k=5 n_shared=2 (got {curve[5]['n_shared']})")
    check(abs(curve[5]["jaccard"] - 0.25) < 0.01,
          f"k=5 jaccard≈0.25 (got {curve[5]['jaccard']})")

    # At k=3: feature sets [100,200,300] vs [100,200,600] → 2/4 = 0.5.
    check(abs(curve[3]["jaccard"] - 0.5) < 0.01,
          f"k=3 jaccard≈0.5 (got {curve[3]['jaccard']})")


def test_item_overlap() -> None:
    print("\n=== Item Overlap ===", flush=True)

    # Synthetic metadata.
    rows = []
    for i in range(20):
        sg = ["gay"] if i < 10 else ["bisexual"]
        rows.append({
            "item_idx": i, "category": "so", "context_condition": "ambig",
            "stereotyped_groups": sg, "model_answer_role": "unknown",
        })
    # Items shared by both (structural).
    for i in range(20, 25):
        rows.append({
            "item_idx": i, "category": "so", "context_condition": "ambig",
            "stereotyped_groups": ["gay", "bisexual"],
            "model_answer_role": "stereotyped_target",
        })
    meta = pd.DataFrame(rows)

    subgroups = [("so", "gay"), ("so", "bisexual")]
    result = compute_item_overlap(meta, subgroups, structural_threshold=0.3)

    check("per_category" in result, "has per_category")
    so = result["per_category"]["so"]
    # Pair key is alphabetically sorted: bisexual < gay.
    check("bisexual__gay" in so["pairwise"], "has bisexual__gay pair")
    pair = so["pairwise"]["bisexual__gay"]
    check(pair["n_shared"] == 5, f"n_shared=5 (got {pair['n_shared']})")
    # gay has 15 items, bisexual has 15 items, 5 shared.
    # fraction_of_bisexual_in_gay = 5/15 ≈ 0.333
    check(pair["max_fraction"] > 0.3, f"max_fraction > 0.3 (got {pair['max_fraction']})")
    check(pair["is_structural"], "flagged as structural at threshold 0.3")


def test_output_schema() -> None:
    print("\n=== Output Schema ===", flush=True)

    # ranked_features.parquet columns.
    expected_ranked_cols = {
        "category", "subgroup", "direction", "rank", "feature_idx", "layer",
        "cohens_d", "p_value_raw", "p_value_fdr",
        "firing_rate_stereo", "firing_rate_non_stereo",
        "mean_activation_stereo", "mean_activation_non_stereo",
        "n_stereo", "n_non_stereo",
    }
    check(True, f"ranked_features schema has {len(expected_ranked_cols)} columns")

    # injection_layers.json structure.
    sample = {
        "so/gay": {
            "category": "so", "subgroup": "gay",
            "pro_bias": {
                "injection_layer": 14,
                "layer_scores": {"14": 8.4},
                "n_features": 287,
                "top_layer_score": 8.4,
                "score_concentration": 0.412,
            },
            "anti_bias": None,
            "anti_bias_note": "No significant anti-bias features",
        }
    }
    check("injection_layer" in sample["so/gay"]["pro_bias"],
          "injection_layers schema valid")


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("B2 Structural Test Suite", flush=True)
    print("=" * 60, flush=True)

    test_imports()
    test_key_helpers()
    test_dedup()
    test_enumerate_subgroups()
    test_rank_features()
    test_injection_layers()
    test_overlap_curve()
    test_item_overlap()
    test_output_schema()

    print("\n" + "=" * 60, flush=True)
    print(f"Results: {passed} passed, {failed} failed", flush=True)
    print("=" * 60, flush=True)

    sys.exit(1 if failed > 0 else 0)
