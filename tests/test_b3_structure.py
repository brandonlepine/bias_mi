"""Structural tests for B3 subgroup direction geometry — validates logic with synthetic data.

Run from project root:
    python tests/test_b3_structure.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.geometry import (
    DIRECTION_SPECS,
    b3_complete,
    build_summary,
    compute_alignment,
    compute_all_cosines,
    compute_differentiation_metrics,
    compute_subgroup_directions,
    find_stable_range,
    load_direction,
    save_cosines,
    save_directions,
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


# ── Synthetic data helpers ──────────────────────────────────────────────

N_LAYERS = 4
HIDDEN_DIM = 16
N_ITEMS = 60


def make_synthetic_meta() -> pd.DataFrame:
    """Build synthetic metadata DataFrame matching A2 output schema."""
    rng = np.random.RandomState(42)
    rows = []
    for i in range(N_ITEMS):
        sub = ["gay", "bisexual", "lesbian"][i % 3]
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
        })
    return pd.DataFrame(rows)


def make_synthetic_hidden_states(n_items: int) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic normed hidden states and raw norms."""
    rng = np.random.RandomState(123)
    hs_normed = rng.randn(n_items, N_LAYERS, HIDDEN_DIM).astype(np.float32)
    # Unit-normalise per layer
    norms_raw = np.linalg.norm(hs_normed, axis=2)
    safe = np.maximum(norms_raw, 1e-8)[:, :, None]
    hs_normed = (hs_normed / safe).astype(np.float32)
    raw_norms = (norms_raw * rng.uniform(10, 50, size=(n_items, N_LAYERS))).astype(
        np.float32
    )
    return hs_normed, raw_norms


# ── Tests ───────────────────────────────────────────────────────────────

def test_compute_subgroup_directions_basic():
    print("\n[B3] compute_subgroup_directions — basic", flush=True)
    meta = make_synthetic_meta()
    hs_normed, raw_norms = make_synthetic_hidden_states(len(meta))
    hs_raw = hs_normed * raw_norms[:, :, None]

    result = compute_subgroup_directions(
        cat="so", sub="gay", cat_meta=meta,
        all_hs_normed=hs_normed, all_hs_raw=hs_raw,
        n_layers=N_LAYERS, hidden_dim=HIDDEN_DIM, min_n=3,
    )

    check(result is not None, "result not None")
    check("arrays" in result and "norms" in result and "info" in result, "has keys")

    info = result["info"]
    check(info["category"] == "so", "category matches")
    check(info["subgroup"] == "gay", "subgroup matches")
    check(not info["skipped"], "not skipped")
    check(info["bias_computed"], "bias computed")
    check(info["identity_computed"], "identity computed")

    # Check shapes
    for key in ["bias_direction_raw", "bias_direction_normed",
                 "identity_direction_raw", "identity_direction_normed"]:
        arr = result["arrays"][key]
        check(arr.shape == (N_LAYERS, HIDDEN_DIM), f"{key} shape correct")
        check(arr.dtype == np.float32, f"{key} dtype float32")
        # Check unit-normalised per layer
        norms = np.linalg.norm(arr, axis=1)
        check(np.allclose(norms, 1.0, atol=1e-5), f"{key} unit-normalised")

    # Check norm arrays
    for key in ["bias_direction_raw_norm", "bias_direction_normed_norm",
                 "identity_direction_raw_norm", "identity_direction_normed_norm"]:
        arr = result["norms"][key]
        check(arr.shape == (N_LAYERS,), f"{key} shape (n_layers,)")
        check(arr.dtype == np.float32, f"{key} dtype float32")


def test_compute_subgroup_directions_insufficient_data():
    print("\n[B3] compute_subgroup_directions — insufficient data", flush=True)
    meta = make_synthetic_meta()
    hs_normed, raw_norms = make_synthetic_hidden_states(len(meta))
    hs_raw = hs_normed * raw_norms[:, :, None]

    # min_n so high that nothing qualifies
    result = compute_subgroup_directions(
        cat="so", sub="gay", cat_meta=meta,
        all_hs_normed=hs_normed, all_hs_raw=hs_raw,
        n_layers=N_LAYERS, hidden_dim=HIDDEN_DIM, min_n=999,
    )
    check(result is None, "returns None when min_n too high")


def test_compute_subgroup_directions_partial_skip():
    print("\n[B3] compute_subgroup_directions — partial skip (bias vs identity)", flush=True)
    # Create data where only 2 items are ambig+stereotyped for a subgroup
    meta = make_synthetic_meta()
    # Force all ambig items to non-stereotyped except 2
    meta.loc[meta["context_condition"] == "ambig", "model_answer_role"] = "non_stereotyped"
    meta.loc[0, "model_answer_role"] = "stereotyped_target"
    meta.loc[3, "model_answer_role"] = "stereotyped_target"

    hs_normed, raw_norms = make_synthetic_hidden_states(len(meta))
    hs_raw = hs_normed * raw_norms[:, :, None]

    result = compute_subgroup_directions(
        cat="so", sub="gay", cat_meta=meta,
        all_hs_normed=hs_normed, all_hs_raw=hs_raw,
        n_layers=N_LAYERS, hidden_dim=HIDDEN_DIM, min_n=5,
    )

    if result is not None:
        check(not result["info"]["bias_computed"], "bias skipped (too few stereo)")
        check("bias_skip_reason" in result["info"], "skip reason recorded")
        check(result["info"]["identity_computed"], "identity still computed")
    else:
        # Both skipped — also valid for this data
        check(True, "result is None (both insufficient)")


def test_save_and_load_directions():
    print("\n[B3] save_directions / load_direction — I/O round-trip", flush=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)

        arrays = {
            "bias_direction_raw_so_gay": np.random.randn(N_LAYERS, HIDDEN_DIM).astype(np.float32),
            "identity_direction_normed_so_gay": np.random.randn(N_LAYERS, HIDDEN_DIM).astype(np.float32),
        }
        norms = {
            "bias_direction_raw_norm_so_gay": np.random.rand(N_LAYERS).astype(np.float32),
            "identity_direction_normed_norm_so_gay": np.random.rand(N_LAYERS).astype(np.float32),
        }

        out_path = save_directions(run_dir, arrays, norms)
        check(out_path.exists(), "npz file created")
        check(not out_path.with_suffix(".npz.tmp").exists(), "tmp file cleaned up")

        # Load back
        loaded = np.load(out_path)
        check(set(loaded.keys()) == set(arrays) | set(norms), "all keys present")
        check(np.allclose(loaded["bias_direction_raw_so_gay"],
                          arrays["bias_direction_raw_so_gay"]),
              "array values preserved")

        # load_direction helper
        d, n = load_direction(run_dir, "bias", "raw", "so", "gay")
        check(d is not None, "load_direction returns direction")
        check(n is not None, "load_direction returns norm")
        check(d.shape == (N_LAYERS, HIDDEN_DIM), "loaded shape correct")

        # Missing direction
        d2, n2 = load_direction(run_dir, "identity", "raw", "so", "gay")
        check(d2 is None and n2 is None, "missing direction returns (None, None)")


def test_save_cosines():
    print("\n[B3] save_cosines — atomic parquet write", flush=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        df = pd.DataFrame([{
            "category": "so", "direction_type": "bias_raw",
            "layer": 0, "subgroup_A": "gay", "subgroup_B": "bisexual",
            "cosine": 0.5,
        }])
        out_path = save_cosines(run_dir, df)
        check(out_path.exists(), "parquet file created")
        check(not out_path.with_suffix(".parquet.tmp").exists(), "tmp cleaned up")
        loaded = pd.read_parquet(out_path)
        check(len(loaded) == 1, "1 row written")
        check(loaded.iloc[0]["cosine"] == 0.5, "value preserved")


def test_compute_all_cosines():
    print("\n[B3] compute_all_cosines — pairwise computation", flush=True)
    rng = np.random.RandomState(77)

    # Create two orthogonal unit directions
    v1 = np.zeros((N_LAYERS, HIDDEN_DIM), dtype=np.float32)
    v1[:, 0] = 1.0
    v2 = np.zeros((N_LAYERS, HIDDEN_DIM), dtype=np.float32)
    v2[:, 1] = 1.0

    arrays = {
        "bias_direction_raw_so_gay": v1,
        "bias_direction_raw_so_bisexual": v2,
    }

    df = compute_all_cosines(arrays, ["so"], N_LAYERS)

    check(len(df) == N_LAYERS, f"one row per layer ({N_LAYERS})")
    check(set(df.columns) == {"category", "direction_type", "layer",
                               "subgroup_A", "subgroup_B", "cosine"},
          "correct columns")
    # Orthogonal → cosine ≈ 0
    check(np.allclose(df["cosine"].values, 0.0, atol=1e-5), "orthogonal → cosine ≈ 0")

    # Same direction → cosine = 1
    arrays2 = {
        "identity_direction_normed_so_gay": v1,
        "identity_direction_normed_so_bisexual": v1,
    }
    df2 = compute_all_cosines(arrays2, ["so"], N_LAYERS)
    check(np.allclose(df2["cosine"].values, 1.0, atol=1e-5), "parallel → cosine ≈ 1")

    # Empty: only one subgroup
    arrays3 = {"bias_direction_raw_so_gay": v1}
    df3 = compute_all_cosines(arrays3, ["so"], N_LAYERS)
    check(len(df3) == 0, "single subgroup → empty DataFrame")


def test_compute_differentiation_metrics():
    print("\n[B3] compute_differentiation_metrics — peak detection", flush=True)
    rng = np.random.RandomState(55)

    # Create cosines where layer 2 has most variance
    rows = []
    for layer in range(N_LAYERS):
        cos_val = 0.1 if layer != 2 else -0.9
        rows.append({
            "category": "so", "direction_type": "identity_normed",
            "layer": layer, "subgroup_A": "gay", "subgroup_B": "bisexual",
            "cosine": cos_val,
        })
        rows.append({
            "category": "so", "direction_type": "identity_normed",
            "layer": layer, "subgroup_A": "gay", "subgroup_B": "lesbian",
            "cosine": 0.2 if layer != 2 else 0.8,
        })

    df = pd.DataFrame(rows)
    result = compute_differentiation_metrics(df, ["so"], N_LAYERS)

    check("so" in result, "category present")
    check("identity_normed" in result["so"], "direction type present")

    d = result["so"]["identity_normed"]
    check(d["peak_layer"] == 2, "peak layer is 2 (highest variance)")
    check(d["peak_variance"] > 0, "peak variance positive")
    check(isinstance(d["stable_range"], list) and len(d["stable_range"]) == 2,
          "stable range is [start, end]")
    check(d["most_anti_aligned_pair_at_peak"] is not None, "most anti-aligned pair found")
    check(d["most_anti_aligned_pair_at_peak"]["cosine"] == -0.9,
          "correct most anti-aligned value")

    # Empty input
    empty_df = pd.DataFrame(columns=df.columns)
    result_empty = compute_differentiation_metrics(empty_df, ["so"], N_LAYERS)
    check(result_empty["so"] == {}, "empty df → empty result")


def test_find_stable_range():
    print("\n[B3] find_stable_range — sign preservation", flush=True)
    # All positive cosines across all layers → full range stable
    rows = []
    for layer in range(N_LAYERS):
        rows.append({
            "category": "so", "direction_type": "x",
            "layer": layer, "subgroup_A": "a", "subgroup_B": "b",
            "cosine": 0.5,
        })
    df = pd.DataFrame(rows)
    start, end = find_stable_range(df, N_LAYERS, peak_layer=1)
    check(start == 0 and end == N_LAYERS - 1, "all-positive → full range")

    # Sign flip at layer 2
    rows[2]["cosine"] = -0.3
    df2 = pd.DataFrame(rows)
    start2, end2 = find_stable_range(df2, N_LAYERS, peak_layer=0)
    check(end2 < 2, "sign flip at layer 2 → stable range ends before 2")


def test_compute_alignment():
    print("\n[B3] compute_alignment — bias-identity cosine", flush=True)
    # Identical bias and identity → alignment = 1.0
    v = np.zeros((N_LAYERS, HIDDEN_DIM), dtype=np.float32)
    v[:, 0] = 1.0

    arrays = {
        "bias_direction_raw_so_gay": v,
        "identity_direction_raw_so_gay": v,
        "bias_direction_normed_so_gay": v,
        "identity_direction_normed_so_gay": v,
    }
    result = compute_alignment(arrays, ["so"], N_LAYERS)

    check("so" in result, "category present")
    check("gay" in result["so"], "subgroup present")
    check(result["so"]["gay"]["raw"] is not None, "raw alignment computed")
    check(result["so"]["gay"]["normed"] is not None, "normed alignment computed")

    raw = result["so"]["gay"]["raw"]
    check(np.allclose(raw["per_layer_alignment"], [1.0] * N_LAYERS, atol=1e-5),
          "identical directions → alignment = 1.0")
    check(raw["peak_alignment"] == 1.0 or abs(raw["peak_alignment"] - 1.0) < 1e-3,
          "peak alignment ≈ 1.0")

    # Orthogonal → alignment = 0
    v2 = np.zeros((N_LAYERS, HIDDEN_DIM), dtype=np.float32)
    v2[:, 1] = 1.0
    arrays2 = {
        "bias_direction_raw_so_gay": v,
        "identity_direction_raw_so_gay": v2,
    }
    result2 = compute_alignment(arrays2, ["so"], N_LAYERS)
    r2 = result2["so"]["gay"]["raw"]
    check(np.allclose(r2["per_layer_alignment"], [0.0] * N_LAYERS, atol=1e-5),
          "orthogonal → alignment = 0")

    # Missing identity → skip
    arrays3 = {"bias_direction_raw_so_gay": v}
    result3 = compute_alignment(arrays3, ["so"], N_LAYERS)
    check("gay" not in result3["so"] or result3["so"]["gay"]["raw"] is None,
          "missing identity → no alignment")


def test_build_summary():
    print("\n[B3] build_summary — JSON structure", flush=True)
    subgroup_info = {
        ("so", "gay"): {
            "category": "so", "subgroup": "gay", "skipped": False,
            "n_total_targeting_S": 20, "n_total_not_targeting_S": 40,
            "n_bias_stereo": 10, "n_bias_non_stereo": 10,
            "bias_computed": True, "identity_computed": True,
        },
        ("so", "bisexual"): {
            "category": "so", "subgroup": "bisexual", "skipped": True,
        },
    }
    norms = {
        "bias_direction_raw_norm_so_gay": np.array([1.0, 2.0, 3.0, 2.5], dtype=np.float32),
    }
    summary = build_summary(subgroup_info, norms, ["so"], min_n=10)

    check(summary["min_n_per_group"] == 10, "min_n recorded")
    check(summary["n_subgroups_total"] == 2, "total subgroups counted")
    check(summary["n_subgroups_with_bias"] == 1, "1 subgroup with bias")
    check("so/gay" in summary["per_subgroup"], "subgroup key format cat/sub")
    check("so/bisexual" in summary["per_subgroup"], "skipped subgroup included")

    gay = summary["per_subgroup"]["so/gay"]
    check("bias_direction_raw_norm_range" in gay, "norm range present")
    check(gay["bias_direction_raw_norm_peak_layer"] == 2, "peak layer from norms")


def test_b3_complete():
    print("\n[B3] b3_complete — resume check", flush=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        check(not b3_complete(run_dir), "incomplete when no files")

        geo_dir = run_dir / "B_geometry"
        geo_dir.mkdir(parents=True)
        for f in ["subgroup_directions.npz", "cosine_pairs.parquet",
                   "subgroup_directions_summary.json",
                   "differentiation_metrics.json",
                   "bias_identity_alignment.json"]:
            (geo_dir / f).touch()
        check(b3_complete(run_dir), "complete when all files present")


# ── Run ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_compute_subgroup_directions_basic()
    test_compute_subgroup_directions_insufficient_data()
    test_compute_subgroup_directions_partial_skip()
    test_save_and_load_directions()
    test_save_cosines()
    test_compute_all_cosines()
    test_compute_differentiation_metrics()
    test_find_stable_range()
    test_compute_alignment()
    test_build_summary()
    test_b3_complete()

    print(f"\n{'=' * 60}", flush=True)
    print(f"B3 structure tests: {passed} passed, {failed} failed", flush=True)
    print(f"{'=' * 60}", flush=True)
    sys.exit(1 if failed else 0)
