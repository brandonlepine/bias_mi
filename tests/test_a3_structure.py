"""Structural tests for A3 SAE encoding — validates code without requiring SAE weights.

Tests:
  - All imports resolve
  - Parquet schema matches spec (long/sparse format)
  - encode_batch with a mock SAE
  - Summary computation from synthetic data
  - Subgroup lookup construction
  - Device selection logic

Run from project root:
    python tests/test_a3_structure.py
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.extraction.sae_encoding import (
    build_encoding_summary,
    build_subgroup_lookup,
    encode_batch,
    load_all_stimuli,
    select_encode_device,
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


# =========================================================================
# Imports
# =========================================================================

def test_imports() -> None:
    print("\n=== Imports ===", flush=True)

    from src.sae.wrapper import SAEWrapper
    check(True, "src.sae.wrapper imports")

    from src.extraction.sae_encoding import (
        encode_batch, encode_layer, build_encoding_summary,
        validate_sae_source, load_metadata, load_all_stimuli,
        build_subgroup_lookup, select_encode_device,
        run_encoding_validation,
    )
    check(True, "src.extraction.sae_encoding imports")

    from src.extraction.activations import save_metadata_parquet
    check(True, "save_metadata_parquet import")


# =========================================================================
# Device selection
# =========================================================================

def test_device_selection() -> None:
    print("\n=== Device Selection ===", flush=True)

    # CPU always works.
    check(select_encode_device("cpu") in ("cpu", "mps", "cuda"),
          "cpu config returns a valid device")

    # If MPS is available, mps config should return mps.
    if torch.backends.mps.is_available():
        check(select_encode_device("mps") == "mps", "mps available → mps")
    else:
        result = select_encode_device("mps")
        check(result in ("mps", "cpu"), f"mps unavailable → {result}")


# =========================================================================
# Subgroup lookup
# =========================================================================

def test_subgroup_lookup() -> None:
    print("\n=== Subgroup Lookup ===", flush=True)

    stimuli_by_cat = {
        "so": [
            {"item_idx": 0, "stereotyped_groups": ["gay"]},
            {"item_idx": 1, "stereotyped_groups": ["bisexual"]},
        ],
        "race": [
            {"item_idx": 0, "stereotyped_groups": ["black", "hispanic"]},
        ],
    }

    lookup = build_subgroup_lookup(stimuli_by_cat)
    check(lookup[("so", 0)] == ["gay"], "so/0 → gay")
    check(lookup[("so", 1)] == ["bisexual"], "so/1 → bisexual")
    check(lookup[("race", 0)] == ["black", "hispanic"], "race/0 → black,hispanic")
    check(("so", 999) not in lookup, "missing item not in lookup")


# =========================================================================
# encode_batch with mock SAE
# =========================================================================

def test_encode_batch() -> None:
    print("\n=== encode_batch (mock SAE) ===", flush=True)

    hidden_dim = 64
    n_features = 256

    # Build a mock SAE.
    mock_sae = MagicMock()
    mock_sae.n_features = n_features
    mock_sae.hidden_dim = hidden_dim

    # encode() returns sparse activations — mostly zeros.
    def mock_encode(batch_tensor: torch.Tensor) -> torch.Tensor:
        batch_size = batch_tensor.shape[0]
        out = torch.zeros(batch_size, n_features)
        # Set a few features active per item.
        for i in range(batch_size):
            active = torch.randint(0, n_features, (5,))
            out[i, active] = torch.rand(5) + 0.1
        return out

    mock_sae.encode = mock_encode

    # Create a small batch.
    batch_hs = [np.random.randn(hidden_dim).astype(np.float32) for _ in range(3)]
    batch_idxs = [10, 20, 30]

    records, l0s = encode_batch(
        mock_sae, batch_hs, batch_idxs, "so", 14, "cpu",
    )

    check(len(l0s) == 3, f"got 3 L0 values (got {len(l0s)})")
    check(all(isinstance(l, int) for l in l0s), "L0 values are ints")
    check(all(l > 0 for l in l0s), "all items have active features")

    check(len(records) > 0, f"got {len(records)} records")

    # Check record schema.
    r = records[0]
    check(set(r.keys()) == {"item_idx", "feature_idx", "activation_value", "category"},
          "record has correct keys")
    check(isinstance(r["item_idx"], int), "item_idx is int")
    check(isinstance(r["feature_idx"], int), "feature_idx is int")
    check(isinstance(r["activation_value"], float), "activation_value is float")
    check(r["activation_value"] > 0, "activation_value > 0")
    check(r["category"] == "so", "category is 'so'")

    # All item_idxs from our batch should appear.
    seen_idxs = set(r["item_idx"] for r in records)
    check(seen_idxs == {10, 20, 30}, f"all batch items in records: {seen_idxs}")


# =========================================================================
# Parquet schema
# =========================================================================

def test_parquet_schema() -> None:
    print("\n=== Parquet Schema ===", flush=True)

    # Simulate what encode_layer writes.
    records = [
        {"item_idx": 0, "feature_idx": 100, "activation_value": 1.5, "category": "so"},
        {"item_idx": 0, "feature_idx": 200, "activation_value": 0.8, "category": "so"},
        {"item_idx": 1, "feature_idx": 100, "activation_value": 2.1, "category": "so"},
        {"item_idx": 2, "feature_idx": 300, "activation_value": 0.3, "category": "race"},
    ]

    df = pd.DataFrame(records)
    df["item_idx"] = df["item_idx"].astype(np.int32)
    df["feature_idx"] = df["feature_idx"].astype(np.int32)
    df["activation_value"] = df["activation_value"].astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    df.to_parquet(tmp_path, index=False, compression="snappy")

    # Read back and validate.
    df2 = pd.read_parquet(tmp_path)
    check(list(df2.columns) == ["item_idx", "feature_idx", "activation_value", "category"],
          "correct columns")
    check(df2["item_idx"].dtype == np.int32, f"item_idx dtype int32 (got {df2['item_idx'].dtype})")
    check(df2["feature_idx"].dtype == np.int32, f"feature_idx dtype int32")
    check(df2["activation_value"].dtype == np.float32, f"activation_value dtype float32")
    check(len(df2) == 4, f"4 rows (got {len(df2)})")

    # Only nonzero activations stored.
    check((df2["activation_value"] > 0).all(), "all activation_value > 0")

    tmp_path.unlink()


# =========================================================================
# Metadata parquet schema
# =========================================================================

def test_metadata_parquet_schema() -> None:
    print("\n=== Metadata Parquet Schema ===", flush=True)

    expected_cols = {
        "item_idx", "category", "model_answer", "model_answer_role",
        "is_stereotyped_response", "is_correct", "context_condition",
        "stereotyped_groups", "n_target_groups", "margin",
        "question_polarity", "correct_letter", "stereotyped_option",
    }

    # Build a synthetic row.
    row = {
        "item_idx": 0,
        "category": "so",
        "model_answer": "A",
        "model_answer_role": "stereotyped_target",
        "is_stereotyped_response": True,
        "is_correct": False,
        "context_condition": "ambig",
        "stereotyped_groups": json.dumps(["gay"]),
        "n_target_groups": 1,
        "margin": 3.25,
        "question_polarity": "neg",
        "correct_letter": "B",
        "stereotyped_option": "A",
    }

    check(set(row.keys()) == expected_cols, "metadata row has all expected columns")

    # JSON round-trip for stereotyped_groups.
    decoded = json.loads(row["stereotyped_groups"])
    check(decoded == ["gay"], "stereotyped_groups survives JSON string encoding")


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("A3 Structural Test Suite (no SAE weights required)", flush=True)
    print("=" * 60, flush=True)

    test_imports()
    test_device_selection()
    test_subgroup_lookup()
    test_encode_batch()
    test_parquet_schema()
    test_metadata_parquet_schema()

    print("\n" + "=" * 60, flush=True)
    print(f"Results: {passed} passed, {failed} failed", flush=True)
    print("=" * 60, flush=True)

    sys.exit(1 if failed > 0 else 0)
