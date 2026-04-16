"""Structural tests for A2 extraction — validates code without requiring a model.

Tests:
  - All imports resolve
  - locate_hidden_tensor handles all documented output formats
  - extract_letter_logits handles tokenizer edge cases
  - Metadata schema matches spec
  - Summary computation from synthetic metadata
  - .npz file schema validation

Run from project root:
    python tests/test_a2_structure.py
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.extraction.activations import (
    build_category_extraction_summary,
    extract_letter_logits,
)
from src.models.wrapper import locate_hidden_tensor

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
# locate_hidden_tensor
# =========================================================================

def test_locate_hidden_tensor() -> None:
    print("\n=== locate_hidden_tensor ===", flush=True)
    hidden_dim = 128

    # Direct tensor output.
    t = torch.randn(1, 10, hidden_dim)
    result = locate_hidden_tensor(t, hidden_dim)
    check(result.shape == (1, 10, hidden_dim), "direct tensor: correct shape")

    # Tuple output, hidden state first.
    result2 = locate_hidden_tensor((t, None, None), hidden_dim)
    check(result2.shape == (1, 10, hidden_dim), "tuple (first element): correct shape")

    # Tuple output, hidden state not first.
    other = torch.randn(1, 10, 64)  # different dim
    result3 = locate_hidden_tensor((other, t, None), hidden_dim)
    check(result3.shape == (1, 10, hidden_dim), "tuple (second element): correct shape")

    # Wrong shape → error.
    wrong = torch.randn(1, 10, 64)
    try:
        locate_hidden_tensor(wrong, hidden_dim)
        check(False, "wrong shape tensor raises ValueError")
    except ValueError:
        check(True, "wrong shape tensor raises ValueError")

    # Tuple with no match → error.
    try:
        locate_hidden_tensor((wrong, None), hidden_dim)
        check(False, "tuple with no match raises ValueError")
    except ValueError:
        check(True, "tuple with no match raises ValueError")

    # Non-tensor input → error.
    try:
        locate_hidden_tensor("not a tensor", hidden_dim)
        check(False, "non-tensor raises ValueError")
    except ValueError:
        check(True, "non-tensor raises ValueError")


# =========================================================================
# extract_letter_logits
# =========================================================================

def test_extract_letter_logits() -> None:
    print("\n=== extract_letter_logits ===", flush=True)

    # Build a mock tokenizer.
    tokenizer = MagicMock()
    # "A" → token 32, " A" → token 330, etc.
    token_map = {
        "A": [32], " A": [330],
        "B": [33], " B": [365],
        "C": [34], " C": [356],
    }
    tokenizer.encode = lambda text, add_special_tokens=False: token_map.get(text, [999, 888])

    vocab_size = 1000
    logits = torch.zeros(vocab_size)
    logits[32] = 9.0    # bare A
    logits[330] = 12.0   # space A — should win
    logits[33] = 7.0     # bare B
    logits[365] = 8.5    # space B
    logits[34] = 6.0     # bare C
    logits[356] = 9.5    # space C

    result = extract_letter_logits(logits, tokenizer)

    check(set(result.keys()) == {"A", "B", "C"}, "has all three letters")
    check(result["A"] == 12.0, f"A takes max(9.0, 12.0) = 12.0 (got {result['A']})")
    check(result["B"] == 8.5, f"B takes max(7.0, 8.5) = 8.5 (got {result['B']})")
    check(result["C"] == 9.5, f"C takes max(6.0, 9.5) = 9.5 (got {result['C']})")

    # Model answer = A (highest logit).
    model_answer = max(result, key=result.get)
    check(model_answer == "A", f"model answer is A (got {model_answer})")

    # Margin = 12.0 - 9.5 = 2.5.
    margin = result[model_answer] - max(v for k, v in result.items() if k != model_answer)
    check(abs(margin - 2.5) < 1e-6, f"margin is 2.5 (got {margin})")

    # Case where a letter has no single-token encoding → -inf.
    bad_tokenizer = MagicMock()
    bad_tokenizer.encode = lambda text, add_special_tokens=False: [999, 888]  # always multi-token
    result2 = extract_letter_logits(logits, bad_tokenizer)
    check(result2["A"] == float("-inf"), "multi-token encoding → -inf")


# =========================================================================
# Metadata schema
# =========================================================================

def test_metadata_schema() -> None:
    print("\n=== Metadata Schema ===", flush=True)

    expected_keys = {
        "item_idx", "category", "model_answer", "model_answer_role",
        "is_stereotyped_response", "is_correct", "answer_logits", "margin",
        "stereotyped_groups", "n_target_groups", "stereotyped_option",
        "context_condition", "correct_letter", "question_polarity",
    }

    # Build a synthetic metadata dict matching what extract_single_item produces.
    meta = {
        "item_idx": 42,
        "category": "so",
        "model_answer": "A",
        "model_answer_role": "stereotyped_target",
        "is_stereotyped_response": True,
        "is_correct": False,
        "answer_logits": {"A": 12.31, "B": 8.72, "C": 9.06},
        "margin": 3.25,
        "stereotyped_groups": ["bisexual"],
        "n_target_groups": 1,
        "stereotyped_option": "A",
        "context_condition": "ambig",
        "correct_letter": "B",
        "question_polarity": "neg",
    }

    check(set(meta.keys()) == expected_keys,
          f"metadata has all expected keys")

    # Verify it serializes to JSON and back.
    json_str = json.dumps(meta, ensure_ascii=False)
    roundtrip = json.loads(json_str)
    check(roundtrip == meta, "metadata survives JSON round-trip")


# =========================================================================
# .npz file schema
# =========================================================================

def test_npz_schema() -> None:
    print("\n=== .npz File Schema ===", flush=True)

    n_layers = 32
    hidden_dim = 4096

    # Simulate what extract_single_item + save produces:
    # generate raw hidden states, compute norms, normalize, cast to float16.
    hs_raw_f32 = np.random.randn(n_layers, hidden_dim).astype(np.float32) * 50
    raw_norms = np.linalg.norm(hs_raw_f32, axis=1).astype(np.float32)
    safe_norms = np.maximum(raw_norms, 1e-8)[:, None]
    hs_normed = (hs_raw_f32 / safe_norms).astype(np.float16)
    meta = {
        "item_idx": 0, "category": "so", "model_answer": "A",
        "model_answer_role": "stereotyped_target",
        "is_stereotyped_response": True, "is_correct": False,
        "answer_logits": {"A": 12.0, "B": 8.0, "C": 9.0},
        "margin": 3.0, "stereotyped_groups": ["gay"],
        "n_target_groups": 1, "stereotyped_option": "A",
        "context_condition": "ambig", "correct_letter": "B",
        "question_polarity": "neg",
    }
    meta_json = json.dumps(meta)

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    np.savez(
        tmp_path,
        hidden_states=hs_normed,
        hidden_states_raw_norms=raw_norms,
        metadata_json=np.array(meta_json),
    )

    # Load and validate.
    data = np.load(tmp_path, allow_pickle=True)

    check("hidden_states" in data, "has hidden_states key")
    check("hidden_states_raw_norms" in data, "has hidden_states_raw_norms key")
    check("metadata_json" in data, "has metadata_json key")

    hs = data["hidden_states"]
    check(hs.dtype == np.float16, f"hidden_states dtype is float16 (got {hs.dtype})")
    check(hs.shape == (n_layers, hidden_dim),
          f"hidden_states shape is ({n_layers}, {hidden_dim}) (got {hs.shape})")

    norms = data["hidden_states_raw_norms"]
    check(norms.dtype == np.float32, f"raw_norms dtype is float32 (got {norms.dtype})")
    check(norms.shape == (n_layers,), f"raw_norms shape is ({n_layers},) (got {norms.shape})")

    # Metadata deserialization.
    raw_meta = data["metadata_json"]
    meta_str = raw_meta.item() if raw_meta.shape == () else str(raw_meta)
    loaded_meta = json.loads(meta_str)
    check(loaded_meta["item_idx"] == 0, "metadata item_idx correct")
    check(loaded_meta["model_answer"] == "A", "metadata model_answer correct")

    # Reconstruction of raw hidden states.
    # float16 introduces quantization error, so use a realistic tolerance.
    hs_f32 = hs.astype(np.float32)
    hs_raw_recon = hs_f32 * norms[:, None]
    reconstructed_norms = np.linalg.norm(hs_raw_recon, axis=1)
    check(
        np.allclose(reconstructed_norms, norms, rtol=5e-2),
        "raw hidden states reconstructable from normed + norms (rtol=5e-2)",
    )

    tmp_path.unlink()


# =========================================================================
# Summary computation
# =========================================================================

def test_summary_computation() -> None:
    print("\n=== Summary Computation ===", flush=True)

    # Build synthetic metadata for 10 items.
    metas = []
    for i in range(10):
        is_ambig = i < 6
        is_stereo = (i % 3 == 0) and is_ambig
        metas.append({
            "item_idx": i,
            "category": "so",
            "model_answer": "A" if is_stereo else "B",
            "model_answer_role": "stereotyped_target" if is_stereo else "unknown",
            "is_stereotyped_response": is_stereo,
            "is_correct": not is_stereo,
            "answer_logits": {"A": 10.0, "B": 8.0, "C": 6.0},
            "margin": 2.0,
            "stereotyped_groups": ["gay"] if i < 5 else ["bisexual"],
            "n_target_groups": 1,
            "stereotyped_option": "A",
            "context_condition": "ambig" if is_ambig else "disambig",
            "correct_letter": "B",
            "question_polarity": "neg" if i % 2 == 0 else "nonneg",
        })

    summary = build_category_extraction_summary(metas, "so")

    check(summary["category"] == "so", "summary category")
    check(summary["n_items"] == 10, f"n_items is 10 (got {summary['n_items']})")
    check(summary["n_ambig"] == 6, f"n_ambig is 6 (got {summary['n_ambig']})")
    check(summary["n_disambig"] == 4, f"n_disambig is 4 (got {summary['n_disambig']})")

    bs = summary["behavioral_summary"]
    check("n_stereotyped_response" in bs, "has n_stereotyped_response")
    check("stereotyped_rate_ambig" in bs, "has stereotyped_rate_ambig")
    check("accuracy_disambig" in bs, "has accuracy_disambig")

    check("per_subgroup_stereotyped_rate_ambig" in summary,
          "has per_subgroup rates")
    check("margin_stats" in summary, "has margin_stats")
    check("margin_by_response_type_ambig" in summary, "has margin by response")

    # Per-subgroup rates.
    rates = summary["per_subgroup_stereotyped_rate_ambig"]
    check("gay" in rates, "has gay subgroup rate")
    check("bisexual" in rates, "has bisexual subgroup rate")

    # Empty case.
    empty_summary = build_category_extraction_summary([], "empty")
    check(empty_summary["n_items"] == 0, "empty summary: n_items == 0")


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("A2 Structural Test Suite (no model required)", flush=True)
    print("=" * 60, flush=True)

    test_locate_hidden_tensor()
    test_extract_letter_logits()
    test_metadata_schema()
    test_npz_schema()
    test_summary_computation()

    print("\n" + "=" * 60, flush=True)
    print(f"Results: {passed} passed, {failed} failed", flush=True)
    print("=" * 60, flush=True)

    sys.exit(1 if failed > 0 else 0)
