"""A2 core: per-item activation extraction and behavioral recording.

Runs each BBQ item through the model, captures the last-token hidden state
at every transformer layer, and records the model's behavioral response
(answer choice, logits, margin).
"""

from __future__ import annotations

import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.models.wrapper import locate_hidden_tensor
from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_path: str,
    device: str,
) -> tuple[Any, Any, int, int, Any]:
    """Load the model and return ``(model, tokenizer, n_layers, hidden_dim, get_layer_fn)``.

    Tries ``ModelWrapper`` first for architecture abstraction, then falls back
    to direct HuggingFace loading with manual layer detection.
    """
    # Attempt 1: ModelWrapper
    try:
        from src.models.wrapper import ModelWrapper
        wrapper = ModelWrapper.from_pretrained(model_path, device=device)
        log(f"Loaded via ModelWrapper: {type(wrapper.model).__name__}")
        log(f"  Layers: {wrapper.n_layers}, Hidden dim: {wrapper.hidden_dim}")
        return (
            wrapper.model, wrapper.tokenizer,
            wrapper.n_layers, wrapper.hidden_dim,
            wrapper.get_layer,
        )
    except (ImportError, AttributeError) as e:
        log(f"ModelWrapper unavailable ({e}), falling back to direct loading")

    # Attempt 2: Direct HuggingFace loading
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float32 if device == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype,
    ).to(device)
    model.eval()

    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    log(f"Loaded: {type(model).__name__}, {n_layers} layers, dim={hidden_dim}")

    # Find decoder layers
    inner = None
    for attr_path in ["model.layers", "transformer.h", "gpt_neox.layers"]:
        obj = model
        for attr in attr_path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            inner = obj
            log(f"  Decoder layers at: {attr_path}")
            break

    if inner is None:
        raise RuntimeError(
            f"Cannot find decoder layers for {type(model).__name__}. "
            "Implement ModelWrapper for this architecture."
        )

    # Validate hook output on a test input
    _validate_hooks(model, tokenizer, inner, hidden_dim, device)

    def get_layer_fn(idx: int) -> Any:
        return inner[idx]

    return model, tokenizer, n_layers, hidden_dim, get_layer_fn


def _validate_hooks(
    model: Any,
    tokenizer: Any,
    layers: Any,
    hidden_dim: int,
    device: str,
) -> None:
    """Verify that a hook on the first layer captures the expected tensor shape."""
    test_input = tokenizer("test", return_tensors="pt").to(device)
    captured: dict[str, Any] = {}

    def hook_fn(module: Any, args: Any, output: Any) -> None:
        captured["output"] = output

    handle = layers[0].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(**test_input)
    handle.remove()

    h = locate_hidden_tensor(captured["output"], hidden_dim)
    if h.shape[-1] != hidden_dim:
        raise RuntimeError(
            f"Hook output dim {h.shape[-1]} != expected {hidden_dim}"
        )
    log(f"  Hook validation passed: output shape {tuple(h.shape)}")


# ---------------------------------------------------------------------------
# Letter logit extraction
# ---------------------------------------------------------------------------

def extract_letter_logits(
    logits: torch.Tensor,
    tokenizer: Any,
    letters: tuple[str, ...] = ("A", "B", "C"),
) -> dict[str, float]:
    """Extract logit values for answer letters from the vocabulary logits.

    Checks both bare (``"A"``) and space-prefixed (``" A"``) token variants.
    Takes the higher logit of the two for each letter.
    """
    letter_logits: dict[str, float] = {}

    for letter in letters:
        candidates: list[float] = []
        for prefix in ["", " "]:
            text = f"{prefix}{letter}"
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) == 1:
                candidates.append(float(logits[token_ids[0]].item()))

        if candidates:
            letter_logits[letter] = max(candidates)
        else:
            letter_logits[letter] = float("-inf")
            log(f"  WARNING: Could not resolve token ID for letter '{letter}'")

    return letter_logits


def log_token_diagnostic(
    logits: torch.Tensor,
    tokenizer: Any,
) -> None:
    """Log detailed token ID mapping for answer letters.  Run once for verification."""
    log("  === Token Diagnostic (first item) ===")
    for letter in ["A", "B", "C"]:
        for prefix in ["", " "]:
            text = f"{prefix}{letter}"
            ids = tokenizer.encode(text, add_special_tokens=False)
            decoded = tokenizer.decode(ids)
            if len(ids) == 1:
                logit_val = f"{float(logits[ids[0]].item()):.4f}"
            else:
                logit_val = f"multi-token (ids={ids})"
            log(f"    '{text}' → token_ids={ids}, decoded='{decoded}', "
                f"logit={logit_val}")
    log("  === End Token Diagnostic ===")


# ---------------------------------------------------------------------------
# Per-item extraction
# ---------------------------------------------------------------------------

def extract_single_item(
    item: dict[str, Any],
    model: Any,
    tokenizer: Any,
    get_layer_fn: Any,
    n_layers: int,
    hidden_dim: int,
    device: str,
) -> dict[str, Any]:
    """Run one BBQ item through the model.

    Returns dict with:
      hidden_states:           np.ndarray float16 (n_layers, hidden_dim), unit-normalized
      hidden_states_raw_norms: np.ndarray float32 (n_layers,)
      metadata:                dict with behavioral data
    """
    prompt = item["prompt"]

    # Step 1: Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    seq_len = inputs["input_ids"].shape[1]
    last_pos = seq_len - 1

    # Step 2: Register hooks on every layer
    hidden_states: dict[int, torch.Tensor] = {}
    hook_fired: dict[int, bool] = {}

    def make_hook(layer_idx: int) -> Any:
        def hook_fn(module: Any, args: Any, output: Any) -> None:
            hook_fired[layer_idx] = True
            h = locate_hidden_tensor(output, hidden_dim)
            hidden_states[layer_idx] = h[0, last_pos, :].detach().cpu().float()
        return hook_fn

    handles = []
    for i in range(n_layers):
        layer_module = get_layer_fn(i)
        handles.append(layer_module.register_forward_hook(make_hook(i)))

    # Step 3: Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Step 4: Remove hooks
    for h in handles:
        h.remove()

    # Validate all hooks fired
    missing = [i for i in range(n_layers) if i not in hook_fired]
    if missing:
        raise RuntimeError(f"Hooks did not fire for layers: {missing}")

    # Step 5: Extract answer logits
    logits = outputs.logits[0, last_pos, :]
    letter_logits = extract_letter_logits(logits, tokenizer)

    # Step 6: Determine model's answer
    model_answer = max(letter_logits, key=letter_logits.get)
    model_answer_role = item["answer_roles"][model_answer]
    is_stereotyped_response = (model_answer_role == "stereotyped_target")
    is_correct = (model_answer == item["correct_letter"])

    # Step 7: Compute margin
    top_logit = letter_logits[model_answer]
    other_logits = [v for k, v in letter_logits.items() if k != model_answer]
    margin = top_logit - max(other_logits) if other_logits else 0.0

    # Step 8: Stack and normalize hidden states
    hs_raw = torch.stack(
        [hidden_states[i] for i in range(n_layers)], dim=0,
    ).numpy()  # (n_layers, hidden_dim), float32

    raw_norms = np.linalg.norm(hs_raw, axis=1).astype(np.float32)  # (n_layers,)
    safe_norms = np.maximum(raw_norms, 1e-8)[:, None]               # (n_layers, 1)
    hs_normed = (hs_raw / safe_norms).astype(np.float16)            # (n_layers, hidden_dim)

    # Build metadata
    metadata: dict[str, Any] = {
        "item_idx": item["item_idx"],
        "category": item["category"],
        "model_answer": model_answer,
        "model_answer_role": model_answer_role,
        "is_stereotyped_response": is_stereotyped_response,
        "is_correct": is_correct,
        "answer_logits": letter_logits,
        "margin": float(margin),
        "stereotyped_groups": item["stereotyped_groups"],
        "n_target_groups": item["n_target_groups"],
        "stereotyped_option": item["stereotyped_option"],
        "context_condition": item["context_condition"],
        "correct_letter": item["correct_letter"],
        "question_polarity": item["question_polarity"],
    }

    return {
        "hidden_states": hs_normed,
        "hidden_states_raw_norms": raw_norms,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Category-level extraction
# ---------------------------------------------------------------------------

def extract_category(
    items: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    get_layer_fn: Any,
    n_layers: int,
    hidden_dim: int,
    device: str,
    output_dir: Path,
    category: str,
) -> dict[str, Any]:
    """Extract activations for all items in one category.

    Resume-safe: skips items with existing ``.npz`` files.
    Returns summary dict for this category.
    """
    n_total = len(items)
    n_extracted = 0
    n_skipped = 0
    all_metadata: list[dict[str, Any]] = []

    log(f"\n{'=' * 60}")
    log(f"Extracting: {category} ({n_total} items)")
    log(f"{'=' * 60}")

    token_diagnostic_done = False

    for item in items:
        item_idx = item["item_idx"]
        out_path = output_dir / f"item_{item_idx:06d}.npz"

        # Resume safety: skip if already extracted.
        if out_path.exists():
            try:
                existing = np.load(out_path, allow_pickle=True)
                raw = existing["metadata_json"]
                meta_str = raw.item() if raw.shape == () else str(raw)
                all_metadata.append(json.loads(meta_str))
            except Exception:
                pass  # corrupted — excluded from summary but don't crash
            n_skipped += 1
            continue

        # Extract
        result = extract_single_item(
            item, model, tokenizer, get_layer_fn,
            n_layers, hidden_dim, device,
        )

        # First-item validation and token diagnostic.
        if not token_diagnostic_done:
            hs = result["hidden_states"]
            norms_check = np.linalg.norm(hs.astype(np.float32), axis=1)
            log(f"  Validation: shape={hs.shape}, expected=({n_layers}, {hidden_dim})")
            log(f"  Norm range after normalization: "
                f"[{norms_check.min():.4f}, {norms_check.max():.4f}]")
            log(f"  Model answer: {result['metadata']['model_answer']} "
                f"(role={result['metadata']['model_answer_role']}, "
                f"correct={result['metadata']['is_correct']})")
            log(f"  Logits: {result['metadata']['answer_logits']}")
            log(f"  Margin: {result['metadata']['margin']:.4f}")

            # Token diagnostic — re-run forward pass for logging only.
            inputs = tokenizer(item["prompt"], return_tensors="pt").to(device)
            with torch.no_grad():
                diag_outputs = model(**inputs)
            last_logits = diag_outputs.logits[0, inputs["input_ids"].shape[1] - 1, :]
            log_token_diagnostic(last_logits, tokenizer)

            token_diagnostic_done = True

        # Save .npz atomically.
        metadata_json_str = json.dumps(
            result["metadata"], ensure_ascii=False,
        )
        tmp_path = out_path.with_suffix(".npz.tmp")
        np.savez(
            tmp_path,
            hidden_states=result["hidden_states"],
            hidden_states_raw_norms=result["hidden_states_raw_norms"],
            metadata_json=np.array(metadata_json_str),
        )
        tmp_path.rename(out_path)

        all_metadata.append(result["metadata"])
        n_extracted += 1

        # Progress logging.
        if n_extracted == 1 or n_extracted % 100 == 0:
            log(f"  [{n_extracted + n_skipped}/{n_total}] "
                f"extracted={n_extracted} skipped={n_skipped} "
                f"last_answer={result['metadata']['model_answer']}")

        # Memory management.
        if device == "mps" and n_extracted % 50 == 0:
            torch.mps.empty_cache()
        elif device.startswith("cuda") and n_extracted % 100 == 0:
            torch.cuda.empty_cache()

    log(f"  Complete: {n_extracted} extracted, {n_skipped} skipped")

    # GC between categories.
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()

    return build_category_extraction_summary(all_metadata, category)


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------

def build_category_extraction_summary(
    all_metadata: list[dict[str, Any]],
    category: str,
) -> dict[str, Any]:
    """Build behavioral summary statistics for a category from extracted metadata."""
    n = len(all_metadata)
    if n == 0:
        return {"category": category, "n_items": 0}

    n_stereo = sum(1 for m in all_metadata if m["is_stereotyped_response"])
    n_non_stereo = sum(
        1 for m in all_metadata if m["model_answer_role"] == "non_stereotyped"
    )
    n_unknown = sum(
        1 for m in all_metadata if m["model_answer_role"] == "unknown"
    )
    n_correct = sum(1 for m in all_metadata if m["is_correct"])

    # Ambig-specific: stereotyped response rate.
    ambig = [m for m in all_metadata if m["context_condition"] == "ambig"]
    n_ambig = len(ambig)
    n_stereo_ambig = sum(1 for m in ambig if m["is_stereotyped_response"])
    stereotyped_rate_ambig = n_stereo_ambig / max(n_ambig, 1)

    # Disambig-specific: accuracy.
    disambig = [m for m in all_metadata if m["context_condition"] == "disambig"]
    n_disambig = len(disambig)
    n_correct_disambig = sum(1 for m in disambig if m["is_correct"])
    accuracy_disambig = n_correct_disambig / max(n_disambig, 1)

    # Per-subgroup stereotyped rates on ambig items.
    subgroup_ambig_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n_total": 0, "n_stereo": 0},
    )
    for m in ambig:
        for sg in m["stereotyped_groups"]:
            subgroup_ambig_counts[sg]["n_total"] += 1
            if m["is_stereotyped_response"]:
                subgroup_ambig_counts[sg]["n_stereo"] += 1

    per_subgroup: dict[str, dict[str, Any]] = {}
    for sg, counts in sorted(subgroup_ambig_counts.items()):
        rate = counts["n_stereo"] / max(counts["n_total"], 1)
        per_subgroup[sg] = {
            "n_ambig_items": counts["n_total"],
            "n_stereotyped": counts["n_stereo"],
            "stereotyped_rate": round(rate, 4),
        }

    # Margin stats.
    margins = [m["margin"] for m in all_metadata]
    margin_stats = {
        "mean": round(float(np.mean(margins)), 4),
        "median": round(float(np.median(margins)), 4),
        "std": round(float(np.std(margins)), 4),
        "min": round(float(np.min(margins)), 4),
        "max": round(float(np.max(margins)), 4),
    }

    # Margin by response type (ambig only).
    stereo_margins = [m["margin"] for m in ambig if m["is_stereotyped_response"]]
    non_stereo_margins = [
        m["margin"] for m in ambig if not m["is_stereotyped_response"]
    ]
    margin_by_response = {
        "stereotyped": {
            "n": len(stereo_margins),
            "mean": round(float(np.mean(stereo_margins)), 4) if stereo_margins else None,
            "median": round(float(np.median(stereo_margins)), 4) if stereo_margins else None,
        },
        "non_stereotyped": {
            "n": len(non_stereo_margins),
            "mean": round(float(np.mean(non_stereo_margins)), 4) if non_stereo_margins else None,
            "median": round(float(np.median(non_stereo_margins)), 4) if non_stereo_margins else None,
        },
    }

    return {
        "category": category,
        "n_items": n,
        "n_ambig": n_ambig,
        "n_disambig": n_disambig,
        "behavioral_summary": {
            "n_stereotyped_response": n_stereo,
            "n_non_stereotyped_response": n_non_stereo,
            "n_unknown_selected": n_unknown,
            "n_correct": n_correct,
            "stereotyped_rate_ambig": round(stereotyped_rate_ambig, 4),
            "accuracy_disambig": round(accuracy_disambig, 4),
        },
        "per_subgroup_stereotyped_rate_ambig": per_subgroup,
        "margin_stats": margin_stats,
        "margin_by_response_type_ambig": margin_by_response,
    }


def build_and_save_extraction_summary(
    run_dir: Path,
    categories: list[str],
    config: dict[str, Any],
) -> None:
    """Build extraction_summary.json from per-category .npz outputs."""
    summary: dict[str, Any] = {
        "model_id": config.get("model_id", ""),
        "model_path": config.get("model_path", ""),
        "device": config.get("device", ""),
        "n_layers": config.get("n_layers"),
        "hidden_dim": config.get("hidden_dim"),
        "per_category": {},
        "total_items": 0,
    }

    for cat in categories:
        act_dir = run_dir / "A_extraction" / "activations" / cat
        if not act_dir.is_dir():
            continue

        npz_files = sorted(act_dir.glob("item_*.npz"))
        n_extracted = len(npz_files)

        all_meta: list[dict[str, Any]] = []
        for npz_path in npz_files:
            try:
                data = np.load(npz_path, allow_pickle=True)
                raw = data["metadata_json"]
                meta_str = raw.item() if raw.shape == () else str(raw)
                all_meta.append(json.loads(meta_str))
            except Exception:
                continue

        cat_summary = build_category_extraction_summary(all_meta, cat)
        cat_summary["n_extracted"] = n_extracted
        cat_summary["n_npz_files"] = n_extracted

        summary["per_category"][cat] = cat_summary
        summary["total_items"] += n_extracted

    summary_path = run_dir / "A_extraction" / "extraction_summary.json"
    atomic_save_json(summary, summary_path)

    log(f"\nExtraction summary -> {summary_path}")
    log(f"Total items: {summary['total_items']}")

    # Log per-subgroup stereotyped rates as a quick overview.
    log(f"\n{'=' * 60}")
    log("Per-subgroup stereotyped rates (ambig items):")
    log(f"{'=' * 60}")
    for cat, cs in sorted(summary["per_category"].items()):
        rates = cs.get("per_subgroup_stereotyped_rate_ambig", {})
        if rates:
            log(f"  {cat}:")
            for sg, info in sorted(
                rates.items(), key=lambda x: -x[1]["stereotyped_rate"],
            ):
                log(
                    f"    {sg:>25s}: {info['stereotyped_rate']:.3f} "
                    f"({info['n_stereotyped']}/{info['n_ambig_items']})"
                )
