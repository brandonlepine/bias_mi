"""C4: Token-level feature interpretability.

For each feature in the optimal steering vectors, determines which tokens
activate the feature, what output tokens the feature promotes/suppresses
(logit decomposition), and which BBQ templates the feature responds to.

Produces neuronpedia-style per-feature characterisations.

Usage:
    python scripts/C4_token_features.py --run_dir runs/llama-3.1-8b_2026-04-15/
    python scripts/C4_token_features.py --run_dir ... --categories so,race
    python scripts/C4_token_features.py --run_dir ... --max_features_per_subgroup 10
    python scripts/C4_token_features.py --run_dir ... --save_full_per_token
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.models.wrapper import ModelWrapper, locate_hidden_tensor
from src.sae.wrapper import SAEWrapper
from src.utils.config import load_config
from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log, progress_bar


# Re-use demographic patterns from C3 for identity-term annotation
try:
    from scripts.C3_generalization import DEMOGRAPHIC_PATTERNS
except ImportError:
    DEMOGRAPHIC_PATTERNS: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MAX_FEATURES_PER_SUBGROUP = 10
STRING_TEMPLATE_THRESHOLD = 0.90
POSITION_TEMPLATE_THRESHOLD = 0.80
TOP_N_LOGIT_TOKENS = 10
TOP_N_ACTIVATING_EXAMPLES = 20


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="C4: Token-level feature interpretability")
    p.add_argument("--run_dir", required=True, type=str)
    p.add_argument("--categories", type=str, default=None,
                   help="Comma-separated category filter")
    p.add_argument("--max_features_per_subgroup", type=int,
                   default=DEFAULT_MAX_FEATURES_PER_SUBGROUP)
    p.add_argument("--save_full_per_token", action="store_true",
                   help="Save full per-item per-token activation parquet (LARGE)")
    p.add_argument("--skip_figures", action="store_true")
    p.add_argument("--max_items", type=int, default=None,
                   help="Max ambig items per category (for quick tests)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Step 0: Collect features to analyse
# ---------------------------------------------------------------------------

def collect_needed_features(
    viable_manifests: list[dict[str, Any]],
    vectors_dir: Path,
    max_features_per_subgroup: int,
    category_filter: set[str] | None = None,
) -> tuple[dict[int, list[int]], list[dict[str, Any]]]:
    """From each viable subgroup's npz, extract (feature_idx, layer) pairs.

    Returns
    -------
    needed_by_layer : dict
        ``{layer: sorted list of unique feature_idxs}``.
    feature_manifest : list[dict]
        Per-entry: category, subgroup, rank_in_vector, feature_idx, layer.
    """
    needed_by_layer: dict[int, set[int]] = defaultdict(set)
    feature_manifest: list[dict[str, Any]] = []

    for m in viable_manifests:
        cat = m["category"]
        if category_filter and cat not in category_filter:
            continue
        sub = m["subgroup"]
        vec_path = vectors_dir / f"{cat}_{sub}.npz"
        if not vec_path.exists():
            continue
        data = np.load(vec_path)

        feature_idxs = data["feature_idxs"]
        feature_layers = data["feature_layers"]
        n_features = min(len(feature_idxs), max_features_per_subgroup)

        for rank in range(n_features):
            fidx = int(feature_idxs[rank])
            flayer = int(feature_layers[rank])
            needed_by_layer[flayer].add(fidx)
            feature_manifest.append({
                "category": cat,
                "subgroup": sub,
                "rank_in_vector": rank,
                "feature_idx": fidx,
                "layer": flayer,
            })

    needed_sorted = {
        layer: sorted(feats) for layer, feats in needed_by_layer.items()
    }

    log(f"Collected {len(feature_manifest)} (subgroup, feature) pairs")
    log(f"Unique (layer, feature_idx) pairs: "
        f"{sum(len(f) for f in needed_sorted.values())}")
    log(f"Layers involved: {sorted(needed_sorted.keys())}")

    return needed_sorted, feature_manifest


# ---------------------------------------------------------------------------
# Step 1: Logit effect decomposition
# ---------------------------------------------------------------------------

def compute_logit_effects(
    wrapper: ModelWrapper,
    sae_cache: dict[int, SAEWrapper],
    feature_manifest: list[dict[str, Any]],
    top_n_tokens: int = TOP_N_LOGIT_TOKENS,
) -> pd.DataFrame:
    """For each unique (layer, feature_idx), compute top promoted/suppressed
    output tokens via unembedding decomposition."""
    tokenizer = wrapper.tokenizer
    W_U = wrapper.get_unembedding_matrix()  # (hidden_dim, vocab_size)
    W_U = W_U.float().cpu()

    rows: list[dict[str, Any]] = []
    processed: set[tuple[int, int]] = set()

    for entry in feature_manifest:
        fidx = entry["feature_idx"]
        flayer = entry["layer"]
        pair = (flayer, fidx)
        if pair in processed:
            continue
        processed.add(pair)

        sae = sae_cache[flayer]
        decoder_col = torch.from_numpy(
            sae.get_feature_decoder_column(fidx),
        ).float()

        # logit_contribution[v] = W_U[:, v] · decoder_col
        logit_contribution = W_U.T @ decoder_col  # (vocab_size,)

        top_pos = torch.topk(logit_contribution, k=top_n_tokens)
        top_neg = torch.topk(-logit_contribution, k=top_n_tokens)

        for rank in range(top_n_tokens):
            pos_tid = int(top_pos.indices[rank].item())
            neg_tid = int(top_neg.indices[rank].item())
            rows.append({
                "layer": flayer, "feature_idx": fidx, "direction": "positive",
                "rank": rank + 1, "token_id": pos_tid,
                "token_str": tokenizer.decode([pos_tid]),
                "logit_contribution": float(top_pos.values[rank].item()),
            })
            rows.append({
                "layer": flayer, "feature_idx": fidx, "direction": "negative",
                "rank": rank + 1, "token_id": neg_tid,
                "token_str": tokenizer.decode([neg_tid]),
                "logit_contribution": float(-top_neg.values[rank].item()),
            })

    log(f"Computed logit effects for {len(processed)} unique features")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 2: Item-centric token activation extraction
# ---------------------------------------------------------------------------

def make_all_token_hook(
    layer_idx: int, hidden_dim: int, storage: dict[int, torch.Tensor],
) -> Any:
    """Hook that stores full per-token hidden states at this layer."""
    def hook_fn(module: Any, args: Any, output: Any) -> None:
        h = locate_hidden_tensor(output, hidden_dim)
        storage[layer_idx] = h[0].detach().cpu().float()
    return hook_fn


def extract_token_activations(
    category: str,
    needed_by_layer: dict[int, list[int]],
    feature_manifest: list[dict[str, Any]],
    wrapper: ModelWrapper,
    sae_cache: dict[int, SAEWrapper],
    metadata_df: pd.DataFrame,
    stimuli: list[dict[str, Any]],
    max_items: int | None = None,
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    """Process all ambig items in a category with item-centric multi-layer hooks.

    Returns accumulator: ``{(layer, feature_idx): [per-item records]}``.
    """
    # Determine layers needed for this category
    cat_features = [e for e in feature_manifest if e["category"] == category]
    cat_layers = sorted(set(e["layer"] for e in cat_features))
    if not cat_layers:
        return {}

    log(f"  {category}: {len(cat_layers)} layers to hook, "
        f"{len(cat_features)} (subgroup, feature) entries")

    # Get ambig items
    cat_meta = metadata_df[
        (metadata_df["category"] == category)
        & (metadata_df["context_condition"] == "ambig")
    ]
    stimuli_by_idx = {s["item_idx"]: s for s in stimuli}

    if max_items:
        cat_meta = cat_meta.head(max_items)

    tokenizer = wrapper.tokenizer
    hidden_dim = wrapper.hidden_dim
    device = wrapper.device

    accumulator: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)

    for i, (_, meta_row) in enumerate(
        progress_bar(cat_meta.iterrows(), total=len(cat_meta),
                     desc=f"  {category}")
    ):
        item_idx = int(meta_row["item_idx"])
        stim = stimuli_by_idx.get(item_idx)
        if stim is None:
            continue

        prompt = stim["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"][0].cpu().tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        # Install hooks on all needed layers
        storage: dict[int, torch.Tensor] = {}
        hooks = []
        for layer in cat_layers:
            layer_module = wrapper.get_layer(layer)
            h = layer_module.register_forward_hook(
                make_all_token_hook(layer, hidden_dim, storage),
            )
            hooks.append(h)

        try:
            with torch.no_grad():
                wrapper.model(**inputs)
        finally:
            for h in hooks:
                h.remove()

        # Extract target features from each layer
        for layer in cat_layers:
            if layer not in storage:
                continue
            hidden_all_tokens = storage[layer]  # (seq_len, hidden_dim)

            target_feat_ids = needed_by_layer.get(layer, [])
            if not target_feat_ids:
                continue

            sae = sae_cache[layer]
            with torch.no_grad():
                feat_acts_full = sae.encode(
                    hidden_all_tokens.to(sae.device),
                )  # (seq_len, n_features_in_sae)

            target_ids_tensor = torch.tensor(
                target_feat_ids, device=feat_acts_full.device,
            )
            target_acts = (
                feat_acts_full[:, target_ids_tensor].cpu().float().numpy()
            )  # (seq_len, n_target_features)

            del feat_acts_full
            if device == "mps":
                torch.mps.empty_cache()
            elif str(device).startswith("cuda"):
                torch.cuda.empty_cache()

            # Parse metadata
            stereo_groups = meta_row.get("stereotyped_groups", "[]")
            if isinstance(stereo_groups, str):
                stereo_groups = json.loads(stereo_groups)

            for feat_i, feature_idx in enumerate(target_feat_ids):
                acts_1d = target_acts[:, feat_i]
                argmax_pos = int(np.argmax(acts_1d))
                accumulator[(layer, feature_idx)].append({
                    "item_idx": item_idx,
                    "question_index": int(meta_row.get("question_index", 0)),
                    "question_polarity": str(
                        meta_row.get("question_polarity", ""),
                    ),
                    "stereotyped_groups": stereo_groups,
                    "model_answer_role": str(
                        meta_row.get("model_answer_role", ""),
                    ),
                    "is_stereotyped_response": bool(
                        meta_row.get("is_stereotyped_response", False),
                    ),
                    "stereotyped_option": stim.get("stereotyped_option"),
                    "tokens": tokens,
                    "activations": acts_1d.tolist(),
                    "max_activation": float(np.max(acts_1d)),
                    "argmax_position": argmax_pos,
                    "argmax_token": tokens[argmax_pos] if argmax_pos < len(tokens) else "",
                })

        del storage

    return dict(accumulator)


# ---------------------------------------------------------------------------
# Step 3: Template filtering
# ---------------------------------------------------------------------------

def compute_string_template_tokens(
    stimuli: list[dict[str, Any]],
    tokenizer: Any,
    threshold: float = STRING_TEMPLATE_THRESHOLD,
) -> set[str]:
    """Tokens appearing in >= threshold fraction of items."""
    n_items = len(stimuli)
    if n_items == 0:
        return set()
    token_presence: Counter = Counter()
    for item in stimuli:
        toks = set(tokenizer.convert_ids_to_tokens(
            tokenizer.encode(item["prompt"]),
        ))
        for tok in toks:
            token_presence[tok] += 1
    return {tok for tok, cnt in token_presence.items()
            if cnt / n_items >= threshold}


def compute_position_template_positions(
    stimuli: list[dict[str, Any]],
    metadata_df: pd.DataFrame,
    tokenizer: Any,
    threshold: float = POSITION_TEMPLATE_THRESHOLD,
) -> dict[int, set[int]]:
    """Per question_index, positions where the same token appears in >threshold
    fraction of items in that template."""
    meta_by_idx = {}
    for _, row in metadata_df.iterrows():
        meta_by_idx[int(row["item_idx"])] = row

    items_by_qidx: dict[int, list[dict]] = defaultdict(list)
    for item in stimuli:
        meta = meta_by_idx.get(item["item_idx"])
        if meta is not None:
            qidx = int(meta.get("question_index", 0))
            items_by_qidx[qidx].append(item)

    template_positions: dict[int, set[int]] = {}
    for qidx, qidx_items in items_by_qidx.items():
        if len(qidx_items) < 5:
            continue
        pos_tok_counts: dict[int, Counter] = defaultdict(Counter)
        for item in qidx_items:
            toks = tokenizer.convert_ids_to_tokens(
                tokenizer.encode(item["prompt"]),
            )
            for pos, tok in enumerate(toks):
                pos_tok_counts[pos][tok] += 1

        invariant = set()
        for pos, counter in pos_tok_counts.items():
            most_common, count = counter.most_common(1)[0]
            if count / len(qidx_items) >= threshold:
                invariant.add(pos)
        template_positions[qidx] = invariant

    return template_positions


# ---------------------------------------------------------------------------
# Step 4: Per-feature aggregation
# ---------------------------------------------------------------------------

def is_identity_match(token: str) -> bool:
    """Check if a token matches any demographic pattern."""
    tok_clean = token.lower().strip().replace("\u2581", "").replace("\u0120", "")
    if len(tok_clean) < 2:
        return False
    for spec in DEMOGRAPHIC_PATTERNS.values():
        for pat_str in spec.get("patterns", []):
            try:
                if re.search(pat_str, tok_clean, re.IGNORECASE):
                    return True
            except re.error:
                continue
    return False


def aggregate_token_rankings(
    feature_records: list[dict[str, Any]],
    string_template_tokens: set[str],
) -> pd.DataFrame:
    """Aggregate per-token statistics across all items for one feature."""
    token_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "activations": [],
        "nonzero_activations": [],
    })

    for rec in feature_records:
        for tok, act in zip(rec["tokens"], rec["activations"]):
            token_stats[tok]["activations"].append(act)
            if act > 0:
                token_stats[tok]["nonzero_activations"].append(act)

    rows: list[dict[str, Any]] = []
    for tok, stats in token_stats.items():
        acts = stats["activations"]
        nonzero = stats["nonzero_activations"]
        rows.append({
            "token": tok,
            "n_occurrences": len(acts),
            "n_nonzero": len(nonzero),
            "mean_activation": float(np.mean(acts)) if acts else 0.0,
            "mean_activation_nonzero": (
                float(np.mean(nonzero)) if nonzero else 0.0
            ),
            "max_activation": float(max(acts)) if acts else 0.0,
            "median_activation_nonzero": (
                float(np.median(nonzero)) if nonzero else 0.0
            ),
            "fraction_firings": len(nonzero) / max(len(acts), 1),
            "is_template_string": tok in string_template_tokens,
            "is_identity_term": is_identity_match(tok),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            "mean_activation_nonzero", ascending=False,
        ).reset_index(drop=True)
        df["rank"] = df.index + 1
    return df


def aggregate_per_template(
    feature_records: list[dict[str, Any]],
) -> pd.DataFrame:
    """Aggregate by question_index for one feature."""
    by_qidx: dict[int, list[dict]] = defaultdict(list)
    for rec in feature_records:
        by_qidx[rec["question_index"]].append(rec)

    rows: list[dict[str, Any]] = []
    for qidx, recs in by_qidx.items():
        max_acts = [rec["max_activation"] for rec in recs]
        n_fired = sum(1 for ma in max_acts if ma > 0)
        rows.append({
            "question_index": qidx,
            "n_items": len(recs),
            "n_items_with_firing": n_fired,
            "fraction_items_with_firing": n_fired / max(len(recs), 1),
            "mean_max_activation": float(np.mean(max_acts)),
            "max_across_items": float(max(max_acts)) if max_acts else 0.0,
            "median_max_activation": float(np.median(max_acts)),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            "mean_max_activation", ascending=False,
        ).reset_index(drop=True)
        df["rank"] = df.index + 1
    return df


def compute_activation_density(
    feature_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build distribution of nonzero activations for one feature."""
    all_acts: list[float] = []
    for rec in feature_records:
        all_acts.extend(rec["activations"])

    arr = np.array(all_acts) if all_acts else np.array([0.0])
    nonzero = arr[arr > 0]

    total_positions = len(arr)
    density = len(nonzero) / max(total_positions, 1)

    if len(nonzero) > 0:
        bin_edges = np.linspace(0, float(nonzero.max()), 51)
        counts, _ = np.histogram(nonzero, bins=bin_edges)
    else:
        bin_edges = np.linspace(0, 1, 51)
        counts = np.zeros(50, dtype=int)

    return {
        "density": round(density, 6),
        "n_total_positions": total_positions,
        "n_nonzero_positions": int(len(nonzero)),
        "max_activation": float(arr.max()),
        "mean_activation_nonzero": (
            float(nonzero.mean()) if len(nonzero) else 0.0
        ),
        "median_activation_nonzero": (
            float(np.median(nonzero)) if len(nonzero) else 0.0
        ),
        "histogram_bin_edges": bin_edges.tolist(),
        "histogram_counts": counts.tolist(),
    }


def select_top_activating_examples(
    feature_records: list[dict[str, Any]],
    stimuli_by_idx: dict[int, dict[str, Any]],
    top_n: int = TOP_N_ACTIVATING_EXAMPLES,
) -> list[dict[str, Any]]:
    """Top-N items by max_activation with full token activation pattern."""
    sorted_recs = sorted(
        feature_records, key=lambda r: -r["max_activation"],
    )[:top_n]

    results: list[dict[str, Any]] = []
    for rec in sorted_recs:
        stim = stimuli_by_idx.get(rec["item_idx"], {})
        results.append({
            "item_idx": rec["item_idx"],
            "max_activation": rec["max_activation"],
            "argmax_position": rec["argmax_position"],
            "argmax_token": rec["argmax_token"],
            "tokens": rec["tokens"],
            "activations": rec["activations"],
            "prompt_preview": stim.get("prompt", "")[:200],
            "stereotyped_groups": rec["stereotyped_groups"],
            "question_polarity": rec["question_polarity"],
            "is_stereotyped_response": rec["is_stereotyped_response"],
        })
    return results


# ---------------------------------------------------------------------------
# Feature key helper
# ---------------------------------------------------------------------------

def _fkey(layer: int, feature_idx: int) -> str:
    return f"L{layer:02d}_F{feature_idx}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    t0 = time.time()

    run_dir = Path(args.run_dir)
    config = load_config(run_dir)
    device = torch.device(config["device"])
    dtype = getattr(torch, config["dtype"])

    log("C4 Token-Level Feature Interpretability")
    log(f"  run_dir: {run_dir}")
    log(f"  device: {device}")

    # Load model
    log("\nLoading model...")
    wrapper = ModelWrapper.from_pretrained(config["model_path"], device=str(device))
    tokenizer = wrapper.tokenizer

    # Load viable manifests
    with open(run_dir / "C_steering" / "steering_manifests.json") as f:
        manifests = json.load(f)
    viable = [m for m in manifests if m.get("steering_viable")]
    log(f"Viable subgroups: {len(viable)}")

    # Category filter
    cat_filter = set(args.categories.split(",")) if args.categories else None

    # Step 0: Collect features
    vectors_dir = run_dir / "C_steering" / "vectors"
    needed_by_layer, feature_manifest = collect_needed_features(
        viable, vectors_dir, args.max_features_per_subgroup, cat_filter,
    )

    if not feature_manifest:
        log("No features to analyse; exiting")
        return

    # Load SAEs for all needed layers
    sae_cache: dict[int, SAEWrapper] = {}
    for layer in sorted(needed_by_layer.keys()):
        log(f"  Loading SAE for layer {layer}...")
        sae_cache[layer] = SAEWrapper(
            config["sae_source"],
            layer=layer,
            expansion=config.get("sae_expansion", 32),
            device=str(device),
        )

    # Output directories
    output_dir = run_dir / "C_token_features"
    ensure_dir(output_dir / "token_rankings")
    ensure_dir(output_dir / "per_template_rankings")
    ensure_dir(output_dir / "top_activating_examples")
    ensure_dir(output_dir / "figures")

    # Save feature manifest
    atomic_save_json(feature_manifest, output_dir / "feature_manifest.json")

    # Step 1: Logit effect decomposition
    log("\nStep 1: Logit effect decomposition...")
    logit_effects_df = compute_logit_effects(
        wrapper, sae_cache, feature_manifest,
    )
    logit_effects_df.to_parquet(
        output_dir / "logit_effects.parquet",
        index=False, compression="snappy",
    )

    # Load metadata
    metadata_df = pd.read_parquet(run_dir / "A_extraction" / "metadata.parquet")

    # Determine categories to process
    categories = sorted(set(e["category"] for e in feature_manifest))
    if cat_filter:
        categories = [c for c in categories if c in cat_filter]

    # Step 2: Extract token activations per category
    log("\nStep 2: Token activation extraction...")
    all_accumulator: dict[tuple[int, int], list[dict[str, Any]]] = {}
    template_filters_data: dict[str, Any] = {
        "string_level_threshold": STRING_TEMPLATE_THRESHOLD,
        "position_level_threshold": POSITION_TEMPLATE_THRESHOLD,
        "per_category": {},
    }
    all_string_templates: dict[str, set[str]] = {}
    all_position_templates: dict[str, dict[int, set[int]]] = {}
    all_stimuli_by_cat: dict[str, dict[int, dict[str, Any]]] = {}

    for category in categories:
        log(f"\n  Category: {category}")
        stimuli_path = run_dir / "A_extraction" / "stimuli" / f"{category}.json"
        with open(stimuli_path) as f:
            stimuli = json.load(f)

        stimuli_by_idx = {s["item_idx"]: s for s in stimuli}
        all_stimuli_by_cat[category] = stimuli_by_idx

        # Template filtering
        string_templates = compute_string_template_tokens(
            stimuli, tokenizer,
        )
        position_templates = compute_position_template_positions(
            stimuli, metadata_df, tokenizer,
        )
        all_string_templates[category] = string_templates
        all_position_templates[category] = position_templates

        template_filters_data["per_category"][category] = {
            "string_level_template_tokens": sorted(string_templates),
            "n_template_tokens": len(string_templates),
        }

        # Extract activations
        cat_accum = extract_token_activations(
            category, needed_by_layer, feature_manifest,
            wrapper, sae_cache, metadata_df, stimuli,
            max_items=args.max_items,
        )
        all_accumulator.update(cat_accum)

    # Save template filters
    atomic_save_json(template_filters_data, output_dir / "template_filters.json")

    # Step 3-4: Aggregate per feature
    log("\nStep 3-4: Per-feature aggregation...")
    activation_densities: dict[str, Any] = {}
    interpretability_rows: list[dict[str, Any]] = []
    full_per_token_rows: list[dict[str, Any]] = []

    # Build reverse lookup: which categories/subgroups use each (layer, feat)?
    feat_to_subs: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for entry in feature_manifest:
        feat_to_subs[(entry["layer"], entry["feature_idx"])].append(entry)

    for (layer, feat_idx), records in progress_bar(
        all_accumulator.items(), desc="  Aggregating features",
    ):
        fk = _fkey(layer, feat_idx)
        entries = feat_to_subs.get((layer, feat_idx), [])
        cats_for_feat = sorted(set(e["category"] for e in entries))
        subs_for_feat = sorted(set(
            f"{e['category']}/{e['subgroup']}" for e in entries
        ))

        # Pick the first category's template tokens for filtering
        cat_for_filter = cats_for_feat[0] if cats_for_feat else ""
        string_tpl = all_string_templates.get(cat_for_filter, set())

        # Token rankings
        token_df = aggregate_token_rankings(records, string_tpl)
        if not token_df.empty:
            token_df.to_parquet(
                output_dir / "token_rankings" / f"{fk}.parquet",
                index=False, compression="snappy",
            )

        # Per-template rankings
        template_df = aggregate_per_template(records)
        if not template_df.empty:
            template_df.to_parquet(
                output_dir / "per_template_rankings" / f"{fk}.parquet",
                index=False, compression="snappy",
            )

        # Activation density
        density = compute_activation_density(records)
        activation_densities[fk] = density

        # Top activating examples
        all_stim = {}
        for cat in cats_for_feat:
            all_stim.update(all_stimuli_by_cat.get(cat, {}))
        top_examples = select_top_activating_examples(records, all_stim)
        atomic_save_json(
            top_examples,
            output_dir / "top_activating_examples" / f"{fk}.json",
        )

        # Logit effect summary for this feature
        feat_logit = logit_effects_df[
            (logit_effects_df["layer"] == layer)
            & (logit_effects_df["feature_idx"] == feat_idx)
        ]
        top_pos_token = ""
        top_neg_token = ""
        if not feat_logit.empty:
            pos_rows = feat_logit[
                (feat_logit["direction"] == "positive") & (feat_logit["rank"] == 1)
            ]
            neg_rows = feat_logit[
                (feat_logit["direction"] == "negative") & (feat_logit["rank"] == 1)
            ]
            if not pos_rows.empty:
                top_pos_token = str(pos_rows["token_str"].iloc[0])
            if not neg_rows.empty:
                top_neg_token = str(neg_rows["token_str"].iloc[0])

        # Top activating token (unfiltered and filtered)
        top_tok_unfiltered = ""
        top_tok_filtered = ""
        if not token_df.empty:
            top_tok_unfiltered = str(token_df.iloc[0]["token"])
            filtered = token_df[~token_df["is_template_string"]]
            if not filtered.empty:
                top_tok_filtered = str(filtered.iloc[0]["token"])

        interpretability_rows.append({
            "layer": layer,
            "feature_idx": feat_idx,
            "categories": ",".join(cats_for_feat),
            "subgroups": ",".join(subs_for_feat),
            "n_items_processed": len(records),
            "density": density["density"],
            "mean_activation_nonzero": density["mean_activation_nonzero"],
            "median_activation_nonzero": density["median_activation_nonzero"],
            "max_activation": density["max_activation"],
            "top_activating_token": top_tok_unfiltered,
            "top_activating_token_filtered": top_tok_filtered,
            "top_positive_logit_token": top_pos_token,
            "top_negative_logit_token": top_neg_token,
        })

        # Full per-token (optional)
        if args.save_full_per_token:
            for rec in records:
                for pos, (tok, act) in enumerate(
                    zip(rec["tokens"], rec["activations"]),
                ):
                    full_per_token_rows.append({
                        "layer": layer,
                        "feature_idx": feat_idx,
                        "item_idx": rec["item_idx"],
                        "position": pos,
                        "token": tok,
                        "activation": act,
                    })

    # Save outputs
    log("\nSaving outputs...")

    # feature_interpretability.parquet
    if interpretability_rows:
        pd.DataFrame(interpretability_rows).to_parquet(
            output_dir / "feature_interpretability.parquet",
            index=False, compression="snappy",
        )

    # activation_densities.json
    atomic_save_json(activation_densities, output_dir / "activation_densities.json")

    # Full per-token (optional)
    if full_per_token_rows:
        pd.DataFrame(full_per_token_rows).to_parquet(
            output_dir / "per_item_per_token_activations.parquet",
            index=False, compression="snappy",
        )
        log(f"  Full per-token: {len(full_per_token_rows)} rows")

    # Figures
    if not args.skip_figures:
        try:
            from src.visualization.token_feature_figures import (
                generate_c4_figures,
            )
            generate_c4_figures(
                output_dir, feature_manifest, logit_effects_df,
                activation_densities, all_accumulator,
                all_string_templates, viable,
            )
        except ImportError:
            log("WARNING: token_feature_figures module not available")
        except Exception as e:
            log(f"WARNING: figure generation failed: {e}")

    runtime = time.time() - t0
    log(f"\nC4 complete in {runtime:.1f}s")
    log(f"  Features characterised: {len(interpretability_rows)}")
    log(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
