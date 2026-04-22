"""A3 core: SAE encoding of hidden states into sparse feature activations.

Reads hidden states from A2's .npz files, passes them through pre-trained
SAE encoders at each layer, and writes sparse feature activations as
parquet files.  Pure matrix computation — no model forward passes.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.sae.wrapper import SAEWrapper
from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log, progress_bar


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def select_encode_device(config_device: str) -> str:
    """Select the best available device for SAE encoding."""
    if config_device == "mps" or (
        config_device != "cpu" and torch.backends.mps.is_available()
    ):
        return "mps"
    elif config_device.startswith("cuda") or torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


# ---------------------------------------------------------------------------
# SAE source validation
# ---------------------------------------------------------------------------

def validate_sae_source(sae_source: str, test_layer: int, expansion: int) -> None:
    """Verify the HuggingFace SAE repo is accessible and correctly structured."""
    log(f"Validating SAE source: {sae_source}")
    try:
        from huggingface_hub import list_repo_files
        files = list_repo_files(sae_source)
        pattern = f"L{test_layer}R-{expansion}x"
        matches = [f for f in files if pattern in f]
        if not matches:
            raise FileNotFoundError(
                f"No files matching pattern '{pattern}' found in {sae_source}. "
                f"Available files (sample): {files[:10]}"
            )
        log(f"  SAE source validated: found {len(matches)} files for layer {test_layer}")
    except Exception as e:
        raise SystemExit(
            f"SAE source validation failed: {e}\n"
            f"Verify the repo exists at: https://huggingface.co/{sae_source}"
        )


# ---------------------------------------------------------------------------
# Metadata and subgroup lookup
# ---------------------------------------------------------------------------

def load_metadata(run_dir: Path) -> pd.DataFrame:
    """Load item metadata.  Prefer parquet; fall back to scanning .npz files."""
    meta_path = run_dir / "A_extraction" / "metadata.parquet"
    if meta_path.exists():
        df = pd.read_parquet(meta_path)
        log(f"Loaded metadata: {len(df)} items from {meta_path.name}")
        return df

    log("metadata.parquet not found; building from .npz files (slow)...")
    records: list[dict[str, Any]] = []
    act_dir = run_dir / "A_extraction" / "activations"
    for cat_dir in sorted(act_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        for npz_path in sorted(cat_dir.glob("item_*.npz")):
            try:
                data = np.load(npz_path, allow_pickle=True)
                raw = data["metadata_json"]
                meta_str = raw.item() if raw.shape == () else str(raw)
                meta = json.loads(meta_str)
                records.append(meta)
            except Exception:
                continue

    df = pd.DataFrame(records)
    df.to_parquet(meta_path, index=False, compression="snappy")
    log(f"  Built metadata.parquet: {len(df)} items")
    return df


def load_all_stimuli(
    run_dir: Path, categories: list[str],
) -> dict[str, list[dict[str, Any]]]:
    """Load stimuli JSONs for all categories."""
    stimuli: dict[str, list[dict[str, Any]]] = {}
    for cat in categories:
        path = run_dir / "A_extraction" / "stimuli" / f"{cat}.json"
        if path.exists():
            with open(path) as f:
                stimuli[cat] = json.load(f)
    return stimuli


def build_subgroup_lookup(
    stimuli_by_cat: dict[str, list[dict[str, Any]]],
) -> dict[tuple[str, int], list[str]]:
    """Build ``(category, item_idx) → stereotyped_groups`` mapping."""
    lookup: dict[tuple[str, int], list[str]] = {}
    for cat, items in stimuli_by_cat.items():
        for item in items:
            lookup[(cat, item["item_idx"])] = item["stereotyped_groups"]
    return lookup


# ---------------------------------------------------------------------------
# Encoding validation
# ---------------------------------------------------------------------------

def run_encoding_validation(
    sae: SAEWrapper,
    hs_raw: np.ndarray,
    feature_activations: torch.Tensor,
    n_active: int,
    layer: int,
    full_reconstruct: bool = False,
) -> None:
    """Validate SAE encoding on a single item."""
    log(f"  === Encoding Validation (layer {layer}) ===")

    log(f"    L0 = {n_active} active features out of {sae.n_features}")
    if n_active < 5:
        raw_norm = float(np.linalg.norm(hs_raw))
        log(f"    WARNING: Very low L0 ({n_active}). Raw norm: {raw_norm:.4f}")
    elif n_active > 500:
        log(f"    WARNING: Very high L0 ({n_active}). Check SAE/model match.")
    else:
        log(f"    L0 in expected range (20-200 for 32x SAE)")

    # Top-5 active features.
    nonzero_mask = feature_activations > 0
    if n_active > 0:
        nonzero_acts = feature_activations[nonzero_mask]
        top_k = min(5, n_active)
        top_vals, top_local_idx = torch.topk(nonzero_acts, top_k)
        global_indices = nonzero_mask.nonzero(as_tuple=True)[0]
        top_global_idx = global_indices[top_local_idx.cpu()]
        log(f"    Top-{top_k} features:")
        for fidx, fval in zip(top_global_idx.tolist(), top_vals.tolist()):
            log(f"      Feature {fidx}: activation = {fval:.4f}")

    # Reconstruction check (expensive).
    if full_reconstruct:
        reconstructed = sae.decode(feature_activations)
        hs_tensor = torch.from_numpy(hs_raw).to(
            reconstructed.dtype
        ).to(reconstructed.device)
        recon_error = float(torch.norm(reconstructed - hs_tensor).item())
        raw_norm = float(torch.norm(hs_tensor).item())
        relative_error = recon_error / max(raw_norm, 1e-8)
        log(f"    Reconstruction error: {recon_error:.4f} "
            f"(relative: {relative_error:.4f})")
        if relative_error > 0.3:
            log(f"    WARNING: High reconstruction error.")
        elif relative_error > 0.1:
            log(f"    Moderate reconstruction error. Acceptable.")
        else:
            log(f"    Reconstruction quality: good")

    log(f"  === End Validation ===")


# ---------------------------------------------------------------------------
# Batch encoding
# ---------------------------------------------------------------------------

def encode_batch(
    sae: SAEWrapper,
    batch_hs: list[np.ndarray],
    batch_idxs: list[int],
    category: str,
    layer: int,
    device: str,
    validate: bool = False,
    full_validate: bool = False,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Encode a batch of hidden states through the SAE.

    Returns ``(records, l0s)`` — parquet row dicts and L0 counts.
    """
    batch_tensor = torch.from_numpy(np.stack(batch_hs)).to(dtype=torch.float32)
    feature_activations = sae.encode(batch_tensor)

    records: list[dict[str, Any]] = []
    l0s: list[int] = []

    for i in range(len(batch_hs)):
        item_acts = feature_activations[i]
        nonzero_mask = item_acts > 0
        n_active = int(nonzero_mask.sum().item())
        l0s.append(n_active)

        if validate and i == 0:
            run_encoding_validation(
                sae, batch_hs[i], item_acts, n_active, layer,
                full_reconstruct=full_validate,
            )

        if n_active > 0:
            feat_indices = nonzero_mask.nonzero(as_tuple=True)[0].cpu().numpy()
            act_values = item_acts[nonzero_mask].cpu().detach().float().numpy()
            for fidx, aval in zip(feat_indices, act_values):
                records.append({
                    "item_idx": batch_idxs[i],
                    "feature_idx": int(fidx),
                    "activation_value": float(aval),
                    "category": category,
                })

    return records, l0s


# ---------------------------------------------------------------------------
# Per-layer encoding
# ---------------------------------------------------------------------------

BATCH_SIZE = 64


def encode_layer(
    layer: int,
    run_dir: Path,
    config: dict[str, Any],
    categories: list[str],
    device: str,
    subgroup_lookup: dict[tuple[str, int], list[str]],
    max_items: int | None,
) -> bool:
    """Encode all items at one layer.  Returns True on success."""
    output_dir = ensure_dir(run_dir / "A_extraction" / "sae_encoding")
    parquet_path = output_dir / f"layer_{layer:02d}.parquet"
    summary_path = output_dir / f"layer_{layer:02d}_summary.json"

    if parquet_path.exists():
        log(f"Layer {layer:02d}: already encoded, skipping")
        return True

    log(f"\n{'=' * 60}")
    log(f"Encoding layer {layer:02d}")
    log(f"{'=' * 60}")

    t0 = time.time()

    # Load SAE.
    try:
        sae = SAEWrapper(
            config["sae_source"],
            layer=layer,
            site=config.get("sae_site", "R"),
            expansion=config["sae_expansion"],
            device=device,
        )
    except Exception as e:
        log(f"  ERROR: Failed to load SAE for layer {layer}: {e}")
        return False

    log(f"  SAE loaded: {sae.n_features} features, hidden_dim={sae.hidden_dim}")

    n_layers = config["n_layers"]
    full_validation_layers = {0, n_layers // 2, n_layers - 1}

    all_records: list[dict[str, Any]] = []
    per_category_stats: dict[str, dict[str, Any]] = {}
    per_subgroup_l0: dict[str, dict[str, Any]] = {}
    validation_done = False

    for cat in categories:
        cat_dir = run_dir / "A_extraction" / "activations" / cat
        if not cat_dir.is_dir():
            log(f"  No activations for {cat}, skipping")
            continue

        npz_paths = sorted(cat_dir.glob("item_*.npz"))
        if max_items:
            npz_paths = npz_paths[:max_items]

        n_items = len(npz_paths)
        log(f"  {cat}: {n_items} items")

        cat_l0s: list[int] = []
        cat_records: list[dict[str, Any]] = []
        batch_hs: list[np.ndarray] = []
        batch_idxs: list[int] = []

        for npz_i, npz_path in enumerate(progress_bar(
            npz_paths, desc=f"    {cat}", unit="items",
        )):
            data = np.load(npz_path, allow_pickle=True)
            hs_normed = data["hidden_states"][layer].astype(np.float32)
            raw_norm = float(data["hidden_states_raw_norms"][layer])
            hs_raw = hs_normed * raw_norm

            item_idx = int(npz_path.stem.split("_")[1])
            batch_hs.append(hs_raw)
            batch_idxs.append(item_idx)

            if len(batch_hs) == BATCH_SIZE or npz_i == len(npz_paths) - 1:
                batch_records, batch_l0s = encode_batch(
                    sae, batch_hs, batch_idxs, cat, layer, device,
                    validate=(not validation_done),
                    full_validate=(
                        layer in full_validation_layers and not validation_done
                    ),
                )
                cat_records.extend(batch_records)
                cat_l0s.extend(batch_l0s)
                if not validation_done and batch_records:
                    validation_done = True
                batch_hs.clear()
                batch_idxs.clear()

        all_records.extend(cat_records)

        # Category-level L0 stats.
        if cat_l0s:
            l0_arr = np.array(cat_l0s)
            cat_stat: dict[str, Any] = {
                "n_items_encoded": n_items,
                "mean_l0": round(float(l0_arr.mean()), 1),
                "std_l0": round(float(l0_arr.std()), 1),
                "median_l0": round(float(np.median(l0_arr)), 1),
                "min_l0": int(l0_arr.min()),
                "max_l0": int(l0_arr.max()),
                "total_nonzero_entries": len(cat_records),
            }
            if cat_records:
                acts = [r["activation_value"] for r in cat_records]
                cat_stat["mean_activation_nonzero"] = round(float(np.mean(acts)), 4)
                cat_stat["max_activation"] = round(float(np.max(acts)), 4)
            per_category_stats[cat] = cat_stat

        # Per-subgroup L0 stats.
        subgroup_l0_accum: dict[str, list[int]] = {}
        item_idxs_for_l0 = [
            int(p.stem.split("_")[1]) for p in npz_paths[: len(cat_l0s)]
        ]
        for idx, l0 in zip(item_idxs_for_l0, cat_l0s):
            groups = subgroup_lookup.get((cat, idx), [])
            for sg in groups:
                subgroup_l0_accum.setdefault(sg, []).append(l0)

        for sg, l0_vals in sorted(subgroup_l0_accum.items()):
            arr = np.array(l0_vals)
            per_subgroup_l0[f"{cat}/{sg}"] = {
                "n_items": len(l0_vals),
                "mean_l0": round(float(arr.mean()), 1),
                "std_l0": round(float(arr.std()), 1),
            }

        if device == "mps":
            torch.mps.empty_cache()

    # Write parquet atomically.
    if all_records:
        df = pd.DataFrame(all_records)
        df["item_idx"] = df["item_idx"].astype(np.int32)
        df["feature_idx"] = df["feature_idx"].astype(np.int32)
        df["activation_value"] = df["activation_value"].astype(np.float32)

        tmp_path = parquet_path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp_path, index=False, compression="snappy")
        tmp_path.rename(parquet_path)

        parquet_size_mb = parquet_path.stat().st_size / (1024 * 1024)
        log(f"  Parquet: {len(df)} rows, {parquet_size_mb:.1f} MB → "
            f"{parquet_path.name}")
    else:
        log(f"  WARNING: No records produced for layer {layer}")
        return False

    # Write per-layer summary.
    elapsed = time.time() - t0
    layer_summary = {
        "layer": layer,
        "sae_source": config["sae_source"],
        "sae_expansion": config["sae_expansion"],
        "n_features": sae.n_features,
        "encoding_device": device,
        "per_category": per_category_stats,
        "per_subgroup_l0": per_subgroup_l0,
        "encoding_time_seconds": round(elapsed, 1),
    }
    atomic_save_json(layer_summary, summary_path)
    log(f"  Complete in {elapsed:.1f}s")

    # Unload SAE to free memory.
    del sae
    if device == "mps":
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()
    gc.collect()

    return True


# ---------------------------------------------------------------------------
# Global encoding summary
# ---------------------------------------------------------------------------

def build_encoding_summary(
    run_dir: Path,
    config: dict[str, Any],
    layers: list[int],
    failed_layers: list[int],
    categories: list[str],
    encode_device: str,
) -> None:
    """Build encoding_summary.json from per-layer summaries."""
    encoding_dir = run_dir / "A_extraction" / "sae_encoding"

    l0_by_layer: dict[str, dict[str, float]] = {}
    total_time = 0.0
    total_size_bytes = 0

    for layer in layers:
        if layer in failed_layers:
            continue
        summary_path = encoding_dir / f"layer_{layer:02d}_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                ls = json.load(f)
            means = [v["mean_l0"] for v in ls["per_category"].values()]
            stds = [v["std_l0"] for v in ls["per_category"].values()]
            l0_by_layer[str(layer)] = {
                "mean": round(float(np.mean(means)), 1),
                "std": round(float(np.mean(stds)), 1),
            }
            total_time += ls.get("encoding_time_seconds", 0)

        parquet_path = encoding_dir / f"layer_{layer:02d}.parquet"
        if parquet_path.exists():
            total_size_bytes += parquet_path.stat().st_size

    items_per_cat: dict[str, int] = {}
    for cat in categories:
        stim_path = run_dir / "A_extraction" / "stimuli" / f"{cat}.json"
        if stim_path.exists():
            with open(stim_path) as f:
                items_per_cat[cat] = len(json.load(f))

    summary = {
        "sae_source": config["sae_source"],
        "sae_expansion": config["sae_expansion"],
        "encode_device": encode_device,
        "layers_encoded": sorted(l for l in layers if l not in failed_layers),
        "failed_layers": failed_layers,
        "total_items_per_category": items_per_cat,
        "total_parquet_size_mb": round(total_size_bytes / (1024 * 1024), 1),
        "total_encoding_time_seconds": round(total_time, 1),
        "l0_by_layer": l0_by_layer,
    }

    summary_path = encoding_dir / "encoding_summary.json"
    atomic_save_json(summary, summary_path)

    log(f"\nEncoding summary → {summary_path}")
    log(f"Layers encoded: {len(summary['layers_encoded'])}/{len(layers)}")
    if failed_layers:
        log(f"Failed layers: {failed_layers}")
    log(f"Total parquet size: {summary['total_parquet_size_mb']:.1f} MB")
    log(f"Total time: {summary['total_encoding_time_seconds']:.0f}s")
