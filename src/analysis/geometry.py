"""B3 core: subgroup direction geometry via Difference-in-Means.

Computes bias and identity directions for each subgroup under two
normalization regimes (raw-based, normed-based).  Produces pairwise cosine
matrices, differentiation metrics, and bias-identity alignment scores.

Independent of the SAE — reads raw activations from A2 directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log, progress_bar


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_category_hidden_states(
    run_dir: Path,
    cat: str,
    item_idxs: list[int],
    n_layers: int,
    hidden_dim: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Load hidden states for all items in a category.

    Returns
    -------
    all_hs_normed : (n_items, n_layers, hidden_dim), float32
    all_norms : (n_items, n_layers), float32
    loaded_idxs : item_idx list in loaded order (may be shorter if files missing)
    """
    cat_dir = run_dir / "A_extraction" / "activations" / cat

    hs_list: list[np.ndarray] = []
    norms_list: list[np.ndarray] = []
    loaded_idxs: list[int] = []

    for idx in progress_bar(item_idxs, desc=f"  Loading {cat}"):
        npz_path = cat_dir / f"item_{idx:06d}.npz"
        if not npz_path.exists():
            log(f"    WARNING: missing {npz_path.name}")
            continue

        data = np.load(npz_path, allow_pickle=True)
        hs_list.append(data["hidden_states"].astype(np.float32))
        norms_list.append(data["hidden_states_raw_norms"].astype(np.float32))
        loaded_idxs.append(idx)

    if not hs_list:
        empty_hs = np.zeros((0, n_layers, hidden_dim), dtype=np.float32)
        empty_norms = np.zeros((0, n_layers), dtype=np.float32)
        return empty_hs, empty_norms, []

    all_hs_normed = np.stack(hs_list, axis=0)
    all_norms = np.stack(norms_list, axis=0)

    return all_hs_normed, all_norms, loaded_idxs


# ---------------------------------------------------------------------------
# Per-subgroup direction computation
# ---------------------------------------------------------------------------

def compute_subgroup_directions(
    cat: str,
    sub: str,
    cat_meta: pd.DataFrame,
    all_hs_normed: np.ndarray,
    all_hs_raw: np.ndarray,
    n_layers: int,
    hidden_dim: int,
    min_n: int,
) -> dict | None:
    """Compute all four directions for one subgroup.

    Returns dict with keys ``"arrays"``, ``"norms"``, ``"info"``
    or ``None`` if all directions were skipped due to insufficient data.
    """
    # Build boolean masks --------------------------------------------------
    is_S = cat_meta["stereotyped_groups"].apply(lambda gs: sub in gs).values
    is_ambig = (cat_meta["context_condition"] == "ambig").values
    is_stereo = (cat_meta["model_answer_role"] == "stereotyped_target").values

    # Bias direction: items targeting S, ambig, split by response
    bias_stereo_mask = is_S & is_ambig & is_stereo
    bias_non_stereo_mask = is_S & is_ambig & ~is_stereo

    n_bias_stereo = int(bias_stereo_mask.sum())
    n_bias_non_stereo = int(bias_non_stereo_mask.sum())

    # Identity direction: items targeting S vs items NOT targeting S
    identity_S_mask = is_S
    identity_not_S_mask = ~is_S

    n_identity_S = int(identity_S_mask.sum())
    n_identity_not_S = int(identity_not_S_mask.sum())

    bias_ok = n_bias_stereo >= min_n and n_bias_non_stereo >= min_n
    identity_ok = n_identity_S >= min_n and n_identity_not_S >= min_n

    if not bias_ok and not identity_ok:
        return None

    arrays: dict[str, np.ndarray] = {}
    norms: dict[str, np.ndarray] = {}

    # Bias directions ------------------------------------------------------
    if bias_ok:
        # Raw-based
        mean_stereo_raw = all_hs_raw[bias_stereo_mask].mean(axis=0)
        mean_non_stereo_raw = all_hs_raw[bias_non_stereo_mask].mean(axis=0)
        bias_raw = mean_stereo_raw - mean_non_stereo_raw
        bias_raw_prenorm = np.linalg.norm(bias_raw, axis=1)
        safe = np.maximum(bias_raw_prenorm, 1e-8)[:, None]
        arrays["bias_direction_raw"] = (bias_raw / safe).astype(np.float32)
        norms["bias_direction_raw_norm"] = bias_raw_prenorm.astype(np.float32)

        # Normed-based
        mean_stereo_normed = all_hs_normed[bias_stereo_mask].mean(axis=0)
        mean_non_stereo_normed = all_hs_normed[bias_non_stereo_mask].mean(axis=0)
        bias_normed = mean_stereo_normed - mean_non_stereo_normed
        bias_normed_prenorm = np.linalg.norm(bias_normed, axis=1)
        safe_n = np.maximum(bias_normed_prenorm, 1e-8)[:, None]
        arrays["bias_direction_normed"] = (bias_normed / safe_n).astype(np.float32)
        norms["bias_direction_normed_norm"] = bias_normed_prenorm.astype(np.float32)

    # Identity directions --------------------------------------------------
    if identity_ok:
        mean_S_raw = all_hs_raw[identity_S_mask].mean(axis=0)
        mean_not_S_raw = all_hs_raw[identity_not_S_mask].mean(axis=0)
        identity_raw = mean_S_raw - mean_not_S_raw
        identity_raw_prenorm = np.linalg.norm(identity_raw, axis=1)
        safe = np.maximum(identity_raw_prenorm, 1e-8)[:, None]
        arrays["identity_direction_raw"] = (identity_raw / safe).astype(np.float32)
        norms["identity_direction_raw_norm"] = identity_raw_prenorm.astype(np.float32)

        mean_S_normed = all_hs_normed[identity_S_mask].mean(axis=0)
        mean_not_S_normed = all_hs_normed[identity_not_S_mask].mean(axis=0)
        identity_normed = mean_S_normed - mean_not_S_normed
        identity_normed_prenorm = np.linalg.norm(identity_normed, axis=1)
        safe_n = np.maximum(identity_normed_prenorm, 1e-8)[:, None]
        arrays["identity_direction_normed"] = (identity_normed / safe_n).astype(np.float32)
        norms["identity_direction_normed_norm"] = identity_normed_prenorm.astype(np.float32)

    # Build info record ----------------------------------------------------
    info: dict[str, Any] = {
        "category": cat,
        "subgroup": sub,
        "skipped": False,
        "n_total_targeting_S": n_identity_S,
        "n_total_not_targeting_S": n_identity_not_S,
        "n_bias_stereo": n_bias_stereo,
        "n_bias_non_stereo": n_bias_non_stereo,
        "bias_computed": bias_ok,
        "identity_computed": identity_ok,
    }

    if not bias_ok:
        info["bias_skip_reason"] = (
            f"n_bias_stereo={n_bias_stereo} < min_n={min_n}"
            if n_bias_stereo < min_n
            else f"n_bias_non_stereo={n_bias_non_stereo} < min_n={min_n}"
        )

    if not identity_ok:
        info["identity_skip_reason"] = (
            f"n_identity_S={n_identity_S} < min_n={min_n}"
            if n_identity_S < min_n
            else f"n_identity_not_S={n_identity_not_S} < min_n={min_n}"
        )

    return {"arrays": arrays, "norms": norms, "info": info}


# ---------------------------------------------------------------------------
# Per-category processing
# ---------------------------------------------------------------------------

def process_category(
    cat: str,
    run_dir: Path,
    meta_df: pd.DataFrame,
    n_layers: int,
    hidden_dim: int,
    min_n: int,
    max_items: int | None,
    directions_arrays: dict[str, np.ndarray],
    directions_norms: dict[str, np.ndarray],
    subgroup_info: dict[tuple[str, str], dict[str, Any]],
) -> None:
    """Load all items for a category and compute directions for every subgroup."""
    log(f"\n{'=' * 60}")
    log(f"Category: {cat}")
    log(f"{'=' * 60}")

    cat_meta = meta_df[meta_df["category"] == cat].copy()

    # Ensure stereotyped_groups is a list
    cat_meta["stereotyped_groups"] = cat_meta["stereotyped_groups"].apply(
        lambda x: x if isinstance(x, list) else json.loads(x)
    )

    if max_items:
        cat_meta = cat_meta.head(max_items)

    # Load hidden states ---------------------------------------------------
    all_hs_normed, all_norms, item_idxs = load_category_hidden_states(
        run_dir, cat, cat_meta["item_idx"].tolist(), n_layers, hidden_dim,
    )

    if len(item_idxs) == 0:
        log(f"  No activations found for category '{cat}', skipping.")
        return

    log(f"  Loaded {len(item_idxs)} items, shape {all_hs_normed.shape}")

    # Reconstruct raw activations
    all_hs_raw = all_hs_normed * all_norms[:, :, None]

    # Re-index cat_meta to match loaded order
    cat_meta = cat_meta.set_index("item_idx").loc[item_idxs].reset_index()

    # Enumerate subgroups --------------------------------------------------
    all_subgroups: set[str] = set()
    for gs in cat_meta["stereotyped_groups"]:
        all_subgroups.update(gs)
    all_subgroups_sorted = sorted(all_subgroups)

    log(f"  Subgroups ({len(all_subgroups_sorted)}): {all_subgroups_sorted}")

    for sub in all_subgroups_sorted:
        result = compute_subgroup_directions(
            cat=cat,
            sub=sub,
            cat_meta=cat_meta,
            all_hs_normed=all_hs_normed,
            all_hs_raw=all_hs_raw,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            min_n=min_n,
        )

        if result is None:
            log(f"    {sub}: insufficient data, all directions skipped")
            subgroup_info[(cat, sub)] = {
                "category": cat,
                "subgroup": sub,
                "skipped": True,
            }
            continue

        for key, arr in result["arrays"].items():
            directions_arrays[f"{key}_{cat}_{sub}"] = arr
        for key, arr in result["norms"].items():
            directions_norms[f"{key}_{cat}_{sub}"] = arr

        subgroup_info[(cat, sub)] = result["info"]

        info = result["info"]
        parts = []
        if info["bias_computed"]:
            parts.append(f"bias(n_s={info['n_bias_stereo']},n_ns={info['n_bias_non_stereo']})")
        else:
            parts.append("bias=SKIP")
        if info["identity_computed"]:
            parts.append(f"identity(n_S={info['n_total_targeting_S']},n_notS={info['n_total_not_targeting_S']})")
        else:
            parts.append("identity=SKIP")
        log(f"    {sub}: {', '.join(parts)}")


# ---------------------------------------------------------------------------
# Saving directions
# ---------------------------------------------------------------------------

def save_directions(
    run_dir: Path,
    directions_arrays: dict[str, np.ndarray],
    directions_norms: dict[str, np.ndarray],
) -> Path:
    """Save all directions and pre-normalization norms to a single .npz."""
    out_dir = ensure_dir(run_dir / "B_geometry")
    out_path = out_dir / "subgroup_directions.npz"

    combined = {**directions_arrays, **directions_norms}

    tmp_path = out_path.with_suffix(".npz.tmp")
    np.savez(tmp_path, **combined)
    tmp_path.rename(out_path)

    log(f"Saved {len(combined)} arrays to {out_path}")
    return out_path


def load_direction(
    run_dir: Path,
    direction_type: str,
    normalize: str,
    category: str,
    subgroup: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load a direction and its pre-normalization norm.

    Parameters
    ----------
    direction_type : "bias" or "identity"
    normalize : "raw" or "normed"

    Returns
    -------
    (unit_direction, raw_norm) or (None, None) if not present.
    unit_direction : (n_layers, hidden_dim) float32
    raw_norm : (n_layers,) float32
    """
    npz_path = run_dir / "B_geometry" / "subgroup_directions.npz"
    data = np.load(npz_path)

    key_dir = f"{direction_type}_direction_{normalize}_{category}_{subgroup}"
    key_norm = f"{direction_type}_direction_{normalize}_norm_{category}_{subgroup}"

    if key_dir not in data:
        return None, None

    return data[key_dir], data[key_norm]


# ---------------------------------------------------------------------------
# Pairwise cosines
# ---------------------------------------------------------------------------

# Direction type specs: (direction_type, normalize, label)
DIRECTION_SPECS = [
    ("bias", "raw", "bias_raw"),
    ("bias", "normed", "bias_normed"),
    ("identity", "raw", "identity_raw"),
    ("identity", "normed", "identity_normed"),
]


def compute_all_cosines(
    directions_arrays: dict[str, np.ndarray],
    categories: list[str],
    n_layers: int,
) -> pd.DataFrame:
    """Compute pairwise cosine similarities for all subgroup pairs.

    Returns DataFrame with columns: category, direction_type, layer,
    subgroup_A, subgroup_B, cosine.
    """
    rows: list[dict[str, Any]] = []

    for cat in categories:
        # Enumerate subgroups per direction type
        subgroups_with_dirs: dict[str, list[str]] = {}
        for dtype, normalize, label in DIRECTION_SPECS:
            subs = []
            prefix = f"{dtype}_direction_{normalize}_{cat}_"
            for key in directions_arrays:
                if key.startswith(prefix):
                    subs.append(key[len(prefix):])
            subgroups_with_dirs[label] = sorted(subs)

        for dtype, normalize, label in DIRECTION_SPECS:
            subs = subgroups_with_dirs[label]
            if len(subs) < 2:
                continue

            # Gather direction arrays
            dir_arrays: dict[str, np.ndarray] = {}
            for sub in subs:
                key = f"{dtype}_direction_{normalize}_{cat}_{sub}"
                dir_arrays[sub] = directions_arrays[key]

            # Compute pairwise cosines (vectorized per pair across layers)
            for i, sub_A in enumerate(subs):
                for sub_B in subs[i + 1:]:
                    arr_A = dir_arrays[sub_A]
                    arr_B = dir_arrays[sub_B]
                    cosines = np.sum(arr_A * arr_B, axis=1)

                    for layer in range(n_layers):
                        rows.append({
                            "category": cat,
                            "direction_type": label,
                            "layer": int(layer),
                            "subgroup_A": sub_A,
                            "subgroup_B": sub_B,
                            "cosine": round(float(cosines[layer]), 6),
                        })

    return pd.DataFrame(rows)


def save_cosines(run_dir: Path, cosine_df: pd.DataFrame) -> Path:
    """Save pairwise cosines to parquet (atomic)."""
    out_dir = ensure_dir(run_dir / "B_geometry")
    out_path = out_dir / "cosine_pairs.parquet"
    tmp_path = out_path.with_suffix(".parquet.tmp")
    cosine_df.to_parquet(tmp_path, index=False, compression="snappy")
    tmp_path.rename(out_path)
    log(f"Saved cosine_pairs.parquet: {len(cosine_df)} rows")
    return out_path


# ---------------------------------------------------------------------------
# Differentiation metrics
# ---------------------------------------------------------------------------

def find_stable_range(
    cosine_sub_df: pd.DataFrame,
    n_layers: int,
    peak_layer: int,
) -> tuple[int, int]:
    """Find contiguous layers around peak where every pair preserves cosine sign."""
    peak_data = cosine_sub_df[cosine_sub_df["layer"] == peak_layer]
    peak_signs: dict[tuple[str, str], float] = {}
    for _, row in peak_data.iterrows():
        pair_key = (row["subgroup_A"], row["subgroup_B"])
        peak_signs[pair_key] = np.sign(row["cosine"])

    def signs_match(layer: int) -> bool:
        layer_data = cosine_sub_df[cosine_sub_df["layer"] == layer]
        for _, row in layer_data.iterrows():
            pair_key = (row["subgroup_A"], row["subgroup_B"])
            if pair_key in peak_signs:
                if np.sign(row["cosine"]) != peak_signs[pair_key]:
                    return False
        return True

    start = peak_layer
    while start > 0 and signs_match(start - 1):
        start -= 1

    end = peak_layer
    while end < n_layers - 1 and signs_match(end + 1):
        end += 1

    return start, end


def compute_differentiation_metrics(
    cosine_df: pd.DataFrame,
    categories: list[str],
    n_layers: int,
) -> dict[str, Any]:
    """Per (category, direction_type): per-layer metrics and peak layer."""
    result: dict[str, Any] = {}

    for cat in categories:
        result[cat] = {}

        for dtype_label in ["bias_raw", "bias_normed", "identity_raw", "identity_normed"]:
            sub_df = cosine_df[
                (cosine_df["category"] == cat)
                & (cosine_df["direction_type"] == dtype_label)
            ]

            if sub_df.empty:
                continue

            per_layer: dict[int, dict[str, Any]] = {}
            for layer in range(n_layers):
                layer_pairs = sub_df[sub_df["layer"] == layer]["cosine"].values
                if len(layer_pairs) == 0:
                    continue

                per_layer[layer] = {
                    "variance": float(np.var(layer_pairs)),
                    "range": float(layer_pairs.max() - layer_pairs.min()),
                    "mean_distance": float(np.mean(1 - layer_pairs)),
                    "min_cosine": float(layer_pairs.min()),
                    "max_cosine": float(layer_pairs.max()),
                    "median_cosine": float(np.median(layer_pairs)),
                    "n_pairs": int(len(layer_pairs)),
                }

            if not per_layer:
                continue

            peak_layer = max(per_layer, key=lambda l: per_layer[l]["variance"])

            stable_start, stable_end = find_stable_range(
                sub_df, n_layers, peak_layer,
            )

            peak_data = sub_df[sub_df["layer"] == peak_layer]
            most_anti_pair: dict[str, Any] | None = None
            if not peak_data.empty:
                most_anti = peak_data.nsmallest(1, "cosine").iloc[0]
                most_anti_pair = {
                    "subgroup_A": most_anti["subgroup_A"],
                    "subgroup_B": most_anti["subgroup_B"],
                    "cosine": float(most_anti["cosine"]),
                }

            # Convert per_layer keys to strings for JSON serialization
            per_layer_str = {str(k): v for k, v in per_layer.items()}

            result[cat][dtype_label] = {
                "per_layer_metrics": per_layer_str,
                "peak_layer": int(peak_layer),
                "peak_variance": float(per_layer[peak_layer]["variance"]),
                "peak_range": float(per_layer[peak_layer]["range"]),
                "peak_mean_distance": float(per_layer[peak_layer]["mean_distance"]),
                "stable_range": [int(stable_start), int(stable_end)],
                "stable_range_length": int(stable_end - stable_start + 1),
                "most_anti_aligned_pair_at_peak": most_anti_pair,
            }

    return result


# ---------------------------------------------------------------------------
# Bias-identity alignment
# ---------------------------------------------------------------------------

def compute_alignment(
    directions_arrays: dict[str, np.ndarray],
    categories: list[str],
    n_layers: int,
) -> dict[str, Any]:
    """Compute cosine(bias, identity) per subgroup per layer, both norm regimes."""
    result: dict[str, Any] = {}

    for cat in categories:
        cat_result: dict[str, Any] = {}

        # Find subgroups that have bias directions (identity may or may not exist)
        subgroups: set[str] = set()
        for key in directions_arrays:
            for norm_type in ["raw", "normed"]:
                prefix = f"bias_direction_{norm_type}_{cat}_"
                if key.startswith(prefix):
                    subgroups.add(key[len(prefix):])

        for sub in sorted(subgroups):
            sub_alignment: dict[str, Any] = {"raw": None, "normed": None}

            for norm_type in ["raw", "normed"]:
                bias_key = f"bias_direction_{norm_type}_{cat}_{sub}"
                id_key = f"identity_direction_{norm_type}_{cat}_{sub}"

                if bias_key not in directions_arrays or id_key not in directions_arrays:
                    continue

                bias = directions_arrays[bias_key]
                identity = directions_arrays[id_key]

                alignments = np.sum(bias * identity, axis=1)
                alignments_sq = alignments ** 2

                peak_layer = int(np.argmax(np.abs(alignments)))
                peak_alignment = float(alignments[peak_layer])

                sub_alignment[norm_type] = {
                    "per_layer_alignment": [round(float(a), 4) for a in alignments],
                    "per_layer_alignment_squared": [round(float(a), 4) for a in alignments_sq],
                    "peak_layer": peak_layer,
                    "peak_alignment": round(peak_alignment, 4),
                    "peak_alignment_squared": round(peak_alignment ** 2, 4),
                    "mean_alignment": round(float(alignments.mean()), 4),
                    "mean_alignment_squared": round(float(alignments_sq.mean()), 4),
                }

            # Only include if at least one norm type computed
            if sub_alignment["raw"] is not None or sub_alignment["normed"] is not None:
                cat_result[sub] = sub_alignment

        result[cat] = cat_result

    return result


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def build_summary(
    subgroup_info: dict[tuple[str, str], dict[str, Any]],
    directions_norms: dict[str, np.ndarray],
    categories: list[str],
    min_n: int,
) -> dict[str, Any]:
    """Build the subgroup_directions_summary.json content."""
    per_subgroup: dict[str, Any] = {}

    for (cat, sub), info in sorted(subgroup_info.items()):
        key = f"{cat}/{sub}"
        entry = dict(info)  # shallow copy

        # Add norm range/peak for each computed direction type
        for dtype in ["bias_direction_raw", "bias_direction_normed",
                       "identity_direction_raw", "identity_direction_normed"]:
            norm_key = f"{dtype}_norm_{cat}_{sub}"
            if norm_key in directions_norms:
                norm_arr = directions_norms[norm_key]
                entry[f"{dtype}_norm_range"] = [
                    round(float(norm_arr.min()), 4),
                    round(float(norm_arr.max()), 4),
                ]
                entry[f"{dtype}_norm_peak_layer"] = int(np.argmax(norm_arr))

        per_subgroup[key] = entry

    return {
        "min_n_per_group": min_n,
        "categories": categories,
        "n_subgroups_total": len(subgroup_info),
        "n_subgroups_with_bias": sum(
            1 for v in subgroup_info.values() if v.get("bias_computed", False)
        ),
        "n_subgroups_with_identity": sum(
            1 for v in subgroup_info.values() if v.get("identity_computed", False)
        ),
        "per_subgroup": per_subgroup,
    }


# ---------------------------------------------------------------------------
# Resume check
# ---------------------------------------------------------------------------

_REQUIRED_OUTPUTS = [
    "B_geometry/subgroup_directions.npz",
    "B_geometry/cosine_pairs.parquet",
    "B_geometry/subgroup_directions_summary.json",
    "B_geometry/differentiation_metrics.json",
    "B_geometry/bias_identity_alignment.json",
]


def b3_complete(run_dir: Path) -> bool:
    """Check whether all required B3 outputs already exist."""
    return all((run_dir / p).exists() for p in _REQUIRED_OUTPUTS)
