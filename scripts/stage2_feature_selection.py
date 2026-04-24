"""Stage 2: SAE Feature Selection.

Identify SAE features causally implicated in demographic subgroup encoding
using two complementary methods per subgroup:

  Method A — Identity features: SAE decoder columns geometrically aligned
             with the subgroup's DIM identity direction from Stage 1.

  Method B — Bias-prediction features: SAE features whose max-pooled
             activations predict biased model output on S-targeted items
             (Movva-style L1-regularised logistic regression).

The intersection identifies directly bias-mediating identity features.

Usage:
    python scripts/stage2_feature_selection.py \
        --run_dir runs/llama-3.1-8b_2026-04-22/

    python scripts/stage2_feature_selection.py \
        --run_dir runs/llama-3.1-8b_2026-04-22/ --categories so,race

    python scripts/stage2_feature_selection.py \
        --run_dir runs/llama-3.1-8b_2026-04-22/ --skip_extraction
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*penalty.*deprecated.*", category=UserWarning)

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.wrapper import ModelWrapper, locate_hidden_tensor
from src.sae.wrapper import SAEWrapper
from src.utils.io import atomic_save_json
from src.utils.logging import progress_bar

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

WONG = {
    "orange": "#E69F00", "sky_blue": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00",
    "purple": "#CC79A7", "black": "#000000",
}
DPI = 200


def log(msg: str) -> None:
    print(f"[stage2] {msg}", flush=True)


def _save_fig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2: SAE Feature Selection")
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--layer", type=int, default=14)
    p.add_argument("--alias_threshold", type=float, default=0.98,
                   help="Cosine threshold for collapsing representational aliases")
    p.add_argument("--categories", type=str, default=None,
                   help="Subset of categories; default uses all except excluded")
    p.add_argument("--excluded_categories", type=str, default="age",
                   help="Categories to always exclude (comma-separated)")
    p.add_argument("--top_k_identity", type=int, default=100,
                   help="Number of top identity features to keep per subgroup")
    p.add_argument("--l1_c_values", type=str, default="0.01,0.1,1.0,10.0",
                   help="C values for L1 cross-validation")
    p.add_argument("--n_cv_folds", type=int, default=5)
    p.add_argument("--min_n_stereo", type=int, default=10,
                   help="Minimum stereotyped-response items to train probe")
    p.add_argument("--min_n_non_stereo", type=int, default=10,
                   help="Minimum non-stereotyped items to train probe")
    p.add_argument("--skip_extraction", action="store_true",
                   help="Assume all-token activations already extracted")
    p.add_argument("--batch_size_extraction", type=int, default=1,
                   help="Forward pass batch size for per-token extraction")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--skip_figures", action="store_true")
    p.add_argument("--random_seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_metadata(run_dir: Path) -> pd.DataFrame:
    meta_path = run_dir / "A_extraction" / "metadata.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.parquet not found at {meta_path}")
    df = pd.read_parquet(meta_path)
    sample = df["stereotyped_groups"].iloc[0]
    if isinstance(sample, str):
        df["stereotyped_groups"] = df["stereotyped_groups"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )
    df["stereotyped_groups"] = df["stereotyped_groups"].apply(
        lambda gs: [s.strip().lower() for s in gs] if gs else []
    )
    return df


def load_stimuli(run_dir: Path, category: str) -> list[dict]:
    stim_path = run_dir / "A_extraction" / "stimuli" / f"{category}.json"
    if not stim_path.exists():
        raise FileNotFoundError(f"Stimuli not found: {stim_path}")
    with open(stim_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Phase 2a: Per-token hidden state extraction
# ---------------------------------------------------------------------------

def make_all_token_hook(
    hidden_dim: int, storage: dict[str, torch.Tensor],
) -> callable:
    """Capture residual stream at all token positions after layer forward."""
    def hook_fn(module: object, args: object, output: object) -> None:
        h = locate_hidden_tensor(output, hidden_dim)
        # h: (batch=1, seq_len, hidden_dim)
        storage["hidden"] = h[0].detach().to("cpu", dtype=torch.float16)
    return hook_fn


def _validate_npz(path: Path, hidden_dim: int) -> bool:
    """Check an existing .npz is readable and has correct shape."""
    try:
        data = np.load(path, allow_pickle=True)
        h = data["hidden_all_tokens"]
        if h.ndim != 2 or h.shape[1] != hidden_dim:
            return False
        if h.shape[0] == 0:
            return False
        return True
    except Exception:
        return False


def extract_all_token_activations(
    run_dir: Path,
    categories: list[str],
    layer: int,
    wrapper: ModelWrapper,
    device: str,
) -> None:
    """For every ambig item in scoped categories, save per-token hidden states."""
    meta_df = load_metadata(run_dir)
    tokenizer = wrapper.tokenizer
    hidden_dim = wrapper.hidden_dim

    output_root = run_dir / "A_extraction" / "activations_all_tokens" / f"L{layer:02d}"
    output_root.mkdir(parents=True, exist_ok=True)

    # Accumulate with prior summary if it exists (supports adding categories)
    summary_path = output_root / "_extraction_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        # Merge category lists
        prev_cats = set(summary.get("categories", []))
        summary["categories"] = sorted(prev_cats | set(categories))
    else:
        summary = {
            "layer": layer,
            "categories": sorted(categories),
            "n_items_extracted": 0,
            "n_items_skipped": 0,
            "items_failed": [],
        }

    # Clean up stale temp files from prior interrupted runs.
    # np.savez silently appends .npz, so a tmp write to "foo.npz.tmp"
    # actually creates "foo.npz.tmp.npz" — clean those up too.
    for cat in categories:
        cat_dir = output_root / cat
        if cat_dir.exists():
            for tmp in list(cat_dir.glob("*.tmp")) + list(cat_dir.glob("*.tmp.npz")):
                tmp.unlink()
                log(f"  Removed stale tmp: {tmp.name}")

    # Validate hook fires before processing thousands of items
    log(f"  Validating hook on layer {layer}...")
    _hook_storage: dict[str, torch.Tensor] = {}
    _hook_handle = wrapper.get_layer(layer).register_forward_hook(
        make_all_token_hook(hidden_dim, _hook_storage)
    )
    _test_inp = tokenizer("test", return_tensors="pt").to(device)
    with torch.no_grad():
        wrapper.model(**_test_inp)
    _hook_handle.remove()
    if "hidden" not in _hook_storage:
        raise RuntimeError(
            f"Hook validation FAILED: layer {layer} hook did not fire. "
            "Check ModelWrapper.get_layer() for this architecture."
        )
    h_test = _hook_storage["hidden"]
    if h_test.shape[-1] != hidden_dim:
        raise RuntimeError(
            f"Hook validation FAILED: expected hidden_dim={hidden_dim}, "
            f"got {h_test.shape[-1]}"
        )
    del _hook_storage, _hook_handle, _test_inp, h_test
    log("  Hook validation passed")

    for cat in categories:
        cat_dir = output_root / cat
        cat_dir.mkdir(parents=True, exist_ok=True)

        try:
            stimuli = load_stimuli(run_dir, cat)
        except FileNotFoundError:
            log(f"  {cat}: stimuli file missing; skipping")
            continue
        stimuli_by_idx = {s["item_idx"]: s for s in stimuli}

        cat_ambig = meta_df[
            (meta_df["category"] == cat)
            & (meta_df["context_condition"] == "ambig")
        ]

        n_cat_extracted = 0
        n_cat_skipped = 0
        n_cat_corrupt = 0
        pbar = progress_bar(
            cat_ambig.iterrows(), total=len(cat_ambig),
            desc=f"  2a extract {cat}", unit="items",
        )
        for _, meta_row in pbar:
            item_idx = int(meta_row["item_idx"])
            out_path = cat_dir / f"item_{item_idx:06d}.npz"

            if out_path.exists():
                if _validate_npz(out_path, hidden_dim):
                    summary["n_items_skipped"] += 1
                    n_cat_skipped += 1
                    pbar.set_postfix_str(
                        f"ext={n_cat_extracted} skip={n_cat_skipped}"
                    )
                    continue
                else:
                    # Corrupt file from prior crash — remove and re-extract
                    out_path.unlink()
                    n_cat_corrupt += 1
                    log(f"    Removed corrupt {out_path.name}")

            stim = stimuli_by_idx.get(item_idx)
            if stim is None:
                summary["items_failed"].append({
                    "cat": cat, "item_idx": item_idx, "reason": "no_stimulus",
                })
                continue

            prompt = stim["prompt"]

            try:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                input_ids = inputs["input_ids"][0].cpu().tolist()
                tokens = tokenizer.convert_ids_to_tokens(input_ids)

                storage: dict[str, torch.Tensor] = {}
                layer_module = wrapper.get_layer(layer)
                handle = layer_module.register_forward_hook(
                    make_all_token_hook(hidden_dim, storage)
                )

                with torch.no_grad():
                    _ = wrapper.model(**inputs)

                handle.remove()

                if "hidden" not in storage:
                    summary["items_failed"].append({
                        "cat": cat, "item_idx": item_idx,
                        "reason": "hook_did_not_fire",
                    })
                    continue

                hidden = storage["hidden"].numpy().astype(np.float16)

                # Shape sanity check
                assert hidden.ndim == 2 and hidden.shape[1] == hidden_dim, (
                    f"Bad hidden shape {hidden.shape}, expected (seq, {hidden_dim})"
                )

                # Atomic write — use .tmp (no .npz in name) so np.savez
                # appends .npz → file is "item_NNNNNN.tmp.npz", then
                # rename to final .npz path.
                tmp_stem = out_path.with_suffix(".tmp")
                np.savez(
                    tmp_stem,
                    hidden_all_tokens=hidden,
                    tokens=np.array(tokens),
                    item_idx=np.array([item_idx], dtype=np.int32),
                    seq_len=np.array([len(tokens)], dtype=np.int32),
                )
                # np.savez appends .npz → actual file is tmp_stem.with_suffix(".tmp.npz")
                tmp_actual = tmp_stem.with_suffix(".tmp.npz")
                tmp_actual.rename(out_path)

                summary["n_items_extracted"] += 1
                n_cat_extracted += 1
                pbar.set_postfix_str(
                    f"ext={n_cat_extracted} skip={n_cat_skipped}"
                )

            except Exception as e:
                summary["items_failed"].append({
                    "cat": cat, "item_idx": item_idx,
                    "reason": str(e)[:200],
                })
                continue

        pbar.close()
        if n_cat_corrupt > 0:
            log(f"    {cat}: removed {n_cat_corrupt} corrupt files")

        # Save summary incrementally after each category
        atomic_save_json(summary, summary_path)

    log(f"Phase 2a complete: extracted={summary['n_items_extracted']}, "
        f"skipped={summary['n_items_skipped']}, "
        f"failed={len(summary['items_failed'])}")


# ---------------------------------------------------------------------------
# Phase 2b: Alias detection and scoping
# ---------------------------------------------------------------------------

def detect_alias_clusters(
    cosines_df: pd.DataFrame,
    alias_threshold: float,
    excluded_categories: set[str],
) -> dict:
    """Per category, find connected components where cosine >= threshold.

    Each component is a cluster; the representative is the subgroup with the
    largest n_targeting.  Returns one dict entry per non-excluded category.
    """
    result: dict = {}

    categories = sorted(cosines_df["category"].unique())

    for cat in categories:
        if cat in excluded_categories:
            continue

        cat_df = cosines_df[cosines_df["category"] == cat]
        if cat_df.empty:
            continue

        subgroups = sorted(cat_df["subgroup_a"].unique())

        # Build undirected edges at or above threshold (off-diagonal only)
        edges: dict[str, set[str]] = defaultdict(set)
        for _, row in cat_df.iterrows():
            if row["subgroup_a"] == row["subgroup_b"]:
                continue
            if row["cosine_normed"] >= alias_threshold:
                a = row["subgroup_a"]
                b = row["subgroup_b"]
                edges[a].add(b)
                edges[b].add(a)

        # Connected components via BFS
        visited: set[str] = set()
        clusters_raw: list[list[str]] = []
        for sub in subgroups:
            if sub in visited:
                continue
            component: set[str] = set()
            stack = [sub]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)
                stack.extend(edges[node] - visited)
            clusters_raw.append(sorted(component))

        # n_targeting per subgroup (from diagonal / self-row entries)
        n_targeting_by_sub: dict[str, int] = {}
        for _, row in cat_df.iterrows():
            if row["subgroup_a"] == row["subgroup_b"]:
                n_targeting_by_sub[row["subgroup_a"]] = int(row["n_a"])

        # Choose representative = largest n_targeting in each cluster
        clusters: list[dict] = []
        scoped: list[str] = []
        dropped: list[str] = []
        for i, members in enumerate(clusters_raw):
            rep = max(members, key=lambda s: n_targeting_by_sub.get(s, 0))
            clusters.append({
                "cluster_id": i,
                "members": members,
                "representative": rep,
                "n_targeting_rep": n_targeting_by_sub.get(rep, 0),
            })
            scoped.append(rep)
            dropped.extend([m for m in members if m != rep])

        result[cat] = {
            "clusters": clusters,
            "scoped_subgroups": scoped,
            "dropped_subgroups": dropped,
            "n_original": len(subgroups),
            "n_scoped": len(scoped),
        }

        log(f"  {cat}: {len(subgroups)} original -> {len(scoped)} scoped "
            f"({len(dropped)} aliased away)")
        for cluster in clusters:
            if len(cluster["members"]) > 1:
                log(f"    Cluster: {cluster['members']} "
                    f"-> keep '{cluster['representative']}'")

    return result


# ---------------------------------------------------------------------------
# Phase 2c: Method A — Identity features
# ---------------------------------------------------------------------------

def compute_identity_features(
    run_dir: Path,
    alias_clusters: dict,
    sae_wrapper: SAEWrapper,
    top_k: int,
    layer: int,
) -> pd.DataFrame:
    """Compute cosine of each SAE decoder column with DIM identity direction.

    Returns DataFrame with top-k features per scoped subgroup ranked by
    |cosine|.
    """
    # Decoder matrix — already L2-normalised rows from SAEWrapper
    W_dec_normed = sae_wrapper.get_decoder_matrix()  # (n_features, hidden_dim)
    n_features, hidden_dim = W_dec_normed.shape
    log(f"  SAE decoder: ({n_features}, {hidden_dim})")

    # Validate decoder norms (should be ~1.0 after normalisation)
    row_norms = np.linalg.norm(W_dec_normed, axis=1)
    if not np.allclose(row_norms, 1.0, atol=0.01):
        log(f"  WARNING: decoder rows not unit-normalised "
            f"(mean norm={row_norms.mean():.4f}, "
            f"range=[{row_norms.min():.4f}, {row_norms.max():.4f}])")

    rows: list[dict] = []
    identity_dir_base = (
        run_dir / "stage1_geometry" / "identity_directions" / f"L{layer:02d}"
    )

    if not identity_dir_base.exists():
        log(f"  WARNING: identity direction dir missing: {identity_dir_base}")
        return pd.DataFrame(rows)

    # Collect all scoped subgroups for progress
    all_subs = [
        (cat, sub)
        for cat, info in alias_clusters.items()
        for sub in info["scoped_subgroups"]
    ]

    for cat, sub in progress_bar(all_subs, desc="  2c identity features",
                                 unit="sub"):
        safe_sub = sub.replace("/", "_").replace(" ", "_")
        dir_path = identity_dir_base / f"{cat}_{safe_sub}.npz"

        if not dir_path.exists():
            log(f"    {cat}/{sub}: identity direction missing; skipping")
            continue

        data = np.load(dir_path, allow_pickle=True)
        direction_normed = data["direction_normed"].astype(np.float32)

        if direction_normed.shape != (hidden_dim,):
            log(f"    {cat}/{sub}: dim mismatch "
                f"({direction_normed.shape} vs {hidden_dim}); skipping")
            continue

        # Validate direction is approximately unit-normalised
        d_norm = float(np.linalg.norm(direction_normed))
        if abs(d_norm - 1.0) > 0.01:
            log(f"    {cat}/{sub}: direction norm={d_norm:.4f} "
                f"(expected ~1.0); re-normalising")
            direction_normed = direction_normed / max(d_norm, 1e-12)

        # Cosine of each (normalised) decoder column with direction
        cosines = W_dec_normed @ direction_normed  # (n_features,)
        abs_cosines = np.abs(cosines)

        # Sanity: cosines should be in [-1, 1]
        if abs_cosines.max() > 1.0 + 1e-5:
            log(f"    {cat}/{sub}: WARNING max |cosine|={abs_cosines.max():.6f} > 1")

        # Top-k by |cosine|
        top_idx = np.argsort(-abs_cosines)[:top_k]

        for rank, feat_idx in enumerate(top_idx):
            rows.append({
                "category": cat,
                "subgroup": sub,
                "feature_idx": int(feat_idx),
                "cosine": float(cosines[feat_idx]),
                "abs_cosine": float(abs_cosines[feat_idx]),
                "rank": rank + 1,
                "layer": layer,
                "method": "identity",
            })

    df = pd.DataFrame(rows)
    n_subs = df["subgroup"].nunique() if len(df) > 0 else 0
    log(f"  Method A complete: {len(df)} feature rows ({n_subs} subgroups)")
    return df


# ---------------------------------------------------------------------------
# Phase 2d: Method B — Bias-prediction features
# ---------------------------------------------------------------------------

def encode_all_items_max_pooled(
    run_dir: Path,
    categories: list[str],
    layer: int,
    sae_wrapper: SAEWrapper,
    device: str,
) -> dict[int, np.ndarray]:
    """Max-pool SAE-encoded per-token activations for each ambig item.

    Returns {item_idx: z_i (n_features,) float32}.
    """
    result: dict[int, np.ndarray] = {}
    cache_base = (
        run_dir / "A_extraction" / "activations_all_tokens" / f"L{layer:02d}"
    )

    expected_hdim = sae_wrapper.hidden_dim
    n_features = sae_wrapper.n_features
    n_load_errors = 0

    for cat in categories:
        cat_dir = cache_base / cat
        if not cat_dir.exists():
            log(f"    {cat}: no activations cache; skipping")
            continue

        item_files = sorted(
            p for p in cat_dir.glob("item_*.npz")
            if ".tmp" not in p.name
        )
        if not item_files:
            log(f"    {cat}: 0 cached files found; skipping")
            continue

        pbar = progress_bar(
            item_files, desc=f"    2d encode {cat}", unit="items",
        )
        for item_path in pbar:
            item_idx = int(item_path.stem.split("_")[1])

            try:
                data = np.load(item_path, allow_pickle=True)
                hidden = data["hidden_all_tokens"]  # (seq_len, hidden_dim) f16
            except Exception as e:
                n_load_errors += 1
                if n_load_errors <= 5:
                    log(f"      Failed to load {item_path.name}: {e}")
                continue

            if hidden.ndim != 2 or hidden.shape[1] != expected_hdim:
                n_load_errors += 1
                if n_load_errors <= 5:
                    log(f"      Bad shape {hidden.shape} in {item_path.name}, "
                        f"expected (*, {expected_hdim})")
                continue

            hidden_gpu = torch.from_numpy(
                hidden.astype(np.float32)
            ).to(device)

            with torch.no_grad():
                # SAE encode: (seq_len, hidden_dim) -> (seq_len, n_features)
                z_all = sae_wrapper.encode(hidden_gpu)
                assert z_all.shape == (hidden.shape[0], n_features), (
                    f"SAE encode shape mismatch: {z_all.shape} vs "
                    f"expected ({hidden.shape[0]}, {n_features})"
                )
                # Max pool across tokens
                z_max = z_all.max(dim=0).values  # (n_features,)
                # Cast to float32 before .numpy() — bfloat16 not supported by numpy
                z_max_cpu = z_max.float().cpu().numpy()

            del hidden_gpu, z_all, z_max

            result[item_idx] = z_max_cpu

        pbar.close()

    if n_load_errors > 0:
        log(f"  WARNING: {n_load_errors} items failed to load or had bad shape")
    log(f"  Max-pool encoding complete: {len(result)} items")
    return result


def compute_bias_prediction_features(
    z_max: dict[int, np.ndarray],
    meta_df: pd.DataFrame,
    alias_clusters: dict,
    l1_c_values: list[float],
    n_cv_folds: int,
    min_n_stereo: int,
    min_n_non_stereo: int,
    seed: int,
    layer: int,
    checkpoint_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """L1-regularised logistic regression predicting biased output from SAE acts.

    For each scoped subgroup, trains on ambig items targeting S.  Label is
    model_answer_role == "stereotyped_target" (biased) vs other.

    If checkpoint_dir is set, saves per-subgroup results incrementally and
    skips subgroups already completed on a prior run.

    Returns (features_df, probe_summary_dict).
    """
    rows: list[dict] = []
    summary: dict = {}

    has_qidx = "question_index" in meta_df.columns
    if not has_qidx:
        log("  WARNING: metadata has no 'question_index' column — "
            "GroupKFold will treat each item as its own group "
            "(equivalent to leave-one-out, which has poor variance)")

    # Build flat list of (cat, sub) for progress tracking
    all_subs = [
        (cat, sub)
        for cat, info in alias_clusters.items()
        for sub in info["scoped_subgroups"]
    ]

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    pbar = progress_bar(all_subs, desc="  2d L1 probes", unit="sub")
    for cat, sub in pbar:
        pbar.set_postfix_str(f"{cat}/{sub}")
        sub_key = f"{cat}/{sub}"

        # --- Per-subgroup checkpoint: skip if already done ---
        if checkpoint_dir is not None:
            safe_name = f"{cat}_{sub.replace('/', '_').replace(' ', '_')}"
            ckpt_path = checkpoint_dir / f"probe_{safe_name}.json"
            if ckpt_path.exists():
                try:
                    with open(ckpt_path) as f:
                        ckpt = json.load(f)
                    summary[sub_key] = ckpt["summary"]
                    rows.extend(ckpt["rows"])
                    continue
                except (json.JSONDecodeError, KeyError):
                    ckpt_path.unlink()  # corrupt checkpoint

        # Items targeting S (ambig only)
        cat_ambig = meta_df[
            (meta_df["category"] == cat)
            & (meta_df["context_condition"] == "ambig")
        ]
        targeting = cat_ambig[
            cat_ambig["stereotyped_groups"].apply(lambda gs: sub in gs)
        ]

        if len(targeting) == 0:
            continue

        # Build X, y, groups from items with cached max-pool activations
        X_rows: list[np.ndarray] = []
        y_rows: list[int] = []
        groups: list[int] = []

        for _, row in targeting.iterrows():
            item_idx = int(row["item_idx"])
            if item_idx not in z_max:
                continue
            X_rows.append(z_max[item_idx])
            y_val = (
                1 if row["model_answer_role"] == "stereotyped_target" else 0
            )
            y_rows.append(y_val)
            if has_qidx:
                groups.append(int(row["question_index"]))

        if len(X_rows) == 0:
            continue

        X = np.stack(X_rows)
        y = np.array(y_rows)
        g = np.array(groups) if has_qidx else np.arange(len(y))

        n_stereo = int((y == 1).sum())
        n_non_stereo = int((y == 0).sum())

        if n_stereo < min_n_stereo or n_non_stereo < min_n_non_stereo:
            log(f"    {sub_key}: insufficient "
                f"(stereo={n_stereo}, non={n_non_stereo}); skipping")
            summary[sub_key] = {
                "status": "insufficient_items",
                "n_stereo": n_stereo,
                "n_non_stereo": n_non_stereo,
            }
            continue

        n_unique_groups = len(np.unique(g))
        effective_folds = min(n_cv_folds, n_unique_groups)
        if effective_folds < 2:
            log(f"    {sub_key}: too few groups ({n_unique_groups})")
            summary[sub_key] = {
                "status": "insufficient_groups",
                "n_unique_groups": n_unique_groups,
            }
            continue

        # Hyperparameter search
        gkf = GroupKFold(n_splits=effective_folds)
        cv_results: dict[float, dict] = {}

        for c in l1_c_values:
            fold_accs: list[float] = []
            for train_idx, test_idx in gkf.split(X, y, g):
                if (len(np.unique(y[train_idx])) < 2
                        or len(np.unique(y[test_idx])) < 2):
                    continue
                try:
                    clf = LogisticRegression(
                        penalty="l1",
                        C=c,
                        solver="liblinear",
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=seed,
                    )
                    clf.fit(X[train_idx], y[train_idx])
                    pred = clf.predict(X[test_idx])
                    fold_accs.append(
                        balanced_accuracy_score(y[test_idx], pred)
                    )
                except Exception as e:
                    log(f"      {sub_key} C={c}: fold failed: {e}")
                    continue

            cv_results[c] = {
                "mean_acc": float(np.mean(fold_accs)) if fold_accs else 0.0,
                "n_folds": len(fold_accs),
            }

        # Pick best C
        if all(r["n_folds"] == 0 for r in cv_results.values()):
            log(f"    {sub_key}: all folds failed")
            summary[sub_key] = {
                "status": "cv_failed",
                "cv_results": {str(c): r for c, r in cv_results.items()},
            }
            continue

        best_c = max(cv_results, key=lambda c: cv_results[c]["mean_acc"])
        best_acc = cv_results[best_c]["mean_acc"]

        # Refit on all data at best C
        clf_final = LogisticRegression(
            penalty="l1",
            C=best_c,
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        )
        clf_final.fit(X, y)

        coefs = clf_final.coef_.flatten()
        nonzero_mask = np.abs(coefs) > 1e-10
        nonzero_idx = np.where(nonzero_mask)[0]
        nonzero_coefs = coefs[nonzero_idx]

        # Sort by |coefficient|
        sort_order = np.argsort(-np.abs(nonzero_coefs))

        log(f"    {sub_key}: best_c={best_c}, cv_acc={best_acc:.3f}, "
            f"n_nonzero={len(nonzero_idx)}, "
            f"n_stereo={n_stereo}, n_non_stereo={n_non_stereo}")

        sub_rows: list[dict] = []
        for rank, order_idx in enumerate(sort_order):
            feat_idx = int(nonzero_idx[order_idx])
            coef = float(nonzero_coefs[order_idx])
            sub_rows.append({
                "category": cat,
                "subgroup": sub,
                "feature_idx": feat_idx,
                "l1_coefficient": coef,
                "abs_coefficient": abs(coef),
                "rank": rank + 1,
                "best_c": best_c,
                "cv_accuracy": best_acc,
                "n_stereo": n_stereo,
                "n_non_stereo": n_non_stereo,
                "layer": layer,
                "method": "bias_prediction",
            })

        sub_summary = {
            "status": "ok",
            "best_c": best_c,
            "cv_accuracy": best_acc,
            "cv_results": {str(c): r for c, r in cv_results.items()},
            "n_nonzero": int(len(nonzero_idx)),
            "n_stereo": n_stereo,
            "n_non_stereo": n_non_stereo,
            "n_unique_groups": n_unique_groups,
            "effective_folds": effective_folds,
        }

        rows.extend(sub_rows)
        summary[sub_key] = sub_summary

        # Save per-subgroup checkpoint
        if checkpoint_dir is not None:
            atomic_save_json(
                {"summary": sub_summary, "rows": sub_rows}, ckpt_path,
            )

    pbar.close()

    df = pd.DataFrame(rows)
    n_subs = df["subgroup"].nunique() if len(df) > 0 else 0
    log(f"  Method B complete: {len(df)} feature rows "
        f"({n_subs} subgroups with nonzero features)")
    return df, summary


# ---------------------------------------------------------------------------
# Phase 2e: Method overlap
# ---------------------------------------------------------------------------

def compute_method_overlap(
    identity_df: pd.DataFrame,
    bias_df: pd.DataFrame,
    top_k_identity: int,
) -> pd.DataFrame:
    """Per subgroup, compute Jaccard between Method A top-k and Method B nonzero."""
    rows: list[dict] = []

    subgroups_a = set(
        zip(identity_df["category"], identity_df["subgroup"])
    ) if len(identity_df) > 0 else set()
    subgroups_b = set(
        zip(bias_df["category"], bias_df["subgroup"])
    ) if len(bias_df) > 0 else set()
    all_subs = subgroups_a | subgroups_b

    for cat, sub in sorted(all_subs):
        feat_a = set(identity_df[
            (identity_df["category"] == cat)
            & (identity_df["subgroup"] == sub)
            & (identity_df["rank"] <= top_k_identity)
        ]["feature_idx"].values) if len(identity_df) > 0 else set()

        feat_b = set(bias_df[
            (bias_df["category"] == cat)
            & (bias_df["subgroup"] == sub)
        ]["feature_idx"].values) if len(bias_df) > 0 else set()

        intersection = feat_a & feat_b
        union = feat_a | feat_b

        jaccard = len(intersection) / len(union) if len(union) > 0 else 0.0

        rows.append({
            "category": cat,
            "subgroup": sub,
            "n_identity_features": len(feat_a),
            "n_bias_prediction_features": len(feat_b),
            "n_intersection": len(intersection),
            "n_union": len(union),
            "jaccard": jaccard,
            "intersection_feature_idxs": sorted(intersection),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Phase 2f: Figures
# ---------------------------------------------------------------------------

def fig_identity_feature_cosines(
    identity_df: pd.DataFrame, category: str, output_dir: Path, top_n: int = 20,
) -> None:
    cat_df = identity_df[identity_df["category"] == category]
    if cat_df.empty:
        return

    subgroups = sorted(cat_df["subgroup"].unique())
    n_subs = len(subgroups)
    n_cols = min(3, n_subs)
    n_rows = (n_subs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False,
    )

    for i, sub in enumerate(subgroups):
        r, c = i // n_cols, i % n_cols
        ax = axes[r, c]

        sub_df = cat_df[cat_df["subgroup"] == sub].head(top_n)
        if sub_df.empty:
            ax.set_visible(False)
            continue

        colors = [
            WONG["blue"] if cos > 0 else WONG["vermillion"]
            for cos in sub_df["cosine"]
        ]

        y_pos = np.arange(len(sub_df))
        ax.barh(y_pos, sub_df["cosine"], color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            [f"F{int(f)}" for f in sub_df["feature_idx"]], fontsize=7,
        )
        ax.set_xlabel("cosine with identity direction", fontsize=9)
        ax.set_title(sub, fontsize=10)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.invert_yaxis()

    for j in range(n_subs, n_rows * n_cols):
        r, c = j // n_cols, j % n_cols
        axes[r, c].set_visible(False)

    fig.suptitle(
        f"Top identity features (Method A) — {category}", fontsize=11,
    )
    fig.tight_layout()
    _save_fig(fig, output_dir / f"fig_identity_feature_cosines_{category}.png")


def fig_bias_prediction_coefficients(
    bias_df: pd.DataFrame, category: str, output_dir: Path, top_n: int = 20,
) -> None:
    cat_df = bias_df[bias_df["category"] == category]
    if cat_df.empty:
        return

    subgroups = sorted(cat_df["subgroup"].unique())
    n_subs = len(subgroups)
    n_cols = min(3, n_subs)
    n_rows = (n_subs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False,
    )

    for i, sub in enumerate(subgroups):
        r, c = i // n_cols, i % n_cols
        ax = axes[r, c]

        sub_df = cat_df[cat_df["subgroup"] == sub].head(top_n)
        if sub_df.empty:
            ax.set_visible(False)
            continue

        colors = [
            WONG["blue"] if coef > 0 else WONG["vermillion"]
            for coef in sub_df["l1_coefficient"]
        ]

        y_pos = np.arange(len(sub_df))
        ax.barh(y_pos, sub_df["l1_coefficient"], color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            [f"F{int(f)}" for f in sub_df["feature_idx"]], fontsize=7,
        )
        ax.set_xlabel("L1 coefficient", fontsize=9)

        best_c = sub_df["best_c"].iloc[0]
        cv_acc = sub_df["cv_accuracy"].iloc[0]
        ax.set_title(f"{sub}\nC={best_c}, CV acc={cv_acc:.3f}", fontsize=9)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.invert_yaxis()

    for j in range(n_subs, n_rows * n_cols):
        r, c = j // n_cols, j % n_cols
        axes[r, c].set_visible(False)

    fig.suptitle(
        f"Top bias-prediction features (Method B) — {category}", fontsize=11,
    )
    fig.tight_layout()
    _save_fig(
        fig, output_dir / f"fig_bias_prediction_coefficients_{category}.png",
    )


def fig_method_overlap_heatmap(
    overlap_df: pd.DataFrame, output_dir: Path,
) -> None:
    if overlap_df.empty:
        return

    df = overlap_df.sort_values(["category", "subgroup"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.4), 4))
    x = np.arange(len(df))
    ax.bar(x, df["jaccard"], color=WONG["purple"], alpha=0.8)

    ax.set_xticks(x)
    labels = [
        f"{row['category']}/{row['subgroup']}" for _, row in df.iterrows()
    ]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Jaccard overlap")
    ax.set_title(
        "Method A (identity) vs Method B (bias-prediction) feature overlap",
    )
    ax.set_ylim(0, max(0.5, df["jaccard"].max() * 1.2))

    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(
            i, row["jaccard"] + 0.005,
            f"{row['n_intersection']}/{row['n_union']}",
            ha="center", fontsize=6,
        )

    plt.tight_layout()
    _save_fig(fig, output_dir / "fig_method_overlap_heatmap.png")


def fig_alias_cluster_summary(
    alias_clusters: dict, output_dir: Path,
) -> None:
    rows: list[dict] = []
    for cat, info in alias_clusters.items():
        for cluster in info["clusters"]:
            rows.append({
                "category": cat,
                "cluster_size": len(cluster["members"]),
                "representative": cluster["representative"],
                "members": ", ".join(cluster["members"]),
            })

    if not rows:
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["category", "cluster_size"], ascending=[True, False],
    ).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.25)))

    colors = [
        WONG["vermillion"] if s > 1 else WONG["blue"]
        for s in df["cluster_size"]
    ]

    y_pos = np.arange(len(df))
    ax.barh(y_pos, df["cluster_size"], color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    labels = [
        f"{row['category']}: {row['representative']}"
        for _, row in df.iterrows()
    ]
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Cluster size (subgroups collapsed)")
    ax.set_title("Representational alias clusters (cosine >= alias_threshold)")
    ax.invert_yaxis()

    for i, (_, row) in enumerate(df.iterrows()):
        if row["cluster_size"] > 1:
            ax.text(
                row["cluster_size"] + 0.1, i, row["members"],
                va="center", fontsize=6,
            )

    plt.tight_layout()
    _save_fig(fig, output_dir / "fig_alias_cluster_summary.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)

    output_dir = run_dir / "stage2_features"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    checkpoint_dir = output_dir / "_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    t_start = time.time()
    log("Stage 2: SAE Feature Selection")
    log(f"Run dir: {run_dir}, layer: {args.layer}")

    # Config
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found at {config_path}")
    with open(config_path) as f:
        config = json.load(f)

    device = args.device

    # === Phase 2b: Alias detection and scoping ===
    log("\n=== Phase 2b: Alias detection ===")
    cosines_path = run_dir / "stage1_geometry" / "cosines.parquet"
    if not cosines_path.exists():
        raise FileNotFoundError(
            f"Stage 1 cosines not found: {cosines_path}. "
            "Run stage1_geometry.py first."
        )
    cosines_df = pd.read_parquet(cosines_path)
    cosines_df = cosines_df[cosines_df["layer"] == args.layer]

    if cosines_df.empty:
        raise ValueError(
            f"No cosine data for layer {args.layer} in {cosines_path}. "
            f"Available layers: "
            f"{sorted(pd.read_parquet(cosines_path)['layer'].unique())}"
        )

    excluded = set(
        c.strip() for c in args.excluded_categories.split(",") if c.strip()
    )

    if args.categories:
        requested = set(c.strip() for c in args.categories.split(","))
        categories_all = sorted(cosines_df["category"].unique())
        categories = sorted(
            (requested & set(categories_all)) - excluded
        )
        missing = requested - set(categories_all) - excluded
        if missing:
            log(f"  WARNING: requested categories not in cosines data: {missing}")
    else:
        categories_all = sorted(cosines_df["category"].unique())
        categories = [c for c in categories_all if c not in excluded]

    if not categories:
        raise ValueError(
            "No categories in scope after filtering. "
            f"Available: {categories_all}, excluded: {sorted(excluded)}"
        )

    log(f"Categories in scope: {categories}")
    log(f"Excluded: {sorted(excluded)}")

    alias_clusters = detect_alias_clusters(
        cosines_df, args.alias_threshold, excluded,
    )
    # Filter to requested categories
    alias_clusters = {
        k: v for k, v in alias_clusters.items() if k in categories
    }

    atomic_save_json(alias_clusters, output_dir / "alias_clusters.json")
    scoped_simple = {
        cat: info["scoped_subgroups"]
        for cat, info in alias_clusters.items()
    }
    atomic_save_json(scoped_simple, output_dir / "scoped_subgroups.json")

    n_total_scoped = sum(
        len(v["scoped_subgroups"]) for v in alias_clusters.values()
    )
    log(f"Total scoped subgroups: {n_total_scoped}")

    if n_total_scoped == 0:
        raise ValueError("No scoped subgroups remain after alias detection")

    # === Phase 2a: Per-token extraction ===
    if not args.skip_extraction:
        log("\n=== Phase 2a: Per-token hidden state extraction ===")

        cache_root = (
            run_dir / "A_extraction" / "activations_all_tokens"
            / f"L{args.layer:02d}"
        )
        summary_path = cache_root / "_extraction_summary.json"

        needs_extraction = True
        if summary_path.exists():
            with open(summary_path) as f:
                existing_summary = json.load(f)
            existing_cats = set(existing_summary.get("categories", []))
            if set(categories).issubset(existing_cats):
                # Verify actual files exist — summary may be stale from
                # a crashed run that left only ghost .tmp files
                n_real = sum(
                    sum(1 for p in (cache_root / cat).glob("item_*.npz")
                        if ".tmp" not in p.name)
                    for cat in categories
                    if (cache_root / cat).exists()
                )
                if n_real > 0:
                    log(f"Extraction cache valid ({n_real} files); skipping")
                    needs_extraction = False
                else:
                    log("Extraction summary exists but no valid .npz files "
                        "found — re-extracting")

        if needs_extraction:
            log("Loading model for extraction...")
            wrapper = ModelWrapper.from_pretrained(
                config["model_path"], device=device,
            )

            extract_all_token_activations(
                run_dir, categories, args.layer, wrapper, device,
            )

            # Free model memory — only SAE needed after this
            del wrapper
            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache()

    # Verify extraction produced files before proceeding
    cache_root = (
        run_dir / "A_extraction" / "activations_all_tokens"
        / f"L{args.layer:02d}"
    )
    n_cached_files = sum(
        sum(1 for p in (cache_root / cat).glob("item_*.npz")
            if ".tmp" not in p.name)
        for cat in categories
        if (cache_root / cat).exists()
    )
    if n_cached_files == 0:
        raise FileNotFoundError(
            f"No cached activation files found under {cache_root}. "
            "Run without --skip_extraction first."
        )
    log(f"Activation cache: {n_cached_files} files across {len(categories)} categories")

    # === Load SAE ===
    log("\n=== Loading SAE ===")
    sae_source = config.get("sae_source", config.get("sae_repo_id", ""))
    sae_expansion = config.get("sae_expansion", 32)
    sae_wrapper = SAEWrapper(
        sae_source,
        layer=args.layer,
        expansion=sae_expansion,
        device=device,
    )
    log(f"SAE loaded: n_features={sae_wrapper.n_features}")

    # === Phase 2c: Method A ===
    phase_2c_done = checkpoint_dir / "_phase_2c_done.json"
    identity_parquet = output_dir / "identity_features.parquet"

    if phase_2c_done.exists() and identity_parquet.exists():
        log("\n=== Phase 2c: LOADED from checkpoint ===")
        identity_df = pd.read_parquet(identity_parquet)
        log(f"  {len(identity_df)} rows, "
            f"{identity_df['subgroup'].nunique() if len(identity_df) > 0 else 0} subgroups")
    else:
        log("\n=== Phase 2c: Method A (identity features) ===")
        identity_df = compute_identity_features(
            run_dir, alias_clusters, sae_wrapper, args.top_k_identity,
            args.layer,
        )
        identity_df.to_parquet(identity_parquet, index=False)
        atomic_save_json(
            {"status": "done", "n_rows": len(identity_df)}, phase_2c_done,
        )

    # === Load metadata ===
    meta_df = load_metadata(run_dir)

    # === Phase 2d: Method B ===
    phase_2d_done = checkpoint_dir / "_phase_2d_done.json"
    bias_parquet = output_dir / "bias_prediction_features.parquet"
    probe_json = output_dir / "probe_summary.json"
    probe_ckpt_dir = checkpoint_dir / "probes"

    if (phase_2d_done.exists() and bias_parquet.exists()
            and probe_json.exists()):
        log("\n=== Phase 2d: LOADED from checkpoint ===")
        bias_df = pd.read_parquet(bias_parquet)
        with open(probe_json) as f:
            probe_summary = json.load(f)
        log(f"  {len(bias_df)} rows, "
            f"{bias_df['subgroup'].nunique() if len(bias_df) > 0 else 0} subgroups")
    else:
        log("\n=== Phase 2d: Method B (bias-prediction features) ===")
        log("  Encoding all items with max-pool SAE...")
        z_max = encode_all_items_max_pooled(
            run_dir, categories, args.layer, sae_wrapper, device,
        )

        if not z_max:
            log("  WARNING: no items encoded — Method B will produce empty results")

        log("  Running L1 probes...")
        l1_c_values = [float(c) for c in args.l1_c_values.split(",")]
        bias_df, probe_summary = compute_bias_prediction_features(
            z_max, meta_df, alias_clusters, l1_c_values, args.n_cv_folds,
            args.min_n_stereo, args.min_n_non_stereo, args.random_seed,
            args.layer, checkpoint_dir=probe_ckpt_dir,
        )
        bias_df.to_parquet(bias_parquet, index=False)
        atomic_save_json(probe_summary, probe_json)
        atomic_save_json(
            {"status": "done", "n_rows": len(bias_df)}, phase_2d_done,
        )

        del z_max

    # === Phase 2e: Method overlap ===
    log("\n=== Phase 2e: Method overlap ===")
    overlap_df = compute_method_overlap(
        identity_df, bias_df, args.top_k_identity,
    )
    overlap_df.to_parquet(
        output_dir / "method_overlap.parquet", index=False,
    )

    log("Overlap summary:")
    for _, row in overlap_df.iterrows():
        log(f"  {row['category']}/{row['subgroup']}: "
            f"|A|={row['n_identity_features']}, "
            f"|B|={row['n_bias_prediction_features']}, "
            f"|A^B|={row['n_intersection']}, "
            f"Jaccard={row['jaccard']:.3f}")

    # === Phase 2f: Figures ===
    if not args.skip_figures:
        log("\n=== Phase 2f: Figures ===")
        fig_dir = output_dir / "figures"

        for cat in categories:
            fig_identity_feature_cosines(identity_df, cat, fig_dir)
            fig_bias_prediction_coefficients(bias_df, cat, fig_dir)

        fig_method_overlap_heatmap(overlap_df, fig_dir)
        fig_alias_cluster_summary(alias_clusters, fig_dir)

    elapsed = time.time() - t_start
    log(f"\nStage 2 complete in {elapsed:.1f}s")
    log(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
