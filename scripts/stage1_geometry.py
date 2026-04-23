"""Stage 1: Representation Geometry.

Characterise subgroup identity directions in the residual stream at a single
chosen SAE layer (default: 14).  Produces pairwise cosines, Gram-Schmidt
decomposition, and identity probe accuracies.

No model loading, no SAE, no forward passes.  All matrix math on cached
activations from Phase A.

Usage:
    python scripts/stage1_geometry.py --run_dir runs/llama-3.1-8b_2026-04-22/
    python scripts/stage1_geometry.py --run_dir ... --categories so --skip_sensitivity
    python scripts/stage1_geometry.py --run_dir ... --layer 14 --skip_figures
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

WONG = {
    "orange": "#E69F00", "sky_blue": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00",
    "purple": "#CC79A7", "black": "#000000",
}
SUBGROUP_COLORS = [
    WONG["blue"], WONG["orange"], WONG["green"], WONG["purple"],
    WONG["vermillion"], WONG["sky_blue"], WONG["yellow"],
]
DPI = 200


def log(msg: str) -> None:
    print(f"[stage1] {msg}", flush=True)


def _save_fig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1: Representation Geometry")
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--layer", type=int, default=14)
    p.add_argument("--sensitivity_layers", type=str, default="12,16")
    p.add_argument("--skip_sensitivity", action="store_true")
    p.add_argument("--categories", type=str, default=None)
    p.add_argument("--min_n_per_group", type=int, default=10)
    p.add_argument("--n_bootstrap", type=int, default=1000)
    p.add_argument("--n_cv_folds", type=int, default=5)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--skip_figures", action="store_true")
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


def load_layer_hidden_states(
    run_dir: Path, categories: list[str], layer: int,
) -> dict[int, np.ndarray]:
    """Load last-token raw hidden states at the given layer for all items."""
    result: dict[int, np.ndarray] = {}
    for cat in categories:
        cat_dir = run_dir / "A_extraction" / "activations" / cat
        if not cat_dir.exists():
            log(f"  WARNING: missing activations dir: {cat_dir}")
            continue
        for item_path in sorted(cat_dir.glob("item_*.npz")):
            item_idx = int(item_path.stem.split("_")[1])
            try:
                data = np.load(item_path, allow_pickle=True)
                # Phase A stores unit-normalised hidden_states + raw_norms
                hs_normed = data["hidden_states"][layer].astype(np.float32)
                raw_norm = float(data["hidden_states_raw_norms"][layer])
                result[item_idx] = hs_normed * raw_norm
            except Exception as e:
                log(f"    Failed to load {item_path.name}: {e}")
    return result


# ---------------------------------------------------------------------------
# Subgroup catalog
# ---------------------------------------------------------------------------

def build_subgroup_catalog(
    meta_df: pd.DataFrame, categories: list[str], min_n: int,
) -> dict[str, dict]:
    catalog: dict[str, dict] = {}
    for cat in categories:
        ambig = meta_df[
            (meta_df["category"] == cat)
            & (meta_df["context_condition"] == "ambig")
        ]
        all_subs: set[str] = set()
        for gs in ambig["stereotyped_groups"]:
            all_subs.update(gs)

        for sub in sorted(all_subs):
            ga = ambig[ambig["stereotyped_groups"].apply(lambda gs: sub in gs)]
            gb = ambig[ambig["stereotyped_groups"].apply(
                lambda gs: (sub not in gs) and len(gs) > 0
            )]
            catalog[sub] = {
                "category": cat,
                "n_targeting": len(ga),
                "n_not_targeting": len(gb),
                "analyzable": len(ga) >= min_n and len(gb) >= min_n,
                "group_a_item_idxs": ga["item_idx"].tolist(),
                "group_b_item_idxs": gb["item_idx"].tolist(),
            }
    return catalog


# ---------------------------------------------------------------------------
# Identity directions (DIM)
# ---------------------------------------------------------------------------

def compute_identity_directions(
    hidden_states: dict[int, np.ndarray],
    catalog: dict[str, dict],
    layer: int,
) -> dict[tuple[str, str], dict]:
    directions: dict[tuple[str, str], dict] = {}
    for sub, entry in catalog.items():
        if not entry["analyzable"]:
            continue
        cat = entry["category"]
        a_vecs = [hidden_states[i] for i in entry["group_a_item_idxs"]
                  if i in hidden_states]
        b_vecs = [hidden_states[i] for i in entry["group_b_item_idxs"]
                  if i in hidden_states]
        if len(a_vecs) < 10 or len(b_vecs) < 10:
            log(f"  {cat}/{sub}: insufficient hidden states "
                f"(a={len(a_vecs)}, b={len(b_vecs)}); skip")
            continue
        mean_a = np.stack(a_vecs).mean(axis=0).astype(np.float32)
        mean_b = np.stack(b_vecs).mean(axis=0).astype(np.float32)
        raw = (mean_a - mean_b).astype(np.float32)
        norm = float(np.linalg.norm(raw))
        if norm < 1e-8:
            continue
        directions[(cat, sub)] = {
            "direction_raw": raw,
            "direction_normed": raw / norm,
            "norm_raw": norm,
            "mean_group_a": mean_a, "mean_group_b": mean_b,
            "n_a": len(a_vecs), "n_b": len(b_vecs),
        }
    log(f"  Layer {layer}: {len(directions)} identity directions computed")
    return directions


def save_identity_directions(
    directions: dict, output_dir: Path, layer: int,
) -> None:
    layer_dir = output_dir / "identity_directions" / f"L{layer:02d}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    for (cat, sub), info in directions.items():
        safe = sub.replace("/", "_").replace(" ", "_")
        np.savez(
            layer_dir / f"{cat}_{safe}.npz",
            direction_raw=info["direction_raw"],
            direction_normed=info["direction_normed"],
            norm_raw=np.float32(info["norm_raw"]),
            n_a=np.int32(info["n_a"]), n_b=np.int32(info["n_b"]),
            category=cat, subgroup=sub, layer=np.int32(layer),
        )


# ---------------------------------------------------------------------------
# Pairwise cosines
# ---------------------------------------------------------------------------

def compute_pairwise_cosines(
    directions: dict, layer: int,
) -> list[dict]:
    by_cat: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for (cat, sub), info in directions.items():
        by_cat[cat].append((sub, info))

    rows: list[dict] = []
    for cat, subs in by_cat.items():
        for i, (sa, ia) in enumerate(subs):
            for j, (sb, ib) in enumerate(subs):
                if i > j:
                    continue
                na, nb = ia["direction_normed"], ib["direction_normed"]
                cos = float(np.dot(na, nb))
                base = {
                    "category": cat, "cosine_raw": cos,
                    "cosine_normed": cos, "layer": layer,
                }
                rows.append({**base, "subgroup_a": sa, "subgroup_b": sb,
                             "n_a": ia["n_a"], "n_b": ib["n_a"]})
                if i != j:
                    rows.append({**base, "subgroup_a": sb, "subgroup_b": sa,
                                 "n_a": ib["n_a"], "n_b": ia["n_a"]})
    return rows


# ---------------------------------------------------------------------------
# Gram-Schmidt decomposition
# ---------------------------------------------------------------------------

def compute_gram_schmidt_decomposition(
    directions: dict, layer: int,
) -> list[dict]:
    by_cat: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for (cat, sub), info in directions.items():
        by_cat[cat].append((sub, info))

    rows: list[dict] = []
    for cat, subs in by_cat.items():
        if len(subs) < 2:
            for sub, info in subs:
                rows.append({
                    "category": cat, "subgroup": sub,
                    "identity_norm": info["norm_raw"],
                    "category_component_norm": None,
                    "subgroup_residual_norm": None,
                    "residual_ratio": None, "projection_scalar": None,
                    "layer": layer, "note": "single_subgroup",
                })
            continue

        raw_mat = np.stack([info["direction_raw"] for _, info in subs])
        centered = raw_mat - raw_mat.mean(axis=0)
        try:
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            top = Vt[0]
            top = top / (np.linalg.norm(top) + 1e-12)
        except np.linalg.LinAlgError:
            log(f"  SVD failed for {cat}; skipping decomposition")
            continue

        for sub, info in subs:
            d = info["direction_raw"]
            proj = float(np.dot(d, top))
            cat_comp = proj * top
            residual = d - cat_comp
            rows.append({
                "category": cat, "subgroup": sub,
                "identity_norm": info["norm_raw"],
                "category_component_norm": float(np.linalg.norm(cat_comp)),
                "subgroup_residual_norm": float(np.linalg.norm(residual)),
                "residual_ratio": float(np.linalg.norm(residual))
                    / (info["norm_raw"] + 1e-12),
                "projection_scalar": proj,
                "layer": layer, "note": "ok",
            })
    return rows


# ---------------------------------------------------------------------------
# Identity probes
# ---------------------------------------------------------------------------

def compute_identity_probes(
    hidden_states: dict[int, np.ndarray],
    meta_df: pd.DataFrame,
    catalog: dict[str, dict],
    layer: int,
    n_folds: int = 5,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    meta_idx = meta_df.set_index("item_idx")
    has_qidx = "question_index" in meta_df.columns
    rows: list[dict] = []

    for sub, entry in catalog.items():
        if not entry["analyzable"]:
            continue
        cat = entry["category"]

        X_list, y_list, groups = [], [], []
        for idx in entry["group_a_item_idxs"]:
            if idx in hidden_states and idx in meta_idx.index:
                X_list.append(hidden_states[idx])
                y_list.append(1)
                if has_qidx:
                    groups.append(int(meta_idx.loc[idx, "question_index"]))
        for idx in entry["group_b_item_idxs"]:
            if idx in hidden_states and idx in meta_idx.index:
                X_list.append(hidden_states[idx])
                y_list.append(0)
                if has_qidx:
                    groups.append(int(meta_idx.loc[idx, "question_index"]))

        if len(X_list) < 20:
            continue

        X = np.stack(X_list)
        y = np.array(y_list)

        # Use GroupKFold if question_index available, else StratifiedKFold
        if has_qidx and groups:
            g = np.array(groups)
            eff_folds = min(n_folds, len(np.unique(g)))
            if eff_folds < 2:
                continue
            cv = GroupKFold(n_splits=eff_folds)
            splits = list(cv.split(X, y, g))
        else:
            g = None
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                  random_state=seed)
            splits = list(cv.split(X, y))

        fold_accs: list[float] = []
        for tr, te in splits:
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            clf = LogisticRegression(
                penalty="l2", C=1.0, max_iter=1000,
                class_weight="balanced", random_state=seed,
            )
            clf.fit(X[tr], y[tr])
            fold_accs.append(balanced_accuracy_score(y[te], clf.predict(X[te])))

        if not fold_accs:
            continue
        mean_acc = float(np.mean(fold_accs))

        # Bootstrap CI
        boot_accs: list[float] = []
        for _ in range(n_bootstrap):
            bi = rng.choice(len(X), size=len(X), replace=True)
            Xb, yb = X[bi], y[bi]
            if len(np.unique(yb)) < 2:
                continue

            if g is not None:
                gb = g[bi]
                uq = np.unique(gb)
                rng.shuffle(uq)
                n_te = max(1, len(uq) // 5)
                te_groups = set(uq[:n_te])
                tr_m = ~np.isin(gb, list(te_groups))
                te_m = np.isin(gb, list(te_groups))
            else:
                # Simple 80/20 split
                n_te = max(5, len(Xb) // 5)
                perm = rng.permutation(len(Xb))
                te_m = np.zeros(len(Xb), dtype=bool)
                te_m[perm[:n_te]] = True
                tr_m = ~te_m

            if tr_m.sum() < 10 or te_m.sum() < 5:
                continue
            if len(np.unique(yb[tr_m])) < 2 or len(np.unique(yb[te_m])) < 2:
                continue
            try:
                clf = LogisticRegression(
                    penalty="l2", C=1.0, max_iter=1000,
                    class_weight="balanced", random_state=seed,
                )
                clf.fit(Xb[tr_m], yb[tr_m])
                boot_accs.append(
                    balanced_accuracy_score(yb[te_m], clf.predict(Xb[te_m]))
                )
            except Exception:
                continue

        ci_lo = float(np.percentile(boot_accs, 2.5)) if len(boot_accs) >= 100 else None
        ci_hi = float(np.percentile(boot_accs, 97.5)) if len(boot_accs) >= 100 else None

        rows.append({
            "category": cat, "subgroup": sub,
            "balanced_accuracy": mean_acc,
            "ci_low": ci_lo, "ci_high": ci_hi,
            "n_targeting": int((y == 1).sum()),
            "n_not_targeting": int((y == 0).sum()),
            "n_cv_folds": len(fold_accs),
            "n_bootstrap_success": len(boot_accs),
            "layer": layer,
            "cv_method": "GroupKFold" if has_qidx else "StratifiedKFold",
        })
    return rows


# ---------------------------------------------------------------------------
# Layer stability
# ---------------------------------------------------------------------------

def compute_layer_stability(
    cosines_df: pd.DataFrame, layers: list[int], categories: list[str],
) -> dict:
    result: dict = {"layers": layers, "per_category": {}}
    all_dist: list[float] = []

    for cat in categories:
        cat_df = cosines_df[cosines_df["category"] == cat]
        if cat_df.empty:
            continue
        subs = sorted(cat_df["subgroup_a"].unique())
        s2i = {s: i for i, s in enumerate(subs)}
        n = len(subs)
        mats: dict[int, np.ndarray] = {}
        for layer in layers:
            mat = np.full((n, n), np.nan)
            for _, r in cat_df[cat_df["layer"] == layer].iterrows():
                mat[s2i[r["subgroup_a"]], s2i[r["subgroup_b"]]] = r["cosine_normed"]
            mats[layer] = mat

        dists: dict[str, float] = {}
        for i, l1 in enumerate(layers):
            for l2 in layers[i + 1:]:
                v = ~(np.isnan(mats[l1]) | np.isnan(mats[l2]))
                if v.sum() == 0:
                    continue
                frob = float(np.sqrt(np.sum((mats[l1][v] - mats[l2][v]) ** 2)))
                dists[f"L{l1}_vs_L{l2}"] = round(frob, 4)
                all_dist.append(frob)

        result["per_category"][cat] = {
            "subgroups": subs, "pairwise_frobenius": dists,
        }
    result["overall_mean_distance"] = (
        round(float(np.mean(all_dist)), 4) if all_dist else None
    )
    return result


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def build_geometry_summary(
    cos_df: pd.DataFrame, decomp_df: pd.DataFrame, probes_df: pd.DataFrame,
    catalog: dict, layers: list[int], primary: int,
) -> dict:
    summary: dict = {
        "primary_layer": primary,
        "all_layers_analyzed": layers,
        "n_analyzable_subgroups": sum(1 for v in catalog.values() if v["analyzable"]),
        "n_total_subgroups": len(catalog),
        "per_category": {},
    }
    for cat in sorted(cos_df["category"].unique()):
        cc = cos_df[
            (cos_df["category"] == cat) & (cos_df["layer"] == primary)
            & (cos_df["subgroup_a"] != cos_df["subgroup_b"])
        ]
        cd = decomp_df[
            (decomp_df["category"] == cat) & (decomp_df["layer"] == primary)
            & (decomp_df["note"] == "ok")
        ]
        cp = probes_df[
            (probes_df["category"] == cat) & (probes_df["layer"] == primary)
        ]
        neg = cc[cc["cosine_normed"] < -0.1]
        seen: set[tuple[str, str]] = set()
        neg_pairs: list[dict] = []
        for _, r in neg.iterrows():
            pair = tuple(sorted([r["subgroup_a"], r["subgroup_b"]]))
            if pair not in seen:
                neg_pairs.append({"subgroups": list(pair),
                                  "cosine": round(float(r["cosine_normed"]), 3)})
                seen.add(pair)

        summary["per_category"][cat] = {
            "n_subgroups": int(cp["subgroup"].nunique()) if not cp.empty else 0,
            "mean_pairwise_cosine": round(float(cc["cosine_normed"].mean()), 3) if not cc.empty else None,
            "min_pairwise_cosine": round(float(cc["cosine_normed"].min()), 3) if not cc.empty else None,
            "max_pairwise_cosine": round(float(cc["cosine_normed"].max()), 3) if not cc.empty else None,
            "n_anti_correlated_pairs": len(neg_pairs),
            "mean_residual_ratio": round(float(cd["residual_ratio"].mean()), 3) if not cd.empty else None,
            "mean_probe_accuracy": round(float(cp["balanced_accuracy"].mean()), 3) if not cp.empty else None,
            "anti_correlated_pairs": neg_pairs,
        }
    return summary


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_cosine_heatmap(cos_df: pd.DataFrame, cat: str, layer: int,
                        out_dir: Path) -> None:
    df = cos_df[(cos_df["category"] == cat) & (cos_df["layer"] == layer)]
    if df.empty:
        return
    subs = sorted(df["subgroup_a"].unique())
    n = len(subs)
    s2i = {s: i for i, s in enumerate(subs)}
    mat = np.full((n, n), np.nan)
    for _, r in df.iterrows():
        mat[s2i[r["subgroup_a"]], s2i[r["subgroup_b"]]] = r["cosine_normed"]

    fig, ax = plt.subplots(figsize=(max(5, n * 0.8), max(4, n * 0.7)))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(subs, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(subs, fontsize=9)
    for i in range(n):
        for j in range(n):
            if not np.isnan(mat[i, j]):
                c = "white" if abs(mat[i, j]) > 0.5 else "black"
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        color=c, fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.8, label="cosine")
    ax.set_title(f"Identity direction cosines — {cat} (layer {layer})")
    plt.tight_layout()
    _save_fig(fig, out_dir / f"fig_cosine_heatmap_{cat}.png")


def fig_decomposition_bars(decomp_df: pd.DataFrame, layer: int,
                            out_dir: Path) -> None:
    df = decomp_df[(decomp_df["layer"] == layer) & (decomp_df["note"] == "ok")]
    if df.empty:
        return
    df = df.sort_values(["category", "subgroup"])
    fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.5), 5))
    x = np.arange(len(df))
    w = 0.4
    ax.bar(x - w / 2, df["category_component_norm"], w,
           color=WONG["blue"], label="Category-shared")
    ax.bar(x + w / 2, df["subgroup_residual_norm"], w,
           color=WONG["orange"], label="Subgroup-residual")
    prev = None
    for i, (_, r) in enumerate(df.iterrows()):
        if r["category"] != prev and i > 0:
            ax.axvline(i - 0.5, color="gray", linestyle=":", alpha=0.5)
        prev = r["category"]
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r['category']}/{r['subgroup']}" for _, r in df.iterrows()],
        rotation=45, ha="right", fontsize=8,
    )
    ax.set_ylabel("Direction component norm")
    ax.set_title(f"Gram-Schmidt decomposition at layer {layer}")
    ax.legend()
    plt.tight_layout()
    _save_fig(fig, out_dir / "fig_decomposition_bars.png")


def fig_probe_accuracy(probes_df: pd.DataFrame, layer: int,
                        out_dir: Path) -> None:
    df = probes_df[probes_df["layer"] == layer].sort_values(["category", "subgroup"])
    if df.empty:
        return
    cats = sorted(df["category"].unique())
    c2c = {c: SUBGROUP_COLORS[i % len(SUBGROUP_COLORS)] for i, c in enumerate(cats)}
    colors = [c2c[c] for c in df["category"]]

    fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.5), 5))
    x = np.arange(len(df))
    ax.bar(x, df["balanced_accuracy"], color=colors, alpha=0.8)
    if df["ci_low"].notna().all():
        ax.errorbar(x, df["balanced_accuracy"],
                    yerr=[df["balanced_accuracy"] - df["ci_low"],
                          df["ci_high"] - df["balanced_accuracy"]],
                    fmt="none", ecolor="black", capsize=3)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.7, label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r['category']}/{r['subgroup']}" for _, r in df.iterrows()],
        rotation=45, ha="right", fontsize=8,
    )
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0.4, 1.02)
    ax.set_title(f"Identity probe accuracy — layer {layer}")
    ax.legend(handles=[Patch(facecolor=c2c[c], label=c) for c in cats],
              loc="lower right", fontsize=7)
    plt.tight_layout()
    _save_fig(fig, out_dir / "fig_probe_accuracy_by_subgroup.png")


def fig_layer_sensitivity(cos_df: pd.DataFrame, categories: list[str],
                           layers: list[int], out_dir: Path) -> None:
    if len(layers) < 2:
        return
    for cat in categories:
        cd = cos_df[cos_df["category"] == cat]
        subs = sorted(cd["subgroup_a"].unique())
        n = len(subs)
        if n < 3:
            continue
        s2i = {s: i for i, s in enumerate(subs)}
        fig, axes = plt.subplots(1, len(layers),
                                 figsize=(4 * len(layers), 4), squeeze=False)
        for ai, layer in enumerate(layers):
            ax = axes[0, ai]
            mat = np.full((n, n), np.nan)
            for _, r in cd[cd["layer"] == layer].iterrows():
                mat[s2i[r["subgroup_a"]], s2i[r["subgroup_b"]]] = r["cosine_normed"]
            im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(subs, rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels(subs, fontsize=7)
            ax.set_title(f"Layer {layer}")
        fig.suptitle(f"Layer sensitivity — {cat}", fontsize=11)
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
        _save_fig(fig, out_dir / f"fig_layer_sensitivity_{cat}.png")


def generate_all_figures(
    cos_df: pd.DataFrame, decomp_df: pd.DataFrame, probes_df: pd.DataFrame,
    categories: list[str], layers: list[int], primary: int, out_dir: Path,
) -> None:
    for cat in categories:
        fig_cosine_heatmap(cos_df, cat, primary, out_dir)
    fig_decomposition_bars(decomp_df, primary, out_dir)
    fig_probe_accuracy(probes_df, primary, out_dir)
    if len(layers) > 1:
        fig_layer_sensitivity(cos_df, categories, layers, out_dir)
    log(f"  All figures saved to {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    t0 = time.time()

    output_dir = run_dir / "stage1_geometry"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "identity_directions").mkdir(exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    log(f"Stage 1: Representation Geometry")
    log(f"  Run dir: {run_dir}")
    log(f"  Primary layer: {args.layer}")

    meta_df = load_metadata(run_dir)
    log(f"  Loaded metadata: {len(meta_df)} items")

    if args.categories:
        categories = [c.strip() for c in args.categories.split(",")]
    else:
        with open(run_dir / "config.json") as f:
            categories = json.load(f)["categories"]
    log(f"  Categories: {categories}")

    if args.skip_sensitivity:
        layers = [args.layer]
    else:
        sens = [int(x) for x in args.sensitivity_layers.split(",")]
        layers = sorted(set([args.layer] + sens))
    log(f"  Layers: {layers}")

    catalog = build_subgroup_catalog(meta_df, categories, args.min_n_per_group)
    n_ok = sum(1 for v in catalog.values() if v["analyzable"])
    log(f"  Analyzable subgroups: {n_ok}/{len(catalog)}")

    all_cosines: list[dict] = []
    all_decomp: list[dict] = []
    all_probes: list[dict] = []

    for layer in layers:
        log(f"\n{'=' * 60}")
        log(f"Layer {layer}")
        log(f"{'=' * 60}")

        hs = load_layer_hidden_states(run_dir, categories, layer)
        log(f"  Loaded hidden states for {len(hs)} items")

        dirs = compute_identity_directions(hs, catalog, layer)
        save_identity_directions(dirs, output_dir, layer)

        all_cosines.extend(compute_pairwise_cosines(dirs, layer))
        all_decomp.extend(compute_gram_schmidt_decomposition(dirs, layer))
        all_probes.extend(
            compute_identity_probes(
                hs, meta_df, catalog, layer,
                n_folds=args.n_cv_folds,
                n_bootstrap=args.n_bootstrap,
                seed=args.random_seed,
            )
        )

    cos_df = pd.DataFrame(all_cosines)
    decomp_df = pd.DataFrame(all_decomp)
    probes_df = pd.DataFrame(all_probes)

    cos_df.to_parquet(output_dir / "cosines.parquet", index=False)
    decomp_df.to_parquet(output_dir / "decomposition.parquet", index=False)
    probes_df.to_parquet(output_dir / "probe_accuracies.parquet", index=False)
    log(f"\nSaved cosines ({len(cos_df)}), decomposition ({len(decomp_df)}), "
        f"probes ({len(probes_df)})")

    if len(layers) > 1:
        stability = compute_layer_stability(cos_df, layers, categories)
        with open(output_dir / "layer_stability.json", "w") as f:
            json.dump(stability, f, indent=2)

    summary = build_geometry_summary(
        cos_df, decomp_df, probes_df, catalog, layers, args.layer,
    )
    with open(output_dir / "geometry_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    if not args.skip_figures:
        log("\nGenerating figures...")
        generate_all_figures(
            cos_df, decomp_df, probes_df,
            categories, layers, args.layer, output_dir / "figures",
        )

    elapsed = time.time() - t0
    log(f"\nStage 1 complete in {elapsed:.1f}s")
    log(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
