"""B2 figures: feature overlap heatmaps, overlap curves, ranked effect sizes,
feature counts, layer distribution, injection layer histogram.

Uses the Wong colorblind-safe palette throughout.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.ranking import make_sub_key, parse_sub_key
from src.utils.io import ensure_dir
from src.utils.logging import log

# Wong colorblind-safe palette.
WONG = {
    "orange": "#E69F00",
    "blue": "#0072B2",
    "green": "#009E73",
    "purple": "#CC79A7",
    "vermillion": "#D55E00",
    "sky_blue": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
}
WONG_CYCLE = list(WONG.values())
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

DPI = 150


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log(f"    Saved {path.name}")


# ---------------------------------------------------------------------------
# Feature overlap heatmap (per category, k=20)
# ---------------------------------------------------------------------------

def fig_feature_overlap(
    overlap_data: dict[str, Any],
    ranked_df: pd.DataFrame,
    fig_dir: Path,
    k: int = 20,
) -> None:
    """Heatmap of Jaccard at k for all subgroup pairs, per category."""
    for cat, cat_data in overlap_data.items():
        subs = cat_data["subgroups"]
        n = len(subs)
        if n < 2:
            continue

        for direction in ["s_marking", "other_marking"]:
            pairs = cat_data.get(direction, {})
            if not pairs:
                continue

            matrix = np.full((n, n), np.nan)
            for i, sub_A in enumerate(subs):
                matrix[i, i] = 1.0
                for j, sub_B in enumerate(subs):
                    if i >= j:
                        continue
                    pair_key = f"{sub_A}__{sub_B}"
                    curve = pairs.get(pair_key, {})
                    val = curve.get(str(k), curve.get(k, {}))
                    if isinstance(val, dict):
                        j_val = val.get("jaccard")
                    else:
                        j_val = None
                    if j_val is not None:
                        matrix[i, j] = j_val
                        matrix[j, i] = j_val

            # Feature counts for labels.
            labels = []
            for s in subs:
                n_feat = len(ranked_df[
                    (ranked_df["category"] == cat)
                    & (ranked_df["subgroup"] == s)
                    & (ranked_df["direction"] == direction)
                ])
                labels.append(f"{s} (N={n_feat})")

            fig, ax = plt.subplots(figsize=(max(6, n * 1.2), max(5, n * 1.0)))
            im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")

            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(labels, fontsize=8)

            for i in range(n):
                for j in range(n):
                    if not np.isnan(matrix[i, j]) and i != j:
                        ax.text(j, i, f"{matrix[i, j]:.2f}",
                                ha="center", va="center", fontsize=7)

            plt.colorbar(im, ax=ax, shrink=0.8, label="Jaccard")
            dir_label = direction.replace("_", "-")
            ax.set_title(f"Top-{k} {dir_label} feature overlap — {cat}")

            _save(fig, fig_dir / f"fig_feature_overlap_{cat}_{direction}.png")


# ---------------------------------------------------------------------------
# Overlap curves (Jaccard vs k)
# ---------------------------------------------------------------------------

def fig_overlap_curves(
    overlap_data: dict[str, Any],
    fig_dir: Path,
) -> None:
    """Line plots: Jaccard as a function of k for all subgroup pairs."""
    for cat, cat_data in overlap_data.items():
        k_values = cat_data.get("k_values", [])
        if not k_values:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        for ax_idx, direction in enumerate(["s_marking", "other_marking"]):
            ax = axes[ax_idx]
            pairs = cat_data.get(direction, {})

            for p_idx, (pair_key, curve) in enumerate(sorted(pairs.items())):
                jaccards = []
                ks = []
                for k in k_values:
                    entry = curve.get(str(k), curve.get(k, {}))
                    if isinstance(entry, dict) and entry.get("jaccard") is not None:
                        ks.append(k)
                        jaccards.append(entry["jaccard"])

                if ks:
                    color = WONG_CYCLE[p_idx % len(WONG_CYCLE)]
                    marker = MARKERS[p_idx % len(MARKERS)]
                    label = pair_key.replace("__", " vs ")
                    ax.plot(ks, jaccards, marker=marker, color=color,
                            label=label, markersize=5, linewidth=1.5)

            ax.axhline(y=0.1, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
            ax.axhline(y=0.3, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
            ax.set_xscale("log")
            ax.set_xlabel("k (top features)")
            ax.set_ylim(-0.02, 1.02)
            dir_label = direction.replace("_", "-")
            ax.set_title(f"{dir_label}")
            ax.legend(fontsize=6, loc="upper left")

        axes[0].set_ylabel("Jaccard overlap")
        fig.suptitle(f"Feature overlap vs. k — {cat}", fontsize=13)
        fig.tight_layout()
        _save(fig, fig_dir / f"fig_overlap_curves_{cat}.png")


# ---------------------------------------------------------------------------
# Ranked effect sizes (|d| vs rank)
# ---------------------------------------------------------------------------

def fig_ranked_effect_sizes(
    ranked_df: pd.DataFrame,
    subgroups: list[tuple[str, str]],
    fig_dir: Path,
    max_rank: int = 100,
) -> None:
    """Overlay effect-size curves for all subgroups per category."""
    by_cat: dict[str, list[str]] = defaultdict(list)
    for cat, sub in subgroups:
        by_cat[cat].append(sub)

    for cat, subs in sorted(by_cat.items()):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        for ax_idx, direction in enumerate(["s_marking", "other_marking"]):
            ax = axes[ax_idx]
            for s_idx, sub in enumerate(sorted(subs)):
                df = ranked_df[
                    (ranked_df["category"] == cat)
                    & (ranked_df["subgroup"] == sub)
                    & (ranked_df["direction"] == direction)
                    & (ranked_df["rank"] <= max_rank)
                ].sort_values("rank")
                if df.empty:
                    continue
                color = WONG_CYCLE[s_idx % len(WONG_CYCLE)]
                marker = MARKERS[s_idx % len(MARKERS)]
                ax.plot(df["rank"], df["cohens_d"].abs(),
                        marker=marker, color=color, label=sub,
                        markersize=3, linewidth=1.2, markevery=max(1, len(df) // 10))

            ax.set_xlabel("Rank")
            dir_label = direction.replace("_", "-")
            ax.set_title(f"{dir_label}")
            ax.legend(fontsize=7)

        axes[0].set_ylabel("|Cohen's d|")
        fig.suptitle(f"Ranked effect sizes — {cat}", fontsize=13)
        fig.tight_layout()
        _save(fig, fig_dir / f"fig_ranked_effect_sizes_{cat}.png")


# ---------------------------------------------------------------------------
# Feature count bar chart
# ---------------------------------------------------------------------------

def fig_feature_count_per_subgroup(
    ranked_df: pd.DataFrame,
    subgroups: list[tuple[str, str]],
    fig_dir: Path,
) -> None:
    """Bar chart of significant feature counts per subgroup."""
    labels = []
    pro_counts = []
    anti_counts = []

    for cat, sub in subgroups:
        key = make_sub_key(cat, sub)
        n_pro = len(ranked_df[
            (ranked_df["category"] == cat)
            & (ranked_df["subgroup"] == sub)
            & (ranked_df["direction"] == "s_marking")
        ])
        n_anti = len(ranked_df[
            (ranked_df["category"] == cat)
            & (ranked_df["subgroup"] == sub)
            & (ranked_df["direction"] == "other_marking")
        ])
        labels.append(key)
        pro_counts.append(n_pro)
        anti_counts.append(n_anti)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.6), 6))
    ax.bar(x - width / 2, pro_counts, width, label="s-marking",
           color=WONG["blue"])
    ax.bar(x + width / 2, anti_counts, width, label="other-marking",
           color=WONG["orange"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("# significant features (FDR < 0.05)")
    ax.set_title("Significant features per subgroup")
    ax.legend()
    fig.tight_layout()
    _save(fig, fig_dir / "fig_feature_count_per_subgroup.png")


# ---------------------------------------------------------------------------
# Injection layer histogram
# ---------------------------------------------------------------------------

def fig_injection_layer_distribution(
    injection_layers: dict[str, dict[str, Any]],
    n_layers: int,
    fig_dir: Path,
    ranked_df: pd.DataFrame | None = None,
) -> None:
    """Two-panel figure: (A) significant feature count by layer across all
    subgroups, (B) injection layer selections.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Panel A: Feature count distribution across layers ---
    ax_a = axes[0]
    if ranked_df is not None and not ranked_df.empty:
        s_marking_by_layer = (
            ranked_df[ranked_df["direction"] == "s_marking"]
            .groupby("layer").size()
            .reindex(range(n_layers), fill_value=0)
        )
        other_marking_by_layer = (
            ranked_df[ranked_df["direction"] == "other_marking"]
            .groupby("layer").size()
            .reindex(range(n_layers), fill_value=0)
        )
        x = np.arange(n_layers)
        width = 0.4
        ax_a.bar(x - width / 2, s_marking_by_layer.values, width,
                 label="s-marking", color=WONG["blue"], alpha=0.8)
        ax_a.bar(x + width / 2, other_marking_by_layer.values, width,
                 label="other-marking", color=WONG["orange"], alpha=0.8)
    ax_a.set_xlabel("Layer")
    ax_a.set_ylabel("# significant features")
    ax_a.set_title("A. Significant features by layer")
    ax_a.legend(fontsize=8)

    # --- Panel B: Injection layer selections (one dot per subgroup) ---
    ax_b = axes[1]
    pro_layers: list[int] = []
    anti_layers: list[int] = []
    pro_labels: list[str] = []
    anti_labels: list[str] = []
    for key, rec in injection_layers.items():
        if rec.get("s_marking") and rec["s_marking"].get("injection_layer") is not None:
            pro_layers.append(rec["s_marking"]["injection_layer"])
            pro_labels.append(key)
        if rec.get("other_marking") and rec["other_marking"].get("injection_layer") is not None:
            anti_layers.append(rec["other_marking"]["injection_layer"])
            anti_labels.append(key)

    bins = np.arange(-0.5, n_layers + 0.5, 1)
    ax_b.hist(pro_layers, bins=bins, alpha=0.7, label="s-marking",
              color=WONG["blue"])
    ax_b.hist(anti_layers, bins=bins, alpha=0.7, label="other-marking",
              color=WONG["orange"])
    ax_b.set_xlabel("Layer")
    ax_b.set_ylabel("# subgroups")
    ax_b.set_title("B. Selected injection layers")
    ax_b.legend(fontsize=8)

    fig.suptitle("Layer distribution of significant features", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, fig_dir / "fig_injection_layer_distribution.png")


# ---------------------------------------------------------------------------
# Master figure generation
# ---------------------------------------------------------------------------

def generate_all_b2_figures(
    ranked_df: pd.DataFrame,
    subgroups: list[tuple[str, str]],
    overlap_data: dict[str, Any],
    injection_layers: dict[str, dict[str, Any]],
    n_layers: int,
    fig_dir: Path,
) -> None:
    """Generate all B2 figures."""
    fig_dir = ensure_dir(fig_dir)
    log("Generating B2 figures...")

    fig_feature_overlap(overlap_data, ranked_df, fig_dir)
    fig_overlap_curves(overlap_data, fig_dir)
    fig_ranked_effect_sizes(ranked_df, subgroups, fig_dir)
    fig_feature_count_per_subgroup(ranked_df, subgroups, fig_dir)
    fig_injection_layer_distribution(injection_layers, n_layers, fig_dir,
                                     ranked_df=ranked_df)

    log("  B2 figures complete")
