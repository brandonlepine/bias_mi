"""B5 figures: cross-subgroup activation heatmaps, specificity distributions,
matched-pairs deltas, co-occurrence heatmaps, artifact flag summary.

Uses the Wong colorblind-safe palette throughout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage

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


def _color(idx: int) -> str:
    return WONG_CYCLE[idx % len(WONG_CYCLE)]


def _marker(idx: int) -> str:
    return MARKERS[idx % len(MARKERS)]


# ---------------------------------------------------------------------------
# Cross-subgroup activation heatmap (clustered, per category)
# ---------------------------------------------------------------------------

def fig_cross_subgroup_activation(
    cross_matrices: dict[str, Any],
    fig_dir: Path,
) -> None:
    """Clustered heatmap of feature × subgroup mean activation per category."""
    for cat, data in cross_matrices.items():
        matrix = np.array(data["matrix"])
        feature_labels = data["feature_labels"]
        target_subs = data["target_subgroups"]
        ari = data["adjusted_rand_index"]
        bds = data["block_diagonal_strength"]

        n_feat, n_sub = matrix.shape
        if n_feat < 2 or n_sub < 2:
            continue

        # Row clustering
        row_maxes = np.maximum(matrix.max(axis=1, keepdims=True), 1e-8)
        normalised = matrix / row_maxes

        try:
            link = linkage(normalised, method="ward")
        except Exception:
            link = None

        fig_height = max(5, n_feat * 0.3)
        fig_width = max(6, n_sub * 1.0 + 3)

        if link is not None:
            fig, (ax_dend, ax_heat) = plt.subplots(
                1, 2, figsize=(fig_width, fig_height),
                gridspec_kw={"width_ratios": [1, 4]},
            )
            dend = dendrogram(link, orientation="left", labels=feature_labels,
                              ax=ax_dend, leaf_font_size=6, no_labels=True)
            ax_dend.set_xticks([])
            row_order = dend["leaves"]
        else:
            fig, ax_heat = plt.subplots(figsize=(fig_width, fig_height))
            row_order = list(range(n_feat))

        ordered_matrix = matrix[row_order, :]
        ordered_labels = [feature_labels[i] for i in row_order]

        vmax = float(matrix.max()) if matrix.max() > 0 else 1.0
        im = ax_heat.imshow(ordered_matrix, cmap="YlOrRd", vmin=0, vmax=vmax,
                            aspect="auto")
        ax_heat.set_xticks(range(n_sub))
        ax_heat.set_xticklabels(target_subs, rotation=45, ha="right", fontsize=8)
        ax_heat.set_yticks(range(n_feat))
        ax_heat.set_yticklabels(ordered_labels, fontsize=6)
        fig.colorbar(im, ax=ax_heat, shrink=0.6, label="Mean activation")

        ax_heat.set_title(
            f"Cross-subgroup activations — {cat} (ARI={ari:.2f}, BDS={bds:.1f})",
            fontsize=10,
        )
        fig.tight_layout()
        _save(fig, fig_dir / f"fig_cross_subgroup_activation_{cat}.png")


# ---------------------------------------------------------------------------
# Subgroup specificity distribution
# ---------------------------------------------------------------------------

def fig_subgroup_specificity_distribution(
    stats_df: pd.DataFrame,
    fig_dir: Path,
) -> None:
    """Histogram of subgroup_specificity across all pro-bias features."""
    pro = stats_df[
        (stats_df["direction"] == "s_marking")
        & stats_df["subgroup_specificity"].notna()
    ]
    if pro.empty:
        return

    vals = pro["subgroup_specificity"].values

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(vals, bins=40, color=WONG["blue"], edgecolor="white", alpha=0.85)
    ax.axvline(0.8, color=WONG["vermillion"], linestyle="--", linewidth=1,
               label="Threshold 0.8")
    ax.axvline(1.5, color=WONG["green"], linestyle="--", linewidth=1,
               label="Threshold 1.5")

    median_val = float(np.median(vals))
    q25, q75 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
    frac_above = float((vals > 1.5).mean())

    ax.set_xlabel("Subgroup specificity")
    ax.set_ylabel("Count")
    ax.set_title("Subgroup specificity distribution (all pro-bias features)", fontsize=11)
    ax.legend(fontsize=8)

    text = f"Median={median_val:.2f}, IQR=[{q25:.2f},{q75:.2f}]\n>1.5: {frac_above:.1%}"
    ax.text(0.97, 0.95, text, transform=ax.transAxes, ha="right", va="top",
            fontsize=8, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    _save(fig, fig_dir / "fig_subgroup_specificity_distribution.png")


# ---------------------------------------------------------------------------
# Category specificity ratio distribution
# ---------------------------------------------------------------------------

def fig_category_specificity_ratio(
    stats_df: pd.DataFrame,
    fig_dir: Path,
) -> None:
    """Histogram of category_specificity_ratio (log-scale recommended)."""
    valid = stats_df[stats_df["category_specificity_ratio"].notna()]
    if valid.empty:
        return

    vals = valid["category_specificity_ratio"].values
    # Clip for display
    vals_clipped = np.clip(vals, 0.01, None)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(np.log10(vals_clipped), bins=40, color=WONG["orange"],
            edgecolor="white", alpha=0.85)
    ax.axvline(np.log10(2.0), color=WONG["vermillion"], linestyle="--",
               linewidth=1, label="Threshold 2.0")

    n_below = int((vals < 2.0).sum())
    ax.set_xlabel("log10(category specificity ratio)")
    ax.set_ylabel("Count")
    ax.set_title("Category specificity ratio (all characterised features)", fontsize=11)
    ax.legend(fontsize=8)

    ax.text(0.97, 0.95, f"Below threshold: {n_below}/{len(vals)}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    _save(fig, fig_dir / "fig_category_specificity_ratio.png")


# ---------------------------------------------------------------------------
# Matched-pairs delta
# ---------------------------------------------------------------------------

def fig_matched_pairs_delta(
    stats_df: pd.DataFrame,
    categories: list[str],
    fig_dir: Path,
) -> None:
    """Per-category: feature rank vs mean matched-pairs delta, per subgroup."""
    pro = stats_df[
        (stats_df["direction"] == "s_marking")
        & stats_df["matched_mean_delta"].notna()
    ]
    if pro.empty:
        return

    cats_with_data = [c for c in categories if c in pro["category"].values]
    if not cats_with_data:
        return

    n_cats = len(cats_with_data)
    cols = min(3, n_cats)
    rows = (n_cats + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for idx, cat in enumerate(cats_with_data):
        ax = axes[idx // cols, idx % cols]
        cat_df = pro[pro["category"] == cat]

        subs = sorted(cat_df["subgroup"].unique())
        for s_idx, sub in enumerate(subs):
            sub_df = cat_df[cat_df["subgroup"] == sub].sort_values("rank")
            ax.plot(sub_df["rank"], sub_df["matched_mean_delta"],
                    color=_color(s_idx), marker=_marker(s_idx), markersize=3,
                    linewidth=1.0, label=sub)

        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xlabel("Feature rank")
        ax.set_ylabel("Mean delta (ambig − disambig)")
        ax.set_title(f"Matched-pairs delta — {cat}", fontsize=10)
        ax.legend(fontsize=6, loc="best")

    for idx in range(len(cats_with_data), rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)

    fig.tight_layout()
    _save(fig, fig_dir / "fig_matched_pairs_delta.png")


# ---------------------------------------------------------------------------
# Feature co-occurrence heatmaps per category
# ---------------------------------------------------------------------------

def fig_feature_cooccurrence(
    cooccurrence: dict[str, Any],
    categories: list[str],
    fig_dir: Path,
) -> None:
    """Correlation heatmaps for top-N features per subgroup, grouped by category."""
    # Group by category
    cat_subs: dict[str, list[str]] = {}
    for key in sorted(cooccurrence.keys()):
        cat = key.split("/")[0]
        cat_subs.setdefault(cat, []).append(key)

    for cat in categories:
        keys = cat_subs.get(cat, [])
        if not keys:
            continue

        n = len(keys)
        cols = min(3, n)
        rows_needed = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows_needed, cols, figsize=(4 * cols, 3.5 * rows_needed),
                                 squeeze=False)

        for idx, key in enumerate(keys):
            ax = axes[idx // cols, idx % cols]
            data = cooccurrence[key]
            matrix = np.array(data["matrix"])
            labels = data["feature_labels"]
            m = len(labels)

            im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(m))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=5)
            ax.set_yticks(range(m))
            ax.set_yticklabels(labels, fontsize=5)

            for i in range(m):
                for j in range(m):
                    color = "white" if abs(matrix[i, j]) > 0.5 else "black"
                    ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center",
                            fontsize=5, color=color)

            sub_name = key.split("/", 1)[1] if "/" in key else key
            ax.set_title(sub_name, fontsize=8)

        for idx in range(len(keys), rows_needed * cols):
            axes[idx // cols, idx % cols].set_visible(False)

        fig.suptitle(f"Feature co-occurrence — {cat}", fontsize=11)
        fig.tight_layout()
        _save(fig, fig_dir / f"fig_feature_cooccurrence_{cat}.png")


# ---------------------------------------------------------------------------
# Artifact flag summary bar chart
# ---------------------------------------------------------------------------

def fig_artifact_flag_summary(
    stats_df: pd.DataFrame,
    categories: list[str],
    fig_dir: Path,
) -> None:
    """Bar chart of artifact flag rates per category."""
    if stats_df.empty:
        return

    cats_with_data = [c for c in categories if c in stats_df["category"].values]
    if not cats_with_data:
        return

    flag_types = ["low_category_specificity", "length_correlation",
                  "high_firing_rate"]
    flag_colors = [WONG["blue"], WONG["orange"], WONG["green"]]

    x = np.arange(len(cats_with_data))
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(8, len(cats_with_data) * 1.5), 5))

    for f_idx, (flag_type, flag_color) in enumerate(zip(flag_types, flag_colors)):
        rates = []
        for cat in cats_with_data:
            cat_rows = stats_df[stats_df["category"] == cat]
            if len(cat_rows) == 0:
                rates.append(0.0)
            else:
                flagged = cat_rows["artifact_flags"].fillna("").str.contains(flag_type)
                rates.append(float(flagged.mean()))
        ax.bar(x + f_idx * width, rates, width, label=flag_type.replace("_", " "),
               color=flag_color)

    # Any flag
    any_rates = []
    for cat in cats_with_data:
        cat_rows = stats_df[stats_df["category"] == cat]
        if len(cat_rows) == 0:
            any_rates.append(0.0)
        else:
            any_rates.append(float(cat_rows["is_artifact_flagged"].mean()))
    ax.bar(x + len(flag_types) * width, any_rates, width, label="any flag",
           color=WONG["vermillion"])

    ax.set_xlabel("Category")
    ax.set_ylabel("Fraction flagged")
    ax.set_title("Artifact flag rates by category", fontsize=12)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(cats_with_data, rotation=45, ha="right")
    ax.legend(fontsize=8)
    fig.tight_layout()

    _save(fig, fig_dir / "fig_artifact_flag_summary.png")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_all_figures(
    run_dir: Path,
    stats_df: pd.DataFrame,
    cross_matrices: dict[str, Any],
    cooccurrence: dict[str, Any],
    categories: list[str],
) -> None:
    """Generate all B5 figures."""
    fig_dir = ensure_dir(run_dir / "B_feature_interpretability" / "figures")
    log(f"\nGenerating figures in {fig_dir}")

    fig_cross_subgroup_activation(cross_matrices, fig_dir)
    fig_subgroup_specificity_distribution(stats_df, fig_dir)
    fig_category_specificity_ratio(stats_df, fig_dir)
    fig_matched_pairs_delta(stats_df, categories, fig_dir)
    fig_feature_cooccurrence(cooccurrence, categories, fig_dir)
    fig_artifact_flag_summary(stats_df, categories, fig_dir)

    log("Figure generation complete.")
