"""C2 figures: universal backfire scatter, transfer heatmaps, cosine comparisons,
per-category/source faceted scatters, stable-range robustness.

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

from src.utils.io import ensure_dir
from src.utils.logging import log


# ---------------------------------------------------------------------------
# Palette & style
# ---------------------------------------------------------------------------

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

CATEGORY_COLORS: dict[str, str] = {
    "so": WONG["orange"],
    "gi": WONG["blue"],
    "race": WONG["green"],
    "religion": WONG["purple"],
    "disability": WONG["vermillion"],
    "age": WONG["sky_blue"],
    "ses": WONG["yellow"],
    "nationality": WONG["black"],
    "physical_appearance": "#999999",
}

CATEGORY_MARKERS: dict[str, str] = {
    "so": "o",
    "gi": "s",
    "race": "^",
    "religion": "D",
    "disability": "v",
    "age": "P",
    "ses": "X",
    "nationality": "p",
    "physical_appearance": "h",
}

DPI = 150


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log(f"    Saved {path.name}")


def _color_for(cat: str) -> str:
    return CATEGORY_COLORS.get(cat, "#666666")


def _marker_for(cat: str) -> str:
    return CATEGORY_MARKERS.get(cat, "o")


# ---------------------------------------------------------------------------
# fig_universal_backfire_scatter
# ---------------------------------------------------------------------------

def _plot_scatter_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    regression: dict[str, Any] | None,
    title: str,
) -> None:
    """Plot one panel of the universal backfire scatter."""
    # Self-pairs as gold stars
    self_df = df[df["is_self"]]
    non_self = df[~df["is_self"]]

    # Non-self points by category
    for cat in sorted(non_self["category"].unique()):
        cat_df = non_self[non_self["category"] == cat]
        sizes = np.log1p(cat_df["n_items"].values) * 10 + 15
        ax.scatter(
            cat_df["cosine_dim_identity_normed"],
            cat_df["bias_change"],
            c=_color_for(cat), marker=_marker_for(cat),
            s=sizes, alpha=0.7, label=cat, zorder=3,
        )

    # Self-pairs
    if not self_df.empty:
        ax.scatter(
            self_df["cosine_dim_identity_normed"],
            self_df["bias_change"],
            marker="*", s=120, c="gold", edgecolors="black",
            linewidths=0.8, zorder=5, label="self",
        )

    # Regression line + CI
    if regression and regression.get("n", 0) >= 5:
        x_line = np.linspace(-1, 1, 100)
        y_line = regression["slope"] * x_line + regression["intercept"]
        ax.plot(x_line, y_line, "k--", linewidth=1.2, zorder=4)

        # Bootstrap CI band
        slope_lo, slope_hi = regression["slope_ci_95"]
        int_lo, int_hi = regression["intercept_ci_95"]
        y_lo = slope_lo * x_line + int_lo
        y_hi = slope_hi * x_line + int_hi
        ax.fill_between(x_line, y_lo, y_hi, color="gray", alpha=0.15, zorder=2)

        # Annotation
        r2 = regression["r_squared"]
        p = regression["p_value"]
        n = regression["n"]
        ax.annotate(
            f"r² = {r2:.3f}, p = {p:.1e}, n = {n}",
            (0.95, 0.05), xycoords="axes fraction",
            ha="right", va="bottom", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    # Reference lines
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", zorder=1)
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--", zorder=1)

    # Quadrant labels
    ax.text(-0.9, ax.get_ylim()[1] * 0.85, "BACKFIRE",
            fontsize=8, color="gray", ha="left", style="italic")
    ax.text(0.9, ax.get_ylim()[0] * 0.85, "CROSS-\nDEBIASING",
            fontsize=8, color="gray", ha="right", style="italic")

    ax.set_xlabel("Pairwise cosine (DIM identity-normed)")
    ax.set_ylabel("Bias change")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-1.1, 1.1)


def fig_universal_backfire_scatter(
    scatter_df: pd.DataFrame,
    regression_results: dict[str, Any],
    fig_dir: Path,
) -> None:
    """Two panels: all categories, excluding disability."""
    has_dim = scatter_df.dropna(subset=["cosine_dim_identity_normed"])
    if has_dim.empty:
        log("    SKIP: no DIM cosines for scatter")
        return

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5.5))

    _plot_scatter_panel(
        ax_a, has_dim,
        regression_results.get("primary_dim_all"),
        "Panel A: All categories",
    )

    no_dis = has_dim[has_dim["category"] != "disability"]
    _plot_scatter_panel(
        ax_b, no_dis,
        regression_results.get("sensitivity_dim_no_disability"),
        "Panel B: Excluding disability",
    )

    # Shared legend
    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 8),
               fontsize=7, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig, fig_dir / "fig_universal_backfire_scatter.png")


# ---------------------------------------------------------------------------
# fig_cosine_vs_backfire_by_category
# ---------------------------------------------------------------------------

def fig_cosine_vs_backfire_by_category(
    scatter_df: pd.DataFrame,
    regression_results: dict[str, Any],
    fig_dir: Path,
) -> None:
    """Faceted scatter, one panel per category."""
    per_cat = regression_results.get("per_category", {})
    has_dim = scatter_df.dropna(subset=["cosine_dim_identity_normed"])
    cats_with_data = [
        cat for cat in sorted(has_dim["category"].unique())
        if cat in per_cat and per_cat[cat].get("n_pairs", 0) >= 4
    ]

    if not cats_with_data:
        return

    n = len(cats_with_data)
    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows),
                             squeeze=False)

    for idx, cat in enumerate(cats_with_data):
        ax = axes[idx // n_cols][idx % n_cols]
        cat_df = has_dim[has_dim["category"] == cat]

        non_self = cat_df[~cat_df["is_self"]]
        self_df = cat_df[cat_df["is_self"]]

        ax.scatter(
            non_self["cosine_dim_identity_normed"],
            non_self["bias_change"],
            c=_color_for(cat), marker=_marker_for(cat),
            s=40, alpha=0.7,
        )
        if not self_df.empty:
            ax.scatter(
                self_df["cosine_dim_identity_normed"],
                self_df["bias_change"],
                marker="*", s=100, c="gray", alpha=0.5,
            )

        reg = per_cat.get(cat, {})
        if reg.get("r_squared") is not None:
            x_line = np.linspace(-1, 1, 50)
            y_line = reg["slope"] * x_line + reg["intercept"]
            ax.plot(x_line, y_line, "k--", linewidth=1)
            ax.annotate(
                f"r² = {reg['r_squared']:.3f}",
                (0.95, 0.05), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=8,
            )

        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_xlabel("DIM cosine")
        ax.set_ylabel("Bias change")
        ax.set_title(cat, fontsize=10)

    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle("Cosine vs. backfire by category", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, fig_dir / "fig_cosine_vs_backfire_by_category.png")


# ---------------------------------------------------------------------------
# fig_cosine_vs_backfire_by_source_subgroup
# ---------------------------------------------------------------------------

def fig_cosine_vs_backfire_by_source_subgroup(
    scatter_df: pd.DataFrame,
    regression_results: dict[str, Any],
    fig_dir: Path,
) -> None:
    """One panel per source subgroup with >= 3 target pairs."""
    per_source = regression_results.get("per_source_subgroup", {})
    has_dim = scatter_df.dropna(subset=["cosine_dim_identity_normed"])

    sources_with_data = []
    for key, reg in per_source.items():
        if reg.get("n_pairs", 0) >= 3 and reg.get("r_squared") is not None:
            sources_with_data.append(key)

    if not sources_with_data:
        return

    n = len(sources_with_data)
    n_cols = min(4, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows),
                             squeeze=False)

    for idx, src_key in enumerate(sorted(sources_with_data)):
        ax = axes[idx // n_cols][idx % n_cols]
        cat, src = src_key.split("/", 1)

        src_df = has_dim[
            (has_dim["category"] == cat)
            & (has_dim["source_subgroup"] == src)
        ]
        non_self = src_df[~src_df["is_self"]]
        self_df = src_df[src_df["is_self"]]

        # Color by target subgroup
        targets = sorted(non_self["target_subgroup"].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(targets), 1)))
        for ti, tgt in enumerate(targets):
            tgt_df = non_self[non_self["target_subgroup"] == tgt]
            ax.scatter(
                tgt_df["cosine_dim_identity_normed"],
                tgt_df["bias_change"],
                c=[colors[ti]], s=40, label=tgt, alpha=0.8,
            )

        if not self_df.empty:
            ax.scatter(1.0, self_df["bias_change"].iloc[0],
                       marker="*", s=120, c="gold", edgecolors="black",
                       linewidths=0.8, zorder=5)

        reg = per_source[src_key]
        if reg.get("slope") is not None:
            x_line = np.linspace(-1, 1, 50)
            y_line = reg["slope"] * x_line + reg["intercept"]
            ax.plot(x_line, y_line, "k--", linewidth=1)
            ax.annotate(
                f"r²={reg['r_squared']:.2f}\nslope={reg['slope']:+.2f}",
                (0.95, 0.95), xycoords="axes fraction",
                ha="right", va="top", fontsize=6,
            )

        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_xlabel("DIM cosine")
        ax.set_ylabel("Bias change")
        ax.set_title(src_key, fontsize=9)
        if len(targets) <= 6:
            ax.legend(fontsize=5, loc="lower left")

    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle("Per-source subgroup transfer effects", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, fig_dir / "fig_cosine_vs_backfire_by_source_subgroup.png")


# ---------------------------------------------------------------------------
# fig_transfer_heatmaps
# ---------------------------------------------------------------------------

def fig_transfer_heatmaps(
    transfer_df: pd.DataFrame,
    viable_manifests: list[dict[str, Any]],
    fig_dir: Path,
) -> None:
    """Grid of heatmaps, one per category with >= 2 subgroups."""
    by_cat: dict[str, list[str]] = {}
    for m in viable_manifests:
        by_cat.setdefault(m["category"], []).append(m["subgroup"])

    cats_with_multi = [c for c, s in by_cat.items() if len(s) >= 2]
    if not cats_with_multi:
        return

    n = len(cats_with_multi)
    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows),
                             squeeze=False)

    panel_labels = "ABCDEFGHIJKLMNOP"

    for idx, cat in enumerate(sorted(cats_with_multi)):
        ax = axes[idx // n_cols][idx % n_cols]
        subs = sorted(by_cat[cat])
        n_subs = len(subs)

        matrix = np.full((n_subs, n_subs), np.nan)
        for _, row in transfer_df[transfer_df["category"] == cat].iterrows():
            src = row["source_subgroup"]
            tgt = row["target_subgroup"]
            if src in subs and tgt in subs:
                si = subs.index(src)
                ti = subs.index(tgt)
                matrix[si, ti] = row["bias_change"]

        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

        # Annotate cells
        for si in range(n_subs):
            for ti in range(n_subs):
                val = matrix[si, ti]
                if not np.isnan(val):
                    color = "white" if abs(val) > 0.5 else "black"
                    ax.text(ti, si, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color=color)

        ax.set_xticks(range(n_subs))
        ax.set_xticklabels(subs, fontsize=7, rotation=45, ha="right")
        ax.set_yticks(range(n_subs))
        ax.set_yticklabels(subs, fontsize=7)
        ax.set_xlabel("Target")
        ax.set_ylabel("Source")
        label = panel_labels[idx] if idx < len(panel_labels) else ""
        ax.set_title(f"{label}. {cat}", fontsize=10)

    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.colorbar(im, ax=axes, shrink=0.6, label="Bias change")
    fig.suptitle("Cross-subgroup transfer heatmaps", fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])
    _save(fig, fig_dir / "fig_transfer_heatmaps.png")


# ---------------------------------------------------------------------------
# fig_sae_vs_dim_cosine
# ---------------------------------------------------------------------------

def fig_sae_vs_dim_cosine(
    scatter_df: pd.DataFrame,
    sae_dim_comparison: dict[str, Any],
    fig_dir: Path,
) -> None:
    """X = DIM cosine, Y = SAE cosine, diagonal reference."""
    paired = scatter_df[~scatter_df["is_self"]].dropna(
        subset=["cosine_dim_identity_normed", "cosine_sae_steering"],
    )
    if len(paired) < 3:
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    for cat in sorted(paired["category"].unique()):
        cat_df = paired[paired["category"] == cat]
        ax.scatter(
            cat_df["cosine_dim_identity_normed"],
            cat_df["cosine_sae_steering"],
            c=_color_for(cat), marker=_marker_for(cat),
            s=40, alpha=0.7, label=cat,
        )

    ax.plot([-1, 1], [-1, 1], "k--", linewidth=0.8, alpha=0.5, label="y=x")

    pearson = sae_dim_comparison.get("pearson_r", "n/a")
    spearman = sae_dim_comparison.get("spearman_rho", "n/a")
    ax.annotate(
        f"Pearson r = {pearson}\nSpearman ρ = {spearman}",
        (0.05, 0.95), xycoords="axes fraction",
        ha="left", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel("DIM identity-normed cosine")
    ax.set_ylabel("SAE steering vector cosine")
    ax.set_title("SAE steering vector geometry vs. DIM identity geometry")
    ax.legend(fontsize=7)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

    fig.tight_layout()
    _save(fig, fig_dir / "fig_sae_vs_dim_cosine.png")


# ---------------------------------------------------------------------------
# fig_stable_range_robustness
# ---------------------------------------------------------------------------

def fig_stable_range_robustness(
    stable_range_results: dict[str, Any],
    fig_dir: Path,
) -> None:
    """r² and slope across layers in stable range per category."""
    cats = [c for c in stable_range_results
            if isinstance(stable_range_results[c], dict)
            and "per_layer" in stable_range_results[c]]
    if not cats:
        return

    n = len(cats)
    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                             squeeze=False)

    for idx, cat in enumerate(sorted(cats)):
        ax = axes[idx // n_cols][idx % n_cols]
        data = stable_range_results[cat]
        per_layer = data["per_layer"]
        stable_start, stable_end = data["stable_range"]

        layers = sorted(int(l) for l in per_layer.keys())
        r2s = [per_layer[str(l)]["r_squared"] for l in layers]
        slopes = [per_layer[str(l)]["slope"] for l in layers]

        ax.plot(layers, r2s, "o-", color=WONG["blue"], label="r²")
        ax.set_ylabel("r²", color=WONG["blue"])
        ax.set_xlabel("Layer")

        ax2 = ax.twinx()
        ax2.plot(layers, slopes, "s--", color=WONG["orange"], label="slope")
        ax2.set_ylabel("Slope", color=WONG["orange"])

        # Shade stable range
        ax.axvspan(stable_start, stable_end, alpha=0.1, color="green")

        ax.set_title(cat, fontsize=10)

    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle("Stable-range robustness of cosine-backfire relationship", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, fig_dir / "fig_stable_range_robustness.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_c2_figures(
    output_dir: Path,
    scatter_df: pd.DataFrame,
    regression_results: dict[str, Any],
    sae_dim_comparison: dict[str, Any],
    stable_range_results: dict[str, Any],
    transfer_df: pd.DataFrame,
    viable_manifests: list[dict[str, Any]],
) -> None:
    """Generate all C2 figures."""
    fig_dir = ensure_dir(output_dir / "figures")
    log("\nGenerating C2 figures...")

    fig_universal_backfire_scatter(scatter_df, regression_results, fig_dir)
    fig_cosine_vs_backfire_by_category(scatter_df, regression_results, fig_dir)
    fig_cosine_vs_backfire_by_source_subgroup(
        scatter_df, regression_results, fig_dir,
    )
    fig_transfer_heatmaps(transfer_df, viable_manifests, fig_dir)
    fig_sae_vs_dim_cosine(scatter_df, sae_dim_comparison, fig_dir)
    if stable_range_results:
        fig_stable_range_robustness(stable_range_results, fig_dir)

    log("C2 figures complete")
