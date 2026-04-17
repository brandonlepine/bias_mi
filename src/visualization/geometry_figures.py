"""B3 figures: cosine heatmaps, layer curves, direction norms, alignment, differentiation.

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
# Cosine heatmap at peak layer
# ---------------------------------------------------------------------------

def fig_cosine_heatmap(
    cosine_df: pd.DataFrame,
    differentiation: dict[str, Any],
    fig_dir: Path,
) -> None:
    """Heatmap of pairwise cosines at peak layer, per (cat, direction_type)."""
    for cat, cat_diff in differentiation.items():
        for dtype_label, dtype_data in cat_diff.items():
            peak_layer = dtype_data["peak_layer"]

            sub_df = cosine_df[
                (cosine_df["category"] == cat)
                & (cosine_df["direction_type"] == dtype_label)
                & (cosine_df["layer"] == peak_layer)
            ]
            if sub_df.empty:
                continue

            subs = sorted(
                set(sub_df["subgroup_A"].tolist() + sub_df["subgroup_B"].tolist())
            )
            n = len(subs)
            if n < 2:
                continue

            sub_idx = {s: i for i, s in enumerate(subs)}
            matrix = np.eye(n)
            for _, row in sub_df.iterrows():
                i = sub_idx[row["subgroup_A"]]
                j = sub_idx[row["subgroup_B"]]
                matrix[i, j] = row["cosine"]
                matrix[j, i] = row["cosine"]

            fig, ax = plt.subplots(figsize=(max(6, n * 1.2), max(5, n * 1.0)))
            im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(n))
            ax.set_xticklabels(subs, rotation=45, ha="right", fontsize=9)
            ax.set_yticks(range(n))
            ax.set_yticklabels(subs, fontsize=9)

            for i in range(n):
                for j in range(n):
                    color = "white" if abs(matrix[i, j]) > 0.5 else "black"
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                            fontsize=8, color=color)

            fig.colorbar(im, ax=ax, shrink=0.8, label="Cosine similarity")
            ax.set_title(
                f"Pairwise cosines — {cat} {dtype_label} at layer {peak_layer}",
                fontsize=11,
            )
            fig.tight_layout()

            fname = f"fig_cosine_heatmap_{cat}_{dtype_label}_L{peak_layer}.png"
            _save(fig, fig_dir / fname)


# ---------------------------------------------------------------------------
# Cosine by layer (line plot)
# ---------------------------------------------------------------------------

def fig_cosine_by_layer(
    cosine_df: pd.DataFrame,
    differentiation: dict[str, Any],
    n_layers: int,
    fig_dir: Path,
) -> None:
    """Line plot of each pair's cosine across layers."""
    for cat, cat_diff in differentiation.items():
        for dtype_label, dtype_data in cat_diff.items():
            peak_layer = dtype_data["peak_layer"]
            stable_start, stable_end = dtype_data["stable_range"]

            sub_df = cosine_df[
                (cosine_df["category"] == cat)
                & (cosine_df["direction_type"] == dtype_label)
            ]
            if sub_df.empty:
                continue

            pairs = sub_df.groupby(["subgroup_A", "subgroup_B"]).size().index.tolist()
            if not pairs:
                continue

            fig, ax = plt.subplots(figsize=(10, 5))

            for idx, (sub_A, sub_B) in enumerate(pairs):
                pair_df = sub_df[
                    (sub_df["subgroup_A"] == sub_A) & (sub_df["subgroup_B"] == sub_B)
                ].sort_values("layer")

                ax.plot(
                    pair_df["layer"], pair_df["cosine"],
                    color=_color(idx), marker=_marker(idx), markersize=3,
                    linewidth=1.2, label=f"{sub_A} vs {sub_B}",
                )

            ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.axvline(peak_layer, color="gray", linestyle="--", linewidth=0.8,
                       alpha=0.7, label=f"Peak L{peak_layer}")
            ax.axvspan(stable_start, stable_end, alpha=0.08, color="gray",
                       label=f"Stable [{stable_start}-{stable_end}]")

            ax.set_xlabel("Layer")
            ax.set_ylabel("Cosine similarity")
            ax.set_xlim(0, n_layers - 1)
            ax.set_ylim(-1.05, 1.05)
            ax.set_title(
                f"Subgroup cosines across layers — {cat} {dtype_label}",
                fontsize=11,
            )
            ax.legend(fontsize=7, loc="best", ncol=max(1, len(pairs) // 5))
            fig.tight_layout()

            fname = f"fig_cosine_by_layer_{cat}_{dtype_label}.png"
            _save(fig, fig_dir / fname)


# ---------------------------------------------------------------------------
# Direction norms (pre-normalization magnitudes)
# ---------------------------------------------------------------------------

def fig_direction_norms(
    directions_norms: dict[str, np.ndarray],
    categories: list[str],
    n_layers: int,
    fig_dir: Path,
) -> None:
    """2x2 grid of norm plots per category (bias/identity × raw/normed)."""
    grid_specs = [
        ("bias_direction_raw_norm", "Bias (raw)"),
        ("bias_direction_normed_norm", "Bias (normed)"),
        ("identity_direction_raw_norm", "Identity (raw)"),
        ("identity_direction_normed_norm", "Identity (normed)"),
    ]

    layers = np.arange(n_layers)

    for cat in categories:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        axes_flat = axes.flatten()
        any_data = False

        for ax_idx, (prefix, title) in enumerate(grid_specs):
            ax = axes_flat[ax_idx]

            # Find all subgroups with this norm type for this category
            subs = []
            for key in sorted(directions_norms.keys()):
                full_prefix = f"{prefix}_{cat}_"
                if key.startswith(full_prefix):
                    subs.append(key[len(full_prefix):])

            for s_idx, sub in enumerate(subs):
                norm_key = f"{prefix}_{cat}_{sub}"
                norm_arr = directions_norms[norm_key]
                ax.plot(layers, norm_arr, color=_color(s_idx), marker=_marker(s_idx),
                        markersize=2, linewidth=1.0, label=sub)
                any_data = True

            ax.set_title(title, fontsize=10)
            ax.set_ylabel("Norm (log scale)")
            ax.set_yscale("log")
            if subs:
                ax.legend(fontsize=7, loc="best")

        for ax in axes_flat[2:]:
            ax.set_xlabel("Layer")

        fig.suptitle(f"{cat} — direction pre-normalization magnitudes", fontsize=12)
        fig.tight_layout()

        if any_data:
            _save(fig, fig_dir / f"fig_direction_norms_{cat}.png")
        else:
            plt.close(fig)


# ---------------------------------------------------------------------------
# Bias-identity alignment
# ---------------------------------------------------------------------------

def fig_bias_identity_alignment(
    alignment: dict[str, Any],
    n_layers: int,
    fig_dir: Path,
) -> None:
    """Panel per category, two subplots (raw/normed), one line per subgroup."""
    cats_with_data = [cat for cat in alignment if alignment[cat]]
    if not cats_with_data:
        return

    n_cats = len(cats_with_data)
    fig, axes = plt.subplots(n_cats, 2, figsize=(14, 4 * n_cats), squeeze=False)
    layers = np.arange(n_layers)

    for row_idx, cat in enumerate(cats_with_data):
        cat_data = alignment[cat]
        subs = sorted(cat_data.keys())

        for col_idx, norm_type in enumerate(["raw", "normed"]):
            ax = axes[row_idx, col_idx]

            for s_idx, sub in enumerate(subs):
                sub_data = cat_data[sub].get(norm_type)
                if sub_data is None:
                    continue

                align_vals = sub_data["per_layer_alignment"]
                ax.plot(layers[:len(align_vals)], align_vals,
                        color=_color(s_idx), marker=_marker(s_idx),
                        markersize=2, linewidth=1.0, label=sub)

            ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.axhline(1, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)
            ax.axhline(-1, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)
            ax.set_xlabel("Layer")
            ax.set_ylabel("Alignment (cosine)")
            ax.set_ylim(-1.05, 1.05)
            ax.set_title(f"{cat} — {norm_type}", fontsize=10)
            if subs:
                ax.legend(fontsize=7, loc="best")

    fig.suptitle("Bias-identity alignment across layers", fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, fig_dir / "fig_bias_identity_alignment.png")


# ---------------------------------------------------------------------------
# Differentiation metrics summary bar chart
# ---------------------------------------------------------------------------

def fig_differentiation_summary(
    differentiation: dict[str, Any],
    fig_dir: Path,
) -> None:
    """Bar chart of peak variance by category, identity_normed vs bias_normed."""
    cats = sorted(differentiation.keys())
    identity_vals = []
    bias_vals = []
    identity_peaks = []
    bias_peaks = []

    for cat in cats:
        id_data = differentiation[cat].get("identity_normed", {})
        bi_data = differentiation[cat].get("bias_normed", {})
        identity_vals.append(id_data.get("peak_variance", 0))
        bias_vals.append(bi_data.get("peak_variance", 0))
        identity_peaks.append(id_data.get("peak_layer", ""))
        bias_peaks.append(bi_data.get("peak_layer", ""))

    if not any(v > 0 for v in identity_vals + bias_vals):
        return

    x = np.arange(len(cats))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(cats) * 1.5), 5))
    bars_id = ax.bar(x - width / 2, identity_vals, width, label="identity_normed",
                     color=WONG["blue"])
    bars_bi = ax.bar(x + width / 2, bias_vals, width, label="bias_normed",
                     color=WONG["orange"])

    # Annotate peak layers
    for bar, peak in zip(bars_id, identity_peaks):
        if peak != "":
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"L{peak}", ha="center", va="bottom", fontsize=7)
    for bar, peak in zip(bars_bi, bias_peaks):
        if peak != "":
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"L{peak}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Category")
    ax.set_ylabel("Peak variance")
    ax.set_title("Peak differentiation by category", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()

    _save(fig, fig_dir / "fig_differentiation_metrics.png")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_all_figures(
    run_dir: Path,
    directions_norms: dict[str, np.ndarray],
    cosine_df: pd.DataFrame,
    differentiation: dict[str, Any],
    alignment: dict[str, Any],
    categories: list[str],
    n_layers: int,
) -> None:
    """Generate all B3 figures."""
    fig_dir = ensure_dir(run_dir / "B_geometry" / "figures")
    log(f"\nGenerating figures in {fig_dir}")

    fig_cosine_heatmap(cosine_df, differentiation, fig_dir)
    fig_cosine_by_layer(cosine_df, differentiation, n_layers, fig_dir)
    fig_direction_norms(directions_norms, categories, n_layers, fig_dir)
    fig_bias_identity_alignment(alignment, n_layers, fig_dir)
    fig_differentiation_summary(differentiation, fig_dir)

    log("Figure generation complete.")
