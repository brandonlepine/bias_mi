"""B4 figures: probe selectivity, binary subgroup, structural comparison,
raw-vs-SAE, cross-category heatmap, within-category generalization.

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
# fig_probe_selectivity: multiclass subgroup probe + permutation baseline
# ---------------------------------------------------------------------------

def fig_probe_selectivity(
    probe_df: pd.DataFrame,
    categories: list[str],
    fig_dir: Path,
) -> None:
    """One panel per category: multiclass probe balanced accuracy vs permutation."""
    multi = probe_df[probe_df["probe_type"] == "subgroup_multiclass"].copy()
    if multi.empty:
        return

    cats_with_data = [c for c in categories if c in multi["category"].values]
    if not cats_with_data:
        return

    n_cats = len(cats_with_data)
    cols = min(3, n_cats)
    rows = (n_cats + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for idx, cat in enumerate(cats_with_data):
        ax = axes[idx // cols, idx % cols]
        cat_df = multi[multi["category"] == cat].sort_values("layer")

        if cat_df.empty:
            ax.set_visible(False)
            continue

        layers = cat_df["layer"].values
        ba = cat_df["mean_balanced_accuracy"].values
        perm_mean = cat_df["permutation_mean_balanced_accuracy"].values

        ax.plot(layers, ba, color=WONG["blue"], marker="o", markersize=3,
                linewidth=1.2, label="Probe")

        # Permutation baseline with ±1σ band
        perm_valid = ~np.isnan(perm_mean.astype(float))
        if perm_valid.any():
            ax.plot(layers[perm_valid], perm_mean[perm_valid],
                    color="gray", linestyle="--", linewidth=1.0, label="Permutation")

        # Peak selectivity marker
        sel = cat_df["selectivity"].values
        sel_valid = ~pd.isna(sel)
        if sel_valid.any():
            peak_idx = np.nanargmax(sel)
            ax.axvline(layers[peak_idx], color=WONG["vermillion"], linestyle=":",
                       linewidth=0.8, alpha=0.7)

        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Balanced accuracy")
        ax.set_ylim(0, 1.05)
        ax.set_title(cat, fontsize=10)
        ax.legend(fontsize=7, loc="best")

    # Hide unused axes
    for idx in range(len(cats_with_data), rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)

    fig.suptitle("Subgroup multiclass probe — selectivity", fontsize=12)
    fig.tight_layout()
    _save(fig, fig_dir / "fig_probe_selectivity.png")


# ---------------------------------------------------------------------------
# fig_probe_binary_subgroup: one line per subgroup within each category
# ---------------------------------------------------------------------------

def fig_probe_binary_subgroup(
    probe_df: pd.DataFrame,
    categories: list[str],
    fig_dir: Path,
) -> None:
    """Binary subgroup detection balanced accuracy across layers."""
    binary = probe_df[probe_df["probe_type"] == "subgroup_binary"].copy()
    if binary.empty:
        return

    cats_with_data = [c for c in categories if c in binary["category"].values]
    if not cats_with_data:
        return

    n_cats = len(cats_with_data)
    cols = min(3, n_cats)
    rows = (n_cats + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for idx, cat in enumerate(cats_with_data):
        ax = axes[idx // cols, idx % cols]
        cat_df = binary[binary["category"] == cat]

        subs = sorted(cat_df["subgroup"].dropna().unique())
        for s_idx, sub in enumerate(subs):
            sub_df = cat_df[cat_df["subgroup"] == sub].sort_values("layer")
            ax.plot(sub_df["layer"], sub_df["mean_balanced_accuracy"],
                    color=_color(s_idx), marker=_marker(s_idx), markersize=2,
                    linewidth=1.0, label=sub)

        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Balanced accuracy")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Binary subgroup detection — {cat}", fontsize=10)
        ax.legend(fontsize=6, loc="best")

    for idx in range(len(cats_with_data), rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)

    fig.tight_layout()
    _save(fig, fig_dir / "fig_probe_binary_subgroup.png")


# ---------------------------------------------------------------------------
# fig_probe_structural_comparison: all probe types on one panel per category
# ---------------------------------------------------------------------------

def fig_probe_structural_comparison(
    probe_df: pd.DataFrame,
    categories: list[str],
    fig_dir: Path,
) -> None:
    """Compare probe types: multiclass, stereotyped, context, template."""
    probe_types = [
        ("subgroup_multiclass", "Subgroup MC", WONG["blue"], "o"),
        ("stereotyped_response_binary", "Stereotyped", WONG["orange"], "s"),
        ("context_condition", "Context", WONG["green"], "^"),
        ("template_id", "Template ID", WONG["purple"], "D"),
    ]

    cats_with_data = sorted(probe_df["category"].dropna().unique())
    cats_with_data = [c for c in categories if c in cats_with_data]
    if not cats_with_data:
        return

    n_cats = len(cats_with_data)
    cols = min(3, n_cats)
    rows = (n_cats + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for idx, cat in enumerate(cats_with_data):
        ax = axes[idx // cols, idx % cols]

        for pt_name, pt_label, pt_color, pt_marker in probe_types:
            pt_df = probe_df[
                (probe_df["probe_type"] == pt_name)
                & (probe_df["category"] == cat)
                & probe_df["mean_balanced_accuracy"].notna()
            ].sort_values("layer")

            if pt_df.empty:
                continue

            ax.plot(pt_df["layer"], pt_df["mean_balanced_accuracy"],
                    color=pt_color, marker=pt_marker, markersize=3,
                    linewidth=1.0, label=pt_label)

        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Balanced accuracy")
        ax.set_ylim(0, 1.05)
        ax.set_title(cat, fontsize=10)
        ax.legend(fontsize=7, loc="best")

    for idx in range(len(cats_with_data), rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)

    fig.suptitle("Structural comparison — probe types", fontsize=12)
    fig.tight_layout()
    _save(fig, fig_dir / "fig_probe_structural_comparison.png")


# ---------------------------------------------------------------------------
# fig_raw_vs_sae_probes: paired raw/SAE lines per subgroup
# ---------------------------------------------------------------------------

def fig_raw_vs_sae_probes(
    probe_df: pd.DataFrame,
    categories: list[str],
    fig_dir: Path,
) -> None:
    """Compare raw hidden-state vs SAE-feature binary subgroup probes."""
    raw = probe_df[probe_df["probe_type"] == "subgroup_binary"].copy()
    sae = probe_df[probe_df["probe_type"] == "sae_subgroup_binary"].copy()
    if raw.empty or sae.empty:
        return

    cats_with_data = [c for c in categories if c in raw["category"].values]
    if not cats_with_data:
        return

    n_cats = len(cats_with_data)
    cols = min(3, n_cats)
    rows = (n_cats + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for idx, cat in enumerate(cats_with_data):
        ax = axes[idx // cols, idx % cols]
        cat_raw = raw[raw["category"] == cat]
        cat_sae = sae[sae["category"] == cat]

        subs = sorted(cat_raw["subgroup"].dropna().unique())
        for s_idx, sub in enumerate(subs):
            # Raw line (solid)
            sub_raw = cat_raw[cat_raw["subgroup"] == sub].sort_values("layer")
            if not sub_raw.empty:
                ax.plot(sub_raw["layer"], sub_raw["mean_balanced_accuracy"],
                        color=_color(s_idx), marker=_marker(s_idx), markersize=2,
                        linewidth=1.0, label=f"{sub} (raw)")

            # SAE point (layer=-1 means across layers; plot as horizontal dashed)
            sub_sae = cat_sae[cat_sae["subgroup"] == sub]
            if not sub_sae.empty:
                sae_ba = sub_sae["mean_balanced_accuracy"].values[0]
                if sae_ba is not None and not np.isnan(sae_ba):
                    ax.axhline(sae_ba, color=_color(s_idx), linestyle="--",
                               linewidth=0.8, alpha=0.7)

        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Balanced accuracy")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Raw vs SAE — {cat}", fontsize=10)
        ax.legend(fontsize=6, loc="best")

    for idx in range(len(cats_with_data), rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)

    fig.tight_layout()
    _save(fig, fig_dir / "fig_raw_vs_sae_probes.png")


# ---------------------------------------------------------------------------
# fig_cross_category_matrix: heatmap at selected layers
# ---------------------------------------------------------------------------

def fig_cross_category_matrix(
    cross_df: pd.DataFrame,
    fig_dir: Path,
) -> None:
    """Heatmap of cross-category stereotyped-response generalization."""
    if cross_df.empty:
        return

    # Pick up to 4 representative layers
    all_layers = sorted(cross_df["layer"].unique())
    if not all_layers:
        return

    # Select evenly-spaced layers
    n_panels = min(4, len(all_layers))
    step = max(1, len(all_layers) // n_panels)
    selected_layers = [all_layers[i * step] for i in range(n_panels)]

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5), squeeze=False)

    for p_idx, layer in enumerate(selected_layers):
        ax = axes[0, p_idx]
        layer_df = cross_df[cross_df["layer"] == layer]
        if layer_df.empty:
            ax.set_visible(False)
            continue

        cats = sorted(set(layer_df["train_category"].tolist()
                          + layer_df["test_category"].tolist()))
        n = len(cats)
        cat_idx = {c: i for i, c in enumerate(cats)}

        matrix = np.full((n, n), np.nan)
        for _, row in layer_df.iterrows():
            i = cat_idx.get(row["train_category"])
            j = cat_idx.get(row["test_category"])
            if i is not None and j is not None:
                matrix[i, j] = row["balanced_accuracy"]

        im = ax.imshow(matrix, cmap="Blues", vmin=0.4, vmax=1.0, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(n))
        ax.set_yticklabels(cats, fontsize=7)

        for i in range(n):
            for j in range(n):
                if not np.isnan(matrix[i, j]):
                    color = "white" if matrix[i, j] > 0.7 else "black"
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                            fontsize=7, color=color)

        ax.set_title(f"Layer {layer}", fontsize=10)
        if p_idx == 0:
            ax.set_ylabel("Train category")

    fig.suptitle("Cross-category is_stereotyped generalization", fontsize=12)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label="Balanced accuracy")
    fig.tight_layout()
    _save(fig, fig_dir / "fig_cross_category_matrix.png")


# ---------------------------------------------------------------------------
# fig_within_category_generalization: heatmap per category at peak layer
# ---------------------------------------------------------------------------

def fig_within_category_generalization(
    within_df: pd.DataFrame,
    differentiation: dict[str, Any] | None,
    fig_dir: Path,
) -> None:
    """Heatmap of within-category cross-subgroup generalization."""
    if within_df.empty:
        return

    cats = sorted(within_df["category"].unique())
    cats_with_data = []
    for cat in cats:
        subs = set(within_df[within_df["category"] == cat]["train_subgroup"].tolist()
                    + within_df[within_df["category"] == cat]["test_subgroup"].tolist())
        if len(subs) >= 2:
            cats_with_data.append(cat)

    if not cats_with_data:
        return

    n_cats = len(cats_with_data)
    cols = min(3, n_cats)
    rows = (n_cats + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows), squeeze=False)

    for idx, cat in enumerate(cats_with_data):
        ax = axes[idx // cols, idx % cols]
        cat_df = within_df[within_df["category"] == cat]

        # Pick peak layer from differentiation if available, else use median layer
        if differentiation and cat in differentiation:
            id_data = differentiation[cat].get("identity_normed", {})
            peak_layer = id_data.get("peak_layer", int(cat_df["layer"].median()))
        else:
            peak_layer = int(cat_df["layer"].median())

        layer_df = cat_df[cat_df["layer"] == peak_layer]
        if layer_df.empty:
            # Fall back to closest available layer
            available = sorted(cat_df["layer"].unique())
            peak_layer = min(available, key=lambda l: abs(l - peak_layer))
            layer_df = cat_df[cat_df["layer"] == peak_layer]

        subs = sorted(set(layer_df["train_subgroup"].tolist()
                          + layer_df["test_subgroup"].tolist()))
        n = len(subs)
        sub_idx = {s: i for i, s in enumerate(subs)}

        matrix = np.full((n, n), np.nan)
        for _, row in layer_df.iterrows():
            i = sub_idx.get(row["train_subgroup"])
            j = sub_idx.get(row["test_subgroup"])
            if i is not None and j is not None:
                matrix[i, j] = row["balanced_accuracy"]

        im = ax.imshow(matrix, cmap="RdBu_r", vmin=0.3, vmax=0.7, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_xticklabels(subs, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(n))
        ax.set_yticklabels(subs, fontsize=7)

        for i in range(n):
            for j in range(n):
                if not np.isnan(matrix[i, j]):
                    color = "white" if abs(matrix[i, j] - 0.5) > 0.15 else "black"
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                            fontsize=7, color=color)

        ax.set_title(f"{cat} — layer {peak_layer}", fontsize=10)
        ax.set_ylabel("Train subgroup")
        ax.set_xlabel("Test subgroup")

    for idx in range(len(cats_with_data), rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)

    fig.suptitle("Within-category cross-subgroup generalization", fontsize=12)
    fig.tight_layout()
    _save(fig, fig_dir / "fig_within_category_generalization.png")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_all_figures(
    run_dir: Path,
    probe_df: pd.DataFrame,
    cross_df: pd.DataFrame,
    within_df: pd.DataFrame,
    categories: list[str],
    differentiation: dict[str, Any] | None = None,
) -> None:
    """Generate all B4 figures."""
    fig_dir = ensure_dir(run_dir / "B_probes" / "figures")
    log(f"\nGenerating figures in {fig_dir}")

    fig_probe_selectivity(probe_df, categories, fig_dir)
    fig_probe_binary_subgroup(probe_df, categories, fig_dir)
    fig_probe_structural_comparison(probe_df, categories, fig_dir)
    fig_raw_vs_sae_probes(probe_df, categories, fig_dir)
    fig_cross_category_matrix(cross_df, fig_dir)
    fig_within_category_generalization(within_df, differentiation, fig_dir)

    log("Figure generation complete.")
