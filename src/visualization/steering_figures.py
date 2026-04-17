"""C1 figures: Pareto frontiers, stepwise correction, marginal analysis,
optimal-k distribution, alpha-vs-k heatmaps, margin-conditioned bars,
exacerbation asymmetry.

Uses the Wong colorblind-safe palette throughout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
WONG_CYCLE = list(WONG.values())

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

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
DPI = 150


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log(f"    Saved {path.name}")


def _color_for_category(cat: str) -> str:
    return CATEGORY_COLORS.get(cat, "#666666")


# ---------------------------------------------------------------------------
# fig_pareto_frontier_{cat}
# ---------------------------------------------------------------------------

def fig_pareto_frontier(
    grid_records: list[dict[str, Any]],
    manifests: list[dict[str, Any]],
    fig_dir: Path,
) -> None:
    """One subplot per subgroup.  X = ||v||, Y = RCR_1.0. Color by k."""
    # Group grid records by category
    by_cat: dict[str, dict[str, list[dict]]] = {}
    for r in grid_records:
        cat = r["category"]
        sub = r["subgroup"]
        by_cat.setdefault(cat, {}).setdefault(sub, []).append(r)

    # Build optimal lookup
    optimal_lookup: dict[str, dict[str, Any]] = {}
    for m in manifests:
        if m.get("steering_viable"):
            optimal_lookup[f"{m['category']}/{m['subgroup']}"] = m

    for cat, subs_data in by_cat.items():
        subs = sorted(subs_data.keys())
        n_subs = len(subs)
        if n_subs == 0:
            continue

        n_cols = min(4, n_subs)
        n_rows = (n_subs + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                                 squeeze=False)

        for idx, sub in enumerate(subs):
            ax = axes[idx // n_cols][idx % n_cols]
            records = subs_data[sub]

            # Collect unique k values for colormap
            k_values = sorted(set(r["k"] for r in records))
            cmap = plt.cm.viridis
            k_to_color = {
                k: cmap(i / max(len(k_values) - 1, 1))
                for i, k in enumerate(k_values)
            }

            for r in records:
                is_unsafe = r["degeneration_rate"] >= 0.05 or r["corruption_rate"] >= 0.05
                alpha = 0.3 if is_unsafe else 0.8
                ax.scatter(
                    r["vector_norm"], r["metrics"]["rcr_1.0"]["rcr"],
                    c=[k_to_color[r["k"]]], s=40, alpha=alpha,
                    edgecolors="gray" if is_unsafe else "none",
                    linewidths=0.5 if is_unsafe else 0,
                )

            # Mark optimum
            opt_key = f"{cat}/{sub}"
            if opt_key in optimal_lookup:
                m = optimal_lookup[opt_key]
                ax.scatter(
                    m["optimal_vector_norm"], m["optimal_rcr_1.0"],
                    marker="*", s=200, c="gold", edgecolors="black",
                    linewidths=1, zorder=10,
                )
                ax.annotate(
                    f"η*={m['optimal_eta']:.3f}",
                    (m["optimal_vector_norm"], m["optimal_rcr_1.0"]),
                    textcoords="offset points", xytext=(8, 8), fontsize=7,
                )

            ax.set_xlabel("||v||₂")
            ax.set_ylabel("RCR₁.₀")
            ax.set_title(sub, fontsize=10)
            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

        # Hide unused axes
        for idx in range(n_subs, n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].set_visible(False)

        fig.suptitle(f"Pareto frontier — {cat}", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        _save(fig, fig_dir / f"fig_pareto_frontier_{cat}.png")


# ---------------------------------------------------------------------------
# fig_stepwise_correction_{cat}
# ---------------------------------------------------------------------------

def fig_stepwise_correction(
    grid_records: list[dict[str, Any]],
    manifests: list[dict[str, Any]],
    fig_dir: Path,
) -> None:
    """X = k, Y1 = RCR, Y2 = corruption rate, at optimal target_norm."""
    optimal_lookup = {
        f"{m['category']}/{m['subgroup']}": m
        for m in manifests if m.get("steering_viable")
    }

    by_cat: dict[str, dict[str, list[dict]]] = {}
    for r in grid_records:
        by_cat.setdefault(r["category"], {}).setdefault(r["subgroup"], []).append(r)

    for cat, subs_data in by_cat.items():
        viable_subs = [s for s in subs_data if f"{cat}/{s}" in optimal_lookup]
        if not viable_subs:
            continue

        n_subs = len(viable_subs)
        n_cols = min(4, n_subs)
        n_rows = (n_subs + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                                 squeeze=False)

        for idx, sub in enumerate(sorted(viable_subs)):
            ax = axes[idx // n_cols][idx % n_cols]
            m = optimal_lookup[f"{cat}/{sub}"]
            opt_tn = m["optimal_target_norm"]

            # Filter to optimal target_norm
            relevant = [r for r in subs_data[sub] if r["target_norm"] == opt_tn]
            relevant.sort(key=lambda r: r["k"])

            if not relevant:
                ax.set_visible(False)
                continue

            ks = [r["k"] for r in relevant]
            rcrs = [r["metrics"]["rcr_1.0"]["rcr"] for r in relevant]
            corrupts = [r["corruption_rate"] for r in relevant]

            ax.plot(ks, rcrs, "o-", color=WONG["blue"], label="RCR₁.₀")
            for ki, ri, ni in zip(ks, rcrs, [r["n_items"] for r in relevant]):
                ax.annotate(f"n={ni}", (ki, ri), textcoords="offset points",
                            xytext=(0, 6), fontsize=6, ha="center")

            ax2 = ax.twinx()
            ax2.plot(ks, corrupts, "s--", color=WONG["vermillion"],
                     label="Corruption")
            ax2.set_ylabel("Corruption rate", color=WONG["vermillion"])

            ax.axvline(m["optimal_k"], color="gray", linestyle="--", alpha=0.5)
            ax.set_xlabel("k")
            ax.set_ylabel("RCR₁.₀", color=WONG["blue"])
            ax.set_title(sub, fontsize=10)

        for idx in range(n_subs, n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].set_visible(False)

        fig.suptitle(f"Stepwise correction — {cat}", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        _save(fig, fig_dir / f"fig_stepwise_correction_{cat}.png")


# ---------------------------------------------------------------------------
# fig_marginal_analysis_{cat}
# ---------------------------------------------------------------------------

def fig_marginal_analysis(
    manifests: list[dict[str, Any]],
    fig_dir: Path,
) -> None:
    """At selected tau*, show RCR, ||v||, and marginal efficiency vs k."""
    by_cat: dict[str, list[dict]] = {}
    for m in manifests:
        if m.get("steering_viable") and m.get("marginal_analysis"):
            by_cat.setdefault(m["category"], []).append(m)

    for cat, cat_manifests in by_cat.items():
        n_subs = len(cat_manifests)
        n_cols = min(4, n_subs)
        n_rows = (n_subs + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                                 squeeze=False)

        for idx, m in enumerate(sorted(cat_manifests, key=lambda x: x["subgroup"])):
            ax = axes[idx // n_cols][idx % n_cols]
            marginal = m["marginal_analysis"]

            ks = [e["k"] for e in marginal]
            rcrs = [e["rcr_1.0"] for e in marginal]
            norms = [e["vector_norm"] for e in marginal]

            ax.plot(ks, rcrs, "o-", color=WONG["blue"], label="RCR₁.₀")
            ax.set_ylabel("RCR₁.₀", color=WONG["blue"])
            ax.set_xlabel("k")

            ax2 = ax.twinx()
            ax2.plot(ks, norms, "s--", color=WONG["orange"], label="||v||₂")
            ax2.set_ylabel("||v||₂", color=WONG["orange"])

            # Marginal efficiency on third axis (plotted as scatter)
            marg_eff = [e.get("marginal_efficiency") for e in marginal]
            valid_me = [(k, me) for k, me in zip(ks, marg_eff) if me is not None]
            if valid_me:
                me_ks, me_vals = zip(*valid_me)
                ax.scatter(me_ks, [0] * len(me_ks), marker="d",
                           color=WONG["green"], s=15, zorder=5)

            ax.axvline(m["optimal_k"], color="gray", linestyle="--", alpha=0.5)
            ax.set_title(m["subgroup"], fontsize=10)

        for idx in range(n_subs, n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].set_visible(False)

        fig.suptitle(f"Marginal analysis — {cat}", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        _save(fig, fig_dir / f"fig_marginal_analysis_{cat}.png")


# ---------------------------------------------------------------------------
# fig_optimal_k_distribution
# ---------------------------------------------------------------------------

def fig_optimal_k_distribution(
    manifests: list[dict[str, Any]],
    fig_dir: Path,
) -> None:
    """Histogram of optimal k across all viable subgroups."""
    viable = [m for m in manifests if m.get("steering_viable")]
    if not viable:
        return

    fig, ax = plt.subplots(figsize=(6, 4))

    ks = [m["optimal_k"] for m in viable]
    cats = [m["category"] for m in viable]
    unique_cats = sorted(set(cats))

    # Stacked by category
    k_bins = sorted(set(ks))
    bin_edges = [k - 0.5 for k in k_bins] + [max(k_bins) + 0.5]

    for ci, cat in enumerate(unique_cats):
        cat_ks = [k for k, c in zip(ks, cats) if c == cat]
        ax.hist(cat_ks, bins=bin_edges, alpha=0.7,
                color=_color_for_category(cat), label=cat)

    median_k = float(np.median(ks))
    ax.axvline(median_k, color="black", linestyle="--", linewidth=1)
    ax.annotate(f"median={median_k:.0f}", (median_k, ax.get_ylim()[1] * 0.9),
                fontsize=8)

    ax.set_xlabel("Optimal k")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of optimal k across subgroups")
    ax.legend(fontsize=7)

    fig.tight_layout()
    _save(fig, fig_dir / "fig_optimal_k_distribution.png")


# ---------------------------------------------------------------------------
# fig_alpha_vs_k_heatmaps_{cat}
# ---------------------------------------------------------------------------

def fig_alpha_vs_k_heatmaps(
    grid_records: list[dict[str, Any]],
    manifests: list[dict[str, Any]],
    fig_dir: Path,
) -> None:
    """Heatmap of eta across (k, target_norm) grid per subgroup."""
    optimal_lookup = {
        f"{m['category']}/{m['subgroup']}": m
        for m in manifests if m.get("steering_viable")
    }

    by_cat: dict[str, dict[str, list[dict]]] = {}
    for r in grid_records:
        by_cat.setdefault(r["category"], {}).setdefault(r["subgroup"], []).append(r)

    for cat, subs_data in by_cat.items():
        subs = sorted(subs_data.keys())
        n_subs = len(subs)
        if n_subs == 0:
            continue

        n_cols = min(4, n_subs)
        n_rows = (n_subs + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                                 squeeze=False)

        for idx, sub in enumerate(subs):
            ax = axes[idx // n_cols][idx % n_cols]
            records = subs_data[sub]

            # Build grid
            all_k = sorted(set(r["k"] for r in records))
            all_tn = sorted(set(r["target_norm"] for r in records))

            if not all_k or not all_tn:
                ax.set_visible(False)
                continue

            eta_matrix = np.full((len(all_k), len(all_tn)), np.nan)
            unsafe_mask = np.zeros((len(all_k), len(all_tn)), dtype=bool)

            for r in records:
                ki = all_k.index(r["k"])
                ti = all_tn.index(r["target_norm"])
                if r["degeneration_rate"] >= 0.05 or r["corruption_rate"] >= 0.05:
                    unsafe_mask[ki, ti] = True
                else:
                    eta_matrix[ki, ti] = r["eta"]

            im = ax.imshow(eta_matrix, aspect="auto", cmap="YlGnBu",
                           origin="lower")
            # Gray out unsafe cells
            gray_overlay = np.zeros((*eta_matrix.shape, 4))
            gray_overlay[unsafe_mask] = [0.7, 0.7, 0.7, 0.6]
            ax.imshow(gray_overlay, aspect="auto", origin="lower")

            ax.set_xticks(range(len(all_tn)))
            ax.set_xticklabels([f"{tn}" for tn in all_tn], fontsize=6, rotation=45)
            ax.set_yticks(range(len(all_k)))
            ax.set_yticklabels([str(k) for k in all_k], fontsize=6)
            ax.set_xlabel("target_norm")
            ax.set_ylabel("k")
            ax.set_title(sub, fontsize=10)

            # Star on optimum
            opt_key = f"{cat}/{sub}"
            if opt_key in optimal_lookup:
                m = optimal_lookup[opt_key]
                if m["optimal_k"] in all_k and m["optimal_target_norm"] in all_tn:
                    ki = all_k.index(m["optimal_k"])
                    ti = all_tn.index(m["optimal_target_norm"])
                    ax.scatter(ti, ki, marker="*", s=150, c="gold",
                               edgecolors="black", linewidths=1, zorder=10)

            fig.colorbar(im, ax=ax, shrink=0.8)

        for idx in range(n_subs, n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].set_visible(False)

        fig.suptitle(f"η heatmap (k × target_norm) — {cat}", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        _save(fig, fig_dir / f"fig_alpha_vs_k_heatmaps_{cat}.png")


# ---------------------------------------------------------------------------
# fig_margin_conditioned_{cat}
# ---------------------------------------------------------------------------

def fig_margin_conditioned(
    manifests: list[dict[str, Any]],
    fig_dir: Path,
) -> None:
    """Grouped bars per subgroup: RCR at each margin bin."""
    by_cat: dict[str, list[dict]] = {}
    for m in manifests:
        if m.get("steering_viable") and m.get("optimal_logit_shift"):
            by_cat.setdefault(m["category"], []).append(m)

    bin_names = ["near_indifferent", "moderate", "confident"]
    bin_colors = [WONG["orange"], WONG["blue"], WONG["green"]]

    for cat, cat_manifests in by_cat.items():
        n_subs = len(cat_manifests)
        n_cols = min(4, n_subs)
        n_rows = (n_subs + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                                 squeeze=False)

        for idx, m in enumerate(sorted(cat_manifests, key=lambda x: x["subgroup"])):
            ax = axes[idx // n_cols][idx % n_cols]

            per_bin = m["optimal_logit_shift"].get("per_margin_bin", {})
            x_pos = np.arange(len(bin_names))
            values = []
            ns = []
            for bn in bin_names:
                bd = per_bin.get(bn, {})
                values.append(abs(bd.get("mean_shift", 0)))
                ns.append(bd.get("n", 0))

            bars = ax.bar(x_pos, values, color=bin_colors, width=0.6)
            for bi, (bar, n) in enumerate(zip(bars, ns)):
                ax.annotate(f"n={n}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            ha="center", va="bottom", fontsize=7)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(bin_names, fontsize=7, rotation=20)
            ax.set_ylabel("|Mean logit shift|")
            ax.set_title(m["subgroup"], fontsize=10)

        for idx in range(n_subs, n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].set_visible(False)

        fig.suptitle(f"Margin-conditioned effects — {cat}", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        _save(fig, fig_dir / f"fig_margin_conditioned_{cat}.png")


# ---------------------------------------------------------------------------
# fig_exacerbation_asymmetry
# ---------------------------------------------------------------------------

def fig_exacerbation_asymmetry(
    manifests: list[dict[str, Any]],
    fig_dir: Path,
) -> None:
    """Paired bars: RCR (debiasing) vs corruption rate (exacerbation)."""
    viable = [
        m for m in manifests
        if m.get("steering_viable") and m.get("exacerbation")
    ]
    if not viable:
        return

    viable.sort(key=lambda m: (m["category"], m["subgroup"]))
    labels = [f"{m['category']}/{m['subgroup']}" for m in viable]

    rcrs = [m["optimal_rcr_1.0"] for m in viable]
    corrupts = [m["exacerbation"]["corruption_rate_non_stereo"] for m in viable]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.6), 5))
    ax.bar(x - width / 2, rcrs, width, color=WONG["blue"], label="RCR₁.₀ (debiasing)")
    ax.bar(x + width / 2, corrupts, width, color=WONG["vermillion"],
           label="Corruption (exacerbation)")

    # Annotate n
    for i, m in enumerate(viable):
        ax.annotate(
            f"n={m['n_stereo_items']}", (x[i] - width / 2, rcrs[i]),
            ha="center", va="bottom", fontsize=5,
        )
        ax.annotate(
            f"n={m['exacerbation']['n_non_stereo_items']}",
            (x[i] + width / 2, corrupts[i]),
            ha="center", va="bottom", fontsize=5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Rate")
    ax.set_title("Debiasing vs. exacerbation effects — all viable subgroups")
    ax.legend()
    ax.axhline(0, color="gray", linewidth=0.5)

    fig.tight_layout()
    _save(fig, fig_dir / "fig_exacerbation_asymmetry.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_c1_figures(
    output_dir: Path,
    manifests: list[dict[str, Any]],
    grid_records: list[dict[str, Any]],
) -> None:
    """Generate all C1 figures."""
    fig_dir = ensure_dir(output_dir / "figures")
    log("\nGenerating C1 figures...")

    fig_pareto_frontier(grid_records, manifests, fig_dir)
    fig_stepwise_correction(grid_records, manifests, fig_dir)
    fig_marginal_analysis(manifests, fig_dir)
    fig_optimal_k_distribution(manifests, fig_dir)
    fig_alpha_vs_k_heatmaps(grid_records, manifests, fig_dir)
    fig_margin_conditioned(manifests, fig_dir)
    fig_exacerbation_asymmetry(manifests, fig_dir)

    log("C1 figures complete")
