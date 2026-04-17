"""C3 figures: MedQA conditions comparison, debias vs exacerbation,
logit shift distributions, MMLU supercategory heatmap, side-effect heatmap,
BBQ vs MedQA scatter plots.

Uses the Wong colorblind-safe palette throughout.
"""

from __future__ import annotations

import json
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
# Palette
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

CONDITION_COLORS = {
    "matched": WONG["blue"],
    "within_cat_mismatched": WONG["orange"],
    "cross_cat_mismatched": WONG["green"],
    "no_demographic": WONG["vermillion"],
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

DPI = 150


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log(f"    Saved {path.name}")


# ---------------------------------------------------------------------------
# fig_medqa_conditions_comparison
# ---------------------------------------------------------------------------

def fig_medqa_conditions_comparison(
    medqa_agg: dict[str, Any],
    fig_dir: Path,
) -> None:
    """Grouped bars per subgroup: accuracy delta by MedQA condition."""
    conditions = ["matched", "within_cat_mismatched",
                   "cross_cat_mismatched", "no_demographic"]
    cond_labels = ["Matched", "Within-cat\nmismatch", "Cross-cat\nmismatch",
                    "No demo"]

    # Group debiasing results by category
    by_cat: dict[str, list[tuple[str, dict]]] = {}
    for key, result in medqa_agg.items():
        if not key.endswith("_debiasing"):
            continue
        vec = result["steering_vector"]
        cat = vec.split("/")[0]
        by_cat.setdefault(cat, []).append((vec, result))

    cats_with_data = [c for c in sorted(by_cat) if by_cat[c]]
    if not cats_with_data:
        return

    n_cats = len(cats_with_data)
    fig, axes = plt.subplots(1, n_cats, figsize=(6 * n_cats, 5), squeeze=False)

    for ci, cat in enumerate(cats_with_data):
        ax = axes[0][ci]
        vecs = sorted(by_cat[cat], key=lambda x: x[0])
        sub_labels = [v.split("/")[1] for v, _ in vecs]
        x = np.arange(len(sub_labels))
        width = 0.18

        for cond_i, cond in enumerate(conditions):
            deltas = []
            ns = []
            for _, result in vecs:
                cond_data = result.get("per_condition", {}).get(cond, {})
                deltas.append(cond_data.get("accuracy_delta", 0))
                ns.append(cond_data.get("n", 0))

            bars = ax.bar(
                x + cond_i * width - 1.5 * width, deltas, width,
                color=CONDITION_COLORS.get(cond, "#999"),
                label=cond_labels[cond_i] if ci == 0 else "",
            )
            for bi, (bar, n) in enumerate(zip(bars, ns)):
                if n > 0:
                    ax.annotate(
                        f"n={n}", (bar.get_x() + bar.get_width() / 2,
                                   bar.get_height()),
                        ha="center", va="bottom", fontsize=5, rotation=90,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(sub_labels, fontsize=7, rotation=30, ha="right")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_ylabel("Accuracy delta")
        ax.set_title(cat, fontsize=10)

    axes[0][0].legend(fontsize=6, loc="lower left")
    fig.suptitle("MedQA: accuracy delta by demographic condition", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, fig_dir / "fig_medqa_conditions_comparison.png")


# ---------------------------------------------------------------------------
# fig_medqa_debias_vs_exacerbation
# ---------------------------------------------------------------------------

def fig_medqa_debias_vs_exacerbation(
    medqa_agg: dict[str, Any],
    fig_dir: Path,
) -> None:
    """Paired bars: matched debias delta vs matched exac delta."""
    pairs: list[tuple[str, float, float, int, int]] = []
    vec_keys = set()
    for key in medqa_agg:
        vec = medqa_agg[key]["steering_vector"]
        vec_keys.add(vec)

    for vec in sorted(vec_keys):
        debias = medqa_agg.get(f"{vec}_debiasing", {})
        exac = medqa_agg.get(f"{vec}_exacerbation", {})
        d_matched = debias.get("per_condition", {}).get("matched", {})
        e_matched = exac.get("per_condition", {}).get("matched", {})
        if d_matched.get("n", 0) >= 10 or e_matched.get("n", 0) >= 10:
            pairs.append((
                vec,
                d_matched.get("accuracy_delta", 0),
                e_matched.get("accuracy_delta", 0),
                d_matched.get("n", 0),
                e_matched.get("n", 0),
            ))

    if not pairs:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(pairs) * 0.8), 5))
    labels = [p[0] for p in pairs]
    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width / 2, [p[1] for p in pairs], width,
           color=WONG["blue"], label="Debiasing")
    ax.bar(x + width / 2, [p[2] for p in pairs], width,
           color=WONG["vermillion"], label="Exacerbation")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Accuracy delta (matched items)")
    ax.set_title("MedQA: debiasing vs exacerbation on matched items")
    ax.legend()

    fig.tight_layout()
    _save(fig, fig_dir / "fig_medqa_debias_vs_exacerbation.png")


# ---------------------------------------------------------------------------
# fig_medqa_logit_shift_distributions
# ---------------------------------------------------------------------------

def fig_medqa_logit_shift_distributions(
    output_dir: Path,
    fig_dir: Path,
) -> None:
    """Violin plots of correct_logit_shift for matched items."""
    parquet_path = output_dir / "medqa" / "per_item.parquet"
    if not parquet_path.exists():
        return

    df = pd.read_parquet(parquet_path)
    matched = df[df["condition"] == "matched"]
    if len(matched) < 20:
        return

    fig, ax = plt.subplots(figsize=(max(8, matched["steering_vector"].nunique() * 1.2), 5))

    vecs = sorted(matched["steering_vector"].unique())
    data_debias = []
    data_exac = []
    labels = []

    for vec in vecs:
        d = matched[(matched["steering_vector"] == vec) & (matched["direction"] == "debiasing")]
        e = matched[(matched["steering_vector"] == vec) & (matched["direction"] == "exacerbation")]
        if len(d) >= 5:
            data_debias.append(d["correct_logit_shift"].values)
            data_exac.append(e["correct_logit_shift"].values if len(e) >= 5 else np.array([]))
            labels.append(vec)

    if not data_debias:
        plt.close(fig)
        return

    positions = np.arange(len(labels))
    parts_d = ax.violinplot(data_debias, positions=positions - 0.15,
                             widths=0.25, showmedians=True)
    for pc in parts_d["bodies"]:
        pc.set_facecolor(WONG["blue"])
        pc.set_alpha(0.6)

    non_empty_exac = [d for d in data_exac if len(d) > 0]
    if non_empty_exac:
        exac_pos = [p + 0.15 for p, d in zip(positions, data_exac) if len(d) > 0]
        parts_e = ax.violinplot(non_empty_exac, positions=exac_pos,
                                 widths=0.25, showmedians=True)
        for pc in parts_e["bodies"]:
            pc.set_facecolor(WONG["vermillion"])
            pc.set_alpha(0.6)

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
    ax.set_ylabel("Correct logit shift")
    ax.set_title("MedQA: logit shift distribution on matched items")

    fig.tight_layout()
    _save(fig, fig_dir / "fig_medqa_logit_shift_distributions.png")


# ---------------------------------------------------------------------------
# fig_mmlu_supercategory_heatmap
# ---------------------------------------------------------------------------

def fig_mmlu_supercategory_heatmap(
    mmlu_agg: dict[str, Any],
    fig_dir: Path,
) -> None:
    """Rows = vectors, columns = supercategories, color = accuracy delta."""
    supercats = ["STEM", "humanities", "social_sciences", "other"]

    vecs = sorted(set(
        r["steering_vector"] for r in mmlu_agg.values()
        if r.get("direction") == "debiasing"
    ))
    if not vecs:
        return

    matrix = np.full((len(vecs), len(supercats)), np.nan)
    for vi, vec in enumerate(vecs):
        result = mmlu_agg.get(f"{vec}_debiasing", {})
        for si, sc in enumerate(supercats):
            sc_data = result.get("per_supercategory", {}).get(sc, {})
            if "accuracy_delta" in sc_data:
                matrix[vi, si] = sc_data["accuracy_delta"]

    fig, ax = plt.subplots(figsize=(6, max(4, len(vecs) * 0.4)))
    vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 0.05)
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    for vi in range(len(vecs)):
        for si in range(len(supercats)):
            val = matrix[vi, si]
            if not np.isnan(val):
                color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(si, vi, f"{val:.3f}", ha="center", va="center",
                        fontsize=6, color=color)

    ax.set_xticks(range(len(supercats)))
    ax.set_xticklabels(supercats, fontsize=8)
    ax.set_yticks(range(len(vecs)))
    ax.set_yticklabels(vecs, fontsize=6)
    ax.set_title("Capability tax across MMLU supercategories")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Accuracy delta")

    fig.tight_layout()
    _save(fig, fig_dir / "fig_mmlu_supercategory_heatmap.png")


# ---------------------------------------------------------------------------
# fig_side_effect_heatmap
# ---------------------------------------------------------------------------

def fig_side_effect_heatmap(
    medqa_agg: dict[str, Any],
    mmlu_agg: dict[str, Any],
    fig_dir: Path,
) -> None:
    """Rows = vectors, columns = side-effect domains, color = accuracy delta."""
    columns = ["MedQA\nno-demo", "MMLU\nSTEM", "MMLU\nhumanities",
                "MMLU\nsocial_sci", "MMLU\nother"]

    vecs = sorted(set(
        r["steering_vector"]
        for agg in [medqa_agg, mmlu_agg]
        for r in agg.values()
        if r.get("direction") == "debiasing"
    ))
    if not vecs:
        return

    matrix = np.full((len(vecs), len(columns)), np.nan)
    for vi, vec in enumerate(vecs):
        # MedQA no-demographic
        medqa_r = medqa_agg.get(f"{vec}_debiasing", {})
        no_demo = medqa_r.get("per_condition", {}).get("no_demographic", {})
        if "accuracy_delta" in no_demo:
            matrix[vi, 0] = no_demo["accuracy_delta"]

        # MMLU supercategories
        mmlu_r = mmlu_agg.get(f"{vec}_debiasing", {})
        for si, sc in enumerate(["STEM", "humanities", "social_sciences", "other"]):
            sc_data = mmlu_r.get("per_supercategory", {}).get(sc, {})
            if "accuracy_delta" in sc_data:
                matrix[vi, si + 1] = sc_data["accuracy_delta"]

    fig, ax = plt.subplots(figsize=(7, max(4, len(vecs) * 0.4)))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-0.1, vmax=0.1, aspect="auto")

    for vi in range(len(vecs)):
        for ci in range(len(columns)):
            val = matrix[vi, ci]
            if not np.isnan(val):
                color = "white" if abs(val) > 0.06 else "black"
                ax.text(ci, vi, f"{val:.3f}", ha="center", va="center",
                        fontsize=6, color=color)

    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, fontsize=7)
    ax.set_yticks(range(len(vecs)))
    ax.set_yticklabels(vecs, fontsize=6)
    ax.set_title("Side effects on unrelated content (debiasing direction)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Accuracy delta")

    fig.tight_layout()
    _save(fig, fig_dir / "fig_side_effect_heatmap.png")


# ---------------------------------------------------------------------------
# fig_bbq_vs_medqa_debiasing / exacerbation
# ---------------------------------------------------------------------------

def _fig_bbq_vs_medqa(
    manifests_path: Path,
    fig_dir: Path,
    y_field: str,
    y_label: str,
    title: str,
    filename: str,
) -> None:
    """Scatter: X = BBQ RCR_1.0, Y = MedQA accuracy delta."""
    if not manifests_path.exists():
        return
    with open(manifests_path) as f:
        manifests = json.load(f)

    viable = [m for m in manifests if m.get("steering_viable")]
    points = []
    for m in viable:
        x_val = m.get("optimal_rcr_1.0")
        y_val = m.get(y_field)
        if x_val is not None and y_val is not None:
            points.append({
                "x": x_val, "y": y_val,
                "cat": m["category"], "sub": m["subgroup"],
            })

    if len(points) < 3:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    for p in points:
        ax.scatter(
            p["x"], p["y"],
            c=CATEGORY_COLORS.get(p["cat"], "#666"),
            s=40, alpha=0.7,
        )
        ax.annotate(p["sub"], (p["x"], p["y"]), fontsize=5,
                     textcoords="offset points", xytext=(3, 3))

    x_arr = np.array([p["x"] for p in points])
    y_arr = np.array([p["y"] for p in points])
    if len(x_arr) >= 5:
        from scipy.stats import linregress
        slope, intercept, r_value, _, _ = linregress(x_arr, y_arr)
        x_line = np.linspace(x_arr.min(), x_arr.max(), 50)
        ax.plot(x_line, slope * x_line + intercept, "k--", linewidth=1)
        ax.annotate(f"r = {r_value:.3f}", (0.05, 0.95),
                     xycoords="axes fraction", fontsize=8, va="top")

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("BBQ RCR₁.₀ (debiasing effectiveness)")
    ax.set_ylabel(y_label)
    ax.set_title(title)

    fig.tight_layout()
    _save(fig, fig_dir / filename)


def fig_bbq_vs_medqa_debiasing(
    manifests_path: Path, fig_dir: Path,
) -> None:
    _fig_bbq_vs_medqa(
        manifests_path, fig_dir,
        y_field="medqa_matched_debias_delta",
        y_label="MedQA matched accuracy delta (debiasing)",
        title="BBQ debiasing vs MedQA matched effect",
        filename="fig_bbq_vs_medqa_debiasing.png",
    )


def fig_bbq_vs_medqa_exacerbation(
    manifests_path: Path, fig_dir: Path,
) -> None:
    _fig_bbq_vs_medqa(
        manifests_path, fig_dir,
        y_field="medqa_matched_exac_delta",
        y_label="MedQA matched accuracy delta (exacerbation)",
        title="BBQ debiasing vs MedQA exacerbation effect",
        filename="fig_bbq_vs_medqa_exacerbation.png",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_c3_figures(
    output_dir: Path,
    viable_vectors: list[dict[str, Any]],
) -> None:
    """Generate all C3 figures."""
    fig_dir = ensure_dir(output_dir / "figures")
    log("\nGenerating C3 figures...")

    # Load aggregated results
    medqa_agg: dict[str, Any] = {}
    mmlu_agg: dict[str, Any] = {}

    medqa_agg_path = output_dir / "medqa" / "aggregated_results.json"
    if medqa_agg_path.exists():
        with open(medqa_agg_path) as f:
            medqa_agg = json.load(f)

    mmlu_agg_path = output_dir / "mmlu" / "aggregated_results.json"
    if mmlu_agg_path.exists():
        with open(mmlu_agg_path) as f:
            mmlu_agg = json.load(f)

    if medqa_agg:
        fig_medqa_conditions_comparison(medqa_agg, fig_dir)
        fig_medqa_debias_vs_exacerbation(medqa_agg, fig_dir)
        fig_medqa_logit_shift_distributions(output_dir, fig_dir)

    if mmlu_agg:
        fig_mmlu_supercategory_heatmap(mmlu_agg, fig_dir)

    if medqa_agg and mmlu_agg:
        fig_side_effect_heatmap(medqa_agg, mmlu_agg, fig_dir)

    manifests_path = output_dir / "manifests_with_generalization.json"
    fig_bbq_vs_medqa_debiasing(manifests_path, fig_dir)
    fig_bbq_vs_medqa_exacerbation(manifests_path, fig_dir)

    log("C3 figures complete")
