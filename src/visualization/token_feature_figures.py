"""C4 figures: neuronpedia-style feature cards, token ranking grids,
per-template heatmaps, identity token specificity.

Uses the Wong colorblind-safe palette throughout.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

CATEGORY_COLORS: dict[str, str] = {
    "so": WONG["orange"], "gi": WONG["blue"], "race": WONG["green"],
    "religion": WONG["purple"], "disability": WONG["vermillion"],
    "age": WONG["sky_blue"], "ses": WONG["yellow"],
    "nationality": WONG["black"], "physical_appearance": "#999999",
}

DPI = 150


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log(f"    Saved {path.name}")


def _fkey(layer: int, feature_idx: int) -> str:
    return f"L{layer:02d}_F{feature_idx}"


def _clean_token(tok: str, max_len: int = 20) -> str:
    """Clean BPE token for display."""
    tok = tok.replace("\u2581", " ").replace("\u0120", " ").strip()
    if len(tok) > max_len:
        tok = tok[:max_len] + "..."
    return tok if tok else "<empty>"


# ---------------------------------------------------------------------------
# fig_feature_card (per feature)
# ---------------------------------------------------------------------------

def fig_feature_card(
    layer: int,
    feature_idx: int,
    logit_effects_df: pd.DataFrame,
    density_data: dict[str, Any],
    token_ranking_path: Path,
    top_examples_path: Path,
    subs_using: list[str],
    fig_dir: Path,
) -> None:
    """Neuronpedia-style per-feature card with four panels."""
    fk = _fkey(layer, feature_idx)
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

    # Panel A: Logit effect decomposition
    ax_logit = fig.add_subplot(gs[0, 0])
    feat_logit = logit_effects_df[
        (logit_effects_df["layer"] == layer)
        & (logit_effects_df["feature_idx"] == feature_idx)
    ]
    if not feat_logit.empty:
        pos = feat_logit[feat_logit["direction"] == "positive"].sort_values("rank")
        neg = feat_logit[feat_logit["direction"] == "negative"].sort_values("rank")

        y_pos = np.arange(len(pos))
        ax_logit.barh(y_pos, pos["logit_contribution"].values,
                      color=WONG["green"], alpha=0.7)
        ax_logit.set_yticks(y_pos)
        ax_logit.set_yticklabels(
            [_clean_token(t) for t in pos["token_str"].values],
            fontsize=6,
        )
        ax_logit.invert_yaxis()
        ax_logit.set_xlabel("Logit contribution")
        ax_logit.set_title("Top promoted tokens", fontsize=9)

    # Panel A2: Negative logit effects
    ax_neg = fig.add_subplot(gs[0, 1])
    if not feat_logit.empty and not neg.empty:
        y_neg = np.arange(len(neg))
        ax_neg.barh(y_neg, neg["logit_contribution"].values,
                     color=WONG["vermillion"], alpha=0.7)
        ax_neg.set_yticks(y_neg)
        ax_neg.set_yticklabels(
            [_clean_token(t) for t in neg["token_str"].values],
            fontsize=6,
        )
        ax_neg.invert_yaxis()
        ax_neg.set_xlabel("Logit contribution")
        ax_neg.set_title("Top suppressed tokens", fontsize=9)

    # Panel B: Activation density histogram
    ax_hist = fig.add_subplot(gs[1, 0])
    if density_data and density_data.get("histogram_counts"):
        edges = np.array(density_data["histogram_bin_edges"])
        counts = np.array(density_data["histogram_counts"])
        centres = (edges[:-1] + edges[1:]) / 2
        ax_hist.bar(centres, counts, width=np.diff(edges), color=WONG["orange"],
                    alpha=0.7, edgecolor="none")
        ax_hist.set_xlabel("Activation magnitude")
        ax_hist.set_ylabel("Count")
        density_pct = density_data.get("density", 0) * 100
        ax_hist.set_title(f"Activation density: {density_pct:.3f}%", fontsize=9)

    # Panel C: Top token rankings (filtered)
    ax_tok = fig.add_subplot(gs[1, 1])
    if token_ranking_path.exists():
        tok_df = pd.read_parquet(token_ranking_path)
        filtered = tok_df[~tok_df["is_template_string"]].head(15)
        if not filtered.empty:
            y = np.arange(len(filtered))
            colors = []
            for _, row in filtered.iterrows():
                if row.get("is_identity_term", False):
                    colors.append(WONG["blue"])
                else:
                    colors.append("#999999")

            ax_tok.barh(y, filtered["mean_activation_nonzero"].values,
                        color=colors)
            labels = [
                f"{_clean_token(t)} (n={n})"
                for t, n in zip(
                    filtered["token"].values,
                    filtered["n_nonzero"].values,
                )
            ]
            ax_tok.set_yticks(y)
            ax_tok.set_yticklabels(labels, fontsize=6)
            ax_tok.invert_yaxis()
            ax_tok.set_xlabel("Mean activation (nonzero)")
            ax_tok.set_title("Top tokens (content only)", fontsize=9)

    # Panel D: Top activating examples (text snippets)
    ax_ex = fig.add_subplot(gs[2, :])
    ax_ex.axis("off")
    if top_examples_path.exists():
        with open(top_examples_path) as f:
            examples = json.load(f)
        text_lines = []
        for ex in examples[:5]:
            preview = ex.get("prompt_preview", "")[:150]
            max_act = ex.get("max_activation", 0)
            argmax_tok = _clean_token(ex.get("argmax_token", ""))
            text_lines.append(
                f"[act={max_act:.2f}] token=\"{argmax_tok}\"  {preview}"
            )
        text = "\n".join(text_lines) if text_lines else "(no examples)"
        ax_ex.text(
            0, 1, text, transform=ax_ex.transAxes,
            fontsize=7, verticalalignment="top", fontfamily="monospace",
            wrap=True,
        )
        ax_ex.set_title("Top activating examples", fontsize=9)

    subs_str = ", ".join(subs_using[:5])
    if len(subs_using) > 5:
        subs_str += f" +{len(subs_using)-5} more"
    fig.suptitle(f"Feature {fk} — used by {subs_str}", fontsize=12)

    _save(fig, fig_dir / f"fig_feature_card_{fk}.png")


# ---------------------------------------------------------------------------
# fig_token_rankings_{category}
# ---------------------------------------------------------------------------

def fig_token_rankings_category(
    category: str,
    feature_manifest: list[dict[str, Any]],
    output_dir: Path,
    fig_dir: Path,
) -> None:
    """Grid of small token-ranking bar charts for all features in a category."""
    cat_entries = [e for e in feature_manifest if e["category"] == category]
    if not cat_entries:
        return

    # Deduplicate by (subgroup, layer, feature_idx)
    seen: set[tuple[str, int, int]] = set()
    unique_entries: list[dict[str, Any]] = []
    for e in cat_entries:
        key = (e["subgroup"], e["layer"], e["feature_idx"])
        if key not in seen:
            seen.add(key)
            unique_entries.append(e)

    n = len(unique_entries)
    n_cols = min(4, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows), squeeze=False,
    )

    for idx, entry in enumerate(unique_entries):
        ax = axes[idx // n_cols][idx % n_cols]
        fk = _fkey(entry["layer"], entry["feature_idx"])
        path = output_dir / "token_rankings" / f"{fk}.parquet"

        if not path.exists():
            ax.set_visible(False)
            continue

        tok_df = pd.read_parquet(path)
        filtered = tok_df[~tok_df["is_template_string"]].head(10)
        if filtered.empty:
            ax.text(0.5, 0.5, "No content tokens", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
            ax.set_title(f"{entry['subgroup']}: {fk}", fontsize=8)
            continue

        y = np.arange(len(filtered))
        colors = [
            WONG["blue"] if row.get("is_identity_term", False) else "#999999"
            for _, row in filtered.iterrows()
        ]
        ax.barh(y, filtered["mean_activation_nonzero"].values, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(
            [_clean_token(t, 15) for t in filtered["token"].values],
            fontsize=5,
        )
        ax.invert_yaxis()
        ax.set_xlabel("Mean act (nz)", fontsize=7)
        ax.set_title(f"{entry['subgroup']}: {fk}", fontsize=8)

    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle(f"Token rankings — {category}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, fig_dir / f"fig_token_rankings_{category}.png")


# ---------------------------------------------------------------------------
# fig_per_template_heatmap_{category}
# ---------------------------------------------------------------------------

def fig_per_template_heatmap_category(
    category: str,
    feature_manifest: list[dict[str, Any]],
    output_dir: Path,
    fig_dir: Path,
) -> None:
    """Heatmap: rows = features, columns = question_index, color = mean_max_act."""
    cat_entries = [e for e in feature_manifest if e["category"] == category]
    if not cat_entries:
        return

    # Collect data
    feat_labels: list[str] = []
    all_qidxs: set[int] = set()
    feat_data: list[dict[int, float]] = []

    seen: set[tuple[str, int, int]] = set()
    for e in cat_entries:
        key = (e["subgroup"], e["layer"], e["feature_idx"])
        if key in seen:
            continue
        seen.add(key)

        fk = _fkey(e["layer"], e["feature_idx"])
        path = output_dir / "per_template_rankings" / f"{fk}.parquet"
        if not path.exists():
            continue

        df = pd.read_parquet(path)
        qidx_to_act = dict(zip(df["question_index"], df["mean_max_activation"]))
        all_qidxs.update(qidx_to_act.keys())
        feat_data.append(qidx_to_act)
        feat_labels.append(f"{e['subgroup']}: {fk}")

    if not feat_data or not all_qidxs:
        return

    qidxs_sorted = sorted(all_qidxs)
    matrix = np.zeros((len(feat_labels), len(qidxs_sorted)))
    for fi, qmap in enumerate(feat_data):
        for qi, qidx in enumerate(qidxs_sorted):
            matrix[fi, qi] = qmap.get(qidx, 0.0)

    fig, ax = plt.subplots(
        figsize=(max(8, len(qidxs_sorted) * 0.3), max(4, len(feat_labels) * 0.4)),
    )
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(qidxs_sorted)))
    ax.set_xticklabels(qidxs_sorted, fontsize=5, rotation=90)
    ax.set_yticks(range(len(feat_labels)))
    ax.set_yticklabels(feat_labels, fontsize=6)
    ax.set_xlabel("question_index")
    ax.set_title(f"Per-template feature activations — {category}", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Mean max activation")

    fig.tight_layout()
    _save(fig, fig_dir / f"fig_per_template_heatmap_{category}.png")


# ---------------------------------------------------------------------------
# fig_identity_token_specificity
# ---------------------------------------------------------------------------

def fig_identity_token_specificity(
    feature_manifest: list[dict[str, Any]],
    output_dir: Path,
    fig_dir: Path,
) -> None:
    """For each (subgroup, feature): fraction of top-20 nonzero content tokens
    that are identity terms."""
    seen: set[tuple[str, int, int]] = set()
    bars: list[dict[str, Any]] = []

    for e in feature_manifest:
        key = (e["subgroup"], e["layer"], e["feature_idx"])
        if key in seen:
            continue
        seen.add(key)

        fk = _fkey(e["layer"], e["feature_idx"])
        path = output_dir / "token_rankings" / f"{fk}.parquet"
        if not path.exists():
            continue

        df = pd.read_parquet(path)
        filtered = df[~df["is_template_string"]].head(20)
        if filtered.empty:
            continue

        n_identity = int(filtered["is_identity_term"].sum())
        bars.append({
            "label": f"{e['subgroup']}/{fk}",
            "category": e["category"],
            "fraction_identity": n_identity / len(filtered),
            "n_identity": n_identity,
            "n_total": len(filtered),
        })

    if not bars:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(bars) * 0.5), 5))
    x = np.arange(len(bars))
    colors = [CATEGORY_COLORS.get(b["category"], "#666") for b in bars]
    fracs = [b["fraction_identity"] for b in bars]
    labels = [b["label"] for b in bars]

    ax.bar(x, fracs, color=colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5, rotation=60, ha="right")
    ax.set_ylabel("Fraction identity terms in top-20 tokens")
    ax.set_title("Identity token specificity per feature")

    fig.tight_layout()
    _save(fig, fig_dir / "fig_identity_token_specificity.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_c4_figures(
    output_dir: Path,
    feature_manifest: list[dict[str, Any]],
    logit_effects_df: pd.DataFrame,
    activation_densities: dict[str, Any],
    all_accumulator: dict[tuple[int, int], list[dict[str, Any]]],
    all_string_templates: dict[str, set[str]],
    viable_manifests: list[dict[str, Any]],
) -> None:
    """Generate all C4 figures."""
    fig_dir = ensure_dir(output_dir / "figures")
    log("\nGenerating C4 figures...")

    # Build reverse lookup: (layer, feat) -> list of subgroup labels
    feat_to_subs: dict[tuple[int, int], list[str]] = defaultdict(list)
    for e in feature_manifest:
        feat_to_subs[(e["layer"], e["feature_idx"])].append(
            f"{e['category']}/{e['subgroup']}"
        )

    # Per-feature cards
    processed: set[tuple[int, int]] = set()
    for e in feature_manifest:
        pair = (e["layer"], e["feature_idx"])
        if pair in processed:
            continue
        processed.add(pair)
        fk = _fkey(pair[0], pair[1])

        fig_feature_card(
            layer=pair[0],
            feature_idx=pair[1],
            logit_effects_df=logit_effects_df,
            density_data=activation_densities.get(fk, {}),
            token_ranking_path=output_dir / "token_rankings" / f"{fk}.parquet",
            top_examples_path=output_dir / "top_activating_examples" / f"{fk}.json",
            subs_using=sorted(set(feat_to_subs[pair])),
            fig_dir=fig_dir,
        )

    # Per-category token ranking grids
    categories = sorted(set(e["category"] for e in feature_manifest))
    for cat in categories:
        fig_token_rankings_category(cat, feature_manifest, output_dir, fig_dir)
        fig_per_template_heatmap_category(cat, feature_manifest, output_dir, fig_dir)

    # Identity token specificity
    fig_identity_token_specificity(feature_manifest, output_dir, fig_dir)

    log("C4 figures complete")
