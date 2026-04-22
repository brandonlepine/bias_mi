"""B2 core: feature ranking per subgroup, injection layer computation, overlap analysis.

Collects FDR-significant features from B1, ranks by effect size, determines
effect-weighted injection layers, and computes cross-subgroup Jaccard overlap.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log, progress_bar


# ---------------------------------------------------------------------------
# Identity key helpers
# ---------------------------------------------------------------------------

def make_sub_key(category: str, subgroup: str) -> str:
    return f"{category}/{subgroup}"


def parse_sub_key(key: str) -> tuple[str, str]:
    cat, sub = key.split("/", 1)
    return cat, sub


# ---------------------------------------------------------------------------
# Step 1: Load all significant features
# ---------------------------------------------------------------------------

def load_all_significant(run_dir: Path, n_layers: int) -> pd.DataFrame:
    """Load all FDR-significant features across all B1 layers."""
    differential_dir = run_dir / "B_differential"
    dfs: list[pd.DataFrame] = []

    for layer in progress_bar(range(n_layers), desc="Loading B1 layers", unit="layer"):
        parquet_path = differential_dir / f"layer_{layer:02d}.parquet"
        if not parquet_path.exists():
            continue
        df = pd.read_parquet(parquet_path)
        sig = df[df["is_significant"]].copy()
        dfs.append(sig)

    if not dfs:
        raise SystemExit("No B1 parquets found. Run B1 first.")

    combined = pd.concat(dfs, ignore_index=True)

    # Defensive normalization.
    combined["subgroup"] = combined["subgroup"].astype(str).str.strip().str.lower()
    combined["category"] = combined["category"].astype(str).str.strip().str.lower()

    log(f"Loaded {len(combined)} significant features across {len(dfs)} layers")
    return combined


# ---------------------------------------------------------------------------
# Step 2: Defensive deduplication
# ---------------------------------------------------------------------------

def deduplicate_defensive(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate (category, subgroup, feature_idx, layer) rows if any."""
    before = len(df)
    df = df.assign(abs_d=df["cohens_d"].abs())
    df = df.sort_values("abs_d", ascending=False)
    df = df.drop_duplicates(
        subset=["category", "subgroup", "feature_idx", "layer"], keep="first",
    )
    df = df.drop(columns=["abs_d"])
    after = len(df)
    if before != after:
        log(f"  WARNING: deduplication removed {before - after} duplicate rows")
    return df


# ---------------------------------------------------------------------------
# Step 3: Enumerate subgroups
# ---------------------------------------------------------------------------

def enumerate_subgroups(
    combined_df: pd.DataFrame,
    b1_summary: dict[str, Any],
) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    """Enumerate subgroups with significant features, cross-referencing B1 catalog."""
    in_parquet: set[tuple[str, str]] = set()
    for _, row in combined_df[["category", "subgroup"]].drop_duplicates().iterrows():
        in_parquet.add((row["category"], row["subgroup"]))

    in_catalog: set[tuple[str, str]] = set()
    catalog = b1_summary.get("subgroup_catalog", {})
    for sub_key, entry in catalog.items():
        if entry.get("analyzable"):
            sub_norm = str(sub_key).strip().lower()
            in_catalog.add((entry["category"], sub_norm))

    report: dict[str, Any] = {
        "total_subgroups": len(in_parquet | in_catalog),
        "in_both": sorted(make_sub_key(c, s) for c, s in in_parquet & in_catalog),
        "only_in_parquet": sorted(make_sub_key(c, s) for c, s in in_parquet - in_catalog),
        "only_in_catalog": sorted(make_sub_key(c, s) for c, s in in_catalog - in_parquet),
    }

    if report["only_in_parquet"]:
        log(f"  WARNING: subgroups in parquet but not B1 catalog: "
            f"{report['only_in_parquet']}")
    if report["only_in_catalog"]:
        log(f"  NOTE: subgroups in B1 catalog but no significant features: "
            f"{report['only_in_catalog']}")

    return sorted(in_parquet | in_catalog), report


# ---------------------------------------------------------------------------
# Step 4: Rank features per subgroup
# ---------------------------------------------------------------------------

def rank_features_all(
    combined_df: pd.DataFrame,
    subgroups: list[tuple[str, str]],
) -> pd.DataFrame:
    """Rank all significant features per (category, subgroup, direction)."""
    ranked_rows: list[dict[str, Any]] = []

    for cat, sub in progress_bar(subgroups, desc="Ranking features", unit="sub"):
        sub_df = combined_df[
            (combined_df["category"] == cat) & (combined_df["subgroup"] == sub)
        ]
        for direction in ["s_marking", "other_marking"]:
            dir_df = sub_df[sub_df["direction"] == direction].copy()
            if dir_df.empty:
                continue

            dir_df["abs_d"] = dir_df["cohens_d"].abs()
            dir_df = dir_df.sort_values("abs_d", ascending=False).reset_index(drop=True)
            dir_df["rank"] = dir_df.index + 1

            for _, row in dir_df.iterrows():
                ranked_rows.append({
                    "category": cat,
                    "subgroup": sub,
                    "direction": direction,
                    "rank": int(row["rank"]),
                    "feature_idx": int(row["feature_idx"]),
                    "layer": int(row["layer"]),
                    "cohens_d": float(row["cohens_d"]),
                    "p_value_raw": float(row["p_value_raw"]),
                    "p_value_fdr": float(row["p_value_fdr"]),
                    "firing_rate_targeting": float(row["firing_rate_targeting"]),
                    "firing_rate_not_targeting": float(row["firing_rate_not_targeting"]),
                    "mean_activation_targeting": float(row["mean_activation_targeting"]),
                    "mean_activation_not_targeting": float(row["mean_activation_not_targeting"]),
                    "n_targeting": int(row["n_targeting"]),
                    "n_not_targeting": int(row["n_not_targeting"]),
                })

    return pd.DataFrame(ranked_rows)


# ---------------------------------------------------------------------------
# Step 5: Injection layers
# ---------------------------------------------------------------------------

def compute_injection_layer_weighted(features: pd.DataFrame) -> dict[str, Any] | None:
    """Determine injection layer from effect-weighted sum across all significant features."""
    if features.empty:
        return None

    layer_scores = features.groupby("layer")["cohens_d"].apply(
        lambda x: float(np.sum(np.abs(x)))
    ).to_dict()

    if not layer_scores:
        return None

    max_score = max(layer_scores.values())
    total_score = sum(layer_scores.values())

    # Ties broken by preferring deeper layers.
    candidates = [l for l, s in layer_scores.items() if s == max_score]
    injection_layer = max(candidates)

    return {
        "injection_layer": int(injection_layer),
        "layer_scores": {int(k): round(v, 4) for k, v in layer_scores.items()},
        "n_features": int(len(features)),
        "top_layer_score": round(float(max_score), 4),
        "score_concentration": round(
            float(max_score / max(total_score, 1e-12)), 4,
        ),
    }


def build_injection_layers(
    ranked_df: pd.DataFrame,
    subgroups: list[tuple[str, str]],
) -> dict[str, dict[str, Any]]:
    """Build injection layer records for all subgroups."""
    results: dict[str, dict[str, Any]] = {}

    for cat, sub in subgroups:
        key = make_sub_key(cat, sub)
        pro = ranked_df[
            (ranked_df["category"] == cat)
            & (ranked_df["subgroup"] == sub)
            & (ranked_df["direction"] == "s_marking")
        ]
        anti = ranked_df[
            (ranked_df["category"] == cat)
            & (ranked_df["subgroup"] == sub)
            & (ranked_df["direction"] == "other_marking")
        ]

        record: dict[str, Any] = {
            "category": cat,
            "subgroup": sub,
            "s_marking": compute_injection_layer_weighted(pro),
            "other_marking": compute_injection_layer_weighted(anti),
        }
        if record["s_marking"] is None:
            record["s_marking_note"] = "No significant pro-bias features"
        if record["other_marking"] is None:
            record["other_marking_note"] = "No significant anti-bias features"

        results[key] = record

    return results


# ---------------------------------------------------------------------------
# Step 6: Cross-subgroup feature overlap (Jaccard curves)
# ---------------------------------------------------------------------------

K_VALUES_DEFAULT = [5, 10, 20, 50, 100, 200]


def compute_overlap_curve(
    ranked_A: pd.DataFrame,
    ranked_B: pd.DataFrame,
    k_values: list[int],
) -> dict[int, dict[str, Any]]:
    """Compute Jaccard overlap at multiple k values for one pair."""
    results: dict[int, dict[str, Any]] = {}

    for k in k_values:
        set_A = set(
            (int(r["feature_idx"]), int(r["layer"]))
            for _, r in ranked_A.head(k).iterrows()
        )
        set_B = set(
            (int(r["feature_idx"]), int(r["layer"]))
            for _, r in ranked_B.head(k).iterrows()
        )

        if not set_A or not set_B:
            results[k] = {
                "jaccard": None,
                "directed_A_to_B": None,
                "directed_B_to_A": None,
                "n_shared": 0,
                "effective_k_A": len(set_A),
                "effective_k_B": len(set_B),
            }
            continue

        intersection = set_A & set_B
        union = set_A | set_B

        results[k] = {
            "jaccard": round(len(intersection) / len(union), 4),
            "directed_A_to_B": round(len(intersection) / len(set_A), 4),
            "directed_B_to_A": round(len(intersection) / len(set_B), 4),
            "n_shared": int(len(intersection)),
            "effective_k_A": int(len(set_A)),
            "effective_k_B": int(len(set_B)),
        }

    return results


def compute_all_overlaps(
    ranked_df: pd.DataFrame,
    subgroups: list[tuple[str, str]],
    k_values: list[int],
) -> dict[str, dict[str, Any]]:
    """Compute overlap curves for all subgroup pairs within each category."""
    by_category: dict[str, list[str]] = defaultdict(list)
    for cat, sub in subgroups:
        by_category[cat].append(sub)

    overlap_results: dict[str, dict[str, Any]] = {}

    for cat, subs_in_cat in by_category.items():
        if len(subs_in_cat) < 2:
            continue

        sorted_subs = sorted(subs_in_cat)
        overlap_results[cat] = {
            "subgroups": sorted_subs,
            "k_values": k_values,
            "s_marking": {},
            "other_marking": {},
        }

        for direction in ["s_marking", "other_marking"]:
            for i, sub_A in enumerate(sorted_subs):
                for sub_B in sorted_subs[i + 1 :]:
                    pair_key = f"{sub_A}__{sub_B}"

                    ranked_A = ranked_df[
                        (ranked_df["category"] == cat)
                        & (ranked_df["subgroup"] == sub_A)
                        & (ranked_df["direction"] == direction)
                    ].sort_values("rank")

                    ranked_B = ranked_df[
                        (ranked_df["category"] == cat)
                        & (ranked_df["subgroup"] == sub_B)
                        & (ranked_df["direction"] == direction)
                    ].sort_values("rank")

                    curve = compute_overlap_curve(ranked_A, ranked_B, k_values)
                    overlap_results[cat][direction][pair_key] = curve

    return overlap_results


# ---------------------------------------------------------------------------
# Step 7: Item overlap report
# ---------------------------------------------------------------------------

def compute_item_overlap(
    metadata_df: pd.DataFrame,
    subgroups: list[tuple[str, str]],
    structural_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute item-level overlap for subgroup pairs within each category."""
    ambig_df = metadata_df[metadata_df["context_condition"] == "ambig"].copy()
    ambig_df["stereotyped_groups"] = ambig_df["stereotyped_groups"].apply(
        lambda x: x if isinstance(x, list) else json.loads(x)
    )

    sub_items: dict[tuple[str, str], set[int]] = {}
    for cat, sub in subgroups:
        items = ambig_df[
            (ambig_df["category"] == cat)
            & (ambig_df["stereotyped_groups"].apply(lambda gs: sub in gs))
        ]["item_idx"].tolist()
        sub_items[(cat, sub)] = set(items)

    by_category: dict[str, list[str]] = defaultdict(list)
    for cat, sub in subgroups:
        by_category[cat].append(sub)

    result: dict[str, Any] = {
        "structural_threshold": structural_threshold,
        "per_category": {},
    }

    for cat, subs_in_cat in by_category.items():
        if len(subs_in_cat) < 2:
            continue

        sorted_subs = sorted(subs_in_cat)
        cat_result: dict[str, Any] = {
            "subgroups": sorted_subs,
            "item_counts": {s: len(sub_items[(cat, s)]) for s in sorted_subs},
            "pairwise": {},
            "structural_pairs": [],
        }

        for i, sub_A in enumerate(sorted_subs):
            for sub_B in sorted_subs[i + 1 :]:
                items_A = sub_items[(cat, sub_A)]
                items_B = sub_items[(cat, sub_B)]
                shared = items_A & items_B

                jaccard = len(shared) / max(len(items_A | items_B), 1)
                frac_A_in_B = len(shared) / max(len(items_A), 1)
                frac_B_in_A = len(shared) / max(len(items_B), 1)
                max_frac = max(frac_A_in_B, frac_B_in_A)

                pair_key = f"{sub_A}__{sub_B}"
                pair_entry = {
                    "n_shared": int(len(shared)),
                    "jaccard": round(jaccard, 4),
                    "fraction_of_A_in_B": round(frac_A_in_B, 4),
                    "fraction_of_B_in_A": round(frac_B_in_A, 4),
                    "max_fraction": round(max_frac, 4),
                    "is_structural": bool(max_frac >= structural_threshold),
                }
                cat_result["pairwise"][pair_key] = pair_entry

                if pair_entry["is_structural"]:
                    cat_result["structural_pairs"].append(pair_key)

        if cat_result["structural_pairs"]:
            log(f"  {cat}: structural item overlap in pairs "
                f"{cat_result['structural_pairs']}")

        result["per_category"][cat] = cat_result

    return result


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def build_ranking_summary(
    n_layers: int,
    subgroups: list[tuple[str, str]],
    ranked_df: pd.DataFrame,
    injection_layers: dict[str, dict[str, Any]],
    enumeration_report: dict[str, Any],
    k_values: list[int],
    structural_threshold: float,
    elapsed: float,
) -> dict[str, Any]:
    """Build the top-level ranking_summary.json."""
    n_pro = sum(
        1 for _, v in injection_layers.items() if v.get("s_marking") is not None
    )
    n_anti = sum(
        1 for _, v in injection_layers.items() if v.get("other_marking") is not None
    )
    no_features = [
        make_sub_key(c, s) for c, s in subgroups
        if make_sub_key(c, s) not in injection_layers
        or (
            injection_layers[make_sub_key(c, s)].get("s_marking") is None
            and injection_layers[make_sub_key(c, s)].get("other_marking") is None
        )
    ]

    # Per-category summary.
    by_cat: dict[str, list[str]] = defaultdict(list)
    for cat, sub in subgroups:
        by_cat[cat].append(sub)

    per_cat_summary: dict[str, dict[str, Any]] = {}
    for cat, subs in sorted(by_cat.items()):
        cat_ranked = ranked_df[ranked_df["category"] == cat]
        n_subs = len(subs)
        n_pro_cat = sum(
            1 for s in subs
            if injection_layers.get(make_sub_key(cat, s), {}).get("s_marking") is not None
        )

        pro_counts = []
        anti_counts = []
        inj_layers_pro: dict[int, int] = defaultdict(int)
        for s in subs:
            key = make_sub_key(cat, s)
            rec = injection_layers.get(key, {})
            pro = cat_ranked[
                (cat_ranked["subgroup"] == s) & (cat_ranked["direction"] == "s_marking")
            ]
            anti = cat_ranked[
                (cat_ranked["subgroup"] == s) & (cat_ranked["direction"] == "other_marking")
            ]
            pro_counts.append(len(pro))
            anti_counts.append(len(anti))
            if rec.get("s_marking") and rec["s_marking"].get("injection_layer") is not None:
                inj_layers_pro[rec["s_marking"]["injection_layer"]] += 1

        per_cat_summary[cat] = {
            "n_subgroups": n_subs,
            "n_subgroups_with_s_marking": n_pro_cat,
            "mean_n_s_marking_features": round(float(np.mean(pro_counts)), 0) if pro_counts else 0,
            "mean_n_other_marking_features": round(float(np.mean(anti_counts)), 0) if anti_counts else 0,
            "injection_layer_distribution_s_marking": dict(sorted(inj_layers_pro.items())),
        }

    return {
        "n_layers_loaded": n_layers,
        "n_subgroups_total": len(subgroups),
        "n_subgroups_with_s_marking": n_pro,
        "n_subgroups_with_other_marking": n_anti,
        "n_subgroups_no_features": len(no_features),
        "subgroups_no_features": no_features,
        "k_values_used": k_values,
        "structural_threshold": structural_threshold,
        "enumeration_report": enumeration_report,
        "per_category_summary": per_cat_summary,
        "total_runtime_seconds": round(elapsed, 1),
    }
