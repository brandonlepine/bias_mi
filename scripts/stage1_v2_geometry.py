"""Stage 1 v2: Role / Mention / Bias-Mediation Decomposition.

Computes THREE distinct DIM directions per subgroup using enriched metadata:
  1. DIM_mention(S)        — S is mentioned anywhere in the prompt
  2. DIM_role(S)           — S is the bias target, given S is mentioned
  3. DIM_bias_response(S)  — model gave a bias-aligned response on S-targeted items

Outputs go to {run_dir}/stage1_v2_geometry/ (v1 outputs are untouched).

No model loading, no SAE, no forward passes — pure matrix math on cached
Phase A activations.

Usage:
    python scripts/stage1_v2_geometry.py --run_dir runs/llama-3.1-8b_2026-04-22/
    python scripts/stage1_v2_geometry.py --run_dir ... --categories so
    python scripts/stage1_v2_geometry.py --run_dir ... --skip_figures
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import atomic_save_json

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WONG = {
    "orange": "#E69F00", "sky_blue": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00",
    "purple": "#CC79A7", "black": "#000000",
}
DIRECTION_TYPE_COLORS = {
    "mention": WONG["blue"],
    "role": WONG["orange"],
    "bias_response": WONG["green"],
}
ALIGNMENT_PAIR_COLORS = {
    "mention_vs_role": WONG["blue"],
    "mention_vs_bias_response": WONG["orange"],
    "role_vs_bias_response": WONG["green"],
}
DPI = 200

# Short name ↔ BBQ category name
CATEGORY_FILE_MAP: dict[str, str] = {
    "age": "Age",
    "disability": "Disability_status",
    "gi": "Gender_identity",
    "nationality": "Nationality",
    "physical_appearance": "Physical_appearance",
    "race": "Race_ethnicity",
    "religion": "Religion",
    "ses": "SES",
    "so": "Sexual_orientation",
}
BBQ_TO_SHORT: dict[str, str] = {v: k for k, v in CATEGORY_FILE_MAP.items()}

# Categories that need dual granularity
DUAL_GRANULARITY_CATEGORIES: set[str] = {"nationality", "disability"}

DIRECTION_TYPES: list[str] = ["mention", "role", "bias_response"]


def log(msg: str) -> None:
    print(f"[stage1v2] {msg}", flush=True)


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 1 v2: Role / Mention / Bias-Mediation Decomposition",
    )
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--layer", type=int, default=14)
    p.add_argument("--categories", type=str, default=None,
                   help="Comma-separated subset of categories (short names)")
    p.add_argument("--min_n_per_group", type=int, default=10)
    p.add_argument("--n_bootstrap", type=int, default=0,
                   help="Bootstrap iterations for CIs; default 0 (skip). "
                        "Set to 100 only when CIs are needed for final figures.")
    p.add_argument("--n_cv_folds", type=int, default=5)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--probe_max_iter", type=int, default=200,
                   help="max_iter for LogisticRegression")
    p.add_argument("--race_intersectional_threshold", type=float, default=0.10,
                   help="If race-only items < threshold, strip gender prefix")
    p.add_argument("--skip_figures", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _resolve_category_name(cat: str) -> str:
    """Given either a short name or BBQ name, return the short name."""
    if cat in CATEGORY_FILE_MAP:
        return cat
    if cat in BBQ_TO_SHORT:
        return BBQ_TO_SHORT[cat]
    return cat


def _ensure_list(val: Any) -> list[str]:
    """Coerce a possibly-serialised list column to list[str], lowercase.

    Handles: Python lists, numpy arrays, pyarrow ListScalars, JSON strings,
    ast-parsable strings, and bare string values.
    """
    # Handle pyarrow scalars (pandas ArrowDtype backend)
    if hasattr(val, "as_py"):
        val = val.as_py()
    # Handle None / NaN
    if val is None:
        return []
    if isinstance(val, float) and np.isnan(val):
        return []
    # Handle numpy arrays (parquet list columns round-trip as ndarray)
    if isinstance(val, np.ndarray):
        return [str(s).strip().lower() for s in val if s]
    # Handle Python lists / tuples
    if isinstance(val, (list, tuple)):
        return [str(s).strip().lower() for s in val if s]
    # Handle JSON-serialised strings (e.g. '["gay", "lesbian"]')
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
        except (json.JSONDecodeError, ValueError):
            try:
                parsed = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                return [val.strip().lower()] if val.strip() else []
        if isinstance(parsed, (list, tuple)):
            return [str(s).strip().lower() for s in parsed if s]
        return [str(parsed).strip().lower()] if parsed else []
    return []


def load_enriched_metadata(run_dir: Path) -> pd.DataFrame:
    """Load metadata_enriched.parquet with list-column deserialization."""
    meta_path = run_dir / "A_extraction" / "metadata_enriched.parquet"
    if not meta_path.exists():
        log(f"FATAL: metadata_enriched.parquet not found at {meta_path}")
        log("Run enrich_metadata.py first (Prompt 1).")
        sys.exit(1)

    df = pd.read_parquet(meta_path)

    # Diagnostic: report raw types before parsing
    for col in ["stereotyped_groups", "mentioned_subgroups"]:
        if col in df.columns:
            sample_val = df[col].dropna().iloc[0] if df[col].notna().any() else None
            log(f"  {col} raw dtype={df[col].dtype}, "
                f"sample type={type(sample_val).__name__}, "
                f"sample={sample_val!r}")

    df["stereotyped_groups"] = df["stereotyped_groups"].apply(_ensure_list)
    df["mentioned_subgroups"] = df["mentioned_subgroups"].apply(_ensure_list)

    # Normalize category to short names for activation directory lookup
    df["category_short"] = df["category"].apply(_resolve_category_name)

    log(f"Loaded metadata_enriched: {len(df)} rows, "
        f"categories={sorted(df['category_short'].unique())}")
    return df


def load_layer_hidden_states(
    run_dir: Path, categories_short: list[str], layer: int,
) -> dict[int, np.ndarray]:
    """Load last-token raw hidden states at the given layer for all items."""
    result: dict[int, np.ndarray] = {}
    for cat in categories_short:
        cat_dir = run_dir / "A_extraction" / "activations" / cat
        if not cat_dir.exists():
            log(f"  WARNING: missing activations dir: {cat_dir}")
            continue
        items = sorted(cat_dir.glob("item_*.npz"))
        for item_path in tqdm(items, desc=f"  Loading {cat}", unit="items",
                              leave=False):
            item_idx = int(item_path.stem.split("_")[1])
            try:
                data = np.load(item_path, allow_pickle=True)
                hs_normed = data["hidden_states"][layer].astype(np.float32)
                raw_norm = float(data["hidden_states_raw_norms"][layer])
                result[item_idx] = hs_normed * raw_norm
            except Exception as e:
                log(f"    Failed to load {item_path.name}: {e}")
    return result


def load_question_index_from_bbq(
    bbq_data_dir: Path, categories_short: list[str],
) -> dict[str, dict[int, int]]:
    """Load question_index from BBQ JSONL, keyed by (category_short, example_id).

    Returns {category_short: {example_id: question_index}}.
    """
    result: dict[str, dict[int, int]] = {}
    for cat in categories_short:
        bbq_name = CATEGORY_FILE_MAP.get(cat)
        if not bbq_name:
            continue
        path = bbq_data_dir / f"{bbq_name}.jsonl"
        if not path.is_file():
            log(f"  WARNING: BBQ file not found for question_index: {path}")
            continue
        qi_map: dict[int, int] = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                eid = int(item["example_id"])
                qi = int(item["question_index"])
                qi_map[eid] = qi
        result[cat] = qi_map
    return result


def build_item_to_question_index(
    meta_df: pd.DataFrame, qi_maps: dict[str, dict[int, int]],
    stimuli_bridges: dict[str, dict[int, int]],
) -> dict[int, int]:
    """Map item_idx → question_index using stimuli bridge and BBQ lookup."""
    result: dict[int, int] = {}
    per_cat_mapped: Counter = Counter()
    per_cat_total: Counter = Counter()

    for _, row in meta_df.iterrows():
        idx = int(row["item_idx"])
        cat_short = row["category_short"]
        per_cat_total[cat_short] += 1
        # item_idx → example_id via bridge
        eid = stimuli_bridges.get(cat_short, {}).get(idx, idx)
        qi = qi_maps.get(cat_short, {}).get(int(eid))
        if qi is not None:
            result[idx] = qi
            per_cat_mapped[cat_short] += 1

    for cat in sorted(per_cat_total):
        mapped = per_cat_mapped[cat]
        total = per_cat_total[cat]
        pct = 100 * mapped / total if total > 0 else 0
        if pct < 100:
            log(f"  question_index {cat}: {mapped}/{total} ({pct:.0f}%) mapped")

    return result


def load_stimuli_bridges(
    run_dir: Path, categories_short: list[str],
) -> dict[str, dict[int, int]]:
    """Load item_idx → example_id mapping from stimuli JSONs."""
    bridges: dict[str, dict[int, int]] = {}
    for cat in categories_short:
        candidates = [
            run_dir / "A_extraction" / "stimuli" / f"{cat}.json",
            run_dir / "stimuli" / f"{cat}.json",
        ]
        # Also check data/processed with date stamps
        processed_dir = PROJECT_ROOT / "data" / "processed"
        if processed_dir.is_dir():
            for p in sorted(processed_dir.glob(f"stimuli_{cat}_*.json")):
                candidates.append(p)

        found = False
        for path in candidates:
            if path.is_file():
                with open(path) as f:
                    items = json.load(f)
                bridges[cat] = {
                    int(it["item_idx"]): int(it["example_id"])
                    for it in items if "example_id" in it
                }
                log(f"  Stimuli bridge {cat}: {len(bridges[cat])} items from {path}")
                found = True
                break
        if not found:
            log(f"  WARNING: no stimuli file found for {cat}; "
                f"question_index grouping may be incomplete")
    return bridges


# ---------------------------------------------------------------------------
# Race intersectional diagnostic
# ---------------------------------------------------------------------------

def strip_gender_prefix(label: str) -> str:
    """Strip gender prefix from intersectional labels: 'f-asian' → 'asian'."""
    if isinstance(label, str) and len(label) > 2 and label[1] == "-":
        return label.split("-", 1)[1]
    return label


def diagnose_race_intersectional(
    meta_df: pd.DataFrame, threshold: float = 0.10,
) -> dict[str, Any]:
    """Categorize race ambig items by mention-pattern type."""
    race_ambig = meta_df[
        (meta_df["category_short"] == "race")
        & (meta_df["context_condition"] == "ambig")
    ]
    n_total = len(race_ambig)
    n_intersectional = 0
    n_race_only = 0
    n_mixed = 0

    for _, row in race_ambig.iterrows():
        ms = row["mentioned_subgroups"] if isinstance(row["mentioned_subgroups"], list) else []
        if not ms:
            continue
        has_prefix = sum(
            1 for label in ms
            if isinstance(label, str) and len(label) > 2 and label[1] == "-"
        )
        no_prefix = len(ms) - has_prefix

        if has_prefix > 0 and no_prefix == 0:
            n_intersectional += 1
        elif has_prefix == 0 and no_prefix > 0:
            n_race_only += 1
        else:
            n_mixed += 1

    race_only_frac = n_race_only / n_total if n_total > 0 else 0.0
    handling = "both" if race_only_frac >= threshold else "stripped"

    return {
        "n_total_race_items": int(n_total),
        "n_intersectional": int(n_intersectional),
        "n_race_only": int(n_race_only),
        "n_mixed": int(n_mixed),
        "race_only_fraction": float(race_only_frac),
        "threshold": float(threshold),
        "handling_strategy": handling,
    }


def apply_race_stripping(meta_df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of meta_df with gender-prefix-stripped labels for race rows."""
    df = meta_df.copy()
    race_mask = df["category_short"] == "race"

    df.loc[race_mask, "mentioned_subgroups"] = df.loc[race_mask, "mentioned_subgroups"].apply(
        lambda ms: sorted(set(strip_gender_prefix(l) for l in ms)) if isinstance(ms, list) else ms
    )
    df.loc[race_mask, "stereotyped_groups"] = df.loc[race_mask, "stereotyped_groups"].apply(
        lambda gs: sorted(set(strip_gender_prefix(l) for l in gs)) if isinstance(gs, list) else gs
    )
    return df


# ---------------------------------------------------------------------------
# Subgroup vocabulary
# ---------------------------------------------------------------------------

def build_subgroup_vocabulary(
    meta_df: pd.DataFrame, category_short: str, granularity: str = "fine",
) -> list[str]:
    """Build the subgroup vocabulary for a category at the given granularity.

    fine: labels from stereotyped_groups
    aggregated: labels from mentioned_subgroups
    """
    cat_ambig = meta_df[
        (meta_df["category_short"] == category_short)
        & (meta_df["context_condition"] == "ambig")
    ]

    labels: set[str] = set()
    col = "stereotyped_groups" if granularity == "fine" else "mentioned_subgroups"

    for groups in cat_ambig[col]:
        if isinstance(groups, list):
            for g in groups:
                g_str = str(g).strip().lower()
                if g_str:
                    labels.add(g_str)

    return sorted(labels)


def should_run_aggregated(
    meta_df: pd.DataFrame, category_short: str,
) -> bool:
    """Check if fine and aggregated vocabularies differ for this category."""
    if category_short in DUAL_GRANULARITY_CATEGORIES:
        fine = set(build_subgroup_vocabulary(meta_df, category_short, "fine"))
        agg = set(build_subgroup_vocabulary(meta_df, category_short, "aggregated"))
        return fine != agg
    return False


# ---------------------------------------------------------------------------
# Group construction for three direction types
# ---------------------------------------------------------------------------

def build_groups(
    meta_df: pd.DataFrame,
    category_short: str,
    subgroup: str,
    direction_type: str,
    granularity: str,
) -> tuple[list[int], list[int]]:
    """Return (group_a_item_idxs, group_b_item_idxs) for the given direction.

    direction_type ∈ {"mention", "role", "bias_response"}
    """
    cat_ambig = meta_df[
        (meta_df["category_short"] == category_short)
        & (meta_df["context_condition"] == "ambig")
    ]

    target_col = "stereotyped_groups" if granularity == "fine" else "mentioned_subgroups"
    mention_col = "mentioned_subgroups"

    def _in_col(row: pd.Series, col: str) -> bool:
        gs = row[col]
        return isinstance(gs, list) and subgroup in gs

    if direction_type == "mention":
        group_a = cat_ambig[
            cat_ambig[mention_col].apply(lambda gs: subgroup in gs if isinstance(gs, list) else False)
        ]
        group_b = cat_ambig[
            cat_ambig[mention_col].apply(lambda gs: subgroup not in gs if isinstance(gs, list) else True)
        ]

    elif direction_type == "role":
        # A: S is the target.
        group_a = cat_ambig[
            cat_ambig[target_col].apply(lambda gs: subgroup in gs if isinstance(gs, list) else False)
        ]
        # B: S is mentioned but NOT the target.
        group_b = cat_ambig[
            cat_ambig.apply(
                lambda r: (
                    isinstance(r[mention_col], list) and subgroup in r[mention_col]
                ) and not (
                    isinstance(r[target_col], list) and subgroup in r[target_col]
                ),
                axis=1,
            )
        ]

    elif direction_type == "bias_response":
        # Both groups: S is the target. A: biased response. B: counter-biased.
        targeted = cat_ambig[
            cat_ambig[target_col].apply(lambda gs: subgroup in gs if isinstance(gs, list) else False)
        ]
        group_a = targeted[targeted["is_biased_response"] == 1]
        group_b = targeted[targeted["is_biased_response"] == 0]

    else:
        raise ValueError(f"Unknown direction_type: {direction_type}")

    return (
        group_a["item_idx"].astype(int).tolist(),
        group_b["item_idx"].astype(int).tolist(),
    )


# ---------------------------------------------------------------------------
# Direction computation
# ---------------------------------------------------------------------------

DirectionKey = tuple[str, str, str, str]  # (category_short, subgroup, direction_type, granularity)


def compute_directions(
    meta_df: pd.DataFrame,
    hidden_states: dict[int, np.ndarray],
    categories_short: list[str],
    layer: int,
    min_n: int,
) -> tuple[dict[DirectionKey, dict], list[dict]]:
    """Compute all three direction types for all subgroups.

    Returns:
        directions: keyed by (cat, sub, dir_type, granularity) → direction info dict
        summary_rows: list of dicts for directions_summary.parquet
    """
    directions: dict[DirectionKey, dict] = {}
    summary_rows: list[dict] = []

    for cat in categories_short:
        granularities_to_run = ["fine"]
        if should_run_aggregated(meta_df, cat):
            granularities_to_run.append("aggregated")
            log(f"  {cat}: running both fine and aggregated granularities")

        for gran in granularities_to_run:
            vocab = build_subgroup_vocabulary(meta_df, cat, gran)
            log(f"  {cat} [{gran}]: {len(vocab)} subgroups: {vocab}")

            for sub in vocab:
                for dir_type in DIRECTION_TYPES:
                    key: DirectionKey = (cat, sub, dir_type, gran)

                    group_a_idxs, group_b_idxs = build_groups(
                        meta_df, cat, sub, dir_type, gran,
                    )

                    # Filter to items with cached hidden states
                    a_vecs = [hidden_states[i] for i in group_a_idxs
                              if i in hidden_states]
                    b_vecs = [hidden_states[i] for i in group_b_idxs
                              if i in hidden_states]

                    row_base = {
                        "category": cat, "subgroup": sub,
                        "direction_type": dir_type, "granularity": gran,
                        "layer": layer,
                        "n_a_metadata": len(group_a_idxs),
                        "n_b_metadata": len(group_b_idxs),
                        "n_a_cached": len(a_vecs),
                        "n_b_cached": len(b_vecs),
                    }

                    if len(a_vecs) < min_n or len(b_vecs) < min_n:
                        summary_rows.append({
                            **row_base,
                            "status": "insufficient_items",
                            "norm_raw": None,
                        })
                        continue

                    mean_a = np.stack(a_vecs).mean(axis=0).astype(np.float32)
                    mean_b = np.stack(b_vecs).mean(axis=0).astype(np.float32)
                    raw = (mean_a - mean_b).astype(np.float32)
                    norm = float(np.linalg.norm(raw))

                    if norm < 1e-8:
                        summary_rows.append({
                            **row_base, "status": "zero_norm", "norm_raw": 0.0,
                        })
                        continue

                    directions[key] = {
                        "direction_raw": raw,
                        "direction_normed": raw / norm,
                        "norm_raw": norm,
                        "mean_group_a": mean_a,
                        "mean_group_b": mean_b,
                        "n_a": len(a_vecs),
                        "n_b": len(b_vecs),
                        "group_a_idxs": group_a_idxs,
                        "group_b_idxs": group_b_idxs,
                    }
                    summary_rows.append({
                        **row_base, "status": "ok", "norm_raw": norm,
                    })

    n_ok = sum(1 for r in summary_rows if r["status"] == "ok")
    log(f"  Computed {n_ok}/{len(summary_rows)} directions successfully")
    return directions, summary_rows


def save_direction_vectors(
    directions: dict[DirectionKey, dict], output_dir: Path, layer: int,
) -> None:
    """Save each direction as a separate .npz file."""
    for (cat, sub, dir_type, gran), info in directions.items():
        safe_sub = sub.replace("/", "_").replace(" ", "_")
        cat_dir = output_dir / "direction_vectors" / f"L{layer}" / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            cat_dir / f"{safe_sub}_{dir_type}_{gran}.npz",
            direction_raw=info["direction_raw"],
            direction_normed=info["direction_normed"],
            norm_raw=np.float32(info["norm_raw"]),
            n_a=np.int32(info["n_a"]),
            n_b=np.int32(info["n_b"]),
            category=cat, subgroup=sub,
            direction_type=dir_type, granularity=gran,
            layer=np.int32(layer),
        )


# ---------------------------------------------------------------------------
# Pairwise cosines (within direction type)
# ---------------------------------------------------------------------------

def compute_pairwise_cosines(
    directions: dict[DirectionKey, dict], layer: int,
) -> list[dict]:
    """Pairwise cosines between subgroups for each (category, direction_type, granularity)."""
    # Group by (cat, dir_type, gran)
    grouped: dict[tuple[str, str, str], list[tuple[str, dict]]] = defaultdict(list)
    for (cat, sub, dir_type, gran), info in directions.items():
        grouped[(cat, dir_type, gran)].append((sub, info))

    rows: list[dict] = []
    for (cat, dir_type, gran), subs in grouped.items():
        for i, (sa, ia) in enumerate(subs):
            for j, (sb, ib) in enumerate(subs):
                if i > j:
                    continue
                cos = float(np.dot(ia["direction_normed"], ib["direction_normed"]))
                base = {
                    "category": cat, "direction_type": dir_type,
                    "granularity": gran, "layer": layer,
                }
                rows.append({
                    **base, "subgroup_a": sa, "subgroup_b": sb,
                    "cosine": cos, "n_a": ia["n_a"], "n_b": ib["n_a"],
                })
                if i != j:
                    rows.append({
                        **base, "subgroup_a": sb, "subgroup_b": sa,
                        "cosine": cos, "n_a": ib["n_a"], "n_b": ia["n_a"],
                    })
    return rows


# ---------------------------------------------------------------------------
# Cross-direction-type alignment
# ---------------------------------------------------------------------------

def compute_direction_alignment(
    directions: dict[DirectionKey, dict],
) -> list[dict]:
    """For each subgroup, compute cosines between the three direction types."""
    # Group by (cat, sub, gran)
    by_sub: dict[tuple[str, str, str], dict[str, dict]] = defaultdict(dict)
    for (cat, sub, dir_type, gran), info in directions.items():
        by_sub[(cat, sub, gran)][dir_type] = info

    pairs = [
        ("mention", "role", "mention_vs_role"),
        ("mention", "bias_response", "mention_vs_bias_response"),
        ("role", "bias_response", "role_vs_bias_response"),
    ]

    rows: list[dict] = []
    for (cat, sub, gran), type_map in by_sub.items():
        for dt_a, dt_b, pair_name in pairs:
            if dt_a not in type_map or dt_b not in type_map:
                continue
            da = type_map[dt_a]["direction_normed"]
            db = type_map[dt_b]["direction_normed"]
            cos = float(np.dot(da, db))
            rows.append({
                "category": cat, "subgroup": sub, "granularity": gran,
                "pair": pair_name, "cosine": cos,
                "norm_a": type_map[dt_a]["norm_raw"],
                "norm_b": type_map[dt_b]["norm_raw"],
            })
    return rows


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def make_probe_pipeline(seed: int, max_iter: int = 200) -> Pipeline:
    """Per-fold standardization + L2 logistic regression with liblinear solver."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="liblinear",
            max_iter=max_iter,
            class_weight="balanced",
            random_state=seed,
        )),
    ])


def compute_directional_probe(
    hidden_states: dict[int, np.ndarray],
    group_a_idxs: list[int],
    group_b_idxs: list[int],
    qi_map: dict[int, int],
    n_folds: int = 5,
    n_bootstrap: int = 100,
    seed: int = 42,
    max_iter: int = 200,
) -> dict | None:
    """Train probe to distinguish Group A from Group B using hidden states.

    StandardScaler is fit inside each fold via the Pipeline (no leakage).
    Returns dict with probe metrics, or None if insufficient data.
    """
    rng = np.random.default_rng(seed)

    X_list, y_list, groups = [], [], []
    for idx in group_a_idxs:
        if idx in hidden_states:
            X_list.append(hidden_states[idx])
            y_list.append(1)
            groups.append(qi_map.get(idx, idx))
    for idx in group_b_idxs:
        if idx in hidden_states:
            X_list.append(hidden_states[idx])
            y_list.append(0)
            groups.append(qi_map.get(idx, idx))

    if len(X_list) < 20:
        return None

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list)
    g = np.array(groups)

    n_a = int((y == 1).sum())
    n_b = int((y == 0).sum())

    n_unique_groups = len(np.unique(g))
    eff_folds = min(n_folds, n_unique_groups)
    if eff_folds < 2:
        return None

    gkf = GroupKFold(n_splits=eff_folds)
    fold_ba: list[float] = []
    fold_auroc: list[float] = []

    for tr, te in gkf.split(X, y, g):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        try:
            pipe = make_probe_pipeline(seed=seed, max_iter=max_iter)
            pipe.fit(X[tr], y[tr])
            proba = pipe.predict_proba(X[te])[:, 1]
            pred = (proba >= 0.5).astype(int)
            fold_ba.append(balanced_accuracy_score(y[te], pred))
            fold_auroc.append(roc_auc_score(y[te], proba))
        except Exception:
            continue

    if not fold_ba:
        return None

    mean_ba = float(np.mean(fold_ba))
    mean_auroc = float(np.mean(fold_auroc))

    # Bootstrap CIs (skipped entirely when n_bootstrap == 0)
    boot_ba: list[float] = []
    boot_auroc: list[float] = []

    for _ in range(n_bootstrap):  # noqa: no-op when n_bootstrap == 0
        bi = rng.choice(len(X), size=len(X), replace=True)
        Xb, yb, gb = X[bi], y[bi], g[bi]
        if len(np.unique(yb)) < 2:
            continue

        uq = np.unique(gb)
        if len(uq) < 5:
            continue
        rng.shuffle(uq)
        n_te = max(1, len(uq) // 5)
        te_groups = set(uq[:n_te])
        tr_m = ~np.isin(gb, list(te_groups))
        te_m = np.isin(gb, list(te_groups))

        if (tr_m.sum() < 10 or te_m.sum() < 5
                or len(np.unique(yb[tr_m])) < 2
                or len(np.unique(yb[te_m])) < 2):
            continue
        try:
            pipe = make_probe_pipeline(seed=seed, max_iter=max_iter)
            pipe.fit(Xb[tr_m], yb[tr_m])
            proba = pipe.predict_proba(Xb[te_m])[:, 1]
            pred = (proba >= 0.5).astype(int)
            boot_ba.append(balanced_accuracy_score(yb[te_m], pred))
            boot_auroc.append(roc_auc_score(yb[te_m], proba))
        except Exception:
            continue

    if len(boot_ba) >= 20:
        ba_ci_lo = float(np.percentile(boot_ba, 2.5))
        ba_ci_hi = float(np.percentile(boot_ba, 97.5))
        auroc_ci_lo = float(np.percentile(boot_auroc, 2.5))
        auroc_ci_hi = float(np.percentile(boot_auroc, 97.5))
    else:
        ba_ci_lo = ba_ci_hi = auroc_ci_lo = auroc_ci_hi = None

    return {
        "balanced_accuracy": mean_ba,
        "auroc": mean_auroc,
        "ba_ci_low": ba_ci_lo, "ba_ci_high": ba_ci_hi,
        "auroc_ci_low": auroc_ci_lo, "auroc_ci_high": auroc_ci_hi,
        "n_a": n_a, "n_b": n_b,
        "n_cv_folds": len(fold_ba),
        "n_bootstrap_success": len(boot_ba),
    }


def _json_safe(obj: Any) -> Any:
    """JSON serializer for numpy scalars."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if not np.isnan(obj) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def load_probe_progress(progress_path: Path) -> dict[tuple, dict]:
    """Load already-completed probes from JSONL. Returns dict keyed by
    (category, subgroup, direction_type, granularity)."""
    if not progress_path.exists():
        return {}
    completed: dict[tuple, dict] = {}
    with open(progress_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                key = (row["category"], row["subgroup"],
                       row["direction_type"], row["granularity"])
                completed[key] = row
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def append_probe_result(progress_path: Path, row: dict) -> None:
    """Append a single probe result to the JSONL progress file."""
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_path, "a") as f:
        f.write(json.dumps(row, default=_json_safe) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run_all_probes(
    directions: dict[DirectionKey, dict],
    hidden_states: dict[int, np.ndarray],
    qi_map: dict[int, int],
    output_dir: Path,
    layer: int = 14,
    n_folds: int = 5,
    n_bootstrap: int = 0,
    seed: int = 42,
    max_iter: int = 200,
) -> list[dict]:
    """Run probes for all directions with JSONL incremental checkpointing.

    Completed probes are appended to _probe_progress.jsonl as they finish.
    On restart, already-completed probes are loaded and skipped.
    """
    progress_path = output_dir / "_probe_progress.jsonl"
    completed = load_probe_progress(progress_path)

    if completed:
        log(f"  Resuming: {len(completed)} probes already completed in "
            f"{progress_path.name}")

    rows: list[dict] = list(completed.values())
    skipped = 0

    items = list(directions.items())
    for (cat, sub, dir_type, gran), info in tqdm(
        items, desc="  Probes", unit="probe",
    ):
        key = (cat, sub, dir_type, gran)
        if key in completed:
            skipped += 1
            continue

        result = compute_directional_probe(
            hidden_states,
            info["group_a_idxs"],
            info["group_b_idxs"],
            qi_map,
            n_folds=n_folds,
            n_bootstrap=n_bootstrap,
            seed=seed,
            max_iter=max_iter,
        )

        row: dict[str, Any] = {
            "category": cat, "subgroup": sub,
            "direction_type": dir_type, "granularity": gran,
            "layer": layer,
        }
        if result is None:
            row.update({
                "balanced_accuracy": None, "auroc": None,
                "ba_ci_low": None, "ba_ci_high": None,
                "auroc_ci_low": None, "auroc_ci_high": None,
                "n_a": 0, "n_b": 0,
                "n_cv_folds": 0, "n_bootstrap_success": 0,
                "status": "insufficient_data",
            })
        else:
            row.update(result)
            row["status"] = "ok"

        # Persist immediately
        append_probe_result(progress_path, row)
        rows.append(row)

    log(f"  Probes complete: {len(rows)} total, {skipped} resumed from disk")
    return rows


# ---------------------------------------------------------------------------
# Geometry summary
# ---------------------------------------------------------------------------

def build_geometry_summary(
    cos_df: pd.DataFrame,
    align_df: pd.DataFrame,
    probes_df: pd.DataFrame,
    summary_rows: list[dict],
    layer: int,
    race_diag: dict,
) -> dict:
    """Build the high-level geometry summary JSON."""
    summary: dict[str, Any] = {
        "primary_layer": layer,
        "race_handling": race_diag.get("handling_strategy", "unknown"),
        "n_subgroups_per_category": {},
        "per_category": {},
    }

    # Count subgroups per category/granularity
    dirs_df = pd.DataFrame(summary_rows)
    for cat in sorted(dirs_df["category"].unique()):
        cat_dirs = dirs_df[dirs_df["category"] == cat]
        counts: dict[str, int] = {}
        for gran in cat_dirs["granularity"].unique():
            n = cat_dirs[
                (cat_dirs["granularity"] == gran) & (cat_dirs["status"] == "ok")
            ]["subgroup"].nunique()
            counts[gran] = int(n)
        summary["n_subgroups_per_category"][cat] = counts

    # Per-category stats
    for cat in sorted(dirs_df["category"].unique()):
        cat_summary: dict[str, Any] = {}

        for gran in dirs_df[dirs_df["category"] == cat]["granularity"].unique():
            gran_stats: dict[str, Any] = {}
            gran_stats["n_subgroups"] = int(
                dirs_df[
                    (dirs_df["category"] == cat)
                    & (dirs_df["granularity"] == gran)
                    & (dirs_df["status"] == "ok")
                ]["subgroup"].nunique()
            )

            # Mean pairwise cosines per direction type
            for dt in DIRECTION_TYPES:
                cc = cos_df[
                    (cos_df["category"] == cat)
                    & (cos_df["direction_type"] == dt)
                    & (cos_df["granularity"] == gran)
                    & (cos_df["subgroup_a"] != cos_df["subgroup_b"])
                ]
                if not cc.empty:
                    gran_stats[f"mean_pairwise_cosine_{dt}"] = round(
                        float(cc["cosine"].mean()), 3,
                    )

            # Mean alignment per pair type
            for pair in ["mention_vs_role", "mention_vs_bias_response",
                         "role_vs_bias_response"]:
                aa = align_df[
                    (align_df["category"] == cat)
                    & (align_df["granularity"] == gran)
                    & (align_df["pair"] == pair)
                ]
                if not aa.empty:
                    gran_stats[f"mean_{pair}_alignment"] = round(
                        float(aa["cosine"].mean()), 3,
                    )

            # Mean probe AUROC per direction type
            for dt in DIRECTION_TYPES:
                pp = probes_df[
                    (probes_df["category"] == cat)
                    & (probes_df["direction_type"] == dt)
                    & (probes_df["granularity"] == gran)
                ]
                if not pp.empty and pp["auroc"].notna().any():
                    gran_stats[f"mean_probe_auroc_{dt}"] = round(
                        float(pp["auroc"].dropna().mean()), 3,
                    )

            cat_summary[gran] = gran_stats
        summary["per_category"][cat] = cat_summary

    return summary


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_cosine_heatmap(
    cos_df: pd.DataFrame, cat: str, dir_type: str, gran: str,
    layer: int, out_dir: Path,
) -> None:
    """Pairwise cosine heatmap for one (category, direction_type, granularity)."""
    df = cos_df[
        (cos_df["category"] == cat)
        & (cos_df["direction_type"] == dir_type)
        & (cos_df["granularity"] == gran)
        & (cos_df["layer"] == layer)
    ]
    if df.empty:
        return
    subs = sorted(df["subgroup_a"].unique())
    n = len(subs)
    if n < 2:
        return
    s2i = {s: i for i, s in enumerate(subs)}
    mat = np.full((n, n), np.nan)
    for _, r in df.iterrows():
        mat[s2i[r["subgroup_a"]], s2i[r["subgroup_b"]]] = r["cosine"]

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
    title = f"DIM_{dir_type} cosines — {cat} [{gran}] (L{layer})"
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    _save_fig(fig, out_dir / f"fig_pairwise_cosines_{cat}_{dir_type}_{gran}.png")


def fig_direction_alignment(
    align_df: pd.DataFrame, cat: str, gran: str, out_dir: Path,
) -> None:
    """Grouped bar chart of cross-type cosines per subgroup."""
    df = align_df[
        (align_df["category"] == cat) & (align_df["granularity"] == gran)
    ]
    if df.empty:
        return

    subs = sorted(df["subgroup"].unique())
    pair_names = ["mention_vs_role", "mention_vs_bias_response",
                  "role_vs_bias_response"]
    pair_labels = ["mention vs role", "mention vs bias", "role vs bias"]
    pair_colors = [ALIGNMENT_PAIR_COLORS[p] for p in pair_names]

    n_subs = len(subs)
    n_pairs = len(pair_names)
    x = np.arange(n_subs)
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(6, n_subs * 1.2), 5))
    for pi, (pname, plabel, pcol) in enumerate(zip(pair_names, pair_labels, pair_colors)):
        vals = []
        for sub in subs:
            row = df[(df["subgroup"] == sub) & (df["pair"] == pname)]
            vals.append(float(row["cosine"].iloc[0]) if not row.empty else 0.0)
        ax.bar(x + (pi - 1) * width, vals, width, label=plabel, color=pcol, alpha=0.85)

    ax.axhline(0, color="gray", linestyle=":", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(subs, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Cosine similarity")
    ax.set_title(f"Direction alignment — {cat} [{gran}]", fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save_fig(fig, out_dir / f"fig_direction_alignment_{cat}_{gran}.png")


def fig_probe_accuracies(
    probes_df: pd.DataFrame, out_dir: Path,
) -> None:
    """Grouped bar chart: AUROC per subgroup, three bars per subgroup, faceted by category."""
    df = probes_df[probes_df["auroc"].notna()].copy()
    if df.empty:
        return

    categories = sorted(df["category"].unique())
    n_cats = len(categories)
    if n_cats == 0:
        return

    n_cols = min(3, n_cats)
    n_rows = (n_cats + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows),
                              squeeze=False)

    for ci, cat in enumerate(categories):
        ax = axes[ci // n_cols, ci % n_cols]
        cat_df = df[df["category"] == cat]
        # Use fine granularity preferentially
        grans = sorted(cat_df["granularity"].unique())
        gran = "fine" if "fine" in grans else grans[0]
        cat_df = cat_df[cat_df["granularity"] == gran]

        subs = sorted(cat_df["subgroup"].unique())
        n_subs = len(subs)
        x = np.arange(n_subs)
        width = 0.25

        for di, dt in enumerate(DIRECTION_TYPES):
            vals = []
            for sub in subs:
                row = cat_df[(cat_df["subgroup"] == sub) & (cat_df["direction_type"] == dt)]
                vals.append(float(row["auroc"].iloc[0]) if not row.empty and row["auroc"].notna().any() else np.nan)
            ax.bar(x + (di - 1) * width, vals, width,
                   label=dt, color=DIRECTION_TYPE_COLORS[dt], alpha=0.85)

        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.7, label="chance")
        ax.set_xticks(x)
        ax.set_xticklabels(subs, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("AUROC")
        ax.set_ylim(0.35, 1.05)
        ax.set_title(f"{cat} [{gran}]", fontsize=10)
        if ci == 0:
            ax.legend(fontsize=7, loc="lower right")

    # Hide unused axes
    for ci in range(n_cats, n_rows * n_cols):
        axes[ci // n_cols, ci % n_cols].set_visible(False)

    fig.suptitle("Probe AUROC by direction type", fontsize=12)
    plt.tight_layout()
    _save_fig(fig, out_dir / "fig_probe_accuracies.png")


def fig_norm_comparison(
    summary_rows: list[dict], out_dir: Path,
) -> None:
    """Direction norms across types per subgroup."""
    df = pd.DataFrame(summary_rows)
    df = df[df["status"] == "ok"]
    if df.empty:
        return

    categories = sorted(df["category"].unique())
    n_cats = len(categories)
    n_cols = min(3, n_cats)
    n_rows = (n_cats + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows),
                              squeeze=False)

    for ci, cat in enumerate(categories):
        ax = axes[ci // n_cols, ci % n_cols]
        cat_df = df[df["category"] == cat]
        grans = sorted(cat_df["granularity"].unique())
        gran = "fine" if "fine" in grans else grans[0]
        cat_df = cat_df[cat_df["granularity"] == gran]

        subs = sorted(cat_df["subgroup"].unique())
        n_subs = len(subs)
        x = np.arange(n_subs)
        width = 0.25

        for di, dt in enumerate(DIRECTION_TYPES):
            vals = []
            for sub in subs:
                row = cat_df[(cat_df["subgroup"] == sub) & (cat_df["direction_type"] == dt)]
                vals.append(float(row["norm_raw"].iloc[0]) if not row.empty and row["norm_raw"].notna().any() else 0.0)
            ax.bar(x + (di - 1) * width, vals, width,
                   label=dt, color=DIRECTION_TYPE_COLORS[dt], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(subs, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Direction norm")
        ax.set_title(f"{cat} [{gran}]", fontsize=10)
        if ci == 0:
            ax.legend(fontsize=7)

    for ci in range(n_cats, n_rows * n_cols):
        axes[ci // n_cols, ci % n_cols].set_visible(False)

    fig.suptitle("Direction norms by type", fontsize=12)
    plt.tight_layout()
    _save_fig(fig, out_dir / "fig_norm_comparison.png")


def fig_race_diagnostic(race_diag: dict, out_dir: Path) -> None:
    """Pie chart of race item breakdown."""
    labels = ["Intersectional", "Race-only", "Mixed"]
    sizes = [
        race_diag.get("n_intersectional", 0),
        race_diag.get("n_race_only", 0),
        race_diag.get("n_mixed", 0),
    ]
    if sum(sizes) == 0:
        return
    colors = [WONG["blue"], WONG["orange"], WONG["green"]]
    fig, ax = plt.subplots(figsize=(5, 5))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, textprops={"fontsize": 10},
    )
    ax.set_title(
        f"Race items — {race_diag.get('handling_strategy', '?')} "
        f"(n={race_diag.get('n_total_race_items', 0)})",
        fontsize=11,
    )
    _save_fig(fig, out_dir / "fig_race_intersectional_diagnostic.png")


def generate_all_figures(
    cos_df: pd.DataFrame,
    align_df: pd.DataFrame,
    probes_df: pd.DataFrame,
    summary_rows: list[dict],
    race_diag: dict,
    categories_short: list[str],
    layer: int,
    out_dir: Path,
) -> None:
    """Generate all figures."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pairwise cosine heatmaps
    for cat in categories_short:
        for gran in cos_df[cos_df["category"] == cat]["granularity"].unique():
            for dt in DIRECTION_TYPES:
                fig_cosine_heatmap(cos_df, cat, dt, gran, layer, out_dir)

    # Direction alignment bar charts
    for cat in categories_short:
        for gran in align_df[align_df["category"] == cat]["granularity"].unique():
            fig_direction_alignment(align_df, cat, gran, out_dir)

    # Probe AUROC
    fig_probe_accuracies(probes_df, out_dir)

    # Norm comparison
    fig_norm_comparison(summary_rows, out_dir)

    # Race diagnostic
    if race_diag.get("n_total_race_items", 0) > 0:
        fig_race_diagnostic(race_diag, out_dir)

    log(f"  All figures saved to {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    t0 = time.time()

    output_dir = run_dir / "stage1_v2_geometry"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    log("Stage 1 v2: Role / Mention / Bias-Mediation Decomposition")
    log(f"  Run dir: {run_dir}")
    log(f"  Layer: {args.layer}")

    # ── Load enriched metadata ──────────────────────────────────────────
    meta_df = load_enriched_metadata(run_dir)

    # Determine categories
    if args.categories:
        categories_short = [c.strip() for c in args.categories.split(",")]
    else:
        categories_short = sorted(meta_df["category_short"].unique())
    log(f"  Categories: {categories_short}")

    # Check for empty mentioned_subgroups
    for cat in categories_short:
        cat_df = meta_df[meta_df["category_short"] == cat]
        n_empty = cat_df["mentioned_subgroups"].apply(
            lambda ms: not isinstance(ms, list) or len(ms) == 0
        ).sum()
        pct = 100 * n_empty / len(cat_df) if len(cat_df) > 0 else 0
        if pct > 5:
            log(f"  WARNING: {cat} has {pct:.1f}% empty mentioned_subgroups")

    # ── BBQ data for question_index ─────────────────────────────────────
    # Try standard BBQ data locations
    bbq_candidates = [
        PROJECT_ROOT / "datasets" / "bbq" / "data",
        PROJECT_ROOT.parent / "bbqmi" / "data" / "BBQ" / "data",
    ]
    bbq_data_dir = None
    for p in bbq_candidates:
        if p.is_dir():
            bbq_data_dir = p
            break

    qi_map: dict[int, int] = {}
    if bbq_data_dir is not None:
        log(f"  Loading question_index from BBQ: {bbq_data_dir}")
        qi_maps = load_question_index_from_bbq(bbq_data_dir, categories_short)
        bridges = load_stimuli_bridges(run_dir, categories_short)
        qi_map = build_item_to_question_index(meta_df, qi_maps, bridges)
        log(f"  Mapped question_index for {len(qi_map)} items")
    else:
        log("  WARNING: BBQ data not found; probes will use StratifiedKFold")

    # ── Race intersectional diagnostic ──────────────────────────────────
    race_diag: dict[str, Any] = {"n_total_race_items": 0, "handling_strategy": "n/a"}
    if "race" in categories_short:
        log("\nRace intersectional diagnostic:")
        race_diag = diagnose_race_intersectional(
            meta_df, threshold=args.race_intersectional_threshold,
        )
        log(f"  Total: {race_diag['n_total_race_items']}")
        log(f"  Intersectional: {race_diag['n_intersectional']}")
        log(f"  Race-only: {race_diag['n_race_only']}")
        log(f"  Mixed: {race_diag['n_mixed']}")
        log(f"  Race-only fraction: {race_diag['race_only_fraction']:.3f}")
        log(f"  → Handling: {race_diag['handling_strategy']}")

        atomic_save_json(
            race_diag,
            output_dir / "race_intersectional_diagnostic.json",
        )
    else:
        # Write an empty diagnostic
        atomic_save_json(
            race_diag,
            output_dir / "race_intersectional_diagnostic.json",
        )

    # Apply race stripping if needed
    if race_diag.get("handling_strategy") == "stripped":
        log("  Applying gender-prefix stripping to race labels")
        meta_df = apply_race_stripping(meta_df)

    # ── Load hidden states ──────────────────────────────────────────────
    log(f"\nLoading hidden states at layer {args.layer}...")
    hidden_states = load_layer_hidden_states(
        run_dir, categories_short, args.layer,
    )
    log(f"  Loaded {len(hidden_states)} item hidden states")

    if len(hidden_states) == 0:
        log("FATAL: No hidden states loaded. Ensure A_extraction/activations/ exists.")
        sys.exit(1)

    # ── Compute or reload directions / cosines / alignment ───────────────
    directions_path = output_dir / "directions_summary.parquet"
    cosines_path = output_dir / "pairwise_cosines.parquet"
    alignment_path = output_dir / "direction_alignment.parquet"

    resume_directions = (
        directions_path.exists()
        and cosines_path.exists()
        and alignment_path.exists()
    )

    if resume_directions:
        log("\nFound existing direction outputs; skipping direction computation.")
        log(f"  {directions_path.name}, {cosines_path.name}, {alignment_path.name}")
        log(f"  If you want to recompute, delete those files first.")

        dirs_df = pd.read_parquet(directions_path)
        cos_df = pd.read_parquet(cosines_path)
        align_df = pd.read_parquet(alignment_path)
        summary_rows = dirs_df.to_dict("records")

        # Rebuild the directions dict with group indices for the probe loop.
        # Group indices come from metadata, same logic as compute_directions.
        log("\nRebuilding direction groups from metadata for probe phase...")
        directions, _ = compute_directions(
            meta_df, hidden_states, categories_short,
            args.layer, args.min_n_per_group,
        )
        log(f"  Rebuilt {len(directions)} direction group specs")

    else:
        log(f"\nComputing directions from scratch...")
        directions, summary_rows = compute_directions(
            meta_df, hidden_states, categories_short,
            args.layer, args.min_n_per_group,
        )

        save_direction_vectors(directions, output_dir, args.layer)

        dirs_df = pd.DataFrame(summary_rows)
        tmp = output_dir / ".directions_summary.parquet.tmp"
        dirs_df.to_parquet(tmp, index=False, compression="snappy")
        os.rename(tmp, output_dir / "directions_summary.parquet")
        log(f"  Saved directions_summary: {len(dirs_df)} rows")

        log("\nComputing pairwise cosines...")
        cos_rows = compute_pairwise_cosines(directions, args.layer)
        cos_df = pd.DataFrame(cos_rows)
        if not cos_df.empty:
            tmp = output_dir / ".pairwise_cosines.parquet.tmp"
            cos_df.to_parquet(tmp, index=False, compression="snappy")
            os.rename(tmp, output_dir / "pairwise_cosines.parquet")
        log(f"  Saved pairwise_cosines: {len(cos_df)} rows")

        log("\nComputing cross-type alignment...")
        align_rows = compute_direction_alignment(directions)
        align_df = pd.DataFrame(align_rows)
        if not align_df.empty:
            tmp = output_dir / ".direction_alignment.parquet.tmp"
            align_df.to_parquet(tmp, index=False, compression="snappy")
            os.rename(tmp, output_dir / "direction_alignment.parquet")
        log(f"  Saved direction_alignment: {len(align_df)} rows")

    # ── Probes ──────────────────────────────────────────────────────────
    log("\nComputing probes...")
    probe_rows = run_all_probes(
        directions, hidden_states, qi_map,
        output_dir=output_dir,
        layer=args.layer,
        n_folds=args.n_cv_folds,
        n_bootstrap=args.n_bootstrap,
        seed=args.random_seed,
        max_iter=args.probe_max_iter,
    )
    probes_df = pd.DataFrame(probe_rows)
    if not probes_df.empty:
        tmp = output_dir / ".probe_accuracies.parquet.tmp"
        probes_df.to_parquet(tmp, index=False, compression="snappy")
        os.rename(tmp, output_dir / "probe_accuracies.parquet")
    log(f"  Saved probe_accuracies: parquet with {len(probes_df)} rows")

    # ── Summary ─────────────────────────────────────────────────────────
    log("\nBuilding geometry summary...")
    summary = build_geometry_summary(
        cos_df, align_df, probes_df, summary_rows, args.layer, race_diag,
    )
    atomic_save_json(summary, output_dir / "geometry_summary.json")
    log("  Saved geometry_summary.json")

    # Print summary to stdout
    log("\n" + "=" * 72)
    log("GEOMETRY SUMMARY")
    log("=" * 72)
    for cat, cat_info in summary.get("per_category", {}).items():
        for gran, ginfo in cat_info.items():
            log(f"\n  {cat} [{gran}]:")
            for k, v in ginfo.items():
                log(f"    {k}: {v}")

    # ── Figures ─────────────────────────────────────────────────────────
    if not args.skip_figures:
        log("\nGenerating figures...")
        generate_all_figures(
            cos_df, align_df, probes_df, summary_rows,
            race_diag, categories_short, args.layer, fig_dir,
        )

    elapsed = time.time() - t0
    log(f"\nStage 1 v2 complete in {elapsed:.1f}s")
    log(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
