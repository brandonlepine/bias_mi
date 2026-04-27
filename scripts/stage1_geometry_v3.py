"""Stage 1 v3: Corrected Geometry Pipeline.

Computes per-subgroup DIM directions, pairwise cosines, cross-direction
alignment, and probe AUROCs using enriched BBQ data with explicit status
codes and hard validity checks.

Reads enriched JSONL files from data/enriched/, joins with activations
and model predictions, outputs to stage1_geometry_v3/.

Usage:
    python scripts/stage1_geometry_v3.py --run_dir runs/llama-3.1-8b_2026-04-22/ \\
        --layers 12,14,16 --min_n 10 \\
        2>&1 | tee runs/llama-3.1-8b_2026-04-22/logs/stage1_geometry_v3.log
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
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
from matplotlib.patches import Patch
import matplotlib.patches as mpatches

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import atomic_save_json


# ===================================================================
# 1. Constants, CLI, Status
# ===================================================================

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
DIRECTION_TYPE_MARKERS = {
    "mention": "o",
    "role": "s",
    "bias_response": "^",
}
ALIGNMENT_PAIR_COLORS = {
    "mention_vs_role": WONG["blue"],
    "mention_vs_bias_response": WONG["orange"],
    "role_vs_bias_response": WONG["green"],
}
DPI = 200
DIRECTION_TYPES = ["mention", "role", "bias_response"]
ALIGNMENT_PAIRS = [
    ("mention", "role", "mention_vs_role"),
    ("mention", "bias_response", "mention_vs_bias_response"),
    ("role", "bias_response", "role_vs_bias_response"),
]

CATEGORY_FILE_MAP: dict[str, str] = {
    "age": "Age", "disability": "Disability_status", "gi": "Gender_identity",
    "nationality": "Nationality", "physical_appearance": "Physical_appearance",
    "race": "Race_ethnicity", "religion": "Religion", "ses": "SES",
    "so": "Sexual_orientation",
}
BBQ_TO_SHORT: dict[str, str] = {v: k for k, v in CATEGORY_FILE_MAP.items()}
ALL_SHORT_NAMES = list(CATEGORY_FILE_MAP.keys())


class S:
    """Direction status constants."""
    COMPUTABLE = "computable"
    INSUFFICIENT_ITEMS = "insufficient_items"
    STRUCTURALLY_UNDEFINED = "structurally_undefined"
    DEGENERATE = "degenerate"
    NEVER_TARGETED = "never_targeted"
    NORM_NEAR_ZERO = "norm_near_zero"
    PAIRED_MENTION_DEGENERATE = "paired_mention_degenerate"
    PAIRED_TARGET_DEGENERATE = "paired_target_degenerate"


ALL_STATUSES = [
    S.COMPUTABLE, S.INSUFFICIENT_ITEMS, S.STRUCTURALLY_UNDEFINED,
    S.DEGENERATE, S.NEVER_TARGETED, S.NORM_NEAR_ZERO,
    S.PAIRED_MENTION_DEGENERATE, S.PAIRED_TARGET_DEGENERATE,
]

STATUS_COLORS = {
    S.COMPUTABLE: WONG["green"],
    S.INSUFFICIENT_ITEMS: WONG["orange"],
    S.STRUCTURALLY_UNDEFINED: WONG["purple"],
    S.DEGENERATE: WONG["sky_blue"],
    S.NEVER_TARGETED: WONG["vermillion"],
    S.NORM_NEAR_ZERO: WONG["yellow"],
    S.PAIRED_MENTION_DEGENERATE: "#888888",
    S.PAIRED_TARGET_DEGENERATE: "#AAAAAA",
}
STATUS_SHORT = {
    S.COMPUTABLE: "OK",
    S.INSUFFICIENT_ITEMS: "n<min",
    S.STRUCTURALLY_UNDEFINED: "undef",
    S.DEGENERATE: "degen",
    S.NEVER_TARGETED: "no tgt",
    S.NORM_NEAR_ZERO: "||d||~0",
    S.PAIRED_MENTION_DEGENERATE: "paired",
    S.PAIRED_TARGET_DEGENERATE: "co-tgt",
}

NORM_EPSILON = 1e-6


def log(msg: str) -> None:
    print(f"[v3] {msg}", flush=True)


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved {path.name}")


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if not np.isnan(obj) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1 v3: Corrected Geometry Pipeline")
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--layers", type=str, default="14",
                   help="Comma-separated layer indices")
    p.add_argument("--primary_layer", type=int, default=14,
                   help="Primary layer for summaries and figures")
    p.add_argument("--min_n", type=int, default=10)
    p.add_argument("--race_handling", type=str, default="stripped",
                   choices=["stripped", "intersectional"])
    p.add_argument("--probe_n_folds", type=int, default=5)
    p.add_argument("--predictions_source", type=str, default="auto",
                   choices=["auto", "predictions_dir", "metadata_enriched"])
    p.add_argument("--activations_layout", type=str, default="auto",
                   choices=["auto", "config_key", "default_path"])
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip_probes", action="store_true")
    p.add_argument("--probes_only", action="store_true",
                   help="Load existing directions, run only the probe stage")
    p.add_argument("--regenerate_outputs", action="store_true",
                   help="Regenerate figures + summary JSONs from existing parquets only")
    return p.parse_args()


# ===================================================================
# 2. Pre-flight Checks
# ===================================================================

ENRICHED_CATEGORIES = [
    "age", "disability", "gi", "nationality",
    "physical_appearance", "race", "religion", "ses", "so",
]


def preflight(run_dir: Path, args: argparse.Namespace) -> None:
    """Verify all required files exist and there's enough disk space."""
    enriched_dir = run_dir / "data" / "enriched"
    missing: list[str] = []

    # Enriched JSONL files
    for cat in ENRICHED_CATEGORIES:
        fname = cat + ".jsonl"
        if cat == "race" and args.race_handling == "intersectional":
            fname = "race_intersectional.jsonl"
        path = enriched_dir / fname
        if not path.is_file():
            missing.append(str(path))

    # Audit files
    for p in [
        enriched_dir / "enrichment_audit.json",
        run_dir / "diagnostics" / "bbq_vocabulary_audit.json",
    ]:
        if not p.is_file():
            missing.append(str(p))

    if missing:
        log("ERROR: Required enriched data files not found.")
        log("")
        log("Run this on the local machine to sync:")
        log("")
        log(f"  rsync -avz --progress \\")
        log(f"    <local_run_dir>/data/enriched/ \\")
        log(f"    <user>@<runpod_host>:{run_dir}/data/enriched/")
        log("")
        log(f"  rsync -avz --progress \\")
        log(f"    <local_run_dir>/diagnostics/bbq_vocabulary_audit.json \\")
        log(f"    <user>@<runpod_host>:{run_dir}/diagnostics/")
        log("")
        log("Missing files:")
        for m in missing:
            log(f"  - {m}")
        log("")
        log("After syncing, re-run this script.")
        sys.exit(1)

    # Output dir
    output_dir = run_dir / "stage1_geometry_v3"
    if output_dir.exists() and not args.force:
        if not (args.probes_only or args.regenerate_outputs):
            log(f"ERROR: Output directory already exists: {output_dir}")
            log("Use --force to overwrite, or --probes_only / --regenerate_outputs.")
            sys.exit(1)

    # Disk space
    usage = shutil.disk_usage(run_dir)
    free_gb = usage.free / (1024 ** 3)
    if free_gb < 5.0:
        log(f"ERROR: Only {free_gb:.1f} GB free under {run_dir}. Need >= 5 GB.")
        sys.exit(1)

    log(f"Pre-flight OK: {free_gb:.1f} GB free, all required files present")


# ===================================================================
# 3. Data Loading
# ===================================================================

def detect_activations_dir(
    run_dir: Path, config: dict, mode: str,
) -> Path:
    """Auto-detect or explicitly resolve the activations directory."""
    candidates: list[tuple[str, Path]] = []

    if mode in ("auto", "config_key"):
        if "activations_dir" in config:
            p = Path(config["activations_dir"])
            if not p.is_absolute():
                p = run_dir / p
            candidates.append(("config_key", p))
        elif mode == "config_key":
            log("FATAL: --activations_layout=config_key but config.json has no activations_dir")
            sys.exit(1)

    if mode in ("auto", "default_path"):
        candidates.append(("default_path", run_dir / "A_extraction" / "activations"))

    if mode == "auto":
        for label, p in [
            ("activations/", run_dir / "activations"),
            ("data/activations/", run_dir / "data" / "activations"),
            ("extraction/activations/", run_dir / "extraction" / "activations"),
        ]:
            candidates.append((label, p))

    for label, p in candidates:
        if p.is_dir():
            npz_count = len(list(p.rglob("*.npz"))[:5])
            if npz_count > 0:
                log(f"  Activations dir: {p} (detected via {label})")
                return p
            else:
                log(f"  Tried {p} ({label}): exists but no .npz files")
        else:
            log(f"  Tried {p} ({label}): not found")

    log("FATAL: Could not find activations directory. Paths tried:")
    for label, p in candidates:
        log(f"  {p} ({label})")
    sys.exit(1)


def detect_activation_pattern(
    act_dir: Path, enriched_items_by_cat: dict[str, list[dict]],
    stimuli_bridges: dict[str, dict[int, int]],
) -> str:
    """Detect the naming pattern for activation .npz files."""
    # Try: {cat}/item_{item_idx}.npz
    total = 0
    found = 0
    for cat in list(enriched_items_by_cat.keys())[:3]:
        items = enriched_items_by_cat[cat][:20]
        bridge = stimuli_bridges.get(cat, {})
        for item in items:
            eid = item["example_id"]
            # Try reverse bridge: find item_idx for this example_id
            item_idx = None
            for iidx, eidx in bridge.items():
                if eidx == eid:
                    item_idx = iidx
                    break
            if item_idx is None:
                item_idx = eid  # fallback: assume item_idx == example_id

            total += 1
            cat_dir = act_dir / cat
            if (cat_dir / f"item_{item_idx:06d}.npz").is_file():
                found += 1
            elif (cat_dir / f"item_{item_idx}.npz").is_file():
                found += 1

    if total > 0 and found / total >= 0.95:
        log(f"  Activation pattern: {{cat}}/item_{{item_idx}}.npz ({found}/{total} resolved)")
        return "item_{idx}"

    log(f"  WARNING: item_{{idx}} pattern resolved only {found}/{total}")
    log("  Falling back to glob-based resolution")
    return "glob"


def load_enriched_jsonl(
    enriched_dir: Path, categories: list[str], race_handling: str,
) -> dict[str, list[dict]]:
    """Load enriched JSONL files. Returns {cat_short: [items]}."""
    result: dict[str, list[dict]] = {}
    for cat in categories:
        fname = cat + ".jsonl"
        if cat == "race" and race_handling == "intersectional":
            fname = "race_intersectional.jsonl"
        path = enriched_dir / fname
        items = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        result[cat] = items
        log(f"  {cat}: {len(items)} enriched items")
    return result


def load_stimuli_bridges(
    run_dir: Path, categories: list[str],
) -> dict[str, dict[int, int]]:
    """Load item_idx → example_id mapping from stimuli JSONs."""
    bridges: dict[str, dict[int, int]] = {}
    for cat in categories:
        candidates = [
            run_dir / "A_extraction" / "stimuli" / f"{cat}.json",
            run_dir / "stimuli" / f"{cat}.json",
        ]
        processed_dir = PROJECT_ROOT / "data" / "processed"
        if processed_dir.is_dir():
            for p in sorted(processed_dir.glob(f"stimuli_{cat}_*.json")):
                candidates.append(p)
        for path in candidates:
            if path.is_file():
                with open(path) as f:
                    items = json.load(f)
                bridges[cat] = {
                    int(it["item_idx"]): int(it["example_id"])
                    for it in items if "example_id" in it
                }
                log(f"  Stimuli bridge {cat}: {len(bridges[cat])} items from {path.name}")
                break
        if cat not in bridges:
            log(f"  WARNING: no stimuli bridge for {cat}")
    return bridges


def build_reverse_bridges(
    bridges: dict[str, dict[int, int]],
) -> dict[str, dict[int, list[int]]]:
    """Reverse: example_id → [item_idx, ...] per category."""
    rev: dict[str, dict[int, list[int]]] = {}
    for cat, bridge in bridges.items():
        rev_cat: dict[int, list[int]] = defaultdict(list)
        for item_idx, example_id in bridge.items():
            rev_cat[example_id].append(item_idx)
        rev[cat] = dict(rev_cat)
    return rev


def detect_predictions_source(
    run_dir: Path, mode: str,
) -> tuple[str, Path]:
    """Find the predictions / is_biased_response source."""
    meta_enriched = run_dir / "A_extraction" / "metadata_enriched.parquet"
    predictions_dir = run_dir / "predictions"

    if mode == "metadata_enriched":
        if not meta_enriched.is_file():
            log(f"FATAL: --predictions_source=metadata_enriched but "
                f"{meta_enriched} not found")
            sys.exit(1)
        return "metadata_enriched", meta_enriched

    if mode == "predictions_dir":
        if not predictions_dir.is_dir():
            log(f"FATAL: --predictions_source=predictions_dir but "
                f"{predictions_dir} not found")
            sys.exit(1)
        return "predictions_dir", predictions_dir

    # Auto mode
    if meta_enriched.is_file():
        log(f"  Predictions source: metadata_enriched.parquet")
        return "metadata_enriched", meta_enriched

    if predictions_dir.is_dir():
        parquets = list(predictions_dir.glob("*.parquet"))
        if parquets:
            log(f"  Predictions source: predictions/ ({len(parquets)} files)")
            return "predictions_dir", predictions_dir

    log("FATAL: Could not find predictions source. Searched:")
    log(f"  {meta_enriched}")
    log(f"  {predictions_dir}")
    sys.exit(1)


def load_predictions(
    source_type: str, source_path: Path,
) -> pd.DataFrame:
    """Load predictions into a DataFrame with item_idx, category, is_biased_response."""
    if source_type == "metadata_enriched":
        df = pd.read_parquet(source_path)
        required = {"item_idx", "category"}
        if not required.issubset(df.columns):
            log(f"FATAL: metadata_enriched.parquet missing columns: "
                f"{required - set(df.columns)}")
            sys.exit(1)

        # Derive is_biased_response if missing
        if "is_biased_response" not in df.columns:
            if "model_answer_role" not in df.columns:
                log("FATAL: metadata_enriched.parquet has neither "
                    "is_biased_response nor model_answer_role")
                sys.exit(1)
            log("  Deriving is_biased_response from model_answer_role + question_polarity")
            df["is_biased_response"] = df.apply(
                lambda r: _compute_biased(r.get("model_answer_role"),
                                          r.get("question_polarity")),
                axis=1,
            )

        # Normalize category to short name
        df["category_short"] = df["category"].apply(
            lambda c: BBQ_TO_SHORT.get(c, c) if c in BBQ_TO_SHORT else c
        )
        return df

    elif source_type == "predictions_dir":
        frames = []
        for p in sorted(source_path.glob("*.parquet")):
            frames.append(pd.read_parquet(p))
        if not frames:
            log("FATAL: No parquet files in predictions/")
            sys.exit(1)
        df = pd.concat(frames, ignore_index=True)
        df["category_short"] = df["category"].apply(
            lambda c: BBQ_TO_SHORT.get(c, c) if c in BBQ_TO_SHORT else c
        )
        return df

    log(f"FATAL: Unknown predictions source type: {source_type}")
    sys.exit(1)


def _compute_biased(role: Any, polarity: Any) -> int:
    """Derive is_biased_response from model_answer_role and question_polarity."""
    if role is None or (isinstance(role, float) and math.isnan(role)):
        return -1
    if polarity is None or (isinstance(polarity, float) and math.isnan(polarity)):
        return -1
    role = str(role).strip().lower()
    pol = str(polarity).strip().lower()
    if role == "unknown":
        return -1
    if pol == "neg":
        return 1 if role in ("target", "stereotyped_target") else 0
    elif pol in ("nonneg",):
        return 1 if role in ("non_target", "non_stereotyped", "nontarget") else 0
    return -1


# Unified item record used throughout
ItemRecord = dict[str, Any]  # keys: item_idx, example_id, cat, is_biased_response, + enriched fields


def build_unified_items(
    enriched_by_cat: dict[str, list[dict]],
    rev_bridges: dict[str, dict[int, list[int]]],
    predictions_df: pd.DataFrame,
    categories: list[str],
) -> dict[str, list[ItemRecord]]:
    """Join enriched JSONL items with item_idx and is_biased_response.

    Returns {cat_short: [ItemRecord, ...]} for ambig items only.
    """
    # Build prediction lookup: (category_short, item_idx) → is_biased_response
    pred_lookup: dict[tuple[str, int], int] = {}
    for _, row in predictions_df.iterrows():
        cat_s = row.get("category_short", "")
        idx = int(row["item_idx"])
        ibr = int(row.get("is_biased_response", -1))
        pred_lookup[(cat_s, idx)] = ibr

    result: dict[str, list[ItemRecord]] = {}
    for cat in categories:
        items = enriched_by_cat.get(cat, [])
        rev = rev_bridges.get(cat, {})
        records: list[ItemRecord] = []
        n_matched = 0
        n_total = 0

        for item in items:
            if item.get("context_condition") != "ambig":
                continue
            n_total += 1
            eid = int(item["example_id"])

            # Find item_idx via reverse bridge
            item_idxs = rev.get(eid)
            if not item_idxs:
                # Fallback: item_idx == example_id
                item_idxs = [eid]

            # Match by context_condition if multiple
            matched_idx = None
            for iidx in item_idxs:
                matched_idx = iidx
                break  # Take first match (ambig items are unique per example_id)

            if matched_idx is None:
                continue

            ibr = pred_lookup.get((cat, matched_idx), -1)
            n_matched += 1

            records.append({
                "item_idx": matched_idx,
                "example_id": eid,
                "category": cat,
                "question_index": item.get("question_index", "0"),
                "question_polarity": item.get("question_polarity", ""),
                "context_condition": "ambig",
                "mentioned_subgroups_fine": item.get("mentioned_subgroups_fine", []),
                "mentioned_subgroups_aggregated": item.get("mentioned_subgroups_aggregated", []),
                "bias_target_fine": item.get("bias_target_fine", []),
                "bias_target_aggregated": item.get("bias_target_aggregated", []),
                "is_biased_response": ibr,
            })

        result[cat] = records
        match_pct = 100 * n_matched / n_total if n_total > 0 else 0
        log(f"  {cat}: {n_matched}/{n_total} ambig items joined ({match_pct:.0f}%)")
        if match_pct < 99 and n_total > 0:
            log(f"    WARNING: match rate < 99%")

    return result


def load_layer_hidden_states(
    act_dir: Path, categories: list[str], layer: int,
    stimuli_bridges: dict[str, dict[int, int]],
) -> dict[int, np.ndarray]:
    """Load last-token raw hidden states at the given layer for all items."""
    result: dict[int, np.ndarray] = {}
    for cat in categories:
        cat_dir = act_dir / cat
        if not cat_dir.exists():
            log(f"  WARNING: missing activations dir: {cat_dir}")
            continue
        npz_files = sorted(cat_dir.glob("item_*.npz"))
        for item_path in tqdm(npz_files, desc=f"  Loading {cat} L{layer}",
                              unit="items", leave=False):
            try:
                stem = item_path.stem
                item_idx = int(stem.split("_")[1])
                data = np.load(item_path, allow_pickle=True)
                hs_normed = data["hidden_states"][layer].astype(np.float32)
                raw_norm = float(data["hidden_states_raw_norms"][layer])
                result[item_idx] = hs_normed * raw_norm
            except Exception as e:
                log(f"    Failed to load {item_path.name}: {e}")
    return result


# ===================================================================
# 4. Status Determination
# ===================================================================

DirectionKey = tuple[str, str, str, str]  # (cat, gran, subgroup, direction_type)


def _detect_paired_degeneracy(
    records: list[ItemRecord], subgroup: str, field: str,
) -> list[str]:
    """Find subgroups mentioned/targeted in EXACTLY the same item set as `subgroup`."""
    s_items = frozenset(i for i, r in enumerate(records) if subgroup in r.get(field, []))
    if not s_items:
        return []
    all_subs: set[str] = set()
    for r in records:
        all_subs.update(r.get(field, []))
    paired = []
    for other in sorted(all_subs):
        if other == subgroup:
            continue
        other_items = frozenset(i for i, r in enumerate(records) if other in r.get(field, []))
        if other_items == s_items:
            paired.append(other)
    return paired


def determine_all_statuses(
    unified_items: dict[str, list[ItemRecord]],
    categories: list[str],
    min_n: int,
    enrichment_audit: dict,
) -> dict[DirectionKey, dict]:
    """Determine status for every (cat, gran, sub, dir_type) tuple.

    Returns {key: {"status": str, "n_a": int, "n_b": int, "reason": str,
                    "group_a_idxs": list, "group_b_idxs": list,
                    "paired_with": list (optional)}}.
    """
    status_map: dict[DirectionKey, dict] = {}

    for cat in categories:
        records = unified_items.get(cat, [])
        if not records:
            continue

        for gran in ("fine", "aggregated"):
            mention_col = f"mentioned_subgroups_{gran}"
            target_col = f"bias_target_{gran}"

            # Gather all subgroups
            all_subs: set[str] = set()
            for rec in records:
                all_subs.update(rec.get(mention_col, []))
                all_subs.update(rec.get(target_col, []))

            # Check for degenerate mention (all items mention same set)
            mention_sets = [frozenset(r.get(mention_col, [])) for r in records]
            all_same_mention = len(set(mention_sets)) == 1 and len(mention_sets) > 0

            # Pre-compute paired-mention clusters (once per category×granularity)
            mention_paired_cache: dict[str, list[str]] = {}
            target_paired_cache: dict[str, list[str]] = {}
            for sub in sorted(all_subs):
                mention_paired_cache[sub] = _detect_paired_degeneracy(
                    records, sub, mention_col)
                target_paired_cache[sub] = _detect_paired_degeneracy(
                    records, sub, target_col)

            for sub in sorted(all_subs):
                for dir_type in DIRECTION_TYPES:
                    key: DirectionKey = (cat, gran, sub, dir_type)
                    a_idxs, b_idxs = _build_groups(
                        records, sub, dir_type, mention_col, target_col,
                    )

                    # --- Status determination (ordered per spec) ---

                    # 1. STRUCTURALLY_UNDEFINED
                    if len(a_idxs) == 0 and len(b_idxs) == 0:
                        status_map[key] = {
                            "status": S.STRUCTURALLY_UNDEFINED,
                            "n_a": 0, "n_b": 0,
                            "reason": "Both groups empty",
                            "group_a_idxs": [], "group_b_idxs": [],
                        }
                        continue

                    # 2. NEVER_TARGETED
                    if dir_type in ("role", "bias_response"):
                        n_targeting = sum(
                            1 for r in records if sub in r.get(target_col, [])
                        )
                        if n_targeting == 0:
                            status_map[key] = {
                                "status": S.NEVER_TARGETED, "n_a": 0,
                                "n_b": 0,
                                "reason": f"'{sub}' never in {target_col}",
                                "group_a_idxs": [], "group_b_idxs": [],
                            }
                            continue

                    # 3. PAIRED_MENTION_DEGENERATE / PAIRED_TARGET_DEGENERATE
                    if dir_type == "mention":
                        paired = mention_paired_cache.get(sub, [])
                        if paired:
                            status_map[key] = {
                                "status": S.PAIRED_MENTION_DEGENERATE,
                                "n_a": len(a_idxs), "n_b": len(b_idxs),
                                "reason": f"Mention-paired with {paired}",
                                "group_a_idxs": a_idxs, "group_b_idxs": b_idxs,
                                "paired_with": paired,
                            }
                            continue
                    elif dir_type in ("role", "bias_response"):
                        paired = target_paired_cache.get(sub, [])
                        if paired:
                            status_map[key] = {
                                "status": S.PAIRED_TARGET_DEGENERATE,
                                "n_a": len(a_idxs), "n_b": len(b_idxs),
                                "reason": f"Co-targeted with {paired}",
                                "group_a_idxs": a_idxs, "group_b_idxs": b_idxs,
                                "paired_with": paired,
                            }
                            continue

                    # 4. DEGENERATE (all items mention same set)
                    if dir_type == "mention" and all_same_mention:
                        status_map[key] = {
                            "status": S.DEGENERATE, "n_a": len(a_idxs),
                            "n_b": len(b_idxs),
                            "reason": "All items mention the same subgroup set",
                            "group_a_idxs": a_idxs, "group_b_idxs": b_idxs,
                        }
                        continue

                    # 5. INSUFFICIENT_ITEMS
                    if len(a_idxs) < min_n or len(b_idxs) < min_n:
                        status_map[key] = {
                            "status": S.INSUFFICIENT_ITEMS,
                            "n_a": len(a_idxs), "n_b": len(b_idxs),
                            "reason": f"min(|A|={len(a_idxs)}, |B|={len(b_idxs)}) < {min_n}",
                            "group_a_idxs": a_idxs, "group_b_idxs": b_idxs,
                        }
                        continue

                    # 6. NORM_NEAR_ZERO checked after direction computation
                    # 7. COMPUTABLE (tentative — may be downgraded after compute)
                    status_map[key] = {
                        "status": S.COMPUTABLE,
                        "n_a": len(a_idxs), "n_b": len(b_idxs),
                        "reason": "OK",
                        "group_a_idxs": a_idxs, "group_b_idxs": b_idxs,
                    }

    return status_map


def _build_groups(
    records: list[ItemRecord],
    sub: str,
    dir_type: str,
    mention_col: str,
    target_col: str,
) -> tuple[list[int], list[int]]:
    """Build Group A and Group B item_idx lists for a direction."""
    a_idxs: list[int] = []
    b_idxs: list[int] = []

    if dir_type == "mention":
        for r in records:
            if sub in r.get(mention_col, []):
                a_idxs.append(r["item_idx"])
            else:
                b_idxs.append(r["item_idx"])

    elif dir_type == "role":
        for r in records:
            in_target = sub in r.get(target_col, [])
            in_mention = sub in r.get(mention_col, [])
            if in_target:
                a_idxs.append(r["item_idx"])
            elif in_mention and not in_target:
                b_idxs.append(r["item_idx"])

    elif dir_type == "bias_response":
        for r in records:
            if sub not in r.get(target_col, []):
                continue
            ibr = r.get("is_biased_response", -1)
            if ibr == 1:
                a_idxs.append(r["item_idx"])
            elif ibr == 0:
                b_idxs.append(r["item_idx"])
            # ibr == -1 (unknown) excluded from both

    return a_idxs, b_idxs


# ===================================================================
# 5. Direction Computation
# ===================================================================

def compute_directions(
    status_map: dict[DirectionKey, dict],
    hidden_states: dict[int, np.ndarray],
    layer: int,
) -> tuple[dict[DirectionKey, np.ndarray], list[dict]]:
    """Compute directions for all COMPUTABLE keys.

    Returns:
        dir_vectors: {key: unit-normalized direction vector}
        summary_rows: list of dicts for directions_summary.parquet
    """
    dir_vectors: dict[DirectionKey, np.ndarray] = {}
    summary_rows: list[dict] = []

    for key, info in status_map.items():
        cat, gran, sub, dir_type = key
        row_base = {
            "category": cat, "granularity": gran, "subgroup": sub,
            "direction_type": dir_type, "layer": layer,
            "status": info["status"],
            "group_a_n": info["n_a"], "group_b_n": info["n_b"],
        }

        if info["status"] != S.COMPUTABLE:
            row_base["direction_norm"] = None
            row_base["direction_unit_l2"] = None
            row_base["mean_act_norm_layer"] = None
            summary_rows.append(row_base)
            continue

        # Gather vectors
        a_vecs = [hidden_states[i] for i in info["group_a_idxs"] if i in hidden_states]
        b_vecs = [hidden_states[i] for i in info["group_b_idxs"] if i in hidden_states]

        if len(a_vecs) < 1 or len(b_vecs) < 1:
            info["status"] = S.INSUFFICIENT_ITEMS
            info["reason"] = f"No cached activations (a={len(a_vecs)}, b={len(b_vecs)})"
            row_base["status"] = S.INSUFFICIENT_ITEMS
            row_base["direction_norm"] = None
            row_base["direction_unit_l2"] = None
            row_base["mean_act_norm_layer"] = None
            summary_rows.append(row_base)
            continue

        mean_a = np.stack(a_vecs).mean(axis=0).astype(np.float32)
        mean_b = np.stack(b_vecs).mean(axis=0).astype(np.float32)
        raw = (mean_a - mean_b).astype(np.float32)
        norm = float(np.linalg.norm(raw))

        if norm < NORM_EPSILON:
            info["status"] = S.NORM_NEAR_ZERO
            info["reason"] = f"||d|| = {norm:.2e} < {NORM_EPSILON}"
            row_base["status"] = S.NORM_NEAR_ZERO
            row_base["direction_norm"] = norm
            row_base["direction_unit_l2"] = None
            row_base["mean_act_norm_layer"] = None
            summary_rows.append(row_base)
            continue

        unit = raw / norm
        dir_vectors[key] = unit

        all_norms = [np.linalg.norm(v) for v in a_vecs + b_vecs]
        mean_act_norm = float(np.mean(all_norms)) if all_norms else None

        row_base["direction_norm"] = norm
        row_base["direction_unit_l2"] = float(np.linalg.norm(unit))
        row_base["mean_act_norm_layer"] = mean_act_norm
        summary_rows.append(row_base)

    return dir_vectors, summary_rows


def save_directions_npz(
    dir_vectors: dict[DirectionKey, np.ndarray], output_dir: Path,
) -> None:
    """Save all direction vectors to a single .npz file."""
    arrays: dict[str, np.ndarray] = {}
    for (cat, gran, sub, dir_type), vec in dir_vectors.items():
        safe_sub = sub.replace("/", "_").replace(" ", "_")
        key = f"{cat}__{gran}__{safe_sub}__{dir_type}"
        arrays[key] = vec
    if arrays:
        path = output_dir / "directions.npz"
        np.savez(path, **arrays)
        log(f"  Saved {len(arrays)} direction vectors to directions.npz")


# ===================================================================
# 6. Pairwise Cosines
# ===================================================================

def compute_pairwise_cosines(
    dir_vectors: dict[DirectionKey, np.ndarray],
    status_map: dict[DirectionKey, dict],
    layer: int,
) -> list[dict]:
    """Pairwise cosines between subgroups, per (cat, gran, dir_type)."""
    # Group by (cat, gran, dir_type)
    grouped: dict[tuple[str, str, str], list[tuple[str, DirectionKey]]] = defaultdict(list)
    for key in status_map:
        cat, gran, sub, dir_type = key
        grouped[(cat, gran, dir_type)].append((sub, key))

    rows: list[dict] = []
    for (cat, gran, dir_type), subs in grouped.items():
        subs_sorted = sorted(subs, key=lambda x: x[0])
        for i, (sa, ka) in enumerate(subs_sorted):
            for j, (sb, kb) in enumerate(subs_sorted):
                if i >= j:
                    continue
                sta = status_map[ka]["status"]
                stb = status_map[kb]["status"]

                # Hard validity check
                cos_val = float("nan")
                if sta == S.COMPUTABLE and stb == S.COMPUTABLE:
                    va = dir_vectors.get(ka)
                    vb = dir_vectors.get(kb)
                    if va is not None and vb is not None:
                        na = np.linalg.norm(va)
                        nb = np.linalg.norm(vb)
                        if na > NORM_EPSILON and nb > NORM_EPSILON:
                            cos_val = float(np.dot(va, vb))

                rows.append({
                    "category": cat, "granularity": gran,
                    "direction_type": dir_type, "layer": layer,
                    "subgroup_i": sa, "subgroup_j": sb,
                    "cosine": cos_val,
                    "status_i": sta, "status_j": stb,
                })

    return rows


# ===================================================================
# 7. Cross-type Alignment
# ===================================================================

def compute_alignment(
    dir_vectors: dict[DirectionKey, np.ndarray],
    status_map: dict[DirectionKey, dict],
    layer: int,
) -> list[dict]:
    """Cross-direction-type cosines per subgroup."""
    # Group by (cat, gran, sub)
    by_sub: dict[tuple[str, str, str], dict[str, DirectionKey]] = defaultdict(dict)
    for key in status_map:
        cat, gran, sub, dir_type = key
        by_sub[(cat, gran, sub)][dir_type] = key

    rows: list[dict] = []
    for (cat, gran, sub), type_map in by_sub.items():
        for dt_a, dt_b, pair_name in ALIGNMENT_PAIRS:
            ka = type_map.get(dt_a)
            kb = type_map.get(dt_b)
            if ka is None or kb is None:
                continue

            sta = status_map[ka]["status"]
            stb = status_map[kb]["status"]

            cos_val = float("nan")
            if sta == S.COMPUTABLE and stb == S.COMPUTABLE:
                va = dir_vectors.get(ka)
                vb = dir_vectors.get(kb)
                if va is not None and vb is not None:
                    cos_val = float(np.dot(va, vb))

            rows.append({
                "category": cat, "granularity": gran, "subgroup": sub,
                "layer": layer, "alignment_pair": pair_name,
                "cosine": cos_val,
                "status_left": sta, "status_right": stb,
            })

    return rows


# ===================================================================
# 8. Probes
# ===================================================================

def make_probe_pipeline(seed: int, max_iter: int = 1000) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2", C=1.0, solver="liblinear",
            max_iter=max_iter, class_weight="balanced",
            random_state=seed,
        )),
    ])


def load_probe_progress(path: Path) -> dict[tuple, list[dict]]:
    """Load completed probes from JSONL. Returns {(cat,gran,sub,dt,layer): [rows]}."""
    if not path.exists():
        return {}
    completed: dict[tuple, list[dict]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                key = (row["category"], row["granularity"], row["subgroup"],
                       row["direction_type"], row["layer"])
                completed[key].append(row)
            except (json.JSONDecodeError, KeyError):
                continue
    return dict(completed)


def append_probe_result(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, default=_json_safe) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run_all_probes(
    status_map: dict[DirectionKey, dict],
    hidden_states: dict[int, np.ndarray],
    unified_items: dict[str, list[ItemRecord]],
    output_dir: Path,
    layer: int,
    n_folds: int = 5,
    seed: int = 42,
) -> list[dict]:
    """Run probes with per-layer JSONL checkpointing.

    Each layer gets its own checkpoint file. Per-fold rows AND a mean row
    are checkpointed per direction. n_train and n_test are from real
    GroupKFold splits — never zero for computable directions.
    """
    progress_path = output_dir / f"_probe_progress_L{layer}.jsonl"
    completed = load_probe_progress(progress_path)
    if completed:
        log(f"  Resuming L{layer}: {len(completed)} probe directions already completed")

    # Build question_index lookup from unified items
    qi_lookup: dict[int, int] = {}
    for cat, records in unified_items.items():
        for r in records:
            qi = r.get("question_index", "0")
            qi_lookup[r["item_idx"]] = int(qi) if str(qi).isdigit() else hash(qi) % 10000

    rows: list[dict] = []
    for ckpt_rows in completed.values():
        rows.extend(ckpt_rows)
    skipped = 0

    computable_keys = [k for k, v in status_map.items() if v["status"] == S.COMPUTABLE]

    for key in tqdm(computable_keys, desc=f"  Probes L{layer}", unit="probe"):
        cat, gran, sub, dir_type = key
        ckpt_key = (cat, gran, sub, dir_type, layer)
        if ckpt_key in completed:
            skipped += 1
            continue

        info = status_map[key]
        X_list, y_list, groups = [], [], []
        for idx in info["group_a_idxs"]:
            if idx in hidden_states:
                X_list.append(hidden_states[idx])
                y_list.append(1)
                groups.append(qi_lookup.get(idx, idx))
        for idx in info["group_b_idxs"]:
            if idx in hidden_states:
                X_list.append(hidden_states[idx])
                y_list.append(0)
                groups.append(qi_lookup.get(idx, idx))

        row_base: dict[str, Any] = {
            "category": cat, "granularity": gran, "subgroup": sub,
            "direction_type": dir_type, "layer": layer,
        }

        if len(X_list) < 20:
            skip_row = {
                **row_base, "auroc": None, "auroc_ci_low": None,
                "auroc_ci_high": None, "n_train": 0, "n_test": 0,
                "status": S.INSUFFICIENT_ITEMS, "fold_idx": -1,
                "group_a_train": 0, "group_b_train": 0,
                "group_a_test": 0, "group_b_test": 0,
            }
            append_probe_result(progress_path, skip_row)
            rows.append(skip_row)
            continue

        X = np.stack(X_list).astype(np.float32)
        y = np.array(y_list)
        g = np.array(groups)

        n_unique_groups = len(np.unique(g))
        eff_folds = min(n_folds, n_unique_groups)
        if eff_folds < 2:
            skip_row = {
                **row_base, "auroc": None, "auroc_ci_low": None,
                "auroc_ci_high": None, "n_train": int(len(X)), "n_test": 0,
                "status": S.INSUFFICIENT_ITEMS, "fold_idx": -1,
                "group_a_train": 0, "group_b_train": 0,
                "group_a_test": 0, "group_b_test": 0,
            }
            append_probe_result(progress_path, skip_row)
            rows.append(skip_row)
            continue

        gkf = GroupKFold(n_splits=eff_folds)
        fold_aurocs: list[float] = []
        direction_rows: list[dict] = []
        total_n_test = 0

        for fold_i, (tr_idx, te_idx) in enumerate(gkf.split(X, y, g)):
            X_train, X_test = X[tr_idx], X[te_idx]
            y_train, y_test = y[tr_idx], y[te_idx]
            n_train = int(len(tr_idx))
            n_test = int(len(te_idx))

            # Hard assertions
            if n_test == 0:
                log(f"    WARNING: fold {fold_i} has n_test=0 for "
                    f"{cat}/{sub}/{dir_type}. Skipping fold.")
                continue
            if len(np.unique(y_train)) < 2:
                continue
            if len(np.unique(y_test)) < 2:
                # Both classes not in test set — AUROC is undefined
                fold_row = {
                    **row_base, "fold_idx": fold_i, "auroc": None,
                    "auroc_ci_low": None, "auroc_ci_high": None,
                    "n_train": n_train, "n_test": n_test,
                    "group_a_train": int((y_train == 1).sum()),
                    "group_b_train": int((y_train == 0).sum()),
                    "group_a_test": int((y_test == 1).sum()),
                    "group_b_test": int((y_test == 0).sum()),
                    "status": "degenerate_fold",
                }
                direction_rows.append(fold_row)
                total_n_test += n_test
                continue

            try:
                pipe = make_probe_pipeline(seed=seed)
                pipe.fit(X_train, y_train)
                y_score = pipe.predict_proba(X_test)[:, 1]

                # Sanity: y_test and y_score must be same length
                assert len(y_test) == len(y_score) == n_test

                auroc = float(roc_auc_score(y_test, y_score))
                fold_aurocs.append(auroc)
                total_n_test += n_test

                fold_row = {
                    **row_base, "fold_idx": fold_i, "auroc": auroc,
                    "auroc_ci_low": None, "auroc_ci_high": None,
                    "n_train": n_train, "n_test": n_test,
                    "group_a_train": int((y_train == 1).sum()),
                    "group_b_train": int((y_train == 0).sum()),
                    "group_a_test": int((y_test == 1).sum()),
                    "group_b_test": int((y_test == 0).sum()),
                    "status": S.COMPUTABLE,
                }
                direction_rows.append(fold_row)
            except Exception as e:
                log(f"    WARNING: fold {fold_i} failed for "
                    f"{cat}/{sub}/{dir_type}: {e}")
                continue

        # Mean summary row
        mean_auroc = float(np.mean(fold_aurocs)) if fold_aurocs else None
        mean_n_test = total_n_test // max(len(fold_aurocs), 1) if fold_aurocs else 0
        mean_row = {
            **row_base, "fold_idx": -1, "auroc": mean_auroc,
            "auroc_ci_low": None, "auroc_ci_high": None,
            "n_train": int(len(X)) - mean_n_test if fold_aurocs else int(len(X)),
            "n_test": mean_n_test,
            "group_a_train": int((y == 1).sum()),
            "group_b_train": int((y == 0).sum()),
            "group_a_test": 0, "group_b_test": 0,
            "status": S.COMPUTABLE if fold_aurocs else S.INSUFFICIENT_ITEMS,
        }
        direction_rows.append(mean_row)

        # Checkpoint ALL rows for this direction atomically
        for dr in direction_rows:
            append_probe_result(progress_path, dr)
        rows.extend(direction_rows)

    log(f"  Probes L{layer} complete: {len(rows)} total rows, {skipped} resumed")
    return rows


# ===================================================================
# 9. Summary & Audit Discrepancy
# ===================================================================

def build_geometry_summary(
    summary_rows: list[dict],
    cos_rows: list[dict],
    align_rows: list[dict],
    probe_rows: list[dict],
    layers: list[int],
    primary_layer: int,
    race_handling: str,
    min_n: int,
) -> dict:
    """Build geometry_summary.json."""
    dirs_df = pd.DataFrame(summary_rows)
    cos_df = pd.DataFrame(cos_rows) if cos_rows else pd.DataFrame()
    align_df = pd.DataFrame(align_rows) if align_rows else pd.DataFrame()
    probes_df = pd.DataFrame(probe_rows) if probe_rows else pd.DataFrame()

    summary: dict[str, Any] = {
        "primary_layer": primary_layer,
        "layers_computed": layers,
        "race_handling": race_handling,
        "min_n": min_n,
        "n_subgroups_per_category": {},
        "per_category": {},
    }

    for cat in sorted(dirs_df["category"].unique()):
        cat_dirs = dirs_df[dirs_df["category"] == cat]
        cat_summary: dict[str, Any] = {}

        for gran in sorted(cat_dirs["granularity"].unique()):
            gd = cat_dirs[(cat_dirs["granularity"] == gran) & (cat_dirs["layer"] == primary_layer)]
            gs: dict[str, Any] = {"n_subgroups": int(gd["subgroup"].nunique())}

            for dt in DIRECTION_TYPES:
                dt_rows = gd[gd["direction_type"] == dt]
                for st in [S.COMPUTABLE, S.INSUFFICIENT_ITEMS, S.STRUCTURALLY_UNDEFINED,
                           S.DEGENERATE, S.NEVER_TARGETED, S.NORM_NEAR_ZERO]:
                    n = int((dt_rows["status"] == st).sum())
                    if n > 0:
                        gs[f"n_{dt}_{st}"] = n

                # Mean pairwise cosine
                if not cos_df.empty:
                    cc = cos_df[
                        (cos_df["category"] == cat) & (cos_df["granularity"] == gran)
                        & (cos_df["direction_type"] == dt) & (cos_df["layer"] == primary_layer)
                    ]
                    valid = cc["cosine"].dropna()
                    gs[f"mean_pairwise_cosine_{dt}"] = round(float(valid.mean()), 3) if len(valid) > 0 else None

            # Alignment
            for pair_name in ["mention_vs_role", "mention_vs_bias_response", "role_vs_bias_response"]:
                if not align_df.empty:
                    aa = align_df[
                        (align_df["category"] == cat) & (align_df["granularity"] == gran)
                        & (align_df["alignment_pair"] == pair_name) & (align_df["layer"] == primary_layer)
                    ]
                    valid = aa["cosine"].dropna()
                    gs[f"mean_{pair_name}_alignment"] = round(float(valid.mean()), 3) if len(valid) > 0 else None

            # Probe AUROC
            if not probes_df.empty:
                for dt in DIRECTION_TYPES:
                    pp = probes_df[
                        (probes_df["category"] == cat) & (probes_df["granularity"] == gran)
                        & (probes_df["direction_type"] == dt) & (probes_df["layer"] == primary_layer)
                        & (probes_df["fold_idx"] == -1)
                    ]
                    valid = pp["auroc"].dropna()
                    gs[f"mean_probe_auroc_{dt}"] = round(float(valid.mean()), 3) if len(valid) > 0 else None

            cat_summary[gran] = gs

        summary["per_category"][cat] = cat_summary
        n_per_gran = {
            gran: int(cat_dirs[cat_dirs["granularity"] == gran]["subgroup"].nunique())
            for gran in cat_dirs["granularity"].unique()
        }
        summary["n_subgroups_per_category"][cat] = n_per_gran

    return summary


def build_audit_discrepancy(
    status_map: dict[DirectionKey, dict],
    enrichment_audit: dict,
) -> dict:
    """Compare enrichment audit expectations vs script actual."""
    discrepancies: list[dict] = []
    n_audit_yes_script_no = 0
    n_audit_no_script_yes = 0

    for gran_key in ("fine", "aggregated"):
        audit_cats = enrichment_audit.get(gran_key, {})
        for cat, subgroups in audit_cats.items():
            for sg_info in subgroups:
                sub = sg_info["subgroup"]
                for dir_type in DIRECTION_TYPES:
                    key = (cat, gran_key, sub, dir_type)

                    # Audit expectation
                    if dir_type == "mention":
                        audit_computable = sg_info.get("mention", {}).get("computable", False)
                    elif dir_type == "role":
                        audit_computable = sg_info.get("role", {}).get("computable", False)
                    elif dir_type == "bias_response":
                        audit_computable = sg_info.get("bias_response_universe", {}).get(
                            "computable_in_principle", False)
                    else:
                        continue

                    script_info = status_map.get(key)
                    if script_info is None:
                        if audit_computable:
                            n_audit_yes_script_no += 1
                            discrepancies.append({
                                "category": cat, "granularity": gran_key,
                                "subgroup": sub, "direction_type": dir_type,
                                "audit_says": True, "script_status": "not_found",
                                "diagnostic": "Key not in status_map (no enriched items?)",
                            })
                        continue

                    script_computable = script_info["status"] == S.COMPUTABLE
                    if audit_computable and not script_computable:
                        n_audit_yes_script_no += 1
                        discrepancies.append({
                            "category": cat, "granularity": gran_key,
                            "subgroup": sub, "direction_type": dir_type,
                            "audit_says": True, "script_status": script_info["status"],
                            "diagnostic": script_info.get("reason", ""),
                        })
                    elif not audit_computable and script_computable:
                        n_audit_no_script_yes += 1
                        discrepancies.append({
                            "category": cat, "granularity": gran_key,
                            "subgroup": sub, "direction_type": dir_type,
                            "audit_says": False, "script_status": S.COMPUTABLE,
                            "diagnostic": "Script found enough items despite audit expectation",
                        })

    return {
        "n_audit_says_computable_but_script_failed": n_audit_yes_script_no,
        "n_audit_says_uncomputable_but_script_succeeded": n_audit_no_script_yes,
        "discrepancies": discrepancies,
    }


# ===================================================================
# 10. Figures
# ===================================================================

def fig_cosine_heatmap(
    cos_df: pd.DataFrame, status_map: dict[DirectionKey, dict],
    cat: str, dir_type: str, gran: str, layer: int, out_dir: Path,
) -> None:
    df = cos_df[
        (cos_df["category"] == cat) & (cos_df["direction_type"] == dir_type)
        & (cos_df["granularity"] == gran) & (cos_df["layer"] == layer)
    ]
    if df.empty:
        return
    subs = sorted(set(df["subgroup_i"].tolist() + df["subgroup_j"].tolist()))
    n = len(subs)
    if n < 2:
        return
    s2i = {s: i for i, s in enumerate(subs)}
    mat = np.full((n, n), np.nan)
    for _, r in df.iterrows():
        i, j = s2i[r["subgroup_i"]], s2i[r["subgroup_j"]]
        mat[i, j] = r["cosine"]
        mat[j, i] = r["cosine"]
    np.fill_diagonal(mat, 1.0)

    fig, ax = plt.subplots(figsize=(max(5, n * 0.8), max(4, n * 0.7)))

    # Grey hatching for NaN cells
    for i in range(n):
        for j in range(n):
            if np.isnan(mat[i, j]):
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=True,
                    facecolor="#D0D0D0", edgecolor="none",
                ))
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=False,
                    hatch="///", edgecolor="grey", linewidth=0.5,
                ))

    masked = np.ma.array(mat, mask=np.isnan(mat))
    im = ax.imshow(masked, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(subs, rotation=45, ha="right", fontsize=max(6, 10 - n // 5))
    ax.set_yticklabels(subs, fontsize=max(6, 10 - n // 5))

    for i in range(n):
        for j in range(n):
            if not np.isnan(mat[i, j]):
                c = "white" if abs(mat[i, j]) > 0.5 else "black"
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        color=c, fontsize=max(5, 8 - n // 5))

    plt.colorbar(im, ax=ax, shrink=0.8, label="cosine")

    # Footnote: list non-computable subgroups
    non_comp = []
    for s in subs:
        key = (cat, gran, s, dir_type)
        st = status_map.get(key, {}).get("status", "?")
        if st != S.COMPUTABLE:
            non_comp.append(f"{s}={STATUS_SHORT.get(st, st)}")
    if non_comp:
        footnote = "Non-computable: " + ", ".join(non_comp)
        fig.text(0.5, -0.02, footnote, ha="center", fontsize=7, color="grey",
                 transform=ax.transAxes)

    ax.set_title(f"DIM_{dir_type} cosines — {cat} [{gran}] (L{layer})", fontsize=10)
    plt.tight_layout()
    _save_fig(fig, out_dir / f"fig_pairwise_cosines_{cat}_{dir_type}_{gran}.png")


def fig_direction_alignment(
    align_df: pd.DataFrame, status_map: dict[DirectionKey, dict],
    cat: str, gran: str, layer: int, out_dir: Path,
) -> None:
    df = align_df[
        (align_df["category"] == cat) & (align_df["granularity"] == gran)
        & (align_df["layer"] == layer)
    ]
    if df.empty:
        return

    subs = sorted(df["subgroup"].unique())
    pair_names = [p[2] for p in ALIGNMENT_PAIRS]
    pair_labels = ["mention vs role", "mention vs bias", "role vs bias"]
    pair_colors = [ALIGNMENT_PAIR_COLORS[p] for p in pair_names]
    n_subs = len(subs)
    width = 0.25
    x = np.arange(n_subs)

    fig, ax = plt.subplots(figsize=(max(6, n_subs * 1.2), 5))
    for pi, (pname, plabel, pcol) in enumerate(zip(pair_names, pair_labels, pair_colors)):
        vals = []
        for sub in subs:
            row = df[(df["subgroup"] == sub) & (df["alignment_pair"] == pname)]
            if not row.empty and not np.isnan(row["cosine"].iloc[0]):
                vals.append(float(row["cosine"].iloc[0]))
            else:
                vals.append(None)

        for si, v in enumerate(vals):
            xpos = x[si] + (pi - 1) * width
            if v is not None:
                ax.bar(xpos, v, width, color=pcol, alpha=0.85,
                       label=plabel if si == 0 else None)
            else:
                # Missing bar annotation
                ax.bar(xpos, 0, width, color="#D0D0D0", alpha=0.4)
                # Determine status
                dt_a, dt_b = ALIGNMENT_PAIRS[pi][0], ALIGNMENT_PAIRS[pi][1]
                st_a = status_map.get((cat, gran, subs[si], dt_a), {}).get("status", "?")
                st_b = status_map.get((cat, gran, subs[si], dt_b), {}).get("status", "?")
                reason = STATUS_SHORT.get(st_a, st_a) if st_a != S.COMPUTABLE else STATUS_SHORT.get(st_b, st_b)
                ax.text(xpos, 0.02, f"[{reason}]", ha="center", va="bottom",
                        fontsize=5, rotation=90, color="grey")

    ax.axhline(0, color="gray", linestyle=":", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(subs, rotation=45, ha="right", fontsize=max(6, 9 - n_subs // 5))
    ax.set_ylabel("Cosine similarity")
    ax.set_title(f"Direction alignment — {cat} [{gran}] (L{layer})", fontsize=10)
    handles, labels = ax.get_legend_handles_labels()
    # Add "no data" entry
    handles.append(mpatches.Patch(facecolor="#D0D0D0", alpha=0.4, label="no data"))
    labels.append("no data")
    ax.legend(handles, labels, fontsize=7, loc="best")
    plt.tight_layout()
    _save_fig(fig, out_dir / f"fig_direction_alignment_{cat}_{gran}.png")


def fig_norm_comparison(
    summary_rows: list[dict], status_map: dict[DirectionKey, dict],
    primary_layer: int, out_dir: Path,
) -> None:
    df = pd.DataFrame(summary_rows)
    df = df[df["layer"] == primary_layer]
    if df.empty:
        return

    categories = sorted(df["category"].unique())
    n_cats = len(categories)
    n_cols = min(3, n_cats)
    n_rows = (n_cats + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)

    for ci, cat in enumerate(categories):
        ax = axes[ci // n_cols, ci % n_cols]
        cat_df = df[df["category"] == cat]
        grans = sorted(cat_df["granularity"].unique())
        gran = "fine" if "fine" in grans else grans[0]
        cat_df = cat_df[cat_df["granularity"] == gran]
        subs = sorted(cat_df["subgroup"].unique())
        n_subs = len(subs)
        x_arr = np.arange(n_subs)
        width = 0.25

        for di, dt in enumerate(DIRECTION_TYPES):
            vals = []
            for sub in subs:
                row = cat_df[(cat_df["subgroup"] == sub) & (cat_df["direction_type"] == dt)]
                if not row.empty and row["status"].iloc[0] == S.COMPUTABLE:
                    vals.append(float(row["direction_norm"].iloc[0]) if row["direction_norm"].notna().any() else 0)
                else:
                    vals.append(None)

            for si, v in enumerate(vals):
                xpos = x_arr[si] + (di - 1) * width
                if v is not None:
                    ax.bar(xpos, v, width, color=DIRECTION_TYPE_COLORS[dt],
                           alpha=0.85, label=dt if si == 0 else None)
                else:
                    ax.bar(xpos, 0, width, color="#D0D0D0", alpha=0.3)

        ax.set_xticks(x_arr)
        ax.set_xticklabels(subs, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Direction norm")
        ax.set_title(f"{cat} [{gran}]", fontsize=10)
        if ci == 0:
            ax.legend(fontsize=7)

    for ci in range(n_cats, n_rows * n_cols):
        axes[ci // n_cols, ci % n_cols].set_visible(False)

    fig.suptitle("Direction norms by type", fontsize=12)
    plt.tight_layout()
    _save_fig(fig, out_dir / "fig_norm_comparison.png")


def fig_probe_accuracies(
    probe_rows: list[dict], out_dir: Path,
) -> None:
    df = pd.DataFrame(probe_rows)
    df = df[(df["fold_idx"] == -1) & df["auroc"].notna()]
    if df.empty:
        return

    categories = sorted(df["category"].unique())
    n_cats = len(categories)
    n_cols = min(3, n_cats)
    n_rows = (n_cats + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)

    for ci, cat in enumerate(categories):
        ax = axes[ci // n_cols, ci % n_cols]
        cat_df = df[df["category"] == cat]
        grans = sorted(cat_df["granularity"].unique())
        gran = "fine" if "fine" in grans else grans[0]
        cat_df = cat_df[cat_df["granularity"] == gran]
        subs = sorted(cat_df["subgroup"].unique())
        n_subs = len(subs)
        x_arr = np.arange(n_subs)
        width = 0.25

        for di, dt in enumerate(DIRECTION_TYPES):
            vals = []
            for sub in subs:
                row = cat_df[(cat_df["subgroup"] == sub) & (cat_df["direction_type"] == dt)]
                vals.append(float(row["auroc"].iloc[0]) if not row.empty else np.nan)
            ax.bar(x_arr + (di - 1) * width, vals, width,
                   color=DIRECTION_TYPE_COLORS[dt], alpha=0.85,
                   label=dt if ci == 0 else None)

        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.7)
        ax.set_xticks(x_arr)
        ax.set_xticklabels(subs, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("AUROC")
        ax.set_ylim(0.35, 1.05)
        ax.set_title(f"{cat} [{gran}]", fontsize=10)
        if ci == 0:
            ax.legend(fontsize=7, loc="lower right")

    for ci in range(n_cats, n_rows * n_cols):
        axes[ci // n_cols, ci % n_cols].set_visible(False)

    fig.suptitle("Probe AUROC by direction type", fontsize=12)
    plt.tight_layout()
    _save_fig(fig, out_dir / "fig_probe_accuracies.png")


def fig_status_summary(
    status_map: dict[DirectionKey, dict],
    primary_layer: int,
    out_dir: Path,
) -> None:
    """Stacked bar showing status distribution per category."""
    # Filter to primary layer only (status_map doesn't have layer, but directions do)
    # Status is layer-independent for most statuses, so just use the map directly
    cat_status_counts: dict[str, Counter] = defaultdict(Counter)
    for (cat, gran, sub, dir_type), info in status_map.items():
        cat_status_counts[cat][info["status"]] += 1

    categories = sorted(cat_status_counts.keys())
    all_statuses = [S.COMPUTABLE, S.INSUFFICIENT_ITEMS, S.STRUCTURALLY_UNDEFINED,
                    S.DEGENERATE, S.NEVER_TARGETED, S.NORM_NEAR_ZERO]

    fig, ax = plt.subplots(figsize=(max(8, len(categories) * 1.5), 5))
    x = np.arange(len(categories))
    bottoms = np.zeros(len(categories))

    for st in all_statuses:
        vals = [cat_status_counts[cat].get(st, 0) for cat in categories]
        if sum(vals) == 0:
            continue
        ax.bar(x, vals, bottom=bottoms, color=STATUS_COLORS.get(st, "#999"),
               label=st, alpha=0.85)
        bottoms += np.array(vals, dtype=float)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Count (gran x sub x dir_type)")
    ax.set_title("Direction status distribution by category", fontsize=11)
    ax.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    _save_fig(fig, out_dir / "fig_status_summary.png")


def generate_all_figures(
    cos_rows: list[dict], align_rows: list[dict],
    summary_rows: list[dict], probe_rows: list[dict],
    status_map: dict[DirectionKey, dict],
    categories: list[str], primary_layer: int,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cos_df = pd.DataFrame(cos_rows) if cos_rows else pd.DataFrame()
    align_df = pd.DataFrame(align_rows) if align_rows else pd.DataFrame()

    # Cosine heatmaps
    if not cos_df.empty:
        for cat in categories:
            for gran in cos_df[cos_df["category"] == cat]["granularity"].unique():
                for dt in DIRECTION_TYPES:
                    fig_cosine_heatmap(cos_df, status_map, cat, dt, gran,
                                       primary_layer, out_dir)

    # Alignment
    if not align_df.empty:
        for cat in categories:
            for gran in align_df[align_df["category"] == cat]["granularity"].unique():
                fig_direction_alignment(align_df, status_map, cat, gran,
                                        primary_layer, out_dir)

    # Norms
    fig_norm_comparison(summary_rows, status_map, primary_layer, out_dir)

    # Probes
    if probe_rows:
        fig_probe_accuracies(probe_rows, out_dir)

    # Status summary
    fig_status_summary(status_map, primary_layer, out_dir)

    log(f"  All figures saved to {out_dir}")


# ===================================================================
# 11. Self-validation
# ===================================================================

def self_validate(
    output_dir: Path,
    cos_rows: list[dict],
    summary_rows: list[dict],
    probe_rows: list[dict],
    status_map: dict[DirectionKey, dict],
    categories: list[str],
    primary_layer: int,
    discrepancy: dict,
) -> bool:
    """Run the 11-check self-validation suite. Returns True if all hard checks pass."""
    checks: list[tuple[str, bool, str, bool]] = []  # (name, passed, detail, is_hard_fail)

    # 1. All output files exist and non-empty
    for name in ["directions.npz", "directions_summary.parquet",
                  "pairwise_cosines.parquet", "direction_alignment.parquet",
                  "geometry_summary.json", "audit_discrepancy_report.json",
                  "status_inventory.json"]:
        p = output_dir / name
        exists = p.is_file()
        sz = p.stat().st_size if exists else 0
        checks.append((f"{name} exists", exists and sz > 0,
                        f"exists={exists}, size={sz}", True))
    fig_dir = output_dir / "figures"
    fig_count = len(list(fig_dir.glob("*.png"))) if fig_dir.exists() else 0
    checks.append(("Figures exist", fig_count > 0, f"{fig_count} PNGs", True))

    # 2. All 9 categories at primary layer
    dirs_df = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()
    for cat in categories:
        has = False
        if not dirs_df.empty:
            has = len(dirs_df[
                (dirs_df["category"] == cat) & (dirs_df["layer"] == primary_layer)
            ]) > 0
        checks.append((f"{cat} at L{primary_layer}", has, "", True))

    # 3. No cosine=1.0 between distinct subgroups
    cos_df = pd.DataFrame(cos_rows) if cos_rows else pd.DataFrame()
    if not cos_df.empty:
        exact_ones = cos_df[
            (cos_df["cosine"] == 1.0)
            & (cos_df["subgroup_i"] != cos_df["subgroup_j"])
        ]
        n_ones = len(exact_ones)
        checks.append((
            "No cosine=1.0 between distinct subgroups",
            n_ones == 0,
            f"{n_ones} pairs" if n_ones > 0 else "OK", True,
        ))
        if n_ones > 0:
            for _, r in exact_ones.head(10).iterrows():
                log(f"    cos=1.0: {r['category']}/{r['subgroup_i']} vs {r['subgroup_j']} "
                    f"({r['direction_type']}, {r['granularity']})")
    else:
        checks.append(("No cosine=1.0 (empty df)", True, "OK", True))

    # 4. No probe n_test==0 for computable directions
    probes_df = pd.DataFrame(probe_rows) if probe_rows else pd.DataFrame()
    if not probes_df.empty:
        bad_probes = probes_df[
            (probes_df["status"] == S.COMPUTABLE)
            & (probes_df["fold_idx"] >= 0)
            & (probes_df["n_test"] == 0)
        ]
        checks.append((
            "No computable probe fold with n_test=0",
            len(bad_probes) == 0,
            f"{len(bad_probes)} bad rows" if len(bad_probes) > 0 else "OK", True,
        ))
    else:
        checks.append(("Probe n_test check (no probes)", True, "skipped", False))

    # 5. AUROC distribution (warn, not hard fail)
    if not probes_df.empty:
        mean_probes = probes_df[
            (probes_df["fold_idx"] == -1) & probes_df["auroc"].notna()
            & (probes_df["status"] == S.COMPUTABLE)
        ]
        if not mean_probes.empty:
            log("\n  Per-direction-type mean AUROC distribution:")
            for dt in DIRECTION_TYPES:
                dt_aurocs = mean_probes[mean_probes["direction_type"] == dt]["auroc"]
                if len(dt_aurocs) > 0:
                    below_04 = int((dt_aurocs < 0.4).sum())
                    above_095 = int((dt_aurocs > 0.95).sum())
                    log(f"    {dt:15s} min={dt_aurocs.min():.3f} median={dt_aurocs.median():.3f} "
                        f"max={dt_aurocs.max():.3f} n_below_0.4={below_04} n_above_0.95={above_095}")
                    if below_04 > 0:
                        bad = mean_probes[
                            (mean_probes["direction_type"] == dt) & (mean_probes["auroc"] < 0.4)
                        ]
                        for _, r in bad.iterrows():
                            log(f"      AUROC<0.4: {r['category']}/{r['subgroup']} = {r['auroc']:.3f}")
            median_all = mean_probes["auroc"].median()
            checks.append((
                "AUROC median in [0.5, 0.95]",
                0.5 <= median_all <= 0.95,
                f"median={median_all:.3f}", False,
            ))

    # 6. Status counts add up
    if not dirs_df.empty:
        for cat in categories:
            for gran in dirs_df[dirs_df["category"] == cat]["granularity"].unique():
                layer_df = dirs_df[
                    (dirs_df["category"] == cat) & (dirs_df["granularity"] == gran)
                    & (dirs_df["layer"] == primary_layer)
                ]
                n_subs = layer_df["subgroup"].nunique()
                for dt in DIRECTION_TYPES:
                    n_dt = len(layer_df[layer_df["direction_type"] == dt])
                    if n_dt != n_subs:
                        checks.append((
                            f"Status count {cat}/{gran}/{dt}",
                            False, f"{n_dt} rows vs {n_subs} subgroups", True,
                        ))
                        break
        else:
            checks.append(("Status counts consistent", True, "OK", True))

    # 7. Non-COMPUTABLE → NaN norm
    norm_ok = True
    for row in summary_rows:
        if row["status"] != S.COMPUTABLE and row.get("direction_norm") is not None:
            if not (isinstance(row["direction_norm"], float) and math.isnan(row["direction_norm"])):
                norm_ok = False
                break
    checks.append(("Non-COMPUTABLE → NaN norm", norm_ok, "", True))

    # 8. Audit discrepancies surfaced (info only, not hard fail)
    checks.append((
        "Audit discrepancies reported",
        True,
        f"yes→no={discrepancy.get('n_audit_says_computable_but_script_failed', '?')}, "
        f"no→yes={discrepancy.get('n_audit_says_uncomputable_but_script_succeeded', '?')}",
        False,
    ))
    # Print full list
    for d in discrepancy.get("discrepancies", []):
        log(f"    {d['category']}/{d['subgroup']}/{d['direction_type']} "
            f"[{d['granularity']}]: audit={d['audit_says']}, script={d['script_status']} "
            f"— {d.get('diagnostic', '')}")

    # 9. Probe AUROC sanity
    if not probes_df.empty:
        mean_probes = probes_df[
            (probes_df["fold_idx"] == -1) & probes_df["auroc"].notna()
            & (probes_df["status"] == S.COMPUTABLE)
        ]
        n_below_04 = int((mean_probes["auroc"] < 0.4).sum()) if not mean_probes.empty else 0
        n_total = len(mean_probes)
        pct = 100 * n_below_04 / n_total if n_total > 0 else 0
        checks.append((
            "Probe AUROC<0.4 count",
            pct <= 5.0,
            f"{n_below_04}/{n_total} ({pct:.1f}%)", pct > 5.0,
        ))

    # 10. Paired degeneracy counts
    paired_mention = sum(1 for v in status_map.values()
                         if v["status"] == S.PAIRED_MENTION_DEGENERATE)
    paired_target = sum(1 for v in status_map.values()
                        if v["status"] == S.PAIRED_TARGET_DEGENERATE)
    log(f"\n  Paired degeneracy: {paired_mention} mention, {paired_target} target")
    for cat in categories:
        pm = sum(1 for (c, g, s, d), v in status_map.items()
                 if c == cat and v["status"] == S.PAIRED_MENTION_DEGENERATE)
        pt = sum(1 for (c, g, s, d), v in status_map.items()
                 if c == cat and v["status"] == S.PAIRED_TARGET_DEGENERATE)
        if pm + pt > 0:
            log(f"    {cat}: {pm} mention-paired, {pt} target-paired")
    checks.append(("Paired degeneracy reported", True,
                    f"mention={paired_mention}, target={paired_target}", False))

    # 11. Primary layer = 14
    summary_path = output_dir / "geometry_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            gs = json.load(f)
        actual_primary = gs.get("primary_layer")
        checks.append((
            "Primary layer = 14",
            actual_primary == 14,
            f"actual={actual_primary}", True,
        ))

    # Print results
    log("\n" + "=" * 72)
    log("SELF-VALIDATION")
    log("=" * 72)
    all_pass = True
    for name, passed, detail, is_hard in checks:
        status = "PASS" if passed else ("FAIL" if is_hard else "WARN")
        if not passed and is_hard:
            all_pass = False
        log(f"  [{status}] {name}: {detail}")

    return all_pass


# ===================================================================
# 12. Main
# ===================================================================

def _load_common_data(args: argparse.Namespace, run_dir: Path) -> tuple[
    dict, list[str], dict, dict, dict, Path, pd.DataFrame, dict,
]:
    """Load data shared across all modes. Returns (config, categories,
    enriched_by_cat, bridges, rev_bridges, act_dir, predictions_df,
    enrichment_audit)."""
    config: dict = {}
    config_path = run_dir / "config.json"
    if config_path.is_file():
        with open(config_path) as f:
            config = json.load(f)
        log(f"  Loaded config.json")
    else:
        log(f"  WARNING: config.json not found at {config_path}")

    enriched_dir = run_dir / "data" / "enriched"
    categories = list(ENRICHED_CATEGORIES)

    log("\nLoading enriched JSONL files...")
    enriched_by_cat = load_enriched_jsonl(enriched_dir, categories, args.race_handling)

    log("\nLoading stimuli bridges...")
    bridges = load_stimuli_bridges(run_dir, categories)
    rev_bridges = build_reverse_bridges(bridges)

    log("\nDetecting activations directory...")
    act_dir = detect_activations_dir(run_dir, config, args.activations_layout)

    log("\nDetecting predictions source...")
    pred_type, pred_path = detect_predictions_source(run_dir, args.predictions_source)
    predictions_df = load_predictions(pred_type, pred_path)
    log(f"  Loaded {len(predictions_df)} prediction rows")

    with open(enriched_dir / "enrichment_audit.json") as f:
        enrichment_audit = json.load(f)

    return (config, categories, enriched_by_cat, bridges, rev_bridges,
            act_dir, predictions_df, enrichment_audit)


def _generate_post_outputs(
    output_dir: Path, all_summary_rows: list[dict], all_cos_rows: list[dict],
    all_align_rows: list[dict], all_probe_rows: list[dict],
    global_status_map: dict[DirectionKey, dict], enrichment_audit: dict,
    categories: list[str], layers: list[int], primary_layer: int,
    race_handling: str, min_n: int,
) -> dict:
    """Generate summary JSONs, figures, and run validation. Returns discrepancy."""
    # ── Summary JSONs ───────────────────────────────────────────────────
    log("\nBuilding summaries...")
    summary = build_geometry_summary(
        all_summary_rows, all_cos_rows, all_align_rows, all_probe_rows,
        layers, primary_layer, race_handling, min_n,
    )
    atomic_save_json(summary, output_dir / "geometry_summary.json")

    status_inv = {
        f"{cat}__{gran}__{sub}__{dt}": {
            "status": info["status"], "reason": info["reason"],
            "n_a": info["n_a"], "n_b": info["n_b"],
        }
        for (cat, gran, sub, dt), info in global_status_map.items()
    }
    atomic_save_json(status_inv, output_dir / "status_inventory.json")

    discrepancy = build_audit_discrepancy(global_status_map, enrichment_audit)
    atomic_save_json(discrepancy, output_dir / "audit_discrepancy_report.json")

    log(f"\nAudit discrepancy: "
        f"{discrepancy['n_audit_says_computable_but_script_failed']} audit-yes/script-no, "
        f"{discrepancy['n_audit_says_uncomputable_but_script_succeeded']} audit-no/script-yes")
    if discrepancy["discrepancies"]:
        for d in discrepancy["discrepancies"][:10]:
            log(f"  {d['category']}/{d['subgroup']}/{d['direction_type']} "
                f"[{d['granularity']}]: audit={d['audit_says']}, "
                f"script={d['script_status']}")

    log("\n" + "=" * 72)
    log("GEOMETRY SUMMARY")
    log("=" * 72)
    for cat, cat_info in summary.get("per_category", {}).items():
        for gran, ginfo in cat_info.items():
            log(f"\n  {cat} [{gran}]:")
            for k, v in ginfo.items():
                log(f"    {k}: {v}")

    # ── Figures ─────────────────────────────────────────────────────────
    log("\nGenerating figures...")
    try:
        generate_all_figures(
            all_cos_rows, all_align_rows, all_summary_rows, all_probe_rows,
            global_status_map, categories, primary_layer,
            output_dir / "figures",
        )
    except Exception as e:
        log(f"  ERROR during figure generation: {e}")
        import traceback
        traceback.print_exc()

    return discrepancy


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    t0 = time.time()

    layers = [int(x) for x in args.layers.split(",")]
    primary_layer = args.primary_layer
    if primary_layer not in layers:
        layers.append(primary_layer)
        layers.sort()

    log("Stage 1 v3: Corrected Geometry Pipeline")
    log(f"  Run dir: {run_dir}")
    log(f"  Layers: {layers} (primary: {primary_layer})")
    log(f"  Min N: {args.min_n}")
    log(f"  Race handling: {args.race_handling}")
    if args.probes_only:
        log(f"  Mode: PROBES ONLY")
    elif args.regenerate_outputs:
        log(f"  Mode: REGENERATE OUTPUTS ONLY")
    else:
        log(f"  Mode: FULL PIPELINE")

    # ── Pre-flight ──────────────────────────────────────────────────────
    preflight(run_dir, args)

    output_dir = run_dir / "stage1_geometry_v3"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)

    # ================================================================
    # MODE: --regenerate_outputs
    # ================================================================
    if args.regenerate_outputs:
        log("\nRegenerate mode: loading existing parquets...")
        enriched_dir = run_dir / "data" / "enriched"
        categories = list(ENRICHED_CATEGORIES)

        # Load parquets
        all_summary_rows = pd.read_parquet(
            output_dir / "directions_summary.parquet").to_dict("records")
        all_cos_rows = pd.read_parquet(
            output_dir / "pairwise_cosines.parquet").to_dict("records")
        all_align_rows = pd.read_parquet(
            output_dir / "direction_alignment.parquet").to_dict("records")

        probe_path = output_dir / "probe_accuracies.parquet"
        all_probe_rows = pd.read_parquet(probe_path).to_dict("records") if probe_path.exists() else []

        with open(enriched_dir / "enrichment_audit.json") as f:
            enrichment_audit = json.load(f)

        # Rebuild status map from directions_summary
        global_status_map: dict[DirectionKey, dict] = {}
        for row in all_summary_rows:
            key = (row["category"], row["granularity"], row["subgroup"], row["direction_type"])
            global_status_map[key] = {
                "status": row["status"], "n_a": row.get("group_a_n", 0),
                "n_b": row.get("group_b_n", 0), "reason": row["status"],
                "group_a_idxs": [], "group_b_idxs": [],
            }

        discrepancy = _generate_post_outputs(
            output_dir, all_summary_rows, all_cos_rows, all_align_rows,
            all_probe_rows, global_status_map, enrichment_audit,
            categories, layers, primary_layer, args.race_handling, args.min_n,
        )

        all_pass = self_validate(
            output_dir, all_cos_rows, all_summary_rows,
            all_probe_rows if 'all_probe_rows' in dir() else [],
            global_status_map, categories, primary_layer, discrepancy,
        )
        elapsed = time.time() - t0
        log(f"\nRegenerate complete in {elapsed:.1f}s")
        if not all_pass:
            sys.exit(1)
        return

    # ================================================================
    # Shared data loading (full mode and probes_only)
    # ================================================================
    (config, categories, enriched_by_cat, bridges, rev_bridges,
     act_dir, predictions_df, enrichment_audit) = _load_common_data(args, run_dir)

    log("\nBuilding unified item table...")
    unified_items = build_unified_items(
        enriched_by_cat, rev_bridges, predictions_df, categories,
    )

    # ================================================================
    # MODE: --probes_only
    # ================================================================
    if args.probes_only:
        log("\nProbes-only mode: loading existing direction outputs...")

        # Load existing parquets
        all_summary_rows = pd.read_parquet(
            output_dir / "directions_summary.parquet").to_dict("records")
        all_cos_rows = pd.read_parquet(
            output_dir / "pairwise_cosines.parquet").to_dict("records")
        all_align_rows = pd.read_parquet(
            output_dir / "direction_alignment.parquet").to_dict("records")

        # Rebuild status map from metadata (need group_a/b_idxs for probes)
        log("  Rebuilding status map...")
        global_status_map = determine_all_statuses(
            unified_items, categories, args.min_n, enrichment_audit,
        )

        # Delete stale probe JSONL files so we start clean
        for old_jsonl in output_dir.glob("_probe_progress_L*.jsonl"):
            old_jsonl.unlink()
            log(f"  Deleted stale {old_jsonl.name}")
        old_flat = output_dir / "_probe_progress.jsonl"
        if old_flat.exists():
            old_flat.unlink()
            log(f"  Deleted stale _probe_progress.jsonl")

        # Run probes — one layer at a time, no duplicate
        all_probe_rows: list[dict] = []
        for layer in layers:
            log(f"\nLoading hidden states for L{layer}...")
            hs = load_layer_hidden_states(act_dir, categories, layer, bridges)
            log(f"  Loaded {len(hs)} items")
            probe_rows = run_all_probes(
                global_status_map, hs, unified_items, output_dir,
                layer=layer, n_folds=args.probe_n_folds, seed=42,
            )
            all_probe_rows.extend(probe_rows)
            del hs

        probes_df = pd.DataFrame(all_probe_rows)
        if not probes_df.empty:
            tmp = output_dir / ".probe_accuracies.parquet.tmp"
            probes_df.to_parquet(tmp, index=False, compression="snappy")
            os.rename(tmp, output_dir / "probe_accuracies.parquet")
        log(f"\n  Saved probe_accuracies.parquet: {len(probes_df)} rows")

        # Regenerate all post-outputs (figures, summaries)
        discrepancy = _generate_post_outputs(
            output_dir, all_summary_rows, all_cos_rows, all_align_rows,
            all_probe_rows, global_status_map, enrichment_audit,
            categories, layers, primary_layer, args.race_handling, args.min_n,
        )

        all_pass = self_validate(
            output_dir, all_cos_rows, all_summary_rows,
            all_probe_rows if 'all_probe_rows' in dir() else [],
            global_status_map, categories, primary_layer, discrepancy,
        )
        elapsed = time.time() - t0
        log(f"\nProbes-only complete in {elapsed:.1f}s")
        if not all_pass:
            sys.exit(1)
        return

    # ================================================================
    # MODE: FULL PIPELINE
    # ================================================================
    all_summary_rows: list[dict] = []
    all_cos_rows: list[dict] = []
    all_align_rows: list[dict] = []
    all_dir_vectors: dict[DirectionKey, np.ndarray] = {}
    global_status_map: dict[DirectionKey, dict] = {}

    for layer in layers:
        ckpt_path = output_dir / "checkpoints" / f"layer_{layer}_done.json"
        if ckpt_path.exists() and not args.force:
            log(f"\nLayer {layer}: LOADED from checkpoint")
            ckpt_summary = output_dir / "checkpoints" / f"summary_L{layer}.parquet"
            ckpt_cos = output_dir / "checkpoints" / f"cosines_L{layer}.parquet"
            ckpt_align = output_dir / "checkpoints" / f"alignment_L{layer}.parquet"
            if ckpt_summary.exists():
                all_summary_rows.extend(pd.read_parquet(ckpt_summary).to_dict("records"))
            if ckpt_cos.exists():
                all_cos_rows.extend(pd.read_parquet(ckpt_cos).to_dict("records"))
            if ckpt_align.exists():
                all_align_rows.extend(pd.read_parquet(ckpt_align).to_dict("records"))
            continue

        log(f"\n{'=' * 60}")
        log(f"Layer {layer}")
        log(f"{'=' * 60}")

        hs = load_layer_hidden_states(act_dir, categories, layer, bridges)
        log(f"  Loaded hidden states for {len(hs)} items")

        if len(hs) == 0:
            log("  WARNING: No hidden states loaded for this layer, skipping")
            continue

        log("  Determining direction statuses...")
        status_map = determine_all_statuses(
            unified_items, categories, args.min_n, enrichment_audit,
        )
        n_computable = sum(1 for v in status_map.values() if v["status"] == S.COMPUTABLE)
        n_total = len(status_map)
        log(f"  {n_computable}/{n_total} directions computable")

        global_status_map.update(status_map)

        log("  Computing directions...")
        dir_vectors, summary_rows = compute_directions(status_map, hs, layer)
        all_dir_vectors.update(dir_vectors)
        log(f"  {len(dir_vectors)} direction vectors computed")

        log("  Computing pairwise cosines...")
        cos_rows = compute_pairwise_cosines(dir_vectors, status_map, layer)

        log("  Computing cross-type alignment...")
        align_rows = compute_alignment(dir_vectors, status_map, layer)

        pd.DataFrame(summary_rows).to_parquet(
            output_dir / "checkpoints" / f"summary_L{layer}.parquet", index=False)
        pd.DataFrame(cos_rows).to_parquet(
            output_dir / "checkpoints" / f"cosines_L{layer}.parquet", index=False)
        pd.DataFrame(align_rows).to_parquet(
            output_dir / "checkpoints" / f"alignment_L{layer}.parquet", index=False)
        with open(ckpt_path, "w") as f:
            json.dump({"layer": layer, "status": "done"}, f)
        log(f"  Layer {layer} checkpoint saved")

        all_summary_rows.extend(summary_rows)
        all_cos_rows.extend(cos_rows)
        all_align_rows.extend(align_rows)

        del hs

    # Save consolidated direction outputs
    log("\nSaving consolidated outputs...")
    dirs_df = pd.DataFrame(all_summary_rows)
    tmp = output_dir / ".directions_summary.parquet.tmp"
    dirs_df.to_parquet(tmp, index=False, compression="snappy")
    os.rename(tmp, output_dir / "directions_summary.parquet")

    cos_df = pd.DataFrame(all_cos_rows)
    tmp = output_dir / ".pairwise_cosines.parquet.tmp"
    cos_df.to_parquet(tmp, index=False, compression="snappy")
    os.rename(tmp, output_dir / "pairwise_cosines.parquet")

    align_df = pd.DataFrame(all_align_rows)
    tmp = output_dir / ".direction_alignment.parquet.tmp"
    align_df.to_parquet(tmp, index=False, compression="snappy")
    os.rename(tmp, output_dir / "direction_alignment.parquet")

    save_directions_npz(all_dir_vectors, output_dir)

    # Probes — single pass, one layer at a time
    all_probe_rows: list[dict] = []
    if not args.skip_probes:
        log("\nComputing probes...")
        for layer in layers:
            hs = load_layer_hidden_states(act_dir, categories, layer, bridges)
            probe_rows = run_all_probes(
                global_status_map, hs, unified_items, output_dir,
                layer=layer, n_folds=args.probe_n_folds, seed=42,
            )
            all_probe_rows.extend(probe_rows)
            del hs

        probes_df = pd.DataFrame(all_probe_rows)
        if not probes_df.empty:
            tmp = output_dir / ".probe_accuracies.parquet.tmp"
            probes_df.to_parquet(tmp, index=False, compression="snappy")
            os.rename(tmp, output_dir / "probe_accuracies.parquet")
        log(f"  Saved probe_accuracies.parquet: {len(probes_df)} rows")

    # Post-outputs (summaries, figures) — always run
    discrepancy = _generate_post_outputs(
        output_dir, all_summary_rows, all_cos_rows, all_align_rows,
        all_probe_rows, global_status_map, enrichment_audit,
        categories, layers, primary_layer, args.race_handling, args.min_n,
    )

    all_pass = self_validate(
        output_dir, all_cos_rows, all_summary_rows, all_probe_rows,
        global_status_map, categories, primary_layer, discrepancy,
    )

    elapsed = time.time() - t0
    log(f"\nStage 1 v3 complete in {elapsed:.1f}s")
    log(f"  Output: {output_dir}")

    if not all_pass:
        log("\nWARNING: Self-validation had failures. Inspect output.")
        sys.exit(1)


if __name__ == "__main__":
    main()
