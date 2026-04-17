"""C3: Generalization evaluation on MedQA and MMLU.

Applies C1's saved steering vectors to out-of-distribution benchmarks.
Tests whether bias-specific interventions transfer to real-world tasks and
quantifies the "capability tax" — how much general performance suffers.

Usage:
    python scripts/C3_generalization.py --run_dir runs/llama-3.1-8b_2026-04-15/

    # Limit items
    python scripts/C3_generalization.py --run_dir ... --max_medqa_items 500 --max_mmlu_items 1000

    # Specific vectors
    python scripts/C3_generalization.py --run_dir ... --subgroups so/gay,race/black

    # Only one dataset
    python scripts/C3_generalization.py --run_dir ... --datasets medqa
    python scripts/C3_generalization.py --run_dir ... --datasets mmlu

    # Skip exacerbation
    python scripts/C3_generalization.py --run_dir ... --skip_exacerbation
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.data.medqa_loader import load_medqa_items
from src.data.mmlu_loader import load_mmlu_items
from src.metrics.bias_metrics import is_degenerated
from src.models.wrapper import ModelWrapper
from src.sae.wrapper import SAEWrapper
from src.sae_localization.steering import SAESteerer
from src.utils.config import load_config
from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MEDQA_EXCLUDED_SUBGROUPS = {
    "physical_appearance/obese",
    "physical_appearance/pregnant",
    "physical_appearance/underweight",
    "disability/disabled",
    "disability/physically disabled",
    "disability/cognitively disabled",
}

DEMOGRAPHIC_PATTERNS: dict[str, dict[str, Any]] = {
    # Sexual Orientation
    "gay": {"category": "so", "patterns": [r"\bgay\b", r"\bhomosexual (man|male)\b", r"\bmsm\b", r"\bmen who have sex with men\b"], "fp_risk": "low"},
    "lesbian": {"category": "so", "patterns": [r"\blesbian\b", r"\bhomosexual (woman|female)\b", r"\bwsw\b"], "fp_risk": "low"},
    "bisexual": {"category": "so", "patterns": [r"\bbisexual\b"], "fp_risk": "low"},
    "pansexual": {"category": "so", "patterns": [r"\bpansexual\b"], "fp_risk": "low"},
    # Gender Identity
    "transgender women": {"category": "gi", "patterns": [r"\btransgender wom(?:a|e)n\b", r"\btrans wom(?:a|e)n\b", r"\bmale.to.female\b", r"\bmtf\b"], "fp_risk": "low"},
    "transgender men": {"category": "gi", "patterns": [r"\btransgender m(?:a|e)n\b", r"\btrans m(?:a|e)n\b", r"\bfemale.to.male\b", r"\bftm\b"], "fp_risk": "low"},
    "non-binary": {"category": "gi", "patterns": [r"\bnon.binary\b", r"\bnonbinary\b", r"\bgenderqueer\b"], "fp_risk": "low"},
    # Race
    "black": {"category": "race", "patterns": [r"\bblack\b", r"\bafrican.?american\b"], "fp_risk": "medium"},
    "white": {"category": "race", "patterns": [r"\bwhite (?:m(?:a|e)n|wom(?:a|e)n|person|patient|male|female)\b", r"\bcaucasian\b"], "fp_risk": "medium"},
    "asian": {"category": "race", "patterns": [r"\basian\b"], "fp_risk": "low"},
    "middle eastern": {"category": "race", "patterns": [r"\bmiddle.eastern\b", r"\barab\b"], "fp_risk": "low"},
    "hispanic": {"category": "race", "patterns": [r"\bhispanic\b", r"\blatin(?:o|a|x|e)\b"], "fp_risk": "low"},
    "native american": {"category": "race", "patterns": [r"\bnative american\b", r"\bamerican indian\b"], "fp_risk": "low"},
    # Religion
    "muslim": {"category": "religion", "patterns": [r"\bmuslim\b", r"\bislamic\b"], "fp_risk": "low"},
    "jewish": {"category": "religion", "patterns": [r"\bjewish\b", r"\bashkenazi\b", r"\bsephardic\b"], "fp_risk": "low"},
    "christian": {"category": "religion", "patterns": [r"\bchristian\b", r"\bcatholic\b", r"\bprotestant\b", r"\bevangelical\b"], "fp_risk": "low"},
    "hindu": {"category": "religion", "patterns": [r"\bhindu\b"], "fp_risk": "low"},
    "buddhist": {"category": "religion", "patterns": [r"\bbuddhist\b"], "fp_risk": "low"},
    "atheist": {"category": "religion", "patterns": [r"\batheist\b", r"\bagnostic\b"], "fp_risk": "low"},
    # Age
    "old": {"category": "age", "patterns": [r"\b(?:elderly|aged|senior|geriatric)\b", r"\b\d{2,3}.year.old\b"], "fp_risk": "high"},
    "young": {"category": "age", "patterns": [r"\byoung (?:patient|adult|person|m(?:a|e)n|wom(?:a|e)n|child)\b", r"\badolescent\b", r"\bteenage\b"], "fp_risk": "medium"},
    # SES
    "low ses": {"category": "ses", "patterns": [r"\blow.income\b", r"\bpoor (?:patient|family|neighborhood)\b", r"\bhomeless\b", r"\buninsured\b", r"\bmedicaid\b"], "fp_risk": "medium"},
    "high ses": {"category": "ses", "patterns": [r"\bwealthy\b", r"\bupper.class\b", r"\baffluent\b"], "fp_risk": "medium"},
}

MMLU_SUPERCATEGORIES: dict[str, set[str]] = {
    "STEM": {
        "abstract_algebra", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_physics",
        "computer_security", "conceptual_physics", "electrical_engineering",
        "elementary_mathematics", "high_school_biology", "high_school_chemistry",
        "high_school_computer_science", "high_school_mathematics",
        "high_school_physics", "high_school_statistics", "machine_learning",
        "astronomy",
    },
    "humanities": {
        "formal_logic", "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "international_law", "jurisprudence",
        "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
        "prehistory", "professional_law", "world_religions",
    },
    "social_sciences": {
        "econometrics", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_microeconomics", "high_school_psychology",
        "human_sexuality", "professional_psychology", "public_relations",
        "security_studies", "sociology", "us_foreign_policy",
    },
    "other": {
        "anatomy", "business_ethics", "clinical_knowledge", "college_medicine",
        "global_facts", "human_aging", "management", "marketing",
        "medical_genetics", "miscellaneous", "nutrition",
        "professional_accounting", "professional_medicine", "virology",
    },
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="C3: Generalization evaluation")
    p.add_argument("--run_dir", required=True, type=str)
    p.add_argument("--subgroups", type=str, default=None,
                   help="Comma-separated subgroup filter (cat/sub)")
    p.add_argument("--max_medqa_items", type=int, default=None)
    p.add_argument("--max_mmlu_items", type=int, default=None)
    p.add_argument("--datasets", type=str, default=None,
                   help="Comma-separated: medqa,mmlu (default: both)")
    p.add_argument("--skip_exacerbation", action="store_true")
    p.add_argument("--skip_figures", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Demographic classification (MedQA)
# ---------------------------------------------------------------------------

def classify_medqa_demographics(
    medqa_items: list[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Classify each MedQA item for demographic mentions using regex."""
    compiled: dict[str, dict[str, Any]] = {}
    for sub_label, spec in DEMOGRAPHIC_PATTERNS.items():
        compiled[sub_label] = {
            "category": spec["category"],
            "patterns": [re.compile(p, re.IGNORECASE) for p in spec["patterns"]],
            "fp_risk": spec["fp_risk"],
        }

    for item in medqa_items:
        text = (item.get("prompt", "") or item.get("question", "")).lower()
        matches: list[dict[str, str]] = []
        for sub_label, spec in compiled.items():
            for pat in spec["patterns"]:
                if pat.search(text):
                    matches.append({
                        "subgroup": sub_label,
                        "category": spec["category"],
                        "fp_risk": spec["fp_risk"],
                    })
                    break
        item["demographic_matches"] = matches
        item["mentions_demographic"] = len(matches) > 0
        item["has_low_fp_match"] = any(m["fp_risk"] == "low" for m in matches)

    n_demo = sum(1 for it in medqa_items if it["mentions_demographic"])
    n_low_fp = sum(1 for it in medqa_items if it.get("has_low_fp_match"))
    log(f"  Demographic classification: {n_demo}/{len(medqa_items)} items with any match")
    log(f"  Low-FP-risk matches only: {n_low_fp}/{len(medqa_items)} items")

    sub_counts: Counter = Counter()
    for it in medqa_items:
        for m in it["demographic_matches"]:
            sub_counts[m["subgroup"]] += 1
    log(f"  Per-subgroup counts:")
    for sub, cnt in sub_counts.most_common():
        log(f"    {sub}: {cnt}")

    # Save classification for audit
    classification_out = {
        "pattern_catalog": {
            sub: {"category": spec["category"], "fp_risk": spec["fp_risk"],
                  "patterns": [p.pattern for p in spec["patterns"]]}
            for sub, spec in compiled.items()
        },
        "excluded_subgroups": sorted(MEDQA_EXCLUDED_SUBGROUPS),
        "summary": {
            "total_items": len(medqa_items),
            "items_with_match": n_demo,
            "items_with_low_fp_match": n_low_fp,
            "per_subgroup_counts": dict(sub_counts),
        },
    }
    atomic_save_json(
        classification_out,
        output_dir / "medqa" / "demographic_classification.json",
    )

    return medqa_items


def annotate_mmlu_supercategories(
    mmlu_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach supercategory label to each MMLU item."""
    subject_to_super: dict[str, str] = {}
    for super_name, subjects in MMLU_SUPERCATEGORIES.items():
        for s in subjects:
            subject_to_super[s] = super_name

    for item in mmlu_items:
        subject = item.get("subject", "").strip().lower()
        item["supercategory"] = subject_to_super.get(subject, "uncategorized")

    counts = Counter(it["supercategory"] for it in mmlu_items)
    log(f"  MMLU supercategory counts: {dict(counts)}")
    return mmlu_items


def compute_medqa_condition(
    item: dict[str, Any], vec_cat: str, vec_sub: str,
) -> str:
    """Determine the condition label for a (MedQA item, vector) pair."""
    key = f"{vec_cat}/{vec_sub}"
    if key in MEDQA_EXCLUDED_SUBGROUPS:
        return "excluded"
    matches = item.get("demographic_matches", [])
    if not matches:
        return "no_demographic"
    match_subs = [m["subgroup"] for m in matches]
    match_cats = [m["category"] for m in matches]
    if vec_sub in match_subs:
        return "matched"
    elif vec_cat in match_cats:
        return "within_cat_mismatched"
    else:
        return "cross_cat_mismatched"


# ---------------------------------------------------------------------------
# Viable vector loading
# ---------------------------------------------------------------------------

def load_viable_vectors(
    run_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    subgroup_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Load all viable steering vectors from C1."""
    with open(run_dir / "C_steering" / "steering_manifests.json") as f:
        manifests = json.load(f)

    filter_set = None
    if subgroup_filter:
        filter_set = set(subgroup_filter.split(","))

    vectors: list[dict[str, Any]] = []
    for m in manifests:
        if not m.get("steering_viable"):
            continue
        key = f"{m['category']}/{m['subgroup']}"
        if filter_set and key not in filter_set:
            continue

        vec_path = run_dir / "C_steering" / "vectors" / f"{m['category']}_{m['subgroup']}.npz"
        if not vec_path.exists():
            log(f"  WARNING: vector file missing for {key}")
            continue

        data = np.load(vec_path)
        vec = torch.from_numpy(data["vector"]).to(device=device, dtype=dtype)
        vectors.append({
            "category": m["category"],
            "subgroup": m["subgroup"],
            "vector": vec,
            "injection_layer": int(data["injection_layer"]),
        })

    log(f"Loaded {len(vectors)} viable steering vectors")
    return vectors


# ---------------------------------------------------------------------------
# Baseline computation
# ---------------------------------------------------------------------------

def compute_or_load_baselines(
    items: list[dict[str, Any]],
    wrapper: ModelWrapper,
    sae_cache: dict[int, SAEWrapper],
    letters: tuple[str, ...],
    cache_path: Path,
    dataset_name: str,
) -> pd.DataFrame:
    """Compute baseline forward passes for all items, or load from cache."""
    if cache_path.exists():
        log(f"  Loading {dataset_name} baselines from cache: {cache_path}")
        df = pd.read_parquet(cache_path)
        if len(df) == len(items):
            return df
        log(f"  Cache size mismatch ({len(df)} vs {len(items)} items); recomputing")

    # Use any SAE layer for baseline (no hooks needed)
    first_layer = next(iter(sae_cache))
    steerer = SAESteerer(wrapper, sae_cache[first_layer], first_layer)

    rows: list[dict[str, Any]] = []
    for i, item in enumerate(items):
        prompt = item.get("prompt", "")
        correct = str(item.get("answer", "")).strip().upper()

        baseline = steerer.evaluate_baseline(prompt, letters=letters)

        baseline_degen = is_degenerated(baseline["answer_logits"], options=letters)

        row: dict[str, Any] = {
            "item_idx": i,
            "correct_answer": correct,
            "baseline_top_answer": baseline["model_answer"],
            "baseline_top_logit": float(max(baseline["answer_logits"].values())),
            "baseline_correct_logit": float(
                baseline["answer_logits"].get(correct, float("-inf"))
            ),
            "baseline_correct": int(baseline["model_answer"] == correct),
            "baseline_degenerated": int(baseline_degen),
        }
        for letter in letters:
            row[f"baseline_logit_{letter}"] = float(
                baseline["answer_logits"].get(letter, float("-inf"))
            )
        rows.append(row)

        if (i + 1) % 100 == 0:
            log(f"    {dataset_name} baselines: {i + 1}/{len(items)}")

    df = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False, compression="snappy")

    n_correct = int(df["baseline_correct"].sum())
    log(f"  {dataset_name} baseline accuracy: "
        f"{n_correct / len(df):.4f} ({n_correct}/{len(df)})")

    return df


# ---------------------------------------------------------------------------
# Per-vector evaluation
# ---------------------------------------------------------------------------

def run_dataset_evaluation(
    dataset_name: str,
    items: list[dict[str, Any]],
    baselines_df: pd.DataFrame,
    vec_info: dict[str, Any],
    sae_cache: dict[int, SAEWrapper],
    wrapper: ModelWrapper,
    output_dir: Path,
    letters: tuple[str, ...],
    skip_exacerbation: bool = False,
) -> list[dict[str, Any]]:
    """Evaluate one steering vector on a full dataset."""
    cat = vec_info["category"]
    sub = vec_info["subgroup"]
    vec = vec_info["vector"]
    exac_vec = -vec
    injection_layer = vec_info["injection_layer"]

    ckpt_path = (
        output_dir / dataset_name / "per_vector_checkpoints" / f"{cat}_{sub}.json"
    )
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            cached = json.load(f)
        log(f"  {dataset_name} {cat}/{sub}: LOADED from checkpoint")
        return cached["records"]

    steerer = SAESteerer(wrapper, sae_cache[injection_layer], injection_layer)
    records: list[dict[str, Any]] = []

    directions = [("debiasing", vec)]
    if not skip_exacerbation:
        directions.append(("exacerbation", exac_vec))

    for direction_name, direction_vec in directions:
        for i, item in enumerate(items):
            prompt = item.get("prompt", "")
            correct = str(item.get("answer", "")).strip().upper()

            baseline_row = baselines_df.iloc[i]

            steered = steerer.steer_and_evaluate(
                prompt, direction_vec, letters=letters,
            )
            steered_correct_logit = float(
                steered["answer_logits"].get(correct, float("-inf"))
            )
            baseline_correct_logit = float(baseline_row["baseline_correct_logit"])

            record: dict[str, Any] = {
                "item_idx": i,
                "dataset": dataset_name,
                "steering_vector": f"{cat}/{sub}",
                "steering_category": cat,
                "steering_subgroup": sub,
                "direction": direction_name,
                "correct_answer": correct,
                "baseline_top_answer": baseline_row["baseline_top_answer"],
                "baseline_top_logit": float(baseline_row["baseline_top_logit"]),
                "baseline_correct_logit": baseline_correct_logit,
                "baseline_correct": int(baseline_row["baseline_correct"]),
                "baseline_degenerated": int(baseline_row["baseline_degenerated"]),
                "steered_top_answer": steered["model_answer"],
                "steered_top_logit": float(
                    max(steered["answer_logits"].values())
                ),
                "steered_correct_logit": steered_correct_logit,
                "steered_correct": int(steered["model_answer"] == correct),
                "steered_degenerated": int(steered.get("degenerated", False)),
                "correct_logit_shift": steered_correct_logit - baseline_correct_logit,
                "top_logit_shift": (
                    float(max(steered["answer_logits"].values()))
                    - float(baseline_row["baseline_top_logit"])
                ),
                "flipped": int(
                    steered["model_answer"] != baseline_row["baseline_top_answer"]
                ),
                "correctness_delta": (
                    int(steered["model_answer"] == correct)
                    - int(baseline_row["baseline_correct"])
                ),
            }

            # Per-letter logits
            for letter in letters:
                record[f"baseline_logit_{letter}"] = float(
                    baseline_row[f"baseline_logit_{letter}"]
                )
                record[f"steered_logit_{letter}"] = float(
                    steered["answer_logits"].get(letter, float("-inf"))
                )

            # Dataset-specific fields
            if dataset_name == "medqa":
                record["demographic_matches"] = json.dumps(
                    item.get("demographic_matches", [])
                )
                record["mentions_demographic"] = int(
                    item.get("mentions_demographic", False)
                )
                record["has_low_fp_match"] = int(
                    item.get("has_low_fp_match", False)
                )
                record["condition"] = compute_medqa_condition(item, cat, sub)
            elif dataset_name == "mmlu":
                record["subject"] = item.get("subject", "")
                record["supercategory"] = item.get("supercategory", "")

            records.append(record)

        n_flipped = sum(
            1 for r in records
            if r["direction"] == direction_name and r["flipped"]
        )
        log(f"  {dataset_name} {cat}/{sub} [{direction_name}]: "
            f"{n_flipped}/{len(items)} items flipped")

    atomic_save_json({"records": records}, ckpt_path)
    return records


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_group(df: pd.DataFrame) -> dict[str, Any]:
    """Compute aggregated statistics for a group of per-item records."""
    n = len(df)
    if n == 0:
        return {"n": 0}

    baseline_acc = float(df["baseline_correct"].mean())
    steered_acc = float(df["steered_correct"].mean())
    accuracy_delta = steered_acc - baseline_acc

    logit_shifts = df["correct_logit_shift"].values
    mean_logit_shift = float(np.mean(logit_shifts))
    median_logit_shift = float(np.median(logit_shifts))
    std_logit_shift = float(np.std(logit_shifts))

    flip_rate = float(df["flipped"].mean())
    n_correct_to_wrong = int(
        ((df["baseline_correct"] == 1) & (df["steered_correct"] == 0)).sum()
    )
    n_wrong_to_correct = int(
        ((df["baseline_correct"] == 0) & (df["steered_correct"] == 1)).sum()
    )
    n_degenerated = int(df["steered_degenerated"].sum())

    # Bootstrap CIs
    rng = np.random.default_rng(42)
    boot_acc_deltas: list[float] = []
    boot_mean_shifts: list[float] = []
    for _ in range(1000):
        idx = rng.choice(n, size=n, replace=True)
        sample = df.iloc[idx]
        boot_acc_deltas.append(
            float(sample["steered_correct"].mean() - sample["baseline_correct"].mean())
        )
        boot_mean_shifts.append(float(sample["correct_logit_shift"].mean()))

    return {
        "n": n,
        "baseline_accuracy": round(baseline_acc, 4),
        "steered_accuracy": round(steered_acc, 4),
        "accuracy_delta": round(accuracy_delta, 4),
        "accuracy_delta_ci_95": [
            round(float(np.percentile(boot_acc_deltas, 2.5)), 4),
            round(float(np.percentile(boot_acc_deltas, 97.5)), 4),
        ],
        "mean_correct_logit_shift": round(mean_logit_shift, 4),
        "median_correct_logit_shift": round(median_logit_shift, 4),
        "std_correct_logit_shift": round(std_logit_shift, 4),
        "mean_shift_ci_95": [
            round(float(np.percentile(boot_mean_shifts, 2.5)), 4),
            round(float(np.percentile(boot_mean_shifts, 97.5)), 4),
        ],
        "flip_rate": round(flip_rate, 4),
        "n_correct_to_wrong": n_correct_to_wrong,
        "n_wrong_to_correct": n_wrong_to_correct,
        "n_degenerated": n_degenerated,
    }


def aggregate_medqa(
    per_item_df: pd.DataFrame, output_dir: Path,
) -> dict[str, Any]:
    """Per-vector, per-condition MedQA aggregation."""
    results: dict[str, Any] = {}

    for (vec, direction), grp in per_item_df.groupby(
        ["steering_vector", "direction"]
    ):
        vec_result: dict[str, Any] = {
            "steering_vector": vec,
            "direction": direction,
            "n_items_total": len(grp),
            "overall": aggregate_group(grp),
            "per_condition": {},
        }

        for cond, cond_grp in grp.groupby("condition"):
            if len(cond_grp) < 10:
                vec_result["per_condition"][cond] = {
                    "n": len(cond_grp), "status": "n_insufficient",
                }
                continue
            vec_result["per_condition"][cond] = aggregate_group(cond_grp)

        # Low-FP matched subset
        matched_low_fp = grp[
            (grp["condition"] == "matched") & (grp["has_low_fp_match"] == 1)
        ]
        if len(matched_low_fp) >= 10:
            vec_result["matched_low_fp_only"] = aggregate_group(matched_low_fp)

        results[f"{vec}_{direction}"] = vec_result

    atomic_save_json(results, output_dir / "medqa" / "aggregated_results.json")
    return results


def aggregate_mmlu(
    per_item_df: pd.DataFrame, output_dir: Path,
) -> dict[str, Any]:
    """Per-vector, per-subject, per-supercategory MMLU aggregation."""
    results: dict[str, Any] = {}

    for (vec, direction), grp in per_item_df.groupby(
        ["steering_vector", "direction"]
    ):
        vec_result: dict[str, Any] = {
            "steering_vector": vec,
            "direction": direction,
            "n_items_total": len(grp),
            "overall": aggregate_group(grp),
            "per_supercategory": {},
            "per_subject": {},
            "worst_subject": None,
            "worst_subject_delta": None,
        }

        for supercat, sc_grp in grp.groupby("supercategory"):
            vec_result["per_supercategory"][supercat] = aggregate_group(sc_grp)

        per_subject: dict[str, Any] = {}
        for subj, sj_grp in grp.groupby("subject"):
            if len(sj_grp) < 10:
                per_subject[subj] = {"n": len(sj_grp), "status": "n_insufficient"}
                continue
            per_subject[subj] = aggregate_group(sj_grp)
        vec_result["per_subject"] = per_subject

        eligible = {
            s: v for s, v in per_subject.items()
            if isinstance(v, dict) and v.get("n", 0) >= 10
            and "accuracy_delta" in v
        }
        if eligible:
            worst = min(eligible, key=lambda s: eligible[s]["accuracy_delta"])
            vec_result["worst_subject"] = worst
            vec_result["worst_subject_delta"] = eligible[worst]["accuracy_delta"]

        results[f"{vec}_{direction}"] = vec_result

    atomic_save_json(results, output_dir / "mmlu" / "aggregated_results.json")
    return results


# ---------------------------------------------------------------------------
# Manifest update
# ---------------------------------------------------------------------------

def update_manifests(run_dir: Path, output_dir: Path) -> None:
    """Add generalization fields to each viable subgroup manifest."""
    with open(run_dir / "C_steering" / "steering_manifests.json") as f:
        manifests = json.load(f)

    medqa_path = output_dir / "medqa" / "aggregated_results.json"
    mmlu_path = output_dir / "mmlu" / "aggregated_results.json"

    medqa_results = {}
    if medqa_path.exists():
        with open(medqa_path) as f:
            medqa_results = json.load(f)

    mmlu_results = {}
    if mmlu_path.exists():
        with open(mmlu_path) as f:
            mmlu_results = json.load(f)

    for m in manifests:
        if not m.get("steering_viable"):
            continue
        vec_key = f"{m['category']}/{m['subgroup']}"
        medqa_debias = medqa_results.get(f"{vec_key}_debiasing", {})
        medqa_exac = medqa_results.get(f"{vec_key}_exacerbation", {})
        mmlu_debias = mmlu_results.get(f"{vec_key}_debiasing", {})

        m["medqa_overall_debias_delta"] = (
            medqa_debias.get("overall", {}).get("accuracy_delta")
        )
        m["medqa_overall_exac_delta"] = (
            medqa_exac.get("overall", {}).get("accuracy_delta")
        )
        m["medqa_overall_debias_logit_shift"] = (
            medqa_debias.get("overall", {}).get("mean_correct_logit_shift")
        )
        m["medqa_overall_exac_logit_shift"] = (
            medqa_exac.get("overall", {}).get("mean_correct_logit_shift")
        )

        for cond in ["matched", "within_cat_mismatched",
                      "cross_cat_mismatched", "no_demographic"]:
            cond_debias = medqa_debias.get("per_condition", {}).get(cond, {})
            cond_exac = medqa_exac.get("per_condition", {}).get(cond, {})
            m[f"medqa_{cond}_debias_delta"] = cond_debias.get("accuracy_delta")
            m[f"medqa_{cond}_exac_delta"] = cond_exac.get("accuracy_delta")
            m[f"medqa_{cond}_debias_logit_shift"] = cond_debias.get(
                "mean_correct_logit_shift",
            )
            m[f"medqa_{cond}_exac_logit_shift"] = cond_exac.get(
                "mean_correct_logit_shift",
            )
            m[f"medqa_{cond}_n"] = cond_debias.get("n")

        mmlu_exac = mmlu_results.get(f"{vec_key}_exacerbation", {})
        m["mmlu_overall_debias_delta"] = (
            mmlu_debias.get("overall", {}).get("accuracy_delta")
        )
        m["mmlu_overall_exac_delta"] = (
            mmlu_exac.get("overall", {}).get("accuracy_delta")
        )
        m["mmlu_worst_subject_debias"] = mmlu_debias.get("worst_subject")
        m["mmlu_worst_subject_debias_delta"] = mmlu_debias.get("worst_subject_delta")

        for supercat in ["STEM", "humanities", "social_sciences", "other"]:
            sc_debias = mmlu_debias.get("per_supercategory", {}).get(supercat, {})
            sc_exac = mmlu_exac.get("per_supercategory", {}).get(supercat, {})
            m[f"mmlu_{supercat}_debias_delta"] = sc_debias.get("accuracy_delta")
            m[f"mmlu_{supercat}_exac_delta"] = sc_exac.get("accuracy_delta")

    atomic_save_json(manifests, output_dir / "manifests_with_generalization.json")


# ---------------------------------------------------------------------------
# Top-impact items
# ---------------------------------------------------------------------------

def identify_top_impact_items(output_dir: Path) -> None:
    """Rank (item, vector) pairs by |correct_logit_shift|."""
    for dataset in ["medqa", "mmlu"]:
        parquet_path = output_dir / dataset / "per_item.parquet"
        if not parquet_path.exists():
            continue
        df = pd.read_parquet(parquet_path)
        df["abs_logit_shift"] = df["correct_logit_shift"].abs()
        top100 = df.nlargest(100, "abs_logit_shift").reset_index(drop=True)
        out_path = output_dir / dataset / "top_impact_items.parquet"
        top100.to_parquet(out_path, index=False, compression="snappy")
        log(f"  Saved top-100 impact items for {dataset}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    t0 = time.time()

    run_dir = Path(args.run_dir)
    config = load_config(run_dir)

    device = torch.device(config["device"])
    dtype = getattr(torch, config["dtype"])

    log("C3 Generalization Evaluation")
    log(f"  run_dir: {run_dir}")
    log(f"  device: {device}")

    # Load model
    log("\nLoading model...")
    wrapper = ModelWrapper.from_pretrained(config["model_path"], device=str(device))

    # Load viable vectors
    viable_vectors = load_viable_vectors(
        run_dir, device, dtype, args.subgroups,
    )

    # Load SAEs for needed layers
    needed_layers = set(v["injection_layer"] for v in viable_vectors)
    sae_cache: dict[int, SAEWrapper] = {}
    for layer in sorted(needed_layers):
        log(f"  Loading SAE for layer {layer}...")
        sae_cache[layer] = SAEWrapper(
            config["sae_source"],
            layer=layer,
            expansion=config.get("sae_expansion", 32),
            device=str(device),
        )

    output_dir = run_dir / "C_generalization"
    ensure_dir(output_dir / "baselines")
    ensure_dir(output_dir / "medqa" / "per_vector_checkpoints")
    ensure_dir(output_dir / "mmlu" / "per_vector_checkpoints")
    ensure_dir(output_dir / "figures")

    datasets_to_run = (
        args.datasets.split(",") if args.datasets else ["medqa", "mmlu"]
    )

    # Load datasets
    medqa_items: list[dict[str, Any]] | None = None
    mmlu_items: list[dict[str, Any]] | None = None

    if "medqa" in datasets_to_run:
        try:
            medqa_path = config.get("medqa_path", "datasets/medqa")
            medqa_items = load_medqa_items(medqa_path)
            if args.max_medqa_items:
                medqa_items = medqa_items[:args.max_medqa_items]
            log(f"Loaded {len(medqa_items)} MedQA items")
            medqa_items = classify_medqa_demographics(medqa_items, output_dir)
        except Exception as e:
            log(f"WARNING: MedQA loading failed: {e}")
            medqa_items = None

    if "mmlu" in datasets_to_run:
        try:
            mmlu_path = config.get("mmlu_path", "datasets/mmlu")
            mmlu_items = load_mmlu_items(mmlu_path)
            if args.max_mmlu_items:
                mmlu_items = mmlu_items[:args.max_mmlu_items]
            log(f"Loaded {len(mmlu_items)} MMLU items")
            mmlu_items = annotate_mmlu_supercategories(mmlu_items)
        except Exception as e:
            log(f"WARNING: MMLU loading failed: {e}")
            mmlu_items = None

    # Determine letters per dataset
    medqa_letters = ("A", "B", "C", "D", "E")
    mmlu_letters = ("A", "B", "C", "D")

    # Step 1: Compute and cache baselines
    medqa_baselines_df: pd.DataFrame | None = None
    mmlu_baselines_df: pd.DataFrame | None = None

    if medqa_items:
        log("\nStep 1a: Computing MedQA baselines...")
        medqa_baselines_df = compute_or_load_baselines(
            medqa_items, wrapper, sae_cache, medqa_letters,
            output_dir / "baselines" / "medqa_baselines.parquet",
            "medqa",
        )

    if mmlu_items:
        log("\nStep 1b: Computing MMLU baselines...")
        mmlu_baselines_df = compute_or_load_baselines(
            mmlu_items, wrapper, sae_cache, mmlu_letters,
            output_dir / "baselines" / "mmlu_baselines.parquet",
            "mmlu",
        )

    # Step 2: Evaluate each steering vector
    medqa_per_item_records: list[dict[str, Any]] = []
    mmlu_per_item_records: list[dict[str, Any]] = []

    log("\nStep 2: Evaluating steering vectors...")
    for vi, vec_info in enumerate(viable_vectors):
        log(f"\nVector {vi + 1}/{len(viable_vectors)}: "
            f"{vec_info['category']}/{vec_info['subgroup']}")

        if medqa_items is not None and medqa_baselines_df is not None:
            records = run_dataset_evaluation(
                "medqa", medqa_items, medqa_baselines_df,
                vec_info, sae_cache, wrapper, output_dir,
                medqa_letters, args.skip_exacerbation,
            )
            medqa_per_item_records.extend(records)

        if mmlu_items is not None and mmlu_baselines_df is not None:
            records = run_dataset_evaluation(
                "mmlu", mmlu_items, mmlu_baselines_df,
                vec_info, sae_cache, wrapper, output_dir,
                mmlu_letters, args.skip_exacerbation,
            )
            mmlu_per_item_records.extend(records)

    # Step 3: Save per-item parquets
    log("\nStep 3: Saving per-item parquets...")
    if medqa_per_item_records:
        medqa_df = pd.DataFrame(medqa_per_item_records)
        medqa_df.to_parquet(
            output_dir / "medqa" / "per_item.parquet",
            index=False, compression="snappy",
        )
        log(f"  MedQA: {len(medqa_df)} rows")

    if mmlu_per_item_records:
        mmlu_df = pd.DataFrame(mmlu_per_item_records)
        mmlu_df.to_parquet(
            output_dir / "mmlu" / "per_item.parquet",
            index=False, compression="snappy",
        )
        log(f"  MMLU: {len(mmlu_df)} rows")

    # Step 4: Aggregate
    log("\nStep 4: Aggregating results...")
    medqa_agg = {}
    mmlu_agg = {}
    if medqa_per_item_records:
        medqa_agg = aggregate_medqa(
            pd.DataFrame(medqa_per_item_records), output_dir,
        )
    if mmlu_per_item_records:
        mmlu_agg = aggregate_mmlu(
            pd.DataFrame(mmlu_per_item_records), output_dir,
        )

    # Step 5: Update manifests
    log("Step 5: Updating manifests...")
    update_manifests(run_dir, output_dir)

    # Step 6: Top-impact items
    log("Step 6: Identifying top-impact items...")
    identify_top_impact_items(output_dir)

    # Figures
    if not args.skip_figures:
        try:
            from src.visualization.generalization_figures import (
                generate_c3_figures,
            )
            generate_c3_figures(output_dir, viable_vectors)
        except ImportError:
            log("WARNING: generalization_figures module not available")
        except Exception as e:
            log(f"WARNING: figure generation failed: {e}")

    runtime = time.time() - t0
    log(f"\nC3 complete in {runtime:.1f}s")
    log(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
