"""BBQ Vocabulary & Subgroup Counting Diagnostic.

Reads raw BBQ JSONL files and audits:
  1. Role-tag vocabulary (answer_info[ans_X][1])
  2. Stereotyped-groups vocabulary (additional_metadata.stereotyped_groups)
  3. Subcategory vocabulary (additional_metadata.subcategory)
  4. Cross-vocabulary alignment after normalization
  5. Per-subgroup item counts for mention/role/bias_response direction types

Writes a single diagnostic JSON and prints a human-readable summary.

Usage:
    python scripts/diagnose_bbq_vocabulary.py --bbq_dir datasets/bbq/data/
    python scripts/diagnose_bbq_vocabulary.py --run_dir runs/llama-3.1-8b_2026-04-22/
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORY_MAP: dict[str, str] = {
    "Age": "age",
    "Disability_status": "disability",
    "Gender_identity": "gi",
    "Nationality": "nationality",
    "Physical_appearance": "physical_appearance",
    "Race_ethnicity": "race",
    "Religion": "religion",
    "SES": "ses",
    "Sexual_orientation": "so",
}

# Expected approximate item counts per category (BBQ paper).
EXPECTED_COUNTS: dict[str, int] = {
    "age": 3680, "disability": 1556, "gi": 5672, "nationality": 3080,
    "physical_appearance": 1576, "race": 6880, "religion": 1200,
    "ses": 6864, "so": 864,
}

MIN_N_DEFAULT = 10


def log(msg: str) -> None:
    print(f"[bbq_diag] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize(s: str) -> str:
    """Normalize a subgroup label for comparison."""
    return s.strip().lower().replace(" ", "").replace("-", "").replace("/", "")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BBQ Vocabulary & Subgroup Counting Diagnostic",
    )
    p.add_argument("--run_dir", type=str, default=None)
    p.add_argument("--bbq_dir", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--min_n", type=int, default=MIN_N_DEFAULT)
    return p.parse_args()


def resolve_bbq_dir(args: argparse.Namespace) -> tuple[Path, Path | None]:
    """Returns (bbq_dir, run_dir_or_None)."""
    run_dir = Path(args.run_dir) if args.run_dir else None

    if args.bbq_dir:
        return Path(args.bbq_dir), run_dir

    if run_dir:
        config_path = run_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            if "bbq_data_dir" in cfg:
                return Path(cfg["bbq_data_dir"]), run_dir

        # Try default locations
        for candidate in [
            Path("datasets/bbq/data"),
            run_dir.parent.parent / "datasets" / "bbq" / "data",
        ]:
            if candidate.is_dir():
                return candidate, run_dir

    log("FATAL: Must provide --bbq_dir or --run_dir with config.json containing bbq_data_dir")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Load BBQ items
# ---------------------------------------------------------------------------

def load_category(bbq_dir: Path, bbq_stem: str) -> list[dict]:
    path = bbq_dir / f"{bbq_stem}.jsonl"
    if not path.is_file():
        log(f"  WARNING: {path} not found, skipping")
        return []
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ---------------------------------------------------------------------------
# Per-item enrichment
# ---------------------------------------------------------------------------

def enrich_item(item: dict, short_name: str, country_to_region: dict[str, str]) -> dict:
    """Compute per-item derived fields for the audit."""
    answer_info = item.get("answer_info", {})

    raw_role_tags = []
    for ans_key in ("ans0", "ans1", "ans2"):
        info = answer_info.get(ans_key)
        if isinstance(info, list) and len(info) >= 2:
            raw_role_tags.append(str(info[1]))
        else:
            raw_role_tags.append("")

    norm_role_tags_excl_unknown = sorted(set(
        normalize(t) for t in raw_role_tags
        if normalize(t) != "unknown" and t.strip()
    ))

    meta = item.get("additional_metadata", {})
    raw_stereo = meta.get("stereotyped_groups", [])
    if isinstance(raw_stereo, str):
        try:
            raw_stereo = json.loads(raw_stereo)
        except (json.JSONDecodeError, ValueError):
            raw_stereo = [raw_stereo]
    if not isinstance(raw_stereo, list):
        raw_stereo = []
    norm_stereo = sorted(set(normalize(s) for s in raw_stereo if s))

    # mentioned_subgroups = non-unknown role tags
    mentioned_fine = norm_role_tags_excl_unknown[:]
    mentioned_agg = mentioned_fine[:]  # same for most categories

    # bias_target_fine = normalized stereotyped_groups
    bias_target_fine = norm_stereo[:]
    bias_target_agg = norm_stereo[:]

    # Category-specific aggregation
    if short_name == "nationality":
        # Aggregate mentioned to regions (role tags are already regional)
        mentioned_agg = mentioned_fine[:]  # already regional
        # Aggregate bias targets: country → region
        bias_target_agg = sorted(set(
            country_to_region.get(ns, ns) for ns in norm_stereo
        ))
    elif short_name == "disability":
        # Aggregate bias targets: all fine labels → "disabled"
        if any(normalize(s) != "nondisabled" for s in raw_stereo if s):
            bias_target_agg = ["disabled"]
        else:
            bias_target_agg = ["nondisabled"]

    return {
        "example_id": item.get("example_id"),
        "question_index": item.get("question_index"),
        "question_polarity": item.get("question_polarity", ""),
        "context_condition": item.get("context_condition", ""),
        "raw_role_tags": raw_role_tags,
        "normalized_role_tags_excluding_unknown": norm_role_tags_excl_unknown,
        "raw_stereotyped_groups": raw_stereo,
        "normalized_stereotyped_groups": norm_stereo,
        "mentioned_subgroups_fine": mentioned_fine,
        "mentioned_subgroups_aggregated": mentioned_agg,
        "bias_target_fine": bias_target_fine,
        "bias_target_aggregated": bias_target_agg,
        "subcategory": str(meta.get("subcategory", "None")),
    }


# ---------------------------------------------------------------------------
# Learn country→region mapping for nationality
# ---------------------------------------------------------------------------

def learn_country_to_region(items: list[dict]) -> dict[str, str]:
    """For nationality items, learn which region each country co-occurs with."""
    # Each item has role tags (regions) and stereotyped_groups (countries).
    # Map each country to the region that appears most often in items targeting it.
    country_region_counts: dict[str, Counter] = defaultdict(Counter)

    for item in items:
        answer_info = item.get("answer_info", {})
        role_tags = set()
        for ans_key in ("ans0", "ans1", "ans2"):
            info = answer_info.get(ans_key)
            if isinstance(info, list) and len(info) >= 2:
                tag = normalize(str(info[1]))
                if tag != "unknown":
                    role_tags.add(tag)

        meta = item.get("additional_metadata", {})
        stereo = meta.get("stereotyped_groups", [])
        if isinstance(stereo, str):
            try:
                stereo = json.loads(stereo)
            except (json.JSONDecodeError, ValueError):
                stereo = [stereo]
        if not isinstance(stereo, list):
            stereo = []

        for country_raw in stereo:
            country_norm = normalize(country_raw)
            for region in role_tags:
                country_region_counts[country_norm][region] += 1

    mapping: dict[str, str] = {}
    for country, region_counts in sorted(country_region_counts.items()):
        if region_counts:
            mapping[country] = region_counts.most_common(1)[0][0]

    return mapping


# ---------------------------------------------------------------------------
# Section 1: Vocabulary inventories
# ---------------------------------------------------------------------------

def build_vocab_inventory(items: list[dict], short_name: str) -> dict:
    """Build vocabulary inventories for a single category."""
    role_tag_counter: Counter = Counter()
    stereo_counter: Counter = Counter()
    subcat_counter: Counter = Counter()
    polarity_counter: Counter = Counter()
    condition_counter: Counter = Counter()
    qi_set: set = set()

    for item in items:
        answer_info = item.get("answer_info", {})
        for ans_key in ("ans0", "ans1", "ans2"):
            info = answer_info.get(ans_key)
            if isinstance(info, list) and len(info) >= 2:
                role_tag_counter[str(info[1])] += 1

        meta = item.get("additional_metadata", {})
        stereo = meta.get("stereotyped_groups", [])
        if isinstance(stereo, str):
            try:
                stereo = json.loads(stereo)
            except (json.JSONDecodeError, ValueError):
                stereo = [stereo]
        if isinstance(stereo, list):
            for s in stereo:
                if s:
                    stereo_counter[str(s)] += 1

        subcat_counter[str(meta.get("subcategory", "None"))] += 1
        polarity_counter[item.get("question_polarity", "?")] += 1
        condition_counter[item.get("context_condition", "?")] += 1
        qi_set.add(item.get("question_index"))

    # Build normalized_to_raw maps
    role_norm_to_raw: dict[str, list[str]] = defaultdict(list)
    for raw in sorted(role_tag_counter.keys()):
        n = normalize(raw)
        if raw not in role_norm_to_raw[n]:
            role_norm_to_raw[n].append(raw)

    stereo_norm_to_raw: dict[str, list[str]] = defaultdict(list)
    for raw in sorted(stereo_counter.keys()):
        n = normalize(raw)
        if raw not in stereo_norm_to_raw[n]:
            stereo_norm_to_raw[n].append(raw)

    return {
        "category": short_name,
        "n_items": len(items),
        "n_unique_question_indices": len(qi_set),
        "role_tag_vocab": {
            "raw": sorted(role_tag_counter.keys()),
            "raw_with_counts": dict(sorted(role_tag_counter.items())),
            "normalized_to_raw": dict(sorted(role_norm_to_raw.items())),
        },
        "stereotyped_groups_vocab": {
            "raw": sorted(stereo_counter.keys()),
            "raw_with_counts": dict(sorted(stereo_counter.items())),
            "normalized_to_raw": dict(sorted(stereo_norm_to_raw.items())),
        },
        "subcategory_vocab": {
            "raw": sorted(subcat_counter.keys()),
            "raw_with_counts": dict(sorted(subcat_counter.items())),
        },
        "polarity_distribution": dict(sorted(polarity_counter.items())),
        "context_condition_distribution": dict(sorted(condition_counter.items())),
    }


# ---------------------------------------------------------------------------
# Section 2: Cross-vocabulary alignment
# ---------------------------------------------------------------------------

def build_alignment(vocab_inv: dict) -> dict:
    """Check how role-tag and stereotyped-groups vocabularies align."""
    role_norm = set(vocab_inv["role_tag_vocab"]["normalized_to_raw"].keys())
    stereo_norm = set(vocab_inv["stereotyped_groups_vocab"]["normalized_to_raw"].keys())

    # Remove "unknown" from role tags for alignment
    role_norm_clean = role_norm - {"unknown"}

    matches: dict[str, str] = {}
    role_unmatched: list[str] = []
    stereo_unmatched: list[str] = []

    for rt in sorted(role_norm_clean):
        if rt in stereo_norm:
            matches[rt] = rt
        else:
            role_unmatched.append(rt)

    for st in sorted(stereo_norm):
        if st not in role_norm_clean:
            stereo_unmatched.append(st)

    exact = (set(matches.keys()) == role_norm_clean == stereo_norm)

    # Classify relationship
    if exact:
        signal = "identical"
    elif stereo_norm.issubset(role_norm_clean) or role_norm_clean.issubset(stereo_norm):
        if len(stereo_norm) > len(role_norm_clean):
            signal = "fine_subset_of_aggregated"
        else:
            signal = "fine_subset_of_aggregated"
    elif len(matches) == 0:
        signal = "disjoint"
    else:
        signal = "complex"

    return {
        "category": vocab_inv["category"],
        "alignment": {
            "role_tags_in_stereotyped_groups": {
                "matches": matches,
                "role_tags_unmatched": role_unmatched,
                "stereo_tokens_unmatched": stereo_unmatched,
            },
            "exact_match": exact,
            "fine_aggregated_signal": signal,
        },
    }


# ---------------------------------------------------------------------------
# Section 3: Subgroup item-count audit
# ---------------------------------------------------------------------------

def build_subgroup_vocab(
    enriched_items: list[dict], short_name: str, vocab_inv: dict,
) -> tuple[list[dict], list[dict]]:
    """Build fine and aggregated subgroup vocabularies and compute item counts."""
    # Fine vocabulary: all normalized role tags (excl unknown) + any normalized
    # stereotyped_groups tokens not already present.
    role_norm = set(vocab_inv["role_tag_vocab"]["normalized_to_raw"].keys()) - {"unknown"}
    stereo_norm = set(vocab_inv["stereotyped_groups_vocab"]["normalized_to_raw"].keys())
    fine_vocab = sorted(role_norm | stereo_norm)

    # Build raw label lookup for traceability
    all_norm_to_raw: dict[str, str] = {}
    for n, raws in vocab_inv["role_tag_vocab"]["normalized_to_raw"].items():
        all_norm_to_raw[n] = raws[0]
    for n, raws in vocab_inv["stereotyped_groups_vocab"]["normalized_to_raw"].items():
        if n not in all_norm_to_raw:
            all_norm_to_raw[n] = raws[0]

    # Aggregated vocabulary: for most categories, same as fine.
    # For nationality: use role tags (regional labels).
    # For disability: use role tags (disabled/nondisabled).
    if short_name == "nationality":
        agg_vocab = sorted(role_norm)
    elif short_name == "disability":
        agg_vocab = sorted(role_norm)
    else:
        agg_vocab = fine_vocab[:]

    # Filter to ambig items only
    ambig = [e for e in enriched_items if e["context_condition"] == "ambig"]

    fine_results = _count_subgroup_items(ambig, fine_vocab, "fine", all_norm_to_raw, short_name)
    agg_results = _count_subgroup_items(ambig, agg_vocab, "aggregated", all_norm_to_raw, short_name)

    return fine_results, agg_results


def _count_subgroup_items(
    ambig_items: list[dict],
    vocab: list[str],
    granularity: str,
    norm_to_raw: dict[str, str],
    short_name: str,
) -> list[dict]:
    """Count mention/role/bias_response groups for each subgroup."""
    min_n = MIN_N_DEFAULT  # will be overridden in main
    results = []

    mention_col = f"mentioned_subgroups_{granularity}"
    target_col = f"bias_target_{granularity}"

    for sub in vocab:
        # Mention: Group A = S in mentioned_subgroups, Group B = S not in mentioned
        mention_a = sum(1 for e in ambig_items if sub in e[mention_col])
        mention_b = sum(1 for e in ambig_items if sub not in e[mention_col])

        # Role: Group A = S is bias target, Group B = S mentioned but not target
        role_a = sum(1 for e in ambig_items if sub in e[target_col])
        role_b = sum(1 for e in ambig_items
                     if sub in e[mention_col] and sub not in e[target_col])

        # Bias response: universe of items targeting S
        bias_universe = sum(1 for e in ambig_items if sub in e[target_col])

        results.append({
            "category": short_name,
            "granularity": granularity,
            "subgroup": sub,
            "raw_subgroup_label": norm_to_raw.get(sub, sub),
            "mention": {
                "group_a_count": mention_a,
                "group_b_count": mention_b,
                "computable": mention_a >= min_n and mention_b >= min_n,
            },
            "role": {
                "group_a_count": role_a,
                "group_b_count": role_b,
                "computable": role_a >= min_n and role_b >= min_n,
            },
            "bias_response_universe": {
                "n_items_targeting_S": bias_universe,
                "computable_in_principle": bias_universe >= min_n,
            },
        })

    return results


# ---------------------------------------------------------------------------
# Section 4: Summary & bug candidates
# ---------------------------------------------------------------------------

def build_summary(
    alignments: dict[str, dict],
    fine_counts: dict[str, list[dict]],
    agg_counts: dict[str, list[dict]],
) -> dict:
    """Build the top-level diagnostic summary."""
    clean = []
    needs_split = []
    no_role_fine: list[dict] = []
    no_bias_response: list[dict] = []
    aliased: list[dict] = []
    candidate_bugs: list[str] = []

    for short, al in alignments.items():
        signal = al["alignment"]["fine_aggregated_signal"]
        if signal == "identical":
            clean.append(short)
        else:
            needs_split.append(short)

    for short, counts in fine_counts.items():
        for c in counts:
            sub = c["subgroup"]
            # No role data at fine granularity
            if c["role"]["group_a_count"] == 0 and c["role"]["group_b_count"] == 0:
                if c["mention"]["group_a_count"] > 0:
                    no_role_fine.append({
                        "category": short, "subgroup": sub,
                        "reason": f"Fine label '{sub}' has {c['mention']['group_a_count']} "
                                  f"mention items but 0 role items. Label may only appear "
                                  f"in stereotyped_groups, not in role tags.",
                    })

            # No bias response data
            if c["bias_response_universe"]["n_items_targeting_S"] == 0:
                no_bias_response.append({
                    "category": short, "subgroup": sub,
                    "reason": f"'{sub}' never appears in stereotyped_groups — "
                              f"cannot compute bias_response direction.",
                })

    # Check for aliased directions in nationality
    if "nationality" in fine_counts:
        # Group countries by their bias_target sets
        target_to_subs: dict[str, list[str]] = defaultdict(list)
        for c in fine_counts["nationality"]:
            if c["role"]["group_a_count"] > 0:
                key = str(c["role"]["group_a_count"])
                target_to_subs[key].append(c["subgroup"])
        for key, subs in target_to_subs.items():
            if len(subs) > 2:
                aliased.append({
                    "category": "nationality",
                    "subgroups": subs,
                    "reason": f"Co-targeted countries with identical role group_a count ({key}); "
                              f"fine directions will be mathematically identical if they share "
                              f"the same items.",
                })

    # Candidate bugs
    if "ses" in fine_counts:
        ses_subs = {c["subgroup"] for c in fine_counts["ses"]}
        if "highses" not in ses_subs and "lowses" in ses_subs:
            candidate_bugs.append(
                "ses: only lowses subgroup detected (highSES role tag may be "
                "missing from vocabulary)"
            )
        ses_counts = {c["subgroup"]: c for c in fine_counts["ses"]}
        if "highses" in ses_counts:
            hc = ses_counts["highses"]
            if hc["role"]["group_a_count"] == 0:
                candidate_bugs.append(
                    "ses: highses has 0 role-target items (highSES never in "
                    "stereotyped_groups — only 'low SES' is targeted)"
                )

    if "disability" in fine_counts:
        dis_fine = fine_counts["disability"]
        n_no_role = sum(1 for c in dis_fine
                        if c["role"]["group_a_count"] == 0
                        and c["mention"]["group_a_count"] > 0)
        if n_no_role > 0:
            candidate_bugs.append(
                f"disability fine: {n_no_role} fine subgroups have mention data "
                f"but 0 role-target items (fine labels only in stereotyped_groups)"
            )

    if "physical_appearance" in fine_counts:
        pa_subs = {c["subgroup"]: c for c in fine_counts["physical_appearance"]}
        for contrast in ["nonobese", "notpregnant", "novisible difference", "posdress", "tall"]:
            norm_c = normalize(contrast)
            if norm_c in pa_subs and pa_subs[norm_c]["role"]["group_a_count"] == 0:
                candidate_bugs.append(
                    f"physical_appearance: '{contrast}' contrast tag has 0 "
                    f"role-target items (expected — it's the non-stereotyped side)"
                )

    return {
        "categories_with_clean_alignment": sorted(clean),
        "categories_needing_fine_aggregated_split": sorted(needs_split),
        "subgroups_with_no_role_data_at_fine": no_role_fine,
        "subgroups_with_no_bias_response_data": no_bias_response,
        "subgroups_with_aliased_directions": aliased,
        "candidate_bugs_in_current_pipeline": candidate_bugs,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def run_validations(
    vocab_invs: dict[str, dict],
    alignments: dict[str, dict],
) -> list[tuple[str, bool, str]]:
    """Run validation checks. Returns list of (check_name, passed, detail)."""
    results: list[tuple[str, bool, str]] = []

    # 1. SES role tags
    ses_role = set(vocab_invs.get("ses", {}).get("role_tag_vocab", {}).get("raw", []))
    results.append((
        "SES role tags include lowSES and highSES",
        "lowSES" in ses_role and "highSES" in ses_role,
        f"actual: {sorted(ses_role)}",
    ))

    # 2. SES stereotyped groups
    ses_stereo = set(vocab_invs.get("ses", {}).get("stereotyped_groups_vocab", {}).get("raw", []))
    results.append((
        "SES stereotyped_groups includes 'low SES'",
        "low SES" in ses_stereo,
        f"actual: {sorted(ses_stereo)}",
    ))

    # 3. SES alignment
    ses_al = alignments.get("ses", {}).get("alignment", {})
    ses_matches = ses_al.get("role_tags_in_stereotyped_groups", {}).get("matches", {})
    ses_unmatched = ses_al.get("role_tags_in_stereotyped_groups", {}).get("role_tags_unmatched", [])
    results.append((
        "SES alignment: lowses matches, highses unmatched",
        "lowses" in ses_matches and "highses" in ses_unmatched,
        f"matches={ses_matches}, unmatched={ses_unmatched}",
    ))

    # 4. Disability role tags are binary
    dis_role = set(vocab_invs.get("disability", {}).get("role_tag_vocab", {}).get("raw", []))
    dis_role_clean = dis_role - {"unknown"}
    results.append((
        "Disability role tags are {disabled, nonDisabled}",
        dis_role_clean == {"disabled", "nonDisabled"},
        f"actual: {sorted(dis_role_clean)}",
    ))

    # 5. Disability stereotyped_groups has fine labels
    dis_stereo = set(vocab_invs.get("disability", {}).get(
        "stereotyped_groups_vocab", {}).get("raw", []))
    expected_dis = {"disabled", "physically disabled", "mentally-ill",
                    "autistic people", "D/deaf", "people with blindness or low-vision"}
    results.append((
        "Disability stereotyped_groups has fine labels",
        expected_dis.issubset(dis_stereo),
        f"missing: {expected_dis - dis_stereo}" if not expected_dis.issubset(dis_stereo)
        else f"all present, total: {len(dis_stereo)}",
    ))

    # 6. Nationality role tags are regional
    nat_role = set(vocab_invs.get("nationality", {}).get("role_tag_vocab", {}).get("raw", []))
    expected_nat_regions = {"Africa", "ArabStates", "AsiaPacific", "Europe",
                            "LatinSouthAmerica", "MiddleEast", "NorthAmerica"}
    results.append((
        "Nationality role tags include regional labels",
        expected_nat_regions.issubset(nat_role),
        f"missing: {expected_nat_regions - nat_role}" if not expected_nat_regions.issubset(nat_role)
        else "all present",
    ))

    # 7. Nationality stereotyped_groups has country names, NOT regional
    nat_stereo = set(vocab_invs.get("nationality", {}).get(
        "stereotyped_groups_vocab", {}).get("raw", []))
    results.append((
        "Nationality stereotyped_groups has country names not regions",
        not expected_nat_regions.intersection(nat_stereo),
        f"regions found in stereo: {expected_nat_regions.intersection(nat_stereo)}"
        if expected_nat_regions.intersection(nat_stereo)
        else f"correct, {len(nat_stereo)} country names",
    ))

    # 8. SO and Religion alignment is "identical"
    for cat in ("so", "religion"):
        al = alignments.get(cat, {}).get("alignment", {})
        sig = al.get("fine_aggregated_signal", "?")
        results.append((
            f"{cat.upper()} alignment is 'identical'",
            sig == "identical",
            f"actual: {sig}",
        ))

    # 9. Item counts within 5% of expected
    for short, expected in EXPECTED_COUNTS.items():
        actual = vocab_invs.get(short, {}).get("n_items", 0)
        pct_off = abs(actual - expected) / expected if expected > 0 else 1.0
        results.append((
            f"{short} item count ≈ {expected}",
            pct_off <= 0.05,
            f"actual={actual}, off by {pct_off*100:.1f}%",
        ))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global MIN_N_DEFAULT
    args = parse_args()
    MIN_N_DEFAULT = args.min_n

    bbq_dir, run_dir = resolve_bbq_dir(args)
    log(f"BBQ data dir: {bbq_dir}")

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    elif run_dir:
        diag_dir = run_dir / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        output_path = diag_dir / "bbq_vocabulary_audit.json"
    else:
        output_path = Path("bbq_vocabulary_audit.json")

    # ── Load all categories ─────────────────────────────────────────────
    all_items: dict[str, list[dict]] = {}
    for bbq_stem, short in CATEGORY_MAP.items():
        items = load_category(bbq_dir, bbq_stem)
        all_items[short] = items
        log(f"  {short}: {len(items)} items")

    # ── Learn nationality country→region mapping ───────────────────────
    country_to_region: dict[str, str] = {}
    if all_items.get("nationality"):
        country_to_region = learn_country_to_region(all_items["nationality"])
        log(f"\nNationality country→region mapping ({len(country_to_region)} countries):")
        for country, region in sorted(country_to_region.items()):
            log(f"  {country} → {region}")

    # ── Section 1: Vocabulary inventories ───────────────────────────────
    log("\n" + "=" * 72)
    log("SECTION 1: VOCABULARY INVENTORIES")
    log("=" * 72)
    vocab_invs: dict[str, dict] = {}
    for short, items in all_items.items():
        inv = build_vocab_inventory(items, short)
        vocab_invs[short] = inv
        log(f"\n--- {short} ({inv['n_items']} items, "
            f"{inv['n_unique_question_indices']} question indices) ---")
        log(f"  Role tags: {inv['role_tag_vocab']['raw']}")
        log(f"  Stereotyped groups: {inv['stereotyped_groups_vocab']['raw']}")
        log(f"  Subcategories: {inv['subcategory_vocab']['raw']}")
        log(f"  Polarity: {inv['polarity_distribution']}")

    # ── Section 2: Cross-vocabulary alignment ───────────────────────────
    log("\n" + "=" * 72)
    log("SECTION 2: CROSS-VOCABULARY ALIGNMENT")
    log("=" * 72)
    alignments: dict[str, dict] = {}
    for short, inv in vocab_invs.items():
        al = build_alignment(inv)
        alignments[short] = al
        sig = al["alignment"]["fine_aggregated_signal"]
        matches = al["alignment"]["role_tags_in_stereotyped_groups"]["matches"]
        role_un = al["alignment"]["role_tags_in_stereotyped_groups"]["role_tags_unmatched"]
        stereo_un = al["alignment"]["role_tags_in_stereotyped_groups"]["stereo_tokens_unmatched"]
        log(f"\n  {short}: {sig}")
        if matches:
            log(f"    Matches: {matches}")
        if role_un:
            log(f"    Role-only (no stereo equiv): {role_un}")
        if stereo_un:
            log(f"    Stereo-only (no role equiv): {stereo_un}")

    # ── Enrich items ────────────────────────────────────────────────────
    enriched: dict[str, list[dict]] = {}
    for short, items in all_items.items():
        enriched[short] = [enrich_item(it, short, country_to_region) for it in items]

    # ── Section 3: Subgroup item-count audit ────────────────────────────
    log("\n" + "=" * 72)
    log("SECTION 3: SUBGROUP ITEM-COUNT AUDIT (ambig items only)")
    log("=" * 72)
    fine_counts: dict[str, list[dict]] = {}
    agg_counts: dict[str, list[dict]] = {}

    for short in sorted(all_items.keys()):
        fine, agg = build_subgroup_vocab(enriched[short], short, vocab_invs[short])
        fine_counts[short] = fine
        agg_counts[short] = agg

        log(f"\n--- {short} FINE ---")
        for c in fine:
            m = c["mention"]
            r = c["role"]
            b = c["bias_response_universe"]
            flags = []
            if not m["computable"]:
                flags.append("mention:SKIP")
            if not r["computable"]:
                flags.append("role:SKIP")
            if not b["computable_in_principle"]:
                flags.append("bias:SKIP")
            flag_str = f"  [{', '.join(flags)}]" if flags else ""
            log(f"  {c['subgroup']:30s} mention=({m['group_a_count']:4d},{m['group_b_count']:4d}) "
                f"role=({r['group_a_count']:4d},{r['group_b_count']:4d}) "
                f"bias_target={b['n_items_targeting_S']:4d}{flag_str}")

        if agg != fine:
            log(f"\n--- {short} AGGREGATED ---")
            for c in agg:
                m = c["mention"]
                r = c["role"]
                b = c["bias_response_universe"]
                flags = []
                if not m["computable"]:
                    flags.append("mention:SKIP")
                if not r["computable"]:
                    flags.append("role:SKIP")
                if not b["computable_in_principle"]:
                    flags.append("bias:SKIP")
                flag_str = f"  [{', '.join(flags)}]" if flags else ""
                log(f"  {c['subgroup']:30s} mention=({m['group_a_count']:4d},{m['group_b_count']:4d}) "
                    f"role=({r['group_a_count']:4d},{r['group_b_count']:4d}) "
                    f"bias_target={b['n_items_targeting_S']:4d}{flag_str}")

    # ── Sample items ────────────────────────────────────────────────────
    rng = random.Random(42)
    samples: dict[str, list[dict]] = {}
    for short, eitems in enriched.items():
        pool = [e for e in eitems if e["context_condition"] == "ambig"]
        k = min(3, len(pool))
        samples[short] = rng.sample(pool, k) if pool else []

    # ── Section 4: Summary ──────────────────────────────────────────────
    log("\n" + "=" * 72)
    log("SECTION 4: DIAGNOSTIC SUMMARY")
    log("=" * 72)
    summary = build_summary(alignments, fine_counts, agg_counts)

    log(f"\n  Clean alignment: {summary['categories_with_clean_alignment']}")
    log(f"  Need fine/agg split: {summary['categories_needing_fine_aggregated_split']}")
    log(f"  Subgroups with no role data (fine): {len(summary['subgroups_with_no_role_data_at_fine'])}")
    for s in summary["subgroups_with_no_role_data_at_fine"]:
        log(f"    {s['category']}/{s['subgroup']}: {s['reason']}")
    log(f"  Subgroups with no bias_response: {len(summary['subgroups_with_no_bias_response_data'])}")
    for s in summary["subgroups_with_no_bias_response_data"]:
        log(f"    {s['category']}/{s['subgroup']}")
    log(f"  Candidate bugs: {len(summary['candidate_bugs_in_current_pipeline'])}")
    for b in summary["candidate_bugs_in_current_pipeline"]:
        log(f"    - {b}")

    # ── Validation checks ───────────────────────────────────────────────
    log("\n" + "=" * 72)
    log("VALIDATION CHECKS")
    log("=" * 72)
    checks = run_validations(vocab_invs, alignments)
    n_pass = 0
    n_fail = 0
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        if passed:
            n_pass += 1
        else:
            n_fail += 1
        log(f"  [{status}] {name}: {detail}")

    # ── Write output JSON ───────────────────────────────────────────────
    output = {
        "section_1_vocabulary_inventories": vocab_invs,
        "section_2_alignment": alignments,
        "section_3_subgroup_counts": {
            "fine": fine_counts,
            "aggregated": agg_counts,
        },
        "section_3_sample_items": samples,
        "section_4_summary": summary,
        "nationality_country_to_region_map": country_to_region,
        "validation": {
            "n_pass": n_pass,
            "n_fail": n_fail,
            "checks": [
                {"name": name, "passed": passed, "detail": detail}
                for name, passed, detail in checks
            ],
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(output, f, indent=2, sort_keys=True, default=str)
    os.rename(tmp, output_path)

    log(f"\nOutput written to: {output_path}")
    log(f"Validation: {n_pass} PASS, {n_fail} FAIL")

    total_items = sum(inv["n_items"] for inv in vocab_invs.values())
    n_clean = len(summary["categories_with_clean_alignment"])
    n_bugs = len(summary["candidate_bugs_in_current_pipeline"])
    log(f"\nSummary: {len(vocab_invs)} categories, {total_items} total items, "
        f"{n_clean} with clean alignment, {n_bugs} candidate bugs identified.")

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
