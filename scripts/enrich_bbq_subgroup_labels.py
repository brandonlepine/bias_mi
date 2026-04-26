"""BBQ Subgroup-Label Enrichment.

Reads raw BBQ JSONL files and adds explicit, normalized subgroup labels to each
item without modifying original fields. Produces enriched JSONL files consumed
by the geometry pipeline.

Usage:
    python scripts/enrich_bbq_subgroup_labels.py --bbq_dir datasets/bbq/data/
    python scripts/enrich_bbq_subgroup_labels.py --run_dir runs/llama-3.1-8b_2026-04-22/
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
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

NATIONALITY_COUNTRY_TO_REGION: dict[str, str] = {
    "afghan": "middleeast",
    "american": "northamerica",
    "british": "europe",
    "burmese": "asiapacific",
    "chinese": "asiapacific",
    "eritrean": "africa",
    "ethiopian": "africa",
    "guinean": "africa",
    "indian": "asiapacific",
    "indonesian": "asiapacific",
    "iranian": "middleeast",
    "iraqi": "middleeast",
    "irish": "europe",
    "italian": "europe",
    "japanese": "asiapacific",
    "kenyan": "africa",
    "korean": "asiapacific",
    "libyan": "middleeast",
    "malian": "africa",
    "moroccan": "middleeast",
    "mozambican": "africa",
    "namibian": "africa",
    "nigerian": "africa",
    "pakistani": "asiapacific",
    "palestinian": "middleeast",
    "saudi": "middleeast",
    "srilankan": "asiapacific",
    "syrian": "middleeast",
    "thai": "asiapacific",
    "vietnamese": "asiapacific",
    "yemeni": "middleeast",
}

DISABILITY_SUBCATEGORY_MAP: dict[str, str] = {
    "Physical": "physicallydisabled",
    "MentalIllness": "mentallyill",
    "Neurodiversity": "autistic",
    "Cognitive": "peoplewithcognitivedisabilities",
    "Blindness/low-vision": "peoplewithblindnessorlowvision",
    "D/deaf": "ddeaf",
}

GI_STEREO_ALIASES: dict[str, str] = {
    "transgenderwomen": "trans_f",
    "transgendermen": "trans_m",
}

MIN_N_DEFAULT = 10


def log(msg: str) -> None:
    print(f"[enrich_bbq] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize(s: str) -> str:
    return s.strip().lower().replace(" ", "").replace("-", "").replace("/", "").replace("'", "")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_role_tags(item: dict) -> list[str]:
    """Extract all 3 role tags from answer_info."""
    ai = item.get("answer_info", {})
    tags = []
    for key in ("ans0", "ans1", "ans2"):
        info = ai.get(key)
        if isinstance(info, list) and len(info) >= 2:
            tags.append(str(info[1]))
        else:
            tags.append("")
    return tags


def get_answer_texts(item: dict) -> list[str]:
    """Extract answer_info[ans_X][0] texts."""
    ai = item.get("answer_info", {})
    texts = []
    for key in ("ans0", "ans1", "ans2"):
        info = ai.get(key)
        if isinstance(info, list) and len(info) >= 1:
            texts.append(str(info[0]))
        else:
            texts.append("")
    return texts


def get_stereotyped_groups(item: dict) -> list[str]:
    """Extract stereotyped_groups from additional_metadata."""
    meta = item.get("additional_metadata", {})
    sg = meta.get("stereotyped_groups", [])
    if isinstance(sg, str):
        try:
            sg = json.loads(sg)
        except (json.JSONDecodeError, ValueError):
            sg = [sg]
    if not isinstance(sg, list):
        sg = []
    return [str(s) for s in sg if s]


def non_unknown_role_tags_normalized(item: dict) -> list[str]:
    """Normalized non-unknown role tags, sorted and deduped."""
    tags = get_role_tags(item)
    return sorted(set(normalize(t) for t in tags if normalize(t) != "unknown" and t.strip()))


def make_provenance(handler: str, race_handling: str = "n/a") -> dict:
    return {
        "version": "v1",
        "category_handler": handler,
        "race_handling": race_handling,
    }


# ---------------------------------------------------------------------------
# Per-category enrichment handlers
# ---------------------------------------------------------------------------

def enrich_age(item: dict) -> dict:
    """Age: binary (old vs nonold). Mention direction is structurally degenerate
    because every item mentions both subgroups."""
    mentioned = non_unknown_role_tags_normalized(item)
    stereo_norm = sorted(set(normalize(s) for s in get_stereotyped_groups(item)))

    item["mentioned_subgroups_fine"] = mentioned
    item["mentioned_subgroups_aggregated"] = mentioned
    item["bias_target_fine"] = stereo_norm
    item["bias_target_aggregated"] = stereo_norm
    item["subcategory_normalized"] = str(item.get("additional_metadata", {}).get("subcategory", "None"))
    item["enrichment_provenance"] = make_provenance("age_standard")
    return item


def enrich_disability(item: dict) -> dict:
    """Disability: subcategory-driven fine subtype. Role tags are binary
    (disabled/nonDisabled) but subcategory gives fine-grained disability type."""
    meta = item.get("additional_metadata", {})
    subcategory = str(meta.get("subcategory", "None"))
    stereo_raw = get_stereotyped_groups(item)
    stereo_norm = sorted(set(normalize(s) for s in stereo_raw))

    # Determine fine subtype from subcategory
    subtype = DISABILITY_SUBCATEGORY_MAP.get(subcategory)

    if subtype is None:
        # Subcategory not in map — try to derive from stereotyped_groups directly.
        # Use any non-"disabled" token as the fine subtype.
        fine_tokens = [normalize(s) for s in stereo_raw if normalize(s) != "disabled"]
        if fine_tokens:
            subtype = fine_tokens[0]

    if subtype is None:
        # Fallback: only "disabled" in stereotyped_groups, no subcategory
        subtype = "disabled"

    item["mentioned_subgroups_fine"] = sorted(set([subtype, "nondisabled"]))
    item["mentioned_subgroups_aggregated"] = ["disabled", "nondisabled"]
    item["bias_target_fine"] = [subtype] if "disabled" in stereo_norm or subtype in stereo_norm else []
    item["bias_target_aggregated"] = ["disabled"] if "disabled" in stereo_norm or any(
        s != "nondisabled" for s in stereo_norm
    ) else []
    item["subcategory_normalized"] = subcategory
    item["enrichment_provenance"] = make_provenance("disability_subcategory_driven")
    return item


def enrich_gi(item: dict) -> dict:
    """Gender identity: alias stereotyped_groups tokens to match role-tag forms.
    'Transgender women' → trans_f, 'transgender men' → trans_m."""
    mentioned = non_unknown_role_tags_normalized(item)
    stereo_raw = get_stereotyped_groups(item)

    # Apply alias map to stereotyped_groups
    bias_fine = []
    for s in stereo_raw:
        n = normalize(s)
        aliased = GI_STEREO_ALIASES.get(n, n)
        # Also normalize underscores for consistency
        bias_fine.append(aliased)
    bias_fine = sorted(set(bias_fine))

    item["mentioned_subgroups_fine"] = mentioned
    item["mentioned_subgroups_aggregated"] = mentioned
    item["bias_target_fine"] = bias_fine
    item["bias_target_aggregated"] = bias_fine
    item["subcategory_normalized"] = str(item.get("additional_metadata", {}).get("subcategory", "None"))
    item["enrichment_provenance"] = make_provenance("gi_alias_map")
    return item


def enrich_nationality(
    item: dict, country_to_region: dict[str, str],
) -> dict:
    """Nationality: extract country names from answer_info[ans_X][0] for fine
    granularity. Role tags (answer_info[ans_X][1]) are regional aggregates."""
    ai = item.get("answer_info", {})
    stereo_raw = get_stereotyped_groups(item)

    # Fine mentions: country names from answer_info[ans_X][0]
    fine_mentions = []
    agg_mentions = []
    for key in ("ans0", "ans1", "ans2"):
        info = ai.get(key)
        if not isinstance(info, list) or len(info) < 2:
            continue
        country_text = str(info[0]).strip()
        region_tag = str(info[1]).strip()
        if normalize(region_tag) == "unknown":
            continue
        fine_mentions.append(normalize(country_text))
        agg_mentions.append(normalize(region_tag))

    # Bias targets
    bias_fine = sorted(set(normalize(s) for s in stereo_raw))
    bias_agg = sorted(set(
        country_to_region.get(normalize(s), normalize(s)) for s in stereo_raw
    ))

    item["mentioned_subgroups_fine"] = sorted(set(fine_mentions))
    item["mentioned_subgroups_aggregated"] = sorted(set(agg_mentions))
    item["bias_target_fine"] = bias_fine
    item["bias_target_aggregated"] = bias_agg
    item["subcategory_normalized"] = str(item.get("additional_metadata", {}).get("subcategory", "None"))
    item["enrichment_provenance"] = make_provenance("nationality_with_country_extraction")
    return item


def enrich_physical_appearance(item: dict) -> dict:
    """Physical appearance: standard enrichment. Stereotyped_groups aligns
    with role tags after normalization."""
    mentioned = non_unknown_role_tags_normalized(item)
    stereo_norm = sorted(set(normalize(s) for s in get_stereotyped_groups(item)))

    item["mentioned_subgroups_fine"] = mentioned
    item["mentioned_subgroups_aggregated"] = mentioned
    item["bias_target_fine"] = stereo_norm
    item["bias_target_aggregated"] = stereo_norm
    item["subcategory_normalized"] = str(item.get("additional_metadata", {}).get("subcategory", "None"))
    item["enrichment_provenance"] = make_provenance("physical_appearance_standard")
    return item


def enrich_race(item: dict, strip_prefix: bool = True) -> dict:
    """Race: strip F/M gender prefix from role tags (default) or preserve
    for intersectional analysis."""
    role_tags = get_role_tags(item)
    stereo_norm = sorted(set(normalize(s) for s in get_stereotyped_groups(item)))

    mentioned = []
    for tag in role_tags:
        if normalize(tag) == "unknown" or not tag.strip():
            continue
        if strip_prefix:
            # Strip F- or M- prefix
            stripped = re.sub(r"^[FM]-", "", tag)
            mentioned.append(normalize(stripped))
        else:
            mentioned.append(normalize(tag))

    mentioned = sorted(set(mentioned))

    handling = "stripped" if strip_prefix else "intersectional"
    item["mentioned_subgroups_fine"] = mentioned
    item["mentioned_subgroups_aggregated"] = mentioned
    item["bias_target_fine"] = stereo_norm
    item["bias_target_aggregated"] = stereo_norm
    item["subcategory_normalized"] = str(item.get("additional_metadata", {}).get("subcategory", "None"))
    item["enrichment_provenance"] = make_provenance("race_fm_handling", race_handling=handling)
    return item


def enrich_religion(item: dict) -> dict:
    """Religion: standard enrichment. Vocabularies match cleanly after
    normalization."""
    mentioned = non_unknown_role_tags_normalized(item)
    stereo_norm = sorted(set(normalize(s) for s in get_stereotyped_groups(item)))

    item["mentioned_subgroups_fine"] = mentioned
    item["mentioned_subgroups_aggregated"] = mentioned
    item["bias_target_fine"] = stereo_norm
    item["bias_target_aggregated"] = stereo_norm
    item["subcategory_normalized"] = str(item.get("additional_metadata", {}).get("subcategory", "None"))
    item["enrichment_provenance"] = make_provenance("religion_standard")
    return item


def enrich_ses(item: dict) -> dict:
    """SES: degenerate category. Every item mentions both highses and lowses.
    Only lowses is ever targeted. Role direction has no Group B for highses."""
    mentioned = non_unknown_role_tags_normalized(item)
    stereo_norm = sorted(set(normalize(s) for s in get_stereotyped_groups(item)))

    item["mentioned_subgroups_fine"] = mentioned
    item["mentioned_subgroups_aggregated"] = mentioned
    item["bias_target_fine"] = stereo_norm
    item["bias_target_aggregated"] = stereo_norm
    item["subcategory_normalized"] = str(item.get("additional_metadata", {}).get("subcategory", "None"))
    item["enrichment_provenance"] = make_provenance("ses_standard")
    return item


def enrich_so(item: dict) -> dict:
    """Sexual orientation: standard enrichment."""
    mentioned = non_unknown_role_tags_normalized(item)
    stereo_norm = sorted(set(normalize(s) for s in get_stereotyped_groups(item)))

    item["mentioned_subgroups_fine"] = mentioned
    item["mentioned_subgroups_aggregated"] = mentioned
    item["bias_target_fine"] = stereo_norm
    item["bias_target_aggregated"] = stereo_norm
    item["subcategory_normalized"] = str(item.get("additional_metadata", {}).get("subcategory", "None"))
    item["enrichment_provenance"] = make_provenance("so_standard")
    return item


# ---------------------------------------------------------------------------
# Post-enrichment audit
# ---------------------------------------------------------------------------

def run_post_enrichment_audit(
    output_dir: Path, min_n: int,
) -> dict:
    """Re-run subgroup counting on enriched files."""
    audit: dict[str, Any] = {"fine": {}, "aggregated": {}}

    for jsonl_path in sorted(output_dir.glob("*.jsonl")):
        short = jsonl_path.stem
        if short == "race_intersectional":
            continue  # audit only the canonical race.jsonl

        items = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))

        ambig = [it for it in items if it.get("context_condition") == "ambig"]

        for granularity in ("fine", "aggregated"):
            mention_col = f"mentioned_subgroups_{granularity}"
            target_col = f"bias_target_{granularity}"

            # Gather all subgroups
            all_subs: set[str] = set()
            for it in ambig:
                all_subs.update(it.get(mention_col, []))
                all_subs.update(it.get(target_col, []))

            counts = []
            for sub in sorted(all_subs):
                mention_a = sum(1 for it in ambig if sub in it.get(mention_col, []))
                mention_b = sum(1 for it in ambig if sub not in it.get(mention_col, []))
                role_a = sum(1 for it in ambig if sub in it.get(target_col, []))
                role_b = sum(1 for it in ambig
                             if sub in it.get(mention_col, [])
                             and sub not in it.get(target_col, []))
                bias_universe = role_a

                counts.append({
                    "category": short,
                    "granularity": granularity,
                    "subgroup": sub,
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

            audit.setdefault(granularity, {})[short] = counts

    return audit


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def run_validations(
    output_dir: Path, audit: dict, country_to_region: dict[str, str],
) -> list[tuple[str, bool, str]]:
    """Run the 13 validation checks."""
    results: list[tuple[str, bool, str]] = []
    rng = random.Random(42)

    # Helper to find a subgroup's counts in audit
    def find_sub(cat: str, sub: str, gran: str = "fine") -> dict | None:
        for c in audit.get(gran, {}).get(cat, []):
            if c["subgroup"] == sub:
                return c
        return None

    # 1. nationality: japanese mention.group_a >= 200
    c = find_sub("nationality", "japanese")
    val = c["mention"]["group_a_count"] if c else 0
    results.append(("nationality japanese mention.group_a >= 200", val >= 200, f"actual={val}"))

    # 2. nationality: japanese role.group_a >= 200 and role.group_b >= 100
    ra = c["role"]["group_a_count"] if c else 0
    rb = c["role"]["group_b_count"] if c else 0
    results.append(("nationality japanese role.group_a>=200 AND role.group_b>=100",
                     ra >= 200 and rb >= 100, f"group_a={ra}, group_b={rb}"))

    # 3. nationality: Japanese-targeted items have asiapacific in bias_target_aggregated
    nat_path = output_dir / "nationality.jsonl"
    if nat_path.exists():
        japanese_items = []
        with open(nat_path) as f:
            for line in f:
                it = json.loads(line.strip())
                if "japanese" in [normalize(s) for s in get_stereotyped_groups(it)]:
                    japanese_items.append(it)
        sample = rng.sample(japanese_items, min(5, len(japanese_items)))
        all_have_asiapacific = all("asiapacific" in it.get("bias_target_aggregated", []) for it in sample)
        results.append(("nationality Japanese→asiapacific in bias_target_aggregated",
                         all_have_asiapacific, f"checked {len(sample)} items"))
    else:
        results.append(("nationality Japanese→asiapacific", False, "nationality.jsonl not found"))

    # 4. disability: mentallyill mention.group_a >= 400
    c = find_sub("disability", "mentallyill")
    val = c["mention"]["group_a_count"] if c else 0
    results.append(("disability mentallyill mention.group_a >= 400", val >= 400,
                     f"actual={val}"))

    # 5. disability: mentallyill role.group_a >= 200 AND role.group_b >= 50
    ra = c["role"]["group_a_count"] if c else 0
    rb = c["role"]["group_b_count"] if c else 0
    results.append(("disability mentallyill role.group_a>=200 AND role.group_b>=50",
                     ra >= 200 and rb >= 50, f"group_a={ra}, group_b={rb}"))

    # 6. disability: every item has mentioned_subgroups_aggregated == ["disabled", "nondisabled"]
    dis_path = output_dir / "disability.jsonl"
    if dis_path.exists():
        all_match = True
        n_checked = 0
        with open(dis_path) as f:
            for line in f:
                it = json.loads(line.strip())
                if it.get("mentioned_subgroups_aggregated") != ["disabled", "nondisabled"]:
                    all_match = False
                    break
                n_checked += 1
        results.append(("disability all items mentioned_agg==[disabled,nondisabled]",
                         all_match, f"checked {n_checked} items"))
    else:
        results.append(("disability aggregated check", False, "disability.jsonl not found"))

    # 7. gi: Transgender women → trans_f in bias_target_fine
    gi_path = output_dir / "gi.jsonl"
    if gi_path.exists():
        tw_items = []
        with open(gi_path) as f:
            for line in f:
                it = json.loads(line.strip())
                if "Transgender women" in get_stereotyped_groups(it):
                    tw_items.append(it)
        sample = rng.sample(tw_items, min(5, len(tw_items)))
        all_have = all("trans_f" in it.get("bias_target_fine", []) for it in sample)
        results.append(("gi 'Transgender women'→trans_f in bias_target_fine",
                         all_have, f"checked {len(sample)} items"))
    else:
        results.append(("gi trans_f check", False, "gi.jsonl not found"))

    # 8. race (stripped): no fblack in mentioned_subgroups_fine
    race_path = output_dir / "race.jsonl"
    if race_path.exists():
        found_fblack = False
        with open(race_path) as f:
            for line in f:
                it = json.loads(line.strip())
                if "fblack" in it.get("mentioned_subgroups_fine", []):
                    found_fblack = True
                    break
        results.append(("race stripped: no fblack in mentioned_subgroups_fine",
                         not found_fblack, f"found_fblack={found_fblack}"))
    else:
        results.append(("race stripped check", False, "race.jsonl not found"))

    # 9. race_intersectional: fblack DOES appear
    ri_path = output_dir / "race_intersectional.jsonl"
    if ri_path.exists():
        found_fblack = False
        with open(ri_path) as f:
            for line in f:
                it = json.loads(line.strip())
                if "fblack" in it.get("mentioned_subgroups_fine", []):
                    found_fblack = True
                    break
        results.append(("race_intersectional: fblack present",
                         found_fblack, f"found_fblack={found_fblack}"))
    else:
        results.append(("race_intersectional check", False, "race_intersectional.jsonl not found"))

    # 10. ses: every item has bias_target_fine==["lowses"] and mentioned==["highses","lowses"]
    ses_path = output_dir / "ses.jsonl"
    if ses_path.exists():
        all_match = True
        n_checked = 0
        with open(ses_path) as f:
            for line in f:
                it = json.loads(line.strip())
                if (it.get("bias_target_fine") != ["lowses"]
                        or it.get("mentioned_subgroups_fine") != ["highses", "lowses"]):
                    all_match = False
                    break
                n_checked += 1
        results.append(("ses all items: bias_target=[lowses], mentioned=[highses,lowses]",
                         all_match, f"checked {n_checked} items"))
    else:
        results.append(("ses check", False, "ses.jsonl not found"))

    # 11. Schema: every enriched item has all 5 new fields, no nulls
    required_fields = [
        "mentioned_subgroups_fine", "mentioned_subgroups_aggregated",
        "bias_target_fine", "bias_target_aggregated", "subcategory_normalized",
    ]
    schema_ok = True
    schema_detail = ""
    for jsonl_path in sorted(output_dir.glob("*.jsonl")):
        with open(jsonl_path) as f:
            for line_no, line in enumerate(f, 1):
                it = json.loads(line.strip())
                for field in required_fields:
                    if field not in it:
                        schema_ok = False
                        schema_detail = f"{jsonl_path.name}:{line_no} missing {field}"
                        break
                    if it[field] is None:
                        schema_ok = False
                        schema_detail = f"{jsonl_path.name}:{line_no} {field} is null"
                        break
                if not schema_ok:
                    break
        if not schema_ok:
            break
    results.append(("schema: all items have 5 new fields, no nulls",
                     schema_ok, schema_detail or "all OK"))

    # 12. Provenance version
    prov_ok = True
    prov_detail = ""
    for jsonl_path in sorted(output_dir.glob("*.jsonl")):
        with open(jsonl_path) as f:
            for line_no, line in enumerate(f, 1):
                it = json.loads(line.strip())
                prov = it.get("enrichment_provenance", {})
                if prov.get("version") != "v1":
                    prov_ok = False
                    prov_detail = f"{jsonl_path.name}:{line_no} version={prov.get('version')}"
                    break
        if not prov_ok:
            break
    results.append(("provenance version == v1", prov_ok, prov_detail or "all OK"))

    # 13. >=80% reduction in subgroups_with_no_bias_response_data
    # Count subgroups with 0 bias-response targeting in enriched audit
    n_no_bias = 0
    n_total_subs = 0
    for cat_counts in audit.get("fine", {}).values():
        for c in cat_counts:
            n_total_subs += 1
            if c["bias_response_universe"]["n_items_targeting_S"] == 0:
                n_no_bias += 1
    # Pre-enrichment audit had 42 subgroups with no bias response
    pre_count = 42
    reduction = 1.0 - (n_no_bias / pre_count) if pre_count > 0 else 0.0
    results.append((
        ">=80% reduction in no-bias-response subgroups",
        reduction >= 0.80,
        f"pre={pre_count}, post={n_no_bias}, reduction={reduction*100:.0f}%",
    ))

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BBQ Subgroup-Label Enrichment")
    p.add_argument("--run_dir", type=str, default=None)
    p.add_argument("--bbq_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--min_n", type=int, default=MIN_N_DEFAULT)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else None

    # Resolve bbq_dir
    if args.bbq_dir:
        bbq_dir = Path(args.bbq_dir)
    elif run_dir:
        config_path = run_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            bbq_dir = Path(cfg.get("bbq_data_dir", "datasets/bbq/data"))
        else:
            bbq_dir = Path("datasets/bbq/data")
    else:
        bbq_dir = Path("datasets/bbq/data")

    if not bbq_dir.is_dir():
        log(f"FATAL: BBQ data dir not found: {bbq_dir}")
        sys.exit(1)

    # Resolve output_dir
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif run_dir:
        output_dir = run_dir / "data" / "enriched"
    else:
        output_dir = Path("enriched")

    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"BBQ dir: {bbq_dir}")
    log(f"Output dir: {output_dir}")

    # Load country-to-region map from audit if available
    country_to_region = dict(NATIONALITY_COUNTRY_TO_REGION)
    if run_dir:
        audit_path = run_dir / "diagnostics" / "bbq_vocabulary_audit.json"
        if audit_path.exists():
            with open(audit_path) as f:
                audit_data = json.load(f)
            learned = audit_data.get("nationality_country_to_region_map", {})
            if learned:
                country_to_region.update(learned)
                log(f"  Loaded {len(learned)} country→region mappings from audit")

    # Log file
    log_path = output_dir / "enrichment_log.txt"
    log_lines: list[str] = []

    def log_and_save(msg: str) -> None:
        log(msg)
        log_lines.append(msg)

    # ── Process each category ───────────────────────────────────────────
    for bbq_stem, short in sorted(CATEGORY_MAP.items()):
        src_path = bbq_dir / f"{bbq_stem}.jsonl"
        if not src_path.is_file():
            log_and_save(f"WARNING: {src_path} not found, skipping {short}")
            continue

        out_path = output_dir / f"{short}.jsonl"
        out_intersectional = output_dir / "race_intersectional.jsonl" if short == "race" else None

        n_items = 0
        fine_subs: Counter = Counter()
        agg_subs: Counter = Counter()
        target_fine_subs: Counter = Counter()
        warnings: list[str] = []

        out_f = open(out_path, "w")
        out_i = open(out_intersectional, "w") if out_intersectional else None

        with open(src_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                n_items += 1

                # Apply category handler
                if short == "age":
                    item = enrich_age(item)
                elif short == "disability":
                    item = enrich_disability(item)
                elif short == "gi":
                    item = enrich_gi(item)
                elif short == "nationality":
                    item = enrich_nationality(item, country_to_region)
                elif short == "physical_appearance":
                    item = enrich_physical_appearance(item)
                elif short == "race":
                    # Write stripped version
                    item_stripped = enrich_race(json.loads(json.dumps(item)), strip_prefix=True)
                    out_f.write(json.dumps(item_stripped, ensure_ascii=False) + "\n")
                    # Write intersectional version
                    item_inter = enrich_race(item, strip_prefix=False)
                    if out_i:
                        out_i.write(json.dumps(item_inter, ensure_ascii=False) + "\n")
                    # Track stats from stripped version
                    for s in item_stripped.get("mentioned_subgroups_fine", []):
                        fine_subs[s] += 1
                    for s in item_stripped.get("mentioned_subgroups_aggregated", []):
                        agg_subs[s] += 1
                    for s in item_stripped.get("bias_target_fine", []):
                        target_fine_subs[s] += 1
                    continue
                elif short == "religion":
                    item = enrich_religion(item)
                elif short == "ses":
                    item = enrich_ses(item)
                elif short == "so":
                    item = enrich_so(item)

                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

                for s in item.get("mentioned_subgroups_fine", []):
                    fine_subs[s] += 1
                for s in item.get("mentioned_subgroups_aggregated", []):
                    agg_subs[s] += 1
                for s in item.get("bias_target_fine", []):
                    target_fine_subs[s] += 1

        out_f.close()
        if out_i:
            out_i.close()

        log_and_save(f"\n{'='*60}")
        log_and_save(f"{short}: {n_items} items enriched")
        log_and_save(f"  Fine subgroups in mentioned: {len(fine_subs)} unique")
        log_and_save(f"    {dict(fine_subs.most_common(15))}")
        log_and_save(f"  Aggregated subgroups in mentioned: {len(agg_subs)} unique")
        log_and_save(f"  Fine subgroups in bias_target: {len(target_fine_subs)} unique")
        log_and_save(f"    {dict(target_fine_subs.most_common(15))}")
        if warnings:
            for w in warnings:
                log_and_save(f"  WARNING: {w}")

    # Write log
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    log(f"\nEnrichment log: {log_path}")

    # ── Post-enrichment audit ───────────────────────────────────────────
    log("\nRunning post-enrichment audit...")
    audit = run_post_enrichment_audit(output_dir, args.min_n)

    audit_path = output_dir / "enrichment_audit.json"
    tmp = audit_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(audit, f, indent=2, sort_keys=True, default=str)
    os.rename(tmp, audit_path)
    log(f"Post-enrichment audit: {audit_path}")

    # Print key audit results
    for gran in ("fine", "aggregated"):
        for cat, counts in sorted(audit.get(gran, {}).items()):
            computable_mention = sum(1 for c in counts if c["mention"]["computable"])
            computable_role = sum(1 for c in counts if c["role"]["computable"])
            computable_bias = sum(1 for c in counts if c["bias_response_universe"]["computable_in_principle"])
            log(f"  {cat} [{gran}]: {len(counts)} subs, "
                f"mention={computable_mention}, role={computable_role}, bias={computable_bias}")

    # ── Validation ──────────────────────────────────────────────────────
    log("\n" + "=" * 72)
    log("VALIDATION CHECKS")
    log("=" * 72)
    checks = run_validations(output_dir, audit, country_to_region)
    n_pass = 0
    n_fail = 0
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        if passed:
            n_pass += 1
        else:
            n_fail += 1
        log(f"  [{status}] {name}: {detail}")

    log(f"\nValidation: {n_pass} PASS, {n_fail} FAIL")

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
