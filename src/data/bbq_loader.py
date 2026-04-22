"""BBQ JSONL parsing and standardization for all bias categories.

Implements all steps of the A1_prepare_stimuli spec:
  1. File Discovery — map category short names to JSONL file paths
  2. Load Raw Items — parse JSONL into dicts
  3. Classify Answer Roles — determine stereotyped_target / non_stereotyped / unknown
  4. Shuffle Answer Positions — randomize A/B/C ordering with seeded RNG
  5. Normalize Subgroup Labels — collapse synonyms, deduplicate
  6. Determine Derived Fields — stereotyped_option, non_stereotyped_identity, etc.
  7. Build Prompt String — zero-shot MC prompt
  8. Assemble Output Item — final dict per item
  9. Filtering — drop malformed items with reason tracking
"""

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from src.utils.logging import log, progress_bar


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Mapping from our short names to BBQ JSONL filenames (without extension).
# The 9 non-intersectional categories only.
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

ALL_CATEGORIES: list[str] = list(CATEGORY_FILE_MAP.keys())

# Role tags that indicate the "unknown" / "can't be determined" answer.
UNKNOWN_TAGS: set[str] = {"unknown"}

# Subgroup label normalization map.
# Keys are lowercased, stripped labels; values are the canonical form.
# Anything not in this map is kept as-is (lowercased, stripped).
SUBGROUP_NORMALIZATION: dict[str, str] = {
    # Race collapses
    "black": "black",
    "african american": "black",
    "african-american": "black",
    "arab": "middle eastern",
    "middle eastern": "middle eastern",
}


# ---------------------------------------------------------------------------
# Step 1: File Discovery
# ---------------------------------------------------------------------------

def find_bbq_files(
    bbq_data_dir: Path,
    categories: list[str],
) -> dict[str, Path]:
    """Map category short names to JSONL file paths.

    Args:
        bbq_data_dir: Directory containing BBQ JSONL files.
        categories: List of short category names to look up.

    Returns:
        Dict mapping each found category short name to its JSONL path.
        Categories that are unknown or whose file is missing are skipped
        with a warning.
    """
    found: dict[str, Path] = {}
    for cat in categories:
        bbq_name = CATEGORY_FILE_MAP.get(cat)
        if bbq_name is None:
            log(f"WARNING: unknown category '{cat}', skipping")
            continue
        path = bbq_data_dir / f"{bbq_name}.jsonl"
        if path.exists():
            found[cat] = path
        else:
            log(f"WARNING: file not found for '{cat}': {path}")
    return found


# ---------------------------------------------------------------------------
# Step 2: Load Raw Items
# ---------------------------------------------------------------------------

def load_raw_items(jsonl_path: Path) -> list[dict[str, Any]]:
    """Load all items from a BBQ JSONL file.

    Every parseable non-empty line is loaded; no filtering at this stage.
    """
    items: list[dict[str, Any]] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ---------------------------------------------------------------------------
# Step 3: Classify Answer Roles
# ---------------------------------------------------------------------------

def classify_answer_role(
    role_tag: str,
    stereotyped_groups: list[str],
) -> str:
    """Classify an answer's role based on its answer_info role_tag and the
    item's stereotyped_groups.

    Returns one of: ``"stereotyped_target"``, ``"non_stereotyped"``,
    ``"unknown"``.

    Ported from lgbtqmi/src/data/bbq_loader.py ``_classify_answer_role``
    with all edge-case handling preserved:
      - GI items where stereotyped_groups contains "F"/"M" but role tags
        are "woman"/"man"
      - GI items with "trans" in stereotyped_groups and tags like
        "trans_F", "nonTrans_M"
      - Race items with compound tags like "F-Black" (gender prefix)
      - Multi-word groups like "African American" requiring substring
        matching
    """
    tag_lower = role_tag.lower().strip()

    # Unknown answer
    if tag_lower in UNKNOWN_TAGS:
        return "unknown"

    stereo_lower = [g.lower().strip() for g in stereotyped_groups]

    # GI: stereotyped_groups is ["F"] or ["M"] but tags are "woman"/"man".
    if any(sg in ("f", "m") for sg in stereo_lower):
        if "f" in stereo_lower and tag_lower in ("woman", "girl", "female"):
            return "stereotyped_target"
        if "m" in stereo_lower and tag_lower in ("man", "boy", "male"):
            return "stereotyped_target"

    # GI: tags like "trans_F" / "nonTrans_M" with "trans" in stereotyped_groups.
    if any("trans" in sg for sg in stereo_lower):
        if "trans" in tag_lower and not tag_lower.startswith("nontrans"):
            return "stereotyped_target"

    # Exact match
    if tag_lower in stereo_lower:
        return "stereotyped_target"

    # Compound tag: "F-Black", "M-Latino" — identity after the hyphen.
    for sg in stereo_lower:
        if "-" in tag_lower and tag_lower.split("-", 1)[1] == sg:
            return "stereotyped_target"

    # Multi-word group matching: "African American" as tag or group.
    # Only apply substring matching when tag or group contains a space,
    # to avoid false positives like "old" matching "nonOld".
    for sg in stereo_lower:
        if " " in sg or " " in tag_lower:
            if sg in tag_lower or tag_lower in sg:
                return "stereotyped_target"

    # Space-stripped comparison: handles SES where role_tag is "lowSES"
    # but stereotyped_groups is "low SES".  Strip all spaces/hyphens/
    # underscores from both sides and compare.
    tag_stripped = tag_lower.replace(" ", "").replace("-", "").replace("_", "")
    for sg in stereo_lower:
        sg_stripped = sg.replace(" ", "").replace("-", "").replace("_", "")
        if tag_stripped == sg_stripped:
            return "stereotyped_target"

    return "non_stereotyped"


def classify_item_roles(
    raw: dict[str, Any],
) -> tuple[list[str], list[str]] | None:
    """Classify all three answer roles for a raw BBQ item.

    First attempts classification using role_tags (answer_info[ansN][1]).
    If that fails to produce a valid triple (exactly one of each role),
    falls back to using descriptive_tags (answer_info[ansN][0]).  This
    handles Nationality items where role_tags are geographic regions
    (e.g. "Europe") but descriptive_tags are nationalities (e.g. "British").

    Returns:
        ``(effective_tags, roles)`` — two parallel lists of length 3, or
        ``None`` if the answer_info is malformed (fewer than 3 answers).
        ``effective_tags`` are whichever tags (role or descriptive) were
        used for the successful classification.
    """
    answer_info = raw["answer_info"]
    stereotyped_groups = raw["additional_metadata"]["stereotyped_groups"]

    try:
        role_tags = [
            answer_info["ans0"][1],
            answer_info["ans1"][1],
            answer_info["ans2"][1],
        ]
    except (KeyError, IndexError):
        return None

    # Primary: classify using role_tags.
    roles = [classify_answer_role(tag, stereotyped_groups) for tag in role_tags]
    expected = {"stereotyped_target", "non_stereotyped", "unknown"}
    if set(roles) == expected:
        return role_tags, roles

    # Fallback: hybrid classification for categories like Nationality where
    # role_tags are geographic regions (can't distinguish stereotyped from
    # non-stereotyped) but correctly identify "unknown", while descriptive
    # tags (answer_info[ansN][0]) carry the actual identity labels but use
    # varied text for the unknown answer ("Not known", "Can't answer", etc.).
    #
    # Strategy: use role_tag to detect "unknown" answers, then use
    # descriptive_tags for the remaining two answers.
    try:
        desc_tags = [
            answer_info["ans0"][0],
            answer_info["ans1"][0],
            answer_info["ans2"][0],
        ]
    except (KeyError, IndexError):
        return role_tags, roles

    hybrid_roles: list[str] = []
    hybrid_tags: list[str] = []
    for i in range(3):
        if role_tags[i].lower().strip() in UNKNOWN_TAGS:
            # Role tag correctly identifies unknown — use it.
            hybrid_roles.append("unknown")
            hybrid_tags.append(role_tags[i])
        else:
            # Use descriptive tag for stereotyped/non-stereotyped distinction.
            hybrid_roles.append(
                classify_answer_role(desc_tags[i], stereotyped_groups)
            )
            hybrid_tags.append(desc_tags[i])

    if set(hybrid_roles) == expected:
        return hybrid_tags, hybrid_roles

    # Last resort: pure descriptive_tag classification (no hybrid).
    desc_roles = [classify_answer_role(tag, stereotyped_groups) for tag in desc_tags]
    if set(desc_roles) == expected:
        return desc_tags, desc_roles

    # Nothing worked — return the primary result (caller handles the failure).
    return role_tags, roles


# ---------------------------------------------------------------------------
# Step 4: Shuffle Answer Positions
# ---------------------------------------------------------------------------

def shuffle_answers(
    ans_texts: list[str],
    ans_roles: list[str],
    ans_role_tags: list[str],
    correct_idx: int,
    rng: random.Random,
) -> tuple[dict[str, str], str, dict[str, str], dict[str, str]]:
    """Shuffle answer positions and return shuffled mappings.

    Ported from lgbtqmi/src/data/bbq_loader.py ``_shuffle_answers``.

    Args:
        ans_texts: [ans0_text, ans1_text, ans2_text] in original order.
        ans_roles: [ans0_role, ans1_role, ans2_role] in original order.
        ans_role_tags: [ans0_tag, ans1_tag, ans2_tag] in original order.
        correct_idx: Index of the correct answer in the original ordering
                     (from ``raw["label"]``).
        rng: Seeded ``random.Random`` instance.

    Returns:
        Tuple of:
          answers:          {"A": text, "B": text, "C": text}
          correct_letter:   "A", "B", or "C"
          answer_roles:     {"A": role, "B": role, "C": role}
          answer_role_tags: {"A": tag, "B": tag, "C": tag}
    """
    indices = [0, 1, 2]
    rng.shuffle(indices)
    letters = ["A", "B", "C"]

    answers: dict[str, str] = {}
    answer_roles: dict[str, str] = {}
    answer_role_tags_by_letter: dict[str, str] = {}
    correct_letter = ""

    for letter, idx in zip(letters, indices):
        answers[letter] = ans_texts[idx]
        answer_roles[letter] = ans_roles[idx]
        answer_role_tags_by_letter[letter] = ans_role_tags[idx]
        if idx == correct_idx:
            correct_letter = letter

    return answers, correct_letter, answer_roles, answer_role_tags_by_letter


# ---------------------------------------------------------------------------
# Step 5: Normalize Subgroup Labels
# ---------------------------------------------------------------------------

def normalize_subgroup(label: str) -> str:
    """Normalize a single subgroup label.

    Lowercases, strips whitespace, and applies canonical collapses
    (e.g. "African American" → "black", "Arab" → "middle eastern").
    Labels not in the normalization map are returned lowercased/stripped.
    """
    cleaned = label.strip().lower()
    return SUBGROUP_NORMALIZATION.get(cleaned, cleaned)


def normalize_subgroups(
    stereotyped_groups_raw: list[str],
) -> tuple[list[str], bool, bool]:
    """Normalize and deduplicate a list of subgroup labels.

    Args:
        stereotyped_groups_raw: The raw ``stereotyped_groups`` list from
            BBQ ``additional_metadata``.

    Returns:
        Tuple of:
          normalized: Deduplicated list preserving first-occurrence order.
          was_normalized: True if any label changed during normalization.
          was_deduped: True if deduplication removed any entries.
    """
    raw_lower = [g.strip().lower() for g in stereotyped_groups_raw]
    normalized_all = [normalize_subgroup(g) for g in stereotyped_groups_raw]

    was_normalized = normalized_all != raw_lower

    # Deduplicate while preserving order (dict.fromkeys trick).
    normalized = list(dict.fromkeys(normalized_all))

    was_deduped = len(normalized) < len(normalized_all)

    return normalized, was_normalized, was_deduped


# ---------------------------------------------------------------------------
# Step 6: Determine Derived Fields
# ---------------------------------------------------------------------------

def derive_fields(
    answer_roles: dict[str, str],
    answer_role_tags: dict[str, str],
    stereotyped_groups: list[str],
) -> tuple[str, int, str | None]:
    """Compute derived fields from the shuffled answer mappings.

    Returns:
        Tuple of:
          stereotyped_option: Letter ("A"/"B"/"C") for stereotyped_target.
          n_target_groups: Number of (normalized, deduped) target groups.
          non_stereotyped_identity: The role tag of the non-stereotyped answer.
    """
    stereotyped_option = ""
    non_stereotyped_identity: str | None = None
    for letter, role in answer_roles.items():
        if role == "stereotyped_target":
            stereotyped_option = letter
        if role == "non_stereotyped":
            non_stereotyped_identity = answer_role_tags[letter]

    n_target_groups = len(stereotyped_groups)
    return stereotyped_option, n_target_groups, non_stereotyped_identity


# ---------------------------------------------------------------------------
# Step 7: Build Prompt String
# ---------------------------------------------------------------------------

def build_prompt(
    context: str,
    question: str,
    answers: dict[str, str],
) -> str:
    """Build the zero-shot multiple-choice prompt string.

    No trailing space or newline after "Answer:".  The model's next-token
    prediction at this position determines the answer.
    """
    return (
        f"Context: {context}\n"
        f"Question: {question}\n"
        f"A. {answers['A']}\n"
        f"B. {answers['B']}\n"
        f"C. {answers['C']}\n"
        f"Answer:"
    )


# ---------------------------------------------------------------------------
# Steps 8 + 9: Process a Single Item (assembly + filtering)
# ---------------------------------------------------------------------------

def process_item(
    raw: dict[str, Any],
    category_short: str,
    rng: random.Random,
) -> tuple[dict[str, Any] | None, str | None, bool, bool]:
    """Process a single raw BBQ item through Steps 3-8, with Step 9 filtering.

    Args:
        raw: A single raw item dict from the JSONL.
        category_short: Our short category name (e.g. "so").
        rng: Seeded RNG instance (shared across items — consumed sequentially).

    Returns:
        Tuple of:
          item: The processed item dict, or ``None`` if dropped.
          drop_reason: One of the drop reason keys, or ``None`` if kept.
          was_normalized: Whether subgroup normalization changed any label.
          was_deduped: Whether dedup removed any subgroup entries.
    """
    was_normalized = False
    was_deduped = False

    # --- Step 9 check: missing context or question ---
    context = raw.get("context", "")
    question = raw.get("question", "")
    if not context or not question:
        return None, "missing_context_or_question", was_normalized, was_deduped

    # --- Step 3: Classify answer roles ---
    classify_result = classify_item_roles(raw)
    if classify_result is None:
        return None, "malformed_answer_info", was_normalized, was_deduped

    role_tags, roles = classify_result
    expected = {"stereotyped_target", "non_stereotyped", "unknown"}
    if set(roles) != expected:
        return None, "role_assignment_failure", was_normalized, was_deduped

    # --- Step 5: Normalize subgroup labels ---
    stereotyped_groups_raw = raw["additional_metadata"]["stereotyped_groups"]
    stereotyped_groups, was_normalized, was_deduped = normalize_subgroups(
        stereotyped_groups_raw
    )

    # --- Step 9 check: empty stereotyped_groups after normalization ---
    if not stereotyped_groups:
        return None, "empty_stereotyped_groups", was_normalized, was_deduped

    # --- Step 4: Shuffle answer positions ---
    ans_texts = [raw["ans0"], raw["ans1"], raw["ans2"]]
    correct_idx = raw["label"]

    answers, correct_letter, answer_roles, answer_role_tags = shuffle_answers(
        ans_texts, roles, role_tags, correct_idx, rng,
    )

    # --- Step 6: Derived fields ---
    stereotyped_option, n_target_groups, non_stereotyped_identity = derive_fields(
        answer_roles, answer_role_tags, stereotyped_groups,
    )

    # --- Step 7: Build prompt ---
    prompt = build_prompt(context, question, answers)

    # --- Step 8: Assemble output ---
    processed_item: dict[str, Any] = {
        # Primary key
        "item_idx": raw["example_id"],
        # Category and condition
        "category": category_short,
        "context_condition": raw["context_condition"],
        "question_polarity": raw["question_polarity"],
        "subcategory": raw["additional_metadata"].get("subcategory", ""),
        # Prompt
        "prompt": prompt,
        # Answers (shuffled)
        "answers": answers,
        "answer_roles": answer_roles,
        "answer_role_tags": answer_role_tags,
        "correct_letter": correct_letter,
        "stereotyped_option": stereotyped_option,
        # Subgroup info
        "stereotyped_groups": stereotyped_groups,
        "n_target_groups": n_target_groups,
        "non_stereotyped_identity": non_stereotyped_identity,
        # BBQ metadata (preserved for auditing)
        "example_id": raw["example_id"],
        "question_index": raw["question_index"],
    }

    return processed_item, None, was_normalized, was_deduped


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_category(items: list[dict[str, Any]], category: str) -> list[str]:
    """Run post-processing validation checks on a category's items.

    Returns a list of warning messages.  Empty list = all good.
    Warnings are informational — they do NOT cause items to be dropped.
    """
    warnings: list[str] = []

    # 1. Every item has exactly one of each role.
    for it in items:
        roles = set(it["answer_roles"].values())
        if roles != {"stereotyped_target", "non_stereotyped", "unknown"}:
            warnings.append(
                f"Item {it['item_idx']}: unexpected roles {roles}"
            )

    # 2. Ambig items: correct answer should be "unknown".
    for it in items:
        correct_role = it["answer_roles"][it["correct_letter"]]
        if it["context_condition"] == "ambig" and correct_role != "unknown":
            warnings.append(
                f"Item {it['item_idx']}: ambig item correct answer is "
                f"{correct_role}, expected unknown"
            )

    # 3. stereotyped_option matches stereotyped_target role.
    for it in items:
        if it["answer_roles"].get(it["stereotyped_option"]) != "stereotyped_target":
            warnings.append(
                f"Item {it['item_idx']}: stereotyped_option mismatch"
            )

    # 4. No duplicate item_idx.
    idxs = [it["item_idx"] for it in items]
    if len(idxs) != len(set(idxs)):
        warnings.append("Duplicate item_idx values found")

    # 5. At least 2 subgroups with >= 10 items each.
    sub_counts: Counter[str] = Counter()
    for it in items:
        for sg in it["stereotyped_groups"]:
            sub_counts[sg] += 1
    viable = {sg: n for sg, n in sub_counts.items() if n >= 10}
    if len(viable) < 2:
        warnings.append(
            f"Fewer than 2 viable subgroups (>=10 items): {viable}"
        )

    # 6. Balanced ambig/disambig.
    n_ambig = sum(1 for it in items if it["context_condition"] == "ambig")
    n_disambig = sum(1 for it in items if it["context_condition"] == "disambig")
    if abs(n_ambig - n_disambig) > 0.1 * len(items):
        warnings.append(
            f"Imbalanced conditions: {n_ambig} ambig, {n_disambig} disambig"
        )

    return warnings


# ---------------------------------------------------------------------------
# Category-level orchestration
# ---------------------------------------------------------------------------

def process_category(
    category_short: str,
    jsonl_path: Path,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, int], int, int]:
    """Process all items for one category through the full pipeline.

    Implements the spec's processing flow: load → shuffle item order →
    process each item (Steps 3-8 with Step 9 filtering) → sort by item_idx.

    Args:
        category_short: Our short category name.
        jsonl_path: Path to the BBQ JSONL file.
        rng: Seeded RNG instance (shared across categories).

    Returns:
        Tuple of:
          items: Processed items, sorted by item_idx.
          drop_counts: Dict of drop reason → count.
          n_normalized: Number of items where subgroup labels were normalized.
          n_deduped: Number of items where dedup removed entries.
    """
    raw_items = load_raw_items(jsonl_path)
    log(f"  Loaded {len(raw_items)} raw items from {jsonl_path.name}")

    # Shuffle item processing order for better subgroup coverage in
    # small subsets (per spec).  The RNG is consumed here before per-item
    # answer shuffles, preserving deterministic sequencing.
    indexed = list(enumerate(raw_items))
    rng.shuffle(indexed)

    drop_counts: dict[str, int] = {
        "malformed_answer_info": 0,
        "role_assignment_failure": 0,
        "empty_stereotyped_groups": 0,
        "missing_context_or_question": 0,
    }
    n_normalized = 0
    n_deduped = 0
    processed: list[dict[str, Any]] = []

    for _shuffled_pos, raw in progress_bar(
        indexed, desc=f"  {category_short}", unit="items",
    ):
        item, drop_reason, was_norm, was_dedup = process_item(
            raw, category_short, rng,
        )
        if was_norm:
            n_normalized += 1
        if was_dedup:
            n_deduped += 1

        if drop_reason is not None:
            drop_counts[drop_reason] += 1
            continue

        assert item is not None
        processed.append(item)

    # Sort by item_idx (BBQ's example_id) for deterministic output.
    processed.sort(key=lambda it: it["item_idx"])

    # Log summary.
    total_dropped = sum(drop_counts.values())
    log(f"  Kept {len(processed)}/{len(raw_items)} items "
        f"(dropped {total_dropped})")
    if total_dropped > 0:
        for reason, count in drop_counts.items():
            if count > 0:
                log(f"    {reason}: {count}")

    return processed, drop_counts, n_normalized, n_deduped


def build_category_summary(
    items: list[dict[str, Any]],
    n_raw: int,
    drop_counts: dict[str, int],
    n_normalized: int,
    n_deduped: int,
) -> dict[str, Any]:
    """Build the per-category section of preparation_summary.json."""
    n_ambig = sum(1 for it in items if it["context_condition"] == "ambig")
    n_disambig = sum(1 for it in items if it["context_condition"] == "disambig")
    n_neg = sum(1 for it in items if it["question_polarity"] == "neg")
    n_nonneg = sum(1 for it in items if it["question_polarity"] == "nonneg")

    # Subgroup stats.
    sub_counts: Counter[str] = Counter()
    sub_single: Counter[str] = Counter()
    sub_multi: Counter[str] = Counter()
    for it in items:
        is_multi = it["n_target_groups"] > 1
        for sg in it["stereotyped_groups"]:
            sub_counts[sg] += 1
            if is_multi:
                sub_multi[sg] += 1
            else:
                sub_single[sg] += 1

    subgroups: dict[str, dict[str, Any]] = {}
    for sg in sorted(sub_counts.keys()):
        subgroups[sg] = {
            "n_items": sub_counts[sg],
            "n_single_group": sub_single[sg],
            "n_multi_group": sub_multi[sg],
        }

    return {
        "n_items_raw": n_raw,
        "n_items_kept": len(items),
        "n_dropped": drop_counts,
        "n_ambig": n_ambig,
        "n_disambig": n_disambig,
        "n_neg_polarity": n_neg,
        "n_nonneg_polarity": n_nonneg,
        "subgroups": subgroups,
        "n_items_normalized": n_normalized,
        "n_items_deduped": n_deduped,
    }
