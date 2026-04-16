"""Tests for A1 prepare_stimuli Steps 1-5.

Run from project root:
    python -m pytest tests/test_a1_steps1_5.py -v
    # or simply:
    python tests/test_a1_steps1_5.py
"""

import json
import random
import sys
import tempfile
from pathlib import Path

# Ensure project root is on sys.path for src imports.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.bbq_loader import (
    ALL_CATEGORIES,
    CATEGORY_FILE_MAP,
    classify_answer_role,
    classify_item_roles,
    find_bbq_files,
    load_raw_items,
    normalize_subgroup,
    normalize_subgroups,
    shuffle_answers,
)

BBQ_DATA_DIR = PROJECT_ROOT / "datasets" / "bbq" / "data"

# ── helpers ──────────────────────────────────────────────────────────────

passed = 0
failed = 0


def check(condition: bool, name: str) -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✓ {name}", flush=True)
    else:
        failed += 1
        print(f"  ✗ FAIL: {name}", flush=True)


# =========================================================================
# Step 1: File Discovery
# =========================================================================

def test_step1_file_discovery() -> None:
    print("\n=== Step 1: File Discovery ===", flush=True)

    # 1a. All 9 categories should resolve to existing files.
    found = find_bbq_files(BBQ_DATA_DIR, ALL_CATEGORIES)
    check(len(found) == 9, f"found all 9 categories (got {len(found)})")
    for cat in ALL_CATEGORIES:
        check(cat in found, f"  category '{cat}' found")
        check(found[cat].exists(), f"  file exists: {found[cat].name}")

    # 1b. Unknown category is skipped (with warning).
    found2 = find_bbq_files(BBQ_DATA_DIR, ["so", "FAKE_CATEGORY"])
    check(len(found2) == 1, f"unknown category skipped (got {len(found2)} files)")
    check("so" in found2, "valid category still returned")

    # 1c. Intersectional files should NOT appear.
    check("Race_x_gender" not in [p.stem for p in found.values()],
          "Race_x_gender not in results")
    check("Race_x_SES" not in [p.stem for p in found.values()],
          "Race_x_SES not in results")

    # 1d. Category map covers exactly the 9 non-intersectional categories.
    check(set(CATEGORY_FILE_MAP.keys()) == {
        "age", "disability", "gi", "nationality", "physical_appearance",
        "race", "religion", "ses", "so",
    }, "CATEGORY_FILE_MAP has exactly 9 entries")


# =========================================================================
# Step 2: Load Raw Items
# =========================================================================

def test_step2_load_raw_items() -> None:
    print("\n=== Step 2: Load Raw Items ===", flush=True)

    # 2a. Load SO and check basic properties.
    so_path = BBQ_DATA_DIR / "Sexual_orientation.jsonl"
    items = load_raw_items(so_path)
    check(len(items) > 0, f"loaded {len(items)} SO items")

    # 2b. Every item has required fields.
    required_fields = {
        "example_id", "question_index", "question_polarity",
        "context_condition", "category", "answer_info",
        "additional_metadata", "context", "question",
        "ans0", "ans1", "ans2", "label",
    }
    first = items[0]
    missing = required_fields - set(first.keys())
    check(len(missing) == 0, f"first item has all required fields (missing: {missing})")

    # 2c. answer_info has ans0, ans1, ans2 with [text, role_tag] each.
    ai = first["answer_info"]
    check(all(k in ai for k in ["ans0", "ans1", "ans2"]),
          "answer_info has ans0/ans1/ans2")
    check(all(len(ai[k]) == 2 for k in ["ans0", "ans1", "ans2"]),
          "each answer_info entry is a 2-element list")

    # 2d. additional_metadata has stereotyped_groups.
    check("stereotyped_groups" in first["additional_metadata"],
          "additional_metadata has stereotyped_groups")

    # 2e. Empty lines are skipped gracefully.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        tmp.write('{"example_id": 0, "test": true}\n')
        tmp.write("\n")
        tmp.write('{"example_id": 1, "test": true}\n')
        tmp.write("   \n")
        tmp_path = Path(tmp.name)
    loaded = load_raw_items(tmp_path)
    check(len(loaded) == 2, f"empty lines skipped (loaded {len(loaded)} items)")
    tmp_path.unlink()

    # 2f. Load every category to confirm all files are parseable.
    for cat, bbq_name in CATEGORY_FILE_MAP.items():
        p = BBQ_DATA_DIR / f"{bbq_name}.jsonl"
        cat_items = load_raw_items(p)
        check(len(cat_items) > 0, f"  {cat}: loaded {len(cat_items)} items")


# =========================================================================
# Step 3: Classify Answer Roles
# =========================================================================

def test_step3_classify_answer_roles() -> None:
    print("\n=== Step 3: Classify Answer Roles ===", flush=True)

    # 3a. Basic: "unknown" tag → "unknown".
    check(classify_answer_role("unknown", ["gay"]) == "unknown",
          "unknown tag → unknown")

    # 3b. Direct match: "gay" with stereotyped_groups=["gay"].
    check(classify_answer_role("gay", ["gay"]) == "stereotyped_target",
          "exact match: gay → stereotyped_target")

    # 3c. Non-match: "lesbian" not in stereotyped_groups=["gay"].
    check(classify_answer_role("lesbian", ["gay"]) == "non_stereotyped",
          "non-match: lesbian with groups=[gay] → non_stereotyped")

    # 3d. GI edge case: stereotyped_groups=["F"], tag="woman".
    check(classify_answer_role("woman", ["F"]) == "stereotyped_target",
          "GI: woman with groups=[F] → stereotyped_target")
    check(classify_answer_role("man", ["F"]) == "non_stereotyped",
          "GI: man with groups=[F] → non_stereotyped")

    # 3e. GI edge case: stereotyped_groups=["M"], tag="man".
    check(classify_answer_role("man", ["M"]) == "stereotyped_target",
          "GI: man with groups=[M] → stereotyped_target")
    check(classify_answer_role("woman", ["M"]) == "non_stereotyped",
          "GI: woman with groups=[M] → non_stereotyped")

    # 3f. Trans handling: stereotyped_groups contains "trans".
    check(classify_answer_role("trans_F", ["transgender women"]) == "stereotyped_target",
          "trans: trans_F with groups=[transgender women] → stereotyped_target")
    check(classify_answer_role("nonTrans_M", ["transgender women"]) == "non_stereotyped",
          "trans: nonTrans_M with groups=[transgender women] → non_stereotyped")

    # 3g. Compound race tag: "F-Black" with groups=["Black"].
    check(classify_answer_role("F-Black", ["Black"]) == "stereotyped_target",
          "compound: F-Black with groups=[Black] → stereotyped_target")
    check(classify_answer_role("M-White", ["Black"]) == "non_stereotyped",
          "compound: M-White with groups=[Black] → non_stereotyped")

    # 3h. Multi-word group: "African American".
    check(classify_answer_role("African American", ["African American"]) == "stereotyped_target",
          "multi-word: African American exact match → stereotyped_target")

    # 3i. Case insensitivity.
    check(classify_answer_role("GAY", ["gay"]) == "stereotyped_target",
          "case insensitive: GAY with groups=[gay] → stereotyped_target")
    check(classify_answer_role("Unknown", ["gay"]) == "unknown",
          "case insensitive: Unknown → unknown")

    # 3j. classify_item_roles: full item classification on real SO data.
    so_path = BBQ_DATA_DIR / "Sexual_orientation.jsonl"
    so_items = load_raw_items(so_path)
    n_valid = 0
    n_malformed = 0
    for raw in so_items:
        result = classify_item_roles(raw)
        if result is None:
            n_malformed += 1
            continue
        role_tags, roles = result
        role_set = set(roles)
        expected = {"stereotyped_target", "non_stereotyped", "unknown"}
        if role_set == expected:
            n_valid += 1
    check(n_valid > 0, f"SO: {n_valid} items with valid role triples")
    check(n_malformed == 0, f"SO: {n_malformed} malformed items")
    print(f"    SO valid: {n_valid}/{len(so_items)} "
          f"({100*n_valid/len(so_items):.1f}%)", flush=True)

    # 3k. Run on all categories to check role coverage.
    print("  Role classification across all categories:", flush=True)
    for cat, bbq_name in CATEGORY_FILE_MAP.items():
        p = BBQ_DATA_DIR / f"{bbq_name}.jsonl"
        cat_items = load_raw_items(p)
        valid = 0
        malformed = 0
        role_fail = 0
        for raw in cat_items:
            result = classify_item_roles(raw)
            if result is None:
                malformed += 1
                continue
            _, roles = result
            if set(roles) == {"stereotyped_target", "non_stereotyped", "unknown"}:
                valid += 1
            else:
                role_fail += 1
        total = len(cat_items)
        pct = 100 * valid / total if total > 0 else 0
        check(valid > 0, f"  {cat}: {valid}/{total} valid ({pct:.1f}%), "
              f"{role_fail} role failures, {malformed} malformed")


# =========================================================================
# Step 4: Shuffle Answer Positions
# =========================================================================

def test_step4_shuffle_answers() -> None:
    print("\n=== Step 4: Shuffle Answer Positions ===", flush=True)

    texts = ["The lesbian woman", "Can't be determined", "The gay man"]
    roles = ["non_stereotyped", "unknown", "stereotyped_target"]
    tags = ["lesbian", "unknown", "gay"]
    correct_idx = 1  # "Can't be determined" is correct

    # 4a. Determinism: same seed → same result.
    rng1 = random.Random(42)
    result1 = shuffle_answers(texts, roles, tags, correct_idx, rng1)

    rng2 = random.Random(42)
    result2 = shuffle_answers(texts, roles, tags, correct_idx, rng2)

    check(result1 == result2, "same seed produces identical shuffle")

    # 4b. Different seed → different result (with high probability).
    rng3 = random.Random(99)
    result3 = shuffle_answers(texts, roles, tags, correct_idx, rng3)
    # It's possible but very unlikely the shuffles are identical.
    # We test that the function at least runs; strict inequality isn't guaranteed.
    check(result3 is not None, "different seed produces a result")

    # 4c. Output structure: answers, correct_letter, answer_roles, answer_role_tags.
    answers, correct_letter, answer_roles, answer_role_tags = result1
    check(set(answers.keys()) == {"A", "B", "C"}, "answers has keys A, B, C")
    check(set(answer_roles.keys()) == {"A", "B", "C"}, "answer_roles has keys A, B, C")
    check(set(answer_role_tags.keys()) == {"A", "B", "C"}, "answer_role_tags has keys A, B, C")
    check(correct_letter in ("A", "B", "C"), f"correct_letter is valid: {correct_letter}")

    # 4d. correct_letter actually maps to the correct answer text.
    check(answers[correct_letter] == texts[correct_idx],
          f"correct_letter '{correct_letter}' maps to correct text")

    # 4e. All three original texts are present (no duplication/loss).
    check(set(answers.values()) == set(texts), "all original texts preserved")

    # 4f. All three roles are present.
    check(set(answer_roles.values()) == set(roles), "all original roles preserved")

    # 4g. Tags and roles are paired correctly after shuffle.
    for letter in ["A", "B", "C"]:
        # Find which original index this letter maps to.
        orig_text = answers[letter]
        orig_idx = texts.index(orig_text)
        check(answer_roles[letter] == roles[orig_idx],
              f"  {letter}: role matches original index {orig_idx}")
        check(answer_role_tags[letter] == tags[orig_idx],
              f"  {letter}: tag matches original index {orig_idx}")

    # 4h. Sequential RNG usage: item-order shuffle then answer shuffle.
    #     Verify that the RNG is consumed in the expected order.
    rng_seq = random.Random(42)
    # Simulate what the full pipeline does: shuffle item order, then per-item
    # answer shuffle.
    so_items = load_raw_items(BBQ_DATA_DIR / "Sexual_orientation.jsonl")
    indexed = list(enumerate(so_items))
    rng_seq.shuffle(indexed)  # item-order shuffle consumes RNG

    # Now do the first item's answer shuffle.
    first_orig_idx, first_raw = indexed[0]
    ai = first_raw["answer_info"]
    first_texts = [first_raw["ans0"], first_raw["ans1"], first_raw["ans2"]]
    first_tags = [ai["ans0"][1], ai["ans1"][1], ai["ans2"][1]]
    first_roles = [classify_answer_role(t, first_raw["additional_metadata"]["stereotyped_groups"])
                   for t in first_tags]
    first_result = shuffle_answers(first_texts, first_roles, first_tags,
                                   first_raw["label"], rng_seq)

    # Repeat with a fresh RNG of the same seed — should get identical state.
    rng_verify = random.Random(42)
    indexed2 = list(enumerate(so_items))
    rng_verify.shuffle(indexed2)
    first2_orig_idx, first2_raw = indexed2[0]
    ai2 = first2_raw["answer_info"]
    first2_texts = [first2_raw["ans0"], first2_raw["ans1"], first2_raw["ans2"]]
    first2_tags = [ai2["ans0"][1], ai2["ans1"][1], ai2["ans2"][1]]
    first2_roles = [classify_answer_role(t, first2_raw["additional_metadata"]["stereotyped_groups"])
                    for t in first2_tags]
    first2_result = shuffle_answers(first2_texts, first2_roles, first2_tags,
                                    first2_raw["label"], rng_verify)

    check(first_orig_idx == first2_orig_idx,
          "sequential RNG: same first item after item-order shuffle")
    check(first_result == first2_result,
          "sequential RNG: same answer shuffle for first item")


# =========================================================================
# Step 5: Normalize Subgroup Labels
# =========================================================================

def test_step5_normalize_subgroups() -> None:
    print("\n=== Step 5: Normalize Subgroup Labels ===", flush=True)

    # 5a. Basic normalization.
    check(normalize_subgroup("African American") == "black",
          "African American → black")
    check(normalize_subgroup("african-american") == "black",
          "african-american → black")
    check(normalize_subgroup("Black") == "black",
          "Black → black")
    check(normalize_subgroup("Arab") == "middle eastern",
          "Arab → middle eastern")
    check(normalize_subgroup("Middle Eastern") == "middle eastern",
          "Middle Eastern → middle eastern")

    # 5b. Pass-through: labels not in normalization map are lowered/stripped.
    check(normalize_subgroup("gay") == "gay", "gay → gay (pass-through)")
    check(normalize_subgroup("  Hindu  ") == "hindu", "Hindu stripped/lowered")
    check(normalize_subgroup("transgender women") == "transgender women",
          "transgender women → unchanged (not collapsed)")
    check(normalize_subgroup("lesbian") == "lesbian",
          "lesbian → unchanged (not collapsed)")

    # 5c. Deduplication: ["Black", "African American"] → ["black"].
    norm, was_normalized, was_deduped = normalize_subgroups(
        ["Black", "African American"]
    )
    check(norm == ["black"], f"dedup: ['Black','African American'] → {norm}")
    check(was_normalized, "was_normalized=True for African American")
    check(was_deduped, "was_deduped=True when entries collapse")

    # 5d. No dedup needed: ["gay", "lesbian"].
    norm2, was_norm2, was_dedup2 = normalize_subgroups(["gay", "lesbian"])
    check(norm2 == ["gay", "lesbian"], f"no dedup: {norm2}")
    check(not was_norm2, "was_normalized=False when no change")
    check(not was_dedup2, "was_deduped=False when no duplicates")

    # 5e. Order preservation: first occurrence wins.
    norm3, _, _ = normalize_subgroups(
        ["African American", "Black", "Hispanic"]
    )
    check(norm3 == ["black", "hispanic"],
          f"order preserved, first occurrence wins: {norm3}")

    # 5f. Empty input.
    norm4, was_norm4, was_dedup4 = normalize_subgroups([])
    check(norm4 == [], "empty input → empty output")

    # 5g. Real data: check Race_ethnicity items for normalization effects.
    race_items = load_raw_items(BBQ_DATA_DIR / "Race_ethnicity.jsonl")
    n_normalized = 0
    n_deduped = 0
    all_subgroups: set[str] = set()
    for raw in race_items:
        groups_raw = raw["additional_metadata"]["stereotyped_groups"]
        norm, was_n, was_d = normalize_subgroups(groups_raw)
        if was_n:
            n_normalized += 1
        if was_d:
            n_deduped += 1
        all_subgroups.update(norm)
    check(n_normalized > 0,
          f"race: {n_normalized} items had labels normalized")
    check(n_deduped > 0,
          f"race: {n_deduped} items had labels deduplicated")
    print(f"    race subgroups after normalization: {sorted(all_subgroups)}",
          flush=True)

    # 5h. Verify things we do NOT collapse remain separate.
    check("transgender women" not in CATEGORY_FILE_MAP,
          "transgender women is not a category (sanity)")
    norm_tw, _, _ = normalize_subgroups(["transgender women", "transgender men"])
    check(norm_tw == ["transgender women", "transgender men"],
          "trans groups NOT collapsed")

    # 5i. Check SO subgroups: gay, lesbian, bisexual, pansexual remain separate.
    so_items = load_raw_items(BBQ_DATA_DIR / "Sexual_orientation.jsonl")
    so_subgroups: set[str] = set()
    for raw in so_items:
        norm, _, _ = normalize_subgroups(
            raw["additional_metadata"]["stereotyped_groups"]
        )
        so_subgroups.update(norm)
    check("gay" in so_subgroups, "SO: 'gay' subgroup present")
    check("lesbian" in so_subgroups, "SO: 'lesbian' subgroup present")
    check("bisexual" in so_subgroups, "SO: 'bisexual' subgroup present")
    print(f"    SO subgroups: {sorted(so_subgroups)}", flush=True)


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("A1 Steps 1-5 Test Suite", flush=True)
    print("=" * 60, flush=True)

    test_step1_file_discovery()
    test_step2_load_raw_items()
    test_step3_classify_answer_roles()
    test_step4_shuffle_answers()
    test_step5_normalize_subgroups()

    print("\n" + "=" * 60, flush=True)
    print(f"Results: {passed} passed, {failed} failed", flush=True)
    print("=" * 60, flush=True)

    sys.exit(1 if failed > 0 else 0)
