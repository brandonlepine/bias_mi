"""A1: Prepare Stimuli — transform raw BBQ JSONL files into clean, analysis-ready JSON.

Pure data processing — no model, no GPU.

Usage:
    python scripts/A1_prepare_stimuli.py --run_dir runs/llama-3.1-8b_2026-04-15/

    # Override categories
    python scripts/A1_prepare_stimuli.py --run_dir runs/... --categories so,race

    # Custom seed
    python scripts/A1_prepare_stimuli.py --run_dir runs/... --seed 42
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

# Ensure project root is importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.bbq_loader import (
    ALL_CATEGORIES,
    CATEGORY_FILE_MAP,
    SUBGROUP_NORMALIZATION,
    build_category_summary,
    find_bbq_files,
    load_raw_items,
    process_category,
    validate_category,
)
from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A1: Prepare BBQ stimuli for a run."
    )
    parser.add_argument(
        "--run_dir", type=str, required=True,
        help="Path to the run directory (must contain config.json).",
    )
    parser.add_argument(
        "--categories", type=str, default=None,
        help="Comma-separated category short names to process (default: from config.json).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for answer shuffling (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)

    # ── Load config ──────────────────────────────────────────────────
    config_path = run_dir / "config.json"
    if not config_path.exists():
        log(f"ERROR: config.json not found at {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    bbq_data_dir = Path(config["bbq_data_dir"])
    if not bbq_data_dir.is_absolute():
        bbq_data_dir = PROJECT_ROOT / bbq_data_dir

    # ── Resolve categories ───────────────────────────────────────────
    if args.categories is not None:
        categories = [c.strip() for c in args.categories.split(",")]
    elif "categories" in config:
        categories = config["categories"]
    else:
        categories = list(ALL_CATEGORIES)

    log(f"A1: Preparing stimuli for {len(categories)} categories")
    log(f"  run_dir: {run_dir}")
    log(f"  bbq_data_dir: {bbq_data_dir}")
    log(f"  seed: {args.seed}")
    log(f"  categories: {categories}")

    # ── File discovery ───────────────────────────────────────────────
    found = find_bbq_files(bbq_data_dir, categories)
    if not found:
        log("ERROR: no BBQ files found. Check bbq_data_dir in config.json.")
        sys.exit(1)

    # ── Output directory ─────────────────────────────────────────────
    stimuli_dir = ensure_dir(run_dir / "A_extraction" / "stimuli")

    # ── Single RNG for the entire run (shared across categories) ─────
    rng = random.Random(args.seed)

    # ── Process each category ────────────────────────────────────────
    per_category_summary: dict[str, dict] = {}
    total_kept = 0
    total_dropped = 0

    for cat in categories:
        if cat not in found:
            continue  # warning already logged by find_bbq_files

        log(f"\n[{cat}] Processing...")
        jsonl_path = found[cat]

        # Resume safety: skip if output already exists with expected count.
        output_path = stimuli_dir / f"{cat}.json"
        n_raw = len(load_raw_items(jsonl_path))

        if output_path.exists():
            try:
                with open(output_path) as f:
                    existing = json.load(f)
                # Re-process if item count looks wrong (could be partial).
                # "Expected" count is unknown until we process, so we use
                # a basic sanity check: file exists and has items.
                if len(existing) > 0:
                    log(f"  Output exists with {len(existing)} items — skipping. "
                        f"Delete {output_path.name} to reprocess.")

                    # Still need to consume the RNG the same way for determinism.
                    # Re-run process_category but discard results to advance RNG.
                    rng_check = random.Random(0)  # throwaway
                    # We need to advance the REAL rng as if we processed this category.
                    # This is tricky — we must consume the same RNG calls.
                    # Safest: just process and discard.
                    _items, _drops, _nn, _nd = process_category(cat, jsonl_path, rng)

                    # Build summary from the existing file.
                    per_category_summary[cat] = build_category_summary(
                        existing, n_raw, _drops, _nn, _nd,
                    )
                    total_kept += len(existing)
                    total_dropped += sum(_drops.values())
                    continue
            except (json.JSONDecodeError, KeyError):
                log(f"  Existing output is corrupt — reprocessing.")

        # Process the category.
        items, drop_counts, n_normalized, n_deduped = process_category(
            cat, jsonl_path, rng,
        )

        # Validate.
        warnings = validate_category(items, cat)
        if warnings:
            log(f"  Validation warnings ({len(warnings)}):")
            for w in warnings:
                log(f"    {w}")
        else:
            log(f"  Validation: all checks passed")

        # Save (atomic write).
        atomic_save_json(items, output_path)
        log(f"  Saved {len(items)} items to {output_path.name}")

        # Summary.
        per_category_summary[cat] = build_category_summary(
            items, n_raw, drop_counts, n_normalized, n_deduped,
        )
        total_kept += len(items)
        total_dropped += sum(drop_counts.values())

    # ── Build and save preparation summary ───────────────────────────
    # Only include normalization entries that actually changed something.
    normalization_applied = {
        k: v for k, v in SUBGROUP_NORMALIZATION.items() if k != v
    }

    summary = {
        "bbq_data_dir": str(config["bbq_data_dir"]),
        "seed": args.seed,
        "categories_processed": [c for c in categories if c in found],
        "intersectional_skipped": ["Race_x_gender", "Race_x_SES"],
        "subgroup_normalization_applied": normalization_applied,
        "per_category": per_category_summary,
        "total_items_kept": total_kept,
        "total_items_dropped": total_dropped,
    }

    summary_path = stimuli_dir / "preparation_summary.json"
    atomic_save_json(summary, summary_path)
    log(f"\nSaved preparation summary to {summary_path.name}")
    log(f"Total: {total_kept} items kept, {total_dropped} dropped")


if __name__ == "__main__":
    main()
