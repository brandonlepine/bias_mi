"""Phase A unified runner: A1 → A2 → A3 sequentially.

Orchestrates the full extraction pipeline with a single config.json,
consistent device handling, data provenance tracking, and progress bars.

Usage:
    # Full pipeline (all three stages)
    python scripts/run_phase_a.py --run_dir runs/llama-3.1-8b_2026-04-16/

    # A1 only (no model/GPU needed)
    python scripts/run_phase_a.py --run_dir runs/... --stages A1

    # A1 + A2 (skip SAE encoding)
    python scripts/run_phase_a.py --run_dir runs/... --stages A1,A2

    # Quick test (20 items per category)
    python scripts/run_phase_a.py --run_dir runs/... --max_items 20 --categories so

    # Auto-detect device (or override)
    python scripts/run_phase_a.py --run_dir runs/... --device auto
    python scripts/run_phase_a.py --run_dir runs/... --device cuda
    python scripts/run_phase_a.py --run_dir runs/... --device mps

    # Create a new run from scratch
    python scripts/run_phase_a.py \\
        --run_dir runs/llama-3.1-8b_2026-04-16/ \\
        --model_path /path/to/model \\
        --bbq_data_dir datasets/bbq/data/ \\
        --sae_source OpenMOSS-Team/Llama3_1-8B-Base-LXR-32x \\
        --sae_expansion 32
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

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
from src.utils.config import (
    build_provenance,
    detect_device,
    load_config,
    save_config,
    setup_run_dir,
    validate_config,
)
from src.utils.io import atomic_save_json, ensure_dir
from src.utils.logging import log, progress_bar


# =========================================================================
# Argument parsing
# =========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase A: unified extraction pipeline (A1 → A2 → A3).",
    )
    # Run directory.
    parser.add_argument(
        "--run_dir", type=str, required=True,
        help="Path to the run directory.",
    )
    parser.add_argument(
        "--stages", type=str, default="A1,A2,A3",
        help="Comma-separated stages to run (default: A1,A2,A3).",
    )

    # Config overrides — used to create config.json if it doesn't exist.
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--bbq_data_dir", type=str, default=None)
    parser.add_argument("--sae_source", type=str, default=None)
    parser.add_argument("--sae_expansion", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, mps, cpu (default: auto).")
    parser.add_argument("--categories", type=str, default=None,
                        help="Comma-separated category short names.")

    # Execution control.
    parser.add_argument("--max_items", type=int, default=None,
                        help="Max items per category (for quick testing).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for A1 answer shuffling.")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices for A3 (default: all).")

    return parser.parse_args()


# =========================================================================
# Config resolution
# =========================================================================

def resolve_config(args: argparse.Namespace) -> dict:
    """Load existing config or build one from CLI args."""
    run_dir = Path(args.run_dir)
    config_path = run_dir / "config.json"

    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        log(f"Loaded existing config.json from {config_path}")
    else:
        config = {}
        log("No existing config.json — building from CLI args")

    # Apply CLI overrides.
    if args.model_path is not None:
        config["model_path"] = args.model_path
    if args.model_id is not None:
        config["model_id"] = args.model_id
    if args.bbq_data_dir is not None:
        config["bbq_data_dir"] = args.bbq_data_dir
    if args.sae_source is not None:
        config["sae_source"] = args.sae_source
    if args.sae_expansion is not None:
        config["sae_expansion"] = args.sae_expansion
    if args.categories is not None:
        config["categories"] = [c.strip() for c in args.categories.split(",")]

    # Defaults.
    config.setdefault("categories", list(ALL_CATEGORIES))
    config.setdefault("bbq_data_dir", "datasets/bbq/data/")
    config.setdefault("sae_expansion", 32)

    return config


# =========================================================================
# Stage A1: Prepare Stimuli
# =========================================================================

def run_a1(
    run_dir: Path,
    config: dict,
    seed: int,
) -> None:
    """A1: Transform raw BBQ JSONL files into clean, analysis-ready JSON."""
    log(f"\n{'=' * 70}")
    log(f"  STAGE A1: Prepare Stimuli")
    log(f"{'=' * 70}")
    t0 = time.time()

    bbq_data_dir = Path(config["bbq_data_dir"])
    if not bbq_data_dir.is_absolute():
        bbq_data_dir = PROJECT_ROOT / bbq_data_dir

    categories = config["categories"]

    found = find_bbq_files(bbq_data_dir, categories)
    if not found:
        raise RuntimeError("No BBQ files found. Check bbq_data_dir in config.")

    stimuli_dir = ensure_dir(run_dir / "A_extraction" / "stimuli")
    rng = random.Random(seed)

    per_category_summary: dict[str, dict] = {}
    total_kept = 0
    total_dropped = 0

    for cat in categories:
        if cat not in found:
            continue

        log(f"\n[A1/{cat}] Processing...")
        jsonl_path = found[cat]
        output_path = stimuli_dir / f"{cat}.json"
        n_raw = len(load_raw_items(jsonl_path))

        # Resume safety.
        if output_path.exists():
            try:
                with open(output_path) as f:
                    existing = json.load(f)
                if len(existing) > 0:
                    log(f"  Output exists ({len(existing)} items) — skipping. "
                        f"Delete {output_path.name} to reprocess.")
                    # Must still advance RNG for determinism.
                    _items, _drops, _nn, _nd = process_category(cat, jsonl_path, rng)
                    per_category_summary[cat] = build_category_summary(
                        existing, n_raw, _drops, _nn, _nd,
                    )
                    total_kept += len(existing)
                    total_dropped += sum(_drops.values())
                    continue
            except (json.JSONDecodeError, KeyError):
                log(f"  Existing output corrupt — reprocessing.")

        items, drop_counts, n_normalized, n_deduped = process_category(
            cat, jsonl_path, rng,
        )

        warnings = validate_category(items, cat)
        if warnings:
            log(f"  Validation warnings ({len(warnings)}):")
            for w in warnings:
                log(f"    {w}")
        else:
            log(f"  Validation: all checks passed")

        atomic_save_json(items, output_path)
        log(f"  Saved {len(items)} items → {output_path.name}")

        per_category_summary[cat] = build_category_summary(
            items, n_raw, drop_counts, n_normalized, n_deduped,
        )
        total_kept += len(items)
        total_dropped += sum(drop_counts.values())

    # Preparation summary.
    normalization_applied = {
        k: v for k, v in SUBGROUP_NORMALIZATION.items() if k != v
    }
    summary = {
        "bbq_data_dir": str(config["bbq_data_dir"]),
        "seed": seed,
        "categories_processed": [c for c in categories if c in found],
        "intersectional_skipped": ["Race_x_gender", "Race_x_SES"],
        "subgroup_normalization_applied": normalization_applied,
        "per_category": per_category_summary,
        "total_items_kept": total_kept,
        "total_items_dropped": total_dropped,
    }
    atomic_save_json(summary, stimuli_dir / "preparation_summary.json")

    elapsed = time.time() - t0
    log(f"\nA1 complete: {total_kept} items kept, {total_dropped} dropped "
        f"({elapsed:.1f}s)")


# =========================================================================
# Stage A2: Extract Activations
# =========================================================================

def run_a2(
    run_dir: Path,
    config: dict,
    device: str,
    max_items: int | None,
) -> None:
    """A2: Run items through the model and capture hidden states."""
    log(f"\n{'=' * 70}")
    log(f"  STAGE A2: Extract Activations (device: {device})")
    log(f"{'=' * 70}")
    t0 = time.time()

    from src.extraction.activations import (
        build_and_save_extraction_summary,
        extract_category,
        load_model,
    )

    model_path = config["model_path"]
    categories = config["categories"]

    log(f"Loading model: {model_path}")
    model, tokenizer, n_layers, hidden_dim, get_layer_fn = load_model(
        model_path, device,
    )

    # Update config with architecture info.
    if config.get("n_layers") is None:
        config["n_layers"] = n_layers
        config["hidden_dim"] = hidden_dim
        save_config(config, run_dir)
        log(f"  Updated config: n_layers={n_layers}, hidden_dim={hidden_dim}")

    for cat in categories:
        stimuli_path = run_dir / "A_extraction" / "stimuli" / f"{cat}.json"
        if not stimuli_path.exists():
            log(f"No stimuli for {cat} — run A1 first. Skipping.")
            continue

        output_dir = ensure_dir(
            run_dir / "A_extraction" / "activations" / cat,
        )
        with open(stimuli_path) as f:
            items = json.load(f)
        if max_items:
            items = items[:max_items]

        extract_category(
            items, model, tokenizer, get_layer_fn,
            n_layers, hidden_dim, device, output_dir, cat,
        )

    build_and_save_extraction_summary(run_dir, categories, config)

    elapsed = time.time() - t0
    log(f"\nA2 complete ({elapsed:.1f}s)")


# =========================================================================
# Stage A3: SAE Encoding
# =========================================================================

def run_a3(
    run_dir: Path,
    config: dict,
    device: str,
    max_items: int | None,
    layers: list[int] | None,
) -> None:
    """A3: Encode hidden states through SAEs at each layer."""
    log(f"\n{'=' * 70}")
    log(f"  STAGE A3: SAE Encoding (device: {device})")
    log(f"{'=' * 70}")
    t0 = time.time()

    from src.extraction.sae_encoding import (
        build_encoding_summary,
        build_subgroup_lookup,
        encode_layer,
        load_all_stimuli,
        load_metadata,
        select_encode_device,
        validate_sae_source,
    )

    n_layers = config.get("n_layers")
    if n_layers is None:
        raise RuntimeError("config.json missing n_layers. Run A2 first.")

    encode_device = select_encode_device(device)
    log(f"SAE encoding device: {encode_device}")

    if layers is None:
        layers = list(range(n_layers))
    categories = config["categories"]

    validate_sae_source(config["sae_source"], layers[0], config["sae_expansion"])

    load_metadata(run_dir)
    stimuli_by_cat = load_all_stimuli(run_dir, categories)
    subgroup_lookup = build_subgroup_lookup(stimuli_by_cat)

    failed_layers: list[int] = []
    for layer in progress_bar(layers, desc="  SAE layers", unit="layer"):
        success = encode_layer(
            layer, run_dir, config, categories, encode_device,
            subgroup_lookup, max_items,
        )
        if not success:
            failed_layers.append(layer)

    build_encoding_summary(
        run_dir, config, layers, failed_layers, categories, encode_device,
    )

    elapsed = time.time() - t0
    log(f"\nA3 complete ({elapsed:.1f}s)")
    if failed_layers:
        log(f"  WARNING: {len(failed_layers)} layers failed: {failed_layers}")


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    args = parse_args()
    stages = [s.strip().upper() for s in args.stages.split(",")]
    valid_stages = {"A1", "A2", "A3"}
    invalid = set(stages) - valid_stages
    if invalid:
        log(f"ERROR: invalid stages: {invalid}. Valid: {valid_stages}")
        sys.exit(1)

    # Resolve config.
    config = resolve_config(args)
    run_dir = setup_run_dir(args.run_dir, config)

    # Validate config for requested stages.
    errors = validate_config(config, stages, PROJECT_ROOT)
    if errors:
        log("Config validation errors:")
        for e in errors:
            log(f"  - {e}")
        sys.exit(1)

    # Detect device.
    device = detect_device(args.device)
    config["device"] = device
    save_config(config, run_dir)

    # Resolve layers for A3.
    a3_layers = None
    if args.layers is not None:
        a3_layers = [int(x) for x in args.layers.split(",")]

    # Provenance.
    provenance = build_provenance(config, device, stages)
    provenance_path = run_dir / "provenance.json"
    # Append to provenance log (don't overwrite).
    existing_provenance: list = []
    if provenance_path.exists():
        try:
            with open(provenance_path) as f:
                existing_provenance = json.load(f)
        except (json.JSONDecodeError, TypeError):
            existing_provenance = []
    existing_provenance.append(provenance)
    atomic_save_json(existing_provenance, provenance_path)

    # Banner.
    log(f"\n{'=' * 70}")
    log(f"  Phase A Pipeline")
    log(f"{'=' * 70}")
    log(f"  Run dir:    {run_dir}")
    log(f"  Stages:     {' → '.join(stages)}")
    log(f"  Device:     {device}")
    log(f"  Categories: {config['categories']}")
    if args.max_items:
        log(f"  Max items:  {args.max_items}")
    log(f"{'=' * 70}")

    pipeline_t0 = time.time()

    # Execute stages sequentially.
    if "A1" in stages:
        run_a1(run_dir, config, args.seed)

    if "A2" in stages:
        run_a2(run_dir, config, device, args.max_items)

    if "A3" in stages:
        run_a3(run_dir, config, device, args.max_items, a3_layers)

    total_elapsed = time.time() - pipeline_t0
    log(f"\n{'=' * 70}")
    log(f"  Phase A complete — {total_elapsed:.1f}s total")
    log(f"{'=' * 70}")


if __name__ == "__main__":
    main()
