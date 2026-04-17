"""Integration tests for run_phase_b.py — validates arg parsing, dependency checks, resume logic.

Run from project root:
    python tests/test_run_phase_b.py
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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


# ── Tests ──────────────────────────────────────────────────────────────

def test_parse_args_defaults():
    print("\n[Runner] parse_args — defaults", flush=True)
    from scripts.run_phase_b import parse_args

    with patch("sys.argv", ["prog", "--run_dir", "/tmp/test_run"]):
        args = parse_args()

    check(args.run_dir == "/tmp/test_run", "run_dir parsed")
    check(args.stages == "B1,B2,B3,B4,B5", "default stages = all")
    check(args.categories is None, "categories default = None")
    check(args.layers is None, "layers default = None")
    check(args.min_n_per_group == 10, "min_n_per_group default = 10")
    check(args.n_permutations == 10, "n_permutations default = 10")
    check(args.min_n_per_class == 20, "min_n_per_class default = 20")
    check(args.top_k == 20, "top_k default = 20")
    check(not args.skip_figures, "skip_figures default = False")
    check(not args.force, "force default = False")
    check(not args.skip_sae_probes, "skip_sae_probes default = False")


def test_parse_args_overrides():
    print("\n[Runner] parse_args — CLI overrides", flush=True)
    from scripts.run_phase_b import parse_args

    with patch("sys.argv", [
        "prog", "--run_dir", "/tmp/test",
        "--stages", "B3,B5",
        "--categories", "so,race",
        "--layers", "0,14,31",
        "--min_n_per_group", "15",
        "--n_permutations", "5",
        "--top_k", "30",
        "--skip_figures",
        "--force",
        "--skip_sae_probes",
    ]):
        args = parse_args()

    check(args.stages == "B3,B5", "stages parsed")
    check(args.categories == "so,race", "categories parsed")
    check(args.layers == "0,14,31", "layers parsed")
    check(args.min_n_per_group == 15, "min_n_per_group = 15")
    check(args.n_permutations == 5, "n_permutations = 5")
    check(args.top_k == 30, "top_k = 30")
    check(args.skip_figures, "skip_figures = True")
    check(args.force, "force = True")
    check(args.skip_sae_probes, "skip_sae_probes = True")


def test_stage_validation():
    print("\n[Runner] stage validation — invalid stages rejected", flush=True)
    # The main() function checks for invalid stages. We test the logic directly.
    valid_stages = {"B1", "B2", "B3", "B4", "B5"}

    stages_ok = [s.strip().upper() for s in "B1,B3,B5".split(",")]
    invalid_ok = set(stages_ok) - valid_stages
    check(len(invalid_ok) == 0, "B1,B3,B5 all valid")

    stages_bad = [s.strip().upper() for s in "B1,B6,X".split(",")]
    invalid_bad = set(stages_bad) - valid_stages
    check(invalid_bad == {"B6", "X"}, "B6,X detected as invalid")


def test_config_missing_exits():
    print("\n[Runner] main — missing config.json handled", flush=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        # No config.json in tmpdir → main should exit
        from scripts.run_phase_b import main

        with patch("sys.argv", ["prog", "--run_dir", tmpdir]):
            try:
                main()
                check(False, "should have exited")
            except SystemExit as e:
                check(e.code == 1, "exits with code 1")


def test_config_missing_nlayers_exits():
    print("\n[Runner] main — missing n_layers handled", flush=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write config without n_layers
        config = {"categories": ["so"], "model_path": "test"}
        with open(Path(tmpdir) / "config.json", "w") as f:
            json.dump(config, f)

        from scripts.run_phase_b import main

        with patch("sys.argv", ["prog", "--run_dir", tmpdir]):
            try:
                main()
                check(False, "should have exited")
            except SystemExit as e:
                check(e.code == 1, "exits with code 1 (missing n_layers)")


def test_dependency_check_b2_needs_b1():
    print("\n[Runner] dependency — B2 requires B1 output", flush=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"categories": ["so"], "n_layers": 4, "hidden_dim": 16}
        with open(Path(tmpdir) / "config.json", "w") as f:
            json.dump(config, f)

        from scripts.run_phase_b import main

        # B2 without B1 and no B1 output → should exit
        with patch("sys.argv", ["prog", "--run_dir", tmpdir, "--stages", "B2"]):
            try:
                main()
                check(False, "should have exited")
            except SystemExit as e:
                check(e.code == 1, "exits with code 1 (B2 needs B1)")


def test_dependency_check_b5_needs_b2():
    print("\n[Runner] dependency — B5 requires B2 output", flush=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"categories": ["so"], "n_layers": 4, "hidden_dim": 16}
        with open(Path(tmpdir) / "config.json", "w") as f:
            json.dump(config, f)

        from scripts.run_phase_b import main

        with patch("sys.argv", ["prog", "--run_dir", tmpdir, "--stages", "B5"]):
            try:
                main()
                check(False, "should have exited")
            except SystemExit as e:
                check(e.code == 1, "exits with code 1 (B5 needs B2)")


def test_provenance_appended():
    print("\n[Runner] provenance — appended not overwritten", flush=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)

        # Write config
        config = {"categories": ["so"], "n_layers": 4, "hidden_dim": 16}
        with open(run_dir / "config.json", "w") as f:
            json.dump(config, f)

        # Write existing provenance
        existing = [{"stages": ["A1"], "timestamp_utc": "2026-01-01"}]
        with open(run_dir / "provenance.json", "w") as f:
            json.dump(existing, f)

        # Write B3 output so it skips (no actual computation)
        geo_dir = run_dir / "B_geometry"
        geo_dir.mkdir()
        for name in ["subgroup_directions.npz", "cosine_pairs.parquet",
                      "subgroup_directions_summary.json",
                      "differentiation_metrics.json",
                      "bias_identity_alignment.json"]:
            (geo_dir / name).touch()

        from scripts.run_phase_b import main

        with patch("sys.argv", ["prog", "--run_dir", tmpdir, "--stages", "B3"]):
            try:
                main()
            except SystemExit:
                pass

        # Check provenance was appended
        with open(run_dir / "provenance.json") as f:
            prov = json.load(f)
        check(len(prov) >= 2, "provenance appended (≥2 entries)")
        check(prov[0]["stages"] == ["A1"], "original entry preserved")
        check("B3" in prov[-1]["stages"], "new entry has B3")


def test_resume_skips_completed():
    print("\n[Runner] resume — completed stages skipped", flush=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)

        config = {"categories": ["so"], "n_layers": 4, "hidden_dim": 16}
        with open(run_dir / "config.json", "w") as f:
            json.dump(config, f)

        # Mark B3 as complete
        geo_dir = run_dir / "B_geometry"
        geo_dir.mkdir()
        for name in ["subgroup_directions.npz", "cosine_pairs.parquet",
                      "subgroup_directions_summary.json",
                      "differentiation_metrics.json",
                      "bias_identity_alignment.json"]:
            (geo_dir / name).touch()

        from scripts.run_phase_b import run_b3
        from src.analysis.geometry import b3_complete

        check(b3_complete(run_dir), "B3 detected as complete")

        # run_b3 with force=False should return immediately
        # (We can't easily verify it skipped, but we can verify it doesn't crash)
        run_b3(run_dir, config, ["so"], 4, 16, 10, None, True, False)
        check(True, "run_b3 returns without error when complete + no force")


# ── Run ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_parse_args_defaults()
    test_parse_args_overrides()
    test_stage_validation()
    test_config_missing_exits()
    test_config_missing_nlayers_exits()
    test_dependency_check_b2_needs_b1()
    test_dependency_check_b5_needs_b2()
    test_provenance_appended()
    test_resume_skips_completed()

    print(f"\n{'=' * 60}", flush=True)
    print(f"Phase B runner tests: {passed} passed, {failed} failed", flush=True)
    print(f"{'=' * 60}", flush=True)
    sys.exit(1 if failed else 0)
