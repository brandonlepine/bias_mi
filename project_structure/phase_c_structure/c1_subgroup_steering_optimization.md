# C1: Subgroup-Specific Steering Optimization — Full Implementation Specification

## Purpose

For each subgroup with significant pro-bias features from B1/B2, find the optimal single-hook steering configuration (k features, target vector norm τ) that maximizes debiasing efficiency η = RCR_τ / ‖v‖. Produces per-subgroup steering vectors that C2 and C3 consume. Also runs exacerbation tests (positive-direction steering) to characterize asymmetry between debiasing and bias-amplification.

## Invocation

```bash
python scripts/C1_steering.py --run_dir runs/llama-3.1-8b_2026-04-15/

# Quick test (limited items per subgroup)
python scripts/C1_steering.py --run_dir runs/llama-3.1-8b_2026-04-15/ --max_items 20

# Specific categories only
python scripts/C1_steering.py --run_dir runs/llama-3.1-8b_2026-04-15/ --categories so,disability

# Specific subgroups (format: cat/sub)
python scripts/C1_steering.py --run_dir runs/llama-3.1-8b_2026-04-15/ --subgroups so/gay,race/black

# Override minimum item count
python scripts/C1_steering.py --run_dir runs/llama-3.1-8b_2026-04-15/ --min_n_per_group 15

# Skip exacerbation tests
python scripts/C1_steering.py --run_dir runs/llama-3.1-8b_2026-04-15/ --skip_exacerbation

# Override target norm grid
python scripts/C1_steering.py --run_dir runs/llama-3.1-8b_2026-04-15/ --target_norms -0.5,-1,-2,-5,-10,-20,-40
```

Reads `model_path`, `sae_source`, `sae_expansion`, `device`, `dtype` from config.json.

---

## Input

- Ranked features: `{run}/B_feature_ranking/ranked_features.parquet`
- Injection layers: `{run}/B_feature_ranking/injection_layers.json`
- Artifact flags: `{run}/B_feature_interpretability/artifact_flags.json`
- Metadata: `{run}/A_extraction/metadata.parquet`
- Stimuli: `{run}/A_extraction/stimuli/{category}.json`
- Model + SAE checkpoints (via `config.json`)

## Dependencies

- `torch` (model, SAEs, hooks)
- `pandas`, `pyarrow`
- `numpy`
- `matplotlib`
- Shared modules ported from old repo:
  - `src/metrics/bias_metrics.py` — `compute_all_metrics`, `build_result_dict`
  - `src/sae_localization/steering.py` — `SAESteerer`
  - `src/sae_localization/sae_wrapper.py` — `SAEWrapper`
  - `src/models/wrapper.py` — `ModelWrapper`

---

## Steering Convention

**Single-hook steering.** One vector injected at B2's pre-computed injection layer.

**Sign convention:**
- `α < 0` → pushes model AWAY from stereotyped response (debiasing)
- `α > 0` → pushes model TOWARD stereotyped response (exacerbation)

This follows from: pro-bias features have positive cohen's d (fire more on stereotyped responses). Their decoder columns are directions the residual stream moves when the features fire. Subtracting them (α < 0) should reduce bias; adding them (α > 0) should amplify bias.

**Vector construction:**
```python
def build_subgroup_steering_vector(
    top_k_features: list[dict],
    sae_cache: dict[int, SAEWrapper],
    alpha: float,
    device, dtype,
) -> tuple[torch.Tensor, float]:
    """
    Construct single steering vector as α × mean of unit-normalized decoder columns.
    
    Features may come from different layers; their decoder columns are all in the same
    ambient space (ℝ^hidden_dim) so averaging is well-defined. The resulting vector is
    injected at B2's designated injection_layer (not the features' source layers).
    
    Returns:
        vec: torch.Tensor of shape (hidden_dim,)
        mean_direction_norm: float, ||mean_of_unit_directions|| — the scalar needed to
                             convert target_norm → alpha
    """
    directions = []
    for f in top_k_features:
        sae = sae_cache[f["layer"]]
        d = sae.get_feature_direction(f["feature_idx"])  # unit-normalized (hidden_dim,)
        directions.append(torch.from_numpy(d).float())
    
    stacked = torch.stack(directions, dim=0)  # (k, hidden_dim)
    mean_dir = stacked.mean(dim=0)             # (hidden_dim,)
    mean_dir_norm = float(mean_dir.norm().item())
    
    vec = (alpha * mean_dir).to(dtype=dtype, device=device)
    return vec, mean_dir_norm
```

**Target-norm to alpha conversion:**
```python
def alpha_for_target_norm(target_norm: float, mean_dir_norm: float) -> float:
    """Compute alpha so that ||alpha * mean_dir|| == |target_norm|, sign preserved."""
    # ||alpha * mean_dir|| = |alpha| * mean_dir_norm = |target_norm|
    # sign(vec) follows sign(alpha) = sign(target_norm)
    if mean_dir_norm < 1e-8:
        return 0.0
    return target_norm / mean_dir_norm
```

---

## Steering Efficiency Metric

**Primary optimization metric: η**

```python
η = RCR_τ=1.0 / ‖v‖_2
```

Where:
- RCR_τ=1.0 is Robust Correction Rate at margin threshold 1.0 (fraction of stereotyped-response items with baseline margin ≥ 1.0 that are corrected by steering)
- ‖v‖_2 is the L2 norm of the steering vector as applied

**Safety constraints** (configurations failing either are excluded from selection):
- `degeneration_rate < 0.05`
- `corruption_rate < 0.05`

Degeneration is defined as the model's output distribution collapsing (see below). Corruption is when a non-stereotyped response becomes stereotyped under steering — the opposite of what we want.

**Selection:** Maximum η among safe configurations. Tie-breaking within 1% of best η: prefer smaller ‖v‖.

**Reported at optimum:** η, RCR_τ=1.0, RCR_τ=0.5, RCR_τ=2.0, MWCS, logit_shift (per margin bin), degeneration rate, corruption rate, ‖v‖.

---

## Degeneration Definition

An item's result has `degenerated = True` if ANY of:
1. Max logit across {A, B, C} < -10.0 (model assigns extremely low probability to all answer options)
2. Entropy of softmax over {A, B, C} > 1.05 (near-uniform; log(3) ≈ 1.099)
3. Any of the {A, B, C} logits is NaN or ±inf

Implemented in `src/metrics/bias_metrics.py::build_result_dict`:

```python
def is_degenerated(logits_steered: dict) -> bool:
    values = [logits_steered.get(k, float('-inf')) for k in ["A", "B", "C"]]
    arr = np.array(values, dtype=np.float64)
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        return True
    if arr.max() < -10.0:
        return True
    # Softmax entropy
    shifted = arr - arr.max()
    exp = np.exp(shifted)
    probs = exp / exp.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    return entropy > 1.05
```

---

## Script Structure

```python
def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = load_config(run_dir)
    
    device = torch.device(config["device"])
    dtype = getattr(torch, config["dtype"])
    
    # Load model and SAE framework
    wrapper = ModelWrapper(config["model_path"], device=device, dtype=dtype)
    
    # Load B2 outputs
    ranked_features_df = pd.read_parquet(run_dir / "B_feature_ranking" / "ranked_features.parquet")
    with open(run_dir / "B_feature_ranking" / "injection_layers.json") as f:
        injection_layers = json.load(f)
    
    # Load B5 artifact flags
    artifact_flags_path = run_dir / "B_feature_interpretability" / "artifact_flags.json"
    flagged_set = set()
    if artifact_flags_path.exists():
        with open(artifact_flags_path) as f:
            flags_data = json.load(f)
        for entry in flags_data["flagged_features"]:
            flagged_set.add((entry["feature_idx"], entry["layer"]))
        log(f"Loaded {len(flagged_set)} flagged feature (idx, layer) pairs for exclusion")
    
    # Filter features
    ranked_features_df = filter_flagged_features(ranked_features_df, flagged_set)
    
    # Load metadata
    metadata_df = load_metadata(run_dir)
    
    # Determine subgroups to process
    subgroups_to_process = determine_subgroups(
        ranked_features_df, injection_layers,
        filter_categories=args.categories, filter_subgroups=args.subgroups,
    )
    
    log(f"Will process {len(subgroups_to_process)} subgroups")
    
    # Pre-identify all SAE layers we'll need, load them once
    needed_layers = identify_needed_layers(ranked_features_df, subgroups_to_process, max_k=55)
    sae_cache = {}
    for layer in sorted(needed_layers):
        log(f"  Loading SAE for layer {layer}...")
        sae_cache[layer] = SAEWrapper(
            config["sae_source"],
            layer=layer,
            expansion=config["sae_expansion"],
            device=device,
            dtype=dtype,
        )
    
    # Output directory structure
    output_dir = run_dir / "C_steering"
    ensure_dir(output_dir / "checkpoints")
    ensure_dir(output_dir / "vectors")
    ensure_dir(output_dir / "per_item")
    ensure_dir(output_dir / "figures")
    
    # Process each subgroup
    all_manifests = []
    all_phase1_results = {}
    all_grid_records = []
    
    for (cat, sub) in subgroups_to_process:
        manifest = process_subgroup(
            cat=cat, sub=sub,
            ranked_df=ranked_features_df,
            injection_layers=injection_layers,
            metadata_df=metadata_df,
            wrapper=wrapper,
            sae_cache=sae_cache,
            run_dir=run_dir,
            output_dir=output_dir,
            config=config,
            args=args,
        )
        
        all_manifests.append(manifest)
        if "phase1_results" in manifest:
            all_phase1_results[f"{cat}/{sub}"] = manifest["phase1_results"]
        if "phase2_grid" in manifest:
            all_grid_records.extend(manifest["phase2_grid"])
    
    # Save top-level outputs
    save_outputs(output_dir, all_manifests, all_phase1_results, all_grid_records)
    
    if not args.skip_figures:
        generate_figures(output_dir, all_manifests, all_grid_records)
```

---

## Per-Subgroup Processing

```python
def process_subgroup(cat, sub, ranked_df, injection_layers, metadata_df,
                    wrapper, sae_cache, run_dir, output_dir, config, args) -> dict:
    """
    Run full C1 pipeline for one subgroup.
    
    Returns manifest dict.
    """
    sub_key = f"{cat}/{sub}"
    log(f"\n{'='*60}")
    log(f"Subgroup: {sub_key}")
    log(f"{'='*60}")
    
    # Initialize manifest
    manifest = {
        "subgroup": sub,
        "category": cat,
        "steering_viable": False,
        "steering_skip_reason": None,
    }
    
    # Step 0: Get injection layer from B2
    inj_entry = injection_layers.get(sub_key, {})
    pro_bias_entry = inj_entry.get("pro_bias") if inj_entry else None
    
    if pro_bias_entry is None:
        manifest["steering_skip_reason"] = "no_significant_features"
        log(f"  SKIP: no significant pro-bias features")
        return manifest
    
    injection_layer = pro_bias_entry["injection_layer"]
    manifest["injection_layer"] = injection_layer
    
    # Step 0b: Get top pro-bias features (after artifact filter)
    pro_bias_features = get_ranked_pro_bias_features(ranked_df, cat, sub)
    
    if len(pro_bias_features) == 0:
        manifest["steering_skip_reason"] = "no_features_after_filter"
        log(f"  SKIP: no features remaining after artifact filter")
        return manifest
    
    log(f"  Injection layer: {injection_layer} (from B2 effect-weighted)")
    log(f"  Pro-bias features after filter: {len(pro_bias_features)}")
    
    # Step 1: Partition items
    stereo_items, non_stereo_items = partition_items(metadata_df, run_dir, cat, sub)
    
    if args.max_items:
        stereo_items = stereo_items[:args.max_items]
        non_stereo_items = non_stereo_items[:args.max_items]
    
    n_stereo = len(stereo_items)
    n_non_stereo = len(non_stereo_items)
    
    manifest["n_stereo_items"] = n_stereo
    manifest["n_non_stereo_items"] = n_non_stereo
    
    if n_stereo < args.min_n_per_group or n_non_stereo < args.min_n_per_group:
        manifest["steering_skip_reason"] = f"insufficient_items (stereo={n_stereo}, non_stereo={n_non_stereo})"
        log(f"  SKIP: insufficient items (need ≥{args.min_n_per_group} per group)")
        return manifest
    
    log(f"  n_stereo={n_stereo}, n_non_stereo={n_non_stereo}")
    
    # Build steerer for injection layer
    steerer = SAESteerer(wrapper, sae_cache[injection_layer], injection_layer)
    
    # Compute baseline results for stereo_items (used across all Phase 1/2 configs)
    baselines_stereo = compute_baselines(steerer, stereo_items)
    
    # Step 2: Phase 1 — alpha viability at k=1
    log(f"  Phase 1: target_norm sweep at k=1...")
    phase1_results = run_phase1(
        pro_bias_features, sae_cache, baselines_stereo, stereo_items,
        steerer, args.target_norms, config["dtype"], output_dir,
        cat, sub,
    )
    manifest["phase1_results"] = phase1_results
    
    viable_target_norms = identify_viable_target_norms(phase1_results)
    
    if len(viable_target_norms) < 2:
        manifest["steering_skip_reason"] = "no_viable_target_norms_in_phase1"
        log(f"  SKIP: no viable target_norms after Phase 1")
        return manifest
    
    log(f"  Phase 1: {len(viable_target_norms)}/{len(args.target_norms)} viable target_norms: {viable_target_norms}")
    
    # Step 3: Phase 2 — joint (k, target_norm) sweep
    log(f"  Phase 2: joint (k, target_norm) sweep...")
    k_steps = build_k_steps(len(pro_bias_features))
    
    phase2_grid = run_phase2(
        pro_bias_features, sae_cache, baselines_stereo, stereo_items,
        steerer, k_steps, viable_target_norms, config["dtype"],
        output_dir, cat, sub, injection_layer,
    )
    manifest["phase2_grid"] = phase2_grid
    
    # Step 4: Phase 3 — select optimal
    optimal = select_optimal(phase2_grid)
    
    if optimal is None:
        manifest["steering_viable"] = False
        manifest["steering_skip_reason"] = "no_safe_config"
        log(f"  SKIP: no config with degen<0.05 AND corrupt<0.05")
        return manifest
    
    if optimal["metrics"]["rcr_1.0"]["rcr"] <= 0.0:
        manifest["steering_viable"] = False
        manifest["steering_skip_reason"] = "rcr_zero"
        log(f"  WARNING: optimal config has RCR=0; subgroup unsteerable")
        return manifest
    
    manifest["steering_viable"] = True
    manifest["optimal_k"] = optimal["k"]
    manifest["optimal_target_norm"] = optimal["target_norm"]
    manifest["optimal_alpha"] = optimal["alpha"]
    manifest["optimal_vector_norm"] = optimal["vector_norm"]
    manifest["optimal_eta"] = optimal["eta"]
    manifest["optimal_rcr_1.0"] = optimal["metrics"]["rcr_1.0"]["rcr"]
    manifest["optimal_rcr_0.5"] = optimal["metrics"]["rcr_0.5"]["rcr"]
    manifest["optimal_rcr_2.0"] = optimal["metrics"]["rcr_2.0"]["rcr"]
    manifest["optimal_mwcs_1.0"] = optimal["metrics"]["mwcs_1.0"]["mwcs"]
    manifest["optimal_logit_shift"] = optimal["metrics"]["logit_shift"]
    manifest["optimal_degeneration_rate"] = optimal["degeneration_rate"]
    manifest["optimal_corruption_rate"] = optimal["corruption_rate"]
    manifest["optimal_features"] = pro_bias_features[:optimal["k"]]
    
    log(f"  OPTIMAL: k={optimal['k']} τ={optimal['target_norm']} α={optimal['alpha']:.2f}")
    log(f"    η={optimal['eta']:.3f} RCR₁.₀={optimal['metrics']['rcr_1.0']['rcr']:.3f} ||v||={optimal['vector_norm']:.3f}")
    log(f"    degen={optimal['degeneration_rate']:.3f} corrupt={optimal['corruption_rate']:.3f}")
    
    # Step 5: Phase 4 — marginal analysis at optimal target_norm
    manifest["marginal_analysis"] = compute_marginal_analysis(phase2_grid, optimal)
    
    # Step 6: Save optimal steering vector
    vec, mean_norm = build_subgroup_steering_vector(
        pro_bias_features[:optimal["k"]],
        sae_cache, optimal["alpha"],
        device=torch.device(config["device"]),
        dtype=getattr(torch, config["dtype"]),
    )
    save_steering_vector(output_dir, cat, sub, vec, optimal, pro_bias_features, injection_layer)
    
    # Step 7: Exacerbation test
    if not args.skip_exacerbation:
        log(f"  Step 7: exacerbation test at +|optimal_target_norm|={abs(optimal['target_norm'])}...")
        exac_manifest = run_exacerbation_test(
            optimal, pro_bias_features, sae_cache,
            steerer, stereo_items, non_stereo_items, baselines_stereo,
            config, output_dir, cat, sub,
        )
        manifest["exacerbation"] = exac_manifest
    
    # Step 8: Save consolidated per-item parquet
    save_per_item_parquet(output_dir, cat, sub, optimal, manifest)
    
    # Placeholders for C3 (filled in later)
    for key in [
        "medqa_matched_delta", "medqa_within_cat_mismatched_delta",
        "medqa_cross_cat_mismatched_delta", "medqa_nodemo_delta",
        "medqa_exacerbation_matched_delta",
        "mmlu_delta", "mmlu_worst_subject", "mmlu_worst_subject_delta",
    ]:
        manifest[key] = None
    
    return manifest
```

---

## Phase 1: Target Norm Viability at k=1

```python
DEFAULT_TARGET_NORMS = [-0.5, -1.0, -2.0, -5.0, -10.0, -20.0, -40.0, -80.0]

def run_phase1(
    pro_bias_features, sae_cache, baselines, stereo_items, steerer,
    target_norms, dtype, output_dir, cat, sub,
) -> dict:
    """
    Sweep target_norms at k=1 to identify viable magnitudes.
    
    A target_norm is viable if:
        - RCR_1.0 > 0 (some correction occurs)
        - degeneration_rate < 0.05
    
    If fewer than 2 pass, relax degeneration threshold to 0.10.
    """
    # Checkpoint path
    ckpt_dir = output_dir / "checkpoints"
    phase1_ckpt = ckpt_dir / f"{cat}_{sub}_phase1.json"
    
    if phase1_ckpt.exists():
        with open(phase1_ckpt) as f:
            return json.load(f)
    
    # Compute mean_dir_norm at k=1
    top_1 = pro_bias_features[:1]
    _, mean_norm_k1 = build_subgroup_steering_vector(
        top_1, sae_cache, alpha=1.0,
        device=steerer.device, dtype=dtype,
    )
    
    results = {}
    device = steerer.device
    
    for tn in target_norms:
        alpha = alpha_for_target_norm(tn, mean_norm_k1)
        vec, _ = build_subgroup_steering_vector(
            top_1, sae_cache, alpha,
            device=device, dtype=dtype,
        )
        vec_norm = float(vec.norm().item())
        
        per_item = []
        for item, baseline in zip(stereo_items, baselines):
            steered = steerer.steer_and_evaluate(item["prompt"], vec)
            per_item.append(build_result_dict(item, baseline, steered, vec))
        
        metrics = compute_all_metrics(per_item)
        degen = sum(1 for r in per_item if r["degenerated"]) / len(per_item)
        corrupt = sum(1 for r in per_item if r["corrupted"]) / len(per_item)
        
        results[str(tn)] = {
            "target_norm": tn,
            "alpha": alpha,
            "vector_norm": vec_norm,
            "rcr_1.0": metrics["rcr_1.0"]["rcr"],
            "degeneration_rate": degen,
            "corruption_rate": corrupt,
            "n_items": len(per_item),
        }
        
        log(f"    τ={tn:+.1f} α={alpha:+.2f} ||v||={vec_norm:.2f}: "
            f"RCR₁.₀={metrics['rcr_1.0']['rcr']:.3f} degen={degen:.3f}")
    
    # Save checkpoint
    atomic_write_json(phase1_ckpt, results)
    return results


def identify_viable_target_norms(phase1_results: dict) -> list[float]:
    """Select target_norms passing viability criteria."""
    viable = [
        float(tn) for tn, r in phase1_results.items()
        if r["rcr_1.0"] > 0 and r["degeneration_rate"] < 0.05
    ]
    
    if len(viable) < 2:
        # Relax to degen < 0.10
        viable = [
            float(tn) for tn, r in phase1_results.items()
            if r["degeneration_rate"] < 0.10
        ]
    
    return sorted(viable, reverse=True)  # ascending magnitude: [-0.5, -1, -2, ...]
```

**Phase 1 uses only negative target_norms** (debiasing direction). Positive target_norms would produce corruption (opposite of what we want) and would be filtered out immediately. The exacerbation test (Step 7) handles positive direction separately.

---

## Phase 2: Joint (k, Target Norm) Sweep

```python
BASE_K_STEPS = [1, 2, 3, 5, 8, 13, 21, 34, 55]

def build_k_steps(n_features: int) -> list[int]:
    """Extract k values from base list that don't exceed n_features."""
    return [k for k in BASE_K_STEPS if k <= n_features]


def run_phase2(
    pro_bias_features, sae_cache, baselines, stereo_items, steerer,
    k_steps, viable_target_norms, dtype, output_dir, cat, sub,
    injection_layer,
) -> list[dict]:
    """
    Full (k, target_norm) sweep with per-config checkpointing.
    """
    ckpt_dir = output_dir / "checkpoints"
    grid = []
    device = steerer.device
    
    for k in k_steps:
        # Compute mean_dir_norm for this k (depends on features, so changes with k)
        top_k = pro_bias_features[:k]
        _, mean_norm_k = build_subgroup_steering_vector(
            top_k, sae_cache, alpha=1.0,
            device=device, dtype=dtype,
        )
        
        for tn in viable_target_norms:
            ckpt_name = config_ckpt_name(cat, sub, k, tn)
            ckpt_path = ckpt_dir / ckpt_name
            
            if ckpt_path.exists():
                with open(ckpt_path) as f:
                    record = json.load(f)
                grid.append(record)
                log(f"    k={k:03d} τ={tn:+.1f}: LOADED from checkpoint")
                continue
            
            alpha = alpha_for_target_norm(tn, mean_norm_k)
            vec, _ = build_subgroup_steering_vector(
                top_k, sae_cache, alpha,
                device=device, dtype=dtype,
            )
            vec_norm = float(vec.norm().item())
            
            per_item = []
            for item, baseline in zip(stereo_items, baselines):
                steered = steerer.steer_and_evaluate(item["prompt"], vec)
                per_item.append(build_result_dict(item, baseline, steered, vec))
            
            metrics = compute_all_metrics(per_item)
            degen = sum(1 for r in per_item if r["degenerated"]) / len(per_item)
            corrupt = sum(1 for r in per_item if r["corrupted"]) / len(per_item)
            
            rcr_1 = metrics["rcr_1.0"]["rcr"]
            eta = rcr_1 / max(vec_norm, 1e-8)
            
            record = {
                "category": cat,
                "subgroup": sub,
                "k": k,
                "target_norm": tn,
                "alpha": alpha,
                "injection_layer": injection_layer,
                "vector_norm": vec_norm,
                "eta": eta,
                "metrics": metrics,
                "degeneration_rate": degen,
                "corruption_rate": corrupt,
                "n_items": len(per_item),
                "per_item_results": per_item,  # kept for consolidation into per-item parquet
            }
            
            atomic_write_json(ckpt_path, record)
            grid.append(record)
            
            log(f"    k={k:03d} τ={tn:+.1f} α={alpha:+.2f}: "
                f"RCR₁.₀={rcr_1:.3f} η={eta:.3f} ||v||={vec_norm:.2f} "
                f"degen={degen:.3f} corrupt={corrupt:.3f}")
    
    return grid


def config_ckpt_name(cat: str, sub: str, k: int, target_norm: float) -> str:
    """
    Build deterministic checkpoint filename with integer-based target_norm encoding.
    
    target_norm = -1.25 → "norm-0125"
    target_norm = -10.0 → "norm-1000"
    target_norm = +5.0  → "norm+0500"
    """
    norm_int = int(round(target_norm * 100))
    return f"{cat}_{sub}_k{k:03d}_norm{norm_int:+05d}.json"
```

---

## Phase 3: Optimal Selection

```python
def select_optimal(grid: list[dict]) -> dict | None:
    """
    Select optimal configuration.
    
    Safety constraints:
        - degeneration_rate < 0.05
        - corruption_rate < 0.05
    
    Primary metric: max eta
    Tie-breaking (within 1% of max eta): smaller vector_norm, then higher eta
    """
    safe = [
        r for r in grid
        if r["degeneration_rate"] < 0.05 and r["corruption_rate"] < 0.05
    ]
    
    if not safe:
        # Relax degeneration threshold
        safe = [r for r in grid if r["degeneration_rate"] < 0.10 and r["corruption_rate"] < 0.05]
        if safe:
            log(f"    WARNING: relaxed degen<0.10 to find safe configs")
    
    if not safe:
        return None
    
    best_eta = max(r["eta"] for r in safe)
    
    # Within 1% of best
    candidates = [r for r in safe if r["eta"] >= best_eta * 0.99]
    candidates.sort(key=lambda r: (r["vector_norm"], -r["eta"]))
    
    return candidates[0]
```

---

## Phase 4: Marginal Analysis

```python
def compute_marginal_analysis(grid: list[dict], optimal: dict) -> list[dict]:
    """
    At optimal target_norm, show how RCR and ||v|| evolve with k.
    
    Returns list of records with marginal gain/cost between successive k values.
    """
    optimal_tn = optimal["target_norm"]
    relevant = [r for r in grid if r["target_norm"] == optimal_tn]
    relevant.sort(key=lambda r: r["k"])
    
    marginal = []
    for i, r in enumerate(relevant):
        entry = {
            "k": r["k"],
            "rcr_1.0": r["metrics"]["rcr_1.0"]["rcr"],
            "vector_norm": r["vector_norm"],
            "eta": r["eta"],
        }
        if i > 0:
            prev = relevant[i - 1]
            entry["marginal_rcr_gain"] = r["metrics"]["rcr_1.0"]["rcr"] - prev["metrics"]["rcr_1.0"]["rcr"]
            entry["marginal_norm_cost"] = r["vector_norm"] - prev["vector_norm"]
            denom = max(abs(entry["marginal_norm_cost"]), 1e-8)
            entry["marginal_efficiency"] = entry["marginal_rcr_gain"] / denom
        marginal.append(entry)
    
    return marginal
```

---

## Exacerbation Test (Step 7)

```python
def run_exacerbation_test(
    optimal, pro_bias_features, sae_cache,
    steerer, stereo_items, non_stereo_items, baselines_stereo,
    config, output_dir, cat, sub,
) -> dict:
    """
    At +|optimal_target_norm| (positive, exacerbation direction):
        1. Test corruption on non-stereotyped items
        2. Test amplification on stereotyped items
    """
    exac_target_norm = abs(optimal["target_norm"])  # flip sign of optimal (which was negative)
    k = optimal["k"]
    
    ckpt_path = output_dir / "checkpoints" / f"{cat}_{sub}_exac.json"
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            return json.load(f)
    
    top_k = pro_bias_features[:k]
    _, mean_norm = build_subgroup_steering_vector(
        top_k, sae_cache, alpha=1.0,
        device=steerer.device, dtype=getattr(torch, config["dtype"]),
    )
    exac_alpha = alpha_for_target_norm(exac_target_norm, mean_norm)
    vec, _ = build_subgroup_steering_vector(
        top_k, sae_cache, exac_alpha,
        device=steerer.device, dtype=getattr(torch, config["dtype"]),
    )
    vec_norm = float(vec.norm().item())
    
    # Baselines for non_stereo_items
    baselines_non_stereo = compute_baselines(steerer, non_stereo_items)
    
    # Test on non_stereo items (corruption check)
    per_item_non_stereo = []
    for item, baseline in zip(non_stereo_items, baselines_non_stereo):
        steered = steerer.steer_and_evaluate(item["prompt"], vec)
        per_item_non_stereo.append(build_result_dict(item, baseline, steered, vec))
    
    metrics_non_stereo = compute_all_metrics(per_item_non_stereo)
    corruption_rate = sum(1 for r in per_item_non_stereo if r["corrupted"]) / max(len(per_item_non_stereo), 1)
    
    # Test on stereo items (amplification check)
    per_item_stereo = []
    for item, baseline in zip(stereo_items, baselines_stereo):
        steered = steerer.steer_and_evaluate(item["prompt"], vec)
        per_item_stereo.append(build_result_dict(item, baseline, steered, vec))
    
    # Amplification: did logit for stereotyped_option INCREASE under exacerbation steering?
    stereo_logit_shifts = []
    for r in per_item_stereo:
        opt = r.get("stereotyped_option")
        if opt and opt in r["logit_baseline"] and opt in r["logit_steered"]:
            stereo_logit_shifts.append(r["logit_steered"][opt] - r["logit_baseline"][opt])
    
    mean_amplification = float(np.mean(stereo_logit_shifts)) if stereo_logit_shifts else None
    frac_amplified = (
        float(sum(1 for s in stereo_logit_shifts if s > 0) / len(stereo_logit_shifts))
        if stereo_logit_shifts else None
    )
    
    result = {
        "exac_target_norm": exac_target_norm,
        "exac_alpha": exac_alpha,
        "vector_norm": vec_norm,
        "n_non_stereo_items": len(non_stereo_items),
        "n_stereo_items": len(stereo_items),
        "corruption_rate_non_stereo": corruption_rate,
        "metrics_non_stereo": metrics_non_stereo,
        "mean_logit_amplification_stereo": mean_amplification,
        "fraction_amplified_stereo": frac_amplified,
        "per_item_non_stereo": per_item_non_stereo,  # for consolidation
        "per_item_stereo": per_item_stereo,
    }
    
    atomic_write_json(ckpt_path, result)
    return result
```

---

## Output Files

### `vectors/{cat}_{sub}.npz`

Per-subgroup steering vector and metadata. Self-contained for C2/C3 consumption.

```python
np.savez(
    output_dir / "vectors" / f"{cat}_{sub}.npz",
    vector=vec.float().cpu().numpy(),       # (hidden_dim,) float32
    injection_layer=np.int32(injection_layer),
    target_norm=np.float32(optimal_target_norm),
    alpha=np.float32(optimal_alpha),
    k=np.int32(optimal_k),
    vector_norm=np.float32(optimal_vector_norm),
    eta=np.float32(optimal_eta),
    rcr_at_optimal=np.float32(optimal_rcr_1_0),
    feature_idxs=np.array([f["feature_idx"] for f in optimal_features], dtype=np.int32),
    feature_layers=np.array([f["layer"] for f in optimal_features], dtype=np.int32),
    category=cat,
    subgroup=sub,
)
```

### `per_item/{cat}_{sub}.parquet`

Consolidated per-item results across all conditions.

Columns:
- `item_idx`, `category`, `subgroup`, `condition` (string)
- `k`, `target_norm`, `alpha`, `injection_layer`, `vector_norm`
- `baseline_answer`, `steered_answer`, `baseline_role`, `steered_role`
- `corrected` (bool), `corrupted` (bool), `degenerated` (bool)
- `margin` (float), `margin_bin` (string)
- `logit_baseline_A/B/C`, `logit_steered_A/B/C` (individual columns)
- `stereotyped_option`

**Condition values:**
- `debiasing_optimal` — at (k*, τ*) on stereo_items
- `exacerbation_optimal` — at (k*, +|τ*|) on stereo_items
- `exacerbation_on_non_stereo` — at (k*, +|τ*|) on non_stereo_items

### `steering_manifests.json`

List of per-subgroup manifest dicts. One dict per subgroup processed.

```json
[
  {
    "subgroup": "gay",
    "category": "so",
    "steering_viable": true,
    "injection_layer": 18,
    "n_stereo_items": 487,
    "n_non_stereo_items": 593,
    "optimal_k": 5,
    "optimal_target_norm": -2.0,
    "optimal_alpha": -3.42,
    "optimal_vector_norm": 2.0,
    "optimal_eta": 0.264,
    "optimal_rcr_1.0": 0.528,
    "optimal_rcr_0.5": 0.612,
    "optimal_rcr_2.0": 0.341,
    "optimal_mwcs_1.0": 0.402,
    "optimal_logit_shift": {...},
    "optimal_degeneration_rate": 0.012,
    "optimal_corruption_rate": 0.023,
    "optimal_features": [...],
    "marginal_analysis": [...],
    "exacerbation": {...},
    "medqa_matched_delta": null,
    ...
  },
  {
    "subgroup": "pansexual",
    "category": "so",
    "steering_viable": false,
    "steering_skip_reason": "insufficient_items (stereo=7, non_stereo=45)"
  }
]
```

### `phase1_results.json`

Per-subgroup Phase 1 target_norm sweep at k=1.

```json
{
  "so/gay": {
    "-0.5": {"target_norm": -0.5, "alpha": -0.78, "vector_norm": 0.5, "rcr_1.0": 0.12, ...},
    "-1.0": {...},
    ...
  },
  ...
}
```

### `phase2_grid.parquet`

All Phase 2 configuration records flattened. One row per (category, subgroup, k, target_norm).

Columns: `category`, `subgroup`, `k`, `target_norm`, `alpha`, `injection_layer`, `vector_norm`, `eta`, `rcr_0.5`, `rcr_1.0`, `rcr_2.0`, `mwcs_1.0`, `mean_logit_shift`, `degeneration_rate`, `corruption_rate`, `n_items`.

### `marginal_analysis.json`

Per-subgroup marginal k-by-k RCR gain and vector-norm cost at optimal target_norm.

### `c1_summary.json`

Top-level run summary.

```json
{
  "config": {...},
  "n_subgroups_processed": 40,
  "n_subgroups_viable": 32,
  "n_subgroups_skipped": 8,
  "skip_reasons": {
    "no_significant_features": 2,
    "insufficient_items": 4,
    "no_viable_target_norms_in_phase1": 1,
    "no_safe_config": 1,
    "rcr_zero": 0
  },
  "optimal_k_distribution": {"1": 3, "2": 5, "3": 8, "5": 12, "8": 4, "13": 0, "21": 0},
  "median_optimal_k": 5,
  "median_optimal_eta": 0.183,
  "median_optimal_rcr_1.0": 0.42,
  "runtime_seconds": 27843.0
}
```

---

## Figures

### `fig_pareto_frontier_{cat}.png`

One subplot per subgroup in this category.
- X: ‖v‖_2 (intervention strength)
- Y: RCR_1.0 (debiasing benefit)
- Points: all (k, τ) configurations from Phase 2 grid
- Color: k (viridis ordinal colormap)
- Marker: distinct per target_norm
- Star: selected optimum (large, gold)
- Thin gray lines connecting points of constant α
- Gray region: degen > 0.05 or corrupt > 0.05 (shaded unsafe zone)
- Annotation at optimum: `η*={value:.3f}`, `k={k}`, `τ={tn:+.1f}`
- Title: subgroup name

### `fig_stepwise_correction_{cat}.png`

One subplot per subgroup.
- X: k
- Y (primary): RCR_1.0 at optimal τ (blue #0072B2, circle markers)
- Y (secondary): corruption rate at optimal τ (vermillion #D55E00, dashed, square markers)
- n count labels above each RCR point
- Vertical dashed line at selected k*
- Legend: RCR, corruption rate
- Title: subgroup name

### `fig_marginal_analysis_{cat}.png`

One subplot per subgroup. At selected τ*:
- X: k
- Three lines:
  - RCR_1.0(k) in blue (circle markers)
  - ‖v‖_2(k) in orange (square markers, secondary axis)
  - Marginal efficiency (marginal_rcr_gain / marginal_norm_cost) in green (diamond markers, tertiary axis)
- Vertical dashed line at selected k*
- Horizontal dashed line at y=0 for marginal_efficiency
- Legend
- Title: subgroup name

### `fig_optimal_k_distribution.png`

Single histogram across all subgroups with `steering_viable=True`.
- X: optimal k (log-scale or discrete bins matching k_steps)
- Y: count
- Bars colored by category (Wong palette)
- Median k annotated with dashed line
- Title: "Distribution of optimal k across subgroups"

### `fig_alpha_vs_k_heatmaps_{cat}.png`

One subplot per subgroup.
- Heatmap of η values across (k, τ) grid
- X: target_norm (log-scale)
- Y: k (log-scale)
- Color: η (YlGnBu sequential colormap)
- Star marker on selected optimum
- Cells where degen > 0.05 or corrupt > 0.05 shown in gray
- Cells not evaluated (outside viable_target_norms ∪ k_steps) shown in white
- Title: subgroup name

### `fig_margin_conditioned_{cat}.png`

One subplot per subgroup.
- Three grouped bars (near_indifferent, moderate, confident)
- Each bar: RCR at that margin bin at selected optimum
- Wong palette (orange for near_indifferent, blue for moderate, green for confident)
- N counts annotated per bar
- Horizontal line at overall RCR_1.0 for reference
- Title: subgroup name

### `fig_exacerbation_asymmetry.png`

Single figure across all viable subgroups.
- Paired bars per subgroup: left = RCR_1.0 (debiasing), right = corruption_rate (exacerbation)
- Blue #0072B2 (debiasing), vermillion #D55E00 (exacerbation)
- Grouped by category with separation lines
- N counts annotated
- Title: "Debiasing vs. exacerbation effects — all viable subgroups"

---

## Output Structure

```
{run}/C_steering/
├── checkpoints/
│   ├── {cat}_{sub}_phase1.json                 # Per-subgroup Phase 1 results
│   ├── {cat}_{sub}_k{k:03d}_norm{tn_int:+05d}.json  # Per-config Phase 2 results
│   ├── {cat}_{sub}_exac.json                   # Exacerbation test
│   └── ...
├── vectors/
│   ├── {cat}_{sub}.npz                         # Optimal steering vector + metadata
│   └── ...
├── per_item/
│   ├── {cat}_{sub}.parquet                     # Consolidated per-item results
│   └── ...
├── steering_manifests.json                      # List of all subgroup manifests
├── phase1_results.json                          # All Phase 1 sweeps
├── phase2_grid.parquet                          # All Phase 2 configurations, flat
├── marginal_analysis.json                       # Per-subgroup k-vs-RCR data
├── c1_summary.json                              # Top-level summary
└── figures/
    ├── fig_pareto_frontier_{cat}.png/.pdf
    ├── fig_stepwise_correction_{cat}.png/.pdf
    ├── fig_marginal_analysis_{cat}.png/.pdf
    ├── fig_optimal_k_distribution.png/.pdf
    ├── fig_alpha_vs_k_heatmaps_{cat}.png/.pdf
    ├── fig_margin_conditioned_{cat}.png/.pdf
    └── fig_exacerbation_asymmetry.png/.pdf
```

---

## Resume Safety

**Per-configuration checkpoints.** Before evaluating any (subgroup, k, target_norm) or Phase 1 sweep or exacerbation test, check if its checkpoint JSON exists. If yes, load and skip. If no, compute and save atomically.

**Checkpoint granularity:**
- Phase 1: one JSON per subgroup (`{cat}_{sub}_phase1.json`)
- Phase 2: one JSON per (subgroup, k, target_norm) — typically 7-9 k × 4-8 τ = 28-72 per subgroup
- Exacerbation: one JSON per subgroup (`{cat}_{sub}_exac.json`)

**Atomic writes.** Every checkpoint and final output uses tmp-then-rename to prevent corruption on crash.

**SAE caching.** SAEs are loaded once per run (not per subgroup) and kept in `sae_cache`. Restarting the script reloads SAEs but doesn't reprocess completed configurations.

**Handling config changes.** If `target_norms` or `k_steps` change between runs, old checkpoints from the previous config are still valid — they represent completed evaluations of specific (k, τ) points. New configs just add to the grid.

---

## Compute Estimate (M4 Max MPS)

- Baseline per item: ~0.2s
- Steered per item: ~0.3s (extra hook overhead)
- Per (k, τ) config: ~100 items × 0.3s = ~30s
- Phase 1 per subgroup: 8 τ × 30s = ~4 min
- Phase 2 per subgroup: 9 k × 6 τ × 30s = ~27 min
- Exacerbation per subgroup: ~3 min
- Total per subgroup: ~35 min
- 40 viable subgroups × 35 min = **~23 hours on MPS**

On RunPod CUDA (A100): ~6-8 hours (≈3x speedup).

**Mitigation with --max_items 50:** Total drops to ~12 hours on MPS, ~4 hours on CUDA.

---

## Assumptions Summary

| # | Decision | Value |
|---|---|---|
| 1 | Steering type | Single-hook |
| 2 | Injection layer source | B2's effect-weighted layer from `injection_layers.json` (authoritative) |
| 3 | Sign convention | α < 0 debias, α > 0 exacerbate |
| 4 | Parameterization | Signed target vector norms, converted to α per-k |
| 5 | Phase 1 target_norms | [-0.5, -1, -2, -5, -10, -20, -40, -80] (negative only) |
| 6 | k_steps | [1, 2, 3, 5, 8, 13, 21, 34, 55] capped at n_features |
| 7 | Viability criteria | RCR_1.0 > 0 AND degen < 0.05 (Phase 1); degen < 0.05 AND corrupt < 0.05 (Phase 3) |
| 8 | Primary metric | η = RCR_1.0 / ‖v‖ |
| 9 | Tie-breaking | Smaller ‖v‖ within 1% of best η, then larger η |
| 10 | Degeneration definition | max logit < -10 OR entropy > 1.05 OR NaN/inf |
| 11 | Minimum items per group | 10 (configurable) |
| 12 | Artifact filtering | Remove (feature_idx, layer) pairs flagged in B5 |
| 13 | Exacerbation direction | Positive τ of equal magnitude to optimal |
| 14 | Checkpoint naming | Integer-based: `norm{norm_int:+05d}.json` |
| 15 | Per-item output | Parquet with `condition` column distinguishing cases |
| 16 | Steering viability flag | Explicit `steering_viable` bool + `steering_skip_reason` |
| 17 | Vector storage | npz per subgroup with all metadata for C2/C3 consumption |

---

## Test Commands

```bash
# Quick smoke test: single subgroup, 20 items per group
python scripts/C1_steering.py \
    --run_dir runs/llama-3.1-8b_2026-04-15/ \
    --subgroups so/gay \
    --max_items 20

# Single category
python scripts/C1_steering.py \
    --run_dir runs/llama-3.1-8b_2026-04-15/ \
    --categories so

# Full run
python scripts/C1_steering.py --run_dir runs/llama-3.1-8b_2026-04-15/ 2>&1 | tee logs/C1_steering.log

# Verify outputs
python -c "
import json
import pandas as pd
import numpy as np

with open('runs/llama-3.1-8b_2026-04-15/C_steering/steering_manifests.json') as f:
    manifests = json.load(f)

viable = [m for m in manifests if m['steering_viable']]
skipped = [m for m in manifests if not m['steering_viable']]

print(f'Viable: {len(viable)}/{len(manifests)}')
print(f'Skipped: {len(skipped)}')
for m in skipped:
    print(f'  {m[\"category\"]}/{m[\"subgroup\"]}: {m[\"steering_skip_reason\"]}')

print(f'\\nMedian optimal k: {np.median([m[\"optimal_k\"] for m in viable])}')
print(f'Median η: {np.median([m[\"optimal_eta\"] for m in viable]):.3f}')
print(f'Median RCR_1.0: {np.median([m[\"optimal_rcr_1.0\"] for m in viable]):.3f}')

# Load a vector
import os
vec_files = os.listdir('runs/llama-3.1-8b_2026-04-15/C_steering/vectors')
if vec_files:
    data = np.load(f'runs/llama-3.1-8b_2026-04-15/C_steering/vectors/{vec_files[0]}')
    print(f'\\nSample vector {vec_files[0]}:')
    print(f'  shape={data[\"vector\"].shape}, norm={np.linalg.norm(data[\"vector\"]):.3f}')
    print(f'  injection_layer={data[\"injection_layer\"]}, k={data[\"k\"]}, α={data[\"alpha\"]:+.2f}')

# Exacerbation summary
if viable and 'exacerbation' in viable[0]:
    exac_corrupts = [m['exacerbation']['corruption_rate_non_stereo'] for m in viable if m.get('exacerbation')]
    print(f'\\nMean exacerbation corruption rate on non-stereo items: {np.mean(exac_corrupts):.3f}')
"
```