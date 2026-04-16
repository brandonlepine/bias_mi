# Phase C: Steering & Evaluation Pipeline — Detailed Specification

## Overview

Phase C requires the model loaded on device. It reads Phase B outputs (ranked features, directions, manifests) and produces causal evidence: do the features identified in Phase B actually influence model behavior when steered? How do interventions transfer across subgroups? How do they affect performance on out-of-distribution tasks?

**Stages:**
- **C1** — Subgroup-specific steering optimization (joint k,α sweep per subgroup on BBQ)
- **C2** — Cross-subgroup transfer & universal backfire prediction
- **C3** — Generalization evaluation (MedQA + MMLU with saved steering vectors)
- **C4** — Token-level feature interpretability (per-token SAE activations via model hooks)

**Dependencies:**
```
B2 (ranked features) ──→ C1 (steering optimization)
B3 (cosine geometry) ──→ C2 (universal backfire — needs C1 vectors + B3 cosines)
C1 (steering vectors) ──→ C3 (generalization)
B2 + C1 ────────────────→ C4 (token-level — needs optimal features + model)
```

C1 must run first. C2, C3, C4 can run in any order after C1 (all need the model but read different Phase B outputs).

**Resource requirements:**
- C1: Model + SAE loaded. ~50 (k,α) configurations per subgroup × ~40 subgroups × ~100 items per subgroup × 2 forward passes (baseline + steered) = ~400K forward passes. At ~0.3s each on MPS = ~33 hours. BUT: Phase 1 alpha pruning + resume safety means this can be chunked across sessions. On CUDA (RunPod): ~8 hours.
- C2: Model + SAE. ~40 subgroups × ~3 other subgroups × ~100 items × 2 passes = ~24K forward passes. ~2 hours MPS, ~30 min CUDA.
- C3: Model + SAE. ~40 vectors × ~1000 MedQA items × 2 directions × 2 passes = ~160K forward passes. ~13 hours MPS, ~3 hours CUDA. Cap with --max_items if needed.
- C4: Model + SAE with per-token hooks. ~40 subgroups × 3 features × ~100 items × 1 pass = ~12K passes (but each captures all tokens, slightly slower). ~1.5 hours MPS.

---

## Shared Infrastructure: Confidence-Aware Metrics

All steering evaluations in Phase C use the metrics module created in Parts 0-3:

```
src/metrics/bias_metrics.py
```

Every item evaluated during steering produces a result dict:

```python
{
    "item_idx": 42,
    "baseline_answer": "A",
    "steered_answer": "C",
    "baseline_role": "stereotyped_target",
    "steered_role": "unknown",
    "corrected": True,          # was stereotyped → now non-stereo/unknown
    "corrupted": False,         # was non-stereo → now stereotyped
    "margin": 1.73,             # baseline logit margin (pre-steering)
    "margin_bin": "moderate",   # near_indifferent / moderate / confident
    "logit_baseline": {"A": 2.1, "B": -0.3, "C": 0.37},
    "logit_steered": {"A": 0.8, "B": -0.1, "C": 1.9},
    "stereotyped_option": "A",
    "degenerated": False,
    "vector_norm": 0.342,
}
```

This dict is passed to `compute_all_metrics()` which returns:

```python
{
    "rcr_0.5": {"rcr": 0.52, "n_eligible": 45, "n_corrected": 23, "tau": 0.5},
    "rcr_1.0": {"rcr": 0.42, "n_eligible": 31, "n_corrected": 13, "tau": 1.0},
    "rcr_2.0": {"rcr": 0.18, "n_eligible": 11, "n_corrected": 2, "tau": 2.0},
    "mwcs_1.0": {"mwcs": 0.39, "tau": 1.0},
    "logit_shift": {
        "mean_shift": -1.34, "std_shift": 0.82, "median_shift": -1.12, "n": 72,
        "per_margin_bin": {
            "near_indifferent": {"mean_shift": -0.42, "n": 23},
            "moderate": {"mean_shift": -1.87, "n": 31},
            "confident": {"mean_shift": -0.91, "n": 18},
        }
    },
    "raw_correction_rate": 0.59,
    "raw_corruption_rate": 0.03,
    "n_items": 72,
}
```

The three metrics:
- **RCR (Robust Correction Rate):** Only counts corrections on items where baseline margin ≥ τ. Filters out "cheap" flips on near-indifferent items.
- **MWCS (Margin-Weighted Correction Score):** Soft weighting via sigmoid — low-margin corrections count less.
- **Logit Shift:** Continuous measure of how much the stereotyped option's logit moved. Negative = good (model moved away from stereotype).

**RCR at τ=1.0 is the primary metric** used for steering efficiency optimization.

---

## Shared Infrastructure: Steering Mechanics

All steering in Phase C follows the same pattern:

```python
from src.sae_localization.steering import SAESteerer

# Build steerer for a specific layer
steerer = SAESteerer(wrapper, sae, injection_layer)

# For each item:
prompt = item["prompt"]
baseline = steerer.evaluate_baseline(prompt)       # Forward pass, no hooks
result = steerer.steer_and_evaluate(prompt, vec)   # Forward pass with steering hook

# The hook adds vec to the residual stream at the injection layer, last-token position
```

The steering vector `vec` is pre-computed (from `build_subgroup_steering_vector` in C1, loaded from .npz in C2/C3). It's a tensor of shape `(hidden_dim,)` that gets added to the hidden state.

---

## C1: Subgroup-Specific Steering Optimization

### Purpose

For each subgroup, find the optimal steering configuration (k features, α coefficient) that maximizes debiasing benefit per unit of representational perturbation. This is the joint optimization over k and α using steering efficiency η.

### Invocation

```bash
python scripts/C1_steering.py --run_dir runs/llama-3.1-8b_2026-04-15/

# Quick test
python scripts/C1_steering.py --run_dir runs/llama-3.1-8b_2026-04-15/ --max_items 20

# Specific categories
python scripts/C1_steering.py --run_dir runs/llama-3.1-8b_2026-04-15/ --categories so,disability
```

Reads model_path, sae_source, device from config.json.

### Input

- Ranked features: `{run}/B_feature_ranking/ranked_features_by_subgroup.json`
- Injection layers: `{run}/B_feature_ranking/injection_layers.json`
- Artifact flags: `{run}/B_feature_interpretability/artifact_flags.json` (if available)
- Stimuli: `{run}/A_extraction/stimuli/{category}.json`
- Behavioral metadata: `{run}/A_extraction/activations/{category}/item_*.npz`
- Model + SAE checkpoints

### Step 0: Load and Filter Features

```python
with open(ranked_features_path) as f:
    ranked = json.load(f)

# If artifact flags exist, remove flagged features from ranked lists
if artifact_flags_path.exists():
    with open(artifact_flags_path) as f:
        flagged = json.load(f)
    flagged_set = {(f["feature_idx"], f["layer"]) for f in flagged}
    
    for cat in ranked:
        for sub in ranked[cat]:
            for direction in ["pro_bias", "anti_bias"]:
                ranked[cat][sub][direction] = [
                    f for f in ranked[cat][sub][direction]
                    if (f["feature_idx"], f["layer"]) not in flagged_set
                ]
```

This ensures features identified as template artifacts in B5 don't end up in steering vectors.

### Step 1: Load SAEs for All Needed Layers

Features may span multiple layers. Load SAEs for all layers that appear in any subgroup's ranked list:

```python
needed_layers = set()
for cat in ranked:
    for sub in ranked[cat]:
        for f in ranked[cat][sub].get("pro_bias", [])[:21]:  # max k we'll test
            needed_layers.add(f["layer"])

sae_cache = {}
for layer in sorted(needed_layers):
    sae_cache[layer] = SAEWrapper(config["sae_source"], layer=layer,
                                   expansion=config["sae_expansion"], device=device)
```

### Step 2: Prepare Items Per Subgroup

For each subgroup S, partition items into:
- **Stereotyped-response items (ambig only):** For correction testing. Items where S ∈ stereotyped_groups AND context_condition == "ambig" AND model_answer_role == "stereotyped_target".
- **Non-stereotyped-response items (ambig only):** For corruption testing. Items where S ∈ stereotyped_groups AND context_condition == "ambig" AND model_answer_role != "stereotyped_target".

```python
def partition_subgroup_items(meta_df, stimuli, subgroup):
    targeting = [
        it for it in stimuli
        if subgroup in it["stereotyped_groups"]
        and it["context_condition"] == "ambig"
    ]
    
    stereo = [it for it in targeting 
              if meta[it["item_idx"]]["model_answer_role"] == "stereotyped_target"]
    non_stereo = [it for it in targeting 
                  if meta[it["item_idx"]]["model_answer_role"] != "stereotyped_target"]
    
    return stereo, non_stereo
```

**Minimum item count:** Skip subgroups with fewer than 10 stereotyped-response items. Can't meaningfully optimize steering with fewer data points.

### Step 3: Build Steering Vectors

For a given (k, α), the steering vector is built by `build_subgroup_steering_vector`:

```python
def build_subgroup_steering_vector(feature_list, sae_cache, k, alpha, device, dtype):
    top_k = feature_list[:k]
    
    # Determine injection layer: mode of layers in top-k
    layer_counts = Counter(f["layer"] for f in top_k)
    injection_layer = max(layer_counts, key=lambda l: (layer_counts[l], l))
    
    # Collect unit-normalized decoder columns
    directions = []
    for f in top_k:
        sae = sae_cache[f["layer"]]
        d = sae.get_feature_direction(f["feature_idx"])  # unit-normalized (hidden_dim,)
        directions.append(torch.from_numpy(d))
    
    # Mean of unit directions, scaled by alpha
    mean_dir = torch.stack(directions).mean(dim=0)
    vec = alpha * mean_dir
    
    return vec.to(dtype=dtype, device=device), injection_layer
```

**Magnitude convention:** The vector has magnitude ≈ |α| × ‖mean_of_unit_vectors‖. For k unit vectors, ‖mean‖ ≈ 1/√k if the directions are roughly orthogonal, or ≈ 1 if they're aligned. The actual ‖v‖₂ is recorded for each configuration and used in the efficiency metric.

**Multi-layer features:** Features from different layers have decoder columns in the same dimensional space (the model's residual stream, ℝ^4096). They can be meaningfully averaged even though they come from different SAEs. The injection layer is where the averaged vector is injected — chosen as the mode layer among the top-k features.

### Step 4: The Optimization — Three Phases

#### Phase 1: Alpha Range Identification (cheap)

At k=1 (single best pro-bias feature), sweep the full alpha range:

```python
alpha_grid_full = [-80, -60, -40, -20, -10, -5, 5, 10, 20, 40, 60, 80]

for alpha in alpha_grid_full:
    vec, inj_layer = build_subgroup_steering_vector(pro_bias_features, sae_cache, k=1, alpha=alpha, ...)
    steerer = SAESteerer(wrapper, sae_cache[inj_layer], inj_layer)
    
    results = []
    for item in stereo_items:
        baseline = steerer.evaluate_baseline(item["prompt"])
        steered = steerer.steer_and_evaluate(item["prompt"], vec)
        results.append(build_result_dict(item, baseline, steered, vec))
    
    metrics = compute_all_metrics(results)
    degen_rate = sum(1 for r in results if r["degenerated"]) / len(results)
    
    phase1_results[alpha] = {
        "rcr_1.0": metrics["rcr_1.0"]["rcr"],
        "degeneration_rate": degen_rate,
        "vector_norm": float(vec.norm()),
    }
```

Identify viable alphas:
```python
viable_alphas = [
    alpha for alpha, res in phase1_results.items()
    if res["rcr_1.0"] > 0 and res["degeneration_rate"] < 0.05
]

# Keep at least 4 alphas; if fewer are viable, keep all that have degen < 0.10
if len(viable_alphas) < 4:
    viable_alphas = [a for a, r in phase1_results.items() if r["degeneration_rate"] < 0.10]

log(f"  Phase 1: {len(viable_alphas)} viable alphas out of {len(alpha_grid_full)}")
log(f"  Viable: {viable_alphas}")
```

This prunes the alpha grid from ~12 values to ~4-8, saving substantial compute in Phase 2.

#### Phase 2: Joint (k, α) Sweep

```python
k_steps = [1, 2, 3, 5, 8, 13, 21]

for k in k_steps:
    if k > len(pro_bias_features):
        break
    
    for alpha in viable_alphas:
        # Resume checkpoint
        ckpt_path = ckpt_dir / f"{cat}_{sub}_k{k}_a{alpha}.json"
        if ckpt_path.exists():
            grid.append(load_json(ckpt_path))
            continue
        
        vec, inj_layer = build_subgroup_steering_vector(
            pro_bias_features, sae_cache, k, alpha, ...)
        vec_norm = float(vec.norm())
        
        steerer = SAESteerer(wrapper, sae_cache[inj_layer], inj_layer)
        
        per_item_results = []
        for item in stereo_items:
            baseline = steerer.evaluate_baseline(item["prompt"])
            steered = steerer.steer_and_evaluate(item["prompt"], vec)
            per_item_results.append(build_result_dict(item, baseline, steered, vec))
        
        metrics = compute_all_metrics(per_item_results)
        degen_rate = sum(1 for r in per_item_results if r["degenerated"]) / len(per_item_results)
        corrupt_count = sum(1 for r in per_item_results if r["corrupted"])
        corrupt_rate = corrupt_count / max(len(per_item_results), 1)
        
        # Compute steering efficiency
        rcr_1 = metrics["rcr_1.0"]["rcr"]
        eta = rcr_1 / max(vec_norm, 1e-8)
        
        record = {
            "subgroup": sub, "category": cat,
            "k": k, "alpha": alpha,
            "injection_layer": inj_layer,
            "vector_norm": vec_norm,
            "eta": eta,
            "metrics": metrics,
            "degeneration_rate": degen_rate,
            "corruption_rate": corrupt_rate,
            "n_items": len(per_item_results),
            "features_used": pro_bias_features[:k],
        }
        
        # Save checkpoint
        save_json(record, ckpt_path)
        grid.append(record)
        
        log(f"    k={k} α={alpha}: RCR₁.₀={rcr_1:.3f} η={eta:.3f} "
            f"‖v‖={vec_norm:.3f} degen={degen_rate:.3f} corrupt={corrupt_rate:.3f}")
```

#### Phase 3: Selection

```python
# Filter to safe configurations
safe = [r for r in grid 
        if r["degeneration_rate"] < 0.05 
        and r["corruption_rate"] < 0.05]

if not safe:
    # Relax constraints
    safe = [r for r in grid if r["degeneration_rate"] < 0.10]
    log(f"  WARNING: no config with degen<0.05 AND corrupt<0.05; relaxed to degen<0.10")

if not safe:
    log(f"  WARNING: no viable config for {sub}; skipping")
    continue

# Select by maximum η
best_eta = max(r["eta"] for r in safe)

# Tie-breaking: within 1% of best η, prefer smaller ||v||
candidates = [r for r in safe if r["eta"] >= best_eta * 0.99]
candidates.sort(key=lambda r: (r["vector_norm"], -r["eta"]))
optimal = candidates[0]

log(f"  OPTIMAL for {sub}: k={optimal['k']}, α={optimal['alpha']}, "
    f"η={optimal['eta']:.3f}, RCR₁.₀={optimal['metrics']['rcr_1.0']['rcr']:.3f}, "
    f"‖v‖={optimal['vector_norm']:.3f}")
```

#### Phase 4: Marginal Analysis

At the selected α*, show how correction and cost evolve with k:

```python
optimal_alpha = optimal["alpha"]
marginal_data = []

for k in k_steps:
    matches = [r for r in grid if r["alpha"] == optimal_alpha and r["k"] == k]
    if not matches:
        continue
    r = matches[0]
    marginal_data.append({
        "k": k,
        "rcr_1.0": r["metrics"]["rcr_1.0"]["rcr"],
        "vector_norm": r["vector_norm"],
        "eta": r["eta"],
    })

# Compute marginal gain / marginal cost between successive k values
for i in range(1, len(marginal_data)):
    prev = marginal_data[i-1]
    curr = marginal_data[i]
    curr["marginal_rcr_gain"] = curr["rcr_1.0"] - prev["rcr_1.0"]
    curr["marginal_norm_cost"] = curr["vector_norm"] - prev["vector_norm"]
    curr["marginal_efficiency"] = (
        curr["marginal_rcr_gain"] / max(curr["marginal_norm_cost"], 1e-8)
    )
```

This shows whether adding features produces diminishing returns (marginal_efficiency drops) and identifies the "elbow" where it's no longer worth adding more.

### Step 5: Exacerbation Test (runs by default)

For the selected (k*, α*), run with FLIPPED alpha sign on ALL ambiguous items (both stereotyped and non-stereotyped response groups):

```python
exac_alpha = -optimal["alpha"]  # flip sign
exac_vec, exac_layer = build_subgroup_steering_vector(
    pro_bias_features, sae_cache, optimal["k"], exac_alpha, ...)

steerer = SAESteerer(wrapper, sae_cache[exac_layer], exac_layer)

# Test on non-stereotyped items: does exacerbation corrupt them?
exac_results_non_stereo = []
for item in non_stereo_items:
    baseline = steerer.evaluate_baseline(item["prompt"])
    steered = steerer.steer_and_evaluate(item["prompt"], exac_vec)
    exac_results_non_stereo.append(build_result_dict(item, baseline, steered, exac_vec))

exac_metrics = compute_all_metrics(exac_results_non_stereo)

# Test on stereotyped items: does exacerbation make them more confident?
exac_results_stereo = []
for item in stereo_items:
    baseline = steerer.evaluate_baseline(item["prompt"])
    steered = steerer.steer_and_evaluate(item["prompt"], exac_vec)
    exac_results_stereo.append(build_result_dict(item, baseline, steered, exac_vec))

# For stereo items, "exacerbation success" = logit for stereotyped option INCREASES
exac_logit_shifts_stereo = [
    r["logit_steered"][r["stereotyped_option"]] - r["logit_baseline"][r["stereotyped_option"]]
    for r in exac_results_stereo
    if r["stereotyped_option"] in r["logit_steered"] and r["stereotyped_option"] in r["logit_baseline"]
]
```

**What the exacerbation tells us:**
- If corruption rate on non-stereo items is HIGH: the features are powerful levers — they can push the model toward bias as easily as away from it.
- If corruption is high but debiasing (RCR) was low: the model is more vulnerable to bias amplification than amenable to correction. The features may be entangled with capability.
- If exacerbation logit shifts on already-stereotyped items are large and positive: the model can be pushed to more extreme bias, not just marginal.

### Step 6: Save Optimal Steering Vector

```python
vec, inj_layer = build_subgroup_steering_vector(
    pro_bias_features, sae_cache, optimal["k"], optimal["alpha"], ...)

np.savez(
    vectors_dir / f"{cat}_{sub}.npz",
    vector=vec.float().cpu().numpy(),
    injection_layer=inj_layer,
    alpha=optimal["alpha"],
    k=optimal["k"],
)
```

This .npz is what C2, C3 load.

### Step 7: Save Per-Item Results at Optimal Configuration

```python
per_item_df = pd.DataFrame(per_item_results_at_optimal)
per_item_df.to_parquet(per_item_dir / f"{cat}_{sub}_optimal.parquet", index=False)
```

Columns: item_idx, category, subgroup, k, alpha, baseline_answer, steered_answer, baseline_role, steered_role, corrected, corrupted, margin, margin_bin, logit_baseline (as JSON string), logit_steered (as JSON string), stereotyped_option, degenerated, vector_norm.

### Step 8: Build Steering Manifests

```python
manifest = {
    "subgroup": sub,
    "category": cat,
    "optimal_k": optimal["k"],
    "optimal_alpha": optimal["alpha"],
    "injection_layer": optimal.get("injection_layer"),
    "steering_efficiency_eta": optimal["eta"],
    "steering_vector_norm": optimal["vector_norm"],
    "features": optimal["features_used"],
    "phase1_viable_alphas": viable_alphas,
    "metrics": optimal["metrics"],
    "margin_bins": optimal["metrics"]["logit_shift"]["per_margin_bin"],
    "degeneration_rate": optimal["degeneration_rate"],
    "corruption_rate": optimal["corruption_rate"],
    "exacerbation": {
        "alpha": exac_alpha,
        "corruption_rate_non_stereo": exac_metrics["raw_corruption_rate"],
        "mean_logit_shift_stereo": float(np.mean(exac_logit_shifts_stereo)) if exac_logit_shifts_stereo else None,
        "n_non_stereo": len(exac_results_non_stereo),
        "n_stereo": len(exac_results_stereo),
    },
    # Placeholders for C3
    "medqa_matched_delta": None,
    "medqa_within_cat_mismatched_delta": None,
    "medqa_cross_cat_mismatched_delta": None,
    "medqa_nodemo_delta": None,
    "medqa_exacerbation_matched_delta": None,
    "mmlu_delta": None,
    "mmlu_worst_subject": None,
    "mmlu_worst_subject_delta": None,
}
```

### Output

```
{run}/C_steering/
├── phase1_results.json                   # Per-subgroup alpha viability at k=1
├── stepwise_results.json                 # Full k×α grid, all subgroups
├── optimal_configs.json                  # {cat: {sub: optimal record}}
├── steering_manifests.json               # List of complete manifests
├── marginal_analysis.json                # Per-subgroup marginal gain/cost data
├── vectors/
│   ├── so_gay.npz                        # {vector, injection_layer, alpha, k}
│   ├── so_bisexual.npz
│   └── ...
├── per_item/
│   ├── so_gay_optimal.parquet
│   ├── so_bisexual_optimal.parquet
│   └── ...
├── checkpoints/                          # Per-(subgroup, k, alpha) resume checkpoints
│   ├── so_gay_k1_a-20.json
│   └── ...
└── figures/
    ├── fig_pareto_frontier_so.png/.pdf
    ├── fig_pareto_frontier_race.png/.pdf
    ├── fig_stepwise_correction_so.png/.pdf
    ├── fig_marginal_analysis_so.png/.pdf
    ├── fig_optimal_k_distribution.png/.pdf
    ├── fig_alpha_vs_k_heatmaps_so.png/.pdf
    ├── fig_margin_conditioned_so.png/.pdf
    └── fig_exacerbation_asymmetry.png/.pdf
```

### Figures

**fig_pareto_frontier_{category}.png:** One subplot per subgroup. X = ‖v‖₂ (intervention strength). Y = RCR₁.₀ (debiasing benefit). Points = (k, α) configurations. Color by k (use viridis or similar ordinal colormap). Star marker on the selected optimum. Thin lines connecting points of constant α. Annotate optimum with "η*={value:.2f}". Gray region where degen > 0.05 or corrupt > 0.05.

**fig_stepwise_correction_{category}.png:** One subplot per subgroup. X = k. Y (primary axis) = RCR₁.₀ at optimal α (blue line, circle markers). Y (secondary axis) = corruption rate at optimal α (vermillion dashed, square markers). n count labels above each point. Vertical dashed line at selected k*.

**fig_marginal_analysis_{category}.png:** One subplot per subgroup. At the selected α*: X = k. Blue line = RCR₁.₀(k). Orange dashed = ‖v‖₂(k). Green dotted = marginal efficiency (marginal_rcr_gain / marginal_norm_cost). Vertical dashed line at selected k*.

**fig_optimal_k_distribution.png:** Histogram across all subgroups. X = optimal k, Y = count. Annotate median k.

**fig_alpha_vs_k_heatmaps_{category}.png:** One subplot per subgroup. Heatmap of η values across (k, α) grid. X = α, Y = k. Color = η (YlGnBu sequential). Star marker on optimum. Cells where degen > 0.05 or corrupt > 0.05 shown in gray.

**fig_margin_conditioned_{category}.png:** One subplot per subgroup. Grouped bars: three bars per subgroup (near_indifferent, moderate, confident) showing RCR at that margin bin. Include n counts per bar. Shows whether steering corrects only "cheap" low-margin items or also confident ones.

**fig_exacerbation_asymmetry.png:** Single figure across all categories. Paired bars per subgroup: left = RCR₁.₀ under debiasing, right = corruption rate under exacerbation. Blue and vermillion coloring. Grouped by category. Annotate with n for each bar. Shows whether it's easier to push bias in than pull it out.

### Resume Safety

Per-(subgroup, k, α) checkpointing. Before evaluating a configuration, check if the checkpoint JSON exists. Skip if it does. This means a crashed run resumes at the exact configuration it failed on, not re-running anything.

Per-layer SAE caching. SAEs are loaded once and reused across subgroups.

---

## C2: Cross-Subgroup Transfer & Universal Backfire Prediction

### Purpose

Test whether subgroup fragmentation is causally operative: when you apply subgroup A's steering vector to items targeting subgroup B, does the bias change depend on the cosine similarity between A and B's directions? This produces the universal backfire scatter — the headline figure.

### Invocation

```bash
python scripts/C2_transfer.py --run_dir runs/llama-3.1-8b_2026-04-15/

# Specific categories
python scripts/C2_transfer.py --run_dir runs/llama-3.1-8b_2026-04-15/ --categories so,race,religion
```

### Input

- Steering vectors: `{run}/C_steering/vectors/*.npz`
- Optimal configs: `{run}/C_steering/optimal_configs.json`
- Subgroup directions + cosines: `{run}/B_geometry/subgroup_directions.npz`, `{run}/B_geometry/cosine_matrices/*.json`
- Stimuli + metadata: from A1/A2
- Model + SAE

### Step 1: Compute Pairwise Cosines (Two Methods)

**Method A — SAE-based directions:** For each subgroup, the steering vector direction is the mean of its top-k* unit-normalized decoder columns (before α scaling). Compute pairwise cosines between subgroup steering vector directions:

```python
sae_directions = {}
for cat, subs in optimal_configs.items():
    for sub, config in subs.items():
        # Load the steering vector and normalize to unit length
        data = np.load(vectors_dir / f"{cat}_{sub}.npz")
        vec = data["vector"]
        vec_norm = np.linalg.norm(vec)
        if vec_norm > 1e-8:
            sae_directions[f"{cat}/{sub}"] = vec / vec_norm

sae_cosines = {}
for cat in categories:
    subs_in_cat = [s for s in sae_directions if s.startswith(f"{cat}/")]
    for i, s1 in enumerate(subs_in_cat):
        for j, s2 in enumerate(subs_in_cat):
            if j > i:
                cos = float(np.dot(sae_directions[s1], sae_directions[s2]))
                sae_cosines[(s1, s2)] = cos
```

**Method B — DIM-based directions:** Load identity directions from B3 at the injection layer:

```python
directions_npz = np.load(geometry_dir / "subgroup_directions.npz")

dim_cosines = {}
for cat in categories:
    # Load identity direction cosine matrix at the peak layer for this category
    with open(cosine_matrices_dir / f"{cat}_identity.json") as f:
        cos_data = json.load(f)
    
    # Use the injection layer from C1 for this category (or peak layer from B3)
    layer_key = f"layer_{injection_layer}"
    if layer_key in cos_data:
        matrix = cos_data[layer_key]["matrix"]
        subgroups = cos_data[layer_key]["subgroups"]
        for i, s1 in enumerate(subgroups):
            for j, s2 in enumerate(subgroups):
                if j > i:
                    dim_cosines[(f"{cat}/{s1}", f"{cat}/{s2}")] = matrix[i][j]
```

**Both methods produce the X-axis data for the scatter.** If they agree (high correlation between SAE and DIM cosines), the SAE decomposition captures the same geometry as raw DIM. If they disagree, report both and investigate why.

### Step 2: Cross-Subgroup Steering Transfer

For each (source_subgroup, target_subgroup) pair within a category, apply source's steering vector to target's items and measure bias change:

```python
for cat in categories:
    subs = subgroups_in(cat)
    if len(subs) < 2:
        continue
    
    for source_sub in subs:
        # Load source's optimal vector
        vec_data = np.load(vectors_dir / f"{cat}_{source_sub}.npz")
        vec = torch.from_numpy(vec_data["vector"]).to(device=device, dtype=model_dtype)
        inj_layer = int(vec_data["injection_layer"])
        
        sae = sae_cache[inj_layer]
        steerer = SAESteerer(wrapper, sae, inj_layer)
        
        for target_sub in subs:
            # Resume checkpoint
            ckpt = transfer_dir / f"{cat}_{source_sub}_to_{target_sub}.json"
            if ckpt.exists():
                continue
            
            # Get target subgroup's stereotyped-response items (ambig only)
            target_stereo = [
                it for it in cat_stimuli
                if target_sub in it["stereotyped_groups"]
                and it["context_condition"] == "ambig"
                and meta[it["item_idx"]]["model_answer_role"] == "stereotyped_target"
            ]
            
            if len(target_stereo) < 5:
                log(f"    {source_sub}→{target_sub}: <5 target items, skipping")
                continue
            
            # Evaluate: apply source's vector to target's items
            results = []
            for item in target_stereo:
                baseline = steerer.evaluate_baseline(item["prompt"])
                steered = steerer.steer_and_evaluate(item["prompt"], vec)
                results.append(build_result_dict(item, baseline, steered, vec))
            
            metrics = compute_all_metrics(results)
            
            # Compute bias score change
            # Baseline bias score: fraction of non-unknown responses that are stereotyped
            # (For stereotyped-response items, baseline bias is ~1.0 by definition.
            #  After steering, some may flip. The bias change is the difference.)
            n_still_stereo = sum(1 for r in results if r["steered_role"] == "stereotyped_target")
            n_total = len(results)
            steered_stereo_rate = n_still_stereo / max(n_total, 1)
            bias_change = steered_stereo_rate - 1.0  # negative = debiased
            
            record = {
                "source_subgroup": source_sub,
                "target_subgroup": target_sub,
                "category": cat,
                "is_self": source_sub == target_sub,
                "bias_change": bias_change,
                "mean_logit_shift": metrics["logit_shift"]["mean_shift"],
                "rcr_1.0": metrics["rcr_1.0"]["rcr"],
                "mwcs": metrics["mwcs_1.0"]["mwcs"],
                "n_items": n_total,
                "metrics": metrics,
            }
            save_json(record, ckpt)
```

**Note on bias_change computation:** Since we're starting from stereotyped-response items (baseline stereotyped rate ≈ 1.0), the bias_change is essentially -(correction rate). A more negative value means more debiasing. For cross-subgroup pairs, we expect:
- Self (source=target): large negative bias_change (strong debiasing)
- Aligned pairs (positive cosine): moderate negative bias_change (some cross-debiasing)
- Orthogonal pairs (zero cosine): ~zero bias_change
- Anti-correlated pairs (negative cosine): positive bias_change (BACKFIRE — bias increased)

### Step 3: Build the Universal Scatter

```python
scatter_data = []

for ckpt_path in sorted(transfer_dir.glob("*.json")):
    record = load_json(ckpt_path)
    cat = record["category"]
    src = record["source_subgroup"]
    tgt = record["target_subgroup"]
    
    if record["is_self"]:
        continue  # Exclude self-steering from the scatter (it's always effective)
    
    # Look up cosine
    key_fwd = (f"{cat}/{src}", f"{cat}/{tgt}")
    key_rev = (f"{cat}/{tgt}", f"{cat}/{src}")
    
    cosine_sae = sae_cosines.get(key_fwd) or sae_cosines.get(key_rev)
    cosine_dim = dim_cosines.get(key_fwd) or dim_cosines.get(key_rev)
    
    if cosine_sae is not None:
        scatter_data.append({
            "source": src, "target": tgt, "category": cat,
            "cosine_sae": cosine_sae,
            "cosine_dim": cosine_dim,
            "bias_change": record["bias_change"],
            "mean_logit_shift": record["mean_logit_shift"],
            "n_items": record["n_items"],
        })
```

### Step 4: Regression Analysis

```python
from scipy.stats import linregress
import numpy as np

# Using SAE cosines as X
x = np.array([d["cosine_sae"] for d in scatter_data])
y = np.array([d["bias_change"] for d in scatter_data])

slope, intercept, r_value, p_value, std_err = linregress(x, y)
r_squared = r_value ** 2

# Bootstrap CI for the regression line
n_bootstrap = 1000
rng = np.random.default_rng(42)
boot_slopes, boot_intercepts = [], []
for _ in range(n_bootstrap):
    idx = rng.choice(len(x), size=len(x), replace=True)
    s, i, _, _, _ = linregress(x[idx], y[idx])
    boot_slopes.append(s)
    boot_intercepts.append(i)

slope_ci = (np.percentile(boot_slopes, 2.5), np.percentile(boot_slopes, 97.5))
```

**Run twice:**
1. All categories included
2. Disability excluded (sensitivity test — Disability showed hypersensitivity to interventions)

Report both r² values and both p-values.

**Also run per-category regressions** for categories with ≥4 (source, target) pairs. Report per-category r² to show the relationship holds within categories, not just in aggregate.

### Step 5: SAE vs. DIM Cosine Comparison

If both cosine sets are available:

```python
# Scatter: X = SAE cosine, Y = DIM cosine, for all pairs
pairs_with_both = [d for d in scatter_data if d["cosine_dim"] is not None]
x_sae = [d["cosine_sae"] for d in pairs_with_both]
y_dim = [d["cosine_dim"] for d in pairs_with_both]

sae_dim_corr = np.corrcoef(x_sae, y_dim)[0, 1]
```

If correlation > 0.8: the SAE decomposition faithfully represents the same geometric structure as raw DIM. If < 0.5: the SAE imposes its own structure — warrants investigation.

### Output

```
{run}/C_transfer/
├── sae_cosines.json
├── dim_cosines.json
├── transfer_effects/
│   ├── so_gay_to_bisexual.json
│   ├── so_gay_to_lesbian.json
│   ├── so_bisexual_to_gay.json
│   └── ...
├── universal_scatter_data.json
├── regression_results.json
│     {all_categories: {r2, p, slope, intercept, slope_ci, n_pairs},
│      excluding_disability: {r2, p, slope, ...},
│      per_category: {so: {r2, p, ...}, race: {r2, p, ...}, ...}}
├── sae_vs_dim_comparison.json
└── figures/
    ├── fig_universal_backfire_scatter.png/.pdf
    ├── fig_transfer_heatmaps.png/.pdf
    ├── fig_cosine_vs_backfire_by_category.png/.pdf
    └── fig_sae_vs_dim_cosine.png/.pdf
```

### Figures

**fig_universal_backfire_scatter.png:** Two panels (A: all categories, B: excluding Disability).

Each panel:
- X = pairwise cosine (SAE-based)
- Y = bias_change (negative = debiasing, positive = backfire)
- One point per (source, target) pair, EXCLUDING self-steering
- Color by category (CATEGORY_COLORS)
- Distinct marker per category
- OLS regression line in black, dashed
- 95% CI band as gray shading (from bootstrap)
- Horizontal dashed line at y = 0 (no effect)
- Vertical dashed line at x = 0 (orthogonal directions)
- Annotate in bottom-right: "r² = {val:.3f}, p = {val:.1e}, n = {n_pairs}"
- Upper-left quadrant label: "BACKFIRE"
- Lower-right quadrant label: "CROSS-DEBIASING"

**fig_transfer_heatmaps.png:** Grid of heatmaps, one per category with ≥2 subgroups. Rows = source subgroup (whose vector was applied). Columns = target subgroup (whose bias was measured). Color = bias_change (RdBu_r, centered at 0; blue = debiasing, red = backfire). Annotate cells with values. Diagonal = self-steering (should be strongly blue). Panel labels (A, B, C, ...).

**fig_cosine_vs_backfire_by_category.png:** Faceted scatter — one panel per category with ≥4 data points. Same axes as universal scatter. Per-category regression line. Annotate per-category r². Shows whether the universal relationship holds within individual categories.

**fig_sae_vs_dim_cosine.png:** Single scatter. X = SAE cosine, Y = DIM cosine. One point per subgroup pair. Color by category. Annotate Pearson r. Diagonal reference line (y=x). If points cluster along the diagonal, SAE and DIM agree.

---

## C3: Generalization Evaluation

### Purpose

Apply the saved steering vectors from C1 to out-of-distribution benchmarks (MedQA and MMLU). Tests whether bias-specific interventions transfer to real-world tasks and measures the "capability tax" — how much general performance suffers.

### Invocation

```bash
python scripts/C3_generalization.py --run_dir runs/llama-3.1-8b_2026-04-15/

# Test with limited items
python scripts/C3_generalization.py --run_dir runs/llama-3.1-8b_2026-04-15/ --max_items 50
```

Reads medqa_path, mmlu_path from config.json.

### Input

- Steering vectors: `{run}/C_steering/vectors/*.npz`
- Manifests: `{run}/C_steering/steering_manifests.json`
- MedQA dataset (from config)
- MMLU dataset (from config)
- Model + SAE

### Step 0: Load and Validate External Datasets

**MedQA loading:**
```python
from src.data.medqa_loader import load_medqa_items

medqa_items = load_medqa_items(config["medqa_path"])
assert len(medqa_items) > 0, f"No MedQA items loaded from {config['medqa_path']}"
log(f"Loaded {len(medqa_items)} MedQA items")
log(f"  Sample: {medqa_items[0].keys()}")
log(f"  Expected fields: prompt, answer, options")
```

**Verify the loader returns items with the expected schema:** Each item must have `prompt` (the formatted question), `answer` (correct letter), and ideally `question` (raw question text for demographic classification). If the loader returns an empty list, log the file format expected vs. found and exit with an actionable error.

**MMLU loading:**
```python
from src.data.mmlu_loader import load_mmlu_items

mmlu_items = load_mmlu_items(config["mmlu_path"])
assert len(mmlu_items) > 0, f"No MMLU items loaded from {config['mmlu_path']}"
log(f"Loaded {len(mmlu_items)} MMLU items")
log(f"  Subjects: {len(set(it.get('subject','') for it in mmlu_items))} unique")
```

**If either dataset fails to load, log the error and continue with the other.** Don't crash the entire generalization pipeline because one dataset is missing.

### Step 1: Classify MedQA Items by Demographic Content

For each MedQA item, determine whether any BBQ-relevant subgroup is mentioned:

```python
# Build subgroup keyword patterns
SUBGROUP_PATTERNS = {
    "gay": [r"\bgay\b", r"\bhomosexual\b"],
    "lesbian": [r"\blesbian\b"],
    "bisexual": [r"\bbisexual\b"],
    "transgender": [r"\btransgender\b", r"\btranssexual\b", r"\bgender dysphoria\b"],
    "African American": [r"\bafrican.american\b"],
    "Asian": [r"\basian\b"],
    "Hispanic": [r"\bhispanic\b", r"\blatino\b", r"\blatina\b"],
    "Muslim": [r"\bmuslim\b", r"\bislam\b"],
    "Jewish": [r"\bjewish\b", r"\bashkenazi\b"],
    "obese": [r"\bobese\b", r"\bobesity\b"],
    "pregnant": [r"\bpregnant\b", r"\bpregnancy\b", r"\bgestational\b"],
    # ... etc, matching the subgroup labels used in BBQ
}

# Map each subgroup to its BBQ category
SUBGROUP_TO_CATEGORY = {
    "gay": "so", "lesbian": "so", "bisexual": "so",
    "transgender": "gi",
    "African American": "race", "Asian": "race", "Hispanic": "race",
    "Muslim": "religion", "Jewish": "religion",
    "obese": "physical_appearance", "pregnant": "physical_appearance",
    # ...
}

def classify_medqa_item(text: str) -> list[dict]:
    """Return list of {subgroup, category} matches found in text."""
    matches = []
    text_lower = text.lower()
    for subgroup, patterns in SUBGROUP_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text_lower):
                matches.append({
                    "subgroup": subgroup,
                    "category": SUBGROUP_TO_CATEGORY[subgroup],
                })
                break  # don't double-count same subgroup
    return matches
```

Classify all items:
```python
for item in medqa_items:
    text = item.get("prompt", "") or item.get("question", "")
    item["demographic_matches"] = classify_medqa_item(text)
    item["mentions_demographic"] = len(item["demographic_matches"]) > 0
```

Log statistics:
```python
n_demo = sum(1 for it in medqa_items if it["mentions_demographic"])
log(f"  Demographic classification: {n_demo}/{len(medqa_items)} items mention a subgroup")
for sub in sorted(SUBGROUP_PATTERNS.keys()):
    n = sum(1 for it in medqa_items if any(m["subgroup"] == sub for m in it["demographic_matches"]))
    if n > 0:
        log(f"    {sub}: {n} items")
```

**Save classification for audit:**
```python
save_json(
    {it.get("item_idx", i): it["demographic_matches"] for i, it in enumerate(medqa_items)},
    output_dir / "medqa" / "demographic_classification.json"
)
```

### Step 2: Compute Baseline Performance (once, reuse across all vectors)

Run all MedQA and MMLU items through the model WITHOUT steering. Cache the results:

```python
medqa_baselines = {}
for i, item in enumerate(medqa_items):
    prompt = item["prompt"]
    letters = tuple(item.get("letters", ("A", "B", "C", "D", "E")))
    baseline = steerer.evaluate_baseline_mcq(prompt, letters=letters)
    medqa_baselines[i] = {
        "answer": baseline["model_answer"],
        "logits": baseline["answer_logits"],
        "correct": int(baseline["model_answer"] == item.get("answer", "")),
    }

mmlu_baselines = {}
for i, item in enumerate(mmlu_items):
    prompt = build_mmlu_prompt(item)  # "Subject: ...\nQuestion: ...\nA. ...\n..."
    baseline = steerer.evaluate_baseline_mcq(prompt, letters=("A", "B", "C", "D"))
    mmlu_baselines[i] = {
        "answer": baseline["model_answer"],
        "logits": baseline["answer_logits"],
        "correct": int(baseline["model_answer"] == item.get("answer", "")),
    }

baseline_medqa_acc = sum(b["correct"] for b in medqa_baselines.values()) / max(len(medqa_baselines), 1)
baseline_mmlu_acc = sum(b["correct"] for b in mmlu_baselines.values()) / max(len(mmlu_baselines), 1)
log(f"  Baseline MedQA accuracy: {baseline_medqa_acc:.4f}")
log(f"  Baseline MMLU accuracy: {baseline_mmlu_acc:.4f}")
```

**This is the most important optimization in C3.** Without caching baselines, each steering vector would require re-computing baselines — that's ~40 vectors × ~1000 items = ~40K redundant forward passes. Caching makes it ~1000 + 40×1000 = ~41K passes instead of ~80K.

### Step 3: Evaluate Each Steering Vector

For each subgroup's optimal steering vector, evaluate under multiple conditions:

```python
for vec_key in sorted(vectors.keys()):  # e.g., "so_gay"
    cat, sub = vec_key.split("_", 1)
    vec_data = load_npz(vectors_dir / f"{vec_key}.npz")
    vec = torch.from_numpy(vec_data["vector"]).to(device=device, dtype=model_dtype)
    neg_vec = -vec  # exacerbation direction
    inj_layer = int(vec_data["injection_layer"])
    
    sae = sae_cache[inj_layer]
    steerer = SAESteerer(wrapper, sae, inj_layer)
    
    # ---- MedQA evaluation ----
    if medqa_items:
        # Condition 1: Matched — items mentioning THIS subgroup
        matched = [i for i, it in enumerate(medqa_items) 
                   if any(m["subgroup"] == sub for m in it["demographic_matches"])]
        
        # Condition 2: Within-category mismatched — items mentioning a DIFFERENT subgroup from SAME category
        cat_subs_in_vectors = [k.split("_", 1)[1] for k in vectors if k.startswith(f"{cat}_")]
        within_mismatched = [
            i for i, it in enumerate(medqa_items)
            if any(m["category"] == cat and m["subgroup"] != sub for m in it["demographic_matches"])
            and not any(m["subgroup"] == sub for m in it["demographic_matches"])
        ]
        
        # Condition 3: Cross-category mismatched — items mentioning a subgroup from DIFFERENT category
        cross_mismatched = [
            i for i, it in enumerate(medqa_items)
            if any(m["category"] != cat for m in it["demographic_matches"])
            and not any(m["category"] == cat for m in it["demographic_matches"])
        ]
        
        # Condition 4: No demographic content
        no_demo = [i for i, it in enumerate(medqa_items) if not it["mentions_demographic"]]
        
        conditions = {
            "matched": matched,
            "within_cat_mismatched": within_mismatched,
            "cross_cat_mismatched": cross_mismatched,
            "no_demographic": no_demo,
        }
        
        medqa_results_entry = {"vector": vec_key, "category": cat, "subgroup": sub}
        
        for cond_name, item_indices in conditions.items():
            if not item_indices:
                log(f"    MedQA {cond_name}: 0 items, skipping")
                continue
            
            # Debiasing direction
            debiasing_results = evaluate_with_vector(
                steerer, medqa_items, item_indices, medqa_baselines,
                vec, letters=("A", "B", "C", "D", "E")
            )
            
            # Exacerbation direction
            exac_results = evaluate_with_vector(
                steerer, medqa_items, item_indices, medqa_baselines,
                neg_vec, letters=("A", "B", "C", "D", "E")
            )
            
            medqa_results_entry[f"{cond_name}_debiasing"] = summarize_generalization(debiasing_results)
            medqa_results_entry[f"{cond_name}_exacerbation"] = summarize_generalization(exac_results)
            
            log(f"    MedQA {cond_name}: n={len(item_indices)}, "
                f"debias_delta={medqa_results_entry[f'{cond_name}_debiasing']['delta']:.4f}, "
                f"exac_delta={medqa_results_entry[f'{cond_name}_exacerbation']['delta']:.4f}")
```

**The evaluation helper:**

```python
def evaluate_with_vector(steerer, items, indices, baselines, vec, letters):
    """Evaluate steering on a subset of items, reusing cached baselines."""
    results = []
    for idx in indices:
        item = items[idx]
        prompt = item["prompt"]
        bl = baselines[idx]
        
        # Steered forward pass
        steered = steerer.steer_and_evaluate(prompt, vec, letters=letters)
        
        correct_answer = item.get("answer", "")
        results.append({
            "item_idx": idx,
            "baseline_answer": bl["answer"],
            "steered_answer": steered["model_answer"],
            "baseline_correct": bl["correct"],
            "steered_correct": int(steered["model_answer"] == correct_answer),
            "flipped": bl["answer"] != steered["model_answer"],
            "baseline_logits": bl["logits"],
            "steered_logits": steered["answer_logits"],
            "degenerated": steered.get("degenerated", False),
            "subject": item.get("subject", ""),
            "demographic_matches": item.get("demographic_matches", []),
        })
    return results

def summarize_generalization(results):
    """Compute accuracy metrics for generalization."""
    n = len(results)
    if n == 0:
        return {"n": 0}
    
    b_correct = sum(r["baseline_correct"] for r in results)
    s_correct = sum(r["steered_correct"] for r in results)
    n_flipped = sum(r["flipped"] for r in results)
    
    return {
        "n": n,
        "accuracy_baseline": round(b_correct / n, 4),
        "accuracy_steered": round(s_correct / n, 4),
        "delta": round((s_correct - b_correct) / n, 4),
        "flip_rate": round(n_flipped / n, 4),
        "n_degraded": sum(r["steered_correct"] < r["baseline_correct"] for r in results),
        "n_improved": sum(r["steered_correct"] > r["baseline_correct"] for r in results),
    }
```

**MMLU evaluation — same structure but without demographic conditions:**

```python
for vec_key, vec, neg_vec, inj_layer in vectors:
    # Debiasing direction on all MMLU items
    debiasing_results = evaluate_with_vector(
        steerer, mmlu_items, range(len(mmlu_items)), mmlu_baselines,
        vec, letters=("A", "B", "C", "D")
    )
    
    # Exacerbation direction
    exac_results = evaluate_with_vector(
        steerer, mmlu_items, range(len(mmlu_items)), mmlu_baselines,
        neg_vec, letters=("A", "B", "C", "D")
    )
    
    # Per-subject breakdown
    per_subject = {}
    for r in debiasing_results:
        s = r.get("subject", "other")
        per_subject.setdefault(s, {"n": 0, "b_correct": 0, "s_correct": 0, "flipped": 0})
        per_subject[s]["n"] += 1
        per_subject[s]["b_correct"] += r["baseline_correct"]
        per_subject[s]["s_correct"] += r["steered_correct"]
        per_subject[s]["flipped"] += r["flipped"]
    
    # Identify worst-affected subject
    worst_subject = min(per_subject, key=lambda s: 
        (per_subject[s]["s_correct"] - per_subject[s]["b_correct"]) / max(per_subject[s]["n"], 1))
    worst_delta = (per_subject[worst_subject]["s_correct"] - per_subject[worst_subject]["b_correct"]) / max(per_subject[worst_subject]["n"], 1)
```

### Step 4: Save Per-Item Parquets

```python
# MedQA per-item parquet
medqa_per_item_rows = []
for vec_key, results_by_condition in all_medqa_results.items():
    for cond, results in results_by_condition.items():
        for r in results:
            medqa_per_item_rows.append({
                "item_idx": r["item_idx"],
                "steering_vector": vec_key,
                "condition": cond,
                "baseline_answer": r["baseline_answer"],
                "steered_answer": r["steered_answer"],
                "baseline_correct": r["baseline_correct"],
                "steered_correct": r["steered_correct"],
                "flipped": r["flipped"],
                "demographic_matches": json.dumps(r["demographic_matches"]),
            })

pd.DataFrame(medqa_per_item_rows).to_parquet(output_dir / "medqa" / "per_item.parquet")
```

Same for MMLU, adding `subject` column.

### Step 5: Update Steering Manifests

```python
manifests = load_json(steering_dir / "steering_manifests.json")

for m in manifests:
    key = f"{m['category']}_{m['subgroup']}"
    medqa = all_medqa_results.get(key, {})
    mmlu = all_mmlu_results.get(key, {})
    
    m["medqa_matched_delta"] = medqa.get("matched_debiasing", {}).get("delta")
    m["medqa_within_cat_mismatched_delta"] = medqa.get("within_cat_mismatched_debiasing", {}).get("delta")
    m["medqa_cross_cat_mismatched_delta"] = medqa.get("cross_cat_mismatched_debiasing", {}).get("delta")
    m["medqa_nodemo_delta"] = medqa.get("no_demographic_debiasing", {}).get("delta")
    m["medqa_exacerbation_matched_delta"] = medqa.get("matched_exacerbation", {}).get("delta")
    m["mmlu_delta"] = mmlu.get("debiasing", {}).get("delta")
    m["mmlu_worst_subject"] = mmlu.get("worst_subject")
    m["mmlu_worst_subject_delta"] = mmlu.get("worst_delta")

save_json(manifests, output_dir / "manifests_with_generalization.json")
```

### Output

```
{run}/C_generalization/
├── medqa/
│   ├── per_item.parquet
│   ├── results_by_vector.json
│   └── demographic_classification.json
├── mmlu/
│   ├── per_item.parquet
│   └── results_by_vector.json
├── baselines.json                         # Cached baseline accuracies
├── manifests_with_generalization.json
└── figures/
    ├── fig_medqa_matched_vs_mismatched.png/.pdf
    ├── fig_medqa_exacerbation.png/.pdf
    ├── fig_side_effect_heatmap.png/.pdf
    └── fig_debiasing_vs_exacerbation.png/.pdf
```

### Figures

**fig_medqa_matched_vs_mismatched.png:** One panel per category (only categories with ≥1 matched MedQA item). Grouped bars per subgroup showing accuracy delta under: matched_debiasing, within_cat_mismatched_debiasing, cross_cat_mismatched_debiasing, no_demographic_debiasing. Horizontal dashed line at 0. Include n counts per bar. Color scheme: distinct color per condition. Error bars via bootstrap 95% CI where n ≥ 20.

**fig_medqa_exacerbation.png:** One panel per category. Paired bars per subgroup: left = matched_debiasing delta, right = matched_exacerbation delta. Blue and vermillion coloring. Shows the asymmetry: is it easier to push bias in (large negative exacerbation delta) than pull it out (small positive debiasing delta)?

**fig_side_effect_heatmap.png:** Single heatmap. Rows = steering vectors (one per subgroup, labeled "{category}/{subgroup}"). Columns = {MedQA no-demo, MMLU overall, MMLU STEM, MMLU humanities, MMLU social science, MMLU other}. Color = accuracy delta (RdBu_r, centered at 0). Annotate cells. Identifies which vectors are most/least harmful to which knowledge domains.

**fig_debiasing_vs_exacerbation.png:** Single scatter. X = BBQ RCR₁.₀ from C1 (debiasing effectiveness on BBQ). Y = MedQA matched exacerbation delta from C3 (vulnerability to bias amplification on medical content). Each point = one subgroup. Color by category. Annotate Pearson r. If correlated: features are bias-specific. If Y is large but X is small: features are entangled with medical reasoning.

---

## C4: Token-Level Feature Interpretability

### Purpose

For the top features in each subgroup's optimal steering vector, determine which SPECIFIC TOKENS in the prompt most strongly activate each feature. This provides the most granular interpretability evidence: "Feature L14_F45021 fires most strongly on the token 'bisexual' at position 17."

### Invocation

```bash
python scripts/C4_token_features.py --run_dir runs/llama-3.1-8b_2026-04-15/

# Specific categories
python scripts/C4_token_features.py --run_dir runs/llama-3.1-8b_2026-04-15/ --categories so,race
```

### Input

- Optimal configs: `{run}/C_steering/optimal_configs.json`
- Ranked features: `{run}/B_feature_ranking/ranked_features_by_subgroup.json`
- Stimuli: `{run}/A_extraction/stimuli/{category}.json`
- Model + SAE

### Scope

For each subgroup, analyze the **top-3 features** in the optimal steering vector (or fewer if k* < 3). This keeps the compute tractable — each feature requires a full forward pass with per-token hooks for ~100 items.

### Per-Feature Processing

**Step 1: Register per-token hooks.**

Unlike A2 (which captures only the last-token hidden state), C4 captures the hidden state at ALL token positions at the feature's layer:

```python
def make_all_token_hook(layer_idx, hidden_dim, storage):
    def hook_fn(module, args, output):
        h = locate_hidden_tensor(output, hidden_dim)
        # h shape: (1, seq_len, hidden_dim)
        storage[layer_idx] = h[0].detach().cpu().float()  # (seq_len, hidden_dim)
    return hook_fn
```

**Step 2: For each item, run forward pass and capture per-token hidden states.**

```python
for item in category_items[:max_items]:
    prompt = item["prompt"]
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    seq_len = len(tokens)
    
    storage = {}
    hook = get_layer_fn(feature_layer).register_forward_hook(
        make_all_token_hook(feature_layer, hidden_dim, storage)
    )
    
    with torch.no_grad():
        model(**inputs)
    
    hook.remove()
    
    # Per-token hidden states at the feature's layer: (seq_len, hidden_dim)
    all_token_hs = storage[feature_layer]
```

**Step 3: Encode each token position through the SAE.**

```python
# Encode all positions at once (batch)
feature_activations = sae.encode(all_token_hs)  # (seq_len, n_features)

# Get activation of our target feature at each position
target_activations = feature_activations[:, feature_idx].numpy()  # (seq_len,)
```

**Step 4: Record per-token activations.**

```python
token_records = []
for pos in range(seq_len):
    token_records.append({
        "position": pos,
        "token": tokens[pos],
        "activation": float(target_activations[pos]),
    })

# Sort by activation descending
token_records.sort(key=lambda r: -r["activation"])
```

**Step 5: Filter template tokens.**

Identify tokens that appear at the same position in >80% of items (template invariants):

```python
# Build position→token frequency table across all items
position_token_counts = defaultdict(Counter)
for item in all_items:
    toks = tokenizer.convert_ids_to_tokens(tokenizer.encode(item["prompt"]))
    for pos, tok in enumerate(toks):
        position_token_counts[pos][tok] += 1

n_items = len(all_items)
template_positions = set()
for pos, counter in position_token_counts.items():
    most_common_tok, count = counter.most_common(1)[0]
    if count / n_items > 0.80:
        template_positions.add(pos)

# Filter: exclude template positions from top-activating tokens
filtered_tokens = [r for r in token_records if r["position"] not in template_positions]
```

This removes "Context", ":", "Question", "Answer", "A", "B", "C", ".", "\n", and other prompt-structure tokens that appear in nearly every item. What remains are the content-variable tokens: names, identity terms, scenario descriptions, answer texts.

**Step 6: Aggregate across items.**

For each feature, aggregate across all items to produce a token frequency/activation table:

```python
# Group by token string, compute statistics
from collections import defaultdict
token_stats = defaultdict(list)

for item_result in all_item_results:
    for r in item_result["filtered_tokens"]:
        token_stats[r["token"]].append(r["activation"])

token_summary = []
for token, activations in token_stats.items():
    nonzero = [a for a in activations if a > 0]
    token_summary.append({
        "token": token,
        "n_occurrences": len(activations),         # how many items have this token
        "n_nonzero": len(nonzero),                  # how many times the feature fires ON this token
        "mean_activation": float(np.mean(activations)),
        "mean_activation_nonzero": float(np.mean(nonzero)) if nonzero else 0,
        "max_activation": float(max(activations)),
        "median_activation": float(np.median(activations)),
    })

# Sort by mean_activation_nonzero descending (most characteristic tokens)
token_summary.sort(key=lambda r: -r["mean_activation_nonzero"])
```

This produces the answer to "what tokens does this feature care about?" as a ranked list sorted by mean activation when the feature fires.

### Output

```
{run}/C_token_features/
├── token_activations/
│   ├── so_gay_L14_F45021.json         # Per-item, per-token activations + aggregated summary
│   ├── so_gay_L14_F88012.json
│   ├── so_bisexual_L16_F72301.json
│   └── ...
├── template_positions.json             # Per-category: which positions are template-invariant
├── token_summaries.json                # Aggregated token rankings per feature
└── figures/
    ├── fig_token_activations_so.png/.pdf
    ├── fig_token_activations_race.png/.pdf
    └── ...
```

### Figures

**fig_token_activations_{category}.png:** Grid of subplots, one per (subgroup, feature) combination (up to 3 per subgroup × ~4 subgroups = 12 panels per category).

Each panel:
- Horizontal bar chart of top-15 tokens by mean_activation_nonzero
- Y-axis = token string (cleaned: replace special characters, truncate long tokens)
- X-axis = mean activation when nonzero
- Color bars by category of the token: identity terms (blue), stereotype-related words (orange), other content (gray). This coloring requires a simple keyword list for identity terms — if the token matches any SUBGROUP_PATTERNS regex, it's an identity term; if it matches a manually curated stereotype keyword list (e.g., "promiscuous", "criminal", "lazy", "terrorist"), it's stereotype-related; else other.
- Annotate each bar with n_nonzero (how many items the feature fires on for this token)
- Title: "Feature L{layer}_F{idx} — selected for {subgroup}"

This is the "money figure" for interpretability. If the top tokens for the bisexual pro-bias feature are ["bisexual", "promiscuous", "confused", "phase"], that's compelling evidence the SAE has decomposed the bisexual stereotyping mechanism into an interpretable feature.

---

## Phase C Output Structure (Complete)

```
{run}/
├── C_steering/
│   ├── phase1_results.json
│   ├── stepwise_results.json
│   ├── optimal_configs.json
│   ├── steering_manifests.json
│   ├── marginal_analysis.json
│   ├── vectors/*.npz
│   ├── per_item/*.parquet
│   ├── checkpoints/
│   └── figures/
│
├── C_transfer/
│   ├── sae_cosines.json
│   ├── dim_cosines.json
│   ├── transfer_effects/*.json
│   ├── universal_scatter_data.json
│   ├── regression_results.json
│   ├── sae_vs_dim_comparison.json
│   └── figures/
│
├── C_generalization/
│   ├── medqa/{per_item.parquet, results_by_vector.json, demographic_classification.json}
│   ├── mmlu/{per_item.parquet, results_by_vector.json}
│   ├── baselines.json
│   ├── manifests_with_generalization.json
│   └── figures/
│
└── C_token_features/
    ├── token_activations/*.json
    ├── template_positions.json
    ├── token_summaries.json
    └── figures/
```

---

## Assumptions and Methodological Choices Summary

| # | Choice | Rationale | Downstream Impact |
|---|--------|-----------|-------------------|
| 1 | η = RCR₁.₀ / ‖v‖₂ as optimization objective | Maximizes correction per unit perturbation; naturally penalizes unnecessary features and excessive alpha | Produces minimal-intervention steering vectors |
| 2 | Phase 1 alpha pruning (k=1 sweep) before full grid | Reduces Phase 2 compute by ~50% by eliminating clearly non-viable alphas | May miss configurations where k>1 enables an alpha that fails at k=1; mitigated by keeping ≥4 alphas |
| 3 | Tie-breaking: within 1% of best η, prefer smaller ‖v‖₂ | Minimal intervention when equally effective | May miss slightly higher correction at slightly larger norm |
| 4 | Exacerbation runs by default (flipped alpha) | Asymmetry between debiasing and exacerbation is a finding; needed for C3 asymmetry figure | Adds ~50% more forward passes per subgroup |
| 5 | Exclude self-steering from universal scatter | Self-steering is trivially effective (high cosine, high debiasing); including it would inflate r² | Scatter tests cross-subgroup prediction only |
| 6 | Both SAE-based and DIM-based cosines for scatter | Cross-validation: if both produce same r², the geometric relationship is method-independent | Requires B3 to have been run |
| 7 | Baseline caching in C3 | Single baseline pass for all items, reused across all steering vectors | Halves total forward passes; baselines must not use hooks (verified by evaluate_baseline clearing hooks) |
| 8 | Four MedQA conditions: matched, within-cat mismatch, cross-cat mismatch, no-demographic | Tests specificity at three levels: subgroup, category, and no-overlap baseline | Low N for matched condition (few MedQA items mention specific BBQ subgroups); report N per condition |
| 9 | Template position filtering in C4 (>80% invariant) | Removes "Question", "Answer", "the" etc. from token analysis | Threshold is heuristic; 80% is conservative (keeps tokens that vary in 20%+ of items) |
| 10 | Top-3 features per subgroup for C4 | Balances interpretability depth with compute cost | If k* < 3, analyze all k* features; if k* > 3, the top 3 are the most impactful |
| 11 | Token aggregation by mean_activation_nonzero | More informative than raw mean (which is diluted by items where feature doesn't fire) | Biased toward tokens that co-occur with feature firing, which is what we want |
| 12 | Both debiasing AND exacerbation on ALL C3 conditions | Exacerbation on medical content tests vulnerability to adversarial bias injection | Doubles the steered forward passes in C3 |