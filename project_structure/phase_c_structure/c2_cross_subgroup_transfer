# C2: Cross-Subgroup Transfer & Universal Backfire — Full Implementation Specification

## Purpose

Test whether subgroup fragmentation is causally operative: when you apply subgroup A's steering vector to items targeting subgroup B, does the bias change depend on the cosine similarity between A's and B's identity representations? The core product is the **universal backfire scatter** — the headline figure of the paper. Anti-correlated subgroup pairs should exhibit backfire (steering A's bias-reduction vector INCREASES bias on B), while aligned pairs should show cross-debiasing.

## Invocation

```bash
python scripts/C2_transfer.py --run_dir runs/llama-3.1-8b_2026-04-15/

# Specific categories
python scripts/C2_transfer.py --run_dir runs/llama-3.1-8b_2026-04-15/ --categories so,race,religion

# Quick test (limited items per pair)
python scripts/C2_transfer.py --run_dir runs/llama-3.1-8b_2026-04-15/ --max_items 20

# Override minimum items per target
python scripts/C2_transfer.py --run_dir runs/llama-3.1-8b_2026-04-15/ --min_n_per_target 15

# Skip sensitivity analyses (primary only)
python scripts/C2_transfer.py --run_dir runs/llama-3.1-8b_2026-04-15/ --primary_only
```

Reads `model_path`, `sae_source`, `device`, `dtype` from config.json.

---

## Input

- Steering manifests: `{run}/C_steering/steering_manifests.json` (for `steering_viable` filter + injection layers)
- Steering vectors: `{run}/C_steering/vectors/{cat}_{sub}.npz`
- DIM directions: `{run}/B_geometry/subgroup_directions.npz` (identity_normed variants)
- Cosine data: `{run}/B_geometry/cosine_pairs.parquet`
- Differentiation metrics: `{run}/B_geometry/differentiation_metrics.json` (peak layers, stable range)
- Metadata: `{run}/A_extraction/metadata.parquet`
- Stimuli: `{run}/A_extraction/stimuli/{category}.json`
- Model + SAE

## Dependencies

- `torch`
- `pandas`, `pyarrow`
- `numpy`, `scipy.stats.linregress`
- `matplotlib`
- Ported from old repo: `SAESteerer`, `SAEWrapper`, `ModelWrapper`
- Shared: `src/metrics/bias_metrics.py`

---

## Conceptual Framework

**What's being tested.** If subgroup fragmentation is real — i.e., different subgroups within a category have anti-correlated representations — then applying a steering vector designed for one subgroup should have predictably different effects on another subgroup depending on whether they're aligned or anti-correlated in representation space.

**X-axis: DIM identity_normed cosines.** Measures subgroup representational similarity independent of the steering vectors. This avoids circularity: we're not asking "does similarity between two vectors predict similarity of their effects" (tautological) but rather "does subgroup geometry in model activation space predict transfer effects."

**Y-axis: bias_change under cross-subgroup steering.** Evaluated on all ambig items targeting the target subgroup. Can be positive (backfire — steering made it MORE biased) or negative (cross-debiasing — steering reduced bias on a subgroup it wasn't designed for).

**Diagnostic: SAE vs DIM cosine comparison.** Does the SAE decomposition preserve the same geometric structure as raw DIM? If yes, SAE features reflect the model's own representational structure.

---

## Step 1: Load Inputs and Filter to Viable Subgroups

```python
def load_viable_manifests(run_dir: Path) -> list[dict]:
    """Load steering manifests from C1, filter to steering_viable subgroups."""
    with open(run_dir / "C_steering" / "steering_manifests.json") as f:
        manifests = json.load(f)
    viable = [m for m in manifests if m.get("steering_viable")]
    log(f"Loaded {len(viable)} viable subgroups (of {len(manifests)} total)")
    return viable


def load_steering_vector(run_dir: Path, cat: str, sub: str, device, dtype) -> tuple[torch.Tensor, int]:
    """Load pre-computed steering vector from C1 output."""
    data = np.load(run_dir / "C_steering" / "vectors" / f"{cat}_{sub}.npz")
    vec = torch.from_numpy(data["vector"]).to(device=device, dtype=dtype)
    injection_layer = int(data["injection_layer"])
    return vec, injection_layer
```

---

## Step 2: Compute Primary Cosines (DIM Identity, Normed)

The X-axis data. For each subgroup pair within a category, get the DIM cosine at the category's peak differentiation layer.

```python
def get_primary_cosines(
    run_dir: Path,
    viable_subgroups: list[tuple[str, str]],
) -> pd.DataFrame:
    """
    Build primary cosine table from B3 output.
    
    Uses DIM identity_normed direction cosines at each category's peak differentiation layer.
    
    Returns DataFrame with columns:
        category, subgroup_A, subgroup_B, peak_layer, cosine_dim_identity_normed
    """
    # Load B3 cosine pairs parquet
    cosine_df = pd.read_parquet(run_dir / "B_geometry" / "cosine_pairs.parquet")
    
    # Load peak layers per category from differentiation_metrics.json
    with open(run_dir / "B_geometry" / "differentiation_metrics.json") as f:
        diff_metrics = json.load(f)
    
    rows = []
    for cat, cat_metrics in diff_metrics.items():
        if "identity_normed" not in cat_metrics:
            continue
        peak_layer = cat_metrics["identity_normed"]["peak_layer"]
        
        # Filter cosine pairs for this category, direction type, and peak layer
        cat_cosines = cosine_df[
            (cosine_df["category"] == cat) &
            (cosine_df["direction_type"] == "identity_normed") &
            (cosine_df["layer"] == peak_layer)
        ]
        
        for _, row in cat_cosines.iterrows():
            rows.append({
                "category": cat,
                "subgroup_A": row["subgroup_A"],
                "subgroup_B": row["subgroup_B"],
                "peak_layer": int(peak_layer),
                "cosine_dim_identity_normed": float(row["cosine"]),
            })
    
    return pd.DataFrame(rows)
```

The peak differentiation layer is a category-level property (the layer where subgroups within that category are most distinguishable in representation space). Each category uses its own peak layer.

---

## Step 3: Compute Secondary Cosines (SAE Steering Vectors)

For comparison/diagnostic purposes.

```python
def get_sae_cosines(
    run_dir: Path,
    viable_manifests: list[dict],
    device, dtype,
) -> pd.DataFrame:
    """
    Compute pairwise cosines between SAE steering vector directions.
    
    The "direction" is the unit-normalized steering vector from C1.
    Pairs are within-category only.
    """
    # Group viable manifests by category
    by_cat = {}
    for m in viable_manifests:
        by_cat.setdefault(m["category"], []).append(m["subgroup"])
    
    # Load all steering vectors, normalize to unit
    sae_directions = {}  # (cat, sub) -> unit vector
    for m in viable_manifests:
        vec, _ = load_steering_vector(run_dir, m["category"], m["subgroup"], device, dtype)
        vec_np = vec.float().cpu().numpy()
        norm = np.linalg.norm(vec_np)
        if norm > 1e-8:
            sae_directions[(m["category"], m["subgroup"])] = vec_np / norm
    
    rows = []
    for cat, subs in by_cat.items():
        subs_sorted = sorted(subs)
        for i, sub_A in enumerate(subs_sorted):
            for sub_B in subs_sorted[i+1:]:
                key_A = (cat, sub_A)
                key_B = (cat, sub_B)
                if key_A not in sae_directions or key_B not in sae_directions:
                    continue
                cos = float(np.dot(sae_directions[key_A], sae_directions[key_B]))
                rows.append({
                    "category": cat,
                    "subgroup_A": sub_A,
                    "subgroup_B": sub_B,
                    "cosine_sae_steering": round(cos, 6),
                })
    
    return pd.DataFrame(rows)
```

---

## Step 4: Cross-Subgroup Steering Transfer

For each (source_subgroup, target_subgroup) pair within a category, apply source's pre-computed steering vector to target's items and measure bias change.

**Key change from original spec:** Evaluate on ALL ambig items targeting the target subgroup, not just stereotyped-response items. This allows backfire to manifest as positive bias_change (items that were unknown/non-stereo baseline becoming stereo under steering).

```python
def run_transfer_evaluation(
    viable_manifests: list[dict],
    metadata_df: pd.DataFrame,
    stimuli_by_cat: dict,
    wrapper, sae_cache: dict,
    run_dir: Path,
    output_dir: Path,
    args,
) -> pd.DataFrame:
    """
    For each (source, target) subgroup pair within a category, run cross-subgroup steering.
    Self-pairs (source == target) are INCLUDED so they appear as anchors in the scatter.
    Checkpoints per pair; resume-safe.
    
    Returns a flat DataFrame with one row per (source, target) pair.
    """
    transfer_dir = output_dir / "per_pair_checkpoints"
    ensure_dir(transfer_dir)
    
    device = wrapper.device
    dtype = wrapper.dtype
    
    # Group viable manifests by category
    by_cat = {}
    for m in viable_manifests:
        by_cat.setdefault(m["category"], []).append(m)
    
    all_records = []
    
    for cat, cat_manifests in by_cat.items():
        log(f"\n{'='*60}")
        log(f"Category: {cat} ({len(cat_manifests)} viable subgroups)")
        log(f"{'='*60}")
        
        subs_in_cat = sorted(m["subgroup"] for m in cat_manifests)
        
        if len(subs_in_cat) < 2:
            log(f"  SKIP: fewer than 2 viable subgroups")
            continue
        
        # Metadata and stimuli for category
        cat_meta = metadata_df[metadata_df["category"] == cat]
        cat_stimuli = stimuli_by_cat[cat]
        stimuli_by_idx = {s["item_idx"]: s for s in cat_stimuli}
        
        for source_m in cat_manifests:
            source_sub = source_m["subgroup"]
            
            # Load source's steering vector
            source_vec, source_inj_layer = load_steering_vector(
                run_dir, cat, source_sub, device, dtype
            )
            
            # Build steerer at source's injection layer
            source_steerer = SAESteerer(wrapper, sae_cache[source_inj_layer], source_inj_layer)
            
            for target_sub in subs_in_cat:
                is_self = (source_sub == target_sub)
                
                ckpt_path = transfer_dir / f"{cat}_{source_sub}_to_{target_sub}.json"
                if ckpt_path.exists():
                    with open(ckpt_path) as f:
                        record = json.load(f)
                    all_records.append(record)
                    log(f"  {source_sub}→{target_sub}: LOADED from checkpoint")
                    continue
                
                # Get ALL ambig items targeting target_sub (not just stereotyped-response ones)
                target_ambig_items = cat_meta[
                    (cat_meta["context_condition"] == "ambig") &
                    (cat_meta["stereotyped_groups"].apply(
                        lambda gs: target_sub in (gs if isinstance(gs, list) else json.loads(gs))
                    ))
                ]
                
                if args.max_items:
                    target_ambig_items = target_ambig_items.head(args.max_items)
                
                n_items = len(target_ambig_items)
                if n_items < args.min_n_per_target:
                    log(f"  {source_sub}→{target_sub}: SKIP (n={n_items} < {args.min_n_per_target})")
                    continue
                
                # Run evaluation
                log(f"  {source_sub}→{target_sub}: {n_items} items, inj_layer={source_inj_layer}")
                
                per_item_results = []
                for _, meta_row in target_ambig_items.iterrows():
                    item_idx = int(meta_row["item_idx"])
                    stim = stimuli_by_idx.get(item_idx)
                    if stim is None:
                        continue
                    
                    baseline = source_steerer.evaluate_baseline(stim["prompt"])
                    steered = source_steerer.steer_and_evaluate(stim["prompt"], source_vec)
                    
                    # Build result dict with the source's vector
                    item_for_result = {
                        "item_idx": item_idx,
                        "prompt": stim["prompt"],
                        "stereotyped_option": stim.get("stereotyped_option"),
                    }
                    r = build_result_dict(item_for_result, baseline, steered, source_vec)
                    r["target_subgroup"] = target_sub
                    r["source_subgroup"] = source_sub
                    r["category"] = cat
                    r["baseline_is_stereotyped_response"] = bool(
                        meta_row["model_answer_role"] == "stereotyped_target"
                    )
                    per_item_results.append(r)
                
                # Compute bias_change: change in stereotyped_response rate across ALL items
                n_baseline_stereo = sum(1 for r in per_item_results if r["baseline_role"] == "stereotyped_target")
                n_steered_stereo = sum(1 for r in per_item_results if r["steered_role"] == "stereotyped_target")
                baseline_stereo_rate = n_baseline_stereo / max(len(per_item_results), 1)
                steered_stereo_rate = n_steered_stereo / max(len(per_item_results), 1)
                bias_change = steered_stereo_rate - baseline_stereo_rate
                
                # Continuous metric: mean logit shift of stereotyped option
                logit_shifts = []
                for r in per_item_results:
                    opt = r.get("stereotyped_option")
                    if opt and opt in r.get("logit_baseline", {}) and opt in r.get("logit_steered", {}):
                        shift = r["logit_steered"][opt] - r["logit_baseline"][opt]
                        logit_shifts.append(shift)
                mean_logit_shift = float(np.mean(logit_shifts)) if logit_shifts else None
                
                # Standard metrics (computed on stereotyped-response baseline subset for RCR consistency)
                stereo_subset = [r for r in per_item_results if r["baseline_role"] == "stereotyped_target"]
                metrics = compute_all_metrics(stereo_subset) if stereo_subset else None
                
                # Degeneration/corruption tracking
                n_degen = sum(1 for r in per_item_results if r.get("degenerated", False))
                degen_rate = n_degen / max(len(per_item_results), 1)
                
                record = {
                    "category": cat,
                    "source_subgroup": source_sub,
                    "target_subgroup": target_sub,
                    "is_self": is_self,
                    "n_items": len(per_item_results),
                    "source_injection_layer": source_inj_layer,
                    "baseline_stereotyped_rate": round(baseline_stereo_rate, 4),
                    "steered_stereotyped_rate": round(steered_stereo_rate, 4),
                    "bias_change": round(bias_change, 4),
                    "mean_logit_shift_stereotyped_option": round(mean_logit_shift, 4) if mean_logit_shift is not None else None,
                    "degeneration_rate": round(degen_rate, 4),
                    "rcr_1.0": round(metrics["rcr_1.0"]["rcr"], 4) if metrics else None,
                    "rcr_1.0_n_eligible": metrics["rcr_1.0"]["n_eligible"] if metrics else 0,
                    "per_item_results": per_item_results,  # for consolidation
                }
                
                atomic_write_json(ckpt_path, record)
                all_records.append(record)
                
                log(f"    bias_change={bias_change:+.3f}, logit_shift={mean_logit_shift:+.3f}, "
                    f"RCR₁.₀={metrics['rcr_1.0']['rcr']:.3f if metrics else 'n/a'}, degen={degen_rate:.3f}")
    
    # Build flat DataFrame (excluding per_item_results — those go to separate parquet)
    flat_records = []
    for r in all_records:
        flat = {k: v for k, v in r.items() if k != "per_item_results"}
        flat_records.append(flat)
    
    return pd.DataFrame(flat_records)
```

**Why evaluate on all ambig items:** Allows backfire to be measurable. If source's vector pushes target's unknown/non-stereo items INTO stereotyped territory, that's a positive bias_change — backfire visible on the Y-axis. The original spec only evaluated on stereo-response items, which bounds bias_change at [-1, 0] and makes backfire invisible.

**Why still compute RCR on the stereo subset:** RCR is defined on items where a correction is possible (stereotyped baseline). Reporting it here for consistency with C1 metrics. But the **primary Y-axis is bias_change**, which is defined over all items.

---

## Step 5: Assemble Scatter Data

```python
def build_scatter_data(
    transfer_df: pd.DataFrame,
    primary_cosines: pd.DataFrame,
    sae_cosines: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge transfer results with cosines to produce scatter plot data.
    
    One row per (source, target) pair. Cosine is matched regardless of A/B order.
    """
    rows = []
    
    for _, t in transfer_df.iterrows():
        cat = t["category"]
        src = t["source_subgroup"]
        tgt = t["target_subgroup"]
        
        # Look up cosine (cosine_pairs has subgroup_A < subgroup_B alphabetically)
        cos_row_fwd = primary_cosines[
            (primary_cosines["category"] == cat) &
            (primary_cosines["subgroup_A"] == src) &
            (primary_cosines["subgroup_B"] == tgt)
        ]
        cos_row_rev = primary_cosines[
            (primary_cosines["category"] == cat) &
            (primary_cosines["subgroup_A"] == tgt) &
            (primary_cosines["subgroup_B"] == src)
        ]
        
        if len(cos_row_fwd) > 0:
            cos_dim = float(cos_row_fwd["cosine_dim_identity_normed"].iloc[0])
            peak_layer = int(cos_row_fwd["peak_layer"].iloc[0])
        elif len(cos_row_rev) > 0:
            cos_dim = float(cos_row_rev["cosine_dim_identity_normed"].iloc[0])
            peak_layer = int(cos_row_rev["peak_layer"].iloc[0])
        else:
            # Self-pair or missing cosine
            cos_dim = 1.0 if t["is_self"] else None
            peak_layer = None
        
        # SAE cosine (same lookup logic)
        sae_fwd = sae_cosines[
            (sae_cosines["category"] == cat) &
            (sae_cosines["subgroup_A"] == src) &
            (sae_cosines["subgroup_B"] == tgt)
        ]
        sae_rev = sae_cosines[
            (sae_cosines["category"] == cat) &
            (sae_cosines["subgroup_A"] == tgt) &
            (sae_cosines["subgroup_B"] == src)
        ]
        if len(sae_fwd) > 0:
            cos_sae = float(sae_fwd["cosine_sae_steering"].iloc[0])
        elif len(sae_rev) > 0:
            cos_sae = float(sae_rev["cosine_sae_steering"].iloc[0])
        else:
            cos_sae = 1.0 if t["is_self"] else None
        
        rows.append({
            "category": cat,
            "source_subgroup": src,
            "target_subgroup": tgt,
            "is_self": t["is_self"],
            "cosine_dim_identity_normed": cos_dim,
            "cosine_dim_peak_layer": peak_layer,
            "cosine_sae_steering": cos_sae,
            "bias_change": t["bias_change"],
            "baseline_stereotyped_rate": t["baseline_stereotyped_rate"],
            "steered_stereotyped_rate": t["steered_stereotyped_rate"],
            "mean_logit_shift": t["mean_logit_shift_stereotyped_option"],
            "degeneration_rate": t["degeneration_rate"],
            "rcr_1.0": t["rcr_1.0"],
            "n_items": t["n_items"],
        })
    
    return pd.DataFrame(rows)
```

---

## Step 6: Regression Analysis

Regress `bias_change` on cosine. **Exclude self-pairs from regression** but include in scatter visualization. Report multiple analyses.

```python
from scipy.stats import linregress

def run_regression_analyses(scatter_df: pd.DataFrame, args) -> dict:
    """
    Run multiple regression analyses:
      1. Primary: all categories, cosine_dim_identity_normed as X
      2. Sensitivity: excluding disability category
      3. Secondary: cosine_sae_steering as X (same filters)
      4. Per-source-subgroup: regression conditioning on source identity
      5. Per-category: regression within each category with ≥4 non-self pairs
    """
    # Exclude self-pairs and rows with missing cosines/bias_change
    non_self = scatter_df[~scatter_df["is_self"]].copy()
    non_self = non_self.dropna(subset=["cosine_dim_identity_normed", "bias_change"])
    
    results = {
        "n_total_non_self_pairs": int(len(non_self)),
    }
    
    # Primary regression (DIM cosine, all categories)
    results["primary_dim_all"] = do_regression(
        non_self["cosine_dim_identity_normed"].values,
        non_self["bias_change"].values,
        label="DIM cosine, all categories"
    )
    
    # Sensitivity: exclude disability
    no_dis = non_self[non_self["category"] != "disability"]
    if len(no_dis) >= 5:
        results["sensitivity_dim_no_disability"] = do_regression(
            no_dis["cosine_dim_identity_normed"].values,
            no_dis["bias_change"].values,
            label="DIM cosine, excluding disability"
        )
    
    # Secondary: SAE cosine
    has_sae = non_self.dropna(subset=["cosine_sae_steering"])
    if len(has_sae) >= 5:
        results["secondary_sae_all"] = do_regression(
            has_sae["cosine_sae_steering"].values,
            has_sae["bias_change"].values,
            label="SAE cosine, all categories"
        )
        
        has_sae_no_dis = has_sae[has_sae["category"] != "disability"]
        if len(has_sae_no_dis) >= 5:
            results["secondary_sae_no_disability"] = do_regression(
                has_sae_no_dis["cosine_sae_steering"].values,
                has_sae_no_dis["bias_change"].values,
                label="SAE cosine, excluding disability"
            )
    
    # Per-source-subgroup regressions
    per_source = {}
    for source_key, grp in non_self.groupby(["category", "source_subgroup"]):
        cat, src = source_key
        if len(grp) < 3:
            per_source[f"{cat}/{src}"] = {"n_pairs": len(grp), "status": "insufficient_data"}
            continue
        reg = do_regression(
            grp["cosine_dim_identity_normed"].values,
            grp["bias_change"].values,
            label=f"{cat}/{src}"
        )
        reg["n_pairs"] = len(grp)
        per_source[f"{cat}/{src}"] = reg
    results["per_source_subgroup"] = per_source
    
    # Per-category regressions
    per_category = {}
    for cat, grp in non_self.groupby("category"):
        if len(grp) < 4:
            per_category[cat] = {"n_pairs": len(grp), "status": "insufficient_data"}
            continue
        reg = do_regression(
            grp["cosine_dim_identity_normed"].values,
            grp["bias_change"].values,
            label=f"category={cat}"
        )
        reg["n_pairs"] = len(grp)
        per_category[cat] = reg
    results["per_category"] = per_category
    
    return results


def do_regression(x: np.ndarray, y: np.ndarray, label: str = "") -> dict:
    """Linear regression with bootstrap CI."""
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
    # Bootstrap for 95% CI on slope and intercept
    rng = np.random.default_rng(42)
    boot_slopes, boot_intercepts = [], []
    n = len(x)
    for _ in range(1000):
        idx = rng.choice(n, size=n, replace=True)
        s, i, _, _, _ = linregress(x[idx], y[idx])
        boot_slopes.append(s)
        boot_intercepts.append(i)
    
    return {
        "label": label,
        "n": int(n),
        "slope": float(slope),
        "intercept": float(intercept),
        "r_value": float(r_value),
        "r_squared": float(r_value ** 2),
        "p_value": float(p_value),
        "std_err": float(std_err),
        "slope_ci_95": [float(np.percentile(boot_slopes, 2.5)),
                        float(np.percentile(boot_slopes, 97.5))],
        "intercept_ci_95": [float(np.percentile(boot_intercepts, 2.5)),
                            float(np.percentile(boot_intercepts, 97.5))],
    }
```

---

## Step 7: SAE vs DIM Cosine Comparison

Diagnostic: do the SAE steering vectors capture the same pairwise geometry as raw DIM?

```python
def compare_sae_vs_dim_cosines(scatter_df: pd.DataFrame) -> dict:
    """
    Correlate SAE cosines with DIM cosines across all pairs.
    
    High correlation (>0.8): SAE preserves DIM geometry.
    Low correlation (<0.5): SAE imposes different structure.
    """
    paired = scatter_df[~scatter_df["is_self"]].dropna(
        subset=["cosine_dim_identity_normed", "cosine_sae_steering"]
    )
    
    if len(paired) < 5:
        return {"status": "insufficient_data", "n_pairs": len(paired)}
    
    x = paired["cosine_dim_identity_normed"].values
    y = paired["cosine_sae_steering"].values
    
    pearson = float(np.corrcoef(x, y)[0, 1])
    
    # Also Spearman (rank-based, more robust)
    from scipy.stats import spearmanr
    spearman_rho, spearman_p = spearmanr(x, y)
    
    return {
        "n_pairs": int(len(paired)),
        "pearson_r": round(pearson, 4),
        "spearman_rho": round(float(spearman_rho), 4),
        "spearman_p": float(spearman_p),
        "mean_dim_cosine": round(float(x.mean()), 4),
        "mean_sae_cosine": round(float(y.mean()), 4),
    }
```

---

## Step 8: Stable-Range Sensitivity Analysis

Verify the primary finding isn't dependent on the specific "peak layer" choice.

```python
def stable_range_sensitivity(
    run_dir: Path,
    transfer_df: pd.DataFrame,
    scatter_df: pd.DataFrame,
) -> dict:
    """
    Re-run primary regression at each layer in B3's stable range per category.
    
    Uses DIM identity_normed cosines at each layer instead of just the peak layer.
    Reports r² and slope across all stable-range layers.
    """
    cosine_df = pd.read_parquet(run_dir / "B_geometry" / "cosine_pairs.parquet")
    
    with open(run_dir / "B_geometry" / "differentiation_metrics.json") as f:
        diff_metrics = json.load(f)
    
    results = {}
    
    for cat, cat_metrics in diff_metrics.items():
        if "identity_normed" not in cat_metrics:
            continue
        stable_range = cat_metrics["identity_normed"]["stable_range"]
        stable_start, stable_end = stable_range
        
        cat_transfers = transfer_df[transfer_df["category"] == cat]
        if len(cat_transfers) < 4:
            continue
        
        per_layer = {}
        for layer in range(stable_start, stable_end + 1):
            layer_cos = cosine_df[
                (cosine_df["category"] == cat) &
                (cosine_df["direction_type"] == "identity_normed") &
                (cosine_df["layer"] == layer)
            ]
            
            # Merge transfer data with layer-specific cosines
            merged_rows = []
            for _, t in cat_transfers.iterrows():
                if t["is_self"]:
                    continue
                src, tgt = t["source_subgroup"], t["target_subgroup"]
                cos_fwd = layer_cos[
                    (layer_cos["subgroup_A"] == src) &
                    (layer_cos["subgroup_B"] == tgt)
                ]
                cos_rev = layer_cos[
                    (layer_cos["subgroup_A"] == tgt) &
                    (layer_cos["subgroup_B"] == src)
                ]
                if len(cos_fwd) > 0:
                    cos = float(cos_fwd["cosine"].iloc[0])
                elif len(cos_rev) > 0:
                    cos = float(cos_rev["cosine"].iloc[0])
                else:
                    continue
                merged_rows.append((cos, t["bias_change"]))
            
            if len(merged_rows) < 4:
                continue
            
            x = np.array([m[0] for m in merged_rows])
            y = np.array([m[1] for m in merged_rows])
            reg = do_regression(x, y, label=f"{cat} L{layer}")
            per_layer[layer] = {
                "r_squared": reg["r_squared"],
                "slope": reg["slope"],
                "p_value": reg["p_value"],
                "n": reg["n"],
            }
        
        if per_layer:
            results[cat] = {
                "stable_range": stable_range,
                "per_layer": per_layer,
                "r_squared_min": min(v["r_squared"] for v in per_layer.values()),
                "r_squared_max": max(v["r_squared"] for v in per_layer.values()),
                "r_squared_mean": float(np.mean([v["r_squared"] for v in per_layer.values()])),
            }
    
    return results
```

---

## Output Files

### `transfer_pairs.parquet`

Primary flat output. One row per (source, target) pair.

```
{run}/C_transfer/transfer_pairs.parquet
```

| Column | Type | Description |
|---|---|---|
| `category` | string | |
| `source_subgroup` | string | Subgroup whose steering vector was applied |
| `target_subgroup` | string | Subgroup whose items were evaluated |
| `is_self` | bool | source == target |
| `n_items` | int32 | Items evaluated (all ambig items targeting target) |
| `source_injection_layer` | int32 | Layer at which source's vector was injected |
| `baseline_stereotyped_rate` | float32 | Fraction of items with stereo baseline response |
| `steered_stereotyped_rate` | float32 | Fraction with stereo response after steering |
| `bias_change` | float32 | Signed change; positive = backfire, negative = debiasing |
| `mean_logit_shift` | float32 | Mean shift of stereotyped option logit |
| `degeneration_rate` | float32 | Fraction degenerated |
| `rcr_1.0` | float32 | RCR on stereo-baseline subset (or null if none) |

### `per_pair/` — per-pair per-item results

```
{run}/C_transfer/per_pair/{cat}_{source}_to_{target}.parquet
```

Flat parquet per pair with item-level details. Columns: `item_idx`, `category`, `source_subgroup`, `target_subgroup`, `baseline_answer`, `steered_answer`, `baseline_role`, `steered_role`, `corrected`, `corrupted`, `degenerated`, `margin`, `margin_bin`, logit columns, `baseline_is_stereotyped_response`, `stereotyped_option`.

### `cosines.parquet`

Merged primary (DIM) and secondary (SAE) cosines per pair.

```
{run}/C_transfer/cosines.parquet
```

| Column | Type |
|---|---|
| `category` | string |
| `subgroup_A` | string |
| `subgroup_B` | string |
| `cosine_dim_identity_normed` | float32 |
| `cosine_dim_peak_layer` | int32 |
| `cosine_sae_steering` | float32 |

### `scatter_data.parquet`

Final analysis-ready data for the scatter plot.

```
{run}/C_transfer/scatter_data.parquet
```

Columns: `category`, `source_subgroup`, `target_subgroup`, `is_self`, `cosine_dim_identity_normed`, `cosine_dim_peak_layer`, `cosine_sae_steering`, `bias_change`, `baseline_stereotyped_rate`, `steered_stereotyped_rate`, `mean_logit_shift`, `degeneration_rate`, `rcr_1.0`, `n_items`.

### `regression_results.json`

```json
{
  "n_total_non_self_pairs": 87,
  "primary_dim_all": {
    "label": "DIM cosine, all categories",
    "n": 87,
    "slope": -0.412,
    "intercept": 0.057,
    "r_squared": 0.538,
    "p_value": 2.1e-15,
    "slope_ci_95": [-0.487, -0.334],
    "intercept_ci_95": [0.025, 0.089]
  },
  "sensitivity_dim_no_disability": {
    "r_squared": 0.521,
    "slope": -0.398,
    "p_value": 4.3e-13,
    "n": 78,
    ...
  },
  "secondary_sae_all": {
    "r_squared": 0.412,
    "slope": -0.287,
    ...
  },
  "secondary_sae_no_disability": {...},
  "per_source_subgroup": {
    "so/gay": {"n_pairs": 3, "r_squared": 0.72, "slope": -0.51, ...},
    "so/bisexual": {...},
    ...
  },
  "per_category": {
    "so": {"n_pairs": 12, "r_squared": 0.64, "slope": -0.48, ...},
    "race": {...},
    ...
  }
}
```

### `sae_vs_dim_comparison.json`

```json
{
  "n_pairs": 87,
  "pearson_r": 0.791,
  "spearman_rho": 0.763,
  "spearman_p": 3.2e-17,
  "mean_dim_cosine": 0.187,
  "mean_sae_cosine": 0.201
}
```

### `stable_range_sensitivity.json`

Per-category, the r² for the primary regression across each layer in B3's stable range. Demonstrates robustness of the finding to layer choice.

```json
{
  "so": {
    "stable_range": [11, 19],
    "per_layer": {
      "11": {"r_squared": 0.48, "slope": -0.38, "p_value": 0.002, "n": 12},
      "12": {"r_squared": 0.52, "slope": -0.41, ...},
      ...
      "19": {"r_squared": 0.56, "slope": -0.43, ...}
    },
    "r_squared_min": 0.48,
    "r_squared_max": 0.59,
    "r_squared_mean": 0.53
  },
  ...
}
```

### `c2_summary.json`

```json
{
  "config": {...},
  "n_viable_subgroups_used": 32,
  "n_total_non_self_pairs": 87,
  "n_pairs_skipped_insufficient_items": 3,
  "primary_finding": {
    "r_squared": 0.538,
    "slope": -0.412,
    "p_value": 2.1e-15,
    "conclusion": "Strong negative relationship between representational similarity and transfer benefit (backfire for anti-correlated pairs)"
  },
  "sae_vs_dim_pearson_r": 0.791,
  "per_source_subgroup_regression_r_squared_median": 0.48,
  "runtime_seconds": 8432.0
}
```

---

## Figures

All figures use Wong colorblind-safe palette. Distinct markers per category.

### `fig_universal_backfire_scatter.png`

The headline figure. Two panels side by side.

**Panel A (all categories):**
- X: `cosine_dim_identity_normed` (-1 to 1)
- Y: `bias_change` (negative = debiasing, positive = backfire)
- Points: every (source, target) pair including self-pairs
- Self-pairs marked with large gold stars, excluded from regression
- Non-self pairs colored by category (Wong palette), with distinct markers
- Marker size proportional to `log(n_items)` for visual clarity
- OLS regression line (non-self only) in black, dashed
- 95% bootstrap CI band as gray shading
- Horizontal dashed line at y=0 (no effect)
- Vertical dashed line at x=0 (orthogonal directions)
- Quadrant labels:
  - Upper-left: "BACKFIRE"
  - Lower-right: "CROSS-DEBIASING"
- Bottom-right annotation: `r² = 0.54, p = 2.1e-15, n = 87`
- Title: "Panel A: All categories"

**Panel B (excluding disability):**
- Same axes and content, but filtering out disability
- Separate regression and CI
- Bottom-right: `r² = 0.52, p = 4.3e-13, n = 78`
- Title: "Panel B: Excluding disability"

Legend below the two panels, shared (category color + marker key).

### `fig_cosine_vs_backfire_by_category.png`

Faceted scatter. One panel per category with ≥4 non-self pairs.
- Same axes as universal scatter
- Per-category regression line
- `r²` annotated
- Self-pairs shown as gray stars (not in regression)
- Title per panel: category name

### `fig_cosine_vs_backfire_by_source_subgroup.png`

Faceted scatter. One panel per source subgroup with ≥3 target pairs.
- X: DIM cosine between this source and the target
- Y: bias_change when applying this source's vector to the target
- One point per target; color by target subgroup
- Per-source regression line
- Self-pair shown as large star at x=1
- Title per panel: source subgroup name
- Annotated `r²` and `slope` per panel

This figure directly addresses the per-source analysis — shows whether the relationship holds uniformly or is driven by specific subgroups.

### `fig_transfer_heatmaps.png`

Grid of heatmaps, one per category with ≥2 viable subgroups.
- Rows: source subgroup
- Columns: target subgroup
- Color: `bias_change` (RdBu_r diverging, vmin=-1, vmax=1)
- Diagonal (self-pairs): should be strongly blue (negative bias_change, debiasing)
- Annotated cells with values to 2 decimals
- Panel labels (A, B, C, ...)
- Shared colorbar
- Title per panel: category name

### `fig_sae_vs_dim_cosine.png`

Single scatter.
- X: `cosine_dim_identity_normed`
- Y: `cosine_sae_steering`
- One point per non-self pair
- Color by category
- Diagonal reference line y=x
- Annotated Pearson r and Spearman ρ
- Title: "SAE steering vector geometry vs. DIM identity geometry"

### `fig_stable_range_robustness.png`

One subplot per category in the stable-range sensitivity analysis.
- X: layer (across stable range)
- Y (primary): `r_squared` of bias_change ~ cosine_dim regression at that layer
- Y (secondary): slope of regression
- Shaded range: stable range
- Vertical dashed line at peak layer
- Shows robustness of finding across layers

---

## Output Structure

```
{run}/C_transfer/
├── per_pair_checkpoints/
│   ├── {cat}_{source}_to_{target}.json
│   └── ...
├── per_pair/                          # Consolidated per-pair per-item parquets
│   ├── {cat}_{source}_to_{target}.parquet
│   └── ...
├── transfer_pairs.parquet              # Flat summary, one row per pair
├── cosines.parquet                     # Merged DIM and SAE cosines
├── scatter_data.parquet                # Analysis-ready scatter input
├── regression_results.json             # All regression analyses
├── sae_vs_dim_comparison.json          # Cosine correlation diagnostic
├── stable_range_sensitivity.json       # Layer robustness analysis
├── c2_summary.json                     # Top-level summary
└── figures/
    ├── fig_universal_backfire_scatter.png/.pdf
    ├── fig_cosine_vs_backfire_by_category.png/.pdf
    ├── fig_cosine_vs_backfire_by_source_subgroup.png/.pdf
    ├── fig_transfer_heatmaps.png/.pdf
    ├── fig_sae_vs_dim_cosine.png/.pdf
    └── fig_stable_range_robustness.png/.pdf
```

---

## Resume Safety

**Per-pair checkpoints.** Before each (source, target) evaluation, check for the JSON checkpoint. Skip if present. Atomic writes ensure no corruption on crash.

**SAE cache shared across pairs.** Loaded once at startup for all needed injection layers.

**Cosines + regression analyses are cheap.** Re-run on every C2 invocation. No caching.

---

## Compute Estimate (M4 Max MPS)

- Baseline per item: ~0.2s
- Steered per item: ~0.3s
- Per pair (~100 items, 2 passes each): ~50s
- Non-self pairs: ~40 viable subgroups × avg 4 in-category = ~12 pairs per category × 9 categories = ~120 non-self + 40 self = ~160 pairs
- Total: 160 × 50s = ~2.2 hours MPS, ~35 min CUDA

---

## Assumptions Summary

| # | Decision | Value |
|---|---|---|
| 1 | Primary X-axis | DIM identity_normed cosines |
| 2 | Primary Y-axis | bias_change (allows backfire as positive) |
| 3 | Items evaluated | ALL ambig items targeting target subgroup |
| 4 | Layer for primary cosines | Peak differentiation layer per category (from B3) |
| 5 | Secondary X-axis | SAE steering vector cosines (diagnostic) |
| 6 | Self-pairs | Included as data points; EXCLUDED from regression |
| 7 | Minimum N per target | 10 items (configurable) |
| 8 | Regression variants | Primary, no-disability sensitivity, SAE secondary, per-source, per-category |
| 9 | Stable-range sensitivity | r² reported at each layer in B3's stable range per category |
| 10 | Output format | Parquets for flat data, JSON for nested/regression results |
| 11 | Viable subgroups filter | Only process subgroups with steering_viable=True in C1 |
| 12 | Bootstrap CI | 1000 resamples, fixed seed |

---

## Test Commands

```bash
# Quick test: single category with 20 items per pair
python scripts/C2_transfer.py \
    --run_dir runs/llama-3.1-8b_2026-04-15/ \
    --categories so \
    --max_items 20

# Verify outputs
python -c "
import pandas as pd
import json
import numpy as np

# Scatter data
sd = pd.read_parquet('runs/llama-3.1-8b_2026-04-15/C_transfer/scatter_data.parquet')
print(f'Scatter pairs: {len(sd)}')
print(f'Self pairs: {sd[\"is_self\"].sum()}')
print(f'Non-self with DIM cosine: {(~sd[\"is_self\"] & sd[\"cosine_dim_identity_normed\"].notna()).sum()}')

# Primary regression
with open('runs/llama-3.1-8b_2026-04-15/C_transfer/regression_results.json') as f:
    reg = json.load(f)
p = reg['primary_dim_all']
print(f'\\nPrimary DIM regression:')
print(f'  r² = {p[\"r_squared\"]:.3f}, slope = {p[\"slope\"]:+.3f}, p = {p[\"p_value\"]:.2e}, n = {p[\"n\"]}')

# SAE vs DIM
with open('runs/llama-3.1-8b_2026-04-15/C_transfer/sae_vs_dim_comparison.json') as f:
    comp = json.load(f)
print(f'\\nSAE vs DIM cosine correlation: Pearson={comp[\"pearson_r\"]:.3f}, Spearman={comp[\"spearman_rho\"]:.3f}')

# Count backfire pairs
non_self = sd[~sd['is_self']]
backfire = non_self[non_self['bias_change'] > 0]
print(f'\\nBackfire pairs (bias_change > 0): {len(backfire)}/{len(non_self)}')
if len(backfire) > 0:
    print('Top backfire pairs:')
    top = backfire.nlargest(5, 'bias_change')[['category', 'source_subgroup', 'target_subgroup', 'cosine_dim_identity_normed', 'bias_change']]
    print(top.to_string(index=False))
"
```