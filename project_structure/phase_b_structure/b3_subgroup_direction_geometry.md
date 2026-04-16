# B3: Subgroup Direction Geometry — Full Implementation Specification

## Purpose

Compute Difference-in-Means (DIM) directions for each subgroup from raw activation differences, completely independent of the SAE. Produces two types of directions (bias, identity) under two normalization regimes (raw-based, normed-based), plus pairwise cosine matrices and bias-identity alignment scores. This provides a second geometric view of subgroup representations that cross-validates against B1/B2's SAE-based analysis.

## Invocation

```bash
python scripts/B3_geometry.py --run_dir runs/llama-3.1-8b_2026-04-15/

# Quick test
python scripts/B3_geometry.py --run_dir runs/llama-3.1-8b_2026-04-15/ --max_items 100

# Single category
python scripts/B3_geometry.py --run_dir runs/llama-3.1-8b_2026-04-15/ --categories so

# Override minimum sample size
python scripts/B3_geometry.py --run_dir runs/llama-3.1-8b_2026-04-15/ --min_n_per_group 15
```

Reads `categories`, `n_layers`, `hidden_dim` from config.json.

---

## Input

- Activations: `{run}/A_extraction/activations/{category}/item_*.npz`
- Metadata parquet: `{run}/A_extraction/metadata.parquet`
- Stimuli: `{run}/A_extraction/stimuli/{category}.json` (for subgroup membership)

Does NOT use SAE encodings. Independent of A3/B1/B2.

## Dependencies

- `numpy`
- `pandas`, `pyarrow`
- `matplotlib` (for figures)

---

## Two Direction Types × Two Normalization Regimes = Four Directions Per Subgroup

### Bias direction

"Which direction distinguishes stereotyped-response hidden states from non-stereotyped-response hidden states, among items targeting S?"

```
bias_direction(S, layer) = mean(h for items targeting S, ambig, stereotyped response)
                         - mean(h for items targeting S, ambig, non-stereotyped response)
```

Used by C1 for steering (against this direction reduces bias). Steering magnitude matters, so the **raw-based** version is primary for C1.

### Identity direction

"Which direction distinguishes content about subgroup S from content about other subgroups in the same category?"

```
identity_direction(S, layer) = mean(h for items targeting S)
                             - mean(h for items NOT targeting S in same category)
```

Used by C2 for cosine geometry (the X-axis of the universal backfire scatter). Angular/geometric comparisons are what matter, so the **normed-based** version is primary for C2.

**"NOT targeting S" — granular definition:** An item is in the "not targeting S" group iff S is not in `stereotyped_groups`. Items targeting overlapping labels (e.g., "physically disabled" when S = "disabled") contribute to the "not targeting disabled" group because "disabled" is not in their stereotyped_groups list. This uses the most granular representation — each subgroup gets its own clean contrast, even when labels structurally overlap.

### Raw-based vs Normed-based DIM

Two ways to compute the mean per group:

**Raw-based DIM:**
```python
# Reconstruct raw activations first, then take mean
hs_raw = hs_normed * raw_norms[:, None]
mean_raw = hs_raw[mask].mean(axis=0)
direction_raw = mean_stereo_raw - mean_non_stereo_raw
direction_raw_unit = direction_raw / ||direction_raw||
```

Items with larger-magnitude hidden states contribute more to the mean. Captures magnitude-weighted differences. Used for steering (C1), where the direction is added as `α × direction` to the residual stream.

**Normed-based DIM:**
```python
# Use pre-normalized (unit-sphere) hidden states
hs_normed = data["hidden_states"].astype(np.float32)  # already unit-per-layer
mean_normed = hs_normed[mask].mean(axis=0)
direction_normed = mean_stereo_normed - mean_non_stereo_normed
direction_normed_unit = direction_normed / ||direction_normed||
```

Each item contributes equally regardless of magnitude. Captures angular structure only. Used for cosine geometry (C2).

**All four directions are computed and stored per subgroup per layer:**
- `bias_direction_raw_{cat}_{sub}`
- `bias_direction_normed_{cat}_{sub}`
- `identity_direction_raw_{cat}_{sub}`
- `identity_direction_normed_{cat}_{sub}`

Each is shape `(n_layers, hidden_dim)`, float32, unit-normalized per-layer.

**Also stored: raw (pre-normalization) magnitudes:**
- `bias_direction_raw_norm_{cat}_{sub}` — shape `(n_layers,)`, the ||mean_stereo - mean_non_stereo|| before unit-normalization
- `bias_direction_normed_norm_{cat}_{sub}` — same, computed on normed activations
- `identity_direction_raw_norm_{cat}_{sub}`
- `identity_direction_normed_norm_{cat}_{sub}`

These tell you how much signal is in each direction BEFORE normalization. A direction with raw_norm = 0.01 relative to hidden state magnitudes (~20-50) is noise. A direction with raw_norm = 3.0 is substantive.

---

## Minimum Item Counts

Matches B1 conventions:
- `min_n_per_group = 10` (default, configurable via `--min_n_per_group`)

### For bias direction computation:
- Require ≥10 items in both stereo-response group AND non-stereo-response group
- Otherwise, skip bias direction for that subgroup, log as skipped

### For identity direction computation:
- Require ≥10 items in both "targeting S" and "NOT targeting S" groups
- Otherwise, skip identity direction for that subgroup, log as skipped

Subgroups may have bias_direction computable but not identity_direction (e.g., a category with only one large subgroup — nothing to contrast identity against). Or vice versa. The output JSON clearly indicates which directions each subgroup has.

---

## Script Structure

```python
def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = load_config(run_dir)
    
    categories = parse_categories(args.categories) if args.categories else config["categories"]
    min_n = args.min_n_per_group  # default 10
    max_items = args.max_items
    
    n_layers = config["n_layers"]
    hidden_dim = config["hidden_dim"]
    
    # Load metadata once
    meta_df = load_metadata(run_dir)
    
    # Storage for all directions
    directions_arrays = {}        # key: "{type}_{normalize}_{cat}_{sub}"
    directions_norms = {}         # key: same, value: (n_layers,) raw magnitude pre-normalization
    subgroup_info = {}            # per (cat, sub): n_items, n_stereo, etc.
    
    for cat in categories:
        process_category(
            cat=cat,
            run_dir=run_dir,
            meta_df=meta_df,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            min_n=min_n,
            max_items=max_items,
            directions_arrays=directions_arrays,
            directions_norms=directions_norms,
            subgroup_info=subgroup_info,
        )
    
    # Save directions as single .npz
    save_directions(run_dir, directions_arrays, directions_norms)
    
    # Compute pairwise cosines and save as parquet
    cosine_df = compute_all_cosines(directions_arrays, categories, n_layers)
    save_cosines(run_dir, cosine_df)
    
    # Compute differentiation metrics per category
    differentiation = compute_differentiation_metrics(cosine_df, categories, n_layers)
    
    # Compute bias-identity alignment
    alignment = compute_alignment(directions_arrays, categories, n_layers)
    
    # Save summary
    save_summary(run_dir, subgroup_info, differentiation, alignment, min_n)
    
    # Generate figures
    if not args.skip_figures:
        generate_figures(run_dir, directions_arrays, cosine_df, differentiation, alignment)
```

---

## Per-Category Processing

```python
def process_category(
    cat: str,
    run_dir: Path,
    meta_df: pd.DataFrame,
    n_layers: int,
    hidden_dim: int,
    min_n: int,
    max_items: int | None,
    directions_arrays: dict,
    directions_norms: dict,
    subgroup_info: dict,
):
    """Load all items for category, compute directions for all subgroups."""
    log(f"\n{'='*60}")
    log(f"Category: {cat}")
    log(f"{'='*60}")
    
    # Filter metadata to this category
    cat_meta = meta_df[meta_df["category"] == cat].copy()
    
    # Parse stereotyped_groups from JSON strings (should already be lists after load_metadata)
    # But ensure:
    cat_meta["stereotyped_groups"] = cat_meta["stereotyped_groups"].apply(
        lambda x: x if isinstance(x, list) else json.loads(x)
    )
    
    if max_items:
        cat_meta = cat_meta.head(max_items)
    
    # Load all hidden states for this category in one pass
    log(f"  Loading activations...")
    all_hs_normed, all_norms, item_idxs = load_category_hidden_states(
        run_dir, cat, cat_meta["item_idx"].tolist(), n_layers, hidden_dim
    )
    # Shape: (n_items, n_layers, hidden_dim) float32, and (n_items, n_layers) float32
    
    log(f"  Loaded {len(item_idxs)} items, hidden states shape {all_hs_normed.shape}")
    
    # Reconstruct raw activations (in-place where possible)
    all_hs_raw = all_hs_normed * all_norms[:, :, None]  # broadcast norms over hidden_dim
    
    # Build lookup: item_idx → position in all_hs arrays
    idx_to_pos = {idx: i for i, idx in enumerate(item_idxs)}
    
    # Reorder cat_meta to match
    cat_meta = cat_meta.set_index("item_idx").loc[item_idxs].reset_index()
    
    # Identify all subgroups in this category (from stereotyped_groups values)
    all_subgroups = set()
    for gs in cat_meta["stereotyped_groups"]:
        all_subgroups.update(gs)
    all_subgroups = sorted(all_subgroups)
    
    log(f"  Subgroups: {all_subgroups}")
    
    # Compute directions for each subgroup
    for sub in all_subgroups:
        result = compute_subgroup_directions(
            cat=cat,
            sub=sub,
            cat_meta=cat_meta,
            all_hs_normed=all_hs_normed,
            all_hs_raw=all_hs_raw,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            min_n=min_n,
        )
        
        if result is None:
            log(f"    {sub}: insufficient data, all directions skipped")
            subgroup_info[(cat, sub)] = {"category": cat, "subgroup": sub, "skipped": True}
            continue
        
        # Store arrays
        for key, arr in result["arrays"].items():
            directions_arrays[f"{key}_{cat}_{sub}"] = arr
        for key, arr in result["norms"].items():
            directions_norms[f"{key}_{cat}_{sub}"] = arr
        
        # Store info
        subgroup_info[(cat, sub)] = result["info"]
        
        log(f"    {sub}: {result['info']}")
```

### Loading Hidden States Efficiently

```python
def load_category_hidden_states(
    run_dir: Path,
    cat: str,
    item_idxs: list[int],
    n_layers: int,
    hidden_dim: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Load hidden states for all items in a category.
    
    Returns:
        all_hs_normed: (n_items, n_layers, hidden_dim), float32
        all_norms: (n_items, n_layers), float32
        loaded_idxs: item_idx list in loaded order (may be shorter if some files missing)
    """
    cat_dir = run_dir / "A_extraction" / "activations" / cat
    
    hs_list = []
    norms_list = []
    loaded_idxs = []
    
    for idx in item_idxs:
        npz_path = cat_dir / f"item_{idx:06d}.npz"
        if not npz_path.exists():
            log(f"    WARNING: missing {npz_path.name}")
            continue
        
        data = np.load(npz_path, allow_pickle=True)
        hs_list.append(data["hidden_states"].astype(np.float32))  # (n_layers, hidden_dim)
        norms_list.append(data["hidden_states_raw_norms"].astype(np.float32))  # (n_layers,)
        loaded_idxs.append(idx)
    
    if not hs_list:
        return np.zeros((0, n_layers, hidden_dim), dtype=np.float32), np.zeros((0, n_layers), dtype=np.float32), []
    
    all_hs_normed = np.stack(hs_list, axis=0)  # (n_items, n_layers, hidden_dim)
    all_norms = np.stack(norms_list, axis=0)   # (n_items, n_layers)
    
    return all_hs_normed, all_norms, loaded_idxs
```

**Memory estimate:** For ~8640 items × 32 layers × 4096 dim × 4 bytes (float32) = ~4.3 GB per category for `all_hs_normed`. Plus the same again for `all_hs_raw`. Total ~8.6 GB per category, trivially fits in 128GB RAM. Processing is category-by-category so this is the peak footprint.

For larger models (Llama-3.1-70B with 8192 hidden dim and 80 layers), memory would be ~43 GB per category — still manageable on the 128GB machine but needing careful handling. For now (8B model), no concern.

---

## Per-Subgroup Direction Computation

```python
def compute_subgroup_directions(
    cat: str,
    sub: str,
    cat_meta: pd.DataFrame,
    all_hs_normed: np.ndarray,
    all_hs_raw: np.ndarray,
    n_layers: int,
    hidden_dim: int,
    min_n: int,
) -> dict | None:
    """
    Compute all four directions for one subgroup.
    
    Returns dict with:
        "arrays": {"bias_direction_raw": ..., "bias_direction_normed": ...,
                   "identity_direction_raw": ..., "identity_direction_normed": ...}
        "norms":  {"bias_direction_raw_norm": ..., ...}
        "info":   subgroup info dict for summary
    
    Or None if all directions were skipped.
    """
    n_items = len(cat_meta)
    
    # Build boolean masks
    is_S = cat_meta["stereotyped_groups"].apply(lambda gs: sub in gs).values  # (n_items,)
    is_ambig = (cat_meta["context_condition"] == "ambig").values
    is_stereo = (cat_meta["model_answer_role"] == "stereotyped_target").values
    
    # Bias direction masks: items targeting S, ambig condition, split by response
    bias_stereo_mask = is_S & is_ambig & is_stereo
    bias_non_stereo_mask = is_S & is_ambig & ~is_stereo
    
    n_bias_stereo = int(bias_stereo_mask.sum())
    n_bias_non_stereo = int(bias_non_stereo_mask.sum())
    
    # Identity direction masks: items targeting S vs items NOT targeting S (all context conditions)
    # "NOT targeting S" means S is not in stereotyped_groups, even if overlapping labels are present
    identity_S_mask = is_S
    identity_not_S_mask = ~is_S
    
    n_identity_S = int(identity_S_mask.sum())
    n_identity_not_S = int(identity_not_S_mask.sum())
    
    # Compute bias directions if enough data
    bias_raw_unit = None
    bias_normed_unit = None
    bias_raw_prenorm = None
    bias_normed_prenorm = None
    bias_ok = n_bias_stereo >= min_n and n_bias_non_stereo >= min_n
    
    if bias_ok:
        # Raw-based bias direction
        mean_stereo_raw = all_hs_raw[bias_stereo_mask].mean(axis=0)      # (n_layers, hidden_dim)
        mean_non_stereo_raw = all_hs_raw[bias_non_stereo_mask].mean(axis=0)
        bias_raw = mean_stereo_raw - mean_non_stereo_raw                  # (n_layers, hidden_dim)
        bias_raw_prenorm = np.linalg.norm(bias_raw, axis=1)               # (n_layers,)
        safe = np.maximum(bias_raw_prenorm, 1e-8)[:, None]
        bias_raw_unit = (bias_raw / safe).astype(np.float32)
        
        # Normed-based bias direction
        mean_stereo_normed = all_hs_normed[bias_stereo_mask].mean(axis=0)
        mean_non_stereo_normed = all_hs_normed[bias_non_stereo_mask].mean(axis=0)
        bias_normed = mean_stereo_normed - mean_non_stereo_normed
        bias_normed_prenorm = np.linalg.norm(bias_normed, axis=1)
        safe_n = np.maximum(bias_normed_prenorm, 1e-8)[:, None]
        bias_normed_unit = (bias_normed / safe_n).astype(np.float32)
    
    # Compute identity directions if enough data
    identity_raw_unit = None
    identity_normed_unit = None
    identity_raw_prenorm = None
    identity_normed_prenorm = None
    identity_ok = n_identity_S >= min_n and n_identity_not_S >= min_n
    
    if identity_ok:
        mean_S_raw = all_hs_raw[identity_S_mask].mean(axis=0)
        mean_not_S_raw = all_hs_raw[identity_not_S_mask].mean(axis=0)
        identity_raw = mean_S_raw - mean_not_S_raw
        identity_raw_prenorm = np.linalg.norm(identity_raw, axis=1)
        safe = np.maximum(identity_raw_prenorm, 1e-8)[:, None]
        identity_raw_unit = (identity_raw / safe).astype(np.float32)
        
        mean_S_normed = all_hs_normed[identity_S_mask].mean(axis=0)
        mean_not_S_normed = all_hs_normed[identity_not_S_mask].mean(axis=0)
        identity_normed = mean_S_normed - mean_not_S_normed
        identity_normed_prenorm = np.linalg.norm(identity_normed, axis=1)
        safe_n = np.maximum(identity_normed_prenorm, 1e-8)[:, None]
        identity_normed_unit = (identity_normed / safe_n).astype(np.float32)
    
    # If nothing computable, return None
    if not bias_ok and not identity_ok:
        return None
    
    # Package results
    arrays = {}
    norms = {}
    
    if bias_ok:
        arrays["bias_direction_raw"] = bias_raw_unit
        arrays["bias_direction_normed"] = bias_normed_unit
        norms["bias_direction_raw_norm"] = bias_raw_prenorm.astype(np.float32)
        norms["bias_direction_normed_norm"] = bias_normed_prenorm.astype(np.float32)
    
    if identity_ok:
        arrays["identity_direction_raw"] = identity_raw_unit
        arrays["identity_direction_normed"] = identity_normed_unit
        norms["identity_direction_raw_norm"] = identity_raw_prenorm.astype(np.float32)
        norms["identity_direction_normed_norm"] = identity_normed_prenorm.astype(np.float32)
    
    info = {
        "category": cat,
        "subgroup": sub,
        "skipped": False,
        "n_total_targeting_S": n_identity_S,
        "n_total_not_targeting_S": n_identity_not_S,
        "n_bias_stereo": n_bias_stereo,
        "n_bias_non_stereo": n_bias_non_stereo,
        "bias_computed": bias_ok,
        "identity_computed": identity_ok,
    }
    
    return {"arrays": arrays, "norms": norms, "info": info}
```

---

## Saving Directions

Single .npz containing all directions for all subgroups:

```python
def save_directions(
    run_dir: Path,
    directions_arrays: dict[str, np.ndarray],
    directions_norms: dict[str, np.ndarray],
):
    """Save all directions and their pre-normalization norms to a single .npz."""
    out_dir = ensure_dir(run_dir / "B_geometry")
    out_path = out_dir / "subgroup_directions.npz"
    
    combined = {**directions_arrays, **directions_norms}
    
    tmp_path = out_path.with_suffix(".npz.tmp")
    np.savez(tmp_path, **combined)
    tmp_path.rename(out_path)
    
    log(f"Saved {len(combined)} arrays to {out_path}")
```

Keys in the .npz follow the pattern:
- `{direction_type}_{normalize}_{category}_{subgroup}` for unit-normalized directions
  - e.g., `bias_direction_raw_so_gay`, `identity_direction_normed_race_black`
- `{direction_type}_{normalize}_norm_{category}_{subgroup}` for pre-normalization magnitudes
  - e.g., `bias_direction_raw_norm_so_gay`

**Loading helper** (for downstream scripts):

```python
def load_direction(
    run_dir: Path,
    direction_type: str,     # "bias" or "identity"
    normalize: str,           # "raw" or "normed"
    category: str,
    subgroup: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Load a direction and its pre-normalization norm.
    
    Returns:
        (unit_direction, raw_norm) or (None, None) if not present
        unit_direction: (n_layers, hidden_dim) float32
        raw_norm: (n_layers,) float32
    """
    npz_path = run_dir / "B_geometry" / "subgroup_directions.npz"
    data = np.load(npz_path)
    
    key_dir = f"{direction_type}_direction_{normalize}_{category}_{subgroup}"
    key_norm = f"{direction_type}_direction_{normalize}_norm_{category}_{subgroup}"
    
    if key_dir not in data:
        return None, None
    
    return data[key_dir], data[key_norm]
```

---

## Pairwise Cosines

Compute all pairwise cosines for each category at each layer, for all four direction types (bias_raw, bias_normed, identity_raw, identity_normed). Store as a single parquet.

```python
def compute_all_cosines(
    directions_arrays: dict[str, np.ndarray],
    categories: list[str],
    n_layers: int,
) -> pd.DataFrame:
    """
    Compute pairwise cosine similarities for all subgroup pairs within each category,
    for all four direction types, at all layers.
    
    Returns DataFrame with columns:
        category, direction_type (bias_raw/bias_normed/identity_raw/identity_normed),
        layer, subgroup_A, subgroup_B, cosine
    """
    rows = []
    
    direction_specs = [
        ("bias", "raw", "bias_raw"),
        ("bias", "normed", "bias_normed"),
        ("identity", "raw", "identity_raw"),
        ("identity", "normed", "identity_normed"),
    ]
    
    for cat in categories:
        # Enumerate subgroups that have directions for this category
        subgroups_with_dirs = {}
        for dtype, normalize, label in direction_specs:
            subs = []
            for key in directions_arrays:
                prefix = f"{dtype}_direction_{normalize}_{cat}_"
                if key.startswith(prefix):
                    sub = key[len(prefix):]
                    subs.append(sub)
            subgroups_with_dirs[label] = sorted(subs)
        
        for dtype, normalize, label in direction_specs:
            subs = subgroups_with_dirs[label]
            if len(subs) < 2:
                continue
            
            # Get all direction arrays for this (category, direction_type)
            dir_arrays = {}
            for sub in subs:
                key = f"{dtype}_direction_{normalize}_{cat}_{sub}"
                dir_arrays[sub] = directions_arrays[key]  # (n_layers, hidden_dim)
            
            # Compute cosines for all pairs at all layers
            for i, sub_A in enumerate(subs):
                for sub_B in subs[i+1:]:
                    arr_A = dir_arrays[sub_A]  # (n_layers, hidden_dim)
                    arr_B = dir_arrays[sub_B]
                    
                    # Both are unit-normalized per-layer, so cosine = dot product
                    cosines = np.sum(arr_A * arr_B, axis=1)  # (n_layers,)
                    
                    for layer in range(n_layers):
                        rows.append({
                            "category": cat,
                            "direction_type": label,
                            "layer": int(layer),
                            "subgroup_A": sub_A,
                            "subgroup_B": sub_B,
                            "cosine": round(float(cosines[layer]), 6),
                        })
    
    return pd.DataFrame(rows)
```

**Parquet output:**

```
{run}/B_geometry/cosine_pairs.parquet
```

| Column | Type | Description |
|---|---|---|
| `category` | string | Category short name |
| `direction_type` | string | `bias_raw`, `bias_normed`, `identity_raw`, `identity_normed` |
| `layer` | int32 | Transformer layer |
| `subgroup_A` | string | First subgroup (alphabetically) |
| `subgroup_B` | string | Second subgroup (alphabetically) |
| `cosine` | float32 | Cosine similarity at this layer |

Expected size: ~9 categories × 4 direction_types × 32 layers × ~10 pairs = ~12K rows. ~500KB parquet.

Downstream queries:
```python
# Get identity_normed cosine matrix for "so" at layer 14
df = pd.read_parquet("cosine_pairs.parquet")
subset = df[
    (df["category"] == "so") &
    (df["direction_type"] == "identity_normed") &
    (df["layer"] == 14)
]
# Pivot to matrix form as needed for display
```

---

## Differentiation Metrics

For each (category, direction_type), find the layer where subgroups are most differentiated. Compute multiple metrics; report all; pick peak by variance.

```python
def compute_differentiation_metrics(
    cosine_df: pd.DataFrame,
    categories: list[str],
    n_layers: int,
) -> dict:
    """
    For each (category, direction_type), compute per-layer differentiation metrics
    and find the peak layer.
    """
    result = {}
    
    for cat in categories:
        result[cat] = {}
        
        for dtype in ["bias_raw", "bias_normed", "identity_raw", "identity_normed"]:
            sub_df = cosine_df[
                (cosine_df["category"] == cat) &
                (cosine_df["direction_type"] == dtype)
            ]
            
            if sub_df.empty:
                continue
            
            per_layer = {}
            for layer in range(n_layers):
                layer_pairs = sub_df[sub_df["layer"] == layer]["cosine"].values
                if len(layer_pairs) == 0:
                    continue
                
                per_layer[layer] = {
                    "variance": float(np.var(layer_pairs)),
                    "range": float(layer_pairs.max() - layer_pairs.min()),
                    "mean_distance": float(np.mean(1 - layer_pairs)),  # 1 - cosine = angular dissimilarity
                    "min_cosine": float(layer_pairs.min()),
                    "max_cosine": float(layer_pairs.max()),
                    "median_cosine": float(np.median(layer_pairs)),
                    "n_pairs": int(len(layer_pairs)),
                }
            
            if not per_layer:
                continue
            
            # Peak layer by variance
            peak_layer = max(per_layer, key=lambda l: per_layer[l]["variance"])
            
            # Stable range (contiguous around peak where all pair signs preserved)
            stable_start, stable_end = find_stable_range(
                sub_df, n_layers, peak_layer
            )
            
            # Most anti-aligned pair at peak
            peak_data = sub_df[sub_df["layer"] == peak_layer]
            if not peak_data.empty:
                most_anti = peak_data.nsmallest(1, "cosine").iloc[0]
                most_anti_pair = {
                    "subgroup_A": most_anti["subgroup_A"],
                    "subgroup_B": most_anti["subgroup_B"],
                    "cosine": float(most_anti["cosine"]),
                }
            else:
                most_anti_pair = None
            
            result[cat][dtype] = {
                "per_layer_metrics": per_layer,
                "peak_layer": int(peak_layer),
                "peak_variance": float(per_layer[peak_layer]["variance"]),
                "peak_range": float(per_layer[peak_layer]["range"]),
                "peak_mean_distance": float(per_layer[peak_layer]["mean_distance"]),
                "stable_range": [int(stable_start), int(stable_end)],
                "stable_range_length": int(stable_end - stable_start + 1),
                "most_anti_aligned_pair_at_peak": most_anti_pair,
            }
    
    return result


def find_stable_range(
    cosine_df_for_cat_dtype: pd.DataFrame,
    n_layers: int,
    peak_layer: int,
) -> tuple[int, int]:
    """
    Find the contiguous range of layers around peak_layer where every subgroup pair
    maintains its cosine sign.
    """
    # Get pair signs at peak layer
    peak_data = cosine_df_for_cat_dtype[cosine_df_for_cat_dtype["layer"] == peak_layer]
    peak_signs = {}
    for _, row in peak_data.iterrows():
        pair_key = (row["subgroup_A"], row["subgroup_B"])
        peak_signs[pair_key] = np.sign(row["cosine"])
    
    def signs_match(layer: int) -> bool:
        layer_data = cosine_df_for_cat_dtype[cosine_df_for_cat_dtype["layer"] == layer]
        for _, row in layer_data.iterrows():
            pair_key = (row["subgroup_A"], row["subgroup_B"])
            if pair_key in peak_signs:
                if np.sign(row["cosine"]) != peak_signs[pair_key]:
                    return False
        return True
    
    # Expand left
    L_start = peak_layer
    while L_start > 0 and signs_match(L_start - 1):
        L_start -= 1
    
    # Expand right
    L_end = peak_layer
    while L_end < n_layers - 1 and signs_match(L_end + 1):
        L_end += 1
    
    return L_start, L_end
```

---

## Bias-Identity Alignment

For each subgroup at each layer, compute the cosine between its bias direction and its identity direction. Do this for both normalization regimes.

```python
def compute_alignment(
    directions_arrays: dict[str, np.ndarray],
    categories: list[str],
    n_layers: int,
) -> dict:
    """
    Compute cosine(bias_direction, identity_direction) per subgroup per layer.
    
    Do this for both normalization regimes (raw and normed).
    """
    result = {}
    
    for cat in categories:
        cat_result = {}
        
        # Find all subgroups that have BOTH bias and identity directions
        subgroups = set()
        for key in directions_arrays:
            if key.startswith(f"bias_direction_raw_{cat}_"):
                sub = key[len(f"bias_direction_raw_{cat}_"):]
                subgroups.add(sub)
        
        for sub in sorted(subgroups):
            sub_alignment = {"raw": None, "normed": None}
            
            for norm_type in ["raw", "normed"]:
                bias_key = f"bias_direction_{norm_type}_{cat}_{sub}"
                id_key = f"identity_direction_{norm_type}_{cat}_{sub}"
                
                if bias_key not in directions_arrays or id_key not in directions_arrays:
                    continue
                
                bias = directions_arrays[bias_key]      # (n_layers, hidden_dim)
                identity = directions_arrays[id_key]
                
                # Unit-normalized, so cosine = dot product
                alignments = np.sum(bias * identity, axis=1)  # (n_layers,)
                alignments_sq = alignments ** 2
                
                # Peak |alignment| layer
                peak_layer = int(np.argmax(np.abs(alignments)))
                peak_alignment = float(alignments[peak_layer])
                
                sub_alignment[norm_type] = {
                    "per_layer_alignment": [round(float(a), 4) for a in alignments],
                    "per_layer_alignment_squared": [round(float(a), 4) for a in alignments_sq],
                    "peak_layer": peak_layer,
                    "peak_alignment": round(peak_alignment, 4),
                    "peak_alignment_squared": round(peak_alignment ** 2, 4),
                    "mean_alignment": round(float(alignments.mean()), 4),
                    "mean_alignment_squared": round(float(alignments_sq.mean()), 4),
                }
            
            cat_result[sub] = sub_alignment
        
        result[cat] = cat_result
    
    return result
```

**Interpretation:**
- `alignment = 1.0` — Bias direction = identity direction. Fully entangled.
- `alignment = 0.0` — Bias is orthogonal to identity. Cleanly separable.
- `alignment_squared` — Fraction of shared variance between bias and identity (0 to 1).
- `alignment = -1.0` — Anti-aligned. Invoking identity pushes AWAY from stereotyped response. Unusual finding.

Note that positive alignment is the null expectation (bias direction uses S-targeted items, so it should lean toward the identity direction). The question is magnitude — 0.3 (weak shared component) vs 0.95 (nearly identical).

---

## Output Files

### `subgroup_directions.npz`

Single .npz file with all direction arrays and pre-normalization norms.

```
{run}/B_geometry/subgroup_directions.npz
```

Keys (per subgroup with bias and identity directions):
- `bias_direction_raw_{cat}_{sub}` — (n_layers, hidden_dim) float32
- `bias_direction_normed_{cat}_{sub}` — (n_layers, hidden_dim) float32
- `identity_direction_raw_{cat}_{sub}` — (n_layers, hidden_dim) float32
- `identity_direction_normed_{cat}_{sub}` — (n_layers, hidden_dim) float32
- `bias_direction_raw_norm_{cat}_{sub}` — (n_layers,) float32
- `bias_direction_normed_norm_{cat}_{sub}` — (n_layers,) float32
- `identity_direction_raw_norm_{cat}_{sub}` — (n_layers,) float32
- `identity_direction_normed_norm_{cat}_{sub}` — (n_layers,) float32

Estimated size: 40 subgroups × 8 arrays × avg 530KB = ~170MB. Manageable.

### `cosine_pairs.parquet`

All pairwise cosines. Schema above.

```
{run}/B_geometry/cosine_pairs.parquet
```

### `subgroup_directions_summary.json`

Per-subgroup metadata: item counts, which directions computed, raw norms summary.

```json
{
  "min_n_per_group": 10,
  "categories": ["so", "race", ...],
  "per_subgroup": {
    "so/gay": {
      "category": "so",
      "subgroup": "gay",
      "skipped": false,
      "n_total_targeting_S": 2160,
      "n_total_not_targeting_S": 6480,
      "n_bias_stereo": 487,
      "n_bias_non_stereo": 593,
      "bias_computed": true,
      "identity_computed": true,
      "bias_direction_raw_norm_range": [0.234, 3.821],
      "bias_direction_raw_norm_peak_layer": 18,
      "identity_direction_raw_norm_range": [0.489, 5.217],
      "identity_direction_raw_norm_peak_layer": 16
    },
    "so/pansexual": {
      ...
      "bias_computed": false,
      "bias_skip_reason": "n_bias_stereo=7 < min_n=10"
    }
  }
}
```

### `differentiation_metrics.json`

Per-category, per-direction-type metrics.

```json
{
  "so": {
    "identity_normed": {
      "peak_layer": 14,
      "peak_variance": 0.284,
      "peak_range": 1.214,
      "peak_mean_distance": 0.687,
      "stable_range": [11, 19],
      "stable_range_length": 9,
      "most_anti_aligned_pair_at_peak": {
        "subgroup_A": "bisexual",
        "subgroup_B": "gay",
        "cosine": -0.342
      },
      "per_layer_metrics": {
        "0": {"variance": 0.012, "range": 0.18, "mean_distance": 0.08, ...},
        "14": {"variance": 0.284, ...},
        ...
      }
    },
    "identity_raw": {...},
    "bias_normed": {...},
    "bias_raw": {...}
  },
  ...
}
```

### `bias_identity_alignment.json`

Per-subgroup alignment between bias and identity directions.

```json
{
  "so": {
    "gay": {
      "raw": {
        "per_layer_alignment": [0.12, 0.18, ..., 0.87, 0.91, 0.89, ...],
        "per_layer_alignment_squared": [0.014, 0.032, ..., 0.757, 0.828, 0.792, ...],
        "peak_layer": 18,
        "peak_alignment": 0.912,
        "peak_alignment_squared": 0.832,
        "mean_alignment": 0.645,
        "mean_alignment_squared": 0.521
      },
      "normed": {...}
    }
  }
}
```

---

## Figures

All figures use Wong colorblind-safe palette. Distinct markers per subgroup/category.

### `fig_cosine_heatmap_{cat}_{direction_type}_L{peak_layer}.png`

Heatmap of pairwise cosines at the peak differentiation layer. One figure per (category, direction_type) that has ≥2 subgroups.

- Rows and columns: subgroups (sorted)
- Cell values: cosine similarity
- Diagonal: 1.0 (or masked)
- Colormap: `RdBu_r` diverging, vmin=-1, vmax=1, centered at 0
- Cells annotated with values to 2 decimals
- Title: "Pairwise cosines — {category} {direction_type} at layer {peak_layer}"

Generate for all four direction types (bias_raw, bias_normed, identity_raw, identity_normed) for each category.

### `fig_cosine_by_layer_{cat}_{direction_type}.png`

Line plot showing each subgroup pair's cosine across layers. One figure per (category, direction_type).

- X-axis: layer (0 to n_layers-1)
- Y-axis: cosine similarity (-1 to 1)
- One line per pair, distinct color + marker
- Horizontal dashed line at y=0
- Vertical dashed line at peak layer
- Shaded region for stable range
- Legend: subgroup pairs
- Title: "Subgroup cosines across layers — {category} {direction_type}"

### `fig_direction_norms_{cat}.png`

Plot of raw direction norms (pre-normalization magnitudes) across layers. Sanity check for direction strength.

- 2×2 grid of subplots: (bias/identity) × (raw/normed)
- X-axis: layer
- Y-axis: raw norm (log scale)
- One line per subgroup, distinct color + marker
- Title: "{category} — direction pre-normalization magnitudes"

A subgroup with norms near zero across all layers has effectively no signal.

### `fig_bias_identity_alignment.png`

Panel per category. Within each panel, one line per subgroup showing bias-identity alignment across layers.

- X-axis: layer
- Y-axis: alignment cosine (-1 to 1, or alignment² for 0 to 1 display)
- One line per subgroup, distinct color + marker
- Horizontal dashed lines at y=0 and y=1
- Two subplots per figure: one for raw, one for normed
- Title: "Bias-identity alignment across layers"

### `fig_differentiation_metrics.png`

Summary figure showing peak differentiation metrics across categories.

- Bar chart grouped by category
- Paired bars per category: `identity_normed` variance (primary) and `bias_normed` variance (secondary)
- X-axis: category
- Y-axis: peak variance
- Annotate peak layer on each bar
- Title: "Peak differentiation by category"

---

## Output Structure

```
{run}/B_geometry/
├── subgroup_directions.npz               # All directions + pre-norm magnitudes
├── cosine_pairs.parquet                   # All pairwise cosines, long format
├── subgroup_directions_summary.json       # Per-subgroup metadata
├── differentiation_metrics.json           # Peak layer analysis per (cat, direction_type)
├── bias_identity_alignment.json           # Per-subgroup alignment curves
└── figures/
    ├── fig_cosine_heatmap_{cat}_{direction_type}_L{peak_layer}.png/.pdf
    ├── fig_cosine_by_layer_{cat}_{direction_type}.png/.pdf
    ├── fig_direction_norms_{cat}.png/.pdf
    ├── fig_bias_identity_alignment.png/.pdf
    └── fig_differentiation_metrics.png/.pdf
```

---

## Resume Safety

Per-category granularity. Check if all expected outputs exist:

```python
def b3_complete(run_dir: Path) -> bool:
    required = [
        "B_geometry/subgroup_directions.npz",
        "B_geometry/cosine_pairs.parquet",
        "B_geometry/subgroup_directions_summary.json",
        "B_geometry/differentiation_metrics.json",
        "B_geometry/bias_identity_alignment.json",
    ]
    return all((run_dir / p).exists() for p in required)
```

If complete and `--force` not passed, skip. If incomplete, re-run from scratch (B3 is fast — ~10-20 min total).

Atomic writes for all outputs (tmp-then-rename).

---

## Compute Estimate

- Per category: ~1-2 min (loading, mean computation, cosine calculation)
- All 9 categories: ~10-20 min
- Figure generation: ~1-2 min
- Total: ~15-25 min

---

## Assumptions Summary

| # | Decision | Value |
|---|---|---|
| 1 | "NOT targeting S" definition | Items where S is not in stereotyped_groups, regardless of overlapping labels |
| 2 | Multi-group item handling | Items contribute to all targeted subgroups' directions |
| 3 | Minimum items per group | 10 (configurable via `--min_n_per_group`) |
| 4 | Direction computation regimes | Both raw-based (for C1 steering) and normed-based (for C2 geometry) |
| 5 | Unit normalization | Applied at end of computation, per-layer |
| 6 | Raw pre-normalization norms stored | Yes — ||direction|| before unit-norming, as (n_layers,) arrays |
| 7 | Cosine storage | Single parquet with all (cat, direction_type, layer, sub_A, sub_B, cosine) rows |
| 8 | Peak differentiation metric | Variance of off-diagonal cosines; range and mean-distance also reported |
| 9 | Stable range definition | Contiguous layers around peak where all pair signs preserved |
| 10 | Bias direction items | Ambig only, split by stereotyped vs non-stereotyped response |
| 11 | Identity direction items | All items (both context conditions included) |
| 12 | Alignment reporting | Both cosine and cosine² (fraction shared variance) |
| 13 | Subgroup enumeration | Unique subgroups in each category's stereotyped_groups values |

---

## Test Command

```bash
# Run B3
python scripts/B3_geometry.py --run_dir runs/llama-3.1-8b_2026-04-15/

# Verify outputs
python -c "
import numpy as np
import pandas as pd
import json

# Check directions
data = np.load('runs/llama-3.1-8b_2026-04-15/B_geometry/subgroup_directions.npz')
keys = list(data.keys())
print(f'Total arrays: {len(keys)}')
print(f'Sample keys: {keys[:5]}')

# Check a specific direction
if 'bias_direction_raw_so_gay' in keys:
    arr = data['bias_direction_raw_so_gay']
    print(f'bias_direction_raw_so_gay shape: {arr.shape}')
    # Should be unit-normalized per layer
    norms = np.linalg.norm(arr, axis=1)
    print(f'  Per-layer norms: min={norms.min():.4f}, max={norms.max():.4f} (should be ~1.0)')

# Check cosines
df = pd.read_parquet('runs/llama-3.1-8b_2026-04-15/B_geometry/cosine_pairs.parquet')
print(f'\\nCosine pairs: {len(df)} rows')
print(f'Direction types: {sorted(df[\"direction_type\"].unique())}')
print(f'Categories: {sorted(df[\"category\"].unique())}')

# Peak differentiation
with open('runs/llama-3.1-8b_2026-04-15/B_geometry/differentiation_metrics.json') as f:
    diff = json.load(f)

for cat in sorted(diff.keys()):
    if 'identity_normed' in diff[cat]:
        d = diff[cat]['identity_normed']
        print(f'{cat} identity_normed: peak L={d[\"peak_layer\"]}, var={d[\"peak_variance\"]:.3f}')
        mp = d.get('most_anti_aligned_pair_at_peak')
        if mp:
            print(f'  Most anti-aligned: {mp[\"subgroup_A\"]}__{mp[\"subgroup_B\"]} cos={mp[\"cosine\"]:.3f}')

# Bias-identity alignment
with open('runs/llama-3.1-8b_2026-04-15/B_geometry/bias_identity_alignment.json') as f:
    align = json.load(f)

print(f'\\nAlignment sample (so/gay):')
if 'so' in align and 'gay' in align['so']:
    for norm_type in ['raw', 'normed']:
        if norm_type in align['so']['gay'] and align['so']['gay'][norm_type]:
            a = align['so']['gay'][norm_type]
            print(f'  {norm_type}: peak L={a[\"peak_layer\"]}, peak align={a[\"peak_alignment\"]:.3f}')
"
```