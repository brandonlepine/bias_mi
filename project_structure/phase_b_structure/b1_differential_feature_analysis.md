# B1: Differential Feature Analysis — Full Implementation Specification

## Purpose

For each subgroup at each layer, identify SAE features that activate differently when the model produces a stereotyped response vs. when it doesn't. This is the core statistical test that surfaces "bias-associated features" for downstream ranking (B2), interpretability (B5), and steering (C1).

## Invocation

```bash
python scripts/B1_differential.py --run_dir runs/llama-3.1-8b_2026-04-15/

# Quick test (limited items per category)
python scripts/B1_differential.py --run_dir runs/llama-3.1-8b_2026-04-15/ --max_items 100

# Single category
python scripts/B1_differential.py --run_dir runs/llama-3.1-8b_2026-04-15/ --categories so

# Specific layers only
python scripts/B1_differential.py --run_dir runs/llama-3.1-8b_2026-04-15/ --layers 12,14,16

# Override minimum sample size
python scripts/B1_differential.py --run_dir runs/llama-3.1-8b_2026-04-15/ --min_n_per_group 15
```

Reads `categories` and `n_layers` from config.json.

---

## Input

- SAE encodings: `{run}/A_extraction/sae_encoding/layer_{NN:02d}.parquet`
- Metadata parquet (primary): `{run}/A_extraction/metadata.parquet`
- Metadata .npz fallback: `{run}/A_extraction/activations/{category}/item_*.npz`

## Dependencies

- `pandas`, `pyarrow`
- `numpy`
- `scipy` (≥1.8 for vectorized Mann-Whitney U with axis support)
- `statsmodels` (for Benjamini-Hochberg FDR correction)

---

## Script Structure

```python
def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = load_config(run_dir)
    
    # Resolve parameters
    categories = parse_categories(args.categories) if args.categories else config["categories"]
    layers = parse_layers(args.layers, config["n_layers"])  # default: all layers
    min_n = args.min_n_per_group  # default 10
    
    # Step 0: Load metadata
    meta_df = load_metadata(run_dir)
    
    # Step 0b: Identify subgroups to analyze
    subgroup_catalog = build_subgroup_catalog(meta_df, categories, min_n)
    
    log(f"Subgroups to analyze: {len(subgroup_catalog)} across {len(categories)} categories")
    for cat in categories:
        subs = [s for s in subgroup_catalog if subgroup_catalog[s]["category"] == cat]
        log(f"  {cat}: {len(subs)} subgroups")
    
    # Process each layer
    all_summaries = {}
    for layer in layers:
        parquet_path = run_dir / "B_differential" / f"layer_{layer:02d}.parquet"
        
        # Resume check
        if parquet_path.exists():
            log(f"Layer {layer:02d}: already processed, skipping")
            # Still load for summary computation
            all_summaries[layer] = load_layer_summary(run_dir, layer)
            continue
        
        layer_results, layer_summary = process_layer(
            layer=layer,
            run_dir=run_dir,
            meta_df=meta_df,
            categories=categories,
            subgroup_catalog=subgroup_catalog,
            min_n=min_n,
            max_items=args.max_items,
        )
        
        # Write parquet
        save_layer_parquet(run_dir, layer, layer_results)
        
        # Write per-layer summary
        save_layer_summary(run_dir, layer, layer_summary)
        all_summaries[layer] = layer_summary
    
    # Build global summary
    build_differential_summary(run_dir, layers, subgroup_catalog, all_summaries)
```

---

## Step 0: Load Metadata

Primary path: load from A2's `metadata.parquet`. Fall back to scanning .npz files if missing.

```python
def load_metadata(run_dir: Path) -> pd.DataFrame:
    """Load item metadata. Prefer parquet; fall back to .npz scanning."""
    meta_path = run_dir / "A_extraction" / "metadata.parquet"
    
    if meta_path.exists():
        df = pd.read_parquet(meta_path)
        log(f"Loaded metadata: {len(df)} items from metadata.parquet")
    else:
        log(f"metadata.parquet not found; scanning .npz files (slow)...")
        df = build_metadata_from_npz(run_dir)
    
    # Deserialize stereotyped_groups from JSON string to list
    df["stereotyped_groups"] = df["stereotyped_groups"].apply(json.loads)
    
    return df
```

The metadata DataFrame has columns:
- `item_idx` (int32)
- `category` (string)
- `model_answer_role` (string: "stereotyped_target", "non_stereotyped", "unknown")
- `is_stereotyped_response` (bool)
- `is_correct` (bool)
- `context_condition` (string: "ambig", "disambig")
- `stereotyped_groups` (list of strings, deserialized from JSON)
- `n_target_groups` (int32)
- `margin` (float32)
- `question_polarity` (string)
- `correct_letter`, `stereotyped_option` (string)

---

## Step 0b: Build Subgroup Catalog

Enumerate all subgroups that have sufficient items to analyze. The subgroup labels come from the metadata — whatever normalized labels A1 produced are what we iterate over.

```python
def build_subgroup_catalog(
    meta_df: pd.DataFrame,
    categories: list[str],
    min_n: int,
) -> dict[str, dict]:
    """
    Build catalog of analyzable subgroups.
    
    Returns: {subgroup_name: {category, n_stereo, n_non_stereo, n_unknown, total_n}}
    
    A subgroup is analyzable if it has >= min_n items in BOTH the stereotyped-response
    group and the non-stereotyped-response group (ambig items only).
    """
    catalog = {}
    
    for cat in categories:
        cat_df = meta_df[meta_df["category"] == cat]
        ambig = cat_df[cat_df["context_condition"] == "ambig"]
        
        # Collect all subgroups appearing in ambig items
        all_subs_in_cat = set()
        for gs in ambig["stereotyped_groups"]:
            all_subs_in_cat.update(gs)
        
        for sub in sorted(all_subs_in_cat):
            # Items targeting this subgroup
            targeting = ambig[ambig["stereotyped_groups"].apply(lambda gs: sub in gs)]
            
            n_stereo = int((targeting["model_answer_role"] == "stereotyped_target").sum())
            n_non_stereo = int((targeting["model_answer_role"] != "stereotyped_target").sum())
            n_unknown = int((targeting["model_answer_role"] == "unknown").sum())
            n_non_stereo_strict = int((targeting["model_answer_role"] == "non_stereotyped").sum())
            
            entry = {
                "category": cat,
                "n_stereo": n_stereo,
                "n_non_stereo": n_non_stereo,
                "n_non_stereo_strict": n_non_stereo_strict,
                "n_unknown_in_non_stereo_group": n_unknown,
                "total_ambig": int(len(targeting)),
                "analyzable": n_stereo >= min_n and n_non_stereo >= min_n,
            }
            
            catalog[sub] = entry
    
    # Filter to analyzable subgroups (but keep the full catalog for reporting)
    n_analyzable = sum(1 for v in catalog.values() if v["analyzable"])
    n_skipped = len(catalog) - n_analyzable
    
    if n_skipped > 0:
        log(f"Subgroups with <{min_n} items in either group (skipped):")
        for sub, entry in sorted(catalog.items()):
            if not entry["analyzable"]:
                log(f"  {entry['category']}/{sub}: "
                    f"n_stereo={entry['n_stereo']}, n_non_stereo={entry['n_non_stereo']}")
    
    return catalog
```

**Note on subgroup labels:** The catalog uses whatever normalized labels appear in `stereotyped_groups`. For race, this would include "black" (collapsed from "african american") and "middle eastern" (collapsed from "arab"). For GI, "transgender women" and "transgender men" remain separate.

**Duplicate subgroup names across categories:** In principle, a subgroup name could appear in multiple categories (e.g., "old" in Age, "old" in another category). The catalog keys should be subgroup names alone since BBQ doesn't actually have this overlap, but if it did, we'd need composite keys. For safety, we track the category on each entry and verify uniqueness:

```python
# Verify: subgroup names should be unique across categories
name_to_cats = defaultdict(list)
for sub, entry in catalog.items():
    name_to_cats[sub].append(entry["category"])

for name, cats in name_to_cats.items():
    if len(cats) > 1:
        log(f"WARNING: subgroup '{name}' appears in multiple categories: {cats}")
        # Treat each as separate — this shouldn't actually happen with BBQ
```

---

## Step 1: Define Comparison Groups (Question A)

**Methodological choice — Question A:**

For subgroup S, the comparison is:

**Stereotyped-response group:** Items where S ∈ `stereotyped_groups` AND `context_condition == "ambig"` AND `model_answer_role == "stereotyped_target"` (model chose the biased answer).

**Non-stereotyped-response group:** Items where S ∈ `stereotyped_groups` AND `context_condition == "ambig"` AND `model_answer_role != "stereotyped_target"` (model chose unknown OR non_stereotyped).

This tests: "which features drive the model's choice of the biased answer?" The non-stereotyped group intentionally lumps together "correctly uncertain" (unknown) and "actively anti-stereotypical" (non_stereotyped) responses — both represent the model NOT being biased.

The per-item results retain `model_answer_role` and `is_stereotyped_response` so post-hoc analysis can split the non-stereotyped group into unknown-specific vs non_stereotyped-specific if desired.

```python
def get_comparison_groups(
    meta_df: pd.DataFrame,
    category: str,
    subgroup: str,
) -> tuple[list[int], list[int]]:
    """
    Return (stereotyped_item_idxs, non_stereotyped_item_idxs) for one subgroup.
    Both lists contain item_idx values for ambig items targeting the subgroup.
    """
    cat_df = meta_df[meta_df["category"] == category]
    targeting = cat_df[
        cat_df["stereotyped_groups"].apply(lambda gs: subgroup in gs)
        & (cat_df["context_condition"] == "ambig")
    ]
    
    stereo_mask = targeting["model_answer_role"] == "stereotyped_target"
    
    stereo_idxs = targeting.loc[stereo_mask, "item_idx"].tolist()
    non_stereo_idxs = targeting.loc[~stereo_mask, "item_idx"].tolist()
    
    return stereo_idxs, non_stereo_idxs
```

---

## Step 2: Load and Build Sparse Matrix Per (Category, Layer)

For efficient computation, build ONE sparse item×feature matrix per (category, layer), then reuse it for all subgroups within that category.

```python
from scipy.sparse import csr_matrix

def build_sparse_matrix(
    cat_df: pd.DataFrame,
    all_item_idxs: list[int],
) -> tuple[csr_matrix, dict[int, int], np.ndarray]:
    """
    Build a sparse (n_items, n_active_features) matrix for one (category, layer).
    
    Args:
        cat_df: parquet rows for one category at one layer (long format)
        all_item_idxs: item indices to include as rows (in order)
    
    Returns:
        matrix: csr_matrix of activations, shape (n_items, n_active_features)
        item_idx_to_row: dict mapping item_idx -> row index
        col_to_feature: array mapping column index -> feature_idx
    """
    # Item index mapping (matrix rows)
    item_idx_to_row = {idx: i for i, idx in enumerate(all_item_idxs)}
    
    # Only include features that fire on at least one item in this category
    active_features = np.sort(cat_df["feature_idx"].unique())
    feature_idx_to_col = {fidx: i for i, fidx in enumerate(active_features)}
    
    # Build COO arrays
    rows = cat_df["item_idx"].map(item_idx_to_row).values
    cols = cat_df["feature_idx"].map(feature_idx_to_col).values
    vals = cat_df["activation_value"].values.astype(np.float32)
    
    # Filter out items not in our mapping (shouldn't happen but defensive)
    valid_mask = ~(pd.isna(rows) | pd.isna(cols))
    
    matrix = csr_matrix(
        (vals[valid_mask], (rows[valid_mask].astype(int), cols[valid_mask].astype(int))),
        shape=(len(all_item_idxs), len(active_features)),
        dtype=np.float32,
    )
    
    return matrix, item_idx_to_row, active_features
```

**Why active features only:** Most SAE features never fire on any item in a given category. We skip the ~120K "always zero" features since they have no signal. Typical retention: ~5K-15K active features per (category, layer) out of 131K total.

---

## Step 3: Vectorized Statistical Tests

For each subgroup within the category, slice the matrix rows and run vectorized stats across all features simultaneously.

```python
def test_subgroup_vectorized(
    matrix: csr_matrix,
    item_idx_to_row: dict[int, int],
    active_features: np.ndarray,
    stereo_idxs: list[int],
    non_stereo_idxs: list[int],
) -> pd.DataFrame:
    """
    Run vectorized differential tests for all active features for one subgroup.
    
    Returns a DataFrame with one row per active feature, containing:
        feature_idx, cohens_d, p_value_raw, firing_rate_stereo, firing_rate_non_stereo,
        mean_activation_stereo, mean_activation_non_stereo, n_stereo, n_non_stereo,
        passes_firing_filter
    """
    # Map item indices to row indices
    stereo_rows = [item_idx_to_row[idx] for idx in stereo_idxs if idx in item_idx_to_row]
    non_stereo_rows = [item_idx_to_row[idx] for idx in non_stereo_idxs if idx in item_idx_to_row]
    
    if len(stereo_rows) == 0 or len(non_stereo_rows) == 0:
        return None  # no items for this subgroup at this layer
    
    # Extract submatrices as dense arrays (they're small — just a few hundred items × ~10K features)
    stereo_mat = matrix[stereo_rows, :].toarray()          # (n_stereo, n_active_features)
    non_stereo_mat = matrix[non_stereo_rows, :].toarray()  # (n_non_stereo, n_active_features)
    
    n_stereo = len(stereo_rows)
    n_non_stereo = len(non_stereo_rows)
    
    # Vectorized Cohen's d
    mean_stereo = stereo_mat.mean(axis=0)
    mean_non_stereo = non_stereo_mat.mean(axis=0)
    var_stereo = stereo_mat.var(axis=0, ddof=1) if n_stereo > 1 else np.zeros_like(mean_stereo)
    var_non_stereo = non_stereo_mat.var(axis=0, ddof=1) if n_non_stereo > 1 else np.zeros_like(mean_non_stereo)
    
    pooled_var = (
        (n_stereo - 1) * var_stereo + (n_non_stereo - 1) * var_non_stereo
    ) / max(n_stereo + n_non_stereo - 2, 1)
    pooled_std = np.sqrt(np.maximum(pooled_var, 1e-12))
    cohens_d = (mean_stereo - mean_non_stereo) / pooled_std
    
    # Vectorized firing rates
    firing_stereo = (stereo_mat > 0).mean(axis=0)
    firing_non_stereo = (non_stereo_mat > 0).mean(axis=0)
    
    # Firing rate filter
    max_firing = np.maximum(firing_stereo, firing_non_stereo)
    combined_nonzero = (stereo_mat > 0).sum(axis=0) + (non_stereo_mat > 0).sum(axis=0)
    passes_filter = (max_firing >= 0.05) | (combined_nonzero >= 10)
    
    # Vectorized Mann-Whitney U
    # scipy>=1.8 supports axis parameter for batched testing
    pvalues = np.ones(len(active_features), dtype=np.float64)
    
    if passes_filter.any():
        # Only test features that pass the firing rate filter
        filtered_stereo = stereo_mat[:, passes_filter]
        filtered_non_stereo = non_stereo_mat[:, passes_filter]
        
        try:
            from scipy.stats import mannwhitneyu
            _, pvals_filtered = mannwhitneyu(
                filtered_stereo, filtered_non_stereo,
                alternative="two-sided",
                axis=0,
                nan_policy="omit",
            )
            # pvals_filtered shape: (n_passing_features,)
            pvalues[passes_filter] = pvals_filtered
        except (ValueError, TypeError) as e:
            # Fall back to per-feature loop if vectorized fails
            log(f"    Vectorized Mann-Whitney failed ({e}), falling back to loop")
            idx_passing = np.where(passes_filter)[0]
            for j in idx_passing:
                try:
                    _, p = mannwhitneyu(
                        stereo_mat[:, j], non_stereo_mat[:, j],
                        alternative="two-sided",
                    )
                    pvalues[j] = p
                except Exception:
                    pvalues[j] = 1.0
    
    # Features failing filter get p=1.0 (already set)
    
    # Build result DataFrame
    result = pd.DataFrame({
        "feature_idx": active_features.astype(np.int32),
        "cohens_d": cohens_d.astype(np.float32),
        "p_value_raw": pvalues,
        "firing_rate_stereo": firing_stereo.astype(np.float32),
        "firing_rate_non_stereo": firing_non_stereo.astype(np.float32),
        "mean_activation_stereo": mean_stereo.astype(np.float32),
        "mean_activation_non_stereo": mean_non_stereo.astype(np.float32),
        "passes_firing_filter": passes_filter,
    })
    
    result["n_stereo"] = n_stereo
    result["n_non_stereo"] = n_non_stereo
    
    return result
```

**Methodological choice — Mann-Whitney U over t-test:**

SAE feature activations are heavily zero-inflated: for any given feature, 90-99% of items have exactly zero activation. The remaining items have continuous positive values with a right-skewed tail. A t-test assumes normality, which is badly violated here.

Mann-Whitney tests rank differences between groups. If both groups have mostly zeros, ranks don't differ and p-value is high. If one group has more nonzero items or higher activations among its nonzero items, ranks differ and p-value is low. This is robust to zero-inflation.

**Vectorized implementation:** scipy ≥ 1.8 supports `mannwhitneyu(a, b, axis=0)` for batched testing across feature columns. This is dramatically faster than looping: ~5-10 minutes per layer vs. ~30 minutes.

---

## Step 4: FDR Correction

Apply Benjamini-Hochberg correction per (subgroup, layer) across all features that passed the firing rate filter.

```python
from statsmodels.stats.multitest import multipletests

def apply_fdr(result_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Apply Benjamini-Hochberg FDR correction to features passing firing filter.
    
    Adds columns: p_value_fdr, is_significant, direction
    """
    result_df = result_df.copy()
    result_df["p_value_fdr"] = 1.0
    result_df["is_significant"] = False
    
    mask = result_df["passes_firing_filter"].values
    
    if mask.any():
        raw_pvals = result_df.loc[mask, "p_value_raw"].values
        rejected, p_adj, _, _ = multipletests(raw_pvals, alpha=alpha, method="fdr_bh")
        
        result_df.loc[mask, "p_value_fdr"] = p_adj
        result_df.loc[mask, "is_significant"] = rejected
    
    # Direction based on cohen's d sign
    result_df["direction"] = np.where(
        result_df["cohens_d"] > 0, "pro_bias", "anti_bias"
    )
    
    return result_df
```

**Scope: per (subgroup, layer).** Each combination is an independent scientific question. Correcting globally across all layers and subgroups would be overly conservative.

With ~5K-15K features passing the firing rate filter per (subgroup, layer), BH at α=0.05 means the top feature needs raw p < ~5×10⁻⁶ to survive. This is conservative enough to be credible.

---

## Step 5: Per-Layer Processing Loop

```python
def process_layer(
    layer: int,
    run_dir: Path,
    meta_df: pd.DataFrame,
    categories: list[str],
    subgroup_catalog: dict,
    min_n: int,
    max_items: int | None,
) -> tuple[pd.DataFrame, dict]:
    """
    Process one layer: test all subgroups in all categories.
    
    Returns:
        all_results: DataFrame with one row per (feature, subgroup) that passed firing filter
        layer_summary: dict with per-subgroup counts and stats
    """
    t0 = time.time()
    log(f"\n{'='*60}")
    log(f"Layer {layer:02d}")
    log(f"{'='*60}")
    
    # Load layer parquet
    parquet_path = run_dir / "A_extraction" / "sae_encoding" / f"layer_{layer:02d}.parquet"
    if not parquet_path.exists():
        log(f"  ERROR: layer parquet not found: {parquet_path}")
        return pd.DataFrame(), {}
    
    layer_df = pd.read_parquet(parquet_path)
    log(f"  Loaded parquet: {len(layer_df)} rows")
    
    all_results = []
    layer_summary = {"per_subgroup": {}}
    
    for cat in categories:
        # Filter to this category
        cat_df = layer_df[layer_df["category"] == cat]
        if cat_df.empty:
            continue
        
        # Get all ambig items for this category
        cat_ambig_meta = meta_df[
            (meta_df["category"] == cat) &
            (meta_df["context_condition"] == "ambig")
        ]
        all_item_idxs = cat_ambig_meta["item_idx"].tolist()
        
        if max_items:
            all_item_idxs = all_item_idxs[:max_items]
        
        if len(all_item_idxs) == 0:
            continue
        
        # Filter parquet rows to ambig items only
        cat_df = cat_df[cat_df["item_idx"].isin(all_item_idxs)]
        
        if cat_df.empty:
            continue
        
        # Build sparse matrix ONCE for this (category, layer)
        matrix, item_idx_to_row, active_features = build_sparse_matrix(cat_df, all_item_idxs)
        log(f"  {cat}: sparse matrix {matrix.shape}, {len(active_features)} active features")
        
        # Test each subgroup in this category
        for sub, entry in sorted(subgroup_catalog.items()):
            if entry["category"] != cat:
                continue
            if not entry["analyzable"]:
                continue
            
            stereo_idxs, non_stereo_idxs = get_comparison_groups(meta_df, cat, sub)
            
            # Intersect with items we actually have in the matrix
            stereo_idxs = [i for i in stereo_idxs if i in item_idx_to_row]
            non_stereo_idxs = [i for i in non_stereo_idxs if i in item_idx_to_row]
            
            if len(stereo_idxs) < min_n or len(non_stereo_idxs) < min_n:
                continue
            
            # Run tests
            result = test_subgroup_vectorized(
                matrix, item_idx_to_row, active_features,
                stereo_idxs, non_stereo_idxs,
            )
            
            if result is None:
                continue
            
            # Filter to features passing firing filter (drop the ~80% that don't)
            result = result[result["passes_firing_filter"]].copy()
            
            if result.empty:
                continue
            
            # Apply FDR
            result = apply_fdr(result, alpha=0.05)
            
            # Add context columns
            result["layer"] = layer
            result["subgroup"] = sub
            result["category"] = cat
            
            # Record summary
            n_sig = int(result["is_significant"].sum())
            n_pro = int(((result["is_significant"]) & (result["direction"] == "pro_bias")).sum())
            n_anti = int(((result["is_significant"]) & (result["direction"] == "anti_bias")).sum())
            
            layer_summary["per_subgroup"].setdefault(sub, {}).update({
                "category": cat,
                "n_stereo": result["n_stereo"].iloc[0],
                "n_non_stereo": result["n_non_stereo"].iloc[0],
                "n_features_tested": int(result["passes_firing_filter"].sum()),
                "n_significant": n_sig,
                "n_significant_pro_bias": n_pro,
                "n_significant_anti_bias": n_anti,
            })
            
            log(f"    {sub}: n_stereo={result['n_stereo'].iloc[0]}, "
                f"n_non_stereo={result['n_non_stereo'].iloc[0]}, "
                f"n_tested={int(result['passes_firing_filter'].sum())}, "
                f"n_sig={n_sig} (pro={n_pro}, anti={n_anti})")
            
            all_results.append(result)
    
    # Concatenate all subgroup results for this layer
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
    else:
        combined = pd.DataFrame()
    
    elapsed = time.time() - t0
    log(f"  Layer {layer:02d} complete in {elapsed:.1f}s")
    
    layer_summary["layer"] = layer
    layer_summary["elapsed_seconds"] = round(elapsed, 1)
    layer_summary["total_rows"] = len(combined)
    
    return combined, layer_summary
```

---

## Step 6: Output

### Parquet Schema

One parquet per layer, all subgroups combined:

```
{run}/B_differential/layer_{NN:02d}.parquet
```

| Column | Type | Description |
|--------|------|-------------|
| `feature_idx` | int32 | SAE feature index (0 to 131071) |
| `layer` | int32 | Transformer layer |
| `subgroup` | string | Subgroup label (e.g., "bisexual") |
| `category` | string | Category (e.g., "so") |
| `cohens_d` | float32 | Effect size (positive = pro-bias) |
| `p_value_raw` | float64 | Raw Mann-Whitney p-value |
| `p_value_fdr` | float64 | FDR-corrected p-value |
| `is_significant` | bool | p_fdr < 0.05 |
| `direction` | string | "pro_bias" or "anti_bias" |
| `firing_rate_stereo` | float32 | Fraction of stereotyped items with activation > 0 |
| `firing_rate_non_stereo` | float32 | Fraction of non-stereotyped items with activation > 0 |
| `mean_activation_stereo` | float32 | Mean activation across stereotyped items (including zeros) |
| `mean_activation_non_stereo` | float32 | Mean activation across non-stereotyped items |
| `n_stereo` | int32 | Number of stereotyped-response items for this subgroup |
| `n_non_stereo` | int32 | Number of non-stereotyped-response items |

**Only features passing the firing rate filter are stored.** The `is_significant` flag distinguishes FDR-significant from non-significant. Downstream scripts (B2, B5, C1) filter by `is_significant == True` for the fast path; the full distribution is available for sensitivity analysis.

**Expected size per layer:**
- ~40 subgroups × ~3000 avg features passing filter = ~120K rows
- ~80 bytes per row compressed
- ~10MB per layer
- ~320MB total for all 32 layers

### Atomic Writes

```python
def save_layer_parquet(run_dir: Path, layer: int, df: pd.DataFrame):
    """Atomic parquet write."""
    out_dir = ensure_dir(run_dir / "B_differential")
    out_path = out_dir / f"layer_{layer:02d}.parquet"
    tmp_path = out_path.with_suffix(".parquet.tmp")
    
    df.to_parquet(tmp_path, index=False, compression="snappy")
    tmp_path.rename(out_path)
```

### Per-Layer Summary

```
{run}/B_differential/layer_{NN:02d}_summary.json
```

```json
{
  "layer": 14,
  "elapsed_seconds": 287.3,
  "total_rows": 124503,
  "per_subgroup": {
    "gay": {
      "category": "so",
      "n_stereo": 312,
      "n_non_stereo": 508,
      "n_features_tested": 4821,
      "n_significant": 287,
      "n_significant_pro_bias": 164,
      "n_significant_anti_bias": 123
    },
    "bisexual": { ... },
    ...
  }
}
```

### Global Summary

```
{run}/B_differential/differential_summary.json
```

```json
{
  "layers_analyzed": [0, 1, 2, ..., 31],
  "fdr_threshold": 0.05,
  "min_firing_rate_threshold": 0.05,
  "min_nonzero_count_threshold": 10,
  "min_n_per_group": 10,
  "test": "mann_whitney_u",
  "alternative": "two-sided",
  "context_condition_filter": "ambig",
  "comparison_type": "question_A_stereotyped_vs_all_other",
  "subgroup_catalog": {
    "gay": {
      "category": "so",
      "n_stereo": 312,
      "n_non_stereo": 508,
      "n_non_stereo_strict": 287,
      "n_unknown_in_non_stereo_group": 221,
      "total_ambig": 820,
      "analyzable": true
    },
    ...
  },
  "per_subgroup_results": {
    "gay": {
      "category": "so",
      "n_significant_by_layer": {
        "0": 17, "1": 23, "2": 45, ..., "14": 287, ..., "31": 112
      },
      "peak_layer": 14,
      "peak_n_significant": 287,
      "total_significant_pro_bias_all_layers": 1842,
      "total_significant_anti_bias_all_layers": 1356,
      "total_significant_all_layers": 3198
    },
    ...
  },
  "skipped_subgroups": {
    "pansexual": {"reason": "n_stereo=7 < min_n=10", "category": "so"}
  },
  "total_significant_features": 87432,
  "total_runtime_seconds": 8721.3
}
```

---

## Resume Safety

**Per-layer granularity.** Before processing a layer, check if the output parquet exists. If yes, skip. Layers are independent.

```python
if parquet_path.exists():
    log(f"Layer {layer:02d}: already processed, skipping")
    all_summaries[layer] = load_layer_summary(run_dir, layer)
    continue
```

**Atomic parquet writes** (tmp-then-rename) prevent corrupt parquets from crashed writes.

**No within-layer resume.** If a layer crashes mid-processing, it re-runs from scratch. Processing a single layer takes ~5-10 minutes with vectorized Mann-Whitney, so partial resume isn't worth the complexity.

---

## Output Structure

```
{run}/B_differential/
├── layer_00.parquet
├── layer_00_summary.json
├── layer_01.parquet
├── layer_01_summary.json
├── ...
├── layer_31.parquet
├── layer_31_summary.json
└── differential_summary.json
```

**No separate "full" vs "significant-only" parquets.** Single parquet per layer with `is_significant` flag. Downstream code filters as needed.

**No per-category parquets.** Category-general vs subgroup-specific patterns can be derived from subgroup-level results in B2.

---

## Compute Estimate

With vectorized Mann-Whitney and sparse matrix construction:

- Per (subgroup, layer) test: ~1-3 seconds (dominated by matrix slicing and statsmodels FDR)
- Per layer: ~40 subgroups × ~2s = ~80s compute + ~20s I/O = ~100-300s
- All 32 layers: ~1-2 hours total

This is ~3-5x faster than naive per-feature mannwhitneyu loops (~5+ hours).

With `--max_items 100`:
- Per layer: ~30 seconds
- All 32 layers: ~15 minutes

---

## Assumptions Summary

| # | Decision | Value |
|---|---|---|
| 1 | Metadata source | A2's `metadata.parquet` (fallback: .npz scanning) |
| 2 | Comparison type | Question A: stereotyped_target vs (non_stereotyped + unknown) |
| 3 | Context condition filter | ambig only |
| 4 | Multi-group item handling | Items contribute to all groups in stereotyped_groups |
| 5 | Minimum N per group | 10 (configurable via `--min_n_per_group`) |
| 6 | Statistical test | Mann-Whitney U, two-sided |
| 7 | Firing rate filter | max_group_firing ≥ 5% OR combined_nonzero ≥ 10 |
| 8 | FDR method | Benjamini-Hochberg, α=0.05 |
| 9 | FDR scope | Per (subgroup, layer), not global |
| 10 | Per-category analysis | Not included (derivable from subgroup results) |
| 11 | Parquet organization | Single file per layer with `is_significant` flag |
| 12 | Features stored | All passing firing filter (significant + non-significant) |
| 13 | Implementation | Vectorized sparse matrix + vectorized Mann-Whitney |

---

## Test Command

```bash
# Quick test: layer 14 only, 100 items, SO category
python scripts/B1_differential.py \
    --run_dir runs/llama-3.1-8b_2026-04-15/ \
    --layers 14 \
    --categories so \
    --max_items 100

# Verify output
python -c "
import pandas as pd
df = pd.read_parquet('runs/llama-3.1-8b_2026-04-15/B_differential/layer_14.parquet')
print(f'Total rows: {len(df)}')
print(f'Subgroups: {df[\"subgroup\"].unique()}')
print(f'Categories: {df[\"category\"].unique()}')
print(f'Significant: {df[\"is_significant\"].sum()}')
print(f'Pro-bias significant: {((df[\"is_significant\"]) & (df[\"direction\"]==\"pro_bias\")).sum()}')
print(f'Anti-bias significant: {((df[\"is_significant\"]) & (df[\"direction\"]==\"anti_bias\")).sum()}')
print(f'Effect size range (significant only):')
sig = df[df['is_significant']]
print(f'  cohen d: [{sig[\"cohens_d\"].min():.3f}, {sig[\"cohens_d\"].max():.3f}]')
print(f'Top 5 pro-bias features:')
print(sig[sig['direction']=='pro_bias'].nlargest(5, 'cohens_d')[['feature_idx','subgroup','cohens_d','p_value_fdr']])
"

# Full run
python scripts/B1_differential.py --run_dir runs/llama-3.1-8b_2026-04-15/ 2>&1 | tee logs/B1_differential.log
```