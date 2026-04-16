# B2: Feature Ranking Per Subgroup — Full Implementation Specification

## Purpose

Collect FDR-significant features for each subgroup across ALL transformer layers, rank them by effect size, determine effect-weighted injection layers, and compute cross-subgroup feature overlap. All outputs are advisory — C1 makes authoritative decisions about which features and layers to actually use for steering.

## Invocation

```bash
python scripts/B2_rank_features.py --run_dir runs/llama-3.1-8b_2026-04-15/
```

No additional required arguments. Reads B1 output and config.json.

Optional overrides:
- `--overlap_ks 5,10,20,50,100,200` — k values for Jaccard curve computation (default above)
- `--structural_overlap_threshold 0.5` — item-overlap fraction above which pairs are flagged as structural (default 0.5)
- `--skip_figures` — skip figure generation

---

## Input

- B1 parquets: `{run}/B_differential/layer_{NN:02d}.parquet`
- B1 summary: `{run}/B_differential/differential_summary.json`
- Metadata: `{run}/A_extraction/metadata.parquet`
- Config: `{run}/config.json`

## Dependencies

- `pandas`, `pyarrow`
- `numpy`
- `matplotlib` (for figures)
- `json` (stdlib)

---

## Identity Key Convention

Throughout B2, subgroups are identified by the composite string `"{category}/{subgroup}"`. In JSON this appears as a flat key; internally we track `(category, subgroup)` tuples.

```python
def make_sub_key(category: str, subgroup: str) -> str:
    return f"{category}/{subgroup}"

def parse_sub_key(key: str) -> tuple[str, str]:
    cat, sub = key.split("/", 1)
    return cat, sub
```

All subgroup names are normalized at read time:
```python
subgroup = str(subgroup).strip().lower()
```

This catches any whitespace or case inconsistencies from upstream.

---

## Step 1: Load and Collect All Significant Features

Load every B1 parquet and concatenate significant rows.

```python
def load_all_significant(run_dir: Path, n_layers: int) -> pd.DataFrame:
    """
    Load all FDR-significant features across all layers.
    
    Returns DataFrame with columns: feature_idx, layer, subgroup, category,
    cohens_d, p_value_raw, p_value_fdr, direction, firing_rate_stereo,
    firing_rate_non_stereo, mean_activation_stereo, mean_activation_non_stereo,
    n_stereo, n_non_stereo
    """
    differential_dir = run_dir / "B_differential"
    dfs = []
    
    for layer in range(n_layers):
        parquet_path = differential_dir / f"layer_{layer:02d}.parquet"
        if not parquet_path.exists():
            log(f"  WARNING: layer {layer:02d} parquet not found, skipping")
            continue
        
        df = pd.read_parquet(parquet_path)
        sig = df[df["is_significant"]].copy()
        dfs.append(sig)
    
    if not dfs:
        raise SystemExit("No B1 parquets found. Run B1 first.")
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Defensive normalization
    combined["subgroup"] = combined["subgroup"].astype(str).str.strip().str.lower()
    combined["category"] = combined["category"].astype(str).str.strip().str.lower()
    
    log(f"Loaded {len(combined)} significant features across {len(dfs)} layers")
    return combined
```

---

## Step 2: Defensive Deduplication

Within a (category, subgroup, feature_idx, layer) group, only one row should exist from B1. If duplicates appear (shouldn't, but defensive), keep the one with the largest |cohens_d|:

```python
def deduplicate_defensive(df: pd.DataFrame) -> pd.DataFrame:
    """Defensive dedup: shouldn't be needed if B1 is correct, but catches bugs."""
    before = len(df)
    df = df.assign(abs_d=df["cohens_d"].abs())
    df = df.sort_values("abs_d", ascending=False)
    df = df.drop_duplicates(subset=["category", "subgroup", "feature_idx", "layer"], keep="first")
    df = df.drop(columns=["abs_d"])
    after = len(df)
    
    if before != after:
        log(f"  WARNING: deduplication removed {before - after} duplicate rows")
    
    return df
```

**Cross-layer identity:** We do NOT deduplicate the same `feature_idx` across different layers. Feature 45021 at layer 14 and feature 45021 at layer 16 are from different SAEs — different features with the same index. The `(feature_idx, layer)` tuple is the unique feature identity throughout B2.

---

## Step 3: Enumerate Subgroups

Get the list of subgroups from two sources and cross-reference:

```python
def enumerate_subgroups(
    combined_df: pd.DataFrame,
    b1_summary: dict,
) -> tuple[list[tuple[str, str]], dict]:
    """
    Enumerate subgroups to rank.
    
    Returns:
        subgroups: list of (category, subgroup) tuples that have significant features
        enumeration_report: dict documenting the enumeration process
    """
    # Subgroups present in B1's output parquets
    in_parquet = set()
    for _, row in combined_df[["category", "subgroup"]].drop_duplicates().iterrows():
        in_parquet.add((row["category"], row["subgroup"]))
    
    # Subgroups present in B1's summary catalog (analyzable ones)
    in_catalog = set()
    catalog = b1_summary.get("subgroup_catalog", {})
    for sub_key, entry in catalog.items():
        if entry.get("analyzable"):
            sub_norm = str(sub_key).strip().lower()
            in_catalog.add((entry["category"], sub_norm))
    
    # Unions and differences
    all_subgroups = in_parquet | in_catalog
    only_parquet = in_parquet - in_catalog
    only_catalog = in_catalog - in_parquet
    
    report = {
        "total_subgroups": len(all_subgroups),
        "in_both": sorted([f"{c}/{s}" for c, s in in_parquet & in_catalog]),
        "only_in_parquet": sorted([f"{c}/{s}" for c, s in only_parquet]),
        "only_in_catalog": sorted([f"{c}/{s}" for c, s in only_catalog]),
    }
    
    if only_parquet:
        log(f"  WARNING: subgroups in parquet but not B1 catalog: {report['only_in_parquet']}")
    if only_catalog:
        log(f"  NOTE: subgroups in B1 catalog but no significant features found: {report['only_in_catalog']}")
    
    # Return all unique subgroups, sorted by category then subgroup
    return sorted(all_subgroups), report
```

This flags any subgroup-name drift between B1's catalog and its output parquets.

---

## Step 4: Rank Features Per Subgroup

For each (category, subgroup), split features by direction and sort by |cohens_d| descending. **Store ALL FDR-significant features — no k-cutoff.**

```python
def rank_features_all(combined_df: pd.DataFrame, subgroups: list[tuple[str, str]]) -> pd.DataFrame:
    """
    Rank all significant features per (category, subgroup, direction).
    
    Returns a flat DataFrame with one row per (category, subgroup, direction, rank).
    """
    ranked_rows = []
    
    for cat, sub in subgroups:
        sub_df = combined_df[
            (combined_df["category"] == cat) &
            (combined_df["subgroup"] == sub)
        ]
        
        for direction in ["pro_bias", "anti_bias"]:
            dir_df = sub_df[sub_df["direction"] == direction].copy()
            
            if dir_df.empty:
                continue  # no features for this direction
            
            # Sort by |cohens_d| descending
            dir_df["abs_d"] = dir_df["cohens_d"].abs()
            dir_df = dir_df.sort_values("abs_d", ascending=False).reset_index(drop=True)
            dir_df["rank"] = dir_df.index + 1  # 1-indexed rank
            dir_df = dir_df.drop(columns=["abs_d"])
            
            # Append
            for _, row in dir_df.iterrows():
                ranked_rows.append({
                    "category": cat,
                    "subgroup": sub,
                    "direction": direction,
                    "rank": int(row["rank"]),
                    "feature_idx": int(row["feature_idx"]),
                    "layer": int(row["layer"]),
                    "cohens_d": float(row["cohens_d"]),
                    "p_value_raw": float(row["p_value_raw"]),
                    "p_value_fdr": float(row["p_value_fdr"]),
                    "firing_rate_stereo": float(row["firing_rate_stereo"]),
                    "firing_rate_non_stereo": float(row["firing_rate_non_stereo"]),
                    "mean_activation_stereo": float(row["mean_activation_stereo"]),
                    "mean_activation_non_stereo": float(row["mean_activation_non_stereo"]),
                    "n_stereo": int(row["n_stereo"]),
                    "n_non_stereo": int(row["n_non_stereo"]),
                })
    
    return pd.DataFrame(ranked_rows)
```

**Output: `ranked_features.parquet`** — one row per (category, subgroup, direction, rank). Downstream code queries with:

```python
df = pd.read_parquet("ranked_features.parquet")
# Top 13 pro-bias features for so/gay
top_features = df[
    (df["category"] == "so") & 
    (df["subgroup"] == "gay") & 
    (df["direction"] == "pro_bias") & 
    (df["rank"] <= 13)
].sort_values("rank")
```

---

## Step 5: Determine Effect-Weighted Injection Layers

No k-cutoff. Use ALL FDR-significant features in each direction, weight each layer by the sum of |cohens_d| of its features.

```python
def compute_injection_layer_weighted(features: pd.DataFrame) -> dict | None:
    """
    Determine injection layer from effect-weighted sum across all significant features.
    
    Returns:
        {
            "injection_layer": int | None,
            "layer_scores": {layer: total_abs_cohens_d},
            "n_features": int,
            "top_layer_score": float,
            "score_concentration": float  # top_layer_score / total_score
        }
    """
    if features.empty:
        return None
    
    layer_scores = features.groupby("layer")["cohens_d"].apply(
        lambda x: float(np.sum(np.abs(x)))
    ).to_dict()
    
    if not layer_scores:
        return None
    
    max_score = max(layer_scores.values())
    total_score = sum(layer_scores.values())
    
    # Ties broken by preferring deeper layers (more direct effect on logits)
    candidates = [l for l, s in layer_scores.items() if s == max_score]
    injection_layer = max(candidates)
    
    return {
        "injection_layer": int(injection_layer),
        "layer_scores": {int(k): round(v, 4) for k, v in layer_scores.items()},
        "n_features": int(len(features)),
        "top_layer_score": round(float(max_score), 4),
        "score_concentration": round(float(max_score / max(total_score, 1e-12)), 4),
    }
```

**Rationale:** Effect-weighted ranks layers by their total bias signal, not just feature count. A layer with one very strong feature (d=2.0) is weighted equally to a layer with four moderate features (d=0.5 each). This matches the steering use case — we care about which layer has the strongest bias representation, not which has the most features.

**Why prefer deeper layers on tie:** Later layers are closer to the output and have more direct influence on the model's next-token prediction. Steering at deeper layers produces cleaner effects.

**`score_concentration`** measures how focused the bias signal is:
- 1.0: all significant effect lives at one layer (highly concentrated)
- 0.1: effect is spread across ~10 layers roughly evenly (distributed)

This is diagnostic — subgroups with diffuse signals may be harder to steer with a single-layer intervention.

### Per-Subgroup Injection Layer Record

```python
def build_injection_layer_record(
    ranked_df: pd.DataFrame,
    cat: str,
    sub: str,
) -> dict:
    """Build the injection layer record for one subgroup covering both directions."""
    pro = ranked_df[
        (ranked_df["category"] == cat) &
        (ranked_df["subgroup"] == sub) &
        (ranked_df["direction"] == "pro_bias")
    ]
    anti = ranked_df[
        (ranked_df["category"] == cat) &
        (ranked_df["subgroup"] == sub) &
        (ranked_df["direction"] == "anti_bias")
    ]
    
    pro_layer = compute_injection_layer_weighted(pro)
    anti_layer = compute_injection_layer_weighted(anti)
    
    record = {
        "category": cat,
        "subgroup": sub,
        "pro_bias": pro_layer,
        "anti_bias": anti_layer,
    }
    
    if pro_layer is None:
        record["pro_bias_note"] = "No significant pro-bias features"
    if anti_layer is None:
        record["anti_bias_note"] = "No significant anti-bias features"
    
    return record
```

Subgroups with null `pro_bias` cannot be steered to reduce bias (no targetable features). C1 should skip these.

---

## Step 6: Cross-Subgroup Feature Overlap — Jaccard Curves

For each pair of subgroups WITHIN the same category, compute Jaccard overlap at multiple k values to show the full k-dependence.

```python
K_VALUES_DEFAULT = [5, 10, 20, 50, 100, 200]

def compute_overlap_curve(
    ranked_A: pd.DataFrame,
    ranked_B: pd.DataFrame,
    k_values: list[int],
) -> dict:
    """
    Compute Jaccard overlap at multiple k values.
    
    ranked_A, ranked_B: DataFrames of features for one subgroup+direction, sorted by rank
    
    Returns:
        {
            k_value: {jaccard, directed_A_to_B, directed_B_to_A, 
                      n_shared, n_A, n_B, effective_k_A, effective_k_B}
        }
    """
    results = {}
    
    for k in k_values:
        # If a subgroup has fewer than k features, use all available
        set_A = set()
        for _, row in ranked_A.head(k).iterrows():
            set_A.add((int(row["feature_idx"]), int(row["layer"])))
        
        set_B = set()
        for _, row in ranked_B.head(k).iterrows():
            set_B.add((int(row["feature_idx"]), int(row["layer"])))
        
        if not set_A or not set_B:
            results[k] = {
                "jaccard": None,
                "directed_A_to_B": None,
                "directed_B_to_A": None,
                "n_shared": 0,
                "effective_k_A": len(set_A),
                "effective_k_B": len(set_B),
            }
            continue
        
        intersection = set_A & set_B
        union = set_A | set_B
        
        jaccard = len(intersection) / len(union)
        directed_A_to_B = len(intersection) / len(set_A)
        directed_B_to_A = len(intersection) / len(set_B)
        
        results[k] = {
            "jaccard": round(jaccard, 4),
            "directed_A_to_B": round(directed_A_to_B, 4),
            "directed_B_to_A": round(directed_B_to_A, 4),
            "n_shared": int(len(intersection)),
            "effective_k_A": int(len(set_A)),
            "effective_k_B": int(len(set_B)),
        }
    
    return results
```

**Directed overlap uses actual set sizes**, not k. If subgroup A has only 8 pro-bias features and we're computing at k=20, set_A has size 8 and directed_A_to_B = intersection / 8.

### Full Pairwise Overlap

```python
def compute_all_overlaps(
    ranked_df: pd.DataFrame,
    subgroups: list[tuple[str, str]],
    k_values: list[int],
) -> dict:
    """
    Compute overlap curves for all subgroup pairs within each category,
    for both pro_bias and anti_bias directions.
    """
    # Group subgroups by category
    by_category = defaultdict(list)
    for cat, sub in subgroups:
        by_category[cat].append(sub)
    
    overlap_results = {}
    
    for cat, subs_in_cat in by_category.items():
        if len(subs_in_cat) < 2:
            continue
        
        overlap_results[cat] = {
            "subgroups": sorted(subs_in_cat),
            "k_values": k_values,
            "pro_bias": {},
            "anti_bias": {},
        }
        
        for direction in ["pro_bias", "anti_bias"]:
            for i, sub_A in enumerate(sorted(subs_in_cat)):
                for sub_B in sorted(subs_in_cat)[i+1:]:
                    pair_key = f"{sub_A}__{sub_B}"  # double underscore separator
                    
                    ranked_A = ranked_df[
                        (ranked_df["category"] == cat) &
                        (ranked_df["subgroup"] == sub_A) &
                        (ranked_df["direction"] == direction)
                    ].sort_values("rank")
                    
                    ranked_B = ranked_df[
                        (ranked_df["category"] == cat) &
                        (ranked_df["subgroup"] == sub_B) &
                        (ranked_df["direction"] == direction)
                    ].sort_values("rank")
                    
                    curve = compute_overlap_curve(ranked_A, ranked_B, k_values)
                    overlap_results[cat][direction][pair_key] = curve
    
    return overlap_results
```

**Connection to fragmentation:**
- Low Jaccard (< 0.1) at k=20: subgroups have fragmented pro-bias feature sets
- High Jaccard (> 0.3) at k=20: subgroups share bias mechanism
- Jaccard increases with k: top features differ but broader sets converge (weak fragmentation)
- Jaccard flat across k: consistent fragmentation at all scales (robust finding)

---

## Step 7: Item Overlap Report

For each pair of subgroups within a category, compute what fraction of items are shared. Flag "structural" overlap (above threshold — typically disability co-labels).

```python
def compute_item_overlap(
    metadata_df: pd.DataFrame,
    subgroups: list[tuple[str, str]],
    structural_threshold: float = 0.5,
) -> dict:
    """
    Compute item-level overlap for each subgroup pair within a category.
    
    An item targeting both subgroups contributes to both groups' differential
    analyses in B1. High overlap means the feature/direction comparisons for
    those subgroups are based on largely shared data.
    """
    ambig_df = metadata_df[metadata_df["context_condition"] == "ambig"].copy()
    ambig_df["stereotyped_groups"] = ambig_df["stereotyped_groups"].apply(
        lambda x: x if isinstance(x, list) else json.loads(x)
    )
    
    # Build {(category, subgroup): set of item_idxs}
    sub_items = {}
    for cat, sub in subgroups:
        items = ambig_df[
            (ambig_df["category"] == cat) &
            (ambig_df["stereotyped_groups"].apply(lambda gs: sub in gs))
        ]["item_idx"].tolist()
        sub_items[(cat, sub)] = set(items)
    
    # Group by category
    by_category = defaultdict(list)
    for cat, sub in subgroups:
        by_category[cat].append(sub)
    
    result = {}
    for cat, subs_in_cat in by_category.items():
        if len(subs_in_cat) < 2:
            continue
        
        result[cat] = {
            "subgroups": sorted(subs_in_cat),
            "item_counts": {s: len(sub_items[(cat, s)]) for s in subs_in_cat},
            "pairwise": {},
            "structural_pairs": [],
        }
        
        for i, sub_A in enumerate(sorted(subs_in_cat)):
            for sub_B in sorted(subs_in_cat)[i+1:]:
                items_A = sub_items[(cat, sub_A)]
                items_B = sub_items[(cat, sub_B)]
                shared = items_A & items_B
                
                jaccard = len(shared) / max(len(items_A | items_B), 1)
                frac_A_in_B = len(shared) / max(len(items_A), 1)
                frac_B_in_A = len(shared) / max(len(items_B), 1)
                max_frac = max(frac_A_in_B, frac_B_in_A)
                
                pair_key = f"{sub_A}__{sub_B}"
                pair_entry = {
                    "n_shared": int(len(shared)),
                    "jaccard": round(jaccard, 4),
                    "fraction_of_A_in_B": round(frac_A_in_B, 4),
                    "fraction_of_B_in_A": round(frac_B_in_A, 4),
                    "max_fraction": round(max_frac, 4),
                    "is_structural": bool(max_frac >= structural_threshold),
                }
                
                result[cat]["pairwise"][pair_key] = pair_entry
                
                if pair_entry["is_structural"]:
                    result[cat]["structural_pairs"].append(pair_key)
        
        if result[cat]["structural_pairs"]:
            log(f"  {cat}: structural item overlap in pairs {result[cat]['structural_pairs']}")
    
    return result
```

**Expected structural cases:**
- `disability`: "disabled" and "physically disabled" will likely have >50% overlap (many items labeled with both)
- `so`: subgroups usually disjoint (single primary identity per item)

Structural pairs should be interpreted with caution — feature overlap between them reflects shared data, not necessarily shared representational structure.

---

## Step 8: Output Files

### `ranked_features.parquet`

Primary output. Flat table with all ranked features.

```
{run}/B_feature_ranking/ranked_features.parquet
```

| Column | Type | Description |
|---|---|---|
| `category` | string | Category short name |
| `subgroup` | string | Subgroup label (normalized) |
| `direction` | string | "pro_bias" or "anti_bias" |
| `rank` | int32 | 1-indexed rank within (category, subgroup, direction) |
| `feature_idx` | int32 | SAE feature index |
| `layer` | int32 | SAE layer |
| `cohens_d` | float32 | Effect size (signed) |
| `p_value_raw` | float64 | Raw Mann-Whitney p-value |
| `p_value_fdr` | float64 | FDR-corrected p-value |
| `firing_rate_stereo` | float32 | Firing rate in stereotyped-response group |
| `firing_rate_non_stereo` | float32 | Firing rate in non-stereotyped-response group |
| `mean_activation_stereo` | float32 | Mean activation, stereotyped group |
| `mean_activation_non_stereo` | float32 | Mean activation, non-stereotyped group |
| `n_stereo` | int32 | Stereotyped response N for this subgroup |
| `n_non_stereo` | int32 | Non-stereotyped response N for this subgroup |

### `injection_layers.json`

Per-subgroup injection layers for both directions.

```json
{
  "so/gay": {
    "category": "so",
    "subgroup": "gay",
    "pro_bias": {
      "injection_layer": 14,
      "layer_scores": {"10": 0.8, "12": 2.1, "14": 8.4, "16": 3.2, ...},
      "n_features": 287,
      "top_layer_score": 8.4,
      "score_concentration": 0.412
    },
    "anti_bias": {
      "injection_layer": 18,
      "layer_scores": {...},
      "n_features": 142,
      "top_layer_score": 5.1,
      "score_concentration": 0.287
    }
  },
  "so/pansexual": {
    "category": "so",
    "subgroup": "pansexual",
    "pro_bias": null,
    "pro_bias_note": "No significant pro-bias features",
    "anti_bias": {...}
  },
  ...
}
```

### `feature_overlap.json`

Jaccard curves per category, per direction, per pair.

```json
{
  "so": {
    "subgroups": ["bisexual", "gay", "lesbian", "pansexual"],
    "k_values": [5, 10, 20, 50, 100, 200],
    "pro_bias": {
      "bisexual__gay": {
        "5": {"jaccard": 0.0, "directed_A_to_B": 0.0, "directed_B_to_A": 0.0, "n_shared": 0, "effective_k_A": 5, "effective_k_B": 5},
        "10": {"jaccard": 0.053, ...},
        "20": {"jaccard": 0.081, ...},
        "50": {"jaccard": 0.136, ...},
        "100": {"jaccard": 0.189, ...},
        "200": {"jaccard": 0.244, ...}
      },
      "bisexual__lesbian": {...},
      ...
    },
    "anti_bias": {...}
  },
  ...
}
```

### `item_overlap_report.json`

Item-level overlap (not "warnings" — just information).

```json
{
  "structural_threshold": 0.5,
  "per_category": {
    "so": {
      "subgroups": ["bisexual", "gay", "lesbian", "pansexual"],
      "item_counts": {"bisexual": 820, "gay": 820, "lesbian": 820, "pansexual": 820},
      "pairwise": {
        "bisexual__gay": {
          "n_shared": 0,
          "jaccard": 0.0,
          "fraction_of_A_in_B": 0.0,
          "fraction_of_B_in_A": 0.0,
          "max_fraction": 0.0,
          "is_structural": false
        },
        ...
      },
      "structural_pairs": []
    },
    "disability": {
      "subgroups": ["disabled", "physically disabled", "cognitively disabled"],
      "item_counts": {"disabled": 1980, "physically disabled": 1820, "cognitively disabled": 580},
      "pairwise": {
        "disabled__physically disabled": {
          "n_shared": 1820,
          "jaccard": 0.919,
          "fraction_of_A_in_B": 0.919,
          "fraction_of_B_in_A": 1.0,
          "max_fraction": 1.0,
          "is_structural": true
        },
        ...
      },
      "structural_pairs": ["disabled__physically disabled"]
    }
  }
}
```

### `ranking_summary.json`

Top-level summary of the B2 run.

```json
{
  "n_layers_loaded": 32,
  "n_subgroups_total": 38,
  "n_subgroups_with_pro_bias": 36,
  "n_subgroups_with_anti_bias": 34,
  "n_subgroups_no_features": 2,
  "subgroups_no_features": ["gi/nonbinary", "age/young"],
  "k_values_used": [5, 10, 20, 50, 100, 200],
  "structural_threshold": 0.5,
  "enumeration_report": {...},
  "per_category_summary": {
    "so": {
      "n_subgroups": 4,
      "n_subgroups_with_pro_bias": 4,
      "mean_n_pro_bias_features": 234,
      "mean_n_anti_bias_features": 156,
      "injection_layer_distribution_pro_bias": {"14": 2, "16": 1, "18": 1}
    },
    ...
  },
  "total_runtime_seconds": 42.1
}
```

---

## Step 9: Figures

All figures use the Wong colorblind-safe palette (orange `#E69F00`, blue `#0072B2`, green `#009E73`, purple `#CC79A7`, vermillion `#D55E00`, sky blue `#56B4E9`, yellow `#F0E442`, black `#000000`). Distinct markers per subgroup/category as secondary visual channel.

### `fig_feature_overlap_{category}.png`

Heatmap of Jaccard at k=20 for all subgroup pairs in a category. One heatmap per category with ≥2 subgroups. Separate panels for pro_bias and anti_bias (or one figure with two subplots).

- Rows and columns: subgroups (sorted)
- Cell values: Jaccard overlap at k=20
- Diagonal: masked (or shown as gray)
- N annotated next to row labels: "{subgroup} (N={n_features})"
- Colormap: `YlOrRd` sequential, vmin=0, vmax=1
- Cells annotated with values to 2 decimals
- Title: "Top-20 pro-bias feature overlap — {category}"

### `fig_overlap_curves_{category}.png`

Line plots showing Jaccard as a function of k for all subgroup pairs in a category.

- One subplot per direction (pro_bias, anti_bias) — two subplots total per figure
- X-axis: k (log scale: 5, 10, 20, 50, 100, 200)
- Y-axis: Jaccard overlap (0 to 1)
- One line per subgroup pair, with distinct color + marker combinations
- Legend with subgroup pair names
- Horizontal dashed line at Jaccard = 0.1 ("fragmented") and 0.3 ("shared mechanism")
- Title: "Feature overlap vs. k — {category}"

### `fig_ranked_effect_sizes_{category}.png`

For each category, overlay effect-size curves for all its subgroups.

- One subplot per direction
- X-axis: rank (1 to max_rank per subgroup, limited to 100 for display)
- Y-axis: |cohens_d|
- One line per subgroup, distinct color + marker
- Title: "Ranked effect sizes — {category}"

### `fig_feature_count_per_subgroup.png`

Bar chart across all subgroups showing feature counts.

- X-axis: subgroups (grouped by category, separated by spacing)
- Y-axis: number of significant features
- Paired bars per subgroup: pro-bias (blue #0072B2) and anti-bias (orange #E69F00)
- Annotated with values above bars
- Title: "Significant features per subgroup (FDR < 0.05)"
- Rotate x-axis labels 45 degrees for readability

### `fig_layer_distribution_heatmap.png`

Heatmap showing where features concentrate across layers for each subgroup.

- Rows: subgroups (grouped by category, with category separator lines)
- Columns: layers (0 to n_layers-1)
- Cells: effect-weighted score for that subgroup at that layer (sum of |cohens_d| across significant features at that layer)
- Separate heatmaps for pro_bias and anti_bias
- Colormap: `viridis` sequential
- Annotate injection layer (from Step 5) with a star marker
- Title: "Effect-weighted layer distribution — {direction}"

### `fig_injection_layer_distribution.png`

Histogram of injection layer selections across all subgroups.

- X-axis: layer (0 to n_layers-1)
- Y-axis: number of subgroups with that injection layer
- Paired bars: pro-bias (blue) and anti-bias (orange)
- Color by direction
- Title: "Distribution of injection layers across subgroups"

---

## Step 10: Output Structure

```
{run}/B_feature_ranking/
├── ranked_features.parquet              # All FDR-significant features, ranked per (category, subgroup, direction)
├── injection_layers.json                # Effect-weighted injection layers (pro_bias + anti_bias)
├── feature_overlap.json                 # Jaccard curves across k values
├── item_overlap_report.json             # Item-level overlap with structural flag
├── ranking_summary.json                 # Top-level summary
└── figures/
    ├── fig_feature_overlap_{category}.png/.pdf
    ├── fig_overlap_curves_{category}.png/.pdf
    ├── fig_ranked_effect_sizes_{category}.png/.pdf
    ├── fig_feature_count_per_subgroup.png/.pdf
    ├── fig_layer_distribution_heatmap.png/.pdf
    └── fig_injection_layer_distribution.png/.pdf
```

---

## Step 11: Resume Safety

B2 is relatively fast (~1-2 minutes) so resume granularity is coarse — just skip the entire script if `ranking_summary.json` exists and pointing at the correct B1 run.

```python
summary_path = run_dir / "B_feature_ranking" / "ranking_summary.json"
if summary_path.exists() and not args.force:
    log(f"B2 output already exists at {summary_path}. Use --force to rerun.")
    sys.exit(0)
```

Atomic writes for all output files (tmp-then-rename).

---

## Compute Estimate

- Loading 32 parquets: ~10-20s
- Ranking and overlap computation: ~30-60s
- Figure generation: ~30-60s
- Total: ~1-2 minutes

---

## Assumptions Summary

| # | Decision | Value |
|---|---|---|
| 1 | Identity key | `(category, subgroup)` composite, rendered as `"{cat}/{sub}"` |
| 2 | Name normalization | `.strip().lower()` on all subgroup/category names at read time |
| 3 | Ranked features storage | All FDR-significant features, no k-cutoff |
| 4 | Output format | Parquet for ranked features, JSON for summaries |
| 5 | Ranking metric | \|cohens_d\| descending; both raw and FDR p-values preserved |
| 6 | Direction separation | Pro-bias and anti-bias ranked/reported separately |
| 7 | Cross-layer deduplication | Same feature_idx at different layers = different features (not deduplicated) |
| 8 | Injection layer method | Effect-weighted sum of \|cohens_d\|, no k-cutoff |
| 9 | Tie-breaking | Prefer deeper layers (closer to output) |
| 10 | Injection layer advisory | C1 is authoritative; B2 reports are for guidance |
| 11 | Overlap k values | [5, 10, 20, 50, 100, 200] — default, configurable |
| 12 | Overlap directed computation | Uses actual set sizes (not fixed k) |
| 13 | Overlap directions | Pro-bias and anti-bias computed separately |
| 14 | Structural overlap threshold | max_fraction ≥ 0.5 flags pair as structural |
| 15 | Renaming | `item_overlap_report.json` not `item_overlap_warnings.json` |

---

## Test Command

```bash
# Run B2
python scripts/B2_rank_features.py --run_dir runs/llama-3.1-8b_2026-04-15/

# Verify outputs
python -c "
import pandas as pd
import json

# Check ranked features
df = pd.read_parquet('runs/llama-3.1-8b_2026-04-15/B_feature_ranking/ranked_features.parquet')
print(f'Ranked features: {len(df)} rows')
print(f'Categories: {sorted(df[\"category\"].unique())}')
print(f'Subgroups: {df[\"subgroup\"].nunique()}')
print(f'Directions: {sorted(df[\"direction\"].unique())}')
print()

# Top 5 pro-bias features for so/gay
top = df[(df['category']=='so') & (df['subgroup']=='gay') & (df['direction']=='pro_bias')].nsmallest(5, 'rank')
print('Top 5 pro-bias features for so/gay:')
print(top[['rank', 'feature_idx', 'layer', 'cohens_d', 'p_value_fdr']])
print()

# Injection layers
with open('runs/llama-3.1-8b_2026-04-15/B_feature_ranking/injection_layers.json') as f:
    inj = json.load(f)
print(f'Injection layers: {len(inj)} subgroups')
for key, rec in list(inj.items())[:3]:
    pro = rec.get('pro_bias', {})
    print(f'  {key}: pro_bias layer={pro.get(\"injection_layer\") if pro else None}, n_features={pro.get(\"n_features\") if pro else 0}')

# Overlap
with open('runs/llama-3.1-8b_2026-04-15/B_feature_ranking/feature_overlap.json') as f:
    overlap = json.load(f)
for cat in overlap:
    print(f'\\n{cat}: {len(overlap[cat][\"pro_bias\"])} pairs')
    for pair, curve in list(overlap[cat]['pro_bias'].items())[:2]:
        print(f'  {pair}: Jaccard at k=20 = {curve[\"20\"][\"jaccard\"]}')

# Item overlap
with open('runs/llama-3.1-8b_2026-04-15/B_feature_ranking/item_overlap_report.json') as f:
    items = json.load(f)
for cat, rec in items['per_category'].items():
    structural = rec.get('structural_pairs', [])
    if structural:
        print(f'\\n{cat} structural pairs: {structural}')
"
```