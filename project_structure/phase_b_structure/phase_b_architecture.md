# Phase B: Offline Analysis Pipeline — Detailed Specification

## Overview

Phase B is the analysis stage. Everything here reads from Phase A artifacts — no model forward passes, no GPU required (except optionally for faster matrix operations). All stages are re-runnable and iteratable.

**Stages:**
- **B1** — Differential feature analysis (which SAE features are bias-associated per subgroup?)
- **B2** — Feature ranking per subgroup (aggregate across layers, rank, compute overlap)
- **B3** — Subgroup direction geometry (DIM directions, pairwise cosines — independent of SAE)
- **B4** — Probe training + controls (what's linearly decodable, and is it genuine?)
- **B5** — Feature interpretability, item-level (what do the top features respond to?)

**Dependencies:**
```
A1 → A2 → A3 ──┬── B1 → B2 → B5
                ├── B3 (independent of SAE — reads A2 directly)
                └── B4 (independent of SAE — reads A2 directly)
```

B3 and B4 branch from A2 and don't need SAE encodings. B1 needs A3. B2 needs B1. B5 needs both A3 and B2.

**Resource requirements:** CPU only. The most expensive step is B1, which loads 32 parquets and runs ~131K Mann-Whitney tests per (subgroup, layer) combination. With ~40 subgroups across 9 categories and 32 layers, that's ~160M tests total — but each is fast (~0.1ms), so the total is ~4-5 hours. B2-B5 are minutes each.

---

## B1: Differential Feature Analysis

### Purpose

For each subgroup at each layer, identify SAE features that activate differently when the model produces a stereotyped response vs. when it doesn't. This is the core statistical test that surfaces "bias-associated features."

### Invocation

```bash
python scripts/B1_differential.py --run_dir runs/llama-3.1-8b_2026-04-15/

# Quick test
python scripts/B1_differential.py --run_dir runs/llama-3.1-8b_2026-04-15/ --max_items 100

# Single category
python scripts/B1_differential.py --run_dir runs/llama-3.1-8b_2026-04-15/ --categories so
```

### Input

- SAE encodings: `{run}/A_extraction/sae_encoding/layer_{NN}.parquet`
- Behavioral metadata: `{run}/A_extraction/activations/{category}/item_*.npz` (for model_answer_role, stereotyped_groups, context_condition)
- Stimuli: `{run}/A_extraction/stimuli/{category}.json` (for answer_roles, stereotyped_groups)

### Step 0: Build the Item Metadata Table

Before running any statistics, consolidate all behavioral metadata into a single dataframe per category for fast lookup:

```python
item_meta = []
for npz_path in sorted(activations_dir.glob("item_*.npz")):
    data = np.load(npz_path, allow_pickle=True)
    meta = json.loads(data["metadata_json"].item())
    item_meta.append({
        "item_idx": meta["item_idx"],
        "category": meta["category"],
        "model_answer_role": meta["model_answer_role"],
        "is_stereotyped_response": meta["is_stereotyped_response"],
        "context_condition": meta["context_condition"],
        "stereotyped_groups": meta["stereotyped_groups"],
        "n_target_groups": meta["n_target_groups"],
        "margin": meta["margin"],
    })

meta_df = pd.DataFrame(item_meta)
```

This is loaded once and reused for all layers.

### Step 1: Define the Comparison Groups

For a given subgroup S, the comparison is:

**Stereotyped-response group:** Items where:
- S ∈ `stereotyped_groups` (item targets this subgroup — may also target others)
- `context_condition == "ambig"` (ambiguous context only)
- `model_answer_role == "stereotyped_target"` (model chose the biased answer)

**Non-stereotyped-response group:** Items where:
- S ∈ `stereotyped_groups` (same subgroup targeting criterion)
- `context_condition == "ambig"` (same context condition)
- `model_answer_role != "stereotyped_target"` (model chose "unknown" or the non-stereotyped answer)

```python
def get_comparison_groups(meta_df: pd.DataFrame, subgroup: str) -> tuple[list[int], list[int]]:
    """Return (stereotyped_item_idxs, non_stereotyped_item_idxs) for a subgroup."""
    # Items targeting this subgroup in ambiguous context
    targeting = meta_df[
        meta_df["stereotyped_groups"].apply(lambda gs: subgroup in gs)
        & (meta_df["context_condition"] == "ambig")
    ]
    
    stereo_idxs = targeting[targeting["model_answer_role"] == "stereotyped_target"]["item_idx"].tolist()
    non_stereo_idxs = targeting[targeting["model_answer_role"] != "stereotyped_target"]["item_idx"].tolist()
    
    return stereo_idxs, non_stereo_idxs
```

**Methodological choice — ambiguous items only for differential analysis:**

Disambiguated items provide evidential context that should override stereotypes. A stereotyped response on a disambiguated item could indicate the model misunderstood the evidence rather than relying on bias. Ambiguous items are where the model has no evidence-based reason to prefer one answer, so choosing the stereotyped option specifically reveals reliance on prior stereotypical associations. This is the comparison BBQ was designed for.

Disambiguated items are NOT discarded — they remain in the dataset for other uses:
- B4 probes can train on both conditions
- C1 steering can evaluate on both conditions
- The contrast between ambig and disambig performance is itself informative

**Methodological choice — multi-group items contribute to all their groups:**

An item with `stereotyped_groups: ["gay", "lesbian"]` contributes to BOTH the "gay" and "lesbian" comparison groups. This maximizes N per subgroup. The `n_target_groups` field allows downstream analysis to sensitivity-test by restricting to single-group items only.

### Step 2: The Statistical Test

For each feature F (out of 131,072) at layer L for subgroup S:

**2a. Collect activation values:**

```python
# Load the layer parquet
layer_df = pd.read_parquet(f"layer_{L:02d}.parquet")

# Filter to the category containing subgroup S
cat_df = layer_df[layer_df["category"] == category_of(S)]

# Pivot to get activation per (item_idx, feature_idx) — but stay in long format for efficiency
# For a specific feature F:
feat_df = cat_df[cat_df["feature_idx"] == F]
feat_by_item = feat_df.set_index("item_idx")["activation_value"]

# Items NOT in feat_by_item have activation = 0 for this feature
a_stereo = [feat_by_item.get(idx, 0.0) for idx in stereo_idxs]
a_non_stereo = [feat_by_item.get(idx, 0.0) for idx in non_stereo_idxs]
```

**Optimization note:** The naive approach above (looping over 131K features × 40 subgroups × 32 layers) is slow. The efficient approach:

```python
# For all features at once for a given subgroup:
# 1. Build a wide matrix for the subgroup's items: (n_items, n_active_features)
# 2. Run vectorized tests across all features simultaneously

# Practical approach: pivot the parquet to a sparse matrix per category,
# then index into it for each subgroup's item sets.
```

The implementation should process one (category, layer) at a time, building the sparse item×feature matrix once, then testing all subgroups within that category. This avoids re-loading the parquet for each subgroup.

**2b. Compute Cohen's d:**

```python
def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's d (positive means a > b)."""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0
    m1, m2 = np.mean(a), np.mean(b)
    v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_var = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
    pooled_std = np.sqrt(max(pooled_var, 1e-12))
    return (m1 - m2) / pooled_std
```

Positive d → feature fires MORE for stereotyped responses → **pro-bias feature**
Negative d → feature fires MORE for non-stereotyped responses → **anti-bias feature**

**2c. Compute p-value — Mann-Whitney U test:**

```python
from scipy.stats import mannwhitneyu

def mw_test(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided Mann-Whitney U test. Returns p-value."""
    # Handle degenerate cases
    if len(a) < 3 or len(b) < 3:
        return 1.0
    if np.all(a == 0) and np.all(b == 0):
        return 1.0  # both groups have zero activation — no signal
    try:
        _, p = mannwhitneyu(a, b, alternative="two-sided")
        return float(p)
    except ValueError:
        return 1.0  # identical distributions
```

**Methodological choice — Mann-Whitney U over t-test:**

SAE feature activations are heavily zero-inflated: for any given feature, 90-99% of items have exactly zero activation. The remaining items have continuous positive values. This distribution is severely non-normal (a spike at zero plus a right-skewed tail). A t-test assumes normality and would be unreliable here.

Mann-Whitney tests whether the ranks differ between groups. It handles zero-inflation naturally: if both groups have mostly zeros, the test returns a high p-value (no difference). If one group has more nonzero items or higher activation values among its nonzero items, the ranks will differ and the p-value will be low.

The cost is slightly less statistical power than a t-test when normality holds, but the robustness gain is worth it given our distributions.

**2d. Compute firing rates:**

```python
firing_rate_stereo = np.mean(np.array(a_stereo) > 0)
firing_rate_non_stereo = np.mean(np.array(a_non_stereo) > 0)
```

These are interpretability aids:
- Feature with d=0.8, firing_rate=0.02: fires rarely but intensely when it does — a highly specific feature
- Feature with d=0.3, firing_rate=0.6: fires broadly with moderate bias — a general feature with a slight lean

**2e. Compute mean activations (nonzero and overall):**

```python
mean_activation_stereo = np.mean(a_stereo)
mean_activation_non_stereo = np.mean(a_non_stereo)
```

These are useful for B5's specificity analysis — the difference in mean activation is what drives cohen's d, and knowing the absolute levels helps interpret feature behavior.

### Step 3: Minimum Firing Rate Filter

Before including a feature in the results, apply a minimum firing rate threshold to exclude features with too few nonzero activations to be meaningfully characterized:

```python
combined = np.concatenate([a_stereo, a_non_stereo])
n_nonzero = np.sum(combined > 0)
n_total = len(combined)

# Require: at least 5% nonzero in the larger group OR at least 10 nonzero items total
max_group_firing = max(firing_rate_stereo, firing_rate_non_stereo)
passes_filter = (max_group_firing >= 0.05) or (n_nonzero >= 10)
```

Features failing this filter get p_value = 1.0 and are excluded from significance testing but still recorded in the parquet (with `is_significant = False`) for completeness.

**Rationale:** A feature that fires on 2 out of 200 items can produce a low p-value by chance (both nonzero items happened to be in the stereotyped group). The minimum firing rate prevents this.

### Step 4: FDR Correction

Apply Benjamini-Hochberg correction across all features that passed the firing rate filter, within each (subgroup, layer) combination:

```python
from scipy.stats import false_discovery_control  # or statsmodels.stats.multitest

# Collect all raw p-values for features passing the filter at this (subgroup, layer)
raw_pvalues = [...]  # one per feature that passed filter
feature_indices = [...]  # corresponding feature indices

# Benjamini-Hochberg
rejected, p_adjusted, _, _ = multipletests(raw_pvalues, method="fdr_bh", alpha=0.05)

# Assign back
for i, fidx in enumerate(feature_indices):
    results[fidx]["p_value_fdr"] = p_adjusted[i]
    results[fidx]["is_significant"] = rejected[i]
```

**Scope of FDR correction — per (subgroup, layer):**

We correct within each (subgroup, layer) combination independently. NOT globally across all subgroups × all layers.

**Rationale:** The scientific question is specific: "at this layer, for this subgroup, which features are differentially active?" Correcting across all layers would be overly conservative because each layer represents a different stage of computation with independent features. Correcting across subgroups would conflate separate hypotheses.

With 131K features per test, the FDR correction is already stringent. A feature needs a raw p-value of approximately `0.05 * rank / 131072` to survive, which for the top-ranked feature means p < 3.8 × 10⁻⁷. This is conservative enough to be credible.

### Step 5: Assign Direction Labels

```python
direction = "pro_bias" if cohens_d > 0 else "anti_bias"
```

Pro-bias features are candidates for dampening (to reduce bias). Anti-bias features are candidates for amplifying (to counteract bias). Both are interesting and both get ranked in B2.

### Output

**Per-subgroup parquets — one per layer, containing all subgroups and categories:**

```
{run}/B_differential/per_subgroup/layer_00.parquet
{run}/B_differential/per_subgroup/layer_01.parquet
...
{run}/B_differential/per_subgroup/layer_31.parquet
```

**Parquet schema:**

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

**Only significant features are stored** (to keep parquet sizes manageable). Non-significant features are discarded after testing. If a downstream analysis needs the full distribution of effect sizes (including non-significant), we can add a `--save_all` flag, but the default should be significant-only.

**Actually, reconsideration:** Storing only significant features makes it impossible to do post-hoc analysis of the effect size distribution or to re-run with a different FDR threshold. Better approach: store ALL features that passed the minimum firing rate filter (Step 3), with their p-values and is_significant flag. This is more rows per parquet but still manageable:

Expected: ~131K features × ~40 subgroups per layer, but only ~5-20% pass the firing rate filter → ~1M-5M rows per parquet at ~50 bytes/row → ~50-250MB per layer.

That might be too large for 32 layers. **Compromise:** Store all features that passed the firing rate filter in a "full" directory, and a "significant-only" version in a separate directory:

```
{run}/B_differential/
├── per_subgroup/          # Significant features only (small, fast to load)
│   ├── layer_00.parquet
│   └── ...
├── per_subgroup_full/     # All features passing firing rate filter (large, for post-hoc)
│   ├── layer_00.parquet
│   └── ...
└── differential_summary.json
```

The default downstream pipeline (B2, B5, C1) reads from `per_subgroup/`. The `per_subgroup_full/` directory is for methodology audits and sensitivity analyses.

**Per-category parquets** (secondary — same analysis but grouping by category instead of subgroup):

```
{run}/B_differential/per_category/layer_00.parquet
...
```

Same schema but with `subgroup` column absent (or set to the category name). These are for comparison with subgroup-level results: if a feature is significant at the category level but NOT at any subgroup level, it might be a category-general feature. If it's significant for one subgroup but not the category, it's subgroup-specific. This contrast informs the fragmentation narrative.

**Summary JSON:**

```
{run}/B_differential/differential_summary.json
```

```json
{
  "layers_analyzed": [0, 1, 2, ..., 31],
  "fdr_threshold": 0.05,
  "min_firing_rate": 0.05,
  "min_nonzero_count": 10,
  "test": "mann_whitney_u",
  "context_condition_filter": "ambig",
  "per_subgroup": {
    "gay": {
      "category": "so",
      "n_stereo": 312,
      "n_non_stereo": 508,
      "n_significant_by_layer": {
        "0": 17, "1": 23, "2": 45, ..., "14": 287, ..., "31": 112
      },
      "peak_layer": 14,
      "total_significant_pro_bias": 1842,
      "total_significant_anti_bias": 1356,
      "total_significant_all_layers": 3198
    },
    "bisexual": {
      "category": "so",
      "n_stereo": 289,
      "n_non_stereo": 531,
      ...
    },
    ...
  },
  "per_category": {
    "so": {
      "n_significant_by_layer": {...},
      "peak_layer": 14
    },
    ...
  }
}
```

### Performance Optimization Notes

The naive approach (loop over 131K features × 40 subgroups × 32 layers, calling mannwhitneyu each time) is ~160M calls at ~0.1ms each = ~4.5 hours. This is acceptable but can be optimized:

**Optimization 1:** For each (category, layer), build a sparse item×feature matrix once. Then for each subgroup within the category, index into the matrix using the subgroup's item indices. This avoids re-scanning the parquet per subgroup.

```python
# Load layer parquet for this category
cat_df = layer_df[layer_df["category"] == cat]

# Pivot to sparse matrix: rows = items, columns = features
from scipy.sparse import csr_matrix
# Build via COO format from (item_idx, feature_idx, activation_value) triples
...
```

**Optimization 2:** Skip features with zero variance across the combined group (both groups all zeros). This eliminates ~80% of features immediately since most SAE features don't fire on any item in a given category.

```python
# Before running any tests: identify features that fire on at least one item
active_features = cat_df["feature_idx"].unique()  # typically ~5K-15K out of 131K
# Only test these features — skip the ~120K that never fire
```

This reduces the number of Mann-Whitney tests from 131K to ~10K per (subgroup, layer), cutting total time to ~30 minutes.

**Optimization 3:** Use scipy's vectorized rank-sum test or a batch implementation rather than calling mannwhitneyu in a loop.

### Resume Safety

Per-layer: check if the output parquet exists before processing. If it exists, skip the layer. Layers are independent.

---

## B2: Feature Ranking Per Subgroup

### Purpose

Collect significant features for each subgroup across ALL 32 layers, rank them by effect size, determine the optimal injection layer per subgroup, and compute cross-subgroup feature overlap.

### Invocation

```bash
python scripts/B2_rank_features.py --run_dir runs/llama-3.1-8b_2026-04-15/
```

No additional arguments needed — reads from B1 output and config.json.

### Input

- B1 parquets: `{run}/B_differential/per_subgroup/layer_*.parquet`

### Step 1: Collect Significant Features Across All Layers

For each subgroup, concatenate all significant features from all 32 layer parquets:

```python
all_sig = []
for layer in range(32):
    df = pd.read_parquet(f"per_subgroup/layer_{layer:02d}.parquet")
    sig = df[df["is_significant"]].copy()
    all_sig.append(sig)

combined = pd.concat(all_sig, ignore_index=True)
```

### Step 2: Deduplicate

A given feature_idx can appear at multiple layers. This is expected — SAE features at adjacent layers are different features (different SAEs), but the same feature_idx at layer 14 and layer 15 might capture similar phenomena.

**Deduplication rule:** Within a subgroup, if the same (feature_idx, layer) appears multiple times (shouldn't happen from B1, but defensive), keep the entry with the largest |cohen's_d|.

**Cross-layer deduplication:** We do NOT deduplicate the same feature_idx across different layers. Feature 45021 at layer 14 and feature 45021 at layer 16 are **different features** from different SAEs. They just happen to share an index number. Treat them as independent.

### Step 3: Split and Rank

For each subgroup:

```python
sub_df = combined[combined["subgroup"] == subgroup]

# Pro-bias: features that fire more for stereotyped responses
pro_bias = sub_df[sub_df["cohens_d"] > 0].sort_values("cohens_d", ascending=False)

# Anti-bias: features that fire more for non-stereotyped responses  
anti_bias = sub_df[sub_df["cohens_d"] < 0].copy()
anti_bias["abs_d"] = anti_bias["cohens_d"].abs()
anti_bias = anti_bias.sort_values("abs_d", ascending=False)
```

Each entry in the ranked list contains:

```python
{
    "feature_idx": 45021,
    "layer": 14,
    "cohens_d": 1.23,
    "p_fdr": 0.00012,
    "firing_rate_stereo": 0.34,
    "firing_rate_non_stereo": 0.11,
    "mean_activation_stereo": 2.14,
    "mean_activation_non_stereo": 0.87,
    "direction": "pro_bias"
}
```

### Step 4: Determine Optimal Injection Layer Per Subgroup

For steering, we need to inject the steering vector at a specific layer. The optimal layer for subgroup S is determined by where its strongest features concentrate:

```python
def optimal_injection_layer(ranked_pro_bias: list[dict], top_n: int = 10) -> dict:
    """Determine injection layer from top-N pro-bias features."""
    top = ranked_pro_bias[:top_n]
    if not top:
        return {"injection_layer": None, "layer_distribution": {}}
    
    layer_counts = {}
    for f in top:
        layer_counts[f["layer"]] = layer_counts.get(f["layer"], 0) + 1
    
    # Mode layer; ties broken by preferring deeper layers
    max_count = max(layer_counts.values())
    candidates = [l for l, c in layer_counts.items() if c == max_count]
    injection_layer = max(candidates)  # prefer deeper layer on tie
    
    return {
        "injection_layer": injection_layer,
        "layer_distribution_top10": layer_counts,
        "layer_distribution_top20": _count_layers(ranked_pro_bias[:20]),
        "n_features_total": len(ranked_pro_bias),
    }
```

**Why prefer deeper layers on tie:** Later layers are closer to the output and have more influence on the model's next-token prediction. Steering at a deeper layer produces a more direct effect on the answer logits. Earlier layers have their effects diluted by subsequent computation.

### Step 5: Cross-Subgroup Feature Overlap

For every pair of subgroups WITHIN the same category, compute how much their top feature sets overlap:

```python
def feature_overlap(ranked_A: list[dict], ranked_B: list[dict], k: int = 20) -> dict:
    """Compute pairwise feature overlap metrics."""
    # Feature identity is the (feature_idx, layer) tuple
    set_A = {(f["feature_idx"], f["layer"]) for f in ranked_A[:k]}
    set_B = {(f["feature_idx"], f["layer"]) for f in ranked_B[:k]}
    
    intersection = set_A & set_B
    union = set_A | set_B
    
    jaccard = len(intersection) / len(union) if union else 0.0
    
    # Directed overlap: what fraction of A's features appear in B's set
    directed_A_to_B = len(intersection) / k if k > 0 else 0.0
    directed_B_to_A = len(intersection) / k if k > 0 else 0.0
    
    return {
        "jaccard": jaccard,
        "directed_A_to_B": directed_A_to_B,
        "directed_B_to_A": directed_B_to_A,
        "n_shared": len(intersection),
        "shared_features": [{"feature_idx": f[0], "layer": f[1]} for f in intersection],
    }
```

**Connection to the fragmentation hypothesis:** If subgroups within a category (e.g., gay vs. bisexual) have fragmented representations, their top feature sets should have low Jaccard overlap (<0.1). If they share a common bias mechanism, overlap should be high (>0.3). This is the feature-level test of fragmentation — complementary to the geometric test (cosine of DIM directions) in B3.

### Step 6: Track Multi-Group Item Contamination

For each subgroup, record what fraction of its items are shared with other subgroups:

```python
for sub_A in subgroups_in_category:
    items_A = set(item_idxs_targeting(sub_A))
    for sub_B in subgroups_in_category:
        if sub_A == sub_B:
            continue
        items_B = set(item_idxs_targeting(sub_B))
        overlap_frac = len(items_A & items_B) / len(items_A) if items_A else 0
        # Record: "{overlap_frac:.1%} of {sub_A}'s items also target {sub_B}"
```

If >20% of items are shared between two subgroups, their feature overlap and cosine similarity (B3) may be inflated by shared data. This is a caveat to report, not a reason to exclude items.

### Output

```
{run}/B_feature_ranking/
├── ranked_features_by_subgroup.json
├── injection_layers.json
├── feature_overlap.json
├── item_overlap_warnings.json
└── figures/
    ├── fig_feature_overlap_so.png/.pdf
    ├── fig_feature_overlap_race.png/.pdf
    ├── ...  (one per category with ≥2 subgroups)
    ├── fig_feature_layer_distribution.png/.pdf
    └── fig_ranked_effect_sizes_so.png/.pdf
    └── fig_ranked_effect_sizes_race.png/.pdf
    └── ...
```

**ranked_features_by_subgroup.json:**
```json
{
  "so": {
    "gay": {
      "pro_bias": [
        {"feature_idx": 45021, "layer": 14, "cohens_d": 1.23, "p_fdr": 0.00012, ...},
        {"feature_idx": 88012, "layer": 16, "cohens_d": 0.98, ...},
        ...
      ],
      "anti_bias": [
        {"feature_idx": 72301, "layer": 14, "cohens_d": -0.87, ...},
        ...
      ]
    },
    "bisexual": { ... },
    ...
  },
  ...
}
```

**injection_layers.json:**
```json
{
  "so/gay": {"injection_layer": 14, "layer_distribution_top10": {"14": 6, "16": 3, "12": 1}, ...},
  "so/bisexual": {"injection_layer": 16, "layer_distribution_top10": {"16": 5, "14": 3, "18": 2}, ...},
  ...
}
```

**feature_overlap.json:**
```json
{
  "so": {
    "subgroups": ["bisexual", "gay", "lesbian", "pansexual"],
    "pairwise": {
      "gay_bisexual": {"jaccard": 0.05, "directed_A_to_B": 0.10, "n_shared": 2, ...},
      "gay_lesbian": {"jaccard": 0.35, "directed_A_to_B": 0.40, "n_shared": 8, ...},
      ...
    },
    "top_k": 20,
    "direction": "pro_bias"
  },
  ...
}
```

### Figures

**fig_feature_overlap_{category}.png:** One heatmap per category. Rows and columns = subgroups. Cell values = Jaccard overlap of top-20 pro-bias features. Diagonal = number of features (not Jaccard). Annotate all cells. Use YlOrRd sequential colormap. Title: "Top-20 pro-bias feature overlap — {category}".

**fig_feature_layer_distribution.png:** Grid of subplots, one per subgroup (across all categories). Within each subplot, histogram of which layers contribute to the subgroup's top-20 pro-bias features. X = layer, Y = count. Color by category. This reveals whether a subgroup's bias features are concentrated in one layer or distributed.

**fig_ranked_effect_sizes_{category}.png:** One plot per category. Overlaid curves showing |cohen's_d| vs. rank for each subgroup within the category. X = feature rank (1, 2, 3, ..., 30), Y = |cohen's_d|. One line per subgroup with distinct color + marker. Shows whether subgroups have one dominant feature (steep initial drop) or many comparable features (gradual decline). Legend with subgroup names.

---

## B3: Subgroup Direction Geometry

### Purpose

Compute DIM (Difference-in-Means) directions for each subgroup from raw activation differences — completely independent of the SAE. This provides a second geometric view of subgroup representations that can be cross-validated against the SAE-based analysis from B1/B2.

### Invocation

```bash
python scripts/B3_geometry.py --run_dir runs/llama-3.1-8b_2026-04-15/
```

### Input

- Activations: `{run}/A_extraction/activations/{category}/item_*.npz`
- Stimuli: `{run}/A_extraction/stimuli/{category}.json`

Does NOT use SAE encodings. Independent of A3/B1/B2.

### Two Types of Directions

We compute two distinct directions per subgroup per layer. They capture different phenomena:

**Bias direction:** "Which direction in activation space corresponds to the model giving a stereotyped response about subgroup S?"

```
bias_direction(S, layer) = mean(h[layer] for items targeting S with stereotyped response)
                         - mean(h[layer] for items targeting S with non-stereotyped response)
```

This is the direction that distinguishes biased behavior from unbiased behavior for items about subgroup S. Steering AGAINST this direction should reduce bias.

**Identity direction:** "Which direction in activation space distinguishes content about subgroup S from content about other subgroups in the same category?"

```
identity_direction(S, layer) = mean(h[layer] for items targeting S)
                             - mean(h[layer] for items NOT targeting S in same category)
```

This captures how the model represents the identity itself, independent of whether the response is biased. The identity direction may or may not align with the bias direction — and whether they align is itself a finding.

**Why compute both:**
- The bias direction is what C1 steers against. It's directly actionable.
- The identity direction is what the cosine geometry in C2's universal scatter should use. The question "does subgroup similarity predict cross-subgroup steering effects?" is about identity geometry, not bias geometry.
- If bias directions and identity directions are highly aligned (cosine ≈ 1), then the model's bias is encoded along the same axis as identity — bias and identity are entangled. If they're orthogonal, bias can be separated from identity.

### Per-Layer Processing

**Step 1: Load and de-normalize hidden states.**

```python
data = np.load(item_npz)
hs_normed = data["hidden_states"]  # (n_layers, hidden_dim), float16
raw_norms = data["hidden_states_raw_norms"]  # (n_layers,), float32
hs_raw = hs_normed.astype(np.float32) * raw_norms[:, None]  # reconstruct raw
```

**Methodological choice — compute directions on raw activations:**

DIM directions computed on unit-normalized activations capture angular differences only. DIM directions computed on raw activations capture both angular and magnitude differences. Since the residual stream norm grows across layers, raw DIM directions at later layers will be larger in magnitude — but this is controlled by unit-normalizing the final direction.

We compute on raw activations because the magnitude of the mean difference is informative: if the stereotyped group has a much larger mean norm at some layer, that itself is a finding about how bias is encoded. The final direction is unit-normalized for cosine comparisons.

**Step 2: Group items by subgroup.**

```python
# For bias direction: partition items targeting S by model response
for subgroup in subgroups_in_category:
    targeting_S = [item for item in items if subgroup in item["stereotyped_groups"]
                   and item["context_condition"] == "ambig"]
    
    stereo_items = [it for it in targeting_S 
                    if meta[it["item_idx"]]["model_answer_role"] == "stereotyped_target"]
    non_stereo_items = [it for it in targeting_S 
                        if meta[it["item_idx"]]["model_answer_role"] != "stereotyped_target"]
```

**Step 3: Compute directions at each layer.**

```python
for layer in range(n_layers):
    # Bias direction
    mean_stereo = np.mean([hs_raw[layer] for item in stereo_items], axis=0)
    mean_non_stereo = np.mean([hs_raw[layer] for item in non_stereo_items], axis=0)
    bias_dir = mean_stereo - mean_non_stereo
    bias_dir_normed = bias_dir / max(np.linalg.norm(bias_dir), 1e-8)
    
    # Identity direction
    mean_S = np.mean([hs_raw[layer] for item in all_S_items], axis=0)
    mean_not_S = np.mean([hs_raw[layer] for item in all_non_S_items], axis=0)
    id_dir = mean_S - mean_not_S
    id_dir_normed = id_dir / max(np.linalg.norm(id_dir), 1e-8)
```

Result per subgroup: two arrays of shape `(n_layers, hidden_dim)`.

**Minimum item count:** If either group (stereo or non-stereo) has fewer than 10 items, skip that subgroup for bias direction computation. If total items targeting S is fewer than 10, skip identity direction. Log the skip.

### Pairwise Cosine Matrices

For each category at each layer, compute cosine similarity between all subgroup pairs:

```python
for cat in categories:
    subgroups = subgroups_in(cat)
    for layer in range(n_layers):
        n = len(subgroups)
        cos_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                cos_matrix[i, j] = np.dot(
                    identity_directions[subgroups[i]][layer],
                    identity_directions[subgroups[j]][layer]
                )
                # Both are unit-normalized, so dot product = cosine
```

Store the full matrix at every layer. Also identify:
- **Peak differentiation layer:** The layer where the variance of the off-diagonal cosines is maximized (most spread between aligned and anti-aligned pairs).
- **Stable range:** Layers where the cosine structure is qualitatively stable (pairwise cosines don't change sign).

### Bias-Identity Alignment

For each subgroup at each layer, compute the cosine between its bias direction and its identity direction:

```python
alignment(S, layer) = dot(bias_direction_normed[S][layer], identity_direction_normed[S][layer])
```

If alignment ≈ 1: bias is encoded along the identity axis — knowing which identity is discussed predicts whether the response is biased.
If alignment ≈ 0: bias is orthogonal to identity — the model can represent identity without triggering bias.

Report per-subgroup alignment across layers as a curve. This is a finding about entanglement that connects to the interpretability discussion.

### Output

```
{run}/B_geometry/
├── subgroup_directions.npz
│     Keys: bias_direction_so_gay, identity_direction_so_gay, 
│           bias_direction_so_bisexual, identity_direction_so_bisexual, ...
│     Each: shape (n_layers, hidden_dim), float32, unit-normalized per layer
│
├── subgroup_directions_summary.json
│     Per-subgroup: n_items, n_stereo, n_non_stereo, multi-group overlap stats
│     Per-category: peak_differentiation_layer, stable_range
│
├── cosine_matrices/
│   ├── so_identity.json       # {layer_0: {subgroups: [...], matrix: [[...]]}, ...}
│   ├── so_bias.json
│   ├── race_identity.json
│   ├── race_bias.json
│   └── ...
│
├── bias_identity_alignment.json
│     {subgroup: {layer_0: cosine, layer_1: cosine, ...}}
│
└── figures/
    ├── fig_cosine_heatmap_so_identity_layer14.png/.pdf
    ├── fig_cosine_heatmap_so_bias_layer14.png/.pdf
    ├── ...  (at peak layer per category)
    ├── fig_cosine_by_layer_so.png/.pdf          # Pairwise cosines across layers (line plot)
    ├── fig_cosine_by_layer_race.png/.pdf
    ├── fig_bias_identity_alignment.png/.pdf     # Per-subgroup alignment across layers
    └── ...
```

### Figures

**fig_cosine_heatmap_{category}_{direction_type}_layer{NN}.png:** Heatmap of pairwise cosines at the peak differentiation layer. Rows and columns = subgroups. RdBu_r diverging colormap centered at 0. Annotate cells with values. Blue = aligned, red = anti-aligned. This is one of the key paper figures.

**fig_cosine_by_layer_{category}.png:** Line plot showing how each subgroup pair's cosine evolves across layers. X = layer, Y = cosine similarity. One line per subgroup pair. Distinct colors + markers. Shows where the pairwise structure emerges and stabilizes.

**fig_bias_identity_alignment.png:** One panel per category. Within each panel, one line per subgroup showing bias-identity alignment cosine across layers. If all subgroups converge to alignment ≈ 1 at the same layer, bias and identity are entangled at that depth.

---

## B4: Probe Training + Controls

### Purpose

Train linear probes on saved hidden states to test what's linearly decodable from the model's representations at each layer. Three controls establish whether the probes find genuine identity-specific signal vs. surface artifacts.

### Invocation

```bash
python scripts/B4_probes.py --run_dir runs/llama-3.1-8b_2026-04-15/

# Subset of layers for speed
python scripts/B4_probes.py --run_dir runs/llama-3.1-8b_2026-04-15/ --layer_stride 2
```

### Input

- Activations: `{run}/A_extraction/activations/{category}/item_*.npz`
- Stimuli: `{run}/A_extraction/stimuli/{category}.json`

No SAE or model needed.

### Loading and Preparing Features

```python
# For each category, load all hidden states and de-normalize
for cat in categories:
    items = load_stimuli(cat)
    hidden_states = {}  # {item_idx: (n_layers, hidden_dim) raw float32}
    meta = {}
    
    for npz_path in activations_dir(cat).glob("item_*.npz"):
        data = np.load(npz_path, allow_pickle=True)
        hs_normed = data["hidden_states"].astype(np.float32)
        raw_norms = data["hidden_states_raw_norms"]
        hs_raw = hs_normed * raw_norms[:, None]
        
        m = json.loads(data["metadata_json"].item())
        hidden_states[m["item_idx"]] = hs_raw
        meta[m["item_idx"]] = m
```

**De-normalize before probing.** Probes should see the same activation scale the model uses internally. If we probed on normalized activations, we'd lose magnitude information that might carry signal.

### Probe Architecture

PCA dimensionality reduction followed by logistic regression with 5-fold stratified cross-validation:

```python
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

def train_probe(X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> dict:
    """
    X: (n_items, hidden_dim)
    y: labels (string or int)
    Returns: {mean_accuracy, std_accuracy, per_fold_accuracies}
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    # PCA reduction — hidden_dim=4096 >> n_items, direct LR would overfit
    n_components = min(50, X.shape[0] - 1, X.shape[1])
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    accs = []
    
    for train_idx, test_idx in skf.split(X, y_enc):
        # Fit PCA on training fold only (no data leakage)
        pca = PCA(n_components=n_components)
        X_train = pca.fit_transform(X[train_idx])
        X_test = pca.transform(X[test_idx])
        
        clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                                 multi_class="multinomial")
        clf.fit(X_train, y_enc[train_idx])
        accs.append(float(clf.score(X_test, y_enc[test_idx])))
    
    return {
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "per_fold": accs,
        "n_classes": len(le.classes_),
        "classes": list(le.classes_),
        "n_items": len(y),
        "n_components": n_components,
    }
```

**Methodological choice — PCA-50 before probing:**

Standard in the probing literature. The hidden dimension (4096) is much larger than the number of items per category (~800), creating severe overfitting risk for direct logistic regression. PCA to 50 components captures >95% of variance while reducing dimensionality by 80x. The number 50 is a convention; report it and note sensitivity could be tested at 25 and 100.

**PCA fit on training fold only:** PCA is fit within each CV fold on training data only, then applied to the test fold. This prevents data leakage through the dimensionality reduction step.

### Main Probe: Subgroup Classification

At each layer, predict which subgroup an item targets:

```python
for cat in categories:
    for layer in range(0, n_layers, layer_stride):
        X = np.stack([hidden_states[idx][layer] for idx in item_indices])
        y = np.array([stereotyped_groups_primary(idx) for idx in item_indices])
        # stereotyped_groups_primary: first group in stereotyped_groups list
        # For multi-group items, use the first group (consistent assignment for probing)
        
        result = train_probe(X, y)
```

**Note on multi-group items for probing:** For probe training, each item needs exactly one label. Use `stereotyped_groups[0]` as the probe label. This is a different context than B1 (where items contribute to all groups for differential analysis). Probing requires mutually exclusive labels; differential analysis does not.

Run at every layer (or every `layer_stride` layers). Record accuracy per layer to produce the accuracy-vs-layer curve.

### Control A: Permutation Baseline

For each (category, layer), repeat the probe with randomly permuted labels:

```python
def permutation_baseline(X, y, n_permutations=10, n_folds=5):
    rng = np.random.default_rng(42)
    perm_accs = []
    for trial in range(n_permutations):
        y_perm = rng.permutation(y)
        result = train_probe(X, y_perm, n_folds=n_folds)
        perm_accs.append(result["mean_accuracy"])
    return {
        "mean": float(np.mean(perm_accs)),
        "std": float(np.std(perm_accs)),
        "all_trials": perm_accs,
    }
```

**Selectivity:**
```
selectivity(category, layer) = real_accuracy - permutation_mean
```

Interpretation:
- Selectivity ≈ 0 at early layers: probe exploits surface features (same for real and shuffled labels)
- Selectivity > 0 at mid/late layers: genuine identity-specific signal emerges
- Expected permutation accuracy for k classes: ~1/k (random chance)

### Control B: Structural Control Tasks

Two additional probes at the same layers with the same architecture:

**B1 — Context condition classification:** Predict `"ambig"` vs `"disambig"`.

```python
y_context = np.array([meta[idx]["context_condition"] for idx in item_indices])
result_context = train_probe(X, y_context)
```

This is a structural property encoded in sentence length and evidential clause presence.

**B2 — Stereotyped answer position:** Predict which letter (A, B, or C) is the stereotyped option.

```python
y_position = np.array([stimuli[idx]["stereotyped_option"] for idx in item_indices])
result_position = train_probe(X, y_position)
```

This varies across items but may correlate with template structure.

**Identity-attributable excess:**
```
excess(category, layer) = identity_probe_acc(layer) - max(context_probe_acc(layer), position_probe_acc(layer))
```

If excess > 0, the identity probe captures information beyond surface structure.

### Control D: Cross-Category Generalization

Test whether a "bias detector" probe generalizes across categories.

**Binary probe:** Predict `is_stereotyped_response` (True/False) rather than subgroup identity. This is category-agnostic — a stereotyped response is a stereotyped response regardless of which category.

```python
for train_cat in categories:
    for test_cat in categories:
        # Features at the best layer for train_cat's identity probe
        best_layer = best_layer_for(train_cat)
        
        X_train = hidden_states_at_layer(train_cat, best_layer)
        y_train = is_stereotyped_labels(train_cat)
        
        X_test = hidden_states_at_layer(test_cat, best_layer)
        y_test = is_stereotyped_labels(test_cat)
        
        # Fit PCA on training category, transform test category
        pca = PCA(n_components=50)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        clf = LogisticRegression(C=1.0, max_iter=1000)
        clf.fit(X_train_pca, y_train)
        
        generalization_matrix[train_cat][test_cat] = clf.score(X_test_pca, y_test)
```

**Prediction from fragmentation:** Cross-category probes should perform near chance (0.5 for binary). The stereotyping representation is NOT shared across categories.

**Within-category cross-subgroup generalization** (finer-grained):

```python
# Within category C, train on items targeting subgroup A, test on items targeting subgroup B
for cat in categories:
    for train_sub in subgroups(cat):
        for test_sub in subgroups(cat):
            X_train = hidden_states for items targeting train_sub
            y_train = is_stereotyped for those items
            X_test = hidden_states for items targeting test_sub
            y_test = is_stereotyped for those items
            
            # Same PCA-on-train → LR pipeline
            within_cat_matrix[cat][train_sub][test_sub] = accuracy
```

**Prediction from fragmentation:** Anti-correlated subgroup pairs (gay↔bisexual, where DIM cosine is negative) should show poor cross-subgroup generalization. Aligned pairs (gay↔lesbian, where cosine is positive) should generalize. This directly connects probe generalization to the cosine geometry from B3.

### Output

```
{run}/B_probes/
├── identity_probes.json
│     Per (category, layer): accuracy, std, n_classes, classes
│
├── permutation_baselines.json
│     Per (category, layer): permutation mean, std, selectivity
│
├── structural_controls.json
│     Per (category, layer): context_probe_acc, position_probe_acc, excess
│
├── cross_category_generalization.json
│     Matrix: {train_cat: {test_cat: accuracy}}
│
├── within_category_generalization.json
│     Per category: {train_sub: {test_sub: accuracy}}
│
└── figures/
    ├── fig_probe_selectivity.png/.pdf
    ├── fig_probe_structural_comparison.png/.pdf
    ├── fig_probe_generalization_matrix.png/.pdf
    └── fig_within_category_generalization.png/.pdf
```

### Figures

**fig_probe_selectivity.png:** One panel per category. X = layer, Y = accuracy. Blue solid line = real probe accuracy. Gray dashed line with ±1σ band = permutation baseline. Shaded red region between them = selectivity. Annotate peak selectivity layer and value.

**fig_probe_structural_comparison.png:** One panel per category. X = layer, Y = accuracy. Blue solid = identity probe. Orange dashed = context condition probe. Green dotted = answer position probe. The gap between blue and max(orange, green) at each layer is the identity-attributable excess.

**fig_probe_generalization_matrix.png:** Single heatmap. Rows = training category, columns = test category. Color = binary probe accuracy (Blues colormap, vmin=0.4, vmax=1.0). Annotate cells. Diagonal should be high, off-diagonal near 0.5.

**fig_within_category_generalization.png:** One heatmap per category with ≥3 subgroups. Rows = training subgroup, columns = test subgroup. Color = accuracy. Annotate cells. Should show that aligned pairs generalize and anti-correlated pairs fail — directly connecting to B3's cosine matrices.

---

## B5: Feature Interpretability (Item-Level)

### Purpose

For the top-ranked features per subgroup (from B2), characterize WHAT those features respond to by examining which items activate them most, whether they're subgroup-specific or category-general, and whether they pass basic sanity checks for being genuine bias features rather than artifacts.

### Invocation

```bash
python scripts/B5_feature_interpretability.py --run_dir runs/llama-3.1-8b_2026-04-15/

# Limit to specific categories
python scripts/B5_feature_interpretability.py --run_dir runs/llama-3.1-8b_2026-04-15/ --categories so,race
```

### Input

- SAE encodings: `{run}/A_extraction/sae_encoding/layer_*.parquet`
- Ranked features: `{run}/B_feature_ranking/ranked_features_by_subgroup.json`
- Stimuli: `{run}/A_extraction/stimuli/{category}.json`
- Behavioral metadata: from A2 .npz files (loaded via B1's metadata table or directly)

### Which Features to Characterize

For each subgroup, characterize the **top-20 pro-bias features** and **top-10 anti-bias features** from the B2 ranked lists. This covers the features most likely to end up in steering vectors (C1 typically selects top-3 to top-13).

### Analysis A: Activation Distribution Per Feature

For each feature F at its layer L:

```python
# Load the layer parquet
layer_df = pd.read_parquet(f"layer_{F_layer:02d}.parquet")

# Get activation values for this feature across all items in the category
feat_activations = layer_df[
    (layer_df["feature_idx"] == F_idx) & (layer_df["category"] == cat)
].set_index("item_idx")["activation_value"]

# Items not in feat_activations have activation = 0
all_items = stimuli_for_cat
activations = [feat_activations.get(it["item_idx"], 0.0) for it in all_items]
```

Compute:
```python
{
    "mean_all": np.mean(activations),
    "std_all": np.std(activations),
    "median_all": np.median(activations),
    "fraction_nonzero": np.mean(np.array(activations) > 0),
    "mean_stereo": np.mean([a for a, it in zip(activations, items) if is_stereo(it)]),
    "mean_non_stereo": np.mean([a for a, it in zip(activations, items) if not is_stereo(it)]),
    "mean_ambig": np.mean([a for a, it in zip(activations, items) if is_ambig(it)]),
    "mean_disambig": np.mean([a for a, it in zip(activations, items) if is_disambig(it)]),
}
```

### Analysis B: Top-Activating Items

Sort items by activation value descending. Record top-20:

```python
ranked_items = sorted(zip(activations, items), key=lambda x: -x[0])[:20]

top_items = []
for activation, item in ranked_items:
    top_items.append({
        "item_idx": item["item_idx"],
        "activation": float(activation),
        "prompt_preview": item["prompt"][:150],
        "model_answer_role": meta[item["item_idx"]]["model_answer_role"],
        "stereotyped_groups": item["stereotyped_groups"],
        "n_target_groups": item["n_target_groups"],
        "context_condition": item["context_condition"],
    })
```

This shows WHICH BBQ scenarios trigger the feature most strongly. A reviewer can read the prompt previews and assess face validity: does the feature fire on content related to the subgroup's stereotypes?

### Analysis C: Stereotype Specificity Score

For each feature selected for subgroup S, measure whether it fires preferentially on items targeting S vs. items targeting other subgroups in the same category:

```python
def specificity(activations_by_subgroup: dict[str, list[float]], target_subgroup: str) -> float:
    """
    activations_by_subgroup: {subgroup_name: [activation values for items targeting that subgroup]}
    target_subgroup: the subgroup this feature was selected for
    """
    target_mean = np.mean(activations_by_subgroup[target_subgroup])
    all_means = [np.mean(vs) for vs in activations_by_subgroup.values()]
    category_mean = np.mean(all_means)  # unweighted mean across subgroups
    
    if category_mean < 1e-8:
        return 0.0  # feature barely fires on any subgroup
    
    return target_mean / category_mean
```

Interpretation:
- Specificity >> 1: Feature is subgroup-specific (fires on bisexual items, not gay items). Strong evidence for fragmentation at the feature level.
- Specificity ≈ 1: Feature is category-general (fires on all SO items equally). May capture shared bias mechanisms.
- Specificity < 1: Feature fires more on OTHER subgroups. Unexpected — flag for investigation.

### Analysis D: Cross-Subgroup Activation Matrix

For all top features selected by ANY subgroup within a category, compute mean activation per target subgroup:

```python
# Rows = features (labeled by which subgroup selected them + feature_idx)
# Columns = target subgroups
# Cells = mean activation on items targeting that subgroup

for cat in categories:
    all_features = []  # (source_sub, feature_idx, layer)
    for sub in subgroups(cat):
        for feat in ranked_pro_bias[cat][sub][:top_n]:
            all_features.append((sub, feat["feature_idx"], feat["layer"]))
    
    # For each feature, compute per-subgroup activation
    matrix = np.zeros((len(all_features), len(subgroups(cat))))
    for i, (src_sub, fidx, layer) in enumerate(all_features):
        layer_df = pd.read_parquet(f"layer_{layer:02d}.parquet")
        feat_acts = layer_df[
            (layer_df["feature_idx"] == fidx) & (layer_df["category"] == cat)
        ].set_index("item_idx")["activation_value"]
        
        for j, tgt_sub in enumerate(subgroups(cat)):
            tgt_items = [it["item_idx"] for it in items if tgt_sub in it["stereotyped_groups"]]
            vals = [feat_acts.get(idx, 0.0) for idx in tgt_items]
            matrix[i, j] = np.mean(vals) if vals else 0.0
```

**Block-diagonal test:** If fragmentation holds at the feature level, this matrix should show block-diagonal structure — features selected for subgroup A activate strongly on A-targeted items and weakly on B-targeted items. Cluster the rows (features) by their activation profiles and see if the clustering recovers the subgroup-of-origin labels.

### Analysis E: Cross-Category Baseline (Artifact Detection)

For each feature, compute its mean activation on items from a COMPLETELY DIFFERENT category:

```python
def cross_category_activation(feature_idx, feature_layer, source_cat, all_categories):
    """Mean activation of this feature on items from other categories."""
    other_activations = []
    layer_df = pd.read_parquet(f"layer_{feature_layer:02d}.parquet")
    
    for other_cat in all_categories:
        if other_cat == source_cat:
            continue
        other_df = layer_df[
            (layer_df["feature_idx"] == feature_idx) & (layer_df["category"] == other_cat)
        ]
        # Mean including zeros for items not in the parquet
        n_items_other = count_items_in(other_cat)
        total = other_df["activation_value"].sum()
        other_activations.append(total / max(n_items_other, 1))
    
    return np.mean(other_activations) if other_activations else 0.0
```

```python
within_category_mean = mean_activation(feature, items_in_source_category)
cross_category_mean = cross_category_activation(feature, ...)

category_specificity_ratio = within_category_mean / max(cross_category_mean, 1e-8)
```

Interpretation:
- Ratio >> 1: Feature is category-specific. Good — it's not a general template feature.
- Ratio ≈ 1: Feature fires equally on unrelated categories. This is a red flag — likely a prompt structure feature (responds to "Question:", "Answer:", common words). Should NOT be used for steering.
- Ratio < 0.5: Feature fires MORE on other categories. Almost certainly an artifact.

**Flag features with category_specificity_ratio < 2.0** in the output. If these appear in a subgroup's top-ranked list, they should be demoted or excluded from steering vectors in C1.

### Analysis F: Feature Co-occurrence

For subgroups where the optimal k* > 1 (multiple features in the steering vector), compute pairwise correlation of feature activations across items:

```python
# For features F1, F2 both selected for subgroup S:
acts_F1 = [activation of F1 for each item targeting S]
acts_F2 = [activation of F2 for each item targeting S]
correlation = np.corrcoef(acts_F1, acts_F2)[0, 1]
```

High correlation (>0.7): Features capture overlapping information. Adding the second feature to the steering vector may not provide additional benefit (redundancy).
Low correlation (<0.3): Features capture distinct aspects. Both contribute unique signal.

Report as a small correlation matrix per subgroup (only for the top-k* features that end up in steering vectors).

### Output

```
{run}/B_feature_interpretability/
├── feature_reports.json
│     Per feature: activation stats, top-20 items, specificity score,
│     category_specificity_ratio, co-occurrence correlations
│
├── cross_subgroup_activation_matrices.json
│     Per category: matrix (features × subgroups), feature labels, subgroup labels
│
├── specificity_scores.json
│     Per subgroup: {feature_idx: {specificity, category_specificity_ratio, ...}}
│
├── artifact_flags.json
│     Features with category_specificity_ratio < 2.0
│
└── figures/
    ├── fig_cross_subgroup_activation_so.png/.pdf
    ├── fig_cross_subgroup_activation_race.png/.pdf
    ├── ...  (one heatmap per category)
    ├── fig_specificity_distribution.png/.pdf
    ├── fig_category_specificity_ratio.png/.pdf
    └── fig_feature_cooccurrence_so.png/.pdf
    └── ...
```

### Figures

**fig_cross_subgroup_activation_{category}.png:** Heatmap per category. Rows = features (labeled as "{source_subgroup}: L{layer}_F{idx}"). Columns = target subgroups. Color = mean activation (YlOrRd sequential colormap). Annotate cells with values. Add hierarchical clustering dendrogram on the row axis. If fragmentation holds, dendrogram should cluster features by their source subgroup.

**fig_specificity_distribution.png:** Single figure. Histogram of specificity scores across ALL characterized features from ALL categories. Vertical dashed line at specificity = 1.0 (category-general baseline). Annotate median, IQR. If most features have specificity > 1.5, the SAE has decomposed bias into subgroup-specific components.

**fig_category_specificity_ratio.png:** Single figure. Histogram of category_specificity_ratio across all characterized features. Vertical dashed line at ratio = 2.0 (threshold for "category-specific enough"). Features below this line are flagged as potential artifacts. Annotate how many features fall below the threshold.

**fig_feature_cooccurrence_{category}.png:** For categories where ≥1 subgroup has k* > 1, show the pairwise correlation matrix of the selected features. Small heatmap (k × k), annotated cells. One panel per subgroup that has multiple features.

---

## Phase B Output Structure (Complete)

```
{run}/
├── B_differential/
│   ├── per_subgroup/
│   │   ├── layer_00.parquet
│   │   ├── layer_01.parquet
│   │   └── ... (32 files)
│   ├── per_subgroup_full/            # All features passing filter (large, for audits)
│   │   ├── layer_00.parquet
│   │   └── ...
│   ├── per_category/
│   │   ├── layer_00.parquet
│   │   └── ...
│   └── differential_summary.json
│
├── B_feature_ranking/
│   ├── ranked_features_by_subgroup.json
│   ├── injection_layers.json
│   ├── feature_overlap.json
│   ├── item_overlap_warnings.json
│   └── figures/
│       ├── fig_feature_overlap_*.png/.pdf
│       ├── fig_feature_layer_distribution.png/.pdf
│       └── fig_ranked_effect_sizes_*.png/.pdf
│
├── B_geometry/
│   ├── subgroup_directions.npz
│   ├── subgroup_directions_summary.json
│   ├── cosine_matrices/
│   │   ├── so_identity.json
│   │   ├── so_bias.json
│   │   └── ...
│   ├── bias_identity_alignment.json
│   └── figures/
│       ├── fig_cosine_heatmap_*.png/.pdf
│       ├── fig_cosine_by_layer_*.png/.pdf
│       └── fig_bias_identity_alignment.png/.pdf
│
├── B_probes/
│   ├── identity_probes.json
│   ├── permutation_baselines.json
│   ├── structural_controls.json
│   ├── cross_category_generalization.json
│   ├── within_category_generalization.json
│   └── figures/
│       ├── fig_probe_selectivity.png/.pdf
│       ├── fig_probe_structural_comparison.png/.pdf
│       ├── fig_probe_generalization_matrix.png/.pdf
│       └── fig_within_category_generalization.png/.pdf
│
└── B_feature_interpretability/
    ├── feature_reports.json
    ├── cross_subgroup_activation_matrices.json
    ├── specificity_scores.json
    ├── artifact_flags.json
    └── figures/
        ├── fig_cross_subgroup_activation_*.png/.pdf
        ├── fig_specificity_distribution.png/.pdf
        ├── fig_category_specificity_ratio.png/.pdf
        └── fig_feature_cooccurrence_*.png/.pdf
```

---

## Assumptions and Methodological Choices Summary

| # | Choice | Rationale | Downstream Impact |
|---|--------|-----------|-------------------|
| 1 | Ambiguous items only for B1 differential analysis | BBQ design: ambig items test stereotype reliance under uncertainty; disambig items test evidence override | Disambig items available for probes (B4) and steering evaluation (C1) |
| 2 | Mann-Whitney U test (not t-test) for differential analysis | SAE activations are zero-inflated + right-skewed; Mann-Whitney is rank-based and robust | Slightly less power than t-test under normality, but much more reliable for our distributions |
| 3 | FDR correction scoped per (subgroup, layer) | Each (subgroup, layer) is an independent scientific question; global correction too conservative | ~131K tests per scope with BH at 0.05; top feature needs raw p < 3.8×10⁻⁷ |
| 4 | Minimum firing rate filter: ≥5% in larger group or ≥10 items | Prevents spuriously significant features from very low-count coincidences | Features firing on <10 items excluded from significance testing |
| 5 | Store both significant-only and full parquets | Significant-only for fast downstream loading; full for post-hoc sensitivity analysis | ~50-250MB per full parquet per layer; significant-only much smaller |
| 6 | Both bias directions AND identity directions in B3 | Bias = stereotyped vs non-stereotyped within subgroup; Identity = subgroup vs other subgroups | Bias direction used for steering (C1); identity direction used for cosine geometry (C2) |
| 7 | Directions computed on raw (de-normalized) activations | Preserves magnitude information; final directions are unit-normalized for cosine | Must reconstruct raw from normed × norms before computing means |
| 8 | PCA-50 before all probes | Prevents overfitting when hidden_dim >> n_items; standard in probing literature | PCA fit within each CV fold to prevent data leakage |
| 9 | Cross-category generalization uses binary (stereotyped yes/no) probe | Category-specific labels don't transfer; binary is category-agnostic | Tests whether "bias detection" generalizes, not subgroup classification |
| 10 | Within-category cross-subgroup generalization probe | Finer-grained fragmentation test at the probe level | Connects directly to B3 cosine matrices — aligned pairs should generalize, anti-correlated should fail |
| 11 | Cross-category baseline activation in B5 (artifact detection) | Catches features that fire on prompt structure rather than identity content | Features with category_specificity_ratio < 2.0 flagged; may be excluded from steering |
| 12 | Feature co-occurrence correlation | Tests whether multi-feature steering vectors contain redundant vs complementary features | Informs whether adding features improves steering or just inflates the vector |
| 13 | Multi-group items: contribute to all groups in B1, use first group for B4 probes | B1 maximizes N per subgroup (no exclusion bias); B4 needs mutually exclusive labels | Item overlap tracked and reported; high overlap inflates feature similarity |