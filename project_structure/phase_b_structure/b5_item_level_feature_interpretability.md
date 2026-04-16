# B5: Feature Interpretability (Item-Level) — Full Implementation Specification

## Purpose

For the top-ranked features per subgroup from B2, characterize WHAT those features respond to: which items activate them most, whether they're subgroup-specific or category-general, and whether they pass sanity checks for being genuine bias features rather than template artifacts. Produces the artifact_flags that C1 uses to exclude template features from steering candidates.

## Invocation

```bash
python scripts/B5_feature_interpretability.py --run_dir runs/llama-3.1-8b_2026-04-15/

# Specific categories
python scripts/B5_feature_interpretability.py --run_dir runs/llama-3.1-8b_2026-04-15/ --categories so,race

# Override top-K for characterization
python scripts/B5_feature_interpretability.py --run_dir runs/llama-3.1-8b_2026-04-15/ --top_k 30
```

Reads `categories`, `n_layers` from config.json.

---

## Input

- SAE encodings: `{run}/A_extraction/sae_encoding/layer_{NN}.parquet`
- Ranked features: `{run}/B_feature_ranking/ranked_features.parquet`
- Metadata: `{run}/A_extraction/metadata.parquet`
- Stimuli: `{run}/A_extraction/stimuli/{category}.json` (for prompt text, question_index, context_condition)

## Dependencies

- `pandas`, `pyarrow`
- `numpy`
- `scipy.cluster.hierarchy` (for cross-subgroup matrix clustering)
- `sklearn.metrics.adjusted_rand_score`
- `matplotlib` (for figures)

---

## Which Features to Characterize

For each (category, subgroup):
- **Top-20 pro-bias features** (highest positive cohens_d)
- **Top-20 anti-bias features** (largest magnitude negative cohens_d)

Configurable via `--top_k`. K=20 is the default. Both directions use the same K.

```python
def load_characterization_features(run_dir: Path, top_k: int = 20) -> pd.DataFrame:
    """
    Load the top-K features per (category, subgroup, direction) from B2.
    
    Returns DataFrame with columns:
        category, subgroup, direction, rank, feature_idx, layer,
        cohens_d, p_value_fdr, firing_rate_stereo, firing_rate_non_stereo,
        mean_activation_stereo, mean_activation_non_stereo, n_stereo, n_non_stereo
    """
    ranked = pd.read_parquet(run_dir / "B_feature_ranking" / "ranked_features.parquet")
    top = ranked[ranked["rank"] <= top_k].copy()
    return top
```

---

## Terminology Clarification

Two distinct specificity metrics, both computed:

**Subgroup specificity** (within category): 
```
subgroup_specificity(F, S) = mean_activation(F on items targeting S) / mean_activation(F on all items in S's category)
```
High (>1.5): feature is subgroup-specific. Supports fragmentation.
Low (<1.0): feature fires more on other subgroups than on S — unexpected.

**Category specificity ratio** (across categories):
```
category_specificity_ratio(F) = mean_activation(F within source category) / mean_activation(F across other categories)
```
High (>2.0): feature is category-specific. Good.
Low (<2.0): feature fires similarly across unrelated categories → template/artifact candidate.

These measure different things. A feature can be category-specific (ratio=5) but not subgroup-specific (specificity=1.0) — it's a category-level bias feature. A template feature is neither (both ~1.0).

---

## Item Loading Helper

All analyses need per-item activations for specific (feature, layer) combinations. Build a fast lookup:

```python
def load_feature_activations(
    run_dir: Path,
    feature_idx: int,
    layer: int,
) -> pd.Series:
    """
    Load activation values for one feature at one layer across ALL items and categories.
    
    Returns:
        Series indexed by item_idx containing activation values.
        Items not in the index have activation = 0.
    """
    parquet_path = run_dir / "A_extraction" / "sae_encoding" / f"layer_{layer:02d}.parquet"
    df = pd.read_parquet(parquet_path)
    feat_df = df[df["feature_idx"] == feature_idx]
    return feat_df.set_index(["category", "item_idx"])["activation_value"]
```

**Optimization note:** Loading the parquet for each feature is slow if we're characterizing many features at the same layer. Better: group features by layer, load the parquet once per layer, then filter:

```python
# Cache: {layer: full_parquet_df}
layer_cache = {}

def get_layer_df(run_dir, layer):
    if layer not in layer_cache:
        layer_cache[layer] = pd.read_parquet(
            run_dir / "A_extraction" / "sae_encoding" / f"layer_{layer:02d}.parquet"
        )
    return layer_cache[layer]

def feature_activations_from_cache(layer_df, feature_idx):
    feat_df = layer_df[layer_df["feature_idx"] == feature_idx]
    return feat_df.set_index(["category", "item_idx"])["activation_value"]
```

Memory: each layer parquet is ~12MB, and we only cache layers that have features in the top-K. Typically 5-15 distinct layers across all characterized features. Manageable.

---

## Analysis A: Activation Distribution Per Feature

For each characterized feature, compute distribution statistics over items in its source category.

**Restriction: ambig items only for the primary stereo/non-stereo comparison.** This matches B1's primary analysis.

```python
def compute_activation_distribution(
    feature_activations: pd.Series,
    category_items: pd.DataFrame,
    category: str,
) -> dict:
    """
    Compute activation distribution statistics for one feature within its source category.
    
    Args:
        feature_activations: Series indexed by (category, item_idx)
        category_items: metadata DataFrame filtered to source category
    
    Returns:
        Dict of statistics.
    """
    # Get activations for all items in this category (0 for items not in the parquet)
    cat_acts = {}
    for _, row in category_items.iterrows():
        cat_acts[row["item_idx"]] = float(feature_activations.get((category, row["item_idx"]), 0.0))
    
    acts_array = np.array(list(cat_acts.values()))
    
    # Overall distribution
    stats = {
        "mean_all": float(np.mean(acts_array)),
        "std_all": float(np.std(acts_array)),
        "median_all": float(np.median(acts_array)),
        "max_activation": float(np.max(acts_array)),
        "fraction_nonzero": float(np.mean(acts_array > 0)),
        "n_items": len(acts_array),
    }
    
    # Ambig-specific stats (primary comparison, matches B1)
    ambig_items = category_items[category_items["context_condition"] == "ambig"]
    ambig_acts = np.array([cat_acts[i] for i in ambig_items["item_idx"]])
    
    ambig_stereo_items = ambig_items[ambig_items["is_stereotyped_response"] == True]
    ambig_non_stereo_items = ambig_items[ambig_items["is_stereotyped_response"] == False]
    
    stereo_acts = np.array([cat_acts[i] for i in ambig_stereo_items["item_idx"]])
    non_stereo_acts = np.array([cat_acts[i] for i in ambig_non_stereo_items["item_idx"]])
    
    stats["ambig"] = {
        "mean_all": float(ambig_acts.mean()) if len(ambig_acts) else None,
        "mean_stereo_response": float(stereo_acts.mean()) if len(stereo_acts) else None,
        "mean_non_stereo_response": float(non_stereo_acts.mean()) if len(non_stereo_acts) else None,
        "n_stereo": int(len(stereo_acts)),
        "n_non_stereo": int(len(non_stereo_acts)),
    }
    
    # Disambig-specific (secondary, for comparison with ambig)
    disambig_items = category_items[category_items["context_condition"] == "disambig"]
    disambig_acts = np.array([cat_acts[i] for i in disambig_items["item_idx"]])
    stats["disambig"] = {
        "mean_all": float(disambig_acts.mean()) if len(disambig_acts) else None,
        "n_items": int(len(disambig_acts)),
    }
    
    return stats
```

---

## Analysis B: Matched-Pairs Ambig/Disambig Comparison

**New analysis (not in original spec).** BBQ pairs every question_index with ambig and disambig variants. Compare a feature's activation on each matched pair to diagnose how context evidence changes feature firing.

```python
def compute_matched_pairs_comparison(
    feature_activations: pd.Series,
    category_items: pd.DataFrame,
    category: str,
) -> dict:
    """
    For each (question_index, polarity) combination, identify matched ambig/disambig items
    and compute paired activation differences.
    
    Returns:
        Statistics describing how feature activation changes when context evidence is added.
    """
    # Group items by question_index and question_polarity
    # Each group contains matched ambig + disambig variants
    matched_pairs = []
    
    for (qidx, polarity), group in category_items.groupby(["question_index", "question_polarity"]):
        ambig_rows = group[group["context_condition"] == "ambig"]
        disambig_rows = group[group["context_condition"] == "disambig"]
        
        if len(ambig_rows) == 0 or len(disambig_rows) == 0:
            continue
        
        # For each (ambig, disambig) pair (there may be multiple per group)
        # Pair them by order within stereotyped_groups signature
        for _, a_row in ambig_rows.iterrows():
            # Find matching disambig by stereotyped_groups
            a_groups = sorted(a_row["stereotyped_groups"])
            for _, d_row in disambig_rows.iterrows():
                d_groups = sorted(d_row["stereotyped_groups"])
                if a_groups == d_groups:
                    a_act = float(feature_activations.get((category, a_row["item_idx"]), 0.0))
                    d_act = float(feature_activations.get((category, d_row["item_idx"]), 0.0))
                    matched_pairs.append({
                        "question_index": qidx,
                        "polarity": polarity,
                        "stereotyped_groups": a_groups,
                        "ambig_item_idx": a_row["item_idx"],
                        "disambig_item_idx": d_row["item_idx"],
                        "ambig_activation": a_act,
                        "disambig_activation": d_act,
                        "delta": a_act - d_act,  # positive = feature fires more under ambiguity
                    })
                    break  # one-to-one matching
    
    if not matched_pairs:
        return {"n_pairs": 0, "matched_pairs_available": False}
    
    deltas = np.array([p["delta"] for p in matched_pairs])
    ambig_acts = np.array([p["ambig_activation"] for p in matched_pairs])
    disambig_acts = np.array([p["disambig_activation"] for p in matched_pairs])
    
    # Pearson correlation between paired activations
    if len(matched_pairs) >= 2 and (np.std(ambig_acts) > 0 and np.std(disambig_acts) > 0):
        correlation = float(np.corrcoef(ambig_acts, disambig_acts)[0, 1])
    else:
        correlation = None
    
    return {
        "matched_pairs_available": True,
        "n_pairs": len(matched_pairs),
        "mean_ambig_activation": float(ambig_acts.mean()),
        "mean_disambig_activation": float(disambig_acts.mean()),
        "mean_delta": float(deltas.mean()),
        "std_delta": float(deltas.std()),
        "fraction_ambig_higher": float((deltas > 0).mean()),
        "pearson_correlation": correlation,
    }
```

**Interpretation:**
- `mean_delta > 0` with high `fraction_ambig_higher` → feature fires more under ambiguity (context-sensitive feature)
- `mean_delta ≈ 0` and high `correlation` → feature fires on identity content regardless of context (content feature)
- `mean_delta < 0` → feature fires more when evidence is present (unusual; may track evidential linguistic structure)

For the ideal "bias feature" in the subgroup-specific sense, we'd expect:
- Activations higher on ambig than disambig items (model relies on priors more under uncertainty)
- Correlation between ambig and disambig positive (same identity content triggers similar patterns)

---

## Analysis C: Top-Activating Items

```python
def get_top_activating_items(
    feature_activations: pd.Series,
    category_items: pd.DataFrame,
    stimuli: list[dict],
    category: str,
    top_n: int = 20,
) -> list[dict]:
    """
    Return the top-N items by activation value for this feature, with prompt previews.
    
    Uses all items in the category (both ambig and disambig).
    """
    # Build activation list for category items
    item_acts = []
    for _, row in category_items.iterrows():
        act = float(feature_activations.get((category, row["item_idx"]), 0.0))
        item_acts.append((act, row["item_idx"], row))
    
    # Sort descending by activation
    item_acts.sort(key=lambda x: -x[0])
    
    # Build item_idx → stimulus lookup
    stimuli_by_idx = {s["item_idx"]: s for s in stimuli}
    
    top_items = []
    for act, item_idx, meta_row in item_acts[:top_n]:
        stim = stimuli_by_idx.get(item_idx, {})
        top_items.append({
            "item_idx": int(item_idx),
            "activation": round(act, 4),
            "prompt_preview": stim.get("prompt", "")[:150],
            "model_answer_role": meta_row["model_answer_role"],
            "model_answer": meta_row["model_answer"],
            "stereotyped_groups": meta_row["stereotyped_groups"],
            "n_target_groups": int(meta_row["n_target_groups"]),
            "context_condition": meta_row["context_condition"],
            "question_polarity": meta_row["question_polarity"],
            "is_stereotyped_response": bool(meta_row["is_stereotyped_response"]),
        })
    
    return top_items
```

This shows WHICH BBQ scenarios trigger the feature most strongly. For face validity: does the feature fire on content related to the subgroup's stereotypes?

---

## Analysis D: Subgroup Specificity (Within Category)

```python
def compute_subgroup_specificity(
    feature_activations: pd.Series,
    category_items: pd.DataFrame,
    category: str,
    target_subgroup: str,
) -> dict:
    """
    Compute how much more the feature fires on items targeting target_subgroup
    vs. items targeting other subgroups in the same category.
    
    Uses ambig items only (primary comparison).
    """
    ambig = category_items[category_items["context_condition"] == "ambig"]
    
    # Per-subgroup mean activations
    per_subgroup = {}
    all_subgroups = set()
    for gs in ambig["stereotyped_groups"]:
        all_subgroups.update(gs)
    
    for sub in sorted(all_subgroups):
        sub_items = ambig[ambig["stereotyped_groups"].apply(lambda gs: sub in gs)]
        if len(sub_items) == 0:
            continue
        acts = np.array([
            float(feature_activations.get((category, idx), 0.0))
            for idx in sub_items["item_idx"]
        ])
        per_subgroup[sub] = {
            "n_items": int(len(acts)),
            "mean_activation": float(acts.mean()),
            "fraction_nonzero": float((acts > 0).mean()),
        }
    
    if target_subgroup not in per_subgroup:
        return {"per_subgroup": per_subgroup, "subgroup_specificity": None}
    
    target_mean = per_subgroup[target_subgroup]["mean_activation"]
    category_mean = np.mean([v["mean_activation"] for v in per_subgroup.values()])
    
    specificity = target_mean / max(category_mean, 1e-8)
    
    return {
        "per_subgroup_activations": per_subgroup,
        "target_mean": float(target_mean),
        "category_mean": float(category_mean),
        "subgroup_specificity": round(float(specificity), 4),
    }
```

**Thresholds:**
- subgroup_specificity > 1.5: subgroup-specific feature (strong fragmentation signal)
- subgroup_specificity 0.8-1.5: category-general (fires on all subgroups similarly)
- subgroup_specificity < 0.8: fires MORE on other subgroups (unusual; worth investigating)

---

## Analysis E: Cross-Subgroup Activation Matrix

For each category, build a matrix where rows are characterized features (from ANY subgroup in that category) and columns are target subgroups.

```python
def build_cross_subgroup_matrix(
    category: str,
    top_features: pd.DataFrame,
    run_dir: Path,
    metadata_df: pd.DataFrame,
) -> dict:
    """
    For all characterized pro-bias features in a category, compute mean activation
    per target subgroup. Produce matrix, cluster rows, compute ARI and block-diagonal strength.
    
    Returns:
        {
            "feature_labels": list of "{source_sub}:L{layer}_F{feature_idx}",
            "source_subgroups": list of source subgroups (one per row),
            "target_subgroups": list of target subgroups (columns),
            "matrix": (n_features, n_subgroups) activation matrix,
            "cluster_assignments": list of cluster IDs (one per row),
            "adjusted_rand_index": float,
            "block_diagonal_strength": float,
        }
    """
    # Filter to pro-bias features in this category
    cat_features = top_features[
        (top_features["category"] == category) &
        (top_features["direction"] == "pro_bias")
    ].copy()
    
    if cat_features.empty:
        return None
    
    # Identify subgroups
    cat_meta = metadata_df[metadata_df["category"] == category]
    ambig = cat_meta[cat_meta["context_condition"] == "ambig"]
    
    all_subgroups = set()
    for gs in ambig["stereotyped_groups"]:
        all_subgroups.update(gs)
    subgroups_list = sorted(all_subgroups)
    
    # Build matrix
    n_features = len(cat_features)
    n_subgroups = len(subgroups_list)
    matrix = np.zeros((n_features, n_subgroups), dtype=np.float32)
    feature_labels = []
    source_subgroups = []
    
    # Cache layer parquets to avoid re-loading
    layer_cache = {}
    
    for i, (_, feat_row) in enumerate(cat_features.iterrows()):
        fidx = int(feat_row["feature_idx"])
        layer = int(feat_row["layer"])
        src_sub = feat_row["subgroup"]
        
        feature_labels.append(f"{src_sub}:L{layer}_F{fidx}")
        source_subgroups.append(src_sub)
        
        # Load layer parquet once
        if layer not in layer_cache:
            layer_cache[layer] = pd.read_parquet(
                run_dir / "A_extraction" / "sae_encoding" / f"layer_{layer:02d}.parquet"
            )
        layer_df = layer_cache[layer]
        
        feat_acts = layer_df[
            (layer_df["feature_idx"] == fidx) &
            (layer_df["category"] == category)
        ].set_index("item_idx")["activation_value"]
        
        for j, tgt_sub in enumerate(subgroups_list):
            tgt_items = ambig[ambig["stereotyped_groups"].apply(lambda gs: tgt_sub in gs)]
            if len(tgt_items) == 0:
                continue
            vals = np.array([
                float(feat_acts.get(idx, 0.0))
                for idx in tgt_items["item_idx"]
            ])
            matrix[i, j] = float(vals.mean())
    
    # Clustering: normalize rows by max (focus on shape, not scale)
    row_maxes = np.maximum(matrix.max(axis=1, keepdims=True), 1e-8)
    normalized = matrix / row_maxes
    
    # Hierarchical clustering
    from scipy.cluster.hierarchy import linkage, fcluster
    try:
        linkage_matrix = linkage(normalized, method="ward")
        cluster_assignments = fcluster(linkage_matrix, t=n_subgroups, criterion="maxclust")
    except Exception:
        cluster_assignments = np.arange(n_features)
    
    # Adjusted Rand Index: does clustering recover source subgroup labels?
    from sklearn.metrics import adjusted_rand_score
    ari = float(adjusted_rand_score(source_subgroups, cluster_assignments))
    
    # Block-diagonal strength
    diagonal_vals = []
    off_diagonal_vals = []
    for i, src in enumerate(source_subgroups):
        for j, tgt in enumerate(subgroups_list):
            if src == tgt:
                diagonal_vals.append(matrix[i, j])
            else:
                off_diagonal_vals.append(matrix[i, j])
    
    diag_mean = float(np.mean(diagonal_vals)) if diagonal_vals else 0.0
    off_diag_mean = float(np.mean(off_diagonal_vals)) if off_diagonal_vals else 1e-8
    block_diagonal_strength = diag_mean / max(off_diag_mean, 1e-8)
    
    return {
        "feature_labels": feature_labels,
        "source_subgroups": source_subgroups,
        "target_subgroups": subgroups_list,
        "matrix": matrix.tolist(),
        "cluster_assignments": cluster_assignments.tolist(),
        "adjusted_rand_index": round(ari, 4),
        "block_diagonal_strength": round(block_diagonal_strength, 4),
        "diagonal_mean": round(diag_mean, 4),
        "off_diagonal_mean": round(off_diag_mean, 4),
    }
```

**Interpretation:**
- ARI ≈ 1: Clustering recovers source subgroups perfectly → strong feature-level fragmentation
- ARI ≈ 0: Random clustering → features don't align with source subgroups (category-general features)
- Block-diagonal strength >> 1: Features activate strongly on their source subgroup vs others
- Block-diagonal strength ≈ 1: Uniform activation across subgroups (no specificity)
- Block-diagonal strength < 1: Anti-diagonal pattern (features fire MORE on other subgroups)

Both metrics reported because they measure different things: ARI is about the categorical structure of the clustering, block-diagonal strength is about the magnitude of the specificity effect.

---

## Analysis F: Cross-Category Baseline (Artifact Detection)

For each characterized feature, compute mean activation in its source category vs. all other categories.

```python
def compute_category_specificity_ratio(
    feature_activations: pd.Series,
    source_category: str,
    all_categories: list[str],
    metadata_df: pd.DataFrame,
) -> dict:
    """
    Compute ratio of feature's mean activation within source category vs. across other categories.
    
    Within-category: ambig items only (matches B1 comparison scope).
    Cross-category: all items (both conditions, for a robust baseline estimate).
    """
    # Within-category mean (ambig only)
    src_ambig = metadata_df[
        (metadata_df["category"] == source_category) &
        (metadata_df["context_condition"] == "ambig")
    ]
    src_acts = np.array([
        float(feature_activations.get((source_category, idx), 0.0))
        for idx in src_ambig["item_idx"]
    ])
    within_mean = float(src_acts.mean()) if len(src_acts) else 0.0
    
    # Cross-category mean (all items from other categories)
    other_means = []
    for cat in all_categories:
        if cat == source_category:
            continue
        other_items = metadata_df[metadata_df["category"] == cat]
        if len(other_items) == 0:
            continue
        other_acts = np.array([
            float(feature_activations.get((cat, idx), 0.0))
            for idx in other_items["item_idx"]
        ])
        other_means.append(float(other_acts.mean()))
    
    cross_mean = float(np.mean(other_means)) if other_means else 1e-8
    
    ratio = within_mean / max(cross_mean, 1e-8)
    
    return {
        "within_category_mean": round(within_mean, 4),
        "cross_category_mean": round(cross_mean, 4),
        "category_specificity_ratio": round(ratio, 4),
    }
```

---

## Analysis G: Template Feature Detection

Three heuristics combine to flag likely template/artifact features:

```python
def detect_template_artifacts(
    feature_activations: pd.Series,
    source_category: str,
    category_specificity_ratio: float,
    category_items: pd.DataFrame,
    stimuli: list[dict],
) -> dict:
    """
    Flag features that look like template/artifact responses rather than bias features.
    
    Three heuristics:
        1. category_specificity_ratio < 2.0 — fires equally across unrelated categories
        2. |correlation(activation, prompt_length)| > 0.5 — activation tracks prompt length
        3. firing_rate_in_source_category > 0.8 — fires on near-all items
    
    A feature flagged on ANY heuristic is added to artifact_flags.
    """
    flags = []
    
    # Heuristic 1: low category specificity
    if category_specificity_ratio < 2.0:
        flags.append("low_category_specificity")
    
    # Heuristic 2: activation correlates with prompt length
    stimuli_by_idx = {s["item_idx"]: s for s in stimuli}
    activations = []
    lengths = []
    for _, row in category_items.iterrows():
        idx = row["item_idx"]
        act = float(feature_activations.get((source_category, idx), 0.0))
        prompt = stimuli_by_idx.get(idx, {}).get("prompt", "")
        activations.append(act)
        lengths.append(len(prompt))
    
    activations = np.array(activations)
    lengths = np.array(lengths)
    
    length_correlation = None
    if len(activations) >= 10 and np.std(activations) > 0 and np.std(lengths) > 0:
        length_correlation = float(np.corrcoef(activations, lengths)[0, 1])
        if abs(length_correlation) > 0.5:
            flags.append("length_correlation")
    
    # Heuristic 3: high firing rate
    firing_rate = float((activations > 0).mean())
    if firing_rate > 0.8:
        flags.append("high_firing_rate")
    
    return {
        "length_correlation": round(length_correlation, 4) if length_correlation is not None else None,
        "firing_rate_source_category": round(firing_rate, 4),
        "artifact_flags": flags,
        "is_artifact_flagged": len(flags) > 0,
    }
```

**Flagged features are NOT automatically excluded from analysis** — they still appear in the ranked lists. But `artifact_flags.json` provides the list of flagged (category, feature_idx, layer) triples that C1 should exclude from steering candidates.

---

## Analysis H: Feature Co-occurrence

For each subgroup, compute pairwise correlation of feature activations across items for the top-10 pro-bias features.

```python
def compute_feature_cooccurrence(
    category: str,
    subgroup: str,
    top_features: pd.DataFrame,
    run_dir: Path,
    metadata_df: pd.DataFrame,
    top_n_cooccur: int = 10,
) -> dict:
    """
    For one subgroup, compute pairwise activation correlations among top-N pro-bias features.
    
    Correlation is computed across items targeting the subgroup, ambig items only.
    """
    sub_features = top_features[
        (top_features["category"] == category) &
        (top_features["subgroup"] == subgroup) &
        (top_features["direction"] == "pro_bias") &
        (top_features["rank"] <= top_n_cooccur)
    ].sort_values("rank")
    
    if len(sub_features) < 2:
        return {"matrix": None, "feature_labels": [], "n_features": len(sub_features)}
    
    # Items targeting this subgroup (ambig only)
    sub_items = metadata_df[
        (metadata_df["category"] == category) &
        (metadata_df["context_condition"] == "ambig") &
        (metadata_df["stereotyped_groups"].apply(lambda gs: subgroup in gs))
    ]
    item_idxs = sub_items["item_idx"].tolist()
    
    if len(item_idxs) < 10:
        return {"matrix": None, "feature_labels": [], "n_items": len(item_idxs)}
    
    # Build feature × item activation matrix
    n_feat = len(sub_features)
    n_items = len(item_idxs)
    activations = np.zeros((n_feat, n_items), dtype=np.float32)
    
    layer_cache = {}
    feature_labels = []
    
    for i, (_, feat_row) in enumerate(sub_features.iterrows()):
        fidx = int(feat_row["feature_idx"])
        layer = int(feat_row["layer"])
        feature_labels.append(f"L{layer}_F{fidx}")
        
        if layer not in layer_cache:
            layer_cache[layer] = pd.read_parquet(
                run_dir / "A_extraction" / "sae_encoding" / f"layer_{layer:02d}.parquet"
            )
        feat_acts = layer_cache[layer][
            (layer_cache[layer]["feature_idx"] == fidx) &
            (layer_cache[layer]["category"] == category)
        ].set_index("item_idx")["activation_value"]
        
        for j, idx in enumerate(item_idxs):
            activations[i, j] = float(feat_acts.get(idx, 0.0))
    
    # Pairwise correlation
    correlation_matrix = np.corrcoef(activations)
    # Handle NaN from zero-variance rows
    correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
    
    return {
        "feature_labels": feature_labels,
        "matrix": correlation_matrix.tolist(),
        "n_features": int(n_feat),
        "n_items": int(n_items),
    }
```

**Interpretation:**
- High correlation (>0.7) between two features: they fire on the same items → redundant. Adding both to steering vector may not help.
- Low correlation (<0.3): features fire on disjoint items → complementary. Both contribute unique signal.

This informs C1's k-selection: if top features are highly correlated, smaller k may be optimal.

---

## Script Structure

```python
def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = load_config(run_dir)
    
    categories = parse_categories(args.categories) if args.categories else config["categories"]
    top_k = args.top_k
    
    # Load inputs
    metadata_df = load_metadata(run_dir)
    top_features = load_characterization_features(run_dir, top_k=top_k)
    
    # Filter to requested categories
    top_features = top_features[top_features["category"].isin(categories)]
    
    log(f"Characterizing {len(top_features)} feature entries across {len(categories)} categories")
    
    # Load stimuli per category
    stimuli_by_cat = {}
    for cat in categories:
        stimuli_path = run_dir / "A_extraction" / "stimuli" / f"{cat}.json"
        with open(stimuli_path) as f:
            stimuli_by_cat[cat] = json.load(f)
    
    # Per-feature records for feature_stats.parquet
    stats_records = []
    
    # Per-feature top-activating items for top_activating_items.json
    top_items_dict = {}
    
    # Cross-subgroup matrices per category
    cross_subgroup_matrices = {}
    
    # Co-occurrence matrices per subgroup
    cooccurrence = {}
    
    # Artifact flags
    artifact_list = []
    
    # Layer cache (shared across features)
    layer_cache = {}
    
    # Process each characterized feature
    for _, feat_row in top_features.iterrows():
        cat = feat_row["category"]
        sub = feat_row["subgroup"]
        direction = feat_row["direction"]
        rank = int(feat_row["rank"])
        fidx = int(feat_row["feature_idx"])
        layer = int(feat_row["layer"])
        
        # Load layer parquet (cached)
        if layer not in layer_cache:
            layer_cache[layer] = pd.read_parquet(
                run_dir / "A_extraction" / "sae_encoding" / f"layer_{layer:02d}.parquet"
            )
        layer_df = layer_cache[layer]
        
        # Get feature's activations across ALL items
        feat_df = layer_df[layer_df["feature_idx"] == fidx]
        feature_activations = feat_df.set_index(["category", "item_idx"])["activation_value"]
        
        # Get category items
        cat_meta = metadata_df[metadata_df["category"] == cat]
        
        # Run all analyses
        dist = compute_activation_distribution(feature_activations, cat_meta, cat)
        matched = compute_matched_pairs_comparison(feature_activations, cat_meta, cat)
        spec = compute_subgroup_specificity(feature_activations, cat_meta, cat, sub)
        cat_spec = compute_category_specificity_ratio(feature_activations, cat, categories, metadata_df)
        artifact = detect_template_artifacts(
            feature_activations, cat,
            cat_spec["category_specificity_ratio"],
            cat_meta, stimuli_by_cat[cat],
        )
        
        # Assemble stats record
        stats_records.append({
            "category": cat,
            "subgroup": sub,
            "direction": direction,
            "rank": rank,
            "feature_idx": fidx,
            "layer": layer,
            "cohens_d": float(feat_row["cohens_d"]),
            "p_value_fdr": float(feat_row["p_value_fdr"]),
            "mean_all": dist["mean_all"],
            "std_all": dist["std_all"],
            "median_all": dist["median_all"],
            "max_activation": dist["max_activation"],
            "fraction_nonzero": dist["fraction_nonzero"],
            "mean_ambig_stereo_response": dist["ambig"]["mean_stereo_response"],
            "mean_ambig_non_stereo_response": dist["ambig"]["mean_non_stereo_response"],
            "mean_disambig": dist["disambig"]["mean_all"],
            "matched_n_pairs": matched["n_pairs"],
            "matched_mean_delta": matched.get("mean_delta"),
            "matched_correlation": matched.get("pearson_correlation"),
            "subgroup_specificity": spec["subgroup_specificity"],
            "target_mean": spec.get("target_mean"),
            "category_mean": spec.get("category_mean"),
            "within_category_mean": cat_spec["within_category_mean"],
            "cross_category_mean": cat_spec["cross_category_mean"],
            "category_specificity_ratio": cat_spec["category_specificity_ratio"],
            "length_correlation": artifact["length_correlation"],
            "firing_rate_source_category": artifact["firing_rate_source_category"],
            "is_artifact_flagged": artifact["is_artifact_flagged"],
            "artifact_flags": ",".join(artifact["artifact_flags"]),
        })
        
        # Top-activating items
        top_items = get_top_activating_items(
            feature_activations, cat_meta, stimuli_by_cat[cat], cat, top_n=20
        )
        key = f"{cat}/{sub}/{direction}/rank{rank:03d}/L{layer}_F{fidx}"
        top_items_dict[key] = top_items
        
        # Track artifact flags
        if artifact["is_artifact_flagged"]:
            artifact_list.append({
                "category": cat,
                "subgroup": sub,
                "direction": direction,
                "rank": rank,
                "feature_idx": fidx,
                "layer": layer,
                "flags": artifact["artifact_flags"],
                "category_specificity_ratio": cat_spec["category_specificity_ratio"],
                "length_correlation": artifact["length_correlation"],
                "firing_rate": artifact["firing_rate_source_category"],
            })
    
    # Cross-subgroup matrices per category (one per category, pro-bias only)
    for cat in categories:
        matrix_result = build_cross_subgroup_matrix(cat, top_features, run_dir, metadata_df)
        if matrix_result is not None:
            cross_subgroup_matrices[cat] = matrix_result
    
    # Co-occurrence per subgroup
    for cat in categories:
        cat_subs = top_features[top_features["category"] == cat]["subgroup"].unique()
        for sub in sorted(cat_subs):
            cooccur = compute_feature_cooccurrence(cat, sub, top_features, run_dir, metadata_df)
            if cooccur["matrix"] is not None:
                key = f"{cat}/{sub}"
                cooccurrence[key] = cooccur
    
    # Save outputs
    save_outputs(run_dir, stats_records, top_items_dict,
                 cross_subgroup_matrices, cooccurrence, artifact_list)
    
    if not args.skip_figures:
        generate_figures(run_dir)
```

---

## Output Files

### `feature_stats.parquet`

Primary output with per-feature statistics. Flat parquet for fast querying.

```
{run}/B_feature_interpretability/feature_stats.parquet
```

| Column | Type | Description |
|---|---|---|
| `category` | string | |
| `subgroup` | string | Source subgroup that selected this feature |
| `direction` | string | pro_bias / anti_bias |
| `rank` | int32 | Rank within (category, subgroup, direction) |
| `feature_idx` | int32 | |
| `layer` | int32 | |
| `cohens_d` | float32 | From B1 |
| `p_value_fdr` | float64 | From B1 |
| `mean_all` | float32 | Overall mean activation in source category |
| `std_all` | float32 | |
| `median_all` | float32 | |
| `max_activation` | float32 | |
| `fraction_nonzero` | float32 | |
| `mean_ambig_stereo_response` | float32 | |
| `mean_ambig_non_stereo_response` | float32 | |
| `mean_disambig` | float32 | |
| `matched_n_pairs` | int32 | Count of matched ambig/disambig pairs |
| `matched_mean_delta` | float32 | Mean (ambig − disambig) activation across pairs |
| `matched_correlation` | float32 | Correlation between ambig and disambig activations |
| `subgroup_specificity` | float32 | Target / category_mean |
| `target_mean` | float32 | Mean activation on target subgroup (ambig) |
| `category_mean` | float32 | Mean activation across all subgroups in category (ambig) |
| `within_category_mean` | float32 | |
| `cross_category_mean` | float32 | |
| `category_specificity_ratio` | float32 | within / cross |
| `length_correlation` | float32 | Activation vs prompt length correlation |
| `firing_rate_source_category` | float32 | |
| `is_artifact_flagged` | bool | |
| `artifact_flags` | string | Comma-separated reasons |

### `top_activating_items.json`

Keyed by `"{cat}/{sub}/{direction}/rank{rank:03d}/L{layer}_F{fidx}"`, value = list of up to 20 item dicts.

```
{run}/B_feature_interpretability/top_activating_items.json
```

### `cross_subgroup_activation_matrices.json`

One entry per category, containing the cross-subgroup matrix, clustering, ARI, and block-diagonal strength.

```json
{
  "so": {
    "feature_labels": ["gay:L14_F45021", "bisexual:L16_F88012", ...],
    "source_subgroups": ["gay", "gay", "bisexual", ...],
    "target_subgroups": ["bisexual", "gay", "lesbian", "pansexual"],
    "matrix": [[...], [...], ...],
    "cluster_assignments": [1, 1, 2, ...],
    "adjusted_rand_index": 0.723,
    "block_diagonal_strength": 3.41,
    "diagonal_mean": 1.847,
    "off_diagonal_mean": 0.542
  },
  ...
}
```

### `feature_cooccurrence.json`

Per-subgroup co-occurrence matrices for top-10 pro-bias features.

```json
{
  "so/gay": {
    "feature_labels": ["L14_F45021", "L16_F88012", ...],
    "matrix": [[1.0, 0.23, ...], [0.23, 1.0, ...], ...],
    "n_features": 10,
    "n_items": 487
  },
  ...
}
```

### `artifact_flags.json`

List of features flagged as template/artifact candidates. C1 uses this to exclude them from steering candidates.

```json
{
  "n_flagged": 47,
  "flagging_criteria": {
    "low_category_specificity_threshold": 2.0,
    "length_correlation_threshold": 0.5,
    "high_firing_rate_threshold": 0.8
  },
  "flagged_features": [
    {
      "category": "so",
      "subgroup": "gay",
      "direction": "pro_bias",
      "rank": 7,
      "feature_idx": 123456,
      "layer": 18,
      "flags": ["low_category_specificity", "high_firing_rate"],
      "category_specificity_ratio": 1.2,
      "length_correlation": 0.34,
      "firing_rate": 0.91
    },
    ...
  ]
}
```

### `interpretability_summary.json`

Top-level run summary.

```json
{
  "config": {
    "top_k": 20,
    "categories": ["age", "disability", "gi", ...],
    "length_correlation_threshold": 0.5,
    "high_firing_rate_threshold": 0.8,
    "low_category_specificity_threshold": 2.0
  },
  "n_features_characterized": 1200,
  "n_artifact_flagged": 47,
  "artifact_flag_rate_per_category": {
    "so": 0.08,
    "race": 0.12,
    ...
  },
  "block_diagonal_strength_per_category": {
    "so": 3.41,
    "race": 2.87,
    ...
  },
  "adjusted_rand_index_per_category": {
    "so": 0.723,
    ...
  },
  "runtime_seconds": 542.0
}
```

---

## Figures

### `fig_cross_subgroup_activation_{category}.png`

Clustered heatmap per category.
- Rows: features, labeled `{source_sub}:L{layer}_F{idx}`
- Columns: target subgroups (alphabetical)
- Color: mean activation (YlOrRd, vmin=0, vmax=dataset max)
- Hierarchical clustering dendrogram on row axis
- Annotations: ARI and block-diagonal strength in title
- Title: "Cross-subgroup activations — {category} (ARI={ari:.2f}, BDS={bds:.1f})"

### `fig_subgroup_specificity_distribution.png`

Histogram of `subgroup_specificity` across all characterized pro-bias features.
- X: subgroup_specificity score
- Y: count
- Vertical dashed lines at 0.8 and 1.5 (thresholds)
- Annotations: median, IQR, fraction above 1.5
- Title: "Subgroup specificity distribution (all pro-bias features)"

### `fig_category_specificity_ratio.png`

Histogram of `category_specificity_ratio`.
- X: ratio (log scale recommended)
- Y: count
- Vertical dashed line at 2.0 (threshold)
- Annotations: number of features below threshold
- Title: "Category specificity ratio (all characterized features)"

### `fig_matched_pairs_delta.png`

Matched-pairs ambig vs disambig activation deltas.
- One panel per category
- X: feature rank within subgroup (1 to 20)
- Y: mean matched-pairs delta (ambig − disambig)
- One line per subgroup, distinct color + marker
- Horizontal dashed line at y=0
- Title: "Ambig vs disambig activation delta — {category}"

### `fig_feature_cooccurrence_{category}.png`

Correlation heatmaps for each subgroup with ≥2 features.
- Grid of small heatmaps (one per subgroup)
- Each: 10×10 correlation matrix (RdBu_r, vmin=-1, vmax=1)
- Annotated cells
- Title: "Feature co-occurrence — {category}"

### `fig_artifact_flag_summary.png`

Bar chart of artifact flag rates per (category, flag type).
- X: category
- Y: fraction of characterized features flagged
- Grouped bars: low_category_specificity, length_correlation, high_firing_rate, any_flag
- Title: "Artifact flag rates by category"

---

## Output Structure

```
{run}/B_feature_interpretability/
├── feature_stats.parquet                          # Per-feature stats, flat
├── top_activating_items.json                       # Per-feature top-20 items
├── cross_subgroup_activation_matrices.json         # Per-category matrices + ARI + BDS
├── feature_cooccurrence.json                       # Per-subgroup correlation matrices
├── artifact_flags.json                             # Flagged features for C1 to exclude
├── interpretability_summary.json                   # Top-level summary
└── figures/
    ├── fig_cross_subgroup_activation_{cat}.png/.pdf
    ├── fig_subgroup_specificity_distribution.png/.pdf
    ├── fig_category_specificity_ratio.png/.pdf
    ├── fig_matched_pairs_delta.png/.pdf
    ├── fig_feature_cooccurrence_{cat}.png/.pdf
    └── fig_artifact_flag_summary.png/.pdf
```

---

## Resume Safety

Coarse-grained. Check for `interpretability_summary.json`. If present and not `--force`, skip.

Atomic writes for all outputs.

---

## Compute Estimate

- Per feature: ~3-5 seconds (load activations, compute stats, matched pairs)
- Total features: ~1200 (~40 subgroups × 30 features)
- Layer caching reduces redundant parquet loads drastically
- Total: ~10-20 minutes

---

## Assumptions Summary

| # | Decision | Value |
|---|---|---|
| 1 | Top-K for characterization | K=20 for both pro-bias and anti-bias (configurable) |
| 2 | Primary comparison context | Ambig items (matches B1) |
| 3 | Disambig analysis | Matched-pairs comparison per question_index |
| 4 | Subgroup specificity | target_mean / category_mean (ambig items) |
| 5 | Category specificity | within_cat_ambig_mean / cross_cat_all_items_mean |
| 6 | Artifact flag heuristics | cat_spec_ratio<2, length_corr>0.5, firing_rate>0.8 |
| 7 | Artifact flag policy | ANY of three heuristics flags the feature |
| 8 | Cross-subgroup matrix scope | Pro-bias features only |
| 9 | Clustering method | Hierarchical (Ward), matched to n_subgroups clusters |
| 10 | Clustering agreement metric | Adjusted Rand Index (ARI) |
| 11 | Block-diagonal metric | Ratio of diagonal-mean to off-diagonal-mean |
| 12 | Co-occurrence scope | Top-10 pro-bias features per subgroup |
| 13 | Output format | Parquet for flat stats, JSON for nested structures |
| 14 | Prompt previews | Stored in top_activating_items.json (self-contained) |

---

## Test Command

```bash
# Quick test: SO category
python scripts/B5_feature_interpretability.py \
    --run_dir runs/llama-3.1-8b_2026-04-15/ \
    --categories so

# Verify outputs
python -c "
import pandas as pd
import json

stats = pd.read_parquet('runs/llama-3.1-8b_2026-04-15/B_feature_interpretability/feature_stats.parquet')
print(f'Feature stats: {len(stats)} rows')

# Artifact flag rate
print(f'Artifact flag rate: {stats[\"is_artifact_flagged\"].mean():.2%}')

# Specificity distribution
print(f'Subgroup specificity median: {stats[\"subgroup_specificity\"].median():.2f}')
print(f'Subgroup specificity >1.5 fraction: {(stats[\"subgroup_specificity\"] > 1.5).mean():.2%}')

# Matched pairs
n_with_pairs = (stats['matched_n_pairs'] > 0).sum()
print(f'Features with matched pairs: {n_with_pairs}/{len(stats)}')

# Cross-subgroup
with open('runs/llama-3.1-8b_2026-04-15/B_feature_interpretability/cross_subgroup_activation_matrices.json') as f:
    matrices = json.load(f)
for cat, data in matrices.items():
    print(f'{cat}: ARI={data[\"adjusted_rand_index\"]:.2f}, BDS={data[\"block_diagonal_strength\"]:.2f}')
"
```