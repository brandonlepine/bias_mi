# C4: Token-Level Feature Interpretability — Full Implementation Specification

## Purpose

For the features in each subgroup's optimal steering vector, determine which specific tokens activate the feature and how the feature's decoder direction affects output logits. Produces neuronpedia-style per-feature characterizations: logit effects (via unembedding decomposition), activation distribution, top activating examples with highlighted tokens, per-template breakdowns, and token rankings.

## Invocation

```bash
python scripts/C4_token_features.py --run_dir runs/llama-3.1-8b_2026-04-15/

# Specific categories
python scripts/C4_token_features.py --run_dir runs/llama-3.1-8b_2026-04-15/ --categories so,race

# Override per-subgroup feature cap
python scripts/C4_token_features.py --run_dir runs/llama-3.1-8b_2026-04-15/ --max_features_per_subgroup 10

# Save full per-item per-token activations (optional, large output)
python scripts/C4_token_features.py --run_dir runs/llama-3.1-8b_2026-04-15/ --save_full_per_token

# Skip figures
python scripts/C4_token_features.py --run_dir runs/llama-3.1-8b_2026-04-15/ --skip_figures
```

Reads `model_path`, `sae_source`, `sae_expansion`, `device`, `dtype` from config.json.

---

## Input

- Steering manifests: `{run}/C_steering/steering_manifests.json` (for `steering_viable` + optimal k per subgroup)
- Steering vectors: `{run}/C_steering/vectors/{cat}_{sub}.npz` (contains `feature_idxs`, `feature_layers` of the actual features used)
- Metadata: `{run}/A_extraction/metadata.parquet`
- Stimuli: `{run}/A_extraction/stimuli/{category}.json`
- Model + SAEs (per layer)

## Dependencies

- `torch`
- `pandas`, `pyarrow`
- `numpy`, `matplotlib`
- Ported: `SAEWrapper`, `ModelWrapper`
- Shared: demographic patterns from C3's `DEMOGRAPHIC_PATTERNS`

---

## The Neuronpedia-Style Output

For each (subgroup, feature) combination, we produce five analysis views (see neuronpedia UI reference):

1. **Logit effect decomposition.** Feature's decoder column projected onto the model's unembedding matrix — which output tokens does this feature promote / suppress? (Computed from SAE decoder + model's output head, no data needed.)

2. **Activation density histogram.** Distribution of activation magnitudes where the feature fires (nonzero) across all token positions across all items.

3. **Top activating examples.** Real BBQ prompts where the feature fires most strongly, with the activating tokens visibly highlighted (for the paper figures).

4. **Token rankings.** Aggregated ranking of tokens by mean activation when the feature fires on that token.

5. **Per-template breakdown.** Which BBQ templates (`question_index`) does the feature activate most strongly on?

---

## Architectural Strategy: Item-Centric Processing

**The critical efficiency insight:** instead of iterating `for feature: for item: forward_pass()` (naive), we iterate `for item: forward_pass_all_needed_layers(); extract_all_needed_features()`. Items are shared across features, so we amortize forward-pass cost across all features that live at the layers we hooked.

```python
# Collect all (layer, feature_idx) pairs needed across all subgroups
needed_features = collect_needed_features(viable_manifests, args.max_features_per_subgroup)
# needed_features: {layer: set of feature_idxs}

# Items are shared within categories (same ambig items used regardless of which subgroup selected a feature)
for category in categories:
    cat_items = load_ambig_items(category)
    cat_needed_layers = layers_for_features_in_category(needed_features, category)
    
    for item in cat_items:
        # ONE forward pass with hooks at all needed layers for this category
        hidden_states_per_layer = forward_with_multilayer_hooks(item, cat_needed_layers)
        
        for layer in cat_needed_layers:
            # Encode per-token hidden states through SAE
            per_token_all_features = sae_cache[layer].encode(hidden_states_per_layer[layer])
            # shape: (seq_len, n_features)
            
            # Extract only the features we care about
            layer_feature_ids = needed_features[layer]
            target_acts = per_token_all_features[:, layer_feature_ids]  # (seq_len, n_target_features)
            
            # Accumulate per-feature activations + tokens
            for feat_i, feature_idx in enumerate(layer_feature_ids):
                activations = target_acts[:, feat_i]  # (seq_len,)
                accumulator[layer, feature_idx].append({
                    "item_idx": item.item_idx,
                    "tokens": tokens,
                    "activations": activations.numpy(),
                    ...
                })
            
            del per_token_all_features  # free the 131K-feature matrix
```

Per-item cost: one multi-layer forward pass + a few SAE encodings. Total cost dominated by forward passes (~0.4s on MPS per item).

---

## Step 0: Collect Features to Analyze

For each viable subgroup, use the features from its optimal steering vector — these are the features actually doing the work in C1. Cap at `max_features_per_subgroup` (default 10) for pathological cases.

```python
def collect_needed_features(
    viable_manifests: list[dict],
    vectors_dir: Path,
    max_features_per_subgroup: int = 10,
) -> tuple[dict, list]:
    """
    From each viable subgroup's npz, extract (feature_idx, layer) pairs.
    
    Returns:
        needed_by_layer: {layer: sorted list of unique feature_idxs across all subgroups at this layer}
        feature_manifest: list of {category, subgroup, rank_in_vector, feature_idx, layer}
    """
    needed_by_layer = defaultdict(set)
    feature_manifest = []
    
    for m in viable_manifests:
        cat = m["category"]
        sub = m["subgroup"]
        vec_path = vectors_dir / f"{cat}_{sub}.npz"
        if not vec_path.exists():
            continue
        data = np.load(vec_path)
        
        feature_idxs = data["feature_idxs"]  # from C1 output
        feature_layers = data["feature_layers"]
        
        # Cap at max_features_per_subgroup
        n_features = min(len(feature_idxs), max_features_per_subgroup)
        
        for rank in range(n_features):
            fidx = int(feature_idxs[rank])
            flayer = int(feature_layers[rank])
            needed_by_layer[flayer].add(fidx)
            feature_manifest.append({
                "category": cat,
                "subgroup": sub,
                "rank_in_vector": rank,
                "feature_idx": fidx,
                "layer": flayer,
            })
    
    # Convert to sorted lists for deterministic extraction order
    needed_sorted = {layer: sorted(feats) for layer, feats in needed_by_layer.items()}
    
    log(f"Collected {len(feature_manifest)} (subgroup, feature) pairs")
    log(f"Unique (layer, feature_idx) pairs: {sum(len(f) for f in needed_sorted.values())}")
    log(f"Layers involved: {sorted(needed_sorted.keys())}")
    
    return needed_sorted, feature_manifest
```

---

## Step 1: Logit Effect Decomposition

The feature's decoder column W_dec[f] is a vector in ℝ^hidden_dim. When projected onto the model's unembedding matrix W_U (shape `hidden_dim × vocab_size`), we get a vector of vocab_size showing which tokens are promoted (positive logit contribution) or suppressed (negative) when this feature fires.

```python
def compute_logit_effects(
    wrapper: ModelWrapper,
    sae_cache: dict[int, SAEWrapper],
    feature_manifest: list[dict],
    tokenizer,
    top_n_tokens: int = 10,
) -> pd.DataFrame:
    """
    For each (feature, layer) in the manifest, compute which output tokens are
    most promoted/suppressed when the feature fires.
    
    Returns DataFrame with columns:
        layer, feature_idx, direction (positive/negative),
        rank, token_id, token_str, logit_contribution
    """
    # Get unembedding matrix
    W_U = wrapper.get_unembedding_matrix()  # (hidden_dim, vocab_size) or (vocab_size, hidden_dim)
    if W_U.shape[0] == wrapper.config.vocab_size:
        W_U = W_U.T  # ensure (hidden_dim, vocab_size)
    W_U = W_U.float().to("cpu")  # keep on CPU, lightweight matmul
    
    rows = []
    processed_pairs = set()
    
    for entry in feature_manifest:
        fidx = entry["feature_idx"]
        flayer = entry["layer"]
        pair = (flayer, fidx)
        if pair in processed_pairs:
            continue
        processed_pairs.add(pair)
        
        # Get decoder column for this feature
        sae = sae_cache[flayer]
        decoder_col = sae.get_feature_decoder_column(fidx)  # (hidden_dim,)
        decoder_col = torch.from_numpy(decoder_col).float()
        
        # Project onto unembedding: logit contribution per vocab token
        # effect[v] = W_U[:, v].dot(decoder_col)
        logit_contribution = W_U.T @ decoder_col  # (vocab_size,)
        
        # Top N positive and negative
        top_pos = torch.topk(logit_contribution, k=top_n_tokens)
        top_neg = torch.topk(-logit_contribution, k=top_n_tokens)
        
        for rank in range(top_n_tokens):
            rows.append({
                "layer": flayer,
                "feature_idx": fidx,
                "direction": "positive",
                "rank": rank + 1,
                "token_id": int(top_pos.indices[rank].item()),
                "token_str": tokenizer.decode([int(top_pos.indices[rank].item())]),
                "logit_contribution": float(top_pos.values[rank].item()),
            })
            rows.append({
                "layer": flayer,
                "feature_idx": fidx,
                "direction": "negative",
                "rank": rank + 1,
                "token_id": int(top_neg.indices[rank].item()),
                "token_str": tokenizer.decode([int(top_neg.indices[rank].item())]),
                "logit_contribution": float(-top_neg.values[rank].item()),  # sign back
            })
    
    return pd.DataFrame(rows)
```

**Note on decoder column retrieval.** `SAEWrapper.get_feature_decoder_column(idx)` should exist in the ported code. If not, it's `W_dec[idx]` where W_dec is the SAE's decoder weight matrix of shape `(n_features, hidden_dim)` — so column `idx` is the direction added to the residual stream when feature `idx` fires.

This analysis is fast (no forward passes needed) and produces interpretable output. Save as a single parquet keyed by (layer, feature_idx).

---

## Step 2: Item-Centric Token Activation Extraction

Process each category once. For every ambig item, do one forward pass with hooks on all layers that contain features we need for that category. Extract per-token feature activations, discard the full activation matrix immediately, keep only target features.

### Per-Token Hook Installation

```python
def make_all_token_hook(layer_idx: int, hidden_dim: int, storage: dict):
    """Hook that stores the FULL per-token hidden states at this layer."""
    def hook_fn(module, args, output):
        h = locate_hidden_tensor(output, hidden_dim)
        # h shape: (batch_size, seq_len, hidden_dim); batch_size = 1 in our case
        storage[layer_idx] = h[0].detach().cpu().float()  # (seq_len, hidden_dim)
    return hook_fn
```

### Main Extraction Loop

```python
def extract_token_activations(
    category: str,
    needed_features_by_layer: dict[int, list[int]],
    wrapper: ModelWrapper,
    sae_cache: dict[int, SAEWrapper],
    metadata_df: pd.DataFrame,
    stimuli: list[dict],
    tokenizer,
    output_dir: Path,
) -> dict:
    """
    Process all ambig items in a category. For each item, run one forward pass
    with hooks at all layers needed for this category's features, then extract
    per-token activations for each target feature.
    
    Returns: accumulator[(layer, feature_idx)] = list of per-item records
    """
    # Layers needed for this category = layers that contain at least one feature
    # used by any subgroup in this category
    cat_needed_layers = set()
    for entry in feature_manifest_by_category[category]:
        cat_needed_layers.add(entry["layer"])
    cat_needed_layers = sorted(cat_needed_layers)
    
    log(f"  {category}: {len(cat_needed_layers)} layers to hook")
    
    # Get ambig items
    cat_ambig = metadata_df[
        (metadata_df["category"] == category) &
        (metadata_df["context_condition"] == "ambig")
    ]
    stimuli_by_idx = {s["item_idx"]: s for s in stimuli}
    
    accumulator = defaultdict(list)  # (layer, feature_idx) -> list of item records
    
    for i, (_, meta_row) in enumerate(cat_ambig.iterrows()):
        item_idx = int(meta_row["item_idx"])
        stim = stimuli_by_idx.get(item_idx)
        if stim is None:
            continue
        
        prompt = stim["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(wrapper.device)
        input_ids = inputs["input_ids"][0].cpu().tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        seq_len = len(tokens)
        
        # Install hooks on all needed layers
        storage = {}
        hooks = []
        for layer in cat_needed_layers:
            h = wrapper.get_layer_module(layer).register_forward_hook(
                make_all_token_hook(layer, wrapper.hidden_dim, storage)
            )
            hooks.append(h)
        
        try:
            with torch.no_grad():
                _ = wrapper.model(**inputs)
        finally:
            for h in hooks:
                h.remove()
        
        # Now we have per-token hidden states at each needed layer
        # For each layer, encode through SAE and extract target features
        for layer in cat_needed_layers:
            if layer not in storage:
                continue
            hidden_all_tokens = storage[layer]  # (seq_len, hidden_dim)
            
            target_feature_ids = needed_features_by_layer.get(layer, [])
            if not target_feature_ids:
                continue
            
            # Encode through SAE: (seq_len, n_features_total)
            # This is the expensive matmul but only done once per (item, layer)
            sae = sae_cache[layer]
            with torch.no_grad():
                feat_acts_full = sae.encode(hidden_all_tokens.to(sae.device))
                # (seq_len, n_features_in_sae)
            
            # Extract only target features
            target_feat_ids_tensor = torch.tensor(target_feature_ids, device=feat_acts_full.device)
            target_acts = feat_acts_full[:, target_feat_ids_tensor].cpu().float().numpy()
            # (seq_len, n_target_features)
            
            # Free the full matrix immediately
            del feat_acts_full
            if hasattr(torch, 'mps'):
                torch.mps.empty_cache() if wrapper.device.type == "mps" else None
            elif wrapper.device.type == "cuda":
                torch.cuda.empty_cache()
            
            # Accumulate per-feature records
            for feat_i, feature_idx in enumerate(target_feature_ids):
                activations_1d = target_acts[:, feat_i]  # (seq_len,)
                accumulator[(layer, feature_idx)].append({
                    "item_idx": item_idx,
                    "question_index": int(meta_row["question_index"]),
                    "question_polarity": meta_row["question_polarity"],
                    "stereotyped_groups": (
                        meta_row["stereotyped_groups"] if isinstance(meta_row["stereotyped_groups"], list)
                        else json.loads(meta_row["stereotyped_groups"])
                    ),
                    "model_answer_role": meta_row["model_answer_role"],
                    "is_stereotyped_response": bool(meta_row["is_stereotyped_response"]),
                    "stereotyped_option": stim.get("stereotyped_option"),
                    "tokens": tokens,
                    "activations": activations_1d.tolist(),
                    "max_activation": float(np.max(activations_1d)),
                    "argmax_position": int(np.argmax(activations_1d)),
                    "argmax_token": tokens[int(np.argmax(activations_1d))],
                })
        
        if (i + 1) % 100 == 0:
            log(f"    {category}: processed {i+1}/{len(cat_ambig)} items")
    
    return accumulator
```

---

## Step 3: Template Filtering (Two Levels)

### String-Level Template Tokens (Per Category)

Tokens appearing in >90% of items in the category, regardless of position. Examples: `":"`, `"Question"`, `"Answer"`, `"\n"`, `"?"`.

```python
def compute_string_level_template_tokens(
    category_items: list[dict],
    tokenizer,
    threshold: float = 0.90,
) -> set[str]:
    """Tokens appearing in >=threshold fraction of items in this category."""
    n_items = len(category_items)
    token_presence = Counter()
    
    for item in category_items:
        tokens_in_item = set(tokenizer.convert_ids_to_tokens(tokenizer.encode(item["prompt"])))
        for tok in tokens_in_item:
            token_presence[tok] += 1
    
    template_tokens = {tok for tok, count in token_presence.items() 
                       if count / n_items >= threshold}
    return template_tokens
```

### Position-Level Template Positions (Per question_index)

Within each template (shared question_index), positions where the same token appears in >80% of items in that template.

```python
def compute_position_level_template_positions(
    items: list[dict],
    metadata_df: pd.DataFrame,
    tokenizer,
    threshold: float = 0.80,
) -> dict[int, set[int]]:
    """
    Per question_index, compute set of positions that are template-invariant
    (same token in >threshold fraction of items in that template).
    """
    items_by_qidx = defaultdict(list)
    meta_by_idx = metadata_df.set_index("item_idx")
    
    for item in items:
        try:
            qidx = int(meta_by_idx.loc[item["item_idx"], "question_index"])
        except KeyError:
            continue
        items_by_qidx[qidx].append(item)
    
    template_positions = {}
    for qidx, qidx_items in items_by_qidx.items():
        if len(qidx_items) < 5:
            continue
        
        position_token_counts = defaultdict(Counter)
        for item in qidx_items:
            tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(item["prompt"]))
            for pos, tok in enumerate(tokens):
                position_token_counts[pos][tok] += 1
        
        invariant_positions = set()
        for pos, counter in position_token_counts.items():
            most_common, count = counter.most_common(1)[0]
            if count / len(qidx_items) >= threshold:
                invariant_positions.add(pos)
        
        template_positions[qidx] = invariant_positions
    
    return template_positions
```

### Applying the Filters

A token activation is "filtered out" if any of:
1. The token string is in the category's `string_level_template_tokens`
2. The position is in the template_positions for the item's question_index

We report THREE views in the final output:
- **Unfiltered:** all tokens including template
- **String-filtered:** excludes structural tokens like ":" and "Question"
- **Fully-filtered:** excludes structural + position-invariant template tokens

The fully-filtered view is the "content tokens" analysis. Figures default to fully-filtered; supplementary shows all three for transparency.

---

## Step 4: Aggregation Per (Layer, Feature, Subgroup) Tuple

For the final analysis, we aggregate differently depending on the perspective.

### 4a: Token-Level Aggregation (Per Feature, Across All Items)

```python
def aggregate_token_rankings(
    feature_records: list[dict],
    string_template_tokens: set[str],
    template_positions_by_qidx: dict,
    stereotyped_option_contains: dict,  # item_idx -> tokens in stereotyped option
) -> pd.DataFrame:
    """
    For one (layer, feature), aggregate per-token statistics across all items.
    
    Returns DataFrame with one row per unique token, columns:
        token, n_occurrences, n_nonzero, mean_activation, mean_activation_nonzero,
        max_activation, median_activation_nonzero, fraction_firings, fraction_in_stereotyped_option,
        is_template_string, is_identity_term
    """
    token_stats = defaultdict(lambda: {
        "activations": [],
        "nonzero_activations": [],
        "n_in_stereo_option": 0,
        "n_in_nonstereo_option": 0,
    })
    
    for rec in feature_records:
        item_idx = rec["item_idx"]
        qidx = rec["question_index"]
        tokens = rec["tokens"]
        acts = rec["activations"]
        template_pos = template_positions_by_qidx.get(qidx, set())
        stereo_opt_tokens = stereotyped_option_contains.get(item_idx, set())
        
        for pos, (tok, act) in enumerate(zip(tokens, acts)):
            token_stats[tok]["activations"].append(act)
            if act > 0:
                token_stats[tok]["nonzero_activations"].append(act)
                if tok in stereo_opt_tokens:
                    token_stats[tok]["n_in_stereo_option"] += 1
                else:
                    token_stats[tok]["n_in_nonstereo_option"] += 1
    
    rows = []
    for tok, stats in token_stats.items():
        acts = stats["activations"]
        nonzero = stats["nonzero_activations"]
        n_total = len(acts)
        n_nonzero = len(nonzero)
        
        # Classify token
        is_template_string = tok in string_template_tokens
        is_identity_term = is_identity_match(tok)
        
        rows.append({
            "token": tok,
            "n_occurrences": n_total,
            "n_nonzero": n_nonzero,
            "mean_activation": float(np.mean(acts)) if acts else 0.0,
            "mean_activation_nonzero": float(np.mean(nonzero)) if nonzero else 0.0,
            "max_activation": float(max(acts)) if acts else 0.0,
            "median_activation_nonzero": float(np.median(nonzero)) if nonzero else 0.0,
            "fraction_firings": n_nonzero / max(n_total, 1),
            "n_firings_in_stereo_option": stats["n_in_stereo_option"],
            "n_firings_in_nonstereo_option": stats["n_in_nonstereo_option"],
            "fraction_in_stereo_option_when_firing": (
                stats["n_in_stereo_option"] / max(n_nonzero, 1)
            ),
            "is_template_string": is_template_string,
            "is_identity_term": is_identity_term,
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values("mean_activation_nonzero", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


def is_identity_match(token: str) -> bool:
    """Check if the token matches any demographic pattern from C3's catalog."""
    from your_c3_module import DEMOGRAPHIC_PATTERNS
    tok_lower = token.lower().strip().replace("▁", "").replace("Ġ", "")
    for sub_label, spec in DEMOGRAPHIC_PATTERNS.items():
        for pat_str in spec["patterns"]:
            # Simplified match: check if token's raw text is a substring of any pattern keyword
            # For rigorous matching, reconstruct words from BPE tokens first
            if re.search(pat_str, tok_lower, re.IGNORECASE):
                return True
    return False
```

### 4b: Per-Template Aggregation (Feature × question_index)

```python
def aggregate_per_template(
    feature_records: list[dict],
    template_positions_by_qidx: dict,
) -> pd.DataFrame:
    """
    For one (layer, feature), aggregate by question_index.
    
    Shows which BBQ templates the feature activates on most.
    """
    by_qidx = defaultdict(list)
    for rec in feature_records:
        by_qidx[rec["question_index"]].append(rec)
    
    rows = []
    for qidx, recs in by_qidx.items():
        # Use max activation per item (capturing peak firing within the prompt)
        max_acts = [rec["max_activation"] for rec in recs]
        n_fired = sum(1 for ma in max_acts if ma > 0)
        
        rows.append({
            "question_index": qidx,
            "n_items": len(recs),
            "n_items_with_firing": n_fired,
            "fraction_items_with_firing": n_fired / max(len(recs), 1),
            "mean_max_activation": float(np.mean(max_acts)),
            "max_across_items": float(max(max_acts)) if max_acts else 0.0,
            "median_max_activation": float(np.median(max_acts)),
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values("mean_max_activation", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df
```

### 4c: Activation Density Histogram

```python
def compute_activation_density(feature_records: list[dict]) -> dict:
    """
    Build distribution of nonzero activations across all positions across all items.
    
    Returns histogram bins + fraction of total token positions with activation > 0 (density).
    """
    all_activations = []
    for rec in feature_records:
        all_activations.extend(rec["activations"])
    
    all_activations = np.array(all_activations)
    nonzero = all_activations[all_activations > 0]
    
    total_positions = len(all_activations)
    density = len(nonzero) / max(total_positions, 1)
    
    # Compute histogram (log-spaced bins above 0, plus one zero-bin)
    if len(nonzero) > 0:
        min_nz = float(nonzero.min())
        max_nz = float(nonzero.max())
        bin_edges = np.linspace(0, max_nz, 51)  # 50 bins
        counts, _ = np.histogram(nonzero, bins=bin_edges)
    else:
        bin_edges = np.linspace(0, 1, 51)
        counts = np.zeros(50, dtype=int)
    
    return {
        "density": density,
        "n_total_positions": total_positions,
        "n_nonzero_positions": int(len(nonzero)),
        "max_activation": float(all_activations.max()) if len(all_activations) else 0.0,
        "mean_activation_nonzero": float(nonzero.mean()) if len(nonzero) else 0.0,
        "median_activation_nonzero": float(np.median(nonzero)) if len(nonzero) else 0.0,
        "histogram_bin_edges": bin_edges.tolist(),
        "histogram_counts": counts.tolist(),
    }
```

### 4d: Top Activating Examples (for Figures)

```python
def select_top_activating_examples(
    feature_records: list[dict],
    stimuli_by_idx: dict,
    top_n: int = 20,
) -> list[dict]:
    """
    Rank items by max_activation, return top-N with full token activation pattern.
    Used for the "Top Activations" view in figures.
    """
    sorted_recs = sorted(feature_records, key=lambda r: -r["max_activation"])[:top_n]
    
    results = []
    for rec in sorted_recs:
        stim = stimuli_by_idx.get(rec["item_idx"], {})
        results.append({
            "item_idx": rec["item_idx"],
            "max_activation": rec["max_activation"],
            "argmax_position": rec["argmax_position"],
            "argmax_token": rec["argmax_token"],
            "tokens": rec["tokens"],
            "activations": rec["activations"],
            "prompt_preview": stim.get("prompt", "")[:200],
            "stereotyped_groups": rec["stereotyped_groups"],
            "question_polarity": rec["question_polarity"],
            "is_stereotyped_response": rec["is_stereotyped_response"],
        })
    return results
```

---

## Output Files

### `feature_interpretability.parquet`

Per-feature metadata + summary stats. One row per (layer, feature_idx) pair.

```
{run}/C_token_features/feature_interpretability.parquet
```

| Column | Type | Description |
|---|---|---|
| `layer` | int32 | |
| `feature_idx` | int32 | |
| `categories` | string | Comma-separated categories whose subgroups use this feature |
| `subgroups` | string | Comma-separated subgroups using this feature |
| `n_items_processed` | int32 | Items processed for this feature |
| `density` | float32 | Fraction of token positions where feature fires |
| `mean_activation_nonzero` | float32 | |
| `median_activation_nonzero` | float32 | |
| `max_activation` | float32 | |
| `top_activating_token` | string | Token with highest mean_activation_nonzero (unfiltered) |
| `top_activating_token_filtered` | string | Same, fully-filtered (content tokens only) |
| `top_positive_logit_token` | string | Token most promoted in output via unembedding decomposition |
| `top_negative_logit_token` | string | Token most suppressed |

### `logit_effects.parquet`

Logit-effect decomposition, top-10 positive and top-10 negative tokens per feature.

```
{run}/C_token_features/logit_effects.parquet
```

| Column | Type |
|---|---|
| `layer` | int32 |
| `feature_idx` | int32 |
| `direction` | string ("positive" / "negative") |
| `rank` | int32 |
| `token_id` | int32 |
| `token_str` | string |
| `logit_contribution` | float32 |

### `token_rankings/{layer}_{feature_idx}.parquet`

Per-feature token ranking.

```
{run}/C_token_features/token_rankings/L{layer:02d}_F{feature_idx}.parquet
```

Columns: `rank`, `token`, `n_occurrences`, `n_nonzero`, `mean_activation`, `mean_activation_nonzero`, `max_activation`, `median_activation_nonzero`, `fraction_firings`, `n_firings_in_stereo_option`, `n_firings_in_nonstereo_option`, `fraction_in_stereo_option_when_firing`, `is_template_string`, `is_identity_term`.

### `per_template_rankings/{layer}_{feature_idx}.parquet`

Per-feature per-template activation summary.

```
{run}/C_token_features/per_template_rankings/L{layer:02d}_F{feature_idx}.parquet
```

Columns: `rank`, `question_index`, `n_items`, `n_items_with_firing`, `fraction_items_with_firing`, `mean_max_activation`, `max_across_items`, `median_max_activation`.

### `top_activating_examples/{layer}_{feature_idx}.json`

Top-20 items with max_activation, tokens, and per-token activations. Used for figures.

```
{run}/C_token_features/top_activating_examples/L{layer:02d}_F{feature_idx}.json
```

### `activation_densities.json`

```json
{
  "L14_F45021": {
    "density": 0.0023,
    "n_total_positions": 2435789,
    "n_nonzero_positions": 5603,
    "max_activation": 8.41,
    "mean_activation_nonzero": 2.14,
    "median_activation_nonzero": 1.87,
    "histogram_bin_edges": [0.0, 0.17, 0.34, ...],
    "histogram_counts": [2843, 1211, 687, ...]
  },
  ...
}
```

### `template_filters.json`

Documents the filtering applied.

```json
{
  "string_level_threshold": 0.90,
  "position_level_threshold": 0.80,
  "per_category": {
    "so": {
      "string_level_template_tokens": [":", "Question", "Answer", ...],
      "n_template_tokens": 37
    },
    ...
  },
  "per_question_index_positions": {
    "so/12": {"n_template_positions": 24, "positions": [0, 1, 2, 3, ...]},
    ...
  }
}
```

### `per_item_per_token_activations.parquet` (OPTIONAL, via `--save_full_per_token`)

Full per-item per-token activations. LARGE.

Columns: `layer`, `feature_idx`, `item_idx`, `position`, `token`, `activation`.

Default: NOT saved. Enable with flag. Estimated size: ~12K items × ~300 tokens × ~200 features × ~40 bytes ≈ 30 GB raw → ~3 GB parquet with compression.

---

## Figures

All figures use Wong colorblind-safe palette.

### `fig_feature_card_{layer}_{feature_idx}.png` (one per feature)

Neuronpedia-style per-feature card. Large figure with multiple panels.

**Panel A (top-left): Logit effect decomposition.** Two columns: negative logits (left, red) and positive logits (right, green). Each shows top-10 tokens with their logit contribution values, ranked.

**Panel B (top-right): Activation density histogram.** Two histograms in small multiples:
- Top: nonzero activations only (orange), binned
- Bottom: zero + nonzero, showing overall sparsity (green for nonzero bars above baseline)
- Annotation: "Density: {density:.3%}" e.g. "0.162%"

**Panel C (middle): Top token rankings.** Horizontal bar chart of top-20 tokens by mean_activation_nonzero from fully-filtered ranking. Color-coded:
- Identity terms (matched by DEMOGRAPHIC_PATTERNS): Wong blue (#0072B2)
- Tokens appearing in stereotyped option ≥50% of firings: Wong orange (#E69F00)
- Other content tokens: gray
Legend in corner.

**Panel D (bottom): Top activating examples.** 3-5 prompt snippets, tokenized, with activating tokens highlighted via background shading proportional to activation. Shows the "Top Activations" snippets from neuronpedia.

Title: "Feature L{layer}_F{feature_idx} — used by {cat/sub list}"

### `fig_token_rankings_{category}.png`

Grid of small token-ranking bar charts, one per (subgroup, feature) in the category. Keeps analyses visually aligned for side-by-side comparison.

Each small panel:
- Horizontal bar chart: top-10 tokens by mean_activation_nonzero (fully-filtered)
- Subtitle: "{subgroup}: L{layer}_F{feature_idx}"
- Color coding as in feature card Panel C

### `fig_per_template_heatmap_{category}.png`

Heatmap:
- Rows: features used by subgroups in this category, labeled `{subgroup}: L{layer}_F{feature_idx}`
- Columns: question_index values (sorted)
- Color: mean_max_activation (YlOrRd sequential)
- Annotated cells
- Title: "Per-template feature activations — {category}"

Shows which templates each feature responds to. If fragmentation features are truly subgroup-specific, different features within a category should light up different templates.

### `fig_identity_token_specificity.png`

For each (subgroup, feature), compute: among top-20 nonzero tokens, what fraction are identity terms matching the subgroup's labels?

Single bar chart across all (subgroup, feature) pairs. X = subgroup/feature label. Y = fraction of top-20 tokens that are identity terms. Bars colored by category.

Subgroups whose features fire predominantly on their own identity terms provide strong face-validity evidence for the interpretation.

---

## Output Structure

```
{run}/C_token_features/
├── feature_interpretability.parquet              # Summary table, one row per feature
├── logit_effects.parquet                          # Unembedding decomposition
├── activation_densities.json                      # Histograms per feature
├── template_filters.json                          # What was filtered
├── feature_manifest.json                          # Which subgroups use which features
├── token_rankings/
│   └── L{layer:02d}_F{feature_idx}.parquet       # Per-feature token rankings
├── per_template_rankings/
│   └── L{layer:02d}_F{feature_idx}.parquet       # Per-feature per-template summary
├── top_activating_examples/
│   └── L{layer:02d}_F{feature_idx}.json          # Top 20 items with full detail
├── per_item_per_token_activations.parquet         # OPTIONAL, with --save_full_per_token
└── figures/
    ├── fig_feature_card_L{layer:02d}_F{feature_idx}.png/.pdf   # Per-feature neuronpedia-style
    ├── fig_token_rankings_{category}.png/.pdf
    ├── fig_per_template_heatmap_{category}.png/.pdf
    └── fig_identity_token_specificity.png/.pdf
```

---

## Resume Safety

**Per-category checkpoints.** Each category's extraction phase produces an intermediate pickle/parquet that can be reloaded. If C4 crashes mid-extraction, restart at the next unprocessed category.

**Logit effects** are cheap — re-computed on every run (no checkpoint needed).

**Per-feature aggregation and figures** run after all extractions are done. If aggregation crashes, extractions are preserved.

Atomic writes throughout.

---

## Compute Estimate (M4 Max MPS)

- Forward pass per item with multi-layer hooks: ~0.4s
- SAE encoding per item per layer: ~0.01-0.02s (matmul on MPS)
- Token aggregation: fast (CPU, pandas)

Per category (~1000 ambig items, ~5-10 layers):
- Forward passes: 1000 × 0.4s = ~7 min
- SAE encodings: 1000 × 8 × 0.015 = ~2 min
- Total: ~9 min per category

All 9 categories: ~80 min ≈ **1.5 hours total**

Plus figure generation: ~30 min if we make per-feature cards for ~200 features.

**Grand total: ~2 hours on MPS.**

---

## Assumptions Summary

| # | Decision | Value |
|---|---|---|
| 1 | Features analyzed | All features in each subgroup's optimal steering vector, capped at max_features_per_subgroup=10 |
| 2 | Processing strategy | Item-centric: one multi-layer forward pass per item, extract all target features at once |
| 3 | Items analyzed | All ambig items per category (not just items targeting the subgroup) |
| 4 | Logit decomposition | W_dec[f] · W_U for each feature; top-10 positive/negative tokens |
| 5 | Template filter level 1 | Tokens in >90% of category's items (string-level) |
| 6 | Template filter level 2 | Position-invariant positions per question_index (>80%) |
| 7 | Three views reported | Unfiltered, string-filtered, fully-filtered |
| 8 | Stereotype annotation | Token appearing in BBQ's `stereotyped_option` text for the item |
| 9 | Identity annotation | Token matches DEMOGRAPHIC_PATTERNS from C3 |
| 10 | Per-template aggregation | Added — which question_index values activate the feature most |
| 11 | Top activating examples | Top 20 items per feature saved |
| 12 | Full per-item-per-token | Optional only (via `--save_full_per_token` flag) |
| 13 | Feature card figures | Neuronpedia-style: logit effects + density + token rankings + example prompts |
| 14 | Memory management | Encode full 131K features through SAE, extract target subset, immediately free |

---

## Test Command

```bash
# Smoke test: single category, limited features
python scripts/C4_token_features.py \
    --run_dir runs/llama-3.1-8b_2026-04-15/ \
    --categories so \
    --max_features_per_subgroup 3

# Verify outputs
python -c "
import pandas as pd
import json

summary = pd.read_parquet('runs/llama-3.1-8b_2026-04-15/C_token_features/feature_interpretability.parquet')
print(f'Features characterized: {len(summary)}')
print(summary[['layer', 'feature_idx', 'subgroups', 'density', 'top_activating_token_filtered']].head(10))

# Logit effects
lf = pd.read_parquet('runs/llama-3.1-8b_2026-04-15/C_token_features/logit_effects.parquet')
print(f'\\nLogit effect rows: {len(lf)}')

# Sample: top positive logits for first feature
first_key = lf[['layer', 'feature_idx']].drop_duplicates().iloc[0]
first_pos = lf[
    (lf['layer'] == first_key['layer']) &
    (lf['feature_idx'] == first_key['feature_idx']) &
    (lf['direction'] == 'positive')
].sort_values('rank')
print(f'\\nTop positive-logit tokens for L{first_key[\"layer\"]}_F{first_key[\"feature_idx\"]}:')
print(first_pos[['rank', 'token_str', 'logit_contribution']].head())
"
```