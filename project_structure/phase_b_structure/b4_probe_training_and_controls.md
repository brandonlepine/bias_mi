# B4: Probe Training + Controls — Full Implementation Specification

## Purpose

Train linear probes on saved hidden states to test what's linearly decodable from the model's representations at each layer. Controls establish whether the probes find genuine identity-specific signal vs. surface artifacts (question templates, prompt structure). Also runs probes on SAE-encoded features to verify that B1/B2's ranked features capture the same identity structure as raw hidden states.

## Invocation

```bash
python scripts/B4_probes.py --run_dir runs/llama-3.1-8b_2026-04-15/

# Subset of layers for speed (default: all layers)
python scripts/B4_probes.py --run_dir runs/llama-3.1-8b_2026-04-15/ --layers 0,4,8,12,16,20,24,28,31

# Override permutation trials
python scripts/B4_probes.py --run_dir runs/llama-3.1-8b_2026-04-15/ --n_permutations 20

# Override minimum positives/negatives for binary probes
python scripts/B4_probes.py --run_dir runs/llama-3.1-8b_2026-04-15/ --min_n_per_class 20

# Skip SAE-based probes (raw-only)
python scripts/B4_probes.py --run_dir runs/llama-3.1-8b_2026-04-15/ --skip_sae_probes

# Skip figures
python scripts/B4_probes.py --run_dir runs/llama-3.1-8b_2026-04-15/ --skip_figures
```

Reads `categories`, `n_layers`, `hidden_dim` from config.json.

---

## Input

- Activations: `{run}/A_extraction/activations/{category}/item_*.npz`
- Metadata parquet: `{run}/A_extraction/metadata.parquet`
- Stimuli: `{run}/A_extraction/stimuli/{category}.json` (for `question_index`, `answer_roles`, etc.)
- SAE encodings: `{run}/A_extraction/sae_encoding/layer_{NN}.parquet` (for SAE-based probes)
- Ranked features: `{run}/B_feature_ranking/ranked_features.parquet` (for SAE top-K selection)

## Dependencies

- `scikit-learn` (LogisticRegression, PCA, GroupKFold, balanced_accuracy_score)
- `numpy`
- `pandas`, `pyarrow`
- `matplotlib`

---

## Critical Methodological Choice: Group-Aware Cross-Validation

**All probes use `GroupKFold` with `groups = question_index`, NOT `StratifiedKFold`.**

### Why

BBQ organizes items by template (each `question_index` corresponds to a distinct template used multiple times with different identities and context variations). A single template produces many items — the same question phrasing with different names/answers/context.

With naive stratified CV, the same template can appear in both train and test folds. A probe can then memorize template-specific features (word patterns, phrase structures) and inflate its apparent accuracy without actually learning identity representations.

Group-aware CV ensures that no `question_index` value appears in both train and test folds. If the probe succeeds, it's generalizing across templates — evidence of identity-specific (not template-specific) learning.

### Implementation

```python
from sklearn.model_selection import GroupKFold

def group_aware_cv_splits(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5):
    """
    Produce CV splits where no group appears in both train and test.
    
    groups: array of question_index values, one per item.
    """
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(X, y, groups))
```

**Note:** `GroupKFold` does NOT guarantee stratification. With severe class imbalance, some folds may have zero positives. We handle this by:
1. Requiring `min_n_per_class = 20` overall (configurable)
2. Checking each fold for at least one positive and one negative before training
3. Skipping folds that fail this check, and reporting the number of folds actually used

```python
def safe_cv_splits(X, y, groups, n_splits=5, min_class_in_fold=1):
    splits = []
    skipped = 0
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(X, y, groups):
        # Check that both classes appear in both train and test
        y_train = y[train_idx]
        y_test = y[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            skipped += 1
            continue
        # For binary: check minimum positives and negatives in each split
        if set(y_train.tolist()) != set(y_test.tolist()):
            skipped += 1
            continue
        splits.append((train_idx, test_idx))
    return splits, skipped
```

---

## Probe Architecture

Single shared architecture for all probes: PCA-50 → L2-regularized logistic regression.

```python
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score

SEED = 42
N_COMPONENTS = 50
N_FOLDS = 5
LR_C = 1.0
LR_MAX_ITER = 1000

def train_probe(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int = N_FOLDS,
    class_weight: str | None = "balanced",
    seed: int = SEED,
) -> dict:
    """
    Run group-aware cross-validated probe.
    
    Args:
        X: (n_items, n_features) feature matrix (raw hidden states, SAE features, etc.)
        y: (n_items,) labels (strings or ints)
        groups: (n_items,) question_index values for group-aware CV
        n_folds: target number of folds (may be reduced if some folds are degenerate)
        class_weight: "balanced" for imbalanced binary probes, None for multiclass
        seed: random seed
    
    Returns:
        Dict with accuracy (mean, std, per-fold), balanced accuracy, n_classes, etc.
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    # Dimensionality check — PCA components limited by smaller of (n_items, n_features, N_COMPONENTS)
    n_components = min(N_COMPONENTS, X.shape[0] - 1, X.shape[1])
    
    # Get CV splits (skip degenerate folds)
    splits, n_skipped = safe_cv_splits(X, y_enc, groups, n_splits=n_folds)
    if not splits:
        return {
            "mean_accuracy": None,
            "std_accuracy": None,
            "mean_balanced_accuracy": None,
            "std_balanced_accuracy": None,
            "n_folds_used": 0,
            "n_folds_skipped": n_skipped,
            "n_items": len(y),
            "n_classes": len(le.classes_),
            "classes": list(le.classes_),
            "n_components": n_components,
            "status": "no_valid_folds",
        }
    
    accs = []
    balanced_accs = []
    
    for train_idx, test_idx in splits:
        # Fit PCA on training fold only — prevents data leakage
        pca = PCA(n_components=n_components, random_state=seed)
        X_train = pca.fit_transform(X[train_idx])
        X_test = pca.transform(X[test_idx])
        
        clf = LogisticRegression(
            C=LR_C,
            max_iter=LR_MAX_ITER,
            solver="lbfgs",
            multi_class="multinomial" if len(le.classes_) > 2 else "auto",
            class_weight=class_weight,
            random_state=seed,
        )
        clf.fit(X_train, y_enc[train_idx])
        y_pred = clf.predict(X_test)
        
        accs.append(float((y_pred == y_enc[test_idx]).mean()))
        balanced_accs.append(float(balanced_accuracy_score(y_enc[test_idx], y_pred)))
    
    return {
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_balanced_accuracy": float(np.mean(balanced_accs)),
        "std_balanced_accuracy": float(np.std(balanced_accs)),
        "per_fold_accuracy": accs,
        "per_fold_balanced_accuracy": balanced_accs,
        "n_folds_used": len(accs),
        "n_folds_skipped": n_skipped,
        "n_items": len(y),
        "n_classes": len(le.classes_),
        "classes": list(le.classes_),
        "n_components": n_components,
        "status": "ok",
    }
```

**PCA-50:** Standard dimensionality reduction for high-dim probing. Captures ≥95% variance while reducing dimensionality 80x (from 4096 to 50). Fit within each CV fold to prevent leakage.

**L2-regularized logistic regression:** The mech interp convention for classification probes. C=1.0 (sklearn default). Binary and multiclass both handled by same pipeline.

**Balanced accuracy:** Average of per-class recalls. Robust to class imbalance. Used as primary metric for binary one-vs-rest probes. For multiclass probes where classes are roughly balanced, raw accuracy and balanced accuracy are nearly identical.

**`class_weight="balanced"` for binary probes** corrects for imbalanced positive/negative counts automatically.

---

## Probes

### Probe 1: Subgroup Classification, Multi-class (Within-Category)

At each layer, within each category, predict which subgroup an item targets.

**Labels:** `stereotyped_groups[0]` (first element of the list). 

**Critical restriction:** Use ONLY single-group items (`n_target_groups == 1`). Multi-group items have ambiguous multi-class labels. This is necessary for multi-class classification to be well-defined. Items excluded here are still used in the binary per-subgroup probes.

```python
# Per category, per layer:
single_group_items = meta_df[
    (meta_df["category"] == cat) &
    (meta_df["n_target_groups"] == 1) &
    (meta_df["context_condition"] == "ambig")
]

X = np.stack([load_hidden_state(idx, layer) for idx in single_group_items["item_idx"]])
y = np.array([sgroup[0] for sgroup in single_group_items["stereotyped_groups"]])
groups = single_group_items["question_index"].values

result = train_probe(X, y, groups, class_weight=None)
```

**Context filter:** Ambig items only. This matches B1's primary comparison and keeps probe semantics consistent.

**Minimum requirements:**
- At least 2 subgroups in the category after single-group filter
- Each subgroup has at least 20 items total
- At least 5 group-aware CV folds usable

### Probe 2: Subgroup Detection, Binary One-vs-Rest (Within-Category)

For each subgroup S within a category, at each layer, predict "does this item target S?" (binary).

**Key advantage:** Multi-group items naturally handled — an item targeting both "disabled" and "physically disabled" is positive for BOTH binary probes. Matches B1's logic of items contributing to all targeted groups.

```python
# Per category, per subgroup, per layer:
cat_items = meta_df[
    (meta_df["category"] == cat) &
    (meta_df["context_condition"] == "ambig")
]

X = np.stack([load_hidden_state(idx, layer) for idx in cat_items["item_idx"]])
y = np.array([int(subgroup in gs) for gs in cat_items["stereotyped_groups"]])
groups = cat_items["question_index"].values

# Class balance check
n_pos = int((y == 1).sum())
n_neg = int((y == 0).sum())
if n_pos < 20 or n_neg < 20:
    continue  # skip

result = train_probe(X, y, groups, class_weight="balanced")
```

**This is the primary identity probe** because it handles multi-group items cleanly and aligns with the "subgroup-first" orientation of the whole pipeline.

### Probe 3: Stereotyped Response Binary (for Cross-Category/Cross-Subgroup Generalization)

Predict `is_stereotyped_response` (T/F) from hidden state. Ambig items only.

Used for both:
- **Cross-category generalization**: train on one category's items, test on another's
- **Within-category cross-subgroup generalization**: train on subgroup A's items, test on subgroup B's items within same category

```python
# Per-category, per-layer main probe (for baseline accuracy):
cat_ambig = meta_df[
    (meta_df["category"] == cat) &
    (meta_df["context_condition"] == "ambig")
]
X = np.stack([load_hidden_state(idx, layer) for idx in cat_ambig["item_idx"]])
y = cat_ambig["is_stereotyped_response"].values.astype(int)
groups = cat_ambig["question_index"].values

n_pos = int(y.sum())
n_neg = int((y == 0).sum())
if n_pos < 20 or n_neg < 20:
    continue

result = train_probe(X, y, groups, class_weight="balanced")
```

### Control A (Permutation Baseline)

For each probe, repeat training with randomly shuffled labels. Compute selectivity = real accuracy − permutation mean.

```python
def permutation_baseline(
    X, y, groups,
    n_permutations: int = 10,
    class_weight: str | None = "balanced",
    seed: int = SEED,
) -> dict:
    """Run n_permutations probes with shuffled labels."""
    perm_accs = []
    perm_balanced_accs = []
    
    for trial in range(n_permutations):
        # Derive distinct seed per trial
        rng = np.random.default_rng(seed + 1000 + trial)
        y_perm = rng.permutation(y)
        
        result = train_probe(X, y_perm, groups, class_weight=class_weight, seed=seed + trial)
        
        if result["status"] == "ok":
            perm_accs.append(result["mean_accuracy"])
            perm_balanced_accs.append(result["mean_balanced_accuracy"])
    
    if not perm_accs:
        return {"mean": None, "std": None, "n_trials": 0}
    
    return {
        "mean_accuracy": float(np.mean(perm_accs)),
        "std_accuracy": float(np.std(perm_accs)),
        "mean_balanced_accuracy": float(np.mean(perm_balanced_accs)),
        "std_balanced_accuracy": float(np.std(perm_balanced_accs)),
        "n_trials": len(perm_accs),
        "all_accuracies": perm_accs,
    }
```

**Selectivity:**
```python
selectivity = real["mean_balanced_accuracy"] - perm["mean_balanced_accuracy"]
```

Use balanced accuracy for selectivity on binary probes (where class imbalance could inflate raw accuracy).

### Control B1 (Context Condition Probe, Secondary Baseline)

Predict ambig vs disambig. Weak control — context_condition is encoded by prompt length/structure and is trivially decodable. Reported as a baseline for decodability of prompt structural properties.

```python
cat_items = meta_df[meta_df["category"] == cat]  # ALL items, both conditions

X = np.stack([load_hidden_state(idx, layer) for idx in cat_items["item_idx"]])
y = cat_items["context_condition"].values  # "ambig" / "disambig"
groups = cat_items["question_index"].values

result = train_probe(X, y, groups, class_weight="balanced")
```

### Control B2 (Template ID Probe, Primary Structural Control)

Predict the `question_index` — a multi-class problem where each class is a template.

**Important subtlety:** Because question_index is ALSO the CV group variable, training a probe to predict question_index using `GroupKFold(groups=question_index)` means the test fold contains templates that weren't in training — so the probe should achieve NEAR-ZERO accuracy by construction. Holdout templates literally weren't seen.

This makes the group-aware template probe a **negative control**: if it somehow achieves non-chance accuracy, something is wrong with the CV.

**Instead, the informative template probe uses STRATIFIED CV (not group-aware):**

```python
from sklearn.model_selection import StratifiedKFold

def template_probe_stratified(X, y, seed=SEED):
    """
    Probe for question_index using stratified (non-group-aware) CV.
    Tests how much of the representation is template-specific when templates are shared
    between train and test.
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    # ... standard PCA + LR pipeline
```

**Interpretation:**
- High template probe accuracy + high identity probe accuracy (group-aware): representation encodes BOTH template and identity. The group-aware CV has already controlled for template-driven inflation of the identity probe — so a high identity score despite this control is meaningful.
- High template probe accuracy + low identity probe accuracy: representation is dominated by template structure; identity isn't really there.
- Low template probe accuracy at late layers: model has abstracted past template structure.

**This serves as the primary structural control** (stronger than context_condition) because it directly tests the "is this just template-memorization?" concern.

### Control D1 (Cross-Category Generalization)

Train binary `is_stereotyped_response` probe on category A, evaluate on category B. All layers.

```python
for train_cat in categories:
    for test_cat in categories:
        for layer in range(n_layers):
            # Training data
            train_items = meta_df[
                (meta_df["category"] == train_cat) &
                (meta_df["context_condition"] == "ambig")
            ]
            X_train = load_hidden_states(train_items["item_idx"], layer)
            y_train = train_items["is_stereotyped_response"].values.astype(int)
            groups_train = train_items["question_index"].values
            
            # Test data
            test_items = meta_df[
                (meta_df["category"] == test_cat) &
                (meta_df["context_condition"] == "ambig")
            ]
            X_test = load_hidden_states(test_items["item_idx"], layer)
            y_test = test_items["is_stereotyped_response"].values.astype(int)
            
            # Category-specific PCA: fit on train_cat only
            pca = PCA(n_components=N_COMPONENTS, random_state=SEED)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)
            
            # For within-category (train_cat == test_cat), use group-aware CV
            if train_cat == test_cat:
                result = train_probe(X_train, y_train, groups_train, class_weight="balanced")
                accuracy = result["mean_balanced_accuracy"]
            else:
                # Cross-category: fit on all train, evaluate on all test
                clf = LogisticRegression(C=LR_C, max_iter=LR_MAX_ITER,
                                          class_weight="balanced", random_state=SEED)
                clf.fit(X_train_pca, y_train)
                y_pred = clf.predict(X_test_pca)
                accuracy = balanced_accuracy_score(y_test, y_pred)
            
            cross_cat_matrix[train_cat][test_cat][layer] = accuracy
```

**Diagonal** (train_cat == test_cat): uses group-aware CV for the within-category case. This gives the "within-category" baseline accuracy fairly.

**Off-diagonal** (train_cat != test_cat): single train-test split, no CV. The train and test come from completely different categories, so there's no leakage concern. PCA fit on train_cat only.

**Output dimension:** 9 × 9 × 32 tensor. Saved as parquet.

### Control D2 (Within-Category Cross-Subgroup Generalization)

Within each category, train `is_stereotyped_response` probe on subgroup A's items, test on subgroup B's items.

```python
for cat in categories:
    # All ambig items in category, with subgroup assignments (primary = [0])
    cat_items = meta_df[
        (meta_df["category"] == cat) &
        (meta_df["context_condition"] == "ambig") &
        (meta_df["n_target_groups"] == 1)  # single-group items only for clean subgroup assignment
    ]
    
    for train_sub in subgroups_in(cat):
        train_sub_items = cat_items[
            cat_items["stereotyped_groups"].apply(lambda gs: gs[0] == train_sub)
        ]
        
        for test_sub in subgroups_in(cat):
            test_sub_items = cat_items[
                cat_items["stereotyped_groups"].apply(lambda gs: gs[0] == test_sub)
            ]
            
            # Require min_n on each
            if len(train_sub_items) < 20 or len(test_sub_items) < 20:
                continue
            
            for layer in range(n_layers):
                X_train = load_hidden_states(train_sub_items["item_idx"], layer)
                y_train = train_sub_items["is_stereotyped_response"].values.astype(int)
                X_test = load_hidden_states(test_sub_items["item_idx"], layer)
                y_test = test_sub_items["is_stereotyped_response"].values.astype(int)
                
                # Check class balance
                if len(set(y_train)) < 2 or len(set(y_test)) < 2:
                    continue
                
                pca = PCA(n_components=N_COMPONENTS, random_state=SEED)
                X_train_pca = pca.fit_transform(X_train)
                X_test_pca = pca.transform(X_test)
                
                clf = LogisticRegression(C=LR_C, max_iter=LR_MAX_ITER,
                                          class_weight="balanced", random_state=SEED)
                clf.fit(X_train_pca, y_train)
                y_pred = clf.predict(X_test_pca)
                
                within_cat_matrix[cat][train_sub][test_sub][layer] = balanced_accuracy_score(y_test, y_pred)
```

**Prediction from fragmentation:** Anti-correlated subgroup pairs from B3 should show poor generalization (near-chance). Aligned pairs should generalize well. This connects probe results directly to the cosine geometry.

**Why single-group items only:** For clean subgroup assignment. Multi-group items create ambiguity about which subgroup "owns" them.

### Probe 4: SAE-Based Subgroup Detection

Robustness check: if the SAE decomposition captures identity structure, probes on SAE features should match probes on raw hidden states.

**Feature selection:** Top-50 pro-bias features per subgroup from B2's ranked_features.parquet. K=50 fixed.

```python
# For each subgroup S:
ranked = pd.read_parquet(run_dir / "B_feature_ranking" / "ranked_features.parquet")
top_features = ranked[
    (ranked["category"] == cat) &
    (ranked["subgroup"] == sub) &
    (ranked["direction"] == "pro_bias") &
    (ranked["rank"] <= 50)
]
# top_features has columns: feature_idx, layer

# For each item, build feature vector: 50-dim activation
for layer_group in top_features["layer"].unique():
    # Load the corresponding SAE encoding parquet
    sae_df = pd.read_parquet(run_dir / "A_extraction" / "sae_encoding" / f"layer_{layer_group:02d}.parquet")
    ...

# Build X matrix of shape (n_items, 50) where each column is an activation value
# Run group-aware binary probe: does this item target S?
```

**Implementation detail:** Features span multiple layers. For each item, we build a 50-dim vector by looking up the feature's activation at its own layer (not one fixed layer). Items where a feature has zero activation get 0 in that column.

```python
def build_sae_feature_matrix(
    items: pd.DataFrame,
    top_features: pd.DataFrame,
    run_dir: Path,
) -> np.ndarray:
    """
    Build (n_items, n_features) matrix where each column is the activation of a feature
    at its own layer, per item.
    """
    n_items = len(items)
    n_features = len(top_features)
    X = np.zeros((n_items, n_features), dtype=np.float32)
    
    item_idx_to_pos = {idx: i for i, idx in enumerate(items["item_idx"])}
    
    # Group features by layer for efficient parquet loading
    for layer, layer_features in top_features.groupby("layer"):
        parquet_path = run_dir / "A_extraction" / "sae_encoding" / f"layer_{int(layer):02d}.parquet"
        if not parquet_path.exists():
            continue
        sae_df = pd.read_parquet(parquet_path)
        
        # For each feature at this layer
        for _, feat_row in layer_features.iterrows():
            feat_idx = int(feat_row["feature_idx"])
            col_idx = top_features.index.get_loc(feat_row.name)
            
            # Get this feature's activations across items
            feat_activations = sae_df[
                sae_df["feature_idx"] == feat_idx
            ].set_index("item_idx")["activation_value"]
            
            # Assign to X
            for item_idx, activation in feat_activations.items():
                if item_idx in item_idx_to_pos:
                    X[item_idx_to_pos[item_idx], col_idx] = activation
    
    return X
```

**Probe:** Same binary subgroup-detection probe as Probe 2, but on this 50-dim SAE feature matrix instead of raw hidden states. Use group-aware CV with question_index.

**For SAE probes, PCA is NOT needed** since the feature space is already 50-dim (smaller than n_items).

```python
result = train_probe(X_sae, y, groups, class_weight="balanced")
# train_probe will try n_components = min(50, n_items-1, 50) = 50
# Effectively LR on the raw 50-dim feature space
```

**Interpretation:**
- SAE probe balanced accuracy ≈ raw-hidden-state probe balanced accuracy: SAE features preserve identity information
- SAE probe << raw probe: SAE features lose identity information (many identity-relevant dimensions live outside the top-50)
- SAE probe > raw probe: feature selection concentrates identity signal (unlikely but possible)

---

## Script Structure

```python
def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = load_config(run_dir)
    
    categories = config["categories"]
    n_layers = config["n_layers"]
    layers = parse_layers(args.layers, n_layers) if args.layers else list(range(n_layers))
    
    # Load metadata once
    meta_df = load_metadata(run_dir)
    
    # Pre-load hidden states for all categories (big memory but OK on 128GB)
    # Alternative: load one category at a time for lower memory footprint
    hs_cache = {}  # {(cat, layer): np.ndarray (n_items, hidden_dim)}
    idx_cache = {}  # {cat: list of item_idxs in hs_cache order}
    
    all_results = []  # flat records for probe_results.parquet
    cross_cat_records = []
    within_cat_records = []
    
    # Run each probe type
    for cat in categories:
        log(f"\n{'='*60}")
        log(f"Probing category: {cat}")
        log(f"{'='*60}")
        
        # Load category activations into hs_cache
        load_category_activations(cat, run_dir, meta_df, hs_cache, idx_cache, n_layers)
        
        for layer in layers:
            # Probe 1: multi-class subgroup
            result = probe_multiclass_subgroup(cat, layer, hs_cache, idx_cache, meta_df)
            if result: all_results.append(result)
            
            # Probe 2: binary subgroup detection (per-subgroup)
            for sub in subgroups_in(cat, meta_df):
                result = probe_binary_subgroup(cat, sub, layer, hs_cache, idx_cache, meta_df)
                if result: all_results.append(result)
            
            # Probe 3: stereotyped response binary
            result = probe_stereotyped_response(cat, layer, hs_cache, idx_cache, meta_df)
            if result: all_results.append(result)
            
            # Control B1: context condition
            result = probe_context_condition(cat, layer, hs_cache, idx_cache, meta_df)
            if result: all_results.append(result)
            
            # Control B2: template ID (stratified, not group-aware)
            result = probe_template_id(cat, layer, hs_cache, idx_cache, meta_df)
            if result: all_results.append(result)
            
            # SAE probes for binary subgroup detection
            if not args.skip_sae_probes:
                for sub in subgroups_in(cat, meta_df):
                    result = probe_sae_binary_subgroup(cat, sub, layer, run_dir, meta_df)
                    if result: all_results.append(result)
        
        # Control D2: within-category cross-subgroup
        for layer in layers:
            records = probe_within_cat_cross_subgroup(cat, layer, hs_cache, idx_cache, meta_df)
            within_cat_records.extend(records)
        
        # Clear hs_cache for this category before loading next
        for key in list(hs_cache.keys()):
            if key[0] == cat:
                del hs_cache[key]
    
    # Control D1: cross-category (needs all categories loaded — process layer by layer)
    for layer in layers:
        records = probe_cross_category(layer, meta_df, run_dir, categories)
        cross_cat_records.extend(records)
    
    # Save outputs
    save_probe_results(run_dir, all_results)
    save_cross_cat_results(run_dir, cross_cat_records)
    save_within_cat_results(run_dir, within_cat_records)
    save_summary(run_dir, config, args)
    
    if not args.skip_figures:
        generate_figures(run_dir)
```

---

## Output Files

### `probe_results.parquet`

Primary flat output for all standard probes.

```
{run}/B_probes/probe_results.parquet
```

| Column | Type | Description |
|---|---|---|
| `probe_type` | string | `subgroup_multiclass`, `subgroup_binary`, `stereotyped_response_binary`, `context_condition`, `template_id`, `sae_subgroup_binary` |
| `category` | string | Category short name |
| `subgroup` | string or null | Subgroup (null for category-level probes) |
| `layer` | int32 | Layer index |
| `n_items` | int32 | Total items |
| `n_classes` | int32 | Number of classes |
| `n_folds_used` | int32 | Actual CV folds completed |
| `mean_accuracy` | float32 | Mean CV accuracy |
| `std_accuracy` | float32 | Std CV accuracy |
| `mean_balanced_accuracy` | float32 | Mean CV balanced accuracy |
| `std_balanced_accuracy` | float32 | Std CV balanced accuracy |
| `permutation_mean_accuracy` | float32 | Permutation baseline accuracy |
| `permutation_std_accuracy` | float32 | Permutation baseline std |
| `permutation_mean_balanced_accuracy` | float32 | Permutation balanced baseline |
| `selectivity` | float32 | Real balanced accuracy − permutation balanced accuracy |
| `status` | string | `ok`, `insufficient_data`, `no_valid_folds` |

### `cross_category_generalization.parquet`

```
{run}/B_probes/cross_category_generalization.parquet
```

| Column | Type | Description |
|---|---|---|
| `train_category` | string |
| `test_category` | string |
| `layer` | int32 |
| `balanced_accuracy` | float32 |
| `n_train` | int32 |
| `n_test` | int32 |
| `is_within_category` | bool |

### `within_category_cross_subgroup.parquet`

```
{run}/B_probes/within_category_cross_subgroup.parquet
```

| Column | Type | Description |
|---|---|---|
| `category` | string |
| `train_subgroup` | string |
| `test_subgroup` | string |
| `layer` | int32 |
| `balanced_accuracy` | float32 |
| `n_train` | int32 |
| `n_test` | int32 |
| `is_same_subgroup` | bool |

### `probes_summary.json`

Top-level summary of the B4 run.

```json
{
  "config": {
    "n_components": 50,
    "n_folds": 5,
    "cv_method": "GroupKFold(question_index)",
    "template_probe_cv_method": "StratifiedKFold (non-group-aware)",
    "n_permutations": 10,
    "min_n_per_class": 20,
    "lr_C": 1.0,
    "random_seed": 42
  },
  "probes_run": {
    "subgroup_multiclass": 288,       # e.g., 9 categories × 32 layers
    "subgroup_binary": 1280,           # e.g., 40 subgroups × 32 layers
    "stereotyped_response_binary": 288,
    "context_condition": 288,
    "template_id": 288,
    "sae_subgroup_binary": 1280
  },
  "peak_selectivity_per_category": {
    "so": {"peak_layer": 18, "peak_selectivity": 0.47, "probe_type": "subgroup_multiclass"},
    ...
  },
  "peak_binary_subgroup_per_subgroup": {
    "so/gay": {"peak_layer": 16, "peak_balanced_accuracy": 0.82},
    ...
  },
  "runtime_seconds": 2847.0
}
```

---

## Figures

All figures use Wong colorblind-safe palette, distinct markers per subgroup.

### `fig_probe_selectivity.png`

One panel per category.
- X: layer
- Y: balanced accuracy (0 to 1)
- Blue solid: main probe (subgroup_multiclass)
- Gray dashed with ±1σ band: permutation baseline
- Shaded region: selectivity
- Vertical dashed line at peak-selectivity layer
- Title: per-category

### `fig_probe_binary_subgroup.png`

One panel per category. Within each, one line per subgroup.
- X: layer
- Y: balanced accuracy from subgroup_binary probe
- Distinct color + marker per subgroup (Wong palette)
- Horizontal dashed line at 0.5 (chance)
- Legend: subgroup names
- Title: "Binary subgroup detection — {category}"

### `fig_probe_structural_comparison.png`

One panel per category.
- X: layer
- Y: balanced accuracy
- Lines for: subgroup_multiclass (primary), stereotyped_response_binary, context_condition, template_id (stratified)
- Distinct color + marker per probe type
- Legend
- Title: per-category

### `fig_raw_vs_sae_probes.png`

One panel per category. Within each, comparison of raw-hidden-state vs SAE-feature probes.
- X: layer
- Y: balanced accuracy
- For each subgroup: paired lines (raw = solid, SAE = dashed)
- Color by subgroup
- Title: "Raw vs SAE probes — {category}"

### `fig_cross_category_matrix.png`

Heatmap per layer (or one at best-selectivity layer plus a grid of 4 layers).
- Rows: train_category
- Columns: test_category
- Color: balanced accuracy (Blues colormap, vmin=0.4, vmax=1.0)
- Annotated cells
- Title: "Cross-category is_stereotyped generalization — layer {L}"

### `fig_within_category_generalization.png`

One heatmap per category with ≥3 subgroups (at peak layer, from B3's peak differentiation layer).
- Rows: train_subgroup
- Columns: test_subgroup
- Color: balanced accuracy (RdBu_r, centered at 0.5)
- Annotated cells
- Title: "Within-category cross-subgroup generalization — {category} at layer {peak_L}"

---

## Output Structure

```
{run}/B_probes/
├── probe_results.parquet                       # All per-layer per-probe-type results
├── cross_category_generalization.parquet        # Cross-category matrix × layers
├── within_category_cross_subgroup.parquet       # Within-category matrix × layers
├── probes_summary.json                          # Top-level summary
└── figures/
    ├── fig_probe_selectivity.png/.pdf
    ├── fig_probe_binary_subgroup.png/.pdf
    ├── fig_probe_structural_comparison.png/.pdf
    ├── fig_raw_vs_sae_probes.png/.pdf
    ├── fig_cross_category_matrix.png/.pdf
    └── fig_within_category_generalization.png/.pdf
```

---

## Resume Safety

Coarse: check for `probes_summary.json`. If present and not `--force`, skip.

Finer: B4 saves results per-category via incremental parquet writes. Not fully implemented here — default is coarse resume.

Atomic writes for all output files.

---

## Compute Estimate

Approximate times on M4 Max CPU (probes don't use GPU):

- Subgroup multi-class × 9 cat × 32 layers × ~1s probe = ~5 min
- Subgroup binary × ~40 subgroups × 32 layers × ~1s = ~20 min
- Stereotyped response binary × 9 cat × 32 layers × ~1s = ~5 min
- Context condition × 9 × 32 × ~1s = ~5 min
- Template ID × 9 × 32 × ~2s (more classes) = ~10 min
- SAE binary × ~40 × 32 × ~0.5s (smaller features) = ~10 min
- Permutation baselines (10 trials per probe): adds ~10× per above
- Cross-category generalization: 9 × 9 × 32 × ~0.5s = ~20 min
- Within-category cross-subgroup: ~9 cat × ~16 subgroup pairs × 32 × ~0.5s = ~40 min

**Total: ~3-5 hours** with permutation baselines. Configurable to trim (`--layers`, `--n_permutations`, `--skip_sae_probes`).

---

## Assumptions Summary

| # | Decision | Value |
|---|---|---|
| 1 | CV method | `GroupKFold(groups=question_index)` for all probes |
| 2 | Template probe exception | Uses `StratifiedKFold` (non-group-aware) |
| 3 | Dimensionality reduction | PCA-50, fit within each CV fold |
| 4 | Probe model | L2-regularized Logistic Regression (C=1.0) |
| 5 | Binary probe class weighting | `class_weight="balanced"` |
| 6 | Primary metric | Balanced accuracy |
| 7 | Subgroup multi-class | Single-group items only (`n_target_groups == 1`) |
| 8 | Binary subgroup detection | All items; item targeting subgroup S = positive |
| 9 | Context filter for probes | Ambig only (except context_condition probe which uses both) |
| 10 | Minimum positives/negatives | 20 per class for binary probes |
| 11 | Permutation baseline trials | 10 (configurable) |
| 12 | Random seed | 42 (fixed, passed everywhere) |
| 13 | Cross-category PCA | Fit on train_cat only (category-specific projection) |
| 14 | Cross-subgroup eligibility | Single-group items only, min 20 per group |
| 15 | SAE probe feature selection | Top-50 pro-bias features per subgroup from B2 |
| 16 | Probe layer coverage | All layers (configurable via `--layers`) |

---

## Test Command

```bash
# Quick test: layers 0, 14, 31 only, SO category
python scripts/B4_probes.py \
    --run_dir runs/llama-3.1-8b_2026-04-15/ \
    --layers 0,14,31 \
    --n_permutations 5

# Verify outputs
python -c "
import pandas as pd
import json

results = pd.read_parquet('runs/llama-3.1-8b_2026-04-15/B_probes/probe_results.parquet')
print(f'Total probe results: {len(results)}')
print(f'Probe types: {sorted(results[\"probe_type\"].unique())}')
print()

# Subgroup multiclass accuracy by category and layer
print('Subgroup multiclass balanced accuracy:')
multi = results[results['probe_type'] == 'subgroup_multiclass']
print(multi[['category', 'layer', 'mean_balanced_accuracy', 'selectivity']].pivot(
    index='category', columns='layer', values='mean_balanced_accuracy'))

# Template probe vs identity probe comparison
print('\\nTemplate ID balanced accuracy:')
temp = results[results['probe_type'] == 'template_id']
print(temp[['category', 'layer', 'mean_balanced_accuracy']].pivot(
    index='category', columns='layer', values='mean_balanced_accuracy'))

# Cross-category matrix at layer 14
cross = pd.read_parquet('runs/llama-3.1-8b_2026-04-15/B_probes/cross_category_generalization.parquet')
layer_14 = cross[cross['layer'] == 14]
matrix = layer_14.pivot(index='train_category', columns='test_category', values='balanced_accuracy')
print('\\nCross-category matrix at layer 14:')
print(matrix)
"
```