# A3: SAE Encoding — Full Implementation Specification

## Purpose

Pass saved hidden states from A2 through pre-trained Sparse Autoencoder encoders at every transformer layer. Produces sparse feature activations per item per layer as parquet files. This is a pure matrix computation — no model forward passes needed.

## Invocation

```bash
python scripts/A3_sae_encode.py --run_dir runs/llama-3.1-8b_2026-04-15/

# Specific layers only
python scripts/A3_sae_encode.py --run_dir runs/llama-3.1-8b_2026-04-15/ --layers 12,14,16

# Quick test
python scripts/A3_sae_encode.py --run_dir runs/llama-3.1-8b_2026-04-15/ --max_items 20

# Single category
python scripts/A3_sae_encode.py --run_dir runs/llama-3.1-8b_2026-04-15/ --categories so
```

Reads `sae_source`, `sae_expansion`, `n_layers`, `device` from config.json.

---

## Input

- Hidden state .npz files from A2: `{run}/A_extraction/activations/{category}/item_*.npz`
- Metadata parquet from A2: `{run}/A_extraction/metadata.parquet` (for per-subgroup L0 stats)
- SAE checkpoints from HuggingFace

## Dependencies

- `torch`
- `numpy`
- `pandas`, `pyarrow` (parquet I/O)
- `safetensors` (SAE checkpoint loading)
- `huggingface_hub` (SAE download)
- `SAEWrapper` from `src/sae_localization/sae_wrapper.py`

---

## SAE Source

**Confirmed HuggingFace repository:** `OpenMOSS-Team/Llama3_1-8B-Base-LXR-32x`

URL: https://huggingface.co/OpenMOSS-Team/Llama3_1-8B-Base-LXR-32x

This repo contains 32x expansion SAEs (131,072 features) for all 32 layers of Llama-3.1-8B. Each layer's SAE is in a subdirectory like `Llama3_1-8B-Base-L{layer}R-32x/` containing:
- `hyperparams.json`
- `checkpoints/final.safetensors`

**Per-layer checkpoint size:** ~2GB (weights for W_enc, W_dec, biases, thresholds)
**Total download for all 32 layers:** ~64GB
**Download is cached by `huggingface_hub`.** First run downloads; subsequent runs use cache.

---

## Encoding Device Selection

SAE encoding is a matrix multiply — no model needed, but benefits from GPU acceleration:

```python
def select_encode_device(config_device: str) -> str:
    """Select the best available device for SAE encoding."""
    if config_device == "mps" or (config_device != "cpu" and torch.backends.mps.is_available()):
        return "mps"
    elif config_device.startswith("cuda") or torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"
```

**Speed estimates per-item encoding (single item, 32x SAE):**
- CPU (M4 Max): ~0.05-0.1s
- MPS (M4 Max): ~0.005-0.01s
- CUDA (A100): ~0.002s

With batch encoding (batch_size=64), effective per-item time drops further. **MPS is ~10x faster than CPU** for this operation.

**Memory requirement per SAE layer on device:**
- W_enc: (4096, 131072) = ~2.1 GB float32
- W_dec: (131072, 4096) = ~2.1 GB float32
- Biases, thresholds: ~1 MB
- Total: ~4.2 GB per layer

M4 Max with 128GB unified memory handles this easily. Only one SAE layer is loaded at a time — previous layer is deleted before loading the next.

---

## SAE Background

The Sparse Autoencoder decomposes a hidden state vector h ∈ ℝ^d into a sparse vector z ∈ ℝ^F where F >> d:

```
z = JumpReLU(W_enc · h + b_enc)
```

where `JumpReLU(x) = x if x > threshold, else 0` (per-feature thresholds learned during SAE training).

- d = 4096 (Llama-3.1-8B hidden dimension)
- F = 131,072 (32x expansion: 32 × 4096)
- Typical L0 (number of nonzero features per item): 50-100

Each layer of the transformer has its own independently trained SAE. Feature 45021 at layer 14 is a COMPLETELY DIFFERENT feature from feature 45021 at layer 16 — they share an index number but are from different SAEs with different weights.

The SAEWrapper handles:
- Downloading checkpoints from HuggingFace (cached after first download)
- Loading weights from safetensors format
- Dataset-wise normalization (a scaling factor folded into encoder weights during init)
- JumpReLU thresholds (per-feature, loaded from checkpoint)
- The `sparsity_include_decoder_norm` flag (some SAEs scale by decoder column norms)

---

## Script Structure

```python
def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = load_config(run_dir)
    
    encode_device = select_encode_device(config["device"])
    log(f"SAE encoding device: {encode_device}")
    
    n_layers = config["n_layers"]  # populated by A2
    if n_layers is None:
        raise SystemExit("config.json missing n_layers. Run A2 first.")
    
    # Determine which layers to encode
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        layers = list(range(n_layers))  # all 32 layers
    
    categories = parse_categories(args.categories) if args.categories else config["categories"]
    
    # Validate SAE source on startup
    validate_sae_source(config["sae_source"], layers[0], config["sae_expansion"])
    
    # Load metadata for per-subgroup stats
    metadata_df = load_metadata(run_dir)
    
    # Load stimuli for subgroup mapping
    stimuli_by_cat = load_all_stimuli(run_dir, categories)
    
    # Build item_idx → stereotyped_groups lookup
    subgroup_lookup = build_subgroup_lookup(stimuli_by_cat)
    
    # Process each layer
    failed_layers = []
    for layer in layers:
        success = encode_layer(
            layer, run_dir, config, categories, encode_device,
            metadata_df, subgroup_lookup, args.max_items,
        )
        if not success:
            failed_layers.append(layer)
    
    # Build global summary
    build_encoding_summary(run_dir, config, layers, failed_layers, categories)
```

---

## Startup Validation

Before encoding any layers, verify the SAE source is accessible:

```python
def validate_sae_source(sae_source: str, test_layer: int, expansion: int):
    """
    Attempt to locate the SAE checkpoint for one layer to verify 
    the HuggingFace repo is accessible and correctly structured.
    """
    log(f"Validating SAE source: {sae_source}")
    
    try:
        from huggingface_hub import list_repo_files
        files = list_repo_files(sae_source)
        
        # Check for expected subdirectory pattern
        pattern = f"L{test_layer}R-{expansion}x"
        matches = [f for f in files if pattern in f]
        
        if not matches:
            raise FileNotFoundError(
                f"No files matching pattern '{pattern}' found in {sae_source}. "
                f"Available files (sample): {files[:10]}"
            )
        
        log(f"  SAE source validated: found {len(matches)} files for layer {test_layer}")
        log(f"  Sample: {matches[:3]}")
        
    except Exception as e:
        raise SystemExit(
            f"SAE source validation failed: {e}\n"
            f"Verify the repo exists at: https://huggingface.co/{sae_source}\n"
            f"Expected: OpenMOSS-Team/Llama3_1-8B-Base-LXR-32x"
        )
```

---

## Metadata and Subgroup Lookup

A3 needs behavioral metadata and subgroup labels to compute per-subgroup L0 statistics. These come from A2's metadata parquet and A1's stimuli files.

### metadata.parquet (created by A2)

Before A3 runs, A2 should have created:

```
{run}/A_extraction/metadata.parquet
```

If it doesn't exist, A3 should build it from the .npz files (slower but works):

```python
def load_metadata(run_dir: Path) -> pd.DataFrame:
    """Load item metadata. Prefer parquet; fall back to scanning .npz files."""
    meta_path = run_dir / "A_extraction" / "metadata.parquet"
    
    if meta_path.exists():
        df = pd.read_parquet(meta_path)
        log(f"Loaded metadata: {len(df)} items from {meta_path}")
        return df
    
    log(f"metadata.parquet not found; building from .npz files (slow)...")
    records = []
    act_dir = run_dir / "A_extraction" / "activations"
    for cat_dir in sorted(act_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        for npz_path in sorted(cat_dir.glob("item_*.npz")):
            try:
                data = np.load(npz_path, allow_pickle=True)
                meta_str = data["metadata_json"].item() if data["metadata_json"].shape == () else str(data["metadata_json"])
                meta = json.loads(meta_str)
                records.append(meta)
            except Exception:
                continue
    
    df = pd.DataFrame(records)
    # Save for future use
    df.to_parquet(meta_path, index=False)
    log(f"  Built metadata.parquet: {len(df)} items")
    return df
```

### Subgroup lookup

```python
def build_subgroup_lookup(stimuli_by_cat: dict[str, list[dict]]) -> dict[tuple[str, int], list[str]]:
    """
    Build (category, item_idx) → stereotyped_groups mapping.
    Used for per-subgroup L0 statistics.
    """
    lookup = {}
    for cat, items in stimuli_by_cat.items():
        for item in items:
            lookup[(cat, item["item_idx"])] = item["stereotyped_groups"]
    return lookup
```

---

## Core: Per-Layer Encoding

```python
def encode_layer(
    layer: int,
    run_dir: Path,
    config: dict,
    categories: list[str],
    device: str,
    metadata_df: pd.DataFrame,
    subgroup_lookup: dict,
    max_items: int | None,
) -> bool:
    """
    Encode all items at one layer. Returns True on success, False on failure.
    
    Resume-safe: skips if output parquet already exists.
    """
    output_dir = ensure_dir(run_dir / "A_extraction" / "sae_encoding")
    parquet_path = output_dir / f"layer_{layer:02d}.parquet"
    summary_path = output_dir / f"layer_{layer:02d}_summary.json"
    
    # Resume check
    if parquet_path.exists():
        log(f"Layer {layer:02d}: already encoded, skipping")
        return True
    
    log(f"\n{'='*60}")
    log(f"Encoding layer {layer:02d}")
    log(f"{'='*60}")
    
    t0 = time.time()
    
    # Step 1: Load SAE
    try:
        sae = SAEWrapper(
            config["sae_source"],
            layer=layer,
            site=config.get("sae_site", "R"),
            expansion=config["sae_expansion"],
            device=device,
        )
    except Exception as e:
        log(f"  ERROR: Failed to load SAE for layer {layer}: {e}")
        return False
    
    log(f"  SAE loaded: {sae.n_features} features, hidden_dim={sae.hidden_dim}")
    
    # Step 2: Encode all items, batched
    all_records = []          # accumulate parquet rows
    per_category_stats = {}   # for summary JSON
    per_subgroup_l0 = {}      # for per-subgroup stats
    validation_done = False
    full_validation_layers = {0, config["n_layers"] // 2, config["n_layers"] - 1}
    
    for cat in categories:
        cat_dir = run_dir / "A_extraction" / "activations" / cat
        if not cat_dir.is_dir():
            log(f"  No activations for {cat}, skipping")
            continue
        
        npz_paths = sorted(cat_dir.glob("item_*.npz"))
        if max_items:
            npz_paths = npz_paths[:max_items]
        
        n_items = len(npz_paths)
        log(f"  {cat}: {n_items} items")
        
        cat_l0s = []       # L0 per item for this category
        cat_records = []   # parquet rows for this category
        
        # Process in batches
        BATCH_SIZE = 64
        batch_hs = []      # raw hidden states
        batch_idxs = []    # item_idx values
        
        for npz_i, npz_path in enumerate(npz_paths):
            # Load hidden state for this layer only
            data = np.load(npz_path, allow_pickle=True)
            hs_normed = data["hidden_states"][layer].astype(np.float32)  # (hidden_dim,)
            raw_norm = float(data["hidden_states_raw_norms"][layer])
            
            # Reconstruct raw activation
            hs_raw = hs_normed * raw_norm
            
            # Extract item_idx from filename
            item_idx = int(npz_path.stem.split("_")[1])
            
            batch_hs.append(hs_raw)
            batch_idxs.append(item_idx)
            
            # Process batch when full or at end of category
            if len(batch_hs) == BATCH_SIZE or npz_i == len(npz_paths) - 1:
                batch_records, batch_l0s = encode_batch(
                    sae, batch_hs, batch_idxs, cat, layer, device,
                    validate=(not validation_done),
                    full_validate=(layer in full_validation_layers and not validation_done),
                )
                
                cat_records.extend(batch_records)
                cat_l0s.extend(batch_l0s)
                
                if not validation_done and batch_records:
                    validation_done = True
                
                batch_hs.clear()
                batch_idxs.clear()
            
            # Progress logging
            if (npz_i + 1) % 500 == 0 or npz_i == 0:
                log(f"    [{npz_i + 1}/{n_items}]")
        
        all_records.extend(cat_records)
        
        # Category-level L0 stats
        if cat_l0s:
            l0_arr = np.array(cat_l0s)
            per_category_stats[cat] = {
                "n_items_encoded": n_items,
                "mean_l0": round(float(l0_arr.mean()), 1),
                "std_l0": round(float(l0_arr.std()), 1),
                "median_l0": round(float(np.median(l0_arr)), 1),
                "min_l0": int(l0_arr.min()),
                "max_l0": int(l0_arr.max()),
                "total_nonzero_entries": len(cat_records),
            }
            
            # Compute mean activation across nonzero entries
            if cat_records:
                acts = [r["activation_value"] for r in cat_records]
                per_category_stats[cat]["mean_activation_nonzero"] = round(float(np.mean(acts)), 4)
                per_category_stats[cat]["max_activation"] = round(float(np.max(acts)), 4)
        
        # Per-subgroup L0 stats
        subgroup_l0_accum = {}  # {subgroup: [l0_values]}
        for item_idx, l0 in zip([extract_idx(p) for p in npz_paths[:len(cat_l0s)]], cat_l0s):
            groups = subgroup_lookup.get((cat, item_idx), [])
            for sg in groups:
                subgroup_l0_accum.setdefault(sg, []).append(l0)
        
        for sg, l0_vals in sorted(subgroup_l0_accum.items()):
            arr = np.array(l0_vals)
            key = f"{cat}/{sg}"
            per_subgroup_l0[key] = {
                "n_items": len(l0_vals),
                "mean_l0": round(float(arr.mean()), 1),
                "std_l0": round(float(arr.std()), 1),
            }
        
        # Memory cleanup between categories
        if device == "mps":
            torch.mps.empty_cache()
    
    # Step 3: Write parquet (atomic)
    if all_records:
        df = pd.DataFrame(all_records)
        # Enforce dtypes
        df["item_idx"] = df["item_idx"].astype(np.int32)
        df["feature_idx"] = df["feature_idx"].astype(np.int32)
        df["activation_value"] = df["activation_value"].astype(np.float32)
        
        tmp_path = parquet_path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp_path, index=False, compression="snappy")
        tmp_path.rename(parquet_path)
        
        parquet_size_mb = parquet_path.stat().st_size / (1024 * 1024)
        log(f"  Parquet: {len(df)} rows, {parquet_size_mb:.1f} MB → {parquet_path.name}")
    else:
        log(f"  WARNING: No records produced for layer {layer}")
        return False
    
    # Step 4: Write per-layer summary
    elapsed = time.time() - t0
    layer_summary = {
        "layer": layer,
        "sae_source": config["sae_source"],
        "sae_expansion": config["sae_expansion"],
        "n_features": sae.n_features,
        "encoding_device": device,
        "per_category": per_category_stats,
        "per_subgroup_l0": per_subgroup_l0,
        "encoding_time_seconds": round(elapsed, 1),
    }
    
    with open(summary_path, "w") as f:
        json.dump(layer_summary, f, indent=2)
    
    log(f"  Complete in {elapsed:.1f}s")
    
    # Step 5: Unload SAE to free memory
    del sae
    if device == "mps":
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()
    
    import gc
    gc.collect()
    
    return True
```

---

## Core: Batch Encoding

```python
def encode_batch(
    sae,
    batch_hs: list[np.ndarray],
    batch_idxs: list[int],
    category: str,
    layer: int,
    device: str,
    validate: bool = False,
    full_validate: bool = False,
) -> tuple[list[dict], list[int]]:
    """
    Encode a batch of hidden states through the SAE.
    
    Args:
        sae: SAEWrapper instance
        batch_hs: list of raw hidden state arrays, each shape (hidden_dim,)
        batch_idxs: corresponding item_idx values
        category: category short name
        layer: layer index (for logging)
        device: encoding device
        validate: if True, run L0 sanity check on first item
        full_validate: if True, also run reconstruction check on first item
    
    Returns:
        (records, l0s) where records is a list of parquet row dicts
        and l0s is a list of L0 values (one per item in batch)
    """
    # Stack batch: (batch_size, hidden_dim)
    batch_tensor = torch.from_numpy(np.stack(batch_hs)).to(dtype=torch.float32)
    
    # Encode: (batch_size, n_features)
    feature_activations = sae.encode(batch_tensor)
    
    records = []
    l0s = []
    
    for i in range(len(batch_hs)):
        item_acts = feature_activations[i]  # (n_features,)
        
        # Nonzero features
        nonzero_mask = item_acts > 0
        n_active = int(nonzero_mask.sum().item())
        l0s.append(n_active)
        
        # Validation on first item
        if validate and i == 0:
            run_encoding_validation(
                sae, batch_hs[i], item_acts, n_active, layer,
                full_reconstruct=full_validate,
            )
        
        # Extract sparse representation
        if n_active > 0:
            feat_indices = nonzero_mask.nonzero(as_tuple=True)[0].cpu().numpy()
            act_values = item_acts[nonzero_mask].cpu().detach().numpy()
            
            for fidx, aval in zip(feat_indices, act_values):
                records.append({
                    "item_idx": batch_idxs[i],
                    "feature_idx": int(fidx),
                    "activation_value": float(aval),
                    "category": category,
                })
    
    return records, l0s
```

---

## Encoding Validation

```python
def run_encoding_validation(
    sae,
    hs_raw: np.ndarray,
    feature_activations: torch.Tensor,
    n_active: int,
    layer: int,
    full_reconstruct: bool = False,
):
    """
    Validate SAE encoding on a single item.
    
    Always runs:
        - L0 sanity check (expected: 20-200 for 32x SAEs)
        - Top-5 active features log
    
    If full_reconstruct=True, also runs:
        - Decode back to hidden state and check reconstruction error
        - Expected relative error < 0.1 for a well-trained SAE
    """
    log(f"  === Encoding Validation (layer {layer}) ===")
    
    # L0 check
    log(f"    L0 = {n_active} active features out of {sae.n_features}")
    
    if n_active < 5:
        raw_norm = float(np.linalg.norm(hs_raw))
        log(f"    WARNING: Very low L0 ({n_active}). Possible encoding issue.")
        log(f"    Raw activation norm: {raw_norm:.4f}")
        log(f"    If norm is very small, the hidden state may be near-zero at this layer.")
    elif n_active > 500:
        log(f"    WARNING: Very high L0 ({n_active}). SAE thresholds may be miscalibrated.")
        log(f"    Check that sae_source matches the model (Llama-3.1-8B).")
    else:
        log(f"    L0 in expected range (20-200 for 32x SAE)")
    
    # Top-5 active features
    nonzero_mask = feature_activations > 0
    if n_active > 0:
        nonzero_acts = feature_activations[nonzero_mask]
        top_k = min(5, n_active)
        top_vals, top_local_idx = torch.topk(nonzero_acts, top_k)
        # Map local indices back to global feature indices
        global_indices = nonzero_mask.nonzero(as_tuple=True)[0]
        top_global_idx = global_indices[top_local_idx.cpu()]
        
        log(f"    Top-{top_k} features:")
        for fidx, fval in zip(top_global_idx.tolist(), top_vals.tolist()):
            log(f"      Feature {fidx}: activation = {fval:.4f}")
    
    # Full reconstruction (expensive — only at selected layers)
    if full_reconstruct:
        reconstructed = sae.decode(feature_activations)
        hs_tensor = torch.from_numpy(hs_raw).to(reconstructed.dtype).to(reconstructed.device)
        
        recon_error = float(torch.norm(reconstructed - hs_tensor).item())
        raw_norm = float(torch.norm(hs_tensor).item())
        relative_error = recon_error / max(raw_norm, 1e-8)
        
        log(f"    Reconstruction error: {recon_error:.4f} (relative: {relative_error:.4f})")
        
        if relative_error > 0.3:
            log(f"    WARNING: High reconstruction error ({relative_error:.4f} > 0.3).")
            log(f"    The SAE may not match this model's activations.")
            log(f"    Check: correct model (Llama-3.1-8B)? Correct layer?")
        elif relative_error > 0.1:
            log(f"    Moderate reconstruction error. Acceptable but worth noting.")
        else:
            log(f"    Reconstruction quality: good")
    
    log(f"  === End Validation ===")
```

---

## Metadata Parquet Creation (in A2)

A2 should create `metadata.parquet` after extracting all items. Add this to the end of A2's `build_and_save_summary`:

```python
def save_metadata_parquet(run_dir: Path, categories: list[str]):
    """
    Create a flat metadata parquet from all extracted .npz files.
    One row per item with all behavioral metadata.
    Used by A3 (per-subgroup L0), B1 (comparison groups), and all Phase B/C scripts.
    """
    records = []
    act_dir = run_dir / "A_extraction" / "activations"
    
    for cat_dir in sorted(act_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        cat = cat_dir.name
        for npz_path in sorted(cat_dir.glob("item_*.npz")):
            try:
                data = np.load(npz_path, allow_pickle=True)
                raw = data["metadata_json"]
                meta_str = raw.item() if raw.shape == () else str(raw)
                meta = json.loads(meta_str)
                
                records.append({
                    "item_idx": meta["item_idx"],
                    "category": meta["category"],
                    "model_answer": meta["model_answer"],
                    "model_answer_role": meta["model_answer_role"],
                    "is_stereotyped_response": meta["is_stereotyped_response"],
                    "is_correct": meta["is_correct"],
                    "context_condition": meta["context_condition"],
                    "stereotyped_groups": json.dumps(meta["stereotyped_groups"]),  # store as JSON string
                    "n_target_groups": meta["n_target_groups"],
                    "margin": meta["margin"],
                    "question_polarity": meta["question_polarity"],
                    "correct_letter": meta["correct_letter"],
                    "stereotyped_option": meta["stereotyped_option"],
                })
            except Exception as e:
                log(f"  WARNING: failed to read {npz_path}: {e}")
    
    df = pd.DataFrame(records)
    out_path = run_dir / "A_extraction" / "metadata.parquet"
    df.to_parquet(out_path, index=False, compression="snappy")
    log(f"Saved metadata.parquet: {len(df)} items → {out_path}")
```

**Schema for metadata.parquet:**

| Column | Type | Description |
|--------|------|-------------|
| `item_idx` | int32 | BBQ example_id (primary key within category) |
| `category` | string | Category short name |
| `model_answer` | string | Letter the model chose ("A", "B", "C") |
| `model_answer_role` | string | "stereotyped_target", "non_stereotyped", "unknown" |
| `is_stereotyped_response` | bool | model_answer_role == "stereotyped_target" |
| `is_correct` | bool | model_answer == correct_letter |
| `context_condition` | string | "ambig" or "disambig" |
| `stereotyped_groups` | string | JSON-encoded list, e.g. `'["bisexual"]'` |
| `n_target_groups` | int32 | Length of stereotyped_groups |
| `margin` | float32 | Logit margin (confidence) |
| `question_polarity` | string | "neg" or "nonneg" |
| `correct_letter` | string | The correct answer letter |
| `stereotyped_option` | string | Which letter is stereotyped_target |

**Note on `stereotyped_groups` as JSON string:** Parquet doesn't natively support list columns in all readers. Storing as a JSON-encoded string is universally compatible. Downstream code deserializes with `json.loads(row["stereotyped_groups"])`.

---

## Global Encoding Summary

After all layers are complete:

```python
def build_encoding_summary(
    run_dir: Path,
    config: dict,
    layers: list[int],
    failed_layers: list[int],
    categories: list[str],
):
    """Build the global encoding_summary.json from per-layer summaries."""
    encoding_dir = run_dir / "A_extraction" / "sae_encoding"
    
    l0_by_layer = {}
    total_time = 0.0
    total_size_bytes = 0
    
    for layer in layers:
        if layer in failed_layers:
            continue
        
        summary_path = encoding_dir / f"layer_{layer:02d}_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                layer_summary = json.load(f)
            
            # Aggregate L0 across categories
            all_l0_means = [v["mean_l0"] for v in layer_summary["per_category"].values()]
            all_l0_stds = [v["std_l0"] for v in layer_summary["per_category"].values()]
            l0_by_layer[str(layer)] = {
                "mean": round(float(np.mean(all_l0_means)), 1),
                "std": round(float(np.mean(all_l0_stds)), 1),
            }
            
            total_time += layer_summary.get("encoding_time_seconds", 0)
        
        parquet_path = encoding_dir / f"layer_{layer:02d}.parquet"
        if parquet_path.exists():
            total_size_bytes += parquet_path.stat().st_size
    
    # Count items per category
    items_per_cat = {}
    for cat in categories:
        stimuli_path = run_dir / "A_extraction" / "stimuli" / f"{cat}.json"
        if stimuli_path.exists():
            with open(stimuli_path) as f:
                items_per_cat[cat] = len(json.load(f))
    
    summary = {
        "sae_source": config["sae_source"],
        "sae_expansion": config["sae_expansion"],
        "encode_device": "mps",  # or whatever was used
        "layers_encoded": sorted([l for l in layers if l not in failed_layers]),
        "failed_layers": failed_layers,
        "total_items_per_category": items_per_cat,
        "total_parquet_size_mb": round(total_size_bytes / (1024 * 1024), 1),
        "total_encoding_time_seconds": round(total_time, 1),
        "l0_by_layer": l0_by_layer,
    }
    
    summary_path = encoding_dir / "encoding_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    log(f"\nEncoding summary → {summary_path}")
    log(f"Layers encoded: {len(summary['layers_encoded'])}/{len(layers)}")
    if failed_layers:
        log(f"Failed layers: {failed_layers}")
    log(f"Total parquet size: {summary['total_parquet_size_mb']:.1f} MB")
    log(f"Total time: {summary['total_encoding_time_seconds']:.0f}s")
```

---

## Output Structure

```
{run}/A_extraction/sae_encoding/
├── layer_00.parquet                 # Sparse feature activations, all categories
├── layer_00_summary.json            # L0 stats, per-category and per-subgroup
├── layer_01.parquet
├── layer_01_summary.json
├── ...
├── layer_31.parquet
├── layer_31_summary.json
└── encoding_summary.json            # Global summary across all layers
```

**Parquet schema (long/sparse format):**

| Column | Type | Description |
|--------|------|-------------|
| `item_idx` | int32 | BBQ example_id (matches A1/A2) |
| `feature_idx` | int32 | SAE feature index (0 to 131071) |
| `activation_value` | float32 | Feature activation magnitude (always > 0) |
| `category` | string | Category short name |

Only nonzero activations are stored. Items with zero activation for a feature have no row.

**Per-layer summary JSON example:**

```json
{
  "layer": 14,
  "sae_source": "OpenMOSS-Team/Llama3_1-8B-Base-LXR-32x",
  "sae_expansion": 32,
  "n_features": 131072,
  "encoding_device": "mps",
  "per_category": {
    "so": {
      "n_items_encoded": 8640,
      "mean_l0": 73.2,
      "std_l0": 18.4,
      "median_l0": 71.0,
      "min_l0": 12,
      "max_l0": 203,
      "mean_activation_nonzero": 0.94,
      "max_activation": 18.7,
      "total_nonzero_entries": 632448
    }
  },
  "per_subgroup_l0": {
    "so/gay": {"n_items": 2160, "mean_l0": 71.3, "std_l0": 17.2},
    "so/bisexual": {"n_items": 2160, "mean_l0": 74.8, "std_l0": 19.1},
    "so/lesbian": {"n_items": 2160, "mean_l0": 72.1, "std_l0": 16.8},
    "so/pansexual": {"n_items": 2160, "mean_l0": 75.4, "std_l0": 18.9},
    "race/black": {"n_items": 1440, "mean_l0": 68.9, "std_l0": 15.3},
    "race/asian": {"n_items": 1440, "mean_l0": 70.2, "std_l0": 16.1}
  },
  "encoding_time_seconds": 245.3
}
```

**Global encoding_summary.json example:**

```json
{
  "sae_source": "OpenMOSS-Team/Llama3_1-8B-Base-LXR-32x",
  "sae_expansion": 32,
  "encode_device": "mps",
  "layers_encoded": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
  "failed_layers": [],
  "total_items_per_category": {
    "so": 8640, "race": 8640, "disability": 3960, "gi": 8640,
    "religion": 8640, "age": 8640, "nationality": 8640,
    "physical_appearance": 8640, "ses": 8640
  },
  "total_parquet_size_mb": 384.2,
  "total_encoding_time_seconds": 2845.0,
  "l0_by_layer": {
    "0": {"mean": 52.1, "std": 14.3},
    "1": {"mean": 58.7, "std": 15.1},
    "14": {"mean": 73.5, "std": 18.2},
    "31": {"mean": 89.4, "std": 22.7}
  }
}
```

---

## Resume Safety

**Per-layer granularity.** Before processing a layer, check if the output parquet exists:

```python
if parquet_path.exists():
    log(f"Layer {layer:02d}: already encoded, skipping")
    return True
```

If it exists, skip entirely. The per-layer summary JSON is written right after the parquet, so if the parquet exists the summary should too (if not, it's regenerated by `build_encoding_summary`).

**Atomic parquet writes.** Write to `.parquet.tmp`, then rename:

```python
tmp_path = parquet_path.with_suffix(".parquet.tmp")
df.to_parquet(tmp_path, index=False, compression="snappy")
tmp_path.rename(parquet_path)
```

This prevents corrupt parquets from crashed writes.

**No within-layer resume.** If a layer crashes mid-encoding, it re-runs from scratch. Encoding a full layer takes ~1-2 minutes on MPS (with batching), so partial resume within a layer isn't worth the complexity.

**Layer independence.** Each layer is encoded independently. Crash at layer 17 → layers 0-16 are saved, layer 17 re-runs, layers 18-31 proceed normally.

---

## Compute Estimate (with MPS + batching)

- Per-batch (64 items): single matrix multiply `(64, 4096) × (4096, 131072)` ≈ 0.3s on MPS
- Per-item effective: ~0.005s
- Per category (~8640 items): ~43s
- Per layer (all 9 categories, ~70,000 items): ~5-6 minutes
- All 32 layers: ~160-190 minutes ≈ **2.5-3 hours on MPS**

This is substantially faster than the earlier CPU estimate of 5+ hours due to MPS acceleration and batching.

With `--max_items 20`: ~20 × 9 × 32 × 0.005s ≈ 29 seconds (quick test).

---

## Layer 0 Note

Layer 0 activations are token embeddings + positional encoding, not compositional representations. SAE features at layer 0 reflect lexical identity (which words appear in the prompt), not semantic processing.

**Features significant at layer 0 are almost certainly lexical confounds.** For example, a feature that fires on the token "bisexual" will be "significant" in B1's differential analysis because stereotyped-bisexual items literally contain that word more often than non-stereotyped items.

**We encode layer 0 anyway** because it provides a useful baseline for B1:
- Features significant at layer 0 AND at deeper layers: partially lexical, interpret cautiously
- Features significant ONLY at deep layers (not at layer 0): more likely to be compositional/semantic
- This distinction matters for B5's artifact detection (category_specificity_ratio)

---

## Test Command

```bash
# Quick test: layer 14 only, 20 items, SO category
python scripts/A3_sae_encode.py \
    --run_dir runs/llama-3.1-8b_2026-04-15/ \
    --layers 14 \
    --categories so \
    --max_items 20

# Verify output
python -c "
import pandas as pd
df = pd.read_parquet('runs/llama-3.1-8b_2026-04-15/A_extraction/sae_encoding/layer_14.parquet')
print(f'Rows: {len(df)}')
print(f'Columns: {list(df.columns)}')
print(f'Categories: {df[\"category\"].unique()}')
print(f'Items: {df[\"item_idx\"].nunique()}')
print(f'Features: {df[\"feature_idx\"].nunique()}')
print(f'Activation range: [{df[\"activation_value\"].min():.4f}, {df[\"activation_value\"].max():.4f}]')
print(f'L0 (mean features per item):', df.groupby('item_idx')['feature_idx'].count().mean())
print(f'Sample rows:')
print(df.head(5))
"

# Full run: all layers
python scripts/A3_sae_encode.py --run_dir runs/llama-3.1-8b_2026-04-15/ 2>&1 | tee logs/A3_encode.log
```