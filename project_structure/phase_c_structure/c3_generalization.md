# C3: Generalization Evaluation — Full Implementation Specification

## Purpose

Apply C1's saved steering vectors to out-of-distribution benchmarks (MedQA and MMLU). Tests whether bias-specific interventions transfer to real-world tasks and quantifies the "capability tax" — how much general performance suffers. Primary analysis uses item-level paired comparisons (baseline vs steered) of logits on correct answer options, not just accuracy deltas.

## Invocation

```bash
python scripts/C3_generalization.py --run_dir runs/llama-3.1-8b_2026-04-15/

# Limit items per dataset
python scripts/C3_generalization.py --run_dir runs/llama-3.1-8b_2026-04-15/ --max_medqa_items 500 --max_mmlu_items 1000

# Specific steering vectors
python scripts/C3_generalization.py --run_dir runs/llama-3.1-8b_2026-04-15/ --subgroups so/gay,race/black

# Skip exacerbation (only debiasing direction)
python scripts/C3_generalization.py --run_dir runs/llama-3.1-8b_2026-04-15/ --skip_exacerbation

# Only MedQA or only MMLU
python scripts/C3_generalization.py --run_dir runs/llama-3.1-8b_2026-04-15/ --datasets medqa
python scripts/C3_generalization.py --run_dir runs/llama-3.1-8b_2026-04-15/ --datasets mmlu
```

Reads `model_path`, `sae_source`, `device`, `dtype`, `medqa_path`, `mmlu_path` from config.json.

---

## Input

- Steering vectors: `{run}/C_steering/vectors/{cat}_{sub}.npz` (from C1)
- Steering manifests: `{run}/C_steering/steering_manifests.json` (for `steering_viable` filter)
- MedQA dataset at `config.medqa_path`
- MMLU dataset at `config.mmlu_path`
- Model + SAEs loaded per injection layer

## Dependencies

- `torch`
- `pandas`, `pyarrow`
- `numpy`, `scipy.stats`
- `matplotlib`
- Ported modules: `SAESteerer`, `SAEWrapper`, `ModelWrapper`
- Data loaders: `src/data/medqa_loader.py`, `src/data/mmlu_loader.py`

---

## Sign Convention Reminder

From C1:
- Saved vector in `vectors/{cat}_{sub}.npz` has `alpha < 0` (debiasing direction, negative target_norm)
- Loading gives the **debiasing vector** directly
- `exac_vec = -vec` gives the **exacerbation vector**

So:
- `vec` applied: pushes model AWAY from stereotyped responses
- `-vec` applied: pushes model TOWARD stereotyped responses

---

## Item-Level Paired Comparison Framework

**The core analysis unit:** for each (item, steering_vector, direction) triple, we have two forward passes through the model:

1. **Baseline:** no hooks, pure model output
2. **Steered:** single hook at `injection_layer`, adding `(direction_multiplier × vec)` to residual stream at last-token position

Where `direction_multiplier = +1` for debiasing, `-1` for exacerbation.

From each pass we extract:
- Logits for each answer letter (A, B, C, D, [E for MedQA])
- Predicted top answer
- Degeneration flag

For a given item, BASELINE is shared across all steering vectors (independent of which vector is applied). Cache baselines once, reuse across all vectors.

### Per-Item Record (both datasets share this structure)

```python
{
    "item_idx": int,                          # from dataset
    "dataset": "medqa" | "mmlu",
    "correct_answer": "A",                    # ground truth letter
    "baseline_top_answer": "B",               # model's top choice, no steering
    "baseline_top_logit": 3.21,               # logit of top answer
    "baseline_correct_logit": 2.87,           # logit of correct answer
    "baseline_correct": False,                # was top answer correct?
    "baseline_degenerated": False,
    "baseline_all_logits": {"A": 1.2, "B": 3.21, "C": 2.87, "D": -0.5},

    # Per-vector steered data (one entry per applied vector)
    "steered_results": {
        "so/gay_debiasing": {
            "top_answer": "C",
            "top_logit": 3.54,
            "correct_logit": 3.54,
            "correct": True,
            "degenerated": False,
            "all_logits": {"A": 0.9, "B": 2.1, "C": 3.54, "D": -0.8},
            # Derived:
            "correct_logit_shift": 0.67,      # steered - baseline correct_logit
            "flipped": True,                  # top_answer changed
            "correctness_delta": 1,           # steered_correct - baseline_correct (∈ {-1, 0, +1})
        },
        "so/gay_exacerbation": {...},
        "race/black_debiasing": {...},
        ...
    }
}
```

### Item-Level Primary Metrics

Per (item, vector, direction):
- `correct_logit_shift = steered_correct_logit - baseline_correct_logit`
- `|correct_logit_shift|` = magnitude of steering effect on this item
- `correctness_delta ∈ {-1, 0, +1}`
- `flipped ∈ {True, False}`

---

## Script Structure

```python
def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = load_config(run_dir)

    device = torch.device(config["device"])
    dtype = getattr(torch, config["dtype"])

    # Load model
    wrapper = ModelWrapper(config["model_path"], device=device, dtype=dtype)

    # Load viable steering vectors
    viable_vectors = load_viable_vectors(run_dir, device, dtype, args.subgroups)
    log(f"Loaded {len(viable_vectors)} viable steering vectors")

    # Identify unique injection layers
    needed_layers = set(v["injection_layer"] for v in viable_vectors)
    sae_cache = {}
    for layer in sorted(needed_layers):
        log(f"Loading SAE for layer {layer}...")
        sae_cache[layer] = SAEWrapper(
            config["sae_source"], layer=layer,
            expansion=config["sae_expansion"],
            device=device, dtype=dtype,
        )

    output_dir = run_dir / "C_generalization"
    ensure_dir(output_dir / "baselines")
    ensure_dir(output_dir / "medqa" / "per_vector_checkpoints")
    ensure_dir(output_dir / "mmlu" / "per_vector_checkpoints")
    ensure_dir(output_dir / "figures")

    datasets_to_run = args.datasets.split(",") if args.datasets else ["medqa", "mmlu"]

    # Load datasets
    medqa_items = None
    mmlu_items = None

    if "medqa" in datasets_to_run:
        try:
            medqa_items = load_medqa_items(config["medqa_path"])
            if args.max_medqa_items:
                medqa_items = medqa_items[:args.max_medqa_items]
            log(f"Loaded {len(medqa_items)} MedQA items")
            medqa_items = classify_medqa_demographics(medqa_items, output_dir)
        except Exception as e:
            log(f"WARNING: MedQA loading failed: {e}")
            medqa_items = None

    if "mmlu" in datasets_to_run:
        try:
            mmlu_items = load_mmlu_items(config["mmlu_path"])
            if args.max_mmlu_items:
                mmlu_items = mmlu_items[:args.max_mmlu_items]
            log(f"Loaded {len(mmlu_items)} MMLU items")
            mmlu_items = annotate_mmlu_supercategories(mmlu_items)
        except Exception as e:
            log(f"WARNING: MMLU loading failed: {e}")
            mmlu_items = None

    # Step 1: Compute and cache baselines
    medqa_baselines_df = None
    mmlu_baselines_df = None

    if medqa_items:
        medqa_baselines_df = compute_or_load_baselines(
            medqa_items, wrapper, sae_cache,
            letters=("A", "B", "C", "D", "E"),
            cache_path=output_dir / "baselines" / "medqa_baselines.parquet",
            dataset_name="medqa",
        )

    if mmlu_items:
        mmlu_baselines_df = compute_or_load_baselines(
            mmlu_items, wrapper, sae_cache,
            letters=("A", "B", "C", "D"),
            cache_path=output_dir / "baselines" / "mmlu_baselines.parquet",
            dataset_name="mmlu",
        )

    # Step 2: Evaluate each steering vector
    medqa_per_item_records = []
    mmlu_per_item_records = []

    for vec_info in viable_vectors:
        if medqa_items is not None:
            records = run_dataset_evaluation(
                dataset_name="medqa",
                items=medqa_items,
                baselines_df=medqa_baselines_df,
                vec_info=vec_info,
                sae_cache=sae_cache,
                wrapper=wrapper,
                output_dir=output_dir,
                letters=("A", "B", "C", "D", "E"),
                skip_exacerbation=args.skip_exacerbation,
            )
            medqa_per_item_records.extend(records)

        if mmlu_items is not None:
            records = run_dataset_evaluation(
                dataset_name="mmlu",
                items=mmlu_items,
                baselines_df=mmlu_baselines_df,
                vec_info=vec_info,
                sae_cache=sae_cache,
                wrapper=wrapper,
                output_dir=output_dir,
                letters=("A", "B", "C", "D"),
                skip_exacerbation=args.skip_exacerbation,
            )
            mmlu_per_item_records.extend(records)

    # Step 3: Save per-item parquets
    if medqa_per_item_records:
        save_per_item_parquet(
            medqa_per_item_records,
            output_dir / "medqa" / "per_item.parquet",
            dataset="medqa",
        )

    if mmlu_per_item_records:
        save_per_item_parquet(
            mmlu_per_item_records,
            output_dir / "mmlu" / "per_item.parquet",
            dataset="mmlu",
        )

    # Step 4: Aggregate
    aggregate_results(output_dir, viable_vectors)

    # Step 5: Update manifests with generalization fields
    update_manifests(run_dir, output_dir)

    # Step 6: Identify top-impact items for supplementary analysis
    identify_top_impact_items(output_dir)

    # Figures
    if not args.skip_figures:
        generate_figures(output_dir)
```

---

## Step 0: Demographic Classification for MedQA

Systematic pattern-matching approach. Patterns are documented with:
- Their BBQ subgroup label (normalized)
- BBQ category
- Whether they're high-false-positive-risk (matched items may be noisy)

### Pattern Definitions

```python
# All patterns compiled as case-insensitive regex, applied to lowercase text.
# BBQ subgroup labels are normalized: lowercase, singular, specific spelling convention.

DEMOGRAPHIC_PATTERNS = {
    # Sexual Orientation
    "gay": {
        "category": "so",
        "patterns": [r"\bgay\b", r"\bhomosexual (man|male)\b", r"\bmsm\b", r"\bmen who have sex with men\b"],
        "fp_risk": "low",
    },
    "lesbian": {
        "category": "so",
        "patterns": [r"\blesbian\b", r"\bhomosexual (woman|female)\b", r"\bwsw\b", r"\bwomen who have sex with women\b"],
        "fp_risk": "low",
    },
    "bisexual": {
        "category": "so",
        "patterns": [r"\bbisexual\b", r"\bmsmw\b"],
        "fp_risk": "low",
    },
    "pansexual": {
        "category": "so",
        "patterns": [r"\bpansexual\b"],
        "fp_risk": "low",
    },

    # Gender Identity
    "transgender women": {
        "category": "gi",
        "patterns": [r"\btransgender wom(a|e)n\b", r"\btrans wom(a|e)n\b", r"\bmale.to.female\b", r"\bmtf\b"],
        "fp_risk": "low",
    },
    "transgender men": {
        "category": "gi",
        "patterns": [r"\btransgender m(a|e)n\b", r"\btrans m(a|e)n\b", r"\bfemale.to.male\b", r"\bftm\b"],
        "fp_risk": "low",
    },
    "non-binary": {
        "category": "gi",
        "patterns": [r"\bnon.binary\b", r"\bnonbinary\b", r"\bgenderqueer\b"],
        "fp_risk": "low",
    },

    # Race (using BBQ labels: black, white, asian, middle eastern, hispanic, native american)
    "black": {
        "category": "race",
        "patterns": [r"\bblack\b", r"\bafrican.?american\b"],
        "fp_risk": "medium",  # "black" may refer to non-demographic color (black bile, etc.)
    },
    "white": {
        "category": "race",
        "patterns": [r"\bwhite (m(a|e)n|wom(a|e)n|person|patient|male|female)\b", r"\bcaucasian\b"],
        "fp_risk": "medium",
    },
    "asian": {
        "category": "race",
        "patterns": [r"\basian\b", r"\bsoutheast asian\b", r"\bsouth asian\b", r"\beast asian\b"],
        "fp_risk": "low",
    },
    "middle eastern": {
        "category": "race",
        "patterns": [r"\bmiddle.eastern\b", r"\barab\b"],
        "fp_risk": "low",
    },
    "hispanic": {
        "category": "race",
        "patterns": [r"\bhispanic\b", r"\blatin(o|a|x|e)\b"],
        "fp_risk": "low",
    },
    "native american": {
        "category": "race",
        "patterns": [r"\bnative american\b", r"\bamerican indian\b"],
        "fp_risk": "low",
    },

    # Religion (BBQ labels: muslim, jewish, christian, hindu, buddhist, atheist, etc.)
    "muslim": {
        "category": "religion",
        "patterns": [r"\bmuslim\b", r"\bislamic\b"],
        "fp_risk": "low",
    },
    "jewish": {
        "category": "religion",
        "patterns": [r"\bjewish\b", r"\bashkenazi\b", r"\bsephardic\b"],
        "fp_risk": "low",
    },
    "christian": {
        "category": "religion",
        "patterns": [r"\bchristian\b", r"\bcatholic\b", r"\bprotestant\b", r"\bevangelical\b"],
        "fp_risk": "low",
    },
    "hindu": {
        "category": "religion",
        "patterns": [r"\bhindu\b"],
        "fp_risk": "low",
    },
    "buddhist": {
        "category": "religion",
        "patterns": [r"\bbuddhist\b"],
        "fp_risk": "low",
    },
    "atheist": {
        "category": "religion",
        "patterns": [r"\batheist\b", r"\bagnostic\b"],
        "fp_risk": "low",
    },

    # Age (BBQ labels: young, old)
    "old": {
        "category": "age",
        "patterns": [r"\b(elderly|aged|senior|geriatric)\b", r"\b\d{2,3}.year.old\b"],  # will need to filter the age pattern further
        "fp_risk": "high",  # ages appear constantly in clinical content
    },
    "young": {
        "category": "age",
        "patterns": [r"\byoung (patient|adult|person|m(a|e)n|wom(a|e)n|child)\b", r"\badolescent\b", r"\bteenage\b"],
        "fp_risk": "medium",
    },

    # SES (BBQ labels: low ses / high ses)
    "low ses": {
        "category": "ses",
        "patterns": [r"\blow.income\b", r"\bpoor (patient|family|neighborhood)\b", r"\bhomeless\b", r"\buninsured\b", r"\bmedicaid\b"],
        "fp_risk": "medium",
    },
    "high ses": {
        "category": "ses",
        "patterns": [r"\bwealthy\b", r"\bupper.class\b", r"\baffluent\b"],
        "fp_risk": "medium",
    },

    # Nationality — BBQ has many specific nationalities; include a subset relevant to medical content
    "american": {
        "category": "nationality",
        "patterns": [r"\bamerican (patient|citizen|m(a|e)n|wom(a|e)n)\b"],
        "fp_risk": "high",
    },
    # ... more nationality patterns, but many in BBQ don't appear in MedQA

    # Physical Appearance — SKIP "obese" and "pregnant" because they confound clinical diagnosis
    # with demographic mention. These subgroups are structurally excluded from MedQA matched analysis.
}

# Subgroups explicitly excluded from MedQA matched-condition analysis because
# pattern matches are systematically confounded with the clinical content of the item.
MEDQA_EXCLUDED_SUBGROUPS = {
    "physical_appearance/obese",      # obesity is a clinical finding, not demographic context
    "physical_appearance/pregnant",   # pregnancy is a clinical scenario, not demographic context
    "physical_appearance/underweight",
    "disability/disabled",            # disability is often the clinical topic itself
    "disability/physically disabled",
    "disability/cognitively disabled",
}
```

### Classification Function

```python
def classify_medqa_demographics(medqa_items: list[dict], output_dir: Path) -> list[dict]:
    """
    Classify each MedQA item for demographic mentions using regex patterns.
    
    Attaches a `demographic_matches` field to each item:
        [{"subgroup": "gay", "category": "so", "fp_risk": "low"}]
    """
    import re
    
    compiled = {}
    for sub_label, spec in DEMOGRAPHIC_PATTERNS.items():
        compiled[sub_label] = {
            "category": spec["category"],
            "patterns": [re.compile(p, re.IGNORECASE) for p in spec["patterns"]],
            "fp_risk": spec["fp_risk"],
        }
    
    for item in medqa_items:
        text = (item.get("prompt", "") or item.get("question", "")).lower()
        
        matches = []
        for sub_label, spec in compiled.items():
            for pat in spec["patterns"]:
                if pat.search(text):
                    matches.append({
                        "subgroup": sub_label,
                        "category": spec["category"],
                        "fp_risk": spec["fp_risk"],
                    })
                    break
        
        item["demographic_matches"] = matches
        item["mentions_demographic"] = len(matches) > 0
        item["has_low_fp_match"] = any(m["fp_risk"] == "low" for m in matches)
    
    # Log statistics
    n_demo = sum(1 for it in medqa_items if it["mentions_demographic"])
    n_low_fp = sum(1 for it in medqa_items if it.get("has_low_fp_match"))
    log(f"  Demographic classification: {n_demo}/{len(medqa_items)} items with any match")
    log(f"  Low-FP-risk matches only: {n_low_fp}/{len(medqa_items)} items")
    
    from collections import Counter
    sub_counts = Counter()
    for it in medqa_items:
        for m in it["demographic_matches"]:
            sub_counts[m["subgroup"]] += 1
    log(f"  Per-subgroup counts:")
    for sub, cnt in sub_counts.most_common():
        log(f"    {sub}: {cnt}")
    
    # Save classification for audit
    classification_out = {
        "pattern_catalog": {
            sub: {"category": spec["category"], "fp_risk": spec["fp_risk"],
                  "patterns": [p.pattern for p in spec["patterns"]]}
            for sub, spec in compiled.items()
        },
        "excluded_subgroups": sorted(MEDQA_EXCLUDED_SUBGROUPS),
        "item_classifications": {
            i: item["demographic_matches"]
            for i, item in enumerate(medqa_items)
            if item["mentions_demographic"]
        },
        "summary": {
            "total_items": len(medqa_items),
            "items_with_match": n_demo,
            "items_with_low_fp_match": n_low_fp,
            "per_subgroup_counts": dict(sub_counts),
        },
    }
    atomic_write_json(output_dir / "medqa" / "demographic_classification.json", classification_out)
    
    return medqa_items
```

**Limitations documented for the paper:**
1. Regex matching is surface-level; doesn't capture contextual mentions
2. High-false-positive-risk subgroups (marked as such) require careful interpretation
3. Some subgroups confound with clinical content and are excluded (documented in output)
4. Matched-condition results should be treated as suggestive evidence, not definitive

---

## Step 0b: MMLU Supercategory Annotation

```python
MMLU_STEM = {
    "abstract_algebra", "college_biology", "college_chemistry", "college_computer_science",
    "college_mathematics", "college_physics", "computer_security", "conceptual_physics",
    "electrical_engineering", "elementary_mathematics", "high_school_biology", "high_school_chemistry",
    "high_school_computer_science", "high_school_mathematics", "high_school_physics",
    "high_school_statistics", "machine_learning", "astronomy",
}

MMLU_HUMANITIES = {
    "formal_logic", "high_school_european_history", "high_school_us_history", "high_school_world_history",
    "international_law", "jurisprudence", "logical_fallacies", "moral_disputes", "moral_scenarios",
    "philosophy", "prehistory", "professional_law", "world_religions",
}

MMLU_SOCIAL_SCIENCES = {
    "econometrics", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_microeconomics", "high_school_psychology",
    "human_sexuality", "professional_psychology", "public_relations", "security_studies",
    "sociology", "us_foreign_policy",
}

MMLU_OTHER = {
    "anatomy", "business_ethics", "clinical_knowledge", "college_medicine", "global_facts",
    "human_aging", "management", "marketing", "medical_genetics", "miscellaneous", "nutrition",
    "professional_accounting", "professional_medicine", "virology",
}


def annotate_mmlu_supercategories(mmlu_items: list[dict]) -> list[dict]:
    """Attach supercategory label to each MMLU item."""
    for item in mmlu_items:
        subject = item.get("subject", "").strip().lower()
        if subject in MMLU_STEM:
            item["supercategory"] = "STEM"
        elif subject in MMLU_HUMANITIES:
            item["supercategory"] = "humanities"
        elif subject in MMLU_SOCIAL_SCIENCES:
            item["supercategory"] = "social_sciences"
        elif subject in MMLU_OTHER:
            item["supercategory"] = "other"
        else:
            item["supercategory"] = "uncategorized"
    
    from collections import Counter
    counts = Counter(it["supercategory"] for it in mmlu_items)
    log(f"  MMLU supercategory counts: {dict(counts)}")
    
    return mmlu_items
```

---

## Step 1: Compute and Cache Baselines

```python
def compute_or_load_baselines(
    items: list[dict],
    wrapper: ModelWrapper,
    sae_cache: dict[int, SAEWrapper],
    letters: tuple[str, ...],
    cache_path: Path,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Compute baseline forward passes for all items, or load from cache if present.
    
    Returns DataFrame with columns:
        item_idx, correct_answer, baseline_top_answer, baseline_top_logit,
        baseline_correct_logit, baseline_correct, baseline_degenerated,
        logit_A, logit_B, ..., logit_<last_letter>
    """
    if cache_path.exists():
        log(f"  Loading {dataset_name} baselines from cache: {cache_path}")
        df = pd.read_parquet(cache_path)
        if len(df) == len(items):
            return df
        log(f"  Cache size mismatch ({len(df)} vs {len(items)} items); recomputing")
    
    # Use any steerer for baseline evaluation (steerer provides baseline_mcq method; no hooks)
    first_layer = next(iter(sae_cache))
    steerer = SAESteerer(wrapper, sae_cache[first_layer], first_layer)
    
    rows = []
    for i, item in enumerate(items):
        prompt = item.get("prompt") or build_prompt_from_item(item, dataset_name)
        correct = str(item.get("answer", "")).strip().upper()
        
        baseline = steerer.evaluate_baseline_mcq(prompt, letters=letters)
        
        row = {
            "item_idx": i,
            "correct_answer": correct,
            "baseline_top_answer": baseline["model_answer"],
            "baseline_top_logit": float(max(baseline["answer_logits"].values())),
            "baseline_correct_logit": float(baseline["answer_logits"].get(correct, float("-inf"))),
            "baseline_correct": int(baseline["model_answer"] == correct),
            "baseline_degenerated": int(baseline.get("degenerated", False)),
        }
        for letter in letters:
            row[f"baseline_logit_{letter}"] = float(baseline["answer_logits"].get(letter, float("-inf")))
        
        rows.append(row)
        
        if (i + 1) % 100 == 0:
            log(f"    {dataset_name} baselines: {i+1}/{len(items)}")
    
    df = pd.DataFrame(rows)
    df.to_parquet(cache_path, index=False)
    
    n_correct = df["baseline_correct"].sum()
    log(f"  {dataset_name} baseline accuracy: {n_correct/len(df):.4f} ({n_correct}/{len(df)})")
    
    return df
```

Baselines are deterministic (no stochasticity in MCQ scoring). Cache is reused across all C3 runs. If you need to invalidate (model/prompt change), delete the cache file.

---

## Step 2: Per-Vector Evaluation

```python
def run_dataset_evaluation(
    dataset_name: str,
    items: list[dict],
    baselines_df: pd.DataFrame,
    vec_info: dict,
    sae_cache: dict,
    wrapper: ModelWrapper,
    output_dir: Path,
    letters: tuple[str, ...],
    skip_exacerbation: bool = False,
) -> list[dict]:
    """
    Evaluate one steering vector on a full dataset.
    
    Checkpoint granularity: one JSON per (vector, dataset) combining debias + exac.
    """
    cat = vec_info["category"]
    sub = vec_info["subgroup"]
    vec = vec_info["vector"]  # already on device, dtype
    exac_vec = -vec
    injection_layer = vec_info["injection_layer"]
    
    ckpt_path = output_dir / dataset_name / "per_vector_checkpoints" / f"{cat}_{sub}.json"
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            cached = json.load(f)
        log(f"  {dataset_name} {cat}/{sub}: LOADED from checkpoint")
        return cached["records"]
    
    steerer = SAESteerer(wrapper, sae_cache[injection_layer], injection_layer)
    records = []
    
    directions = [("debiasing", vec)]
    if not skip_exacerbation:
        directions.append(("exacerbation", exac_vec))
    
    for direction_name, direction_vec in directions:
        for i, item in enumerate(items):
            prompt = item.get("prompt") or build_prompt_from_item(item, dataset_name)
            correct = str(item.get("answer", "")).strip().upper()
            
            # Baseline already cached; just look up
            baseline_row = baselines_df.iloc[i]
            
            # Steered forward pass
            steered = steerer.steer_and_evaluate(prompt, direction_vec, letters=letters)
            steered_correct_logit = float(steered["answer_logits"].get(correct, float("-inf")))
            baseline_correct_logit = float(baseline_row["baseline_correct_logit"])
            
            record = {
                "item_idx": i,
                "dataset": dataset_name,
                "steering_vector": f"{cat}/{sub}",
                "steering_category": cat,
                "steering_subgroup": sub,
                "direction": direction_name,
                "correct_answer": correct,
                "baseline_top_answer": baseline_row["baseline_top_answer"],
                "baseline_top_logit": float(baseline_row["baseline_top_logit"]),
                "baseline_correct_logit": baseline_correct_logit,
                "baseline_correct": int(baseline_row["baseline_correct"]),
                "baseline_degenerated": int(baseline_row["baseline_degenerated"]),
                "steered_top_answer": steered["model_answer"],
                "steered_top_logit": float(max(steered["answer_logits"].values())),
                "steered_correct_logit": steered_correct_logit,
                "steered_correct": int(steered["model_answer"] == correct),
                "steered_degenerated": int(steered.get("degenerated", False)),
                "correct_logit_shift": steered_correct_logit - baseline_correct_logit,
                "top_logit_shift": float(max(steered["answer_logits"].values())) - float(baseline_row["baseline_top_logit"]),
                "flipped": int(steered["model_answer"] != baseline_row["baseline_top_answer"]),
                "correctness_delta": int(steered["model_answer"] == correct) - int(baseline_row["baseline_correct"]),
            }
            
            # Per-letter logits (for supplementary analysis)
            for letter in letters:
                record[f"baseline_logit_{letter}"] = float(baseline_row[f"baseline_logit_{letter}"])
                record[f"steered_logit_{letter}"] = float(steered["answer_logits"].get(letter, float("-inf")))
            
            # Add dataset-specific fields
            if dataset_name == "medqa":
                record["demographic_matches"] = json.dumps(item.get("demographic_matches", []))
                record["mentions_demographic"] = int(item.get("mentions_demographic", False))
                record["has_low_fp_match"] = int(item.get("has_low_fp_match", False))
                record["condition"] = compute_medqa_condition(item, cat, sub)
            elif dataset_name == "mmlu":
                record["subject"] = item.get("subject", "")
                record["supercategory"] = item.get("supercategory", "")
            
            records.append(record)
        
        n_items = len(items)
        n_flipped = sum(1 for r in records if r["direction"] == direction_name and r["flipped"])
        log(f"  {dataset_name} {cat}/{sub} [{direction_name}]: {n_flipped}/{n_items} items flipped")
    
    atomic_write_json(ckpt_path, {"records": records})
    return records


def compute_medqa_condition(item: dict, vec_cat: str, vec_sub: str) -> str:
    """
    Determine the condition label for this (item, vector) pair:
      - "matched": item mentions vec_sub
      - "within_cat_mismatched": item mentions DIFFERENT subgroup from SAME category
      - "cross_cat_mismatched": item mentions subgroup from DIFFERENT category
      - "no_demographic": item has no demographic mentions
      - "excluded": vec_sub is in MEDQA_EXCLUDED_SUBGROUPS
    """
    key = f"{vec_cat}/{vec_sub}"
    if key in MEDQA_EXCLUDED_SUBGROUPS:
        return "excluded"
    
    matches = item.get("demographic_matches", [])
    if not matches:
        return "no_demographic"
    
    match_subs = [m["subgroup"] for m in matches]
    match_cats = [m["category"] for m in matches]
    
    if vec_sub in match_subs:
        return "matched"
    elif vec_cat in match_cats:
        return "within_cat_mismatched"
    else:
        return "cross_cat_mismatched"
```

---

## Step 3: Save Per-Item Parquets

```python
def save_per_item_parquet(records: list[dict], path: Path, dataset: str):
    """Save all per-item records to parquet. One row per (item, vector, direction)."""
    df = pd.DataFrame(records)
    df.to_parquet(path, index=False, compression="snappy")
    log(f"  Saved {len(df)} rows to {path}")
```

### MedQA per-item parquet columns

| Column | Type | Description |
|---|---|---|
| `item_idx` | int32 | MedQA item index |
| `dataset` | string | "medqa" |
| `steering_vector` | string | "{cat}/{sub}" |
| `steering_category` | string | |
| `steering_subgroup` | string | |
| `direction` | string | "debiasing" or "exacerbation" |
| `condition` | string | matched / within_cat_mismatched / cross_cat_mismatched / no_demographic / excluded |
| `mentions_demographic` | int32 | 0/1 |
| `has_low_fp_match` | int32 | 0/1 (only low-FP-risk matches present) |
| `demographic_matches` | string (JSON) | List of matched subgroups |
| `correct_answer` | string | |
| `baseline_top_answer` | string | |
| `baseline_top_logit` | float32 | |
| `baseline_correct_logit` | float32 | |
| `baseline_correct` | int32 | 0/1 |
| `baseline_degenerated` | int32 | 0/1 |
| `steered_top_answer` | string | |
| `steered_top_logit` | float32 | |
| `steered_correct_logit` | float32 | |
| `steered_correct` | int32 | |
| `steered_degenerated` | int32 | |
| `correct_logit_shift` | float32 | **Primary continuous metric** |
| `top_logit_shift` | float32 | |
| `flipped` | int32 | |
| `correctness_delta` | int32 | -1, 0, +1 |
| `baseline_logit_A, ..., baseline_logit_E` | float32 | |
| `steered_logit_A, ..., steered_logit_E` | float32 | |

### MMLU per-item parquet

Same columns except:
- No `condition`, `demographic_matches`, `mentions_demographic`, `has_low_fp_match`
- Adds `subject` (string) and `supercategory` (string)
- Only `baseline_logit_A..D` and `steered_logit_A..D` (4 options)

---

## Step 4: Aggregate Results

Compute aggregated metrics from per-item parquets.

### MedQA Aggregation

```python
def aggregate_medqa(per_item_df: pd.DataFrame, output_dir: Path) -> dict:
    """
    Aggregate per-item results into per-vector summary.
    
    Reports:
        - accuracy_delta (steered - baseline), bootstrap 95% CI
        - mean correct_logit_shift, bootstrap 95% CI
        - flip rate
        - per-condition breakdown
    """
    results = {}
    
    for (vec, direction), grp in per_item_df.groupby(["steering_vector", "direction"]):
        vec_result = {
            "steering_vector": vec,
            "direction": direction,
            "n_items_total": len(grp),
            "overall": aggregate_group(grp),
            "per_condition": {},
        }
        
        for cond, cond_grp in grp.groupby("condition"):
            if len(cond_grp) < 10:
                vec_result["per_condition"][cond] = {
                    "n": len(cond_grp),
                    "status": "n_insufficient",
                }
                continue
            
            vec_result["per_condition"][cond] = aggregate_group(cond_grp)
        
        # Also: low-FP-risk subset for matched items (extra-stringent filter)
        matched_low_fp = grp[(grp["condition"] == "matched") & (grp["has_low_fp_match"] == 1)]
        if len(matched_low_fp) >= 10:
            vec_result["matched_low_fp_only"] = aggregate_group(matched_low_fp)
        
        results[f"{vec}_{direction}"] = vec_result
    
    atomic_write_json(output_dir / "medqa" / "aggregated_results.json", results)
    return results


def aggregate_group(df: pd.DataFrame) -> dict:
    """Compute aggregated statistics for a group of per-item records."""
    n = len(df)
    if n == 0:
        return {"n": 0}
    
    # Accuracy
    baseline_acc = float(df["baseline_correct"].mean())
    steered_acc = float(df["steered_correct"].mean())
    accuracy_delta = steered_acc - baseline_acc
    
    # Logit shifts
    logit_shifts = df["correct_logit_shift"].values
    mean_logit_shift = float(np.mean(logit_shifts))
    median_logit_shift = float(np.median(logit_shifts))
    std_logit_shift = float(np.std(logit_shifts))
    
    # Flip rate and directional breakdown
    flip_rate = float(df["flipped"].mean())
    n_correct_to_wrong = int(((df["baseline_correct"] == 1) & (df["steered_correct"] == 0)).sum())
    n_wrong_to_correct = int(((df["baseline_correct"] == 0) & (df["steered_correct"] == 1)).sum())
    n_degenerated = int(df["steered_degenerated"].sum())
    
    # Bootstrap CIs
    rng = np.random.default_rng(42)
    n_boot = 1000
    
    boot_acc_deltas = []
    boot_mean_shifts = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        sample = df.iloc[idx]
        boot_acc_deltas.append(float(sample["steered_correct"].mean() - sample["baseline_correct"].mean()))
        boot_mean_shifts.append(float(sample["correct_logit_shift"].mean()))
    
    return {
        "n": n,
        "baseline_accuracy": round(baseline_acc, 4),
        "steered_accuracy": round(steered_acc, 4),
        "accuracy_delta": round(accuracy_delta, 4),
        "accuracy_delta_ci_95": [round(float(np.percentile(boot_acc_deltas, 2.5)), 4),
                                  round(float(np.percentile(boot_acc_deltas, 97.5)), 4)],
        "mean_correct_logit_shift": round(mean_logit_shift, 4),
        "median_correct_logit_shift": round(median_logit_shift, 4),
        "std_correct_logit_shift": round(std_logit_shift, 4),
        "mean_shift_ci_95": [round(float(np.percentile(boot_mean_shifts, 2.5)), 4),
                              round(float(np.percentile(boot_mean_shifts, 97.5)), 4)],
        "flip_rate": round(flip_rate, 4),
        "n_correct_to_wrong": n_correct_to_wrong,
        "n_wrong_to_correct": n_wrong_to_correct,
        "n_degenerated": n_degenerated,
    }
```

### MMLU Aggregation

```python
def aggregate_mmlu(per_item_df: pd.DataFrame, output_dir: Path) -> dict:
    """Per-vector, per-subject, and per-supercategory aggregation."""
    results = {}
    
    for (vec, direction), grp in per_item_df.groupby(["steering_vector", "direction"]):
        vec_result = {
            "steering_vector": vec,
            "direction": direction,
            "n_items_total": len(grp),
            "overall": aggregate_group(grp),
            "per_supercategory": {},
            "per_subject": {},
            "worst_subject": None,
        }
        
        for supercat, sc_grp in grp.groupby("supercategory"):
            vec_result["per_supercategory"][supercat] = aggregate_group(sc_grp)
        
        per_subject = {}
        for subj, sj_grp in grp.groupby("subject"):
            if len(sj_grp) < 10:
                per_subject[subj] = {"n": len(sj_grp), "status": "n_insufficient"}
                continue
            per_subject[subj] = aggregate_group(sj_grp)
        vec_result["per_subject"] = per_subject
        
        # Worst subject (min accuracy_delta, requires N≥10)
        eligible = {s: v for s, v in per_subject.items() if v.get("n", 0) >= 10}
        if eligible:
            worst = min(eligible, key=lambda s: eligible[s]["accuracy_delta"])
            vec_result["worst_subject"] = worst
            vec_result["worst_subject_delta"] = eligible[worst]["accuracy_delta"]
        
        results[f"{vec}_{direction}"] = vec_result
    
    atomic_write_json(output_dir / "mmlu" / "aggregated_results.json", results)
    return results
```

---

## Step 5: Update Manifests

```python
def update_manifests(run_dir: Path, output_dir: Path):
    """Add generalization-specific fields to each viable subgroup manifest."""
    with open(run_dir / "C_steering" / "steering_manifests.json") as f:
        manifests = json.load(f)
    
    with open(output_dir / "medqa" / "aggregated_results.json") as f:
        medqa_results = json.load(f)
    
    with open(output_dir / "mmlu" / "aggregated_results.json") as f:
        mmlu_results = json.load(f)
    
    for m in manifests:
        if not m.get("steering_viable"):
            continue
        
        vec_key = f"{m['category']}/{m['subgroup']}"
        medqa_debias = medqa_results.get(f"{vec_key}_debiasing", {})
        medqa_exac = medqa_results.get(f"{vec_key}_exacerbation", {})
        mmlu_debias = mmlu_results.get(f"{vec_key}_debiasing", {})
        mmlu_exac = mmlu_results.get(f"{vec_key}_exacerbation", {})
        
        # MedQA fields
        m["medqa_overall_debias_delta"] = medqa_debias.get("overall", {}).get("accuracy_delta")
        m["medqa_overall_exac_delta"] = medqa_exac.get("overall", {}).get("accuracy_delta")
        m["medqa_overall_debias_logit_shift"] = medqa_debias.get("overall", {}).get("mean_correct_logit_shift")
        m["medqa_overall_exac_logit_shift"] = medqa_exac.get("overall", {}).get("mean_correct_logit_shift")
        
        for cond in ["matched", "within_cat_mismatched", "cross_cat_mismatched", "no_demographic"]:
            cond_debias = medqa_debias.get("per_condition", {}).get(cond, {})
            cond_exac = medqa_exac.get("per_condition", {}).get(cond, {})
            m[f"medqa_{cond}_debias_delta"] = cond_debias.get("accuracy_delta")
            m[f"medqa_{cond}_exac_delta"] = cond_exac.get("accuracy_delta")
            m[f"medqa_{cond}_debias_logit_shift"] = cond_debias.get("mean_correct_logit_shift")
            m[f"medqa_{cond}_exac_logit_shift"] = cond_exac.get("mean_correct_logit_shift")
            m[f"medqa_{cond}_n"] = cond_debias.get("n")
        
        # MMLU fields
        m["mmlu_overall_debias_delta"] = mmlu_debias.get("overall", {}).get("accuracy_delta")
        m["mmlu_overall_exac_delta"] = mmlu_exac.get("overall", {}).get("accuracy_delta")
        m["mmlu_worst_subject_debias"] = mmlu_debias.get("worst_subject")
        m["mmlu_worst_subject_debias_delta"] = mmlu_debias.get("worst_subject_delta")
        
        for supercat in ["STEM", "humanities", "social_sciences", "other"]:
            sc_debias = mmlu_debias.get("per_supercategory", {}).get(supercat, {})
            sc_exac = mmlu_exac.get("per_supercategory", {}).get(supercat, {})
            m[f"mmlu_{supercat}_debias_delta"] = sc_debias.get("accuracy_delta")
            m[f"mmlu_{supercat}_exac_delta"] = sc_exac.get("accuracy_delta")
    
    atomic_write_json(output_dir / "manifests_with_generalization.json", manifests)
```

---

## Step 6: Top-Impact Items (Supplementary)

Identify items with the largest steering effects across all vectors. Useful for qualitative inspection.

```python
def identify_top_impact_items(output_dir: Path):
    """
    Rank (item, vector) pairs by absolute correct_logit_shift.
    Save top-100 per dataset for supplementary analysis.
    """
    for dataset in ["medqa", "mmlu"]:
        parquet_path = output_dir / dataset / "per_item.parquet"
        if not parquet_path.exists():
            continue
        df = pd.read_parquet(parquet_path)
        
        df["abs_logit_shift"] = df["correct_logit_shift"].abs()
        top100 = df.nlargest(100, "abs_logit_shift").reset_index(drop=True)
        
        out_path = output_dir / dataset / "top_impact_items.parquet"
        top100.to_parquet(out_path, index=False)
        log(f"  Saved top-100 impact items for {dataset}: {out_path}")
```

Top items can be inspected manually to understand what the steering vectors are actually doing on real medical/knowledge content.

---

## Output Files

```
{run}/C_generalization/
├── baselines/
│   ├── medqa_baselines.parquet                       # Cached baselines, item_idx × letter_logits
│   └── mmlu_baselines.parquet
├── medqa/
│   ├── per_item.parquet                              # All per-(item, vector, direction) records
│   ├── per_vector_checkpoints/
│   │   └── {cat}_{sub}.json                         # Resume checkpoints
│   ├── aggregated_results.json                       # Per-vector, per-condition metrics
│   ├── demographic_classification.json               # Pattern catalog + per-item matches
│   └── top_impact_items.parquet                      # Top 100 by |correct_logit_shift|
├── mmlu/
│   ├── per_item.parquet
│   ├── per_vector_checkpoints/
│   │   └── {cat}_{sub}.json
│   ├── aggregated_results.json                       # Per-vector, per-subject, per-supercategory
│   └── top_impact_items.parquet
├── manifests_with_generalization.json                # Updated C1 manifests + C3 fields
└── figures/
    ├── fig_medqa_conditions_comparison.png/.pdf
    ├── fig_medqa_debias_vs_exacerbation.png/.pdf
    ├── fig_medqa_logit_shift_distributions.png/.pdf
    ├── fig_mmlu_supercategory_heatmap.png/.pdf
    ├── fig_side_effect_heatmap.png/.pdf
    ├── fig_bbq_vs_medqa_debiasing.png/.pdf
    └── fig_bbq_vs_medqa_exacerbation.png/.pdf
```

---

## Figures

All figures use Wong colorblind-safe palette.

### `fig_medqa_conditions_comparison.png`

One panel per category with ≥1 viable subgroup that has matched items.
- Grouped bars per subgroup: 4 bars for {matched, within_cat_mismatched, cross_cat_mismatched, no_demographic}
- Y: `accuracy_delta` (debiasing direction)
- Bootstrap 95% CI error bars
- Annotate n per bar
- Horizontal dashed line at y=0
- Colors per condition:
  - matched: Wong blue #0072B2
  - within_cat_mismatched: Wong orange #E69F00
  - cross_cat_mismatched: Wong green #009E73
  - no_demographic: Wong vermillion #D55E00
- Title per panel: category name

### `fig_medqa_debias_vs_exacerbation.png`

One panel per category. Paired bars per subgroup:
- Left: matched_debias_delta
- Right: matched_exac_delta
- Blue vs vermillion coloring
- N counts annotated
- Title: asymmetry comparison

### `fig_medqa_logit_shift_distributions.png`

For all viable vectors with ≥20 matched items, violin plot of `correct_logit_shift` distribution.
- X: subgroup (grouped by category, sorted)
- Y: correct_logit_shift
- Split by direction (debiasing left, exacerbation right)
- Horizontal dashed line at y=0
- Boxplot overlay showing median and IQR
- Shows the full distribution of effects, not just means

### `fig_mmlu_supercategory_heatmap.png`

Heatmap:
- Rows: steering vectors (ordered by category)
- Columns: MMLU supercategories (STEM, humanities, social_sciences, other)
- Color: `accuracy_delta` under debiasing (RdBu_r, centered at 0)
- Annotate cells with values
- Separator lines between categories on the rows
- Title: "Capability tax across MMLU supercategories"

### `fig_side_effect_heatmap.png`

Heatmap showing collateral effects (5 columns):
- Rows: steering vectors
- Columns: MedQA no-demographic, MMLU STEM, MMLU humanities, MMLU social_sciences, MMLU other
- Color: accuracy_delta (RdBu_r, vmin=-0.1, vmax=0.1)
- Annotated cells
- Title: "Side effects on unrelated content (debiasing direction)"

### `fig_bbq_vs_medqa_debiasing.png`

Scatter plot.
- X: BBQ optimal RCR_1.0 from C1 (debiasing effectiveness on BBQ)
- Y: MedQA matched_debias_delta from C3
- One point per viable subgroup
- Color by category
- Annotate Pearson r
- Linear fit line
- Interpretation annotation:
  - "Positive correlation: features are bias-specific (BBQ effects transfer to MedQA)"
  - "Near-zero: features are BBQ-specific"

### `fig_bbq_vs_medqa_exacerbation.png`

Same structure but with exacerbation direction.
- X: BBQ RCR_1.0 (still baseline effectiveness)
- Y: MedQA matched_exac_delta (negative = exacerbation hurt accuracy on matched items)
- Shows whether features that work for BBQ debiasing can be reversed to hurt MedQA performance
- If Y is strongly negative and correlated with X: features capture genuinely bias-relevant content

---

## Resume Safety

- **Baseline caching** at `baselines/{dataset}_baselines.parquet`. Loaded on startup if cache size matches item count; recomputed otherwise.
- **Per-vector checkpoints** at `{dataset}/per_vector_checkpoints/{cat}_{sub}.json`. One checkpoint per vector (debiasing + exacerbation consolidated).
- Atomic writes throughout.
- Restarting resumes at the next unprocessed vector without recomputing baselines or completed vectors.

---

## Compute Estimate (M4 Max MPS)

- Baseline pass per item: ~0.2s
- Steered pass per item: ~0.3s

MedQA (assume 1000 items):
- Baselines: 1000 × 0.2 = ~3 min (one-time)
- Per vector: 1000 items × 0.3s × 2 directions = ~10 min
- 40 vectors: ~7 hours

MMLU (assume 2000 items):
- Baselines: 2000 × 0.2 = ~7 min
- Per vector: 2000 × 0.3 × 2 = ~20 min
- 40 vectors: ~13 hours

**Total: ~20 hours on MPS, ~5 hours on CUDA.**

Mitigation: `--max_medqa_items 500` and `--max_mmlu_items 1000` cut by ~half.

---

## Assumptions Summary

| # | Decision | Value |
|---|---|---|
| 1 | Input vectors | From `C_steering/vectors/{cat}_{sub}.npz` |
| 2 | Viable filter | Only steering_viable=True subgroups |
| 3 | Sign convention | Saved vec = debiasing (α<0); exac = -vec |
| 4 | Primary metric | Item-level `correct_logit_shift` (continuous) |
| 5 | Secondary metric | `accuracy_delta` (discrete) with bootstrap CI |
| 6 | Demographic classification | Regex patterns with fp_risk annotations |
| 7 | Excluded subgroups | Disability + {obese, pregnant, underweight} from MedQA matched analysis |
| 8 | Conditions | matched, within_cat_mismatched, cross_cat_mismatched, no_demographic, excluded |
| 9 | Minimum N per condition | 10 items required to report a delta |
| 10 | MMLU supercategories | Hardcoded from MMLU paper (STEM/humanities/social/other) |
| 11 | Worst subject min N | 10 items required |
| 12 | Baseline caching | Parquet, reloaded on resume |
| 13 | Per-vector checkpoint | One JSON per (vector, dataset) with debias+exac combined |
| 14 | Bootstrap trials | 1000, fixed seed 42 |
| 15 | Top-impact items | Top 100 by |correct_logit_shift| per dataset, supplementary |
| 16 | Per-letter logits stored | Yes, for all answer letters, baseline and steered |

---

## Test Command

```bash
# Smoke test: few vectors, small item counts
python scripts/C3_generalization.py \
    --run_dir runs/llama-3.1-8b_2026-04-15/ \
    --subgroups so/gay,so/bisexual \
    --max_medqa_items 50 \
    --max_mmlu_items 100

# Verify
python -c "
import pandas as pd
import json

medqa = pd.read_parquet('runs/llama-3.1-8b_2026-04-15/C_generalization/medqa/per_item.parquet')
print(f'MedQA records: {len(medqa)}')
print(f'Vectors: {medqa[\"steering_vector\"].unique()}')
print(f'Directions: {medqa[\"direction\"].unique()}')
print(f'Conditions: {medqa[\"condition\"].value_counts().to_dict()}')

mmlu = pd.read_parquet('runs/llama-3.1-8b_2026-04-15/C_generalization/mmlu/per_item.parquet')
print(f'\\nMMLU records: {len(mmlu)}')
print(f'Supercategories: {mmlu[\"supercategory\"].value_counts().to_dict()}')

# Logit shift distribution
debias = medqa[medqa['direction'] == 'debiasing']
print(f'\\nDebiasing correct_logit_shift: mean={debias[\"correct_logit_shift\"].mean():+.3f}, median={debias[\"correct_logit_shift\"].median():+.3f}')

with open('runs/llama-3.1-8b_2026-04-15/C_generalization/medqa/aggregated_results.json') as f:
    agg = json.load(f)
for key, result in list(agg.items())[:3]:
    print(f'\\n{key}: overall acc_delta = {result[\"overall\"][\"accuracy_delta\"]:+.4f}')
"
```