runs/<model_id>_<date>/
│
├── config.json                          # Frozen config for this run
│   {model_path, model_id, sae_source, sae_expansion, device,
│    categories, date, bbq_data_dir, medqa_path, mmlu_path}
│
├── A_extraction/
│   ├── stimuli/                         # Processed BBQ items
│   │   ├── so.json
│   │   ├── race.json
│   │   └── ...
│   ├── activations/                     # Per-item hidden states
│   │   ├── so/
│   │   │   ├── item_0000.npz           # {hidden_states, raw_norms, metadata_json}
│   │   │   └── ...
│   │   ├── race/
│   │   └── ...
│   └── extraction_summary.json
│
├── A_sae_encoding/
│   ├── layer_00/
│   │   ├── so/
│   │   │   ├── item_0000.npz           # {feature_activations (sparse), feature_indices, activation_values}
│   │   │   └── ...                      # or a single so.parquet with item_idx × feature columns
│   │   └── ...
│   ├── layer_02/
│   ├── ...
│   ├── layer_30/
│   └── encoding_summary.json            # {layers_encoded, items_per_category, sparsity_stats}
│
├── B_differential/
│   ├── per_subgroup/
│   │   ├── layer_00.parquet             # feature_idx, category, subcategory, cohens_d, p_fdr, 
│   │   ├── layer_02.parquet             #   firing_rate_stereo, firing_rate_nonstereo, direction, is_significant
│   │   └── ...
│   ├── per_category/
│   │   ├── layer_00.parquet
│   │   └── ...
│   └── differential_summary.json
│
├── B_feature_ranking/
│   ├── ranked_features_by_subgroup.json  # {cat: {sub: {pro_bias: [...], anti_bias: [...]}}}
│   ├── injection_layers.json             # {cat/sub: {layer, distribution}}
│   ├── feature_overlap.json
│   └── figures/
│       ├── fig_feature_overlap_*.png/pdf
│       ├── fig_feature_layer_distribution.png/pdf
│       └── fig_ranked_effect_sizes_*.png/pdf
│
├── B_geometry/
│   ├── subgroup_directions.npz           # DIM directions per subgroup per layer
│   ├── subgroup_directions.json          # Summary with pairwise cosines
│   ├── cosine_matrices/
│   │   ├── so_layer_14.json
│   │   └── ...
│   └── figures/
│       └── fig_cosine_heatmaps_*.png/pdf
│
├── B_probes/
│   ├── identity_probes.json              # Real probe accuracies per layer per category
│   ├── permutation_baselines.json        # Control A results
│   ├── structural_controls.json          # Control B results
│   ├── generalization_matrix.json        # Control D results
│   └── figures/
│       ├── fig_probe_selectivity.png/pdf
│       ├── fig_probe_structural_comparison.png/pdf
│       └── fig_probe_generalization_matrix.png/pdf
│
├── B_feature_interpretability/
│   ├── feature_reports.json              # Per-feature: top items, activation stats, specificity
│   ├── cross_subgroup_activation.json    # Per-category activation matrices
│   └── figures/
│       ├── fig_cross_subgroup_activation_*.png/pdf
│       └── fig_specificity_distribution.png/pdf
│
├── C_steering/
│   ├── stepwise_results.json             # Full k×α grid per subgroup
│   ├── optimal_configs.json              # {cat: {sub: {k, alpha, eta, ...}}}
│   ├── steering_manifests.json           # Complete manifests
│   ├── vectors/
│   │   ├── so_gay.npz
│   │   ├── so_bisexual.npz
│   │   └── ...
│   ├── per_item/
│   │   ├── so_gay_optimal.parquet
│   │   └── ...
│   └── figures/
│       ├── fig_pareto_frontier_*.png/pdf
│       ├── fig_stepwise_correction_*.png/pdf
│       ├── fig_margin_conditioned_*.png/pdf
│       └── fig_exacerbation_asymmetry.png/pdf
│
├── C_transfer/
│   ├── transfer_effects/
│   │   ├── so_gay_to_bisexual.json
│   │   └── ...
│   ├── universal_scatter_data.json
│   ├── regression_results.json
│   └── figures/
│       ├── fig_universal_backfire_scatter.png/pdf
│       ├── fig_transfer_heatmaps_*.png/pdf
│       └── fig_cosine_vs_backfire_by_category.png/pdf
│
├── C_generalization/
│   ├── medqa/
│   │   ├── per_item.parquet
│   │   ├── results_by_vector.json
│   │   └── demographic_classification.json
│   ├── mmlu/
│   │   ├── per_item.parquet
│   │   └── results_by_vector.json
│   ├── manifests_with_generalization.json
│   └── figures/
│       ├── fig_medqa_matched_vs_mismatched.png/pdf
│       ├── fig_side_effect_heatmap.png/pdf
│       └── fig_debiasing_vs_exacerbation.png/pdf
│
├── C_token_features/
│   ├── token_activations/
│   │   ├── so_gay_feature_45021.json     # Per-item, per-token activations
│   │   └── ...
│   └── figures/
│       ├── fig_token_activations_*.png/pdf
│       └── fig_token_feature_specificity.png/pdf
│
└── paper/
    ├── all_figures/                       # Symlinks or copies of every figure for easy LaTeX access
    └── results_tables.json                # Machine-readable summary for table generation


# Stage Contracts: 

A1_prepare_stimuli
  reads:  datasets/bbq/data/*.jsonl
  writes: {run}/A_extraction/stimuli/{category}.json
  
A2_extract_activations
  reads:  {run}/A_extraction/stimuli/*.json, model
  writes: {run}/A_extraction/activations/{category}/item_*.npz
  
A3_sae_encode
  reads:  {run}/A_extraction/activations/{category}/item_*.npz, SAE checkpoints
  writes: {run}/A_sae_encoding/layer_{NN}/{category}/item_*.npz
  NOTE:   No model needed. Just SAE encoder matrix multiply on saved hidden states.
          Run at ALL layers (or every-other: 0,2,4,...,30 for 16 layers).
          
B1_differential
  reads:  {run}/A_sae_encoding/layer_*/{category}/item_*.npz,
          {run}/A_extraction/activations/{category}/item_*.npz (for behavioral labels)
  writes: {run}/B_differential/per_subgroup/layer_*.parquet,
          {run}/B_differential/per_category/layer_*.parquet
  NOTE:   Subgroup-first. Groups items by stereotyped_groups[0], not by category.
          The category column exists in the parquet but the primary grouping is subgroup.
          
B2_rank_features
  reads:  {run}/B_differential/per_subgroup/layer_*.parquet
  writes: {run}/B_feature_ranking/*.json, figures
  
B3_geometry
  reads:  {run}/A_extraction/activations/{category}/item_*.npz,
          {run}/A_extraction/stimuli/*.json
  writes: {run}/B_geometry/*.npz, *.json, figures
  NOTE:   This is the DIM direction computation — independent of SAE.
  
B4_probes
  reads:  {run}/A_extraction/activations/{category}/item_*.npz
  writes: {run}/B_probes/*.json, figures
  NOTE:   No model needed. Includes permutation, structural, cross-category controls.
  
B5_feature_interpretability
  reads:  {run}/A_sae_encoding/layer_*/{category}/item_*.npz,
          {run}/B_feature_ranking/ranked_features_by_subgroup.json,
          {run}/A_extraction/stimuli/*.json
  writes: {run}/B_feature_interpretability/*.json, figures
  NOTE:   Item-level only. Which items activate each feature most?
          Cross-subgroup activation matrices. Specificity scores.
          Filter out template tokens (invariant positions across items).
          
C1_steering
  reads:  {run}/B_feature_ranking/ranked_features_by_subgroup.json,
          {run}/A_extraction/stimuli/*.json,
          {run}/A_extraction/activations/{category}/item_*.npz (for behavioral labels),
          model, SAE checkpoints
  writes: {run}/C_steering/*.json, vectors/*.npz, per_item/*.parquet, figures
  NOTE:   Joint (k,α) optimization using η = RCR₁.₀ / ‖v‖₂.
          All confidence-aware metrics computed.
          Exacerbation runs by default.
          
C2_transfer
  reads:  {run}/C_steering/vectors/*.npz,
          {run}/B_geometry/subgroup_directions.npz (for cosines),
          {run}/A_extraction/stimuli/*.json,
          model
  writes: {run}/C_transfer/*.json, figures
  NOTE:   Cross-subgroup steering → universal backfire scatter.
  
C3_generalization
  reads:  {run}/C_steering/vectors/*.npz,
          {run}/C_steering/steering_manifests.json,
          datasets/medqa/, datasets/mmlu/,
          model
  writes: {run}/C_generalization/*.json, *.parquet, figures
  
C4_token_features
  reads:  {run}/B_feature_ranking/ranked_features_by_subgroup.json,
          {run}/C_steering/optimal_configs.json,
          {run}/A_extraction/stimuli/*.json,
          model, SAE checkpoints
  writes: {run}/C_token_features/*.json, figures
  NOTE:   Per-token SAE activations via model hooks.
          Only top-3 features per subgroup.
          Filters template tokens.
