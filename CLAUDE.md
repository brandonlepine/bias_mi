# CLAUDE.md — lgbtqmi project

## What This Project Is
Mechanistic interpretability study of LGBTQ+ identity representations in LLMs. Uses BBQ benchmark to extract activations, analyze representational geometry (fragmentation, gender decomposition), and perform causal interventions (direction ablation, head ablation). Compares across models: Llama-2-7B/13B/13B-chat, Llama-3.1-8B, Qwen-2.5-7B, Mistral-7B-v0.3. Target: NeurIPS 2026.

## Development Environment
- Local: macOS, Apple M4 Max 128GB, MPS backend
- Remote: RunPod H100 80GB, CUDA
- Python 3.11+, PyTorch, transformers (HuggingFace)
- Models stored locally at `/Users/brandonlepine/Repositories/Research_Repositories/lgbtqmi/models/` or on RunPod at `/workspace/lgbtqmi/models/`

## Commands
```bash
# Run full pipeline for a model
python scripts/run_pipeline.py --model_path /path/to/model --model_id llama2-13b --device cuda

# Run individual stages
python scripts/behavioral_pilot.py --model_path ... --device cuda
python scripts/extract_activations.py --model_path ... --stimuli stimuli_so.json --output_subdir so --device cuda
python scripts/analyze_fragmentation.py --run_dir results/runs/llama2-13b/2026-04-10/
python scripts/analyze_decomposition.py --run_dir ... --device cuda
python scripts/causal_analysis.py --run_dir ... --analysis sign_flip_only --device cuda
python scripts/ablate_heads.py --run_dir ... --device cuda

# Test with small subset first
python scripts/run_pipeline.py --model_path ... --max_items 20 --device cuda
```

## Architecture-Critical: Model Wrapper
**NEVER access model internals directly.** Always use the `ModelWrapper` class in `src/models/wrapper.py`. The previous codebase had a critical bug where hooks on `model.model.layers[N]` worked on Llama but silently failed on Mistral and Qwen because the output tuple structure differs.

The wrapper provides:
- `wrapper.get_layer(n)` — decoder layer module
- `wrapper.get_o_proj(n)` — attention output projection  
- `wrapper.n_layers`, `wrapper.n_heads`, `wrapper.head_dim`
- `wrapper.register_residual_hook(layer_idx, hook_fn)` — validated hook registration
- `wrapper.register_head_ablation(layer_idx, head_indices)` — zero specific attention heads
- `wrapper.validate_hooks()` — run a forward pass and confirm hooks fired

After registering any hook, ALWAYS call `wrapper.validate_hooks()` before running the full experiment. This runs a single forward pass and confirms the hook modified the output.

## Code Style
- All `print()` calls must use `flush=True` (for tee piping): `log()` helper everywhere
- Incremental saving after every condition — never batch-save at the end
- Atomic file writes: write to `.tmp`, then `os.rename()`
- argparse for all CLI scripts with standard flags: `--model_path`, `--model_id`, `--device`, `--run_dir`, `--max_items`
- Type hints on all function signatures
- No wildcard imports

## Key Conventions
- Activations: float32 `.npz` files, one per item, under `activations/{so,gi}/`
- Directions: unit-normalized per layer after computing mean delta
- Gender decomposition: Gram-Schmidt via `(d_gay - d_lesbian) / 2`, then project out
- Bias scores: BBQ formula `2 * (n_stereo / n_non_unknown) - 1`
- Figures: Wong colorblind palette (#E69F00 orange, #0072B2 blue, #009E73 green, #CC79A7 purple), matplotlib Agg backend, 150 DPI
- All results namespaced: `results/runs/{model_id}/{run_date}/`

## Things That Break Silently (Watch For These)
1. **Hook output structure**: Llama output[0] is the hidden state. Other architectures may differ. The wrapper must inspect and validate this.
2. **Layer indexing**: Qwen has 28 layers, Llama-7B has 32, Llama-13B has 40. Scripts that hardcode layer indices (e.g., `target_layer=20`) must be parameterized relative to model depth (e.g., `layer = int(0.5 * n_layers)`).
3. **Identity term BPE**: "bisexual" tokenizes differently across tokenizers. BPE subsequence matching must use the model's own tokenizer, not string matching on decoded text.
4. **torch_dtype is deprecated**: Use `dtype=torch.float16` in `from_pretrained()`.
5. **MPS memory**: On Mac, large models need careful memory management. Use `torch.mps.empty_cache()` between conditions.
6. **Output path collisions**: When running multiple models, output paths MUST include the model_id. Previous codebase had a bug where 7B results overwrote 13B results.
7. **Pansexual sample size**: Pansexual has only 32 ambiguous items (vs 240 for gay). Statistical tests on pansexual subgroups have very low power. Flag this in results.

## Testing
- Always run `--max_items 20` before full runs to catch path/hook/dtype issues
- Validate hook effects: after any intervention, check that at least some items changed prediction vs baseline
- Cross-check: behavioral baseline disambig accuracy should be >0.65 for any reasonable model. If it's near 0.33, the prompt format is wrong for that model's tokenizer.

## What NOT To Do
- Don't hardcode `model.model.layers` — use the wrapper
- Don't hardcode layer indices — parameterize relative to n_layers
- Don't save results only at the end of a long run — save incrementally
- Don't assume hook output structure — validate per architecture
- Don't use `torch_dtype` parameter — use `dtype`
- Don't run causal interventions without first confirming baseline behavioral results look reasonable
- Don't compare bias scores across models without noting sample size differences (pansexual n=32 vs gay n=240)



## Git
- `.gitignore`: results/runs/*/activations/, models/, *.npz, __pycache__/
- Commit JSON result summaries but not raw activations
- Tag runs with model_id and date