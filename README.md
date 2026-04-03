# Temporal Activation Monitors for AI Oversight

**SPAR Fellowship — Spring 2026**

Do LLMs develop internal representations of their position within an unfolding process, and can these be used as real-time oversight signals?

**Yes.** Qwen3.5 models encode temporal horizon in a universal linear subspace of the residual stream that transfers across all tested domains (6/6 positive, mean cross-domain R²=0.65). The representation tracks realistic task complexity rather than stated timelines, updates token-by-token during generation, and can be steered to produce safer outputs.

## Key Findings

| Finding | Result |
|---------|--------|
| Cross-domain transfer | 6/6 domains positive, mean R²=0.65 |
| Deception detection | Probe tracks realistic time, not stated (r=0.80) |
| Safety steering | Long-term steering → safer outputs (p=7×10⁻¹⁷) |
| Nonlinear probe | R²=0.83 with MLP (vs 0.53 linear) |
| Per-token dynamics | Internal clock updates during generation |
| Causal localization | L29-31 account for 94-101% of temporal computation |

## Setup

```bash
git clone https://github.com/justinshenk/temporal-awareness.git
cd temporal-awareness
uv init
uv add torch transformers accelerate scikit-learn matplotlib numpy tqdm scipy
```

## Experiments

All scripts are in `experiments/`. Run from the repo root.

### 1. Generate prompts
```bash
# Kitchen-domain (237 prompts)
uv run python experiments/generate_prompts.py

# Multi-domain (180 prompts across 6 domains)
uv run python experiments/run_domain_transfer.py --generate_only
```

### 2. Train probes
```bash
CUDA_VISIBLE_DEVICES=0 uv run python experiments/run_probes.py --prompts prompts_generated.json --models qwen3.5-9b
```

### 3. Core experiments
```bash
# Per-token probing during generation
CUDA_VISIBLE_DEVICES=0 uv run python experiments/run_per_token.py --models qwen3.5-9b

# Activation patching (causal localization)
CUDA_VISIBLE_DEVICES=0 uv run python experiments/run_patching.py --models qwen3.5-9b

# Steering
CUDA_VISIBLE_DEVICES=0 uv run python experiments/run_steering.py --models qwen3.5-9b

# Deception detection
CUDA_VISIBLE_DEVICES=0 uv run python experiments/run_deception.py --models qwen3.5-9b

# Domain transfer
CUDA_VISIBLE_DEVICES=0 uv run python experiments/run_domain_transfer.py --models qwen3.5-9b

# Safety steering
CUDA_VISIBLE_DEVICES=0 uv run python experiments/run_safety_steering.py --models qwen3.5-9b

# Curvature analysis
CUDA_VISIBLE_DEVICES=0 uv run python experiments/run_curvature.py --models qwen3.5-9b

# Nonlinear probe comparison
uv run python experiments/run_nonlinear_probe.py --models qwen3.5-9b

# Head knockout
CUDA_VISIBLE_DEVICES=0 uv run python experiments/run_head_knockout.py --models qwen3.5-9b

# Cross-family generalization
uv run python experiments/run_cross_family.py --models qwen3.5-9b
```

### 4. Plot results
```bash
uv run python experiments/plot_results.py
```

## Models

| Model | Params | GPU Memory | Status |
|-------|--------|------------|--------|
| Qwen3.5-4B | 4B | ~8GB | ✓ Tested |
| Qwen3.5-9B | 9B | ~18GB | ✓ Tested |
| Qwen3.5-27B | 27B | ~54GB | Requires More RAM |

## Output Structure

```
results/
├── qwen3.5-9b/
│   ├── probe_results.json
│   ├── per_token_results.json
│   ├── patching_results.json
│   ├── steering_results.json
│   ├── deception_results.json
│   ├── domain_transfer_results.json
│   ├── safety_steering_results.json
│   ├── curvature_results.json
│   ├── nonlinear_probe_results.json
│   ├── head_knockout_results.json
│   ├── cross_family_results.json
│   ├── activations.pkl
│   ├── probes.pkl
│   └── *.png
└── all_results.json
```

## References

- Wang et al. (2026). Temporal Straightening for Latent Planning. arXiv:2603.12231
- Hosseini & Fedorenko (2023). Large language models implicitly learn to straighten neural sentence trajectories.
- Hosseini et al. (2026). Context structure reshapes the representational geometry of language models. arXiv:2601.22364
