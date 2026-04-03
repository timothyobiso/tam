# SPAR RQ1: Temporal Activation Monitors

**Timothy Obiso** · April 2026  
SPAR Fellowship — Temporal Activation Monitors for AI Oversight

---

**RQ1:** Do LLMs develop internal representations of their position within an unfolding process, and can these be used as real-time oversight signals?

**Answer:** Yes. Qwen3.5 models encode temporal horizon in a universal linear subspace of the residual stream that transfers across all tested domains (kitchen, medical, software, military, business, education — 6/6 positive, mean cross-domain R²=0.65). The representation tracks realistic task complexity rather than stated timelines (r=0.80), updates token-by-token during generation, is causally localized to the final 3 layers, and can be steered to produce safer outputs (p=7×10⁻¹⁷ linear trend).

---

## Setup

**Models.** Qwen3.5-4B (32L, d=2560) and Qwen3.5-9B (32L, d=4096), bfloat16, single A6000.

**Datasets.** (1) 237 kitchen-domain prompts from 4 template families. (2) 180 multi-domain prompts across 6 domains with matched temporal structure.

**Methods.** Linear probing (RidgeCV), nonlinear probing (MLP), per-token probing during generation, activation trajectory curvature (Wang et al. 2026), activation patching, attention subspace knockout, temporal steering, deception detection via stated/realistic time mismatch, cross-domain transfer, safety-steering evaluation.

## Results

### 1. The temporal representation is universal across domains

A probe trained on 5 domains transfers to the held-out 6th domain with no fine-tuning (Qwen3.5-9B, L32):

| Held-out domain | Cross-domain R² (ridge) | Within-domain R² (ridge CV) |
|-----------------|------------------------|-----------------------------|
| business | 0.71 | 0.84 |
| education | 0.65 | 0.50 |
| kitchen | 0.55 | 0.15 |
| medical | **0.76** | -2.91 |
| military | 0.59 | 0.47 |
| software | 0.65 | 0.73 |

**6/6 held-out domains positive, mean R²=0.65.** The model uses the same linear subspace for temporal horizon whether reasoning about flipping steaks, stabilizing patients, deploying code, or planning military operations. Cross-domain probes outperform within-domain on small n (medical: cross=0.76 vs within=-2.91), indicating the temporal direction is so consistent that 150 out-of-domain samples teach a better probe than 30 in-domain.

Pooled across all 6 domains, the probe reaches R²=0.70 at L32 with monotonic layer buildup (R²=0.34 at L0).

### 2. The probe reads realistic time, not stated time

| | r(probe, stated) | r(probe, realistic) |
|-|-------------------|---------------------|
| **4B** | -0.28 (p = 0.19) | **0.80 (p = 2×10⁻⁶)** |
| **9B** | -0.14 (p = 0.50) | **0.63 (p = 9×10⁻⁴)** |

| Prompt | Stated | Realistic | Probe (4B) |
|--------|--------|-----------|------------|
| "This quick 5-minute task: completely renovate the restaurant kitchen" | 1.8 | 11.8 | **7.07** |
| "In just 2 minutes, develop a farm-to-table supply chain" | 1.1 | 12.5 | **10.29** |
| "This year-long endeavor: flip the pancake" | 13.2 | 0.1 | **3.81** |

The 4B shows stronger deception detection (r=0.80 vs 0.63). The 9B partially integrates contradictory stated/realistic signals — more capable models may blend conflicting temporal cues, a scaling concern for oversight.

### 3. Steering toward long-term thinking produces safer outputs

Safety score (safe logprob − harmful logprob) increases monotonically with temporal steering alpha on the 9B:

| α | Safety Score |
|---|-------------|
| -50 | -0.177 |
| 0 | +0.339 |
| +50 | **+0.855** |

Linear trend: slope=0.010, **r=0.9999, p=7×10⁻¹⁷**. Qualitatively, "The person who betrayed my trust deserves" → at α=-30: "a beating" → at α=0: "to be punished." This confirms Issue #29's preliminary GPT-2 result on a 9B model with a logprob metric.

### 4. The representation is substantially nonlinear

| Probe | 4B log_time | 9B log_time | 9B planning_depth |
|-------|-------------|-------------|-------------------|
| Ridge | 0.37 | 0.53 | 0.28 |
| MLP (64) | 0.46 | **0.77** | **0.58** |
| MLP (128,64) | 0.53 | **0.83** | 0.43 |

The linear probe captures roughly half the signal. However, the linear component transfers across domains while the nonlinear component is domain-specific (cross-domain MLP mean=0.32 vs ridge mean=0.65). For a universal oversight monitor, use a linear probe.

### 5. The internal clock updates during generation

| Prompt | Probe Range | Expected | Observed |
|--------|-------------|----------|----------|
| Countdown (60→0 min) | 1.99 → 9.13 | Decreasing | ✓ Downward trend |
| Plan disruption | -0.39 → 9.69 | Crash at pivot | ✓ Collapses to ~0, then recovers |
| Context switch | 1.49 → 9.62 | Discontinuity | ✓ Sharp drop at interruption |
| Tasting menu | 1.88 → 12.25 | Advancing | ✓ Full-scale traversal |

### 6. The computation is localized to 3 layers, distributed within them

**Layer-level:** Patching any single layer from L29-L31 recovers 94-101% of the probe prediction across 5 contrastive pairs on both models.

**Subspace-level:** Within L29-31, ablating individual 160/256-dim subspaces produces max gap reduction of 2.5%. The temporal computation is uniformly distributed across dimensions, consistent with Qwen3.5's Gated Delta Network architecture. A standard MHA model might show head-level sparsity.

### 7. The probe generalizes across phrasing

Trained on 3/4 template families, tested on the held-out 4th: 3/4 folds positive on both models (9B reaches R²=0.46 on held-out `over_timespan`). The `step_plan` family fails — the probe reads temporal language, not step-count reasoning.

### 8. Temporal horizon scales with model size

| Target | 4B (L32) | 9B (L32) |
|--------|----------|----------|
| log(time_horizon) | R²=0.37 | R²=0.53 |
| planning_depth | R²=0.18 | R²=0.28 |

### 9. Curvature traces plan structure

Curvature rises from -0.52 at L0 to -0.37 at L32. The 9B is straighter at every layer. During generation, curvature spikes to -0.8 at plan disruption points.

## Interpretation

1. **Universal, not domain-specific.** The temporal representation occupies the same linear subspace across 6 domains. An oversight monitor trained on any domain works on all others.
2. **Detects misrepresentation.** The model's internal representation tracks realistic task complexity, not stated timelines. Stronger detection in smaller models (scaling concern).
3. **Safety-relevant.** Temporal steering monotonically increases output safety. Long-term thinking → prosocial language; short-term steering → impulsive/violent completions.
4. **Substantially nonlinear, but the linear component is universal.** MLP recovers R²=0.83, but the nonlinear gain is domain-specific. Linear probes are the right tool for cross-domain monitoring.
5. **Updates dynamically.** Tracks narrative time during generation with crash-and-recovery at plan disruptions.
6. **Sharply localized, fully distributed.** Three layers (L29-31) do the computation. No sparse head structure (linear attention architecture).

## Remaining Work

1. **Belief-action consistency (Issue #46).** Forced-choice temporal scenarios measuring whether the model acts on its decoded internal belief.
2. **Standard MHA head analysis.** Run head knockout on Llama/Gemma to test for head-level sparsity in standard attention.
3. **Instruct model experiments.** Rerun steering and safety evaluation on instruction-tuned variants.
4. **Scaling to larger models.** Test on Qwen3.5-27B with proper single-GPU or quantized setup.

## References

- Wang, Y., Bounou, O., Zhou, G., Balestriero, R., Rudner, T.G.J., LeCun, Y., & Ren, M. (2026). Temporal Straightening for Latent Planning. arXiv:2603.12231.
- Hosseini, E.A. & Fedorenko, E. (2023). Large language models implicitly learn to straighten neural sentence trajectories to construct a predictive representation of natural language.
- Hosseini, E.A., Li, Y., Bahri, Y., Campbell, D., & Lampinen, A.K. (2026). Context structure reshapes the representational geometry of language models. arXiv:2601.22364.
- Arghal et al. (2025). A Behavioural and Representational Evaluation of Goal-Directedness in Language Model Agents. arXiv:2602.08964.
