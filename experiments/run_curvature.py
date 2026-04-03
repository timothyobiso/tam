"""
SPAR RQ1 — Activation Trajectory Curvature
===========================================
Implements the curvature metric from Wang et al. (2026)
"Temporal Straightening for Latent Planning" (arXiv:2603.12231).

Core idea:  Given consecutive hidden states z_t, z_{t+1}, z_{t+2}
at a fixed layer during autoregressive generation, compute velocity
vectors v_t = z_{t+1} - z_t and measure their cosine similarity.
High cosine = straight trajectory.  Low/negative = curved.

This script:
  1. Generates text from seed prompts token-by-token
  2. Records hidden states at every layer for each generated token
  3. Computes per-layer curvature (mean cosine similarity of velocity pairs)
  4. Saves results + plots

Usage:
    python run_curvature.py
    python run_curvature.py --models qwen3.5-9b --max_tokens 200
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from config import MODELS, DEFAULT_MODELS, GENERATION_SEEDS, GEN_MAX_TOKENS, LAYER_STRIDE


# ═══════════════════════════════════════════════════════════════════════════
# 1. Token-by-token generation with hidden state recording
# ═══════════════════════════════════════════════════════════════════════════

def load_model(model_key: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_id = MODELS[model_key]
    print(f"Loading {hf_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    return model, tokenizer, n_layers


@torch.no_grad()
def generate_with_hidden_states(
    model, tokenizer, prompt: str, max_new_tokens: int, layer_indices: list[int]
) -> dict:
    """
    Greedy-generate token by token, recording the last-token hidden state
    at each requested layer at each step.

    Returns:
        {
            "text": str,
            "tokens": list[str],
            "states": {layer_idx: np.ndarray of shape (n_tokens, d_model)},
        }
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    generated_tokens = []
    # states_per_layer[li] collects one vector per generated token
    states_per_layer = {li: [] for li in layer_indices}

    for _ in range(max_new_tokens):
        outputs = model(input_ids, output_hidden_states=True)
        hidden = outputs.hidden_states  # tuple of (n_layers+1) tensors

        # Record last-token hidden state at each layer
        for li in layer_indices:
            vec = hidden[li][0, -1, :].clamp(-1e4, 1e4).float().cpu().numpy()
            states_per_layer[li].append(vec)

        # Greedy next token
        next_id = outputs.logits[0, -1, :].argmax(dim=-1, keepdim=True)
        token_str = tokenizer.decode(next_id[0])
        generated_tokens.append(token_str)

        # Stop on EOS
        if next_id.item() == tokenizer.eos_token_id:
            break

        # Append and continue
        input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=-1)

    return {
        "text": "".join(generated_tokens),
        "tokens": generated_tokens,
        "states": {li: np.stack(vecs) for li, vecs in states_per_layer.items()},
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. Curvature Computation (Wang et al. Eq. 4)
# ═══════════════════════════════════════════════════════════════════════════

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def compute_curvature(states: np.ndarray) -> dict:
    """
    Given states of shape (T, d_model), compute the curvature profile.

    Velocity vectors:  v_t = z_{t+1} - z_t
    Curvature at t:    C_t = cos(v_t, v_{t+1})     (Eq. 4)
    Straightness:      mean(C_t)                    (higher = straighter)

    Returns:
        {
            "cosines": list of per-step cosine similarities,
            "mean_cosine": float,   # overall straightness
            "std_cosine": float,
            "velocity_norms": list of ||v_t||,
        }
    """
    T = states.shape[0]
    if T < 3:
        return {"cosines": [], "mean_cosine": 0.0, "std_cosine": 0.0, "velocity_norms": []}

    velocities = np.diff(states, axis=0)  # (T-1, d)
    norms = np.linalg.norm(velocities, axis=1).tolist()

    cosines = []
    for t in range(len(velocities) - 1):
        c = cosine_sim(velocities[t], velocities[t + 1])
        cosines.append(c)

    return {
        "cosines": cosines,
        "mean_cosine": float(np.mean(cosines)),
        "std_cosine": float(np.std(cosines)),
        "velocity_norms": norms,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. Main
# ═══════════════════════════════════════════════════════════════════════════

def run_curvature_for_model(model_key: str, max_tokens: int, stride: int, outdir: Path):
    model, tokenizer, n_layers = load_model(model_key)
    layer_indices = list(range(0, n_layers + 1, stride))
    model_dir = outdir / model_key
    model_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for seed_prompt in GENERATION_SEEDS:
        print(f"\n  Seed: \"{seed_prompt[:50]}...\"")
        gen = generate_with_hidden_states(
            model, tokenizer, seed_prompt, max_tokens, layer_indices
        )
        print(f"    Generated {len(gen['tokens'])} tokens")

        prompt_result = {
            "seed_prompt": seed_prompt,
            "generated_text": gen["text"][:300],
            "n_tokens": len(gen["tokens"]),
            "curvature_by_layer": {},
        }

        for li in layer_indices:
            curv = compute_curvature(gen["states"][li])
            prompt_result["curvature_by_layer"][str(li)] = {
                "mean_cosine": curv["mean_cosine"],
                "std_cosine": curv["std_cosine"],
                "n_velocity_pairs": len(curv["cosines"]),
            }

        all_results.append(prompt_result)

    # ── Summary: mean curvature per layer across all seeds ──────────
    summary_by_layer = {}
    for li in layer_indices:
        cosines = [r["curvature_by_layer"][str(li)]["mean_cosine"] for r in all_results]
        summary_by_layer[str(li)] = {
            "mean_cosine_across_seeds": round(float(np.mean(cosines)), 4),
            "std_across_seeds": round(float(np.std(cosines)), 4),
        }

    output = {
        "model": model_key,
        "hf_id": MODELS[model_key],
        "n_layers": n_layers,
        "layer_indices": layer_indices,
        "max_tokens": max_tokens,
        "per_seed": all_results,
        "summary_by_layer": summary_by_layer,
    }

    with open(model_dir / "curvature_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  ✓ Saved curvature_results.json → {model_dir}")

    # Print summary
    print(f"\n  Layer curvature summary (mean cosine, higher=straighter):")
    for li in layer_indices:
        s = summary_by_layer[str(li)]
        bar = "█" * int(max(0, s["mean_cosine_across_seeds"]) * 40)
        print(f"    L{li:3d}  {s['mean_cosine_across_seeds']:+.4f} ± {s['std_across_seeds']:.4f}  {bar}")

    del model
    torch.cuda.empty_cache()
    return output


def main():
    parser = argparse.ArgumentParser(description="SPAR RQ1 — Curvature Analysis")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--max_tokens", type=int, default=GEN_MAX_TOKENS)
    parser.add_argument("--stride", type=int, default=LAYER_STRIDE)
    parser.add_argument("--outdir", type=str, default="results")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    for model_key in args.models:
        if model_key not in MODELS:
            print(f"⚠  Unknown model '{model_key}'. Skipping.")
            continue
        run_curvature_for_model(model_key, args.max_tokens, args.stride, outdir)

    print(f"\nDone. All results in {outdir}/")


if __name__ == "__main__":
    main()
