"""
SPAR RQ1 — Per-Token Temporal Probing During Generation
========================================================
Loads a fitted probe from run_probes.py, then generates text token-by-token
and applies the probe at every step. Produces a timeline of the model's
internal temporal representation as it writes out a plan.

Key question: does the model's internal "clock" update as it progresses
through its own plan?

Usage:
    python run_per_token.py
    python run_per_token.py --models qwen3.5-9b --layer 32 --max_tokens 300
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import MODELS, DEFAULT_MODELS, SEED

# ── Prompts designed to elicit multi-step plans with clear temporal structure ──
PLAN_PROMPTS = [
    # Should show internal clock advancing through steps
    {
        "prompt": "Write a detailed hour-by-hour schedule for a restaurant kitchen on a busy Saturday night, from 3pm prep to midnight cleanup.",
        "label": "saturday_service",
        "expected": "advancing_time",
    },
    {
        "prompt": "Plan a 5-course tasting menu dinner service. For each course, describe the timing, preparation, and plating.",
        "label": "tasting_menu",
        "expected": "advancing_steps",
    },
    # Should show a shift from short-horizon to long-horizon
    {
        "prompt": "You're a new head chef. Describe what you'll do in your first day, first week, first month, and first year.",
        "label": "new_chef_horizons",
        "expected": "expanding_horizon",
    },
    # Should show curvature spike when the plan changes
    {
        "prompt": "You're preparing a dinner party for 8 when you realize you're out of the main protein. Describe your original plan and how you adapt.",
        "label": "plan_disruption",
        "expected": "curvature_spike_at_pivot",
    },
    # Interleaved short and long horizon reasoning
    {
        "prompt": "Describe how to make beef bourguignon step by step. For each step, mention both what to do right now and why it matters for the final dish hours later.",
        "label": "interleaved_horizons",
        "expected": "oscillating_horizon",
    },
    # Sudden context switch
    {
        "prompt": "Plan tomorrow's lunch prep. Halfway through, a health inspector arrives. Describe both the prep plan and the inspection response.",
        "label": "context_switch",
        "expected": "curvature_spike_at_switch",
    },
    # Countdown — internal clock should decrease
    {
        "prompt": "Service starts in 60 minutes. Describe the countdown: what happens at 60 min out, 45, 30, 15, 5, and go time.",
        "label": "countdown",
        "expected": "decreasing_horizon",
    },
]


def load_model(model_key):
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


def load_probe(model_key, layer_idx, results_dir="results"):
    """Load the fitted probe + scaler + pca from run_probes.py output."""
    probe_path = Path(results_dir) / model_key / "probes.pkl"
    if not probe_path.exists():
        raise FileNotFoundError(
            f"No probes found at {probe_path}. Run run_probes.py --prompts prompts_generated.json first."
        )
    with open(probe_path, "rb") as f:
        probes = pickle.load(f)

    # Find the probe — check "generated", "kitchen", or "temporal" keys
    for setname in ["generated", "kitchen", "temporal"]:
        if setname in probes and "log_time_horizon" in probes[setname]:
            if layer_idx in probes[setname]["log_time_horizon"]:
                p = probes[setname]["log_time_horizon"][layer_idx]
                print(f"  Loaded probe from {setname}/log_time_horizon @ layer {layer_idx}")
                return p["probe"], p["scaler"], p["pca"]

    raise KeyError(f"No log_time_horizon probe found at layer {layer_idx}")


@torch.no_grad()
def generate_with_per_token_probing(
    model, tokenizer, prompt, probe, scaler, pca, layer_idx, max_tokens=200,
):
    """
    Generate token by token. At each step, extract hidden state at layer_idx,
    apply the fitted probe pipeline (scaler → PCA → ridge), and record the
    predicted log(time_horizon).

    Also computes curvature (cosine similarity of velocity vectors).
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    prompt_len = input_ids.shape[1]

    tokens = []
    probe_predictions = []
    raw_states = []

    for step in range(max_tokens):
        outputs = model(input_ids, output_hidden_states=True)
        hidden = outputs.hidden_states

        # Extract last-token state at target layer
        vec = hidden[layer_idx][0, -1, :].clamp(-1e4, 1e4).float().cpu().numpy()
        raw_states.append(vec)

        # Apply probe pipeline: scale → PCA → predict
        vec_scaled = scaler.transform(vec.reshape(1, -1))
        vec_pca = pca.transform(vec_scaled)
        pred = probe.predict(vec_pca)[0]
        probe_predictions.append(float(pred))

        # Greedy next token
        next_id = outputs.logits[0, -1, :].argmax(dim=-1, keepdim=True)
        token_str = tokenizer.decode(next_id[0])
        tokens.append(token_str)

        if next_id.item() == tokenizer.eos_token_id:
            break
        input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=-1)

    # Compute curvature from raw states
    states_arr = np.stack(raw_states)
    velocities = np.diff(states_arr, axis=0)
    cosines = []
    for t in range(len(velocities) - 1):
        a, b = velocities[t], velocities[t + 1]
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        c = float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0
        cosines.append(c)

    return {
        "tokens": tokens,
        "text": "".join(tokens),
        "probe_predictions": probe_predictions,  # predicted log(time_horizon) per token
        "cosines": cosines,                       # curvature per token (len = n_tokens - 2)
        "velocity_norms": np.linalg.norm(velocities, axis=1).tolist(),
    }


def plot_per_token(result, prompt_info, model_key, outdir):
    """Plot probe predictions and curvature over generated tokens."""
    label = prompt_info["label"]
    n = len(result["probe_predictions"])

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # 1. Probe prediction (predicted temporal horizon)
    ax = axes[0]
    ax.plot(result["probe_predictions"], color="#2E75B6", linewidth=1.2)
    ax.set_ylabel("Predicted\nlog(time_horizon)")
    ax.set_title(f"{model_key} — {label}\nInternal temporal representation over generation")
    ax.grid(True, alpha=0.3)
    ax.axhline(np.mean(result["probe_predictions"]), color="gray", ls="--", lw=0.8, alpha=0.5)

    # 2. Curvature (cosine similarity)
    ax = axes[1]
    if result["cosines"]:
        x_curv = list(range(1, len(result["cosines"]) + 1))
        ax.plot(x_curv, result["cosines"], color="#C0392B", linewidth=1.0, alpha=0.7)
        # Smoothed version
        window = min(10, len(result["cosines"]) // 3 + 1)
        if window > 2:
            smoothed = np.convolve(result["cosines"], np.ones(window)/window, mode="valid")
            x_smooth = list(range(window//2 + 1, window//2 + 1 + len(smoothed)))
            ax.plot(x_smooth, smoothed, color="#C0392B", linewidth=2.0, label=f"smoothed (w={window})")
            ax.legend(fontsize=8)
    ax.set_ylabel("Cosine similarity\n(curvature)")
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.grid(True, alpha=0.3)

    # 3. Velocity norms
    ax = axes[2]
    if result["velocity_norms"]:
        ax.plot(range(1, len(result["velocity_norms"]) + 1),
                result["velocity_norms"], color="#27AE60", linewidth=1.0, alpha=0.7)
    ax.set_ylabel("Velocity norm\n||z_{t+1} - z_t||")
    ax.set_xlabel("Generated token index")
    ax.grid(True, alpha=0.3)

    # Add token text as sparse annotations on the bottom
    text = result["text"]
    # Mark every 20th token
    step = max(1, n // 15)
    for i in range(0, n, step):
        snippet = "".join(result["tokens"][max(0,i-1):i+2]).strip()[:20]
        if snippet:
            axes[2].annotate(snippet, (i, 0), xytext=(i, -0.1),
                           textcoords="axes fraction", fontsize=5, rotation=45,
                           alpha=0.5, ha="left", va="top",
                           xycoords=("data", "axes fraction"))

    plt.tight_layout()
    fname = outdir / f"per_token_{model_key}_{label}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


def main():
    parser = argparse.ArgumentParser(description="SPAR RQ1 — Per-Token Probing")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer index for probe (default: final layer)")
    parser.add_argument("--max_tokens", type=int, default=250)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--outdir", type=str, default="results")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    for model_key in args.models:
        if model_key not in MODELS:
            print(f"⚠  Unknown model '{model_key}'. Skipping.")
            continue

        model, tokenizer, n_layers = load_model(model_key)
        layer_idx = args.layer if args.layer is not None else n_layers  # default: final
        probe, scaler, pca = load_probe(model_key, layer_idx, args.results_dir)

        model_dir = outdir / model_key
        model_dir.mkdir(parents=True, exist_ok=True)
        all_results = []

        for pinfo in PLAN_PROMPTS:
            print(f"\n  [{pinfo['label']}] Generating up to {args.max_tokens} tokens...")
            result = generate_with_per_token_probing(
                model, tokenizer, pinfo["prompt"],
                probe, scaler, pca, layer_idx, args.max_tokens,
            )
            print(f"    Generated {len(result['tokens'])} tokens")
            print(f"    Probe range: {min(result['probe_predictions']):.2f} → "
                  f"{max(result['probe_predictions']):.2f}")
            if result["cosines"]:
                print(f"    Curvature range: {min(result['cosines']):.3f} → "
                      f"{max(result['cosines']):.3f}")

            plot_per_token(result, pinfo, model_key, model_dir)

            all_results.append({
                "label": pinfo["label"],
                "prompt": pinfo["prompt"],
                "expected": pinfo["expected"],
                "n_tokens": len(result["tokens"]),
                "generated_text": result["text"][:500],
                "probe_mean": float(np.mean(result["probe_predictions"])),
                "probe_std": float(np.std(result["probe_predictions"])),
                "probe_min": float(min(result["probe_predictions"])),
                "probe_max": float(max(result["probe_predictions"])),
                "curvature_mean": float(np.mean(result["cosines"])) if result["cosines"] else None,
                "curvature_std": float(np.std(result["cosines"])) if result["cosines"] else None,
            })

        with open(model_dir / "per_token_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✓ Saved per_token_results.json → {model_dir}")

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
