"""
SPAR RQ1 — Curvature as a Plan-Change Detection Signal
=======================================================
Tests whether spikes in activation trajectory curvature correspond to
meaningful transitions in generated text: plan pivots, topic switches,
temporal scale shifts, phase boundaries.

If curvature reliably spikes at transition points, it's a lightweight
real-time oversight signal that doesn't require a trained probe — just
a running cosine similarity computation on the residual stream.

Experimental design:
  1. Generate text from prompts with KNOWN transition points
  2. Record per-token curvature
  3. Annotate where transitions should occur (ground-truth)
  4. Test whether curvature is significantly higher at transitions

Usage:
    python run_curvature_signal.py
    python run_curvature_signal.py --models qwen3.5-9b --max_tokens 300
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import MODELS, DEFAULT_MODELS

# ── Prompts with known transition structure ─────────────────────────────────
# Each has "markers" — substrings we look for in the output to locate
# transition boundaries. We measure curvature around these boundaries.

TRANSITION_PROMPTS = [
    {
        "prompt": (
            "Write a kitchen schedule with three distinct phases. "
            "Phase 1: Morning prep (chopping, stocks, sauces). "
            "Phase 2: Afternoon service (firing orders, plating). "
            "Phase 3: Evening cleanup (sanitizing, inventory). "
            "Describe each phase in detail."
        ),
        "label": "three_phases",
        "markers": ["Phase 2", "phase 2", "Afternoon", "afternoon", "service begins",
                     "Phase 3", "phase 3", "Evening", "evening", "cleanup", "close"],
        "description": "Three distinct operational phases — curvature should spike at phase boundaries.",
    },
    {
        "prompt": (
            "You're making dinner and following a recipe. Halfway through, "
            "you realize you're missing a key ingredient and need to completely "
            "change your plan. Describe the original recipe steps, the moment "
            "of realization, and your new adapted plan."
        ),
        "label": "plan_pivot",
        "markers": ["realize", "missing", "instead", "adapt", "change", "pivot",
                     "alternative", "substitute", "new plan", "switch"],
        "description": "Single plan disruption — curvature should spike at the pivot point.",
    },
    {
        "prompt": (
            "Describe a chef's career in three time scales: "
            "what they do in the next 10 minutes (immediate tasks), "
            "what they do this year (skill development), "
            "and what they do over their career (legacy). "
            "Transition between each scale."
        ),
        "label": "scale_shift",
        "markers": ["this year", "year", "annual", "long term", "career", "decade",
                     "legacy", "over time", "eventually"],
        "description": "Temporal scale shifts — curvature should spike when switching from minutes→years→career.",
    },
    {
        "prompt": (
            "Tell a story about a restaurant. Start with a normal dinner service, "
            "then introduce a crisis (kitchen fire, celebrity arrival, health inspector), "
            "then describe the resolution. Make the transitions dramatic."
        ),
        "label": "crisis_narrative",
        "markers": ["suddenly", "but then", "crisis", "fire", "alarm", "panic",
                     "however", "finally", "resolved", "calm", "after"],
        "description": "Narrative arc with crisis — curvature should spike at crisis onset and resolution.",
    },
    {
        "prompt": (
            "List 5 recipes. For each one, give a brief description. "
            "The recipes should be very different from each other: "
            "a quick snack, a slow braise, a dessert, a breakfast item, and a fermented food."
        ),
        "label": "topic_switches",
        "markers": ["2.", "3.", "4.", "5.", "second", "third", "fourth", "fifth",
                     "next", "braise", "dessert", "breakfast", "ferment"],
        "description": "Rapid topic switches — curvature should spike at each recipe boundary.",
    },
    {
        "prompt": (
            "Describe two completely different approaches to the same problem: "
            "making a restaurant profitable. First describe the cost-cutting approach "
            "in detail. Then say 'Alternatively,' and describe the revenue-growth approach. "
            "The two strategies should feel very different."
        ),
        "label": "strategy_switch",
        "markers": ["Alternatively", "alternatively", "other hand", "instead",
                     "different approach", "second approach", "revenue", "growth"],
        "description": "Binary strategy switch — curvature should spike at 'Alternatively'.",
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


@torch.no_grad()
def generate_and_record(model, tokenizer, prompt, max_tokens, layer_idx):
    """Generate token by token, recording hidden states at target layer."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    tokens = []
    states = []

    for _ in range(max_tokens):
        outputs = model(input_ids, output_hidden_states=True)
        vec = outputs.hidden_states[layer_idx][0, -1, :].clamp(-1e4, 1e4).float().cpu().numpy()
        states.append(vec)

        next_id = outputs.logits[0, -1, :].argmax(dim=-1, keepdim=True)
        tokens.append(tokenizer.decode(next_id[0]))

        if next_id.item() == tokenizer.eos_token_id:
            break
        input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=-1)

    states = np.stack(states)
    velocities = np.diff(states, axis=0)
    cosines = []
    for t in range(len(velocities) - 1):
        a, b = velocities[t], velocities[t + 1]
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        cosines.append(float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0)

    return tokens, cosines, np.linalg.norm(velocities, axis=1).tolist()


def find_marker_positions(tokens, markers):
    """Find token indices where any marker substring appears in the running text."""
    positions = []
    text_so_far = ""
    found_markers = set()
    for i, tok in enumerate(tokens):
        prev_len = len(text_so_far)
        text_so_far += tok
        for marker in markers:
            if marker not in found_markers and marker.lower() in text_so_far[max(0, prev_len-len(marker)):].lower():
                positions.append({"index": i, "marker": marker})
                found_markers.add(marker)
    return positions


def curvature_at_transitions(cosines, marker_positions, window=5):
    """
    Compare curvature near transition points vs baseline.
    Returns stats for a significance test.
    """
    if not cosines or not marker_positions:
        return None

    marker_indices = {mp["index"] for mp in marker_positions}

    # Curvature near markers (within ±window tokens)
    near_transition = []
    baseline = []
    transition_set = set()
    for mi in marker_indices:
        for offset in range(-window, window + 1):
            idx = mi + offset - 1  # cosines are offset by 1 from tokens
            if 0 <= idx < len(cosines):
                transition_set.add(idx)

    for i, c in enumerate(cosines):
        if i in transition_set:
            near_transition.append(c)
        else:
            baseline.append(c)

    if not near_transition or not baseline:
        return None

    return {
        "transition_mean": float(np.mean(near_transition)),
        "transition_std": float(np.std(near_transition)),
        "baseline_mean": float(np.mean(baseline)),
        "baseline_std": float(np.std(baseline)),
        "n_transition": len(near_transition),
        "n_baseline": len(baseline),
        "delta": float(np.mean(near_transition) - np.mean(baseline)),
    }


def plot_curvature_signal(tokens, cosines, vel_norms, marker_positions,
                          prompt_info, model_key, outdir):
    """Plot curvature with vertical lines at detected transition markers."""
    label = prompt_info["label"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # Curvature
    ax = axes[0]
    x = list(range(1, len(cosines) + 1))
    ax.plot(x, cosines, color="#C0392B", alpha=0.4, linewidth=0.8)

    # Smoothed
    window = min(8, max(3, len(cosines) // 20))
    if len(cosines) > window:
        smoothed = np.convolve(cosines, np.ones(window)/window, mode="valid")
        x_smooth = list(range(window//2 + 1, window//2 + 1 + len(smoothed)))
        ax.plot(x_smooth, smoothed, color="#C0392B", linewidth=2.0)

    # Mark transitions
    for mp in marker_positions:
        idx = mp["index"]
        if 0 < idx < len(cosines) + 1:
            ax.axvline(idx, color="#2E75B6", alpha=0.6, linestyle="--", linewidth=1.5)
            ax.annotate(mp["marker"], (idx, ax.get_ylim()[1]),
                       fontsize=7, rotation=45, ha="left", va="bottom", color="#2E75B6")

    ax.set_ylabel("Cosine similarity\n(curvature)")
    ax.set_title(f"{model_key} — {label}\n{prompt_info['description']}")
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.grid(True, alpha=0.3)

    # Velocity norms
    ax = axes[1]
    if vel_norms:
        ax.plot(range(1, len(vel_norms) + 1), vel_norms, color="#27AE60", alpha=0.6, linewidth=0.8)
        if len(vel_norms) > window:
            smoothed_v = np.convolve(vel_norms, np.ones(window)/window, mode="valid")
            x_sv = list(range(window//2 + 1, window//2 + 1 + len(smoothed_v)))
            ax.plot(x_sv, smoothed_v, color="#27AE60", linewidth=2.0)
    for mp in marker_positions:
        idx = mp["index"]
        ax.axvline(idx, color="#2E75B6", alpha=0.6, linestyle="--", linewidth=1.5)

    ax.set_ylabel("Velocity norm")
    ax.set_xlabel("Generated token index")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = outdir / f"curvature_signal_{model_key}_{label}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


def main():
    parser = argparse.ArgumentParser(description="SPAR RQ1 — Curvature as Oversight Signal")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer to monitor (default: final)")
    parser.add_argument("--max_tokens", type=int, default=300)
    parser.add_argument("--outdir", type=str, default="results")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    for model_key in args.models:
        if model_key not in MODELS:
            print(f"⚠  Unknown model '{model_key}'. Skipping.")
            continue

        model, tokenizer, n_layers = load_model(model_key)
        layer_idx = args.layer if args.layer is not None else n_layers
        model_dir = outdir / model_key
        model_dir.mkdir(parents=True, exist_ok=True)

        all_results = []

        for pinfo in TRANSITION_PROMPTS:
            print(f"\n  [{pinfo['label']}] Generating...")
            tokens, cosines, vel_norms = generate_and_record(
                model, tokenizer, pinfo["prompt"], args.max_tokens, layer_idx
            )
            print(f"    Generated {len(tokens)} tokens")

            marker_positions = find_marker_positions(tokens, pinfo["markers"])
            print(f"    Found {len(marker_positions)} transition markers: "
                  f"{[mp['marker'] for mp in marker_positions]}")

            stats = curvature_at_transitions(cosines, marker_positions)
            if stats:
                print(f"    Curvature near transitions: {stats['transition_mean']:.4f} ± {stats['transition_std']:.4f}")
                print(f"    Curvature baseline:         {stats['baseline_mean']:.4f} ± {stats['baseline_std']:.4f}")
                print(f"    Delta:                      {stats['delta']:+.4f}")

            plot_curvature_signal(tokens, cosines, vel_norms, marker_positions,
                                pinfo, model_key, model_dir)

            all_results.append({
                "label": pinfo["label"],
                "description": pinfo["description"],
                "n_tokens": len(tokens),
                "n_markers_found": len(marker_positions),
                "markers_found": [mp["marker"] for mp in marker_positions],
                "marker_indices": [mp["index"] for mp in marker_positions],
                "generated_text": "".join(tokens)[:500],
                "transition_stats": stats,
                "curvature_mean": float(np.mean(cosines)) if cosines else None,
                "curvature_std": float(np.std(cosines)) if cosines else None,
            })

        # Summary: aggregate across all prompts
        print(f"\n{'='*60}")
        print(f"  Summary for {model_key}")
        print(f"{'='*60}")
        deltas = [r["transition_stats"]["delta"]
                  for r in all_results if r["transition_stats"]]
        if deltas:
            print(f"  Mean curvature delta at transitions: {np.mean(deltas):+.4f}")
            print(f"  Prompts with lower curvature at transitions: "
                  f"{sum(1 for d in deltas if d < 0)}/{len(deltas)}")
            print(f"  Prompts with higher curvature at transitions: "
                  f"{sum(1 for d in deltas if d > 0)}/{len(deltas)}")

        with open(model_dir / "curvature_signal_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✓ Saved curvature_signal_results.json → {model_dir}")

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
