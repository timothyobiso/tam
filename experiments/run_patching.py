"""
SPAR RQ1 — Activation Patching for Temporal Horizon
====================================================
Identifies which layers causally compute temporal representations by
patching activations from a short-horizon prompt into a long-horizon
prompt (and vice versa), one layer at a time.

If patching layer L causes the probe prediction to flip, that layer
is causally responsible for the temporal computation — not just a
passive carrier of information from earlier layers.

Method:
  1. Run clean (short-horizon) prompt → record all hidden states
  2. Run corrupted (long-horizon) prompt → record probe prediction
  3. For each layer L: re-run corrupted prompt, but replace layer L's
     output with the clean version → measure probe prediction change
  4. The "recovery" at each layer = how much patching restores the
     clean prediction. High recovery = causally important layer.

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_patching.py --models qwen3.5-9b
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from config import MODELS, DEFAULT_MODELS, SEED


# ── Contrastive prompt pairs: matched structure, different time horizons ────
PATCH_PAIRS = [
    {
        "short": "In the next 30 seconds, the chef needs to",
        "long": "Over the next five years, the restaurant needs to",
        "label": "seconds_vs_years",
    },
    {
        "short": "Immediately flip the steak before it",
        "long": "Plan the seasonal menu rotation for the coming year so that",
        "label": "immediate_vs_annual",
    },
    {
        "short": "The timer is going off right now, quickly",
        "long": "Develop a long-term training program for new kitchen staff that",
        "label": "timer_vs_training",
    },
    {
        "short": "In the next minute, finish plating the",
        "long": "Over the next decade, build the restaurant brand by",
        "label": "minute_vs_decade",
    },
    {
        "short": "Right now, turn down the heat on the",
        "long": "Over the coming months, redesign the entire kitchen workflow to",
        "label": "now_vs_months",
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
    probe_path = Path(results_dir) / model_key / "probes.pkl"
    with open(probe_path, "rb") as f:
        probes = pickle.load(f)
    for setname in ["generated", "kitchen", "temporal"]:
        if setname in probes and "log_time_horizon" in probes[setname]:
            if layer_idx in probes[setname]["log_time_horizon"]:
                p = probes[setname]["log_time_horizon"][layer_idx]
                return p["probe"], p["scaler"], p["pca"]
    raise KeyError(f"No probe at layer {layer_idx}")


def apply_probe(hidden_state, scaler, pca, probe):
    """Apply the probe pipeline to a single hidden state vector."""
    vec = hidden_state.clamp(-1e4, 1e4).float().cpu().numpy().reshape(1, -1)
    return float(probe.predict(pca.transform(scaler.transform(vec)))[0])


@torch.no_grad()
def get_clean_states(model, tokenizer, prompt, n_layers):
    """Run a prompt and save the output hidden state at every layer."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model(**inputs, output_hidden_states=True)
    # Save last-token state at each layer (including embedding = index 0)
    states = {}
    for li in range(n_layers + 1):
        states[li] = outputs.hidden_states[li][0, -1, :].clone()
    return states


@torch.no_grad()
def run_with_patch(model, tokenizer, prompt, clean_states, patch_layer, n_layers):
    """
    Run prompt but replace the output of patch_layer with the clean version.
    Returns the final-layer hidden state (for probe evaluation).
    """
    # We hook the target layer to replace its output
    patched_final = [None]

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
            # Replace last token's hidden state with clean version
            clean = clean_states[patch_layer + 1].to(h.device)  # +1 because hidden_states[0]=embed
            h[0, -1, :] = clean
            return (h,) + output[1:]
        else:
            clean = clean_states[patch_layer + 1].to(output.device)
            output[0, -1, :] = clean
            return output

    # Attach hook
    handle = model.model.layers[patch_layer].register_forward_hook(hook_fn)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model(**inputs, output_hidden_states=True)

    handle.remove()

    # Return the final hidden state (after all layers, index n_layers)
    return outputs.hidden_states[n_layers][0, -1, :]


def run_patching_experiment(model, tokenizer, n_layers, probe, scaler, pca, pair):
    """Run full patching sweep for one contrastive pair."""
    probe_layer = n_layers  # probe reads from final hidden state

    # 1. Get clean (short) and corrupted (long) baseline predictions
    clean_states = get_clean_states(model, tokenizer, pair["short"], n_layers)
    corrupt_states = get_clean_states(model, tokenizer, pair["long"], n_layers)

    clean_pred = apply_probe(clean_states[probe_layer], scaler, pca, probe)
    corrupt_pred = apply_probe(corrupt_states[probe_layer], scaler, pca, probe)

    print(f"    Clean (short): {clean_pred:.2f}  Corrupt (long): {corrupt_pred:.2f}  "
          f"Gap: {corrupt_pred - clean_pred:.2f}")

    # 2. Patch each layer: replace corrupt layer output with clean
    recoveries = []
    for patch_layer in range(n_layers):
        patched_final = run_with_patch(
            model, tokenizer, pair["long"], clean_states, patch_layer, n_layers
        )
        patched_pred = apply_probe(patched_final, scaler, pca, probe)

        # Recovery: how much of the gap does patching close?
        gap = corrupt_pred - clean_pred
        if abs(gap) > 0.01:
            recovery = (corrupt_pred - patched_pred) / gap
        else:
            recovery = 0.0

        recoveries.append({
            "layer": patch_layer,
            "patched_pred": round(float(patched_pred), 4),
            "recovery": round(float(recovery), 4),
        })

    return {
        "clean_pred": round(clean_pred, 4),
        "corrupt_pred": round(corrupt_pred, 4),
        "gap": round(float(corrupt_pred - clean_pred), 4),
        "recoveries": recoveries,
    }


def plot_patching(all_results, model_key, outdir):
    """Plot recovery fraction by layer, all pairs overlaid."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for result in all_results:
        layers = [r["layer"] for r in result["recoveries"]]
        recs = [r["recovery"] for r in result["recoveries"]]
        ax.plot(layers, recs, "o-", markersize=3, linewidth=1.2,
                label=f"{result['label']} (gap={result['gap']:.1f})", alpha=0.8)

    ax.set_xlabel("Patched Layer")
    ax.set_ylabel("Recovery Fraction\n(1.0 = fully recovers short-horizon prediction)")
    ax.set_title(f"{model_key} — Activation Patching: Temporal Horizon")
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.axhline(1, color="gray", ls="--", lw=0.5)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    fname = outdir / f"patching_{model_key}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def main():
    parser = argparse.ArgumentParser(description="SPAR RQ1 — Activation Patching")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    outdir = Path(args.results_dir)

    for model_key in args.models:
        if model_key not in MODELS:
            print(f"⚠  Unknown model '{model_key}'. Skipping.")
            continue

        model, tokenizer, n_layers = load_model(model_key)
        probe_layer = n_layers
        probe, scaler, pca = load_probe(model_key, probe_layer, args.results_dir)

        model_dir = outdir / model_key
        model_dir.mkdir(parents=True, exist_ok=True)

        all_results = []
        for pair in PATCH_PAIRS:
            print(f"\n  [{pair['label']}]")
            print(f"    Short: \"{pair['short'][:60]}\"")
            print(f"    Long:  \"{pair['long'][:60]}\"")

            result = run_patching_experiment(
                model, tokenizer, n_layers, probe, scaler, pca, pair
            )
            result["label"] = pair["label"]
            result["short"] = pair["short"]
            result["long"] = pair["long"]
            all_results.append(result)

            # Print top-3 most causally important layers
            sorted_recs = sorted(result["recoveries"], key=lambda r: r["recovery"], reverse=True)
            print(f"    Top causal layers: "
                  f"L{sorted_recs[0]['layer']}({sorted_recs[0]['recovery']:.2f}), "
                  f"L{sorted_recs[1]['layer']}({sorted_recs[1]['recovery']:.2f}), "
                  f"L{sorted_recs[2]['layer']}({sorted_recs[2]['recovery']:.2f})")

        plot_patching(all_results, model_key, model_dir)

        with open(model_dir / "patching_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✓ Saved patching_results.json → {model_dir}")

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
