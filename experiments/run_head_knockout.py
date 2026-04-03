"""
SPAR RQ1 — Attention Head Knockout
====================================
Ablates one attention head at a time in L29-31 (the causally critical
layers from patching) and measures how much the temporal probe prediction
degrades. Identifies specific heads responsible for temporal computation.

Method: for each head, replace its output with zeros during the forward
pass on contrastive prompt pairs, then measure the probe prediction gap.
Heads where ablation collapses the gap are the temporal circuit.

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_head_knockout.py --models qwen3.5-9b
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from config import MODELS, DEFAULT_MODELS, SEED


CONTRASTIVE_PAIRS = [
    ("In the next 30 seconds, the chef needs to",
     "Over the next five years, the restaurant needs to"),
    ("Immediately flip the steak before it",
     "Plan the seasonal menu rotation for the coming year so that"),
    ("The timer is going off right now, quickly",
     "Develop a long-term training program for new kitchen staff that"),
    ("In the next minute, finish plating the",
     "Over the next decade, build the restaurant brand by"),
    ("Right now, turn down the heat on the",
     "Over the coming months, redesign the entire kitchen workflow to"),
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
    n_heads = model.config.num_attention_heads
    d_model = model.config.hidden_size
    head_dim = d_model // n_heads
    print(f"  layers={n_layers}  heads={n_heads}  d_model={d_model}  head_dim={head_dim}")
    return model, tokenizer, n_layers, n_heads, head_dim


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
    vec = hidden_state.clamp(-1e4, 1e4).float().cpu().numpy().reshape(1, -1)
    return float(probe.predict(pca.transform(scaler.transform(vec)))[0])


@torch.no_grad()
def get_baseline_gap(model, tokenizer, short, long, probe, scaler, pca, probe_layer):
    """Get probe predictions for short and long prompts without ablation."""
    preds = {}
    for label, prompt in [("short", short), ("long", long)]:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model(**inputs, output_hidden_states=True)
        preds[label] = apply_probe(outputs.hidden_states[probe_layer][0, -1, :],
                                    scaler, pca, probe)
    return preds["short"], preds["long"], preds["long"] - preds["short"]


@torch.no_grad()
def get_ablated_gap(model, tokenizer, short, long, probe, scaler, pca,
                     probe_layer, ablate_layer, head_idx, n_heads, head_dim):
    """
    Ablate one attention head by zeroing its output contribution,
    then measure the probe gap.
    """
    def make_hook(hi, hd):
        def hook_fn(module, input, output):
            # output is typically (attn_output, attn_weights, ...) or just attn_output
            # For most HF models, the attention layer output[0] is the projected output
            # of shape (batch, seq, d_model). We zero the slice for head hi.
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output

            start = hi * hd
            end = (hi + 1) * hd
            h[:, :, start:end] = 0.0

            if isinstance(output, tuple):
                return (h,) + output[1:]
            return h
        return hook_fn

    # Find the attention module — different models use different attribute names
    layer = model.model.layers[ablate_layer]
    attn_module = None
    for attr in ["self_attn", "attn", "attention", "self_attention"]:
        if hasattr(layer, attr):
            attn_module = getattr(layer, attr)
            break
    if attn_module is None:
        # Fallback: search for any module with 'attn' in its name
        for name, module in layer.named_children():
            if "attn" in name.lower():
                attn_module = module
                break
    if attn_module is None:
        raise AttributeError(f"Cannot find attention module in {type(layer).__name__}. "
                            f"Children: {[n for n, _ in layer.named_children()]}")

    hook = make_hook(head_idx, head_dim)
    handle = attn_module.register_forward_hook(hook)

    preds = {}
    for label, prompt in [("short", short), ("long", long)]:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model(**inputs, output_hidden_states=True)
        preds[label] = apply_probe(outputs.hidden_states[probe_layer][0, -1, :],
                                    scaler, pca, probe)

    handle.remove()
    return preds["short"], preds["long"], preds["long"] - preds["short"]


def plot_head_importance(results, model_key, outdir):
    """Heatmap: layers × heads, color = gap reduction from ablation."""
    layers = sorted(results.keys())
    n_heads = len(results[layers[0]])

    matrix = np.zeros((len(layers), n_heads))
    for i, layer in enumerate(layers):
        for j in range(n_heads):
            matrix[i, j] = results[layer][j]["gap_reduction"]

    fig, ax = plt.subplots(figsize=(max(12, n_heads * 0.4), 4))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlBu_r", interpolation="nearest")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{l}" for l in layers])
    ax.set_xlabel("Attention Head")
    ax.set_ylabel("Layer")
    ax.set_title(f"{model_key} — Head Knockout: Gap Reduction\n(higher = more important for temporal computation)")
    plt.colorbar(im, ax=ax, label="Gap reduction (fraction)")

    # Annotate top heads
    for i, layer in enumerate(layers):
        for j in range(n_heads):
            val = matrix[i, j]
            if abs(val) > 0.15:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6,
                       color="white" if abs(val) > 0.3 else "black")

    plt.tight_layout()
    fname = outdir / f"head_knockout_{model_key}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def main():
    parser = argparse.ArgumentParser(description="SPAR RQ1 — Head Knockout")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--layers", nargs="+", type=int, default=None,
                        help="Layers to ablate (default: last 3)")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    outdir = Path(args.results_dir)

    for model_key in args.models:
        if model_key not in MODELS:
            print(f"⚠  Unknown model '{model_key}'. Skipping.")
            continue

        model, tokenizer, n_layers, n_heads, head_dim = load_model(model_key)
        probe_layer = n_layers
        probe, scaler, pca = load_probe(model_key, probe_layer, args.results_dir)

        ablate_layers = args.layers if args.layers else [n_layers - 3, n_layers - 2, n_layers - 1]
        # Print detected attention module
        layer0 = model.model.layers[ablate_layers[0]]
        print(f"  Layer children: {[n for n, _ in layer0.named_children()]}")
        print(f"  Ablating layers: {ablate_layers}  ({n_heads} heads each)")

        model_dir = outdir / model_key
        model_dir.mkdir(parents=True, exist_ok=True)

        # Get baselines
        print(f"\n  Computing baselines...")
        baselines = []
        for short, long in CONTRASTIVE_PAIRS:
            s, l, gap = get_baseline_gap(model, tokenizer, short, long, probe, scaler, pca, probe_layer)
            baselines.append({"short": s, "long": l, "gap": gap})
            print(f"    gap={gap:.2f}  short={s:.2f}  long={l:.2f}")
        mean_baseline_gap = np.mean([b["gap"] for b in baselines])
        print(f"  Mean baseline gap: {mean_baseline_gap:.2f}")

        # Ablate each head
        all_results = {}
        for layer_idx in ablate_layers:
            print(f"\n  Layer {layer_idx}: ablating {n_heads} heads...")
            layer_results = {}

            for head_idx in range(n_heads):
                gaps = []
                for (short, long), baseline in zip(CONTRASTIVE_PAIRS, baselines):
                    _, _, ablated_gap = get_ablated_gap(
                        model, tokenizer, short, long, probe, scaler, pca,
                        probe_layer, layer_idx, head_idx, n_heads, head_dim,
                    )
                    gaps.append(ablated_gap)

                mean_ablated_gap = float(np.mean(gaps))
                gap_reduction = (mean_baseline_gap - mean_ablated_gap) / mean_baseline_gap if abs(mean_baseline_gap) > 0.01 else 0.0

                layer_results[head_idx] = {
                    "mean_ablated_gap": round(mean_ablated_gap, 4),
                    "gap_reduction": round(float(gap_reduction), 4),
                }

                if abs(gap_reduction) > 0.1:
                    print(f"    Head {head_idx:2d}: gap={mean_ablated_gap:.2f}  "
                          f"reduction={gap_reduction:+.2f} {'⚠️' if gap_reduction > 0.2 else ''}")

            all_results[layer_idx] = layer_results

        # Summary: top heads
        print(f"\n  {'='*50}")
        print(f"  Top temporal heads (by gap reduction):")
        all_heads = []
        for layer_idx, heads in all_results.items():
            for head_idx, info in heads.items():
                all_heads.append((layer_idx, head_idx, info["gap_reduction"]))
        all_heads.sort(key=lambda x: x[2], reverse=True)
        for layer, head, red in all_heads[:10]:
            print(f"    L{layer}H{head}: {red:+.3f}")

        plot_head_importance(all_results, model_key, model_dir)

        output = {
            "model": model_key,
            "n_heads": n_heads,
            "head_dim": head_dim,
            "ablate_layers": ablate_layers,
            "mean_baseline_gap": round(mean_baseline_gap, 4),
            "baselines": baselines,
            "head_results": {str(k): {str(h): v for h, v in heads.items()}
                            for k, heads in all_results.items()},
        }
        with open(model_dir / "head_knockout_results.json", "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n✓ Saved head_knockout_results.json → {model_dir}")

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
