"""
SPAR RQ1 — Causal Steering with Temporal Probe Direction
=========================================================
Extracts the probe's learned direction in activation space, then adds
or subtracts it from the residual stream during generation via a forward
hook. If the temporal representation is causally efficacious:

  +alpha → model should plan further ahead (longer horizon language)
  -alpha → model should become myopic (immediate/short-term language)

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_steering.py --models qwen3.5-9b
    CUDA_VISIBLE_DEVICES=0 python run_steering.py --models qwen3.5-4b --alphas 0 10 30 50 -30 -50
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from config import MODELS, DEFAULT_MODELS, SEED


# ── Neutral prompts: no strong temporal framing, so steering has room to push ──
STEER_PROMPTS = [
    {
        "prompt": "The chef looked at the kitchen and began to think about",
        "label": "neutral_chef",
    },
    {
        "prompt": "The next thing to focus on is",
        "label": "neutral_next",
    },
    {
        "prompt": "Looking ahead, the most important consideration is",
        "label": "neutral_ahead",
    },
    {
        "prompt": "The restaurant's plan involves",
        "label": "neutral_plan",
    },
    {
        "prompt": "After reviewing the situation, the first priority should be",
        "label": "neutral_priority",
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


def extract_steering_vector(model_key, layer_idx, results_dir="results"):
    """
    Extract the temporal steering direction from the fitted probe.

    The probe pipeline is: raw → StandardScaler → PCA → Ridge.
    The steering vector in the original activation space is:
        v = scaler.scale_^{-1} @ pca.components_.T @ ridge.coef_
    normalized to unit length.
    """
    probe_path = Path(results_dir) / model_key / "probes.pkl"
    with open(probe_path, "rb") as f:
        probes = pickle.load(f)

    # Find the probe
    for setname in ["generated", "kitchen", "temporal"]:
        if setname in probes and "log_time_horizon" in probes[setname]:
            if layer_idx in probes[setname]["log_time_horizon"]:
                p = probes[setname]["log_time_horizon"][layer_idx]
                probe, scaler, pca = p["probe"], p["scaler"], p["pca"]
                print(f"  Loaded probe from {setname}/log_time_horizon @ layer {layer_idx}")
                break
    else:
        raise KeyError(f"No probe at layer {layer_idx}")

    # Reconstruct the direction in original activation space
    # Ridge coefs in PCA space
    ridge_coef = probe.coef_  # shape (n_pca_components,)

    # Project back through PCA
    direction_scaled = pca.components_.T @ ridge_coef  # shape (d_model,)

    # Undo scaling (divide by scale to get direction in raw space)
    direction_raw = direction_scaled / scaler.scale_

    # Normalize
    direction_raw = direction_raw / np.linalg.norm(direction_raw)

    return torch.tensor(direction_raw, dtype=torch.bfloat16)


@torch.no_grad()
def generate_steered(
    model, tokenizer, prompt, steering_vec, layer_idx, alpha, max_tokens=150,
):
    """
    Generate with a steering vector added to the residual stream at layer_idx.
    alpha > 0 pushes toward longer temporal horizons.
    alpha < 0 pushes toward shorter/immediate horizons.
    alpha = 0 is the baseline.
    """
    # Place steering vector on the right device
    # For multi-device models, find which device has this layer
    target_device = None
    if hasattr(model, 'hf_device_map'):
        for name, device in model.hf_device_map.items():
            if f"layers.{layer_idx}" in name or name == f"model.layers.{layer_idx}":
                target_device = f"cuda:{device}" if isinstance(device, int) else device
                break
    if target_device is None:
        target_device = model.device
    
    steer = (steering_vec * alpha).to(target_device)

    # Register hook
    handles = []
    def hook_fn(module, input, output):
        # output is typically a tuple; [0] is the hidden state
        if isinstance(output, tuple):
            h = output[0]
            h[:, :, :] = h + steer  # broadcast across batch and seq
            return (h,) + output[1:]
        else:
            output[:, :, :] = output + steer
            return output

    # Find the layer module
    layer_module = model.model.layers[layer_idx]
    handle = layer_module.register_forward_hook(hook_fn)
    handles.append(handle)

    # Generate
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        do_sample=False,
        temperature=1.0,
    )

    # Remove hook
    for h in handles:
        h.remove()

    generated = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    return generated


def main():
    parser = argparse.ArgumentParser(description="SPAR RQ1 — Causal Steering")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer to steer at (default: final)")
    parser.add_argument("--alphas", nargs="+", type=float,
                        default=[0, 10, 30, 50, 100, -10, -30, -50, -100],
                        help="Steering strengths (0=baseline)")
    parser.add_argument("--max_tokens", type=int, default=150)
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    outdir = Path(args.results_dir)

    for model_key in args.models:
        if model_key not in MODELS:
            print(f"⚠  Unknown model '{model_key}'. Skipping.")
            continue

        model, tokenizer, n_layers = load_model(model_key)
        # Probe was trained on hidden_states[n_layers] (output after last layer)
        # Hook attaches to model.model.layers[n_layers-1] (the last layer module)
        probe_layer = args.layer if args.layer is not None else n_layers
        hook_layer = probe_layer - 1 if probe_layer > 0 else 0
        steering_vec = extract_steering_vector(model_key, probe_layer, args.results_dir)
        print(f"  Probe layer: {probe_layer}  Hook layer: {hook_layer}")

        # Move to model device
        steering_vec = steering_vec.to(model.device)

        model_dir = outdir / model_key
        model_dir.mkdir(parents=True, exist_ok=True)

        all_results = []

        for pinfo in STEER_PROMPTS:
            print(f"\n  [{pinfo['label']}] \"{pinfo['prompt'][:50]}...\"")
            prompt_results = {"label": pinfo["label"], "prompt": pinfo["prompt"], "generations": {}}

            for alpha in sorted(args.alphas):
                text = generate_steered(
                    model, tokenizer, pinfo["prompt"],
                    steering_vec, hook_layer, alpha, args.max_tokens,
                )
                tag = f"α={alpha:+.0f}"
                prompt_results["generations"][tag] = text
                direction = "→ LONG" if alpha > 0 else "→ SHORT" if alpha < 0 else "BASELINE"
                print(f"    {tag:>8s} {direction:>10s}:  {text[:100]}...")

            all_results.append(prompt_results)

        # Save
        output = {
            "model": model_key,
            "probe_layer": probe_layer,
            "hook_layer": hook_layer,
            "alphas": sorted(args.alphas),
            "results": all_results,
        }
        with open(model_dir / "steering_results.json", "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n✓ Saved steering_results.json → {model_dir}")

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
