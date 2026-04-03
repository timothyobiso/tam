"""
SPAR RQ1 — Ridge Regression Slider Probes
==========================================
Extract hidden states from Qwen3.5 / Gemma4 models,
train ridge regression probes for:
  - log(time_horizon)   continuous slider
  - planning_depth      continuous slider
  - urgency             continuous slider

Reports R² per layer, saves probe weights + activations.

Usage:
    python run_probes.py                          # DEFAULT_MODELS
    python run_probes.py --models qwen3.5-4b      # single model
    python run_probes.py --models qwen3.5-9b gemma4-e4b --stride 1
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import warnings
from sklearn.decomposition import PCA

from config import (
    MODELS, DEFAULT_MODELS, PROMPTS, TEMPORAL_PROMPTS,
    RIDGE_ALPHA, CV_FOLDS, LAYER_STRIDE, SEED,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Model Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_model(model_key: str, dtype=torch.bfloat16):
    """Load a HuggingFace model with output_hidden_states=True."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_id = MODELS[model_key]
    print(f"\n{'='*60}")
    print(f"Loading {hf_id}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Figure out layer count from the model config
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    print(f"  layers={n_layers}  d_model={d_model}")
    return model, tokenizer, n_layers, d_model


# ═══════════════════════════════════════════════════════════════════════════
# 2. Hidden State Extraction
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_last_token_states(
    model, tokenizer, prompts: list[str], layer_indices: list[int]
) -> dict[int, np.ndarray]:
    """
    For each prompt, run a forward pass and grab the last-token hidden state
    at each requested layer.

    Returns: {layer_idx: np.ndarray of shape (n_prompts, d_model)}
    """
    all_states = {li: [] for li in layer_indices}

    for prompt in tqdm(prompts, desc="  extracting"):
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        outputs = model(**inputs, output_hidden_states=True)
        # outputs.hidden_states is a tuple of (n_layers+1) tensors
        # index 0 = embedding, index i = after layer i
        hidden = outputs.hidden_states

        for li in layer_indices:
            # last token, squeeze batch dim, clamp bf16 extremes, move to cpu
            vec = hidden[li][0, -1, :].clamp(-1e4, 1e4).float().cpu().numpy()
            all_states[li].append(vec)

    return {li: np.stack(vecs) for li, vecs in all_states.items()}


# ═══════════════════════════════════════════════════════════════════════════
# 3. Probe Training (Ridge Regression Sliders)
# ═══════════════════════════════════════════════════════════════════════════

def train_ridge_probes(
    states: dict[int, np.ndarray],
    targets: dict[str, np.ndarray],
    cv_folds: int = CV_FOLDS,
    max_pca_dims: int = 20,
) -> dict[str, dict]:
    """
    Train a ridge regression probe at each layer for each target variable.
    Applies PCA first to handle n << d (24 prompts vs 3584+ hidden dims).

    Returns nested dict: {target_name: {layer_idx: {"r2": ..., "probe": ...}}}
    """
    results = {}

    for tname, y in targets.items():
        print(f"\n  Probing: {tname}  (n={len(y)})")
        layer_results = {}

        for li in sorted(states.keys()):
            X = states[li].copy()

            # Guard against inf/nan from bf16 overflow
            if not np.all(np.isfinite(X)):
                n_bad = np.sum(~np.isfinite(X))
                print(f"    layer {li:3d}  ⚠ {n_bad} inf/nan values, clipping")
                X = np.nan_to_num(X, nan=0.0, posinf=1e4, neginf=-1e4)

            # StandardScaler → PCA → Ridge
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            n_components = min(max_pca_dims, X_scaled.shape[0] - 2)  # leave room for CV
            pca = PCA(n_components=n_components, random_state=SEED)
            X_pca = pca.fit_transform(X_scaled)
            var_explained = pca.explained_variance_ratio_.sum()

            # RidgeCV picks alpha automatically
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                probe = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=cv_folds)
                probe.fit(X_pca, y)
                r2 = probe.score(X_pca, y)

                cv_scores = cross_val_score(
                    RidgeCV(alphas=np.logspace(-3, 3, 20)),
                    X_pca, y, cv=cv_folds, scoring="r2",
                )
                cv_r2 = cv_scores.mean()

            layer_results[li] = {
                "r2_train": round(r2, 4),
                "r2_cv": round(cv_r2, 4),
                "alpha": round(probe.alpha_, 4),
                "pca_dims": n_components,
                "pca_var": round(var_explained, 4),
                "probe": probe,
                "scaler": scaler,
                "pca": pca,
            }
            print(f"    layer {li:3d}  R²_train={r2:.4f}  R²_cv={cv_r2:.4f}  "
                  f"PCA={n_components}d ({var_explained:.0%} var)  α={probe.alpha_:.2e}")

        results[tname] = layer_results

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 4. Main
# ═══════════════════════════════════════════════════════════════════════════

def build_targets(prompt_list):
    """Extract target arrays from a list of prompt dicts."""
    targets = {}

    # log(time_horizon) — the key continuous variable
    raw_th = np.array([p["time_horizon"] for p in prompt_list])
    targets["log_time_horizon"] = np.log1p(raw_th)

    # planning_depth
    targets["planning_depth"] = np.array(
        [float(p["planning_depth"]) for p in prompt_list]
    )

    # urgency (only if present in all prompts)
    if all("urgency" in p for p in prompt_list):
        targets["urgency"] = np.array([p["urgency"] for p in prompt_list])

    return targets


def run_for_model(model_key: str, stride: int, outdir: Path, prompt_file: str = None):
    model, tokenizer, n_layers, d_model = load_model(model_key)
    layer_indices = list(range(0, n_layers + 1, stride))  # +1 for embedding layer
    model_dir = outdir / model_key
    model_dir.mkdir(parents=True, exist_ok=True)

    all_probe_results = {}
    all_states_saved = {}

    if prompt_file:
        # ── Use generated prompts (one big set) ────────────────────────
        with open(prompt_file) as f:
            gen_prompts = json.load(f)
        print(f"\n--- Generated prompts ({len(gen_prompts)}) from {prompt_file} ---")
        texts = [p["prompt"] for p in gen_prompts]
        states = extract_last_token_states(model, tokenizer, texts, layer_indices)
        targets = build_targets(gen_prompts)
        all_probe_results["generated"] = train_ridge_probes(states, targets)
        all_states_saved["generated_states"] = states
        all_states_saved["generated_targets"] = targets
    else:
        # ── Kitchen prompts ─────────────────────────────────────────────
        print(f"\n--- Kitchen prompts ({len(PROMPTS)}) ---")
        texts = [p["prompt"] for p in PROMPTS]
        states = extract_last_token_states(model, tokenizer, texts, layer_indices)
        targets = build_targets(PROMPTS)
        all_probe_results["kitchen"] = train_ridge_probes(states, targets)
        all_states_saved["kitchen_states"] = states
        all_states_saved["kitchen_targets"] = targets

        # ── Temporal prompts ────────────────────────────────────────────
        print(f"\n--- Temporal prompts ({len(TEMPORAL_PROMPTS)}) ---")
        texts = [p["prompt"] for p in TEMPORAL_PROMPTS]
        states = extract_last_token_states(model, tokenizer, texts, layer_indices)
        targets = build_targets(TEMPORAL_PROMPTS)
        all_probe_results["temporal"] = train_ridge_probes(states, targets)
        all_states_saved["temporal_states"] = states
        all_states_saved["temporal_targets"] = targets

    # ── Save ────────────────────────────────────────────────────────────
    summary = {
        "model": model_key,
        "hf_id": MODELS[model_key],
        "n_layers": n_layers,
        "d_model": d_model,
    }
    for setname, probe_results in all_probe_results.items():
        summary[setname] = {
            tname: {str(li): {
                "r2_train": float(v["r2_train"]),
                "r2_cv": float(v["r2_cv"]),
                "alpha": float(v["alpha"]),
                "pca_dims": int(v["pca_dims"]),
                "pca_var": float(v["pca_var"]),
            } for li, v in lresults.items()}
            for tname, lresults in probe_results.items()
        }

    with open(model_dir / "probe_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  ✓ Saved probe_results.json → {model_dir}")

    with open(model_dir / "activations.pkl", "wb") as f:
        pickle.dump(all_states_saved, f)

    with open(model_dir / "probes.pkl", "wb") as f:
        pickle.dump({
            setname: {
                tname: {li: {"probe": v["probe"], "scaler": v["scaler"], "pca": v["pca"]}
                        for li, v in lresults.items()}
                for tname, lresults in probe_results.items()
            }
            for setname, probe_results in all_probe_results.items()
        }, f)

    del model
    torch.cuda.empty_cache()
    return summary


def main():
    parser = argparse.ArgumentParser(description="SPAR RQ1 — Ridge Slider Probes")
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help=f"Model keys to run. Available: {list(MODELS.keys())}",
    )
    parser.add_argument("--stride", type=int, default=LAYER_STRIDE,
                        help="Probe every Nth layer (1=all)")
    parser.add_argument("--outdir", type=str, default="results",
                        help="Output directory")
    parser.add_argument("--prompts", type=str, default=None,
                        help="Path to prompts_generated.json (overrides config prompts)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    all_summaries = {}
    for model_key in args.models:
        if model_key not in MODELS:
            print(f"⚠  Unknown model key '{model_key}'. Skipping.")
            continue
        summary = run_for_model(model_key, args.stride, outdir, args.prompts)
        all_summaries[model_key] = summary

    with open(outdir / "all_results.json", "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Done. Results in {outdir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
