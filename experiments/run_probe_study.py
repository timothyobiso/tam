"""
SPAR RQ1 — Probe Improvement Study
====================================
Systematically tests probe design choices to find the best temporal
representation decoder:

1. PCA components: 20 vs 50 vs 100 vs no-PCA
2. Pooling: last-token vs mean-pool vs last-4-tokens
3. Layer selection: final layer vs concat L29-31 vs concat L24-32
4. Training data: temporal prompts vs bare recipes (no time words)
5. Regularization: Ridge vs ElasticNet vs kernel ridge

The goal is to both maximize R² AND test whether a probe trained
on bare task descriptions (no temporal words) can decode real
cooking times — which would demonstrate temporal reasoning, not
just temporal language decoding.

Usage:
    # Phase 1: test probe architectures on existing activations
    python run_probe_study.py --models qwen3.5-9b --phase arch

    # Phase 2: train bare-recipe probe (needs GPU for extraction)
    CUDA_VISIBLE_DEVICES=0 python run_probe_study.py --models qwen3.5-9b --phase bare
"""

import argparse
import json
import pickle
import random
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.linear_model import RidgeCV, ElasticNetCV, LassoCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt

from config import MODELS, DEFAULT_MODELS, SEED, CV_FOLDS

random.seed(SEED)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Architecture search on existing activations
# ═══════════════════════════════════════════════════════════════════════════

def load_existing_activations(model_key, results_dir="results"):
    """Load saved activations from run_probes.py"""
    act_path = Path(results_dir) / model_key / "activations.pkl"
    with open(act_path, "rb") as f:
        data = pickle.load(f)
    states = data.get("generated_states", data.get("kitchen_states"))

    prompts_path = Path("prompts_generated.json")
    with open(prompts_path) as f:
        prompts = json.load(f)

    y = np.log1p(np.array([p["time_horizon"] for p in prompts]))
    return states, y, prompts


def load_domain_activations(model_key, results_dir="results"):
    """Load multi-domain activations from run_domain_transfer.py"""
    act_path = Path(results_dir) / model_key / "domain_activations.pkl"
    if not act_path.exists():
        return None, None, None
    with open(act_path, "rb") as f:
        states = pickle.load(f)

    prompts_path = Path("prompts_multidomain.json")
    if not prompts_path.exists():
        return None, None, None
    with open(prompts_path) as f:
        prompts = json.load(f)

    y = np.log1p(np.array([p["time_horizon"] for p in prompts]))
    return states, y, prompts


def test_pca_components(X, y, components_list=[10, 20, 50, 100, 200, None]):
    """Test different PCA dimensionality."""
    results = []
    for n_comp in components_list:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            if n_comp is not None and n_comp < X_scaled.shape[0] - 1:
                pca = PCA(n_components=min(n_comp, X_scaled.shape[1]), random_state=SEED)
                X_proj = pca.fit_transform(X_scaled)
                var_explained = sum(pca.explained_variance_ratio_)
            else:
                X_proj = X_scaled
                n_comp = X_scaled.shape[1]
                var_explained = 1.0

            scores = cross_val_score(
                RidgeCV(alphas=np.logspace(-3, 3, 20)),
                X_proj, y, cv=CV_FOLDS, scoring="r2"
            )

        results.append({
            "n_components": n_comp if n_comp else "full",
            "var_explained": round(float(var_explained), 4),
            "r2_mean": round(float(scores.mean()), 4),
            "r2_std": round(float(scores.std()), 4),
        })
    return results


def test_multi_layer(states, y, layer_combos):
    """Test concatenating activations from multiple layers."""
    results = []
    for combo_name, layer_indices in layer_combos.items():
        available = [li for li in layer_indices if li in states]
        if not available:
            continue

        X_concat = np.concatenate([states[li] for li in available], axis=1)
        X_concat = np.nan_to_num(X_concat, nan=0.0, posinf=1e4, neginf=-1e4)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_concat)
            n_comp = min(50, X_scaled.shape[0] - 2, X_scaled.shape[1])
            pca = PCA(n_components=n_comp, random_state=SEED)
            X_proj = pca.fit_transform(X_scaled)

            ridge_scores = cross_val_score(
                RidgeCV(alphas=np.logspace(-3, 3, 20)),
                X_proj, y, cv=CV_FOLDS, scoring="r2"
            )
            mlp_scores = cross_val_score(
                MLPRegressor(hidden_layer_sizes=(64,), max_iter=2000,
                            random_state=SEED, early_stopping=True,
                            validation_fraction=0.15, alpha=0.01),
                X_proj, y, cv=CV_FOLDS, scoring="r2"
            )

        results.append({
            "combo": combo_name,
            "layers": available,
            "d_concat": X_concat.shape[1],
            "pca_components": n_comp,
            "ridge_r2": round(float(ridge_scores.mean()), 4),
            "mlp_r2": round(float(mlp_scores.mean()), 4),
        })
    return results


def test_regularization(X, y, n_comp=50):
    """Test different regularization methods."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=min(n_comp, X_scaled.shape[0] - 2), random_state=SEED)
        X_proj = pca.fit_transform(X_scaled)

        results = []

        # Ridge
        scores = cross_val_score(
            RidgeCV(alphas=np.logspace(-3, 3, 20)),
            X_proj, y, cv=CV_FOLDS, scoring="r2"
        )
        results.append({"method": "RidgeCV", "r2": round(float(scores.mean()), 4)})

        # Lasso
        scores = cross_val_score(
            LassoCV(alphas=np.logspace(-3, 1, 20), max_iter=5000, random_state=SEED),
            X_proj, y, cv=CV_FOLDS, scoring="r2"
        )
        results.append({"method": "LassoCV", "r2": round(float(scores.mean()), 4)})

        # ElasticNet
        scores = cross_val_score(
            ElasticNetCV(alphas=np.logspace(-3, 1, 10), l1_ratio=[0.1, 0.5, 0.9],
                        max_iter=5000, random_state=SEED),
            X_proj, y, cv=CV_FOLDS, scoring="r2"
        )
        results.append({"method": "ElasticNetCV", "r2": round(float(scores.mean()), 4)})

        # Kernel Ridge (RBF)
        kr = GridSearchCV(
            KernelRidge(kernel="rbf"),
            param_grid={"alpha": [0.1, 1, 10], "gamma": [0.001, 0.01, 0.1]},
            cv=CV_FOLDS, scoring="r2"
        )
        kr.fit(X_proj, y)
        results.append({"method": "KernelRidge(RBF)", "r2": round(float(kr.best_score_), 4)})

        # MLP
        scores = cross_val_score(
            MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=2000,
                        random_state=SEED, early_stopping=True,
                        validation_fraction=0.15, alpha=0.01),
            X_proj, y, cv=CV_FOLDS, scoring="r2"
        )
        results.append({"method": "MLP(128,64)", "r2": round(float(scores.mean()), 4)})

    return results


def run_phase1(model_key, results_dir):
    """Architecture search on existing activations."""
    states, y, prompts = load_existing_activations(model_key, results_dir)
    n_layers_max = max(states.keys())

    print(f"\n{'='*60}")
    print(f"Probe Architecture Study: {model_key}")
    print(f"  n={len(y)}, d={states[n_layers_max].shape[1]}")
    print(f"{'='*60}")

    X_final = states[n_layers_max].copy()
    X_final = np.nan_to_num(X_final, nan=0.0, posinf=1e4, neginf=-1e4)

    # Test 1: PCA components
    print(f"\n  ── PCA Components (final layer) ──")
    pca_results = test_pca_components(X_final, y,
                                       [10, 20, 50, 100, 200])
    for r in pca_results:
        print(f"    PCA({r['n_components']:>4}): R²={r['r2_mean']:.4f} ± {r['r2_std']:.4f}  "
              f"(var={r['var_explained']:.2%})")

    # Test 2: Multi-layer concatenation
    print(f"\n  ── Multi-Layer Concatenation ──")
    combos = {
        f"L{n_layers_max} only": [n_layers_max],
        f"L{n_layers_max-2}+L{n_layers_max}": [n_layers_max - 2, n_layers_max],
        f"L{n_layers_max-4}..L{n_layers_max}": list(range(n_layers_max - 4, n_layers_max + 1, 2)),
    }
    # Add causal layers (L29-31 mapped to even indices in saved activations)
    causal = [li for li in states.keys() if li >= n_layers_max - 4]
    combos[f"causal({','.join(str(l) for l in causal)})"] = causal
    # All layers
    combos["all_layers"] = sorted(states.keys())

    layer_results = test_multi_layer(states, y, combos)
    for r in layer_results:
        print(f"    {r['combo']:>30s}: ridge={r['ridge_r2']:.4f}  mlp={r['mlp_r2']:.4f}  "
              f"(d={r['d_concat']}, pca→{r['pca_components']})")

    # Test 3: Regularization methods
    print(f"\n  ── Regularization Methods (PCA=50, final layer) ──")
    reg_results = test_regularization(X_final, y, n_comp=50)
    for r in reg_results:
        print(f"    {r['method']:>20s}: R²={r['r2']:.4f}")

    # Test 4: Domain data (if available)
    domain_states, domain_y, domain_prompts = load_domain_activations(model_key, results_dir)
    domain_pca_results = None
    if domain_states is not None:
        print(f"\n  ── Multi-Domain Data (n={len(domain_y)}, final layer) ──")
        X_dom = domain_states[n_layers_max].copy()
        X_dom = np.nan_to_num(X_dom, nan=0.0, posinf=1e4, neginf=-1e4)
        domain_pca_results = test_pca_components(X_dom, domain_y,
                                                  [20, 50, 100])
        for r in domain_pca_results:
            print(f"    PCA({r['n_components']:>4}): R²={r['r2_mean']:.4f}")

    return {
        "pca_components": pca_results,
        "multi_layer": layer_results,
        "regularization": reg_results,
        "domain_pca": domain_pca_results,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Bare-recipe probe (temporal reasoning test)
# ═══════════════════════════════════════════════════════════════════════════

def load_bare_recipes(n_recipes=500):
    """Load recipes, create bare prompts (no time words)."""
    from datasets import load_dataset
    import re

    print(f"  Loading recipe dataset...")
    ds = load_dataset("SDMN2001/recipe_difficulties", split="train")

    valid = []
    for row in ds:
        total = row.get("total_minutes", 0)
        title = row.get("title", "")
        ingredients = row.get("clean_ingredients", "")
        if total and total > 0 and title and ingredients:
            # Create bare prompt: recipe name + ingredients, NO time words
            ing_str = ingredients if isinstance(ingredients, str) else \
                      ", ".join(ingredients) if isinstance(ingredients, list) else str(ingredients)
            if len(ing_str) > 300:
                ing_str = ing_str[:300]

            prompt = f"Recipe: {title}\nIngredients: {ing_str}\nThis recipe involves"

            # Verify no temporal words leaked in
            temporal_words = ["minute", "hour", "second", "day", "week", "month",
                            "year", "quick", "fast", "slow", "overnight", "time"]
            has_temporal = any(w in prompt.lower() for w in temporal_words)

            valid.append({
                "prompt": prompt,
                "total_minutes": int(total),
                "title": title,
                "has_temporal_words": has_temporal,
            })

    # Filter out prompts with temporal words
    clean = [r for r in valid if not r["has_temporal_words"]]
    print(f"  {len(valid)} recipes total, {len(clean)} without temporal words")

    # Stratified sample
    bins = [(1, 15), (15, 45), (45, 120), (120, 10000)]
    per_bin = n_recipes // len(bins)
    sampled = []
    for lo, hi in bins:
        pool = [r for r in clean if lo <= r["total_minutes"] < hi]
        chosen = random.sample(pool, min(per_bin, len(pool)))
        sampled.extend(chosen)
        print(f"    {lo}-{hi} min: {len(chosen)} (from {len(pool)})")

    random.shuffle(sampled)
    return sampled


def run_phase2(model_key, results_dir):
    """
    Train probe on bare recipe descriptions → real cooking times.
    If this works, the model represents temporal duration from task
    semantics, not just temporal words.
    """
    import torch
    from tqdm import tqdm

    from config import MODELS

    recipes = load_bare_recipes(500)
    y = np.log1p(np.array([r["total_minutes"] for r in recipes]))

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_id = MODELS[model_key]
    print(f"  Loading {hf_id}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers

    # Extract activations
    print(f"  Extracting bare-recipe activations...")
    layers_to_extract = [n_layers - 4, n_layers - 2, n_layers]
    states = defaultdict(list)

    with torch.no_grad():
        for r in tqdm(recipes, desc="  extracting"):
            inputs = tokenizer(r["prompt"], return_tensors="pt",
                             truncation=True, max_length=1024)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True)
            for li in layers_to_extract:
                vec = outputs.hidden_states[li][0, -1, :].clamp(-1e4, 1e4).float().cpu().numpy()
                states[li].append(vec)

    states = {li: np.stack(v) for li, v in states.items()}

    del model
    torch.cuda.empty_cache()

    # Train and evaluate probes
    print(f"\n  ── Bare-Recipe Probe Results ──")
    bare_results = {}

    for li in layers_to_extract:
        X = states[li].copy()
        X = np.nan_to_num(X, nan=0.0, posinf=1e4, neginf=-1e4)

        for n_comp in [20, 50, 100]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                nc = min(n_comp, X_scaled.shape[0] - 2, X_scaled.shape[1])
                pca = PCA(n_components=nc, random_state=SEED)
                X_proj = pca.fit_transform(X_scaled)

                ridge_scores = cross_val_score(
                    RidgeCV(alphas=np.logspace(-3, 3, 20)),
                    X_proj, y, cv=CV_FOLDS, scoring="r2"
                )
                mlp_scores = cross_val_score(
                    MLPRegressor(hidden_layer_sizes=(64,), max_iter=2000,
                                random_state=SEED, early_stopping=True,
                                validation_fraction=0.15, alpha=0.01),
                    X_proj, y, cv=CV_FOLDS, scoring="r2"
                )

            key = f"L{li}_PCA{nc}"
            bare_results[key] = {
                "layer": li,
                "pca": nc,
                "ridge_r2": round(float(ridge_scores.mean()), 4),
                "mlp_r2": round(float(mlp_scores.mean()), 4),
            }
            print(f"    {key:>15s}: ridge={ridge_scores.mean():.4f}  mlp={mlp_scores.mean():.4f}")

    return bare_results, states, y, recipes


def main():
    parser = argparse.ArgumentParser(description="SPAR RQ1 — Probe Study")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--phase", type=str, default="both",
                        choices=["arch", "bare", "both"])
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    outdir = Path(args.results_dir)

    for model_key in args.models:
        if model_key not in MODELS:
            print(f"⚠  Unknown model '{model_key}'. Skipping.")
            continue

        model_dir = outdir / model_key
        model_dir.mkdir(parents=True, exist_ok=True)

        all_results = {}

        if args.phase in ["arch", "both"]:
            all_results["architecture"] = run_phase1(model_key, args.results_dir)

        if args.phase in ["bare", "both"]:
            bare_results, states, y, recipes = run_phase2(model_key, args.results_dir)
            all_results["bare_recipe"] = bare_results

            # Save bare activations for reuse
            with open(model_dir / "bare_recipe_activations.pkl", "wb") as f:
                pickle.dump({"states": states, "y": y, "recipes": recipes}, f)

        with open(model_dir / "probe_study_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✓ Saved probe_study_results.json → {model_dir}")


if __name__ == "__main__":
    main()
