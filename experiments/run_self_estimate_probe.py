"""
SPAR RQ1 — Probe the Model's Own Time Estimate
=================================================
Instead of probing hidden states for ground-truth cooking time
(which failed - R²=0.58, all category), probe for THE MODEL'S
OWN GENERATED ESTIMATE.

Logic:
  1. The model generates "240" as its cooking time estimate
  2. The hidden state before that token must encode "240"
  3. Train probe: hidden state → model's generated number
  4. Separately, we already know model estimates correlate
     with reality (r=0.55)

If probe → model_estimate works (high R²), then:
  probe reads model's intent → intent correlates with reality
  = indirect but real temporal reasoning monitor

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_self_estimate_probe.py --models qwen3.5-9b
"""

import argparse
import json
import pickle
import random
import re
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from scipy import stats as sp_stats

from config import MODELS, DEFAULT_MODELS, SEED, CV_FOLDS

random.seed(SEED)


def load_recipes(n_recipes=500):
    from datasets import load_dataset
    ds = load_dataset("SDMN2001/recipe_difficulties", split="train")
    valid = []
    for row in ds:
        total = row.get("total_minutes", 0)
        title = row.get("title", "")
        ingredients = row.get("clean_ingredients", "")
        if total and total > 0 and title and ingredients:
            ing_str = ingredients if isinstance(ingredients, str) else \
                      ", ".join(ingredients) if isinstance(ingredients, list) else str(ingredients)
            if len(ing_str) > 300:
                ing_str = ing_str[:300]
            valid.append({"title": title, "ingredients": ing_str, "total_minutes": int(total)})

    bins = [(1, 15), (15, 45), (45, 120), (120, 10000)]
    per_bin = n_recipes // len(bins)
    sampled = []
    for lo, hi in bins:
        pool = [r for r in valid if lo <= r["total_minutes"] < hi]
        sampled.extend(random.sample(pool, min(per_bin, len(pool))))
    random.shuffle(sampled)
    return sampled


def load_model(model_key):
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
    return model, tokenizer, n_layers


def extract_number(text):
    matches = re.findall(r'(\d+(?:\.\d+)?)', text[:100])
    if matches:
        return float(matches[0])
    return None


@torch.no_grad()
def extract_and_generate(model, tokenizer, prompt, n_layers, max_new=20):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model(**inputs, output_hidden_states=True)
    states = {}
    for li in [n_layers - 4, n_layers - 2, n_layers]:
        states[li] = outputs.hidden_states[li][0, -1, :].clamp(-1e4, 1e4).float().cpu().numpy()

    input_ids = inputs["input_ids"]
    gen_ids = model.generate(
        input_ids, max_new_tokens=max_new, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_text = tokenizer.decode(gen_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    return states, gen_text


def train_and_eval(X, y, n_comp=50):
    X = np.nan_to_num(X, nan=0.0, posinf=1e4, neginf=-1e4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        nc = min(n_comp, X_scaled.shape[0] - 2, X_scaled.shape[1])
        pca = PCA(n_components=nc, random_state=SEED)
        X_proj = pca.fit_transform(X_scaled)
        scores = cross_val_score(
            RidgeCV(alphas=np.logspace(-3, 3, 20)),
            X_proj, y, cv=CV_FOLDS, scoring="r2"
        )
    return float(scores.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--n_recipes", type=int, default=500)
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    outdir = Path(args.results_dir)
    recipes = load_recipes(args.n_recipes)

    for model_key in args.models:
        if model_key not in MODELS:
            continue

        model, tokenizer, n_layers = load_model(model_key)
        model_dir = outdir / model_key
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Self-Estimate Probe: {model_key}  ({len(recipes)} recipes)")
        print(f"{'='*60}")

        prompt_template = "Recipe: {title}\nIngredients: {ingredients}\nEstimated cooking time in minutes:"

        states_by_layer = defaultdict(list)
        model_estimates = []
        real_times = []
        valid_indices = []

        for i, recipe in enumerate(tqdm(recipes, desc="  extracting")):
            prompt = prompt_template.format(
                title=recipe["title"], ingredients=recipe["ingredients"]
            )
            states, gen_text = extract_and_generate(model, tokenizer, prompt, n_layers)
            num = extract_number(gen_text)

            if num is not None and 0 < num < 100000:
                for li, vec in states.items():
                    states_by_layer[li].append(vec)
                model_estimates.append(num)
                real_times.append(recipe["total_minutes"])
                valid_indices.append(i)

        del model
        torch.cuda.empty_cache()

        n_valid = len(model_estimates)
        print(f"\n  {n_valid}/{len(recipes)} recipes with valid generated estimates")

        log_model = np.log1p(np.array(model_estimates))
        log_real = np.log1p(np.array(real_times))

        # Baseline: model estimates vs reality
        r_gen, p_gen = sp_stats.pearsonr(log_model, log_real)
        print(f"\n  Model generation accuracy: r={r_gen:.4f} (p={p_gen:.2e})")

        # ── KEY TEST: Probe → model's own estimate ──
        print(f"\n  ── Probe → Model's Own Estimate ──")
        print(f"  {'Target':<25s} {'L28':>8s} {'L30':>8s} {'L32':>8s}")

        for target_name, y_target in [
            ("ground truth", log_real),
            ("model estimate", log_model),
            ("model error", log_model - log_real),
        ]:
            row = f"  {target_name:<25s}"
            for li in sorted(states_by_layer.keys()):
                X = np.stack(states_by_layer[li])
                r2 = train_and_eval(X, y_target, n_comp=50)
                row += f" {r2:>8.4f}"
            print(row)

        # ── Within-category check on model estimate ──
        print(f"\n  ── Within-Category: Model Estimate ──")
        bins = defaultdict(list)
        for i, rt in enumerate(real_times):
            if rt < 15: bins["quick"].append(i)
            elif rt < 45: bins["medium"].append(i)
            elif rt < 120: bins["long"].append(i)
            else: bins["very_long"].append(i)

        # Residual: model estimate after removing bin mean
        pred_y = np.zeros(len(log_model))
        for bname, idxs in bins.items():
            pred_y[idxs] = np.mean(log_model[idxs])
        residual_model = log_model - pred_y

        X_final = np.stack(states_by_layer[n_layers])
        r2_residual = train_and_eval(X_final, residual_model, n_comp=50)
        print(f"  Residual R² (model estimate, after bin means): {r2_residual:.4f}")

        # Also: can probe distinguish model's over vs under estimates?
        error = log_model - log_real
        r2_error = train_and_eval(X_final, error, n_comp=50)
        print(f"  Probe → model error (over/under estimate): R²={r2_error:.4f}")

        # ── Correlation chain ──
        print(f"\n  ── Correlation Chain ──")
        # Train probe, get predictions
        X = np.nan_to_num(X_final, nan=0.0, posinf=1e4, neginf=-1e4)
        scaler = StandardScaler().fit(X)
        pca = PCA(n_components=50, random_state=SEED).fit(scaler.transform(X))
        Xp = pca.transform(scaler.transform(X))
        ridge = RidgeCV(alphas=np.logspace(-3, 3, 20)).fit(Xp, log_model)
        probe_preds = ridge.predict(Xp)

        r_probe_model, _ = sp_stats.pearsonr(probe_preds, log_model)
        r_probe_real, _ = sp_stats.pearsonr(probe_preds, log_real)
        r_model_real, _ = sp_stats.pearsonr(log_model, log_real)

        print(f"  probe → model_estimate:  r={r_probe_model:.4f}")
        print(f"  probe → ground_truth:    r={r_probe_real:.4f}")
        print(f"  model_estimate → ground_truth: r={r_model_real:.4f}")
        print(f"")
        print(f"  Chain: probe --({r_probe_model:.2f})--> model_estimate --({r_model_real:.2f})--> reality")

        output = {
            "model": model_key,
            "n_valid": n_valid,
            "generation_accuracy": {"r": round(float(r_gen), 4), "p": float(p_gen)},
            "r2_residual_model_estimate": round(r2_residual, 4),
            "r2_model_error": round(r2_error, 4),
            "correlation_chain": {
                "probe_to_model": round(float(r_probe_model), 4),
                "probe_to_real": round(float(r_probe_real), 4),
                "model_to_real": round(float(r_model_real), 4),
            },
        }
        with open(model_dir / "self_estimate_probe_results.json", "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n✓ Saved self_estimate_probe_results.json → {model_dir}")


if __name__ == "__main__":
    main()
