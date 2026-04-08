"""
SPAR RQ1 — Temporal Reasoning Probe
=====================================
The bare-recipe probe failed because no temporal reasoning was
triggered. This experiment forces the model to predict a duration,
then probes the hidden state RIGHT BEFORE the number token.

Three prompt conditions:
  BARE:    "Recipe: X. Ingredients: Y. This recipe involves"
  PREDICT: "Recipe: X. Ingredients: Y. Estimated cooking time in minutes:"
  CONTEXT: "Recipe: X. Ingredients: Y. A beginner cook should expect this to take"

The PREDICT state should encode the model's duration estimate.
If the probe works on PREDICT but not BARE, temporal reasoning
is activated by the prediction context, not passively present.

Additionally tests whether the model's GENERATED time estimate
correlates with ground truth — establishing that the model actually
knows cooking times before checking if the probe can read them.

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_reasoning_probe.py --models qwen3.5-9b
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


PROMPT_TEMPLATES = {
    "bare": "Recipe: {title}\nIngredients: {ingredients}\nThis recipe involves",
    "predict": "Recipe: {title}\nIngredients: {ingredients}\nEstimated cooking time in minutes:",
    "context": "Recipe: {title}\nIngredients: {ingredients}\nA beginner cook should expect this to take",
}


def load_recipes(n_recipes=500):
    from datasets import load_dataset
    print(f"  Loading recipe dataset...")
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
            valid.append({
                "title": title,
                "ingredients": ing_str,
                "total_minutes": int(total),
            })

    # Stratified sample
    bins = [(1, 15), (15, 45), (45, 120), (120, 10000)]
    per_bin = n_recipes // len(bins)
    sampled = []
    for lo, hi in bins:
        pool = [r for r in valid if lo <= r["total_minutes"] < hi]
        chosen = random.sample(pool, min(per_bin, len(pool)))
        sampled.extend(chosen)
        print(f"    {lo}-{hi} min: {len(chosen)}")

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


@torch.no_grad()
def extract_and_generate(model, tokenizer, prompt, n_layers, max_new=20):
    """Extract hidden state at last token AND generate continuation."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Extract hidden states
    outputs = model(**inputs, output_hidden_states=True)
    states = {}
    for li in [n_layers - 4, n_layers - 2, n_layers]:
        states[li] = outputs.hidden_states[li][0, -1, :].clamp(-1e4, 1e4).float().cpu().numpy()

    # Generate continuation
    input_ids = inputs["input_ids"]
    gen_ids = model.generate(
        input_ids, max_new_tokens=max_new, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_text = tokenizer.decode(gen_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

    return states, gen_text


def extract_number(text):
    """Try to extract a number from generated text."""
    # Match patterns like "30", "30 minutes", "about 30", "approximately 45"
    matches = re.findall(r'(\d+(?:\.\d+)?)', text[:100])
    if matches:
        return float(matches[0])
    return None


def train_and_eval_probe(X, y, n_comp=50):
    """Train ridge probe with cross-validation."""
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
    return float(scores.mean()), float(scores.std())


def main():
    parser = argparse.ArgumentParser(description="SPAR RQ1 — Reasoning Probe")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--n_recipes", type=int, default=500)
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    outdir = Path(args.results_dir)
    recipes = load_recipes(args.n_recipes)
    y = np.log1p(np.array([r["total_minutes"] for r in recipes]))

    for model_key in args.models:
        if model_key not in MODELS:
            continue

        model, tokenizer, n_layers = load_model(model_key)
        model_dir = outdir / model_key
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Reasoning Probe: {model_key}  ({len(recipes)} recipes)")
        print(f"{'='*60}")

        # Extract states and generations for all conditions
        all_states = {cond: defaultdict(list) for cond in PROMPT_TEMPLATES}
        all_gens = {cond: [] for cond in PROMPT_TEMPLATES}

        for recipe in tqdm(recipes, desc="  extracting"):
            for cond, template in PROMPT_TEMPLATES.items():
                prompt = template.format(
                    title=recipe["title"],
                    ingredients=recipe["ingredients"],
                )
                states, gen_text = extract_and_generate(
                    model, tokenizer, prompt, n_layers
                )
                for li, vec in states.items():
                    all_states[cond][li].append(vec)
                all_gens[cond].append(gen_text)

        # Stack arrays
        for cond in PROMPT_TEMPLATES:
            for li in all_states[cond]:
                all_states[cond][li] = np.stack(all_states[cond][li])

        del model
        torch.cuda.empty_cache()

        # ── Analysis 1: Does the model actually know cooking times? ──
        print(f"\n  ── Model's Generated Time Estimates ──")
        gen_results = {}
        for cond in ["predict", "context"]:
            extracted = []
            for i, gen in enumerate(all_gens[cond]):
                num = extract_number(gen)
                if num is not None and 0 < num < 100000:
                    extracted.append((i, num))

            if len(extracted) > 10:
                idxs, preds = zip(*extracted)
                log_preds = np.log1p(np.array(preds))
                log_real = y[list(idxs)]
                r, p = sp_stats.pearsonr(log_preds, log_real)
                gen_results[cond] = {
                    "r": round(float(r), 4),
                    "p": round(float(p), 6),
                    "n_extracted": len(extracted),
                }
                print(f"    {cond}: r={r:.4f} (p={p:.2e}), {len(extracted)}/{len(recipes)} extracted")
                # Show examples
                for idx, pred in extracted[:5]:
                    real = recipes[idx]["total_minutes"]
                    print(f"      {recipes[idx]['title'][:40]:40s}  real={real:5d}  pred={pred:7.0f}  "
                          f"gen=\"{all_gens[cond][idx][:40]}\"")
            else:
                print(f"    {cond}: only {len(extracted)} numbers extracted, skipping")
                gen_results[cond] = {"r": None, "n_extracted": len(extracted)}

        # ── Analysis 2: Probe R² by condition and layer ──
        print(f"\n  ── Probe R² by Condition ──")
        probe_results = {}
        for cond in PROMPT_TEMPLATES:
            probe_results[cond] = {}
            for li in sorted(all_states[cond].keys()):
                X = all_states[cond][li]
                for nc in [20, 50, 100]:
                    r2_mean, r2_std = train_and_eval_probe(X, y, n_comp=nc)
                    key = f"L{li}_PCA{nc}"
                    probe_results[cond][key] = round(r2_mean, 4)

            # Print best result per condition
            best_key = max(probe_results[cond], key=probe_results[cond].get)
            best_r2 = probe_results[cond][best_key]
            print(f"    {cond:>10s}: best R²={best_r2:.4f} ({best_key})")
            for key, r2 in sorted(probe_results[cond].items()):
                print(f"      {key:>15s}: {r2:.4f}")

        # ── Analysis 3: Within-category check for predict condition ──
        print(f"\n  ── Within-Category Check (predict condition) ──")
        li_best = n_layers
        X_pred = all_states["predict"][li_best]
        X_pred = np.nan_to_num(X_pred, nan=0.0, posinf=1e4, neginf=-1e4)

        bins = defaultdict(list)
        for i, r in enumerate(recipes):
            t = r["total_minutes"]
            if t < 15: b = "quick"
            elif t < 45: b = "medium"
            elif t < 120: b = "long"
            else: b = "very_long"
            bins[b].append(i)

        pred_y = np.zeros(len(y))
        for bname, idxs in bins.items():
            pred_y[idxs] = np.mean(y[idxs])
        residual = y - pred_y

        r2_residual, _ = train_and_eval_probe(X_pred, residual, n_comp=50)
        print(f"    Residual R² (predict, after bin means): {r2_residual:.4f}")

        for bname, idxs in sorted(bins.items()):
            if len(idxs) < 20:
                continue
            Xi = X_pred[idxs]
            yi = y[idxs]
            r2, _ = train_and_eval_probe(Xi, yi, n_comp=min(20, len(idxs) - 5))
            print(f"    {bname:>10s} (n={len(idxs)}): within-bin R²={r2:.4f}")

        # Save
        output = {
            "model": model_key,
            "n_recipes": len(recipes),
            "generation_accuracy": gen_results,
            "probe_results": probe_results,
            "residual_r2_predict": round(r2_residual, 4),
        }
        with open(model_dir / "reasoning_probe_results.json", "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n✓ Saved reasoning_probe_results.json → {model_dir}")


if __name__ == "__main__":
    main()
