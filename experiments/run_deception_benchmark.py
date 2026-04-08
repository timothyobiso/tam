"""
SPAR RQ1 — Temporal Deception Detection (Benchmark Version)
=============================================================
Replaces the 24 hand-written deception prompts with real recipes
that have ground-truth cooking times.

Uses SDMN2001/recipe_difficulties (Food Network recipes with
total_minutes, prep_minutes, cook_minutes fields).

Three conditions per recipe:
  1. BARE: recipe name + ingredients only (no time mentioned)
     → Does the probe predict real cooking time from task semantics alone?
  2. HONEST: "This {real_time}-minute recipe: {name}..."
     → Baseline where stated time = real time
  3. MISLEADING: "This {false_time}-minute recipe: {name}..."
     → Does the probe read stated time or real time?

If the probe correlates with real time in the BARE and MISLEADING
conditions, it reads task semantics, not temporal words.

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_deception_benchmark.py --models qwen3.5-9b
"""

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from tqdm import tqdm

from config import MODELS, DEFAULT_MODELS, SEED

random.seed(SEED)


def load_model(model_key):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_id = MODELS[model_key]
    print(f"  Loading {hf_id} ...")
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


# ═══════════════════════════════════════════════════════════════════════════
# 1. Load recipes with real cooking times
# ═══════════════════════════════════════════════════════════════════════════

def load_recipes(n_recipes=200):
    """
    Load recipes from SDMN2001/recipe_difficulties.
    Filter to recipes with valid cooking times spanning a wide range.
    """
    from datasets import load_dataset
    print(f"  Loading recipe_difficulties dataset...")
    ds = load_dataset("SDMN2001/recipe_difficulties", split="train")

    valid = []
    for row in ds:
        total = row.get("total_minutes", 0)
        title = row.get("title", "")
        ingredients = row.get("clean_ingredients", "")
        if total and total > 0 and title and ingredients:
            valid.append({
                "title": title,
                "ingredients": ingredients if isinstance(ingredients, str) else ", ".join(ingredients) if isinstance(ingredients, list) else str(ingredients),
                "total_minutes": int(total),
                "prep_minutes": int(row.get("prep_minutes", 0) or 0),
                "cook_minutes": int(row.get("cook_minutes", 0) or 0),
                "level": row.get("level", ""),
            })

    print(f"  {len(valid)} recipes with valid times")

    # Stratified sample across time ranges
    bins = [
        ("quick", 1, 20),
        ("medium", 20, 60),
        ("long", 60, 180),
        ("very_long", 180, 10000),
    ]
    per_bin = n_recipes // len(bins)
    sampled = []
    for label, lo, hi in bins:
        bin_recipes = [r for r in valid if lo <= r["total_minutes"] < hi]
        chosen = random.sample(bin_recipes, min(per_bin, len(bin_recipes)))
        for r in chosen:
            r["time_bin"] = label
        sampled.extend(chosen)
        print(f"    {label} ({lo}-{hi} min): {len(chosen)} (from {len(bin_recipes)})")

    random.shuffle(sampled)
    return sampled


# ═══════════════════════════════════════════════════════════════════════════
# 2. Construct prompts in three conditions
# ═══════════════════════════════════════════════════════════════════════════

def make_prompts(recipe):
    """Create BARE, HONEST, and MISLEADING versions of a recipe."""
    title = recipe["title"]
    ingredients = recipe["ingredients"]
    real_mins = recipe["total_minutes"]

    # Truncate ingredients to keep prompts reasonable length
    if len(ingredients) > 300:
        ingredients = ingredients[:300] + "..."

    # BARE: no time mentioned at all
    bare = f"Recipe: {title}\nIngredients: {ingredients}\nThis recipe involves"

    # HONEST: stated time = real time
    honest = f"This {real_mins}-minute recipe for {title} with {ingredients} involves"

    # MISLEADING: false time (swap short↔long)
    if real_mins > 60:
        # Long recipe claimed to be quick
        false_mins = random.choice([5, 10, 15])
    elif real_mins > 20:
        # Medium recipe claimed to be either very quick or very long
        false_mins = random.choice([2, 5, 480, 720])
    else:
        # Quick recipe claimed to take forever
        false_mins = random.choice([240, 480, 720, 1440])
    misleading = f"This {false_mins}-minute recipe for {title} with {ingredients} involves"

    return {
        "bare": bare,
        "honest": honest,
        "misleading": misleading,
        "real_minutes": real_mins,
        "stated_honest": real_mins,
        "stated_misleading": false_mins,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. Probe evaluation
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def get_probe_pred(model, tokenizer, prompt, probe, scaler, pca, layer_idx):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model(**inputs, output_hidden_states=True)
    vec = outputs.hidden_states[layer_idx][0, -1, :].clamp(-1e4, 1e4).float().cpu().numpy()
    return float(probe.predict(pca.transform(scaler.transform(vec.reshape(1, -1))))[0])


def run_experiment(model, tokenizer, probe, scaler, pca, layer_idx, recipes):
    results = []

    for recipe in tqdm(recipes, desc="  evaluating"):
        prompts = make_prompts(recipe)

        pred_bare = get_probe_pred(
            model, tokenizer, prompts["bare"], probe, scaler, pca, layer_idx)
        pred_honest = get_probe_pred(
            model, tokenizer, prompts["honest"], probe, scaler, pca, layer_idx)
        pred_misleading = get_probe_pred(
            model, tokenizer, prompts["misleading"], probe, scaler, pca, layer_idx)

        results.append({
            "title": recipe["title"],
            "time_bin": recipe["time_bin"],
            "real_minutes": recipe["total_minutes"],
            "log_real": float(np.log1p(recipe["total_minutes"])),
            "stated_misleading": prompts["stated_misleading"],
            "log_stated_misleading": float(np.log1p(prompts["stated_misleading"])),
            "pred_bare": round(pred_bare, 4),
            "pred_honest": round(pred_honest, 4),
            "pred_misleading": round(pred_misleading, 4),
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 4. Analysis
# ═══════════════════════════════════════════════════════════════════════════

def analyze(results):
    log_real = np.array([r["log_real"] for r in results])
    log_stated_misleading = np.array([r["log_stated_misleading"] for r in results])
    pred_bare = np.array([r["pred_bare"] for r in results])
    pred_honest = np.array([r["pred_honest"] for r in results])
    pred_misleading = np.array([r["pred_misleading"] for r in results])

    analysis = {}

    # BARE: probe vs real time (no temporal words in prompt)
    r, p = sp_stats.pearsonr(pred_bare, log_real)
    analysis["bare_vs_real"] = {"r": round(float(r), 4), "p": round(float(p), 6)}

    # HONEST: probe vs real time (stated = real)
    r, p = sp_stats.pearsonr(pred_honest, log_real)
    analysis["honest_vs_real"] = {"r": round(float(r), 4), "p": round(float(p), 6)}

    # MISLEADING: probe vs real time
    r, p = sp_stats.pearsonr(pred_misleading, log_real)
    analysis["misleading_vs_real"] = {"r": round(float(r), 4), "p": round(float(p), 6)}

    # MISLEADING: probe vs stated (false) time
    r, p = sp_stats.pearsonr(pred_misleading, log_stated_misleading)
    analysis["misleading_vs_stated"] = {"r": round(float(r), 4), "p": round(float(p), 6)}

    return analysis


def plot_results(results, analysis, model_key, outdir):
    log_real = np.array([r["log_real"] for r in results])
    log_stated = np.array([r["log_stated_misleading"] for r in results])
    pred_bare = np.array([r["pred_bare"] for r in results])
    pred_honest = np.array([r["pred_honest"] for r in results])
    pred_misleading = np.array([r["pred_misleading"] for r in results])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: BARE — probe vs real time (no time words)
    ax = axes[0]
    ax.scatter(log_real, pred_bare, s=20, alpha=0.5, color="#2E75B6")
    r = analysis["bare_vs_real"]["r"]
    p = analysis["bare_vs_real"]["p"]
    ax.set_xlabel("log(real cooking time)")
    ax.set_ylabel("Probe prediction")
    ax.set_title(f"BARE (no time words)\nr={r:.3f}, p={p:.2e}")
    # Fit line
    m, b = np.polyfit(log_real, pred_bare, 1)
    ax.plot(sorted(log_real), [m*x+b for x in sorted(log_real)], "r--", lw=1.5)
    ax.grid(True, alpha=0.3)

    # Panel 2: MISLEADING — probe vs real time
    ax = axes[1]
    ax.scatter(log_real, pred_misleading, s=20, alpha=0.5, color="#27AE60", label="vs real")
    r_real = analysis["misleading_vs_real"]["r"]
    r_stated = analysis["misleading_vs_stated"]["r"]
    ax.set_xlabel("log(real cooking time)")
    ax.set_ylabel("Probe prediction")
    ax.set_title(f"MISLEADING\nvs real: r={r_real:.3f}, vs stated: r={r_stated:.3f}")
    m, b = np.polyfit(log_real, pred_misleading, 1)
    ax.plot(sorted(log_real), [m*x+b for x in sorted(log_real)], "r--", lw=1.5)
    ax.grid(True, alpha=0.3)

    # Panel 3: MISLEADING — probe vs stated (false) time
    ax = axes[2]
    ax.scatter(log_stated, pred_misleading, s=20, alpha=0.5, color="#C0392B")
    ax.set_xlabel("log(stated/false cooking time)")
    ax.set_ylabel("Probe prediction")
    ax.set_title(f"MISLEADING vs STATED\nr={r_stated:.3f}")
    m, b = np.polyfit(log_stated, pred_misleading, 1)
    ax.plot(sorted(log_stated), [m*x+b for x in sorted(log_stated)], "r--", lw=1.5)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"{model_key} — Temporal Deception Detection (Food Network recipes, n={len(results)})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fname = outdir / f"deception_benchmark_{model_key}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def main():
    parser = argparse.ArgumentParser(description="SPAR RQ1 — Deception Benchmark")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--n_recipes", type=int, default=200)
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    outdir = Path(args.results_dir)
    recipes = load_recipes(args.n_recipes)

    for model_key in args.models:
        if model_key not in MODELS:
            print(f"⚠  Unknown model '{model_key}'. Skipping.")
            continue

        model, tokenizer, n_layers = load_model(model_key)
        probe_layer = n_layers
        probe, scaler, pca = load_probe(model_key, probe_layer, args.results_dir)

        model_dir = outdir / model_key
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Deception Benchmark: {model_key}  ({len(recipes)} recipes × 3 conditions)")
        print(f"{'='*60}")

        results = run_experiment(model, tokenizer, probe, scaler, pca, probe_layer, recipes)
        analysis = analyze(results)

        print(f"\n  ── Correlations ──")
        print(f"  BARE (no time words):")
        print(f"    probe vs real time:    r={analysis['bare_vs_real']['r']:.4f}  "
              f"p={analysis['bare_vs_real']['p']:.2e}")
        print(f"  HONEST (stated = real):")
        print(f"    probe vs real time:    r={analysis['honest_vs_real']['r']:.4f}  "
              f"p={analysis['honest_vs_real']['p']:.2e}")
        print(f"  MISLEADING (stated ≠ real):")
        print(f"    probe vs REAL time:    r={analysis['misleading_vs_real']['r']:.4f}  "
              f"p={analysis['misleading_vs_real']['p']:.2e}")
        print(f"    probe vs STATED time:  r={analysis['misleading_vs_stated']['r']:.4f}  "
              f"p={analysis['misleading_vs_stated']['p']:.2e}")

        # Verdict
        r_real = abs(analysis["misleading_vs_real"]["r"])
        r_stated = abs(analysis["misleading_vs_stated"]["r"])
        if r_real > r_stated:
            print(f"\n  → Probe tracks REAL time (r={r_real:.3f} > {r_stated:.3f})")
        elif r_stated > r_real:
            print(f"\n  → Probe tracks STATED time (r={r_stated:.3f} > {r_real:.3f})")
        else:
            print(f"\n  → Probe tracks both equally")

        bare_r = abs(analysis["bare_vs_real"]["r"])
        if bare_r > 0.1:
            print(f"  → BARE condition: probe reads task semantics without any time words (r={bare_r:.3f})")
        else:
            print(f"  → BARE condition: weak/no signal without time words (r={bare_r:.3f})")

        plot_results(results, analysis, model_key, model_dir)

        output = {
            "model": model_key,
            "benchmark": "SDMN2001/recipe_difficulties",
            "n_recipes": len(recipes),
            "analysis": analysis,
            "results": results[:50],  # save subset for inspection
        }
        with open(model_dir / "deception_benchmark_results.json", "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n✓ Saved deception_benchmark_results.json → {model_dir}")

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
