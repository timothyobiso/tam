"""
SPAR RQ1 — Nonlinear Probe Comparison
=======================================
Trains MLP probes alongside the existing ridge probes to quantify
how much temporal signal the linear probe misses.

If MLP R² ≈ Ridge R²  → representation is genuinely linear
If MLP R² >> Ridge R²  → significant nonlinear structure exists

Uses the same PCA-reduced activations from run_probes.py.

Usage:
    python run_nonlinear_probe.py
    python run_nonlinear_probe.py --models qwen3.5-4b qwen3.5-9b
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

from config import MODELS, DEFAULT_MODELS, SEED, CV_FOLDS


def load_activations_and_prompts(model_key, results_dir="results"):
    act_path = Path(results_dir) / model_key / "activations.pkl"
    with open(act_path, "rb") as f:
        data = pickle.load(f)
    states = data.get("generated_states", data.get("kitchen_states"))

    prompts_path = Path("prompts_generated.json")
    with open(prompts_path) as f:
        prompts = json.load(f)

    y_time = np.log1p(np.array([p["time_horizon"] for p in prompts]))
    y_depth = np.array([float(p["planning_depth"]) for p in prompts])
    return states, {"log_time_horizon": y_time, "planning_depth": y_depth}


def run_comparison(model_key, results_dir="results"):
    states, targets = load_activations_and_prompts(model_key, results_dir)
    layers = sorted(states.keys())

    print(f"\n{'='*60}")
    print(f"Nonlinear probe comparison: {model_key}")
    print(f"{'='*60}")

    results = {}

    for tname, y in targets.items():
        print(f"\n  Target: {tname}  (n={len(y)})")
        layer_results = {}

        for li in layers:
            X = states[li].copy()
            X = np.nan_to_num(X, nan=0.0, posinf=1e4, neginf=-1e4)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            n_comp = min(20, X_scaled.shape[0] - 2)
            pca = PCA(n_components=n_comp, random_state=SEED)
            X_pca = pca.fit_transform(X_scaled)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Ridge (linear)
                ridge_scores = cross_val_score(
                    RidgeCV(alphas=np.logspace(-3, 3, 20)),
                    X_pca, y, cv=CV_FOLDS, scoring="r2",
                )
                ridge_r2 = float(ridge_scores.mean())

                # MLP (64 hidden units)
                mlp_64_scores = cross_val_score(
                    MLPRegressor(hidden_layer_sizes=(64,), max_iter=2000,
                                random_state=SEED, early_stopping=True,
                                validation_fraction=0.15, alpha=0.01),
                    X_pca, y, cv=CV_FOLDS, scoring="r2",
                )
                mlp_64_r2 = float(mlp_64_scores.mean())

                # MLP (128, 64) — deeper
                mlp_deep_scores = cross_val_score(
                    MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=2000,
                                random_state=SEED, early_stopping=True,
                                validation_fraction=0.15, alpha=0.01),
                    X_pca, y, cv=CV_FOLDS, scoring="r2",
                )
                mlp_deep_r2 = float(mlp_deep_scores.mean())

            delta_64 = mlp_64_r2 - ridge_r2
            delta_deep = mlp_deep_r2 - ridge_r2

            layer_results[li] = {
                "ridge_r2": round(ridge_r2, 4),
                "mlp_64_r2": round(mlp_64_r2, 4),
                "mlp_128_64_r2": round(mlp_deep_r2, 4),
                "delta_64": round(delta_64, 4),
                "delta_deep": round(delta_deep, 4),
            }

            marker = ""
            if delta_64 > 0.05:
                marker = " ← nonlinear signal"
            elif delta_64 < -0.05:
                marker = " ← MLP underfits"

            print(f"    L{li:3d}  ridge={ridge_r2:.4f}  "
                  f"mlp64={mlp_64_r2:.4f} ({delta_64:+.4f})  "
                  f"mlp128_64={mlp_deep_r2:.4f} ({delta_deep:+.4f}){marker}")

        results[tname] = layer_results

    return results


def plot_comparison(results, model_key, outdir):
    for tname, layer_data in results.items():
        layers = sorted(layer_data.keys(), key=int)
        xs = [int(l) for l in layers]
        ridge = [layer_data[l]["ridge_r2"] for l in layers]
        mlp64 = [layer_data[l]["mlp_64_r2"] for l in layers]
        mlp_deep = [layer_data[l]["mlp_128_64_r2"] for l in layers]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(xs, ridge, "o-", label="Ridge (linear)", markersize=4, linewidth=1.5)
        ax.plot(xs, mlp64, "s-", label="MLP (64)", markersize=4, linewidth=1.5)
        ax.plot(xs, mlp_deep, "^-", label="MLP (128,64)", markersize=4, linewidth=1.5)

        ax.set_xlabel("Layer")
        ax.set_ylabel("R² (cross-validated)")
        ax.set_title(f"{model_key} — Linear vs Nonlinear Probes: {tname}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", ls="--", lw=0.5)

        fname = outdir / f"nonlinear_probe_{model_key}_{tname}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


def main():
    parser = argparse.ArgumentParser(description="SPAR RQ1 — Nonlinear Probe")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    outdir = Path(args.results_dir)
    all_results = {}

    for model_key in args.models:
        if model_key not in MODELS:
            print(f"⚠  Unknown model '{model_key}'. Skipping.")
            continue

        results = run_comparison(model_key, args.results_dir)
        all_results[model_key] = results

        model_dir = outdir / model_key
        model_dir.mkdir(parents=True, exist_ok=True)
        plot_comparison(results, model_key, model_dir)

        with open(model_dir / "nonlinear_probe_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Saved nonlinear_probe_results.json → {model_dir}")

    # Summary
    print(f"\n{'='*60}")
    print("Summary: is the representation linear?")
    print(f"{'='*60}")
    for model_key, results in all_results.items():
        for tname, layer_data in results.items():
            final = max(layer_data.keys(), key=int)
            d = layer_data[final]
            verdict = "LINEAR" if d["delta_64"] < 0.05 else "NONLINEAR"
            print(f"  {model_key} / {tname} @ L{final}: "
                  f"ridge={d['ridge_r2']:.3f}  mlp={d['mlp_64_r2']:.3f}  "
                  f"Δ={d['delta_64']:+.3f}  → {verdict}")


if __name__ == "__main__":
    main()
