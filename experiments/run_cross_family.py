"""
SPAR RQ1 — Cross-Family Generalization Test
=============================================
Loads saved activations from run_probes.py, splits by template family,
trains probe on 3 families, evaluates on the held-out 4th. Rotates
through all 4 folds.

If R² stays positive on held-out families, the probe is reading
temporal structure, not template artifacts.

Usage:
    python run_cross_family.py
    python run_cross_family.py --models qwen3.5-4b qwen3.5-9b
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from config import MODELS, DEFAULT_MODELS, SEED


def load_prompts_and_activations(model_key, results_dir="results"):
    """Load the generated prompts (with family labels) and saved activations."""
    prompts_path = Path("prompts_generated.json")
    act_path = Path(results_dir) / model_key / "activations.pkl"

    if not prompts_path.exists():
        raise FileNotFoundError(f"No prompts at {prompts_path}. Run generate_prompts.py first.")
    if not act_path.exists():
        raise FileNotFoundError(f"No activations at {act_path}. Run run_probes.py --prompts prompts_generated.json first.")

    with open(prompts_path) as f:
        prompts = json.load(f)
    with open(act_path, "rb") as f:
        data = pickle.load(f)

    # Find the activations — check for "generated_states" key
    states = data.get("generated_states", data.get("kitchen_states"))
    if states is None:
        raise KeyError("No activations found. Re-run run_probes.py --prompts prompts_generated.json")

    return prompts, states


def run_cross_family(model_key, results_dir="results"):
    prompts, states = load_prompts_and_activations(model_key, results_dir)

    families = sorted(set(p["family"] for p in prompts))
    family_arr = np.array([p["family"] for p in prompts])
    y_time = np.log1p(np.array([p["time_horizon"] for p in prompts]))
    y_depth = np.array([float(p["planning_depth"]) for p in prompts])

    layers = sorted(states.keys())

    print(f"\n{'='*60}")
    print(f"Cross-family generalization: {model_key}")
    print(f"Families: {families}")
    print(f"Counts: {', '.join(f'{fam}={np.sum(family_arr==fam)}' for fam in families)}")
    print(f"{'='*60}")

    results = {}

    for target_name, y in [("log_time_horizon", y_time), ("planning_depth", y_depth)]:
        print(f"\n  Target: {target_name}")
        fold_results = []

        for held_out in families:
            test_mask = family_arr == held_out
            train_mask = ~test_mask
            n_train, n_test = train_mask.sum(), test_mask.sum()

            # Find best layer by training R² (no peeking at test)
            best_layer = None
            best_train_r2 = -np.inf

            for li in layers:
                X = states[li]
                X_train, X_test = X[train_mask], X[test_mask]
                y_train, y_test = y[train_mask], y[test_mask]

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    scaler = StandardScaler().fit(X_train)
                    X_tr_s = scaler.transform(X_train)
                    n_comp = min(20, X_tr_s.shape[0] - 2)
                    pca = PCA(n_components=n_comp, random_state=SEED).fit(X_tr_s)
                    X_tr_p = pca.transform(X_tr_s)

                    probe = RidgeCV(alphas=np.logspace(-3, 3, 20)).fit(X_tr_p, y_train)
                    tr_r2 = probe.score(X_tr_p, y_train)

                    if tr_r2 > best_train_r2:
                        best_train_r2 = tr_r2
                        best_layer = li

            # Evaluate best layer on held-out family
            X = states[best_layer]
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scaler = StandardScaler().fit(X_train)
                n_comp = min(20, scaler.transform(X_train).shape[0] - 2)
                pca = PCA(n_components=n_comp, random_state=SEED).fit(scaler.transform(X_train))
                probe = RidgeCV(alphas=np.logspace(-3, 3, 20)).fit(
                    pca.transform(scaler.transform(X_train)), y_train)

                X_te_p = pca.transform(scaler.transform(X_test))
                test_r2 = probe.score(X_te_p, y_test)
                train_r2 = probe.score(pca.transform(scaler.transform(X_train)), y_train)

            fold_results.append({
                "held_out": held_out,
                "n_train": int(n_train),
                "n_test": int(n_test),
                "best_layer": int(best_layer),
                "train_r2": round(float(train_r2), 4),
                "test_r2": round(float(test_r2), 4),
            })

            status = "✓" if test_r2 > 0 else "✗"
            print(f"    {status} held_out={held_out:15s}  "
                  f"train={n_train} test={n_test}  "
                  f"L{best_layer}  R²_train={train_r2:.4f}  R²_test={test_r2:.4f}")

        mean_test = np.mean([f["test_r2"] for f in fold_results])
        positive = sum(1 for f in fold_results if f["test_r2"] > 0)
        print(f"    ──────────────────────────────────────────────")
        print(f"    Mean held-out R²: {mean_test:.4f}   ({positive}/{len(families)} folds positive)")

        results[target_name] = {
            "folds": fold_results,
            "mean_test_r2": round(float(mean_test), 4),
            "positive_folds": positive,
            "total_folds": len(families),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="SPAR RQ1 — Cross-Family Generalization")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    outdir = Path(args.results_dir)
    all_results = {}

    for model_key in args.models:
        if model_key not in MODELS:
            print(f"⚠  Unknown model '{model_key}'. Skipping.")
            continue
        results = run_cross_family(model_key, args.results_dir)
        all_results[model_key] = results

        with open(outdir / model_key / "cross_family_results.json", "w") as f:
            json.dump(results, f, indent=2)

    with open(outdir / "cross_family_all.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nDone. Results saved.")


if __name__ == "__main__":
    main()
