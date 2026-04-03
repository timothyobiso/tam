"""
SPAR RQ1 — Visualization
=========================
Reads results from run_probes.py and run_curvature.py,
produces publication-ready figures.

Usage:
    python plot_results.py                     # reads from results/
    python plot_results.py --outdir results/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def load_probe_results(outdir: Path) -> dict:
    """Load all probe_results.json files from model subdirectories."""
    results = {}
    for p in sorted(outdir.iterdir()):
        f = p / "probe_results.json"
        if f.exists():
            with open(f) as fh:
                results[p.name] = json.load(fh)
    return results


def load_curvature_results(outdir: Path) -> dict:
    results = {}
    for p in sorted(outdir.iterdir()):
        f = p / "curvature_results.json"
        if f.exists():
            with open(f) as fh:
                results[p.name] = json.load(fh)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# 1. R² by Layer — one plot per target variable, all models overlaid
# ═══════════════════════════════════════════════════════════════════════════

def plot_r2_by_layer(probe_results: dict, prompt_set: str, outdir: Path):
    """
    prompt_set: "kitchen" or "temporal"
    """
    if not probe_results:
        print("  No probe results to plot.")
        return

    # Collect target names from first model
    first = next(iter(probe_results.values()))
    target_names = list(first[prompt_set].keys())

    for tname in target_names:
        fig, ax = plt.subplots(figsize=(10, 4))

        for model_key, data in probe_results.items():
            layer_data = data[prompt_set][tname]
            layers = sorted(layer_data.keys(), key=lambda x: int(x))
            xs = [int(l) for l in layers]
            r2_cv = [layer_data[l]["r2_cv"] for l in layers]

            ax.plot(xs, r2_cv, "o-", label=model_key, markersize=4, linewidth=1.5)

        ax.set_xlabel("Layer")
        ax.set_ylabel("R² (cross-validated)")
        ax.set_title(f"Slider Probe: {tname}  [{prompt_set} prompts]")
        ax.legend(fontsize=9)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax.grid(True, alpha=0.3)

        fname = outdir / f"r2_{prompt_set}_{tname}.png"
        fig.savefig(fname)
        plt.close(fig)
        print(f"  Saved {fname}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Curvature by Layer — bar chart, all models side by side
# ═══════════════════════════════════════════════════════════════════════════

def plot_curvature_by_layer(curv_results: dict, outdir: Path):
    if not curv_results:
        print("  No curvature results to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    n_models = len(curv_results)
    width = 0.8 / n_models

    for i, (model_key, data) in enumerate(curv_results.items()):
        summary = data["summary_by_layer"]
        layers = sorted(summary.keys(), key=lambda x: int(x))
        xs = np.arange(len(layers))
        means = [summary[l]["mean_cosine_across_seeds"] for l in layers]
        stds = [summary[l]["std_across_seeds"] for l in layers]

        ax.bar(xs + i * width, means, width, yerr=stds,
               label=model_key, alpha=0.85, capsize=2)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Cosine Similarity\n(higher = straighter trajectory)")
    ax.set_title("Activation Trajectory Curvature During Generation\n(Wang et al. 2026 metric)")
    layer_labels = sorted(
        next(iter(curv_results.values()))["summary_by_layer"].keys(),
        key=lambda x: int(x),
    )
    ax.set_xticks(np.arange(len(layer_labels)) + width * (n_models - 1) / 2)
    ax.set_xticklabels([f"L{l}" for l in layer_labels], rotation=45, fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    fname = outdir / "curvature_by_layer.png"
    fig.savefig(fname)
    plt.close(fig)
    print(f"  Saved {fname}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Combined: R² vs Curvature scatter (per layer)
# ═══════════════════════════════════════════════════════════════════════════

def plot_r2_vs_curvature(probe_results: dict, curv_results: dict, outdir: Path):
    """
    For each model, scatter plot: x = layer curvature, y = probe R².
    Tests the hypothesis that straighter layers are more linearly decodable.
    """
    common_models = set(probe_results.keys()) & set(curv_results.keys())
    if not common_models:
        print("  No overlapping models for R² vs curvature plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    for model_key in sorted(common_models):
        pd = probe_results[model_key]
        cd = curv_results[model_key]

        # Use log_time_horizon from whichever prompt set is available
        r2_data = None
        for key in ["generated", "kitchen", "temporal"]:
            if key in pd and "log_time_horizon" in pd[key]:
                r2_data = pd[key]["log_time_horizon"]
                break
        if r2_data is None:
            continue
        curv_data = cd["summary_by_layer"]

        # Find common layers
        common_layers = sorted(
            set(r2_data.keys()) & set(curv_data.keys()), key=lambda x: int(x)
        )
        if not common_layers:
            continue

        curvatures = [curv_data[l]["mean_cosine_across_seeds"] for l in common_layers]
        r2s = [r2_data[l]["r2_cv"] for l in common_layers]

        ax.scatter(curvatures, r2s, label=model_key, s=30, alpha=0.7)
        # Annotate a few points
        for l, cx, ry in zip(common_layers, curvatures, r2s):
            if int(l) % 8 == 0:
                ax.annotate(f"L{l}", (cx, ry), fontsize=7, alpha=0.6)

    ax.set_xlabel("Layer Curvature (mean cosine, higher = straighter)")
    ax.set_ylabel("R² (log time horizon probe, CV)")
    ax.set_title("Trajectory Straightness vs Probe Decodability")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fname = outdir / "r2_vs_curvature.png"
    fig.savefig(fname)
    plt.close(fig)
    print(f"  Saved {fname}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SPAR RQ1 — Plot Results")
    parser.add_argument("--outdir", type=str, default="results")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    if not outdir.exists():
        print(f"No results directory at {outdir}. Run run_probes.py first.")
        return

    print("Loading results...")
    probe_results = load_probe_results(outdir)
    curv_results = load_curvature_results(outdir)

    print(f"  Found probe results for: {list(probe_results.keys())}")
    print(f"  Found curvature results for: {list(curv_results.keys())}")

    # Auto-detect prompt sets from the first model's results
    if probe_results:
        first = next(iter(probe_results.values()))
        prompt_sets = [k for k in first.keys()
                       if k not in ("model", "hf_id", "n_layers", "d_model",
                                    "n_kitchen_prompts", "n_temporal_prompts")]
        for ps in prompt_sets:
            print(f"\nPlotting R² by layer ({ps})...")
            plot_r2_by_layer(probe_results, ps, outdir)

    print("\nPlotting curvature by layer...")
    plot_curvature_by_layer(curv_results, outdir)

    print("\nPlotting R² vs curvature...")
    plot_r2_vs_curvature(probe_results, curv_results, outdir)

    print(f"\nAll plots saved to {outdir}/")


if __name__ == "__main__":
    main()
