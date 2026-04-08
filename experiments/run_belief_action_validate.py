"""
SPAR RQ1 — Belief-Action Consistency Validation
=================================================
Validates the goal-inconsistency finding from run_belief_action.py
by ruling out confounds:

1. POSITION SWAP: Run every scenario with A/B swapped.
   If the model just picks A regardless, the result is positional bias.
2. LOGPROB GAP ANALYSIS: How confident is the model in its choice?
   Tiny gaps = noise, large gaps = genuine preference.
3. FREE-FORM GENERATION: Let the model generate its own answer
   instead of forced-choice. Does it still pick short-term?
4. MULTI-LAYER PROBING: Check belief-action consistency at every
   layer, not just the final one. Does the inconsistency emerge
   at specific layers?

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_belief_action_validate.py --models qwen3.5-9b
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from config import MODELS, DEFAULT_MODELS, SEED

# Import scenarios from the main experiment
from run_belief_action import SCENARIOS, load_model, load_probe


# ═══════════════════════════════════════════════════════════════════════════
# 1. Position Swap Test
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def test_position_bias(model, tokenizer, scenarios):
    """
    Run each scenario twice: original order (A=short, B=long)
    and swapped order (A=long, B=short). If the model consistently
    picks the same position regardless of content, it's positional bias.
    """
    results = []

    for s in scenarios:
        # Original order
        orig = get_choice(model, tokenizer, s["stem"], s["option_a"], s["option_b"])
        # Swapped order
        swap = get_choice(model, tokenizer, s["stem"], s["option_b"], s["option_a"])

        # In original: A=short, B=long. model picks orig["choice"]
        # In swapped: A=long, B=short. model picks swap["choice"]
        # If model always picks "A" position: orig=A, swap=A → picks short both times
        # If model picks based on content: orig=B(long), swap=A(long) → picks long both times

        orig_chose_long = (orig["choice"] == "B")  # B is long in original
        swap_chose_long = (swap["choice"] == "A")  # A is long in swapped

        consistent_content = (orig_chose_long == swap_chose_long)
        consistent_position = (orig["choice"] == swap["choice"])

        results.append({
            "stem": s["stem"][:50],
            "category": s["category"],
            "orig_choice": orig["choice"],
            "swap_choice": swap["choice"],
            "orig_gap": orig["gap"],
            "swap_gap": swap["gap"],
            "consistent_content": consistent_content,
            "consistent_position": consistent_position,
            "orig_chose_long": orig_chose_long,
            "swap_chose_long": swap_chose_long,
        })

    n = len(results)
    content_consistent = sum(r["consistent_content"] for r in results)
    position_consistent = sum(r["consistent_position"] for r in results)
    always_a = sum(1 for r in results if r["orig_choice"] == "A" and r["swap_choice"] == "A")

    return {
        "results": results,
        "content_consistency": round(content_consistent / n, 4),
        "position_consistency": round(position_consistent / n, 4),
        "always_a_rate": round(always_a / n, 4),
        "n": n,
    }


@torch.no_grad()
def get_choice(model, tokenizer, stem, option_a_text, option_b_text):
    """Get model's logprob-based choice between two options."""
    lps = {}
    for label, option in [("A", option_a_text), ("B", option_b_text)]:
        full = stem + option
        inputs = tokenizer(full, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        stem_len = tokenizer(stem, return_tensors="pt")["input_ids"].shape[1]
        outputs = model(**inputs)
        logits = outputs.logits[0]
        input_ids = inputs["input_ids"][0]

        total_lp = 0.0
        n_tokens = 0
        for t in range(stem_len, len(input_ids)):
            log_probs = F.log_softmax(logits[t-1].float(), dim=-1)
            total_lp += log_probs[input_ids[t]].item()
            n_tokens += 1
        lps[label] = total_lp / max(n_tokens, 1)

    choice = "A" if lps["A"] > lps["B"] else "B"
    gap = abs(lps["A"] - lps["B"])
    return {"choice": choice, "gap": round(gap, 4), "lp_a": lps["A"], "lp_b": lps["B"]}


# ═══════════════════════════════════════════════════════════════════════════
# 2. Logprob Gap Analysis
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def analyze_logprob_gaps(model, tokenizer, scenarios):
    """Measure how confident the model is in each choice."""
    results = []
    for s in scenarios:
        r = get_choice(model, tokenizer, s["stem"], s["option_a"], s["option_b"])
        chose_long = (r["choice"] == s["gt_long"])
        results.append({
            "stem": s["stem"][:50],
            "choice": r["choice"],
            "gt_long": s["gt_long"],
            "chose_long": chose_long,
            "gap": r["gap"],
            "lp_a": round(r["lp_a"], 4),
            "lp_b": round(r["lp_b"], 4),
        })

    gaps_when_wrong = [r["gap"] for r in results if not r["chose_long"]]
    gaps_when_right = [r["gap"] for r in results if r["chose_long"]]

    return {
        "results": results,
        "mean_gap_overall": round(float(np.mean([r["gap"] for r in results])), 4),
        "mean_gap_when_wrong": round(float(np.mean(gaps_when_wrong)), 4) if gaps_when_wrong else None,
        "mean_gap_when_right": round(float(np.mean(gaps_when_right)), 4) if gaps_when_right else None,
        "n_confident_wrong": sum(1 for r in results if not r["chose_long"] and r["gap"] > 0.1),
        "n_marginal_wrong": sum(1 for r in results if not r["chose_long"] and r["gap"] <= 0.1),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. Free-Form Generation Test
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def test_freeform_generation(model, tokenizer, scenarios, max_tokens=150):
    """
    Instead of forced-choice logprobs, let the model freely generate
    a response. Manually assess whether it picks short or long term.
    """
    results = []
    for s in scenarios:
        prompt = s["stem"] + "\nChoose A or B and explain your reasoning.\n" + \
                 s["option_a"] + "\n" + s["option_b"] + "\nAnswer:"

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        output_ids = model.generate(input_ids, max_new_tokens=max_tokens, do_sample=False)
        text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

        # Simple heuristic: check if the model mentions A or B first
        text_lower = text.lower().strip()
        if text_lower.startswith("a") or " a)" in text_lower[:30] or "option a" in text_lower[:50]:
            gen_choice = "A"
        elif text_lower.startswith("b") or " b)" in text_lower[:30] or "option b" in text_lower[:50]:
            gen_choice = "B"
        else:
            gen_choice = "unclear"

        chose_long = (gen_choice == s["gt_long"])

        results.append({
            "stem": s["stem"][:50],
            "category": s["category"],
            "gt_long": s["gt_long"],
            "gen_choice": gen_choice,
            "chose_long": chose_long,
            "text": text[:300],
        })

        icon = "✓" if chose_long else ("✗" if gen_choice != "unclear" else "?")
        print(f"    {icon} [{gen_choice}] \"{s['stem'][:40]}...\" → {text[:60]}...")

    n_clear = sum(1 for r in results if r["gen_choice"] != "unclear")
    n_long = sum(1 for r in results if r["chose_long"])

    return {
        "results": results,
        "n_clear": n_clear,
        "n_long": n_long,
        "acc_gt": round(n_long / max(n_clear, 1), 4),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. Multi-Layer Belief-Action Consistency
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def test_multilayer_consistency(model, tokenizer, scenarios, probes_by_layer, n_layers):
    """Check belief-action consistency at every probed layer."""
    layer_metrics = {}

    for layer_idx, (probe, scaler, pca) in probes_by_layer.items():
        n_choice_matches_probe = 0
        n_probe_matches_gt = 0
        n_choice_matches_gt = 0
        n_wrong = 0
        n_wrong_consistent = 0

        for s in scenarios:
            # Get model choice (logprob)
            choice_result = get_choice(model, tokenizer, s["stem"], s["option_a"], s["option_b"])
            model_choice = choice_result["choice"]

            # Get probe belief at this layer
            probe_preds = {}
            for label, option in [("A", s["option_a"]), ("B", s["option_b"])]:
                full = s["stem"] + option
                inputs = tokenizer(full, return_tensors="pt", truncation=True, max_length=2048)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                outputs = model(**inputs, output_hidden_states=True)
                vec = outputs.hidden_states[layer_idx][0, -1, :].clamp(-1e4, 1e4).float().cpu().numpy()
                probe_preds[label] = float(probe.predict(
                    pca.transform(scaler.transform(vec.reshape(1, -1)))
                )[0])

            probe_belief = "A" if probe_preds["A"] > probe_preds["B"] else "B"
            gt_long = s["gt_long"]

            if model_choice == gt_long:
                n_choice_matches_gt += 1
            if model_choice == probe_belief:
                n_choice_matches_probe += 1
            if probe_belief == gt_long:
                n_probe_matches_gt += 1
            if model_choice != gt_long:
                n_wrong += 1
                if model_choice == probe_belief:
                    n_wrong_consistent += 1

        n = len(scenarios)
        layer_metrics[layer_idx] = {
            "acc_gt": round(n_choice_matches_gt / n, 4),
            "acc_dec": round(n_choice_matches_probe / n, 4),
            "agreement": round(n_probe_matches_gt / n, 4),
            "recovery": round(n_wrong_consistent / max(n_wrong, 1), 4),
            "n_wrong": n_wrong,
        }

    return layer_metrics


def load_probes_multilayer(model_key, layer_indices, results_dir="results"):
    """Load probes at multiple layers."""
    probe_path = Path(results_dir) / model_key / "probes.pkl"
    with open(probe_path, "rb") as f:
        probes = pickle.load(f)

    loaded = {}
    for setname in ["generated", "kitchen", "temporal"]:
        if setname in probes and "log_time_horizon" in probes[setname]:
            for li in layer_indices:
                if li in probes[setname]["log_time_horizon"]:
                    p = probes[setname]["log_time_horizon"][li]
                    loaded[li] = (p["probe"], p["scaler"], p["pca"])
    return loaded


# ═══════════════════════════════════════════════════════════════════════════
# 5. Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_validation(position_data, gap_data, layer_data, model_key, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Position bias
    ax = axes[0, 0]
    labels = ["Content\nconsistent", "Position\nconsistent", "Always\npicks A"]
    vals = [position_data["content_consistency"],
            position_data["position_consistency"],
            position_data["always_a_rate"]]
    colors = ["#2E75B6", "#C0392B", "#95A5A6"]
    ax.bar(labels, vals, color=colors, alpha=0.85)
    ax.set_ylabel("Fraction")
    ax.set_title("Position Bias Check")
    ax.axhline(0.5, color="gray", ls="--", lw=0.5)
    ax.set_ylim(0, 1.1)

    # Panel 2: Logprob gap distribution
    ax = axes[0, 1]
    gaps_wrong = [r["gap"] for r in gap_data["results"] if not r["chose_long"]]
    gaps_right = [r["gap"] for r in gap_data["results"] if r["chose_long"]]
    if gaps_wrong:
        ax.hist(gaps_wrong, bins=15, alpha=0.6, label=f"Wrong (n={len(gaps_wrong)})", color="#C0392B")
    if gaps_right:
        ax.hist(gaps_right, bins=15, alpha=0.6, label=f"Right (n={len(gaps_right)})", color="#2E75B6")
    ax.set_xlabel("Logprob gap |lp_A - lp_B|")
    ax.set_ylabel("Count")
    ax.set_title("Choice Confidence")
    ax.legend(fontsize=9)
    ax.axvline(0.1, color="gray", ls="--", lw=0.5, label="marginal threshold")

    # Panel 3: Multi-layer consistency
    ax = axes[1, 0]
    if layer_data:
        layers = sorted(layer_data.keys())
        ax.plot(layers, [layer_data[l]["acc_gt"] for l in layers], "o-", label="Acc. GT", markersize=4)
        ax.plot(layers, [layer_data[l]["acc_dec"] for l in layers], "s-", label="Acc. Dec.", markersize=4)
        ax.plot(layers, [layer_data[l]["agreement"] for l in layers], "^-", label="Agreement", markersize=4)
        ax.plot(layers, [layer_data[l]["recovery"] for l in layers], "D-", label="Recovery", markersize=4)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Fraction")
        ax.set_title("Belief-Action Metrics by Layer")
        ax.legend(fontsize=8)
        ax.axhline(0.5, color="gray", ls="--", lw=0.5)
    ax.grid(True, alpha=0.3)

    # Panel 4: Summary text
    ax = axes[1, 1]
    ax.axis("off")
    summary_text = f"""Validation Summary — {model_key}

Position Bias:
  Content consistent: {position_data['content_consistency']:.0%}
  Position consistent: {position_data['position_consistency']:.0%}
  Always picks A: {position_data['always_a_rate']:.0%}

Logprob Gaps:
  Mean gap (wrong): {gap_data['mean_gap_when_wrong'] or 'N/A'}
  Mean gap (right): {gap_data['mean_gap_when_right'] or 'N/A'}
  Confident wrong: {gap_data['n_confident_wrong']}
  Marginal wrong: {gap_data['n_marginal_wrong']}

Verdict:
  {"⚠️ POSITION BIAS" if position_data['always_a_rate'] > 0.7 else "✓ No strong position bias"}
  {"⚠️ MARGINAL CHOICES" if gap_data['n_marginal_wrong'] > gap_data['n_confident_wrong'] else "✓ Choices are confident"}
"""
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace")

    plt.suptitle(f"{model_key} — Belief-Action Validation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fname = outdir / f"belief_action_validate_{model_key}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SPAR RQ1 — Belief-Action Validation")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--skip_generation", action="store_true",
                        help="Skip free-form generation (slow)")
    parser.add_argument("--skip_multilayer", action="store_true",
                        help="Skip multi-layer analysis (slow)")
    args = parser.parse_args()

    outdir = Path(args.results_dir)

    for model_key in args.models:
        if model_key not in MODELS:
            print(f"⚠  Unknown model '{model_key}'. Skipping.")
            continue

        model, tokenizer, n_layers = load_model(model_key)
        model_dir = outdir / model_key
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Belief-Action Validation: {model_key}")
        print(f"{'='*60}")

        # Test 1: Position bias
        print(f"\n  ── Test 1: Position Swap ──")
        position_data = test_position_bias(model, tokenizer, SCENARIOS)
        print(f"    Content consistent: {position_data['content_consistency']:.0%}")
        print(f"    Position consistent: {position_data['position_consistency']:.0%}")
        print(f"    Always picks A: {position_data['always_a_rate']:.0%}")
        if position_data["always_a_rate"] > 0.7:
            print(f"    ⚠️  STRONG POSITION BIAS DETECTED — results may be confounded")
        else:
            print(f"    ✓ No strong position bias")

        # Test 2: Logprob gap analysis
        print(f"\n  ── Test 2: Logprob Gap Analysis ──")
        gap_data = analyze_logprob_gaps(model, tokenizer, SCENARIOS)
        print(f"    Mean gap overall: {gap_data['mean_gap_overall']}")
        print(f"    Mean gap when wrong: {gap_data['mean_gap_when_wrong']}")
        print(f"    Mean gap when right: {gap_data['mean_gap_when_right']}")
        print(f"    Confident wrong (gap > 0.1): {gap_data['n_confident_wrong']}")
        print(f"    Marginal wrong (gap ≤ 0.1): {gap_data['n_marginal_wrong']}")

        # Test 3: Free-form generation
        freeform_data = None
        if not args.skip_generation:
            print(f"\n  ── Test 3: Free-Form Generation ──")
            freeform_data = test_freeform_generation(model, tokenizer, SCENARIOS[:8])
            print(f"    Clear choices: {freeform_data['n_clear']}/8")
            print(f"    Chose long-term: {freeform_data['n_long']}/8")
            print(f"    Acc. GT: {freeform_data['acc_gt']:.0%}")

        # Test 4: Multi-layer consistency
        layer_data = {}
        if not args.skip_multilayer:
            print(f"\n  ── Test 4: Multi-Layer Consistency ──")
            # Load probes at all available layers
            probe_path = Path(args.results_dir) / model_key / "probes.pkl"
            with open(probe_path, "rb") as f:
                all_probes = pickle.load(f)
            available_layers = []
            for setname in ["generated", "kitchen", "temporal"]:
                if setname in all_probes and "log_time_horizon" in all_probes[setname]:
                    available_layers = sorted(all_probes[setname]["log_time_horizon"].keys())
                    break

            if available_layers:
                probes_by_layer = load_probes_multilayer(model_key, available_layers, args.results_dir)
                print(f"    Probes at layers: {sorted(probes_by_layer.keys())}")
                layer_data = test_multilayer_consistency(
                    model, tokenizer, SCENARIOS, probes_by_layer, n_layers
                )
                print(f"    {'Layer':>6} {'Acc.GT':>7} {'Acc.Dec':>8} {'Agree':>7} {'Recovery':>9}")
                for li in sorted(layer_data.keys()):
                    m = layer_data[li]
                    print(f"    L{li:>4} {m['acc_gt']:>7.3f} {m['acc_dec']:>8.3f} "
                          f"{m['agreement']:>7.3f} {m['recovery']:>9.3f}")

        plot_validation(position_data, gap_data, layer_data, model_key, model_dir)

        output = {
            "model": model_key,
            "position_bias": position_data,
            "logprob_gaps": gap_data,
            "freeform": freeform_data,
            "multilayer": {str(k): v for k, v in layer_data.items()} if layer_data else None,
        }
        with open(model_dir / "belief_action_validate_results.json", "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n✓ Saved belief_action_validate_results.json → {model_dir}")

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
