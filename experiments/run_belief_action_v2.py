"""
SPAR RQ1 — Belief-Action Consistency (Team Dataset)
=====================================================
Uses the team's balanced implicit temporal scope dataset (100 pairs)
with proper forced-choice methodology.

Instead of comparing fluency of full option texts (which confounds
with text length/naturalness), we:
  1. Present the full prompt with both options
  2. Compare logprob of token "A" vs token "B" → model's choice
  3. Apply temporal probe on stem + each option → decoded belief
  4. Compute Arghal et al. metrics

Dataset: temporal_scope_implicit_expanded.json
  - 100 pairs, no temporal words, semantic cues only
  - Options are length-matched and position-randomized

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_belief_action_v2.py --models qwen3.5-9b
"""

import argparse
import json
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from config import MODELS, DEFAULT_MODELS, SEED


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


def load_dataset(path="temporal_scope_implicit_expanded.json"):
    with open(path) as f:
        data = json.load(f)
    print(f"  Loaded {len(data['pairs'])} pairs")
    print(f"  Style: {data['metadata']['style']}")
    print(f"  Position randomized: {data['metadata']['position_randomized']}")
    return data["pairs"]


@torch.no_grad()
def evaluate_pair(model, tokenizer, pair, probe, scaler, pca, probe_layer):
    """
    Evaluate one pair using two methods:

    Method 1 (Token choice): Present both options, compare P("A") vs P("B")
    Method 2 (Continuation logprob): Compare avg logprob of each option text

    Also decode the probe's temporal belief for each option.
    """
    stem = pair["question"]
    opt_a = pair["immediate"]  # A is always immediate in dataset
    opt_b = pair["long_term"]  # B is always long-term

    # ── Method 1: A/B token logprob ─────────────────────────────────────
    choice_prompt = f"{stem}\n{opt_a}\n{opt_b}\nAnswer:"
    inputs = tokenizer(choice_prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model(**inputs, output_hidden_states=True)
    logits = outputs.logits[0, -1, :]  # last token logits
    log_probs = F.log_softmax(logits.float(), dim=-1)

    # Find token IDs for "A" and "B" (try with and without space prefix)
    token_a_ids = []
    token_b_ids = []
    for prefix in ["A", " A", "(A", " (A"]:
        ids = tokenizer.encode(prefix, add_special_tokens=False)
        if ids:
            token_a_ids.append(ids[0])
    for prefix in ["B", " B", "(B", " (B"]:
        ids = tokenizer.encode(prefix, add_special_tokens=False)
        if ids:
            token_b_ids.append(ids[0])

    # Take max logprob across A variants and B variants
    lp_a = max(log_probs[tid].item() for tid in token_a_ids) if token_a_ids else -100
    lp_b = max(log_probs[tid].item() for tid in token_b_ids) if token_b_ids else -100
    token_choice = "A" if lp_a > lp_b else "B"
    token_gap = abs(lp_a - lp_b)

    # ── Method 2: Continuation logprob ──────────────────────────────────
    cont_lps = {}
    for label, option in [("A", opt_a), ("B", opt_b)]:
        full = stem + " " + option
        inp = tokenizer(full, return_tensors="pt", truncation=True, max_length=2048)
        inp = {k: v.to(model.device) for k, v in inp.items()}
        stem_len = tokenizer(stem + " ", return_tensors="pt")["input_ids"].shape[1]
        out = model(**inp)
        ids = inp["input_ids"][0]
        total = 0.0
        n = 0
        for t in range(stem_len, len(ids)):
            lps = F.log_softmax(out.logits[0, t-1].float(), dim=-1)
            total += lps[ids[t]].item()
            n += 1
        cont_lps[label] = total / max(n, 1)
    cont_choice = "A" if cont_lps["A"] > cont_lps["B"] else "B"
    cont_gap = abs(cont_lps["A"] - cont_lps["B"])

    # ── Probe belief ────────────────────────────────────────────────────
    probe_preds = {}
    for label, option in [("A", opt_a), ("B", opt_b)]:
        full = stem + " " + option
        inp = tokenizer(full, return_tensors="pt", truncation=True, max_length=2048)
        inp = {k: v.to(model.device) for k, v in inp.items()}
        out = model(**inp, output_hidden_states=True)
        vec = out.hidden_states[probe_layer][0, -1, :].clamp(-1e4, 1e4).float().cpu().numpy()
        probe_preds[label] = float(probe.predict(
            pca.transform(scaler.transform(vec.reshape(1, -1)))
        )[0])

    # B should have higher probe pred (longer horizon)
    probe_belief = "A" if probe_preds["A"] > probe_preds["B"] else "B"
    probe_gap = abs(probe_preds["A"] - probe_preds["B"])

    gt_long = "B"  # by dataset construction

    return {
        "question": stem,
        "category": pair["category"],
        "gt_long": gt_long,
        # Token choice method
        "token_choice": token_choice,
        "token_gap": round(token_gap, 4),
        "lp_a": round(lp_a, 4),
        "lp_b": round(lp_b, 4),
        # Continuation method
        "cont_choice": cont_choice,
        "cont_gap": round(cont_gap, 4),
        # Probe
        "probe_belief": probe_belief,
        "probe_gap": round(probe_gap, 4),
        "probe_a": round(probe_preds["A"], 4),
        "probe_b": round(probe_preds["B"], 4),
        # Derived (token method)
        "token_matches_gt": token_choice == gt_long,
        "token_matches_probe": token_choice == probe_belief,
        "probe_matches_gt": probe_belief == gt_long,
        # Derived (continuation method)
        "cont_matches_gt": cont_choice == gt_long,
        "cont_matches_probe": cont_choice == probe_belief,
    }


def compute_metrics(results, choice_method="token"):
    """Compute Arghal et al. metrics for a given choice method."""
    n = len(results)
    prefix = choice_method  # "token" or "cont"

    acc_gt = sum(r[f"{prefix}_matches_gt"] for r in results) / n
    acc_dec = sum(r[f"{prefix}_matches_probe"] for r in results) / n
    agreement = sum(r["probe_matches_gt"] for r in results) / n

    wrong = [r for r in results if not r[f"{prefix}_matches_gt"]]
    if wrong:
        recovery = sum(r[f"{prefix}_matches_probe"] for r in wrong) / len(wrong)
    else:
        recovery = None

    return {
        "n": n,
        "acc_gt": round(acc_gt, 4),
        "acc_dec": round(acc_dec, 4),
        "agreement": round(agreement, 4),
        "recovery": round(recovery, 4) if recovery is not None else None,
        "n_wrong": len(wrong),
    }


def plot_results(all_metrics, results, model_key, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Metrics comparison (token vs continuation method)
    ax = axes[0]
    metrics = ["acc_gt", "acc_dec", "agreement", "recovery"]
    labels = ["Acc. GT", "Acc. Dec.", "Agreement", "Recovery"]
    x = np.arange(len(metrics))
    w = 0.35
    token_vals = [all_metrics["token"]["all"].get(m, 0) or 0 for m in metrics]
    cont_vals = [all_metrics["cont"]["all"].get(m, 0) or 0 for m in metrics]
    ax.bar(x - w/2, token_vals, w, label="Token P(A) vs P(B)", color="#2E75B6", alpha=0.85)
    ax.bar(x + w/2, cont_vals, w, label="Continuation logprob", color="#C0392B", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fraction")
    ax.set_title("Overall Metrics")
    ax.legend(fontsize=8)
    ax.axhline(0.5, color="gray", ls="--", lw=0.5)
    ax.set_ylim(0, 1.1)

    # Panel 2: Method agreement
    ax = axes[1]
    both_long = sum(1 for r in results if r["token_choice"] == "B" and r["cont_choice"] == "B")
    both_short = sum(1 for r in results if r["token_choice"] == "A" and r["cont_choice"] == "A")
    disagree = len(results) - both_long - both_short
    ax.bar(["Both long", "Both short", "Disagree"],
           [both_long, both_short, disagree],
           color=["#27AE60", "#C0392B", "#95A5A6"], alpha=0.85)
    ax.set_ylabel("Count")
    ax.set_title("Token vs Continuation Agreement")

    # Panel 3: Probe gap distribution
    ax = axes[2]
    correct_gaps = [r["probe_gap"] for r in results if r["probe_matches_gt"]]
    wrong_gaps = [r["probe_gap"] for r in results if not r["probe_matches_gt"]]
    if correct_gaps:
        ax.hist(correct_gaps, bins=15, alpha=0.6, label=f"Probe correct (n={len(correct_gaps)})",
                color="#2E75B6")
    if wrong_gaps:
        ax.hist(wrong_gaps, bins=15, alpha=0.6, label=f"Probe wrong (n={len(wrong_gaps)})",
                color="#C0392B")
    ax.set_xlabel("Probe gap |pred_A - pred_B|")
    ax.set_ylabel("Count")
    ax.set_title("Probe Confidence")
    ax.legend(fontsize=8)

    plt.suptitle(f"{model_key} — Belief-Action Consistency (100 implicit pairs)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fname = outdir / f"belief_action_v2_{model_key}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def main():
    parser = argparse.ArgumentParser(description="SPAR RQ1 — Belief-Action v2")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--dataset", type=str, default="temporal_scope_implicit_expanded.json")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    outdir = Path(args.results_dir)
    pairs = load_dataset(args.dataset)

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
        print(f"Belief-Action v2: {model_key}  ({len(pairs)} pairs)")
        print(f"{'='*60}")

        results = []
        for i, pair in enumerate(pairs):
            r = evaluate_pair(model, tokenizer, pair, probe, scaler, pca, probe_layer)
            results.append(r)

            if (i + 1) % 20 == 0:
                # Running tally
                so_far_gt = sum(rr["token_matches_gt"] for rr in results) / len(results)
                print(f"  [{i+1}/{len(pairs)}]  running Acc.GT(token)={so_far_gt:.3f}")

        # Compute metrics for both methods
        all_metrics = {
            "token": {"all": compute_metrics(results, "token")},
            "cont": {"all": compute_metrics(results, "cont")},
        }

        # Per-category
        categories = sorted(set(r["category"] for r in results))
        for cat in categories:
            cat_results = [r for r in results if r["category"] == cat]
            if len(cat_results) >= 3:
                all_metrics["token"][cat] = compute_metrics(cat_results, "token")
                all_metrics["cont"][cat] = compute_metrics(cat_results, "cont")

        # Print summary
        print(f"\n  {'='*65}")
        print(f"  TOKEN METHOD (P('A') vs P('B') — proper forced choice)")
        print(f"  {'─'*65}")
        m = all_metrics["token"]["all"]
        print(f"  Acc. GT:    {m['acc_gt']:.3f}  (chose long-term)")
        print(f"  Acc. Dec.:  {m['acc_dec']:.3f}  (consistent with probe)")
        print(f"  Agreement:  {m['agreement']:.3f}  (probe agrees with GT)")
        rec = f"{m['recovery']:.3f}" if m['recovery'] is not None else "N/A"
        print(f"  Recovery:   {rec}  (of {m['n_wrong']} wrong choices)")

        print(f"\n  CONTINUATION METHOD (avg logprob — fluency-confounded)")
        print(f"  {'─'*65}")
        m = all_metrics["cont"]["all"]
        print(f"  Acc. GT:    {m['acc_gt']:.3f}")
        print(f"  Acc. Dec.:  {m['acc_dec']:.3f}")
        print(f"  Agreement:  {m['agreement']:.3f}")
        rec = f"{m['recovery']:.3f}" if m['recovery'] is not None else "N/A"
        print(f"  Recovery:   {rec}  (of {m['n_wrong']} wrong choices)")

        # Method comparison
        n_agree = sum(1 for r in results if r["token_choice"] == r["cont_choice"])
        print(f"\n  Token vs Continuation agree: {n_agree}/{len(results)} ({n_agree/len(results):.0%})")

        # Interpretation
        mt = all_metrics["token"]["all"]
        print(f"\n  Interpretation (token method):")
        if mt["recovery"] is not None and mt["recovery"] > 0.5:
            print(f"  → High recovery: errors are mostly REPRESENTATION FAILURES")
        elif mt["recovery"] is not None:
            print(f"  → Low recovery: errors are mostly GOAL-INCONSISTENCY")
        if mt["acc_dec"] > mt["acc_gt"]:
            print(f"  → Choices align more with internal belief than ground truth")

        plot_results(all_metrics, results, model_key, model_dir)

        output = {
            "model": model_key,
            "probe_layer": probe_layer,
            "n_pairs": len(pairs),
            "metrics": all_metrics,
            "results": results,
        }
        with open(model_dir / "belief_action_v2_results.json", "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n✓ Saved belief_action_v2_results.json → {model_dir}")

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
