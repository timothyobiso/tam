"""
SPAR RQ1 — Multi-Domain Temporal Probing
==========================================
Generates 500+ prompts across 6 domains with matched temporal structure,
then tests cross-domain transfer: train probe on N-1 domains, test on
the held-out domain. If the probe transfers, the temporal representation
is domain-general, not kitchen-specific.

Usage:
    python run_domain_transfer.py --models qwen3.5-9b
    python run_domain_transfer.py --models qwen3.5-4b qwen3.5-9b --generate_only
"""

import argparse
import json
import pickle
import random
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

from config import MODELS, DEFAULT_MODELS, SEED, CV_FOLDS

random.seed(SEED)

# ═══════════════════════════════════════════════════════════════════════════
# 1. Multi-Domain Prompt Generation
# ═══════════════════════════════════════════════════════════════════════════

TIME_SPANS = [
    ("the next 10 seconds", 0.17),
    ("the next 30 seconds", 0.5),
    ("the next minute", 1.0),
    ("the next 5 minutes", 5.0),
    ("the next 15 minutes", 15.0),
    ("the next 30 minutes", 30.0),
    ("the next hour", 60.0),
    ("the next 3 hours", 180.0),
    ("the next day", 1440.0),
    ("the next week", 10080.0),
    ("the next month", 43200.0),
    ("the next 3 months", 129600.0),
    ("the next year", 525600.0),
    ("the next 5 years", 2628000.0),
    ("the next decade", 5256000.0),
]

DOMAINS = {
    "kitchen": {
        "agents": ["the chef", "the sous chef", "the kitchen team", "the head cook"],
        "tasks_short": [
            "plate this dish", "flip the steak", "strain the pasta",
            "deglaze the pan", "taste for seasoning", "pull the bread out",
        ],
        "tasks_med": [
            "prep all the mise en place", "coordinate the appetizer course",
            "prepare three entrees simultaneously", "restock the line",
            "calibrate the oven temperatures", "organize the walk-in fridge",
        ],
        "tasks_long": [
            "develop a new seasonal menu", "overhaul the supply chain",
            "redesign the kitchen layout", "train an entire brigade",
            "establish a farm-to-table sourcing program", "open a second location",
        ],
    },
    "medical": {
        "agents": ["the doctor", "the surgeon", "the nursing team", "the attending physician"],
        "tasks_short": [
            "administer the injection", "check the patient's vitals",
            "apply pressure to the wound", "insert the IV line",
            "take the patient's temperature", "suction the airway",
        ],
        "tasks_med": [
            "stabilize the trauma patient", "complete the surgical prep",
            "coordinate the patient discharge", "review all lab results",
            "prepare the operating room", "run the diagnostic panel",
        ],
        "tasks_long": [
            "implement a new electronic health record system",
            "design a residency training curriculum",
            "restructure the emergency department workflow",
            "launch a clinical trial for the new treatment",
            "develop a hospital-wide infection control protocol",
            "build a telemedicine program for rural areas",
        ],
    },
    "software": {
        "agents": ["the developer", "the engineering team", "the tech lead", "the SRE"],
        "tasks_short": [
            "fix the failing unit test", "merge the pull request",
            "restart the crashed service", "update the config file",
            "add the missing import", "roll back the bad deploy",
        ],
        "tasks_med": [
            "refactor the authentication module", "set up the CI/CD pipeline",
            "migrate the database schema", "implement the API endpoint",
            "write integration tests for the payment flow", "debug the memory leak",
        ],
        "tasks_long": [
            "architect the microservices migration",
            "build a machine learning platform from scratch",
            "design the multi-region disaster recovery system",
            "develop a 3-year technical roadmap",
            "create a comprehensive developer onboarding program",
            "rewrite the legacy monolith in a modern stack",
        ],
    },
    "military": {
        "agents": ["the commander", "the squad leader", "the operations team", "the tactical officer"],
        "tasks_short": [
            "call for covering fire", "signal the retreat",
            "radio the coordinates", "clear the immediate area",
            "deploy the smoke screen", "secure the perimeter gate",
        ],
        "tasks_med": [
            "establish a forward operating base", "coordinate the patrol schedule",
            "plan the convoy route", "organize the supply resupply",
            "brief the incoming shift", "set up the communications relay",
        ],
        "tasks_long": [
            "develop the theater-wide campaign strategy",
            "restructure the force deployment across the region",
            "build alliances with local governance structures",
            "design a multi-year training pipeline for special operations",
            "plan the phased withdrawal and transition of authority",
            "establish a long-term intelligence collection network",
        ],
    },
    "business": {
        "agents": ["the CEO", "the management team", "the project manager", "the CFO"],
        "tasks_short": [
            "approve the pending invoice", "respond to the client email",
            "sign the contract amendment", "schedule the emergency meeting",
            "update the project status", "send the quarterly numbers",
        ],
        "tasks_med": [
            "prepare the board presentation", "negotiate the vendor contract",
            "restructure the marketing budget", "hire for the open positions",
            "launch the product update", "conduct the performance reviews",
        ],
        "tasks_long": [
            "execute the five-year growth strategy",
            "lead the company through an IPO",
            "build the international expansion plan",
            "develop a corporate sustainability initiative",
            "create a succession planning framework",
            "design the digital transformation roadmap",
        ],
    },
    "education": {
        "agents": ["the teacher", "the professor", "the department chair", "the curriculum team"],
        "tasks_short": [
            "take attendance for the class", "hand back the graded exams",
            "answer the student's question", "start the lab demonstration",
            "collect the homework assignments", "set up the projector",
        ],
        "tasks_med": [
            "prepare the midterm exam", "design the group project guidelines",
            "organize the parent-teacher conferences", "review the thesis draft",
            "plan the field trip logistics", "update the syllabus for next unit",
        ],
        "tasks_long": [
            "redesign the entire curriculum for the department",
            "write a grant proposal for the new research lab",
            "develop a 4-year degree program",
            "build an online learning platform for the university",
            "establish an international exchange program",
            "create a tenure-track mentorship pipeline",
        ],
    },
}

TEMPLATES = [
    "Over {timespan}, {agent} will need to {task}",
    "In {timespan}, {agent} must {task}",
    "The priority for {timespan} is for {agent} to {task}",
    "{agent} has {timespan} to {task}",
    "Within {timespan}, the goal is to {task}",
]


def generate_domain_prompts():
    all_prompts = []

    for domain, config in DOMAINS.items():
        for timespan, minutes in TIME_SPANS:
            if minutes < 5:
                tasks = config["tasks_short"]
                depth = random.randint(1, 3)
            elif minutes < 180:
                tasks = config["tasks_med"]
                depth = random.randint(3, 8)
            else:
                tasks = config["tasks_long"]
                depth = random.randint(8, 20)

            for agent in random.sample(config["agents"], min(2, len(config["agents"]))):
                task = random.choice(tasks)
                template = random.choice(TEMPLATES)
                prompt = template.format(timespan=timespan, agent=agent, task=task)
                all_prompts.append({
                    "prompt": prompt,
                    "time_horizon": minutes,
                    "planning_depth": depth,
                    "domain": domain,
                })

    random.shuffle(all_prompts)
    return all_prompts


# ═══════════════════════════════════════════════════════════════════════════
# 2. Extraction + Cross-Domain Transfer
# ═══════════════════════════════════════════════════════════════════════════

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


@torch.no_grad()
def extract_states(model, tokenizer, prompts, layer_indices):
    from collections import defaultdict
    states = defaultdict(list)
    for p in tqdm(prompts, desc="  extracting"):
        inputs = tokenizer(p["prompt"], return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model(**inputs, output_hidden_states=True)
        for li in layer_indices:
            vec = outputs.hidden_states[li][0, -1, :].clamp(-1e4, 1e4).float().cpu().numpy()
            states[li].append(vec)
    return {li: np.stack(v) for li, v in states.items()}


def run_cross_domain(states, prompts, layer_idx, target="log_time_horizon"):
    """Train on N-1 domains, test on held-out domain."""
    domains = sorted(set(p["domain"] for p in prompts))
    domain_arr = np.array([p["domain"] for p in prompts])
    y = np.log1p(np.array([p["time_horizon"] for p in prompts]))

    X = states[layer_idx].copy()
    X = np.nan_to_num(X, nan=0.0, posinf=1e4, neginf=-1e4)

    results = []
    for held_out in domains:
        train_mask = domain_arr != held_out
        test_mask = domain_arr == held_out
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaler = StandardScaler().fit(X_train)
            n_comp = min(20, X_train.shape[0] - 2)
            pca = PCA(n_components=n_comp, random_state=SEED).fit(scaler.transform(X_train))

            X_tr = pca.transform(scaler.transform(X_train))
            X_te = pca.transform(scaler.transform(X_test))

            # Ridge
            ridge = RidgeCV(alphas=np.logspace(-3, 3, 20)).fit(X_tr, y_train)
            ridge_r2 = ridge.score(X_te, y_test)

            # MLP
            mlp = MLPRegressor(hidden_layer_sizes=(64,), max_iter=2000,
                              random_state=SEED, early_stopping=True,
                              validation_fraction=0.15, alpha=0.01)
            mlp.fit(X_tr, y_train)
            mlp_r2 = mlp.score(X_te, y_test)

        results.append({
            "held_out": held_out,
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "ridge_r2": round(float(ridge_r2), 4),
            "mlp_r2": round(float(mlp_r2), 4),
        })

    return results


def run_within_domain(states, prompts, layer_idx):
    """Train and test within each domain separately (CV)."""
    domains = sorted(set(p["domain"] for p in prompts))
    domain_arr = np.array([p["domain"] for p in prompts])
    y = np.log1p(np.array([p["time_horizon"] for p in prompts]))
    X = states[layer_idx].copy()
    X = np.nan_to_num(X, nan=0.0, posinf=1e4, neginf=-1e4)

    results = []
    for domain in domains:
        mask = domain_arr == domain
        Xd, yd = X[mask], y[mask]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaler = StandardScaler().fit(Xd)
            n_comp = min(20, Xd.shape[0] - 2)
            pca = PCA(n_components=n_comp, random_state=SEED).fit(scaler.transform(Xd))
            Xd_pca = pca.transform(scaler.transform(Xd))

            ridge_scores = cross_val_score(
                RidgeCV(alphas=np.logspace(-3, 3, 20)),
                Xd_pca, yd, cv=min(CV_FOLDS, mask.sum() - 1), scoring="r2"
            )

        results.append({
            "domain": domain,
            "n": int(mask.sum()),
            "ridge_cv_r2": round(float(ridge_scores.mean()), 4),
        })

    return results


def plot_domain_transfer(cross_results, within_results, model_key, outdir):
    """Bar chart: held-out domain R² + within-domain R² for comparison."""
    import matplotlib.pyplot as plt

    domains = [r["held_out"] for r in cross_results]
    cross_ridge = [r["ridge_r2"] for r in cross_results]
    cross_mlp = [r["mlp_r2"] for r in cross_results]
    within_r2 = [r["ridge_cv_r2"] for r in within_results]

    x = np.arange(len(domains))
    w = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w, within_r2, w, label="Within-domain (ridge CV)", color="#2E75B6", alpha=0.8)
    ax.bar(x, cross_ridge, w, label="Cross-domain (ridge)", color="#C0392B", alpha=0.8)
    ax.bar(x + w, cross_mlp, w, label="Cross-domain (MLP)", color="#27AE60", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=30)
    ax.set_ylabel("R² (log time horizon)")
    ax.set_title(f"{model_key} — Domain Transfer: Train on 5 domains, test on held-out")
    ax.legend()
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fname = outdir / f"domain_transfer_{model_key}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def main():
    parser = argparse.ArgumentParser(description="SPAR RQ1 — Domain Transfer")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--stride", type=int, default=4,
                        help="Layer stride for extraction (default: 4 for speed)")
    parser.add_argument("--generate_only", action="store_true",
                        help="Only generate prompts, don't run extraction")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    # Generate prompts
    prompts = generate_domain_prompts()
    prompts_path = Path("prompts_multidomain.json")
    with open(prompts_path, "w") as f:
        json.dump(prompts, f, indent=2)

    print(f"Generated {len(prompts)} prompts across {len(DOMAINS)} domains")
    counts = Counter(p["domain"] for p in prompts)
    for d, n in counts.most_common():
        print(f"  {d}: {n}")
    ths = [p["time_horizon"] for p in prompts]
    print(f"  Time range: {min(ths):.2f} – {max(ths):.0f} minutes")

    if args.generate_only:
        print("Done (generate_only).")
        return

    outdir = Path(args.results_dir)

    for model_key in args.models:
        if model_key not in MODELS:
            print(f"⚠  Unknown model '{model_key}'. Skipping.")
            continue

        model, tokenizer, n_layers = load_model(model_key)
        layer_indices = list(range(0, n_layers + 1, args.stride))
        if n_layers not in layer_indices:
            layer_indices.append(n_layers)

        model_dir = outdir / model_key
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Domain transfer: {model_key}")
        print(f"  Extracting at layers: {layer_indices}")
        print(f"{'='*60}")

        states = extract_states(model, tokenizer, prompts, layer_indices)

        # Save activations for reuse
        with open(model_dir / "domain_activations.pkl", "wb") as f:
            pickle.dump(states, f)

        # Cross-domain transfer at final layer
        final_layer = n_layers
        print(f"\n  Cross-domain transfer @ L{final_layer}:")
        cross_results = run_cross_domain(states, prompts, final_layer)
        for r in cross_results:
            status = "✓" if r["ridge_r2"] > 0 else "✗"
            print(f"    {status} held_out={r['held_out']:12s}  "
                  f"ridge={r['ridge_r2']:.4f}  mlp={r['mlp_r2']:.4f}  (n={r['n_test']})")

        mean_ridge = np.mean([r["ridge_r2"] for r in cross_results])
        mean_mlp = np.mean([r["mlp_r2"] for r in cross_results])
        positive = sum(1 for r in cross_results if r["ridge_r2"] > 0)
        print(f"    Mean cross-domain: ridge={mean_ridge:.4f}  mlp={mean_mlp:.4f}  "
              f"({positive}/{len(cross_results)} positive)")

        # Within-domain for comparison
        print(f"\n  Within-domain CV @ L{final_layer}:")
        within_results = run_within_domain(states, prompts, final_layer)
        for r in within_results:
            print(f"    {r['domain']:12s}  ridge_cv={r['ridge_cv_r2']:.4f}  (n={r['n']})")

        plot_domain_transfer(cross_results, within_results, model_key, model_dir)

        # R² by layer (all domains pooled)
        print(f"\n  All-domain probe by layer:")
        y_all = np.log1p(np.array([p["time_horizon"] for p in prompts]))
        for li in layer_indices:
            X = states[li].copy()
            X = np.nan_to_num(X, nan=0.0, posinf=1e4, neginf=-1e4)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scaler = StandardScaler().fit(X)
                pca = PCA(n_components=min(20, X.shape[0]-2), random_state=SEED).fit(scaler.transform(X))
                Xp = pca.transform(scaler.transform(X))
                scores = cross_val_score(RidgeCV(alphas=np.logspace(-3, 3, 20)),
                                        Xp, y_all, cv=CV_FOLDS, scoring="r2")
            print(f"    L{li:3d}  R²_cv={scores.mean():.4f}")

        output = {
            "model": model_key,
            "n_prompts": len(prompts),
            "domains": list(DOMAINS.keys()),
            "domain_counts": dict(counts),
            "cross_domain": cross_results,
            "within_domain": within_results,
            "mean_cross_ridge": round(float(mean_ridge), 4),
            "mean_cross_mlp": round(float(mean_mlp), 4),
        }
        with open(model_dir / "domain_transfer_results.json", "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n✓ Saved domain_transfer_results.json → {model_dir}")

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
