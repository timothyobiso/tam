"""
SPAR RQ1 — Prompt Generator
============================
Generates ~300 labeled prompts from templates with controlled variation.
Each prompt has a known time_horizon and planning_depth.

The key insight: with 24 hand-written prompts you can't train a probe.
With 300 template-generated prompts (and PCA to ~20 dims), ridge
regression actually has enough data to find real signal.

Usage:
    python generate_prompts.py              # writes to prompts_generated.json
    python run_probes.py --prompts prompts_generated.json
"""

import json
import random
import itertools

random.seed(42)

# ── Time spans with known minute values ─────────────────────────────────────
TIME_SPANS = [
    # (phrase, minutes)
    ("the next 10 seconds", 0.17),
    ("the next 30 seconds", 0.5),
    ("the next minute", 1.0),
    ("the next 2 minutes", 2.0),
    ("the next 5 minutes", 5.0),
    ("the next 10 minutes", 10.0),
    ("the next 15 minutes", 15.0),
    ("the next 30 minutes", 30.0),
    ("the next hour", 60.0),
    ("the next 2 hours", 120.0),
    ("the next 3 hours", 180.0),
    ("the next 6 hours", 360.0),
    ("the next 12 hours", 720.0),
    ("the next day", 1440.0),
    ("the next 3 days", 4320.0),
    ("the next week", 10080.0),
    ("the next 2 weeks", 20160.0),
    ("the next month", 43200.0),
    ("the next 3 months", 129600.0),
    ("the next 6 months", 259200.0),
    ("the next year", 525600.0),
    ("the next 5 years", 2628000.0),
    ("the next decade", 5256000.0),
]

# ── Template families ───────────────────────────────────────────────────────

# Family 1: "Over {timespan}, {agent} will need to {task}"
AGENTS = ["the chef", "the team", "we", "the kitchen staff", "the head chef",
          "the sous chef", "the prep cook", "the manager"]
TASKS_SHORT = [
    "plate this dish", "finish the sauce", "sear the protein",
    "check the oven", "strain the pasta", "deglaze the pan",
    "whisk the eggs", "dice the onions", "taste for seasoning",
]
TASKS_MED = [
    "prep all the mise en place", "coordinate the appetizer course",
    "prepare three entrees simultaneously", "set up the dessert station",
    "restock the line", "reorganize the walk-in",
    "calibrate all the equipment", "train the new hire on sauté",
]
TASKS_LONG = [
    "develop a new seasonal menu", "overhaul the supply chain",
    "redesign the kitchen layout", "build relationships with local farms",
    "implement a new inventory system", "train an entire brigade",
    "plan the annual holiday catering schedule",
    "establish a farm-to-table sourcing program",
]

# Family 2: "Step N of M: {action}. Currently at step {K}."
PLAN_ACTIONS = [
    "heat the oil", "chop the vegetables", "brown the meat",
    "add the stock", "reduce the sauce", "season to taste",
    "rest the protein", "plate the components", "garnish and serve",
    "clean the station", "check inventory", "prep the next order",
    "fire the appetizers", "coordinate with front of house",
    "set the timers", "check temperatures", "label and date containers",
    "sharpen the knives", "review the reservation list",
    "assign stations for service",
]

# Family 3: "In {timespan}, the following needs to happen: {description}"
DESCRIPTIONS = [
    "the entire prep list needs to be finished",
    "all stations must be set up and ready",
    "the special of the day needs to be perfected",
    "inventory counts need to be completed",
    "the menu needs to be finalized and printed",
    "all dietary accommodations need to be confirmed",
    "the kitchen must be spotless for inspection",
    "every sauce needs to be tasted and adjusted",
    "the ordering for next week must be submitted",
    "the staff schedule needs to be posted",
]


def generate_family1():
    """'Over {timespan}, {agent} will need to {task}'"""
    prompts = []
    for timespan, minutes in TIME_SPANS:
        # Match task complexity to time horizon
        if minutes < 5:
            tasks = TASKS_SHORT
            depth = random.randint(1, 3)
        elif minutes < 120:
            tasks = TASKS_MED
            depth = random.randint(3, 8)
        else:
            tasks = TASKS_LONG
            depth = random.randint(8, 20)

        for agent in random.sample(AGENTS, min(3, len(AGENTS))):
            task = random.choice(tasks)
            prompt = f"Over {timespan}, {agent} will need to {task}"
            prompts.append({
                "prompt": prompt,
                "time_horizon": minutes,
                "planning_depth": depth,
                "family": "over_timespan",
            })
    return prompts


def generate_family2():
    """Step-count prompts: explicit N-step plans at various stages."""
    prompts = []
    for total_steps in [2, 3, 4, 5, 7, 10, 12, 15, 18, 20]:
        for current_step in [1, total_steps // 2, total_steps - 1]:
            if current_step < 1:
                current_step = 1
            actions = random.sample(PLAN_ACTIONS, min(total_steps, len(PLAN_ACTIONS)))
            steps_str = ". ".join([f"Step {i+1}: {a}" for i, a in enumerate(actions)])
            prompt = (f"Here is a {total_steps}-step plan: {steps_str}. "
                      f"Currently on step {current_step}. Next,")

            # Time horizon scales roughly with step count
            minutes = total_steps * random.uniform(3, 10)
            prompts.append({
                "prompt": prompt,
                "time_horizon": round(minutes, 1),
                "planning_depth": total_steps,
                "family": "step_plan",
            })
    return prompts


def generate_family3():
    """'In {timespan}, {description}'"""
    prompts = []
    for timespan, minutes in TIME_SPANS:
        for desc in random.sample(DESCRIPTIONS, min(3, len(DESCRIPTIONS))):
            prompt = f"In {timespan}, {desc}"
            if minutes < 10:
                depth = random.randint(1, 3)
            elif minutes < 1440:
                depth = random.randint(3, 10)
            else:
                depth = random.randint(10, 20)
            prompts.append({
                "prompt": prompt,
                "time_horizon": minutes,
                "planning_depth": depth,
                "family": "in_timespan",
            })
    return prompts


def generate_family4():
    """Pure temporal: 'The deadline is in {timespan}. Plan accordingly.'"""
    prompts = []
    starters = [
        "The deadline is in {t}. Plan accordingly:",
        "You have {t} to complete everything. Begin with",
        "There are {t} remaining. The priority is",
        "With only {t} left, focus on",
        "{t} from now, everything must be ready. Start by",
    ]
    for timespan, minutes in TIME_SPANS:
        for template in random.sample(starters, min(3, len(starters))):
            prompt = template.format(t=timespan)
            if minutes < 10:
                depth = random.randint(1, 3)
            elif minutes < 1440:
                depth = random.randint(3, 10)
            else:
                depth = random.randint(10, 20)
            prompts.append({
                "prompt": prompt,
                "time_horizon": minutes,
                "planning_depth": depth,
                "family": "deadline",
            })
    return prompts


def main():
    all_prompts = []
    all_prompts.extend(generate_family1())
    all_prompts.extend(generate_family2())
    all_prompts.extend(generate_family3())
    all_prompts.extend(generate_family4())

    # Shuffle
    random.shuffle(all_prompts)

    print(f"Generated {len(all_prompts)} prompts")
    print(f"  Family counts:")
    from collections import Counter
    counts = Counter(p["family"] for p in all_prompts)
    for fam, n in counts.most_common():
        print(f"    {fam}: {n}")

    # Time horizon distribution
    ths = [p["time_horizon"] for p in all_prompts]
    print(f"  Time horizon range: {min(ths):.2f} – {max(ths):.0f} minutes")
    print(f"  Planning depth range: {min(p['planning_depth'] for p in all_prompts)} – "
          f"{max(p['planning_depth'] for p in all_prompts)}")

    with open("prompts_generated.json", "w") as f:
        json.dump(all_prompts, f, indent=2)
    print(f"\nSaved → prompts_generated.json")


if __name__ == "__main__":
    main()
