"""
SPAR RQ1 — Config
Models, prompts with continuous labels, experiment settings.
"""

# ── Models to test ──────────────────────────────────────────────────────────
# Pick the ones that fit your GPU.  For a single 24GB card the small
# dense variants are safest.  For 48-80GB try the larger ones.

MODELS = {
    "qwen3.5-4b":  "Qwen/Qwen3.5-4B",
    "qwen3.5-9b":  "Qwen/Qwen3.5-9B",
    "qwen3.5-27b": "Qwen/Qwen3.5-27B",
}

# Default: smallest dense model, fits on a single 24GB GPU
DEFAULT_MODELS = ["qwen3.5-4b"]


# ── Prompts with continuous slider labels ───────────────────────────────────
# time_horizon  : minutes  (will be log-transformed for regression)
# planning_depth: number of sequential steps
# urgency       : 0 → 1

PROMPTS = [
    # --- immediate (seconds) ---
    {"prompt": "The oil is smoking — flip the steak NOW before it",
     "time_horizon": 0.1, "planning_depth": 1, "urgency": 1.0},
    {"prompt": "The timer just went off, quickly pull the bread from the",
     "time_horizon": 0.2, "planning_depth": 1, "urgency": 0.95},
    {"prompt": "The sauce is about to boil over, turn down the",
     "time_horizon": 0.1, "planning_depth": 1, "urgency": 1.0},
    {"prompt": "Plate is in the window — garnish and send it before",
     "time_horizon": 0.3, "planning_depth": 2, "urgency": 0.9},
    {"prompt": "The soufflé is at peak, serve it immediately or",
     "time_horizon": 0.2, "planning_depth": 1, "urgency": 1.0},
    {"prompt": "Drop the tempura batter in right now before the oil cools",
     "time_horizon": 0.1, "planning_depth": 1, "urgency": 0.95},
    {"prompt": "Torch that crème brûlée before the waiter comes back",
     "time_horizon": 0.5, "planning_depth": 1, "urgency": 0.85},
    {"prompt": "Pull the pasta now, it's 10 seconds from overcooked",
     "time_horizon": 0.15, "planning_depth": 1, "urgency": 1.0},

    # --- near-term (minutes) ---
    {"prompt": "Four orders on the rail. Fire apps for table 6 and start prepping",
     "time_horizon": 15.0, "planning_depth": 6, "urgency": 0.6},
    {"prompt": "The risotto needs 18 minutes. While it cooks, prep the salad and plate the",
     "time_horizon": 18.0, "planning_depth": 4, "urgency": 0.4},
    {"prompt": "Two tables sat down at once. Coordinate so both get apps within",
     "time_horizon": 20.0, "planning_depth": 5, "urgency": 0.5},
    {"prompt": "Fish rests 5 min after searing. Use that window to reduce sauce and plate",
     "time_horizon": 10.0, "planning_depth": 4, "urgency": 0.5},
    {"prompt": "We're behind on table 3. Prioritize mains, push dessert to",
     "time_horizon": 25.0, "planning_depth": 5, "urgency": 0.7},
    {"prompt": "Three apps and two mains fire in the next 10 min. Assign stations:",
     "time_horizon": 10.0, "planning_depth": 5, "urgency": 0.65},
    {"prompt": "Pasta water boiling, sauce almost done, bread out in 2 min. Next,",
     "time_horizon": 5.0, "planning_depth": 3, "urgency": 0.55},
    {"prompt": "Table 8 wants all courses timed together. Coordinate the grill and sauté for",
     "time_horizon": 30.0, "planning_depth": 6, "urgency": 0.5},

    # --- strategic (hours / days) ---
    {"prompt": "Tonight's tasting menu has 12 courses. Start the stock now for course 7 in",
     "time_horizon": 180.0, "planning_depth": 12, "urgency": 0.2},
    {"prompt": "Plan the prep schedule for tomorrow's 40-guest event with five courses and",
     "time_horizon": 1440.0, "planning_depth": 15, "urgency": 0.1},
    {"prompt": "For next week's menu change, source new ingredients, test three dishes, train the team on",
     "time_horizon": 10080.0, "planning_depth": 15, "urgency": 0.05},
    {"prompt": "Design the full kitchen workflow for a restaurant opening in three months covering",
     "time_horizon": 131400.0, "planning_depth": 20, "urgency": 0.02},
    {"prompt": "Over the next year develop a seasonal rotation of 48 dishes accounting for",
     "time_horizon": 525600.0, "planning_depth": 20, "urgency": 0.01},
    {"prompt": "Build a two-year training program for apprentice chefs covering knife skills through",
     "time_horizon": 1051200.0, "planning_depth": 20, "urgency": 0.01},
    {"prompt": "Develop a five-year supply chain strategy for the restaurant group including",
     "time_horizon": 2628000.0, "planning_depth": 20, "urgency": 0.005},
    {"prompt": "Create a decade-long expansion plan for opening 12 locations with",
     "time_horizon": 5256000.0, "planning_depth": 20, "urgency": 0.005},
]

# ── Pure temporal prompts (no kitchen domain) ───────────────────────────────
# Isolates temporal signal from domain-specific features
TEMPORAL_PROMPTS = [
    {"prompt": "In the next few seconds, the ball will",
     "time_horizon": 0.05, "planning_depth": 1},
    {"prompt": "Over the next minute, carefully",
     "time_horizon": 1.0, "planning_depth": 2},
    {"prompt": "Within the next hour, we should",
     "time_horizon": 60.0, "planning_depth": 4},
    {"prompt": "Over the next day, the plan involves",
     "time_horizon": 1440.0, "planning_depth": 6},
    {"prompt": "This week we need to accomplish",
     "time_horizon": 10080.0, "planning_depth": 8},
    {"prompt": "Over the next month, the strategy requires",
     "time_horizon": 43200.0, "planning_depth": 10},
    {"prompt": "By next quarter, we aim to",
     "time_horizon": 131400.0, "planning_depth": 12},
    {"prompt": "Over the next year, the vision includes",
     "time_horizon": 525600.0, "planning_depth": 15},
    {"prompt": "Within the next decade, transformative changes will",
     "time_horizon": 5256000.0, "planning_depth": 20},
    {"prompt": "Across the next century, civilizational shifts in",
     "time_horizon": 52560000.0, "planning_depth": 20},
]

# ── Curvature seeds: prompts the model generates from ──────────────────────
# We auto-regressively generate and collect hidden states at each token
# to compute activation trajectory curvature (Wang et al. 2026).
GENERATION_SEEDS = [
    "Plan a three-course dinner for four people. Start with",
    "Organize tomorrow's kitchen prep schedule. First,",
    "You have 30 minutes before service. Prioritize:",
    "Design a weekly meal prep plan. Monday:",
    "Coordinate these simultaneous orders:",
]

# ── Hyperparameters ─────────────────────────────────────────────────────────
RIDGE_ALPHA = 1.0
CV_FOLDS = 5
LAYER_STRIDE = 2          # probe every Nth layer (set 1 for all)
GEN_MAX_TOKENS = 150       # tokens to generate for curvature analysis
SEED = 42
