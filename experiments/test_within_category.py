"""Quick test: is the bare-recipe probe a food-category classifier or a duration predictor?"""
import pickle
import numpy as np
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from collections import defaultdict

data = pickle.load(open("../results/qwen3.5-9b/bare_recipe_activations.pkl", "rb"))
recipes = data["recipes"]
y = data["y"]
X = data["states"][32]
X = np.nan_to_num(X, nan=0.0, posinf=1e4, neginf=-1e4)

# Bin recipes by cooking time
bins = defaultdict(list)
for i, r in enumerate(recipes):
    t = r["total_minutes"]
    if t < 15: b = "quick"
    elif t < 45: b = "medium"
    elif t < 120: b = "long"
    else: b = "very_long"
    bins[b].append(i)

print("Within-bin probe performance (if just category, R2 ~ 0):")
for bname, idxs in sorted(bins.items()):
    if len(idxs) < 20:
        continue
    Xi = X[idxs]
    yi = y[idxs]
    sc = StandardScaler().fit(Xi)
    pca = PCA(n_components=min(20, len(idxs) - 2)).fit(sc.transform(Xi))
    Xp = pca.transform(sc.transform(Xi))
    scores = cross_val_score(
        RidgeCV(alphas=np.logspace(-3, 3, 20)),
        Xp, yi, cv=min(5, len(idxs) - 1), scoring="r2",
    )
    print(f"  {bname:>10s} (n={len(idxs):3d}, range={min(yi):.1f}-{max(yi):.1f}): R2={scores.mean():.4f}")

# Residual correlation: time after removing bin mean
print()
print("Residual correlation (time after removing bin mean):")
pred_y = np.zeros(len(y))
for bname, idxs in bins.items():
    pred_y[idxs] = np.mean(y[idxs])
residual = y - pred_y

sc = StandardScaler().fit(X)
pca = PCA(n_components=50).fit(sc.transform(X))
Xp = pca.transform(sc.transform(X))
scores = cross_val_score(
    RidgeCV(alphas=np.logspace(-3, 3, 20)),
    Xp, residual, cv=5, scoring="r2",
)
print(f"  Residual R2 (after removing bin means): {scores.mean():.4f}")
print(f"  Original R2: 0.62")
print(f"  If residual R2 > 0 -> probe reads more than just category")
