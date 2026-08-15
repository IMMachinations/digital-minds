"""Synthetic recovery for fit_anchored: 12 pinned anchors + free items measured
only against anchors (the Stage 1-XL protocol shape). CPU, run before GPU time.
Usage: uv run python scripts/test_anchored.py"""
import math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import thurstone

rng = np.random.RandomState(0)
N_FREE, N_ANC = 400, 12
N = N_FREE + N_ANC
mu = rng.normal(0, 1.2, N)
s2 = np.exp(rng.normal(0, 0.5, N))
anchor_idx = list(range(N_FREE, N))
mu[anchor_idx] = np.linspace(-2, 2, N_ANC)  # mu-spanning anchors

def true_p(i, j):
    z = (mu[i] - mu[j]) / math.sqrt(s2[i] + s2[j])
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))

obs = []
for i in range(N_FREE):
    for a in anchor_idx:
        for _ in range(4):  # 4 readouts per item-anchor (2 templates x 2 orders)
            obs.append((i, a, float(np.clip(true_p(i, a) + rng.normal(0, 0.06), 0.01, 0.99))))
fit = thurstone.fit_anchored(N, obs, anchor_idx, mu[anchor_idx].tolist(),
                             s2[anchor_idx].tolist())
r = np.corrcoef(np.array(fit["mu"])[:N_FREE], mu[:N_FREE])[0, 1]
print(f"anchored recovery: pearson(mu_hat, mu_true) = {r:+.4f} over {N_FREE} free items")
assert r > 0.95, r
print("PASS")
