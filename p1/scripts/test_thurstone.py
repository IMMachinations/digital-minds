"""Synthetic-recovery test for the Thurstonian fit — run BEFORE any GPU work.

Simulates 300 items with known mu/sigma2, generates observations with the
same shape as the 1B battery (round-0 random partners + extra random pairs,
fractional p from the true model), fits, and requires:
  - Pearson(mu_hat, mu_true) > 0.95
  - Spearman(sigma2_hat, sigma2_true) > 0.4  (variance is harder; rank-level)
  - scipy L-BFGS cross-check: Pearson(mu_adam, mu_lbfgs) > 0.99
Usage: uv run python scripts/test_thurstone.py
"""
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import thurstone

N = 300
PAIRS_PER_ITEM = 8
EXTRA_PAIRS = 2600  # round-0 (~2400 obs) + adaptive rounds stand-in -> ~5k total


def true_p(mu, s2, i, j):
    z = (mu[i] - mu[j]) / math.sqrt(s2[i] + s2[j])
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def main():
    rng = random.Random(0)
    npr = np.random.RandomState(0)
    mu = npr.normal(0, 1, N)
    mu -= mu.mean()
    s2 = np.exp(npr.normal(0, 0.6, N))
    s2 /= s2.mean()

    pairs = set()
    for i in range(N):
        for _ in range(PAIRS_PER_ITEM):
            j = rng.randrange(N)
            if j != i:
                pairs.add((min(i, j), max(i, j)))
    while len(pairs) < PAIRS_PER_ITEM * N // 2 + EXTRA_PAIRS:
        i, j = rng.randrange(N), rng.randrange(N)
        if i != j:
            pairs.add((min(i, j), max(i, j)))
    obs = [(i, j, true_p(mu, s2, i, j)) for i, j in pairs]
    print(f"{N} items, {len(obs)} pair observations")

    res = thurstone.fit(N, obs, steps=2000)
    mu_hat, s2_hat = np.array(res["mu"]), np.array(res["sigma2"])
    r_mu = pearsonr(mu_hat, mu)[0]
    rho_s2 = spearmanr(s2_hat, s2)[0]
    print(f"adam:  pearson(mu)={r_mu:+.4f}  spearman(sigma2)={rho_s2:+.4f}  nll={res['nll']}")

    # L-BFGS cross-check on the same likelihood
    ii = torch.tensor([o[0] for o in obs])
    jj = torch.tensor([o[1] for o in obs])
    pp = torch.tensor([o[2] for o in obs], dtype=torch.float64)

    def f(x):
        t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        loss = thurstone.nll(t[:N], t[N:], ii, jj, pp)
        loss.backward()
        return loss.item(), t.grad.numpy()

    out = minimize(f, np.zeros(2 * N), jac=True, method="L-BFGS-B",
                   options={"maxiter": 1500})
    mu_lb = out.x[:N] - out.x[:N].mean()
    r_cross = pearsonr(mu_hat, mu_lb)[0]
    print(f"lbfgs: nll={out.fun:.6f}  pearson(mu_adam, mu_lbfgs)={r_cross:+.4f}")

    assert r_mu > 0.95, f"mu recovery too weak: {r_mu}"
    assert rho_s2 > 0.4, f"sigma2 rank recovery too weak: {rho_s2}"
    assert r_cross > 0.99, f"optimizer disagreement: {r_cross}"
    print("PASS")


if __name__ == "__main__":
    main()
