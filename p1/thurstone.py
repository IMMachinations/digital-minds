"""Thurstonian utility fit (Case V relaxed: per-item mean AND variance).

Each item i has utility U_i ~ N(mu_i, sigma2_i); P(i beats j) =
Phi((mu_i - mu_j) / sqrt(sigma2_i + sigma2_j)). Observations are fractional
Bernoullis: each battery prompt yields p in (0,1) (sigmoid of the A/B letter
logit difference), which carries more information than a thresholded choice.

Identification: the likelihood is invariant to shifting all mu and to jointly
scaling mu by 1/sqrt(s) and sigma2 by 1/s, so after every optimizer step we
project back to mean(mu)=0, mean(sigma2)=1. Variance is the
strength-of-preference readout (weak preference = high variance).

CPU float64 throughout; 300 items / 5k pairs fits in seconds.
"""
import torch

EPS = 1e-7


def _win_prob(mu, log_s2, i, j):
    s2 = log_s2.exp()
    z = (mu[i] - mu[j]) / torch.sqrt(s2[i] + s2[j])
    return torch.special.ndtr(z).clamp(EPS, 1 - EPS)


def nll(mu, log_s2, i, j, p):
    """Mean fractional-Bernoulli negative log-likelihood over observations."""
    P = _win_prob(mu, log_s2, i, j)
    return -(p * P.log() + (1 - p) * (1 - P).log()).mean()


def _project(mu, log_s2):
    """Enforce mean(mu)=0, mean(sigma2)=1 without changing the likelihood."""
    with torch.no_grad():
        s = log_s2.exp().mean()
        log_s2 -= s.log()
        mu -= mu.mean()
        mu /= s.sqrt()


def fit(n_items, obs, steps=2000, lr=0.05, seed=0):
    """obs: list of (i, j, p) with p = P(item i preferred over item j).

    Returns dict with per-item mu, sigma2, final mean NLL, and loss trace.
    """
    torch.manual_seed(seed)
    i = torch.tensor([o[0] for o in obs], dtype=torch.long)
    j = torch.tensor([o[1] for o in obs], dtype=torch.long)
    p = torch.tensor([o[2] for o in obs], dtype=torch.float64).clamp(EPS, 1 - EPS)
    mu = torch.zeros(n_items, dtype=torch.float64, requires_grad=True)
    log_s2 = torch.zeros(n_items, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([mu, log_s2], lr=lr)
    trace = []
    for step in range(steps):
        opt.zero_grad()
        loss = nll(mu, log_s2, i, j, p)
        loss.backward()
        opt.step()
        _project(mu, log_s2)
        if step % 100 == 0 or step == steps - 1:
            trace.append(round(loss.item(), 6))
    n_pairs = torch.zeros(n_items, dtype=torch.long)
    for idx in (i, j):
        n_pairs += torch.bincount(idx, minlength=n_items)
    return {
        "mu": mu.detach().tolist(),
        "sigma2": log_s2.detach().exp().tolist(),
        "n_obs": n_pairs.tolist(),
        "nll": trace[-1],
        "trace": trace,
    }


def refit_mu(n_items, obs, sigma2, steps=800, lr=0.05):
    """Fit mu only, holding sigma2 fixed — used for per-template/per-order
    consistency refits so template Spearmans compare like with like."""
    i = torch.tensor([o[0] for o in obs], dtype=torch.long)
    j = torch.tensor([o[1] for o in obs], dtype=torch.long)
    p = torch.tensor([o[2] for o in obs], dtype=torch.float64).clamp(EPS, 1 - EPS)
    log_s2 = torch.tensor(sigma2, dtype=torch.float64).log()
    mu = torch.zeros(n_items, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([mu], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = nll(mu, log_s2, i, j, p)
        loss.backward()
        opt.step()
        with torch.no_grad():
            mu -= mu.mean()
    return mu.detach().tolist()


def fit_anchored(n_items, obs, anchor_idx, anchor_mu, anchor_s2,
                 steps=2500, lr=0.05, seed=0):
    """Thurstonian fit with anchor items' (mu, sigma2) frozen at supplied values
    (e.g. from a prior full battery). Identification is inherited from the
    anchors, so no re-centering/rescaling is applied. obs as in fit()."""
    torch.manual_seed(seed)
    i = torch.tensor([o[0] for o in obs], dtype=torch.long)
    j = torch.tensor([o[1] for o in obs], dtype=torch.long)
    p = torch.tensor([o[2] for o in obs], dtype=torch.float64).clamp(EPS, 1 - EPS)
    a_idx = torch.tensor(anchor_idx, dtype=torch.long)
    a_mu = torch.tensor(anchor_mu, dtype=torch.float64)
    a_ls2 = torch.tensor(anchor_s2, dtype=torch.float64).log()
    mu = torch.zeros(n_items, dtype=torch.float64, requires_grad=True)
    log_s2 = torch.zeros(n_items, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([mu, log_s2], lr=lr)
    trace = []
    for step in range(steps):
        opt.zero_grad()
        loss = nll(mu, log_s2, i, j, p)
        loss.backward()
        opt.step()
        with torch.no_grad():
            mu[a_idx] = a_mu
            log_s2[a_idx] = a_ls2
        if step % 200 == 0 or step == steps - 1:
            trace.append(round(loss.item(), 6))
    return {"mu": mu.detach().tolist(), "sigma2": log_s2.detach().exp().tolist(),
            "nll": trace[-1], "trace": trace}
