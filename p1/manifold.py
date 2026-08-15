"""Manifold ladder: linear PC baseline, closed-spline circumplex, readouts.

Readout convention (documented operationalization of the Manifold Steering
recipe): work in the 64-d PCA space of story activations at a working layer.
Emotion centroids, ordered by norms-derived circumplex angle
atan2(z(arousal), z(valence)), get a closed periodic cubic spline. Any
activation x reads out as: theta = 2*pi*u of the nearest spline point
(emotion quality), r = ||x - center|| (intensity), eps = ||x - s(u)||
(off-manifold residual). Valence prediction from theta uses the circular
basis [cos, sin] fitted on train emotions only.
"""
import numpy as np
from scipy.interpolate import splev, splprep


def pca_basis(X, k=64):
    mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, Vt[:k]


def to_basis(X, mu, Vt):
    return (X - mu) @ Vt.T


def pc_scores(vectors):
    """vectors [E, D] -> (pc1 score per emotion, pc2 score) of the vector cloud."""
    mu = vectors.mean(0)
    U, S, Vt = np.linalg.svd(vectors - mu, full_matrices=False)
    return (vectors - mu) @ Vt[0], (vectors - mu) @ Vt[1]


def circumplex_angles(valence, arousal):
    vz = (valence - valence.mean()) / valence.std()
    az = (arousal - arousal.mean()) / arousal.std()
    return np.arctan2(az, vz)


def fit_spline(centroids, angles, smooth=None, k_fit=8):
    """centroids [E, 64]; fits a closed spline through them in circumplex-angle
    order, inside the top-k_fit PCA subspace of the centroids (scipy splprep
    caps at 10 dims). Returns a spline dict {tck, mu, B, order}; off-subspace
    energy is folded into the readout residual eps."""
    order = np.argsort(angles)
    mu_c = centroids.mean(0)
    _, _, Vt = np.linalg.svd(centroids - mu_c, full_matrices=False)
    B = Vt[:k_fit]
    pts = (centroids[order] - mu_c) @ B.T
    if smooth is None:
        smooth = pts.shape[0] * np.var(pts) * 0.5  # heavy smoothing: ring, not zigzag
    tck, _ = splprep(pts.T, s=smooth, per=True)
    return {"tck": tck, "mu": mu_c, "B": B, "order": order}


def spline_points(sp, m=2000):
    u = np.linspace(0, 1, m, endpoint=False)
    return u, np.stack(splev(u, sp["tck"]), axis=1)


def readout(X, sp, m=2000):
    """X [N, 64] -> theta [N] in [0, 2pi), r [N], eps [N] (eps includes the
    component orthogonal to the spline's fitting subspace)."""
    Xc = X - sp["mu"]
    X8 = Xc @ sp["B"].T
    orth = np.linalg.norm(Xc - X8 @ sp["B"], axis=1)
    u, pts = spline_points(sp, m)
    d = ((X8[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    nearest = d.argmin(1)
    theta = 2 * np.pi * u[nearest]
    r = np.linalg.norm(X8, axis=1)
    eps = np.sqrt(d[np.arange(len(X8)), nearest] + orth ** 2)
    return theta, r, eps


def circ_fit(theta, y):
    A = np.column_stack([np.cos(theta), np.sin(theta), np.ones(len(theta))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


def circ_predict(theta, coef):
    return np.column_stack([np.cos(theta), np.sin(theta), np.ones(len(theta))]) @ coef


def lin_fit(x, y):
    A = np.column_stack([x, np.ones(len(x))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


def procrustes_disparity(A, B):
    from scipy.spatial import procrustes
    _, _, disparity = procrustes(A, B)
    return float(disparity)
