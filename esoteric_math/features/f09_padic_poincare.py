"""
Module 09: p-adic Ultrametric Vladimirov Wavelets & Poincare Ball Hyperbolic Geometry
(`esoteric_math/features/f09_padic_poincare.py`).

Implements:
- Exact hierarchical p-adic ultrametric space (V, d_p) satisfying the Strong Triangle Inequality
  d_p(x, z) <= max(d_p(x, y), d_p(y, z)) for all (x, y, z) in V^3.
- Discrete Vladimirov Pseudo-Differential Operators D^alpha psi for alpha in {0.5, 1.0}.
- 2D Poincare Ball B^2 = {z in R^2 : ||z|| < 1} hyperbolic embedding, Riemannian Frechet mean
  acuity radius d_B(0, z_bar_t), and hyperbolic dispersion.
"""
from __future__ import annotations

from typing import Any, List, Literal, Optional, Union
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from esoteric_math.features.base import CausalBaseTransformer
from esoteric_math.schema_bridge import SchemaBundle


def build_padic_ultrametric_matrix(profiles: np.ndarray, p: int = 2) -> np.ndarray:
    """
    Construct an exact p-adic ultrametric distance matrix `U` of shape (V, V) over V entities
    using hierarchical dendrogram tree valuations p^{-v_p(u, v)}.
    Guaranteed to satisfy the Strong Triangle Inequality:
        U[i, k] <= max(U[i, j], U[j, k])  for all i, j, k.
    """
    v_cnt = len(profiles)
    if v_cnt <= 1:
        return np.zeros((v_cnt, v_cnt), dtype=float)

    d_cond = pdist(profiles, metric="euclidean")
    if np.max(d_cond) < 1e-12:
        # Deterministic binary tree valuation on index order when profiles are identical
        d_cond = pdist(np.arange(v_cnt, dtype=float)[:, None], metric="euclidean")

    Z = linkage(d_cond, method="single")
    # Assign strictly decreasing p-adic valuations p^{-(depth)} from root (merge_idx = V-2 -> dist = 1.0)
    # down to leaves: later merges in linkage have larger ultrametric height!
    n_merges = len(Z)
    members: List[List[int]] = [[i] for i in range(v_cnt)]
    U = np.zeros((v_cnt, v_cnt), dtype=float)

    for m_idx in range(n_merges):
        c1, c2 = int(Z[m_idx, 0]), int(Z[m_idx, 1])
        # Valuation k = n_merges - 1 - m_idx in {0, 1, ..., n_merges-1} -> dist = p^{-k}
        val_k = n_merges - 1 - m_idx
        dist_p = float(p) ** (-float(val_k))
        for u in members[c1]:
            for v in members[c2]:
                U[u, v] = dist_p
                U[v, u] = dist_p
        members.append(members[c1] + members[c2])

    return U


def apply_vladimirov_operator(
    psi: np.ndarray,
    ultrametric_mat: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Discrete p-adic Vladimirov pseudo-differential operator of order alpha > 0:
        (D^alpha psi)(u) = sum_{v != u} (psi(u) - psi(v)) / d_p(u, v)^{1 + alpha} * w_v.
    """
    v_cnt = len(psi)
    if v_cnt <= 1:
        return np.zeros_like(psi)
    with np.errstate(divide="ignore"):
        inv_kernel = np.where(ultrametric_mat > 1e-12, ultrametric_mat ** (-(1.0 + alpha)), 0.0)
    np.fill_diagonal(inv_kernel, 0.0)
    diff = psi[:, None] - psi[None, :]
    return np.sum(diff * inv_kernel * weights[None, :], axis=1)


def poincare_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Exact Riemannian geodesic distance in the Poincare unit ball B^2."""
    nx2 = min(0.999999, float(np.dot(x, x)))
    ny2 = min(0.999999, float(np.dot(y, y)))
    diff2 = float(np.dot(x - y, x - y))
    arg = 1.0 + (2.0 * diff2) / max(1e-12, (1.0 - nx2) * (1.0 - ny2))
    return float(np.arccosh(max(1.0, arg)))


def _poincare_log_map(base: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Riemannian logarithmic map Log_{base}(target) on B^2 via Mobius gyrovector addition."""
    u = -base
    v = target
    uv = float(np.dot(u, v))
    nu2 = float(np.dot(u, u))
    nv2 = float(np.dot(v, v))
    denom = max(1e-12, 1.0 + 2.0 * uv + nu2 * nv2)
    # Mobius addition (-base) (+) target
    mob = ((1.0 + 2.0 * uv + nv2) * u + (1.0 - nu2) * v) / denom
    norm_m = float(np.linalg.norm(mob))
    if norm_m < 1e-12:
        return np.zeros_like(base)
    lam_base = 2.0 / max(1e-8, 1.0 - nu2)
    return (2.0 / (lam_base * norm_m)) * float(np.arctanh(min(0.99999, norm_m))) * mob


def _poincare_exp_map(base: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    """Riemannian exponential map Exp_{base}(tangent) on B^2."""
    norm_t = float(np.linalg.norm(tangent))
    if norm_t < 1e-12:
        return base.copy()
    nb2 = min(0.9999, float(np.dot(base, base)))
    lam_base = 2.0 / (1.0 - nb2)
    direction = tangent / norm_t
    step_vec = float(np.tanh(min(12.0, 0.5 * lam_base * norm_t))) * direction
    # Mobius addition base (+) step_vec
    uv = float(np.dot(base, step_vec))
    ns2 = float(np.dot(step_vec, step_vec))
    denom = max(1e-12, 1.0 + 2.0 * uv + nb2 * ns2)
    out = ((1.0 + 2.0 * uv + ns2) * base + (1.0 - nb2) * step_vec) / denom
    n_out = float(np.linalg.norm(out))
    if n_out >= 0.995:
        out = out * (0.995 / n_out)
    return out


def compute_poincare_frechet_mean(points: np.ndarray, weights: np.ndarray, n_iter: int = 12) -> np.ndarray:
    """Riemannian Karcher / Frechet mean on the 2D Poincare disk B^2."""
    w = weights / max(1e-12, float(np.sum(weights)))
    mu = np.sum(points * w[:, None], axis=0)
    if np.linalg.norm(mu) >= 0.98:
        mu = mu * (0.95 / np.linalg.norm(mu))
    for _ in range(n_iter):
        grad = np.zeros(2, dtype=float)
        for i in range(len(points)):
            grad += float(w[i]) * _poincare_log_map(mu, points[i])
        if np.linalg.norm(grad) < 1e-7:
            break
        mu = _poincare_exp_map(mu, 0.65 * grad)
    return mu


class PadicPoincareTransformer(CausalBaseTransformer):
    """
    Module 09: p-adic Vladimirov Operators & 2D Poincare Ball Hyperbolic Acuity Transformer.
    """

    module_id = "f09"
    module_name = "p-adic Ultrametrics & Poincare Ball Geometry"

    def __init__(
        self,
        prime_p: int = 2,
        causal_mode: Literal["telemetry", "forecast"] = "telemetry",
        train_cutoff: Optional[int] = None,
        return_dataframe: bool = True,
    ) -> None:
        super().__init__(
            causal_mode=causal_mode,
            train_cutoff=train_cutoff,
            return_dataframe=return_dataframe,
        )
        self.prime_p = prime_p

    def fit(self, X: Union[pd.DataFrame, SchemaBundle], y: Any = None) -> "PadicPoincareTransformer":
        bundle = self._ensure_bundle_fit(X)
        tc = bundle.tier_c
        v_cnt = len(tc.node_ids)
        # Learn hierarchical p-adic tree strictly from training-fold vertex stress profiles
        profiles = tc.node_scalar_stress.T  # (V, T_train)
        self.ultrametric_ = build_padic_ultrametric_matrix(profiles, p=self.prime_p)
        self.weights_ = tc.node_weights.copy()
        self.angles_ = np.linspace(0.0, 2.0 * np.pi, num=max(1, v_cnt), endpoint=False)
        self.is_fitted_ = True
        return self

    def transform(self, X: Union[pd.DataFrame, SchemaBundle]) -> Union[pd.DataFrame, np.ndarray]:
        if not getattr(self, "is_fitted_", False):
            self.fit(X)
        bundle = self._ensure_bundle_transform(X)
        n = bundle.n_steps
        S = bundle.tier_c.node_scalar_stress  # (T, V)
        v_cnt = S.shape[1]

        feats = np.zeros((n, 6), dtype=float)
        names = [
            "f09_vladimirov_alpha05_norm",
            "f09_vladimirov_alpha05_max",
            "f09_vladimirov_alpha10_norm",
            "f09_vladimirov_alpha10_max",
            "f09_poincare_frechet_acuity_radius",
            "f09_poincare_hyperbolic_dispersion",
        ]

        cos_a = np.cos(self.angles_)
        sin_a = np.sin(self.angles_)

        for t in range(n):
            psi_t = S[t]
            d05 = apply_vladimirov_operator(psi_t, self.ultrametric_, self.weights_, alpha=0.5)
            d10 = apply_vladimirov_operator(psi_t, self.ultrametric_, self.weights_, alpha=1.0)

            # Map vertex stress acuity into B^2: softplus(psi_t) -> radius in (0, 0.96)
            softplus = np.log1p(np.exp(np.clip(psi_t, -10.0, 10.0)))
            radii = np.tanh(0.35 * softplus) * 0.96
            disk_pts = np.column_stack([radii * cos_a[:v_cnt], radii * sin_a[:v_cnt]])

            mu_b = compute_poincare_frechet_mean(disk_pts, self.weights_, n_iter=10)
            acuity_rad = poincare_distance(np.zeros(2, dtype=float), mu_b)
            disp = sum(
                float(self.weights_[i]) * (poincare_distance(mu_b, disk_pts[i]) ** 2)
                for i in range(v_cnt)
            )

            feats[t] = [
                float(np.linalg.norm(d05)),
                float(np.max(np.abs(d05))) if len(d05) > 0 else 0.0,
                float(np.linalg.norm(d10)),
                float(np.max(np.abs(d10))) if len(d10) > 0 else 0.0,
                acuity_rad,
                float(disp),
            ]

        self.last_diagnostics_ = {"ultrametric_matrix": self.ultrametric_.copy()}
        return self._format_output(feats, names, bundle, X)
