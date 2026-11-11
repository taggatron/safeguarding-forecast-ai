"""
Module 10: Optimal Transport — Sinkhorn Divergence, Sliced W2 Velocity & Seasonal Barycenters
(`esoteric_math/features/f10_optimal_transport.py`).

Captures distributional reorganization and vulnerable cohort tail bifurcation even when
institutional mean referral counts remain flat:
  1. Unbiased Entropic Sinkhorn Divergence S_eps(mu_t, nu_calm) >= 0 via log-domain Sinkhorn-Knopp.
  2. Sliced 2-Wasserstein Distributional Velocity SW_2(mu_t, mu_{t-1}).
  3. Exact 1D-projected 2-Wasserstein distances to training-fold UK FE Seasonal Barycenters
     (Autumn, Spring, and Summer terms).
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.special import logsumexp

from esoteric_math.features.base import CausalBaseTransformer
from esoteric_math.schema_bridge import SchemaBundle


def sinkhorn_potentials_log_domain(
    a: np.ndarray,
    b: np.ndarray,
    cost_matrix: np.ndarray,
    eps: float = 0.25,
    max_iter: int = 60,
    tol: float = 1e-6,
) -> float:
    """
    Compute regularized entropic optimal transport dual value OT_eps(a, b)
    using log-stabilized Sinkhorn-Knopp iterations.
    """
    a_prob = np.maximum(np.asarray(a, dtype=float), 1e-12)
    a_prob = a_prob / np.sum(a_prob)
    b_prob = np.maximum(np.asarray(b, dtype=float), 1e-12)
    b_prob = b_prob / np.sum(b_prob)

    log_a = np.log(a_prob)
    log_b = np.log(b_prob)

    n, m = cost_matrix.shape
    f = np.zeros(n, dtype=float)
    g = np.zeros(m, dtype=float)

    for _ in range(max_iter):
        f_prev = f.copy()
        # f_i = -eps * log sum_j exp((g_j - C_ij)/eps + log b_j)
        f = -eps * logsumexp((g[None, :] - cost_matrix) / eps + log_b[None, :], axis=1)
        g = -eps * logsumexp((f[:, None] - cost_matrix) / eps + log_a[:, None], axis=0)
        if np.max(np.abs(f - f_prev)) < tol:
            break

    # Symmetric single refinement for self-transport exactness
    f_sym = 0.5 * (f - eps * logsumexp((g[None, :] - cost_matrix) / eps + log_b[None, :], axis=1))
    g_sym = 0.5 * (g - eps * logsumexp((f_sym[:, None] - cost_matrix) / eps + log_a[:, None], axis=0))

    log_plan = (f_sym[:, None] + g_sym[None, :] - cost_matrix) / eps + log_a[:, None] + log_b[None, :]
    plan = np.exp(np.clip(log_plan, -80.0, 20.0))
    # Dual + primal regularized objective: <a, f> + <b, g> - eps * (<P, 1> - 1)
    ot_val = float(np.dot(a_prob, f_sym) + np.dot(b_prob, g_sym) - eps * (np.sum(plan) - 1.0))
    return ot_val


def sinkhorn_divergence(
    x: np.ndarray,
    a: np.ndarray,
    y: np.ndarray,
    b: np.ndarray,
    eps: float = 0.25,
    max_iter: int = 60,
) -> float:
    """
    Unbiased Entropic Sinkhorn Divergence:
      S_eps(mu, nu) = OT_eps(mu, nu) - 0.5 * OT_eps(mu, mu) - 0.5 * OT_eps(nu, nu) >= 0.
    """
    x_arr = np.atleast_2d(np.asarray(x, dtype=float))
    y_arr = np.atleast_2d(np.asarray(y, dtype=float))

    c_xy = np.sum((x_arr[:, None, :] - y_arr[None, :, :]) ** 2, axis=-1)
    c_xx = np.sum((x_arr[:, None, :] - x_arr[None, :, :]) ** 2, axis=-1)
    c_yy = np.sum((y_arr[:, None, :] - y_arr[None, :, :]) ** 2, axis=-1)

    ot_xy = sinkhorn_potentials_log_domain(a, b, c_xy, eps=eps, max_iter=max_iter)
    ot_xx = sinkhorn_potentials_log_domain(a, a, c_xx, eps=eps, max_iter=max_iter)
    ot_yy = sinkhorn_potentials_log_domain(b, b, c_yy, eps=eps, max_iter=max_iter)
    return float(max(0.0, ot_xy - 0.5 * ot_xx - 0.5 * ot_yy))


def _weighted_1d_quantiles(values: np.ndarray, weights: np.ndarray, u_grid: np.ndarray) -> np.ndarray:
    """Exact left-continuous empirical quantile function on u_grid in (0, 1)."""
    v = np.asarray(values, dtype=float)
    w = np.maximum(np.asarray(weights, dtype=float), 1e-12)
    w = w / np.sum(w)
    order = np.argsort(v, kind="mergesort")
    v_sorted = v[order]
    w_sorted = w[order]
    cdf = np.cumsum(w_sorted)
    cdf[-1] = 1.0
    idx = np.searchsorted(cdf, u_grid, side="left")
    idx = np.clip(idx, 0, len(v_sorted) - 1)
    return v_sorted[idx]


def sliced_wasserstein_quantiles(
    points: np.ndarray,
    weights: np.ndarray,
    projections: np.ndarray,
    u_grid: np.ndarray,
) -> np.ndarray:
    """
    Project empirical point cloud `points` (M, D) onto `projections` (L, D) and
    return matrix of 1D quantile curves of shape (L, Q).
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    L = projections.shape[0]
    Q = len(u_grid)
    proj_vals = pts @ projections.T  # (M, L)
    q_mat = np.empty((L, Q), dtype=float)
    for ell in range(L):
        q_mat[ell] = _weighted_1d_quantiles(proj_vals[:, ell], weights, u_grid)
    return q_mat


class OptimalTransportTransformer(CausalBaseTransformer):
    """
    Module 10 (`f10_optimal_transport.py`):
    Entropic Sinkhorn Divergence, Sliced 2-Wasserstein Distributional Velocity,
    and Seasonal Wasserstein Barycenter transport distances.
    """

    module_id = "f10"
    module_name = "Optimal Transport (Sinkhorn & Sliced Wasserstein)"

    def __init__(
        self,
        window: int = 12,
        sinkhorn_eps: float = 0.35,
        n_slices: int = 16,
        n_quantiles: int = 32,
        causal_mode: Literal["telemetry", "forecast"] = "telemetry",
        train_cutoff: Optional[int] = None,
        return_dataframe: bool = True,
    ) -> None:
        super().__init__(
            causal_mode=causal_mode,
            train_cutoff=train_cutoff,
            return_dataframe=return_dataframe,
        )
        self.window = window
        self.sinkhorn_eps = sinkhorn_eps
        self.n_slices = n_slices
        self.n_quantiles = n_quantiles

    @staticmethod
    def _season_bucket(date_str: str) -> str:
        ts = pd.Timestamp(date_str)
        m = int(ts.month)
        if m in (9, 10, 11, 12):
            return "autumn"
        if m in (1, 2, 3):
            return "spring"
        return "summer"

    def _build_empirical_cloud_at_t(
        self,
        bundle: SchemaBundle,
        t: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct strictly causal empirical support Z_t in R^{M x D} and probability weights a_t
        combining cross-cohort behavioral stalks at t with trailing causal trajectory cloud [t-W+1, t].
        """
        stalks_t = bundle.tier_c.stalks[t]  # (V, d_stalk)
        stress_t = bundle.tier_c.node_scalar_stress[t][:, None]  # (V, 1)
        cohort_pts = np.hstack([stress_t, stalks_t])  # (V, 1 + d_stalk)
        D = cohort_pts.shape[1]

        # Append trailing causal window snapshots of Tier B standardized channels padded/projected to D
        start = max(0, t - self.window + 1)
        win_cov = bundle.tier_b.covariate_matrix[start : t + 1]  # (W_cur, D_B)
        w_cur = win_cov.shape[0]
        win_pts = np.zeros((w_cur, D), dtype=float)
        d_copy = min(D, win_cov.shape[1])
        win_pts[:, :d_copy] = win_cov[:, :d_copy]

        points = np.vstack([cohort_pts, win_pts])
        v_weights = np.maximum(bundle.tier_c.node_weights, 1e-6)
        v_weights = 0.65 * (v_weights / np.sum(v_weights))
        t_weights = 0.35 * (np.ones(w_cur, dtype=float) / float(w_cur))
        weights = np.concatenate([v_weights, t_weights])
        weights = weights / np.sum(weights)
        return points, weights

    def fit(self, X: Union[pd.DataFrame, SchemaBundle], y: Any = None) -> "OptimalTransportTransformer":
        bundle = self._ensure_bundle_fit(X)
        n_train = bundle.n_steps
        sample_pts, sample_w = self._build_empirical_cloud_at_t(bundle, 0)
        D = sample_pts.shape[1]

        rng = np.random.default_rng(1010)
        raw_dirs = rng.normal(size=(self.n_slices, D))
        norms = np.linalg.norm(raw_dirs, axis=1, keepdims=True)
        self.projections_ = raw_dirs / np.maximum(norms, 1e-12)
        self.u_grid_ = np.linspace(0.5 / self.n_quantiles, 1.0 - 0.5 / self.n_quantiles, self.n_quantiles)

        # Collect calm-period reference cloud and seasonal Sliced-W2 barycenters on training split
        all_q: List[np.ndarray] = []
        season_q: Dict[str, List[np.ndarray]] = {"autumn": [], "spring": [], "summer": []}
        calm_clouds: List[np.ndarray] = []

        y_train = bundle.tier_a.y_rate_per_1000
        active_mask = bundle.tier_a.is_instructional > 0.25
        med_rate = float(np.median(y_train[active_mask])) if np.any(active_mask) else float(np.median(y_train))

        for t in range(n_train):
            pts, w = self._build_empirical_cloud_at_t(bundle, t)
            q_mat = sliced_wasserstein_quantiles(pts, w, self.projections_, self.u_grid_)
            all_q.append(q_mat)
            sb = self._season_bucket(str(bundle.tier_a.dates[t]))
            season_q[sb].append(q_mat)
            if y_train[t] <= med_rate * 1.15 and bundle.tier_a.is_instructional[t] > 0.25:
                # Cohort portion has fixed shape (V, D)
                calm_clouds.append(pts[: len(bundle.tier_c.node_ids)])

        global_bary = np.mean(all_q, axis=0) if all_q else np.zeros((self.n_slices, self.n_quantiles))
        self.seasonal_barycenters_: Dict[str, np.ndarray] = {}
        for sb in ("autumn", "spring", "summer"):
            self.seasonal_barycenters_[sb] = (
                np.mean(season_q[sb], axis=0) if len(season_q[sb]) > 0 else global_bary.copy()
            )

        if calm_clouds:
            self.calm_reference_cloud_ = np.mean(calm_clouds, axis=0)
        else:
            self.calm_reference_cloud_ = sample_pts[: len(bundle.tier_c.node_ids)].copy()
        v_w = np.maximum(bundle.tier_c.node_weights, 1e-6)
        self.calm_reference_weights_ = v_w / np.sum(v_w)
        return self

    def transform(self, X: Union[pd.DataFrame, SchemaBundle]) -> Union[pd.DataFrame, np.ndarray]:
        bundle = self._ensure_bundle_transform(X)
        T = bundle.n_steps
        col_names = [
            "f10_sinkhorn_div_calm",
            "f10_sliced_w2_velocity",
            "f10_w2_bary_autumn",
            "f10_w2_bary_spring",
            "f10_w2_bary_summer",
            "f10_w2_active_season_bary",
            "f10_tail_bifurcation_ratio",
        ]
        feats = np.zeros((T, len(col_names)), dtype=float)

        prev_q: Optional[np.ndarray] = None
        prev_mean: float = 0.0

        for t in range(T):
            pts, w = self._build_empirical_cloud_at_t(bundle, t)
            q_mat = sliced_wasserstein_quantiles(pts, w, self.projections_, self.u_grid_)

            # 1. Unbiased Entropic Sinkhorn Divergence to calm training distribution
            s_div = sinkhorn_divergence(
                pts,
                w,
                self.calm_reference_cloud_,
                self.calm_reference_weights_,
                eps=self.sinkhorn_eps,
                max_iter=45,
            )

            # 2. Sliced 2-Wasserstein distributional velocity W_2(mu_t, mu_{t-1})
            cur_mean = float(np.sum(w[:, None] * pts[:, 0:1]))
            if prev_q is None:
                sw2_vel = 0.0
                mean_delta = 0.0
            else:
                sw2_vel = float(np.sqrt(np.mean((q_mat - prev_q) ** 2)))
                mean_delta = abs(cur_mean - prev_mean)

            # 3. Seasonal Wasserstein barycenter distances
            w2_aut = float(np.sqrt(np.mean((q_mat - self.seasonal_barycenters_["autumn"]) ** 2)))
            w2_spr = float(np.sqrt(np.mean((q_mat - self.seasonal_barycenters_["spring"]) ** 2)))
            w2_sum = float(np.sqrt(np.mean((q_mat - self.seasonal_barycenters_["summer"]) ** 2)))
            sb = self._season_bucket(str(bundle.tier_a.dates[t]))
            w2_active = {"autumn": w2_aut, "spring": w2_spr, "summer": w2_sum}[sb]

            tail_bif = sw2_vel / (mean_delta + 0.15)

            feats[t, 0] = s_div
            feats[t, 1] = sw2_vel
            feats[t, 2] = w2_aut
            feats[t, 3] = w2_spr
            feats[t, 4] = w2_sum
            feats[t, 5] = w2_active
            feats[t, 6] = tail_bif

            prev_q = q_mat
            prev_mean = cur_mean

        self.last_diagnostics_ = {
            "sinkhorn_div_calm": feats[:, 0].copy(),
            "sliced_w2_velocity": feats[:, 1].copy(),
            "w2_active_season_bary": feats[:, 5].copy(),
            "tail_bifurcation_ratio": feats[:, 6].copy(),
        }
        return self._format_output(feats, col_names, bundle, X)
