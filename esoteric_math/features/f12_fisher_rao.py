"""
Module 12: Fisher-Rao Information Geometry — Negative Binomial Statistical Manifold
(`esoteric_math/features/f12_fisher_rao.py`).

Endows the Negative Binomial parameter space theta = (mu, r) in R_+^2 with its canonical
diagonal Fisher-Rao Riemannian metric:
  g_{mu,mu}(mu, r) = r / (mu * (r + mu)) > 0
  g_{r,r}(mu, r)   = E_{Y ~ NB(mu, r)}[ (partial_r log P(Y | mu, r))^2 ] > 0
  g_{mu,r}(mu, r)  = 0  (exact parameter orthogonality of (mu, r))

Outputs:
  - Fisher-Rao Geodesic Velocity ds_t = sqrt(g_{mu,mu} (Delta mu_t)^2 + g_{r,r} (Delta r_t)^2)
  - Dispersion-Curvature Burstiness Energy g_{r,r} (Delta r_t)^2
  - Rolling Geodesic Arc Length and Burstiness Manifold Share
"""
from __future__ import annotations

from typing import Any, Literal, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.special import digamma
from scipy.stats import nbinom

from esoteric_math.features.base import CausalBaseTransformer
from esoteric_math.schema_bridge import SchemaBundle


def nb_fisher_information_diagonal(mu: float, r: float) -> Tuple[float, float]:
    """
    Compute the strictly positive diagonal Fisher Information metric (g_mumu, g_rr)
    of the Negative Binomial NB(mu, r) statistical manifold at (mu > 0, r > 0).

    Uses Bartlett's identity g_rr = E[(s_r(Y))^2] over support / quantile quadrature
    so positivity is mathematically guaranteed without catastrophic cancellation.
    """
    mu_pos = float(max(mu, 1e-4))
    r_pos = float(max(r, 1e-3))

    g_mumu = r_pos / (mu_pos * (r_pos + mu_pos))

    p_nb = r_pos / (r_pos + mu_pos)
    p_nb = float(np.clip(p_nb, 1e-9, 1.0 - 1e-9))
    std_nb = np.sqrt(mu_pos + (mu_pos ** 2) / r_pos)
    upper_y = int(min(600, np.ceil(mu_pos + 7.5 * std_nb + 20)))

    if upper_y <= 500:
        y_grid = np.arange(0, upper_y + 1, dtype=float)
        pmf = nbinom.pmf(y_grid, n=r_pos, p=p_nb)
        pmf_sum = float(np.sum(pmf))
        if pmf_sum > 1e-12:
            pmf = pmf / pmf_sum
        score_r = (
            digamma(y_grid + r_pos)
            - digamma(r_pos)
            + np.log(r_pos / (r_pos + mu_pos))
            + (mu_pos - y_grid) / (r_pos + mu_pos)
        )
        g_rr = float(np.sum(pmf * (score_r ** 2)))
    else:
        # Quantile midpoint quadrature for very high count regimes
        u_q = np.linspace(0.005, 0.995, 99)
        y_q = np.maximum(0.0, nbinom.ppf(u_q, n=r_pos, p=p_nb))
        score_r = (
            digamma(y_q + r_pos)
            - digamma(r_pos)
            + np.log(r_pos / (r_pos + mu_pos))
            + (mu_pos - y_q) / (r_pos + mu_pos)
        )
        g_rr = float(np.mean(score_r ** 2))

    # Asymptotic safety floor ~ 0.5 * (mu / (r * (r + mu)))^2
    asymp_floor = 0.5 * ((mu_pos / (r_pos * (r_pos + mu_pos))) ** 2)
    g_rr = float(max(g_rr, asymp_floor, 1e-12))
    return float(max(g_mumu, 1e-12)), g_rr


def nb_fisher_information_vectorized(mu_arr: np.ndarray, r_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized wrapper returning strictly positive (g_mumu, g_rr) arrays."""
    mu_v = np.asarray(mu_arr, dtype=float).ravel()
    r_v = np.asarray(r_arr, dtype=float).ravel()
    g_mu = np.empty_like(mu_v)
    g_r = np.empty_like(r_v)
    for i in range(len(mu_v)):
        g_mu[i], g_r[i] = nb_fisher_information_diagonal(float(mu_v[i]), float(r_v[i]))
    return g_mu, g_r


class FisherRaoInformationGeometryTransformer(CausalBaseTransformer):
    """
    Module 12 (`f12_fisher_rao.py`):
    Rolling causal Negative Binomial (mu_t, r_t) statistical manifold tracking,
    Fisher-Rao geodesic velocity ds_t, and dispersion-curvature burstiness energy.
    """

    module_id = "f12"
    module_name = "Fisher-Rao Information Geometry (NB Manifold)"

    def __init__(
        self,
        window: int = 12,
        arc_window: int = 6,
        r_prior: float = 6.0,
        r_max: float = 60.0,
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
        self.arc_window = arc_window
        self.r_prior = r_prior
        self.r_max = r_max

    def _estimate_causal_nb_params(self, counts: np.ndarray, is_active: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        T = len(counts)
        mu_hat = np.empty(T, dtype=float)
        r_hat = np.empty(T, dtype=float)

        prior_mu = float(self.prior_mu_)
        prior_r = float(self.prior_r_)

        for t in range(T):
            start = max(0, t - self.window + 1)
            win_y = counts[start : t + 1]
            win_act = is_active[start : t + 1] > 0.20
            vals = win_y[win_act] if np.sum(win_act) >= 3 else win_y

            n_eff = len(vals)
            shrink = float(n_eff) / float(n_eff + 3.0)
            sample_m = float(np.mean(vals)) if n_eff > 0 else prior_mu
            mu_t = max(0.15, shrink * sample_m + (1.0 - shrink) * prior_mu)

            if n_eff >= 3:
                sample_v = float(np.var(vals, ddof=1))
                excess_var = max(sample_v - mu_t, 0.02 * (mu_t ** 2))
                r_mom = (mu_t ** 2) / max(excess_var, 1e-6)
                r_mom = float(np.clip(r_mom, 0.35, self.r_max))
                r_t = shrink * r_mom + (1.0 - shrink) * prior_r
            else:
                r_t = prior_r

            mu_hat[t] = mu_t
            r_hat[t] = float(np.clip(r_t, 0.35, self.r_max))
        return mu_hat, r_hat

    def fit(self, X: Union[pd.DataFrame, SchemaBundle], y: Any = None) -> "FisherRaoInformationGeometryTransformer":
        bundle = self._ensure_bundle_fit(X)
        y_cnt = np.maximum(bundle.tier_a.y_count, 0.0)
        act = bundle.tier_a.is_instructional > 0.20
        y_act = y_cnt[act] if np.sum(act) >= 4 else y_cnt
        m0 = float(max(np.mean(y_act), 0.5))
        v0 = float(np.var(y_act, ddof=1)) if len(y_act) >= 2 else m0 * 1.3
        r0 = (m0 ** 2) / max(v0 - m0, 0.05 * (m0 ** 2))
        self.prior_mu_ = m0
        self.prior_r_ = float(np.clip(r0, 0.5, self.r_max))
        return self

    def transform(self, X: Union[pd.DataFrame, SchemaBundle]) -> Union[pd.DataFrame, np.ndarray]:
        bundle = self._ensure_bundle_transform(X)
        T = bundle.n_steps
        y_cnt = np.maximum(bundle.tier_a.y_count, 0.0)
        mu_hat, r_hat = self._estimate_causal_nb_params(y_cnt, bundle.tier_a.is_instructional)

        col_names = [
            "f12_nb_mu_hat",
            "f12_nb_r_hat",
            "f12_fisher_g_mumu",
            "f12_fisher_g_rr",
            "f12_fisher_rao_velocity",
            "f12_dispersion_burst_energy",
            "f12_cumulative_rao_arc",
            "f12_burst_manifold_share",
        ]
        feats = np.zeros((T, len(col_names)), dtype=float)

        ds_series = np.zeros(T, dtype=float)
        for t in range(T):
            if t == 0:
                g_mu, g_r = nb_fisher_information_diagonal(mu_hat[0], r_hat[0])
                ds_t = 0.0
                burst_e = 0.0
                share = 0.0
            else:
                mu_mid = 0.5 * (mu_hat[t] + mu_hat[t - 1])
                r_mid = 0.5 * (r_hat[t] + r_hat[t - 1])
                g_mu, g_r = nb_fisher_information_diagonal(mu_mid, r_mid)
                d_mu = mu_hat[t] - mu_hat[t - 1]
                d_r = r_hat[t] - r_hat[t - 1]
                e_mu = g_mu * (d_mu ** 2)
                burst_e = g_r * (d_r ** 2)
                ds_t = float(np.sqrt(max(0.0, e_mu + burst_e)))
                share = float(burst_e / (e_mu + burst_e + 1e-12))

            ds_series[t] = ds_t
            arc = float(np.sum(ds_series[max(0, t - self.arc_window + 1) : t + 1]))

            feats[t, 0] = mu_hat[t]
            feats[t, 1] = r_hat[t]
            feats[t, 2] = g_mu
            feats[t, 3] = g_r
            feats[t, 4] = ds_t
            feats[t, 5] = burst_e
            feats[t, 6] = arc
            feats[t, 7] = share

        self.last_diagnostics_ = {
            "mu_hat": mu_hat,
            "r_hat": r_hat,
            "fisher_rao_velocity": ds_series,
            "dispersion_burst_energy": feats[:, 5].copy(),
        }
        return self._format_output(feats, col_names, bundle, X)
