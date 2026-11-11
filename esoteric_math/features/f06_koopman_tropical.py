"""
Module 06: Sliding Koopman Exact DMD & Tropical (Max-Plus) Algebra
(`esoteric_math/features/f06_koopman_tropical.py`).

Implements:
- Sliding truncated-SVD Exact Dynamic Mode Decomposition (DMD) over causal windows {t-W..t},
  extracting modal growth rates, Koopman spectral radius, oscillatory mode frequency, and 1-step residual.
- Tropical (Max-Plus) semiring bottleneck polynomials and Tropical Hilbert Projective Distance
  to training-fold crisis centroids.
"""
from __future__ import annotations

from typing import Any, List, Literal, Optional, Tuple, Union
import numpy as np
import pandas as pd

from esoteric_math.features.base import CausalBaseTransformer
from esoteric_math.schema_bridge import SchemaBundle


def tropical_hilbert_projective_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Tropical Hilbert Projective Metric on R^d / (R * 1):
        d_trop(x, y) = max_i (x_i - y_i) - min_i (x_i - y_i) >= 0.
    Satisfies non-negativity, symmetry, triangle inequality, and d_trop(x + c*1, y) == d_trop(x, y).
    """
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    return float(np.max(diff) - np.min(diff))


def _sliding_exact_dmd_step(snapshots: np.ndarray, rank: int = 3) -> Tuple[float, float, float, float]:
    """
    Compute truncated-SVD Exact DMD on `snapshots` of shape (D, M) (chronological columns 0..M-1 <= t).
    Returns (max_growth_rate, spectral_radius, max_frequency, one_step_residual).
    """
    d_dim, m_cols = snapshots.shape
    if m_cols < 3:
        return 0.0, 0.0, 0.0, 0.0

    X0 = snapshots[:, :-1]
    X1 = snapshots[:, 1:]
    if np.linalg.norm(X0) < 1e-10:
        return 0.0, 0.0, 0.0, float(np.linalg.norm(X1[:, -1]))

    U, s, Vt = np.linalg.svd(X0, full_matrices=False)
    valid_r = int(np.sum(s > 1e-7 * s[0]))
    r = max(1, min(rank, valid_r, len(s)))
    U_r = U[:, :r]
    s_r = s[:r]
    V_r = Vt[:r, :].T

    A_tilde = U_r.T @ X1 @ V_r @ np.diag(1.0 / s_r)
    eigvals = np.linalg.eigvals(A_tilde)
    mags = np.abs(eigvals)
    rho = float(np.max(mags)) if len(mags) > 0 else 0.0
    growth = float(np.max(np.log(np.maximum(1e-4, mags)))) if len(mags) > 0 else 0.0
    freq = float(np.max(np.abs(np.angle(eigvals)))) if len(eigvals) > 0 else 0.0

    z_prev = snapshots[:, -2]
    z_curr = snapshots[:, -1]
    z_pred = U_r @ (A_tilde @ (U_r.T @ z_prev))
    resid = float(np.linalg.norm(z_curr - z_pred))
    return float(np.clip(growth, -5.0, 5.0)), float(np.clip(rho, 0.0, 5.0)), freq, resid


class KoopmanTropicalTransformer(CausalBaseTransformer):
    """
    Module 06: Sliding Koopman DMD + Tropical (Max-Plus) Bottleneck & Projective Geometry.
    """

    module_id = "f06"
    module_name = "Koopman Exact DMD & Tropical Algebra"

    def __init__(
        self,
        dmd_window: int = 12,
        dmd_rank: int = 3,
        causal_mode: Literal["telemetry", "forecast"] = "telemetry",
        train_cutoff: Optional[int] = None,
        return_dataframe: bool = True,
    ) -> None:
        super().__init__(
            causal_mode=causal_mode,
            train_cutoff=train_cutoff,
            return_dataframe=return_dataframe,
        )
        self.dmd_window = dmd_window
        self.dmd_rank = dmd_rank

    @staticmethod
    def _build_state_matrix(bundle: SchemaBundle, rate_mean: float, rate_std: float) -> np.ndarray:
        r_z = (bundle.tier_a.y_rate_per_1000 - rate_mean) / max(1e-4, rate_std)
        cov = bundle.tier_b.covariate_matrix[:, : min(4, bundle.tier_b.covariate_matrix.shape[1])]
        return np.column_stack([r_z, cov])

    def fit(self, X: Union[pd.DataFrame, SchemaBundle], y: Any = None) -> "KoopmanTropicalTransformer":
        bundle = self._ensure_bundle_fit(X)
        r = bundle.tier_a.y_rate_per_1000
        self.rate_mean_ = float(np.mean(r)) if len(r) > 0 else 0.0
        self.rate_std_ = max(1e-4, float(np.std(r))) if len(r) > 0 else 1.0

        Z_tr = self._build_state_matrix(bundle, self.rate_mean_, self.rate_std_)
        if len(Z_tr) >= 4:
            thresh = float(np.percentile(r, 85))
            crisis_mask = r >= thresh
            self.crisis_centroid_ = (
                np.mean(Z_tr[crisis_mask], axis=0) if np.any(crisis_mask) else np.max(Z_tr, axis=0)
            )
            calm_mask = r <= float(np.percentile(r, 35))
            self.calm_centroid_ = (
                np.mean(Z_tr[calm_mask], axis=0) if np.any(calm_mask) else np.min(Z_tr, axis=0)
            )
            # Tropical max-plus weights from correlation with high-rate regime
            self.trop_weights_ = np.clip(self.crisis_centroid_ - self.calm_centroid_, -2.0, 2.0)
        else:
            d = Z_tr.shape[1]
            self.crisis_centroid_ = np.ones(d, dtype=float)
            self.calm_centroid_ = -np.ones(d, dtype=float)
            self.trop_weights_ = np.zeros(d, dtype=float)

        self.is_fitted_ = True
        return self

    def transform(self, X: Union[pd.DataFrame, SchemaBundle]) -> Union[pd.DataFrame, np.ndarray]:
        if not getattr(self, "is_fitted_", False):
            self.fit(X)
        bundle = self._ensure_bundle_transform(X)
        n = bundle.n_steps
        Z = self._build_state_matrix(bundle, self.rate_mean_, self.rate_std_)  # (T, D)
        w_len = max(4, int(self.dmd_window))

        feats = np.zeros((n, 7), dtype=float)
        names = [
            "f06_koopman_modal_growth_rate",
            "f06_koopman_spectral_radius",
            "f06_koopman_dominant_freq",
            "f06_koopman_one_step_residual",
            "f06_tropical_maxplus_linear",
            "f06_tropical_maxplus_quadratic",
            "f06_tropical_hilbert_crisis_dist",
        ]

        w_vec = self.trop_weights_
        c_crisis = self.crisis_centroid_

        for t in range(n):
            win = Z[max(0, t - w_len + 1) : t + 1].T  # (D, <= w_len)
            growth, rho, freq, resid = _sliding_exact_dmd_step(win, rank=self.dmd_rank)

            z_t = Z[t]
            # Max-plus linear polynomial: max_i (w_i + z_{t,i})
            trop_lin = float(np.max(w_vec + z_t))
            # Max-plus degree-2 cascade: max_{i,j} (0.5*(w_i + w_j) + z_{t,i} + z_{t,j})
            pair_sum = z_t[:, None] + z_t[None, :] + 0.5 * (w_vec[:, None] + w_vec[None, :])
            trop_quad = float(np.max(pair_sum))
            # Tropical Hilbert projective distance to crisis centroid
            d_crisis = tropical_hilbert_projective_distance(z_t, c_crisis)

            feats[t] = [growth, rho, freq, resid, trop_lin, trop_quad, d_crisis]

        self.last_diagnostics_ = {"crisis_centroid": self.crisis_centroid_.copy()}
        return self._format_output(feats, names, bundle, X)
