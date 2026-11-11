"""
Module 11: Random Matrix Theory — Marchenko-Pastur Spectral Denoising & Tracy-Widom Edge Gap
(`esoteric_math/features/f11_rmt_marchenko.py`).

Evaluates rolling causal sample correlation matrices C_t in R^{N x N} over window W:
  - Marchenko-Pastur bulk upper edge lambda_+ = (1 + sqrt(N / W_eff))^2
  - Escaping signal eigenvalue count sum_i 1(lambda_i > lambda_+)
  - Systemic institutional synchronization ratio lambda_max / N
  - Tracy-Widom scaled edge gap (lambda_max - lambda_+) * N^{2/3}
  - MP spectral denoising Frobenius norm shift ||C_t - C_t^{MP}||_F
"""
from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Union
import numpy as np
import pandas as pd

from esoteric_math.features.base import CausalBaseTransformer
from esoteric_math.schema_bridge import SchemaBundle


def analyze_marchenko_pastur_window(window_matrix: np.ndarray) -> Dict[str, Any]:
    """
    Given a (W, N) real sample matrix, compute sample correlation spectrum,
    Marchenko-Pastur bulk bounds, Tracy-Widom scaled edge gap, and denoised correlation matrix.
    """
    X_win = np.atleast_2d(np.asarray(window_matrix, dtype=float))
    W_eff, N = X_win.shape
    if W_eff < 3 or N < 2:
        q = float(N) / float(max(W_eff, 1))
        lam_plus = (1.0 + np.sqrt(q)) ** 2
        return {
            "lambda_plus": float(lam_plus),
            "lambda_minus": float((1.0 - np.sqrt(q)) ** 2),
            "lambda_max": 1.0,
            "escaping_count": 0.0,
            "sync_ratio": 1.0 / float(max(N, 1)),
            "tracy_widom_gap": float((1.0 - lam_plus) * (max(N, 1) ** (2.0 / 3.0))),
            "denoised_frob_shift": 0.0,
            "spectral_cond_log": 0.0,
            "eigenvalues": np.ones(N, dtype=float),
        }

    centered = X_win - np.mean(X_win, axis=0, keepdims=True)
    stds = np.std(centered, axis=0, ddof=0)
    valid = stds > 1e-10
    Z = np.zeros_like(centered)
    if np.any(valid):
        Z[:, valid] = centered[:, valid] / stds[valid]

    corr = (Z.T @ Z) / float(W_eff)
    # Constant channels behave as uncorrelated unit-variance noise on diagonal
    for j in range(N):
        if not valid[j]:
            corr[j, :] = 0.0
            corr[:, j] = 0.0
        corr[j, j] = 1.0
    corr = 0.5 * (corr + corr.T)

    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 0.0)

    q = float(N) / float(W_eff)
    lam_plus = float((1.0 + np.sqrt(q)) ** 2)
    lam_minus = float((1.0 - np.sqrt(q)) ** 2)
    lam_max = float(eigvals[-1])

    escaping_mask = eigvals > lam_plus
    escaping_count = float(np.sum(escaping_mask))
    sync_ratio = float(lam_max / float(N))
    tw_gap = float((lam_max - lam_plus) * (float(N) ** (2.0 / 3.0)))

    # Laloux-Cizeau-Bouchaud-Potters spectral denoising: replace bulk <= lam_plus by bulk mean
    eigvals_clean = eigvals.copy()
    bulk_mask = ~escaping_mask
    if np.any(bulk_mask):
        bulk_mean = float(np.mean(eigvals[bulk_mask]))
        eigvals_clean[bulk_mask] = bulk_mean
    else:
        bulk_mean = float(np.mean(eigvals))

    corr_clean = (eigvecs * eigvals_clean[None, :]) @ eigvecs.T
    diag_clean = np.sqrt(np.maximum(np.diag(corr_clean), 1e-12))
    corr_clean = corr_clean / np.outer(diag_clean, diag_clean)
    np.fill_diagonal(corr_clean, 1.0)

    frob_shift = float(np.linalg.norm(corr - corr_clean, ord="fro"))
    cond_log = float(np.log(max(lam_max, 1e-8) / max(bulk_mean, 1e-6)))

    return {
        "lambda_plus": lam_plus,
        "lambda_minus": lam_minus,
        "lambda_max": lam_max,
        "escaping_count": escaping_count,
        "sync_ratio": sync_ratio,
        "tracy_widom_gap": tw_gap,
        "denoised_frob_shift": frob_shift,
        "spectral_cond_log": cond_log,
        "eigenvalues": eigvals,
    }


class RMTMarchenkoPasturTransformer(CausalBaseTransformer):
    """
    Module 11 (`f11_rmt_marchenko.py`):
    Rolling causal Marchenko-Pastur spectral denoising, escaping eigenvalue counts,
    systemic synchronization ratio, and Tracy-Widom edge gap.
    """

    module_id = "f11"
    module_name = "Random Matrix Theory (Marchenko-Pastur & Tracy-Widom)"

    def __init__(
        self,
        window: int = 24,
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

    @staticmethod
    def _assemble_channel_panel(bundle: SchemaBundle) -> np.ndarray:
        parts = [bundle.tier_b.covariate_matrix]
        if bundle.tier_b.category_counts.shape[1] > 1:
            parts.append(bundle.tier_b.category_counts)
        if bundle.tier_c.node_scalar_stress.shape[1] > 1:
            parts.append(bundle.tier_c.node_scalar_stress)
        panel = np.hstack(parts)
        # Cap channel count for crisp aspect ratio q = N/W
        if panel.shape[1] > 18:
            panel = panel[:, :18]
        return panel

    def fit(self, X: Union[pd.DataFrame, SchemaBundle], y: Any = None) -> "RMTMarchenkoPasturTransformer":
        bundle = self._ensure_bundle_fit(X)
        panel = self._assemble_channel_panel(bundle)
        self.n_channels_ = int(panel.shape[1])
        return self

    def transform(self, X: Union[pd.DataFrame, SchemaBundle]) -> Union[pd.DataFrame, np.ndarray]:
        bundle = self._ensure_bundle_transform(X)
        panel = self._assemble_channel_panel(bundle)
        T = bundle.n_steps
        col_names = [
            "f11_mp_upper_edge",
            "f11_mp_escaping_count",
            "f11_sync_ratio",
            "f11_tracy_widom_gap",
            "f11_mp_denoised_frob",
            "f11_spectral_cond_log",
        ]
        feats = np.zeros((T, len(col_names)), dtype=float)

        for t in range(T):
            start = max(0, t - self.window + 1)
            win = panel[start : t + 1]
            spec = analyze_marchenko_pastur_window(win)
            feats[t, 0] = spec["lambda_plus"]
            feats[t, 1] = spec["escaping_count"]
            feats[t, 2] = spec["sync_ratio"]
            feats[t, 3] = spec["tracy_widom_gap"]
            feats[t, 4] = spec["denoised_frob_shift"]
            feats[t, 5] = spec["spectral_cond_log"]

        self.last_diagnostics_ = {
            "mp_upper_edge": feats[:, 0].copy(),
            "escaping_count": feats[:, 1].copy(),
            "sync_ratio": feats[:, 2].copy(),
            "tracy_widom_gap": feats[:, 3].copy(),
        }
        return self._format_output(feats, col_names, bundle, X)
