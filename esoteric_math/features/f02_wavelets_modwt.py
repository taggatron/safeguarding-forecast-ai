"""
Module 02: Boundary-Causal MODWT Wavelets & Morlet Scalogram (`esoteric_math/features/f02_wavelets_modwt.py`).

Implements strictly backward-looking (causal) Maximal Overlap Discrete Wavelet Transform (MODWT)
pyramid filtering across scales j = 1..4 using:
  - Haar wavelet filters (capturing sharp half-term & return step cliffs)
  - Daubechies extremal-phase `db4` (8-tap orthogonal D8 / sqrt(2) filters)
  - Causal one-sided complex Morlet scalogram energy across dyadic scales
"""
from __future__ import annotations

from typing import Any, List, Literal, Optional, Tuple, Union
import numpy as np
import pandas as pd

from esoteric_math.features.base import CausalBaseTransformer
from esoteric_math.schema_bridge import SchemaBundle, _causal_ewma_1d

# Exact Daubechies D8 (`db4`) scaling filter coefficients (unit L2 norm), divided by sqrt(2) for MODWT
_DB4_DWT_SCALING = np.array(
    [
        -0.010597401784997278,
        0.032883011666982945,
        0.030841381835986965,
        -0.18703481171888114,
        -0.02798376941698385,
        0.6308807679295904,
        0.7148465705525415,
        0.23037781330885523,
    ],
    dtype=float,
)
_DB4_MODWT_G = _DB4_DWT_SCALING / np.sqrt(2.0)
# Quadrature mirror wavelet filter h_l = (-1)^l * g_{L - 1 - l}
_DB4_MODWT_H = np.array(
    [((-1.0) ** l_idx) * _DB4_MODWT_G[len(_DB4_MODWT_G) - 1 - l_idx] for l_idx in range(len(_DB4_MODWT_G))],
    dtype=float,
)

_HAAR_MODWT_G = np.array([0.5, 0.5], dtype=float)
_HAAR_MODWT_H = np.array([0.5, -0.5], dtype=float)


def _causal_modwt_pyramid(
    signal: np.ndarray,
    h_filt: np.ndarray,
    g_filt: np.ndarray,
    max_level: int = 4,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Strictly backward-looking (causal) MODWT pyramid decomposition.
    At level j (1..J), filter taps sit at backward indices t - 2^(j-1) * l for l = 0..L-1,
    clamping negative indices to index 0 (initial causal boundary condition).
    Modifying signal[s] for s > t_0 has ZERO effect on W_j[t] or V_j[t] for all t <= t_0.
    """
    n = len(signal)
    L = len(h_filt)
    v_prev = signal.astype(float).copy()
    details: List[np.ndarray] = []

    for j in range(1, max_level + 1):
        step = 2 ** (j - 1)
        w_j = np.zeros(n, dtype=float)
        v_j = np.zeros(n, dtype=float)
        idx_base = np.arange(n, dtype=int)
        for l_idx in range(L):
            lagged_idx = np.maximum(0, idx_base - l_idx * step)
            vals = v_prev[lagged_idx]
            w_j += h_filt[l_idx] * vals
            v_j += g_filt[l_idx] * vals
        details.append(w_j)
        v_prev = v_j

    return details, v_prev


def _causal_morlet_energy(signal: np.ndarray, scale: float, omega0: float = 5.0) -> np.ndarray:
    """
    Causal one-sided complex Morlet wavelet energy |(x * psi_s)(t)|^2 reading strictly lags tau >= 0.
    """
    n = len(signal)
    half_len = max(4, int(np.ceil(3.0 * scale)))
    taus = np.arange(0, half_len + 1, dtype=float) / float(scale)
    # Complex Morlet mother wavelet evaluated on causal support tau >= 0, zero-mean corrected
    env = np.exp(-0.5 * (taus**2))
    carrier = np.exp(-1j * omega0 * taus)
    psi = env * (carrier - np.sum(env * carrier) / max(1e-12, np.sum(env)))
    norm = np.linalg.norm(psi)
    if norm > 1e-12:
        psi = psi / norm

    energy = np.zeros(n, dtype=float)
    idx_base = np.arange(n, dtype=int)
    conv_complex = np.zeros(n, dtype=np.complex128)
    for k_lag in range(len(psi)):
        lag_idx = np.maximum(0, idx_base - k_lag)
        conv_complex += psi[k_lag] * signal[lag_idx]
    energy = np.abs(conv_complex) ** 2
    return np.log1p(energy)


class CausalMODWTWaveletTransformer(CausalBaseTransformer):
    """
    Module 02: Boundary-Causal MODWT (Haar & Daubechies db4 scales 1..4) + Morlet Scalogram Energy.
    """

    module_id = "f02"
    module_name = "Boundary-Causal MODWT & Morlet Wavelets"

    def __init__(
        self,
        max_level: int = 4,
        morlet_scales: Tuple[float, ...] = (2.0, 4.0, 8.0, 16.0),
        causal_mode: Literal["telemetry", "forecast"] = "telemetry",
        train_cutoff: Optional[int] = None,
        return_dataframe: bool = True,
    ) -> None:
        super().__init__(
            causal_mode=causal_mode,
            train_cutoff=train_cutoff,
            return_dataframe=return_dataframe,
        )
        self.max_level = max_level
        self.morlet_scales = morlet_scales

    def fit(self, X: Union[pd.DataFrame, SchemaBundle], y: Any = None) -> "CausalMODWTWaveletTransformer":
        bundle = self._ensure_bundle_fit(X)
        self.train_mean_ = float(np.mean(bundle.tier_a.y_rate_per_1000))
        self.is_fitted_ = True
        return self

    def transform(self, X: Union[pd.DataFrame, SchemaBundle]) -> Union[pd.DataFrame, np.ndarray]:
        if not getattr(self, "is_fitted_", False):
            self.fit(X)
        bundle = self._ensure_bundle_transform(X)
        sig = bundle.tier_a.y_rate_per_1000.astype(float) - self.train_mean_

        haar_w, haar_v = _causal_modwt_pyramid(sig, _HAAR_MODWT_H, _HAAR_MODWT_G, max_level=self.max_level)
        db4_w, db4_v = _causal_modwt_pyramid(sig, _DB4_MODWT_H, _DB4_MODWT_G, max_level=self.max_level)

        cols: List[np.ndarray] = []
        names: List[str] = []

        for j_idx, w_j in enumerate(haar_w, start=1):
            cols.append(w_j)
            cols.append(_causal_ewma_1d(w_j**2, alpha=0.25))
            names.extend([f"f02_haar_w{j_idx}", f"f02_haar_energy_j{j_idx}"])
        cols.append(haar_v)
        names.append(f"f02_haar_smooth_v{self.max_level}")

        for j_idx, w_j in enumerate(db4_w, start=1):
            cols.append(w_j)
            cols.append(_causal_ewma_1d(w_j**2, alpha=0.25))
            names.extend([f"f02_db4_w{j_idx}", f"f02_db4_energy_j{j_idx}"])
        cols.append(db4_v)
        names.append(f"f02_db4_smooth_v{self.max_level}")

        morlet_list: List[np.ndarray] = []
        for sc in self.morlet_scales:
            m_e = _causal_morlet_energy(sig, scale=float(sc))
            morlet_list.append(m_e)
            cols.append(m_e)
            names.append(f"f02_morlet_log_energy_s{int(sc)}")

        if len(morlet_list) >= 2:
            burst_ratio = morlet_list[0] / np.maximum(0.05, morlet_list[-1] + 0.05)
            cols.append(burst_ratio)
            names.append("f02_morlet_high_low_burst_ratio")

        self.last_diagnostics_ = {
            "haar_details": haar_w,
            "db4_details": db4_w,
            "morlet_log_energies": morlet_list,
        }
        return self._format_output(np.column_stack(cols), names, bundle, X)
