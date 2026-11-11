"""
Module 13: Fractional Calculus & Sandpile Self-Organized Criticality (SOC)
(`esoteric_math/features/f13_fractional_soc.py`).

Implements:
  1. Strictly backward-causal Grunwald-Letnikov Fractional Differencing (1 - B)^{d^*} X_t
     across both calendar time and holiday-compressed instructional time (preserving
     multi-month pastoral stress memory across half-term and seasonal breaks).
  2. Bak-Tang-Wiesenfeld Sandpile Critical Slowing Down (CSD) diagnostics:
     - Rolling causal lag-1 autocorrelation rho_1(t)
     - Rolling causal Fano factor Var_W(X)_t / Mean_W(X)_t
     - Causal Hill power-law tail-index estimator xi_Hill(t)
"""
from __future__ import annotations

from typing import Any, List, Literal, Optional, Sequence, Union
import numpy as np
import pandas as pd

from esoteric_math.features.base import CausalBaseTransformer
from esoteric_math.schema_bridge import SchemaBundle


def grunwald_letnikov_weights(d: float, max_lags: int) -> np.ndarray:
    """
    Causal Grunwald-Letnikov binomial filter coefficients w_0, ..., w_{max_lags - 1}:
      w_0 = 1,  w_k = -w_{k-1} * (d - k + 1) / k.
    """
    K = max(1, int(max_lags))
    w = np.empty(K, dtype=float)
    w[0] = 1.0
    for k in range(1, K):
        w[k] = -w[k - 1] * (float(d) - float(k) + 1.0) / float(k)
    return w


def causal_fracdiff_1d(series: np.ndarray, d: float, max_lags: int = 36) -> np.ndarray:
    """Apply causal truncated Grunwald-Letnikov operator (1 - B)^d to 1D series."""
    x = np.asarray(series, dtype=float)
    T = len(x)
    w = grunwald_letnikov_weights(d, min(max_lags, max(T, 1)))
    K = len(w)
    out = np.empty(T, dtype=float)
    for t in range(T):
        k_eff = min(t + 1, K)
        # x[t], x[t-1], ..., x[t - k_eff + 1]
        seg = x[t - k_eff + 1 : t + 1][::-1]
        out[t] = float(np.dot(w[:k_eff], seg))
    return out


def _dickey_fuller_tstat(series: np.ndarray) -> float:
    """Simple OLS Dickey-Fuller t-statistic on Delta y_t = alpha + beta * y_{t-1} + eps_t."""
    y = np.asarray(series, dtype=float)
    if len(y) < 12 or np.std(y) < 1e-10:
        return -5.0
    y_lag = y[:-1]
    dy = np.diff(y)
    X_reg = np.column_stack([np.ones_like(y_lag), y_lag])
    beta, _, _, _ = np.linalg.lstsq(X_reg, dy, rcond=None)
    resid = dy - X_reg @ beta
    dof = max(len(dy) - 2, 1)
    s2 = float(np.dot(resid, resid)) / float(dof)
    xtx_inv = np.linalg.pinv(X_reg.T @ X_reg)
    se_beta1 = float(np.sqrt(max(s2 * xtx_inv[1, 1], 1e-12)))
    return float(beta[1] / se_beta1)


def _causal_hill_estimator(values: np.ndarray) -> float:
    """Causal Hill tail-index estimator over upper order statistics of positive shifted counts."""
    v = np.sort(np.maximum(np.asarray(values, dtype=float), 0.0) + 1.0)
    n = len(v)
    if n < 6:
        return 0.0
    k = max(2, min(int(np.ceil(0.25 * n)), n - 1))
    threshold = v[n - k - 1]
    if threshold <= 0.0:
        return 0.0
    top_k = v[n - k :]
    return float(np.mean(np.log(np.maximum(top_k / threshold, 1.0))))


class FractionalSOCTransformer(CausalBaseTransformer):
    """
    Module 13 (`f13_fractional_soc.py`):
    Grunwald-Letnikov Fractional Differencing (calendar + holiday-warped instructional memory)
    and Bak-Tang-Wiesenfeld Sandpile Critical Slowing Down indicators.
    """

    module_id = "f13"
    module_name = "Fractional Calculus & Sandpile SOC"

    def __init__(
        self,
        candidate_d: Sequence[float] = (0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85),
        max_lags: int = 36,
        csd_window: int = 16,
        causal_mode: Literal["telemetry", "forecast"] = "telemetry",
        train_cutoff: Optional[int] = None,
        return_dataframe: bool = True,
    ) -> None:
        super().__init__(
            causal_mode=causal_mode,
            train_cutoff=train_cutoff,
            return_dataframe=return_dataframe,
        )
        self.candidate_d = tuple(candidate_d)
        self.max_lags = max_lags
        self.csd_window = csd_window

    def fit(self, X: Union[pd.DataFrame, SchemaBundle], y: Any = None) -> "FractionalSOCTransformer":
        bundle = self._ensure_bundle_fit(X)
        y_rate = bundle.tier_a.y_rate_per_1000

        chosen_d = 0.40
        for d in self.candidate_d:
            fd = causal_fracdiff_1d(y_rate, d=float(d), max_lags=self.max_lags)
            t_stat = _dickey_fuller_tstat(fd[min(6, len(fd) // 4) :])
            if t_stat <= -2.86:
                chosen_d = float(d)
                break
        self.optimal_d_ = chosen_d

        # Baseline training statistics for standardized SOC criticality score
        ac1_train, fano_train, hill_train = self._compute_csd_channels(
            bundle.tier_a.y_count, bundle.tier_a.is_instructional
        )
        self.ac1_med_ = float(np.median(ac1_train))
        self.ac1_scale_ = float(max(np.std(ac1_train), 0.10))
        self.fano_med_ = float(np.median(fano_train))
        self.fano_scale_ = float(max(np.std(fano_train), 0.25))
        self.hill_med_ = float(np.median(hill_train))
        self.hill_scale_ = float(max(np.std(hill_train), 0.05))
        return self

    def _compute_csd_channels(
        self,
        y_count: np.ndarray,
        is_instructional: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        T = len(y_count)
        ac1 = np.zeros(T, dtype=float)
        fano = np.ones(T, dtype=float)
        hill = np.zeros(T, dtype=float)

        active_hist: List[float] = []
        for t in range(T):
            if is_instructional[t] > 0.20 or len(active_hist) == 0:
                active_hist.append(float(max(y_count[t], 0.0)))

            win = np.asarray(active_hist[-self.csd_window :], dtype=float)
            if len(win) >= 4:
                x0, x1 = win[:-1], win[1:]
                s0, s1 = float(np.std(x0)), float(np.std(x1))
                if s0 > 1e-9 and s1 > 1e-9:
                    ac1[t] = float(np.clip(np.corrcoef(x0, x1)[0, 1], -0.99, 0.99))
                mean_w = float(np.mean(win))
                var_w = float(np.var(win, ddof=1))
                fano[t] = float(var_w / max(mean_w, 0.25))

            full_prefix = np.asarray(active_hist[-max(self.csd_window * 3, 24) :], dtype=float)
            hill[t] = _causal_hill_estimator(full_prefix)

        return ac1, fano, hill

    def transform(self, X: Union[pd.DataFrame, SchemaBundle]) -> Union[pd.DataFrame, np.ndarray]:
        bundle = self._ensure_bundle_transform(X)
        T = bundle.n_steps
        y_rate = bundle.tier_a.y_rate_per_1000
        y_count = bundle.tier_a.y_count
        is_inst = bundle.tier_a.is_instructional

        # 1. Calendar-time causal fractional differencing at d* and reference d=0.40
        fd_opt = causal_fracdiff_1d(y_rate, d=self.optimal_d_, max_lags=self.max_lags)
        fd_ref = causal_fracdiff_1d(y_rate, d=0.40, max_lags=self.max_lags)

        # 2. Holiday-compressed instructional-time fractional differencing (preserves memory across breaks)
        fd_inst = np.zeros(T, dtype=float)
        w_inst = grunwald_letnikov_weights(self.optimal_d_, self.max_lags)
        inst_buffer: List[float] = []
        last_fd = 0.0
        for t in range(T):
            if is_inst[t] > 0.20 or len(inst_buffer) == 0:
                inst_buffer.append(float(y_rate[t]))
                k_eff = min(len(inst_buffer), len(w_inst))
                seg = np.asarray(inst_buffer[-k_eff:][::-1], dtype=float)
                last_fd = float(np.dot(w_inst[:k_eff], seg))
            fd_inst[t] = last_fd

        # 3. Sandpile Self-Organized Criticality (Critical Slowing Down)
        ac1, fano, hill = self._compute_csd_channels(y_count, is_inst)
        z_ac1 = (ac1 - self.ac1_med_) / self.ac1_scale_
        z_fano = (fano - self.fano_med_) / self.fano_scale_
        z_hill = (hill - self.hill_med_) / self.hill_scale_
        soc_score = (z_ac1 + z_fano + z_hill) / np.sqrt(3.0)

        col_names = [
            "f13_fracdiff_optimal",
            "f13_fracdiff_d040",
            "f13_fracdiff_instructional",
            "f13_csd_ac1",
            "f13_sandpile_fano_factor",
            "f13_hill_tail_index",
            "f13_soc_criticality_score",
        ]
        feats = np.column_stack([fd_opt, fd_ref, fd_inst, ac1, fano, hill, soc_score])

        self.last_diagnostics_ = {
            "optimal_d": float(self.optimal_d_),
            "fracdiff_instructional": fd_inst,
            "csd_ac1": ac1,
            "fano_factor": fano,
            "hill_tail_index": hill,
            "soc_criticality_score": soc_score,
        }
        return self._format_output(feats, col_names, bundle, X)
