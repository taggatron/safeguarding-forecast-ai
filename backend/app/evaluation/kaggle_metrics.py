"""
Phase 4: Kaggle Competition Metric Suite & Operational Safeguarding Evaluation
(DengAI Count Accuracy, Recruit RMSLE, M5 RMSSE/WRMSSE, and M5 Uncertainty Pinball/WSPL/PICP).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import numpy as np
from sklearn.metrics import mean_pinball_loss, mean_tweedie_deviance


EVAL_QUANTILES: Tuple[float, ...] = (0.025, 0.50, 0.90, 0.975)


@dataclass
class ModelEvaluationMetrics:
    model_id: str
    model_name: str
    mae: float
    rmse: float
    surge_mae: float
    nb_deviance: float
    tweedie_deviance_p13: float
    rmsle: float
    rmsse: float
    wrmsse: float
    pinball_025: float
    pinball_500: float
    pinball_900: float
    pinball_975: float
    spl: float
    wspl: float
    picp_95: float
    miw_95: float
    aic: Optional[float] = None
    bic: Optional[float] = None
    alpha_dispersion: Optional[float] = None
    notes: str = ""


def compute_mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(yt - yp)))


def compute_rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def compute_surge_mae(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    is_autumn_transition: Sequence[int],
    is_pre_holiday_week: Sequence[int],
    is_post_holiday_week: Sequence[int],
) -> float:
    """
    DengAI Benchmark: Isolated MAE calculated strictly across Autumn Transition weeks (Weeks 1-6)
    and Pre-/Post-Holiday boundary weeks.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = (
        (np.asarray(is_autumn_transition, dtype=int) > 0)
        | (np.asarray(is_pre_holiday_week, dtype=int) > 0)
        | (np.asarray(is_post_holiday_week, dtype=int) > 0)
    )
    if not np.any(mask):
        return compute_mae(yt, yp)
    return float(np.mean(np.abs(yt[mask] - yp[mask])))


def compute_mean_nb_deviance(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    alpha: float = 0.18,
) -> float:
    """
    Mean Negative Binomial (NB2) unit deviance:
    d(y, mu) = 2 * [ y * ln(max(y, eps)/mu) - (y + 1/alpha) * ln((1 + alpha*y)/(1 + alpha*mu)) ]
    """
    yt = np.maximum(0.0, np.asarray(y_true, dtype=float))
    mu = np.maximum(1e-6, np.asarray(y_pred, dtype=float))
    a = max(1e-6, float(alpha))
    inv_a = 1.0 / a

    term1 = np.where(yt > 0, yt * np.log(np.maximum(yt, 1e-12) / mu), 0.0)
    term2 = (yt + inv_a) * np.log((1.0 + a * yt) / (1.0 + a * mu))
    dev = 2.0 * (term1 - term2)
    return float(np.mean(np.maximum(0.0, dev)))


def compute_mean_tweedie_deviance_safe(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    power: float = 1.3,
) -> float:
    yt = np.maximum(0.0, np.asarray(y_true, dtype=float))
    yp = np.maximum(1e-4, np.asarray(y_pred, dtype=float))
    return float(mean_tweedie_deviance(yt, yp, power=power))


def compute_rmsle(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """
    Recruit Forecasting Benchmark — Root Mean Squared Logarithmic Error (RMSLE):
    Uses np.log1p to safely handle zero-count weeks and penalizes under-prediction
    more heavily than over-prediction.
    """
    yt = np.maximum(0.0, np.asarray(y_true, dtype=float))
    yp = np.maximum(0.0, np.asarray(y_pred, dtype=float))
    log_diff = np.log1p(yp) - np.log1p(yt)
    return float(np.sqrt(np.mean(log_diff**2)))


def compute_rmsse(
    y_train: Sequence[float],
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> float:
    """
    M5 Accuracy Benchmark — Root Mean Squared Scaled Error (RMSSE):
    Scales out-of-time RMSE by the in-sample one-step naive random walk error:
      RMSSE_i = sqrt( (1/h * sum (Y_t - Y_hat_t)^2) / (1/(n-1) * sum_{t=2}^n (Y_t - Y_{t-1})^2) )
    Any score < 1.0 proves the model outperforms a naive week-to-week persistence forecast.
    """
    y_tr = np.asarray(y_train, dtype=float)
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    mse_oos = float(np.mean((yt - yp) ** 2))
    if len(y_tr) < 2:
        return float(np.sqrt(mse_oos))
    naive_diffs_sq = (y_tr[1:] - y_tr[:-1]) ** 2
    denom = float(np.mean(naive_diffs_sq))
    if denom <= 1e-12:
        return 0.0 if mse_oos <= 1e-12 else float(np.sqrt(mse_oos))
    return float(np.sqrt(mse_oos / denom))


def compute_wrmsse(
    series_rmsse: Mapping[str, float],
    series_weights: Mapping[str, float],
) -> float:
    """
    Weighted Root Mean Squared Scaled Error across curriculum areas weighted by active
    learner headcount share w_i = N_i / sum(N_j).
    """
    total_w = float(sum(series_weights.get(k, 0.0) for k in series_rmsse))
    if total_w <= 0:
        return float(np.mean(list(series_rmsse.values()))) if series_rmsse else 0.0
    return float(
        sum((series_weights.get(k, 0.0) / total_w) * val for k, val in series_rmsse.items())
    )


def compute_pinball_loss(
    y_true: Sequence[float],
    q_pred: Sequence[float],
    tau: float,
) -> float:
    """
    M5 Uncertainty Benchmark — Pinball (Quantile) Loss at quantile tau in (0, 1):
      L_tau(Y_t, Q_t) = max(tau * (Y_t - Q_t), (tau - 1) * (Y_t - Q_t))
    """
    yt = np.asarray(y_true, dtype=float)
    qp = np.asarray(q_pred, dtype=float)
    return float(mean_pinball_loss(yt, qp, alpha=float(tau)))


def compute_spl(
    y_train: Sequence[float],
    y_true: Sequence[float],
    quantile_preds: Mapping[float, Sequence[float]],
    quantiles: Sequence[float] = EVAL_QUANTILES,
) -> float:
    """
    M5 Uncertainty Benchmark — Scaled Pinball Loss (SPL) for a single series:
      SPL_i = (1 / (|T| * h)) * sum_{tau in T} sum_{t=n+1}^{n+h} L_tau(Y_t, Q_{tau,t}) / scale_i
    where scale_i = (1 / (n - 1)) * sum_{k=2}^n |Y_k - Y_{k-1}|.
    """
    y_tr = np.asarray(y_train, dtype=float)
    yt = np.asarray(y_true, dtype=float)
    if len(y_tr) >= 2:
        scale = float(np.mean(np.abs(y_tr[1:] - y_tr[:-1])))
    else:
        scale = 1.0
    scale = max(1e-8, scale)

    losses = [compute_pinball_loss(yt, quantile_preds[q], q) for q in quantiles if q in quantile_preds]
    if not losses:
        return 0.0
    return float(np.mean(losses) / scale)


def compute_wspl(
    series_spl: Mapping[str, float],
    series_weights: Mapping[str, float],
) -> float:
    """Weighted Scaled Pinball Loss across curriculum areas weighted by active learner share."""
    total_w = float(sum(series_weights.get(k, 0.0) for k in series_spl))
    if total_w <= 0:
        return float(np.mean(list(series_spl.values()))) if series_spl else 0.0
    return float(
        sum((series_weights.get(k, 0.0) / total_w) * val for k, val in series_spl.items())
    )


def compute_picp_and_miw(
    y_true: Sequence[float],
    q_lower_025: Sequence[float],
    q_upper_975: Sequence[float],
) -> Tuple[float, float]:
    """
    Compute 95% Prediction Interval Coverage Probability (PICP, in %) and
    Mean Interval Width (MIW) for [Q_0.025, Q_0.975].
    """
    yt = np.asarray(y_true, dtype=float)
    lo = np.asarray(q_lower_025, dtype=float)
    hi = np.asarray(q_upper_975, dtype=float)
    covered = (yt >= lo) & (yt <= hi)
    picp_pct = float(np.mean(covered) * 100.0)
    miw = float(np.mean(hi - lo))
    return picp_pct, miw

