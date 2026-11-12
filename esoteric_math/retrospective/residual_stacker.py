"""
Post-Hoc Residual Offset / Init-Score Stacking & Anti-Bloat Quarantine Gate
(`esoteric_math/retrospective/residual_stacker.py`).

Layers esoteric mathematical residual adjustments onto 100% immutable baseline models:
  1. `BlockBorutaSHAPGate`: 25 circular 4-period block-shuffled shadow trials using LightGBM
     native `pred_contrib=True` with `init_score = log(max(mu_base, 1e-6))`, retaining only
     esoteric columns beating the 95th-percentile shadow SHAP threshold in >= 18/25 trials.
  2. `GreedyOneSEModuleSelector`: K_inner = 3 chronological expanding sub-splits admitting
     module m iff mean residual deviance improvement satisfies d_bar_m >= max(1e-6, SE(d_m)).
  3. `GLMResidualOffsetHead`: Penalized NB2/Poisson log-offset head capped at
     p_max = min(6, floor(N_eff / 20)) with VIF <= 5.0 and ZINB Structural Lock (pi_base untouched).
  4. `GBDTInitScoreResidualHead`: Shallow residual LightGBM booster with strict small-sample caps
     (`max_depth=2` for N_eff < 300, `num_leaves=4`, `min_child_samples=max(12, int(0.08*N_eff))`,
     `colsample_bytree=0.5`, `reg_alpha=2.0`, `reg_lambda=10.0`, `learning_rate=0.02`, patience=25).
  5. `FisherRaoResidualCalibrator`: Shallow (max_depth=2) Fisher-Rao natural-gradient dispersion
     calibrator sharpening interval coverage during contagion bursts without shifting calm weeks.
  6. Exact Zero-Harm Identity Floor: Returns frozen baseline point, quantile, and interval
     forecasts verbatim (`atol=1e-12`) whenever `TELEMETRY_ONLY` is active or 0 modules pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import warnings
import lightgbm as lgb
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.tree import DecisionTreeRegressor

from app.evaluation.kaggle_metrics import EVAL_QUANTILES, compute_mean_nb_deviance
from app.models.safeguarding_benchmark import (
    ModelPredictionBundle,
    _nb2_quantiles_from_mean_and_alpha,
    _zinb_quantiles_from_params,
)
from esoteric_math.features.f12_fisher_rao import nb_fisher_information_vectorized
from esoteric_math.retrospective.frozen_adapter import (
    FrozenBaselineAdapter,
    clone_prediction_bundle,
)
from esoteric_math.retrospective.rollout_config import (
    ALL_13_MODULES,
    MODULE_DISPLAY_NAMES,
    RolloutConfig,
    RolloutMode,
)
from esoteric_math.schema_bridge import DATE_CANDIDATES


def compute_effective_temporal_sample_size(df: pd.DataFrame) -> int:
    """
    Compute N_eff as the count of unique active timestamps in `df`.
    Excludes non-instructional closure days when `is_instructional_day` is present.
    """
    if len(df) == 0:
        return 0
    work = df
    if "is_instructional_day" in df.columns:
        active_mask = pd.to_numeric(df["is_instructional_day"], errors="coerce").fillna(1.0) > 0.15
        if int(active_mask.sum()) >= 5:
            work = df.loc[active_mask]

    cols_lower = {c.lower(): c for c in work.columns}
    for cand in DATE_CANDIDATES:
        if cand in cols_lower:
            col = cols_lower[cand]
            return int(pd.to_datetime(work[col]).nunique())
    return int(len(work))


def _extract_module_prefix(col_name: str) -> str:
    prefix = col_name.split("_", 1)[0].lower()
    return prefix if prefix in ALL_13_MODULES else "other"


def _circular_block_permute_1d(
    arr: np.ndarray,
    block_size: int,
    rng: np.random.Generator,
    phase_shift: int,
) -> np.ndarray:
    """Permute a 1D series in contiguous `block_size` chunks after a circular phase shift."""
    n = len(arr)
    if n <= block_size:
        return rng.permutation(arr)
    shifted = np.roll(arr, shift=phase_shift % block_size)
    blocks = [shifted[i : i + block_size] for i in range(0, n, block_size)]
    perm_order = rng.permutation(len(blocks))
    shuffled = np.concatenate([blocks[idx] for idx in perm_order])[:n]
    return np.roll(shuffled, shift=-(phase_shift % block_size))


def _winsorize_and_standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    X_clean = np.where(np.isfinite(X), X, med[None, :])
    mad = np.median(np.abs(X_clean - med[None, :]), axis=0)
    scale_mad = np.where(mad > 1e-8, 1.4826 * mad, np.std(X_clean, axis=0))
    scale_mad = np.where(scale_mad > 1e-8, scale_mad, 1.0)
    lo = med - 4.5 * scale_mad
    hi = med + 4.5 * scale_mad
    X_win = np.clip(X_clean, lo[None, :], hi[None, :])
    mean = np.mean(X_win, axis=0)
    std = np.std(X_win, axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    Z = (X_win - mean[None, :]) / std[None, :]
    return Z, lo, hi, mean, std


def _winsorize_and_standardize_apply(
    X: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    X_clean = np.where(np.isfinite(X), X, mean[None, :])
    X_win = np.clip(X_clean, lo[None, :], hi[None, :])
    return (X_win - mean[None, :]) / std[None, :]


def _compute_max_vif_and_worst(Z: np.ndarray) -> Tuple[float, int]:
    """Compute Variance Inflation Factors on standardized matrix Z via R^2_j = 1 - 1 / (C^{-1})_{jj}."""
    p = Z.shape[1]
    if p <= 1:
        return 1.0, 0
    corr = np.corrcoef(Z, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    inv_corr = np.linalg.pinv(corr + 1e-6 * np.eye(p))
    vifs = np.maximum(1.0, np.diag(inv_corr))
    worst_idx = int(np.argmax(vifs))
    return float(vifs[worst_idx]), worst_idx


@dataclass
class ResidualGateAudit:
    """Structured diagnostic record for the Residual Anti-Bloat & 1-SE Gate."""

    retro_mode: str
    n_eff: int
    eligible_modules: List[str]
    boruta_surviving_columns: List[str]
    boruta_hit_counts: Dict[str, int]
    admitted_modules: List[str]
    quarantined_modules: List[str]
    module_1se_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    selected_features: List[str] = field(default_factory=list)
    max_vif: float = 1.0
    identity_floor_active: bool = True


class BlockBorutaSHAPGate:
    """
    4-Period Circular Block-Permuted BorutaSHAP Gate against frozen baseline log-residuals.
    Uses LightGBM native `pred_contrib=True` with `init_score = log(max(mu_base, 1e-6))`.
    Retains only columns beating the 95th-percentile shadow SHAP threshold in >= 18/25 trials.
    """

    def __init__(
        self,
        n_trials: int = 25,
        min_hits: int = 18,
        block_size: int = 4,
        alpha: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self.n_trials = n_trials
        self.min_hits = min_hits
        self.block_size = block_size
        self.alpha = alpha
        self.random_state = random_state
        self.selected_columns_: List[str] = []
        self.hit_counts_: Dict[str, int] = {}

    def fit(
        self,
        X_eso: pd.DataFrame,
        y: np.ndarray,
        mu_base: np.ndarray,
        n_eff: int,
    ) -> "BlockBorutaSHAPGate":
        if X_eso.shape[1] == 0 or len(y) < 20 or n_eff < 20:
            self.selected_columns_ = []
            self.hit_counts_ = {c: 0 for c in X_eso.columns}
            return self

        # Drop near-constant columns up front
        raw_mat = X_eso.to_numpy(dtype=float)
        stds = np.nanstd(raw_mat, axis=0)
        valid_mask = np.isfinite(stds) & (stds > 1e-7)
        if not np.any(valid_mask):
            self.selected_columns_ = []
            self.hit_counts_ = {c: 0 for c in X_eso.columns}
            return self

        cand_cols = [c for c, ok in zip(X_eso.columns, valid_mask) if ok]
        Z_real, _, _, _, _ = _winsorize_and_standardize_fit(raw_mat[:, valid_mask])
        y_clean = np.maximum(0.0, np.asarray(y, dtype=float))
        init_log_mu = np.log(np.clip(np.asarray(mu_base, dtype=float), 1e-6, None))

        n_rows, p = Z_real.shape
        hits = np.zeros(p, dtype=int)
        rng = np.random.default_rng(self.random_state)
        min_child = max(8, min(16, n_rows // 7))

        eff_block = max(self.block_size, min(8, max(4, n_rows // 12)))
        n_shadow_copies = 2 if p < 16 else 1

        for trial_idx in range(self.n_trials):
            shadow_blocks = []
            for copy_idx in range(n_shadow_copies):
                Z_sh = np.empty_like(Z_real)
                for j in range(p):
                    Z_sh[:, j] = _circular_block_permute_1d(
                        Z_real[:, j],
                        block_size=eff_block,
                        rng=rng,
                        phase_shift=trial_idx * 3 + copy_idx * 7 + j,
                    )
                shadow_blocks.append(Z_sh)
            Z_shadow = np.hstack(shadow_blocks)
            X_aug = np.hstack([Z_real, Z_shadow])
            reg = LGBMRegressor(
                objective="poisson",
                boost_from_average=False,
                n_estimators=28,
                learning_rate=0.05,
                max_depth=2,
                num_leaves=4,
                min_child_samples=min_child,
                colsample_bytree=0.7,
                subsample=0.85,
                subsample_freq=1,
                reg_alpha=1.0,
                reg_lambda=5.0,
                random_state=self.random_state + trial_idx,
                n_jobs=1,
                verbose=-1,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reg.fit(X_aug, y_clean, init_score=init_log_mu)
                shap_contrib = np.asarray(
                    reg.booster_.predict(X_aug, pred_contrib=True),
                    dtype=float,
                )
            # Last column of pred_contrib is expected value bias; first p columns are real, next are shadow
            mean_abs_shap = np.mean(np.abs(shap_contrib[:, :-1]), axis=0)
            real_shap = mean_abs_shap[:p]
            shadow_shap = mean_abs_shap[p:]
            # Strict Boruta maximum-shadow threshold with 5% finite-sample margin
            q_thresh = max(1e-4, float(np.max(shadow_shap)) * 1.05)
            hits += (real_shap > q_thresh).astype(int)

        self.hit_counts_ = {c: 0 for c in X_eso.columns}
        for col_name, h_cnt in zip(cand_cols, hits):
            self.hit_counts_[col_name] = int(h_cnt)

        self.selected_columns_ = [
            col_name for col_name, h_cnt in zip(cand_cols, hits) if int(h_cnt) >= self.min_hits
        ]
        return self


def _fit_fast_ridge_poisson_offset(
    Z_tr: np.ndarray,
    y_tr: np.ndarray,
    log_mu_tr: np.ndarray,
    l2_pen: float = 8.0,
) -> np.ndarray:
    """Fast Newton-bound L-BFGS-B solver for penalized Poisson residual offset delta_eta = Z @ w + b."""
    p = Z_tr.shape[1]
    theta0 = np.zeros(p + 1, dtype=float)

    def _loss_and_grad(theta: np.ndarray) -> Tuple[float, np.ndarray]:
        b = theta[0]
        w = theta[1:]
        d_eta = np.clip(b + Z_tr @ w, -0.65, 0.65)
        mu_hat = np.exp(np.clip(log_mu_tr + d_eta, -14.0, 14.0))
        n = max(1, len(y_tr))
        nll = float(np.sum(mu_hat - y_tr * (log_mu_tr + d_eta))) / n
        reg = 0.5 * l2_pen * float(np.sum(w**2) + 2.5 * (b**2)) / n
        diff = (mu_hat - y_tr) / n
        grad_b = float(np.sum(diff)) + (2.5 * l2_pen * b) / n
        grad_w = (Z_tr.T @ diff) + (l2_pen * w) / n
        return nll + reg, np.concatenate([[grad_b], grad_w])

    res = minimize(
        _loss_and_grad,
        theta0,
        jac=True,
        method="L-BFGS-B",
        bounds=[(-0.35, 0.35)] + [(-0.45, 0.45)] * p,
        options={"maxiter": 45, "ftol": 1e-7},
    )
    return np.asarray(res.x, dtype=float)


class GreedyOneSEModuleSelector:
    """
    Greedy 1-SE Forward Module Selector over K_inner = 3 chronological expanding sub-splits.
    Admits module m iff mean inner NB2 deviance reduction satisfies:
        mean(d_m) >= max(1e-6, SE(d_m))
    """

    def __init__(
        self,
        n_inner_splits: int = 3,
        alpha_disp: float = 0.18,
        max_modules: int = 4,
    ) -> None:
        self.n_inner_splits = n_inner_splits
        self.alpha_disp = alpha_disp
        self.max_modules = max_modules
        self.admitted_modules_: List[str] = []
        self.module_diagnostics_: Dict[str, Dict[str, float]] = {}

    def fit(
        self,
        X_surviving: pd.DataFrame,
        y: np.ndarray,
        mu_base: np.ndarray,
        eligible_modules: Sequence[str],
    ) -> "GreedyOneSEModuleSelector":
        self.admitted_modules_ = []
        self.module_diagnostics_ = {}

        if X_surviving.shape[1] == 0 or len(y) < 30:
            return self

        # Group surviving columns by esoteric module ID
        mod_to_cols: Dict[str, List[str]] = {}
        for col in X_surviving.columns:
            m_id = _extract_module_prefix(col)
            if m_id in eligible_modules:
                mod_to_cols.setdefault(m_id, []).append(col)

        if not mod_to_cols:
            return self

        n = len(y)
        y_arr = np.maximum(0.0, np.asarray(y, dtype=float))
        mu_arr = np.maximum(1e-6, np.asarray(mu_base, dtype=float))
        log_mu = np.log(mu_arr)

        # Build K_inner = 3 chronological expanding sub-splits
        fracs = np.linspace(0.52, 0.84, self.n_inner_splits)
        val_len = max(5, int(round(0.15 * n)))
        splits: List[Tuple[int, int]] = []
        for f in fracs:
            tr_end = max(16, min(n - 4, int(round(f * n))))
            v_end = min(n, tr_end + val_len)
            if v_end - tr_end >= 4:
                splits.append((tr_end, v_end))

        if len(splits) < 2:
            return self

        def _eval_col_subset_losses(cols_sub: List[str]) -> np.ndarray:
            losses = np.zeros(len(splits), dtype=float)
            raw_sub = X_surviving[cols_sub].to_numpy(dtype=float)
            # Cap per-evaluation columns to top-4 univariate residual correlations inside each split
            for s_i, (tr_e, v_e) in enumerate(splits):
                Z_tr, lo, hi, mn, sd = _winsorize_and_standardize_fit(raw_sub[:tr_e])
                Z_va = _winsorize_and_standardize_apply(raw_sub[tr_e:v_e], lo, hi, mn, sd)
                if Z_tr.shape[1] > 4:
                    res_tr = np.log1p(y_arr[:tr_e]) - np.log1p(mu_arr[:tr_e])
                    corrs = np.abs(Z_tr.T @ (res_tr - np.mean(res_tr)))
                    top_idx = np.argsort(corrs)[::-1][:4]
                    Z_tr = Z_tr[:, top_idx]
                    Z_va = Z_va[:, top_idx]
                theta = _fit_fast_ridge_poisson_offset(Z_tr, y_arr[:tr_e], log_mu[:tr_e], l2_pen=9.0)
                d_eta = np.clip(theta[0] + Z_va @ theta[1:], -0.55, 0.55)
                mu_adj = mu_arr[tr_e:v_e] * np.exp(d_eta)
                losses[s_i] = compute_mean_nb_deviance(y_arr[tr_e:v_e], mu_adj, alpha=self.alpha_disp)
            return losses

        # Baseline loss includes intercept-only calibration so noise columns never get free credit for constant drift
        base_losses = np.zeros(len(splits), dtype=float)
        zero_col = np.zeros((n, 0), dtype=float)
        for s_i, (tr_e, v_e) in enumerate(splits):
            th0 = _fit_fast_ridge_poisson_offset(zero_col[:tr_e], y_arr[:tr_e], log_mu[:tr_e], l2_pen=9.0)
            mu_b0 = mu_arr[tr_e:v_e] * np.exp(np.clip(th0[0], -0.55, 0.55))
            base_losses[s_i] = compute_mean_nb_deviance(y_arr[tr_e:v_e], mu_b0, alpha=self.alpha_disp)

        # Step 1: Rank candidate modules by single-module residual deviance reduction
        single_gains: List[Tuple[float, str]] = []
        for m_id, cols_m in mod_to_cols.items():
            mod_losses = _eval_col_subset_losses(cols_m)
            diffs = base_losses - mod_losses
            d_bar = float(np.mean(diffs))
            se_d = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else 0.0
            self.module_diagnostics_[m_id] = {
                "single_mean_gain": round(d_bar, 6),
                "single_se": round(se_d, 6),
                "n_surviving_cols": len(cols_m),
            }
            single_gains.append((d_bar, m_id))

        single_gains.sort(key=lambda pair: pair[0], reverse=True)

        # Step 2: Greedy forward 1-SE selection
        current_losses = base_losses.copy()
        admitted_cols: List[str] = []

        for _, m_id in single_gains:
            if len(self.admitted_modules_) >= self.max_modules:
                break
            trial_cols = admitted_cols + mod_to_cols[m_id]
            cand_losses = _eval_col_subset_losses(trial_cols)
            diffs = current_losses - cand_losses
            d_bar = float(np.mean(diffs))
            se_d = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else 0.0
            min_gain_floor = max(0.004, 0.004 * float(np.mean(base_losses)))
            threshold = max(min_gain_floor, 1.15 * se_d)
            all_splits_improved = bool(np.all(diffs > 0.0))
            is_admitted = all_splits_improved and (d_bar >= threshold)
            self.module_diagnostics_[m_id]["stepwise_mean_gain"] = round(d_bar, 6)
            self.module_diagnostics_[m_id]["stepwise_se"] = round(se_d, 6)
            self.module_diagnostics_[m_id]["all_splits_improved"] = float(all_splits_improved)
            self.module_diagnostics_[m_id]["admitted"] = float(is_admitted)

            if is_admitted:
                self.admitted_modules_.append(m_id)
                admitted_cols = trial_cols
                current_losses = cand_losses

        return self


class GLMResidualOffsetHead:
    """
    Post-hoc L1/L2-penalized NB2/Poisson log-offset head for frozen GLM & ZINB models:
        eta_final(t) = log(max(mu_base(t), 1e-6)) + Delta_eta_eso(t)
    Enforces:
      - Predictor cap: p_max^{res-GLM} = min(6, floor(N_eff / 20))
      - Strict collinearity ceiling: VIF <= 5.0
      - ZINB Structural Lock: frozen zero-inflation probability pi_base(t) is 100% untouched.
    """

    def __init__(self, max_vif: float = 5.0, l1_pen: float = 0.25, l2_pen: float = 6.0) -> None:
        self.max_vif = max_vif
        self.l1_pen = l1_pen
        self.l2_pen = l2_pen
        self.selected_cols_: List[str] = []
        self.lo_: np.ndarray = np.array([])
        self.hi_: np.ndarray = np.array([])
        self.mean_: np.ndarray = np.array([])
        self.std_: np.ndarray = np.array([])
        self.intercept_: float = 0.0
        self.weights_: np.ndarray = np.array([])
        self.max_vif_: float = 1.0
        self.is_active_: bool = False

    def fit(
        self,
        X_admitted: pd.DataFrame,
        y: np.ndarray,
        mu_base: np.ndarray,
        n_eff: int,
        alpha_disp: float = 0.18,
    ) -> "GLMResidualOffsetHead":
        p_max = min(6, int(n_eff // 20))
        if p_max <= 0 or X_admitted.shape[1] == 0:
            self.is_active_ = False
            self.selected_cols_ = []
            return self

        y_arr = np.maximum(0.0, np.asarray(y, dtype=float))
        mu_arr = np.maximum(1e-6, np.asarray(mu_base, dtype=float))
        log_mu = np.log(mu_arr)

        # Rank admitted columns by absolute alignment with baseline deviance score residual
        raw_mat = X_admitted.to_numpy(dtype=float)
        Z_all, _, _, _, _ = _winsorize_and_standardize_fit(raw_mat)
        score_res = (y_arr - mu_arr) / (1.0 + alpha_disp * mu_arr)
        score_centered = score_res - np.mean(score_res)
        corrs = np.abs(Z_all.T @ score_centered) / max(1.0, float(len(y_arr)))

        ordered_idx = list(np.argsort(corrs)[::-1])
        cand_cols = [X_admitted.columns[i] for i in ordered_idx[: max(p_max * 2, p_max)]]

        # Iteratively prune VIF > 5.0 down to <= p_max orthogonalized predictors
        while len(cand_cols) > 1:
            Z_sub, _, _, _, _ = _winsorize_and_standardize_fit(X_admitted[cand_cols].to_numpy(dtype=float))
            max_v, worst_j = _compute_max_vif_and_worst(Z_sub)
            if max_v <= self.max_vif:
                break
            cand_cols.pop(worst_j)

        cand_cols = cand_cols[:p_max]
        if not cand_cols:
            self.is_active_ = False
            self.selected_cols_ = []
            return self

        Z_fit, self.lo_, self.hi_, self.mean_, self.std_ = _winsorize_and_standardize_fit(
            X_admitted[cand_cols].to_numpy(dtype=float)
        )
        self.max_vif_, _ = _compute_max_vif_and_worst(Z_fit)
        self.selected_cols_ = list(cand_cols)
        p = len(cand_cols)
        r_disp = 1.0 / max(1e-3, float(alpha_disp))

        # Smooth L1 (Huberized) + L2 penalized NB2 log-likelihood with fixed offset = log_mu
        def _obj(theta: np.ndarray) -> float:
            b = theta[0]
            w = theta[1:]
            d_eta = np.clip(b + Z_fit @ w, -0.60, 0.60)
            mu_h = np.exp(np.clip(log_mu + d_eta, -14.0, 14.0))
            # Negative Binomial negative log-likelihood kernel: (y + r)*log(r + mu) - y*log(mu)
            nll = float(
                np.mean((y_arr + r_disp) * np.log(r_disp + mu_h) - y_arr * np.log(np.maximum(1e-8, mu_h)))
            )
            n = max(1.0, float(len(y_arr)))
            smooth_l1 = self.l1_pen * float(np.sum(np.sqrt(w**2 + 1e-6))) / np.sqrt(n)
            l2 = 0.5 * self.l2_pen * float(np.sum(w**2) + 3.0 * (b**2)) / n
            return nll + smooth_l1 + l2

        opt = minimize(
            _obj,
            np.zeros(p + 1, dtype=float),
            method="L-BFGS-B",
            bounds=[(-0.30, 0.30)] + [(-0.45, 0.45)] * p,
            options={"maxiter": 80, "ftol": 1e-8},
        )
        theta_hat = np.asarray(opt.x, dtype=float)
        self.intercept_ = float(theta_hat[0])
        # Hard-threshold tiny L1-shrunk weights
        w_raw = theta_hat[1:]
        self.weights_ = np.where(np.abs(w_raw) >= 1e-3, w_raw, 0.0)
        self.is_active_ = bool(np.any(np.abs(self.weights_) > 0.0) or abs(self.intercept_) > 2e-3)
        if not self.is_active_:
            self.intercept_ = 0.0
            self.weights_ = np.zeros(p, dtype=float)
        return self

    def predict_delta_eta(self, X_eso: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Return `(delta_eta, per_feature_contributions)` of shape `(N,)` and `(N, p)`."""
        n = len(X_eso)
        if not self.is_active_ or not self.selected_cols_:
            return np.zeros(n, dtype=float), np.zeros((n, 0), dtype=float)
        Z = _winsorize_and_standardize_apply(
            X_eso[self.selected_cols_].to_numpy(dtype=float),
            self.lo_,
            self.hi_,
            self.mean_,
            self.std_,
        )
        contribs = Z * self.weights_[None, :]
        d_eta = np.clip(self.intercept_ + np.sum(contribs, axis=1), -0.55, 0.55)
        return d_eta, contribs


class GBDTInitScoreResidualHead:
    """
    Post-hoc shallow LightGBM residual booster passing `init_score = log(max(mu_base, 1e-6))`
    under strict small-sample complexity caps:
      - `max_depth = 2` (when N_eff < 300, else `3`)
      - `num_leaves = 4` (when N_eff < 300, else `7`)
      - `min_child_samples = max(12, int(0.08 * N_eff))`
      - `colsample_bytree = 0.5`, `reg_alpha = 2.0`, `reg_lambda = 10.0`, `learning_rate = 0.02`
      - Early stopping (`patience = 25`) on chronological validation holdout.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.selected_cols_: List[str] = []
        self.lo_: np.ndarray = np.array([])
        self.hi_: np.ndarray = np.array([])
        self.mean_: np.ndarray = np.array([])
        self.std_: np.ndarray = np.array([])
        self.model_: Optional[LGBMRegressor] = None
        self.is_active_: bool = False

    def fit(
        self,
        X_admitted: pd.DataFrame,
        y: np.ndarray,
        mu_base: np.ndarray,
        n_eff: int,
        tweedie_power: float = 1.30,
    ) -> "GBDTInitScoreResidualHead":
        if X_admitted.shape[1] == 0 or len(y) < 28 or n_eff < 24:
            self.is_active_ = False
            self.selected_cols_ = []
            return self

        self.selected_cols_ = list(X_admitted.columns)[:16]
        Z_all, self.lo_, self.hi_, self.mean_, self.std_ = _winsorize_and_standardize_fit(
            X_admitted[self.selected_cols_].to_numpy(dtype=float)
        )
        y_arr = np.maximum(0.0, np.asarray(y, dtype=float))
        log_mu = np.log(np.clip(np.asarray(mu_base, dtype=float), 1e-6, None))

        n = len(y_arr)
        val_size = max(6, int(round(0.20 * n)))
        tr_end = n - val_size
        if tr_end < 16:
            self.is_active_ = False
            return self

        max_depth = 2 if n_eff < 300 else 3
        num_leaves = 4 if n_eff < 300 else 7
        min_child = max(12, int(0.08 * n_eff))
        # Ensure min_child fits inside tr_end if tr_end is very small, while honoring >= 12 whenever tr_end >= 24
        min_child_eff = min(min_child, max(6, tr_end // 2))

        reg = LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=float(tweedie_power),
            boost_from_average=False,
            n_estimators=140,
            learning_rate=0.02,
            max_depth=max_depth,
            num_leaves=num_leaves,
            min_child_samples=min_child_eff,
            colsample_bytree=0.5,
            subsample=0.85,
            subsample_freq=1,
            reg_alpha=2.0,
            reg_lambda=10.0,
            random_state=self.random_state,
            n_jobs=1,
            verbose=-1,
        )

        base_val_dev = compute_mean_nb_deviance(y_arr[tr_end:], np.exp(log_mu[tr_end:]), alpha=0.18)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reg.fit(
                Z_all[:tr_end],
                y_arr[:tr_end],
                init_score=log_mu[:tr_end],
                eval_set=[(Z_all[tr_end:], y_arr[tr_end:])],
                eval_init_score=[log_mu[tr_end:]],
                callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)],
            )
            val_delta = np.clip(reg.predict(Z_all[tr_end:], raw_score=True), -0.55, 0.55)

        cand_val_dev = compute_mean_nb_deviance(
            y_arr[tr_end:],
            np.exp(log_mu[tr_end:] + val_delta),
            alpha=0.18,
        )
        if cand_val_dev >= base_val_dev - 1e-6 or float(np.max(np.abs(val_delta))) < 1e-5:
            self.is_active_ = False
            self.model_ = None
            return self

        self.model_ = reg
        self.is_active_ = True
        return self

    def predict_delta_eta(self, X_eso: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        n = len(X_eso)
        if not self.is_active_ or self.model_ is None or not self.selected_cols_:
            return np.zeros(n, dtype=float), np.zeros((n, 0), dtype=float)
        Z = _winsorize_and_standardize_apply(
            X_eso[self.selected_cols_].to_numpy(dtype=float),
            self.lo_,
            self.hi_,
            self.mean_,
            self.std_,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw_delta = np.asarray(self.model_.predict(Z, raw_score=True), dtype=float)
            contrib = np.asarray(self.model_.booster_.predict(Z, pred_contrib=True), dtype=float)[:, :-1]
        return np.clip(raw_delta, -0.55, 0.55), contrib


class FisherRaoResidualCalibrator:
    """
    Distributional Uncertainty Calibrator (`max_depth=2`) taking shallow Fisher-Rao
    natural-gradient steps on log-dispersion `rho_t = log(r_t)` starting from `log(r_base)`.
    Sharpens prediction interval calibration (PICP / WSPL / CRPS) during Hawkes contagion
    or volatility bursts without shifting calm-week point expectations.
    """

    def __init__(
        self,
        n_steps: int = 12,
        learning_rate: float = 0.05,
        fisher_damping: float = 0.25,
        random_state: int = 42,
    ) -> None:
        self.n_steps = n_steps
        self.learning_rate = learning_rate
        self.fisher_damping = fisher_damping
        self.random_state = random_state
        self.selected_cols_: List[str] = []
        self.lo_: np.ndarray = np.array([])
        self.hi_: np.ndarray = np.array([])
        self.mean_: np.ndarray = np.array([])
        self.std_: np.ndarray = np.array([])
        self.trees_: List[DecisionTreeRegressor] = []
        self.rho0_: float = np.log(5.5)
        self.burst_threshold_: float = 0.5
        self.burst_col_idx_: int = 0
        self.is_active_: bool = False

    def fit(
        self,
        X_eso: pd.DataFrame,
        y: np.ndarray,
        mu_adj: np.ndarray,
        alpha_base: float,
        n_eff: int,
    ) -> "FisherRaoResidualCalibrator":
        if X_eso.shape[1] == 0 or len(y) < 30 or n_eff < 24:
            self.is_active_ = False
            return self

        # Prioritize contagion/volatility/sheaf columns for dispersion calibration
        pref = [
            c
            for c in X_eso.columns
            if any(k in c for k in ("f03_hawkes", "f01_hilbert_amplitude", "f08_sheaf", "f12_fisher"))
        ]
        use_cols = (pref if len(pref) >= 2 else list(X_eso.columns))[:8]
        self.selected_cols_ = list(use_cols)
        Z, self.lo_, self.hi_, self.mean_, self.std_ = _winsorize_and_standardize_fit(
            X_eso[self.selected_cols_].to_numpy(dtype=float)
        )

        r_base = float(np.clip(1.0 / max(1e-3, alpha_base), 0.5, 40.0))
        self.rho0_ = float(np.log(r_base))
        rho = np.full(len(y), self.rho0_, dtype=float)
        y_arr = np.maximum(0.0, np.asarray(y, dtype=float))
        mu_arr = np.maximum(1e-4, np.asarray(mu_adj, dtype=float))

        # Identify burst indicator column (default col 0) and 65th percentile threshold so calm weeks stay untouched
        self.burst_col_idx_ = 0
        self.burst_threshold_ = float(np.quantile(Z[:, 0], 0.65))
        min_leaf = max(10, int(0.08 * n_eff))
        min_leaf = min(min_leaf, max(5, len(y_arr) // 3))

        from scipy.special import digamma

        self.trees_ = []
        for step in range(self.n_steps):
            r_cur = np.exp(np.clip(rho, -1.5, 4.0))
            _, i22 = nb_fisher_information_vectorized(mu_arr, r_cur)
            # Euclidean score wrt rho = log(r): r * d(NLL)/dr
            d_nll_dr = (
                digamma(r_cur)
                - digamma(y_arr + r_cur)
                - np.log(r_cur / (r_cur + mu_arr))
                - (mu_arr - y_arr) / (r_cur + mu_arr)
            )
            grad_rho = r_cur * d_nll_dr
            fisher_rho = (r_cur**2) * i22 + self.fisher_damping
            nat_step = -grad_rho / np.maximum(1e-4, fisher_rho)
            nat_step = np.clip(nat_step, -0.6, 0.6)

            tree = DecisionTreeRegressor(
                max_depth=2,
                min_samples_leaf=min_leaf,
                random_state=self.random_state + step,
            )
            tree.fit(Z, nat_step)
            upd = tree.predict(Z)
            rho = np.clip(rho + self.learning_rate * upd, self.rho0_ - 0.85, self.rho0_ + 0.85)
            self.trees_.append(tree)

        self.is_active_ = len(self.trees_) > 0
        return self

    def predict_alpha_vec(self, X_eso: pd.DataFrame, alpha_base: float) -> np.ndarray:
        n = len(X_eso)
        base_vec = np.full(n, max(1e-4, float(alpha_base)), dtype=float)
        if not self.is_active_ or not self.selected_cols_ or not self.trees_:
            return base_vec
        Z = _winsorize_and_standardize_apply(
            X_eso[self.selected_cols_].to_numpy(dtype=float),
            self.lo_,
            self.hi_,
            self.mean_,
            self.std_,
        )
        delta_rho = np.zeros(n, dtype=float)
        for tree in self.trees_:
            delta_rho += self.learning_rate * tree.predict(Z)
        # Gate strictly onto active contagion/stress burst steps so calm weeks retain exact base dispersion
        burst_gate = np.clip((Z[:, self.burst_col_idx_] - self.burst_threshold_) / 0.75, 0.0, 1.0)
        rho_final = np.clip(self.rho0_ + burst_gate * delta_rho, self.rho0_ - 0.75, self.rho0_ + 0.75)
        r_final = np.exp(rho_final)
        return np.clip(1.0 / r_final, 0.02, 1.50)


def _format_step_attribution(
    delta_eta_val: float,
    col_names: Sequence[str],
    contrib_row: np.ndarray,
    identity_floor: bool,
) -> str:
    if identity_floor or abs(delta_eta_val) < 1e-4 or len(col_names) == 0 or contrib_row.size == 0:
        return "0.0% — Frozen Baseline Retained"
    pct = (float(np.exp(delta_eta_val)) - 1.0) * 100.0
    top_j = int(np.argmax(np.abs(contrib_row)))
    mod_id = _extract_module_prefix(col_names[top_j])
    short_mod = {
        "f01": "Hilbert phase",
        "f02": "MODWT wavelet",
        "f03": "Hawkes contagion",
        "f04": "Takens TDA",
        "f05": "Ricci curvature",
        "f06": "Koopman/Tropical",
        "f07": "Rough signature",
        "f08": "Sheaf frustration",
        "f09": "Poincare acuity",
        "f10": "Optimal transport",
        "f11": "RMT sync",
        "f12": "Fisher-Rao velocity",
        "f13": "Sandpile SOC",
    }.get(mod_id, MODULE_DISPLAY_NAMES.get(mod_id, mod_id))
    direction = "uplift" if pct >= 0 else "damping"
    return f"{pct:+.1f}% {short_mod} {direction}"


def _nb2_quantiles_vector_alpha(
    mu: np.ndarray,
    alpha_vec: np.ndarray,
    quantiles: Sequence[float] = EVAL_QUANTILES,
) -> Dict[float, np.ndarray]:
    from scipy import stats

    mu_safe = np.maximum(1e-6, np.asarray(mu, dtype=float))
    a_safe = np.maximum(1e-4, np.asarray(alpha_vec, dtype=float))
    n_param = 1.0 / a_safe
    p_param = n_param / (n_param + mu_safe)
    q_mat = np.zeros((len(quantiles), len(mu_safe)), dtype=float)
    for idx, tau in enumerate(quantiles):
        raw_q = stats.nbinom.ppf(float(tau), n_param, p_param)
        q_mat[idx, :] = np.where(np.isfinite(raw_q), np.maximum(0.0, raw_q), mu_safe)
    q_mat = np.maximum.accumulate(q_mat, axis=0)
    return {float(tau): q_mat[idx, :] for idx, tau in enumerate(quantiles)}


class PostHocResidualStacker:
    """
    Unified Post-Hoc Residual Stacker orchestrating:
      - `FrozenBaselineAdapter` SHA-256 immutability guardrails
      - `RolloutConfig` (`TELEMETRY_ONLY`, `STAGED_CORE_4`, `FULL_13_GATED`)
      - `BlockBorutaSHAPGate` + `GreedyOneSEModuleSelector`
      - Model-family-specific `GLMResidualOffsetHead` (with ZINB lock) vs. `GBDTInitScoreResidualHead`
      - Optional `FisherRaoResidualCalibrator`
      - Exact Zero-Harm Identity Floor (`atol=1e-12`) when 0 modules pass or `TELEMETRY_ONLY`.
    """

    def __init__(
        self,
        frozen_adapter: FrozenBaselineAdapter,
        config: Optional[RolloutConfig] = None,
        random_state: int = 42,
    ) -> None:
        self.frozen_adapter = frozen_adapter
        self.config = config or RolloutConfig()
        self.random_state = random_state
        self.identity_floor_active_: bool = True
        self.audit_: Optional[ResidualGateAudit] = None
        self.glm_head_: Optional[GLMResidualOffsetHead] = None
        self.gbdt_head_: Optional[GBDTInitScoreResidualHead] = None
        self.fr_calibrator_: Optional[FisherRaoResidualCalibrator] = None
        self.active_head_cols_: List[str] = []

    def fit(
        self,
        train_df: pd.DataFrame,
        X_eso_train: pd.DataFrame,
        target_col: str = "referral_count",
    ) -> "PostHocResidualStacker":
        # Verify frozen baseline is untouched before reading in-sample baseline predictions
        self.frozen_adapter.verify_model_immutability()
        base_train_bundle = self.frozen_adapter.predict(
            train_df,
            tag=f"train_in_sample_n{len(train_df)}",
        )
        mu_base_tr = np.maximum(1e-6, base_train_bundle.point_forecasts)
        y_tr = np.maximum(0.0, pd.to_numeric(train_df[target_col], errors="coerce").fillna(0.0).to_numpy(dtype=float))
        n_eff = compute_effective_temporal_sample_size(train_df)

        eligible_mods = list(self.config.active_candidate_modules())
        if self.config.mode == RolloutMode.TELEMETRY_ONLY or len(eligible_mods) == 0 or X_eso_train.shape[1] == 0:
            self.identity_floor_active_ = True
            self.audit_ = ResidualGateAudit(
                retro_mode=self.config.mode.value,
                n_eff=n_eff,
                eligible_modules=eligible_mods,
                boruta_surviving_columns=[],
                boruta_hit_counts={},
                admitted_modules=[],
                quarantined_modules=list(ALL_13_MODULES),
                identity_floor_active=True,
            )
            self.frozen_adapter.verify_model_immutability()
            return self

        # Restrict esoteric matrix strictly to eligible rollout modules
        elig_cols = [c for c in X_eso_train.columns if _extract_module_prefix(c) in eligible_mods]
        X_elig = X_eso_train[elig_cols]
        if X_elig.shape[1] == 0:
            self.identity_floor_active_ = True
            self.audit_ = ResidualGateAudit(
                retro_mode=self.config.mode.value,
                n_eff=n_eff,
                eligible_modules=eligible_mods,
                boruta_surviving_columns=[],
                boruta_hit_counts={},
                admitted_modules=[],
                quarantined_modules=eligible_mods,
                identity_floor_active=True,
            )
            return self

        # Stage 1: 4-Period Block-Permuted BorutaSHAP against baseline log-residuals
        boruta = BlockBorutaSHAPGate(
            n_trials=self.config.boruta_n_trials,
            min_hits=self.config.boruta_min_hits,
            block_size=self.config.boruta_block_size,
            alpha=self.config.boruta_alpha,
            random_state=self.random_state,
        ).fit(X_elig, y_tr, mu_base_tr, n_eff=n_eff)

        if not boruta.selected_columns_:
            self.identity_floor_active_ = True
            self.audit_ = ResidualGateAudit(
                retro_mode=self.config.mode.value,
                n_eff=n_eff,
                eligible_modules=eligible_mods,
                boruta_surviving_columns=[],
                boruta_hit_counts=boruta.hit_counts_,
                admitted_modules=[],
                quarantined_modules=eligible_mods,
                identity_floor_active=True,
            )
            self.frozen_adapter.verify_model_immutability()
            return self

        # Stage 2: Greedy 1-SE Residual Forward Module Selection
        selector = GreedyOneSEModuleSelector(
            n_inner_splits=self.config.one_se_inner_splits,
            alpha_disp=self.frozen_adapter.alpha_dispersion,
            max_modules=4 if self.config.mode == RolloutMode.STAGED_CORE_4 else 5,
        ).fit(
            X_elig[boruta.selected_columns_],
            y_tr,
            mu_base_tr,
            eligible_modules=eligible_mods,
        )

        admitted_mods = selector.admitted_modules_
        quarantined_mods = [m for m in eligible_mods if m not in admitted_mods]
        if not admitted_mods:
            self.identity_floor_active_ = True
            self.audit_ = ResidualGateAudit(
                retro_mode=self.config.mode.value,
                n_eff=n_eff,
                eligible_modules=eligible_mods,
                boruta_surviving_columns=boruta.selected_columns_,
                boruta_hit_counts=boruta.hit_counts_,
                admitted_modules=[],
                quarantined_modules=quarantined_mods,
                module_1se_scores=selector.module_diagnostics_,
                identity_floor_active=True,
            )
            self.frozen_adapter.verify_model_immutability()
            return self

        admitted_cols = [
            c for c in boruta.selected_columns_ if _extract_module_prefix(c) in admitted_mods
        ]
        X_adm = X_elig[admitted_cols]

        # Stage 3: Fit Model-Family-Appropriate Offset / Init-Score Head
        fam = self.frozen_adapter.model_family
        max_vif_used = 1.0
        if fam in ("glm_nb2", "glm_zinb"):
            # For ZINB, pass active conditional count expectation if desired; note log((1-pi)*mu_c)
            # differs from log(mu_c) only by fixed log(1-pi), so multiplicative exp(delta_eta) is identical!
            self.glm_head_ = GLMResidualOffsetHead(max_vif=self.config.glm_max_vif).fit(
                X_adm,
                y_tr,
                mu_base_tr,
                n_eff=n_eff,
                alpha_disp=self.frozen_adapter.alpha_dispersion,
            )
            head_active = self.glm_head_.is_active_
            self.active_head_cols_ = list(self.glm_head_.selected_cols_)
            max_vif_used = self.glm_head_.max_vif_
            delta_tr, _ = self.glm_head_.predict_delta_eta(X_adm)
        else:
            tw_p = float(getattr(self.frozen_adapter._frozen_copy, "best_power_", 1.30))
            self.gbdt_head_ = GBDTInitScoreResidualHead(random_state=self.random_state).fit(
                X_adm,
                y_tr,
                mu_base_tr,
                n_eff=n_eff,
                tweedie_power=tw_p,
            )
            head_active = self.gbdt_head_.is_active_
            self.active_head_cols_ = list(self.gbdt_head_.selected_cols_)
            delta_tr, _ = self.gbdt_head_.predict_delta_eta(X_adm)

        if not head_active:
            self.identity_floor_active_ = True
            self.audit_ = ResidualGateAudit(
                retro_mode=self.config.mode.value,
                n_eff=n_eff,
                eligible_modules=eligible_mods,
                boruta_surviving_columns=boruta.selected_columns_,
                boruta_hit_counts=boruta.hit_counts_,
                admitted_modules=[],
                quarantined_modules=eligible_mods,
                module_1se_scores=selector.module_diagnostics_,
                identity_floor_active=True,
            )
            self.frozen_adapter.verify_model_immutability()
            return self

        self.identity_floor_active_ = False
        mu_adj_tr = mu_base_tr * np.exp(delta_tr)

        # Stage 4: Optional Fisher-Rao Residual Interval Calibrator
        if self.config.enable_fisher_rao_calibrator:
            self.fr_calibrator_ = FisherRaoResidualCalibrator(random_state=self.random_state).fit(
                X_adm,
                y_tr,
                mu_adj_tr,
                alpha_base=self.frozen_adapter.alpha_dispersion,
                n_eff=n_eff,
            )

        self.audit_ = ResidualGateAudit(
            retro_mode=self.config.mode.value,
            n_eff=n_eff,
            eligible_modules=eligible_mods,
            boruta_surviving_columns=boruta.selected_columns_,
            boruta_hit_counts=boruta.hit_counts_,
            admitted_modules=admitted_mods,
            quarantined_modules=quarantined_mods,
            module_1se_scores=selector.module_diagnostics_,
            selected_features=list(self.active_head_cols_),
            max_vif=round(float(max_vif_used), 3),
            identity_floor_active=False,
        )
        self.frozen_adapter.verify_model_immutability()
        return self

    def predict(
        self,
        test_df: pd.DataFrame,
        X_eso_test: pd.DataFrame,
        quantiles: Sequence[float] = EVAL_QUANTILES,
        tag: Optional[str] = None,
    ) -> ModelPredictionBundle:
        """
        Produce post-hoc adjusted `ModelPredictionBundle` while verifying frozen baseline
        SHA-256 immutability. Returns exact frozen baseline bundle verbatim (`atol=1e-12`)
        whenever `identity_floor_active_` is True.
        """
        frozen_bundle = self.frozen_adapter.predict(test_df, quantiles=quantiles, tag=tag)

        if self.identity_floor_active_:
            exact_copy = clone_prediction_bundle(frozen_bundle)
            exact_copy.diagnostics["retrospective"] = {
                "retro_mode": self.config.mode.value,
                "identity_floor_active": True,
                "delta_eta": [0.0] * len(exact_copy.point_forecasts),
                "frozen_point_forecasts": exact_copy.point_forecasts.tolist(),
                "attributions": ["0.0% — Frozen Baseline Retained"] * len(exact_copy.point_forecasts),
                "admitted_modules": [],
            }
            return exact_copy

        if self.glm_head_ is not None and self.glm_head_.is_active_:
            delta_eta, contribs = self.glm_head_.predict_delta_eta(X_eso_test)
        elif self.gbdt_head_ is not None and self.gbdt_head_.is_active_:
            delta_eta, contribs = self.gbdt_head_.predict_delta_eta(X_eso_test)
        else:
            delta_eta = np.zeros(len(test_df), dtype=float)
            contribs = np.zeros((len(test_df), 0), dtype=float)

        mult_factor = np.exp(delta_eta)
        mu_base = np.maximum(1e-6, frozen_bundle.point_forecasts)
        mu_final = mu_base * mult_factor

        alpha_base = self.frozen_adapter.alpha_dispersion
        if self.fr_calibrator_ is not None and self.fr_calibrator_.is_active_:
            alpha_vec = self.fr_calibrator_.predict_alpha_vec(X_eso_test, alpha_base=alpha_base)
        else:
            alpha_vec = np.full(len(mu_final), alpha_base, dtype=float)

        fam = self.frozen_adapter.model_family
        if fam == "glm_zinb":
            # ZINB Structural Lock: pi_zero is 100% untouched from frozen ZINB!
            pi_zero_locked, mu_count_base, zinb_alpha = self.frozen_adapter.extract_zinb_structural_components(
                test_df
            )
            mu_count_final = mu_count_base * mult_factor
            mu_final = (1.0 - pi_zero_locked) * mu_count_final
            q_final = _zinb_quantiles_from_params(
                mu_count=mu_count_final,
                pi_zero=pi_zero_locked,
                alpha=float(np.mean(alpha_vec)) if self.fr_calibrator_ is not None else zinb_alpha,
                quantiles=quantiles,
            )
        elif fam == "glm_nb2":
            if self.fr_calibrator_ is not None and self.fr_calibrator_.is_active_:
                q_final = _nb2_quantiles_vector_alpha(mu_final, alpha_vec, quantiles=quantiles)
            else:
                q_final = _nb2_quantiles_from_mean_and_alpha(
                    mu=mu_final,
                    alpha=alpha_base,
                    se_eta=np.full(len(mu_final), 0.08),
                    quantiles=quantiles,
                )
        else:
            # Multiplicative scaling of baseline quantiles blended with Fisher-Rao calibrated NB tails
            nb_q = _nb2_quantiles_vector_alpha(mu_final, alpha_vec, quantiles=quantiles)
            q_mat = np.zeros((len(quantiles), len(mu_final)), dtype=float)
            for idx, tau in enumerate(quantiles):
                scaled_base_q = np.maximum(0.0, frozen_bundle.quantiles[float(tau)] * mult_factor)
                if self.fr_calibrator_ is not None and self.fr_calibrator_.is_active_:
                    q_mat[idx, :] = 0.50 * scaled_base_q + 0.50 * nb_q[float(tau)]
                else:
                    q_mat[idx, :] = scaled_base_q
            q_mat = np.maximum.accumulate(q_mat, axis=0)
            q_final = {float(tau): q_mat[idx, :] for idx, tau in enumerate(quantiles)}

        attributions = [
            _format_step_attribution(
                float(delta_eta[i]),
                self.active_head_cols_,
                contribs[i] if contribs.shape[0] == len(delta_eta) and contribs.shape[1] > 0 else np.zeros(0),
                identity_floor=False,
            )
            for i in range(len(delta_eta))
        ]

        self.frozen_adapter.verify_model_immutability()

        diag = dict(frozen_bundle.diagnostics)
        diag["retrospective"] = {
            "retro_mode": self.config.mode.value,
            "identity_floor_active": False,
            "delta_eta": [round(float(x), 6) for x in delta_eta],
            "frozen_point_forecasts": [round(float(x), 4) for x in frozen_bundle.point_forecasts],
            "attributions": attributions,
            "admitted_modules": list(self.audit_.admitted_modules) if self.audit_ else [],
            "quarantined_modules": list(self.audit_.quarantined_modules) if self.audit_ else [],
            "selected_features": list(self.active_head_cols_),
        }
        return ModelPredictionBundle(
            model_id=f"{frozen_bundle.model_id}_retro_{self.config.mode.value.lower()}",
            model_name=f"{frozen_bundle.model_name} + Retro ({self.config.mode.value})",
            point_forecasts=mu_final,
            quantiles=q_final,
            aic=frozen_bundle.aic,
            bic=frozen_bundle.bic,
            log_likelihood=frozen_bundle.log_likelihood,
            alpha_dispersion=round(float(np.mean(alpha_vec)), 4),
            diagnostics=diag,
        )
