"""
Phase 3: Fisher-Rao Natural Gradient Negative Binomial Booster
(`esoteric_math/models/fisher_rao_ngboost.py`).

Implements a stagewise Riemannian Natural Gradient Booster on the 2-parameter
Negative Binomial statistical manifold theta_i = (eta_{mu, i}, eta_{r, i}) = (log mu_i, log r_i):

  tilde_nabla_theta L(y_i; theta_i) = G(theta_i)^{-1} nabla_theta L(y_i; theta_i)

where G(theta) = diag(r*mu / (r + mu), r^2 * g_{rr}(mu, r)) is the exact diagonal
Fisher Information tensor in log-parameter coordinates.

Key Architectural Properties:
  1. Exact Multiplicative Exposure Proportionality:
     eta_{mu, i} = log(active_enrolments_i) + eta_{mu, base}(x_i) + sum_m nu * rho_m * h_{mu, m}(x_i).
     Doubling `active_enrolments` strictly doubles predicted counts and distributional scales.
  2. Heteroscedastic Overdispersion Boosting:
     Simultaneously updates log-shape eta_{r, i} along its Fisher-Rao natural gradient so
     high-volatility induction/return regimes widen upper-tail quantiles (Q_0.90, Q_0.975)
     while calm regimes maintain sharp calibrated intervals.
  3. Full Contract Compatibility:
     Matches `NB2GovernanceGLM` / `TweedieLightGBMModel` `.fit(train_df)` and
     `.predict(test_df, quantiles=EVAL_QUANTILES) -> ModelPredictionBundle`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.special import digamma, gammaln
from sklearn.linear_model import PoissonRegressor
from sklearn.tree import DecisionTreeRegressor

from app.evaluation.kaggle_metrics import EVAL_QUANTILES
from app.models.safeguarding_benchmark import DEFAULT_BENCHMARK_FEATURES, ModelPredictionBundle
from esoteric_math.features.f12_fisher_rao import nb_fisher_information_vectorized


def nb_log_likelihood_vec(y: np.ndarray, eta_mu: np.ndarray, eta_r: np.ndarray) -> np.ndarray:
    """Pointwise log-likelihood log P_{NB}(y | mu=exp(eta_mu), r=exp(eta_r))."""
    yv = np.maximum(0.0, np.asarray(y, dtype=float))
    mu = np.exp(np.clip(eta_mu, -12.0, 12.0))
    r = np.exp(np.clip(eta_r, -2.0, 4.5))
    return (
        gammaln(yv + r)
        - gammaln(r)
        - gammaln(yv + 1.0)
        + r * np.log(r / (r + mu))
        + yv * np.log(np.maximum(mu, 1e-12) / (r + mu))
    )


def nb_natural_gradient_step(
    y: np.ndarray,
    eta_mu: np.ndarray,
    eta_r: np.ndarray,
    damping: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute exact Fisher-Rao natural ascent direction tilde_s = G(theta)^{-1} nabla_theta ell(y; theta)
    in log-coordinates theta = (eta_mu, eta_r) = (log mu, log r).

    Returns (nat_grad_mu, nat_grad_r, fisher_weight_mu, fisher_weight_r).
    """
    yv = np.maximum(0.0, np.asarray(y, dtype=float))
    mu = np.exp(np.clip(eta_mu, -12.0, 12.0))
    r = np.exp(np.clip(eta_r, -2.0, 4.5))

    # Ordinary score wrt eta_mu = log(mu): s_mu = r * (y - mu) / (r + mu)
    score_eta_mu = r * (yv - mu) / (r + mu)
    # Exact Fisher metric wrt eta_mu: G_{eta_mu, eta_mu} = r * mu / (r + mu)
    G_eta_mu = (r * mu) / (r + mu)
    # Exact natural gradient wrt eta_mu (with mild low-count Tikhonov floor for zero-count stability):
    nat_mu = score_eta_mu / np.maximum(G_eta_mu + damping, 0.15)
    nat_mu = np.clip(nat_mu, -4.0, 4.0)

    # Ordinary score wrt eta_r = log(r):
    score_r = digamma(yv + r) - digamma(r) + np.log(r / (r + mu)) + (mu - yv) / (r + mu)
    score_eta_r = r * score_r

    # Exact Fisher metric wrt eta_r: G_{eta_r, eta_r} = r^2 * g_{rr}(mu, r)
    _, g_rr = nb_fisher_information_vectorized(mu, r)
    G_eta_r = np.maximum((r ** 2) * g_rr, 1e-4)
    nat_r = score_eta_r / (G_eta_r + 0.05)
    nat_r = np.clip(nat_r, -2.5, 2.5)

    return nat_mu, nat_r, G_eta_mu, G_eta_r


class FisherRaoNaturalGradientBooster:
    """
    Fisher-Rao Natural Gradient Negative Binomial Booster (`fisher_rao_ngboost`).
    Updates multiparameter Negative Binomial predictions (log-mean eta_mu with exact
    log-enrolment offset + heteroscedastic log-shape eta_r) along the Riemannian
    Fisher-Rao natural gradient.
    """

    model_id = "fisher_rao_ngboost"
    model_name = "Fisher-Rao Natural Gradient NB Booster"

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        n_estimators: int = 55,
        learning_rate_mu: float = 0.045,
        learning_rate_r: float = 0.035,
        max_depth_mu: int = 2,
        max_depth_r: int = 2,
        use_linear_anchor: bool = True,
        colsample_rate: float = 0.80,
        random_state: int = 42,
    ) -> None:
        self.feature_cols = feature_cols or list(DEFAULT_BENCHMARK_FEATURES)
        self.n_estimators = n_estimators
        self.learning_rate_mu = learning_rate_mu
        self.learning_rate_r = learning_rate_r
        self.max_depth_mu = max_depth_mu
        self.max_depth_r = max_depth_r
        self.use_linear_anchor = use_linear_anchor
        self.colsample_rate = colsample_rate
        self.random_state = random_state

        self.used_cols_: List[str] = []
        self.anchor_cols_idx_: np.ndarray = np.array([], dtype=int)
        self.anchor_means_: np.ndarray = np.array([])
        self.anchor_stds_: np.ndarray = np.array([])
        self.anchor_model_: Optional[PoissonRegressor] = None
        self.base_log_rate_: float = -5.3
        self.base_eta_r_: float = 1.75
        self.trees_mu_: List[Tuple[DecisionTreeRegressor, np.ndarray, float]] = []
        self.trees_r_: List[Tuple[DecisionTreeRegressor, np.ndarray, float]] = []
        self.best_iteration_: int = 0
        self.llf_: float = 0.0
        self.aic_: float = 0.0
        self.bic_: float = 0.0
        self.alpha_mean_: float = 0.18

    def _compute_anchor_log_rate(self, X: np.ndarray) -> np.ndarray:
        if self.anchor_model_ is None or len(self.anchor_cols_idx_) == 0:
            return np.full(len(X), self.base_log_rate_, dtype=float)
        Z = (X[:, self.anchor_cols_idx_] - self.anchor_means_) / self.anchor_stds_
        Z = np.clip(Z, -5.0, 5.0)
        rate_pred = np.maximum(1e-7, self.anchor_model_.predict(Z))
        return np.log(rate_pred)

    def _fit_stages(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        log_enrol_tr: np.ndarray,
        anchor_tr: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        log_enrol_val: Optional[np.ndarray] = None,
        anchor_val: Optional[np.ndarray] = None,
        max_stages: int = 55,
    ) -> Tuple[List[Tuple[DecisionTreeRegressor, np.ndarray, float]], List[Tuple[DecisionTreeRegressor, np.ndarray, float]], int, float]:
        n_tr, p = X_tr.shape
        mu0 = np.exp(np.clip(log_enrol_tr + anchor_tr, -12.0, 12.0))
        excess_v = float(np.sum((y_tr - mu0) ** 2 - mu0))
        denom_v = float(max(1.0, np.sum(mu0 ** 2)))
        alpha0 = float(np.clip(excess_v / denom_v, 0.04, 0.80))
        eta_r0 = float(np.log(1.0 / alpha0))

        eta_mu_tr = log_enrol_tr + anchor_tr
        eta_r_tr = np.full(n_tr, eta_r0, dtype=float)

        has_val = X_val is not None and y_val is not None and log_enrol_val is not None and anchor_val is not None and len(y_val) >= 4
        if has_val:
            eta_mu_val = log_enrol_val + anchor_val
            eta_r_val = np.full(len(y_val), eta_r0, dtype=float)
            best_val_nll = float(-np.mean(nb_log_likelihood_vec(y_val, eta_mu_val, eta_r_val)))
        else:
            best_val_nll = float("inf")

        best_iter = 0
        trees_mu: List[Tuple[DecisionTreeRegressor, np.ndarray, float]] = []
        trees_r: List[Tuple[DecisionTreeRegressor, np.ndarray, float]] = []

        rng = np.random.default_rng(self.random_state)
        min_leaf = max(4, min(10, n_tr // 8))
        k_sub = max(1, int(np.ceil(self.colsample_rate * p)))

        for m in range(max_stages):
            nat_mu, nat_r, w_mu, w_r = nb_natural_gradient_step(y_tr, eta_mu_tr, eta_r_tr)

            cols_mu = np.sort(rng.choice(p, size=k_sub, replace=False)) if k_sub < p else np.arange(p)
            cols_r = np.sort(rng.choice(p, size=k_sub, replace=False)) if k_sub < p else np.arange(p)

            tree_mu = DecisionTreeRegressor(
                max_depth=self.max_depth_mu,
                min_samples_leaf=min_leaf,
                random_state=self.random_state + m * 2,
            )
            tree_mu.fit(X_tr[:, cols_mu], nat_mu, sample_weight=np.sqrt(np.maximum(w_mu, 0.05)))
            step_mu = tree_mu.predict(X_tr[:, cols_mu])

            tree_r = DecisionTreeRegressor(
                max_depth=self.max_depth_r,
                min_samples_leaf=max(min_leaf, 6),
                random_state=self.random_state + m * 2 + 1,
            )
            tree_r.fit(X_tr[:, cols_r], nat_r, sample_weight=np.sqrt(np.maximum(w_r, 0.01)))
            step_r = tree_r.predict(X_tr[:, cols_r])

            # 1D Exact Line Search along the joint Fisher-Rao Natural Gradient direction
            def _stage_obj(rho: float) -> float:
                cand_mu = eta_mu_tr + rho * step_mu
                cand_r = eta_r_tr + 0.75 * rho * step_r
                return float(-np.mean(nb_log_likelihood_vec(y_tr, cand_mu, cand_r)))

            res = minimize_scalar(_stage_obj, bounds=(0.0, 1.8), method="bounded")
            rho_opt = float(res.x) if res.success else 0.5
            if rho_opt <= 1e-4:
                break

            scale_mu = self.learning_rate_mu * rho_opt
            scale_r = self.learning_rate_r * 0.75 * rho_opt

            eta_mu_tr = np.clip(eta_mu_tr + scale_mu * step_mu, -12.0, 12.0)
            eta_r_tr = np.clip(eta_r_tr + scale_r * step_r, -1.5, 4.2)

            trees_mu.append((tree_mu, cols_mu, scale_mu))
            trees_r.append((tree_r, cols_r, scale_r))

            if has_val:
                assert X_val is not None and y_val is not None
                eta_mu_val = np.clip(eta_mu_val + scale_mu * tree_mu.predict(X_val[:, cols_mu]), -12.0, 12.0)
                eta_r_val = np.clip(eta_r_val + scale_r * tree_r.predict(X_val[:, cols_r]), -1.5, 4.2)
                val_nll = float(-np.mean(nb_log_likelihood_vec(y_val, eta_mu_val, eta_r_val)))
                if val_nll < best_val_nll - 1e-5:
                    best_val_nll = val_nll
                    best_iter = m + 1
            else:
                best_iter = m + 1

        return trees_mu, trees_r, best_iter, eta_r0

    def fit(self, train_df: pd.DataFrame) -> "FisherRaoNaturalGradientBooster":
        cand = [c for c in self.feature_cols if c in train_df.columns]
        self.used_cols_ = [
            c
            for c in cand
            if float(np.std(np.nan_to_num(train_df[c].to_numpy(dtype=float), nan=0.0))) > 1e-7
        ]
        if not self.used_cols_ and cand:
            self.used_cols_ = [cand[0]]

        X = np.nan_to_num(train_df[self.used_cols_].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        y = np.maximum(0.0, train_df["referral_count"].to_numpy(dtype=float))
        enrol = np.maximum(1.0, train_df["active_enrolments"].to_numpy(dtype=float))
        log_enrol = np.log(enrol)
        n_obs = len(train_df)

        self.base_log_rate_ = float(np.log(max(1e-6, np.sum(y) / np.sum(enrol))))

        # Anchor on baseline core predictors with ridge-Poisson log-rate offset (strictly exposure-proportional)
        base_anchor_names = [c for c in DEFAULT_BENCHMARK_FEATURES if c in self.used_cols_]
        if self.use_linear_anchor and len(base_anchor_names) > 0 and n_obs >= 20:
            self.anchor_cols_idx_ = np.array([self.used_cols_.index(c) for c in base_anchor_names], dtype=int)
        elif self.use_linear_anchor and len(self.used_cols_) > 0 and n_obs >= 20:
            self.anchor_cols_idx_ = np.arange(min(8, len(self.used_cols_)), dtype=int)
        else:
            self.anchor_cols_idx_ = np.array([], dtype=int)

        # Inner temporal split to select optimal natural-gradient boosting iteration M^* without overfitting
        opt_iters = min(25, self.n_estimators)
        if n_obs >= 36:
            split = int(n_obs * 0.78)
            if len(self.anchor_cols_idx_) > 0:
                X_sub_tr = X[:split, self.anchor_cols_idx_]
                m_sub = np.mean(X_sub_tr, axis=0)
                s_sub = np.maximum(np.std(X_sub_tr, axis=0), 1e-6)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    anc_inner = PoissonRegressor(alpha=0.12, max_iter=300)
                    anc_inner.fit(np.clip((X_sub_tr - m_sub) / s_sub, -5.0, 5.0), y[:split] / enrol[:split], sample_weight=enrol[:split])
                    anc_tr = np.log(np.maximum(1e-7, anc_inner.predict(np.clip((X_sub_tr - m_sub) / s_sub, -5.0, 5.0))))
                    X_sub_va = X[split:, self.anchor_cols_idx_]
                    anc_va = np.log(np.maximum(1e-7, anc_inner.predict(np.clip((X_sub_va - m_sub) / s_sub, -5.0, 5.0))))
            else:
                r_tr = float(np.log(max(1e-6, np.sum(y[:split]) / np.sum(enrol[:split]))))
                anc_tr = np.full(split, r_tr, dtype=float)
                anc_va = np.full(n_obs - split, r_tr, dtype=float)

            _, _, best_m, _ = self._fit_stages(
                X[:split],
                y[:split],
                log_enrol[:split],
                anc_tr,
                X_val=X[split:],
                y_val=y[split:],
                log_enrol_val=log_enrol[split:],
                anchor_val=anc_va,
                max_stages=self.n_estimators,
            )
            opt_iters = best_m

        # Fit final anchor + natural gradient stages on full training fold for M^* iterations
        if len(self.anchor_cols_idx_) > 0:
            X_anc = X[:, self.anchor_cols_idx_]
            self.anchor_means_ = np.mean(X_anc, axis=0)
            self.anchor_stds_ = np.maximum(np.std(X_anc, axis=0), 1e-6)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.anchor_model_ = PoissonRegressor(alpha=0.12, max_iter=350)
                self.anchor_model_.fit(
                    np.clip((X_anc - self.anchor_means_) / self.anchor_stds_, -5.0, 5.0),
                    y / enrol,
                    sample_weight=enrol,
                )
        else:
            self.anchor_model_ = None

        full_anc = self._compute_anchor_log_rate(X)
        if opt_iters > 0:
            t_mu, t_r, _, eta_r0 = self._fit_stages(
                X, y, log_enrol, full_anc, max_stages=opt_iters
            )
            self.trees_mu_ = t_mu
            self.trees_r_ = t_r
            self.base_eta_r_ = eta_r0
        else:
            mu0 = np.exp(np.clip(log_enrol + full_anc, -12.0, 12.0))
            alpha0 = float(np.clip(np.sum((y - mu0) ** 2 - mu0) / max(1.0, np.sum(mu0 ** 2)), 0.04, 0.80))
            self.base_eta_r_ = float(np.log(1.0 / alpha0))
            self.trees_mu_ = []
            self.trees_r_ = []

        self.best_iteration_ = len(self.trees_mu_)

        # Record in-sample manifold likelihood diagnostics
        eta_mu_in, eta_r_in = self._predict_manifold_params(X, log_enrol)
        ll_vec = nb_log_likelihood_vec(y, eta_mu_in, eta_r_in)
        self.llf_ = float(np.sum(ll_vec))
        eff_k = float(len(self.anchor_cols_idx_) + 2 + 0.5 * self.best_iteration_)
        self.aic_ = float(-2.0 * self.llf_ + 2.0 * eff_k)
        self.bic_ = float(-2.0 * self.llf_ + np.log(max(n_obs, 2)) * eff_k)
        self.alpha_mean_ = float(np.mean(np.exp(-eta_r_in)))
        return self

    def _predict_manifold_params(self, X: np.ndarray, log_enrol: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        eta_mu = log_enrol + self._compute_anchor_log_rate(X)
        eta_r = np.full(len(X), self.base_eta_r_, dtype=float)

        for tree_m, cols_m, sc_m in self.trees_mu_:
            eta_mu = eta_mu + sc_m * tree_m.predict(X[:, cols_m])
        for tree_r, cols_r, sc_r in self.trees_r_:
            eta_r = eta_r + sc_r * tree_r.predict(X[:, cols_r])

        eta_mu = np.clip(eta_mu, -12.0, 12.0)
        eta_r = np.clip(eta_r, -1.5, 4.2)
        return eta_mu, eta_r

    def predict(
        self,
        test_df: pd.DataFrame,
        quantiles: Sequence[float] = EVAL_QUANTILES,
    ) -> ModelPredictionBundle:
        X = np.nan_to_num(test_df[self.used_cols_].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        log_enrol = np.log(np.maximum(1.0, test_df["active_enrolments"].to_numpy(dtype=float)))
        eta_mu, eta_r = self._predict_manifold_params(X, log_enrol)

        mu_pred = np.exp(eta_mu)
        r_pred = np.exp(eta_r)
        # Incorporate small epistemic log-rate uncertainty into effective dispersion (matching NB2 calibration)
        alpha_eff = np.maximum(1e-4, (1.0 / r_pred) + (np.exp(0.08 ** 2) - 1.0))
        n_param = 1.0 / alpha_eff
        p_param = n_param / (n_param + mu_pred)

        q_mat = np.zeros((len(quantiles), len(mu_pred)), dtype=float)
        for q_idx, tau in enumerate(quantiles):
            raw_q = stats.nbinom.ppf(float(tau), n_param, p_param)
            raw_q = np.where(np.isfinite(raw_q), np.maximum(0.0, raw_q), mu_pred)
            q_mat[q_idx, :] = raw_q

        # Strict quantile monotonicity
        q_mat = np.maximum.accumulate(q_mat, axis=0)
        q_dict = {float(tau): q_mat[idx, :] for idx, tau in enumerate(quantiles)}

        return ModelPredictionBundle(
            model_id=self.model_id,
            model_name=f"{self.model_name} (stages={self.best_iteration_})",
            point_forecasts=mu_pred,
            quantiles=q_dict,
            aic=round(self.aic_, 2),
            bic=round(self.bic_, 2),
            log_likelihood=round(self.llf_, 2),
            alpha_dispersion=round(float(np.mean(1.0 / r_pred)), 4),
            diagnostics={
                "best_natural_gradient_stages": self.best_iteration_,
                "mean_r_shape": round(float(np.mean(r_pred)), 3),
                "min_r_shape": round(float(np.min(r_pred)), 3),
                "max_r_shape": round(float(np.max(r_pred)), 3),
            },
        )
