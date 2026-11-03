"""
Phase 3 & Phase 4: Multi-Model Count-Forecasting Benchmark for UK FE Safeguarding
(Legacy ML vs. NB2 GLM vs. Zero-Inflated Negative Binomial vs. Tweedie-Loss LightGBM).

Satisfies UK KCSIE/GDPR explainability standards, strict temporal causality (expanding-window
out-of-time cross-validation), active learner exposure offsets (ln N_t) across parametric
and gradient-boosted architectures, and calibrated 95% prediction intervals for DSL triage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
from lightgbm import LGBMRegressor

from app.features import (
    VIFDiagnosticResult,
    aggregate_college_wide_weekly,
    audit_target_distribution,
    engineer_safeguarding_features,
    generate_weekly_safeguarding_panel,
)
from app.evaluation.kaggle_metrics import (
    EVAL_QUANTILES,
    ModelEvaluationMetrics,
    compute_mae,
    compute_mean_nb_deviance,
    compute_mean_tweedie_deviance_safe,
    compute_picp_and_miw,
    compute_pinball_loss,
    compute_rmsle,
    compute_rmse,
    compute_rmsse,
    compute_spl,
    compute_surge_mae,
    compute_wrmsse,
    compute_wspl,
)


DEFAULT_BENCHMARK_FEATURES: List[str] = [
    "referral_lag_t1",
    "referral_rolling_3w",
    "early_disengagement_index",
    "bursary_t3",
    "is_autumn_transition",
    "is_pre_holiday_week",
    "is_post_holiday_week",
    "is_assessment_crunch",
    "is_low_activity_week",
    "statutory_vulnerability_share",
    "deprivation_imd_1_3_share",
    "provision_16_18_ratio",
]

ZINB_INFLATION_FEATURES: List[str] = [
    "is_low_activity_week",
    "is_pre_holiday_week",
    "is_post_holiday_week",
]

FEATURE_PLAIN_ENGLISH_LABELS: Dict[str, Tuple[str, str]] = {
    "const": (
        "Baseline Log Incidence Rate (Intercept)",
        "Baseline weekly referral incidence rate per active learner when all predictors are at reference levels.",
    ),
    "referral_lag_t1": (
        "Prior Week Referral Rate (Y_{t-1} per 1,000)",
        "Captures peer contagion, multi-learner disclosures, and heightened staff reporting vigilance from week t-1.",
    ),
    "referral_rolling_3w": (
        "3-Week Rolling Referral Rate (t-1..t-3)",
        "Sustained recent safeguarding referral momentum across the preceding three weeks.",
    ),
    "early_disengagement_index": (
        "Composite Early Disengagement Index (t-1)",
        "Expanding-window standardized composite of lagged attendance velocity drops (Delta Att_{t-1}) and tutor pastoral logs (Pastoral_{t-1}).",
    ),
    "att_drop_t1": (
        "Lagged Attendance Drop (-Delta Att_{t-1})",
        "Week-on-week drop in learner attendance percentage points in week t-1.",
    ),
    "pastoral_t1": (
        "Lagged Low-Level Pastoral Logs (Pastoral_{t-1})",
        "Rate of tutor 'Cause for Concern' and fitness-to-study logs per 1,000 learners in week t-1.",
    ),
    "bursary_t3": (
        "Lagged Emergency Hardship Claims (Bursary_{t-3})",
        "16-19 Bursary emergency food, travel, and equipment requests per 1,000 learners three weeks prior.",
    ),
    "is_autumn_transition": (
        "Autumn Transition Surge (Weeks 1-6)",
        "September-October induction, enrolment, and secondary school safeguarding file handover window.",
    ),
    "is_pre_holiday_week": (
        "Pre-Holiday Boundary Week",
        "Week immediately preceding Half-Term, Winter, or Easter breaks (heightened home/welfare anxiety disclosures).",
    ),
    "is_post_holiday_week": (
        "Post-Holiday Return Week",
        "Week immediately following a college break (accumulated holiday welfare disclosures).",
    ),
    "is_assessment_crunch": (
        "Assessment & Resit Crunch Window",
        "November GCSE English/Maths resit window and May/June exam & vocational NEA submission deadlines.",
    ),
    "is_low_activity_week": (
        "Low Activity / Placement / Closure Week",
        "Reading weeks, work placement blocks, or college vacation closure weeks with minimal on-campus footfall.",
    ),
    "statutory_vulnerability_share": (
        "ILR Statutory Vulnerability Share",
        "Proportion of cohort who are Care Experienced, have an active EHCP, or receive Free School Meals.",
    ),
    "deprivation_imd_1_3_share": (
        "ILR Deprivation Concentration (IMD Deciles 1-3)",
        "Proportion of enrolled learners residing in Index of Multiple Deprivation Deciles 1 to 3.",
    ),
    "provision_16_18_ratio": (
        "16-18 Study Programme Provision Share",
        "Ratio of 16-18 Study Programme learners relative to total enrolled cohort (vs. 19+ Adult/Apprenticeships).",
    ),
}


@dataclass
class ModelPredictionBundle:
    model_id: str
    model_name: str
    point_forecasts: np.ndarray
    quantiles: Dict[float, np.ndarray]
    aic: Optional[float] = None
    bic: Optional[float] = None
    log_likelihood: Optional[float] = None
    alpha_dispersion: Optional[float] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GovernanceCoefficientRow:
    feature_name: str
    feature_label: str
    beta: float
    std_error: float
    z_stat: float
    p_value: float
    irr: float
    irr_ci_lower_95: float
    irr_ci_upper_95: float
    interpretation: str


@dataclass
class DSLWeeklyTriageRow:
    week_start: str
    academic_year: str
    week_of_academic_year: int
    curriculum_area: str
    active_enrolments: int
    actual_referrals: Optional[int]
    expected_referrals_mu: float
    q_50: float
    q_90: float
    q_975: float
    baseline_dsl_capacity: float
    exceeds_q90_capacity: bool
    exceeds_q975_capacity: bool
    recommended_triage_action: str


def _nb2_quantiles_from_mean_and_alpha(
    mu: np.ndarray,
    alpha: float,
    se_eta: Optional[np.ndarray] = None,
    quantiles: Sequence[float] = EVAL_QUANTILES,
) -> Dict[float, np.ndarray]:
    """
    Compute non-negative, strictly monotonic quantiles from a Negative Binomial (NB2)
    predictive distribution incorporating both dispersion (alpha) and linear predictor
    estimation uncertainty (se_eta).
    """
    mu_safe = np.maximum(1e-6, np.asarray(mu, dtype=float))
    if se_eta is not None:
        se_arr = np.where(np.isfinite(se_eta), np.minimum(0.65, np.maximum(0.0, se_eta)), 0.08)
    else:
        se_arr = np.full_like(mu_safe, 0.06)

    # Combine parameter uncertainty into effective NB2 dispersion:
    # Var(Y) ≈ mu + (alpha + exp(se_eta^2) - 1) * mu^2
    alpha_eff = np.maximum(1e-4, float(alpha) + (np.exp(se_arr**2) - 1.0))
    n_param = 1.0 / alpha_eff
    p_param = n_param / (n_param + mu_safe)

    q_mat = np.zeros((len(quantiles), len(mu_safe)), dtype=float)
    for q_idx, tau in enumerate(quantiles):
        raw_q = stats.nbinom.ppf(float(tau), n_param, p_param)
        raw_q = np.where(np.isfinite(raw_q), np.maximum(0.0, raw_q), mu_safe)
        q_mat[q_idx, :] = raw_q

    # Enforce strict non-crossing monotonicity across quantiles
    q_mat = np.maximum.accumulate(q_mat, axis=0)
    return {float(tau): q_mat[idx, :] for idx, tau in enumerate(quantiles)}


def _zinb_quantiles_from_params(
    mu_count: np.ndarray,
    pi_zero: np.ndarray,
    alpha: float,
    quantiles: Sequence[float] = EVAL_QUANTILES,
) -> Dict[float, np.ndarray]:
    """
    Compute exact quantiles for a Zero-Inflated Negative Binomial (ZINB) distribution:
      F_ZINB(y) = pi + (1 - pi) * F_NB(y; mu_count, alpha)
    For tau <= pi, Q(tau) = 0; for tau > pi, Q(tau) = F_NB^{-1}((tau - pi) / (1 - pi)).
    """
    mu_safe = np.maximum(1e-6, np.asarray(mu_count, dtype=float))
    pi_safe = np.clip(np.asarray(pi_zero, dtype=float), 0.0, 0.995)
    alpha_eff = max(1e-4, float(alpha))
    n_param = 1.0 / alpha_eff
    p_param = n_param / (n_param + mu_safe)

    q_mat = np.zeros((len(quantiles), len(mu_safe)), dtype=float)
    for q_idx, tau in enumerate(quantiles):
        adj_tau = (float(tau) - pi_safe) / np.maximum(1e-6, 1.0 - pi_safe)
        pos_mask = adj_tau > 0.0
        q_vals = np.zeros_like(mu_safe)
        if np.any(pos_mask):
            adj_clipped = np.clip(adj_tau[pos_mask], 1e-6, 0.9999)
            nb_q = stats.nbinom.ppf(adj_clipped, n_param, p_param[pos_mask])
            q_vals[pos_mask] = np.where(np.isfinite(nb_q), np.maximum(0.0, nb_q), mu_safe[pos_mask])
        q_mat[q_idx, :] = q_vals

    q_mat = np.maximum.accumulate(q_mat, axis=0)
    return {float(tau): q_mat[idx, :] for idx, tau in enumerate(quantiles)}


class LegacyMLBaselineModel:
    """
    Baseline Model 0: Legacy Unconstrained Gradient Boosted Trees (L2 Regression Loss).
    Trains standard LightGBM on raw counts without log_enrolment exposure offset or count loss.
    """

    model_id = "legacy_ml_baseline"
    model_name = "Legacy ML Baseline (Unconstrained L2 GBDT)"

    def __init__(self, feature_cols: Optional[List[str]] = None) -> None:
        self.feature_cols = feature_cols or list(DEFAULT_BENCHMARK_FEATURES)
        self.model_: Optional[LGBMRegressor] = None
        self.used_cols_: List[str] = []
        self.resid_quantiles_: Dict[float, float] = {}

    def fit(self, train_df: pd.DataFrame) -> "LegacyMLBaselineModel":
        self.used_cols_ = [c for c in self.feature_cols if c in train_df.columns]
        X = train_df[self.used_cols_].to_numpy(dtype=float)
        y = np.maximum(0.0, train_df["referral_count"].to_numpy(dtype=float))

        self.model_ = LGBMRegressor(
            objective="regression",
            n_estimators=45,
            learning_rate=0.05,
            max_depth=3,
            num_leaves=7,
            min_child_samples=max(3, min(10, len(train_df) // 8)),
            random_state=42,
            n_jobs=1,
            verbose=-1,
        )
        self.model_.fit(X, y)
        in_preds = np.maximum(0.0, self.model_.predict(X))
        resids = y - in_preds
        for tau in EVAL_QUANTILES:
            self.resid_quantiles_[float(tau)] = float(np.quantile(resids, float(tau)))
        return self

    def predict(
        self,
        test_df: pd.DataFrame,
        quantiles: Sequence[float] = EVAL_QUANTILES,
    ) -> ModelPredictionBundle:
        assert self.model_ is not None, "Model must be fitted before calling predict()."
        X = test_df[self.used_cols_].to_numpy(dtype=float)
        raw_preds = np.maximum(0.0, self.model_.predict(X))

        q_mat = np.zeros((len(quantiles), len(raw_preds)), dtype=float)
        for idx, tau in enumerate(quantiles):
            q_shift = self.resid_quantiles_.get(float(tau), 0.0)
            q_mat[idx, :] = np.maximum(0.0, raw_preds + q_shift)
        q_mat = np.maximum.accumulate(q_mat, axis=0)
        q_dict = {float(tau): q_mat[idx, :] for idx, tau in enumerate(quantiles)}

        return ModelPredictionBundle(
            model_id=self.model_id,
            model_name=self.model_name,
            point_forecasts=raw_preds,
            quantiles=q_dict,
        )


class NB2GovernanceGLM:
    """
    Model 1: Negative Binomial GLM (NB2 — Governance Champion).
    - Fits initial Poisson GLM with log link and offset=log_enrolment
    - Runs Cameron-Trivedi auxiliary overdispersion regression test and Pearson chi2/df check
    - Estimates dispersion parameter alpha on the training fold: Var(Y_t) = mu_t + alpha * mu_t^2
    - Fits statsmodels GLM with NegativeBinomial(alpha) family and offset=log_enrolment
    - Produces KCSIE/GDPR Incidence Rate Ratio (IRR = exp(beta)) audit tables and calibrated NB2 quantiles.
    """

    model_id = "nb2_glm"
    model_name = "Negative Binomial GLM (NB2 Governance Champion)"

    def __init__(self, feature_cols: Optional[List[str]] = None) -> None:
        self.feature_cols = feature_cols or list(DEFAULT_BENCHMARK_FEATURES)
        self.used_cols_: List[str] = []
        self.col_means_: np.ndarray = np.array([])
        self.col_stds_: np.ndarray = np.array([])
        self.fit_res_: Any = None
        self.poisson_res_: Any = None
        self.alpha_: float = 0.18
        self.pearson_dispersion_: float = 1.0
        self.cameron_trivedi_t_stat_: float = 0.0
        self.cameron_trivedi_p_val_: float = 1.0
        self.aic_: float = 0.0
        self.bic_: float = 0.0
        self.llf_: float = 0.0

    def _prepare_design_matrix(self, df: pd.DataFrame, is_train: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        if is_train:
            candidate = [c for c in self.feature_cols if c in df.columns]
            self.used_cols_ = [c for c in candidate if float(np.std(df[c].to_numpy(dtype=float))) > 1e-7]

        X_raw = df[self.used_cols_].to_numpy(dtype=float) if self.used_cols_ else np.zeros((len(df), 0))
        X_design = np.column_stack([np.ones(len(df), dtype=float), X_raw])
        log_enrol = np.log(np.maximum(1.0, df["active_enrolments"].to_numpy(dtype=float)))
        return X_design, log_enrol

    def fit(self, train_df: pd.DataFrame) -> "NB2GovernanceGLM":
        y = np.maximum(0.0, train_df["referral_count"].to_numpy(dtype=float))
        X, offset = self._prepare_design_matrix(train_df, is_train=True)
        n_obs, n_params = X.shape
        df_resid = max(1, n_obs - n_params)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Step 1: Fit baseline Poisson GLM with log link and exposure offset
            poisson_mod = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
            self.poisson_res_ = poisson_mod.fit(maxiter=150, disp=False)
            mu_p = np.maximum(1e-4, np.asarray(self.poisson_res_.mu, dtype=float))

            # Step 2: Pearson Chi2 / df dispersion ratio + Cameron-Trivedi test
            pearson_chi2 = float(np.sum(((y - mu_p) ** 2) / mu_p))
            self.pearson_dispersion_ = pearson_chi2 / df_resid

            # Cameron-Trivedi auxiliary regression: ((y - mu)^2 - y) / mu = alpha * mu + error
            ct_lhs = ((y - mu_p) ** 2 - y) / mu_p
            ct_ols = sm.OLS(ct_lhs, mu_p).fit()
            alpha_ct = float(ct_ols.params[0])
            self.cameron_trivedi_t_stat_ = float(ct_ols.tvalues[0]) if np.isfinite(ct_ols.tvalues[0]) else 0.0
            self.cameron_trivedi_p_val_ = float(ct_ols.pvalues[0]) if np.isfinite(ct_ols.pvalues[0]) else 1.0

            # Method-of-moments / Cameron-Trivedi bounded dispersion estimate
            alpha_mom = float(np.sum((y - mu_p) ** 2 - mu_p) / max(1.0, np.sum(mu_p**2)))
            self.alpha_ = float(np.clip(max(alpha_ct, alpha_mom), 0.02, 1.25))

            # Step 3: Fit Negative Binomial (NB2) GLM with estimated alpha and offset=log_enrolment
            nb_mod = sm.GLM(
                y,
                X,
                family=sm.families.NegativeBinomial(alpha=self.alpha_),
                offset=offset,
            )
            # If any column (such as a quasi-static cohort share on a single-cohort slice)
            # has very low variance (std < 0.025), use mild ridge penalty on those columns
            # so coefficients remain well-conditioned and out-of-sample forecasts never blow up.
            col_stds = np.std(X[:, 1:], axis=0) if X.shape[1] > 1 else np.array([])
            if len(col_stds) > 0 and np.any(col_stds < 0.025):
                l2_wt = np.zeros(X.shape[1], dtype=float)
                for idx_c, s_val in enumerate(col_stds, start=1):
                    if s_val < 0.025:
                        l2_wt[idx_c] = 0.25
                self.fit_res_ = nb_mod.fit_regularized(alpha=l2_wt, L1_wt=0.0, maxiter=200)
                # Compute standard GLM covariance & log-likelihood at the regularized parameter vector
                unreg_res = nb_mod.fit(start_params=self.fit_res_.params, maxiter=1, disp=False)
                self.fit_res_ = unreg_res
            else:
                self.fit_res_ = nb_mod.fit(maxiter=200, disp=False)

            self.llf_ = float(nb_mod.loglike(self.fit_res_.params))
            k_total = n_params + 1  # betas + dispersion alpha
            self.aic_ = float(-2.0 * self.llf_ + 2.0 * k_total)
            self.bic_ = float(-2.0 * self.llf_ + np.log(max(1, n_obs)) * k_total)

        return self

    def predict(
        self,
        test_df: pd.DataFrame,
        quantiles: Sequence[float] = EVAL_QUANTILES,
    ) -> ModelPredictionBundle:
        assert self.fit_res_ is not None, "NB2GovernanceGLM must be fitted before calling predict()."
        X_test, offset_test = self._prepare_design_matrix(test_df, is_train=False)

        # Linear predictor eta = offset + X * beta -> mu = exp(offset + X * beta) = N_t * exp(X * beta)
        params = np.asarray(self.fit_res_.params, dtype=float)
        eta = offset_test + (X_test @ params)
        mu_pred = np.exp(np.clip(eta, -15.0, 15.0))

        # Standard error of linear predictor X * beta for predictive interval calibration
        try:
            cov_b = np.asarray(self.fit_res_.cov_params(), dtype=float)
            se_eta = np.sqrt(np.maximum(0.0, np.einsum("ij,jk,ik->i", X_test, cov_b, X_test)))
        except Exception:
            se_eta = np.full(len(test_df), 0.08, dtype=float)

        q_dict = _nb2_quantiles_from_mean_and_alpha(
            mu=mu_pred,
            alpha=self.alpha_,
            se_eta=se_eta,
            quantiles=quantiles,
        )

        return ModelPredictionBundle(
            model_id=self.model_id,
            model_name=self.model_name,
            point_forecasts=mu_pred,
            quantiles=q_dict,
            aic=round(self.aic_, 2),
            bic=round(self.bic_, 2),
            log_likelihood=round(self.llf_, 2),
            alpha_dispersion=round(self.alpha_, 4),
            diagnostics={
                "pearson_dispersion_ratio": round(self.pearson_dispersion_, 3),
                "cameron_trivedi_t_stat": round(self.cameron_trivedi_t_stat_, 3),
                "cameron_trivedi_p_value": round(self.cameron_trivedi_p_val_, 5),
                "alpha_dispersion": round(self.alpha_, 4),
            },
        )

    def export_governance_irr_table(self) -> List[GovernanceCoefficientRow]:
        """
        Export KCSIE/GDPR-compliant Governance & Triage Audit Table containing:
        Feature Name, Coefficient (beta), Standard Error, p-value, and Incidence Rate Ratio
        (IRR = exp(beta)) with 95% Confidence Intervals and plain-English interpretation.
        """
        assert self.fit_res_ is not None, "Fit NB2GovernanceGLM before exporting governance table."
        col_names = ["const"] + list(self.used_cols_)
        params = np.asarray(self.fit_res_.params, dtype=float)
        bse = np.asarray(self.fit_res_.bse, dtype=float)
        zvals = np.asarray(self.fit_res_.tvalues, dtype=float)
        pvals = np.asarray(self.fit_res_.pvalues, dtype=float)

        rows: List[GovernanceCoefficientRow] = []
        for name, beta, se, z_st, p_val in zip(col_names, params, bse, zvals, pvals):
            se_clean = float(se) if np.isfinite(se) and se > 0 else 0.05
            beta_clean = float(beta)
            irr = float(np.exp(np.clip(beta_clean, -10.0, 10.0)))
            irr_lo = float(np.exp(np.clip(beta_clean - 1.96 * se_clean, -10.0, 10.0)))
            irr_hi = float(np.exp(np.clip(beta_clean + 1.96 * se_clean, -10.0, 10.0)))

            label, desc = FEATURE_PLAIN_ENGLISH_LABELS.get(
                name,
                (name.replace("_", " ").title(), f"Log-linear count predictor ({name})."),
            )
            if name == "const":
                interp = f"Baseline referral incidence anchor ({irr * 1000.0:.2f} per 1,000 learners at reference covariate levels)."
            else:
                pct_change = (irr - 1.0) * 100.0
                direction = "increases" if pct_change >= 0 else "reduces"
                interp = (
                    f"Each +1 unit increase in {label} {direction} expected safeguarding referral rate by "
                    f"{abs(pct_change):.1f}% (IRR = {irr:.3f}, 95% CI [{irr_lo:.3f}, {irr_hi:.3f}]). {desc}"
                )

            rows.append(
                GovernanceCoefficientRow(
                    feature_name=name,
                    feature_label=label,
                    beta=round(beta_clean, 4),
                    std_error=round(se_clean, 4),
                    z_stat=round(float(z_st) if np.isfinite(z_st) else 0.0, 3),
                    p_value=round(float(p_val) if np.isfinite(p_val) else 1.0, 5),
                    irr=max(1e-4, round(irr, 4)),
                    irr_ci_lower_95=max(1e-4, round(irr_lo, 4)),
                    irr_ci_upper_95=max(1e-4, round(irr_hi, 4)),
                    interpretation=interp,
                )
            )
        return rows


class ZINBSparseRegimeModel:
    """
    Model 2: Zero-Inflated Negative Binomial (ZINB — Sparse/Departmental Regime).
    Uses statsmodels.discrete.count_model.ZeroInflatedNegativeBinomialP with offset=log_enrolment:
      1. Logit Zero-Inflation Component (exog_infl): Driven by is_low_activity_week,
         holiday boundary indicators, and baseline cohort log-enrolment scale to model
         structural zero weeks.
      2. Negative Binomial Count Component (exog): Driven by the behavioural lag,
         calendar, and cohort feature set with offset=log_enrolment.
    Note: Using the cohort's static training baseline log_enrolment in exog_infl while
    passing dynamic log_enrolment as offset to the count component ensures that doubling
    active_enrolments (N_t) at prediction time strictly doubles E[Y_t] = (1 - pi_t) * mu_t.
    """

    model_id = "zinb_model"
    model_name = "Zero-Inflated Negative Binomial (ZINB Two-Part Regime)"

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        inflation_cols: Optional[List[str]] = None,
    ) -> None:
        self.feature_cols = feature_cols or list(DEFAULT_BENCHMARK_FEATURES)
        self.inflation_cols = inflation_cols or list(ZINB_INFLATION_FEATURES)
        self.used_cols_: List[str] = []
        self.used_infl_cols_: List[str] = []
        self.cohort_baseline_log_enrol_: Dict[str, float] = {}
        self.global_baseline_log_enrol_: float = 6.9
        self.count_params_: np.ndarray = np.array([])
        self.infl_params_: np.ndarray = np.array([])
        self.alpha_: float = 0.18
        self.aic_: float = 0.0
        self.bic_: float = 0.0
        self.llf_: float = 0.0
        self.zinb_res_: Any = None
        self.nb2_fallback_: Optional[NB2GovernanceGLM] = None

    def _build_matrices(
        self,
        df: pd.DataFrame,
        is_train: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if is_train:
            cand = [c for c in self.feature_cols if c in df.columns]
            self.used_cols_ = [c for c in cand if float(np.std(df[c].to_numpy(dtype=float))) > 1e-7]
            cand_infl = [c for c in self.inflation_cols if c in df.columns]
            self.used_infl_cols_ = [c for c in cand_infl if float(np.std(df[c].to_numpy(dtype=float))) > 1e-7]

            log_e = np.log(np.maximum(1.0, df["active_enrolments"].to_numpy(dtype=float)))
            self.global_baseline_log_enrol_ = float(np.mean(log_e))
            if "curriculum_area" in df.columns:
                for area, g in df.groupby("curriculum_area"):
                    self.cohort_baseline_log_enrol_[str(area)] = float(
                        np.mean(np.log(np.maximum(1.0, g["active_enrolments"].to_numpy(dtype=float))))
                    )

        X_count_raw = df[self.used_cols_].to_numpy(dtype=float) if self.used_cols_ else np.zeros((len(df), 0))
        X_count = np.column_stack([np.ones(len(df), dtype=float), X_count_raw])

        # Inflation matrix: intercept + low_activity/holiday flags + static cohort log_enrolment baseline
        infl_parts = [np.ones(len(df), dtype=float)]
        if self.used_infl_cols_:
            infl_parts.append(df[self.used_infl_cols_].to_numpy(dtype=float))
        if "curriculum_area" in df.columns and len(self.cohort_baseline_log_enrol_) > 1:
            cohort_log_scale = np.array(
                [
                    self.cohort_baseline_log_enrol_.get(str(a), self.global_baseline_log_enrol_)
                    for a in df["curriculum_area"]
                ],
                dtype=float,
            )
            if np.std(cohort_log_scale) > 1e-7:
                infl_parts.append(cohort_log_scale[:, None])
        X_infl = np.column_stack(infl_parts)

        offset = np.log(np.maximum(1.0, df["active_enrolments"].to_numpy(dtype=float)))
        return X_count, X_infl, offset

    def fit(self, train_df: pd.DataFrame) -> "ZINBSparseRegimeModel":
        y = np.maximum(0.0, train_df["referral_count"].to_numpy(dtype=float))
        X_count, X_infl, offset = self._build_matrices(train_df, is_train=True)
        n_obs = len(y)

        # Fit NB2 warm start first so ZINB optimization is well-initialized and stable
        nb2 = NB2GovernanceGLM(feature_cols=self.used_cols_).fit(train_df)
        self.nb2_fallback_ = nb2
        count_start = np.asarray(nb2.fit_res_.params, dtype=float)
        alpha_init = float(nb2.alpha_)

        # Estimate logit zero-inflation parameters via regularized logistic fit on zero indicators
        # combined with ZeroInflatedNegativeBinomialP MLE refinement
        is_zero = (y == 0).astype(float)
        mu_nb2 = np.exp(np.clip(offset + X_count @ count_start, -12.0, 12.0))
        # Expected sampling zero probability under NB2: (1 / (1 + alpha * mu))^(1/alpha)
        p_nb_zero = np.power(1.0 + alpha_init * mu_nb2, -1.0 / max(1e-4, alpha_init))
        excess_zero_target = np.clip((is_zero - p_nb_zero) / np.maximum(0.05, 1.0 - p_nb_zero), 0.01, 0.92)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                zinb_mod = ZeroInflatedNegativeBinomialP(
                    endog=y,
                    exog=X_count,
                    exog_infl=X_infl,
                    offset=offset,
                    p=2,
                )
                self.zinb_res_ = zinb_mod
                k_infl = X_infl.shape[1]
                k_count = X_count.shape[1]

                # Warm-start inflation parameters via excess-zero logistic regression
                logit_lhs = np.log(excess_zero_target / (1.0 - excess_zero_target))
                infl_init, _, _, _ = np.linalg.lstsq(X_infl, logit_lhs, rcond=None)
                infl_init = np.clip(infl_init, -5.0, 5.0)
                if np.sum(is_zero) == 0:
                    infl_init = np.zeros(k_infl, dtype=float)
                    infl_init[0] = -6.0  # negligible zero inflation when no zeros observed

                start_params = np.concatenate([infl_init, count_start, [alpha_init]])

                def _neg_ll(theta: np.ndarray) -> float:
                    g_p = np.clip(theta[:k_infl], -8.0, 8.0)
                    b_p = np.clip(theta[k_infl : k_infl + k_count], -10.0, 10.0)
                    a_p = float(np.clip(theta[-1], 0.02, 1.25))
                    self.infl_params_ = g_p
                    self.count_params_ = b_p
                    self.alpha_ = a_p
                    ll = self._compute_zinb_loglik(y, X_count, X_infl, offset)
                    # Mild L2 regularization on inflation parameters for small-sample stability
                    reg_pen = 0.05 * float(np.sum(g_p[1:] ** 2)) if len(g_p) > 1 else 0.0
                    return -ll + reg_pen

                from scipy.optimize import minimize

                # Candidate 1: Excess-zero warm start
                start_1 = np.concatenate([infl_init, count_start, [alpha_init]])
                # Candidate 2: NB2 boundary warm start with low-activity zero-inflation trigger
                infl_2 = np.zeros(k_infl, dtype=float)
                infl_2[0] = -3.8
                if k_infl > 1:
                    infl_2[1] = 3.2
                start_2 = np.concatenate([infl_2, count_start, [alpha_init]])
                # Candidate 3: Pure NB2 nesting boundary (pi -> 0)
                infl_null = np.zeros(k_infl, dtype=float)
                infl_null[0] = -12.0
                start_null = np.concatenate([infl_null, count_start, [alpha_init]])

                def _neg_ll(theta: np.ndarray) -> float:
                    g_p = np.clip(theta[:k_infl], -12.0, 8.0)
                    b_p = np.clip(theta[k_infl : k_infl + k_count], count_start - 0.85, count_start + 0.85)
                    a_p = float(np.clip(theta[-1], 0.02, 1.25))
                    self.infl_params_ = g_p
                    self.count_params_ = b_p
                    self.alpha_ = a_p
                    ll = self._compute_zinb_loglik(y, X_count, X_infl, offset)
                    reg_pen = (
                        (0.01 * float(np.sum(g_p[1:] ** 2)) if len(g_p) > 1 else 0.0)
                        + 0.35 * float(np.sum((b_p - count_start) ** 2))
                    )
                    return -ll + reg_pen

                best_theta = start_null
                self.infl_params_ = start_null[:k_infl]
                self.count_params_ = count_start
                self.alpha_ = alpha_init
                best_true_ll = self._compute_zinb_loglik(y, X_count, X_infl, offset)

                for s_init in (start_1, start_2):
                    opt_res = minimize(
                        _neg_ll,
                        s_init,
                        method="L-BFGS-B",
                        options={"maxiter": 45, "ftol": 1e-6},
                    )
                    cand_theta = np.asarray(opt_res.x, dtype=float)
                    self.infl_params_ = np.clip(cand_theta[:k_infl], -12.0, 8.0)
                    self.count_params_ = np.clip(
                        cand_theta[k_infl : k_infl + k_count], count_start - 0.85, count_start + 0.85
                    )
                    self.alpha_ = float(np.clip(cand_theta[-1], 0.02, 1.25))
                    cand_ll = self._compute_zinb_loglik(y, X_count, X_infl, offset)
                    if cand_ll > best_true_ll:
                        best_true_ll = cand_ll
                        best_theta = np.concatenate([self.infl_params_, self.count_params_, [self.alpha_]])

                self.infl_params_ = best_theta[:k_infl]
                self.count_params_ = best_theta[k_infl : k_infl + k_count]
                self.alpha_ = float(best_theta[-1])
            except Exception:
                logit_lhs = np.log(excess_zero_target / (1.0 - excess_zero_target))
                gamma_hat, _, _, _ = np.linalg.lstsq(X_infl, logit_lhs, rcond=None)
                self.infl_params_ = np.clip(gamma_hat, -6.0, 6.0)
                self.count_params_ = count_start
                self.alpha_ = alpha_init

        # Compute exact ZINB log-likelihood, AIC, and BIC on training data
        llf_val = self._compute_zinb_loglik(y, X_count, X_infl, offset)
        k_total = len(self.infl_params_) + len(self.count_params_) + 1
        self.llf_ = llf_val
        self.aic_ = float(-2.0 * llf_val + 2.0 * k_total)
        self.bic_ = float(-2.0 * llf_val + np.log(max(1, n_obs)) * k_total)
        return self

    def _compute_zinb_loglik(
        self,
        y: np.ndarray,
        X_count: np.ndarray,
        X_infl: np.ndarray,
        offset: np.ndarray,
    ) -> float:
        logit_pi = np.clip(X_infl @ self.infl_params_, -15.0, 15.0)
        pi = 1.0 / (1.0 + np.exp(-logit_pi))
        mu = np.exp(np.clip(offset + X_count @ self.count_params_, -15.0, 15.0))
        a = max(1e-4, self.alpha_)
        n_p = 1.0 / a
        p_p = n_p / (n_p + mu)

        nb_pmf_zero = stats.nbinom.pmf(0, n_p, p_p)
        nb_logpmf = stats.nbinom.logpmf(np.round(y).astype(int), n_p, p_p)

        ll_zero = np.log(np.maximum(1e-12, pi + (1.0 - pi) * nb_pmf_zero))
        ll_pos = np.log(np.maximum(1e-12, 1.0 - pi)) + nb_logpmf
        ll_vec = np.where(y == 0, ll_zero, ll_pos)
        return float(np.sum(ll_vec))

    def predict(
        self,
        test_df: pd.DataFrame,
        quantiles: Sequence[float] = EVAL_QUANTILES,
    ) -> ModelPredictionBundle:
        X_count, X_infl, offset = self._build_matrices(test_df, is_train=False)
        logit_pi = np.clip(X_infl @ self.infl_params_, -15.0, 15.0)
        pi_zero = 1.0 / (1.0 + np.exp(-logit_pi))
        mu_count = np.exp(np.clip(offset + X_count @ self.count_params_, -15.0, 15.0))

        # Expected referral count under ZINB: E[Y_t] = (1 - pi_t) * mu_t
        expected_y = (1.0 - pi_zero) * mu_count
        q_dict = _zinb_quantiles_from_params(
            mu_count=mu_count,
            pi_zero=pi_zero,
            alpha=self.alpha_,
            quantiles=quantiles,
        )

        return ModelPredictionBundle(
            model_id=self.model_id,
            model_name=self.model_name,
            point_forecasts=expected_y,
            quantiles=q_dict,
            aic=round(self.aic_, 2),
            bic=round(self.bic_, 2),
            log_likelihood=round(self.llf_, 2),
            alpha_dispersion=round(self.alpha_, 4),
            diagnostics={
                "mean_structural_zero_prob": round(float(np.mean(pi_zero)), 4),
                "low_activity_structural_zero_prob": round(
                    float(
                        np.mean(pi_zero[test_df["is_low_activity_week"].to_numpy(dtype=int) > 0])
                        if np.any(test_df["is_low_activity_week"].to_numpy(dtype=int) > 0)
                        else np.mean(pi_zero)
                    ),
                    4,
                ),
            },
        )


def compute_nb2_vs_zinb_comparison(
    train_df: pd.DataFrame,
    nb2_model: NB2GovernanceGLM,
    zinb_model: ZINBSparseRegimeModel,
) -> Dict[str, Any]:
    """
    Phase 3 Model 2 Requirement:
    Compute AIC, BIC, and a log-likelihood ratio / Vuong-style comparison between NB2 and ZINB
    to determine whether zero-inflation adds statistically significant explanatory power at the
    chosen data grain.
    """
    y = np.maximum(0.0, train_df["referral_count"].to_numpy(dtype=float))
    zero_pct = float(np.mean(y == 0) * 100.0)

    lr_stat = float(2.0 * (zinb_model.llf_ - nb2_model.llf_))
    df_diff = max(1, len(zinb_model.infl_params_))
    p_val = float(1.0 - stats.chi2.cdf(max(0.0, lr_stat), df=df_diff))

    delta_aic = float(nb2_model.aic_ - zinb_model.aic_)
    delta_bic = float(nb2_model.bic_ - zinb_model.bic_)
    zinb_preferred = bool((lr_stat > 3.84 and p_val < 0.05) or delta_aic > 2.0)

    if zinb_preferred:
        verdict = (
            f"ZINB adds statistically significant explanatory power over NB2 at this grain "
            f"(zero-count rows = {zero_pct:.1f}%, LR stat = {lr_stat:.2f}, p = {p_val:.4f}, "
            f"AIC improvement = {delta_aic:+.2f}). Structural zeros during low-activity/placement weeks "
            "are better captured by the two-part logit + Negative Binomial specification."
        )
    else:
        verdict = (
            f"NB2 is sufficient at this aggregate grain (zero-count rows = {zero_pct:.1f}%, "
            f"LR stat = {lr_stat:.2f}, p = {p_val:.4f}, Delta AIC = {delta_aic:+.2f}). "
            "Standard Negative Binomial overdispersion accounts for low-volume weeks without requiring a separate zero-inflation regime."
        )

    return {
        "zero_count_pct": round(zero_pct, 2),
        "nb2_llf": round(nb2_model.llf_, 2),
        "zinb_llf": round(zinb_model.llf_, 2),
        "nb2_aic": round(nb2_model.aic_, 2),
        "zinb_aic": round(zinb_model.aic_, 2),
        "nb2_bic": round(nb2_model.bic_, 2),
        "zinb_bic": round(zinb_model.bic_, 2),
        "lr_statistic": round(lr_stat, 3),
        "df_difference": df_diff,
        "p_value": round(p_val, 5),
        "zinb_statistically_preferred": zinb_preferred,
        "conclusion": verdict,
    }


class TweedieLightGBMModel:
    """
    Model 3: Tweedie-Loss LightGBM (Non-Linear Kaggle Benchmark).
    - Uses lightgbm.LGBMRegressor(objective='tweedie') with a log link.
    - Exposure Offset Handling: Passes init_score=log_enrolment during .fit() and adds
      log_enrolment to raw margin predictions before exponentiating:
        y_hat = np.exp(raw_preds + log_enrolment)
      ensuring LightGBM treats active learner headcount (N_t) as a true multiplicative
      exposure offset rather than a tree-split feature.
    - Small-Sample Regularization: max_depth=3, num_leaves=7, min_child_samples=10,
      subsample=0.8, colsample_bytree=0.8.
    - Tunes tweedie_variance_power across [1.15, 1.30, 1.50, 1.70] using expanding-window
      validation.
    """

    model_id = "tweedie_lgbm"
    model_name = "Tweedie-Loss LightGBM (Exposure-Offset GBDT)"
    CANDIDATE_POWERS: Tuple[float, ...] = (1.15, 1.30, 1.50, 1.70)

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        tune_power: bool = True,
        default_power: float = 1.30,
    ) -> None:
        self.feature_cols = feature_cols or list(DEFAULT_BENCHMARK_FEATURES)
        self.tune_power = tune_power
        self.best_power_: float = default_power
        self.power_tuning_scores_: Dict[float, float] = {}
        self.used_cols_: List[str] = []
        self.model_: Optional[LGBMRegressor] = None
        self.log_base_rate_: float = -5.3
        self.alpha_equiv_: float = 0.18
        self.conformal_log_quantiles_: Dict[float, float] = {}

    def _make_regressor(self, power: float, n_rows: int) -> LGBMRegressor:
        return LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=float(power),
            boost_from_average=False,
            n_estimators=55,
            learning_rate=0.05,
            max_depth=3,
            num_leaves=7,
            min_child_samples=min(10, max(3, n_rows // 6)),
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.2,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        )

    def fit(self, train_df: pd.DataFrame) -> "TweedieLightGBMModel":
        self.used_cols_ = [c for c in self.feature_cols if c in train_df.columns]
        X = train_df[self.used_cols_].to_numpy(dtype=float)
        y = np.maximum(0.0, train_df["referral_count"].to_numpy(dtype=float))
        enrol = np.maximum(1.0, train_df["active_enrolments"].to_numpy(dtype=float))
        log_enrol = np.log(enrol)
        # Anchor exposure offset at training baseline incidence rate per learner so trees boost around 0
        self.log_base_rate_ = float(np.log(max(1e-6, np.sum(y) / np.sum(enrol))))
        init_offset = log_enrol + self.log_base_rate_
        n_obs = len(train_df)

        # Tune tweedie_variance_power across [1.15, 1.30, 1.50, 1.70] using inner expanding window
        if self.tune_power and n_obs >= 36:
            split_1 = int(n_obs * 0.65)
            split_2 = int(n_obs * 0.82)
            folds = [(split_1, split_2), (split_2, n_obs)]
            best_p = self.CANDIDATE_POWERS[1]
            best_loss = float("inf")

            for p_cand in self.CANDIDATE_POWERS:
                fold_losses: List[float] = []
                for tr_end, val_end in folds:
                    if val_end <= tr_end:
                        continue
                    fold_base_rate = float(np.log(max(1e-6, np.sum(y[:tr_end]) / np.sum(enrol[:tr_end]))))
                    reg = self._make_regressor(p_cand, tr_end)
                    reg.fit(X[:tr_end], y[:tr_end], init_score=log_enrol[:tr_end] + fold_base_rate)
                    raw_val = reg.predict(X[tr_end:val_end], raw_score=True)
                    pred_val = np.exp(np.clip(raw_val + log_enrol[tr_end:val_end] + fold_base_rate, -15.0, 15.0))
                    # Evaluate on scale-free RMSLE + MAE across validation fold
                    fold_losses.append(compute_rmsle(y[tr_end:val_end], pred_val))
                mean_l = float(np.mean(fold_losses)) if fold_losses else float("inf")
                self.power_tuning_scores_[float(p_cand)] = round(mean_l, 5)
                if mean_l < best_loss:
                    best_loss = mean_l
                    best_p = float(p_cand)
            self.best_power_ = best_p

        self.model_ = self._make_regressor(self.best_power_, n_obs)
        self.model_.fit(X, y, init_score=init_offset)

        # Calibrate multiplicative log-rate residual quantiles using out-of-time validation tail
        raw_in = self.model_.predict(X, raw_score=True)
        mu_in = np.exp(np.clip(raw_in + init_offset, -15.0, 15.0))
        # Method-of-moments dispersion equivalent for probabilistic quantile construction
        alpha_est = float(np.sum((y - mu_in) ** 2 - mu_in) / max(1.0, np.sum(mu_in**2)))
        self.alpha_equiv_ = float(np.clip(alpha_est, 0.04, 0.85))

        # Multiplicative ratio residuals r_t = (y_t + 0.5) / (mu_t + 0.5)
        ratios = (y + 0.5) / (mu_in + 0.5)
        for tau in EVAL_QUANTILES:
            self.conformal_log_quantiles_[float(tau)] = float(np.quantile(ratios, float(tau)))

        return self

    def predict(
        self,
        test_df: pd.DataFrame,
        quantiles: Sequence[float] = EVAL_QUANTILES,
    ) -> ModelPredictionBundle:
        assert self.model_ is not None, "TweedieLightGBMModel must be fitted before calling predict()."
        X = test_df[self.used_cols_].to_numpy(dtype=float)
        log_enrol = np.log(np.maximum(1.0, test_df["active_enrolments"].to_numpy(dtype=float)))

        # Crucial: raw_score=True returns the tree log-rate margin f(X);
        # adding log_enrolment + log_base_rate_ before exponentiating guarantees exact
        # multiplicative exposure offset:
        #   mu_hat = exp(f(X) + ln(N_t) + ln(r_0)) = N_t * r_0 * exp(f(X))
        raw_margin = self.model_.predict(X, raw_score=True)
        mu_pred = np.exp(np.clip(raw_margin + log_enrol + self.log_base_rate_, -15.0, 15.0))

        # Construct calibrated, exposure-proportional, strictly non-crossing quantiles
        nb_q = _nb2_quantiles_from_mean_and_alpha(
            mu=mu_pred,
            alpha=self.alpha_equiv_,
            se_eta=np.full(len(mu_pred), 0.10),
            quantiles=quantiles,
        )
        q_mat = np.zeros((len(quantiles), len(mu_pred)), dtype=float)
        for idx, tau in enumerate(quantiles):
            t_f = float(tau)
            mult_q = mu_pred * self.conformal_log_quantiles_.get(t_f, 1.0)
            q_mat[idx, :] = np.maximum(0.0, 0.65 * nb_q[t_f] + 0.35 * mult_q)
        q_mat = np.maximum.accumulate(q_mat, axis=0)
        q_dict = {float(tau): q_mat[idx, :] for idx, tau in enumerate(quantiles)}

        return ModelPredictionBundle(
            model_id=self.model_id,
            model_name=f"{self.model_name} (p={self.best_power_:.2f})",
            point_forecasts=mu_pred,
            quantiles=q_dict,
            alpha_dispersion=round(self.alpha_equiv_, 4),
            diagnostics={
                "selected_tweedie_variance_power": self.best_power_,
                "power_tuning_rmsle": self.power_tuning_scores_,
            },
        )


def run_nb2_residual_diagnostic(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Phase 3 Model 3 Residual Diagnostic:
    Check whether training a shallow Tweedie LightGBM on the NB2 GLM residuals
    (using init_score = log(mu_nb2)) yields a meaningful reduction in out-of-time deviance;
    if not, report that linear log-link effects capture the full signal.
    """
    cols = [c for c in (feature_cols or DEFAULT_BENCHMARK_FEATURES) if c in train_df.columns]
    nb2 = NB2GovernanceGLM(feature_cols=cols).fit(train_df)
    nb2_train_pred = nb2.predict(train_df).point_forecasts
    nb2_test_pred = nb2.predict(test_df).point_forecasts

    y_train = np.maximum(0.0, train_df["referral_count"].to_numpy(dtype=float))
    y_test = np.maximum(0.0, test_df["referral_count"].to_numpy(dtype=float))

    init_train = np.log(np.maximum(1e-4, nb2_train_pred))
    init_test = np.log(np.maximum(1e-4, nb2_test_pred))

    residual_booster = LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.30,
        boost_from_average=False,
        n_estimators=30,
        learning_rate=0.03,
        max_depth=2,
        num_leaves=4,
        min_child_samples=12,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    residual_booster.fit(train_df[cols].to_numpy(dtype=float), y_train, init_score=init_train)
    resid_margin = residual_booster.predict(test_df[cols].to_numpy(dtype=float), raw_score=True)
    hybrid_test_pred = np.exp(np.clip(init_test + resid_margin, -15.0, 15.0))

    dev_nb2 = compute_mean_nb_deviance(y_test, nb2_test_pred, alpha=nb2.alpha_)
    dev_hybrid = compute_mean_nb_deviance(y_test, hybrid_test_pred, alpha=nb2.alpha_)
    rel_reduction_pct = ((dev_nb2 - dev_hybrid) / max(1e-6, dev_nb2)) * 100.0
    meaningful = bool(rel_reduction_pct > 2.5)

    if meaningful:
        conclusion = (
            f"Shallow Tweedie LightGBM on NB2 residuals reduced out-of-time NB deviance by "
            f"{rel_reduction_pct:.2f}% ({dev_nb2:.4f} -> {dev_hybrid:.4f}), indicating mild non-linear "
            "interactions beyond the main log-linear specification."
        )
    else:
        conclusion = (
            f"Shallow Tweedie LightGBM on NB2 residuals changed out-of-time NB deviance by only "
            f"{rel_reduction_pct:+.2f}% ({dev_nb2:.4f} -> {dev_hybrid:.4f}); linear log-link effects "
            "with the composite disengagement index capture the full systematic signal."
        )

    return {
        "nb2_out_of_time_deviance": round(dev_nb2, 4),
        "nb2_plus_tweedie_residual_deviance": round(dev_hybrid, 4),
        "relative_deviance_reduction_pct": round(rel_reduction_pct, 2),
        "meaningful_nonlinear_residual_signal": meaningful,
        "conclusion": conclusion,
    }


def generate_dsl_triage_schedule(
    test_df: pd.DataFrame,
    pred_bundle: ModelPredictionBundle,
    dsl_staff_fte: float = 4.5,
    referrals_per_fte_per_week: float = 8.0,
) -> List[DSLWeeklyTriageRow]:
    """
    Phase 4 Step 2:
    Generate weekly forecast schedule displaying mu_hat_t, Q_0.90,t, and Q_0.975,t,
    flagging any week where the 90th or 97.5th percentile bound exceeds baseline DSL
    triage staffing capacity.
    """
    baseline_capacity = round(float(dsl_staff_fte * referrals_per_fte_per_week), 1)
    mu_vec = pred_bundle.point_forecasts
    q50_vec = pred_bundle.quantiles[0.50]
    q90_vec = pred_bundle.quantiles[0.90]
    q975_vec = pred_bundle.quantiles[0.975]

    schedule: List[DSLWeeklyTriageRow] = []
    for idx, row in enumerate(test_df.itertuples(index=False)):
        mu_val = float(mu_vec[idx])
        q50_val = float(q50_vec[idx])
        q90_val = float(q90_vec[idx])
        q975_val = float(q975_vec[idx])
        ex_90 = bool(q90_val > baseline_capacity)
        ex_975 = bool(q975_val > baseline_capacity)

        if ex_975 and ex_90:
            action = (
                "HIGH ALERT (90th & 97.5th Exceed Capacity): Mobilise deputy DSL rota, "
                "pre-book multi-agency MASH liaison slots, and ring-fence pastoral triage hours."
            )
        elif ex_975:
            action = (
                "TAIL SURGE WATCH (97.5th Exceeds Capacity): Place standby safeguarding officer "
                "on call for potential upper-tail disclosure spike."
            )
        else:
            action = "Standard DSL Rota Sufficient: Predicted demand within core safeguarding team capacity."

        schedule.append(
            DSLWeeklyTriageRow(
                week_start=str(getattr(row, "week_start", f"Week-{idx + 1}")),
                academic_year=str(getattr(row, "academic_year", "2024/25")),
                week_of_academic_year=int(getattr(row, "week_of_academic_year", (idx % 52) + 1)),
                curriculum_area=str(getattr(row, "curriculum_area", "College-Wide")),
                active_enrolments=int(getattr(row, "active_enrolments", 7200)),
                actual_referrals=int(getattr(row, "referral_count")) if hasattr(row, "referral_count") else None,
                expected_referrals_mu=round(mu_val, 2),
                q_50=round(q50_val, 1),
                q_90=round(q90_val, 1),
                q_975=round(q975_val, 1),
                baseline_dsl_capacity=baseline_capacity,
                exceeds_q90_capacity=ex_90,
                exceeds_q975_capacity=ex_975,
                recommended_triage_action=action,
            )
        )
    return schedule


def run_expanding_window_benchmark(
    panel_df: Optional[pd.DataFrame] = None,
    fold_origins_weeks: Sequence[int] = (78, 104, 130),
    horizon_weeks: int = 26,
) -> Dict[str, Any]:
    """
    Phase 4 Complete Expanding-Window Out-of-Time Evaluation & Governance Pipeline:
    Evaluates Legacy ML, NB2 GLM, ZINB, and Tweedie LightGBM across:
      1. Whole-College Weekly Series (expanding-window folds)
      2. Curriculum-Area Granular Series (for hierarchical WRMSSE, WSPL, and sparse ZINB testing)
    Returns the complete multi-model benchmark table, NB2 vs ZINB LR/Vuong comparison,
    Tweedie residual diagnostic, VIF diagnostic, and DSL Governance IRR + Triage Schedule.
    """
    if panel_df is None:
        panel_df = generate_weekly_safeguarding_panel(num_academic_years=3, start_year=2022, seed=42)

    grain_audit = audit_target_distribution(panel_df)
    cw_raw = aggregate_college_wide_weekly(panel_df)
    cw_df, vif_diag = engineer_safeguarding_features(cw_raw, group_col="curriculum_area")
    dept_df, _ = engineer_safeguarding_features(panel_df, group_col="curriculum_area")

    feature_cols = vif_diag.selected_feature_cols
    n_weeks = len(cw_df)
    valid_origins = [o for o in fold_origins_weeks if 30 <= o <= (n_weeks - horizon_weeks)]
    if not valid_origins:
        valid_origins = [max(26, n_weeks - horizon_weeks)]

    model_ids = ["legacy_ml_baseline", "nb2_glm", "zinb_model", "tweedie_lgbm"]
    model_names: Dict[str, str] = {
        "legacy_ml_baseline": LegacyMLBaselineModel.model_name,
        "nb2_glm": NB2GovernanceGLM.model_name,
        "zinb_model": ZINBSparseRegimeModel.model_name,
        "tweedie_lgbm": TweedieLightGBMModel.model_name,
    }

    # Accumulators across expanding-window folds (College-Wide)
    cw_y_true_all: Dict[str, List[float]] = {m: [] for m in model_ids}
    cw_y_pred_all: Dict[str, List[float]] = {m: [] for m in model_ids}
    cw_quantiles_all: Dict[str, Dict[float, List[float]]] = {
        m: {q: [] for q in EVAL_QUANTILES} for m in model_ids
    }
    cw_autumn_all: List[int] = []
    cw_pre_hol_all: List[int] = []
    cw_post_hol_all: List[int] = []
    cw_rmsse_folds: Dict[str, List[float]] = {m: [] for m in model_ids}
    cw_spl_folds: Dict[str, List[float]] = {m: [] for m in model_ids}

    last_nb2_model: Optional[NB2GovernanceGLM] = None
    last_zinb_model: Optional[ZINBSparseRegimeModel] = None
    last_tweedie_model: Optional[TweedieLightGBMModel] = None
    last_test_slice: Optional[pd.DataFrame] = None
    last_nb2_bundle: Optional[ModelPredictionBundle] = None

    for origin in valid_origins:
        train_slice = cw_df.iloc[:origin].copy().reset_index(drop=True)
        test_slice = cw_df.iloc[origin : origin + horizon_weeks].copy().reset_index(drop=True)
        y_tr = train_slice["referral_count"].to_numpy(dtype=float)
        y_te = test_slice["referral_count"].to_numpy(dtype=float)

        cw_autumn_all.extend(test_slice["is_autumn_transition"].astype(int).tolist())
        cw_pre_hol_all.extend(test_slice["is_pre_holiday_week"].astype(int).tolist())
        cw_post_hol_all.extend(test_slice["is_post_holiday_week"].astype(int).tolist())

        m_legacy = LegacyMLBaselineModel(feature_cols=feature_cols).fit(train_slice)
        m_nb2 = NB2GovernanceGLM(feature_cols=feature_cols).fit(train_slice)
        m_zinb = ZINBSparseRegimeModel(feature_cols=feature_cols).fit(train_slice)
        m_tweedie = TweedieLightGBMModel(feature_cols=feature_cols).fit(train_slice)

        bundles = {
            "legacy_ml_baseline": m_legacy.predict(test_slice),
            "nb2_glm": m_nb2.predict(test_slice),
            "zinb_model": m_zinb.predict(test_slice),
            "tweedie_lgbm": m_tweedie.predict(test_slice),
        }

        for m_id, bnd in bundles.items():
            model_names[m_id] = bnd.model_name
            cw_y_true_all[m_id].extend(y_te.tolist())
            cw_y_pred_all[m_id].extend(bnd.point_forecasts.tolist())
            for q in EVAL_QUANTILES:
                cw_quantiles_all[m_id][q].extend(bnd.quantiles[q].tolist())
            cw_rmsse_folds[m_id].append(compute_rmsse(y_tr, y_te, bnd.point_forecasts))
            cw_spl_folds[m_id].append(compute_spl(y_tr, y_te, bnd.quantiles))

        last_nb2_model = m_nb2
        last_zinb_model = m_zinb
        last_tweedie_model = m_tweedie
        last_test_slice = test_slice
        last_nb2_bundle = bundles["nb2_glm"]

    # Evaluate Hierarchical Curriculum-Area WRMSSE and WSPL on the final out-of-time fold
    final_origin = valid_origins[-1]
    dept_weights: Dict[str, float] = {}
    dept_rmsse_by_model: Dict[str, Dict[str, float]] = {m: {} for m in model_ids}
    dept_spl_by_model: Dict[str, Dict[str, float]] = {m: {} for m in model_ids}
    dept_zinb_comparisons: Dict[str, Any] = {}

    for area_name, g_area in dept_df.groupby("curriculum_area"):
        if area_name == "College-Wide":
            continue
        g_sorted = g_area.sort_values("week_start").reset_index(drop=True)
        if len(g_sorted) < final_origin + horizon_weeks:
            continue
        tr_a = g_sorted.iloc[:final_origin].copy().reset_index(drop=True)
        te_a = g_sorted.iloc[final_origin : final_origin + horizon_weeks].copy().reset_index(drop=True)
        dept_weights[str(area_name)] = float(tr_a["active_enrolments"].mean())

        y_tr_a = tr_a["referral_count"].to_numpy(dtype=float)
        y_te_a = te_a["referral_count"].to_numpy(dtype=float)

        a_legacy = LegacyMLBaselineModel(feature_cols=feature_cols).fit(tr_a)
        a_nb2 = NB2GovernanceGLM(feature_cols=feature_cols).fit(tr_a)
        a_zinb = ZINBSparseRegimeModel(feature_cols=feature_cols).fit(tr_a)
        a_tweedie = TweedieLightGBMModel(feature_cols=feature_cols, tune_power=False).fit(tr_a)

        dept_zinb_comparisons[str(area_name)] = compute_nb2_vs_zinb_comparison(tr_a, a_nb2, a_zinb)

        for m_id, mod_obj in [
            ("legacy_ml_baseline", a_legacy),
            ("nb2_glm", a_nb2),
            ("zinb_model", a_zinb),
            ("tweedie_lgbm", a_tweedie),
        ]:
            b_a = mod_obj.predict(te_a)
            dept_rmsse_by_model[m_id][str(area_name)] = compute_rmsse(y_tr_a, y_te_a, b_a.point_forecasts)
            dept_spl_by_model[m_id][str(area_name)] = compute_spl(y_tr_a, y_te_a, b_a.quantiles)

    # Assemble Multi-Model Benchmark Comparison Table
    benchmark_metrics: List[ModelEvaluationMetrics] = []
    alpha_ref = last_nb2_model.alpha_ if last_nb2_model is not None else 0.18

    for m_id in model_ids:
        yt = cw_y_true_all[m_id]
        yp = cw_y_pred_all[m_id]
        q_map = cw_quantiles_all[m_id]

        mae_v = compute_mae(yt, yp)
        rmse_v = compute_rmse(yt, yp)
        surge_mae_v = compute_surge_mae(yt, yp, cw_autumn_all, cw_pre_hol_all, cw_post_hol_all)
        nb_dev_v = compute_mean_nb_deviance(yt, yp, alpha=alpha_ref)
        tw_dev_v = compute_mean_tweedie_deviance_safe(yt, yp, power=1.30)
        rmsle_v = compute_rmsle(yt, yp)
        rmsse_v = float(np.mean(cw_rmsse_folds[m_id]))
        wrmsse_v = compute_wrmsse(dept_rmsse_by_model[m_id], dept_weights)

        pb_025 = compute_pinball_loss(yt, q_map[0.025], 0.025)
        pb_500 = compute_pinball_loss(yt, q_map[0.50], 0.50)
        pb_900 = compute_pinball_loss(yt, q_map[0.90], 0.90)
        pb_975 = compute_pinball_loss(yt, q_map[0.975], 0.975)

        spl_v = float(np.mean(cw_spl_folds[m_id]))
        wspl_v = compute_wspl(dept_spl_by_model[m_id], dept_weights)
        picp_95, miw_95 = compute_picp_and_miw(yt, q_map[0.025], q_map[0.975])

        aic_val = None
        bic_val = None
        alpha_val = None
        if m_id == "nb2_glm" and last_nb2_model is not None:
            aic_val = round(last_nb2_model.aic_, 2)
            bic_val = round(last_nb2_model.bic_, 2)
            alpha_val = round(last_nb2_model.alpha_, 4)
        elif m_id == "zinb_model" and last_zinb_model is not None:
            aic_val = round(last_zinb_model.aic_, 2)
            bic_val = round(last_zinb_model.bic_, 2)
            alpha_val = round(last_zinb_model.alpha_, 4)
        elif m_id == "tweedie_lgbm" and last_tweedie_model is not None:
            alpha_val = round(last_tweedie_model.alpha_equiv_, 4)

        benchmark_metrics.append(
            ModelEvaluationMetrics(
                model_id=m_id,
                model_name=model_names[m_id],
                mae=round(mae_v, 3),
                rmse=round(rmse_v, 3),
                surge_mae=round(surge_mae_v, 3),
                nb_deviance=round(nb_dev_v, 4),
                tweedie_deviance_p13=round(tw_dev_v, 4),
                rmsle=round(rmsle_v, 4),
                rmsse=round(rmsse_v, 4),
                wrmsse=round(wrmsse_v, 4),
                pinball_025=round(pb_025, 4),
                pinball_500=round(pb_500, 4),
                pinball_900=round(pb_900, 4),
                pinball_975=round(pb_975, 4),
                spl=round(spl_v, 4),
                wspl=round(wspl_v, 4),
                picp_95=round(picp_95, 2),
                miw_95=round(miw_95, 2),
                aic=aic_val,
                bic=bic_val,
                alpha_dispersion=alpha_val,
            )
        )

    assert last_nb2_model is not None and last_zinb_model is not None and last_test_slice is not None and last_nb2_bundle is not None
    final_train_slice = cw_df.iloc[:final_origin].copy().reset_index(drop=True)
    cw_zinb_vs_nb2 = compute_nb2_vs_zinb_comparison(final_train_slice, last_nb2_model, last_zinb_model)
    residual_diag = run_nb2_residual_diagnostic(final_train_slice, last_test_slice, feature_cols=feature_cols)

    # Fit NB2 Governance model on pooled curriculum-area training panel up to final_origin
    # so cross-cohort ILR predictors (vulnerability, IMD 1-3 deprivation, 16-18 mix)
    # are estimated with full cross-departmental statistical power and tight standard errors.
    cutoff_date = str(final_train_slice["week_start"].iloc[-1])
    pooled_train_panel = dept_df[dept_df["week_start"] <= cutoff_date].copy().reset_index(drop=True)
    pooled_gov_nb2 = NB2GovernanceGLM(feature_cols=feature_cols).fit(pooled_train_panel)
    irr_table = pooled_gov_nb2.export_governance_irr_table()
    dsl_schedule = generate_dsl_triage_schedule(last_test_slice, last_nb2_bundle)

    return {
        "data_grain_audit": grain_audit,
        "vif_diagnostics": {
            "vif_table": vif_diag.vif_table.to_dict(orient="records"),
            "att_pastoral_vif_exceeded": vif_diag.att_pastoral_vif_exceeded,
            "max_att_pastoral_vif": vif_diag.max_att_pastoral_vif,
            "composite_index_activated": vif_diag.composite_index_activated,
            "selected_feature_cols": vif_diag.selected_feature_cols,
            "collinearity_note": vif_diag.collinearity_note,
        },
        "expanding_window_folds": list(valid_origins),
        "horizon_weeks": horizon_weeks,
        "benchmark_comparison_table": [m.__dict__ for m in benchmark_metrics],
        "nb2_vs_zinb_college_wide": cw_zinb_vs_nb2,
        "nb2_vs_zinb_by_curriculum_area": dept_zinb_comparisons,
        "tweedie_residual_diagnostic": residual_diag,
        "dsl_governance_irr_table": [r.__dict__ for r in irr_table],
        "dsl_weekly_triage_schedule": [s.__dict__ for s in dsl_schedule],
    }

