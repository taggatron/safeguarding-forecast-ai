"""Statistical baseline and count regression models (Seasonal Naive & Negative Binomial/Poisson GLM)."""
import time
from typing import Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive
from app.models.base import (
    Z_80,
    Z_95,
    BaseForecaster,
    EngineForecastOutput,
    build_future_covariates_df,
    compute_calibrated_annual_interval,
)
from app.schemas.models import (
    ScenarioAssumptions,
    ScenarioSensitivityCapabilities,
)


class SeasonalNaiveForecaster(BaseForecaster):
    """
    Model A1: Seasonal-Naive Benchmark using Nixtla StatsForecast.
    Forecasts each future month as the observed count from the same calendar month
    in the preceding academic year, with empirical seasonal-difference prediction intervals.
    """

    model_id = "seasonal_naive"
    name = "Seasonal Naive Baseline"
    short_name = "Seasonal Naive"
    family = "Statistical Benchmark"
    version = "statsforecast-2.1"
    license_name = "Apache-2.0"
    commercial_production_permitted = True
    supports_95_pi = True
    data_requirements = "Minimum 12 months (24+ months recommended for interval calibration)"
    backend_engine = "Nixtla StatsForecast (SeasonalNaive, m=12)"

    def is_available(self, governance_mode: str = "research_prototype", license_ack: bool = True) -> Tuple[bool, str, str]:
        return (
            True,
            "Available",
            "Active statistical benchmark (StatsForecast SeasonalNaive m=12). Fully permitted for college operational use.",
        )

    def get_scenario_capabilities(self, has_attendance_data: bool = True) -> ScenarioSensitivityCapabilities:
        return ScenarioSensitivityCapabilities(
            supports_enrolment_growth=False,
            enrolment_explanation="Seasonal Naive repeats prior-year monthly counts directly and does not adjust for student population growth.",
            supports_teaching_days=False,
            teaching_days_explanation="Seasonal Naive uses only historical monthly counts and does not incorporate teaching-day changes.",
            supports_attendance_scenario=False,
            attendance_explanation="Seasonal Naive is a univariate benchmark and does not use attendance rates.",
            supports_reporting_flag=False,
        )

    def forecast(
        self,
        hist_df: pd.DataFrame,
        horizon: int = 12,
        scenario: Optional[ScenarioAssumptions] = None,
        academic_year_start_month: int = 9,
        **kwargs: Any,
    ) -> EngineForecastOutput:
        t0 = time.perf_counter()
        future_cov = build_future_covariates_df(hist_df, horizon, scenario, academic_year_start_month)

        sf_df = pd.DataFrame(
            {
                "unique_id": "sdc_series",
                "ds": pd.to_datetime(hist_df["period_start"]),
                "y": hist_df["concern_count"].astype(float),
            }
        )

        sf = StatsForecast(models=[SeasonalNaive(season_length=12)], freq="MS")
        sf.fit(sf_df)
        pred_df = sf.predict(h=horizon)
        raw_preds = np.maximum(0.0, pred_df["SeasonalNaive"].to_numpy(dtype=float))

        # Calibrate prediction intervals from historical year-over-year seasonal residuals (y_t - y_{t-12})
        y_vals = hist_df["concern_count"].to_numpy(dtype=float)
        if len(y_vals) >= 24:
            seasonal_diffs = y_vals[12:] - y_vals[:-12]
            sigma_base = float(np.std(seasonal_diffs, ddof=1))
            if len(seasonal_diffs) >= 4:
                s0 = seasonal_diffs[:-1] - np.mean(seasonal_diffs[:-1])
                s1 = seasonal_diffs[1:] - np.mean(seasonal_diffs[1:])
                denom = float(np.sqrt(np.sum(s0**2) * np.sum(s1**2)))
                rho_lag1 = float(np.clip(np.sum(s0 * s1) / denom, 0.05, 0.65)) if denom > 0 else 0.25
            else:
                rho_lag1 = 0.25
        else:
            sigma_base = float(max(2.0, np.std(y_vals) * 0.35))
            rho_lag1 = 0.25

        point_forecasts: List[int] = []
        lower_80: List[int] = []
        upper_80: List[int] = []
        lower_95: List[int] = []
        upper_95: List[int] = []

        for h_idx, raw_val in enumerate(raw_preds):
            pt = max(0, int(round(raw_val)))
            # Scale uncertainty slightly with seasonal amplitude and horizon
            level_scale = max(0.55, np.sqrt(max(1.0, raw_val) / max(1.0, np.mean(y_vals))))
            sigma_h = max(1.5, sigma_base * level_scale * (1.0 + 0.015 * h_idx))

            l80 = max(0, int(round(raw_val - Z_80 * sigma_h)))
            u80 = max(pt, int(round(raw_val + Z_80 * sigma_h)))
            l95 = max(0, int(round(raw_val - Z_95 * sigma_h)))
            u95 = max(u80, int(round(raw_val + Z_95 * sigma_h)))

            point_forecasts.append(pt)
            lower_80.append(l80)
            upper_80.append(u80)
            lower_95.append(l95)
            upper_95.append(u95)

        ann_total, ann_l80, ann_u80, ann_l95, ann_u95, cal_method = compute_calibrated_annual_interval(
            point_forecasts, lower_80, upper_80, lower_95, upper_95, residual_autocorr_lag1=rho_lag1
        )

        runtime_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        warnings_list: List[str] = []
        if len(hist_df) < 36:
            warnings_list.append(
                "Fewer than 3 full annual seasonal cycles (36 months) available; seasonal benchmark intervals have higher uncertainty."
            )

        return EngineForecastOutput(
            model_id=self.model_id,
            model_name=self.name,
            model_version=self.version,
            dates=future_cov["period_start"].tolist(),
            point_forecasts=point_forecasts,
            raw_forecasts=[round(float(x), 2) for x in raw_preds],
            lower_80=lower_80,
            upper_80=upper_80,
            lower_95=lower_95,
            upper_95=upper_95,
            future_enrolled_students=future_cov["enrolled_students"].tolist(),
            future_teaching_days=future_cov["teaching_days"].tolist(),
            future_attendance_rate=future_cov["attendance_rate"].tolist(),
            future_reporting_flag=future_cov["reporting_change_flag"].tolist(),
            annual_total=ann_total,
            annual_lower_80=ann_l80,
            annual_upper_80=ann_u80,
            annual_lower_95=ann_l95,
            annual_upper_95=ann_u95,
            calibration_method=cal_method,
            runtime_ms=runtime_ms,
            warnings=warnings_list,
        )


class StatisticalCountForecaster(BaseForecaster):
    """
    Model A2: Statistical Count Regression (Negative Binomial / Poisson GLM via statsmodels).
    Models non-negative monthly safeguarding counts using a log-link count GLM with:
    - Month-of-year seasonal effects / Fourier harmonics
    - Teaching days exposure
    - Secular trend & reporting practice step-change
    - Student enrolment log-exposure offset
    Automatically estimates overdispersion parameter alpha and selects Negative Binomial
    when count variance exceeds the Poisson mean.
    """

    model_id = "statistical_count"
    name = "Statistical Count Model (NegBin / Poisson GLM)"
    short_name = "Statistical Count GLM"
    family = "Statistical Count Time-Series"
    version = "statsmodels-0.15-GLM"
    license_name = "BSD-3-Clause"
    commercial_production_permitted = True
    supports_95_pi = True
    data_requirements = "24+ months historical monthly counts + enrolment & teaching days"
    backend_engine = "statsmodels GLM (NegativeBinomial / Poisson Log-Link)"

    def is_available(self, governance_mode: str = "research_prototype", license_ack: bool = True) -> Tuple[bool, str, str]:
        return (
            True,
            "Available",
            "Active parametric count regression engine (Negative Binomial / Poisson GLM with log-enrolment exposure). Fully permitted for operational use.",
        )

    def get_scenario_capabilities(self, has_attendance_data: bool = True) -> ScenarioSensitivityCapabilities:
        return ScenarioSensitivityCapabilities(
            supports_enrolment_growth=True,
            enrolment_explanation="Incorporated directly via log(enrolled_students) exposure offset in the count GLM.",
            supports_teaching_days=True,
            teaching_days_explanation="Incorporated directly as a structural regressor in the log-linear count equation.",
            supports_attendance_scenario=False,
            attendance_explanation="Disabled for Statistical Count GLM: attendance rate is omitted to preserve degrees of freedom and avoid collinearity with seasonal indicators.",
            supports_reporting_flag=True,
        )

    @staticmethod
    def _build_design_matrix(
        df: pd.DataFrame,
        t_start_idx: int,
        total_hist_len: int,
        use_month_dummies: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Construct design matrix X and log-enrolment offset for the count GLM."""
        n = len(df)
        t_idx = np.arange(t_start_idx, t_start_idx + n, dtype=float) / max(12.0, float(total_hist_len))
        months = df["month_num"].to_numpy(dtype=int)
        td = df["teaching_days"].to_numpy(dtype=float) / 20.0
        rep_flag = df["reporting_change_flag"].to_numpy(dtype=float)

        cols = [
            np.ones(n, dtype=float),  # intercept
            t_idx,                    # scaled secular trend
            td,                       # teaching days
            rep_flag,                 # reporting system change
        ]

        if use_month_dummies:
            # Month indicators for months 1..11 (month 12 as reference)
            for m in range(1, 12):
                cols.append((months == m).astype(float))
        else:
            # Harmonic seasonal terms for shorter series (< 24 months)
            for k in (1, 2, 3):
                cols.append(np.sin(2.0 * np.pi * k * months / 12.0))
                cols.append(np.cos(2.0 * np.pi * k * months / 12.0))

        X = np.column_stack(cols)
        enrol = np.maximum(10.0, df["enrolled_students"].to_numpy(dtype=float))
        offset = np.log(enrol / 1000.0)
        return X, offset

    def forecast(
        self,
        hist_df: pd.DataFrame,
        horizon: int = 12,
        scenario: Optional[ScenarioAssumptions] = None,
        academic_year_start_month: int = 9,
        **kwargs: Any,
    ) -> EngineForecastOutput:
        t0 = time.perf_counter()
        future_cov = build_future_covariates_df(hist_df, horizon, scenario, academic_year_start_month)

        y_train = np.maximum(0.0, hist_df["concern_count"].to_numpy(dtype=float))
        n_hist = len(hist_df)
        use_dummies = n_hist >= 24

        X_train, offset_train = self._build_design_matrix(hist_df, 0, n_hist, use_month_dummies=use_dummies)
        X_fut, offset_fut = self._build_design_matrix(future_cov, n_hist, n_hist, use_month_dummies=use_dummies)

        # Drop any zero-variance non-intercept columns in X_train (e.g., if reporting_change_flag is constant)
        keep_cols = [0]
        for c_idx in range(1, X_train.shape[1]):
            if np.std(X_train[:, c_idx]) > 1e-7:
                keep_cols.append(c_idx)
        X_train_clean = X_train[:, keep_cols]
        X_fut_clean = X_fut[:, keep_cols]

        # Step 1: Fit initial Poisson GLM to test overdispersion
        try:
            poisson_mod = sm.GLM(y_train, X_train_clean, family=sm.families.Poisson(), offset=offset_train)
            poisson_res = poisson_mod.fit(maxiter=100, disp=False)
            mu_hat_train = np.maximum(0.5, poisson_res.mu)
            df_resid = max(1, n_hist - X_train_clean.shape[1])
            pearson_chi2 = float(np.sum(((y_train - mu_hat_train) ** 2) / mu_hat_train))
            dispersion_ratio = pearson_chi2 / df_resid

            if dispersion_ratio > 1.15:
                # Method-of-moments estimator for Negative Binomial overdispersion parameter alpha
                # E[(y - mu)^2 - mu] = alpha * mu^2
                alpha_est = float(
                    np.sum(((y_train - mu_hat_train) ** 2) - mu_hat_train) / max(1.0, np.sum(mu_hat_train**2))
                )
                alpha_est = float(np.clip(alpha_est, 0.015, 0.65))
                nb_mod = sm.GLM(
                    y_train,
                    X_train_clean,
                    family=sm.families.NegativeBinomial(alpha=alpha_est),
                    offset=offset_train,
                )
                fit_res = nb_mod.fit(maxiter=100, disp=False)
                is_negbin = True
            else:
                alpha_est = 0.0
                fit_res = poisson_res
                is_negbin = False

            pred_summary = fit_res.get_prediction(X_fut_clean, offset=offset_fut)
            raw_preds = np.maximum(0.0, np.asarray(pred_summary.predicted_mean, dtype=float))
            se_mean = np.asarray(pred_summary.se_mean, dtype=float)
            se_mean = np.where(np.isfinite(se_mean), se_mean, raw_preds * 0.08)

            # Calculate empirical residual autocorrelation on training residuals
            resid_train = y_train - np.asarray(fit_res.mu, dtype=float)
            if len(resid_train) >= 12:
                r0 = resid_train[:-1] - np.mean(resid_train[:-1])
                r1 = resid_train[1:] - np.mean(resid_train[1:])
                denom = float(np.sqrt(np.sum(r0**2) * np.sum(r1**2)))
                rho_lag1 = float(np.clip(np.sum(r0 * r1) / denom, 0.05, 0.60)) if denom > 0 else 0.22
            else:
                rho_lag1 = 0.22
        except Exception:
            # Fallback to harmonic OLS on log1p counts if GLM optimization is singular on sparse slices
            is_negbin = False
            alpha_est = 0.08
            y_log = np.log1p(y_train) - offset_train
            beta, _, _, _ = np.linalg.lstsq(X_train_clean, y_log, rcond=None)
            raw_preds = np.maximum(0.0, np.expm1(X_fut_clean @ beta + offset_fut))
            se_mean = raw_preds * 0.10
            rho_lag1 = 0.25

        point_forecasts: List[int] = []
        lower_80: List[int] = []
        upper_80: List[int] = []
        lower_95: List[int] = []
        upper_95: List[int] = []

        for h_idx, mu_val in enumerate(raw_preds):
            pt = max(0, int(round(mu_val)))
            # Predictive variance combines parameter estimation variance (se_mean^2)
            # and count observation variance (mu + alpha * mu^2)
            obs_var = mu_val + (alpha_est * (mu_val**2) if is_negbin else 0.0)
            total_sd = float(np.sqrt(max(1.0, (se_mean[h_idx] ** 2) + obs_var)))

            l80 = max(0, int(round(mu_val - Z_80 * total_sd)))
            u80 = max(pt, int(round(mu_val + Z_80 * total_sd)))
            l95 = max(0, int(round(mu_val - Z_95 * total_sd)))
            u95 = max(u80, int(round(mu_val + Z_95 * total_sd)))

            point_forecasts.append(pt)
            lower_80.append(l80)
            upper_80.append(u80)
            lower_95.append(l95)
            upper_95.append(u95)

        ann_total, ann_l80, ann_u80, ann_l95, ann_u95, cal_method = compute_calibrated_annual_interval(
            point_forecasts, lower_80, upper_80, lower_95, upper_95, residual_autocorr_lag1=rho_lag1
        )

        runtime_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        dist_label = f"NegativeBinomial(alpha={alpha_est:.3f})" if is_negbin else "Poisson"
        warnings_list: List[str] = []
        if len(hist_df) < 36:
            warnings_list.append(
                "Fewer than 3 full annual seasonal cycles (36 months) available; GLM seasonal parameters have wider confidence intervals."
            )

        return EngineForecastOutput(
            model_id=self.model_id,
            model_name=self.name,
            model_version=f"{self.version} [{dist_label}]",
            dates=future_cov["period_start"].tolist(),
            point_forecasts=point_forecasts,
            raw_forecasts=[round(float(x), 2) for x in raw_preds],
            lower_80=lower_80,
            upper_80=upper_80,
            lower_95=lower_95,
            upper_95=upper_95,
            future_enrolled_students=future_cov["enrolled_students"].tolist(),
            future_teaching_days=future_cov["teaching_days"].tolist(),
            future_attendance_rate=future_cov["attendance_rate"].tolist(),
            future_reporting_flag=future_cov["reporting_change_flag"].tolist(),
            annual_total=ann_total,
            annual_lower_80=ann_l80,
            annual_upper_80=ann_u80,
            annual_lower_95=ann_l95,
            annual_upper_95=ann_u95,
            calibration_method=f"{dist_label} predictive variance + {cal_method}",
            runtime_ms=runtime_ms,
            warnings=warnings_list,
        )

