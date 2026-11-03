"""Base forecasting interfaces, statistical interval calibration, and extensible registry."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from app.data.synthetic_generator import (
    BASE_FE_TEACHING_DAYS,
    get_academic_term,
    get_academic_year_label,
)
from app.schemas.models import (
    FeatureImportanceItem,
    ScenarioAssumptions,
    ScenarioSensitivityCapabilities,
)

Z_80 = 1.2815515655446004
Z_95 = 1.959963984540054


@dataclass
class EngineForecastOutput:
    model_id: str
    model_name: str
    model_version: str
    dates: List[str]
    point_forecasts: List[int]
    raw_forecasts: List[float]
    lower_80: List[int]
    upper_80: List[int]
    lower_95: Optional[List[int]]
    upper_95: Optional[List[int]]
    future_enrolled_students: List[int]
    future_teaching_days: List[int]
    future_attendance_rate: List[Optional[float]]
    future_reporting_flag: List[int]
    annual_total: int
    annual_lower_80: int
    annual_upper_80: int
    annual_lower_95: Optional[int]
    annual_upper_95: Optional[int]
    calibration_method: str
    runtime_ms: float
    feature_importances: List[FeatureImportanceItem] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def build_future_covariates_df(
    hist_df: pd.DataFrame,
    horizon: int,
    scenario: Optional[ScenarioAssumptions] = None,
    academic_year_start_month: int = 9,
) -> pd.DataFrame:
    """
    Construct future monthly calendar and scenario covariates strictly from information
    known at the forecast origin plus explicit user scenario assumptions.
    Never leaks actual future observations.
    """
    if scenario is None:
        scenario = ScenarioAssumptions()

    last_dt = pd.to_datetime(hist_df["period_start"].max())
    future_dates = pd.date_range(
        start=last_dt + pd.DateOffset(months=1),
        periods=horizon,
        freq="MS",
    )

    # Calculate historical baseline by month of year from the most recent 12 months
    recent_12 = hist_df.tail(12).copy()
    recent_by_month = recent_12.set_index("month_num")

    latest_enrolment = float(hist_df["enrolled_students"].iloc[-1])
    latest_rep_flag = int(hist_df["reporting_change_flag"].iloc[-1])
    has_att = hist_df["attendance_rate"].notna().any()
    overall_mean_att = float(hist_df["attendance_rate"].dropna().mean()) if has_att else np.nan

    rows = []
    for h_idx, dt in enumerate(future_dates):
        m = int(dt.month)
        # Base enrolment from same month in prior year (to preserve intra-year retention curve)
        # scaled to latest cohort level, then adjusted by scenario enrolment_growth_pct
        if m in recent_by_month.index:
            ref_row = recent_by_month.loc[m]
            if isinstance(ref_row, pd.DataFrame):
                ref_row = ref_row.iloc[-1]
            base_enrol = float(ref_row["enrolled_students"])
            base_td = int(ref_row["teaching_days"])
            base_att = float(ref_row["attendance_rate"]) if pd.notna(ref_row["attendance_rate"]) else overall_mean_att
        else:
            base_enrol = latest_enrolment
            base_td = BASE_FE_TEACHING_DAYS.get(m, 17)
            base_att = overall_mean_att

        enrol_assumed = int(max(10, round(base_enrol * (1.0 + scenario.enrolment_growth_pct / 100.0))))
        td_assumed = int(np.clip(base_td + scenario.teaching_days_delta, 1, 25))

        if has_att and pd.notna(base_att):
            att_assumed: Optional[float] = round(
                float(np.clip(base_att + scenario.attendance_rate_delta_pp, 50.0, 100.0)),
                1,
            )
        else:
            att_assumed = None

        if scenario.reporting_change_assumed is not None:
            rep_assumed = int(scenario.reporting_change_assumed)
        else:
            rep_assumed = latest_rep_flag

        rows.append(
            {
                "period_start": dt.strftime("%Y-%m-%d"),
                "dt": dt,
                "period_label": dt.strftime("%b %Y"),
                "academic_year": get_academic_year_label(dt.year, m, academic_year_start_month),
                "academic_term": get_academic_term(m),
                "month_num": m,
                "horizon_step": h_idx + 1,
                "enrolled_students": enrol_assumed,
                "teaching_days": td_assumed,
                "attendance_rate": att_assumed,
                "reporting_change_flag": rep_assumed,
            }
        )

    return pd.DataFrame(rows)


def compute_calibrated_annual_interval(
    point_forecasts: List[int],
    lower_80: List[int],
    upper_80: List[int],
    lower_95: Optional[List[int]],
    upper_95: Optional[List[int]],
    residual_autocorr_lag1: float = 0.28,
) -> Tuple[int, int, int, Optional[int], Optional[int], str]:
    """
    Calculate a statistically calibrated aggregate prediction interval across months
    in the forecast horizon by accounting for monthly error variance and cross-month
    residual autocorrelation:
        Var(sum_{h=1}^H e_h) = sum_h sigma_h^2 + 2 * sum_{i < j} rho^{|j-i|} * sigma_i * sigma_j
    Never simply sums monthly lower and upper bounds.
    """
    total_point = int(sum(point_forecasts))
    h_len = len(point_forecasts)
    if h_len == 0:
        return 0, 0, 0, 0, 0, "Empty horizon"
    if h_len == 1:
        return (
            total_point,
            int(lower_80[0]),
            int(upper_80[0]),
            int(lower_95[0]) if lower_95 else None,
            int(upper_95[0]) if upper_95 else None,
            "Single-month direct interval",
        )

    rho = float(np.clip(residual_autocorr_lag1, 0.0, 0.75))

    # Extract horizon-specific standard deviations from 80% intervals
    sigmas_80 = np.array(
        [max(1.0, (u - l) / (2.0 * Z_80)) for l, u in zip(lower_80, upper_80)],
        dtype=float,
    )

    var_sum_80 = 0.0
    for i in range(h_len):
        var_sum_80 += sigmas_80[i] ** 2
        for j in range(i + 1, h_len):
            var_sum_80 += 2.0 * (rho ** (j - i)) * sigmas_80[i] * sigmas_80[j]

    agg_sigma_80 = float(np.sqrt(max(1.0, var_sum_80)))
    ann_lower_80 = max(0, int(round(total_point - Z_80 * agg_sigma_80)))
    ann_upper_80 = max(ann_lower_80, int(round(total_point + Z_80 * agg_sigma_80)))

    ann_lower_95: Optional[int] = None
    ann_upper_95: Optional[int] = None
    if lower_95 is not None and upper_95 is not None:
        sigmas_95 = np.array(
            [max(1.0, (u - l) / (2.0 * Z_95)) for l, u in zip(lower_95, upper_95)],
            dtype=float,
        )
        var_sum_95 = 0.0
        for i in range(h_len):
            var_sum_95 += sigmas_95[i] ** 2
            for j in range(i + 1, h_len):
                var_sum_95 += 2.0 * (rho ** (j - i)) * sigmas_95[i] * sigmas_95[j]
        agg_sigma_95 = float(np.sqrt(max(1.0, var_sum_95)))
        ann_lower_95 = max(0, int(round(total_point - Z_95 * agg_sigma_95)))
        ann_upper_95 = max(ann_lower_95, int(round(total_point + Z_95 * agg_sigma_95)))

    method_desc = (
        f"Covariance-calibrated multi-month aggregation (AR(1) error correlation rho={rho:.2f}; "
        "accounts for cross-month error cancellation rather than summing monthly bounds)"
    )
    return total_point, ann_lower_80, ann_upper_80, ann_lower_95, ann_upper_95, method_desc


class BaseForecaster(ABC):
    """Abstract base interface for all Safeguarding Forecast AI engines."""

    model_id: str
    name: str
    short_name: str
    family: str
    version: str
    license_name: str
    commercial_production_permitted: bool
    supports_95_pi: bool
    data_requirements: str
    backend_engine: str

    @abstractmethod
    def is_available(self, governance_mode: str = "research_prototype", license_ack: bool = True) -> Tuple[bool, str, str]:
        """Return (available, status_label, status_explanation)."""
        raise NotImplementedError

    @abstractmethod
    def get_scenario_capabilities(self, has_attendance_data: bool = True) -> ScenarioSensitivityCapabilities:
        """Return which scenario variables this model genuinely uses in prediction."""
        raise NotImplementedError

    @abstractmethod
    def forecast(
        self,
        hist_df: pd.DataFrame,
        horizon: int = 12,
        scenario: Optional[ScenarioAssumptions] = None,
        academic_year_start_month: int = 9,
        **kwargs: Any,
    ) -> EngineForecastOutput:
        """Generate multi-step monthly count forecasts and calibrated prediction intervals."""
        raise NotImplementedError


class AmazonChronos2AdapterStub(BaseForecaster):
    """
    Extensible adapter interface for Amazon Chronos-2 foundation model.
    Demonstrates plug-and-play architecture without restructuring the application.
    """

    model_id = "chronos_2"
    name = "Amazon Chronos-2 (Extensible Slot)"
    short_name = "Chronos-2"
    family = "Foundation Model (Extensible Interface)"
    version = "2.0-adapter"
    license_name = "Apache-2.0"
    commercial_production_permitted = True
    supports_95_pi = True
    data_requirements = "12+ months univariate or covariate time-series"
    backend_engine = "chronos-forecasting (Optional Extension)"

    def is_available(self, governance_mode: str = "research_prototype", license_ack: bool = True) -> Tuple[bool, str, str]:
        return (
            False,
            "Extensible Interface Ready",
            "Optional extension interface pre-configured for Amazon Chronos-2 (`chronos-forecasting`); not enabled in default prototype build.",
        )

    def get_scenario_capabilities(self, has_attendance_data: bool = True) -> ScenarioSensitivityCapabilities:
        return ScenarioSensitivityCapabilities(
            supports_enrolment_growth=False,
            enrolment_explanation="Chronos-2 adapter slot is not active.",
            supports_teaching_days=False,
            teaching_days_explanation="Chronos-2 adapter slot is not active.",
            supports_attendance_scenario=False,
            attendance_explanation="Chronos-2 adapter slot is not active.",
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
        raise RuntimeError("Amazon Chronos-2 optional extension is not installed.")

