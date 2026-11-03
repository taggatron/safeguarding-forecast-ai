"""Core orchestration service for datasets, historical analytics, forecasting, backtesting, and insights."""
import hashlib
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from app.core.database import (
    IMPORTED_BANNER,
    SYNTHETIC_BANNER,
    DatabaseManager,
    db_manager,
)
from app.data.processor import aggregate_monthly_series
from app.data.synthetic_generator import (
    generate_synthetic_safeguarding_dataset,
    get_academic_term,
)
from app.models.base import EngineForecastOutput
from app.models.lightgbm_forecaster import LightGBMMLForecaster
from app.models.registry import EXTENSIBLE_ADAPTERS, MODEL_REGISTRY, get_forecaster
from app.schemas.models import (
    AcademicYearComparisonItem,
    AnnualForecastSummary,
    BacktestResponse,
    CategoryForecastItem,
    CategorySummaryItem,
    DatasetSummaryResponse,
    ForecastRequest,
    ForecastResponse,
    HistoricalStatsResponse,
    ModelAvailabilityItem,
    ModelsAvailabilityResponse,
    MonthlyForecastPoint,
    MonthlyHistoricalPoint,
    ProvisionForecastItem,
    ProvisionSummaryItem,
    ScenarioAssumptions,
    SeasonalMonthProfile,
    SyntheticGeneratorConfig,
    TrendsInsightsResponse,
)
from app.services.backtesting import run_rolling_backtest

MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}


class ForecastPlatformService:
    """Coordinates dataset management, forecasting models, backtesting cache, and deterministic insights."""

    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or db_manager
        self._backtest_cache: Dict[str, BacktestResponse] = {}
        self._forecast_cache: Dict[str, ForecastResponse] = {}
        self.ensure_default_synthetic_dataset()

    def clear_caches(self) -> None:
        self._backtest_cache.clear()
        self._forecast_cache.clear()

    def ensure_default_synthetic_dataset(self) -> str:
        """Ensure the default 6-year South Devon College synthetic dataset exists."""
        default_id = "synthetic_sdc_default"
        if not self.db.dataset_exists(default_id):
            cfg = SyntheticGeneratorConfig(
                seed=42,
                num_academic_years=6,
                start_academic_year=2020,
                academic_year_start_month=9,
            )
            df = generate_synthetic_safeguarding_dataset(cfg)
            self.db.save_dataset(
                dataset_id=default_id,
                name="South Devon College — Synthetic Demonstration Dataset (6 Academic Years)",
                df=df,
                is_synthetic=True,
                seed=cfg.seed,
                academic_year_start_month=cfg.academic_year_start_month,
                metadata=cfg.model_dump(),
            )
            self.db.set_config("active_dataset_id", default_id)
        return self.db.get_config("active_dataset_id", default_id) or default_id

    def regenerate_synthetic_dataset(self, config: SyntheticGeneratorConfig) -> DatasetSummaryResponse:
        """Generate or update the synthetic demonstration dataset with reproducible seed and settings."""
        default_id = "synthetic_sdc_default"
        df = generate_synthetic_safeguarding_dataset(config)
        self.db.save_dataset(
            dataset_id=default_id,
            name=f"South Devon College — Synthetic Demo ({config.num_academic_years} Academic Years, Seed {config.seed})",
            df=df,
            is_synthetic=True,
            seed=config.seed,
            academic_year_start_month=config.academic_year_start_month,
            metadata=config.model_dump(),
        )
        self.db.set_config("active_dataset_id", default_id)
        self.db.set_config("academic_year_start_month", str(config.academic_year_start_month))
        self.clear_caches()
        return self.get_active_dataset_summary()

    def get_active_dataset_and_df(self) -> Tuple[Dict[str, Any], pd.DataFrame]:
        active_id = self.ensure_default_synthetic_dataset()
        info = self.db.get_dataset_info(active_id)
        if not info:
            self.db.set_config("active_dataset_id", "synthetic_sdc_default")
            active_id = self.ensure_default_synthetic_dataset()
            info = self.db.get_dataset_info(active_id)
        df = self.db.load_records_df(active_id)
        return info, df

    def get_active_dataset_summary(self) -> DatasetSummaryResponse:
        info, df = self.get_active_dataset_and_df()
        ay_start = int(self.db.get_config("academic_year_start_month", str(info["academic_year_start_month"])) or 9)
        supp_enabled = (self.db.get_config("small_group_suppression_enabled", "false") or "false").lower() == "true"

        monthly_all = aggregate_monthly_series(df, "All", "All", ay_start)
        min_dt = pd.to_datetime(monthly_all["period_start"].min())
        max_dt = pd.to_datetime(monthly_all["period_start"].max())
        expected_range = pd.date_range(start=min_dt, end=max_dt, freq="MS")
        actual_set = set(monthly_all["period_start"].tolist())
        missing = [d.strftime("%Y-%m-%d") for d in expected_range if d.strftime("%Y-%m-%d") not in actual_set]

        is_synth = bool(info["is_synthetic"])
        banner = SYNTHETIC_BANNER if is_synth else IMPORTED_BANNER

        return DatasetSummaryResponse(
            dataset_id=info["dataset_id"],
            name=info["name"],
            is_synthetic=is_synth,
            governance_banner=banner,
            created_at=info["created_at"],
            seed=info.get("seed"),
            academic_year_start_month=ay_start,
            date_range_start=str(monthly_all["period_start"].min()),
            date_range_end=str(monthly_all["period_start"].max()),
            total_months=len(monthly_all),
            full_academic_years=len(monthly_all) // 12,
            academic_years=sorted(monthly_all["academic_year"].unique().tolist()),
            provisions=sorted(df["provision"].unique().tolist()),
            categories=sorted(df["safeguarding_category"].unique().tolist()),
            total_recorded_concerns=int(monthly_all["concern_count"].sum()),
            latest_enrolled_students=int(monthly_all["enrolled_students"].iloc[-1]),
            has_attendance_data=bool(df["attendance_rate"].notna().any()),
            has_reporting_change_flag=bool((df["reporting_change_flag"] > 0).any()),
            missing_months=missing,
            small_group_suppression_enabled=supp_enabled,
            available_datasets=self.db.list_datasets(),
        )

    def get_models_availability(self) -> ModelsAvailabilityResponse:
        gov_mode = self.db.get_config("governance_mode", "research_prototype") or "research_prototype"
        lic_ack = (self.db.get_config("timesfm_license_acknowledged", "true") or "true").lower() == "true"

        items: List[ModelAvailabilityItem] = []
        for m_id, forecaster in list(MODEL_REGISTRY.items()) + [(a.model_id, a) for a in EXTENSIBLE_ADAPTERS]:
            avail, status_label, explanation = forecaster.is_available(
                governance_mode=gov_mode, license_ack=lic_ack
            )
            items.append(
                ModelAvailabilityItem(
                    model_id=m_id,
                    name=forecaster.name,
                    short_name=forecaster.short_name,
                    family=forecaster.family,
                    version=forecaster.version,
                    available=avail,
                    status_label=status_label,
                    status_explanation=explanation,
                    license_name=forecaster.license_name,
                    commercial_production_permitted=forecaster.commercial_production_permitted,
                    supports_covariates=(m_id in ("statistical_count", "lightgbm_mlforecast", "timesfm_3")),
                    supports_95_pi=forecaster.supports_95_pi,
                    data_requirements=forecaster.data_requirements,
                    supported_horizons=[3, 6, 12],
                    backend_engine=forecaster.backend_engine,
                )
            )

        return ModelsAvailabilityResponse(
            governance_mode=gov_mode,
            timesfm_license_acknowledged=lic_ack,
            models=items,
            extensible_interfaces=["BaseForecaster ABC", "AmazonChronos2AdapterStub (chronos_2)"],
        )

    @staticmethod
    def _build_academic_year_comparisons(monthly_df: pd.DataFrame) -> List[AcademicYearComparisonItem]:
        items: List[AcademicYearComparisonItem] = []
        grouped = monthly_df.groupby("academic_year", as_index=False)
        ay_list = sorted(monthly_df["academic_year"].unique().tolist())

        prev_total: Optional[int] = None
        for ay in ay_list:
            sub = monthly_df[monthly_df["academic_year"] == ay].sort_values("period_start")
            if sub.empty:
                continue
            tot_concerns = int(sub["concern_count"].sum())
            avg_enrol = int(round(float(sub["enrolled_students"].mean())))
            per_1000 = round((tot_concerns / max(1, avg_enrol)) * 1000.0, 1)
            peak_idx = sub["concern_count"].idxmax()
            peak_row = sub.loc[peak_idx]
            peak_m_label = pd.to_datetime(peak_row["period_start"]).strftime("%B")
            peak_cnt = int(peak_row["concern_count"])
            tot_td = int(sub["teaching_days"].sum())
            avg_att = (
                round(float(sub["attendance_rate"].dropna().mean()), 1)
                if sub["attendance_rate"].notna().any()
                else None
            )
            yoy = (
                round(((tot_concerns - prev_total) / prev_total) * 100.0, 1)
                if (prev_total is not None and prev_total > 0 and len(sub) == 12)
                else None
            )
            if len(sub) == 12:
                prev_total = tot_concerns

            items.append(
                AcademicYearComparisonItem(
                    academic_year=ay,
                    total_concerns=tot_concerns,
                    avg_enrolled_students=avg_enrol,
                    concerns_per_1000=per_1000,
                    peak_month=peak_m_label,
                    peak_month_count=peak_cnt,
                    total_teaching_days=tot_td,
                    avg_attendance_rate=avg_att,
                    yoy_change_pct=yoy,
                )
            )
        return items

    def get_historical_stats(
        self,
        academic_year: str = "All",
        provision: str = "All",
        safeguarding_category: str = "All",
    ) -> HistoricalStatsResponse:
        info, df = self.get_active_dataset_and_df()
        ay_start = int(self.db.get_config("academic_year_start_month", str(info["academic_year_start_month"])) or 9)
        supp_enabled = (self.db.get_config("small_group_suppression_enabled", "false") or "false").lower() == "true"

        monthly_full = aggregate_monthly_series(df, provision, safeguarding_category, ay_start)
        ay_comparisons = self._build_academic_year_comparisons(monthly_full)

        if academic_year != "All":
            monthly_view = monthly_full[monthly_full["academic_year"] == academic_year].copy()
            raw_view = df[df["academic_year"] == academic_year].copy()
        else:
            monthly_view = monthly_full.copy()
            raw_view = df.copy()

        # Provision breakdown
        cat_filtered_raw = (
            raw_view[raw_view["safeguarding_category"] == safeguarding_category]
            if safeguarding_category != "All"
            else raw_view
        )
        total_all_prov = max(1, int(cat_filtered_raw["concern_count"].sum()))
        prov_items: List[ProvisionSummaryItem] = []
        for prov_name in sorted(df["provision"].unique().tolist()):
            p_series = aggregate_monthly_series(raw_view, prov_name, safeguarding_category, ay_start)
            if p_series.empty:
                continue
            p_tot = int(p_series["concern_count"].sum())
            p_enrol = int(round(float(p_series["enrolled_students"].mean())))
            n_years = max(1.0, len(p_series) / 12.0)
            p_rate = round(((p_tot / n_years) / max(1, p_enrol)) * 1000.0, 1)
            prov_items.append(
                ProvisionSummaryItem(
                    provision=prov_name,
                    total_concerns=p_tot,
                    share_pct=round((p_tot / total_all_prov) * 100.0, 1),
                    avg_enrolled_students=p_enrol,
                    concerns_per_1000=p_rate,
                )
            )
        prov_items.sort(key=lambda x: x.total_concerns, reverse=True)

        # Category breakdown
        prov_filtered_raw = (
            raw_view[raw_view["provision"] == provision] if provision != "All" else raw_view
        )
        total_all_cat = max(1, int(prov_filtered_raw["concern_count"].sum()))
        cat_items: List[CategorySummaryItem] = []
        for cat_name in sorted(df["safeguarding_category"].unique().tolist()):
            c_tot = int(
                prov_filtered_raw[prov_filtered_raw["safeguarding_category"] == cat_name]["concern_count"].sum()
            )
            suppressed = bool(supp_enabled and 0 < c_tot < 5)
            cat_items.append(
                CategorySummaryItem(
                    safeguarding_category=cat_name,
                    total_concerns=0 if suppressed else c_tot,
                    share_pct=round((c_tot / total_all_cat) * 100.0, 1),
                    suppressed=suppressed,
                )
            )
        cat_items.sort(key=lambda x: x.share_pct, reverse=True)

        # Identify reporting change transition dates
        rep_dates: List[str] = []
        flags = monthly_full["reporting_change_flag"].to_numpy(dtype=int)
        dates = monthly_full["period_start"].tolist()
        for idx in range(1, len(flags)):
            if flags[idx] == 1 and flags[idx - 1] == 0:
                rep_dates.append(dates[idx])

        hist_points = [
            MonthlyHistoricalPoint(
                period_start=str(r["period_start"]),
                period_label=str(r["period_label"]),
                academic_year=str(r["academic_year"]),
                academic_term=str(r["academic_term"]),
                month_num=int(r["month_num"]),
                concern_count=int(r["concern_count"]),
                enrolled_students=int(r["enrolled_students"]),
                concerns_per_1000=float(r["concerns_per_1000"]),
                teaching_days=int(r["teaching_days"]),
                attendance_rate=float(r["attendance_rate"]) if pd.notna(r["attendance_rate"]) else None,
                reporting_change_flag=int(r["reporting_change_flag"]),
                rolling_3m_avg=float(r["rolling_3m_avg"]) if pd.notna(r["rolling_3m_avg"]) else None,
                rolling_6m_avg=float(r["rolling_6m_avg"]) if pd.notna(r["rolling_6m_avg"]) else None,
            )
            for _, r in monthly_view.iterrows()
        ]

        is_synth = bool(info["is_synthetic"])
        return HistoricalStatsResponse(
            dataset_id=info["dataset_id"],
            is_synthetic=is_synth,
            governance_banner=SYNTHETIC_BANNER if is_synth else IMPORTED_BANNER,
            selected_academic_year=academic_year,
            selected_provision=provision,
            selected_category=safeguarding_category,
            monthly_series=hist_points,
            academic_year_comparisons=ay_comparisons,
            provision_breakdown=prov_items,
            category_breakdown=cat_items,
            reporting_change_dates=rep_dates,
        )

    @staticmethod
    def _build_annual_summary(
        eng_out: EngineForecastOutput,
        hist_df: pd.DataFrame,
        horizon: int,
        pi_level: int,
        academic_year_start_month: int,
    ) -> AnnualForecastSummary:
        # Compare horizon total against the corresponding prior-year window of equal length
        prior_slice = hist_df.tail(horizon)
        prior_total = int(prior_slice["concern_count"].sum())
        pred_total = int(eng_out.annual_total)
        yoy_diff = pred_total - prior_total
        yoy_pct = round((yoy_diff / max(1, prior_total)) * 100.0, 1)

        if pi_level == 95 and eng_out.annual_lower_95 is not None and eng_out.annual_upper_95 is not None:
            lb = int(eng_out.annual_lower_95)
            ub = int(eng_out.annual_upper_95)
        else:
            lb = int(eng_out.annual_lower_80)
            ub = int(eng_out.annual_upper_80)

        # Peak demand month and top 2-month / peak period description
        pts = eng_out.point_forecasts
        dts = pd.to_datetime(eng_out.dates)
        max_idx = int(np.argmax(pts))
        peak_dt = dts[max_idx]
        peak_month_str = peak_dt.strftime("%B %Y")
        peak_cnt = int(pts[max_idx])

        # Also identify second highest month or consecutive high-demand pair
        if len(pts) >= 2:
            sorted_indices = np.argsort(pts)[::-1]
            second_idx = int(sorted_indices[1])
            second_dt = dts[second_idx]
            if abs(max_idx - second_idx) == 1:
                first_m = min(peak_dt, second_dt).strftime("%B")
                sec_m = max(peak_dt, second_dt).strftime("%B %Y")
                peak_desc = f"{first_m}–{sec_m} (consecutive peak window)"
            else:
                peak_desc = f"{peak_dt.strftime('%B')} & {second_dt.strftime('%B')}"
        else:
            peak_desc = peak_month_str

        avg_fut_enrol = int(round(float(np.mean(eng_out.future_enrolled_students))))
        # Annualized or horizon rate per 1,000 enrolled students
        rate_per_1000 = round((pred_total / max(1, avg_fut_enrol)) * 1000.0, 1)

        first_fut_dt = dts[0]
        from app.data.synthetic_generator import get_academic_year_label
        target_ay = get_academic_year_label(first_fut_dt.year, first_fut_dt.month, academic_year_start_month)

        return AnnualForecastSummary(
            target_academic_year=target_ay,
            horizon_months=horizon,
            predicted_total_concerns=pred_total,
            lower_bound=lb,
            upper_bound=ub,
            interval_level=pi_level,
            lower_80=int(eng_out.annual_lower_80),
            upper_80=int(eng_out.annual_upper_80),
            lower_95=eng_out.annual_lower_95,
            upper_95=eng_out.annual_upper_95,
            calibration_method=eng_out.calibration_method,
            prior_year_total=prior_total,
            yoy_change_count=yoy_diff,
            yoy_change_pct=yoy_pct,
            peak_demand_month=peak_month_str,
            peak_demand_count=peak_cnt,
            peak_demand_period_description=peak_desc,
            predicted_concerns_per_1000=rate_per_1000,
            enrolled_students_assumed=avg_fut_enrol,
        )

    @staticmethod
    def _generate_deterministic_insights(
        model_name: str,
        annual_summary: AnnualForecastSummary,
        monthly_forecasts: List[MonthlyForecastPoint],
        provision_forecasts: List[ProvisionForecastItem],
        category_forecasts: List[CategoryForecastItem],
        scenario: Optional[ScenarioAssumptions],
        baseline_summary: Optional[AnnualForecastSummary],
    ) -> List[str]:
        """
        Deterministically generate operational planning insights from actual computed
        forecast numbers and historical summaries. Never uses an LLM.
        """
        insights: List[str] = []

        # 1. Peak months insight
        if len(monthly_forecasts) >= 2:
            top_two = sorted(monthly_forecasts, key=lambda x: x.point_forecast, reverse=True)[:2]
            m1 = pd.to_datetime(top_two[0].period_start).strftime("%B")
            m2 = pd.to_datetime(top_two[1].period_start).strftime("%B")
            insights.append(
                f"The {model_name} model forecasts highest recorded safeguarding demand during {m1} "
                f"({top_two[0].point_forecast} concerns) and {m2} ({top_two[1].point_forecast} concerns), "
                "aligning with term-time intake and post-holiday pastoral disclosure patterns."
            )

        # 2. Horizon volume & planning range insight
        direction_word = "an increase" if annual_summary.yoy_change_count >= 0 else "a decrease"
        insights.append(
            f"Over the {annual_summary.horizon_months}-month horizon ({annual_summary.target_academic_year}), "
            f"total recorded demand is projected at {annual_summary.predicted_total_concerns:,} concerns "
            f"({annual_summary.interval_level}% calibrated planning range: {annual_summary.lower_bound:,} to "
            f"{annual_summary.upper_bound:,}), representing {direction_word} of "
            f"{abs(annual_summary.yoy_change_pct):.1f}% ({annual_summary.yoy_change_count:+d} concerns) "
            f"compared with the prior {annual_summary.horizon_months}-month period ({annual_summary.prior_year_total:,})."
        )

        # 3. Provision concentration insight
        if provision_forecasts:
            top_p = max(provision_forecasts, key=lambda x: x.predicted_total)
            insights.append(
                f"{top_p.provision} accounts for the largest share of forecasted demand at {top_p.share_pct:.1f}% "
                f"({top_p.predicted_total:,} concerns; {top_p.concerns_per_1000:.1f} per 1,000 enrolled learners)."
            )

        # 4. Category concentration insight
        if category_forecasts:
            top_c = max(category_forecasts, key=lambda x: x.predicted_total)
            insights.append(
                f"'{top_c.safeguarding_category}' remains the primary forecasted category at {top_c.share_pct:.1f}% "
                f"({top_c.predicted_total:,} predicted concerns), informing pastoral & mental health specialist triage capacity."
            )

        # 5. Scenario comparison insight if active
        if scenario and baseline_summary:
            delta_tot = annual_summary.predicted_total_concerns - baseline_summary.predicted_total_concerns
            if delta_tot != 0:
                insights.append(
                    f"Scenario Impact: Under the adjusted scenario assumptions (enrolment {scenario.enrolment_growth_pct:+.1f}%, "
                    f"teaching days {scenario.teaching_days_delta:+d}/mo, attendance {scenario.attendance_rate_delta_pp:+.1f}pp), "
                    f"predicted demand shifts by {delta_tot:+d} concerns ({((delta_tot / max(1, baseline_summary.predicted_total_concerns)) * 100.0):+.1f}%) "
                    f"relative to the baseline forecast ({baseline_summary.predicted_total_concerns:,})."
                )

        return insights

    def generate_forecast(self, req: ForecastRequest) -> ForecastResponse:
        info, df = self.get_active_dataset_and_df()
        ay_start = int(self.db.get_config("academic_year_start_month", str(info["academic_year_start_month"])) or 9)
        gov_mode = self.db.get_config("governance_mode", "research_prototype") or "research_prototype"
        lic_ack = (self.db.get_config("timesfm_license_acknowledged", "true") or "true").lower() == "true"

        horizon = req.horizon if req.horizon in (3, 6, 12) else 12
        pi_level = 95 if req.prediction_interval == 95 else 80
        scenario = req.scenario or ScenarioAssumptions()

        cache_key_payload = {
            "dataset_id": info["dataset_id"],
            "created_at": info["created_at"],
            "model_id": req.model_id,
            "horizon": horizon,
            "provision": req.provision,
            "category": req.safeguarding_category,
            "pi_level": pi_level,
            "ay_filter": req.academic_year_filter,
            "timesfm_mv": req.timesfm_multivariate,
            "scenario": scenario.model_dump(),
            "gov_mode": gov_mode,
            "lic_ack": lic_ack,
            "ay_start": ay_start,
        }
        cache_key = hashlib.sha256(json.dumps(cache_key_payload, sort_keys=True).encode()).hexdigest()[:20]
        if cache_key in self._forecast_cache:
            return self._forecast_cache[cache_key]

        forecaster = get_forecaster(req.model_id)
        avail, status_label, status_expl = forecaster.is_available(
            governance_mode=gov_mode, license_ack=lic_ack
        )
        if not avail:
            raise ValueError(f"Model '{forecaster.name}' is not available: {status_expl}")

        monthly_full = aggregate_monthly_series(df, req.provision, req.safeguarding_category, ay_start)
        if monthly_full.empty:
            raise ValueError(
                f"No historical records found for provision='{req.provision}' and category='{req.safeguarding_category}'."
            )

        # Primary forecast (with scenario assumptions if any)
        eng_out = forecaster.forecast(
            monthly_full,
            horizon=horizon,
            scenario=scenario,
            academic_year_start_month=ay_start,
            governance_mode=gov_mode,
            license_ack=lic_ack,
            use_covariates=req.timesfm_multivariate,
        )

        # Baseline forecast (zero scenario deltas) for side-by-side comparison in Forecast Explorer
        has_scenario_delta = (
            abs(scenario.enrolment_growth_pct) > 1e-4
            or scenario.teaching_days_delta != 0
            or abs(scenario.attendance_rate_delta_pp) > 1e-4
            or scenario.reporting_change_assumed is not None
        )
        if has_scenario_delta:
            base_eng_out = forecaster.forecast(
                monthly_full,
                horizon=horizon,
                scenario=ScenarioAssumptions(),
                academic_year_start_month=ay_start,
                governance_mode=gov_mode,
                license_ack=lic_ack,
                use_covariates=req.timesfm_multivariate,
            )
        else:
            base_eng_out = eng_out

        annual_summary = self._build_annual_summary(
            eng_out, monthly_full, horizon, pi_level, ay_start
        )
        baseline_annual_summary = self._build_annual_summary(
            base_eng_out, monthly_full, horizon, pi_level, ay_start
        )

        def _to_monthly_points(out: EngineForecastOutput) -> List[MonthlyForecastPoint]:
            pts_list: List[MonthlyForecastPoint] = []
            for idx, d_str in enumerate(out.dates):
                dt = pd.to_datetime(d_str)
                from app.data.synthetic_generator import get_academic_year_label
                enrol_val = max(1, int(out.future_enrolled_students[idx]))
                pt_val = int(out.point_forecasts[idx])
                l_bound = int(out.lower_95[idx]) if (pi_level == 95 and out.lower_95) else int(out.lower_80[idx])
                u_bound = int(out.upper_95[idx]) if (pi_level == 95 and out.upper_95) else int(out.upper_80[idx])
                pts_list.append(
                    MonthlyForecastPoint(
                        period_start=d_str,
                        period_label=dt.strftime("%b %Y"),
                        academic_year=get_academic_year_label(dt.year, dt.month, ay_start),
                        academic_term=get_academic_term(dt.month),
                        month_num=int(dt.month),
                        point_forecast=pt_val,
                        raw_point_forecast=float(out.raw_forecasts[idx]),
                        lower_80=int(out.lower_80[idx]),
                        upper_80=int(out.upper_80[idx]),
                        lower_95=int(out.lower_95[idx]) if out.lower_95 else None,
                        upper_95=int(out.upper_95[idx]) if out.upper_95 else None,
                        enrolled_students_assumed=enrol_val,
                        teaching_days_assumed=int(out.future_teaching_days[idx]),
                        attendance_rate_assumed=out.future_attendance_rate[idx],
                        concerns_per_1000_forecast=round((pt_val / enrol_val) * 1000.0, 2),
                        lower_per_1000=round((l_bound / enrol_val) * 1000.0, 2),
                        upper_per_1000=round((u_bound / enrol_val) * 1000.0, 2),
                    )
                )
            return pts_list

        monthly_forecasts = _to_monthly_points(eng_out)
        baseline_monthly_forecasts = _to_monthly_points(base_eng_out)

        # Provision-level forecast breakdown (reconciled to total predicted concerns)
        all_provs = sorted(df["provision"].unique().tolist())
        prov_forecasts: List[ProvisionForecastItem] = []
        raw_prov_totals: List[Tuple[str, float, float, float, int, int]] = []
        for p_name in all_provs:
            if req.provision != "All" and p_name != req.provision:
                continue
            p_hist = aggregate_monthly_series(df, p_name, req.safeguarding_category, ay_start)
            if len(p_hist) < 12:
                continue
            # Use recent 24-month historical share & trend combined with model horizon total
            prior_h_tot = int(p_hist.tail(horizon)["concern_count"].sum())
            p_enrol = int(round(float(p_hist.tail(12)["enrolled_students"].mean()) * (1.0 + scenario.enrolment_growth_pct / 100.0)))
            raw_prov_totals.append((p_name, float(prior_h_tot), 0.85, 1.15, prior_h_tot, max(1, p_enrol)))

        sum_prior_provs = max(1.0, sum(x[1] for x in raw_prov_totals))
        allocated_prov_sum = 0
        for idx, (p_name, p_prior_flt, _, _, prior_h_tot, p_enrol) in enumerate(raw_prov_totals):
            share = p_prior_flt / sum_prior_provs
            if idx == len(raw_prov_totals) - 1:
                p_pred = max(0, annual_summary.predicted_total_concerns - allocated_prov_sum)
            else:
                p_pred = max(0, int(round(annual_summary.predicted_total_concerns * share)))
                allocated_prov_sum += p_pred
            half_width_ratio = (annual_summary.upper_bound - annual_summary.lower_bound) / max(
                2.0, 2.0 * annual_summary.predicted_total_concerns
            )
            # Slightly wider relative uncertainty for individual provisions (1 / sqrt(share))
            prov_rel_hw = min(0.85, half_width_ratio * float(np.sqrt(1.0 / max(0.15, share))))
            p_low = max(0, int(round(p_pred * (1.0 - prov_rel_hw))))
            p_high = max(p_pred, int(round(p_pred * (1.0 + prov_rel_hw))))
            p_yoy = round(((p_pred - prior_h_tot) / max(1, prior_h_tot)) * 100.0, 1)
            prov_forecasts.append(
                ProvisionForecastItem(
                    provision=p_name,
                    predicted_total=p_pred,
                    lower_bound=p_low,
                    upper_bound=p_high,
                    share_pct=round((p_pred / max(1, annual_summary.predicted_total_concerns)) * 100.0, 1),
                    concerns_per_1000=round((p_pred / p_enrol) * 1000.0, 1),
                    historical_prior_year=prior_h_tot,
                    yoy_change_pct=p_yoy,
                )
            )
        prov_forecasts.sort(key=lambda x: x.predicted_total, reverse=True)

        # Category-level forecast breakdown with sparse category reliability check
        all_cats = sorted(df["safeguarding_category"].unique().tolist())
        cat_forecasts: List[CategoryForecastItem] = []
        raw_cat_totals: List[Tuple[str, float, int, float]] = []
        for c_name in all_cats:
            if req.safeguarding_category != "All" and c_name != req.safeguarding_category:
                continue
            c_hist = aggregate_monthly_series(df, req.provision, c_name, ay_start)
            if c_hist.empty:
                continue
            prior_c_tot = int(c_hist.tail(horizon)["concern_count"].sum())
            mean_monthly = float(c_hist["concern_count"].mean())
            raw_cat_totals.append((c_name, float(max(0.5, prior_c_tot)), prior_c_tot, mean_monthly))

        sum_prior_cats = max(1.0, sum(x[1] for x in raw_cat_totals))
        allocated_cat_sum = 0
        for idx, (c_name, c_prior_flt, prior_c_tot, mean_monthly) in enumerate(raw_cat_totals):
            share = c_prior_flt / sum_prior_cats
            if idx == len(raw_cat_totals) - 1:
                c_pred = max(0, annual_summary.predicted_total_concerns - allocated_cat_sum)
            else:
                c_pred = max(0, int(round(annual_summary.predicted_total_concerns * share)))
                allocated_cat_sum += c_pred
            half_width_ratio = (annual_summary.upper_bound - annual_summary.lower_bound) / max(
                2.0, 2.0 * annual_summary.predicted_total_concerns
            )
            cat_rel_hw = min(0.90, half_width_ratio * float(np.sqrt(1.0 / max(0.10, share))))
            c_low = max(0, int(round(c_pred * (1.0 - cat_rel_hw))))
            c_high = max(c_pred, int(round(c_pred * (1.0 + cat_rel_hw))))
            c_yoy = round(((c_pred - prior_c_tot) / max(1, prior_c_tot)) * 100.0, 1)
            reliable = mean_monthly >= 3.0
            note = (
                None
                if reliable
                else "Low monthly base count (<3/month); category-specific forecast has high relative variance."
            )
            cat_forecasts.append(
                CategoryForecastItem(
                    safeguarding_category=c_name,
                    predicted_total=c_pred,
                    lower_bound=c_low,
                    upper_bound=c_high,
                    share_pct=round((c_pred / max(1, annual_summary.predicted_total_concerns)) * 100.0, 1),
                    historical_prior_year=prior_c_tot,
                    yoy_change_pct=c_yoy,
                    reliable_estimation=reliable,
                    note=note,
                )
            )
        cat_forecasts.sort(key=lambda x: x.predicted_total, reverse=True)

        # Retrieve historical backtest accuracy for this model to display in KPI Card 4
        bt_res = self.get_backtest_comparison(
            provision=req.provision,
            safeguarding_category=req.safeguarding_category,
            horizon=12,
        )
        model_bt = next((m for m in bt_res.metrics if m.model_id == req.model_id), None)
        accuracy_summary = {
            "mae": model_bt.mae if model_bt else None,
            "wape_pct": model_bt.wape_pct if model_bt else None,
            "pi_80_coverage_pct": model_bt.pi_80_coverage_pct if model_bt else None,
            "outperforms_baseline": model_bt.outperforms_baseline if model_bt else None,
            "improvement_vs_baseline_pct": model_bt.improvement_vs_baseline_pct if model_bt else None,
            "best_model_id": bt_res.best_model_id,
            "best_model_name": bt_res.best_model_name,
            "explanation": (
                f"Historical 12-month rolling backtest MAE: {model_bt.mae:.1f} concerns/month "
                f"(WAPE: {model_bt.wape_pct:.1f}% across {model_bt.total_evaluated_months} out-of-sample months)."
                if (model_bt and model_bt.mae is not None and model_bt.wape_pct is not None)
                else "Evaluated via rolling-origin time-series cross-validation."
            ),
        }

        # Filter historical series for chart display if academic_year_filter != "All"
        if req.academic_year_filter != "All":
            hist_display_df = monthly_full[monthly_full["academic_year"] == req.academic_year_filter].copy()
        else:
            hist_display_df = monthly_full.copy()

        hist_points = [
            MonthlyHistoricalPoint(
                period_start=str(r["period_start"]),
                period_label=str(r["period_label"]),
                academic_year=str(r["academic_year"]),
                academic_term=str(r["academic_term"]),
                month_num=int(r["month_num"]),
                concern_count=int(r["concern_count"]),
                enrolled_students=int(r["enrolled_students"]),
                concerns_per_1000=float(r["concerns_per_1000"]),
                teaching_days=int(r["teaching_days"]),
                attendance_rate=float(r["attendance_rate"]) if pd.notna(r["attendance_rate"]) else None,
                reporting_change_flag=int(r["reporting_change_flag"]),
                rolling_3m_avg=float(r["rolling_3m_avg"]) if pd.notna(r["rolling_3m_avg"]) else None,
                rolling_6m_avg=float(r["rolling_6m_avg"]) if pd.notna(r["rolling_6m_avg"]) else None,
            )
            for _, r in hist_display_df.iterrows()
        ]

        insights = self._generate_deterministic_insights(
            forecaster.short_name,
            annual_summary,
            monthly_forecasts,
            prov_forecasts,
            cat_forecasts,
            scenario if has_scenario_delta else None,
            baseline_annual_summary if has_scenario_delta else None,
        )

        run_id = f"run_{uuid.uuid4().hex[:10]}"
        self.db.log_forecast_run(
            run_id=run_id,
            dataset_id=info["dataset_id"],
            model_id=req.model_id,
            model_version=eng_out.model_version,
            horizon=horizon,
            provision=req.provision,
            safeguarding_category=req.safeguarding_category,
            prediction_interval=pi_level,
            scenario=scenario.model_dump(),
            annual_total=annual_summary.predicted_total_concerns,
            lower_bound=annual_summary.lower_bound,
            upper_bound=annual_summary.upper_bound,
            runtime_ms=eng_out.runtime_ms,
        )

        is_synth = bool(info["is_synthetic"])
        resp = ForecastResponse(
            run_id=run_id,
            dataset_id=info["dataset_id"],
            is_synthetic=is_synth,
            governance_banner=SYNTHETIC_BANNER if is_synth else IMPORTED_BANNER,
            model_id=req.model_id,
            model_name=forecaster.name,
            model_version=eng_out.model_version,
            runtime_ms=eng_out.runtime_ms,
            horizon=horizon,
            provision=req.provision,
            safeguarding_category=req.safeguarding_category,
            prediction_interval=pi_level,
            annual_summary=annual_summary,
            historical_series=hist_points,
            monthly_forecasts=monthly_forecasts,
            baseline_monthly_forecasts=baseline_monthly_forecasts,
            baseline_annual_summary=baseline_annual_summary,
            provision_forecasts=prov_forecasts,
            category_forecasts=cat_forecasts,
            planning_insights=insights,
            model_accuracy_summary=accuracy_summary,
            scenario_capabilities=forecaster.get_scenario_capabilities(
                has_attendance_data=bool(monthly_full["attendance_rate"].notna().any())
            ),
            warnings=eng_out.warnings,
        )
        self._forecast_cache[cache_key] = resp
        return resp

    def get_backtest_comparison(
        self,
        provision: str = "All",
        safeguarding_category: str = "All",
        horizon: int = 12,
    ) -> BacktestResponse:
        info, df = self.get_active_dataset_and_df()
        ay_start = int(self.db.get_config("academic_year_start_month", str(info["academic_year_start_month"])) or 9)
        gov_mode = self.db.get_config("governance_mode", "research_prototype") or "research_prototype"
        lic_ack = (self.db.get_config("timesfm_license_acknowledged", "true") or "true").lower() == "true"

        bt_key = f"{info['dataset_id']}_{info['created_at']}_{provision}_{safeguarding_category}_{horizon}_{gov_mode}_{lic_ack}"
        if bt_key in self._backtest_cache:
            return self._backtest_cache[bt_key]

        monthly_df = aggregate_monthly_series(df, provision, safeguarding_category, ay_start)
        res = run_rolling_backtest(
            monthly_df=monthly_df,
            dataset_id=info["dataset_id"],
            is_synthetic=bool(info["is_synthetic"]),
            provision=provision,
            safeguarding_category=safeguarding_category,
            horizon=horizon,
            academic_year_start_month=ay_start,
            governance_mode=gov_mode,
            license_ack=lic_ack,
        )
        self._backtest_cache[bt_key] = res
        return res

    def get_trends_and_insights(
        self,
        provision: str = "All",
        safeguarding_category: str = "All",
    ) -> TrendsInsightsResponse:
        info, df = self.get_active_dataset_and_df()
        ay_start = int(self.db.get_config("academic_year_start_month", str(info["academic_year_start_month"])) or 9)
        monthly_df = aggregate_monthly_series(df, provision, safeguarding_category, ay_start)

        ay_comparisons = self._build_academic_year_comparisons(monthly_df)

        # Seasonal profile ordered by academic year cycle (Sep..Aug when ay_start=9)
        overall_monthly_mean = max(0.1, float(monthly_df["concern_count"].mean()))
        month_order = [(ay_start + i - 1) % 12 + 1 for i in range(12)]
        seasonal_profile: List[SeasonalMonthProfile] = []

        for m in month_order:
            m_sub = monthly_df[monthly_df["month_num"] == m]
            if m_sub.empty:
                continue
            avg_c = float(m_sub["concern_count"].mean())
            min_c = int(m_sub["concern_count"].min())
            max_c = int(m_sub["concern_count"].max())
            avg_td = float(m_sub["teaching_days"].mean())
            seasonal_profile.append(
                SeasonalMonthProfile(
                    month_num=m,
                    month_name=MONTH_NAMES[m][:3],
                    academic_term=get_academic_term(m),
                    avg_concerns=round(avg_c, 1),
                    min_concerns=min_c,
                    max_concerns=max_c,
                    seasonal_index=round(avg_c / overall_monthly_mean, 2),
                    avg_teaching_days=round(avg_td, 1),
                    avg_concerns_per_teaching_day=round(avg_c / max(1.0, avg_td), 2),
                )
            )

        # Academic-year trend by provision
        prov_trends: List[Dict[str, Any]] = []
        provs = sorted(df["provision"].unique().tolist())
        ays = sorted(monthly_df["academic_year"].unique().tolist())
        for ay in ays:
            row_dict: Dict[str, Any] = {"academic_year": ay}
            ay_raw = df[df["academic_year"] == ay]
            if safeguarding_category != "All":
                ay_raw = ay_raw[ay_raw["safeguarding_category"] == safeguarding_category]
            for p_name in provs:
                p_sub = ay_raw[ay_raw["provision"] == p_name]
                p_tot = int(p_sub["concern_count"].sum())
                p_enrol = int(round(float(p_sub["enrolled_students"].mean()))) if not p_sub.empty else 1
                row_dict[p_name] = p_tot
                row_dict[f"{p_name}_per_1000"] = round((p_tot / max(1, p_enrol)) * 1000.0, 1)
            prov_trends.append(row_dict)

        # Academic-year trend by category
        cat_trends: List[Dict[str, Any]] = []
        cats = sorted(df["safeguarding_category"].unique().tolist())
        for ay in ays:
            row_dict = {"academic_year": ay}
            ay_raw = df[df["academic_year"] == ay]
            if provision != "All":
                ay_raw = ay_raw[ay_raw["provision"] == provision]
            for c_name in cats:
                c_tot = int(ay_raw[ay_raw["safeguarding_category"] == c_name]["concern_count"].sum())
                row_dict[c_name] = c_tot
            cat_trends.append(row_dict)

        # Run LightGBM feature importance extraction on the series
        lgbm_forecaster = LightGBMMLForecaster()
        lgbm_out = lgbm_forecaster.forecast(
            monthly_df,
            horizon=12,
            scenario=ScenarioAssumptions(),
            academic_year_start_month=ay_start,
        )

        # Generate deterministic findings
        peak_prof = max(seasonal_profile, key=lambda x: x.avg_concerns)
        trough_prof = min(seasonal_profile, key=lambda x: x.avg_concerns)
        first_ay = ay_comparisons[0]
        last_ay = ay_comparisons[-1]
        multi_year_growth = round(
            ((last_ay.total_concerns - first_ay.total_concerns) / max(1, first_ay.total_concerns)) * 100.0,
            1,
        )

        key_findings = [
            (
                f"Pronounced Term-Time Seasonality: Recorded safeguarding concerns peak in {peak_prof.month_name} "
                f"(average {peak_prof.avg_concerns:.0f} concerns/month, seasonal index {peak_prof.seasonal_index:.2f}x) "
                f"and reach their lowest level in {trough_prof.month_name} ({trough_prof.avg_concerns:.0f} concerns/month, "
                f"{trough_prof.seasonal_index:.2f}x)."
            ),
            (
                f"Multi-Year Demand Trajectory: Across {len(ay_comparisons)} academic years ({first_ay.academic_year} to "
                f"{last_ay.academic_year}), annual recorded volume moved from {first_ay.total_concerns:,} to "
                f"{last_ay.total_concerns:,} ({multi_year_growth:+.1f}%), while rate per 1,000 students shifted from "
                f"{first_ay.concerns_per_1000:.1f} to {last_ay.concerns_per_1000:.1f}."
            ),
            (
                "Reporting Practice Sensitivity: Shifts in recorded volume after September 2023 align with enhanced "
                "digital safeguarding workflow adoption and mandatory staff awareness training."
            ),
        ]

        hist_points = [
            MonthlyHistoricalPoint(
                period_start=str(r["period_start"]),
                period_label=str(r["period_label"]),
                academic_year=str(r["academic_year"]),
                academic_term=str(r["academic_term"]),
                month_num=int(r["month_num"]),
                concern_count=int(r["concern_count"]),
                enrolled_students=int(r["enrolled_students"]),
                concerns_per_1000=float(r["concerns_per_1000"]),
                teaching_days=int(r["teaching_days"]),
                attendance_rate=float(r["attendance_rate"]) if pd.notna(r["attendance_rate"]) else None,
                reporting_change_flag=int(r["reporting_change_flag"]),
                rolling_3m_avg=float(r["rolling_3m_avg"]) if pd.notna(r["rolling_3m_avg"]) else None,
                rolling_6m_avg=float(r["rolling_6m_avg"]) if pd.notna(r["rolling_6m_avg"]) else None,
            )
            for _, r in monthly_df.iterrows()
        ]

        is_synth = bool(info["is_synthetic"])
        return TrendsInsightsResponse(
            dataset_id=info["dataset_id"],
            is_synthetic=is_synth,
            governance_banner=SYNTHETIC_BANNER if is_synth else IMPORTED_BANNER,
            monthly_series=hist_points,
            seasonal_profile=seasonal_profile,
            academic_year_comparisons=ay_comparisons,
            provision_trends=prov_trends,
            category_trends=cat_trends,
            feature_importances=lgbm_out.feature_importances,
            causation_disclaimer="These relationships are statistical associations, not evidence of causation.",
            reporting_practice_note=(
                "Important Interpretation Note: Increases in recorded safeguarding concerns may reflect changes in "
                "student awareness, pastoral accessibility, staff training, or digital recording practices rather "
                "than increases in underlying harm. Low forecasted demand in quieter months must never be used to "
                "justify reducing statutory safeguarding cover."
            ),
            key_findings=key_findings,
        )


platform_service = ForecastPlatformService()

