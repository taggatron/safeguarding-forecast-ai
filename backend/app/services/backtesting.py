"""Rolling-origin time-series backtesting and model comparison engine."""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from app.core.database import IMPORTED_BANNER, SYNTHETIC_BANNER
from app.models.registry import MODEL_REGISTRY
from app.schemas.models import (
    BacktestComparisonPoint,
    BacktestResponse,
    ModelBacktestMetrics,
    ScenarioAssumptions,
)


def compute_rolling_origin_origins(n_months: int, horizon: int = 12) -> List[int]:
    """
    Determine valid rolling forecast origin indices (number of training months)
    strictly respecting temporal order without any lookahead leakage.
    Requires at least 36 months (3 academic years) of initial training history when
    dataset length permits, or at least 24 months for shorter datasets.
    """
    min_train = 36 if n_months >= (36 + horizon) else max(12, n_months - horizon)
    max_origin = n_months - horizon
    if max_origin < 12:
        return []

    # Step forward by 12 months (annual academic-year origins) or 6 months if fewer origins exist
    origins: List[int] = []
    step = 12 if (max_origin - min_train) >= 12 else max(3, horizon // 2)
    curr = max_origin
    while curr >= min_train:
        origins.append(curr)
        curr -= step
    origins.sort()
    return origins


def run_rolling_backtest(
    monthly_df: pd.DataFrame,
    dataset_id: str,
    is_synthetic: bool,
    provision: str = "All",
    safeguarding_category: str = "All",
    horizon: int = 12,
    academic_year_start_month: int = 9,
    governance_mode: str = "research_prototype",
    license_ack: bool = True,
) -> BacktestResponse:
    """
    Execute rolling-origin time-series backtesting across all registered models.
    At each origin T:
    - Trains strictly on monthly_df.iloc[:T]
    - Forecasts monthly_df.iloc[T : T + horizon]
    - Evaluates MAE, RMSE, WAPE, MASE, 80% & 95% PI coverage, and cumulative horizon total MAE.
    """
    banner = SYNTHETIC_BANNER if is_synthetic else IMPORTED_BANNER
    n_months = len(monthly_df)
    origins = compute_rolling_origin_origins(n_months, horizon=horizon)

    seasonality_warning: Optional[str] = None
    if n_months < 36:
        seasonality_warning = (
            f"Historical dataset contains {n_months} months (< 36 months / 3 full annual cycles). "
            "Rolling-origin evaluation and seasonal parameter stability are limited by short history."
        )

    # Store per-model fold results
    model_errors: Dict[str, List[float]] = {m_id: [] for m_id in MODEL_REGISTRY}
    model_sq_errors: Dict[str, List[float]] = {m_id: [] for m_id in MODEL_REGISTRY}
    model_actuals: Dict[str, List[float]] = {m_id: [] for m_id in MODEL_REGISTRY}
    model_cov80: Dict[str, List[int]] = {m_id: [] for m_id in MODEL_REGISTRY}
    model_cov95: Dict[str, List[int]] = {m_id: [] for m_id in MODEL_REGISTRY}
    model_ann_errors: Dict[str, List[float]] = {m_id: [] for m_id in MODEL_REGISTRY}
    model_runtimes: Dict[str, List[float]] = {m_id: [] for m_id in MODEL_REGISTRY}
    model_avail_info: Dict[str, Tuple[bool, str, str]] = {}

    for m_id, forecaster in MODEL_REGISTRY.items():
        model_avail_info[m_id] = forecaster.is_available(
            governance_mode=governance_mode, license_ack=license_ack
        )

    # Track non-overlapping or latest rolling fold points for the visual comparison chart
    timeline_points_map: Dict[str, BacktestComparisonPoint] = {}

    for origin_idx in origins:
        train_slice = monthly_df.iloc[:origin_idx].copy().reset_index(drop=True)
        test_slice = monthly_df.iloc[origin_idx : origin_idx + horizon].copy().reset_index(drop=True)
        fold_origin_date = str(train_slice["period_start"].iloc[-1])
        actual_vec = test_slice["concern_count"].to_numpy(dtype=float)

        # Initialize timeline points for this test window
        for row_idx, test_row in test_slice.iterrows():
            p_start = str(test_row["period_start"])
            timeline_points_map[p_start] = BacktestComparisonPoint(
                period_start=p_start,
                period_label=str(test_row["period_label"]),
                academic_year=str(test_row["academic_year"]),
                fold_origin=fold_origin_date,
                actual_count=int(test_row["concern_count"]),
                predictions={},
                lower_80_bounds={},
                upper_80_bounds={},
            )

        for m_id, forecaster in MODEL_REGISTRY.items():
            avail, _, _ = model_avail_info[m_id]
            if not avail:
                for test_row in test_slice.itertuples():
                    p_start = str(test_row.period_start)
                    timeline_points_map[p_start].predictions[m_id] = None
                    timeline_points_map[p_start].lower_80_bounds[m_id] = None
                    timeline_points_map[p_start].upper_80_bounds[m_id] = None
                continue

            try:
                #Note: During backtest at origin T, we pass ScenarioAssumptions() (no future leakage)
                res = forecaster.forecast(
                    train_slice,
                    horizon=horizon,
                    scenario=ScenarioAssumptions(),
                    academic_year_start_month=academic_year_start_month,
                    governance_mode=governance_mode,
                    license_ack=license_ack,
                )
                preds = np.array(res.point_forecasts, dtype=float)
                l80 = np.array(res.lower_80, dtype=float)
                u80 = np.array(res.upper_80, dtype=float)
                l95 = np.array(res.lower_95, dtype=float) if res.lower_95 else None
                u95 = np.array(res.upper_95, dtype=float) if res.upper_95 else None

                abs_err = np.abs(actual_vec - preds)
                sq_err = (actual_vec - preds) ** 2
                in_80 = ((actual_vec >= l80) & (actual_vec <= u80)).astype(int)

                model_errors[m_id].extend(abs_err.tolist())
                model_sq_errors[m_id].extend(sq_err.tolist())
                model_actuals[m_id].extend(actual_vec.tolist())
                model_cov80[m_id].extend(in_80.tolist())
                if l95 is not None and u95 is not None:
                    in_95 = ((actual_vec >= l95) & (actual_vec <= u95)).astype(int)
                    model_cov95[m_id].extend(in_95.tolist())

                model_ann_errors[m_id].append(float(abs(np.sum(actual_vec) - np.sum(preds))))
                model_runtimes[m_id].append(float(res.runtime_ms))

                for step_i, test_row in enumerate(test_slice.itertuples()):
                    p_start = str(test_row.period_start)
                    timeline_points_map[p_start].predictions[m_id] = int(res.point_forecasts[step_i])
                    timeline_points_map[p_start].lower_80_bounds[m_id] = int(res.lower_80[step_i])
                    timeline_points_map[p_start].upper_80_bounds[m_id] = int(res.upper_80[step_i])

            except Exception as exc:
                model_avail_info[m_id] = (False, "Not available", f"Backtest execution failed: {exc}")

    baseline_mae = (
        float(np.mean(model_errors["seasonal_naive"]))
        if model_errors.get("seasonal_naive")
        else None
    )

    metrics_list: List[ModelBacktestMetrics] = []
    for m_id, forecaster in MODEL_REGISTRY.items():
        avail, status_label, explanation = model_avail_info[m_id]
        errs = model_errors[m_id]
        if not avail or not errs:
            metrics_list.append(
                ModelBacktestMetrics(
                    model_id=m_id,
                    model_name=forecaster.name,
                    family=forecaster.family,
                    available=False,
                    status_label=status_label,
                    horizon_evaluated=horizon,
                    data_requirements=forecaster.data_requirements,
                    notes=explanation,
                )
            )
            continue

        mae_val = float(np.mean(errs))
        rmse_val = float(np.sqrt(np.mean(model_sq_errors[m_id])))
        sum_actual = float(np.sum(model_actuals[m_id]))
        wape_val = float((np.sum(errs) / sum_actual) * 100.0) if sum_actual > 0 else None
        mase_val = float(mae_val / baseline_mae) if (baseline_mae and baseline_mae > 0) else 1.0
        cov80_val = float(np.mean(model_cov80[m_id]) * 100.0) if model_cov80[m_id] else None
        cov95_val = float(np.mean(model_cov95[m_id]) * 100.0) if model_cov95[m_id] else None
        ann_mae_val = float(np.mean(model_ann_errors[m_id])) if model_ann_errors[m_id] else None
        avg_rt = float(np.mean(model_runtimes[m_id])) if model_runtimes[m_id] else 0.0

        if m_id == "seasonal_naive" or baseline_mae is None or baseline_mae <= 0:
            outperforms = False if m_id == "seasonal_naive" else None
            imp_pct = 0.0 if m_id == "seasonal_naive" else None
        else:
            outperforms = bool(mae_val < baseline_mae)
            imp_pct = round(float(((baseline_mae - mae_val) / baseline_mae) * 100.0), 1)

        metrics_list.append(
            ModelBacktestMetrics(
                model_id=m_id,
                model_name=forecaster.name,
                family=forecaster.family,
                available=True,
                status_label=status_label,
                mae=round(mae_val, 2),
                rmse=round(rmse_val, 2),
                wape_pct=round(wape_val, 2) if wape_val is not None else None,
                mase=round(mase_val, 3),
                pi_80_coverage_pct=round(cov80_val, 1) if cov80_val is not None else None,
                pi_95_coverage_pct=round(cov95_val, 1) if cov95_val is not None else None,
                annual_total_mae=round(ann_mae_val, 1) if ann_mae_val is not None else None,
                outperforms_baseline=outperforms,
                improvement_vs_baseline_pct=imp_pct,
                num_rolling_origins=len(origins),
                total_evaluated_months=len(errs),
                avg_runtime_ms=round(avg_rt, 1),
                horizon_evaluated=horizon,
                data_requirements=forecaster.data_requirements,
                notes=explanation,
            )
        )

    # Identify the model with the lowest empirical MAE among available models
    valid_metrics = [m for m in metrics_list if m.available and m.mae is not None]
    if valid_metrics:
        best_m = min(valid_metrics, key=lambda x: (x.mae if x.mae is not None else 999999.0))
        best_id = best_m.model_id
        best_name = best_m.model_name
        if best_id == "seasonal_naive":
            rationale = (
                f"{best_name} achieved the lowest historical Mean Absolute Error (MAE = {best_m.mae:.2f}, "
                f"WAPE = {best_m.wape_pct:.1f}%) across {best_m.num_rolling_origins} rolling-origin folds "
                f"({best_m.total_evaluated_months} evaluated months). More complex models did not outperform the seasonal benchmark on this slice."
            )
        else:
            imp_str = (
                f"improving on the Seasonal Naive benchmark by {best_m.improvement_vs_baseline_pct:.1f}%"
                if best_m.improvement_vs_baseline_pct is not None and best_m.improvement_vs_baseline_pct > 0
                else "matching the benchmark"
            )
            rationale = (
                f"{best_name} achieved the lowest historical forecasting error (MAE = {best_m.mae:.2f} concerns/month, "
                f"WAPE = {best_m.wape_pct:.1f}%, 80% PI coverage = {best_m.pi_80_coverage_pct:.1f}%), {imp_str} "
                f"across {best_m.num_rolling_origins} rolling-origin folds ({best_m.total_evaluated_months} months)."
            )
    else:
        best_id = "seasonal_naive"
        best_name = "Seasonal Naive Baseline"
        rationale = "Insufficient historical origins to rank models; defaulting to Seasonal Naive benchmark."

    sorted_points = [timeline_points_map[k] for k in sorted(timeline_points_map.keys())]

    methodology = (
        f"Evaluated using strictly causal rolling-origin time-series cross-validation across {len(origins)} "
        f"historical forecast origins ({len(origins) * horizon} total out-of-sample months, horizon = {horizon} months). "
        "At each origin, models are trained exclusively on prior academic years and evaluated on the subsequent unseen period. "
        "Random train/test splitting is never used."
    )

    return BacktestResponse(
        dataset_id=dataset_id,
        is_synthetic=is_synthetic,
        governance_banner=banner,
        provision=provision,
        safeguarding_category=safeguarding_category,
        horizon=horizon,
        best_model_id=best_id,
        best_model_name=best_name,
        best_model_rationale=rationale,
        seasonality_warning=seasonality_warning,
        metrics=metrics_list,
        historical_backtest_series=sorted_points,
        methodology_explanation=methodology,
    )

