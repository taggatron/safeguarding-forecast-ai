"""Comprehensive automated test suite for Safeguarding Forecast AI."""
import io
import json
import os
import pandas as pd
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.database import DatabaseManager, SYNTHETIC_BANNER
from app.data.processor import (
    aggregate_monthly_series,
    detect_prohibited_pii_columns,
    validate_and_parse_csv,
)
from app.data.synthetic_generator import (
    generate_synthetic_safeguarding_dataset,
    get_academic_year_label,
)
from app.models.base import compute_calibrated_annual_interval
from app.models.lightgbm_forecaster import LightGBMMLForecaster
from app.models.statistical import SeasonalNaiveForecaster, StatisticalCountForecaster
from app.models.timesfm_forecaster import TimesFM3ForecasterAdapter
from app.schemas.models import (
    ForecastRequest,
    ScenarioAssumptions,
    SyntheticGeneratorConfig,
)
from app.services.backtesting import compute_rolling_origin_origins, run_rolling_backtest
from app.services.forecast_service import ForecastPlatformService
from app.main import app


@pytest.fixture()
def isolated_service(tmp_path):
    db_file = tmp_path / "test_safeguarding.db"
    db = DatabaseManager(db_path=db_file)
    svc = ForecastPlatformService(db=db)
    return svc


def test_synthetic_data_reproducibility():
    """Verify fixed random seed produces 100% identical datasets and different seeds differ."""
    cfg1 = SyntheticGeneratorConfig(seed=42, num_academic_years=6)
    cfg2 = SyntheticGeneratorConfig(seed=42, num_academic_years=6)
    cfg3 = SyntheticGeneratorConfig(seed=99, num_academic_years=6)

    df1 = generate_synthetic_safeguarding_dataset(cfg1)
    df2 = generate_synthetic_safeguarding_dataset(cfg2)
    df3 = generate_synthetic_safeguarding_dataset(cfg3)

    pd.testing.assert_frame_equal(df1, df2)
    assert not df1["concern_count"].equals(df3["concern_count"])
    assert (df1["concern_count"] >= 0).all()
    assert len(df1["period_start"].unique()) == 72


def test_academic_year_boundaries():
    """Verify default September-August academic year boundaries and configurable start months."""
    assert get_academic_year_label(2024, 8, start_month=9) == "2023/24"
    assert get_academic_year_label(2024, 9, start_month=9) == "2024/25"
    assert get_academic_year_label(2025, 1, start_month=9) == "2024/25"
    # Configurable August start
    assert get_academic_year_label(2024, 8, start_month=8) == "2024/25"
    assert get_academic_year_label(2024, 7, start_month=8) == "2023/24"


def test_monthly_aggregation_no_enrolment_double_counting():
    """Ensure enrolled_students is summed across provisions once per month, never multiplied by categories."""
    cfg = SyntheticGeneratorConfig(seed=42, num_academic_years=3)
    df = generate_synthetic_safeguarding_dataset(cfg)

    # Check a single month: 3 provisions x 5 categories = 15 rows
    first_month = df["period_start"].iloc[0]
    m_rows = df[df["period_start"] == first_month]
    assert len(m_rows) == 15

    expected_enrol = m_rows.groupby("provision")["enrolled_students"].first().sum()
    expected_concerns = m_rows["concern_count"].sum()

    agg = aggregate_monthly_series(df, provision="All", safeguarding_category="All")
    agg_first = agg[agg["period_start"] == first_month].iloc[0]

    assert agg_first["enrolled_students"] == expected_enrol
    assert agg_first["concern_count"] == expected_concerns


def test_csv_validation_pii_rejection_and_missing_months():
    """Verify PII columns are rejected, missing months are identified, and duplicates are consolidated."""
    # 1. Test PII column rejection
    pii_cols = detect_prohibited_pii_columns(["period_start", "student_name", "dob", "case_notes", "concern_count"])
    assert set(pii_cols) == {"student_name", "dob", "case_notes"}

    bad_csv = (
        "period_start,provision,safeguarding_category,concern_count,student_name\n"
        "2024-09-01,EPYP,Online safety,12,John Doe\n"
    ).encode("utf-8")
    report_bad, df_bad = validate_and_parse_csv(bad_csv)
    assert not report_bad.is_valid
    assert not report_bad.can_import
    assert df_bad is None
    assert "student_name" in report_bad.rejected_personal_data_columns

    # 2. Test valid CSV with missing month and duplicate record consolidation
    rows = []
    dates = pd.date_range("2023-09-01", periods=15, freq="MS")
    for idx, dt in enumerate(dates):
        if idx == 5:
            # Skip one month to test missing month detection
            continue
        rows.append(f"{dt.strftime('%Y-%m-%d')},EPYP,Mental health,25,3200,18,89.5,0")
    # Add a duplicate row for first date
    rows.append("2023-09-01,EPYP,Mental health,5,3200,18,89.5,0")

    csv_content = (
        "period_start,provision,safeguarding_category,concern_count,enrolled_students,teaching_days,attendance_rate,reporting_change_flag\n"
        + "\n".join(rows)
    ).encode("utf-8")

    report_ok, df_ok = validate_and_parse_csv(csv_content)
    assert report_ok.is_valid
    assert report_ok.can_import
    assert df_ok is not None
    assert len(report_ok.missing_months) == 1
    assert report_ok.duplicate_records == 2
    assert report_ok.seasonality_sufficiency_warning is not None
    # Verify duplicate concern_count was consolidated (25 + 5 = 30)
    sep_row = df_ok[df_ok["period_start"] == "2023-09-01"].iloc[0]
    assert sep_row["concern_count"] == 30


def test_forecast_horizons_non_negative_and_total_consistency(isolated_service):
    """Verify 3, 6, and 12 month horizons produce non-negative integers and exact monthly-to-annual consistency."""
    for model_id in ["seasonal_naive", "statistical_count", "lightgbm_mlforecast"]:
        for horizon in [3, 6, 12]:
            res = isolated_service.generate_forecast(
                ForecastRequest(model_id=model_id, horizon=horizon, prediction_interval=80)
            )
            assert len(res.monthly_forecasts) == horizon
            monthly_sum = sum(pt.point_forecast for pt in res.monthly_forecasts)
            assert res.annual_summary.predicted_total_concerns == monthly_sum

            for pt in res.monthly_forecasts:
                assert pt.point_forecast >= 0
                assert pt.lower_80 >= 0
                assert pt.upper_80 >= pt.point_forecast
                if pt.lower_95 is not None and pt.upper_95 is not None:
                    assert pt.lower_95 >= 0
                    assert pt.upper_95 >= pt.upper_80

            # Verify annual interval is covariance-calibrated and NOT a naive sum of monthly bounds
            naive_lower_sum = sum(pt.lower_80 for pt in res.monthly_forecasts)
            naive_upper_sum = sum(pt.upper_80 for pt in res.monthly_forecasts)
            assert res.annual_summary.lower_80 > naive_lower_sum
            assert res.annual_summary.upper_80 < naive_upper_sum


def test_no_time_series_data_leakage():
    """
    Verify strictly causal feature engineering and forecasting: modifying future observations
    after forecast origin T must have zero effect on training features or predictions from origin T.
    """
    cfg = SyntheticGeneratorConfig(seed=42, num_academic_years=5)
    df = generate_synthetic_safeguarding_dataset(cfg)
    monthly = aggregate_monthly_series(df, "All", "All", 9)

    origin = 48
    train_clean = monthly.iloc[:origin].copy()

    # Create a mutated full series where future rows after origin have extreme values
    monthly_mutated = monthly.copy()
    monthly_mutated.loc[origin:, "concern_count"] = 99999
    monthly_mutated.loc[origin:, "concerns_per_1000"] = 9999.0
    train_from_mutated = monthly_mutated.iloc[:origin].copy()

    lgbm = LightGBMMLForecaster()
    out_clean = lgbm.forecast(train_clean, horizon=12)
    out_mutated = lgbm.forecast(train_from_mutated, horizon=12)

    assert out_clean.point_forecasts == out_mutated.point_forecasts
    assert out_clean.raw_forecasts == out_mutated.raw_forecasts


def test_sparse_category_handling(isolated_service):
    """Verify sparse categories with low monthly counts (< 3/month) are flagged in category forecasts."""
    cfg = SyntheticGeneratorConfig(seed=42, num_academic_years=4, base_annual_volume=320)
    df = generate_synthetic_safeguarding_dataset(cfg)
    # Make 'Exploitation' very sparse (0 or 1 per month)
    df.loc[df["safeguarding_category"] == "Exploitation", "concern_count"] = 0
    isolated_service.db.save_dataset("sparse_test", "Sparse Test", df, is_synthetic=True)
    isolated_service.db.set_config("active_dataset_id", "sparse_test")

    res = isolated_service.generate_forecast(ForecastRequest(model_id="statistical_count", horizon=12))
    exp_cat = next(c for c in res.category_forecasts if c.safeguarding_category == "Exploitation")
    assert exp_cat.reliable_estimation is False
    assert exp_cat.note is not None


def test_timesfm_unavailable_fallback(isolated_service, monkeypatch):
    """
    Verify that when TimesFM 3.0 is unavailable due to environment, hardware, or governance/license settings,
    it reports 'Not available' with an explanation, never fabricates predictions, and Statistical/LightGBM
    models remain 100% functional.
    """
    # 1. Test forced unavailability via environment variable
    monkeypatch.setenv("SAFEGUARDING_DISABLE_TIMESFM", "1")
    tfm = TimesFM3ForecasterAdapter()
    avail, status_label, expl = tfm.is_available(governance_mode="research_prototype", license_ack=True)
    assert avail is False
    assert "Not available" in status_label
    assert "SAFEGUARDING_DISABLE_TIMESFM" in expl

    with pytest.raises(RuntimeError):
        _, df = isolated_service.get_active_dataset_and_df()
        monthly = aggregate_monthly_series(df)
        tfm.forecast(monthly, horizon=12)

    # 2. Test governance mode restriction (operational_production blocks non-commercial weights)
    monkeypatch.delenv("SAFEGUARDING_DISABLE_TIMESFM", raising=False)
    avail_prod, status_prod, expl_prod = tfm.is_available(
        governance_mode="operational_production", license_ack=True
    )
    assert avail_prod is False
    assert "Not available" in status_prod
    assert "timesfm-non-commercial-license-v1.0" in expl_prod

    # 3. Verify Statistical Count and LightGBM continue to work when governance_mode is operational_production
    isolated_service.db.set_config("governance_mode", "operational_production")
    isolated_service.clear_caches()
    bt = isolated_service.get_backtest_comparison(horizon=12)
    tfm_metric = next(m for m in bt.metrics if m.model_id == "timesfm_3")
    lgbm_metric = next(m for m in bt.metrics if m.model_id == "lightgbm_mlforecast")
    stat_metric = next(m for m in bt.metrics if m.model_id == "statistical_count")

    assert tfm_metric.available is False
    assert tfm_metric.mae is None
    assert lgbm_metric.available is True
    assert lgbm_metric.mae is not None
    assert stat_metric.available is True
    assert stat_metric.mae is not None


def test_end_to_end_api_workflow():
    """Full end-to-end workflow test across all FastAPI endpoints."""
    client = TestClient(app)

    # 1. Health check
    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    h_data = r_health.json()
    assert h_data["application"] == "Safeguarding Forecast AI"
    assert h_data["organisation"] == "South Devon College"
    assert h_data["governance_banner"] == SYNTHETIC_BANNER

    # 2. Dataset summary
    r_sum = client.get("/api/dataset/summary")
    assert r_sum.status_code == 200
    assert r_sum.json()["total_months"] >= 36

    # 3. Download sample CSV template and validate/import it
    r_csv = client.get("/api/dataset/sample-csv")
    assert r_csv.status_code == 200
    csv_bytes = r_csv.content

    r_val = client.post(
        "/api/dataset/validate-csv",
        files={"file": ("sample.csv", csv_bytes, "text/csv")},
    )
    assert r_val.status_code == 200
    assert r_val.json()["can_import"] is True

    r_imp = client.post(
        "/api/dataset/import-csv",
        files={"file": ("sample.csv", csv_bytes, "text/csv")},
        data={"dataset_name": "E2E Test Imported Dataset"},
    )
    assert r_imp.status_code == 200
    imp_summary = r_imp.json()
    assert imp_summary["is_synthetic"] is False
    assert imp_summary["dataset_id"].startswith("imported_")

    # Switch back to synthetic demo dataset to ensure datasets are kept separate
    r_sw = client.post("/api/dataset/switch", json={"dataset_id": "synthetic_sdc_default"})
    assert r_sw.status_code == 200
    assert r_sw.json()["is_synthetic"] is True

    # 4. Forecast generation with baseline and scenario
    r_fc = client.post(
        "/api/forecast",
        json={
            "model_id": "lightgbm_mlforecast",
            "horizon": 12,
            "provision": "All",
            "safeguarding_category": "All",
            "prediction_interval": 80,
            "scenario": {
                "enrolment_growth_pct": 10.0,
                "teaching_days_delta": 1,
                "attendance_rate_delta_pp": -2.0,
            },
        },
    )
    assert r_fc.status_code == 200
    fc_data = r_fc.json()
    assert fc_data["annual_summary"]["predicted_total_concerns"] > fc_data["baseline_annual_summary"]["predicted_total_concerns"]

    # 5. Model comparison backtest
    r_bt = client.get("/api/models/backtest?horizon=12")
    assert r_bt.status_code == 200
    bt_data = r_bt.json()
    assert len(bt_data["metrics"]) == 4
    assert bt_data["best_model_id"] in ("statistical_count", "lightgbm_mlforecast", "timesfm_3", "seasonal_naive")

    # 6. Trends & Insights
    r_tr = client.get("/api/trends")
    assert r_tr.status_code == 200
    tr_data = r_tr.json()
    assert tr_data["causation_disclaimer"] == "These relationships are statistical associations, not evidence of causation."
    assert len(tr_data["feature_importances"]) > 0

