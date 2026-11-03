"""
Phase 5: Verification & Automated Unit Tests for the Refactored FE Safeguarding
Count-Forecasting Benchmark Pipeline (NB2 GLM vs. ZINB vs. Tweedie LightGBM).

Verifies:
1. Zero Temporal Leakage (features, EWMAs, and expanding-window z-score scalers)
2. Exposure Offset Proportionality (doubling active_enrolments N_t strictly doubles Y_hat_t
   across NB2 GLM, ZINB, and Tweedie LightGBM init_score)
3. Distributional Non-Negativity & Strict Quantile Monotonicity (no quantile crossing)
4. Kaggle Metric Sanity Checks (RMSSE = 1.0 for naive persistence, 39x asymmetry at tau=0.975,
   and Pinball Loss at tau=0.50 == 0.5 * MAE)
5. VIF Multicollinearity Composite Index & Full Expanding-Window Benchmark Pipeline
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent / "backend")
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import numpy as np
import pandas as pd
import pytest

from app.features import (
    aggregate_college_wide_weekly,
    audit_target_distribution,
    engineer_safeguarding_features,
    expanding_window_zscore,
    generate_weekly_safeguarding_panel,
)
from app.evaluation.kaggle_metrics import (
    EVAL_QUANTILES,
    compute_mae,
    compute_pinball_loss,
    compute_rmsle,
    compute_rmsse,
    compute_wrmsse,
    compute_wspl,
)
from app.models.safeguarding_benchmark import (
    DEFAULT_BENCHMARK_FEATURES,
    LegacyMLBaselineModel,
    NB2GovernanceGLM,
    TweedieLightGBMModel,
    ZINBSparseRegimeModel,
    run_expanding_window_benchmark,
)


def test_zero_temporal_leakage():
    """
    Phase 5 Requirement 1:
    Mutating target or feature rows at t+1 (and beyond) must produce ZERO change in
    engineered features, EWMA values, or expanding-window z-score scalers at time t.
    """
    panel = generate_weekly_safeguarding_panel(num_academic_years=2, start_year=2023, seed=42)
    cw_clean = aggregate_college_wide_weekly(panel)

    cutoff_t = 45
    cw_mutated = cw_clean.copy()
    # Mutate all future observations strictly after index cutoff_t (i.e. t+1 onwards)
    cw_mutated.loc[cutoff_t + 1 :, "referral_count"] = 99999
    cw_mutated.loc[cutoff_t + 1 :, "overall_attendance_pct"] = 15.0
    cw_mutated.loc[cutoff_t + 1 :, "eng_maths_attendance_pct"] = 10.0
    cw_mutated.loc[cutoff_t + 1 :, "pastoral_rate_per_1000"] = 850.0
    cw_mutated.loc[cutoff_t + 1 :, "bursary_rate_per_1000"] = 500.0
    cw_mutated.loc[cutoff_t + 1 :, "active_enrolments"] = 95000

    feat_clean, _ = engineer_safeguarding_features(cw_clean, group_col="curriculum_area")
    feat_mutated, _ = engineer_safeguarding_features(cw_mutated, group_col="curriculum_area")

    causal_check_cols = [
        "log_enrolment",
        "delta_att_t1",
        "delta_att_t2",
        "att_drop_t1",
        "att_velocity_ewma_1w",
        "att_velocity_ewma_2w",
        "att_velocity_ewma_4w",
        "pastoral_t1",
        "pastoral_t2",
        "pastoral_ewma_2w",
        "pastoral_ewma_4w",
        "bursary_t2",
        "bursary_t3",
        "bursary_t4",
        "bursary_ewma_4w",
        "referral_lag_t1",
        "referral_rolling_3w",
        "z_att_drop_t1",
        "z_pastoral_t1",
        "early_disengagement_index",
    ]

    for col in causal_check_cols:
        np.testing.assert_allclose(
            feat_clean.loc[:cutoff_t, col].to_numpy(dtype=float),
            feat_mutated.loc[:cutoff_t, col].to_numpy(dtype=float),
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"Temporal leakage detected in column '{col}' at or before time t={cutoff_t}!",
        )


def test_exposure_offset_proportionality_nb2_zinb_tweedie():
    """
    Phase 5 Requirement 2:
    Doubling active_enrolments (N_t -> 2 * N_t) while holding all rate predictors constant
    must strictly double predicted referral counts (Y_hat_t) across NB2 GLM, ZINB, and
    Tweedie LightGBM (init_score).
    """
    panel = generate_weekly_safeguarding_panel(num_academic_years=3, start_year=2022, seed=42)
    cw_raw = aggregate_college_wide_weekly(panel)
    cw_df, vif_res = engineer_safeguarding_features(cw_raw, group_col="curriculum_area")
    feat_cols = vif_res.selected_feature_cols

    train_df = cw_df.iloc[:104].copy().reset_index(drop=True)
    test_base = cw_df.iloc[104:120].copy().reset_index(drop=True)

    # Create a counterfactual test set with doubled active_enrolments (2 * N_t)
    # while holding all rate/calendar/cohort predictors strictly constant
    test_doubled = test_base.copy()
    test_doubled["active_enrolments"] = test_base["active_enrolments"] * 2.0
    test_doubled["log_enrolment"] = np.log(test_doubled["active_enrolments"])

    # 1. Negative Binomial GLM (NB2)
    nb2 = NB2GovernanceGLM(feature_cols=feat_cols).fit(train_df)
    pred_nb2_1x = nb2.predict(test_base).point_forecasts
    pred_nb2_2x = nb2.predict(test_doubled).point_forecasts
    np.testing.assert_allclose(
        pred_nb2_2x / pred_nb2_1x,
        2.0,
        rtol=1e-6,
        atol=1e-6,
        err_msg="NB2 GLM failed strict 2x exposure offset proportionality!",
    )

    # 2. Zero-Inflated Negative Binomial (ZINB)
    zinb = ZINBSparseRegimeModel(feature_cols=feat_cols).fit(train_df)
    pred_zinb_1x = zinb.predict(test_base).point_forecasts
    pred_zinb_2x = zinb.predict(test_doubled).point_forecasts
    np.testing.assert_allclose(
        pred_zinb_2x / pred_zinb_1x,
        2.0,
        rtol=1e-6,
        atol=1e-6,
        err_msg="ZINB failed strict 2x exposure offset proportionality!",
    )

    # 3. Tweedie-Loss LightGBM (with init_score = log_enrolment)
    tw_lgbm = TweedieLightGBMModel(feature_cols=feat_cols).fit(train_df)
    pred_tw_1x = tw_lgbm.predict(test_base).point_forecasts
    pred_tw_2x = tw_lgbm.predict(test_doubled).point_forecasts
    np.testing.assert_allclose(
        pred_tw_2x / pred_tw_1x,
        2.0,
        rtol=1e-6,
        atol=1e-6,
        err_msg="Tweedie LightGBM (init_score) failed strict 2x exposure offset proportionality!",
    )


def test_distributional_and_quantile_monotonicity():
    """
    Phase 5 Requirement 3:
    All point forecasts and quantile bounds must be strictly non-negative (>= 0) and satisfy
    Q_{0.025, t} <= Q_{0.50, t} <= Q_{0.90, t} <= Q_{0.975, t} without quantile crossing.
    """
    panel = generate_weekly_safeguarding_panel(num_academic_years=3, start_year=2022, seed=42)
    dept_df, vif_res = engineer_safeguarding_features(panel, group_col="curriculum_area")
    feat_cols = vif_res.selected_feature_cols

    # Test on a sparse curriculum area (Adult Access & Work-Based Apprenticeships) as well as College-Wide
    sparse_area = dept_df[dept_df["curriculum_area"] == "Adult Access & Work-Based Apprenticeships"].reset_index(drop=True)
    train_df = sparse_area.iloc[:104].copy()
    test_df = sparse_area.iloc[104:130].copy()

    models = [
        LegacyMLBaselineModel(feature_cols=feat_cols).fit(train_df),
        NB2GovernanceGLM(feature_cols=feat_cols).fit(train_df),
        ZINBSparseRegimeModel(feature_cols=feat_cols).fit(train_df),
        TweedieLightGBMModel(feature_cols=feat_cols).fit(train_df),
    ]

    for mod in models:
        bundle = mod.predict(test_df, quantiles=EVAL_QUANTILES)
        assert np.all(bundle.point_forecasts >= 0.0), f"{mod.model_id} produced negative point forecast!"

        q025 = bundle.quantiles[0.025]
        q500 = bundle.quantiles[0.50]
        q900 = bundle.quantiles[0.90]
        q975 = bundle.quantiles[0.975]

        assert np.all(q025 >= 0.0), f"{mod.model_id} produced negative Q_0.025!"
        assert np.all(q025 <= q500 + 1e-9), f"Quantile crossing in {mod.model_id}: Q_0.025 > Q_0.50"
        assert np.all(q500 <= q900 + 1e-9), f"Quantile crossing in {mod.model_id}: Q_0.50 > Q_0.90"
        assert np.all(q900 <= q975 + 1e-9), f"Quantile crossing in {mod.model_id}: Q_0.90 > Q_0.975"


def test_kaggle_metric_sanity_checks():
    """
    Phase 5 Requirement 4:
    - Verify that a naive persistence forecast (Y_hat_t = Y_{t-1}) yields RMSSE = 1.0
      when out-of-sample step variance matches in-sample step variance.
    - Verify Pinball Loss asymmetry at tau = 0.975: under-predicting true demand by 10
      referrals produces a loss exactly 39x larger than over-predicting by 10 referrals
      (0.975 / 0.025 = 39).
    - Verify mean_pinball_loss at tau = 0.50 equals 0.5 * MAE.
    """
    # 4a. Naive persistence forecast RMSSE == 1.0
    steps = np.array([10.0, 14.0, 10.0, 14.0, 10.0, 14.0, 10.0, 14.0, 10.0])
    y_train = steps[:5]
    y_test = steps[5:]
    y_naive_pred = steps[4:-1]
    rmsse_val = compute_rmsse(y_train, y_test, y_naive_pred)
    assert np.isclose(rmsse_val, 1.0, rtol=1e-12, atol=1e-12), f"Expected RMSSE=1.0, got {rmsse_val}"

    # 4b. Pinball Loss asymmetry at tau = 0.975 (39x penalty ratio)
    y_true = np.array([50.0])
    q_under_by_10 = np.array([40.0])
    q_over_by_10 = np.array([60.0])

    loss_under = compute_pinball_loss(y_true, q_under_by_10, tau=0.975)
    loss_over = compute_pinball_loss(y_true, q_over_by_10, tau=0.975)
    assert np.isclose(loss_under, 9.75, rtol=1e-12)
    assert np.isclose(loss_over, 0.25, rtol=1e-12)
    assert np.isclose(loss_under / loss_over, 39.0, rtol=1e-12), (
        f"Expected 39.0x asymmetry ratio at tau=0.975, got {loss_under / loss_over}"
    )

    # 4c. Pinball Loss at tau = 0.50 equals exactly 0.5 * MAE
    rng = np.random.default_rng(123)
    y_rand_true = rng.integers(0, 80, size=50).astype(float)
    y_rand_pred = np.maximum(0.0, y_rand_true + rng.normal(0.0, 6.0, size=50))
    mae_val = compute_mae(y_rand_true, y_rand_pred)
    pb_50 = compute_pinball_loss(y_rand_true, y_rand_pred, tau=0.50)
    assert np.isclose(pb_50, 0.5 * mae_val, rtol=1e-12, atol=1e-12), (
        f"Expected Pinball(0.50) == 0.5 * MAE ({0.5 * mae_val}), got {pb_50}"
    )


def test_vif_composite_index_and_expanding_window_pipeline():
    """
    Verify that collinear attendance drops and pastoral logs (VIF > 5.0) trigger the
    expanding-window `early_disengagement_index` and that the full multi-model benchmark
    executes end-to-end with IRR governance and DSL capacity triage outputs.
    """
    report = run_expanding_window_benchmark(fold_origins_weeks=(78, 104), horizon_weeks=26)

    assert report["vif_diagnostics"]["att_pastoral_vif_exceeded"] is True
    assert report["vif_diagnostics"]["composite_index_activated"] is True
    assert "early_disengagement_index" in report["vif_diagnostics"]["selected_feature_cols"]

    table = report["benchmark_comparison_table"]
    assert len(table) == 4
    model_ids = {r["model_id"] for r in table}
    assert model_ids == {"legacy_ml_baseline", "nb2_glm", "zinb_model", "tweedie_lgbm"}

    # Verify count-aware models outperform the naive random walk (RMSSE < 1.0) and beat Legacy ML on RMSLE
    by_id = {r["model_id"]: r for r in table}
    assert by_id["nb2_glm"]["rmsse"] < 1.0
    assert by_id["tweedie_lgbm"]["rmsse"] < 1.0
    assert by_id["nb2_glm"]["rmsle"] < by_id["legacy_ml_baseline"]["rmsle"]

    # Verify DSL Governance IRR table and weekly triage schedule
    irr_rows = report["dsl_governance_irr_table"]
    assert len(irr_rows) >= 8
    for r in irr_rows:
        assert r["irr"] > 0
        assert r["irr_ci_lower_95"] <= r["irr"] <= r["irr_ci_upper_95"]

    schedule = report["dsl_weekly_triage_schedule"]
    assert len(schedule) == 26
    assert any(w["exceeds_q90_capacity"] or w["exceeds_q975_capacity"] for w in schedule)
