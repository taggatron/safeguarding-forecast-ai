"""
Phase 5 Mandatory Verification Gate 2: Point-in-Time Vintage Non-Leakage & September Reset Suite
(`tests/test_vintage_and_september_reset.py`).

Verifies:
  1. Injecting synthetic late-logged Friday/weekend/holiday disclosures (`incident_date = t_0 - 2`,
     `logged_date = t_0 + 4`) leaves all backfilled esoteric features, UI telemetry JSON, and
     residual forecasts at `t_0` 100% invariant (`atol=1e-12`), while causally admitting them
     once replay advances to `t_1 = t_0 + 4`.
  2. Perturbing all records with `logged_date > t_0` by `+10^6` produces zero backward bleed
     into `s <= t_0` (`atol=1e-12`).
  3. Corrupting Year `y+1` cohort graphs produces zero backward bleed into Year `y` (`atol=1e-12`),
     and `AcademicYearTopologyManager` triggers explicit September resets with Weeks 1-4 decayed
     warmup priors (`w_prior > 0` in Weeks 1-4, `w_prior == 0.0` from Week 5 onward).
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from esoteric_math.retrospective import (
    AcademicYearTopologyManager,
    RolloutMode,
    VintageSnapshotFilter,
    extract_vintage_replay_payload,
    fit_vintage_esoteric_suite,
)
from esoteric_math.schema_bridge import build_instructional_day_telemetry_frame


def _assert_nested_json_close(obj_a, obj_b, atol: float = 1e-12) -> None:
    if isinstance(obj_a, dict) and isinstance(obj_b, dict):
        # Ignore audit counters that intentionally record how many future rows were excluded!
        skip_audit_count_keys = {
            "total_rows_input",
            "excluded_future_logged_rows",
            "excluded_future_incident_rows",
        }
        assert set(obj_a.keys()) == set(obj_b.keys())
        for k in obj_a:
            if k in skip_audit_count_keys:
                continue
            _assert_nested_json_close(obj_a[k], obj_b[k], atol=atol)
    elif isinstance(obj_a, list) and isinstance(obj_b, list):
        assert len(obj_a) == len(obj_b)
        for va, vb in zip(obj_a, obj_b):
            _assert_nested_json_close(va, vb, atol=atol)
    elif isinstance(obj_a, (int, float, np.number)) and isinstance(obj_b, (int, float, np.number)):
        assert abs(float(obj_a) - float(obj_b)) <= atol, f"Float drift: {obj_a} vs {obj_b}"
    else:
        assert obj_a == obj_b, f"Value mismatch: {obj_a!r} vs {obj_b!r}"


def test_late_logged_disclosures_excluded_at_t0_and_admitted_at_logged_date() -> None:
    base_df = build_instructional_day_telemetry_frame(
        num_days=150,
        start_date="2024-09-02",
        seed=42,
    )
    # Attach explicit dual timestamps: timely disclosures have logged_date == date
    base_df["incident_date"] = base_df["date"]
    base_df["logged_date"] = base_df["date"]

    t0 = "2024-11-14"
    incident_dt = "2024-11-12"  # t_0 - 2 days
    logged_dt = "2024-11-18"    # t_0 + 4 days (Monday post-weekend logging!)

    clean_res = extract_vintage_replay_payload(
        base_df,
        cutoff_date=t0,
        retro_mode=RolloutMode.STAGED_CORE_4,
        burn_in_rows=48,
    )

    # Construct late-logged Friday/weekend disclosure occurring at `t_0 - 2` but logged at `t_0 + 4`
    late_row = base_df[base_df["date"] == incident_dt].iloc[0].to_dict()
    late_row["date"] = incident_dt
    late_row["incident_date"] = incident_dt
    late_row["logged_date"] = logged_dt
    late_row["referral_count"] = 450
    late_row["cat_online_safety"] = 220
    late_row["cat_mental_health"] = 230
    late_row["cohort_epyp_send_referrals"] = 250

    late_df = pd.DataFrame([late_row])
    injected_res = extract_vintage_replay_payload(
        base_df,
        cutoff_date=t0,
        retro_mode=RolloutMode.STAGED_CORE_4,
        late_disclosures_df=late_df,
        burn_in_rows=48,
    )

    # Verify late disclosure was detected and excluded at t_0
    assert injected_res["vintage_audit"]["has_dual_timestamps"] is True
    assert injected_res["vintage_audit"]["excluded_future_logged_rows"] == 1

    # Every feature, UI telemetry payload, and residual forecast at t_0 MUST be bit-for-bit identical!
    _assert_nested_json_close(clean_res, injected_res, atol=1e-12)

    # Now advance replay clock to t_1 = "2024-11-18" (logged_date <= t_1): late disclosure enters history!
    v_filter = VintageSnapshotFilter()
    snap_t1_clean = v_filter.filter_snapshot(base_df, cutoff_date=logged_dt)
    snap_t1_with_late = v_filter.filter_snapshot(base_df, cutoff_date=logged_dt, late_disclosures_df=late_df)

    cnt_clean = float(snap_t1_clean.loc[snap_t1_clean["date"] == incident_dt, "referral_count"].iloc[0])
    cnt_late = float(snap_t1_with_late.loc[snap_t1_with_late["date"] == incident_dt, "referral_count"].iloc[0])
    assert abs(cnt_late - (cnt_clean + 450.0)) <= 1e-12


def test_million_spike_on_logged_date_after_t0_has_zero_backward_bleed() -> None:
    df = build_instructional_day_telemetry_frame(
        num_days=140,
        start_date="2024-09-02",
        seed=19,
    )
    df["logged_date"] = df["date"]
    t0 = "2024-11-08"

    clean_snap = VintageSnapshotFilter().filter_snapshot(df, cutoff_date=t0)
    clean_suite, _ = fit_vintage_esoteric_suite(clean_snap, causal_mode="forecast")
    clean_eso = clean_suite.transform(clean_snap)
    clean_payload = extract_vintage_replay_payload(df, cutoff_date=t0, burn_in_rows=45)

    # Perturb EVERY record with logged_date > t0 by +1,000,000 (including rows backdated to incident <= t0!)
    corrupted = df.copy()
    future_log_mask = pd.to_datetime(corrupted["logged_date"]) > pd.Timestamp(t0)
    assert int(future_log_mask.sum()) >= 25
    num_cols = [
        c
        for c in corrupted.select_dtypes(include=[np.number]).columns
        if c not in ("is_instructional_day",)
    ]
    for c in num_cols:
        corrupted.loc[future_log_mask, c] = corrupted.loc[future_log_mask, c] + 1_000_000.0

    # Also append backdated bomb rows: date <= t0, logged_date > t0, count = +1,000,000
    backdated_bomb = corrupted.iloc[10:15].copy()
    backdated_bomb["logged_date"] = "2024-12-05"
    for c in num_cols:
        backdated_bomb[c] = 1_000_000.0
    corrupted_all = pd.concat([corrupted, backdated_bomb], ignore_index=True)

    pert_snap = VintageSnapshotFilter().filter_snapshot(corrupted_all, cutoff_date=t0)
    pert_suite, _ = fit_vintage_esoteric_suite(pert_snap, causal_mode="forecast")
    pert_eso = pert_suite.transform(pert_snap)
    pert_payload = extract_vintage_replay_payload(corrupted_all, cutoff_date=t0, burn_in_rows=45)

    assert np.allclose(clean_eso.to_numpy(dtype=float), pert_eso.to_numpy(dtype=float), atol=1e-12, rtol=0.0)
    _assert_nested_json_close(clean_payload, pert_payload, atol=1e-12)


def test_corrupted_next_academic_year_cohort_graph_and_september_reset_lifecycle() -> None:
    """
    Verify:
      1. Corrupting Year `y+1` (`2025/26`) cohort stalks & counts has 0.0 effect (`atol=1e-12`)
         on Year `y` (`2024/25`) features, Sheaf telemetry, and residual forecasts.
      2. September reset activates decayed Year `y-1` (`2023/24`) priors during Weeks 1-4 of
         September `2024/25`, then drops `warmup_prior_weight` to exact `0.0` from Week 5 onward
         (so even Year `y-1` cohort stalk corruption cannot bleed into post-Week-4 Year `y` Tier C!).
    """
    multi_yr = build_instructional_day_telemetry_frame(
        num_days=755,
        start_date="2023-09-04",
        seed=42,
    )
    t_eval_yr2 = "2024-11-12"  # Week 11 of AY 2024/25 (post-warmup)

    clean_yr2 = extract_vintage_replay_payload(
        multi_yr,
        cutoff_date=t_eval_yr2,
        retro_mode=RolloutMode.FULL_13_GATED,
        burn_in_rows=364,
    )

    # Corrupt entire Year y+1 (`2025/26`) cohort graphs by +10^6 and sign flips
    corrupted_yr3 = multi_yr.copy()
    yr3_mask = corrupted_yr3["academic_year"] == "2025/26"
    assert int(yr3_mask.sum()) >= 20
    cohort_cols = [c for c in corrupted_yr3.columns if c.startswith("cohort_")]
    for c in cohort_cols:
        corrupted_yr3.loc[yr3_mask, c] = -999_999.0 if "attendance" in c else 1_000_000.0
    corrupted_yr3.loc[yr3_mask, "referral_count"] = 1_000_000

    pert_yr2 = extract_vintage_replay_payload(
        corrupted_yr3,
        cutoff_date=t_eval_yr2,
        retro_mode=RolloutMode.FULL_13_GATED,
        burn_in_rows=364,
    )
    _assert_nested_json_close(clean_yr2, pert_yr2, atol=1e-12)

    # Verify September Topology Reset warmup decay schedule across Weeks 1..6 of September/October 2024
    topo = AcademicYearTopologyManager(warmup_weeks=4)
    _, st_wk1 = topo.prepare_topology_fit_slice(multi_yr, cutoff_date="2024-09-05")
    _, st_wk3 = topo.prepare_topology_fit_slice(multi_yr, cutoff_date="2024-09-19")
    fit_wk6, st_wk6 = topo.prepare_topology_fit_slice(multi_yr, cutoff_date="2024-10-14")

    assert st_wk1.active_academic_year == "2024/25"
    assert st_wk1.last_september_reset_date == "2024-09-01"
    assert st_wk1.warmup_prior_active is True and st_wk1.warmup_prior_weight == 1.0
    assert st_wk3.warmup_prior_active is True and 0.0 < st_wk3.warmup_prior_weight < st_wk1.warmup_prior_weight
    assert st_wk6.warmup_prior_active is False and st_wk6.warmup_prior_weight == 0.0
    # Post-warmup topology slice must contain ONLY Academic Year 2024/25 rows!
    assert set(fit_wk6["academic_year"].unique().tolist()) == {"2024/25"}

    # Ensure JSON serialization has zero NaNs/Infs
    assert len(json.dumps(clean_yr2, allow_nan=False)) > 500
