"""
Phase 4: Bolt-On Causal Telemetry Overlay & Historical Replay Bus
(`esoteric_math/retrospective/replay_telemetry_bus.py`).

Couples `VintageSnapshotFilter`, `AcademicYearTopologyManager`, `FrozenBaselineAdapter`,
and `PostHocResidualStacker` to the three low-cognitive-load pastoral telemetry panels
(Hilbert Phase Clocks, Hawkes Contagion Cascades, and Sheaf Frustration Alerts) without
touching or mutating any core forecast model or existing UI tab.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from app.models.safeguarding_benchmark import NB2GovernanceGLM
from esoteric_math.features import EsotericFeatureSuite
from esoteric_math.retrospective.frozen_adapter import FrozenBaselineAdapter
from esoteric_math.retrospective.residual_stacker import PostHocResidualStacker
from esoteric_math.retrospective.rollout_config import (
    ALL_13_MODULES,
    MODULE_DISPLAY_NAMES,
    RolloutConfig,
    RolloutMode,
)
from esoteric_math.retrospective.vintage_backfill import (
    AcademicYearTopologyManager,
    SeptemberTopologyState,
    VintageSnapshotFilter,
    fit_vintage_esoteric_suite,
    fit_vintage_telemetry_models,
    run_reporting_lag_sensitivity_audit,
)
from esoteric_math.schema_bridge import (
    _causal_ewma_1d,
    _causal_shift_1d,
    _infer_uk_fe_holiday_for_date,
    build_instructional_day_telemetry_frame,
)
from esoteric_math.viz.telemetry_payload import (
    FittedTelemetryModels,
    extract_telemetry_payload,
)


DAILY_BASELINE_FEATURE_COLS: List[str] = [
    "is_instructional_day",
    "is_autumn_transition",
    "is_post_holiday_week",
    "is_assessment_crunch",
    "lag1_rate_per_1000",
    "lag7_ewma_rate_per_1000",
]


def attach_causal_daily_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach strictly causal (`s < t`) baseline predictors onto a daily or weekly safeguarding
    DataFrame so `NB2GovernanceGLM` can serve as a frozen point-in-time baseline checkpoint.
    """
    out = df.copy().reset_index(drop=True)
    y = pd.to_numeric(out["referral_count"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if "active_enrolments" not in out.columns:
        out["active_enrolments"] = 7370.0
    enrol = np.maximum(1.0, pd.to_numeric(out["active_enrolments"], errors="coerce").fillna(7370.0).to_numpy(dtype=float))
    rate = (y / enrol) * 1000.0

    init_r = float(rate[0]) if len(rate) > 0 else 1.1
    lag1 = _causal_shift_1d(rate, lag=1, init_val=init_r)
    ewma7 = _causal_ewma_1d(lag1, alpha=2.0 / 8.0)

    out["lag1_rate_per_1000"] = np.round(lag1, 5)
    out["lag7_ewma_rate_per_1000"] = np.round(ewma7, 5)
    for col in ("is_instructional_day", "is_autumn_transition", "is_post_holiday_week", "is_assessment_crunch"):
        if col not in out.columns:
            out[col] = 1 if col == "is_instructional_day" else 0
    if "is_low_activity_week" not in out.columns:
        inst = pd.to_numeric(out["is_instructional_day"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        out["is_low_activity_week"] = (inst <= 0.15).astype(int)
    if "is_pre_holiday_week" not in out.columns:
        out["is_pre_holiday_week"] = 0
    return out


def extract_vintage_replay_payload(
    df: pd.DataFrame,
    cutoff_date: str,
    retro_mode: str | RolloutMode = RolloutMode.STAGED_CORE_4,
    lag_buffer_days: int = 0,
    late_disclosures_df: Optional[pd.DataFrame] = None,
    burn_in_rows: int = 60,
) -> Dict[str, Any]:
    """
    Standalone, zero-shared-state causal point-in-time evaluator for arbitrary DataFrames
    (used by strict vintage non-leakage & September cohort reset verification suites).

    Guarantees `atol=1e-12` invariance to:
      - Any late-logged disclosure with `logged_date > cutoff_date` (even if `incident_date <= cutoff_date`)
      - Any future row or corrupted Year `y+1` cohort graph (`date > cutoff_date`).
    """
    v_filter = VintageSnapshotFilter(lag_buffer_days=lag_buffer_days)
    snap_df = v_filter.filter_snapshot(
        df,
        cutoff_date=cutoff_date,
        late_disclosures_df=late_disclosures_df,
        lag_buffer_days=lag_buffer_days,
    )
    if len(snap_df) < 12:
        raise ValueError(f"Insufficient causal history ({len(snap_df)} rows) at cutoff_date={cutoff_date}.")

    snap_feat = attach_causal_daily_baseline_features(snap_df)
    topo_mgr = AcademicYearTopologyManager()
    telem_models, topo_state = fit_vintage_telemetry_models(snap_feat, topology_manager=topo_mgr)
    telemetry = extract_telemetry_payload(
        snap_feat,
        t_date=cutoff_date,
        fitted_models=telem_models,
    )

    cfg = RolloutConfig.from_mode(retro_mode)
    b_cut = max(16, min(int(burn_in_rows), len(snap_feat) - 4))
    base_raw = NB2GovernanceGLM(feature_cols=DAILY_BASELINE_FEATURE_COLS).fit(snap_feat.iloc[:b_cut])
    frozen = FrozenBaselineAdapter(base_raw)

    elig_mods = cfg.active_candidate_modules()
    if len(elig_mods) > 0:
        suite, _ = fit_vintage_esoteric_suite(
            snap_feat,
            enabled_modules=elig_mods,
            causal_mode="forecast",
            topology_manager=topo_mgr,
        )
        X_eso = suite.transform(snap_feat).reset_index(drop=True)
    else:
        X_eso = pd.DataFrame(index=snap_feat.index)

    stacker = PostHocResidualStacker(frozen, config=cfg).fit(snap_feat, X_eso)
    pred_bundle = stacker.predict(snap_feat.tail(1).reset_index(drop=True), X_eso.tail(1).reset_index(drop=True), tag="cutoff_t")
    frozen_bundle = frozen.predict(snap_feat.tail(1).reset_index(drop=True), tag="cutoff_t")
    chk = frozen.verify_all_checksums()

    mu_base = float(frozen_bundle.point_forecasts[0])
    mu_adj = float(pred_bundle.point_forecasts[0])
    delta_pct = ((mu_adj / max(1e-6, mu_base)) - 1.0) * 100.0 if not stacker.identity_floor_active_ else 0.0
    attr_list = pred_bundle.diagnostics.get("retrospective", {}).get("attributions", ["0.0% — Frozen Baseline Retained"])

    return {
        "selected_date": str(telemetry["selected_date"]),
        "retro_mode": cfg.mode.value,
        "telemetry": telemetry,
        "point_forecast_frozen": mu_base,
        "point_forecast_adjusted": mu_adj,
        "quantiles_frozen": {str(k): float(v[0]) for k, v in frozen_bundle.quantiles.items()},
        "quantiles_adjusted": {str(k): float(v[0]) for k, v in pred_bundle.quantiles.items()},
        "delta_pct": round(float(delta_pct), 4),
        "attribution": str(attr_list[0]),
        "vintage_audit": asdict(v_filter.last_audit_) if v_filter.last_audit_ else {},
        "september_topology": asdict(topo_state),
        "baseline_sha256": chk,
    }


class ReplayTelemetryBus:
    """
    Cached Multi-Year Point-in-Time Replay & Telemetry Overlay Controller for FastAPI & UI.

    Timeline spans 756 days (`2023-09-04` through `2025-09-28`):
      - Academic Year 1 (`2023/24`, indices `0..363`): Full 1-Academic-Year Burn-In
      - Academic Year 2 (`2024/25`, indices `364..727`): Includes September Topology Reset
        (`2024-09-02`), Autumn Induction Spike (`2024-09-24`), Pre-Half-Term Masking (`2024-10-18`),
        Autumn Half-Term Freeze (`2024-10-29`), January Return Wave (`2025-01-15`), & May Crunch (`2025-05-14`)
      - Academic Year 3 (`2025/26`, indices `728..755`): Includes second September Topology Reset
        (`2025-09-02`) demonstrating Weeks 1-4 Autumn prior decay warmup.

    Every pre-cached checkpoint at index `c` is fitted strictly on `df.iloc[:c + 1]`,
    guaranteeing zero lookahead (`s > t`) at any scrubber position `t >= c`.
    """

    def __init__(
        self,
        num_days: int = 756,
        start_date: str = "2023-09-04",
        burn_in_days: int = 364,
        seed: int = 42,
    ) -> None:
        self.num_days = num_days
        self.start_date = start_date
        self.burn_in_days = burn_in_days
        self.seed = seed
        self._prov_cache: Dict[str, Dict[str, Any]] = {}

    def _get_or_build_provision(self, provision: str) -> Dict[str, Any]:
        prov_key = provision if provision in ("All", "EPYP", "Adult", "Apprenticeships") else "All"
        if prov_key in self._prov_cache:
            return self._prov_cache[prov_key]

        raw_df = build_instructional_day_telemetry_frame(
            num_days=self.num_days,
            start_date=self.start_date,
            seed=self.seed,
            provision_filter=prov_key,
        )
        df = attach_causal_daily_baseline_features(raw_df)

        # Frozen baseline model trained strictly on Academic Year 1 burn-in (`0..363`, i.e. 2023/24)
        burn_slice = df.iloc[: self.burn_in_days].copy().reset_index(drop=True)
        raw_base = NB2GovernanceGLM(feature_cols=DAILY_BASELINE_FEATURE_COLS).fit(burn_slice)
        frozen_adapter = FrozenBaselineAdapter(raw_base, label="Frozen NB2 Daily Baseline (2023/24 Burn-In)")
        # Record immutable out-of-sample SHA-256 across Years 2 & 3
        _ = frozen_adapter.predict(df.iloc[self.burn_in_days :].reset_index(drop=True), tag="replay_years_2_3")

        # Build strictly causal checkpoints (`ckpt_idx <= t`) across September resets & post-warmup milestones:
        #   - 364: 2024-09-02 (Start of AY 2024/25 — September Reset #1, Week 1 warmup active)
        #   - 415: 2024-10-23 (Post-Warmup AY 2024/25 — pure 2024/25 cohort topology)
        #   - 728: 2025-09-01 (Start of AY 2025/26 — September Reset #2, Week 1 warmup active)
        ckpt_indices = [
            idx
            for idx in (self.burn_in_days, 415, 728)
            if idx < len(df)
        ]
        checkpoints: Dict[int, Dict[str, Any]] = {}
        topo_mgr = AcademicYearTopologyManager()

        for ckpt in ckpt_indices:
            prefix_ckpt = df.iloc[: ckpt + 1].copy().reset_index(drop=True)
            telem_m, topo_st = fit_vintage_telemetry_models(prefix_ckpt, topology_manager=topo_mgr)
            suite_m, _ = fit_vintage_esoteric_suite(
                prefix_ckpt,
                causal_mode="forecast",
                topology_manager=topo_mgr,
            )
            X_eso_ckpt = suite_m.transform(prefix_ckpt).reset_index(drop=True)

            stackers_by_mode: Dict[str, PostHocResidualStacker] = {}
            for r_mode in (RolloutMode.TELEMETRY_ONLY, RolloutMode.STAGED_CORE_4, RolloutMode.FULL_13_GATED):
                cfg_m = RolloutConfig(
                    mode=r_mode,
                    boruta_n_trials=15,
                    boruta_min_hits=11,
                )
                stackers_by_mode[r_mode.value] = PostHocResidualStacker(
                    frozen_adapter,
                    config=cfg_m,
                ).fit(prefix_ckpt, X_eso_ckpt)

            checkpoints[ckpt] = {
                "ckpt_index": ckpt,
                "ckpt_date": str(df["date"].iloc[ckpt]),
                "academic_year": topo_st.active_academic_year,
                "telem_models": telem_m,
                "eso_suite": suite_m,
                "stackers": stackers_by_mode,
            }

        # Verify frozen baseline remained 100% bit-for-bit untouched across all checkpoint stacker fits
        sha_proof = frozen_adapter.verify_all_checksums()

        scrubber = self._build_scrubber_catalog(
            all_dates=df["date"].tolist(),
            min_step_index=self.burn_in_days,
        )

        admitted_by_year = self._summarize_admitted_modules_by_year(checkpoints)

        entry = {
            "df": df,
            "frozen_adapter": frozen_adapter,
            "sha_proof": sha_proof,
            "checkpoints": checkpoints,
            "scrubber": scrubber,
            "admitted_by_year": admitted_by_year,
        }
        self._prov_cache[prov_key] = entry
        return entry

    @staticmethod
    def _summarize_admitted_modules_by_year(
        checkpoints: Dict[int, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Build per-academic-year Admitted vs. Quarantined module ledger for the Audit Accordion."""
        by_mode_year: Dict[str, Dict[str, Any]] = {}
        for mode_val in (
            RolloutMode.TELEMETRY_ONLY.value,
            RolloutMode.STAGED_CORE_4.value,
            RolloutMode.FULL_13_GATED.value,
        ):
            yr_records: List[Dict[str, Any]] = []
            seen_years = set()
            # Prefer post-warmup checkpoint for 2024/25 (`400` or `530`), plus `728` for 2025/26
            ordered_ckpts = sorted(checkpoints.keys(), reverse=True)
            for ckpt in sorted(ordered_ckpts):
                c_obj = checkpoints[ckpt]
                ay = str(c_obj["academic_year"])
                # For 2024/25, use the mature expanding checkpoint as representative year summary
                if ay == "2024/25" and ckpt != max(k for k, v in checkpoints.items() if v["academic_year"] == "2024/25"):
                    continue
                if ay in seen_years:
                    continue
                seen_years.add(ay)
                st: PostHocResidualStacker = c_obj["stackers"][mode_val]
                aud = st.audit_
                adm = list(aud.admitted_modules) if aud else []
                quar = list(aud.quarantined_modules) if aud else list(ALL_13_MODULES)
                yr_records.append(
                    {
                        "academic_year": ay,
                        "checkpoint_date": c_obj["ckpt_date"],
                        "admitted_modules": adm,
                        "admitted_labels": [f"{m} ({MODULE_DISPLAY_NAMES.get(m, m)})" for m in adm],
                        "quarantined_modules": quar,
                        "quarantined_count": len(quar),
                        "selected_features": list(aud.selected_features) if aud else [],
                        "max_vif": float(aud.max_vif) if aud else 1.0,
                        "identity_floor_active": bool(st.identity_floor_active_),
                    }
                )
            by_mode_year[mode_val] = {"years": yr_records}
        return by_mode_year

    @staticmethod
    def _build_scrubber_catalog(all_dates: List[str], min_step_index: int) -> Dict[str, Any]:
        landmarks: List[Dict[str, Any]] = []
        target_specs = [
            ("2024-09-02", "Sept 2024 Topology Reset", "Annual September cohort & Sheaf stalk reset (Weeks 1–4 prior decay)"),
            ("2024-09-24", "Autumn Induction Spike", "Early-Autumn enrolment & school file handover surge"),
            ("2024-10-18", "Pre-Half-Term Masking", "Cohomological contextual masking ahead of October break"),
            ("2024-10-29", "Autumn Half-Term Freeze", "Non-instructional break (Hilbert phase clock frozen)"),
            ("2024-11-04", "Post-Half-Term Return", "Accumulated holiday disclosure step-jump on Monday return"),
            ("2025-01-15", "January Return Wave", "Post-winter termly incubation & peer cascade monitoring"),
            ("2025-05-14", "May Exam Crunch", "GCSE resit & vocational assessment pressure window"),
            ("2025-09-02", "Sept 2025 Topology Reset", "2025/26 intake rollover & Sheaf cohort restriction reset"),
        ]
        for target_dt, title, desc in target_specs:
            eligible = [i for i, d in enumerate(all_dates) if d <= target_dt and i >= min_step_index]
            if eligible:
                idx = eligible[-1]
                landmarks.append(
                    {
                        "step_index": int(idx),
                        "date": all_dates[idx],
                        "label": title,
                        "description": desc,
                    }
                )
        if all_dates:
            landmarks.append(
                {
                    "step_index": len(all_dates) - 1,
                    "date": all_dates[-1],
                    "label": "Latest Validation Step",
                    "description": "Most recent causal validation observation",
                }
            )

        holiday_flags = []
        for d in all_dates:
            frac, lbl = _infer_uk_fe_holiday_for_date(pd.Timestamp(d), "daily")
            holiday_flags.append({"date": d, "is_instructional": bool(frac > 0.5), "holiday_label": lbl})

        return {
            "available_dates": all_dates,
            "min_step_index": int(min_step_index),
            "max_step_index": len(all_dates) - 1,
            "default_date": all_dates[-1] if all_dates else "",
            "landmarks": landmarks,
            "calendar_strip": holiday_flags,
        }

    def _select_causal_checkpoint(
        self,
        checkpoints: Dict[int, Dict[str, Any]],
        eff_idx: int,
    ) -> Dict[str, Any]:
        eligible_keys = [k for k in sorted(checkpoints.keys()) if k <= eff_idx]
        chosen_key = eligible_keys[-1] if eligible_keys else min(checkpoints.keys())
        return checkpoints[chosen_key]

    def get_replay_response(
        self,
        t_date: Optional[str] = None,
        t_index: Optional[int] = None,
        provision: str = "All",
        retro_mode: str | RolloutMode = RolloutMode.STAGED_CORE_4,
        lag_buffer_days: int = 0,
    ) -> Dict[str, Any]:
        entry = self._get_or_build_provision(provision)
        df: pd.DataFrame = entry["df"]
        scrubber: Dict[str, Any] = entry["scrubber"]
        frozen_adapter: FrozenBaselineAdapter = entry["frozen_adapter"]
        all_dates: List[str] = scrubber["available_dates"]

        coerced_mode = RolloutMode.coerce(retro_mode)
        cfg = RolloutConfig.from_mode(coerced_mode)

        # Resolve target causal step index
        if t_date is not None and str(t_date).strip() != "":
            target_iso = pd.Timestamp(str(t_date)).strftime("%Y-%m-%d")
            elig = [i for i, d in enumerate(all_dates) if d <= target_iso]
            eff_idx = elig[-1] if elig else scrubber["min_step_index"]
        elif t_index is not None:
            eff_idx = int(np.clip(int(t_index), 10, len(df) - 1))
        else:
            eff_idx = len(df) - 1

        selected_iso = all_dates[eff_idx]

        # 1. Apply VintageSnapshotFilter strictly <= selected_iso
        v_filter = VintageSnapshotFilter(lag_buffer_days=lag_buffer_days)
        snap_df = v_filter.filter_snapshot(df, cutoff_date=selected_iso, lag_buffer_days=lag_buffer_days)
        snap_feat = attach_causal_daily_baseline_features(snap_df)

        # 2. Compute exact September Topology Reset metadata at `selected_iso`
        topo_mgr = AcademicYearTopologyManager()
        _, topo_state = topo_mgr.prepare_topology_fit_slice(snap_feat, cutoff_date=selected_iso)

        # 3. Select strictly causal checkpoint (`ckpt_idx <= eff_idx`)
        ckpt = self._select_causal_checkpoint(entry["checkpoints"], eff_idx)
        telem_models: FittedTelemetryModels = ckpt["telem_models"]
        eso_suite: EsotericFeatureSuite = ckpt["eso_suite"]
        stacker: PostHocResidualStacker = ckpt["stackers"][coerced_mode.value]

        # 4. Extract standard 3-panel telemetry payload strictly from `snap_feat` (`<= selected_iso`)
        telemetry = extract_telemetry_payload(
            snap_feat,
            t_date=selected_iso,
            fitted_models=telem_models,
        )

        # 5. Build Retrospective Forecast Overlay (`[t - 27 .. t]` observed + `(t, t + 5]` deterministic calendar horizon)
        win_start = max(0, len(snap_feat) - 28)
        future_horizon = 5
        # Construct deterministic future calendar rows without reading future observed referral counts!
        fut_rows: List[Dict[str, Any]] = []
        last_dt = pd.Timestamp(selected_iso)
        last_enrol = float(snap_feat["active_enrolments"].iloc[-1])
        recent_rates = (
            snap_feat["referral_count"].tail(14).to_numpy(dtype=float) / max(1.0, last_enrol)
        ) * 1000.0
        ewma_r = float(_causal_ewma_1d(recent_rates, alpha=0.25)[-1]) if len(recent_rates) > 0 else 1.1
        was_hol = bool(snap_feat["is_instructional_day"].iloc[-1] <= 0.15)

        for h_k in range(1, future_horizon + 1):
            f_dt = last_dt + pd.Timedelta(days=h_k)
            frac_i, hol_l = _infer_uk_fe_holiday_for_date(f_dt, "daily")
            is_inst = 1 if frac_i > 0.5 else 0
            is_ret = 1 if (is_inst == 1 and was_hol and f_dt.dayofweek == 0 and hol_l == "") else 0
            was_hol = (is_inst == 0)
            fut_rows.append(
                {
                    "date": f_dt.strftime("%Y-%m-%d"),
                    "academic_year": f"{f_dt.year}/{str(f_dt.year + 1)[-2:]}" if f_dt.month >= 9 else f"{f_dt.year - 1}/{str(f_dt.year)[-2:]}",
                    "is_instructional_day": is_inst,
                    "is_low_activity_week": 1 - is_inst,
                    "is_pre_holiday_week": 0,
                    "holiday_label": hol_l,
                    "referral_count": 0.0,  # Placeholder never read in forecast mode for h_k=1; projected via ewma for h_k>1
                    "active_enrolments": last_enrol,
                    "is_autumn_transition": 1 if (f_dt.month == 9 or (f_dt.month == 10 and f_dt.day <= 18)) else 0,
                    "is_post_holiday_week": is_ret,
                    "is_assessment_crunch": 1 if f_dt.month in (11, 5) else 0,
                    "lag1_rate_per_1000": ewma_r,
                    "lag7_ewma_rate_per_1000": ewma_r,
                }
            )
            # Carry forward last observed cohort/category state causally so schema_bridge never sees zeros
            for col in snap_feat.columns:
                if col not in fut_rows[-1]:
                    fut_rows[-1][col] = snap_feat[col].iloc[-1]

        combined_eval = pd.concat([snap_feat, pd.DataFrame(fut_rows)], ignore_index=True)
        # Keep trailing 96 steps for fast causal esoteric transformation
        eso_tail_start = max(0, len(snap_feat) - 90)
        eval_sub = combined_eval.iloc[eso_tail_start:].reset_index(drop=True)
        X_eso_sub = eso_suite.transform(eval_sub).reset_index(drop=True)

        # Slice exact overlay window `[win_start .. len(snap_feat) + future_horizon)`
        rel_win_start = win_start - eso_tail_start
        overlay_df = eval_sub.iloc[rel_win_start:].reset_index(drop=True)
        overlay_eso = X_eso_sub.iloc[rel_win_start:].reset_index(drop=True)

        frozen_ov = frozen_adapter.predict(overlay_df)
        adj_ov = stacker.predict(overlay_df, overlay_eso)
        sha_live = frozen_adapter.verify_model_immutability()

        retro_diag = adj_ov.diagnostics.get("retrospective", {})
        attributions = retro_diag.get(
            "attributions",
            ["0.0% — Frozen Baseline Retained"] * len(overlay_df),
        )

        n_hist_in_win = len(snap_feat) - win_start
        overlay_series: List[Dict[str, Any]] = []
        for i in range(len(overlay_df)):
            is_proj = i >= n_hist_in_win
            f_mu = float(frozen_ov.point_forecasts[i])
            a_mu = float(adj_ov.point_forecasts[i])
            d_pct = 0.0 if stacker.identity_floor_active_ else ((a_mu / max(1e-6, f_mu)) - 1.0) * 100.0
            overlay_series.append(
                {
                    "date": str(overlay_df["date"].iloc[i]),
                    "is_projection": bool(is_proj),
                    "is_instructional": bool(overlay_df["is_instructional_day"].iloc[i] > 0.2),
                    "observed_count": None if is_proj else round(float(overlay_df["referral_count"].iloc[i]), 2),
                    "frozen_baseline_mu": round(f_mu, 3),
                    "adjusted_final_mu": round(a_mu, 3),
                    "lower_95_frozen": round(float(frozen_ov.quantiles[0.025][i]), 2),
                    "upper_95_frozen": round(float(frozen_ov.quantiles[0.975][i]), 2),
                    "lower_95_adjusted": round(float(adj_ov.quantiles[0.025][i]), 2),
                    "upper_95_adjusted": round(float(adj_ov.quantiles[0.975][i]), 2),
                    "delta_count": round(a_mu - f_mu, 3),
                    "delta_pct": round(float(d_pct), 2),
                    "attribution": str(attributions[i]),
                }
            )

        cur_ov = overlay_series[max(0, n_hist_in_win - 1)]

        # 6. Fast 3-point single-timestamp Reporting-Lag Sensitivity check at `selected_iso` (`0, 2, 5` days)
        lag_window_df = df.iloc[max(0, eff_idx - 85) : eff_idx + 1].copy().reset_index(drop=True)
        lag_sensitivity = run_reporting_lag_sensitivity_audit(
            lag_window_df,
            cutoff_date=selected_iso,
            lag_buffer_days_list=(0, 2, 5),
            fitted_models=telem_models,
        )

        retrospective_overlay = {
            "retro_mode": coerced_mode.value,
            "identity_floor_active": bool(stacker.identity_floor_active_),
            "current_step": cur_ov,
            "series": overlay_series,
        }

        backfill_audit = {
            "rollout_badge": {
                **cfg.to_badge_dict(),
                "baseline_model_id": frozen_adapter.model_id,
                "baseline_model_name": frozen_adapter.model_name,
                "baseline_model_sha256": sha_live,
                "baseline_prediction_sha256": entry["sha_proof"]["verified_prediction_hashes"].get(
                    "replay_years_2_3", ""
                ),
                "baseline_immutability_verified": True,
            },
            "admitted_vs_quarantined_by_year": entry["admitted_by_year"][coerced_mode.value]["years"],
            "vintage_and_topology": {
                "vintage_cutoff_date": selected_iso,
                "schema_note": v_filter.last_audit_.schema_note if v_filter.last_audit_ else "",
                "has_dual_timestamps": bool(
                    v_filter.last_audit_.has_dual_timestamps if v_filter.last_audit_ else False
                ),
                "lag_buffer_days_active": int(lag_buffer_days),
                "excluded_future_incident_rows": int(
                    v_filter.last_audit_.excluded_future_incident_rows if v_filter.last_audit_ else 0
                ),
                "excluded_future_logged_rows": int(
                    v_filter.last_audit_.excluded_future_logged_rows if v_filter.last_audit_ else 0
                ),
                "active_academic_year": topo_state.active_academic_year,
                "last_september_reset_date": topo_state.last_september_reset_date,
                "weeks_into_academic_year": topo_state.weeks_into_academic_year,
                "instructional_days_into_academic_year": topo_state.instructional_days_into_academic_year,
                "warmup_prior_active": topo_state.warmup_prior_active,
                "warmup_prior_weight": topo_state.warmup_prior_weight,
                "prior_academic_year_used": topo_state.prior_academic_year_used,
            },
            "reporting_lag_sensitivity": lag_sensitivity,
        }

        return {
            "governance_banner": "SYNTHETIC DEMONSTRATION DATA — STRICTLY CAUSAL VINTAGE REPLAY (logged_date <= t)",
            "provision": provision,
            "retro_mode": coerced_mode.value,
            "lag_buffer_days": int(lag_buffer_days),
            "scrubber": scrubber,
            "telemetry": telemetry,
            "retrospective_overlay": retrospective_overlay,
            "backfill_audit": backfill_audit,
        }


replay_telemetry_bus = ReplayTelemetryBus()
