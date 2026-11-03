"""
Phase 1 & Phase 2: Data Grain Audit and Domain-Specific Feature Engineering Pipeline
for UK Further Education Safeguarding Demand Forecasting.

Implements:
1. Weekly & Curriculum-Area FE Safeguarding Panel Generator + Data Grain Dispersion/Zero Audit
2. Exposure Offset (log_enrolment = ln(N_t))
3. ILR Cohort & Vulnerability Baselines (Z_cohort)
4. Academic Calendar & Seasonal Triggers (X_calendar)
5. Strictly Causal Leading Behavioural Lags & EWMA Velocity (t-1 to t-4 weeks)
6. Automated VIF Multicollinearity Diagnostic & Expanding-Window Composite Index Builder
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


CURRICULUM_AREA_PROFILES: Dict[str, Dict[str, float]] = {
    "Foundation Learning & SEND (EPYP)": {
        "base_enrolment": 620.0,
        "base_referral_rate_per_1000": 8.4,
        "statutory_vulnerability_share": 0.34,
        "deprivation_imd_1_3_share": 0.42,
        "provision_16_18_ratio": 0.88,
        "zero_inflation_propensity": 0.08,
    },
    "Health, Social Care & Childcare": {
        "base_enrolment": 1180.0,
        "base_referral_rate_per_1000": 5.6,
        "statutory_vulnerability_share": 0.19,
        "deprivation_imd_1_3_share": 0.35,
        "provision_16_18_ratio": 0.72,
        "zero_inflation_propensity": 0.10,
    },
    "Construction & Building Trades": {
        "base_enrolment": 1350.0,
        "base_referral_rate_per_1000": 4.2,
        "statutory_vulnerability_share": 0.14,
        "deprivation_imd_1_3_share": 0.31,
        "provision_16_18_ratio": 0.68,
        "zero_inflation_propensity": 0.14,
    },
    "Digital, Computing & Creative": {
        "base_enrolment": 980.0,
        "base_referral_rate_per_1000": 4.5,
        "statutory_vulnerability_share": 0.16,
        "deprivation_imd_1_3_share": 0.27,
        "provision_16_18_ratio": 0.74,
        "zero_inflation_propensity": 0.15,
    },
    "Engineering, Marine & Automotive": {
        "base_enrolment": 1120.0,
        "base_referral_rate_per_1000": 3.1,
        "statutory_vulnerability_share": 0.11,
        "deprivation_imd_1_3_share": 0.24,
        "provision_16_18_ratio": 0.58,
        "zero_inflation_propensity": 0.22,
    },
    "Adult Access & Work-Based Apprenticeships": {
        "base_enrolment": 1950.0,
        "base_referral_rate_per_1000": 2.2,
        "statutory_vulnerability_share": 0.07,
        "deprivation_imd_1_3_share": 0.29,
        "provision_16_18_ratio": 0.18,
        "zero_inflation_propensity": 0.26,
    },
}


@dataclass
class DataGrainAuditRow:
    grain_name: str
    n_rows: int
    mean_count: float
    variance_count: float
    dispersion_var_to_mean: float
    zero_count_rows: int
    zero_count_pct: float
    max_count: int


@dataclass
class VIFDiagnosticResult:
    vif_table: pd.DataFrame
    att_pastoral_vif_exceeded: bool
    max_att_pastoral_vif: float
    composite_index_activated: bool
    selected_feature_cols: List[str]
    collinearity_note: str


def generate_weekly_safeguarding_panel(
    num_academic_years: int = 3,
    start_year: int = 2022,
    seed: int = 42,
    include_collinear_disengagement: bool = True,
) -> pd.DataFrame:
    """
    Generate a realistic weekly Further Education safeguarding panel dataset across
    curriculum areas and whole-college totals, incorporating:
    - Active learner headcount exposure (N_t)
    - ILR cohort vulnerability, IMD 1-3 deprivation, and 16-18 vs 19+ mix
    - FE academic calendar triggers (Autumn transition weeks 1-6, pre/post holiday weeks,
      Nov/May-June assessment crunch weeks, and low-activity reading/placement/closure weeks)
    - Leading behavioural drivers (attendance, English/Maths attendance, pastoral logs,
      16-19 bursary emergency hardship claims)
    - Overdispersed Negative Binomial weekly referral counts + structural zeros in low-activity weeks.
    """
    rng = np.random.default_rng(seed)
    total_weeks = num_academic_years * 52
    week_dates = pd.date_range(start=f"{start_year:04d}-09-05", periods=total_weeks, freq="W-MON")

    rows: List[Dict[str, Any]] = []

    # Define academic-year week indices (1..52) for UK FE calendar events:
    # Half-terms & holidays: Oct half-term (w8), Winter break (w16-17), Feb half-term (w24),
    # Easter break (w31-32), May half-term (w39), Summer placement/closure (w44-52)
    holiday_or_closure_weeks = {8, 16, 17, 24, 31, 32, 39, 45, 46, 47, 48, 49, 50, 51, 52}
    reading_or_placement_weeks = {13, 21, 28, 43, 44}
    low_activity_weeks = holiday_or_closure_weeks | reading_or_placement_weeks
    pre_holiday_weeks = {7, 15, 23, 30, 38, 44}
    post_holiday_weeks = {9, 18, 25, 33, 40}
    assessment_crunch_weeks = {10, 11, 12, 36, 37, 38, 40, 41}

    for area_name, prof in CURRICULUM_AREA_PROFILES.items():
        base_n = prof["base_enrolment"]
        base_rate = prof["base_referral_rate_per_1000"]
        vuln_base = prof["statutory_vulnerability_share"]
        imd_base = prof["deprivation_imd_1_3_share"]
        mix_base = prof["provision_16_18_ratio"]
        zero_propensity = prof["zero_inflation_propensity"]

        prev_att = 90.5
        prev_eng_maths_att = 86.0
        prev_referrals = max(1, int(round(base_n * base_rate / 1000.0)))
        latent_stress_history: List[float] = [0.0, 0.0, 0.0, 0.0]

        for w_idx, dt in enumerate(week_dates):
            ay_idx = w_idx // 52
            w_of_ay = (w_idx % 52) + 1
            ay_start = start_year + ay_idx
            ay_label = f"{ay_start}/{str(ay_start + 1)[-2:]}"

            is_autumn_transition = 1 if (1 <= w_of_ay <= 6) else 0
            is_pre_holiday = 1 if (w_of_ay in pre_holiday_weeks) else 0
            is_post_holiday = 1 if (w_of_ay in post_holiday_weeks) else 0
            is_assessment_crunch = 1 if (w_of_ay in assessment_crunch_weeks) else 0
            is_low_activity = 1 if (w_of_ay in low_activity_weeks) else 0

            # Termly ILR cohort updates (updated every ~17 weeks with realistic drift)
            term_idx = min(2, (w_of_ay - 1) // 17)
            rng_term = np.random.default_rng(seed + hash(area_name) % 10000 + ay_idx * 10 + term_idx)
            statutory_vuln = float(np.clip(vuln_base + rng_term.normal(0.0, 0.015) + 0.006 * ay_idx, 0.03, 0.55))
            deprivation_imd = float(np.clip(imd_base + rng_term.normal(0.0, 0.018) + 0.004 * ay_idx, 0.08, 0.65))
            provision_16_18 = float(np.clip(mix_base + rng_term.normal(0.0, 0.012), 0.10, 0.96))

            # Active learner headcount N_t (growth across years + mild intra-year retention curve)
            retention = 1.0 - 0.0012 * min(w_of_ay, 42)
            active_enrolments = int(
                round(base_n * ((1.022) ** ay_idx) * retention * float(rng.normal(1.0, 0.01)))
            )
            active_enrolments = max(120, active_enrolments)

            # Latent weekly student distress factor (drives attendance drops, pastoral logs, and bursary claims)
            seasonal_wave = 0.35 * np.sin(2.0 * np.pi * (w_of_ay - 4) / 52.0)
            distress_t = float(
                0.45 * seasonal_wave
                + 0.55 * is_autumn_transition
                + 0.40 * is_pre_holiday
                + 0.45 * is_post_holiday
                + 0.50 * is_assessment_crunch
                - 0.90 * is_low_activity
                + rng.normal(0.0, 0.55)
            )
            latent_stress_history.append(distress_t)

            # Weekly attendance & English/Maths attendance responds to current distress
            target_att = 90.8 - 1.85 * distress_t - (3.5 if is_low_activity else 0.0) + rng.normal(0.0, 0.45)
            overall_att = float(np.clip(0.35 * prev_att + 0.65 * target_att, 72.0, 98.0))
            eng_maths_att = float(
                np.clip(overall_att - 3.8 - 0.55 * distress_t + rng.normal(0.0, 0.5), 65.0, 96.5)
            )
            att_drop_current = max(-4.0, prev_att - overall_att)
            prev_att = overall_att
            prev_eng_maths_att = eng_maths_att

            # Low-level pastoral logs (Cause for Concern / fitness-to-study)
            # Strongly co-moves with attendance drops when include_collinear_disengagement=True
            if include_collinear_disengagement:
                pastoral_rate = max(
                    0.4,
                    8.5 + 3.2 * att_drop_current + 1.8 * distress_t + rng.normal(0.0, 0.55),
                )
            else:
                pastoral_rate = max(0.4, 8.5 + 2.2 * distress_t + rng.normal(0.0, 1.8))
            pastoral_logs = int(round((pastoral_rate / 1000.0) * active_enrolments))

            # Emergency 16-19 Bursary hardship claims (food/travel/equipment)
            bursary_rate = max(
                0.2,
                3.6
                + 4.5 * deprivation_imd
                + 1.6 * max(0.0, distress_t)
                + (1.4 if is_pre_holiday or is_autumn_transition else 0.0)
                + rng.normal(0.0, 0.65),
            )
            bursary_claims = int(round((bursary_rate / 1000.0) * active_enrolments))

            # True data-generating process for safeguarding referrals at week t
            # Depends strictly on PRIOR weeks' distress/disengagement (t-1) and bursary hardship (t-2..t-3),
            # calendar triggers, cohort vulnerability, and exposure offset N_t
            lag1_stress = latent_stress_history[-2]
            lag2_stress = latent_stress_history[-3]
            lag3_stress = latent_stress_history[-4]

            log_rate_per_learner = (
                np.log(base_rate / 1000.0)
                + 0.42 * is_autumn_transition
                + 0.26 * is_pre_holiday
                + 0.31 * is_post_holiday
                + 0.28 * is_assessment_crunch
                - 1.35 * is_low_activity
                + 0.65 * (statutory_vuln - 0.16)
                + 0.45 * (deprivation_imd - 0.30)
                + 0.30 * (provision_16_18 - 0.65)
                + 0.24 * lag1_stress
                + 0.14 * lag2_stress
                + 0.11 * lag3_stress
                + 0.08 * np.log1p(prev_referrals / max(100.0, active_enrolments) * 1000.0)
            )
            mu_t = float(active_enrolments * np.exp(log_rate_per_learner))

            # Structural zero-inflation regime (ZINB): heightened during low-activity/placement/holiday weeks
            logit_zero = (
                -3.1
                + 3.65 * is_low_activity
                + 0.65 * is_pre_holiday
                + 1.8 * (zero_propensity - 0.15)
                - 0.45 * np.log(active_enrolments / 1000.0)
            )
            prob_structural_zero = float(1.0 / (1.0 + np.exp(-logit_zero)))
            is_structural_zero = bool(rng.random() < prob_structural_zero)

            if is_structural_zero:
                referral_count = 0
            else:
                alpha_nb = 0.18
                n_nb = 1.0 / alpha_nb
                p_nb = n_nb / (n_nb + max(0.05, mu_t))
                referral_count = int(rng.negative_binomial(n_nb, p_nb))

            prev_referrals = referral_count

            rows.append(
                {
                    "week_start": dt.strftime("%Y-%m-%d"),
                    "academic_year": ay_label,
                    "week_of_academic_year": w_of_ay,
                    "curriculum_area": area_name,
                    "active_enrolments": active_enrolments,
                    "referral_count": referral_count,
                    "statutory_vulnerability_share": round(statutory_vuln, 4),
                    "deprivation_imd_1_3_share": round(deprivation_imd, 4),
                    "provision_16_18_ratio": round(provision_16_18, 4),
                    "is_autumn_transition": is_autumn_transition,
                    "is_pre_holiday_week": is_pre_holiday,
                    "is_post_holiday_week": is_post_holiday,
                    "is_assessment_crunch": is_assessment_crunch,
                    "is_low_activity_week": is_low_activity,
                    "overall_attendance_pct": round(overall_att, 2),
                    "eng_maths_attendance_pct": round(eng_maths_att, 2),
                    "pastoral_log_count": pastoral_logs,
                    "pastoral_rate_per_1000": round(pastoral_rate, 3),
                    "bursary_hardship_claims": bursary_claims,
                    "bursary_rate_per_1000": round(bursary_rate, 3),
                }
            )

    panel_df = pd.DataFrame(rows)
    return panel_df


def aggregate_college_wide_weekly(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate curriculum-area weekly panel rows into a Whole-College weekly series."""
    if "curriculum_area" in panel_df.columns and (panel_df["curriculum_area"] == "College-Wide").all():
        return panel_df.copy().reset_index(drop=True)

    work = panel_df[panel_df["curriculum_area"] != "College-Wide"].copy()
    work["weighted_vuln"] = work["statutory_vulnerability_share"] * work["active_enrolments"]
    work["weighted_imd"] = work["deprivation_imd_1_3_share"] * work["active_enrolments"]
    work["weighted_16_18"] = work["provision_16_18_ratio"] * work["active_enrolments"]
    work["weighted_att"] = work["overall_attendance_pct"] * work["active_enrolments"]
    work["weighted_em_att"] = work["eng_maths_attendance_pct"] * work["active_enrolments"]

    grouped = (
        work.groupby(["week_start", "academic_year", "week_of_academic_year"], as_index=False)
        .agg(
            active_enrolments=("active_enrolments", "sum"),
            referral_count=("referral_count", "sum"),
            weighted_vuln=("weighted_vuln", "sum"),
            weighted_imd=("weighted_imd", "sum"),
            weighted_16_18=("weighted_16_18", "sum"),
            weighted_att=("weighted_att", "sum"),
            weighted_em_att=("weighted_em_att", "sum"),
            is_autumn_transition=("is_autumn_transition", "max"),
            is_pre_holiday_week=("is_pre_holiday_week", "max"),
            is_post_holiday_week=("is_post_holiday_week", "max"),
            is_assessment_crunch=("is_assessment_crunch", "max"),
            is_low_activity_week=("is_low_activity_week", "max"),
            pastoral_log_count=("pastoral_log_count", "sum"),
            bursary_hardship_claims=("bursary_hardship_claims", "sum"),
        )
        .sort_values("week_start")
        .reset_index(drop=True)
    )

    enrol = np.maximum(1.0, grouped["active_enrolments"].to_numpy(dtype=float))
    grouped["curriculum_area"] = "College-Wide"
    grouped["statutory_vulnerability_share"] = np.round(grouped["weighted_vuln"] / enrol, 4)
    grouped["deprivation_imd_1_3_share"] = np.round(grouped["weighted_imd"] / enrol, 4)
    grouped["provision_16_18_ratio"] = np.round(grouped["weighted_16_18"] / enrol, 4)
    grouped["overall_attendance_pct"] = np.round(grouped["weighted_att"] / enrol, 2)
    grouped["eng_maths_attendance_pct"] = np.round(grouped["weighted_em_att"] / enrol, 2)
    grouped["pastoral_rate_per_1000"] = np.round((grouped["pastoral_log_count"] / enrol) * 1000.0, 3)
    grouped["bursary_rate_per_1000"] = np.round((grouped["bursary_hardship_claims"] / enrol) * 1000.0, 3)

    drop_cols = ["weighted_vuln", "weighted_imd", "weighted_16_18", "weighted_att", "weighted_em_att"]
    return grouped.drop(columns=drop_cols)


def audit_target_distribution(
    panel_df: pd.DataFrame,
    target_col: str = "referral_count",
    group_col: str = "curriculum_area",
) -> Dict[str, Any]:
    """
    Phase 1 Data Grain Audit:
    Inspect the target variable distribution across whole-college and curriculum-area grains,
    calculating mean, variance, dispersion ratio (Var / Mean), and zero-count percentage.
    """
    cw_df = aggregate_college_wide_weekly(panel_df)
    audit_rows: List[DataGrainAuditRow] = []

    # 1. College-Wide Weekly Aggregate
    y_cw = cw_df[target_col].to_numpy(dtype=float)
    mean_cw = float(np.mean(y_cw))
    var_cw = float(np.var(y_cw, ddof=1)) if len(y_cw) > 1 else 0.0
    zeros_cw = int(np.sum(y_cw == 0))
    audit_rows.append(
        DataGrainAuditRow(
            grain_name="College-Wide (Weekly Aggregate)",
            n_rows=len(y_cw),
            mean_count=round(mean_cw, 2),
            variance_count=round(var_cw, 2),
            dispersion_var_to_mean=round(var_cw / max(1e-6, mean_cw), 2),
            zero_count_rows=zeros_cw,
            zero_count_pct=round((zeros_cw / max(1, len(y_cw))) * 100.0, 2),
            max_count=int(np.max(y_cw)),
        )
    )

    # 2. All Curriculum Areas Pooled
    y_all = panel_df[target_col].to_numpy(dtype=float)
    mean_all = float(np.mean(y_all))
    var_all = float(np.var(y_all, ddof=1)) if len(y_all) > 1 else 0.0
    zeros_all = int(np.sum(y_all == 0))
    audit_rows.append(
        DataGrainAuditRow(
            grain_name="All Curriculum Areas (Pooled Panel)",
            n_rows=len(y_all),
            mean_count=round(mean_all, 2),
            variance_count=round(var_all, 2),
            dispersion_var_to_mean=round(var_all / max(1e-6, mean_all), 2),
            zero_count_rows=zeros_all,
            zero_count_pct=round((zeros_all / max(1, len(y_all))) * 100.0, 2),
            max_count=int(np.max(y_all)),
        )
    )

    # 3. Per Curriculum Area
    if group_col in panel_df.columns:
        for area, g_df in panel_df.groupby(group_col):
            if area == "College-Wide":
                continue
            y_g = g_df[target_col].to_numpy(dtype=float)
            m_g = float(np.mean(y_g))
            v_g = float(np.var(y_g, ddof=1)) if len(y_g) > 1 else 0.0
            z_g = int(np.sum(y_g == 0))
            audit_rows.append(
                DataGrainAuditRow(
                    grain_name=str(area),
                    n_rows=len(y_g),
                    mean_count=round(m_g, 2),
                    variance_count=round(v_g, 2),
                    dispersion_var_to_mean=round(v_g / max(1e-6, m_g), 2),
                    zero_count_rows=z_g,
                    zero_count_pct=round((z_g / max(1, len(y_g))) * 100.0, 2),
                    max_count=int(np.max(y_g)),
                )
            )

    return {
        "college_wide_dispersion": audit_rows[0].dispersion_var_to_mean,
        "college_wide_zero_pct": audit_rows[0].zero_count_pct,
        "pooled_panel_dispersion": audit_rows[1].dispersion_var_to_mean,
        "pooled_panel_zero_pct": audit_rows[1].zero_count_pct,
        "rows": [r.__dict__ for r in audit_rows],
    }


def expanding_window_zscore(series: pd.Series, min_periods: int = 3) -> pd.Series:
    """
    Strictly causal expanding-window standardization:
    At time index t, computes (x_t - mean(x_0..t)) / std(x_0..t) using ONLY
    observations up to and including t (when series itself is already a lagged predictor
    such as X_{t-1}, this uses only historical lags <= t-1 with zero look-ahead bias).
    """
    vals = series.to_numpy(dtype=float)
    out = np.zeros(len(vals), dtype=float)
    for t in range(len(vals)):
        window = vals[: t + 1]
        valid = window[np.isfinite(window)]
        if len(valid) < max(2, min_periods):
            out[t] = 0.0
        else:
            mu = float(np.mean(valid))
            sd = float(np.std(valid, ddof=1))
            out[t] = (vals[t] - mu) / sd if sd > 1e-8 else 0.0
    return pd.Series(out, index=series.index, dtype=float)


def _causal_ewma(series: pd.Series, span: float) -> pd.Series:
    """
    Compute a strictly causal Exponentially Weighted Moving Average on a series.
    Because adjust=False updates recursively as y_t = alpha * x_t + (1 - alpha) * y_{t-1},
    modifying x_{t+1} has zero effect on y_t.
    """
    return series.ewm(span=span, adjust=False, min_periods=1).mean()


def engineer_safeguarding_features(
    df: pd.DataFrame,
    group_col: Optional[str] = "curriculum_area",
    vif_threshold: float = 5.0,
) -> Tuple[pd.DataFrame, VIFDiagnosticResult]:
    """
    Phase 2: Build the 4 structured feature groups with strict temporal boundaries
    (zero look-ahead bias) plus automated VIF multicollinearity check & composite index builder:
      1. Exposure Offset: log_enrolment = ln(N_t)
      2. ILR Cohort & Vulnerability Baselines (Z_cohort)
      3. Academic Calendar & Seasonal Triggers (X_calendar)
      4. Leading Behavioural Lags & EWMA Velocity (t-1 to t-4 weeks)
      5. Multicollinearity Diagnostic (VIF > 5.0 -> early_disengagement_index)
    """
    work = df.copy()
    if "week_start" in work.columns:
        work["_sort_dt"] = pd.to_datetime(work["week_start"])
    elif "period_start" in work.columns:
        work["_sort_dt"] = pd.to_datetime(work["period_start"])
    else:
        work["_sort_dt"] = np.arange(len(work))

    if group_col and group_col in work.columns:
        work = work.sort_values([group_col, "_sort_dt"]).reset_index(drop=True)
        groups = [g for _, g in work.groupby(group_col, sort=False)]
    else:
        work = work.sort_values("_sort_dt").reset_index(drop=True)
        groups = [work]

    engineered_parts: List[pd.DataFrame] = []

    for g_df in groups:
        g = g_df.copy().reset_index(drop=True)

        # 1. Exposure Offset: ln(N_t)
        enrol_col = "active_enrolments" if "active_enrolments" in g.columns else "enrolled_students"
        g["active_enrolments"] = np.maximum(1.0, pd.to_numeric(g[enrol_col], errors="coerce").fillna(1000.0))
        g["log_enrolment"] = np.log(g["active_enrolments"].to_numpy(dtype=float))

        # Target & Referral rate per 1,000 learners (for scale-free autoregressive lags)
        target_col = "referral_count" if "referral_count" in g.columns else "concern_count"
        g["referral_count"] = pd.to_numeric(g[target_col], errors="coerce").fillna(0.0)
        referral_rate_per_1000 = (g["referral_count"] / g["active_enrolments"]) * 1000.0

        # 2. ILR Cohort & Vulnerability Baselines (Z_cohort)
        for col, default_val in [
            ("statutory_vulnerability_share", 0.18),
            ("deprivation_imd_1_3_share", 0.30),
            ("provision_16_18_ratio", 0.65),
        ]:
            if col not in g.columns:
                g[col] = default_val
            else:
                g[col] = pd.to_numeric(g[col], errors="coerce").ffill().fillna(default_val)

        # 3. Academic Calendar & Seasonal Triggers (X_calendar)
        for cal_col in [
            "is_autumn_transition",
            "is_pre_holiday_week",
            "is_post_holiday_week",
            "is_assessment_crunch",
            "is_low_activity_week",
        ]:
            if cal_col not in g.columns:
                g[cal_col] = 0
            else:
                g[cal_col] = pd.to_numeric(g[cal_col], errors="coerce").fillna(0).astype(int)

        # 4. Leading Behavioural Lags & EWMA Velocity (t-1 to t-4 weeks)
        # 4a. Attendance Velocity (Delta Att_{t-1}, Delta Att_{t-2}, and 1w, 2w, 4w EWMAs)
        att_series = pd.to_numeric(
            g.get("overall_attendance_pct", pd.Series(90.0, index=g.index)), errors="coerce"
        ).ffill().fillna(90.0)
        em_att_series = pd.to_numeric(
            g.get("eng_maths_attendance_pct", att_series - 3.5), errors="coerce"
        ).ffill().fillna(86.5)

        # Week-on-week percentage change at step k: Att_k - Att_{k-1}
        delta_att = att_series.diff(1).fillna(0.0)
        delta_em_att = em_att_series.diff(1).fillna(0.0)

        # Strictly lagged attendance velocity (shifted by >= 1 week so time t never sees time t)
        g["delta_att_t1"] = delta_att.shift(1).fillna(0.0)
        g["delta_att_t2"] = delta_att.shift(2).fillna(0.0)
        g["eng_maths_delta_att_t1"] = delta_em_att.shift(1).fillna(0.0)
        # Positive disengagement direction: attendance drop at t-1 (-Delta Att_{t-1})
        g["att_drop_t1"] = -g["delta_att_t1"]

        g["att_velocity_ewma_1w"] = _causal_ewma(g["delta_att_t1"], span=1.0)
        g["att_velocity_ewma_2w"] = _causal_ewma(g["delta_att_t1"], span=2.0)
        g["att_velocity_ewma_4w"] = _causal_ewma(g["delta_att_t1"], span=4.0)

        # 4b. Low-Level Pastoral Logs (Pastoral_{t-1}, Pastoral_{t-2}, and 2w/4w EWMAs)
        if "pastoral_rate_per_1000" in g.columns:
            pastoral_rate = pd.to_numeric(g["pastoral_rate_per_1000"], errors="coerce").ffill().fillna(8.0)
        elif "pastoral_log_count" in g.columns:
            pastoral_rate = (pd.to_numeric(g["pastoral_log_count"], errors="coerce").fillna(0.0) / g["active_enrolments"]) * 1000.0
        else:
            pastoral_rate = pd.Series(8.0, index=g.index)

        g["pastoral_t1"] = pastoral_rate.shift(1).bfill().fillna(8.0)
        g["pastoral_t2"] = pastoral_rate.shift(2).bfill().fillna(8.0)
        g["pastoral_ewma_2w"] = _causal_ewma(g["pastoral_t1"], span=2.0)
        g["pastoral_ewma_4w"] = _causal_ewma(g["pastoral_t1"], span=4.0)

        # 4c. Emergency Hardship Claims (Bursary_{t-2} to Bursary_{t-4} and 4w EWMA)
        if "bursary_rate_per_1000" in g.columns:
            bursary_rate = pd.to_numeric(g["bursary_rate_per_1000"], errors="coerce").ffill().fillna(4.0)
        elif "bursary_hardship_claims" in g.columns:
            bursary_rate = (pd.to_numeric(g["bursary_hardship_claims"], errors="coerce").fillna(0.0) / g["active_enrolments"]) * 1000.0
        else:
            bursary_rate = pd.Series(4.0, index=g.index)

        g["bursary_t2"] = bursary_rate.shift(2).bfill().fillna(4.0)
        g["bursary_t3"] = bursary_rate.shift(3).bfill().fillna(4.0)
        g["bursary_t4"] = bursary_rate.shift(4).bfill().fillna(4.0)
        g["bursary_ewma_4w"] = _causal_ewma(g["bursary_t2"], span=4.0)

        # 4d. Autoregressive Referral Lag (Y_{t-1} and 3-week rolling mean up to t-1)
        # Store both raw lagged count and rate-per-1000 lag (rate lag preserves strict exposure proportionality)
        g["referral_lag_t1_count"] = g["referral_count"].shift(1).bfill().fillna(0.0)
        g["referral_rolling_3w_count"] = (
            g["referral_count"].shift(1).rolling(window=3, min_periods=1).mean().bfill().fillna(0.0)
        )
        g["referral_lag_t1"] = referral_rate_per_1000.shift(1).bfill().fillna(float(referral_rate_per_1000.iloc[0]))
        g["referral_rolling_3w"] = (
            referral_rate_per_1000.shift(1)
            .rolling(window=3, min_periods=1)
            .mean()
            .bfill()
            .fillna(float(referral_rate_per_1000.iloc[0]))
        )

        # 5. Expanding-window standardized disengagement components (strictly causal)
        g["z_att_drop_t1"] = expanding_window_zscore(g["att_drop_t1"], min_periods=4)
        g["z_pastoral_t1"] = expanding_window_zscore(g["pastoral_t1"], min_periods=4)
        g["early_disengagement_index"] = 0.5 * (g["z_att_drop_t1"] + g["z_pastoral_t1"])

        engineered_parts.append(g)

    full_df = pd.concat(engineered_parts, axis=0, ignore_index=True)
    full_df = full_df.drop(columns=["_sort_dt"], errors="ignore")

    vif_result = run_vif_Diagnostic_and_select_features(full_df, vif_threshold=vif_threshold)
    return full_df, vif_result


def run_vif_Diagnostic_and_select_features(
    df: pd.DataFrame,
    vif_threshold: float = 5.0,
) -> VIFDiagnosticResult:
    """
    Phase 2 Step 5:
    Run an automated Variance Inflation Factor (VIF) check across candidate linear predictors.
    If VIF > 5.0 between lagged attendance drops (att_drop_t1 / delta_att_t1) and lagged
    pastoral logs (pastoral_t1), replace them with the expanding-window standardized
    composite `early_disengagement_index`.
    """
    candidate_cols = [
        "referral_lag_t1",
        "referral_rolling_3w",
        "att_drop_t1",
        "pastoral_t1",
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
    available_cols = [c for c in candidate_cols if c in df.columns and df[c].std() > 1e-8]

    X_mat = df[available_cols].copy().fillna(0.0)
    X_with_const = np.column_stack([np.ones(len(X_mat)), X_mat.to_numpy(dtype=float)])

    vif_rows: List[Dict[str, Any]] = []
    vif_map: Dict[str, float] = {}
    for idx, col_name in enumerate(available_cols, start=1):
        try:
            v_val = float(variance_inflation_factor(X_with_const, idx))
            if not np.isfinite(v_val):
                v_val = 99.9
        except Exception:
            v_val = 99.9
        vif_map[col_name] = round(v_val, 3)
        vif_rows.append({"feature": col_name, "vif": round(v_val, 3)})

    vif_df = pd.DataFrame(vif_rows)
    max_pair_vif = max(vif_map.get("att_drop_t1", 1.0), vif_map.get("pastoral_t1", 1.0))
    exceeded = bool(max_pair_vif > vif_threshold)

    if exceeded:
        selected_cols = [
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
        note = (
            f"VIF between lagged attendance drop (att_drop_t1 VIF={vif_map.get('att_drop_t1', 0):.2f}) "
            f"and lagged pastoral logs (pastoral_t1 VIF={vif_map.get('pastoral_t1', 0):.2f}) exceeded {vif_threshold:.1f}. "
            "Standardized both series using strictly causal expanding-window statistics and combined into "
            "`early_disengagement_index`."
        )
    else:
        selected_cols = available_cols
        note = (
            f"All linear predictors satisfied VIF <= {vif_threshold:.1f} "
            f"(max att/pastoral VIF={max_pair_vif:.2f}); composite index available alongside individual lags."
        )

    return VIFDiagnosticResult(
        vif_table=vif_df,
        att_pastoral_vif_exceeded=exceeded,
        max_att_pastoral_vif=round(max_pair_vif, 3),
        composite_index_activated=exceeded,
        selected_feature_cols=[c for c in selected_cols if c in df.columns],
        collinearity_note=note,
    )
