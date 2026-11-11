"""
Adaptive Three-Tier Schema Bridge (`esoteric_math/schema_bridge.py`).

Dynamically binds existing Safeguarding Forecast AI DataFrames (weekly curriculum-area panels,
college-wide aggregates, monthly operational datasets, or daily UK FE instructional streams)
into strictly causal Tier A, Tier B, and Tier C mathematical bundles without hard-coding
synthetic column names or leaking future observations (s > t).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple
import numpy as np
import pandas as pd


DATE_CANDIDATES: Tuple[str, ...] = (
    "date",
    "instructional_date",
    "week_start",
    "period_start",
    "ds",
    "timestamp",
)

COUNT_CANDIDATES: Tuple[str, ...] = (
    "referral_count",
    "concern_count",
    "y",
    "count",
    "incidents",
    "referrals",
)

EXPOSURE_CANDIDATES: Tuple[str, ...] = (
    "active_enrolments",
    "enrolled_students",
    "enrolment",
    "headcount",
    "learners",
)

COHORT_GROUP_CANDIDATES: Tuple[str, ...] = (
    "curriculum_area",
    "provision",
    "cohort",
    "department",
    "campus",
    "lsoa_cluster",
)

CATEGORY_GROUP_CANDIDATES: Tuple[str, ...] = (
    "safeguarding_category",
    "category",
    "concern_category",
)

COHORT_DISPLAY_SHORT: Dict[str, str] = {
    "Foundation Learning & SEND (EPYP)": "EPYP & SEND",
    "Health, Social Care & Childcare": "Health & Social Care",
    "Construction & Building Trades": "Construction Trades",
    "Digital, Computing & Creative": "Digital & Creative",
    "Engineering, Marine & Automotive": "Engineering & Marine",
    "Adult Access & Work-Based Apprenticeships": "Adult & Apprentices",
    "epyp_send": "EPYP & SEND",
    "epyp_and_send": "EPYP & SEND",
    "health_social_care": "Health & Social Care",
    "health_and_social_care": "Health & Social Care",
    "construction_trades": "Construction Trades",
    "digital_creative": "Digital & Creative",
    "digital_and_creative": "Digital & Creative",
    "engineering_marine": "Engineering & Marine",
    "engineering_and_marine": "Engineering & Marine",
    "adult_apprentices": "Adult & Apprentices",
    "adult_and_apprentices": "Adult & Apprentices",
}

CATEGORY_DISPLAY_SHORT: Dict[str, str] = {
    "mental_health": "Mental Health",
    "abuse_neglect": "Abuse & Neglect",
    "online_safety": "Online Safety",
    "exploitation": "Exploitation",
    "other_concerns": "Other Concerns",
}


@dataclass
class TierABundle:
    """Tier A — Temporal & Count Dynamics (Always Active)."""

    dates: np.ndarray                        # (T,) ISO date strings
    cadence: Literal["daily", "weekly", "monthly"]
    calendar_days_elapsed: np.ndarray        # (T,) float days since start
    calendar_cycles: np.ndarray              # (T,) elapsed 365.25-day calendar years
    is_instructional: np.ndarray             # (T,) float in [0, 1] (0 during holiday/weekend freeze)
    instructional_days_step: np.ndarray      # (T,) instructional days contributed by step t
    instructional_days_cum: np.ndarray       # (T,) cumulative instructional days t_acad (freezes on holidays)
    warped_acad_cycles: np.ndarray           # (T,) t_acad / 190.0 active academic-year cycles
    holiday_labels: List[str]                # len T ("" when active teaching period)
    y_count: np.ndarray                      # (T,) causal aggregate count signal
    y_raw_observed: np.ndarray               # (T,) unshifted observed count at t (for telemetry <= t)
    exposure: np.ndarray                     # (T,) active enrolled learners N_t
    y_rate_per_1000: np.ndarray              # (T,) causal referrals per 1,000 learners
    calendar_flags: Dict[str, np.ndarray]    # deterministic UK FE calendar flags at t


@dataclass
class TierBBundle:
    """Tier B — Multivariate Covariates & Sub-Categories."""

    covariate_matrix: np.ndarray             # (T, D_B) standardized causal covariate channels
    raw_covariate_matrix: np.ndarray         # (T, D_B) unscaled causal covariate channels
    covariate_names: List[str]               # len D_B
    category_counts: np.ndarray              # (T, C) causal category referral counts
    raw_category_counts: np.ndarray          # (T, C) unshifted category counts (for telemetry <= t)
    category_names: List[str]                # len C
    synthesized_univariate: bool             # True if covariates were causally synthesized from univariate counts


@dataclass
class TierCBundle:
    """Tier C — Cohort, Simulated Student, or Spatial/LSOA Hierarchy (or Latent Regime Graph)."""

    topology_mode: Literal["Cohort Hierarchy Graph", "Latent Feature/Regime Graph"]
    node_ids: List[str]                      # len V
    node_labels: List[str]                   # len V human-readable labels
    node_weights: np.ndarray                 # (V,) positive cohort size / importance weights summing to 1
    adjacency: np.ndarray                    # (V, V) symmetric non-negative weighted adjacency (learned on train)
    stalks: np.ndarray                       # (T, V, d_stalk) standardized vertex stalks x_v(t)
    stalk_dim_names: List[str]               # len d_stalk
    node_scalar_stress: np.ndarray           # (T, V) scalar vertex pressure signal


@dataclass
class SchemaBundle:
    """Unified Three-Tier Causal Mathematical Bundle."""

    tier_a: TierABundle
    tier_b: TierBBundle
    tier_c: TierCBundle
    n_steps: int
    causal_mode: Literal["telemetry", "forecast"]
    row_to_step_idx: Optional[np.ndarray] = None


def _causal_ffill_1d(arr: np.ndarray, default_val: float = 0.0) -> np.ndarray:
    """Strictly causal forward-fill (never looks ahead to s > t)."""
    out = np.empty(len(arr), dtype=float)
    last = float(default_val)
    for i in range(len(arr)):
        v = float(arr[i])
        if np.isfinite(v):
            last = v
        out[i] = last
    return out


def _causal_ewma_1d(arr: np.ndarray, alpha: float) -> np.ndarray:
    """Strictly causal recursive EWMA: y_t = alpha * x_t + (1 - alpha) * y_{t-1}."""
    out = np.empty(len(arr), dtype=float)
    if len(arr) == 0:
        return out
    out[0] = float(arr[0])
    for i in range(1, len(arr)):
        out[i] = alpha * float(arr[i]) + (1.0 - alpha) * out[i - 1]
    return out


def _causal_shift_1d(arr: np.ndarray, lag: int, init_val: float) -> np.ndarray:
    """Shift series backward by `lag` steps using a deterministic training/initial constant for indices < lag."""
    if lag <= 0:
        return arr.copy()
    out = np.empty(len(arr), dtype=float)
    n_fill = min(lag, len(arr))
    out[:n_fill] = float(init_val)
    if len(arr) > lag:
        out[lag:] = arr[:-lag]
    return out


def _infer_uk_fe_holiday_for_date(dt: pd.Timestamp, cadence: str) -> Tuple[float, str]:
    """
    Deterministic UK Further Education academic calendar rule when explicit teaching-day
    or holiday indicators are absent. Depends strictly on calendar date `dt` alone.
    Returns (instructional_fraction_in_0_1, holiday_label).
    """
    m, d, dow = int(dt.month), int(dt.day), int(dt.dayofweek)
    if cadence == "daily":
        if dow >= 5:
            return 0.0, "Weekend"
        if (m == 10 and d >= 25) or (m == 11 and d <= 1):
            return 0.0, "Autumn Half-Term"
        if (m == 12 and d >= 20) or (m == 1 and d <= 3):
            return 0.0, "Winter Break"
        if m == 2 and 14 <= d <= 21:
            return 0.0, "Spring Half-Term"
        if m == 4 and 1 <= d <= 14:
            return 0.0, "Easter Break"
        if m == 5 and d >= 25:
            return 0.0, "May Half-Term"
        if (m == 7 and d >= 18) or m == 8:
            return 0.0, "Summer Vacation"
        return 1.0, ""

    if cadence == "weekly":
        if (m == 10 and d >= 24):
            return 0.0, "Autumn Half-Term"
        if (m == 12 and d >= 19) or (m == 1 and d <= 2):
            return 0.0, "Winter Break"
        if m == 2 and 13 <= d <= 20:
            return 0.0, "Spring Half-Term"
        if m == 4 and 1 <= d <= 14:
            return 0.0, "Easter Break"
        if m == 5 and d >= 25:
            return 0.0, "May Half-Term"
        if (m == 7 and d >= 15) or m == 8:
            return 0.0, "Summer Vacation"
        return 1.0, ""

    # Monthly cadence: fraction of active FE teaching month (out of 20 standard days)
    monthly_td = {9: 19, 10: 18, 11: 21, 12: 14, 1: 19, 2: 15, 3: 20, 4: 13, 5: 17, 6: 18, 7: 6, 8: 2}
    td = monthly_td.get(m, 17)
    if m in (7, 8):
        return float(td) / 20.0, "Summer Period"
    return min(1.0, float(td) / 20.0), ""


def _ensure_connected_symmetric_graph(corr_mat: np.ndarray, top_k: int = 2) -> np.ndarray:
    """
    Build a deterministic, connected, non-negative weighted adjacency matrix strictly from
    training-fold similarity matrix `corr_mat` (shape V x V).
    """
    v_cnt = corr_mat.shape[0]
    if v_cnt <= 1:
        return np.zeros((v_cnt, v_cnt), dtype=float)

    sim = np.clip(0.5 * (corr_mat + corr_mat.T), -0.99, 0.99)
    np.fill_diagonal(sim, 0.0)
    # Positive affinity weights in [0.10, 1.0]
    aff = 0.15 + 0.85 * np.maximum(0.0, (sim + 0.35) / 1.35)
    np.fill_diagonal(aff, 0.0)

    adj = np.zeros_like(aff)
    k_eff = min(max(1, top_k), v_cnt - 1)
    for i in range(v_cnt):
        order = np.argsort(aff[i])[::-1]
        for j in order[:k_eff]:
            if i != j:
                w = float(aff[i, j])
                adj[i, j] = max(adj[i, j], w)
                adj[j, i] = max(adj[j, i], w)

    # Guarantee graph connectivity via greedy maximum-weight spanning tree
    visited = {0}
    while len(visited) < v_cnt:
        best_u, best_v, best_w = 0, 0, -1.0
        for u in visited:
            for v in range(v_cnt):
                if v not in visited and aff[u, v] > best_w:
                    best_u, best_v, best_w = u, v, float(aff[u, v])
        w_link = max(0.18, best_w)
        adj[best_u, best_v] = max(adj[best_u, best_v], w_link)
        adj[best_v, best_u] = max(adj[best_v, best_u], w_link)
        visited.add(best_v)

    return adj


class AdaptiveSchemaBridge:
    """
    Fits training-fold scaling & graph topology parameters without future leakage and
    transforms arbitrary safeguarding DataFrames into causal Tier A / B / C bundles.
    """

    def __init__(
        self,
        causal_mode: Literal["telemetry", "forecast"] = "telemetry",
        train_cutoff: Optional[int] = None,
    ) -> None:
        self.causal_mode = causal_mode
        self.train_cutoff = train_cutoff
        self.is_fitted_: bool = False

        # Learned strictly on training fold
        self.init_count_: float = 10.0
        self.init_rate_: float = 4.0
        self.rate_median_: float = 4.0
        self.rate_mad_: float = 1.5
        self.cov_names_: List[str] = []
        self.cov_medians_: np.ndarray = np.array([])
        self.cov_iqrs_: np.ndarray = np.array([])
        self.cat_names_: List[str] = []
        self.cat_inits_: np.ndarray = np.array([])

        self.topology_mode_: Literal["Cohort Hierarchy Graph", "Latent Feature/Regime Graph"] = (
            "Latent Feature/Regime Graph"
        )
        self.node_ids_: List[str] = []
        self.node_labels_: List[str] = []
        self.node_weights_: np.ndarray = np.array([])
        self.adjacency_: np.ndarray = np.array([])
        self.stalk_dim_names_: List[str] = []
        self.stalk_medians_: np.ndarray = np.array([])  # (V, d)
        self.stalk_iqrs_: np.ndarray = np.array([])     # (V, d)
        self.stalk_inits_: np.ndarray = np.array([])    # (V, d)

    @staticmethod
    def _find_col(cols: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
        lower_map = {c.lower(): c for c in cols}
        for cand in candidates:
            if cand.lower() in lower_map:
                return lower_map[cand.lower()]
        return None

    def _canonicalize_timeline(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[np.ndarray], str]:
        """
        Detect whether `df` is already 1-row-per-timestamp or a multi-row cohort/category panel.
        Preserves exact input row order when timestamps are unique.
        """
        cols = list(df.columns)
        date_col = self._find_col(cols, DATE_CANDIDATES)
        cohort_col = self._find_col(cols, COHORT_GROUP_CANDIDATES)
        cat_col = self._find_col(cols, CATEGORY_GROUP_CANDIDATES)

        if date_col is None:
            work = df.copy().reset_index(drop=True)
            synth_dates = pd.date_range("2023-09-04", periods=len(work), freq="W-MON")
            work["_bridge_dt"] = synth_dates
            return work, None, None, "_bridge_dt"

        # Check if multiple rows share the same date (panel format)
        raw_dates = df[date_col].astype(str).to_numpy()
        unique_dates = pd.unique(raw_dates)
        has_multi_cohort = (
            cohort_col is not None
            and df[cohort_col].nunique(dropna=True) > 1
            and len(unique_dates) < len(df)
        )
        has_multi_cat = (
            cat_col is not None
            and df[cat_col].nunique(dropna=True) > 1
            and len(unique_dates) < len(df)
        )

        if not has_multi_cohort and not has_multi_cat:
            work = df.copy().reset_index(drop=True)
            work["_bridge_dt"] = pd.to_datetime(work[date_col], errors="coerce")
            if work["_bridge_dt"].isna().any():
                work["_bridge_dt"] = pd.date_range("2023-09-04", periods=len(work), freq="D")
            return work, None, None, date_col

        # Aggregate long panel into unique chronological steps while preserving panel for Tier B/C
        panel = df.copy().reset_index(drop=True)
        panel["_bridge_dt"] = pd.to_datetime(panel[date_col], errors="coerce")
        sorted_unique_dts = np.sort(panel["_bridge_dt"].unique())
        dt_to_idx = {dt_v: idx for idx, dt_v in enumerate(sorted_unique_dts)}
        row_to_step = np.array([dt_to_idx[d] for d in panel["_bridge_dt"]], dtype=int)

        count_col = self._find_col(cols, COUNT_CANDIDATES) or cols[0]
        exp_col = self._find_col(cols, EXPOSURE_CANDIDATES)

        agg_rows: List[Dict[str, Any]] = []
        for dt_v in sorted_unique_dts:
            sub = panel[panel["_bridge_dt"] == dt_v]
            row_dict: Dict[str, Any] = {
                date_col: str(sub[date_col].iloc[0]),
                "_bridge_dt": dt_v,
                count_col: float(pd.to_numeric(sub[count_col], errors="coerce").fillna(0.0).sum()),
            }
            if exp_col:
                if cohort_col and cohort_col in sub.columns:
                    # Sum unique cohort enrolments once per timestamp
                    enrol_sum = float(
                        sub.groupby(cohort_col)[exp_col]
                        .first()
                        .pipe(pd.to_numeric, errors="coerce")
                        .fillna(1000.0)
                        .sum()
                    )
                else:
                    enrol_sum = float(pd.to_numeric(sub[exp_col], errors="coerce").fillna(1000.0).mean())
                row_dict[exp_col] = max(1.0, enrol_sum)

            # Carry numeric covariates by mean/max
            for num_col in (
                "teaching_days",
                "attendance_rate",
                "overall_attendance_pct",
                "eng_maths_attendance_pct",
                "pastoral_rate_per_1000",
                "bursary_rate_per_1000",
                "statutory_vulnerability_share",
                "deprivation_imd_1_3_share",
                "provision_16_18_ratio",
                "is_autumn_transition",
                "is_pre_holiday_week",
                "is_post_holiday_week",
                "is_assessment_crunch",
                "is_low_activity_week",
                "reporting_change_flag",
            ):
                if num_col in sub.columns:
                    vals = pd.to_numeric(sub[num_col], errors="coerce").dropna()
                    row_dict[num_col] = float(vals.mean()) if len(vals) > 0 else 0.0

            agg_rows.append(row_dict)

        flat_df = pd.DataFrame(agg_rows)
        return flat_df, panel, row_to_step, date_col

    def _extract_raw_tier_a(self, flat_df: pd.DataFrame, date_col: str) -> TierABundle:
        cols = list(flat_df.columns)
        n = len(flat_df)
        dts = pd.to_datetime(flat_df["_bridge_dt"])
        date_strs = dts.dt.strftime("%Y-%m-%d").to_numpy(dtype=object)

        # Detect cadence from initial spacing (deterministic from dates)
        if n >= 2:
            diffs = np.diff(dts.values).astype("timedelta64[D]").astype(float)
            pos_diffs = diffs[diffs > 0]
            med_diff = float(np.median(pos_diffs[: min(12, len(pos_diffs))])) if len(pos_diffs) > 0 else 7.0
        else:
            med_diff = 7.0

        if med_diff <= 3.5:
            cadence: Literal["daily", "weekly", "monthly"] = "daily"
            default_step_days = 1.0
        elif med_diff <= 14.0:
            cadence = "weekly"
            default_step_days = 5.0
        else:
            cadence = "monthly"
            default_step_days = 18.0

        cal_days = (dts - dts.iloc[0]).dt.total_seconds().to_numpy(dtype=float) / 86400.0
        cal_cycles = cal_days / 365.25

        count_col = self._find_col(cols, COUNT_CANDIDATES)
        if count_col is not None:
            raw_counts = np.maximum(
                0.0,
                _causal_ffill_1d(
                    pd.to_numeric(flat_df[count_col], errors="coerce").to_numpy(dtype=float),
                    default_val=0.0,
                ),
            )
        else:
            num_cols = flat_df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                raw_counts = np.maximum(0.0, _causal_ffill_1d(flat_df[num_cols[0]].to_numpy(dtype=float), 0.0))
            else:
                raw_counts = np.zeros(n, dtype=float)

        exp_col = self._find_col(cols, EXPOSURE_CANDIDATES)
        if exp_col is not None:
            exposure = np.maximum(
                1.0,
                _causal_ffill_1d(
                    pd.to_numeric(flat_df[exp_col], errors="coerce").to_numpy(dtype=float),
                    default_val=1000.0,
                ),
            )
        else:
            exposure = np.full(n, 1000.0, dtype=float)

        # Instructional activity & holiday labels (strictly causal)
        is_inst = np.ones(n, dtype=float)
        inst_step = np.full(n, default_step_days, dtype=float)
        hol_labels: List[str] = [""] * n
        has_explicit_cal = any(
            c in flat_df.columns
            for c in ("is_instructional_day", "is_low_activity_week", "teaching_days", "holiday_label")
        )

        for i in range(n):
            dt_i = pd.Timestamp(dts.iloc[i])
            rule_frac, rule_lbl = _infer_uk_fe_holiday_for_date(dt_i, cadence)
            lbl_i = rule_lbl
            if "holiday_label" in flat_df.columns:
                raw_lbl = flat_df["holiday_label"].iloc[i]
                if pd.notna(raw_lbl) and str(raw_lbl).strip():
                    lbl_i = str(raw_lbl).strip()

            if "is_instructional_day" in flat_df.columns:
                flag_v = float(pd.to_numeric(flat_df["is_instructional_day"].iloc[i], errors="coerce") or 0.0)
                frac_i = float(np.clip(flag_v, 0.0, 1.0))
                step_i = frac_i * default_step_days
                if frac_i <= 0.05 and not lbl_i:
                    lbl_i = "Non-Instructional Closure"
                elif frac_i > 0.05:
                    lbl_i = ""
            elif "is_low_activity_week" in flat_df.columns:
                low_v = int(pd.to_numeric(flat_df["is_low_activity_week"].iloc[i], errors="coerce") or 0)
                frac_i = 0.0 if low_v > 0 else 1.0
                step_i = frac_i * default_step_days
                if low_v > 0 and not lbl_i:
                    lbl_i = "Holiday / Placement Freeze"
                elif low_v == 0:
                    lbl_i = ""
            elif "teaching_days" in flat_df.columns:
                td_v = float(pd.to_numeric(flat_df["teaching_days"].iloc[i], errors="coerce") or 0.0)
                step_i = max(0.0, td_v)
                frac_i = float(np.clip(td_v / 20.0, 0.0, 1.0)) if cadence == "monthly" else float(td_v > 0)
                if td_v <= 6 and cadence == "monthly" and not lbl_i:
                    lbl_i = "Holiday Reduced Period"
            else:
                frac_i = rule_frac
                step_i = frac_i * default_step_days
                # Causal zero-referral holiday block inference if active day has consecutive zeros
                if not has_explicit_cal and i >= 1 and raw_counts[i] == 0.0 and raw_counts[i - 1] == 0.0:
                    frac_i = 0.0
                    step_i = 0.0
                    lbl_i = lbl_i or "Inferred Zero-Referral Break"

            is_inst[i] = frac_i
            inst_step[i] = step_i
            hol_labels[i] = lbl_i

        inst_cum = np.cumsum(inst_step)
        warped_cycles = inst_cum / 190.0

        # Apply forecast causal lag if causal_mode == "forecast"
        init_c = self.init_count_ if self.is_fitted_ else (float(raw_counts[0]) if n > 0 else 10.0)
        if self.causal_mode == "forecast":
            y_causal = _causal_shift_1d(raw_counts, lag=1, init_val=init_c)
            exp_prev = _causal_shift_1d(exposure, lag=1, init_val=float(exposure[0]) if n > 0 else 1000.0)
            if "referral_lag_t1" in flat_df.columns:
                y_rate = _causal_ffill_1d(
                    pd.to_numeric(flat_df["referral_lag_t1"], errors="coerce").to_numpy(dtype=float),
                    default_val=self.init_rate_,
                )
                y_causal = (y_rate / 1000.0) * exposure
            else:
                y_rate = (y_causal / np.maximum(1.0, exp_prev)) * 1000.0
        else:
            y_causal = raw_counts.copy()
            y_rate = (y_causal / np.maximum(1.0, exposure)) * 1000.0

        # Deterministic calendar flags
        months = dts.dt.month.to_numpy(dtype=int)
        is_hol = (is_inst <= 0.15).astype(float)
        is_post_hol = np.zeros(n, dtype=float)
        for i in range(1, n):
            if is_hol[i] < 0.5 and is_hol[i - 1] >= 0.5:
                is_post_hol[i] = 1.0
        if "is_post_holiday_week" in flat_df.columns:
            is_post_hol = np.maximum(
                is_post_hol,
                pd.to_numeric(flat_df["is_post_holiday_week"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
            )

        is_autumn = (
            pd.to_numeric(flat_df["is_autumn_transition"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            if "is_autumn_transition" in flat_df.columns
            else ((months == 9) | ((months == 10) & (dts.dt.day.to_numpy(dtype=int) <= 20))).astype(float)
        )
        is_crunch = (
            pd.to_numeric(flat_df["is_assessment_crunch"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            if "is_assessment_crunch" in flat_df.columns
            else ((months == 11) | (months == 5) | (months == 6)).astype(float)
        )

        cal_flags = {
            "is_holiday": is_hol,
            "is_post_holiday": is_post_hol,
            "is_autumn_induction": is_autumn,
            "is_assessment_crunch": is_crunch,
        }

        return TierABundle(
            dates=date_strs,
            cadence=cadence,
            calendar_days_elapsed=cal_days,
            calendar_cycles=cal_cycles,
            is_instructional=is_inst,
            instructional_days_step=inst_step,
            instructional_days_cum=inst_cum,
            warped_acad_cycles=warped_cycles,
            holiday_labels=hol_labels,
            y_count=y_causal,
            y_raw_observed=raw_counts,
            exposure=exposure,
            y_rate_per_1000=y_rate,
            calendar_flags=cal_flags,
        )

    def _extract_raw_tier_b(
        self,
        flat_df: pd.DataFrame,
        panel_df: Optional[pd.DataFrame],
        tier_a: TierABundle,
    ) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray, List[str], bool]:
        n = len(flat_df)
        cols = list(flat_df.columns)

        # 1. Category streams (wide columns `cat_*` OR panel `safeguarding_category` OR cohort counts)
        wide_cat_cols = [c for c in cols if c.startswith("cat_")]
        cat_col = self._find_col(panel_df.columns, CATEGORY_GROUP_CANDIDATES) if panel_df is not None else None
        count_col = self._find_col(panel_df.columns, COUNT_CANDIDATES) if panel_df is not None else None

        if wide_cat_cols:
            cat_names = [
                CATEGORY_DISPLAY_SHORT.get(
                    c[len("cat_") :], c.replace("cat_", "").replace("_", " ").title()
                )
                for c in wide_cat_cols
            ]
            raw_cats = np.column_stack(
                [
                    np.maximum(
                        0.0,
                        _causal_ffill_1d(pd.to_numeric(flat_df[c], errors="coerce").to_numpy(dtype=float), 0.0),
                    )
                    for c in wide_cat_cols
                ]
            )
        elif panel_df is not None and cat_col is not None and count_col is not None:
            unique_cats = sorted(panel_df[cat_col].dropna().astype(str).unique().tolist())
            dts_sorted = pd.to_datetime(flat_df["_bridge_dt"])
            raw_cats = np.zeros((n, len(unique_cats)), dtype=float)
            for c_i, c_name in enumerate(unique_cats):
                sub = panel_df[panel_df[cat_col].astype(str) == c_name]
                by_dt = sub.groupby("_bridge_dt")[count_col].sum()
                raw_cats[:, c_i] = np.array([float(by_dt.get(d, 0.0)) for d in dts_sorted], dtype=float)
            cat_names = unique_cats
        else:
            # Check if wide cohort referral columns exist to act as contagion channels
            wide_coh_ref = [c for c in cols if c.startswith("cohort_") and c.endswith("_referrals")]
            if len(wide_coh_ref) >= 2:
                cat_names = [
                    c.replace("cohort_", "").replace("_referrals", "").replace("_", " ").title()
                    for c in wide_coh_ref
                ]
                raw_cats = np.column_stack(
                    [
                        np.maximum(
                            0.0,
                            _causal_ffill_1d(pd.to_numeric(flat_df[c], errors="coerce").to_numpy(dtype=float), 0.0),
                        )
                        for c in wide_coh_ref
                    ]
                )
            else:
                cat_names = ["Aggregate Safeguarding Referrals"]
                raw_cats = tier_a.y_raw_observed[:, None].copy()

        if self.causal_mode == "forecast":
            causal_cats = np.zeros_like(raw_cats)
            for j in range(raw_cats.shape[1]):
                init_v = (
                    float(self.cat_inits_[j])
                    if (self.is_fitted_ and j < len(self.cat_inits_))
                    else (float(raw_cats[0, j]) if n > 0 else 1.0)
                )
                causal_cats[:, j] = _causal_shift_1d(raw_cats[:, j], lag=1, init_val=init_v)
        else:
            causal_cats = raw_cats.copy()

        # 2. Exogenous behavioral covariates
        preferred_forecast_covs = [
            "att_drop_t1",
            "pastoral_t1",
            "bursary_t3",
            "early_disengagement_index",
            "statutory_vulnerability_share",
            "deprivation_imd_1_3_share",
            "provision_16_18_ratio",
        ]
        preferred_telemetry_covs = [
            "overall_attendance_pct",
            "attendance_rate",
            "eng_maths_attendance_pct",
            "pastoral_rate_per_1000",
            "bursary_rate_per_1000",
            "teaching_days",
            "statutory_vulnerability_share",
            "deprivation_imd_1_3_share",
            "provision_16_18_ratio",
        ]

        cov_series_list: List[np.ndarray] = []
        cov_names: List[str] = []

        if self.causal_mode == "forecast":
            # Prefer already-lagged predictors if present; otherwise shift contemporaneous covariates by 1 step
            for c in preferred_forecast_covs:
                if c in cols:
                    arr = _causal_ffill_1d(pd.to_numeric(flat_df[c], errors="coerce").to_numpy(dtype=float), 0.0)
                    cov_series_list.append(arr)
                    cov_names.append(c)
            if len(cov_series_list) < 3:
                for c in preferred_telemetry_covs:
                    if c in cols and c not in cov_names:
                        raw_arr = _causal_ffill_1d(
                            pd.to_numeric(flat_df[c], errors="coerce").to_numpy(dtype=float), 0.0
                        )
                        init_v = float(raw_arr[0]) if n > 0 else 0.0
                        lagged = raw_arr if c == "teaching_days" else _causal_shift_1d(raw_arr, lag=1, init_val=init_v)
                        cov_series_list.append(lagged)
                        cov_names.append(f"{c}_causal")
        else:
            for c in preferred_telemetry_covs + preferred_forecast_covs:
                if c in cols and c not in cov_names:
                    arr = _causal_ffill_1d(pd.to_numeric(flat_df[c], errors="coerce").to_numpy(dtype=float), 0.0)
                    cov_series_list.append(arr)
                    cov_names.append(c)

        synthesized = False
        # Always guarantee >= 4 non-degenerate causal channels for multivariate modules (TDA, DMD, Signatures, RMT, OT)
        if len(cov_series_list) < 4:
            synthesized = len(cov_series_list) == 0
            r = tier_a.y_rate_per_1000
            ewma_fast = _causal_ewma_1d(r, alpha=0.45)
            ewma_slow = _causal_ewma_1d(r, alpha=0.15)
            vel = np.zeros(n, dtype=float)
            if n > 1:
                vel[1:] = r[1:] - r[:-1]
            vol = _causal_ewma_1d(np.abs(r - ewma_slow), alpha=0.25)

            synth_channels = [
                ("synth_rate_ewma_fast", ewma_fast),
                ("synth_rate_ewma_slow", ewma_slow),
                ("synth_rate_velocity", vel),
                ("synth_rate_volatility", vol),
            ]
            for s_name, s_arr in synth_channels:
                if len(cov_series_list) < 5:
                    cov_series_list.append(s_arr)
                    cov_names.append(s_name)

        raw_cov_mat = np.column_stack(cov_series_list) if cov_series_list else np.zeros((n, 4), dtype=float)
        return raw_cov_mat, cov_names, causal_cats, raw_cats, cat_names, synthesized

    def _extract_raw_tier_c(
        self,
        flat_df: pd.DataFrame,
        panel_df: Optional[pd.DataFrame],
        tier_a: TierABundle,
        raw_cov_mat: np.ndarray,
        cov_names: List[str],
    ) -> Tuple[
        Literal["Cohort Hierarchy Graph", "Latent Feature/Regime Graph"],
        List[str],
        List[str],
        np.ndarray,
        np.ndarray,
        List[str],
    ]:
        """
        Extract raw vertex stalks of shape (T, V, d_stalk) and node weights.
        Supports:
          1. Wide cohort columns: `cohort_<id>_attendance`, `cohort_<id>_engagement`, `cohort_<id>_pastoral`
          2. Long panel rows grouped by `curriculum_area` / `provision`
          3. Fallback `"Latent Feature/Regime Graph"` across 5 multi-scale behavioral channels
        """
        n = len(flat_df)
        cols = list(flat_df.columns)

        # Case 1: Wide cohort stalk columns (used by instructional-day frame & wide weekly panel)
        att_cols = sorted([c for c in cols if c.startswith("cohort_") and c.endswith("_attendance")])
        if len(att_cols) >= 3:
            node_ids = [c[len("cohort_") : -len("_attendance")] for c in att_cols]
            node_labels = [
                COHORT_DISPLAY_SHORT.get(nid, nid.replace("_", " ").title()) for nid in node_ids
            ]
            v_cnt = len(node_ids)
            stalk_dims = [
                "Attendance Stability",
                "Academic / Workshop Engagement",
                "Pastoral Harmony (Inv. Friction)",
            ]
            raw_stalks = np.zeros((n, v_cnt, 3), dtype=float)
            weights = np.ones(v_cnt, dtype=float)

            for idx_v, nid in enumerate(node_ids):
                c_att = f"cohort_{nid}_attendance"
                c_eng = f"cohort_{nid}_engagement"
                c_pas = f"cohort_{nid}_pastoral"
                c_enr = f"cohort_{nid}_enrolment"

                s_att = _causal_ffill_1d(pd.to_numeric(flat_df[c_att], errors="coerce").to_numpy(dtype=float), 90.0)
                if c_eng in cols:
                    s_eng = _causal_ffill_1d(
                        pd.to_numeric(flat_df[c_eng], errors="coerce").to_numpy(dtype=float), 86.0
                    )
                else:
                    s_eng = s_att - 3.5
                if c_pas in cols:
                    # Invert pastoral friction so all 3 stalk dimensions point positively toward "healthy consistency"
                    # making Contextual Masking (+Attendance vs -Pastoral Harmony) crystal clear!
                    s_pas_raw = _causal_ffill_1d(
                        pd.to_numeric(flat_df[c_pas], errors="coerce").to_numpy(dtype=float), 8.0
                    )
                    s_pas = -s_pas_raw
                else:
                    s_pas = -0.1 * (100.0 - s_att)

                if self.causal_mode == "forecast":
                    s_att = _causal_shift_1d(s_att, 1, float(s_att[0]) if n > 0 else 90.0)
                    s_eng = _causal_shift_1d(s_eng, 1, float(s_eng[0]) if n > 0 else 86.0)
                    s_pas = _causal_shift_1d(s_pas, 1, float(s_pas[0]) if n > 0 else -8.0)

                raw_stalks[:, idx_v, 0] = s_att
                raw_stalks[:, idx_v, 1] = s_eng
                raw_stalks[:, idx_v, 2] = s_pas

                if c_enr in cols:
                    weights[idx_v] = max(
                        1.0, float(pd.to_numeric(flat_df[c_enr].iloc[0], errors="coerce") or 1000.0)
                    )

            weights = weights / np.sum(weights)
            return "Cohort Hierarchy Graph", node_ids, node_labels, weights, raw_stalks, stalk_dims

        # Case 2: Long panel DataFrame with cohort column (`curriculum_area`, `provision`, etc.)
        cohort_col = self._find_col(panel_df.columns, COHORT_GROUP_CANDIDATES) if panel_df is not None else None
        if panel_df is not None and cohort_col is not None:
            unique_cohorts = [
                c for c in sorted(panel_df[cohort_col].dropna().astype(str).unique().tolist())
                if c != "College-Wide"
            ]
            if len(unique_cohorts) >= 2:
                v_cnt = len(unique_cohorts)
                node_labels = [COHORT_DISPLAY_SHORT.get(c, c) for c in unique_cohorts]
                stalk_dims = [
                    "Attendance Stability",
                    "Academic / Workshop Engagement",
                    "Pastoral Harmony (Inv. Friction)",
                ]
                raw_stalks = np.zeros((n, v_cnt, 3), dtype=float)
                weights = np.ones(v_cnt, dtype=float)
                dts_sorted = pd.to_datetime(flat_df["_bridge_dt"])
                exp_col = self._find_col(panel_df.columns, EXPOSURE_CANDIDATES)
                cnt_col = self._find_col(panel_df.columns, COUNT_CANDIDATES)

                for idx_v, coh_name in enumerate(unique_cohorts):
                    sub = panel_df[panel_df[cohort_col].astype(str) == coh_name].sort_values("_bridge_dt")
                    if exp_col and len(sub) > 0:
                        weights[idx_v] = max(
                            1.0, float(pd.to_numeric(sub[exp_col].iloc[0], errors="coerce") or 1000.0)
                        )
                    att_c = "overall_attendance_pct" if "overall_attendance_pct" in sub.columns else "attendance_rate"
                    by_dt_att = sub.groupby("_bridge_dt")[att_c].mean() if att_c in sub.columns else {}
                    by_dt_eng = (
                        sub.groupby("_bridge_dt")["eng_maths_attendance_pct"].mean()
                        if "eng_maths_attendance_pct" in sub.columns
                        else {}
                    )
                    if "pastoral_rate_per_1000" in sub.columns:
                        by_dt_pas = sub.groupby("_bridge_dt")["pastoral_rate_per_1000"].mean()
                    elif cnt_col and exp_col:
                        by_dt_pas = (
                            sub.groupby("_bridge_dt")[cnt_col].sum()
                            / np.maximum(1.0, sub.groupby("_bridge_dt")[exp_col].first())
                            * 1000.0
                        )
                    else:
                        by_dt_pas = {}

                    s_att = _causal_ffill_1d(
                        np.array([float(by_dt_att.get(d, np.nan)) for d in dts_sorted], dtype=float), 90.0
                    )
                    s_eng = _causal_ffill_1d(
                        np.array([float(by_dt_eng.get(d, np.nan)) for d in dts_sorted], dtype=float),
                        float(s_att[0]) - 3.5 if len(s_att) > 0 else 86.5,
                    )
                    s_pas = -_causal_ffill_1d(
                        np.array([float(by_dt_pas.get(d, np.nan)) for d in dts_sorted], dtype=float), 6.0
                    )

                    if self.causal_mode == "forecast":
                        s_att = _causal_shift_1d(s_att, 1, float(s_att[0]) if n > 0 else 90.0)
                        s_eng = _causal_shift_1d(s_eng, 1, float(s_eng[0]) if n > 0 else 86.5)
                        s_pas = _causal_shift_1d(s_pas, 1, float(s_pas[0]) if n > 0 else -6.0)

                    raw_stalks[:, idx_v, 0] = s_att
                    raw_stalks[:, idx_v, 1] = s_eng
                    raw_stalks[:, idx_v, 2] = s_pas

                weights = weights / np.sum(weights)
                return "Cohort Hierarchy Graph", unique_cohorts, node_labels, weights, raw_stalks, stalk_dims

        # Case 3: Purely flat institutional series -> Latent Feature/Regime Graph (5 nodes x 3 multi-scale stalks)
        latent_nodes = [
            ("regime_attendance", "Attendance & Commute Regime"),
            ("regime_engagement", "Curriculum Engagement Pulse"),
            ("regime_pastoral", "Pastoral Triage Pressure"),
            ("regime_hardship", "Bursary & Welfare Friction"),
            ("regime_momentum", "Referral Contagion Momentum"),
        ]
        v_cnt = len(latent_nodes)
        stalk_dims = ["Short-Term Level", "Medium-Term Trend (EWMA)", "Local Stability (-Volatility)"]
        raw_stalks = np.zeros((n, v_cnt, 3), dtype=float)

        # Build 5 base causal signals from available covariates + rate dynamics
        base_signals: List[np.ndarray] = []
        for idx_v in range(v_cnt):
            if idx_v < raw_cov_mat.shape[1]:
                sig = raw_cov_mat[:, idx_v].copy()
            else:
                sig = _causal_ewma_1d(tier_a.y_rate_per_1000, alpha=0.18 * (idx_v + 1))
            base_signals.append(sig)

        for idx_v, sig in enumerate(base_signals):
            med_ewma = _causal_ewma_1d(sig, alpha=0.25)
            loc_vol = -_causal_ewma_1d(np.abs(sig - med_ewma), alpha=0.30)
            raw_stalks[:, idx_v, 0] = sig
            raw_stalks[:, idx_v, 1] = med_ewma
            raw_stalks[:, idx_v, 2] = loc_vol

        weights = np.array([0.24, 0.20, 0.22, 0.16, 0.18], dtype=float)
        return (
            "Latent Feature/Regime Graph",
            [x[0] for x in latent_nodes],
            [x[1] for x in latent_nodes],
            weights,
            raw_stalks,
            stalk_dims,
        )

    def fit(self, df: pd.DataFrame, y: Any = None) -> "AdaptiveSchemaBridge":
        train_df = df.iloc[: self.train_cutoff].copy() if self.train_cutoff is not None else df.copy()
        self.is_fitted_ = False

        flat_tr, panel_tr, _, date_col = self._canonicalize_timeline(train_df)
        tier_a_tr = self._extract_raw_tier_a(flat_tr, date_col)

        self.init_count_ = float(np.median(tier_a_tr.y_raw_observed[: max(1, min(6, len(flat_tr)))]))
        self.init_rate_ = float(np.median(tier_a_tr.y_rate_per_1000[: max(1, min(6, len(flat_tr)))]))
        self.rate_median_ = float(np.median(tier_a_tr.y_rate_per_1000))
        mad_r = float(np.median(np.abs(tier_a_tr.y_rate_per_1000 - self.rate_median_)))
        self.rate_mad_ = max(0.25, 1.4826 * mad_r)

        raw_cov_tr, cov_names, causal_cats_tr, raw_cats_tr, cat_names, _ = self._extract_raw_tier_b(
            flat_tr, panel_tr, tier_a_tr
        )
        self.cov_names_ = cov_names
        self.cov_medians_ = np.median(raw_cov_tr, axis=0)
        q75 = np.percentile(raw_cov_tr, 75, axis=0)
        q25 = np.percentile(raw_cov_tr, 25, axis=0)
        stds = np.std(raw_cov_tr, axis=0)
        self.cov_iqrs_ = np.maximum(1e-4, np.where((q75 - q25) > 1e-4, (q75 - q25) / 1.349, np.maximum(1e-4, stds)))
        self.cat_names_ = cat_names
        self.cat_inits_ = np.median(raw_cats_tr[: max(1, min(6, len(flat_tr)))], axis=0)

        top_mode, node_ids, node_labels, weights, raw_stalks_tr, stalk_dims = self._extract_raw_tier_c(
            flat_tr, panel_tr, tier_a_tr, raw_cov_tr, cov_names
        )
        self.topology_mode_ = top_mode
        self.node_ids_ = node_ids
        self.node_labels_ = node_labels
        self.node_weights_ = weights
        self.stalk_dim_names_ = stalk_dims

        # Robust training-fold stalk normalization (V, d)
        self.stalk_medians_ = np.median(raw_stalks_tr, axis=0)
        sq75 = np.percentile(raw_stalks_tr, 75, axis=0)
        sq25 = np.percentile(raw_stalks_tr, 25, axis=0)
        s_stds = np.std(raw_stalks_tr, axis=0)
        self.stalk_iqrs_ = np.maximum(
            1e-4, np.where((sq75 - sq25) > 1e-4, (sq75 - sq25) / 1.349, np.maximum(1e-4, s_stds))
        )
        self.stalk_inits_ = raw_stalks_tr[0].copy() if len(raw_stalks_tr) > 0 else np.zeros_like(self.stalk_medians_)

        norm_stalks_tr = (raw_stalks_tr - self.stalk_medians_[None, :, :]) / self.stalk_iqrs_[None, :, :]
        # Flatten stalk dimensions across training time to compute training-fold vertex similarity
        v_cnt = len(node_ids)
        flat_vertex_profiles = np.zeros((v_cnt, max(2, len(flat_tr) * raw_stalks_tr.shape[2])), dtype=float)
        for v_i in range(v_cnt):
            vec = norm_stalks_tr[:, v_i, :].reshape(-1)
            flat_vertex_profiles[v_i, : len(vec)] = vec

        if flat_vertex_profiles.shape[1] >= 2 and v_cnt >= 2:
            norms = np.linalg.norm(flat_vertex_profiles - np.mean(flat_vertex_profiles, axis=1, keepdims=True), axis=1)
            if np.all(norms > 1e-8):
                corr_v = np.corrcoef(flat_vertex_profiles)
                corr_v = np.nan_to_num(corr_v, nan=0.0)
            else:
                corr_v = np.eye(v_cnt, dtype=float)
        else:
            corr_v = np.eye(v_cnt, dtype=float)

        self.adjacency_ = _ensure_connected_symmetric_graph(corr_v, top_k=2)
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> SchemaBundle:
        if not self.is_fitted_:
            self.fit(df)

        flat_df, panel_df, row_to_step, date_col = self._canonicalize_timeline(df)
        tier_a = self._extract_raw_tier_a(flat_df, date_col)
        raw_cov_mat, cov_names, causal_cats, raw_cats, cat_names, synthesized = self._extract_raw_tier_b(
            flat_df, panel_df, tier_a
        )

        # Align covariate columns to fitted training schema
        n = len(flat_df)
        d_fit = len(self.cov_names_)
        aligned_raw_cov = np.zeros((n, d_fit), dtype=float)
        name_to_idx = {nm: idx for idx, nm in enumerate(cov_names)}
        for j, fit_nm in enumerate(self.cov_names_):
            if fit_nm in name_to_idx:
                aligned_raw_cov[:, j] = raw_cov_mat[:, name_to_idx[fit_nm]]
            else:
                aligned_raw_cov[:, j] = float(self.cov_medians_[j])

        norm_cov = (aligned_raw_cov - self.cov_medians_[None, :]) / self.cov_iqrs_[None, :]
        norm_cov = np.clip(norm_cov, -25.0, 25.0)

        # Align categories
        c_fit = len(self.cat_names_)
        aligned_causal_cats = np.zeros((n, c_fit), dtype=float)
        aligned_raw_cats = np.zeros((n, c_fit), dtype=float)
        cat_map = {nm: idx for idx, nm in enumerate(cat_names)}
        for j, c_nm in enumerate(self.cat_names_):
            if c_nm in cat_map:
                aligned_causal_cats[:, j] = causal_cats[:, cat_map[c_nm]]
                aligned_raw_cats[:, j] = raw_cats[:, cat_map[c_nm]]
            elif c_fit == 1:
                aligned_causal_cats[:, 0] = tier_a.y_count
                aligned_raw_cats[:, 0] = tier_a.y_raw_observed

        tier_b = TierBBundle(
            covariate_matrix=norm_cov,
            raw_covariate_matrix=aligned_raw_cov,
            covariate_names=list(self.cov_names_),
            category_counts=aligned_causal_cats,
            raw_category_counts=aligned_raw_cats,
            category_names=list(self.cat_names_),
            synthesized_univariate=synthesized,
        )

        # Tier C stalks normalized strictly using training-fold medians & IQRs
        _, node_ids_cur, _, _, raw_stalks_cur, _ = self._extract_raw_tier_c(
            flat_df, panel_df, tier_a, aligned_raw_cov, self.cov_names_
        )
        v_fit = len(self.node_ids_)
        d_stalk = len(self.stalk_dim_names_)
        aligned_stalks = np.zeros((n, v_fit, d_stalk), dtype=float)
        cur_node_map = {nid: idx for idx, nid in enumerate(node_ids_cur)}
        for v_i, fit_nid in enumerate(self.node_ids_):
            if fit_nid in cur_node_map and raw_stalks_cur.shape[2] == d_stalk:
                aligned_stalks[:, v_i, :] = raw_stalks_cur[:, cur_node_map[fit_nid], :]
            else:
                aligned_stalks[:, v_i, :] = self.stalk_medians_[v_i, :][None, :]

        norm_stalks = (aligned_stalks - self.stalk_medians_[None, :, :]) / self.stalk_iqrs_[None, :, :]
        norm_stalks = np.clip(norm_stalks, -25.0, 25.0)
        # Scalar stress per vertex: negative mean of healthy stalk dimensions -> positive when stressed
        node_scalar_stress = -np.mean(norm_stalks, axis=2)

        tier_c = TierCBundle(
            topology_mode=self.topology_mode_,
            node_ids=list(self.node_ids_),
            node_labels=list(self.node_labels_),
            node_weights=self.node_weights_.copy(),
            adjacency=self.adjacency_.copy(),
            stalks=norm_stalks,
            stalk_dim_names=list(self.stalk_dim_names_),
            node_scalar_stress=node_scalar_stress,
        )

        return SchemaBundle(
            tier_a=tier_a,
            tier_b=tier_b,
            tier_c=tier_c,
            n_steps=n,
            causal_mode=self.causal_mode,
            row_to_step_idx=row_to_step,
        )


def attach_panel_hierarchy_wide(cw_df: pd.DataFrame, panel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Non-destructively attach wide curriculum-area cohort columns from `panel_df` onto
    `cw_df` (1 row per `week_start`) so Tier C `Cohort Hierarchy Graph` operates at full
    departmental resolution during weekly College-Wide benchmarks.
    """
    out = cw_df.copy().reset_index(drop=True)
    if "curriculum_area" not in panel_df.columns or "week_start" not in panel_df.columns:
        return out

    areas = [a for a in sorted(panel_df["curriculum_area"].unique().tolist()) if a != "College-Wide"]
    for area in areas:
        slug = (
            COHORT_DISPLAY_SHORT.get(area, area)
            .lower()
            .replace("&", "and")
            .replace(" ", "_")
            .replace("__", "_")
        )
        sub = panel_df[panel_df["curriculum_area"] == area].set_index("week_start")
        out[f"cohort_{slug}_referrals"] = out["week_start"].map(sub["referral_count"]).fillna(0.0).to_numpy(dtype=float)
        out[f"cohort_{slug}_enrolment"] = (
            out["week_start"].map(sub["active_enrolments"]).ffill().fillna(1000.0).to_numpy(dtype=float)
        )
        out[f"cohort_{slug}_attendance"] = (
            out["week_start"].map(sub["overall_attendance_pct"]).ffill().fillna(90.0).to_numpy(dtype=float)
        )
        out[f"cohort_{slug}_engagement"] = (
            out["week_start"].map(sub["eng_maths_attendance_pct"]).ffill().fillna(86.0).to_numpy(dtype=float)
        )
        out[f"cohort_{slug}_pastoral"] = (
            out["week_start"].map(sub["pastoral_rate_per_1000"]).ffill().fillna(8.0).to_numpy(dtype=float)
        )
    return out


def build_instructional_day_telemetry_frame(
    num_days: int = 420,
    start_date: str = "2024-09-02",
    seed: int = 42,
    provision_filter: str = "All",
) -> pd.DataFrame:
    """
    Generate a deterministic, strictly causal UK Further Education daily safeguarding
    telemetry DataFrame for South Devon College spanning instructional days, weekends,
    half-term breaks, winter/easter/summer closures, 5 safeguarding categories (with
    subcritical Hawkes peer/online contagion), and 6 FE curriculum cohorts (with
    contextual masking episodes preceding referral crests).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, periods=num_days, freq="D")

    cat_specs = [
        ("mental_health", 0.42),
        ("abuse_neglect", 0.21),
        ("online_safety", 0.16),
        ("exploitation", 0.09),
        ("other_concerns", 0.12),
    ]
    n_cats = len(cat_specs)

    # Subcritical cross-exciting Hawkes branching parameters (spectral radius ~ 0.68 < 1.0)
    # Online safety (idx 2) sparks Mental Health (idx 0) and Exploitation (idx 3) cascades;
    # Abuse/Neglect (idx 1) sparks Mental Health (idx 0) secondary pastoral referrals.
    beta_mat = np.array(
        [
            [0.32, 0.28, 0.35, 0.30, 0.40],
            [0.35, 0.30, 0.40, 0.32, 0.42],
            [0.38, 0.40, 0.26, 0.34, 0.45],
            [0.40, 0.34, 0.29, 0.28, 0.45],
            [0.42, 0.38, 0.36, 0.38, 0.35],
        ],
        dtype=float,
    )
    gamma_target = np.array(
        [
            [0.24, 0.18, 0.29, 0.12, 0.06],
            [0.08, 0.21, 0.07, 0.14, 0.04],
            [0.09, 0.05, 0.28, 0.11, 0.05],
            [0.05, 0.10, 0.19, 0.20, 0.03],
            [0.07, 0.06, 0.09, 0.05, 0.14],
        ],
        dtype=float,
    )
    alpha_mat = gamma_target * beta_mat

    cohorts = [
        ("epyp_send", 640.0, 91.2, 87.5, 9.2),
        ("health_social_care", 1210.0, 90.4, 86.8, 6.8),
        ("construction_trades", 1380.0, 89.1, 84.5, 5.4),
        ("digital_creative", 1010.0, 90.8, 87.2, 5.8),
        ("engineering_marine", 1150.0, 91.5, 88.0, 4.4),
        ("adult_apprentices", 1980.0, 92.6, 89.8, 3.3),
    ]

    prov_scale = 1.0
    if provision_filter == "EPYP":
        prov_scale = 0.61
    elif provision_filter == "Adult":
        prov_scale = 0.25
    elif provision_filter == "Apprenticeships":
        prov_scale = 0.14

    memory = np.zeros((n_cats, n_cats), dtype=float)
    inst_cum = 0.0
    was_holiday = False
    rows: List[Dict[str, Any]] = []

    for t_idx, dt in enumerate(dates):
        frac_inst, hol_lbl = _infer_uk_fe_holiday_for_date(dt, "daily")
        is_inst = 1 if frac_inst > 0.5 else 0
        is_post_return = bool(is_inst == 1 and was_holiday and hol_lbl == "" and dt.dayofweek == 0)
        # Check if previous Friday was a multi-day holiday break (not just normal weekend)
        if t_idx >= 3 and is_inst == 1:
            prev_3_hol = all(
                _infer_uk_fe_holiday_for_date(dates[t_idx - k], "daily")[1] not in ("", "Weekend")
                for k in (3,)
            )
            is_post_return = prev_3_hol
        was_holiday = (is_inst == 0)

        if is_inst:
            inst_cum += 1.0

        # Warped instructional cycle wave (peaks during mid-Autumn, January return, and May exam crunch)
        acad_phase = 2.0 * np.pi * (inst_cum % 190.0) / 190.0
        term_wave = 1.0 + 0.34 * np.sin(acad_phase - 0.35) + 0.18 * np.cos(2.0 * acad_phase - 0.6)
        induction_boost = 1.32 if (dt.month == 9 or (dt.month == 10 and dt.day <= 18)) else 1.0
        return_boost = 1.42 if is_post_return else 1.0
        crunch_boost = 1.24 if dt.month in (11, 5) else 1.0

        # Pre-crest contextual masking episode (e.g. ~8-12 days before half-term or exam peaks)
        masking_pulse = 0.0
        if is_inst and (
            (dt.month == 10 and 8 <= dt.day <= 22)
            or (dt.month == 1 and 8 <= dt.day <= 22)
            or (dt.month == 5 and 6 <= dt.day <= 20)
            or (t_idx >= len(dates) - 18 and is_inst)
        ):
            masking_pulse = 1.0

        # Update causal Hawkes decay memory from strictly prior days (< t_idx)
        decay = np.exp(-beta_mat)
        memory = memory * decay

        base_daily_total = (8.2 * prov_scale * term_wave * induction_boost * return_boost * crunch_boost)
        if not is_inst:
            # Minimal out-of-hours duty referrals on weekends/holidays
            base_daily_total = 0.35 * prov_scale if hol_lbl == "Weekend" else 0.55 * prov_scale

        cat_counts_t = np.zeros(n_cats, dtype=int)
        for c_i, (_, share) in enumerate(cat_specs):
            mu_c = base_daily_total * share
            # Occasional online/community shock on active teaching days
            if is_inst and c_i == 2 and (t_idx in (26, 74, 142, 248, 368, max(10, num_days - 9))):
                mu_c += 6.5 * prov_scale
            endog_c = float(np.sum(memory[c_i, :])) if is_inst else 0.15 * float(np.sum(memory[c_i, :]))
            lam_c = max(0.02, mu_c + endog_c)
            # Overdispersed NB sampling
            r_nb = 6.5
            p_nb = r_nb / (r_nb + lam_c)
            cnt_c = int(rng.negative_binomial(r_nb, p_nb))
            cat_counts_t[c_i] = cnt_c

        # Add today's shock into memory AFTER sampling today (so M(t) at step t reflects s < t!)
        for j_i in range(n_cats):
            if cat_counts_t[j_i] > 0 and is_inst:
                excess_shock = max(0.0, float(cat_counts_t[j_i]) - 0.55 * base_daily_total * cat_specs[j_i][1])
                memory[:, j_i] += alpha_mat[:, j_i] * excess_shock

        total_ref = int(np.sum(cat_counts_t))
        total_enrol = int(sum(c[1] for c in cohorts))

        row: Dict[str, Any] = {
            "date": dt.strftime("%Y-%m-%d"),
            "academic_year": f"{dt.year}/{str(dt.year + 1)[-2:]}" if dt.month >= 9 else f"{dt.year - 1}/{str(dt.year)[-2:]}",
            "is_instructional_day": is_inst,
            "holiday_label": hol_lbl,
            "referral_count": total_ref,
            "active_enrolments": total_enrol,
            "is_autumn_transition": 1 if (dt.month == 9 or (dt.month == 10 and dt.day <= 18)) else 0,
            "is_post_holiday_week": 1 if is_post_return else 0,
            "is_assessment_crunch": 1 if dt.month in (11, 5) else 0,
        }
        for c_i, (c_slug, _) in enumerate(cat_specs):
            row[f"cat_{c_slug}"] = int(cat_counts_t[c_i])

        # Cohort stalks with realistic coherent movement on calm days & contextual masking on pre-surge days
        common_att_shift = -1.2 * (term_wave - 1.0) + float(rng.normal(0.0, 0.28))
        att_list: List[float] = []
        pas_list: List[float] = []

        for v_i, (c_slug, c_enrol, b_att, b_eng, b_pas) in enumerate(cohorts):
            att_v = b_att + common_att_shift + float(rng.normal(0.0, 0.32))
            eng_v = b_eng + 0.92 * common_att_shift + float(rng.normal(0.0, 0.35))
            pas_v = b_pas - 0.85 * common_att_shift + 0.08 * total_ref + float(rng.normal(0.0, 0.35))

            # Inject realistic Contextual Masking in EPYP & Digital cohorts during masking windows:
            # Surface attendance remains high (+0.9% above baseline), while workshop engagement drops
            # and low-level pastoral friction surges relative to connected peer cohorts!
            if masking_pulse > 0.0 and v_i in (0, 3):
                att_v += 1.35 * masking_pulse
                eng_v -= 3.40 * masking_pulse
                pas_v += 4.60 * masking_pulse
            elif masking_pulse > 0.0 and v_i == 1:
                att_v += 0.65 * masking_pulse
                eng_v -= 1.80 * masking_pulse
                pas_v += 2.40 * masking_pulse

            att_v = float(np.clip(att_v, 74.0, 98.5))
            eng_v = float(np.clip(eng_v, 68.0, 97.5))
            pas_v = float(max(0.5, pas_v))

            coh_share = c_enrol / total_enrol
            row[f"cohort_{c_slug}_enrolment"] = int(c_enrol)
            row[f"cohort_{c_slug}_referrals"] = int(round(total_ref * coh_share))
            row[f"cohort_{c_slug}_attendance"] = round(att_v, 2)
            row[f"cohort_{c_slug}_engagement"] = round(eng_v, 2)
            row[f"cohort_{c_slug}_pastoral"] = round(pas_v, 3)
            att_list.append(att_v * c_enrol)
            pas_list.append(pas_v * c_enrol)

        row["overall_attendance_pct"] = round(sum(att_list) / total_enrol, 2)
        row["pastoral_rate_per_1000"] = round(sum(pas_list) / total_enrol, 3)
        row["bursary_rate_per_1000"] = round(
            float(max(1.0, 4.8 + 0.9 * (term_wave - 1.0) + 0.8 * masking_pulse + rng.normal(0.0, 0.25))),
            3,
        )
        rows.append(row)

    return pd.DataFrame(rows)
