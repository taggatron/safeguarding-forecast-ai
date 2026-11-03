"""Data processing, aggregation, CSV validation, and PII guardrails."""
import io
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from app.data.synthetic_generator import (
    BASE_FE_TEACHING_DAYS,
    get_academic_term,
    get_academic_year_label,
)
from app.schemas.models import DataValidationIssue, DataValidationReport

# Strictly prohibited personal / individual case-level column patterns
PROHIBITED_PII_PATTERNS = [
    r"\b(student_?id|learner_?id|uln|upn|pupil_?id|person_?id|candidate_?no)\b",
    r"\b(name|first_?name|last_?name|surname|forename|full_?name|student_?name|learner_?name)\b",
    r"\b(dob|date_?of_?birth|birth_?date)\b",
    r"\b(narrative|case_?notes?|notes|incident_?details?|description|free_?text|comment|summary_?text)\b",
    r"\b(email|phone|mobile|postcode|zip_?code|home_?address|address|nhs_?number|ni_?number)\b",
    r"\b(social_?worker|allocated_?worker|staff_?name|dsl_?name)\b",
]

# Automatic column alias recognition map
COLUMN_ALIASES: Dict[str, List[str]] = {
    "period_start": [
        "period_start",
        "month",
        "date",
        "period",
        "month_start",
        "reporting_month",
        "year_month",
    ],
    "provision": [
        "provision",
        "provision_type",
        "cohort",
        "study_type",
        "funding_stream",
        "department",
        "faculty",
    ],
    "safeguarding_category": [
        "safeguarding_category",
        "category",
        "concern_category",
        "concern_type",
        "issue_type",
        "safeguarding_type",
    ],
    "concern_count": [
        "concern_count",
        "concerns",
        "count",
        "incident_count",
        "incidents",
        "recorded_concerns",
        "total_concerns",
        "referrals",
    ],
    "enrolled_students": [
        "enrolled_students",
        "enrolment",
        "enrollment",
        "student_count",
        "students",
        "headcount",
        "learners",
    ],
    "teaching_days": [
        "teaching_days",
        "college_days",
        "open_days",
        "term_days",
        "instructional_days",
    ],
    "attendance_rate": [
        "attendance_rate",
        "attendance",
        "attendance_pct",
        "avg_attendance",
    ],
    "reporting_change_flag": [
        "reporting_change_flag",
        "reporting_change",
        "system_change_flag",
        "policy_change_flag",
    ],
}


def detect_prohibited_pii_columns(columns: List[str]) -> List[str]:
    """Identify any column names that indicate student-level PII or case narratives."""
    flagged = []
    for col in columns:
        normalized = col.strip().lower()
        for pattern in PROHIBITED_PII_PATTERNS:
            if re.search(pattern, normalized):
                flagged.append(col)
                break
    return flagged


def auto_map_columns(columns: List[str]) -> Dict[str, str]:
    """Automatically map uploaded CSV column headers to standard schema fields."""
    mapping: Dict[str, str] = {}
    lower_to_orig = {c.strip().lower(): c for c in columns}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lower_to_orig:
                mapping[canonical] = lower_to_orig[alias]
                break
    return mapping


def aggregate_monthly_series(
    df: pd.DataFrame,
    provision: str = "All",
    safeguarding_category: str = "All",
    academic_year_start_month: int = 9,
) -> pd.DataFrame:
    """
    Aggregate granular (period_start, provision, safeguarding_category) records
    into a clean monthly time series without double-counting enrolled_students.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "period_start",
                "academic_year",
                "academic_term",
                "concern_count",
                "enrolled_students",
                "teaching_days",
                "attendance_rate",
                "reporting_change_flag",
                "concerns_per_1000",
                "rolling_3m_avg",
                "rolling_6m_avg",
            ]
        )

    work = df.copy()

    # Step 1: Compute monthly enrolled_students and attendance_rate at the (period_start, provision) level
    # BEFORE filtering by safeguarding_category, because enrolled_students is a provision-level denominator.
    if provision != "All":
        prov_df = work[work["provision"] == provision].copy()
    else:
        prov_df = work.copy()

    if prov_df.empty:
        return pd.DataFrame()

    # Take unique (period_start, provision) enrolment & attendance first, then aggregate across provisions
    prov_monthly = (
        prov_df.groupby(["period_start", "provision"], as_index=False)
        .agg(
            enrolled_students=("enrolled_students", "first"),
            teaching_days=("teaching_days", "first"),
            attendance_rate=("attendance_rate", "mean"),
            reporting_change_flag=("reporting_change_flag", "max"),
        )
    )

    denom_monthly = (
        prov_monthly.groupby("period_start", as_index=False)
        .agg(
            enrolled_students=("enrolled_students", "sum"),
            teaching_days=("teaching_days", "max"),
            attendance_rate=("attendance_rate", "mean"),
            reporting_change_flag=("reporting_change_flag", "max"),
        )
    )

    # Step 2: Filter for concern_count numerator
    num_df = prov_df.copy()
    if safeguarding_category != "All":
        num_df = num_df[num_df["safeguarding_category"] == safeguarding_category].copy()
    else:
        # Exclude any synthetic/imported 'Total' or 'All' summary rows if granular categories also exist
        unique_cats = set(num_df["safeguarding_category"].unique())
        if len(unique_cats) > 1:
            total_aliases = {"total", "all", "all categories", "college total"}
            non_total_mask = ~num_df["safeguarding_category"].str.strip().str.lower().isin(total_aliases)
            if non_total_mask.any():
                num_df = num_df[non_total_mask]

    num_monthly = (
        num_df.groupby("period_start", as_index=False)
        .agg(concern_count=("concern_count", "sum"))
    )

    merged = pd.merge(denom_monthly, num_monthly, on="period_start", how="left")
    merged["concern_count"] = merged["concern_count"].fillna(0).astype(int)
    merged["dt"] = pd.to_datetime(merged["period_start"])
    merged = merged.sort_values("dt").reset_index(drop=True)

    merged["academic_year"] = merged["dt"].apply(
        lambda d: get_academic_year_label(d.year, d.month, academic_year_start_month)
    )
    merged["academic_term"] = merged["dt"].apply(lambda d: get_academic_term(d.month))
    merged["month_num"] = merged["dt"].dt.month
    merged["period_label"] = merged["dt"].dt.strftime("%b %Y")

    merged["concerns_per_1000"] = np.where(
        merged["enrolled_students"] > 0,
        np.round((merged["concern_count"] / merged["enrolled_students"]) * 1000.0, 2),
        0.0,
    )

    merged["rolling_3m_avg"] = merged["concern_count"].rolling(window=3, min_periods=1).mean().round(1)
    merged["rolling_6m_avg"] = merged["concern_count"].rolling(window=6, min_periods=1).mean().round(1)

    return merged


def validate_and_parse_csv(
    csv_bytes: bytes,
    column_mapping: Optional[Dict[str, str]] = None,
    academic_year_start_month: int = 9,
) -> Tuple[DataValidationReport, Optional[pd.DataFrame]]:
    """
    Validate an uploaded CSV file against Safeguarding Forecast AI governance and schema rules.
    Rejects personal data (PII), detects missing months, checks duplicates, prevents double-counting,
    and returns both a structured validation report and a normalized DataFrame when valid.
    """
    issues: List[DataValidationIssue] = []

    try:
        text = csv_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = csv_bytes.decode("latin-1", errors="replace")

    try:
        raw_df = pd.read_csv(io.StringIO(text))
    except Exception as exc:
        return (
            DataValidationReport(
                is_valid=False,
                can_import=False,
                row_count=0,
                issues=[
                    DataValidationIssue(
                        severity="error",
                        code="CSV_PARSE_ERROR",
                        message=f"Could not parse CSV file: {exc}",
                    )
                ],
            ),
            None,
        )

    if raw_df.empty:
        return (
            DataValidationReport(
                is_valid=False,
                can_import=False,
                row_count=0,
                issues=[
                    DataValidationIssue(
                        severity="error",
                        code="EMPTY_FILE",
                        message="The uploaded CSV file contains no rows.",
                    )
                ],
            ),
            None,
        )

    detected_cols = list(raw_df.columns)
    pii_cols = detect_prohibited_pii_columns(detected_cols)
    if pii_cols:
        issues.append(
            DataValidationIssue(
                severity="error",
                code="PROHIBITED_PERSONAL_DATA",
                message=(
                    "Personal data or individual student case fields were detected and rejected: "
                    f"{', '.join(pii_cols)}. This system only accepts anonymised, monthly aggregated "
                    "safeguarding counts. Remove all student identifiers, names, dates of birth, and case notes."
                ),
                affected_rows_or_fields=pii_cols,
            )
        )

    # Scan first 50 rows for accidental long free-text narratives or email patterns
    for col in detected_cols:
        if col in pii_cols:
            continue
        sample_vals = raw_df[col].dropna().astype(str).head(50)
        for val in sample_vals:
            if "@" in val and "." in val.split("@")[-1]:
                issues.append(
                    DataValidationIssue(
                        severity="error",
                        code="PII_EMAIL_DETECTED",
                        message=f"Column '{col}' appears to contain email addresses. Only aggregated statistics are permitted.",
                        affected_rows_or_fields=[col],
                    )
                )
                pii_cols.append(col)
                break
            if len(val) > 140:
                issues.append(
                    DataValidationIssue(
                        severity="error",
                        code="CASE_NARRATIVE_DETECTED",
                        message=f"Column '{col}' contains long text (>140 chars) resembling case notes or narratives.",
                        affected_rows_or_fields=[col],
                    )
                )
                pii_cols.append(col)
                break

    suggested_mapping = auto_map_columns(detected_cols)
    effective_mapping = {**suggested_mapping, **(column_mapping or {})}
    # Remove empty mappings
    effective_mapping = {k: v for k, v in effective_mapping.items() if v and v in detected_cols}

    required_fields = ["period_start", "concern_count"]
    missing_required = [f for f in required_fields if f not in effective_mapping]
    if missing_required:
        issues.append(
            DataValidationIssue(
                severity="error",
                code="MISSING_REQUIRED_COLUMNS",
                message=f"Missing required column mappings for: {', '.join(missing_required)}. Please map these columns.",
                affected_rows_or_fields=missing_required,
            )
        )

    if any(i.severity == "error" for i in issues):
        preview_safe = (
            raw_df.drop(columns=list(set(pii_cols)), errors="ignore")
            .head(10)
            .fillna("")
            .to_dict(orient="records")
        )
        return (
            DataValidationReport(
                is_valid=False,
                can_import=False,
                row_count=len(raw_df),
                detected_columns=detected_cols,
                suggested_mapping=effective_mapping,
                rejected_personal_data_columns=list(set(pii_cols)),
                issues=issues,
                preview_rows=preview_safe,
            ),
            None,
        )

    # Normalize into canonical dataframe
    norm = pd.DataFrame()
    norm["raw_period"] = raw_df[effective_mapping["period_start"]]
    parsed_dates = pd.to_datetime(norm["raw_period"], errors="coerce")
    bad_date_count = int(parsed_dates.isna().sum())
    if bad_date_count > 0:
        issues.append(
            DataValidationIssue(
                severity="error",
                code="INVALID_DATES",
                message=f"{bad_date_count} row(s) have invalid date values in '{effective_mapping['period_start']}'. Use YYYY-MM-DD or YYYY-MM.",
            )
        )
        return (
            DataValidationReport(
                is_valid=False,
                can_import=False,
                row_count=len(raw_df),
                detected_columns=detected_cols,
                suggested_mapping=effective_mapping,
                issues=issues,
                preview_rows=raw_df.head(10).fillna("").to_dict(orient="records"),
            ),
            None,
        )

    # Normalize dates to first of month (YYYY-MM-01)
    norm["dt"] = parsed_dates.dt.to_period("M").dt.to_timestamp()
    norm["period_start"] = norm["dt"].dt.strftime("%Y-%m-%d")

    if "provision" in effective_mapping:
        norm["provision"] = raw_df[effective_mapping["provision"]].fillna("College-Wide").astype(str).str.strip()
    else:
        norm["provision"] = "College-Wide"
        issues.append(
            DataValidationIssue(
                severity="info",
                code="DEFAULT_PROVISION_APPLIED",
                message="No 'provision' column mapped; records will be imported as 'College-Wide'.",
            )
        )

    if "safeguarding_category" in effective_mapping:
        norm["safeguarding_category"] = (
            raw_df[effective_mapping["safeguarding_category"]].fillna("All Recorded Concerns").astype(str).str.strip()
        )
    else:
        norm["safeguarding_category"] = "All Recorded Concerns"
        issues.append(
            DataValidationIssue(
                severity="info",
                code="DEFAULT_CATEGORY_APPLIED",
                message="No 'safeguarding_category' column mapped; records will be imported under 'All Recorded Concerns'.",
            )
        )

    # Concern count validation
    raw_counts = pd.to_numeric(raw_df[effective_mapping["concern_count"]], errors="coerce")
    if raw_counts.isna().any():
        issues.append(
            DataValidationIssue(
                severity="error",
                code="NON_NUMERIC_CONCERN_COUNT",
                message=f"{int(raw_counts.isna().sum())} row(s) have non-numeric values in 'concern_count'.",
            )
        )
    if (raw_counts < 0).any():
        issues.append(
            DataValidationIssue(
                severity="error",
                code="NEGATIVE_CONCERN_COUNT",
                message="Safeguarding concern counts must be non-negative integers (>= 0).",
            )
        )

    norm["concern_count"] = raw_counts.fillna(0).round().astype(int)

    # Enrolled students
    if "enrolled_students" in effective_mapping:
        enrol = pd.to_numeric(raw_df[effective_mapping["enrolled_students"]], errors="coerce")
        norm["enrolled_students"] = enrol.fillna(1000).clip(lower=1).round().astype(int)
    else:
        norm["enrolled_students"] = 2500
        issues.append(
            DataValidationIssue(
                severity="warning",
                code="DEFAULT_ENROLMENT_USED",
                message="No 'enrolled_students' column mapped; using placeholder headcount (2,500 per provision) for per-1,000 calculations.",
            )
        )

    # Teaching days
    if "teaching_days" in effective_mapping:
        td = pd.to_numeric(raw_df[effective_mapping["teaching_days"]], errors="coerce")
        norm["teaching_days"] = td.fillna(norm["dt"].dt.month.map(BASE_FE_TEACHING_DAYS)).clip(lower=0, upper=31).round().astype(int)
    else:
        norm["teaching_days"] = norm["dt"].dt.month.map(BASE_FE_TEACHING_DAYS).astype(int)
        issues.append(
            DataValidationIssue(
                severity="info",
                code="STANDARD_FE_TEACHING_DAYS_APPLIED",
                message="No 'teaching_days' column mapped; standard UK FE monthly teaching days calendar applied.",
            )
        )

    # Attendance rate (optional)
    has_attendance = "attendance_rate" in effective_mapping
    if has_attendance:
        att = pd.to_numeric(raw_df[effective_mapping["attendance_rate"]], errors="coerce")
        # Convert 0-1 scale to 0-100 if needed
        if att.dropna().max() <= 1.05:
            att = att * 100.0
        norm["attendance_rate"] = att.clip(lower=0.0, upper=100.0).round(1)
    else:
        norm["attendance_rate"] = np.nan

    # Reporting change flag (optional)
    has_rep_flag = "reporting_change_flag" in effective_mapping
    if has_rep_flag:
        rf = pd.to_numeric(raw_df[effective_mapping["reporting_change_flag"]], errors="coerce").fillna(0)
        norm["reporting_change_flag"] = (rf > 0).astype(int)
    else:
        norm["reporting_change_flag"] = 0

    # Check for duplicate (period_start, provision, safeguarding_category) rows
    dup_mask = norm.duplicated(subset=["period_start", "provision", "safeguarding_category"], keep=False)
    duplicate_records = int(dup_mask.sum())
    if duplicate_records > 0:
        issues.append(
            DataValidationIssue(
                severity="warning",
                code="DUPLICATE_RECORDS_CONSOLIDATED",
                message=(
                    f"Detected {duplicate_records} duplicate rows sharing the same (period_start, provision, safeguarding_category). "
                    "Duplicate rows will be consolidated by summing concern_count and taking the mean enrolment/attendance."
                ),
            )
        )
        norm = (
            norm.groupby(["period_start", "dt", "provision", "safeguarding_category"], as_index=False)
            .agg(
                concern_count=("concern_count", "sum"),
                enrolled_students=("enrolled_students", "mean"),
                teaching_days=("teaching_days", "first"),
                attendance_rate=("attendance_rate", "mean"),
                reporting_change_flag=("reporting_change_flag", "max"),
            )
        )
        norm["enrolled_students"] = norm["enrolled_students"].round().astype(int)

    # Check for overlapping 'Total'/'All' category rows alongside granular categories
    unique_cats = set(norm["safeguarding_category"].unique())
    total_aliases = {"total", "all", "all categories", "college total"}
    overlapping_totals = [c for c in unique_cats if c.strip().lower() in total_aliases]
    if len(unique_cats) > 1 and overlapping_totals:
        issues.append(
            DataValidationIssue(
                severity="warning",
                code="AUTHORITATIVE_TOTAL_SEPARATED",
                message=(
                    f"Detected aggregate category label(s) {overlapping_totals} alongside specific categories. "
                    "To prevent double-counting concerns, summary category rows are excluded when summing granular categories."
                ),
            )
        )
        norm = norm[~norm["safeguarding_category"].str.strip().str.lower().isin(total_aliases)].copy()

    # Check date range and missing months
    min_dt = norm["dt"].min()
    max_dt = norm["dt"].max()
    expected_range = pd.date_range(start=min_dt, end=max_dt, freq="MS")
    actual_months = set(norm["period_start"].unique())
    missing_months = [d.strftime("%Y-%m-%d") for d in expected_range if d.strftime("%Y-%m-%d") not in actual_months]

    if missing_months:
        issues.append(
            DataValidationIssue(
                severity="warning",
                code="MISSING_MONTHS_DETECTED",
                message=(
                    f"{len(missing_months)} missing month(s) detected between {min_dt.strftime('%Y-%m-%d')} "
                    f"and {max_dt.strftime('%Y-%m-%d')}: {', '.join(missing_months[:6])}"
                    + ("..." if len(missing_months) > 6 else "")
                    + ". Missing months will be zero-filled with interpolated exposure during time-series construction."
                ),
                affected_rows_or_fields=missing_months,
            )
        )

    total_months = len(actual_months)
    full_academic_years = total_months // 12

    seasonality_warning = None
    if total_months < 36:
        seasonality_warning = (
            f"Dataset contains {total_months} months ({full_academic_years} full annual cycles). "
            "Because fewer than three full annual seasonal cycles (36 months) are present, reliable annual "
            "seasonality estimation and 12-month rolling backtesting may be limited."
        )
        issues.append(
            DataValidationIssue(
                severity="warning",
                code="LIMITED_SEASONAL_CYCLES",
                message=seasonality_warning,
            )
        )

    norm["academic_year"] = norm["dt"].apply(
        lambda d: get_academic_year_label(d.year, d.month, academic_year_start_month)
    )
    norm["academic_term"] = norm["dt"].apply(lambda d: get_academic_term(d.month))
    norm = norm.sort_values(["period_start", "provision", "safeguarding_category"]).reset_index(drop=True)

    is_valid = not any(i.severity == "error" for i in issues)
    can_import = is_valid and total_months >= 12
    if total_months < 12:
        issues.append(
            DataValidationIssue(
                severity="error",
                code="INSUFFICIENT_HISTORY",
                message=f"At least 12 months of historical data are required for forecasting (found {total_months}).",
            )
        )
        is_valid = False
        can_import = False

    preview_cols = [
        "period_start",
        "academic_year",
        "provision",
        "safeguarding_category",
        "concern_count",
        "enrolled_students",
        "teaching_days",
        "attendance_rate",
        "reporting_change_flag",
    ]
    preview_rows = norm[preview_cols].head(12).fillna("").to_dict(orient="records")

    report = DataValidationReport(
        is_valid=is_valid,
        can_import=can_import,
        row_count=len(norm),
        date_range_start=min_dt.strftime("%Y-%m-%d"),
        date_range_end=max_dt.strftime("%Y-%m-%d"),
        total_months=total_months,
        full_academic_years=full_academic_years,
        missing_months=missing_months,
        duplicate_records=duplicate_records,
        rejected_personal_data_columns=[],
        detected_columns=detected_cols,
        suggested_mapping=effective_mapping,
        provisions_found=sorted(norm["provision"].unique().tolist()),
        categories_found=sorted(norm["safeguarding_category"].unique().tolist()),
        total_concerns=int(norm["concern_count"].sum()),
        has_attendance_rate=bool(has_attendance and norm["attendance_rate"].notna().any()),
        has_reporting_change_flag=bool(has_rep_flag),
        seasonality_sufficiency_warning=seasonality_warning,
        issues=issues,
        preview_rows=preview_rows,
    )

    return report, norm if can_import else None

