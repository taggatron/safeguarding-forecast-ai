"""Pydantic schemas for Safeguarding Forecast AI."""
from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    application: str
    subtitle: str
    organisation: str
    active_dataset_id: str
    is_synthetic: bool
    governance_banner: str
    academic_year_start_month: int
    timestamp: str


class SyntheticGeneratorConfig(BaseModel):
    seed: int = Field(default=42, ge=1, le=999999, description="Fixed random seed for reproducibility")
    num_academic_years: int = Field(default=6, ge=3, le=10, description="Number of full academic years to generate")
    start_academic_year: int = Field(default=2020, ge=2015, le=2025, description="Starting calendar year of first academic year")
    academic_year_start_month: int = Field(default=9, ge=1, le=12, description="Start month of academic year (default 9 = September)")
    provisions: List[str] = Field(
        default=["EPYP", "Adult", "Apprenticeships"],
        description="Provision types to simulate"
    )
    categories: List[str] = Field(
        default=[
            "Mental health and wellbeing concerns",
            "Abuse or neglect",
            "Exploitation",
            "Online safety",
            "Other recorded concerns",
        ],
        description="Broad safeguarding categories to simulate"
    )
    include_reporting_change: bool = Field(
        default=True,
        description="Simulate a known reporting system/training change in Sep 2023"
    )
    base_annual_volume: int = Field(
        default=1450,
        ge=300,
        le=10000,
        description="Approximate initial annual college-wide concern volume"
    )


class DatasetRecordPreview(BaseModel):
    period_start: str
    academic_year: str
    academic_term: str
    provision: str
    safeguarding_category: str
    concern_count: int
    enrolled_students: int
    teaching_days: int
    attendance_rate: Optional[float] = None
    reporting_change_flag: int = 0


class DataValidationIssue(BaseModel):
    severity: Literal["error", "warning", "info"]
    code: str
    message: str
    affected_rows_or_fields: Optional[List[str]] = None


class DataValidationReport(BaseModel):
    is_valid: bool
    can_import: bool
    row_count: int
    date_range_start: Optional[str] = None
    date_range_end: Optional[str] = None
    total_months: int = 0
    full_academic_years: int = 0
    missing_months: List[str] = []
    duplicate_records: int = 0
    rejected_personal_data_columns: List[str] = []
    detected_columns: List[str] = []
    suggested_mapping: Dict[str, str] = {}
    provisions_found: List[str] = []
    categories_found: List[str] = []
    total_concerns: int = 0
    has_attendance_rate: bool = False
    has_reporting_change_flag: bool = False
    seasonality_sufficiency_warning: Optional[str] = None
    issues: List[DataValidationIssue] = []
    preview_rows: List[Dict[str, Any]] = []


class DatasetSummaryResponse(BaseModel):
    dataset_id: str
    name: str
    is_synthetic: bool
    governance_banner: str
    created_at: str
    seed: Optional[int] = None
    academic_year_start_month: int = 9
    date_range_start: str
    date_range_end: str
    total_months: int
    full_academic_years: int
    academic_years: List[str]
    provisions: List[str]
    categories: List[str]
    total_recorded_concerns: int
    latest_enrolled_students: int
    has_attendance_data: bool
    has_reporting_change_flag: bool
    missing_months: List[str]
    small_group_suppression_enabled: bool
    available_datasets: List[Dict[str, Any]]


class MonthlyHistoricalPoint(BaseModel):
    period_start: str
    period_label: str
    academic_year: str
    academic_term: str
    month_num: int
    concern_count: int
    enrolled_students: int
    concerns_per_1000: float
    teaching_days: int
    attendance_rate: Optional[float] = None
    reporting_change_flag: int = 0
    rolling_3m_avg: Optional[float] = None
    rolling_6m_avg: Optional[float] = None


class AcademicYearComparisonItem(BaseModel):
    academic_year: str
    total_concerns: int
    avg_enrolled_students: int
    concerns_per_1000: float
    peak_month: str
    peak_month_count: int
    total_teaching_days: int
    avg_attendance_rate: Optional[float] = None
    yoy_change_pct: Optional[float] = None


class ProvisionSummaryItem(BaseModel):
    provision: str
    total_concerns: int
    share_pct: float
    avg_enrolled_students: int
    concerns_per_1000: float


class CategorySummaryItem(BaseModel):
    safeguarding_category: str
    total_concerns: int
    share_pct: float
    suppressed: bool = False


class HistoricalStatsResponse(BaseModel):
    dataset_id: str
    is_synthetic: bool
    governance_banner: str
    selected_academic_year: str
    selected_provision: str
    selected_category: str
    monthly_series: List[MonthlyHistoricalPoint]
    academic_year_comparisons: List[AcademicYearComparisonItem]
    provision_breakdown: List[ProvisionSummaryItem]
    category_breakdown: List[CategorySummaryItem]
    reporting_change_dates: List[str]


class ScenarioAssumptions(BaseModel):
    enrolment_growth_pct: float = Field(default=0.0, ge=-30.0, le=50.0, description="Percentage change in future student enrolment")
    teaching_days_delta: int = Field(default=0, ge=-5, le=5, description="Monthly adjustment to scheduled teaching days")
    attendance_rate_delta_pp: float = Field(default=0.0, ge=-15.0, le=10.0, description="Percentage point change in future attendance rate")
    reporting_change_assumed: Optional[int] = Field(default=None, description="Override reporting_change_flag in forecast horizon (0 or 1)")


class ForecastRequest(BaseModel):
    model_id: str = Field(default="lightgbm_mlforecast", description="Model identifier: seasonal_naive, statistical_count, lightgbm_mlforecast, timesfm_3")
    horizon: int = Field(default=12, description="Forecast horizon in months: 3, 6, or 12")
    provision: str = Field(default="All", description="'All' for college-wide or specific provision name")
    safeguarding_category: str = Field(default="All", description="'All' for all categories or specific category name")
    prediction_interval: int = Field(default=80, description="Prediction interval level: 80 or 95")
    academic_year_filter: str = Field(default="All", description="Historical display filter or 'All'")
    timesfm_multivariate: bool = Field(default=True, description="Whether TimesFM 3.0 uses covariates/multivariate mode when selected")
    scenario: Optional[ScenarioAssumptions] = None


class MonthlyForecastPoint(BaseModel):
    period_start: str
    period_label: str
    academic_year: str
    academic_term: str
    month_num: int
    point_forecast: int
    raw_point_forecast: float
    lower_80: int
    upper_80: int
    lower_95: Optional[int] = None
    upper_95: Optional[int] = None
    enrolled_students_assumed: int
    teaching_days_assumed: int
    attendance_rate_assumed: Optional[float] = None
    concerns_per_1000_forecast: float
    lower_per_1000: float
    upper_per_1000: float


class ProvisionForecastItem(BaseModel):
    provision: str
    predicted_total: int
    lower_bound: int
    upper_bound: int
    share_pct: float
    concerns_per_1000: float
    historical_prior_year: int
    yoy_change_pct: float


class CategoryForecastItem(BaseModel):
    safeguarding_category: str
    predicted_total: int
    lower_bound: int
    upper_bound: int
    share_pct: float
    historical_prior_year: int
    yoy_change_pct: float
    reliable_estimation: bool = True
    note: Optional[str] = None


class AnnualForecastSummary(BaseModel):
    target_academic_year: str
    horizon_months: int
    predicted_total_concerns: int
    lower_bound: int
    upper_bound: int
    interval_level: int
    lower_80: int
    upper_80: int
    lower_95: Optional[int] = None
    upper_95: Optional[int] = None
    calibration_method: str
    prior_year_total: int
    yoy_change_count: int
    yoy_change_pct: float
    peak_demand_month: str
    peak_demand_count: int
    peak_demand_period_description: str
    predicted_concerns_per_1000: float
    enrolled_students_assumed: int


class ScenarioSensitivityCapabilities(BaseModel):
    supports_enrolment_growth: bool
    enrolment_explanation: str
    supports_teaching_days: bool
    teaching_days_explanation: str
    supports_attendance_scenario: bool
    attendance_explanation: str
    supports_reporting_flag: bool


class ForecastResponse(BaseModel):
    run_id: str
    dataset_id: str
    is_synthetic: bool
    governance_banner: str
    model_id: str
    model_name: str
    model_version: str
    runtime_ms: float
    horizon: int
    provision: str
    safeguarding_category: str
    prediction_interval: int
    annual_summary: AnnualForecastSummary
    historical_series: List[MonthlyHistoricalPoint]
    monthly_forecasts: List[MonthlyForecastPoint]
    baseline_monthly_forecasts: Optional[List[MonthlyForecastPoint]] = None
    baseline_annual_summary: Optional[AnnualForecastSummary] = None
    provision_forecasts: List[ProvisionForecastItem]
    category_forecasts: List[CategoryForecastItem]
    planning_insights: List[str]
    model_accuracy_summary: Dict[str, Any]
    scenario_capabilities: ScenarioSensitivityCapabilities
    warnings: List[str] = []


class ModelAvailabilityItem(BaseModel):
    model_id: str
    name: str
    short_name: str
    family: str
    version: str
    available: bool
    status_label: str
    status_explanation: str
    license_name: str
    commercial_production_permitted: bool
    supports_covariates: bool
    supports_95_pi: bool
    data_requirements: str
    supported_horizons: List[int]
    backend_engine: str


class ModelsAvailabilityResponse(BaseModel):
    governance_mode: str
    timesfm_license_acknowledged: bool
    models: List[ModelAvailabilityItem]
    extensible_interfaces: List[str]


class ModelBacktestMetrics(BaseModel):
    model_id: str
    model_name: str
    family: str
    available: bool
    status_label: str
    mae: Optional[float] = None
    rmse: Optional[float] = None
    wape_pct: Optional[float] = None
    mase: Optional[float] = None
    pi_80_coverage_pct: Optional[float] = None
    pi_95_coverage_pct: Optional[float] = None
    annual_total_mae: Optional[float] = None
    outperforms_baseline: Optional[bool] = None
    improvement_vs_baseline_pct: Optional[float] = None
    num_rolling_origins: int = 0
    total_evaluated_months: int = 0
    avg_runtime_ms: Optional[float] = None
    horizon_evaluated: int = 12
    data_requirements: str
    notes: str


class BacktestComparisonPoint(BaseModel):
    period_start: str
    period_label: str
    academic_year: str
    fold_origin: str
    actual_count: int
    predictions: Dict[str, Optional[int]]
    lower_80_bounds: Dict[str, Optional[int]]
    upper_80_bounds: Dict[str, Optional[int]]


class BacktestResponse(BaseModel):
    dataset_id: str
    is_synthetic: bool
    governance_banner: str
    provision: str
    safeguarding_category: str
    horizon: int
    best_model_id: str
    best_model_name: str
    best_model_rationale: str
    seasonality_warning: Optional[str] = None
    metrics: List[ModelBacktestMetrics]
    historical_backtest_series: List[BacktestComparisonPoint]
    methodology_explanation: str


class FeatureImportanceItem(BaseModel):
    feature_key: str
    feature_label: str
    category: str
    importance_pct: float
    raw_importance: float
    direction_hint: str
    explanation: str


class SeasonalMonthProfile(BaseModel):
    month_num: int
    month_name: str
    academic_term: str
    avg_concerns: float
    min_concerns: int
    max_concerns: int
    seasonal_index: float
    avg_teaching_days: float
    avg_concerns_per_teaching_day: float


class TrendsInsightsResponse(BaseModel):
    dataset_id: str
    is_synthetic: bool
    governance_banner: str
    monthly_series: List[MonthlyHistoricalPoint]
    seasonal_profile: List[SeasonalMonthProfile]
    academic_year_comparisons: List[AcademicYearComparisonItem]
    provision_trends: List[Dict[str, Any]]
    category_trends: List[Dict[str, Any]]
    feature_importances: List[FeatureImportanceItem]
    causation_disclaimer: str
    reporting_practice_note: str
    key_findings: List[str]

