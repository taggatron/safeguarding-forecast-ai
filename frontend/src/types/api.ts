export interface DatasetSummaryResponse {
  dataset_id: string;
  name: string;
  is_synthetic: boolean;
  governance_banner: string;
  created_at: string;
  seed?: number | null;
  academic_year_start_month: number;
  date_range_start: string;
  date_range_end: string;
  total_months: number;
  full_academic_years: number;
  academic_years: string[];
  provisions: string[];
  categories: string[];
  total_recorded_concerns: number;
  latest_enrolled_students: number;
  has_attendance_data: boolean;
  has_reporting_change_flag: boolean;
  missing_months: string[];
  small_group_suppression_enabled: boolean;
  available_datasets: {
    dataset_id: string;
    name: string;
    is_synthetic: boolean;
    seed?: number | null;
    academic_year_start_month: number;
    created_at: string;
    row_count: number;
    total_concerns: number;
  }[];
}

export interface MonthlyHistoricalPoint {
  period_start: string;
  period_label: string;
  academic_year: string;
  academic_term: string;
  month_num: number;
  concern_count: number;
  enrolled_students: number;
  concerns_per_1000: number;
  teaching_days: number;
  attendance_rate?: number | null;
  reporting_change_flag: number;
  rolling_3m_avg?: number | null;
  rolling_6m_avg?: number | null;
}

export interface MonthlyForecastPoint {
  period_start: string;
  period_label: string;
  academic_year: string;
  academic_term: string;
  month_num: number;
  point_forecast: number;
  raw_point_forecast: number;
  lower_80: number;
  upper_80: number;
  lower_95?: number | null;
  upper_95?: number | null;
  enrolled_students_assumed: number;
  teaching_days_assumed: number;
  attendance_rate_assumed?: number | null;
  concerns_per_1000_forecast: number;
  lower_per_1000: number;
  upper_per_1000: number;
}

export interface AnnualForecastSummary {
  target_academic_year: string;
  horizon_months: number;
  predicted_total_concerns: number;
  lower_bound: number;
  upper_bound: number;
  interval_level: number;
  lower_80: number;
  upper_80: number;
  lower_95?: number | null;
  upper_95?: number | null;
  calibration_method: string;
  prior_year_total: number;
  yoy_change_count: number;
  yoy_change_pct: number;
  peak_demand_month: string;
  peak_demand_count: number;
  peak_demand_period_description: string;
  predicted_concerns_per_1000: number;
  enrolled_students_assumed: number;
}

export interface ProvisionForecastItem {
  provision: string;
  predicted_total: number;
  lower_bound: number;
  upper_bound: number;
  share_pct: number;
  concerns_per_1000: number;
  historical_prior_year: number;
  yoy_change_pct: number;
}

export interface CategoryForecastItem {
  safeguarding_category: string;
  predicted_total: number;
  lower_bound: number;
  upper_bound: number;
  share_pct: number;
  historical_prior_year: number;
  yoy_change_pct: number;
  reliable_estimation: boolean;
  note?: string | null;
}

export interface ScenarioSensitivityCapabilities {
  supports_enrolment_growth: boolean;
  enrolment_explanation: string;
  supports_teaching_days: boolean;
  teaching_days_explanation: string;
  supports_attendance_scenario: boolean;
  attendance_explanation: string;
  supports_reporting_flag: boolean;
}

export interface ScenarioAssumptions {
  enrolment_growth_pct: number;
  teaching_days_delta: number;
  attendance_rate_delta_pp: number;
  reporting_change_assumed?: number | null;
}

export interface ForecastRequest {
  model_id: string;
  horizon: number;
  provision: string;
  safeguarding_category: string;
  prediction_interval: number;
  academic_year_filter?: string;
  timesfm_multivariate?: boolean;
  scenario?: ScenarioAssumptions;
}

export interface ForecastResponse {
  run_id: string;
  dataset_id: string;
  is_synthetic: boolean;
  governance_banner: string;
  model_id: string;
  model_name: string;
  model_version: string;
  runtime_ms: number;
  horizon: number;
  provision: string;
  safeguarding_category: string;
  prediction_interval: number;
  annual_summary: AnnualForecastSummary;
  historical_series: MonthlyHistoricalPoint[];
  monthly_forecasts: MonthlyForecastPoint[];
  baseline_monthly_forecasts?: MonthlyForecastPoint[];
  baseline_annual_summary?: AnnualForecastSummary;
  provision_forecasts: ProvisionForecastItem[];
  category_forecasts: CategoryForecastItem[];
  planning_insights: string[];
  model_accuracy_summary: {
    mae?: number | null;
    wape_pct?: number | null;
    pi_80_coverage_pct?: number | null;
    outperforms_baseline?: boolean | null;
    improvement_vs_baseline_pct?: number | null;
    best_model_id: string;
    best_model_name: string;
    explanation: string;
  };
  scenario_capabilities: ScenarioSensitivityCapabilities;
  warnings: string[];
}

export interface AcademicYearComparisonItem {
  academic_year: string;
  total_concerns: number;
  avg_enrolled_students: number;
  concerns_per_1000: number;
  peak_month: string;
  peak_month_count: number;
  total_teaching_days: number;
  avg_attendance_rate?: number | null;
  yoy_change_pct?: number | null;
}

export interface HistoricalStatsResponse {
  dataset_id: string;
  is_synthetic: boolean;
  governance_banner: string;
  selected_academic_year: string;
  selected_provision: string;
  selected_category: string;
  monthly_series: MonthlyHistoricalPoint[];
  academic_year_comparisons: AcademicYearComparisonItem[];
  provision_breakdown: {
    provision: string;
    total_concerns: number;
    share_pct: number;
    avg_enrolled_students: number;
    concerns_per_1000: number;
  }[];
  category_breakdown: {
    safeguarding_category: string;
    total_concerns: number;
    share_pct: number;
    suppressed: boolean;
  }[];
  reporting_change_dates: string[];
}

export interface ModelAvailabilityItem {
  model_id: string;
  name: string;
  short_name: string;
  family: string;
  version: string;
  available: boolean;
  status_label: string;
  status_explanation: string;
  license_name: string;
  commercial_production_permitted: boolean;
  supports_covariates: boolean;
  supports_95_pi: boolean;
  data_requirements: string;
  supported_horizons: number[];
  backend_engine: string;
}

export interface ModelsAvailabilityResponse {
  governance_mode: string;
  timesfm_license_acknowledged: boolean;
  models: ModelAvailabilityItem[];
  extensible_interfaces: string[];
}

export interface ModelBacktestMetrics {
  model_id: string;
  model_name: string;
  family: string;
  available: boolean;
  status_label: string;
  mae?: number | null;
  rmse?: number | null;
  wape_pct?: number | null;
  mase?: number | null;
  pi_80_coverage_pct?: number | null;
  pi_95_coverage_pct?: number | null;
  annual_total_mae?: number | null;
  outperforms_baseline?: boolean | null;
  improvement_vs_baseline_pct?: number | null;
  num_rolling_origins: number;
  total_evaluated_months: number;
  avg_runtime_ms?: number | null;
  horizon_evaluated: number;
  data_requirements: string;
  notes: string;
}

export interface BacktestComparisonPoint {
  period_start: string;
  period_label: string;
  academic_year: string;
  fold_origin: string;
  actual_count: number;
  predictions: Record<string, number | null>;
  lower_80_bounds: Record<string, number | null>;
  upper_80_bounds: Record<string, number | null>;
}

export interface BacktestResponse {
  dataset_id: string;
  is_synthetic: boolean;
  governance_banner: string;
  provision: string;
  safeguarding_category: string;
  horizon: number;
  best_model_id: string;
  best_model_name: string;
  best_model_rationale: string;
  seasonality_warning?: string | null;
  metrics: ModelBacktestMetrics[];
  historical_backtest_series: BacktestComparisonPoint[];
  methodology_explanation: string;
}

export interface FeatureImportanceItem {
  feature_key: string;
  feature_label: string;
  category: string;
  importance_pct: number;
  raw_importance: number;
  direction_hint: string;
  explanation: string;
}

export interface SeasonalMonthProfile {
  month_num: number;
  month_name: string;
  academic_term: string;
  avg_concerns: number;
  min_concerns: number;
  max_concerns: number;
  seasonal_index: number;
  avg_teaching_days: number;
  avg_concerns_per_teaching_day: number;
}

export interface TrendsInsightsResponse {
  dataset_id: string;
  is_synthetic: boolean;
  governance_banner: string;
  monthly_series: MonthlyHistoricalPoint[];
  seasonal_profile: SeasonalMonthProfile[];
  academic_year_comparisons: AcademicYearComparisonItem[];
  provision_trends: Record<string, any>[];
  category_trends: Record<string, any>[];
  feature_importances: FeatureImportanceItem[];
  causation_disclaimer: string;
  reporting_practice_note: string;
  key_findings: string[];
}

export interface DataValidationReport {
  is_valid: boolean;
  can_import: boolean;
  row_count: number;
  date_range_start?: string | null;
  date_range_end?: string | null;
  total_months: number;
  full_academic_years: number;
  missing_months: string[];
  duplicate_records: number;
  rejected_personal_data_columns: string[];
  detected_columns: string[];
  suggested_mapping: Record<string, string>;
  provisions_found: string[];
  categories_found: string[];
  total_concerns: number;
  has_attendance_rate: boolean;
  has_reporting_change_flag: boolean;
  seasonality_sufficiency_warning?: string | null;
  issues: {
    severity: 'error' | 'warning' | 'info';
    code: string;
    message: string;
    affected_rows_or_fields?: string[] | null;
  }[];
  preview_rows: Record<string, any>[];
}

export interface EsotericModuleCatalogItem {
  id: string;
  short_name: string;
  name: string;
  family: string;
  role: string;
}

export interface EsotericStrategyItem {
  id: string;
  name: string;
  short_name: string;
  default_modules: string[];
  description: string;
}

export interface EsotericAdmittedFeature {
  feature: string;
  module_id: string;
  module_name: string;
  weight: number;
  direction: string;
  description: string;
}

export interface EsotericHistoricalPoint {
  period_start: string;
  period_label: string;
  academic_year: string;
  academic_term: string;
  actual_count: number;
  enrolled_students: number;
  teaching_days: number;
  rate_per_1000: number;
}

export interface EsotericForecastPoint {
  step: number;
  period_start: string;
  period_label: string;
  academic_year: string;
  academic_term: string;
  baseline_forecast: number;
  baseline_lower: number;
  baseline_upper: number;
  esoteric_forecast: number;
  esoteric_raw: number;
  lower_bound: number;
  upper_bound: number;
  lower_80: number;
  upper_80: number;
  lower_95: number;
  upper_95: number;
  delta_count: number;
  delta_pct: number;
  enrolled_students: number;
  teaching_days: number;
  rate_per_1000: number;
  rough_levy_area: number;
  rough_sig_depth3: number;
  hilbert_phase_deg: number;
  hawkes_intensity: number;
  dispersion_r: number;
  dominant_driver: string;
}

export interface EsotericForecastRequest {
  cadence: 'monthly' | 'weekly';
  strategy_id: string;
  base_model_id: string;
  horizon: number;
  provision: string;
  safeguarding_category: string;
  prediction_interval: number;
  custom_modules?: string[] | null;
  scenario?: ScenarioAssumptions;
}

export interface EsotericForecastResponse {
  cadence: 'monthly' | 'weekly';
  period_unit: 'months' | 'weeks';
  dataset_id: string;
  is_synthetic: boolean;
  governance_banner: string;
  strategy: EsotericStrategyItem;
  base_model_id: string;
  base_model_name: string;
  baseline_sha256: string;
  active_modules: string[];
  module_catalog: EsotericModuleCatalogItem[];
  strategies_catalog: EsotericStrategyItem[];
  horizon: number;
  provision: string;
  safeguarding_category: string;
  prediction_interval: number;
  runtime_ms: number;
  summary: {
    esoteric_total: number;
    esoteric_lower: number;
    esoteric_upper: number;
    baseline_total: number;
    baseline_lower: number;
    baseline_upper: number;
    prior_horizon_total: number;
    yoy_change_pct: number;
    net_esoteric_delta: number;
    net_esoteric_delta_pct: number;
    peak_period_label: string;
    peak_period_count: number;
    calibration_method: string;
  };
  conditioning_diagnostics: {
    funnel_type: string;
    n_train_steps: number;
    n_candidate_esoteric_features: number;
    n_admitted_esoteric_features: number;
    n_quarantined_esoteric_features: number;
    max_vif_after: number;
    condition_number_after: number;
    base_nb_dispersion_r: number;
    identity_floor_active: boolean;
    admitted_features: EsotericAdmittedFeature[];
  };
  walk_forward_backtest: {
    evaluated_periods: number;
    folds: number;
    baseline_mae: number;
    esoteric_mae: number;
    mae_delta_pct: number;
    baseline_rmse: number;
    esoteric_rmse: number;
    baseline_wape_pct: number;
    esoteric_wape_pct: number;
    baseline_surge_mae: number;
    esoteric_surge_mae: number;
    baseline_pi_coverage_pct: number;
    esoteric_pi_coverage_pct: number;
    points: {
      period_label: string;
      actual: number;
      baseline: number;
      esoteric: number;
    }[];
  };
  historical_series: EsotericHistoricalPoint[];
  forecast_series: EsotericForecastPoint[];
  mathematical_notes: string[];
}

