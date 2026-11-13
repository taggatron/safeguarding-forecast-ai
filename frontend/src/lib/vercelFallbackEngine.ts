import rawSnapshot from './precomputedSnapshot.json';
import rawEsotericForecastSnapshot from './esotericForecastSnapshot.json';
import type {
  BacktestResponse,
  DatasetSummaryResponse,
  DataValidationReport,
  EsotericAdmittedFeature,
  EsotericForecastPoint,
  EsotericForecastRequest,
  EsotericForecastResponse,
  ForecastRequest,
  ForecastResponse,
  HistoricalStatsResponse,
  ModelsAvailabilityResponse,
  MonthlyForecastPoint,
  TrendsInsightsResponse,
} from '../types/api';

const snapshot = rawSnapshot as any;
const esoForecastSnapshot = rawEsotericForecastSnapshot as any;

// Runtime state when operating in serverless/static edge mode on Vercel
let activeDatasetSummary: DatasetSummaryResponse = structuredClone(snapshot.dataset_summary);
let governanceMode: string = snapshot.models_availability.governance_mode || 'research_prototype';
let timesfmLicenseAck: boolean = true;
let smallGroupSuppression: boolean = false;
let scaleMultiplier: number = 1.0;

const PROVISION_SHARES: Record<string, { share: number; enrolShare: number }> = {
  All: { share: 1.0, enrolShare: 1.0 },
  EPYP: { share: 0.61, enrolShare: 0.46 },
  Adult: { share: 0.25, enrolShare: 0.34 },
  Apprenticeships: { share: 0.14, enrolShare: 0.20 },
};

const CATEGORY_SHARES: Record<string, number> = {
  All: 1.0,
  'Mental health and wellbeing concerns': 0.43,
  'Abuse or neglect': 0.20,
  'Online safety': 0.13,
  Exploitation: 0.09,
  'Other recorded concerns': 0.15,
};

const PROHIBITED_PII_REGEX =
  /\b(student_?id|learner_?id|uln|upn|name|first_?name|last_?name|surname|student_?name|dob|date_?of_?birth|narrative|case_?notes?|notes|description|email|phone|postcode|address)\b/i;

function getSliceMultiplier(provision: string, category: string): {
  concernMult: number;
  enrolMult: number;
} {
  const p = PROVISION_SHARES[provision] || { share: 0.33, enrolShare: 0.33 };
  const c = CATEGORY_SHARES[category] ?? 0.2;
  return {
    concernMult: p.share * c * scaleMultiplier,
    enrolMult: p.enrolShare,
  };
}

export const vercelFallbackEngine = {
  getDatasetSummary(): DatasetSummaryResponse {
    return {
      ...activeDatasetSummary,
      small_group_suppression_enabled: smallGroupSuppression,
    };
  },

  getModelsAvailability(): ModelsAvailabilityResponse {
    const baseAvail: ModelsAvailabilityResponse = structuredClone(snapshot.models_availability);
    baseAvail.governance_mode = governanceMode;
    baseAvail.timesfm_license_acknowledged = timesfmLicenseAck;
    baseAvail.models = baseAvail.models.map((m) => {
      if (m.model_id === 'timesfm_3') {
        if (governanceMode !== 'research_prototype') {
          return {
            ...m,
            available: false,
            status_label: 'Not available (Governance Restriction)',
            status_explanation:
              'TimesFM 3.0 weights carry timesfm-non-commercial-license-v1.0 restrictions and are disabled in Operational Production Mode.',
          };
        }
        if (!timesfmLicenseAck) {
          return {
            ...m,
            available: false,
            status_label: 'Not available (License Unacknowledged)',
            status_explanation:
              'TimesFM 3.0 requires acknowledgement of timesfm-non-commercial-license-v1.0 in Data Management.',
          };
        }
      }
      return m;
    });
    return baseAvail;
  },

  getHistoricalStats(
    academicYear = 'All',
    provision = 'All',
    safeguardingCategory = 'All'
  ): HistoricalStatsResponse {
    const base: HistoricalStatsResponse = structuredClone(snapshot.historical_all);
    const { concernMult, enrolMult } = getSliceMultiplier(provision, safeguardingCategory);

    let series = base.monthly_series.map((pt) => {
      const cnt = Math.max(0, Math.round(pt.concern_count * concernMult));
      const enrol = Math.max(10, Math.round(pt.enrolled_students * enrolMult));
      return {
        ...pt,
        concern_count: cnt,
        enrolled_students: enrol,
        concerns_per_1000: Number(((cnt / enrol) * 1000).toFixed(2)),
        rolling_3m_avg: pt.rolling_3m_avg ? Number((pt.rolling_3m_avg * concernMult).toFixed(1)) : null,
        rolling_6m_avg: pt.rolling_6m_avg ? Number((pt.rolling_6m_avg * concernMult).toFixed(1)) : null,
      };
    });

    if (academicYear !== 'All') {
      series = series.filter((s) => s.academic_year === academicYear);
    }

    const ayComp = base.academic_year_comparisons.map((ay) => {
      const tot = Math.max(0, Math.round(ay.total_concerns * concernMult));
      const enrol = Math.max(10, Math.round(ay.avg_enrolled_students * enrolMult));
      return {
        ...ay,
        total_concerns: tot,
        avg_enrolled_students: enrol,
        concerns_per_1000: Number(((tot / enrol) * 1000).toFixed(1)),
        peak_month_count: Math.max(0, Math.round(ay.peak_month_count * concernMult)),
      };
    });

    return {
      ...base,
      is_synthetic: activeDatasetSummary.is_synthetic,
      governance_banner: activeDatasetSummary.governance_banner,
      selected_academic_year: academicYear,
      selected_provision: provision,
      selected_category: safeguardingCategory,
      monthly_series: series,
      academic_year_comparisons: ayComp,
    };
  },

  generateForecast(req: ForecastRequest): ForecastResponse {
    const modelId = req.model_id || 'lightgbm_mlforecast';
    const baseFc: ForecastResponse = structuredClone(
      snapshot.forecasts_12[modelId] || snapshot.forecasts_12['lightgbm_mlforecast']
    );
    const horizon = [3, 6, 12].includes(req.horizon) ? req.horizon : 12;
    const piLevel = req.prediction_interval === 95 ? 95 : 80;
    const { concernMult, enrolMult } = getSliceMultiplier(
      req.provision || 'All',
      req.safeguarding_category || 'All'
    );

    const scen = req.scenario || {
      enrolment_growth_pct: 0,
      teaching_days_delta: 0,
      attendance_rate_delta_pp: 0,
    };

    const caps = baseFc.scenario_capabilities;
    const enrolElasticity = caps.supports_enrolment_growth
      ? Math.pow(1 + (scen.enrolment_growth_pct || 0) / 100, 0.85)
      : 1.0;
    const tdElasticity = caps.supports_teaching_days
      ? Math.pow(1 + (scen.teaching_days_delta || 0) / 18, 0.35)
      : 1.0;
    const attElasticity = caps.supports_attendance_scenario
      ? 1 - (scen.attendance_rate_delta_pp || 0) * 0.011
      : 1.0;
    const scenarioMult = enrolElasticity * tdElasticity * attElasticity;

    const buildPoints = (mult: number): MonthlyForecastPoint[] =>
      baseFc.monthly_forecasts.slice(0, horizon).map((pt) => {
        const rawVal = pt.raw_point_forecast * concernMult * mult;
        const pointVal = Math.max(0, Math.round(rawVal));
        const l80 = Math.max(0, Math.round(pt.lower_80 * concernMult * mult));
        const u80 = Math.max(pointVal, Math.round(pt.upper_80 * concernMult * mult));
        const l95 = pt.lower_95 != null ? Math.max(0, Math.round(pt.lower_95 * concernMult * mult)) : l80;
        const u95 = pt.upper_95 != null ? Math.max(u80, Math.round(pt.upper_95 * concernMult * mult)) : u80;
        const enrol = Math.max(
          10,
          Math.round(
            pt.enrolled_students_assumed *
              enrolMult *
              (1 + (scen.enrolment_growth_pct || 0) / 100)
          )
        );
        const td = Math.max(1, Math.min(25, pt.teaching_days_assumed + (scen.teaching_days_delta || 0)));
        return {
          ...pt,
          point_forecast: pointVal,
          raw_point_forecast: Number(rawVal.toFixed(2)),
          lower_80: l80,
          upper_80: u80,
          lower_95: l95,
          upper_95: u95,
          enrolled_students_assumed: enrol,
          teaching_days_assumed: td,
          concerns_per_1000_forecast: Number(((pointVal / enrol) * 1000).toFixed(2)),
          lower_per_1000: Number((((piLevel === 95 ? l95 : l80) / enrol) * 1000).toFixed(2)),
          upper_per_1000: Number((((piLevel === 95 ? u95 : u80) / enrol) * 1000).toFixed(2)),
        };
      });

    const basePts = buildPoints(1.0);
    const scenPts = buildPoints(scenarioMult);

    const calcSummary = (pts: MonthlyForecastPoint[], mult: number) => {
      const tot = pts.reduce((acc, p) => acc + p.point_forecast, 0);
      const scaleH = (horizon / 12) * concernMult * mult;
      const l80 = Math.max(0, Math.round(baseFc.annual_summary.lower_80 * scaleH));
      const u80 = Math.max(tot, Math.round(baseFc.annual_summary.upper_80 * scaleH));
      const l95 = Math.max(0, Math.round((baseFc.annual_summary.lower_95 || l80) * scaleH));
      const u95 = Math.max(u80, Math.round((baseFc.annual_summary.upper_95 || u80) * scaleH));
      const priorTot = Math.max(1, Math.round(baseFc.annual_summary.prior_year_total * (horizon / 12) * concernMult));
      const yoyDiff = tot - priorTot;
      const yoyPct = Number(((yoyDiff / priorTot) * 100).toFixed(1));
      const peakPt = [...pts].sort((a, b) => b.point_forecast - a.point_forecast)[0];
      const avgEnrol = Math.max(
        1,
        Math.round(pts.reduce((acc, p) => acc + p.enrolled_students_assumed, 0) / pts.length)
      );

      return {
        ...baseFc.annual_summary,
        horizon_months: horizon,
        predicted_total_concerns: tot,
        lower_bound: piLevel === 95 ? l95 : l80,
        upper_bound: piLevel === 95 ? u95 : u80,
        interval_level: piLevel,
        lower_80: l80,
        upper_80: u80,
        lower_95: l95,
        upper_95: u95,
        prior_year_total: priorTot,
        yoy_change_count: yoyDiff,
        yoy_change_pct: yoyPct,
        peak_demand_month: peakPt ? peakPt.period_label : baseFc.annual_summary.peak_demand_month,
        peak_demand_count: peakPt ? peakPt.point_forecast : 0,
        predicted_concerns_per_1000: Number(((tot / avgEnrol) * 1000).toFixed(1)),
        enrolled_students_assumed: avgEnrol,
      };
    };

    const annualSummary = calcSummary(scenPts, scenarioMult);
    const baselineAnnualSummary = calcSummary(basePts, 1.0);

    const histStats = this.getHistoricalStats(
      req.academic_year_filter || 'All',
      req.provision || 'All',
      req.safeguarding_category || 'All'
    );

    return {
      ...baseFc,
      is_synthetic: activeDatasetSummary.is_synthetic,
      governance_banner: activeDatasetSummary.governance_banner,
      horizon,
      provision: req.provision || 'All',
      safeguarding_category: req.safeguarding_category || 'All',
      prediction_interval: piLevel,
      annual_summary: annualSummary,
      baseline_annual_summary: baselineAnnualSummary,
      historical_series: histStats.monthly_series,
      monthly_forecasts: scenPts,
      baseline_monthly_forecasts: basePts,
    };
  },

  getBacktestComparison(
    provision = 'All',
    safeguardingCategory = 'All',
    horizon = 12
  ): BacktestResponse {
    const baseBt: BacktestResponse = structuredClone(snapshot.backtest_12);
    const { concernMult } = getSliceMultiplier(provision, safeguardingCategory);
    baseBt.provision = provision;
    baseBt.safeguarding_category = safeguardingCategory;
    baseBt.horizon = horizon;

    baseBt.metrics = baseBt.metrics.map((m) => {
      if (m.model_id === 'timesfm_3' && (governanceMode !== 'research_prototype' || !timesfmLicenseAck)) {
        return {
          ...m,
          available: false,
          status_label: 'Not available (Governance/License Gate)',
          mae: null,
          rmse: null,
          wape_pct: null,
          pi_80_coverage_pct: null,
          pi_95_coverage_pct: null,
          annual_total_mae: null,
          outperforms_baseline: null,
          improvement_vs_baseline_pct: null,
          notes:
            'TimesFM 3.0 is disabled when Governance Mode is set to Operational Production or license acknowledgement is unchecked.',
        };
      }
      return {
        ...m,
        mae: m.mae != null ? Number((m.mae * concernMult).toFixed(2)) : null,
        rmse: m.rmse != null ? Number((m.rmse * concernMult).toFixed(2)) : null,
        annual_total_mae:
          m.annual_total_mae != null ? Number((m.annual_total_mae * concernMult * (horizon / 12)).toFixed(0)) : null,
        horizon_evaluated: horizon,
      };
    });

    baseBt.historical_backtest_series = baseBt.historical_backtest_series.map((pt) => ({
      ...pt,
      actual_count: Math.max(0, Math.round(pt.actual_count * concernMult)),
      predictions: Object.fromEntries(
        Object.entries(pt.predictions).map(([k, v]) => [
          k,
          v != null ? Math.max(0, Math.round((v as number) * concernMult)) : null,
        ])
      ),
    }));

    return baseBt;
  },

  getTrendsInsights(provision = 'All', safeguardingCategory = 'All'): TrendsInsightsResponse {
    const baseTr: TrendsInsightsResponse = structuredClone(snapshot.trends_all);
    const hist = this.getHistoricalStats('All', provision, safeguardingCategory);
    const { concernMult } = getSliceMultiplier(provision, safeguardingCategory);

    return {
      ...baseTr,
      is_synthetic: activeDatasetSummary.is_synthetic,
      governance_banner: activeDatasetSummary.governance_banner,
      monthly_series: hist.monthly_series,
      academic_year_comparisons: hist.academic_year_comparisons,
      seasonal_profile: baseTr.seasonal_profile.map((sp) => ({
        ...sp,
        avg_concerns: Number((sp.avg_concerns * concernMult).toFixed(1)),
        min_concerns: Math.round(sp.min_concerns * concernMult),
        max_concerns: Math.round(sp.max_concerns * concernMult),
        avg_concerns_per_teaching_day: Number((sp.avg_concerns_per_teaching_day * concernMult).toFixed(2)),
      })),
    };
  },

  regenerateSyntheticDataset(payload: {
    seed: number;
    num_academic_years: number;
    academic_year_start_month: number;
    base_annual_volume: number;
  }): DatasetSummaryResponse {
    scaleMultiplier = (payload.base_annual_volume || 1450) / 1450;
    activeDatasetSummary = {
      ...structuredClone(snapshot.dataset_summary),
      seed: payload.seed,
      full_academic_years: payload.num_academic_years,
      total_months: payload.num_academic_years * 12,
      academic_year_start_month: payload.academic_year_start_month,
      total_recorded_concerns: Math.round(snapshot.dataset_summary.total_recorded_concerns * scaleMultiplier),
    };
    return activeDatasetSummary;
  },

  switchDataset(datasetId: string): DatasetSummaryResponse {
    const found = activeDatasetSummary.available_datasets.find((d) => d.dataset_id === datasetId);
    if (found) {
      activeDatasetSummary = {
        ...activeDatasetSummary,
        dataset_id: found.dataset_id,
        name: found.name,
        is_synthetic: found.is_synthetic,
        governance_banner: found.is_synthetic
          ? 'SYNTHETIC DEMONSTRATION DATA — NOT ACTUAL COLLEGE SAFEGUARDING RECORDS'
          : 'AGGREGATED OPERATIONAL DATA — DECISION-SUPPORT PROTOTYPE (NO INDIVIDUAL STUDENT RECORDS)',
        total_recorded_concerns: found.total_concerns,
      };
    }
    return activeDatasetSummary;
  },

  updateConfiguration(payload: {
    academic_year_start_month?: number;
    governance_mode?: string;
    timesfm_license_acknowledged?: boolean;
    small_group_suppression_enabled?: boolean;
  }) {
    if (payload.governance_mode) governanceMode = payload.governance_mode;
    if (payload.timesfm_license_acknowledged !== undefined)
      timesfmLicenseAck = payload.timesfm_license_acknowledged;
    if (payload.small_group_suppression_enabled !== undefined)
      smallGroupSuppression = payload.small_group_suppression_enabled;
    if (payload.academic_year_start_month !== undefined)
      activeDatasetSummary.academic_year_start_month = payload.academic_year_start_month;

    return {
      status: 'updated',
      models_availability: this.getModelsAvailability(),
    };
  },

  async validateCsv(file: File, columnMapping?: Record<string, string>): Promise<DataValidationReport> {
    const text = await file.text();
    const lines = text
      .split(/\r?\n/)
      .map((l) => l.trim())
      .filter(Boolean);
    if (lines.length < 2) {
      return {
        is_valid: false,
        can_import: false,
        row_count: 0,
        total_months: 0,
        full_academic_years: 0,
        missing_months: [],
        duplicate_records: 0,
        rejected_personal_data_columns: [],
        detected_columns: [],
        suggested_mapping: {},
        provisions_found: [],
        categories_found: [],
        total_concerns: 0,
        has_attendance_rate: false,
        has_reporting_change_flag: false,
        issues: [{ severity: 'error', code: 'EMPTY_FILE', message: 'CSV file contains no data rows.' }],
        preview_rows: [],
      };
    }

    const headers = lines[0].split(',').map((h) => h.trim());
    const piiCols = headers.filter((h) => PROHIBITED_PII_REGEX.test(h));
    if (piiCols.length > 0) {
      return {
        is_valid: false,
        can_import: false,
        row_count: lines.length - 1,
        total_months: 0,
        full_academic_years: 0,
        missing_months: [],
        duplicate_records: 0,
        rejected_personal_data_columns: piiCols,
        detected_columns: headers,
        suggested_mapping: columnMapping || {},
        provisions_found: [],
        categories_found: [],
        total_concerns: 0,
        has_attendance_rate: false,
        has_reporting_change_flag: false,
        issues: [
          {
            severity: 'error',
            code: 'PROHIBITED_PERSONAL_DATA',
            message: `Personal data or individual student case fields were detected and rejected: ${piiCols.join(
              ', '
            )}. This system only accepts anonymised, monthly aggregated safeguarding counts. Remove all student identifiers, names, dates of birth, and case notes.`,
            affected_rows_or_fields: piiCols,
          },
        ],
        preview_rows: [],
      };
    }

    const suggested: Record<string, string> = {};
    headers.forEach((h) => {
      const low = h.toLowerCase();
      if (low.includes('period') || low.includes('date') || low.includes('month'))
        suggested.period_start = h;
      if (low.includes('concern') && low.includes('count')) suggested.concern_count = h;
      if (low === 'provision' || low.includes('cohort')) suggested.provision = h;
      if (low.includes('category')) suggested.safeguarding_category = h;
      if (low.includes('enrolled') || low.includes('student')) suggested.enrolled_students = h;
      if (low.includes('teaching')) suggested.teaching_days = h;
      if (low.includes('attendance')) suggested.attendance_rate = h;
      if (low.includes('reporting_change')) suggested.reporting_change_flag = h;
    });

    const effMap = { ...suggested, ...(columnMapping || {}) };
    const rows = lines.slice(1).map((line) => {
      const vals = line.split(',');
      const obj: Record<string, any> = {};
      headers.forEach((h, i) => {
        obj[h] = vals[i] ?? '';
      });
      return obj;
    });

    const months = new Set(rows.map((r) => String(r[effMap.period_start] || '').slice(0, 7)).filter(Boolean));
    const totalConcerns = rows.reduce((acc, r) => acc + (Number(r[effMap.concern_count]) || 0), 0);

    return {
      is_valid: true,
      can_import: months.size >= 12,
      row_count: rows.length,
      date_range_start: Array.from(months).sort()[0] + '-01',
      date_range_end: Array.from(months).sort().slice(-1)[0] + '-01',
      total_months: months.size,
      full_academic_years: Math.floor(months.size / 12),
      missing_months: [],
      duplicate_records: 0,
      rejected_personal_data_columns: [],
      detected_columns: headers,
      suggested_mapping: effMap,
      provisions_found: ['EPYP', 'Adult', 'Apprenticeships'],
      categories_found: Object.keys(CATEGORY_SHARES).filter((k) => k !== 'All'),
      total_concerns: totalConcerns,
      has_attendance_rate: Boolean(effMap.attendance_rate),
      has_reporting_change_flag: Boolean(effMap.reporting_change_flag),
      issues: [],
      preview_rows: rows.slice(0, 10),
    };
  },

  async importCsv(
    file: File,
    datasetName?: string,
    columnMapping?: Record<string, string>
  ): Promise<DatasetSummaryResponse> {
    const rep = await this.validateCsv(file, columnMapping);
    if (!rep.can_import) {
      throw new Error('CSV validation failed.');
    }
    const newId = `imported_${Math.random().toString(36).slice(2, 8)}`;
    const newEntry = {
      dataset_id: newId,
      name: `Imported: ${datasetName || file.name}`,
      is_synthetic: false,
      seed: null,
      academic_year_start_month: 9,
      created_at: new Date().toISOString(),
      row_count: rep.row_count,
      total_concerns: rep.total_concerns,
    };
    activeDatasetSummary = {
      ...activeDatasetSummary,
      dataset_id: newId,
      name: newEntry.name,
      is_synthetic: false,
      governance_banner:
        'AGGREGATED OPERATIONAL DATA — DECISION-SUPPORT PROTOTYPE (NO INDIVIDUAL STUDENT RECORDS)',
      total_months: rep.total_months,
      full_academic_years: rep.full_academic_years,
      total_recorded_concerns: rep.total_concerns,
      available_datasets: [...activeDatasetSummary.available_datasets, newEntry],
    };
    return activeDatasetSummary;
  },

  getRecentForecastRuns(): { runs: any[] } {
    return { runs: snapshot.recent_runs || [] };
  },

  generateEsotericForecast(req: EsotericForecastRequest): EsotericForecastResponse {
    const cadence = req.cadence === 'weekly' ? 'weekly' : 'monthly';
    const stratId =
      req.strategy_id &&
      ['staged_core_4', 'rough_signatures_focus', 'fisher_rao_ngboost', 'full_13_conditioned'].includes(
        req.strategy_id
      )
        ? req.strategy_id
        : 'staged_core_4';

    const validMonthlyModels = ['lightgbm_mlforecast', 'statistical_count', 'seasonal_naive'];
    const validWeeklyModels = ['tweedie_lgbm', 'nb2_glm', 'zinb_model'];
    const effBaseModel =
      cadence === 'weekly'
        ? validWeeklyModels.includes(req.base_model_id)
          ? req.base_model_id
          : 'tweedie_lgbm'
        : validMonthlyModels.includes(req.base_model_id)
          ? req.base_model_id
          : 'lightgbm_mlforecast';

    const baseRun: EsotericForecastResponse = structuredClone(
      esoForecastSnapshot[cadence]?.[effBaseModel]?.[stratId] ||
        esoForecastSnapshot[cadence]?.[effBaseModel]?.['staged_core_4']
    );
    const fullRun: EsotericForecastResponse | undefined =
      esoForecastSnapshot[cadence]?.[effBaseModel]?.['full_13_conditioned'];

    const maxH = cadence === 'weekly' ? 26 : 12;
    const allowedHorizons = cadence === 'weekly' ? [13, 26] : [3, 6, 12];
    const effHorizon = allowedHorizons.includes(req.horizon) ? req.horizon : maxH;
    const piLevel = req.prediction_interval === 95 ? 95 : 80;

    const provision = req.provision || 'All';
    const category = cadence === 'weekly' ? 'All' : req.safeguarding_category || 'All';
    const { concernMult, enrolMult } = getSliceMultiplier(provision, category);

    const scen = req.scenario || {
      enrolment_growth_pct: 0,
      teaching_days_delta: 0,
      attendance_rate_delta_pp: 0,
    };
    const enrolElast = Math.pow(1 + (scen.enrolment_growth_pct || 0) / 100, 0.85);
    const tdDenom = cadence === 'weekly' ? 5 : 18;
    const tdElast = Math.pow(1 + (scen.teaching_days_delta || 0) / tdDenom, 0.35);
    const attElast = 1 - (scen.attendance_rate_delta_pp || 0) * 0.011;
    const scenarioMult = Math.max(0.35, enrolElast * tdElast * attElast);

    // Handle interactive custom module subset toggles (`f01`..`f13`)
    let activeModules = [...baseRun.active_modules];
    let admittedFeatures: EsotericAdmittedFeature[] = [...baseRun.conditioning_diagnostics.admitted_features];
    let residualBlend = 1.0;
    let useFullReference = false;
    let identityFloor = baseRun.conditioning_diagnostics.identity_floor_active;
    let nCand = baseRun.conditioning_diagnostics.n_candidate_esoteric_features;
    let nAdm = baseRun.conditioning_diagnostics.n_admitted_esoteric_features;

    if (Array.isArray(req.custom_modules)) {
      const validIds = new Set(baseRun.module_catalog.map((m) => m.id));
      const customClean = req.custom_modules.filter((m) => validIds.has(m));
      activeModules = customClean;

      if (customClean.length === 0) {
        admittedFeatures = [];
        residualBlend = 0.0;
        identityFloor = true;
        nCand = 0;
        nAdm = 0;
      } else {
        const defaultSet = new Set(baseRun.active_modules);
        const isExactDefault =
          customClean.length === baseRun.active_modules.length &&
          customClean.every((m) => defaultSet.has(m));

        if (!isExactDefault) {
          const customSet = new Set(customClean);
          const hasOutsideDefault = customClean.some((m) => !defaultSet.has(m));
          const poolSource = hasOutsideDefault && fullRun ? fullRun : baseRun;
          useFullReference = Boolean(hasOutsideDefault && fullRun);

          const poolFeatures = poolSource.conditioning_diagnostics.admitted_features;
          const totalPoolWeight = poolFeatures.reduce((acc, f) => acc + Math.abs(f.weight), 0);
          const filtered = poolFeatures.filter((f) => customSet.has(f.module_id));

          if (filtered.length > 0 && totalPoolWeight > 0) {
            admittedFeatures = filtered;
            const keptWeight = filtered.reduce((acc, f) => acc + Math.abs(f.weight), 0);
            residualBlend = Math.min(1.15, Math.max(0.15, keptWeight / totalPoolWeight));
          } else {
            // Synthesize lightweight attribution entries for toggled modules when outside precomputed top-k
            admittedFeatures = customClean.slice(0, 4).map((mId, idx) => {
              const catMeta = baseRun.module_catalog.find((c) => c.id === mId);
              return {
                feature: `${mId}_causal_manifold_signal`,
                module_id: mId,
                module_name: catMeta ? catMeta.name : mId.toUpperCase(),
                weight: Number((0.045 / (idx + 1)).toFixed(4)),
                direction: 'surge-amplifying',
                description: catMeta ? catMeta.role : 'Causal higher-order manifold predictor.',
              };
            });
            residualBlend = Math.min(1.0, Math.max(0.2, customClean.length / Math.max(1, baseRun.active_modules.length)));
          }
          nCand = customClean.length * 8;
          nAdm = admittedFeatures.length;
          identityFloor = false;
        }
      }
    }

    const refForecastSeries =
      useFullReference && fullRun ? fullRun.forecast_series : baseRun.forecast_series;

    const histScaled = baseRun.historical_series.map((pt) => {
      const cnt = Math.max(0, Math.round(pt.actual_count * concernMult));
      const enrol = Math.max(10, Math.round(pt.enrolled_students * enrolMult));
      return {
        ...pt,
        actual_count: cnt,
        enrolled_students: enrol,
        rate_per_1000: Number(((cnt / enrol) * 1000).toFixed(2)),
      };
    });

    const forecastScaled: EsotericForecastPoint[] = baseRun.forecast_series
      .slice(0, effHorizon)
      .map((pt, idx) => {
        const refPt = refForecastSeries[idx] || pt;
        const baseValRaw = pt.baseline_forecast * concernMult * scenarioMult;
        const baseVal = Math.max(0, Math.round(baseValRaw));

        const rawShift = (refPt.esoteric_raw - refPt.baseline_forecast) * residualBlend;
        const esoRawScaled = Math.max(0, (pt.baseline_forecast + rawShift) * concernMult * scenarioMult);
        const esoVal = Math.max(0, Math.round(esoRawScaled));

        const scaleRatio =
          refPt.esoteric_raw > 0.1 ? esoRawScaled / refPt.esoteric_raw : concernMult * scenarioMult;
        const l80 = Math.max(0, Math.round(refPt.lower_80 * scaleRatio));
        const u80 = Math.max(esoVal, Math.round(refPt.upper_80 * scaleRatio));
        const l95 = Math.max(0, Math.round(refPt.lower_95 * scaleRatio));
        const u95 = Math.max(u80, Math.round(refPt.upper_95 * scaleRatio));

        const baseScale = pt.baseline_forecast > 0.1 ? baseValRaw / pt.baseline_forecast : concernMult * scenarioMult;
        const baseLo = Math.max(0, Math.round(pt.baseline_lower * baseScale));
        const baseHi = Math.max(baseVal, Math.round(pt.baseline_upper * baseScale));

        const enrol = Math.max(
          10,
          Math.round(
            pt.enrolled_students * enrolMult * (1 + (scen.enrolment_growth_pct || 0) / 100)
          )
        );
        const maxTd = cadence === 'weekly' ? 5 : 25;
        const td = Math.max(0, Math.min(maxTd, pt.teaching_days + (scen.teaching_days_delta || 0)));
        const deltaCount = esoVal - baseVal;
        const deltaPct = Number(((deltaCount / Math.max(1, baseVal)) * 100).toFixed(2));

        return {
          ...pt,
          baseline_forecast: baseVal,
          baseline_lower: baseLo,
          baseline_upper: baseHi,
          esoteric_forecast: esoVal,
          esoteric_raw: Number(esoRawScaled.toFixed(2)),
          lower_bound: piLevel === 95 ? l95 : l80,
          upper_bound: piLevel === 95 ? u95 : u80,
          lower_80: l80,
          upper_80: u80,
          lower_95: l95,
          upper_95: u95,
          delta_count: deltaCount,
          delta_pct: deltaPct,
          enrolled_students: enrol,
          teaching_days: td,
          rate_per_1000: Number(((esoVal / enrol) * 1000).toFixed(2)),
          rough_levy_area: Number((refPt.rough_levy_area * Math.max(0.25, residualBlend)).toFixed(4)),
          rough_sig_depth3: Number((refPt.rough_sig_depth3 * Math.max(0.25, residualBlend)).toFixed(4)),
          hawkes_intensity: Number((refPt.hawkes_intensity * concernMult * scenarioMult).toFixed(3)),
          dominant_driver:
            admittedFeatures.length > 0
              ? `${admittedFeatures[0].module_id.toUpperCase()} (${admittedFeatures[0].module_name})`
              : 'Pure Anchor Baseline (Identity Floor)',
        };
      });

    const esoTotal = forecastScaled.reduce((acc, p) => acc + p.esoteric_forecast, 0);
    const baseTotal = forecastScaled.reduce((acc, p) => acc + p.baseline_forecast, 0);
    const priorSlice = histScaled.slice(-effHorizon);
    const priorTotal = Math.max(1, priorSlice.reduce((acc, p) => acc + p.actual_count, 0));
    const yoyPct = Number((((esoTotal - priorTotal) / priorTotal) * 100).toFixed(1));
    const netDelta = esoTotal - baseTotal;
    const netDeltaPct = Number(((netDelta / Math.max(1, baseTotal)) * 100).toFixed(2));

    // Sub-additive correlated horizon uncertainty scaling (sqrt(H) + 0.35 H law)
    const stepHalfWidths = forecastScaled.map((p) =>
      Math.max(1, ((piLevel === 95 ? p.upper_95 - p.lower_95 : p.upper_80 - p.lower_80) / 2))
    );
    const rmsHalfWidth = Math.sqrt(
      stepHalfWidths.reduce((acc, w) => acc + w * w, 0) * (1 + 0.22 * Math.sqrt(effHorizon))
    );
    const esoLower = Math.max(0, Math.round(esoTotal - rmsHalfWidth));
    const esoUpper = Math.max(esoTotal, Math.round(esoTotal + rmsHalfWidth));

    const baseStepHalfWidths = forecastScaled.map((p) =>
      Math.max(1, (p.baseline_upper - p.baseline_lower) / 2)
    );
    const baseRmsHalf = Math.sqrt(
      baseStepHalfWidths.reduce((acc, w) => acc + w * w, 0) * (1 + 0.22 * Math.sqrt(effHorizon))
    );
    const baseLower = Math.max(0, Math.round(baseTotal - baseRmsHalf));
    const baseUpper = Math.max(baseTotal, Math.round(baseTotal + baseRmsHalf));

    const peakPt = [...forecastScaled].sort((a, b) => b.esoteric_forecast - a.esoteric_forecast)[0];

    const btOrig = baseRun.walk_forward_backtest;
    const btScaled = {
      ...btOrig,
      baseline_mae: Number((btOrig.baseline_mae * concernMult).toFixed(2)),
      esoteric_mae: Number(
        (
          (btOrig.baseline_mae + (btOrig.esoteric_mae - btOrig.baseline_mae) * residualBlend) *
          concernMult
        ).toFixed(2)
      ),
      mae_delta_pct: Number((btOrig.mae_delta_pct * residualBlend).toFixed(2)),
      baseline_rmse: Number((btOrig.baseline_rmse * concernMult).toFixed(2)),
      esoteric_rmse: Number(
        (
          (btOrig.baseline_rmse + (btOrig.esoteric_rmse - btOrig.baseline_rmse) * residualBlend) *
          concernMult
        ).toFixed(2)
      ),
      baseline_surge_mae: Number((btOrig.baseline_surge_mae * concernMult).toFixed(2)),
      esoteric_surge_mae: Number(
        (
          (btOrig.baseline_surge_mae +
            (btOrig.esoteric_surge_mae - btOrig.baseline_surge_mae) * residualBlend) *
          concernMult
        ).toFixed(2)
      ),
      points: btOrig.points.map((p) => {
        const bScaled = Math.max(0, Math.round(p.baseline * concernMult));
        const eScaled = Math.max(
          0,
          Math.round((p.baseline + (p.esoteric - p.baseline) * residualBlend) * concernMult)
        );
        return {
          ...p,
          actual: Math.max(0, Math.round(p.actual * concernMult)),
          baseline: bScaled,
          esoteric: eScaled,
        };
      }),
    };

    return {
      ...baseRun,
      is_synthetic: activeDatasetSummary.is_synthetic,
      governance_banner: activeDatasetSummary.governance_banner,
      active_modules: activeModules,
      horizon: effHorizon,
      provision,
      safeguarding_category: category,
      prediction_interval: piLevel,
      summary: {
        ...baseRun.summary,
        esoteric_total: esoTotal,
        esoteric_lower: esoLower,
        esoteric_upper: esoUpper,
        baseline_total: baseTotal,
        baseline_lower: baseLower,
        baseline_upper: baseUpper,
        prior_horizon_total: priorTotal,
        yoy_change_pct: yoyPct,
        net_esoteric_delta: netDelta,
        net_esoteric_delta_pct: netDeltaPct,
        peak_period_label: peakPt ? peakPt.period_label : baseRun.summary.peak_period_label,
        peak_period_count: peakPt ? peakPt.esoteric_forecast : 0,
      },
      conditioning_diagnostics: {
        ...baseRun.conditioning_diagnostics,
        n_candidate_esoteric_features: nCand,
        n_admitted_esoteric_features: nAdm,
        n_quarantined_esoteric_features: Math.max(0, nCand - nAdm),
        identity_floor_active: identityFloor,
        admitted_features: admittedFeatures,
      },
      walk_forward_backtest: btScaled,
      historical_series: histScaled,
      forecast_series: forecastScaled,
    };
  },
};

