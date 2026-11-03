import React, { useMemo, useState } from 'react';
import {
  ComposedChart,
  Line,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  Cell,
} from 'recharts';
import {
  TrendingUp,
  TrendingDown,
  ShieldAlert,
  Calendar,
  Activity,
  Award,
  Layers,
  Users,
  Info,
  CheckCircle2,
  AlertTriangle,
  Sparkles,
} from 'lucide-react';
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  Badge,
} from '../components/ui/primitives';
import type {
  DatasetSummaryResponse,
  ForecastResponse,
  HistoricalStatsResponse,
  ModelAvailabilityItem,
} from '../types/api';

interface OverviewPageProps {
  datasetSummary: DatasetSummaryResponse;
  forecastData: ForecastResponse;
  historicalStats: HistoricalStatsResponse;
  models: ModelAvailabilityItem[];
  selectedModel: string;
  onSelectModel: (m: string) => void;
  selectedHorizon: number;
  onSelectHorizon: (h: number) => void;
  selectedProvision: string;
  onSelectProvision: (p: string) => void;
  selectedCategory: string;
  onSelectCategory: (c: string) => void;
  selectedPI: number;
  onSelectPI: (pi: number) => void;
  selectedAcademicYearFilter: string;
  onSelectAcademicYearFilter: (ay: string) => void;
  isLoading: boolean;
}

const PROVISION_COLORS = ['#0f172a', '#0d9488', '#2563eb', '#7c3aed', '#f59e0b'];
const CATEGORY_COLORS = ['#1e3a8a', '#0d9488', '#3b82f6', '#7c3aed', '#64748b'];

export const OverviewPage: React.FC<OverviewPageProps> = ({
  datasetSummary,
  forecastData,
  historicalStats,
  models,
  selectedModel,
  onSelectModel,
  selectedHorizon,
  onSelectHorizon,
  selectedProvision,
  onSelectProvision,
  selectedCategory,
  onSelectCategory,
  selectedPI,
  onSelectPI,
  selectedAcademicYearFilter,
  onSelectAcademicYearFilter,
  isLoading,
}) => {
  const [granularity, setGranularity] = useState<'monthly' | 'quarterly'>('monthly');
  const [historyWindow, setHistoryWindow] = useState<'all' | '36m' | '24m'>('36m');
  const [showReportingChangeMarkers, setShowReportingChangeMarkers] = useState<boolean>(true);

  const summary = forecastData.annual_summary;
  const acc = forecastData.model_accuracy_summary;

  // Build unified timeline for primary chart
  const primaryChartData = useMemo(() => {
    let histSlice = [...forecastData.historical_series];
    if (selectedAcademicYearFilter === 'All') {
      if (historyWindow === '36m') histSlice = histSlice.slice(-36);
      if (historyWindow === '24m') histSlice = histSlice.slice(-24);
    }

    const monthlyItems: any[] = histSlice.map((h, idx) => {
      const isLastHist = idx === histSlice.length - 1;
      return {
        period_start: h.period_start,
        label: h.period_label,
        academic_year: h.academic_year,
        academic_term: h.academic_term,
        actual: h.concern_count,
        // Bridge the last historical point to the forecast line for seamless visual continuity
        forecast: isLastHist ? h.concern_count : null,
        pi_band: isLastHist ? [h.concern_count, h.concern_count] : null,
        lower_bound: isLastHist ? h.concern_count : null,
        upper_bound: isLastHist ? h.concern_count : null,
        concerns_per_1000: h.concerns_per_1000,
        enrolled_students: h.enrolled_students,
        teaching_days: h.teaching_days,
        is_forecast: false,
      };
    });

    forecastData.monthly_forecasts.forEach((f) => {
      const lb = selectedPI === 95 && f.lower_95 != null ? f.lower_95 : f.lower_80;
      const ub = selectedPI === 95 && f.upper_95 != null ? f.upper_95 : f.upper_80;
      monthlyItems.push({
        period_start: f.period_start,
        label: f.period_label,
        academic_year: f.academic_year,
        academic_term: f.academic_term,
        actual: null,
        forecast: f.point_forecast,
        pi_band: [lb, ub],
        lower_bound: lb,
        upper_bound: ub,
        concerns_per_1000: f.concerns_per_1000_forecast,
        enrolled_students: f.enrolled_students_assumed,
        teaching_days: f.teaching_days_assumed,
        is_forecast: true,
      });
    });

    if (granularity === 'monthly') {
      return monthlyItems;
    }

    // Aggregate into 3-month quarters if quarterly presentation is selected
    const quarters: any[] = [];
    const histOnly = monthlyItems.filter((x) => !x.is_forecast);
    const futOnly = monthlyItems.filter((x) => x.is_forecast);

    for (let i = 0; i < histOnly.length; i += 3) {
      const chunk = histOnly.slice(i, i + 3);
      if (chunk.length === 0) continue;
      const sumActual = chunk.reduce((accSum, r) => accSum + (r.actual || 0), 0);
      const isLastChunk = i + 3 >= histOnly.length;
      quarters.push({
        period_start: chunk[0].period_start,
        label: `${chunk[0].label.split(' ')[0]}–${chunk[chunk.length - 1].label}`,
        academic_year: chunk[0].academic_year,
        actual: sumActual,
        forecast: isLastChunk ? sumActual : null,
        pi_band: isLastChunk ? [sumActual, sumActual] : null,
        lower_bound: isLastChunk ? sumActual : null,
        upper_bound: isLastChunk ? sumActual : null,
        is_forecast: false,
      });
    }

    for (let i = 0; i < futOnly.length; i += 3) {
      const chunk = futOnly.slice(i, i + 3);
      if (chunk.length === 0) continue;
      const sumPred = chunk.reduce((accSum, r) => accSum + (r.forecast || 0), 0);
      // Calibrated quarterly interval width using covariance factor
      const sumVar = chunk.reduce((accVar, r) => {
        const half = ((r.upper_bound || 0) - (r.lower_bound || 0)) / 2;
        return accVar + half * half * 1.35;
      }, 0);
      const qHalf = Math.round(Math.sqrt(sumVar));
      const qLow = Math.max(0, sumPred - qHalf);
      const qHigh = sumPred + qHalf;
      quarters.push({
        period_start: chunk[0].period_start,
        label: `${chunk[0].label.split(' ')[0]}–${chunk[chunk.length - 1].label}`,
        academic_year: chunk[0].academic_year,
        actual: null,
        forecast: sumPred,
        pi_band: [qLow, qHigh],
        lower_bound: qLow,
        upper_bound: qHigh,
        is_forecast: true,
      });
    }
    return quarters;
  }, [
    forecastData.historical_series,
    forecastData.monthly_forecasts,
    selectedAcademicYearFilter,
    historyWindow,
    selectedPI,
    granularity,
  ]);

  const dividerLabel = useMemo(() => {
    const histItems = primaryChartData.filter((d) => !d.is_forecast);
    return histItems.length > 0 ? histItems[histItems.length - 1].label : undefined;
  }, [primaryChartData]);

  const reportingMarkerLabels = useMemo(() => {
    if (!showReportingChangeMarkers || granularity !== 'monthly') return [];
    const dateSet = new Set(historicalStats.reporting_change_dates);
    return primaryChartData
      .filter((d) => dateSet.has(d.period_start))
      .map((d) => d.label);
  }, [primaryChartData, historicalStats.reporting_change_dates, showReportingChangeMarkers, granularity]);

  return (
    <div className="space-y-6">
      {/* Page Header & Top Filter Bar */}
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between bg-white p-5 rounded-2xl border border-slate-200/80 shadow-xs">
        <div>
          <div className="flex flex-wrap items-center gap-2.5">
            <h1 className="fluid-title font-display font-extrabold text-slate-900 tracking-tight">
              Safeguarding Demand Forecast
            </h1>
            <Badge variant={datasetSummary.is_synthetic ? 'amber' : 'teal'}>
              {datasetSummary.is_synthetic ? 'Synthetic Demo Dataset' : 'Imported College Data'}
            </Badge>
            <Badge variant="slate">
              {datasetSummary.total_months} Months ({datasetSummary.full_academic_years} Academic Years)
            </Badge>
          </div>
          <p className="mt-1 text-sm text-slate-600">
            AI-powered forecasting for proactive safeguarding resource planning at{' '}
            <span className="font-semibold text-slate-800">South Devon College</span>.
          </p>
        </div>

        {/* Global Filters */}
        <div className="flex flex-wrap items-center gap-2.5">
          <div className="flex flex-col">
            <label htmlFor="overview-ay-select" className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-1">
              Historical View
            </label>
            <select
              id="overview-ay-select"
              value={selectedAcademicYearFilter}
              onChange={(e) => onSelectAcademicYearFilter(e.target.value)}
              className="rounded-xl border border-slate-300 bg-slate-50 px-3 py-1.5 text-xs font-medium text-slate-800 focus:border-teal-600 focus:outline-none"
            >
              <option value="All">All Academic Years ({datasetSummary.date_range_start.slice(0, 4)}–{datasetSummary.date_range_end.slice(0, 4)})</option>
              {datasetSummary.academic_years.map((ay) => (
                <option key={ay} value={ay}>
                  Academic Year {ay}
                </option>
              ))}
            </select>
          </div>

          <div className="flex flex-col">
            <label htmlFor="overview-prov-select" className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-1">
              Provision Scope
            </label>
            <select
              id="overview-prov-select"
              value={selectedProvision}
              onChange={(e) => onSelectProvision(e.target.value)}
              className="rounded-xl border border-slate-300 bg-slate-50 px-3 py-1.5 text-xs font-medium text-slate-800 focus:border-teal-600 focus:outline-none"
            >
              <option value="All">College-Wide (All Provisions)</option>
              {datasetSummary.provisions.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </div>

          <div className="flex flex-col">
            <label htmlFor="overview-cat-select" className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-1">
              Safeguarding Category
            </label>
            <select
              id="overview-cat-select"
              value={selectedCategory}
              onChange={(e) => onSelectCategory(e.target.value)}
              className="rounded-xl border border-slate-300 bg-slate-50 px-3 py-1.5 text-xs font-medium text-slate-800 focus:border-teal-600 focus:outline-none"
            >
              <option value="All">All Safeguarding Categories</option>
              {datasetSummary.categories.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Top 4 KPI Cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {/* Card 1: Predicted Annual Concerns */}
        <Card className="relative overflow-hidden border-l-4 border-l-teal-600">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                Predicted {summary.horizon_months === 12 ? 'Annual' : `${summary.horizon_months}-Month`} Concerns
              </span>
              <div className="rounded-lg bg-teal-50 p-2 text-teal-700">
                <Activity className="h-4 w-4" />
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="flex items-baseline gap-2.5">
              <span className="font-display text-3xl font-extrabold text-slate-900">
                {summary.predicted_total_concerns.toLocaleString()}
              </span>
              <Badge variant={summary.yoy_change_pct >= 0 ? 'amber' : 'teal'}>
                {summary.yoy_change_pct >= 0 ? (
                  <TrendingUp className="h-3 w-3" />
                ) : (
                  <TrendingDown className="h-3 w-3" />
                )}
                {summary.yoy_change_pct >= 0 ? `+${summary.yoy_change_pct}%` : `${summary.yoy_change_pct}%`} vs prior period
              </Badge>
            </div>
            <p className="mt-2 text-xs text-slate-500">
              Target Period: <strong className="text-slate-700">{summary.target_academic_year}</strong> ({summary.predicted_concerns_per_1000} per 1,000 students; prior period: {summary.prior_year_total.toLocaleString()})
            </p>
          </CardContent>
        </Card>

        {/* Card 2: Expected Planning Range */}
        <Card className="relative overflow-hidden border-l-4 border-l-blue-600">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                Expected Planning Range ({summary.interval_level}% PI)
              </span>
              <div className="rounded-lg bg-blue-50 p-2 text-blue-700">
                <Layers className="h-4 w-4" />
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="flex items-baseline gap-2">
              <span className="font-display text-3xl font-extrabold text-slate-900">
                {summary.lower_bound.toLocaleString()} – {summary.upper_bound.toLocaleString()}
              </span>
            </div>
            <p className="mt-2 text-xs text-slate-500" title={summary.calibration_method}>
              Calibrated multi-month interval (±{Math.round((summary.upper_bound - summary.lower_bound) / 2).toLocaleString()} concerns; accounts for cross-month error covariance)
            </p>
          </CardContent>
        </Card>

        {/* Card 3: Peak Demand Period */}
        <Card className="relative overflow-hidden border-l-4 border-l-purple-600">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                Peak Demand Period
              </span>
              <div className="rounded-lg bg-purple-50 p-2 text-purple-700">
                <Calendar className="h-4 w-4" />
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="flex items-baseline gap-2">
              <span className="font-display text-2xl font-extrabold text-slate-900">
                {summary.peak_demand_month}
              </span>
              <Badge variant="purple">{summary.peak_demand_count} concerns</Badge>
            </div>
            <p className="mt-2 text-xs text-slate-500">
              High-demand window: <strong className="text-slate-700">{summary.peak_demand_period_description}</strong>
            </p>
          </CardContent>
        </Card>

        {/* Card 4: Forecast Accuracy */}
        <Card className="relative overflow-hidden border-l-4 border-l-slate-800">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                Historical Backtest Accuracy
              </span>
              <div className="rounded-lg bg-slate-100 p-2 text-slate-800">
                <Award className="h-4 w-4" />
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="flex items-baseline gap-2.5">
              <span className="font-display text-3xl font-extrabold text-slate-900">
                {acc.mae != null ? `±${acc.mae}` : 'N/A'}
              </span>
              <span className="text-xs font-medium text-slate-500">MAE / month</span>
              {acc.wape_pct != null && (
                <Badge variant={acc.outperforms_baseline ? 'emerald' : 'slate'}>
                  {acc.wape_pct}% WAPE
                </Badge>
              )}
            </div>
            <p className="mt-2 text-xs text-slate-500">
              {acc.explanation}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Primary Interactive Time-Series Chart */}
      <Card className="dashboard-section-contain">
        <CardHeader className="border-b border-slate-100 pb-4">
          <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
            <div>
              <div className="flex items-center gap-2">
                <CardTitle className="text-lg">
                  Historical Recorded Concerns & Future Demand Trajectory
                </CardTitle>
                {isLoading && (
                  <Badge variant="teal" className="animate-pulse">
                    Recalculating model...
                  </Badge>
                )}
              </div>
              <CardDescription className="mt-0.5">
                Solid navy line shows recorded monthly safeguarding volume; contrasting teal line and shaded band show{' '}
                <strong className="text-slate-700">{forecastData.model_name}</strong> predictions with {selectedPI}% prediction intervals.
              </CardDescription>
            </div>

            {/* Interactive Chart Switches */}
            <div className="flex flex-wrap items-center gap-2">
              {/* Model Switch */}
              <div className="flex items-center gap-1.5 rounded-xl bg-slate-100 p-1">
                {models
                  .filter((m) => m.model_id !== 'chronos_2')
                  .map((m) => (
                    <button
                      key={m.model_id}
                      disabled={!m.available}
                      onClick={() => onSelectModel(m.model_id)}
                      title={m.available ? m.status_explanation : `Unavailable: ${m.status_explanation}`}
                      className={`rounded-lg px-2.5 py-1 text-xs font-semibold transition-all cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed ${
                        selectedModel === m.model_id
                          ? 'bg-slate-900 text-white shadow-xs'
                          : 'text-slate-600 hover:text-slate-900'
                      }`}
                    >
                      {m.short_name}
                    </button>
                  ))}
              </div>

              {/* Horizon Switch */}
              <div className="flex items-center gap-1 rounded-xl bg-slate-100 p-1">
                {[3, 6, 12].map((h) => (
                  <button
                    key={h}
                    onClick={() => onSelectHorizon(h)}
                    className={`rounded-lg px-2.5 py-1 text-xs font-semibold transition-all cursor-pointer ${
                      selectedHorizon === h
                        ? 'bg-teal-600 text-white shadow-xs'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    {h}m
                  </button>
                ))}
              </div>

              {/* Confidence Range Switch */}
              <div className="flex items-center gap-1 rounded-xl bg-slate-100 p-1">
                {[80, 95].map((pi) => (
                  <button
                    key={pi}
                    onClick={() => onSelectPI(pi)}
                    className={`rounded-lg px-2.5 py-1 text-xs font-semibold transition-all cursor-pointer ${
                      selectedPI === pi
                        ? 'bg-blue-600 text-white shadow-xs'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    {pi}% PI
                  </button>
                ))}
              </div>

              {/* Historical Window Switch */}
              {selectedAcademicYearFilter === 'All' && (
                <div className="flex items-center gap-1 rounded-xl bg-slate-100 p-1">
                  {[
                    { id: '24m', label: '2Y Hist' },
                    { id: '36m', label: '3Y Hist' },
                    { id: 'all', label: 'All 6Y' },
                  ].map((w) => (
                    <button
                      key={w.id}
                      onClick={() => setHistoryWindow(w.id as any)}
                      className={`rounded-lg px-2 py-1 text-xs font-semibold transition-all cursor-pointer ${
                        historyWindow === w.id
                          ? 'bg-white text-slate-900 shadow-xs'
                          : 'text-slate-600 hover:text-slate-900'
                      }`}
                    >
                      {w.label}
                    </button>
                  ))}
                </div>
              )}

              {/* Monthly / Quarterly Presentation Toggle */}
              <div className="flex items-center gap-1 rounded-xl bg-slate-100 p-1">
                <button
                  onClick={() => setGranularity('monthly')}
                  className={`rounded-lg px-2.5 py-1 text-xs font-semibold transition-all cursor-pointer ${
                    granularity === 'monthly'
                      ? 'bg-white text-slate-900 shadow-xs'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  Monthly
                </button>
                <button
                  onClick={() => setGranularity('quarterly')}
                  className={`rounded-lg px-2.5 py-1 text-xs font-semibold transition-all cursor-pointer ${
                    granularity === 'quarterly'
                      ? 'bg-white text-slate-900 shadow-xs'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  Quarterly
                </button>
              </div>
            </div>
          </div>
        </CardHeader>

        <CardContent className="pt-4">
          <div className="h-[390px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={primaryChartData} margin={{ top: 15, right: 24, left: 4, bottom: 10 }}>
                <defs>
                  <linearGradient id="piBandGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#0d9488" stopOpacity={0.26} />
                    <stop offset="95%" stopColor="#0d9488" stopOpacity={0.07} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                <XAxis
                  dataKey="label"
                  tick={{ fontSize: 11, fill: '#475569' }}
                  tickMargin={8}
                  minTickGap={22}
                />
                <YAxis
                  tick={{ fontSize: 11, fill: '#475569' }}
                  axisLine={false}
                  tickLine={false}
                  label={{
                    value: 'Recorded Concerns',
                    angle: -90,
                    position: 'insideLeft',
                    style: { fontSize: 11, fill: '#64748b', fontWeight: 600 },
                  }}
                />
                <Tooltip
                  content={({ active, payload, label }) => {
                    if (!active || !payload || payload.length === 0) return null;
                    const row = payload[0].payload;
                    return (
                      <div className="rounded-xl border border-slate-200 bg-white/95 p-3.5 shadow-lg backdrop-blur-xs text-xs">
                        <div className="flex items-center justify-between gap-4 border-b border-slate-100 pb-1.5 mb-2">
                          <span className="font-bold text-slate-900">{label}</span>
                          <Badge variant={row.is_forecast ? 'teal' : 'slate'}>
                            {row.is_forecast ? 'Forecast' : 'Historical'} • {row.academic_year}
                          </Badge>
                        </div>
                        {row.actual != null && (
                          <div className="flex items-center justify-between gap-6 py-0.5">
                            <span className="text-slate-600 flex items-center gap-1.5">
                              <span className="h-2.5 w-2.5 rounded-full bg-slate-900 inline-block" />
                              Recorded Concerns:
                            </span>
                            <span className="font-bold text-slate-900">{row.actual.toLocaleString()}</span>
                          </div>
                        )}
                        {row.is_forecast && row.forecast != null && (
                          <>
                            <div className="flex items-center justify-between gap-6 py-0.5">
                              <span className="text-teal-700 flex items-center gap-1.5 font-medium">
                                <span className="h-2.5 w-2.5 rounded-full bg-teal-600 inline-block" />
                                Predicted Concerns:
                              </span>
                              <span className="font-bold text-teal-800">{row.forecast.toLocaleString()}</span>
                            </div>
                            <div className="flex items-center justify-between gap-6 py-0.5 text-slate-600">
                              <span>{selectedPI}% Prediction Interval:</span>
                              <span className="font-semibold text-slate-800">
                                {row.lower_bound?.toLocaleString()} – {row.upper_bound?.toLocaleString()}
                              </span>
                            </div>
                          </>
                        )}
                        {row.concerns_per_1000 != null && (
                          <div className="mt-1.5 pt-1.5 border-t border-slate-100 flex items-center justify-between gap-6 text-slate-500">
                            <span>Rate per 1,000 Students:</span>
                            <span className="font-medium text-slate-700">{row.concerns_per_1000}</span>
                          </div>
                        )}
                        {row.teaching_days != null && (
                          <div className="flex items-center justify-between gap-6 text-slate-500">
                            <span>Scheduled Teaching Days:</span>
                            <span className="font-medium text-slate-700">{row.teaching_days} days</span>
                          </div>
                        )}
                      </div>
                    );
                  }}
                />
                <Legend wrapperStyle={{ fontSize: 12, paddingTop: 10 }} />

                {/* Shaded Prediction Interval Band */}
                <Area
                  type="monotone"
                  dataKey="pi_band"
                  name={`${selectedPI}% Prediction Interval`}
                  stroke="none"
                  fill="url(#piBandGrad)"
                  connectNulls={false}
                />

                {/* Vertical Divider between History and Forecast */}
                {dividerLabel && (
                  <ReferenceLine
                    x={dividerLabel}
                    stroke="#0d9488"
                    strokeDasharray="4 4"
                    strokeWidth={1.5}
                    label={{
                      value: 'Forecast Origin →',
                      position: 'top',
                      fill: '#0f766e',
                      fontSize: 11,
                      fontWeight: 700,
                    }}
                  />
                )}

                {/* Optional Markers for Reporting-System Changes */}
                {reportingMarkerLabels.map((lbl) => (
                  <ReferenceLine
                    key={lbl}
                    x={lbl}
                    stroke="#7c3aed"
                    strokeDasharray="3 3"
                    label={{
                      value: 'Reporting System Upgrade',
                      position: 'insideTopLeft',
                      fill: '#6d28d9',
                      fontSize: 10,
                      fontWeight: 600,
                    }}
                  />
                ))}

                {/* Historical Line (Deep Navy) */}
                <Line
                  type="monotone"
                  dataKey="actual"
                  name="Historical Recorded Concerns"
                  stroke="#0f172a"
                  strokeWidth={2.5}
                  dot={{ r: 2.5, fill: '#0f172a' }}
                  activeDot={{ r: 5 }}
                />

                {/* Future Forecast Line (Teal) */}
                <Line
                  type="monotone"
                  dataKey="forecast"
                  name={`Forecast (${forecastData.model_name})`}
                  stroke="#0d9488"
                  strokeWidth={3}
                  strokeDasharray="6 3"
                  dot={{ r: 3.5, fill: '#0d9488', stroke: '#ffffff', strokeWidth: 1.5 }}
                  activeDot={{ r: 6 }}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          <div className="mt-3 flex flex-wrap items-center justify-between gap-2 border-t border-slate-100 pt-3 text-xs text-slate-500">
            <div className="flex items-center gap-4">
              <label className="inline-flex items-center gap-1.5 cursor-pointer">
                <input
                  type="checkbox"
                  checked={showReportingChangeMarkers}
                  onChange={(e) => setShowReportingChangeMarkers(e.target.checked)}
                  className="rounded border-slate-300"
                />
                <span>Show reporting-system & training change markers (Sep 2023)</span>
              </label>
            </div>
            <div>
              Engine: <strong className="text-slate-700">{forecastData.model_version}</strong> • Runtime:{' '}
              <strong className="text-slate-700">{forecastData.runtime_ms} ms</strong>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Deterministic Planning Insights + Warnings */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2 bg-gradient-to-br from-slate-900 to-slate-800 text-white border-none">
          <CardHeader className="pb-2">
            <div className="flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-teal-400" />
              <CardTitle className="text-white text-base">
                Deterministic Operational Planning Insights
              </CardTitle>
              <Badge variant="teal" className="ml-auto bg-teal-500/20 text-teal-200 border-teal-400/30">
                Calculated from Model Output
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-2.5 text-xs sm:text-sm text-slate-200">
            {forecastData.planning_insights.map((insight, idx) => (
              <div key={idx} className="flex items-start gap-2.5 rounded-xl bg-white/5 p-3 border border-white/10">
                <CheckCircle2 className="h-4 w-4 text-teal-400 shrink-0 mt-0.5" />
                <p className="leading-relaxed">{insight}</p>
              </div>
            ))}
          </CardContent>
        </Card>

        <Card className="border-amber-200 bg-amber-50/50">
          <CardHeader className="pb-2">
            <div className="flex items-center gap-2">
              <ShieldAlert className="h-4 w-4 text-amber-700" />
              <CardTitle className="text-amber-950 text-base">
                Safeguarding Governance & Interpretation
              </CardTitle>
            </div>
          </CardHeader>
          <CardContent className="space-y-2.5 text-xs text-amber-900 leading-relaxed">
            <p>
              <strong>Aggregate Workload Planning Only:</strong> This platform forecasts college-wide and provision-level volume of recorded safeguarding concerns. It <strong>never</strong> scores individual students or automates casework decisions.
            </p>
            <p>
              <strong>Statutory Floor Protection:</strong> Lower predicted demand during summer or holiday periods must <strong>never</strong> be used to reduce statutory DSL cover or pastoral responsiveness.
            </p>
            {forecastData.warnings.map((w, i) => (
              <div key={i} className="rounded-lg bg-amber-100/80 p-2.5 text-[11px] font-medium text-amber-950 border border-amber-300">
                {w}
              </div>
            ))}
          </CardContent>
        </Card>
      </div>

      {/* Secondary Visualisations (2x2 Grid) */}
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2 dashboard-section-contain">
        {/* A. Monthly Forecast Bar Chart */}
        <Card>
          <CardHeader>
            <CardTitle>A. Monthly Forecast Volume & Academic Term Profile</CardTitle>
            <CardDescription>
              Whole-number predicted safeguarding concerns by month over the next {selectedHorizon} months ({summary.target_academic_year}).
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[270px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={forecastData.monthly_forecasts} margin={{ top: 10, right: 16, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                  <XAxis dataKey="period_label" tick={{ fontSize: 11, fill: '#475569' }} />
                  <YAxis tick={{ fontSize: 11, fill: '#475569' }} axisLine={false} tickLine={false} />
                  <Tooltip
                    formatter={(val: any, name: string) => [val, name === 'point_forecast' ? 'Predicted Concerns' : name]}
                    labelStyle={{ fontWeight: 700, color: '#0f172a' }}
                  />
                  <Bar dataKey="point_forecast" name="Predicted Concerns" radius={[6, 6, 0, 0]}>
                    {forecastData.monthly_forecasts.map((entry, idx) => (
                      <Cell
                        key={`cell-${idx}`}
                        fill={
                          entry.point_forecast === summary.peak_demand_count
                            ? '#7c3aed'
                            : entry.academic_term === 'Autumn Term'
                            ? '#0d9488'
                            : entry.academic_term === 'Spring Term'
                            ? '#2563eb'
                            : entry.academic_term === 'Summer Term'
                            ? '#3b82f6'
                            : '#94a3b8'
                        }
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="mt-2 flex flex-wrap items-center gap-3 text-[11px] text-slate-500">
              <span className="inline-flex items-center gap-1"><span className="h-2.5 w-2.5 rounded-xs bg-purple-600" /> Peak Demand Month</span>
              <span className="inline-flex items-center gap-1"><span className="h-2.5 w-2.5 rounded-xs bg-teal-600" /> Autumn Term</span>
              <span className="inline-flex items-center gap-1"><span className="h-2.5 w-2.5 rounded-xs bg-blue-600" /> Spring Term</span>
              <span className="inline-flex items-center gap-1"><span className="h-2.5 w-2.5 rounded-xs bg-blue-400" /> Summer Term</span>
              <span className="inline-flex items-center gap-1"><span className="h-2.5 w-2.5 rounded-xs bg-slate-400" /> Summer Period</span>
            </div>
          </CardContent>
        </Card>

        {/* B. Historical Academic-Year Comparison */}
        <Card>
          <CardHeader>
            <CardTitle>B. Historical Academic-Year Comparison</CardTitle>
            <CardDescription>
              Total recorded concerns and rate per 1,000 enrolled students across historical academic years.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[270px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={historicalStats.academic_year_comparisons} margin={{ top: 10, right: 18, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                  <XAxis dataKey="academic_year" tick={{ fontSize: 11, fill: '#475569' }} />
                  <YAxis yAxisId="left" tick={{ fontSize: 11, fill: '#475569' }} axisLine={false} tickLine={false} />
                  <YAxis
                    yAxisId="right"
                    orientation="right"
                    tick={{ fontSize: 11, fill: '#0d9488' }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <Tooltip />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Bar
                    yAxisId="left"
                    dataKey="total_concerns"
                    name="Total Annual Concerns"
                    fill="#1e293b"
                    radius={[6, 6, 0, 0]}
                    barSize={38}
                  />
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="concerns_per_1000"
                    name="Concerns per 1,000 Students"
                    stroke="#0d9488"
                    strokeWidth={2.5}
                    dot={{ r: 4, fill: '#0d9488' }}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* C. Forecast Demand by Provision */}
        <Card>
          <CardHeader>
            <CardTitle>C. Forecast Demand by Provision ({summary.target_academic_year})</CardTitle>
            <CardDescription>
              Predicted safeguarding volume, planning range, and incidence per 1,000 learners by college provision.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3.5">
              {forecastData.provision_forecasts.map((pf, idx) => (
                <div key={pf.provision} className="rounded-xl border border-slate-200/80 p-3.5 bg-slate-50/50">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span
                        className="h-3 w-3 rounded-full"
                        style={{ backgroundColor: PROVISION_COLORS[idx % PROVISION_COLORS.length] }}
                      />
                      <span className="font-bold text-sm text-slate-900">{pf.provision}</span>
                      <Badge variant="slate">{pf.share_pct}% share</Badge>
                    </div>
                    <div className="text-right">
                      <span className="font-display text-lg font-extrabold text-slate-900">
                        {pf.predicted_total.toLocaleString()}
                      </span>
                      <span className="ml-1.5 text-xs text-slate-500">
                        ({pf.lower_bound.toLocaleString()}–{pf.upper_bound.toLocaleString()})
                      </span>
                    </div>
                  </div>
                  <div className="mt-2 h-2 w-full rounded-full bg-slate-200 overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-500"
                      style={{
                        width: `${Math.min(100, pf.share_pct)}%`,
                        backgroundColor: PROVISION_COLORS[idx % PROVISION_COLORS.length],
                      }}
                    />
                  </div>
                  <div className="mt-2 flex items-center justify-between text-xs text-slate-500">
                    <span>
                      Forecast Rate: <strong className="text-slate-700">{pf.concerns_per_1000} per 1,000 students</strong>
                    </span>
                    <span>
                      Prior Period: {pf.historical_prior_year.toLocaleString()} ({pf.yoy_change_pct >= 0 ? `+${pf.yoy_change_pct}%` : `${pf.yoy_change_pct}%`})
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* D. Safeguarding Category Distribution */}
        <Card>
          <CardHeader>
            <CardTitle>D. Forecast Safeguarding Category Distribution</CardTitle>
            <CardDescription>
              Estimated breakdown by broad safeguarding category with data-sufficiency verification.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {forecastData.category_forecasts.map((cf, idx) => (
                <div key={cf.safeguarding_category} className="space-y-1.5">
                  <div className="flex items-center justify-between text-xs sm:text-sm">
                    <div className="flex items-center gap-2">
                      <span
                        className="h-2.5 w-2.5 rounded-full shrink-0"
                        style={{ backgroundColor: CATEGORY_COLORS[idx % CATEGORY_COLORS.length] }}
                      />
                      <span className="font-semibold text-slate-800">{cf.safeguarding_category}</span>
                      {!cf.reliable_estimation && (
                        <Badge variant="amber" title={cf.note || ''}>
                          Low Base Count
                        </Badge>
                      )}
                    </div>
                    <div className="text-right">
                      <span className="font-bold text-slate-900">{cf.predicted_total.toLocaleString()}</span>
                      <span className="ml-1.5 text-xs text-slate-500">
                        [{cf.lower_bound}–{cf.upper_bound}] • {cf.share_pct}%
                      </span>
                    </div>
                  </div>
                  <div className="h-2 w-full rounded-full bg-slate-100 overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${Math.min(100, cf.share_pct)}%`,
                        backgroundColor: CATEGORY_COLORS[idx % CATEGORY_COLORS.length],
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

