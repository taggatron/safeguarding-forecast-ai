import React from 'react';
import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import {
  Sliders,
  RotateCcw,
  Lock,
  TrendingUp,
  TrendingDown,
  Users,
  Calendar,
  Activity,
  Info,
  CheckCircle2,
  AlertCircle,
} from 'lucide-react';
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  Badge,
  Button,
} from '../components/ui/primitives';
import type {
  DatasetSummaryResponse,
  ForecastResponse,
  ModelAvailabilityItem,
  ScenarioAssumptions,
} from '../types/api';

interface ForecastExplorerPageProps {
  datasetSummary: DatasetSummaryResponse;
  forecastData: ForecastResponse;
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
  scenario: ScenarioAssumptions;
  onChangeScenario: (s: ScenarioAssumptions) => void;
  onResetScenario: () => void;
  isLoading: boolean;
}

export const ForecastExplorerPage: React.FC<ForecastExplorerPageProps> = ({
  datasetSummary,
  forecastData,
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
  scenario,
  onChangeScenario,
  onResetScenario,
  isLoading,
}) => {
  const caps = forecastData.scenario_capabilities;
  const adjSummary = forecastData.annual_summary;
  const baseSummary = forecastData.baseline_annual_summary || adjSummary;

  const deltaCount = adjSummary.predicted_total_concerns - baseSummary.predicted_total_concerns;
  const deltaPct =
    baseSummary.predicted_total_concerns > 0
      ? Number(((deltaCount / baseSummary.predicted_total_concerns) * 100).toFixed(1))
      : 0;

  const comparisonChartData = forecastData.monthly_forecasts.map((f, idx) => {
    const basePoint = forecastData.baseline_monthly_forecasts?.[idx] || f;
    const lb = selectedPI === 95 && f.lower_95 != null ? f.lower_95 : f.lower_80;
    const ub = selectedPI === 95 && f.upper_95 != null ? f.upper_95 : f.upper_80;
    return {
      period_start: f.period_start,
      label: f.period_label,
      academic_term: f.academic_term,
      baseline_forecast: basePoint.point_forecast,
      scenario_forecast: f.point_forecast,
      scenario_pi: [lb, ub],
      lower_bound: lb,
      upper_bound: ub,
      delta: f.point_forecast - basePoint.point_forecast,
      enrolment_assumed: f.enrolled_students_assumed,
      teaching_days_assumed: f.teaching_days_assumed,
      attendance_assumed: f.attendance_rate_assumed,
      rate_per_1000: f.concerns_per_1000_forecast,
    };
  });

  const applyPreset = (preset: ScenarioAssumptions) => {
    onChangeScenario(preset);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between bg-white p-5 rounded-2xl border border-slate-200/80 shadow-xs">
        <div>
          <div className="flex items-center gap-2.5">
            <h1 className="fluid-title font-display font-extrabold text-slate-900">
              Forecast Explorer & Scenario Lab
            </h1>
            <Badge variant="teal">Model-Validated Scenario Simulation</Badge>
          </div>
          <p className="mt-1 text-sm text-slate-600">
            Test operational demand scenarios (enrolment growth, teaching-day adjustments, and attendance shifts) and compare against the baseline forecast.
          </p>
        </div>
        <Button variant="outline" size="md" onClick={onResetScenario}>
          <RotateCcw className="h-4 w-4" />
          Reset Scenario to Baseline
        </Button>
      </div>

      {/* Main Layout: Left Control Panel + Right Comparison Workspace */}
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-12">
        {/* Scenario Controls Sidebar (4 cols) */}
        <div className="space-y-4 xl:col-span-4">
          <Card>
            <CardHeader className="border-b border-slate-100 pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center gap-2">
                  <Sliders className="h-4 w-4 text-teal-600" />
                  Scenario & Model Controls
                </CardTitle>
                {isLoading && (
                  <Badge variant="teal" className="animate-pulse">
                    Updating...
                  </Badge>
                )}
              </div>
              <CardDescription>
                Controls are automatically enabled or disabled based on the mathematical predictors supported by the active model.
              </CardDescription>
            </CardHeader>

            <CardContent className="space-y-4 pt-4">
              {/* Model Selector */}
              <div>
                <label htmlFor="explorer-model-select" className="block text-xs font-semibold text-slate-700 mb-1.5">
                  Forecasting Engine
                </label>
                <select
                  id="explorer-model-select"
                  value={selectedModel}
                  onChange={(e) => onSelectModel(e.target.value)}
                  className="w-full rounded-xl border border-slate-300 bg-slate-50 px-3 py-2 text-xs sm:text-sm font-medium text-slate-900 focus:border-teal-600 focus:outline-none"
                >
                  {models
                    .filter((m) => m.model_id !== 'chronos_2')
                    .map((m) => (
                      <option key={m.model_id} value={m.model_id} disabled={!m.available}>
                        {m.name} {!m.available ? '(Not Available)' : ''}
                      </option>
                    ))}
                </select>
              </div>

              {/* Horizon & PI */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                    Forecast Horizon
                  </label>
                  <div className="flex rounded-xl bg-slate-100 p-1">
                    {[3, 6, 12].map((h) => (
                      <button
                        key={h}
                        onClick={() => onSelectHorizon(h)}
                        className={`flex-1 rounded-lg py-1.5 text-xs font-semibold transition-all cursor-pointer ${
                          selectedHorizon === h
                            ? 'bg-slate-900 text-white shadow-xs'
                            : 'text-slate-600 hover:text-slate-900'
                        }`}
                      >
                        {h}m
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                    Prediction Interval
                  </label>
                  <div className="flex rounded-xl bg-slate-100 p-1">
                    {[80, 95].map((pi) => (
                      <button
                        key={pi}
                        onClick={() => onSelectPI(pi)}
                        className={`flex-1 rounded-lg py-1.5 text-xs font-semibold transition-all cursor-pointer ${
                          selectedPI === pi
                            ? 'bg-teal-600 text-white shadow-xs'
                            : 'text-slate-600 hover:text-slate-900'
                        }`}
                      >
                        {pi}%
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* Provision & Category Scope */}
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                <div>
                  <label htmlFor="explorer-prov" className="block text-xs font-semibold text-slate-700 mb-1">
                    Provision Filter
                  </label>
                  <select
                    id="explorer-prov"
                    value={selectedProvision}
                    onChange={(e) => onSelectProvision(e.target.value)}
                    className="w-full rounded-xl border border-slate-300 bg-slate-50 px-2.5 py-1.5 text-xs font-medium text-slate-800"
                  >
                    <option value="All">All Provisions</option>
                    {datasetSummary.provisions.map((p) => (
                      <option key={p} value={p}>
                        {p}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label htmlFor="explorer-cat" className="block text-xs font-semibold text-slate-700 mb-1">
                    Category Filter
                  </label>
                  <select
                    id="explorer-cat"
                    value={selectedCategory}
                    onChange={(e) => onSelectCategory(e.target.value)}
                    className="w-full rounded-xl border border-slate-300 bg-slate-50 px-2.5 py-1.5 text-xs font-medium text-slate-800"
                  >
                    <option value="All">All Categories</option>
                    {datasetSummary.categories.map((c) => (
                      <option key={c} value={c}>
                        {c}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <hr className="border-slate-100" />

              {/* Control 1: Student Population Growth Assumption */}
              <div
                className={`rounded-xl border p-3.5 transition-all ${
                  caps.supports_enrolment_growth
                    ? 'border-slate-200 bg-slate-50/60'
                    : 'border-slate-200 bg-slate-100/70 opacity-75'
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
                    <Users className="h-3.5 w-3.5 text-teal-600" />
                    Student Enrolment Growth
                  </span>
                  {caps.supports_enrolment_growth ? (
                    <Badge variant={scenario.enrolment_growth_pct !== 0 ? 'teal' : 'slate'}>
                      {scenario.enrolment_growth_pct > 0
                        ? `+${scenario.enrolment_growth_pct}%`
                        : `${scenario.enrolment_growth_pct}%`}
                    </Badge>
                  ) : (
                    <Badge variant="amber">
                      <Lock className="h-3 w-3" /> Disabled
                    </Badge>
                  )}
                </div>
                <input
                  type="range"
                  aria-label="Student Enrolment Growth Percentage"
                  min={-20}
                  max={30}
                  step={1}
                  disabled={!caps.supports_enrolment_growth}
                  value={scenario.enrolment_growth_pct}
                  onChange={(e) =>
                    onChangeScenario({
                      ...scenario,
                      enrolment_growth_pct: Number(e.target.value),
                    })
                  }
                  className="w-full my-1.5 cursor-pointer disabled:cursor-not-allowed"
                />
                <div className="flex justify-between text-[10px] text-slate-400 mb-1.5">
                  <span>-20%</span>
                  <span>0% (Baseline)</span>
                  <span>+30%</span>
                </div>
                <p className="text-[11px] text-slate-600 leading-snug">
                  {caps.enrolment_explanation}
                </p>
              </div>

              {/* Control 2: Teaching-Day Assumptions */}
              <div
                className={`rounded-xl border p-3.5 transition-all ${
                  caps.supports_teaching_days
                    ? 'border-slate-200 bg-slate-50/60'
                    : 'border-slate-200 bg-slate-100/70 opacity-75'
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
                    <Calendar className="h-3.5 w-3.5 text-blue-600" />
                    Monthly Teaching Days Shift
                  </span>
                  {caps.supports_teaching_days ? (
                    <Badge variant={scenario.teaching_days_delta !== 0 ? 'blue' : 'slate'}>
                      {scenario.teaching_days_delta > 0
                        ? `+${scenario.teaching_days_delta} days/mo`
                        : `${scenario.teaching_days_delta} days/mo`}
                    </Badge>
                  ) : (
                    <Badge variant="amber">
                      <Lock className="h-3 w-3" /> Disabled
                    </Badge>
                  )}
                </div>
                <input
                  type="range"
                  aria-label="Monthly Teaching Days Adjustment"
                  min={-4}
                  max={4}
                  step={1}
                  disabled={!caps.supports_teaching_days}
                  value={scenario.teaching_days_delta}
                  onChange={(e) =>
                    onChangeScenario({
                      ...scenario,
                      teaching_days_delta: Number(e.target.value),
                    })
                  }
                  className="w-full my-1.5 cursor-pointer disabled:cursor-not-allowed"
                />
                <div className="flex justify-between text-[10px] text-slate-400 mb-1.5">
                  <span>-4 days/mo</span>
                  <span>0 (Standard Calendar)</span>
                  <span>+4 days/mo</span>
                </div>
                <p className="text-[11px] text-slate-600 leading-snug">
                  {caps.teaching_days_explanation}
                </p>
              </div>

              {/* Control 3: Future Attendance Scenario */}
              <div
                className={`rounded-xl border p-3.5 transition-all ${
                  caps.supports_attendance_scenario
                    ? 'border-slate-200 bg-slate-50/60'
                    : 'border-slate-200 bg-slate-100/70 opacity-75'
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
                    <Activity className="h-3.5 w-3.5 text-purple-600" />
                    Future Attendance Scenario
                  </span>
                  {caps.supports_attendance_scenario ? (
                    <Badge variant={scenario.attendance_rate_delta_pp !== 0 ? 'purple' : 'slate'}>
                      {scenario.attendance_rate_delta_pp > 0
                        ? `+${scenario.attendance_rate_delta_pp} pp`
                        : `${scenario.attendance_rate_delta_pp} pp`}
                    </Badge>
                  ) : (
                    <Badge variant="amber">
                      <Lock className="h-3 w-3" /> Disabled
                    </Badge>
                  )}
                </div>
                <input
                  type="range"
                  aria-label="Future Attendance Rate Shift"
                  min={-10}
                  max={6}
                  step={0.5}
                  disabled={!caps.supports_attendance_scenario}
                  value={scenario.attendance_rate_delta_pp}
                  onChange={(e) =>
                    onChangeScenario({
                      ...scenario,
                      attendance_rate_delta_pp: Number(e.target.value),
                    })
                  }
                  className="w-full my-1.5 cursor-pointer disabled:cursor-not-allowed"
                />
                <div className="flex justify-between text-[10px] text-slate-400 mb-1.5">
                  <span>-10 pp (Dip)</span>
                  <span>0 pp</span>
                  <span>+6 pp</span>
                </div>
                <p className="text-[11px] text-slate-600 leading-snug">
                  {caps.attendance_explanation}
                </p>
              </div>

              {/* Quick Scenario Presets */}
              <div className="pt-1">
                <span className="block text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-2">
                  Quick Scenario Presets
                </span>
                <div className="grid grid-cols-1 gap-1.5">
                  <button
                    onClick={() =>
                      applyPreset({
                        enrolment_growth_pct: 8,
                        teaching_days_delta: 0,
                        attendance_rate_delta_pp: 0,
                        reporting_change_assumed: null,
                      })
                    }
                    className="text-left rounded-lg border border-slate-200 px-3 py-2 text-xs hover:bg-teal-50/60 hover:border-teal-300 transition-colors cursor-pointer"
                  >
                    <strong className="text-slate-800">High 16–18 Cohort Growth:</strong>{' '}
                    <span className="text-slate-500">+8% student enrolment</span>
                  </button>
                  <button
                    onClick={() =>
                      applyPreset({
                        enrolment_growth_pct: 5,
                        teaching_days_delta: 1,
                        attendance_rate_delta_pp: -3.5,
                        reporting_change_assumed: null,
                      })
                    }
                    className="text-left rounded-lg border border-slate-200 px-3 py-2 text-xs hover:bg-purple-50/60 hover:border-purple-300 transition-colors cursor-pointer"
                  >
                    <strong className="text-slate-800">Winter Attendance & Welfare Pressure:</strong>{' '}
                    <span className="text-slate-500">+5% enrolment, -3.5pp attendance</span>
                  </button>
                  <button
                    onClick={() =>
                      applyPreset({
                        enrolment_growth_pct: -6,
                        teaching_days_delta: 0,
                        attendance_rate_delta_pp: 1.5,
                        reporting_change_assumed: null,
                      })
                    }
                    className="text-left rounded-lg border border-slate-200 px-3 py-2 text-xs hover:bg-blue-50/60 hover:border-blue-300 transition-colors cursor-pointer"
                  >
                    <strong className="text-slate-800">Demographic Contraction & High Attendance:</strong>{' '}
                    <span className="text-slate-500">-6% enrolment, +1.5pp attendance</span>
                  </button>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Right Comparison Workspace (8 cols) */}
        <div className="space-y-5 xl:col-span-8">
          {/* Top Comparison Cards */}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <Card className="bg-slate-50/70">
              <CardHeader className="pb-1">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Baseline Forecast ({selectedHorizon}m)
                </span>
              </CardHeader>
              <CardContent>
                <div className="font-display text-2xl font-extrabold text-slate-800">
                  {baseSummary.predicted_total_concerns.toLocaleString()}
                </div>
                <p className="mt-1 text-xs text-slate-500">
                  {selectedPI}% Range: {baseSummary.lower_bound.toLocaleString()}–{baseSummary.upper_bound.toLocaleString()}
                </p>
              </CardContent>
            </Card>

            <Card className="border-teal-300 bg-teal-50/30">
              <CardHeader className="pb-1">
                <span className="text-xs font-semibold uppercase tracking-wider text-teal-800">
                  Adjusted Scenario Forecast
                </span>
              </CardHeader>
              <CardContent>
                <div className="font-display text-2xl font-extrabold text-teal-950">
                  {adjSummary.predicted_total_concerns.toLocaleString()}
                </div>
                <p className="mt-1 text-xs text-teal-800">
                  {selectedPI}% Range: {adjSummary.lower_bound.toLocaleString()}–{adjSummary.upper_bound.toLocaleString()}
                </p>
              </CardContent>
            </Card>

            <Card className={deltaCount !== 0 ? 'border-purple-300 bg-purple-50/30' : ''}>
              <CardHeader className="pb-1">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Scenario Volume Impact
                </span>
              </CardHeader>
              <CardContent>
                <div className="flex items-baseline gap-2">
                  <span className="font-display text-2xl font-extrabold text-slate-900">
                    {deltaCount > 0 ? `+${deltaCount.toLocaleString()}` : deltaCount.toLocaleString()}
                  </span>
                  <Badge variant={deltaCount > 0 ? 'purple' : deltaCount < 0 ? 'teal' : 'slate'}>
                    {deltaPct > 0 ? `+${deltaPct}%` : `${deltaPct}%`}
                  </Badge>
                </div>
                <p className="mt-1 text-xs text-slate-500">
                  Assumed Cohort: {adjSummary.enrolled_students_assumed.toLocaleString()} students ({adjSummary.predicted_concerns_per_1000}/1k)
                </p>
              </CardContent>
            </Card>
          </div>

          {/* Baseline vs Adjusted Scenario Chart */}
          <Card>
            <CardHeader>
              <CardTitle>Baseline vs. Adjusted Scenario Forecast Trajectory</CardTitle>
              <CardDescription>
                Comparing unadjusted baseline prediction (navy dashed) against the scenario-adjusted model prediction (solid teal with {selectedPI}% interval band).
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[330px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={comparisonChartData} margin={{ top: 12, right: 20, left: 0, bottom: 8 }}>
                    <defs>
                      <linearGradient id="scenBandGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#0d9488" stopOpacity={0.25} />
                        <stop offset="95%" stopColor="#0d9488" stopOpacity={0.05} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                    <XAxis dataKey="label" tick={{ fontSize: 11, fill: '#475569' }} />
                    <YAxis tick={{ fontSize: 11, fill: '#475569' }} axisLine={false} tickLine={false} />
                    <Tooltip
                      content={({ active, payload, label }) => {
                        if (!active || !payload || payload.length === 0) return null;
                        const r = payload[0].payload;
                        return (
                          <div className="rounded-xl border border-slate-200 bg-white p-3.5 shadow-lg text-xs space-y-1">
                            <div className="font-bold text-slate-900 border-b border-slate-100 pb-1 mb-1">
                              {label} ({r.academic_term})
                            </div>
                            <div className="flex justify-between gap-6">
                              <span className="text-slate-600">Baseline Forecast:</span>
                              <span className="font-semibold text-slate-900">{r.baseline_forecast} concerns</span>
                            </div>
                            <div className="flex justify-between gap-6">
                              <span className="text-teal-700 font-medium">Scenario Forecast:</span>
                              <span className="font-bold text-teal-800">{r.scenario_forecast} concerns</span>
                            </div>
                            <div className="flex justify-between gap-6">
                              <span className="text-purple-700">Scenario Difference:</span>
                              <span className="font-bold text-purple-800">
                                {r.delta >= 0 ? `+${r.delta}` : r.delta} concerns
                              </span>
                            </div>
                            <div className="flex justify-between gap-6 text-slate-500 pt-1 border-t border-slate-100">
                              <span>Scenario {selectedPI}% Interval:</span>
                              <span>{r.lower_bound} – {r.upper_bound}</span>
                            </div>
                          </div>
                        );
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <Area
                      type="monotone"
                      dataKey="scenario_pi"
                      name={`Scenario ${selectedPI}% Interval`}
                      stroke="none"
                      fill="url(#scenBandGrad)"
                    />
                    <Line
                      type="monotone"
                      dataKey="baseline_forecast"
                      name="Baseline Forecast"
                      stroke="#1e293b"
                      strokeWidth={2}
                      strokeDasharray="5 5"
                      dot={{ r: 3, fill: '#1e293b' }}
                    />
                    <Line
                      type="monotone"
                      dataKey="scenario_forecast"
                      name="Adjusted Scenario Forecast"
                      stroke="#0d9488"
                      strokeWidth={3}
                      dot={{ r: 4, fill: '#0d9488' }}
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* Practical Implications for Planning & Monthly Table */}
          <Card>
            <CardHeader>
              <CardTitle>Practical Operational Implications & Monthly Scenario Breakdown</CardTitle>
              <CardDescription>
                Month-by-month comparison of baseline vs. scenario safeguarding concern volumes and assumed cohort parameters.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="rounded-xl bg-slate-50 p-3.5 border border-slate-200/80 text-xs text-slate-700 space-y-1.5">
                <div className="font-bold text-slate-900 flex items-center gap-1.5">
                  <Info className="h-4 w-4 text-teal-600" />
                  Resource Planning Implications
                </div>
                {deltaCount === 0 ? (
                  <p>
                    Scenario assumptions currently match the baseline forecast (or the selected model does not adjust for the modified covariate). Use the sliders on the left to test cohort growth, calendar changes, or attendance shifts.
                  </p>
                ) : deltaCount > 0 ? (
                  <p>
                    Under this scenario, safeguarding triage demand increases by <strong>+{deltaCount} recorded concerns</strong> ({deltaPct > 0 ? `+${deltaPct}%` : `${deltaPct}%`}) over the {selectedHorizon}-month horizon, with peak monthly pressure reaching <strong>{adjSummary.peak_demand_count} concerns in {adjSummary.peak_demand_month}</strong>. Leadership may consider proactive pastoral duty allocation and early-intervention mental health capacity ahead of {adjSummary.peak_demand_period_description}.
                  </p>
                ) : (
                  <p>
                    Under this scenario, predicted recorded volume is <strong>{deltaCount} concerns</strong> ({deltaPct}%) relative to baseline. Note: In accordance with South Devon College safeguarding governance, lower scenario projections must <strong>never</strong> be used to reduce statutory DSL staffing below baseline safety thresholds.
                  </p>
                )}
              </div>

              {/* Accessible Responsive Table */}
              <div
                tabIndex={0}
                role="region"
                aria-labelledby="scenario-table-caption"
                className="table-wrapper rounded-xl border border-slate-200"
              >
                <table className="responsive-stack text-left text-xs">
                  <caption id="scenario-table-caption" className="sr-only">
                    Monthly comparison of baseline forecast and scenario-adjusted forecast
                  </caption>
                  <thead className="bg-slate-100 text-slate-700 font-semibold border-b border-slate-200">
                    <tr>
                      <th className="py-2.5 px-3">Month</th>
                      <th className="py-2.5 px-3">Term</th>
                      <th className="py-2.5 px-3 text-right">Baseline</th>
                      <th className="py-2.5 px-3 text-right">Scenario</th>
                      <th className="py-2.5 px-3 text-right">Delta</th>
                      <th className="py-2.5 px-3 text-right">{selectedPI}% Range</th>
                      <th className="py-2.5 px-3 text-right">Enrolment</th>
                      <th className="py-2.5 px-3 text-right">Teaching Days</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {comparisonChartData.map((row) => (
                      <tr key={row.period_start} className="hover:bg-slate-50/80">
                        <th className="py-2 px-3 font-semibold text-slate-900">{row.label}</th>
                        <td data-label="Term" className="py-2 px-3 text-slate-600">{row.academic_term}</td>
                        <td data-label="Baseline" className="py-2 px-3 text-right font-medium text-slate-700">
                          {row.baseline_forecast}
                        </td>
                        <td data-label="Scenario" className="py-2 px-3 text-right font-bold text-teal-800">
                          {row.scenario_forecast}
                        </td>
                        <td
                          data-label="Delta"
                          className={`py-2 px-3 text-right font-semibold ${
                            row.delta > 0
                              ? 'text-purple-700'
                              : row.delta < 0
                              ? 'text-teal-700'
                              : 'text-slate-400'
                          }`}
                        >
                          {row.delta > 0 ? `+${row.delta}` : row.delta}
                        </td>
                        <td data-label="Range" className="py-2 px-3 text-right text-slate-600">
                          {row.lower_bound} – {row.upper_bound}
                        </td>
                        <td data-label="Enrolment" className="py-2 px-3 text-right text-slate-600">
                          {row.enrolment_assumed.toLocaleString()}
                        </td>
                        <td data-label="Teaching Days" className="py-2 px-3 text-right text-slate-600">
                          {row.teaching_days_assumed}d
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

