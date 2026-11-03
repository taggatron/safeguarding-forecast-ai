import React, { useState } from 'react';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import {
  Trophy,
  CheckCircle2,
  XCircle,
  Clock,
  ShieldCheck,
  AlertTriangle,
  Scale,
  Database,
  Cpu,
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
  BacktestResponse,
  DatasetSummaryResponse,
  ModelAvailabilityItem,
} from '../types/api';
import countBenchmarkSnapshot from '../lib/countBenchmarkSnapshot.json';

interface ModelComparisonPageProps {
  datasetSummary: DatasetSummaryResponse;
  backtestData: BacktestResponse;
  modelsAvailability: ModelAvailabilityItem[];
  selectedProvision: string;
  onSelectProvision: (p: string) => void;
  selectedCategory: string;
  onSelectCategory: (c: string) => void;
  backtestHorizon: number;
  onSelectBacktestHorizon: (h: number) => void;
  onSelectActiveModel: (m: string) => void;
  activeModelId: string;
  isLoading: boolean;
}

const MODEL_COLORS: Record<string, string> = {
  seasonal_naive: '#64748b',
  statistical_count: '#2563eb',
  lightgbm_mlforecast: '#0d9488',
  timesfm_3: '#7c3aed',
};

export const ModelComparisonPage: React.FC<ModelComparisonPageProps> = ({
  datasetSummary,
  backtestData,
  modelsAvailability,
  selectedProvision,
  onSelectProvision,
  selectedCategory,
  onSelectCategory,
  backtestHorizon,
  onSelectBacktestHorizon,
  onSelectActiveModel,
  activeModelId,
  isLoading,
}) => {
  const [visibleModels, setVisibleModels] = useState<Record<string, boolean>>({
    seasonal_naive: true,
    statistical_count: true,
    lightgbm_mlforecast: true,
    timesfm_3: true,
  });

  const toggleModelLine = (mId: string) => {
    setVisibleModels((prev) => ({ ...prev, [mId]: !prev[mId] }));
  };

  // Transform historical backtest points for Recharts
  const chartData = backtestData.historical_backtest_series.map((pt) => ({
    period_start: pt.period_start,
    label: pt.period_label,
    academic_year: pt.academic_year,
    fold_origin: pt.fold_origin,
    actual: pt.actual_count,
    seasonal_naive: pt.predictions['seasonal_naive'] ?? null,
    statistical_count: pt.predictions['statistical_count'] ?? null,
    lightgbm_mlforecast: pt.predictions['lightgbm_mlforecast'] ?? null,
    timesfm_3: pt.predictions['timesfm_3'] ?? null,
  }));

  const availableMetrics = backtestData.metrics.filter((m) => m.available && m.mae != null);

  return (
    <div className="space-y-6">
      {/* Header & Slice Controls */}
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between bg-white p-5 rounded-2xl border border-slate-200/80 shadow-xs">
        <div>
          <div className="flex items-center gap-2.5">
            <h1 className="fluid-title font-display font-extrabold text-slate-900">
              Model Benchmarking & Rolling-Origin Validation
            </h1>
            {isLoading && (
              <Badge variant="teal" className="animate-pulse">
                Running Backtest...
              </Badge>
            )}
          </div>
          <p className="mt-1 text-sm text-slate-600">
            Rigorous out-of-sample evaluation across rolling historical forecast origins. Random train/test splitting is never used.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2.5">
          <div>
            <label htmlFor="bt-prov" className="block text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-1">
              Provision Scope
            </label>
            <select
              id="bt-prov"
              value={selectedProvision}
              onChange={(e) => onSelectProvision(e.target.value)}
              className="rounded-xl border border-slate-300 bg-slate-50 px-3 py-1.5 text-xs font-medium text-slate-800"
            >
              <option value="All">College-Wide (All Provisions)</option>
              {datasetSummary.provisions.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label htmlFor="bt-cat" className="block text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-1">
              Category Scope
            </label>
            <select
              id="bt-cat"
              value={selectedCategory}
              onChange={(e) => onSelectCategory(e.target.value)}
              className="rounded-xl border border-slate-300 bg-slate-50 px-3 py-1.5 text-xs font-medium text-slate-800"
            >
              <option value="All">All Safeguarding Categories</option>
              {datasetSummary.categories.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-1">
              Backtest Horizon
            </label>
            <div className="flex rounded-xl bg-slate-100 p-1">
              {[3, 6, 12].map((h) => (
                <button
                  key={h}
                  onClick={() => onSelectBacktestHorizon(h)}
                  className={`rounded-lg px-2.5 py-1 text-xs font-semibold transition-all cursor-pointer ${
                    backtestHorizon === h
                      ? 'bg-slate-900 text-white shadow-xs'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  {h}m
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Seasonality Sufficiency Warning if < 36 months */}
      {backtestData.seasonality_warning && (
        <div className="rounded-2xl border border-amber-300 bg-amber-50 p-4 flex items-start gap-3 text-xs sm:text-sm text-amber-950">
          <AlertTriangle className="h-5 w-5 text-amber-600 shrink-0 mt-0.5" />
          <div>
            <strong className="font-bold">Limited Historical Seasonal Cycles: </strong>
            {backtestData.seasonality_warning}
          </div>
        </div>
      )}

      {/* Lowest Error Model Banner */}
      <Card className="bg-gradient-to-r from-slate-900 via-slate-900 to-teal-950 text-white border-none">
        <CardContent className="p-5 flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex items-start gap-3.5">
            <div className="rounded-2xl bg-teal-500/20 p-3 text-teal-300 border border-teal-400/30">
              <Trophy className="h-6 w-6" />
            </div>
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-xs font-bold uppercase tracking-wider text-teal-300">
                  Lowest Historical Forecasting Error
                </span>
                <Badge variant="teal" className="bg-teal-500/20 text-teal-200 border-teal-400/30">
                  Empirical Winner on Active Slice
                </Badge>
              </div>
              <h2 className="mt-1 font-display text-xl font-extrabold text-white">
                {backtestData.best_model_name}
              </h2>
              <p className="mt-1 text-xs sm:text-sm text-slate-300 max-w-3xl leading-relaxed">
                {backtestData.best_model_rationale}
              </p>
            </div>
          </div>

          <div className="shrink-0">
            <button
              onClick={() => onSelectActiveModel(backtestData.best_model_id)}
              className={`rounded-xl px-4 py-2.5 text-xs sm:text-sm font-bold transition-all cursor-pointer ${
                activeModelId === backtestData.best_model_id
                  ? 'bg-teal-500 text-slate-950'
                  : 'bg-white text-slate-900 hover:bg-teal-50'
              }`}
            >
              {activeModelId === backtestData.best_model_id
                ? '✓ Currently Selected for Dashboard'
                : 'Select This Model for Forecasting'}
            </button>
          </div>
        </CardContent>
      </Card>

      {/* Comprehensive Benchmarking Table */}
      <Card>
        <CardHeader>
          <CardTitle>Rolling-Origin Backtesting Benchmark Matrix</CardTitle>
          <CardDescription>
            {backtestData.methodology_explanation}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div
            tabIndex={0}
            role="region"
            aria-labelledby="benchmark-table-caption"
            className="table-wrapper rounded-xl border border-slate-200"
          >
            <table className="responsive-stack text-left text-xs">
              <caption id="benchmark-table-caption" className="sr-only">
                Model comparison matrix showing MAE, WAPE, prediction interval coverage, runtime, and availability
              </caption>
              <thead className="bg-slate-100 text-slate-700 font-semibold border-b border-slate-200">
                <tr>
                  <th className="py-3 px-3.5">Forecasting Model</th>
                  <th className="py-3 px-3 text-center">Availability</th>
                  <th className="py-3 px-3 text-right">MAE (Concerns/Mo)</th>
                  <th className="py-3 px-3 text-right">WAPE (%)</th>
                  <th className="py-3 px-3 text-right">Annual Total MAE</th>
                  <th className="py-3 px-3 text-right">80% PI Coverage</th>
                  <th className="py-3 px-3 text-right">95% PI Coverage</th>
                  <th className="py-3 px-3 text-center">vs. Benchmark</th>
                  <th className="py-3 px-3 text-right">Avg Runtime</th>
                  <th className="py-3 px-3">Data & Governance Requirements</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {backtestData.metrics.map((m) => {
                  const isBest = m.model_id === backtestData.best_model_id;
                  const meta = modelsAvailability.find((x) => x.model_id === m.model_id);
                  return (
                    <tr
                      key={m.model_id}
                      className={
                        isBest
                          ? 'bg-teal-50/40 hover:bg-teal-50/70'
                          : 'hover:bg-slate-50/80'
                      }
                    >
                      <th className="py-3 px-3.5 font-bold text-slate-900">
                        <div className="flex items-center gap-2">
                          <span
                            className="h-3 w-3 rounded-full shrink-0"
                            style={{ backgroundColor: MODEL_COLORS[m.model_id] || '#334155' }}
                          />
                          <div>
                            <div className="flex items-center gap-1.5">
                              <span>{m.model_name}</span>
                              {isBest && <Badge variant="teal">Lowest MAE</Badge>}
                            </div>
                            <span className="text-[11px] font-normal text-slate-500">
                              {m.family}
                            </span>
                          </div>
                        </div>
                      </th>
                      <td data-label="Availability" className="py-3 px-3 text-center">
                        {m.available ? (
                          <Badge variant={m.model_id === 'timesfm_3' ? 'purple' : 'emerald'}>
                            <CheckCircle2 className="h-3 w-3" />
                            {m.status_label}
                          </Badge>
                        ) : (
                          <Badge variant="rose">
                            <XCircle className="h-3 w-3" />
                            {m.status_label}
                          </Badge>
                        )}
                      </td>
                      <td data-label="MAE" className="py-3 px-3 text-right font-display font-bold text-sm text-slate-900">
                        {m.mae != null ? m.mae.toFixed(2) : '—'}
                      </td>
                      <td data-label="WAPE" className="py-3 px-3 text-right font-semibold text-slate-800">
                        {m.wape_pct != null ? `${m.wape_pct.toFixed(1)}%` : '—'}
                      </td>
                      <td data-label="Annual MAE" className="py-3 px-3 text-right font-medium text-slate-700">
                        {m.annual_total_mae != null ? `±${m.annual_total_mae.toFixed(0)}` : '—'}
                      </td>
                      <td data-label="80% PI Coverage" className="py-3 px-3 text-right text-slate-700">
                        {m.pi_80_coverage_pct != null ? `${m.pi_80_coverage_pct.toFixed(1)}%` : '—'}
                      </td>
                      <td data-label="95% PI Coverage" className="py-3 px-3 text-right text-slate-700">
                        {m.pi_95_coverage_pct != null ? `${m.pi_95_coverage_pct.toFixed(1)}%` : '—'}
                      </td>
                      <td data-label="vs. Benchmark" className="py-3 px-3 text-center">
                        {m.model_id === 'seasonal_naive' ? (
                          <Badge variant="slate">Baseline Benchmark</Badge>
                        ) : m.outperforms_baseline ? (
                          <Badge variant="emerald">
                            +{m.improvement_vs_baseline_pct}% Better
                          </Badge>
                        ) : m.improvement_vs_baseline_pct != null ? (
                          <Badge variant="amber">
                            {m.improvement_vs_baseline_pct}% vs Baseline
                          </Badge>
                        ) : (
                          '—'
                        )}
                      </td>
                      <td data-label="Avg Runtime" className="py-3 px-3 text-right text-slate-600">
                        {m.avg_runtime_ms != null ? `${m.avg_runtime_ms} ms` : '—'}
                      </td>
                      <td data-label="Requirements" className="py-3 px-3 text-slate-600 max-w-xs">
                        <div>{m.data_requirements}</div>
                        <div className="text-[11px] text-slate-400 mt-0.5">
                          License: {meta?.license_name || 'Standard Open Source'}
                        </div>
                        {!m.available && (
                          <div className="text-[11px] text-rose-700 font-medium mt-1">
                            {m.notes}
                          </div>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Historical Backtest Predictions vs Actual Observations Chart */}
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
        <Card className="xl:col-span-2">
          <CardHeader>
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <CardTitle>Out-of-Sample Predictions vs. Actual Recorded Concerns</CardTitle>
                <CardDescription>
                  Comparing rolling-origin historical model forecasts against actual recorded safeguarding counts across evaluated periods.
                </CardDescription>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {backtestData.metrics.map((m) => (
                  <button
                    key={m.model_id}
                    disabled={!m.available}
                    onClick={() => toggleModelLine(m.model_id)}
                    className={`rounded-lg px-2.5 py-1 text-xs font-semibold border transition-all cursor-pointer disabled:opacity-40 ${
                      visibleModels[m.model_id]
                        ? 'bg-slate-900 text-white border-slate-900'
                        : 'bg-white text-slate-600 border-slate-200'
                    }`}
                  >
                    {m.model_name.split(' ')[0]}
                  </button>
                ))}
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="h-[340px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                  <XAxis dataKey="label" tick={{ fontSize: 11, fill: '#475569' }} minTickGap={20} />
                  <YAxis tick={{ fontSize: 11, fill: '#475569' }} axisLine={false} tickLine={false} />
                  <Tooltip />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Line
                    type="monotone"
                    dataKey="actual"
                    name="Actual Recorded Concerns"
                    stroke="#0f172a"
                    strokeWidth={3}
                    dot={{ r: 3, fill: '#0f172a' }}
                  />
                  {visibleModels.seasonal_naive && (
                    <Line
                      type="monotone"
                      dataKey="seasonal_naive"
                      name="Seasonal Naive"
                      stroke={MODEL_COLORS.seasonal_naive}
                      strokeWidth={1.8}
                      strokeDasharray="4 4"
                      dot={false}
                    />
                  )}
                  {visibleModels.statistical_count && (
                    <Line
                      type="monotone"
                      dataKey="statistical_count"
                      name="Statistical Count GLM"
                      stroke={MODEL_COLORS.statistical_count}
                      strokeWidth={2.2}
                      dot={{ r: 2 }}
                    />
                  )}
                  {visibleModels.lightgbm_mlforecast && (
                    <Line
                      type="monotone"
                      dataKey="lightgbm_mlforecast"
                      name="LightGBM + MLForecast"
                      stroke={MODEL_COLORS.lightgbm_mlforecast}
                      strokeWidth={2.4}
                      dot={{ r: 2 }}
                    />
                  )}
                  {visibleModels.timesfm_3 && (
                    <Line
                      type="monotone"
                      dataKey="timesfm_3"
                      name="Google TimesFM 3.0"
                      stroke={MODEL_COLORS.timesfm_3}
                      strokeWidth={2.2}
                      strokeDasharray="6 2"
                      dot={{ r: 2 }}
                    />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Error Comparison Bar Chart & Licensing Notes */}
        <Card>
          <CardHeader>
            <CardTitle>Mean Absolute Error (MAE) Comparison</CardTitle>
            <CardDescription>
              Lower MAE indicates higher historical accuracy (fewer average monthly error counts).
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="h-[200px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={availableMetrics} layout="vertical" margin={{ top: 5, right: 20, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" horizontal={false} />
                  <XAxis type="number" tick={{ fontSize: 11, fill: '#475569' }} />
                  <YAxis
                    type="category"
                    dataKey="model_name"
                    width={120}
                    tick={{ fontSize: 10, fill: '#1e293b', fontWeight: 600 }}
                    tickFormatter={(v: string) => v.split('(')[0].trim()}
                  />
                  <Tooltip />
                  <Bar dataKey="mae" name="MAE (Concerns/Mo)" radius={[0, 6, 6, 0]} barSize={24}>
                    {availableMetrics.map((entry) => (
                      <Cell key={entry.model_id} fill={MODEL_COLORS[entry.model_id] || '#0d9488'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="rounded-xl bg-purple-50/70 border border-purple-200 p-3.5 text-xs text-purple-950 space-y-1.5">
              <div className="font-bold flex items-center gap-1.5">
                <Scale className="h-4 w-4 text-purple-700" />
                TimesFM 3.0 Licensing & Evaluation Integrity
              </div>
              <p className="leading-relaxed">
                Google TimesFM 3.0 weights (<code className="font-mono text-[11px]">google/timesfm-3.0-pytorch</code>) are distributed under <code className="font-mono text-[11px]">timesfm-non-commercial-license-v1.0</code>. Self-hosted weights are restricted to non-commercial research and prototype benchmarking.
              </p>
              <p className="leading-relaxed">
                For live operational FE college deployment, use <strong>Statistical Count GLM</strong> or <strong>LightGBM + MLForecast</strong> (permissive open-source licenses), or access TimesFM 3.0 commercially via Google Cloud BigQuery ML.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Weekly Count-Forecasting Benchmark (NB2 GLM vs. ZINB vs. Tweedie LightGBM) */}
      <Card className="border-teal-200/80">
        <CardHeader className="bg-slate-50/70 border-b border-slate-100 rounded-t-2xl">
          <div className="flex flex-col gap-2 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <div className="flex items-center gap-2">
                <Badge variant="teal">Kaggle Count Benchmark Suite</Badge>
                <Badge variant="blue">Expanding-Window Out-of-Time CV</Badge>
              </div>
              <CardTitle className="mt-1.5">
                Weekly Count-Forecasting Benchmark: NB2 GLM vs. ZINB vs. Tweedie LightGBM
              </CardTitle>
              <CardDescription>
                Evaluated across 3 expanding-window weekly origins (78, 104, 130 weeks; 26-week horizon) using DengAI, Recruit (RMSLE), M5 Accuracy (RMSSE / WRMSSE), and M5 Uncertainty (Pinball Loss, WSPL, 95% PICP) metrics with exact active learner exposure offsets (<code className="font-mono text-[11px]">ln N_t</code>).
              </CardDescription>
            </div>
            <div className="rounded-xl bg-teal-50 border border-teal-200 px-3.5 py-2 text-xs text-teal-950 max-w-md">
              <strong className="font-bold block">VIF Multicollinearity Diagnostic:</strong>
              {countBenchmarkSnapshot.vif_diagnostics.collinearity_note}
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-6 pt-5">
          <div tabIndex={0} role="region" aria-label="Weekly count-forecasting benchmark table" className="table-wrapper rounded-xl border border-slate-200">
            <table className="responsive-stack text-left text-xs">
              <thead className="bg-slate-100 text-slate-700 font-semibold border-b border-slate-200">
                <tr>
                  <th className="py-2.5 px-3">Count Architecture</th>
                  <th className="py-2.5 px-2.5 text-right">MAE</th>
                  <th className="py-2.5 px-2.5 text-right">Surge MAE</th>
                  <th className="py-2.5 px-2.5 text-right">NB Dev</th>
                  <th className="py-2.5 px-2.5 text-right">RMSLE</th>
                  <th className="py-2.5 px-2.5 text-right">RMSSE</th>
                  <th className="py-2.5 px-2.5 text-right">WRMSSE</th>
                  <th className="py-2.5 px-2.5 text-right">Pinball (0.90)</th>
                  <th className="py-2.5 px-2.5 text-right">Pinball (0.975)</th>
                  <th className="py-2.5 px-2.5 text-right">WSPL</th>
                  <th className="py-2.5 px-2.5 text-right">95% PICP</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {countBenchmarkSnapshot.benchmark_comparison_table.map((row) => (
                  <tr
                    key={row.model_id}
                    className={
                      row.model_id === 'nb2_glm' || row.model_id === 'zinb_model'
                        ? 'bg-teal-50/30 hover:bg-teal-50/60'
                        : 'hover:bg-slate-50'
                    }
                  >
                    <th className="py-2.5 px-3 font-bold text-slate-900">
                      <div className="flex items-center gap-1.5">
                        <span>{row.model_name}</span>
                        {row.model_id === 'nb2_glm' && <Badge variant="blue">Governance Champion</Badge>}
                        {row.model_id === 'zinb_model' && <Badge variant="teal">Best Tail / Sparse Regime</Badge>}
                      </div>
                    </th>
                    <td data-label="MAE" className="py-2.5 px-2.5 text-right font-semibold">{row.mae.toFixed(3)}</td>
                    <td data-label="Surge MAE" className="py-2.5 px-2.5 text-right">{row.surge_mae.toFixed(3)}</td>
                    <td data-label="NB Deviance" className="py-2.5 px-2.5 text-right">{row.nb_deviance.toFixed(4)}</td>
                    <td data-label="RMSLE" className="py-2.5 px-2.5 text-right font-semibold text-teal-800">{row.rmsle.toFixed(4)}</td>
                    <td data-label="RMSSE" className="py-2.5 px-2.5 text-right">{row.rmsse.toFixed(4)}</td>
                    <td data-label="WRMSSE" className="py-2.5 px-2.5 text-right">{row.wrmsse.toFixed(4)}</td>
                    <td data-label="Pinball 0.90" className="py-2.5 px-2.5 text-right">{row.pinball_900.toFixed(4)}</td>
                    <td data-label="Pinball 0.975" className="py-2.5 px-2.5 text-right font-bold text-slate-900">{row.pinball_975.toFixed(4)}</td>
                    <td data-label="WSPL" className="py-2.5 px-2.5 text-right font-semibold">{row.wspl.toFixed(4)}</td>
                    <td data-label="95% PICP" className="py-2.5 px-2.5 text-right">
                      <Badge variant={row.picp_95 >= 92 ? 'emerald' : 'amber'}>
                        {row.picp_95.toFixed(1)}%
                      </Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* KCSIE / GDPR Governance IRR Audit Table & Weekly DSL Triage Capacity Schedule */}
          <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
            <div className="space-y-2.5">
              <div className="flex items-center justify-between">
                <h3 className="font-display text-sm font-bold text-slate-900 flex items-center gap-1.5">
                  <ShieldCheck className="h-4 w-4 text-blue-600" />
                  KCSIE / GDPR Governance Audit: NB2 Incidence Rate Ratios (IRR)
                </h3>
                <Badge variant="blue">Exact Exposure Offset ln(N_t)</Badge>
              </div>
              <div tabIndex={0} role="region" aria-label="NB2 GLM Incidence Rate Ratio table" className="table-wrapper rounded-xl border border-slate-200 max-h-[360px]">
                <table className="responsive-stack text-left text-xs">
                  <thead className="bg-slate-100 text-slate-700 font-semibold border-b border-slate-200 sticky top-0">
                    <tr>
                      <th className="py-2 px-2.5">Predictor</th>
                      <th className="py-2 px-2 text-right">Beta</th>
                      <th className="py-2 px-2 text-right">IRR = exp(β)</th>
                      <th className="py-2 px-2 text-right">95% CI</th>
                      <th className="py-2 px-2 text-right">p-val</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {countBenchmarkSnapshot.dsl_governance_irr_table
                      .filter((r) => r.feature_name !== 'const')
                      .map((r) => (
                        <tr key={r.feature_name} className="hover:bg-slate-50">
                          <th className="py-2 px-2.5 font-semibold text-slate-800">
                            <div>{r.feature_label}</div>
                            <div className="text-[10px] font-mono text-slate-400">{r.feature_name}</div>
                          </th>
                          <td data-label="Beta" className="py-2 px-2 text-right font-mono">{r.beta.toFixed(3)}</td>
                          <td data-label="IRR" className="py-2 px-2 text-right font-bold text-slate-900">{r.irr.toFixed(3)}x</td>
                          <td data-label="95% CI" className="py-2 px-2 text-right font-mono text-[11px] text-slate-600">
                            [{r.irr_ci_lower_95.toFixed(2)}, {r.irr_ci_upper_95.toFixed(2)}]
                          </td>
                          <td data-label="p-value" className="py-2 px-2 text-right font-mono text-[11px]">
                            {r.p_value < 0.001 ? '<0.001' : r.p_value.toFixed(3)}
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="space-y-2.5">
              <div className="flex items-center justify-between">
                <h3 className="font-display text-sm font-bold text-slate-900 flex items-center gap-1.5">
                  <Clock className="h-4 w-4 text-teal-600" />
                  DSL Weekly Triage Capacity Schedule (90th & 97.5th Tail Bounds)
                </h3>
                <Badge variant="amber">Baseline DSL Capacity: 36.0 / wk</Badge>
              </div>
              <div tabIndex={0} role="region" aria-label="DSL weekly triage schedule" className="table-wrapper rounded-xl border border-slate-200 max-h-[360px]">
                <table className="responsive-stack text-left text-xs">
                  <thead className="bg-slate-100 text-slate-700 font-semibold border-b border-slate-200 sticky top-0">
                    <tr>
                      <th className="py-2 px-2.5">Week Start</th>
                      <th className="py-2 px-2 text-right">Expected (μ̂)</th>
                      <th className="py-2 px-2 text-right">Q90</th>
                      <th className="py-2 px-2 text-right">Q97.5</th>
                      <th className="py-2 px-2.5">Triage Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {countBenchmarkSnapshot.dsl_weekly_triage_schedule.slice(0, 12).map((w) => (
                      <tr key={w.week_start} className={w.exceeds_q90_capacity ? 'bg-rose-50/40' : 'hover:bg-slate-50'}>
                        <th className="py-2 px-2.5 font-mono font-semibold text-slate-800">
                          {w.week_start} <span className="text-[10px] text-slate-400">(W{w.week_of_academic_year})</span>
                        </th>
                        <td data-label="Expected" className="py-2 px-2 text-right font-semibold">{w.expected_referrals_mu.toFixed(1)}</td>
                        <td data-label="Q90" className="py-2 px-2 text-right font-semibold text-amber-800">{w.q_90.toFixed(0)}</td>
                        <td data-label="Q97.5" className="py-2 px-2 text-right font-bold text-rose-800">{w.q_975.toFixed(0)}</td>
                        <td data-label="Status" className="py-2 px-2.5">
                          {w.exceeds_q90_capacity ? (
                            <Badge variant="rose">High Alert (&gt;36/wk)</Badge>
                          ) : (
                            <Badge variant="emerald">Within Capacity</Badge>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

