import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  Calendar,
  CheckCircle2,
  Cpu,
  GitBranch,
  Info,
  Layers,
  Lock,
  RotateCcw,
  Scale,
  Sliders,
  Sparkles,
  TrendingUp,
} from 'lucide-react';
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  Badge,
  Button,
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '../components/ui/primitives';
import { apiClient } from '../lib/api';
import type {
  DatasetSummaryResponse,
  EsotericForecastResponse,
  ScenarioAssumptions,
} from '../types/api';

interface EsotericForecastPageProps {
  datasetSummary: DatasetSummaryResponse;
  selectedProvision: string;
  onSelectProvision: (p: string) => void;
  selectedCategory: string;
  onSelectCategory: (c: string) => void;
  selectedPI: number;
  onSelectPI: (pi: number) => void;
}

const DEFAULT_SCENARIO: ScenarioAssumptions = {
  enrolment_growth_pct: 0,
  teaching_days_delta: 0,
  attendance_rate_delta_pp: 0,
  reporting_change_assumed: null,
};

const MONTHLY_BASELINES = [
  { id: 'lightgbm_mlforecast', label: 'LightGBM + Nixtla MLForecast (Recommended)' },
  { id: 'statistical_count', label: 'Negative Binomial / Count GLM' },
  { id: 'seasonal_naive', label: 'Academic Seasonal Naive Benchmark' },
];

const WEEKLY_BASELINES = [
  { id: 'tweedie_lgbm', label: 'Tweedie LightGBM (p=1.5 Weekly Anchor)' },
  { id: 'nb2_glm', label: 'NB2 Governance GLM (Exposure Offset)' },
  { id: 'zinb_model', label: 'Zero-Inflated NB (Holiday Structural Zeros)' },
];

export const EsotericForecastPage: React.FC<EsotericForecastPageProps> = ({
  datasetSummary,
  selectedProvision,
  onSelectProvision,
  selectedCategory,
  onSelectCategory,
  selectedPI,
  onSelectPI,
}) => {
  const [cadence, setCadence] = useState<'monthly' | 'weekly'>('monthly');
  const [strategyId, setStrategyId] = useState<string>('staged_core_4');
  const [baseModelId, setBaseModelId] = useState<string>('lightgbm_mlforecast');
  const [horizon, setHorizon] = useState<number>(12);
  const [customModules, setCustomModules] = useState<string[] | null>(null);
  const [scenario, setScenario] = useState<ScenarioAssumptions>(DEFAULT_SCENARIO);

  const [data, setData] = useState<EsotericForecastResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const handleSwitchCadence = (nextCadence: 'monthly' | 'weekly') => {
    if (nextCadence === cadence) return;
    setCadence(nextCadence);
    setBaseModelId(nextCadence === 'weekly' ? 'tweedie_lgbm' : 'lightgbm_mlforecast');
    setHorizon(nextCadence === 'weekly' ? 26 : 12);
    setCustomModules(null);
  };

  const handleSelectStrategy = (nextStrategy: string) => {
    setStrategyId(nextStrategy);
    setCustomModules(null);
  };

  const fetchEsotericForecast = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await apiClient.generateEsotericForecast({
        cadence,
        strategy_id: strategyId,
        base_model_id: baseModelId,
        horizon,
        provision: selectedProvision,
        safeguarding_category: cadence === 'weekly' ? 'All' : selectedCategory,
        prediction_interval: selectedPI,
        custom_modules: customModules,
        scenario,
      });
      setData(res);
    } catch (err: any) {
      setError(err.message || 'Failed to compute esoteric multi-step forecast.');
    } finally {
      setIsLoading(false);
    }
  }, [
    cadence,
    strategyId,
    baseModelId,
    horizon,
    selectedProvision,
    selectedCategory,
    selectedPI,
    customModules,
    scenario,
  ]);

  useEffect(() => {
    fetchEsotericForecast();
  }, [fetchEsotericForecast]);

  const toggleModule = (modId: string) => {
    if (!data) return;
    const currentList = customModules !== null ? customModules : data.active_modules;
    const exists = currentList.includes(modId);
    const nextList = exists
      ? currentList.filter((m) => m !== modId)
      : [...currentList, modId].sort();
    setCustomModules(nextList);
  };

  const hasScenarioChanges =
    scenario.enrolment_growth_pct !== 0 ||
    scenario.teaching_days_delta !== 0 ||
    scenario.attendance_rate_delta_pp !== 0;

  // Build combined historical + forward multi-step chart series
  const { trajectoryData, originLabel } = useMemo(() => {
    if (!data) return { trajectoryData: [], originLabel: '' };
    const histTailCount = data.cadence === 'weekly' ? 40 : 24;
    const histTail = data.historical_series.slice(-histTailCount);
    const lastHist = histTail[histTail.length - 1];

    const histRows = histTail.map((pt, idx) => {
      const isLast = idx === histTail.length - 1;
      return {
        label: pt.period_label,
        academic_term: pt.academic_term,
        is_forecast: false,
        is_origin: isLast,
        actual_count: pt.actual_count,
        baseline_forecast: isLast ? pt.actual_count : null,
        esoteric_forecast: isLast ? pt.actual_count : null,
        esoteric_pi: isLast ? [pt.actual_count, pt.actual_count] : null,
        lower_bound: isLast ? pt.actual_count : null,
        upper_bound: isLast ? pt.actual_count : null,
        delta_count: 0,
        dominant_driver: 'Historical Observed',
      };
    });

    const fcRows = data.forecast_series.map((pt) => ({
      label: pt.period_label,
      academic_term: pt.academic_term,
      is_forecast: true,
      is_origin: false,
      actual_count: null as number | null,
      baseline_forecast: pt.baseline_forecast,
      esoteric_forecast: pt.esoteric_forecast,
      esoteric_pi: [pt.lower_bound, pt.upper_bound] as [number, number],
      lower_bound: pt.lower_bound,
      upper_bound: pt.upper_bound,
      delta_count: pt.delta_count,
      dominant_driver: pt.dominant_driver,
    }));

    return {
      trajectoryData: [...histRows, ...fcRows],
      originLabel: lastHist ? lastHist.period_label : '',
    };
  }, [data]);

  const maxWeight = useMemo(() => {
    if (!data || data.conditioning_diagnostics.admitted_features.length === 0) return 0.1;
    return Math.max(
      0.01,
      ...data.conditioning_diagnostics.admitted_features.map((f) => Math.abs(f.weight))
    );
  }, [data]);

  if (!data && isLoading) {
    return (
      <div className="flex h-96 flex-col items-center justify-center gap-3 rounded-2xl border border-slate-200 bg-white p-8">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-purple-600 border-t-transparent" />
        <p className="text-sm font-semibold text-slate-700">
          Executing Causal Recursive Rollout & Higher-Order Manifold Conditioning...
        </p>
      </div>
    );
  }

  if (error && !data) {
    return (
      <Card className="border-rose-300 bg-rose-50/60 p-6">
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-3 text-rose-900 text-sm">
            <AlertTriangle className="h-5 w-5 text-rose-600 shrink-0" />
            <span>{error}</span>
          </div>
          <Button variant="danger" size="sm" onClick={fetchEsotericForecast}>
            Retry Rollout
          </Button>
        </div>
      </Card>
    );
  }

  if (!data) return null;

  const { summary, conditioning_diagnostics: cond, walk_forward_backtest: bt } = data;
  const horizonOptions = cadence === 'weekly' ? [13, 26] : [3, 6, 12];
  const baselineOptions = cadence === 'weekly' ? WEEKLY_BASELINES : MONTHLY_BASELINES;
  const unitSuffix = cadence === 'weekly' ? 'w' : 'm';
  const btImproved = bt.esoteric_mae <= bt.baseline_mae;

  return (
    <div className="space-y-6">
      {/* Architectural Isolation & Cadence Header */}
      <div className="rounded-2xl border border-purple-200 bg-gradient-to-r from-slate-900 via-slate-900 to-purple-950 p-5 text-white shadow-sm">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div className="space-y-1.5 max-w-3xl">
            <div className="flex flex-wrap items-center gap-2">
              <span className="inline-flex items-center gap-1.5 rounded-full bg-purple-500/25 px-3 py-0.5 text-xs font-bold text-purple-200 border border-purple-400/30">
                <Sparkles className="h-3.5 w-3.5 text-purple-300" />
                Higher-Order Mathematical Forecasting Lab
              </span>
              <span className="inline-flex items-center gap-1 rounded-full bg-teal-500/20 px-2.5 py-0.5 text-[11px] font-semibold text-teal-200 border border-teal-400/30">
                <Lock className="h-3 w-3" />
                Core Operational Forecasts Locked (SHA-256: {data.baseline_sha256.slice(0, 12)}…)
              </span>
              {isLoading && (
                <span className="inline-flex items-center gap-1 rounded-full bg-amber-400/20 px-2.5 py-0.5 text-[11px] font-semibold text-amber-200">
                  Recomputing Manifold…
                </span>
              )}
            </div>
            <h1 className="font-display text-xl sm:text-2xl font-extrabold tracking-tight text-white">
              Esoteric & Rough-Path Multi-Step Time-Series Forecasting
            </h1>
            <p className="text-xs sm:text-sm text-slate-300 leading-relaxed">
              Integrates Depth-3 Chen Rough Path Signatures &amp; Antisymmetric Lévy Area (<code className="text-purple-200">f07</code>), Warped Instructional Hilbert Phase (<code className="text-purple-200">f01</code>), Boundary-Causal MODWT Wavelets (<code className="text-purple-200">f02</code>), Hawkes Contagion (<code className="text-purple-200">f03</code>), and Fisher-Rao Natural Gradient NB Boosting (<code className="text-purple-200">f12</code>) into strictly causal recursive forward rollouts — while keeping the core operational tabs untouched as an immutable control benchmark.
            </p>
          </div>

          {/* Cadence Switcher */}
          <div className="shrink-0 flex flex-col gap-1.5">
            <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
              Time-Series Cadence
            </span>
            <div className="inline-flex rounded-xl bg-slate-800/90 p-1 border border-slate-700">
              <button
                onClick={() => handleSwitchCadence('monthly')}
                className={`inline-flex items-center gap-1.5 rounded-lg px-3.5 py-2 text-xs font-bold transition-all cursor-pointer ${
                  cadence === 'monthly'
                    ? 'bg-purple-500 text-white shadow-xs'
                    : 'text-slate-300 hover:text-white'
                }`}
              >
                <Calendar className="h-3.5 w-3.5" />
                Monthly Strategic (3–12m)
              </button>
              <button
                onClick={() => handleSwitchCadence('weekly')}
                className={`inline-flex items-center gap-1.5 rounded-lg px-3.5 py-2 text-xs font-bold transition-all cursor-pointer ${
                  cadence === 'weekly'
                    ? 'bg-purple-500 text-white shadow-xs'
                    : 'text-slate-300 hover:text-white'
                }`}
              >
                <Activity className="h-3.5 w-3.5" />
                Weekly UK FE Panel (13–26w)
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* 4-Card Executive & Mathematical KPI Ribbon */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {/* Card 1: Esoteric Horizon Total */}
        <Card className="border-purple-200 bg-purple-50/25">
          <CardHeader className="pb-1.5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider text-purple-900">
                Esoteric Horizon Total ({data.horizon}
                {unitSuffix})
              </span>
              <Badge variant="purple">{data.strategy.short_name}</Badge>
            </div>
          </CardHeader>
          <CardContent>
            <div className="flex items-baseline gap-2.5">
              <span className="font-display text-3xl font-extrabold text-slate-900">
                {summary.esoteric_total.toLocaleString()}
              </span>
              <Badge variant={summary.yoy_change_pct >= 0 ? 'amber' : 'teal'}>
                {summary.yoy_change_pct >= 0
                  ? `+${summary.yoy_change_pct}% vs prior`
                  : `${summary.yoy_change_pct}% vs prior`}
              </Badge>
            </div>
            <p className="mt-1.5 text-xs font-medium text-slate-700">
              Fisher-Rao {data.prediction_interval}% Interval:{' '}
              <strong>
                {summary.esoteric_lower.toLocaleString()} – {summary.esoteric_upper.toLocaleString()}
              </strong>
            </p>
            <p className="mt-1 text-[11px] text-slate-500">
              Peak Period: <strong>{summary.peak_period_label}</strong> ({summary.peak_period_count}{' '}
              concerns)
            </p>
          </CardContent>
        </Card>

        {/* Card 2: Net Shift vs Locked Baseline */}
        <Card>
          <CardHeader className="pb-1.5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-500">
                Shift vs. Locked Baseline
              </span>
              <Badge variant="teal">Control Anchor</Badge>
            </div>
          </CardHeader>
          <CardContent>
            <div className="flex items-baseline gap-2.5">
              <span className="font-display text-3xl font-extrabold text-slate-900">
                {summary.net_esoteric_delta > 0
                  ? `+${summary.net_esoteric_delta.toLocaleString()}`
                  : summary.net_esoteric_delta.toLocaleString()}
              </span>
              <Badge
                variant={
                  summary.net_esoteric_delta > 0
                    ? 'purple'
                    : summary.net_esoteric_delta < 0
                      ? 'teal'
                      : 'slate'
                }
              >
                {summary.net_esoteric_delta_pct > 0
                  ? `+${summary.net_esoteric_delta_pct}%`
                  : `${summary.net_esoteric_delta_pct}%`}
              </Badge>
            </div>
            <p className="mt-1.5 text-xs text-slate-700">
              Pure Baseline Total: <strong>{summary.baseline_total.toLocaleString()}</strong> (
              {summary.baseline_lower.toLocaleString()}–{summary.baseline_upper.toLocaleString()})
            </p>
            <p className="mt-1 text-[11px] text-slate-500 truncate" title={data.base_model_name}>
              Anchor: {data.base_model_name}
            </p>
          </CardContent>
        </Card>

        {/* Card 3: Walk-Forward Out-of-Sample Backtest */}
        <Card>
          <CardHeader className="pb-1.5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-500">
                Walk-Forward Ablation MAE
              </span>
              <Badge variant={btImproved ? 'emerald' : 'amber'}>
                {bt.mae_delta_pct <= 0 ? `${bt.mae_delta_pct}% Error` : `+${bt.mae_delta_pct}% Error`}
              </Badge>
            </div>
          </CardHeader>
          <CardContent>
            <div className="flex items-baseline gap-2">
              <span className="font-display text-3xl font-extrabold text-slate-900">
                {bt.esoteric_mae.toFixed(2)}
              </span>
              <span className="text-xs text-slate-500">
                vs <strong>{bt.baseline_mae.toFixed(2)}</strong> baseline MAE
              </span>
            </div>
            <p className="mt-1.5 text-xs text-slate-700">
              Top-Quartile Surge MAE: <strong>{bt.esoteric_surge_mae.toFixed(2)}</strong> (base{' '}
              {bt.baseline_surge_mae.toFixed(2)})
            </p>
            <p className="mt-1 text-[11px] text-slate-500">
              {bt.folds} causal folds ({bt.evaluated_periods} holdout steps) • 80% PICP:{' '}
              {bt.esoteric_pi_coverage_pct}%
            </p>
          </CardContent>
        </Card>

        {/* Card 4: Anti-Bloat Conditioning Funnel */}
        <Card>
          <CardHeader className="pb-1.5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-500">
                Anti-Bloat Conditioning
              </span>
              <Badge variant={cond.identity_floor_active ? 'amber' : 'blue'}>
                {cond.identity_floor_active ? 'Identity Floor' : 'QR + VIF + L1'}
              </Badge>
            </div>
          </CardHeader>
          <CardContent>
            <div className="flex items-baseline gap-2">
              <span className="font-display text-3xl font-extrabold text-slate-900">
                {cond.n_admitted_esoteric_features}
              </span>
              <span className="text-xs text-slate-500">
                / {cond.n_candidate_esoteric_features} admitted ({cond.n_quarantined_esoteric_features}{' '}
                quarantined)
              </span>
            </div>
            <p className="mt-1.5 text-xs text-slate-700">
              Post-Funnel Max VIF: <strong>{cond.max_vif_after.toFixed(2)}</strong> • κ(X):{' '}
              <strong>{cond.condition_number_after.toFixed(1)}</strong>
            </p>
            <p className="mt-1 text-[11px] text-slate-500">
              Fisher-Rao Base NB Dispersion r = {cond.base_nb_dispersion_r.toFixed(2)} (N_train ={' '}
              {cond.n_train_steps})
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Main Two-Column Analytical Layout */}
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-12">
        {/* Left Sidebar: Strategy, Baseline, 13-Module Matrix & Scenario Controls */}
        <div className="space-y-5 xl:col-span-4">
          {/* Strategy & Baseline Card */}
          <Card>
            <CardHeader className="border-b border-slate-100 pb-3">
              <div className="flex items-center gap-2">
                <Cpu className="h-4 w-4 text-purple-600" />
                <CardTitle>Esoteric Forecasting Architecture</CardTitle>
              </div>
              <CardDescription>
                Select how higher-order geometric, spectral, and rough-path predictors couple with the anchor count baseline.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4 pt-4">
              {/* 4 Strategy Cards */}
              <div className="space-y-2">
                {data.strategies_catalog.map((st) => {
                  const active = strategyId === st.id;
                  return (
                    <button
                      key={st.id}
                      onClick={() => handleSelectStrategy(st.id)}
                      className={`w-full text-left rounded-xl border p-3 transition-all cursor-pointer ${
                        active
                          ? 'border-purple-500 bg-purple-50/60 shadow-xs'
                          : 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50/60'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-xs font-bold text-slate-900">{st.name}</span>
                        <Badge variant={active ? 'purple' : 'slate'}>
                          {st.default_modules.length} mods
                        </Badge>
                      </div>
                      <p className="mt-1 text-[11px] text-slate-600 leading-snug">{st.description}</p>
                    </button>
                  );
                })}
              </div>

              {/* Anchor Baseline Selector */}
              <div>
                <label
                  htmlFor="eso-base-model"
                  className="block text-xs font-semibold text-slate-700 mb-1.5"
                >
                  Locked Control Anchor Baseline
                </label>
                <select
                  id="eso-base-model"
                  value={baseModelId}
                  onChange={(e) => setBaseModelId(e.target.value)}
                  className="w-full rounded-xl border border-slate-300 bg-slate-50 px-3 py-2 text-xs font-medium text-slate-900 focus:border-purple-600 focus:outline-none"
                >
                  {baselineOptions.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.label}
                    </option>
                  ))}
                </select>
              </div>

              {/* Horizon & Prediction Interval */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                    Rollout Horizon
                  </label>
                  <div className="flex rounded-xl bg-slate-100 p-1">
                    {horizonOptions.map((h) => (
                      <button
                        key={h}
                        onClick={() => setHorizon(h)}
                        className={`flex-1 rounded-lg py-1.5 text-xs font-semibold transition-all cursor-pointer ${
                          horizon === h
                            ? 'bg-slate-900 text-white shadow-xs'
                            : 'text-slate-600 hover:text-slate-900'
                        }`}
                      >
                        {h}
                        {unitSuffix}
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
                            ? 'bg-purple-600 text-white shadow-xs'
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
                  <label htmlFor="eso-prov" className="block text-xs font-semibold text-slate-700 mb-1">
                    Provision Cohort
                  </label>
                  <select
                    id="eso-prov"
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
                  <label htmlFor="eso-cat" className="block text-xs font-semibold text-slate-700 mb-1">
                    Category Scope
                  </label>
                  <select
                    id="eso-cat"
                    value={cadence === 'weekly' ? 'All' : selectedCategory}
                    disabled={cadence === 'weekly'}
                    onChange={(e) => onSelectCategory(e.target.value)}
                    className="w-full rounded-xl border border-slate-300 bg-slate-50 px-2.5 py-1.5 text-xs font-medium text-slate-800 disabled:opacity-60"
                  >
                    <option value="All">
                      {cadence === 'weekly' ? 'All Curriculum Concerns' : 'All Categories'}
                    </option>
                    {cadence === 'monthly' &&
                      datasetSummary.categories.map((c) => (
                        <option key={c} value={c}>
                          {c}
                        </option>
                      ))}
                  </select>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Interactive 13-Module Mathematical Matrix */}
          <Card>
            <CardHeader className="border-b border-slate-100 pb-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Layers className="h-4 w-4 text-purple-600" />
                  <CardTitle>Causal Manifold Modules (f01–f13)</CardTitle>
                </div>
                {customModules !== null && (
                  <button
                    onClick={() => setCustomModules(null)}
                    className="inline-flex items-center gap-1 text-[11px] font-semibold text-purple-700 hover:text-purple-900 cursor-pointer"
                  >
                    <RotateCcw className="h-3 w-3" />
                    Reset Default
                  </button>
                )}
              </div>
              <CardDescription>
                Toggle individual mathematical frameworks in or out of the candidate conditioning funnel. Disabling all triggers the Identity Floor (pure anchor baseline).
              </CardDescription>
            </CardHeader>
            <CardContent className="pt-3">
              <div className="grid grid-cols-1 gap-1.5 sm:grid-cols-2 xl:grid-cols-1 max-h-[340px] overflow-y-auto pr-1">
                {data.module_catalog.map((mod) => {
                  const isActive = data.active_modules.includes(mod.id);
                  const isAdmitted = cond.admitted_features.some((f) => f.module_id === mod.id);
                  return (
                    <button
                      key={mod.id}
                      onClick={() => toggleModule(mod.id)}
                      title={mod.role}
                      className={`flex items-start justify-between gap-2 rounded-xl border px-3 py-2 text-left transition-all cursor-pointer ${
                        isActive
                          ? 'border-purple-300 bg-purple-50/50 text-slate-900'
                          : 'border-slate-200 bg-slate-50/50 text-slate-400 hover:border-slate-300 hover:text-slate-600'
                      }`}
                    >
                      <div className="min-w-0">
                        <div className="flex items-center gap-1.5">
                          <span
                            className={`rounded px-1.5 py-0.5 font-mono text-[10px] font-bold uppercase ${
                              isActive
                                ? 'bg-purple-700 text-white'
                                : 'bg-slate-200 text-slate-600'
                            }`}
                          >
                            {mod.id}
                          </span>
                          <span className="text-xs font-bold truncate">{mod.short_name}</span>
                        </div>
                        <p className="mt-0.5 text-[10px] leading-tight text-slate-500 line-clamp-1">
                          {mod.family} — {mod.role}
                        </p>
                      </div>
                      <div className="shrink-0 pt-0.5">
                        {isActive && isAdmitted ? (
                          <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-[10px] font-bold text-emerald-800 border border-emerald-200">
                            Admitted
                          </span>
                        ) : isActive ? (
                          <span className="rounded-full bg-amber-50 px-2 py-0.5 text-[10px] font-semibold text-amber-800 border border-amber-200">
                            Pruned
                          </span>
                        ) : (
                          <span className="text-[10px] font-medium text-slate-400">Off</span>
                        )}
                      </div>
                    </button>
                  );
                })}
              </div>
            </CardContent>
          </Card>

          {/* Scenario Stress-Test Sliders */}
          <Card>
            <CardHeader className="border-b border-slate-100 pb-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Sliders className="h-4 w-4 text-teal-600" />
                  <CardTitle>Cohort & Calendar Stress-Testing</CardTitle>
                </div>
                {hasScenarioChanges && (
                  <button
                    onClick={() => setScenario(DEFAULT_SCENARIO)}
                    className="inline-flex items-center gap-1 text-[11px] font-semibold text-teal-700 hover:text-teal-900 cursor-pointer"
                  >
                    <RotateCcw className="h-3 w-3" />
                    Reset
                  </button>
                )}
              </div>
              <CardDescription>
                Propagate enrolment, instructional days, and attendance shocks through both the baseline and recursive esoteric manifold.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3.5 pt-4">
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="font-semibold text-slate-700">Student Enrolment Growth</span>
                  <span className="font-mono font-bold text-purple-700">
                    {scenario.enrolment_growth_pct > 0
                      ? `+${scenario.enrolment_growth_pct}%`
                      : `${scenario.enrolment_growth_pct}%`}
                  </span>
                </div>
                <input
                  type="range"
                  min={-15}
                  max={20}
                  step={1}
                  value={scenario.enrolment_growth_pct}
                  onChange={(e) =>
                    setScenario((s) => ({ ...s, enrolment_growth_pct: Number(e.target.value) }))
                  }
                  className="w-full accent-purple-600 cursor-pointer"
                />
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="font-semibold text-slate-700">
                    Instructional Days Shift ({cadence === 'weekly' ? 'days/wk' : 'days/mo'})
                  </span>
                  <span className="font-mono font-bold text-purple-700">
                    {scenario.teaching_days_delta > 0
                      ? `+${scenario.teaching_days_delta}d`
                      : `${scenario.teaching_days_delta}d`}
                  </span>
                </div>
                <input
                  type="range"
                  min={cadence === 'weekly' ? -2 : -4}
                  max={cadence === 'weekly' ? 2 : 4}
                  step={1}
                  value={scenario.teaching_days_delta}
                  onChange={(e) =>
                    setScenario((s) => ({ ...s, teaching_days_delta: Number(e.target.value) }))
                  }
                  className="w-full accent-purple-600 cursor-pointer"
                />
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="font-semibold text-slate-700">Cohort Attendance Shift</span>
                  <span className="font-mono font-bold text-purple-700">
                    {scenario.attendance_rate_delta_pp > 0
                      ? `+${scenario.attendance_rate_delta_pp} pp`
                      : `${scenario.attendance_rate_delta_pp} pp`}
                  </span>
                </div>
                <input
                  type="range"
                  min={-5}
                  max={5}
                  step={0.5}
                  value={scenario.attendance_rate_delta_pp}
                  onChange={(e) =>
                    setScenario((s) => ({ ...s, attendance_rate_delta_pp: Number(e.target.value) }))
                  }
                  className="w-full accent-purple-600 cursor-pointer"
                />
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Right Analytical Workspace (8 cols) */}
        <div className="space-y-6 xl:col-span-8">
          {/* Primary Chart: Historical + Esoteric Forecast + Fisher-Rao PI vs Locked Baseline */}
          <Card>
            <CardHeader>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <CardTitle>
                    Causal Recursive Multi-Step Forecast vs. Locked Anchor Baseline
                  </CardTitle>
                  <CardDescription>
                    Observed safeguarding volume (slate), locked pure control baseline (teal dashed), and causal esoteric forecast (purple solid with Fisher-Rao {data.prediction_interval}% NB dispersion band).
                  </CardDescription>
                </div>
                <Badge variant="purple">
                  {data.active_modules.length} Active Frameworks • {data.runtime_ms.toFixed(0)} ms
                </Badge>
              </div>
            </CardHeader>
            <CardContent>
              <div className="h-[360px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart
                    data={trajectoryData}
                    margin={{ top: 12, right: 20, left: 0, bottom: 8 }}
                  >
                    <defs>
                      <linearGradient id="esotericPiGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#7e22ce" stopOpacity={0.24} />
                        <stop offset="95%" stopColor="#7e22ce" stopOpacity={0.04} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                    <XAxis
                      dataKey="label"
                      tick={{ fontSize: 11, fill: '#475569' }}
                      minTickGap={18}
                    />
                    <YAxis
                      tick={{ fontSize: 11, fill: '#475569' }}
                      axisLine={false}
                      tickLine={false}
                    />
                    {originLabel && (
                      <ReferenceLine
                        x={originLabel}
                        stroke="#94a3b8"
                        strokeDasharray="4 4"
                        label={{
                          value: 'Forecast Origin T',
                          position: 'insideTopLeft',
                          fill: '#475569',
                          fontSize: 10,
                        }}
                      />
                    )}
                    <Tooltip
                      content={({ active, payload, label }) => {
                        if (!active || !payload || payload.length === 0) return null;
                        const row = payload[0].payload;
                        return (
                          <div className="rounded-xl border border-slate-200 bg-white p-3.5 shadow-lg text-xs space-y-1.5 min-w-[230px]">
                            <div className="font-bold text-slate-900 border-b border-slate-100 pb-1">
                              {label} <span className="text-slate-500 font-medium">({row.academic_term})</span>
                            </div>
                            {!row.is_forecast ? (
                              <div className="flex justify-between gap-4">
                                <span className="text-slate-600">Observed Concerns:</span>
                                <span className="font-bold text-slate-900">
                                  {row.actual_count} concerns
                                </span>
                              </div>
                            ) : (
                              <>
                                <div className="flex justify-between gap-4">
                                  <span className="text-purple-700 font-semibold">
                                    Esoteric Forecast:
                                  </span>
                                  <span className="font-extrabold text-purple-900">
                                    {row.esoteric_forecast} concerns
                                  </span>
                                </div>
                                <div className="flex justify-between gap-4">
                                  <span className="text-teal-700">Locked Anchor Baseline:</span>
                                  <span className="font-semibold text-teal-900">
                                    {row.baseline_forecast} concerns
                                  </span>
                                </div>
                                <div className="flex justify-between gap-4">
                                  <span className="text-slate-600">Higher-Order Shift:</span>
                                  <span className="font-bold text-slate-900">
                                    {row.delta_count > 0 ? `+${row.delta_count}` : row.delta_count}{' '}
                                    concerns
                                  </span>
                                </div>
                                <div className="flex justify-between gap-4 text-slate-500 pt-1 border-t border-slate-100">
                                  <span>Fisher-Rao {data.prediction_interval}% PI:</span>
                                  <span>
                                    {row.lower_bound} – {row.upper_bound}
                                  </span>
                                </div>
                                <div className="text-[10px] text-purple-700 font-medium pt-0.5">
                                  Lead Signal: {row.dominant_driver}
                                </div>
                              </>
                            )}
                          </div>
                        );
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <Area
                      type="monotone"
                      dataKey="esoteric_pi"
                      name={`Fisher-Rao ${data.prediction_interval}% Interval`}
                      stroke="none"
                      fill="url(#esotericPiGrad)"
                    />
                    <Line
                      type="monotone"
                      dataKey="actual_count"
                      name="Historical Observed"
                      stroke="#334155"
                      strokeWidth={2}
                      dot={{ r: 2, fill: '#334155' }}
                    />
                    <Line
                      type="monotone"
                      dataKey="baseline_forecast"
                      name="Locked Anchor Baseline"
                      stroke="#0d9488"
                      strokeWidth={2}
                      strokeDasharray="5 5"
                      dot={{ r: 2.5, fill: '#0d9488' }}
                    />
                    <Line
                      type="monotone"
                      dataKey="esoteric_forecast"
                      name="Esoteric Multi-Step Forecast"
                      stroke="#7e22ce"
                      strokeWidth={2.75}
                      dot={{ r: 3, fill: '#7e22ce' }}
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* Row 2: Step-by-Step Esoteric Delta + Forward Rough-Path & Hawkes Manifold Evolution */}
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            {/* Step-by-Step Esoteric Adjustment Bar Chart */}
            <Card>
              <CardHeader>
                <div className="flex items-center gap-2">
                  <Scale className="h-4 w-4 text-purple-600" />
                  <CardTitle>Step-by-Step Esoteric Count Shift</CardTitle>
                </div>
                <CardDescription>
                  Per-step difference (Esoteric Forecast minus Locked Baseline) across horizon steps T+1…T+{data.horizon}.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-[250px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={data.forecast_series}
                      margin={{ top: 8, right: 12, left: -12, bottom: 4 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                      <XAxis
                        dataKey="period_label"
                        tick={{ fontSize: 10, fill: '#475569' }}
                        minTickGap={14}
                      />
                      <YAxis tick={{ fontSize: 10, fill: '#475569' }} axisLine={false} tickLine={false} />
                      <ReferenceLine y={0} stroke="#64748b" />
                      <Tooltip
                        content={({ active, payload }) => {
                          if (!active || !payload || payload.length === 0) return null;
                          const pt = payload[0].payload;
                          return (
                            <div className="rounded-xl border border-slate-200 bg-white p-3 shadow-md text-xs space-y-1">
                              <div className="font-bold text-slate-900">
                                Step +{pt.step}: {pt.period_label}
                              </div>
                              <div>
                                Net Adjustment:{' '}
                                <strong
                                  className={
                                    pt.delta_count >= 0 ? 'text-purple-700' : 'text-teal-700'
                                  }
                                >
                                  {pt.delta_count > 0 ? `+${pt.delta_count}` : pt.delta_count}{' '}
                                  concerns ({pt.delta_pct > 0 ? `+${pt.delta_pct}%` : `${pt.delta_pct}%`})
                                </strong>
                              </div>
                              <div className="text-[11px] text-slate-500">
                                Driver: {pt.dominant_driver}
                              </div>
                            </div>
                          );
                        }}
                      />
                      <Bar dataKey="delta_count" name="Esoteric − Baseline Shift" radius={[4, 4, 0, 0]}>
                        {data.forecast_series.map((entry, idx) => (
                          <Cell
                            key={`cell-${idx}`}
                            fill={entry.delta_count >= 0 ? '#7e22ce' : '#0d9488'}
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            {/* Forward Rough-Path Levy Area & Signature Norm Evolution */}
            <Card>
              <CardHeader>
                <div className="flex items-center gap-2">
                  <GitBranch className="h-4 w-4 text-purple-600" />
                  <CardTitle>Forward Rough-Path &amp; Manifold Trajectory</CardTitle>
                </div>
                <CardDescription>
                  Recursive forward evolution of Antisymmetric Lévy Area |A(2)| (f07), Depth-3 Chen Signature Norm (f07), and NB Dispersion r_t (f12).
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-[250px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart
                      data={data.forecast_series}
                      margin={{ top: 8, right: 12, left: -12, bottom: 4 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                      <XAxis
                        dataKey="period_label"
                        tick={{ fontSize: 10, fill: '#475569' }}
                        minTickGap={14}
                      />
                      <YAxis
                        yAxisId="left"
                        tick={{ fontSize: 10, fill: '#475569' }}
                        axisLine={false}
                        tickLine={false}
                      />
                      <YAxis
                        yAxisId="right"
                        orientation="right"
                        tick={{ fontSize: 10, fill: '#64748b' }}
                        axisLine={false}
                        tickLine={false}
                      />
                      <Tooltip
                        content={({ active, payload }) => {
                          if (!active || !payload || payload.length === 0) return null;
                          const pt = payload[0].payload;
                          return (
                            <div className="rounded-xl border border-slate-200 bg-white p-3 shadow-md text-xs space-y-1">
                              <div className="font-bold text-slate-900">{pt.period_label}</div>
                              <div className="flex justify-between gap-4">
                                <span className="text-purple-700">Rough Lévy Area |A_ij|:</span>
                                <span className="font-mono font-bold">{pt.rough_levy_area}</span>
                              </div>
                              <div className="flex justify-between gap-4">
                                <span className="text-blue-700">Depth-3 Signature Norm:</span>
                                <span className="font-mono font-bold">{pt.rough_sig_depth3}</span>
                              </div>
                              <div className="flex justify-between gap-4">
                                <span className="text-amber-700">Hilbert Stress Phase:</span>
                                <span className="font-mono font-bold">{pt.hilbert_phase_deg}°</span>
                              </div>
                              <div className="flex justify-between gap-4">
                                <span className="text-teal-700">Fisher-Rao Dispersion r_t:</span>
                                <span className="font-mono font-bold">{pt.dispersion_r}</span>
                              </div>
                            </div>
                          );
                        }}
                      />
                      <Legend wrapperStyle={{ fontSize: 11 }} />
                      <Line
                        yAxisId="left"
                        type="monotone"
                        dataKey="rough_levy_area"
                        name="Rough Lévy Area |A|"
                        stroke="#7e22ce"
                        strokeWidth={2.2}
                        dot={{ r: 2 }}
                      />
                      <Line
                        yAxisId="left"
                        type="monotone"
                        dataKey="rough_sig_depth3"
                        name="Depth-3 Signature ‖S³‖"
                        stroke="#2563eb"
                        strokeWidth={2}
                        dot={false}
                      />
                      <Line
                        yAxisId="right"
                        type="monotone"
                        dataKey="dispersion_r"
                        name="NB Dispersion r_t"
                        stroke="#0d9488"
                        strokeWidth={1.8}
                        strokeDasharray="4 4"
                        dot={false}
                      />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Card 3: Admitted Higher-Order Predictors & Anti-Bloat Quarantine Ledger */}
          <Card>
            <CardHeader>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <CardTitle>
                    Admitted Higher-Order Predictors &amp; Anti-Bloat Conditioning Ledger
                  </CardTitle>
                  <CardDescription>
                    Predictors surviving rank-revealing QR pivoting, iterative VIF &le; 8 pruning, and L1/Riemannian shrinkage ({cond.funnel_type}).
                  </CardDescription>
                </div>
                <Badge variant="emerald">
                  {cond.n_admitted_esoteric_features} Admitted / {cond.n_quarantined_esoteric_features}{' '}
                  Quarantined
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              {cond.admitted_features.length === 0 ? (
                <div className="rounded-xl border border-amber-200 bg-amber-50/70 p-4 text-xs text-amber-950 flex items-center gap-2.5">
                  <Info className="h-4 w-4 text-amber-700 shrink-0" />
                  <span>
                    <strong>Identity Floor Active:</strong> Zero esoteric predictors passed the anti-bloat conditioning gate (or all modules were toggled off). The forecast safely collapses to the pure locked baseline ({data.base_model_name}) with zero variance inflation.
                  </span>
                </div>
              ) : (
                <div className="space-y-2.5">
                  {cond.admitted_features.map((feat) => {
                    const widthPct = Math.min(100, Math.max(8, (Math.abs(feat.weight) / maxWeight) * 100));
                    const isPositive = !feat.direction.includes('dampening');
                    return (
                      <div
                        key={feat.feature}
                        className="rounded-xl border border-slate-200/90 bg-slate-50/60 p-3"
                      >
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <div className="flex items-center gap-2">
                            <span className="rounded bg-purple-700 px-1.5 py-0.5 font-mono text-[10px] font-bold uppercase text-white">
                              {feat.module_id}
                            </span>
                            <code className="text-xs font-bold text-slate-900">{feat.feature}</code>
                            <span className="text-xs text-slate-500 hidden sm:inline">
                              • {feat.module_name}
                            </span>
                          </div>
                          <div className="flex items-center gap-2">
                            <Badge variant={isPositive ? 'purple' : 'teal'}>{feat.direction}</Badge>
                            <span className="font-mono text-xs font-bold text-slate-800">
                              {feat.weight >= 0 ? `+${feat.weight.toFixed(4)}` : feat.weight.toFixed(4)}
                            </span>
                          </div>
                        </div>
                        <div className="mt-2 h-1.5 w-full rounded-full bg-slate-200 overflow-hidden">
                          <div
                            className={`h-full rounded-full ${
                              isPositive ? 'bg-purple-600' : 'bg-teal-600'
                            }`}
                            style={{ width: `${widthPct}%` }}
                          />
                        </div>
                        <p className="mt-1.5 text-xs text-slate-600">{feat.description}</p>
                      </div>
                    );
                  })}
                </div>
              )}

              {/* Mathematical Notes & Governance Verification */}
              <div className="rounded-xl border border-slate-200 bg-white p-3.5 space-y-1.5">
                <div className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
                  <CheckCircle2 className="h-3.5 w-3.5 text-teal-600" />
                  Causal Windowing &amp; Statistical Conditioning Audit Notes
                </div>
                <ul className="list-disc pl-5 space-y-1 text-xs text-slate-600">
                  {data.mathematical_notes.map((note, i) => (
                    <li key={i}>{note}</li>
                  ))}
                </ul>
              </div>
            </CardContent>
          </Card>

          {/* Card 4: Walk-Forward Holdout Chart + Period-by-Period Forecast Schedule Table */}
          <Card>
            <CardHeader>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <div className="flex items-center gap-2">
                    <TrendingUp className="h-4 w-4 text-teal-600" />
                    <CardTitle>
                      Out-of-Sample Walk-Forward Backtest &amp; Horizon Schedule
                    </CardTitle>
                  </div>
                  <CardDescription>
                    Expanding-origin holdout comparison ({bt.evaluated_periods} out-of-sample periods across {bt.folds} causal folds) followed by the full {data.horizon}-{data.period_unit} forward schedule.
                  </CardDescription>
                </div>
                <Badge variant={btImproved ? 'emerald' : 'slate'}>
                  WAPE: {bt.esoteric_wape_pct}% vs {bt.baseline_wape_pct}% Baseline
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-5">
              {/* Walk-Forward Holdout Chart */}
              <div className="h-[230px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart
                    data={bt.points}
                    margin={{ top: 8, right: 16, left: -10, bottom: 4 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                    <XAxis dataKey="period_label" tick={{ fontSize: 10, fill: '#475569' }} />
                    <YAxis tick={{ fontSize: 10, fill: '#475569' }} axisLine={false} tickLine={false} />
                    <Tooltip />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Line
                      type="monotone"
                      dataKey="actual"
                      name="Holdout Actual"
                      stroke="#0f172a"
                      strokeWidth={2.2}
                      dot={{ r: 2.5 }}
                    />
                    <Line
                      type="monotone"
                      dataKey="baseline"
                      name="Pure Baseline Holdout"
                      stroke="#0d9488"
                      strokeWidth={1.8}
                      strokeDasharray="4 4"
                      dot={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="esoteric"
                      name="Esoteric Stack Holdout"
                      stroke="#7e22ce"
                      strokeWidth={2.2}
                      dot={{ r: 2 }}
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>

              {/* Period-by-Period Schedule Table */}
              <div className="overflow-x-auto rounded-xl border border-slate-200">
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-100/90 text-[11px] font-bold uppercase text-slate-600 border-b border-slate-200">
                    <tr>
                      <th className="px-3 py-2.5">Step</th>
                      <th className="px-3 py-2.5">Period</th>
                      <th className="px-3 py-2.5">Term</th>
                      <th className="px-3 py-2.5 text-right">Locked Base</th>
                      <th className="px-3 py-2.5 text-right">Esoteric Fcst</th>
                      <th className="px-3 py-2.5 text-right">Shift</th>
                      <th className="px-3 py-2.5 text-right">{data.prediction_interval}% Interval</th>
                      <th className="px-3 py-2.5 text-right">Lévy |A|</th>
                      <th className="px-3 py-2.5 text-right">Hawkes λ</th>
                      <th className="px-3 py-2.5 text-right">Disp r_t</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 bg-white">
                    {data.forecast_series.map((pt) => (
                      <tr key={pt.step} className="hover:bg-slate-50/80">
                        <td className="px-3 py-2 font-mono text-slate-500">+{pt.step}</td>
                        <td className="px-3 py-2 font-semibold text-slate-900">{pt.period_label}</td>
                        <td className="px-3 py-2 text-slate-600">{pt.academic_term}</td>
                        <td className="px-3 py-2 text-right font-mono text-teal-800">
                          {pt.baseline_forecast}
                        </td>
                        <td className="px-3 py-2 text-right font-mono font-bold text-purple-900">
                          {pt.esoteric_forecast}
                        </td>
                        <td
                          className={`px-3 py-2 text-right font-mono font-semibold ${
                            pt.delta_count > 0
                              ? 'text-purple-700'
                              : pt.delta_count < 0
                                ? 'text-teal-700'
                                : 'text-slate-500'
                          }`}
                        >
                          {pt.delta_count > 0 ? `+${pt.delta_count}` : pt.delta_count}
                        </td>
                        <td className="px-3 py-2 text-right font-mono text-slate-600">
                          {pt.lower_bound}–{pt.upper_bound}
                        </td>
                        <td className="px-3 py-2 text-right font-mono text-slate-700">
                          {pt.rough_levy_area.toFixed(3)}
                        </td>
                        <td className="px-3 py-2 text-right font-mono text-slate-700">
                          {pt.hawkes_intensity.toFixed(1)}
                        </td>
                        <td className="px-3 py-2 text-right font-mono text-slate-600">
                          {pt.dispersion_r.toFixed(1)}
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

