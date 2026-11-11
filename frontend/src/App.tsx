import React, { useEffect, useState, useCallback } from 'react';
import {
  Shield,
  LayoutDashboard,
  Sliders,
  Scale,
  TrendingUp,
  Database,
  Activity,
  AlertTriangle,
  CheckCircle2,
  Building2,
  Lock,
} from 'lucide-react';
import { apiClient } from './lib/api';
import { OverviewPage } from './pages/OverviewPage';
import { ForecastExplorerPage } from './pages/ForecastExplorerPage';
import { ModelComparisonPage } from './pages/ModelComparisonPage';
import { TrendsInsightsPage } from './pages/TrendsInsightsPage';
import { DataManagementPage } from './pages/DataManagementPage';
import { EsotericTelemetryPage } from './pages/EsotericTelemetryPage';
import type {
  BacktestResponse,
  DatasetSummaryResponse,
  ForecastResponse,
  HistoricalStatsResponse,
  ModelsAvailabilityResponse,
  ScenarioAssumptions,
  TrendsInsightsResponse,
} from './types/api';

type NavTab = 'overview' | 'explorer' | 'comparison' | 'trends' | 'telemetry' | 'data';

const DEFAULT_SCENARIO: ScenarioAssumptions = {
  enrolment_growth_pct: 0,
  teaching_days_delta: 0,
  attendance_rate_delta_pp: 0,
  reporting_change_assumed: null,
};

export function App() {
  const [activeTab, setActiveTab] = useState<NavTab>('overview');

  // Shared filter and model states
  const [selectedModel, setSelectedModel] = useState<string>('lightgbm_mlforecast');
  const [selectedHorizon, setSelectedHorizon] = useState<number>(12);
  const [selectedProvision, setSelectedProvision] = useState<string>('All');
  const [selectedCategory, setSelectedCategory] = useState<string>('All');
  const [selectedPI, setSelectedPI] = useState<number>(80);
  const [selectedAcademicYearFilter, setSelectedAcademicYearFilter] = useState<string>('All');
  const [scenario, setScenario] = useState<ScenarioAssumptions>(DEFAULT_SCENARIO);
  const [backtestHorizon, setBacktestHorizon] = useState<number>(12);

  // API Data states
  const [datasetSummary, setDatasetSummary] = useState<DatasetSummaryResponse | null>(null);
  const [modelsAvailability, setModelsAvailability] = useState<ModelsAvailabilityResponse | null>(null);
  const [forecastData, setForecastData] = useState<ForecastResponse | null>(null);
  const [historicalStats, setHistoricalStats] = useState<HistoricalStatsResponse | null>(null);
  const [backtestData, setBacktestData] = useState<BacktestResponse | null>(null);
  const [trendsData, setTrendsData] = useState<TrendsInsightsResponse | null>(null);
  const [recentRuns, setRecentRuns] = useState<any[]>([]);

  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Initial metadata & dataset summary fetch
  const fetchCoreMetadata = useCallback(async () => {
    try {
      const [dsSum, modAvail, runsRes] = await Promise.all([
        apiClient.getDatasetSummary(),
        apiClient.getModelsAvailability(),
        apiClient.getRecentForecastRuns(15),
      ]);
      setDatasetSummary(dsSum);
      setModelsAvailability(modAvail);
      setRecentRuns(runsRes.runs || []);

      // If current selectedModel became unavailable (e.g. TimesFM disabled by governance toggle), fallback cleanly
      const currMod = modAvail.models.find((m) => m.model_id === selectedModel);
      if (currMod && !currMod.available) {
        setSelectedModel('lightgbm_mlforecast');
      }
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to connect to Safeguarding Forecast AI backend.');
    }
  }, [selectedModel]);

  // Fetch forecast + historical stats when controls change
  const fetchForecastAndHistory = useCallback(async () => {
    setIsLoading(true);
    setErrorMsg(null);
    try {
      const activeScenario = activeTab === 'explorer' ? scenario : DEFAULT_SCENARIO;
      const [fRes, hRes] = await Promise.all([
        apiClient.generateForecast({
          model_id: selectedModel,
          horizon: selectedHorizon,
          provision: selectedProvision,
          safeguarding_category: selectedCategory,
          prediction_interval: selectedPI,
          academic_year_filter: selectedAcademicYearFilter,
          timesfm_multivariate: true,
          scenario: activeScenario,
        }),
        apiClient.getHistoricalStats(
          selectedAcademicYearFilter,
          selectedProvision,
          selectedCategory
        ),
      ]);
      setForecastData(fRes);
      setHistoricalStats(hRes);
    } catch (err: any) {
      setErrorMsg(err.message || 'Error generating forecast.');
    } finally {
      setIsLoading(false);
    }
  }, [
    selectedModel,
    selectedHorizon,
    selectedProvision,
    selectedCategory,
    selectedPI,
    selectedAcademicYearFilter,
    scenario,
    activeTab,
  ]);

  // Fetch backtest and trends when slice changes
  const fetchBacktestAndTrends = useCallback(async () => {
    try {
      const [btRes, trRes] = await Promise.all([
        apiClient.getBacktestComparison(selectedProvision, selectedCategory, backtestHorizon),
        apiClient.getTrendsInsights(selectedProvision, selectedCategory),
      ]);
      setBacktestData(btRes);
      setTrendsData(trRes);
    } catch (err: any) {
      console.error('Backtest/Trends fetch error:', err);
    }
  }, [selectedProvision, selectedCategory, backtestHorizon]);

  useEffect(() => {
    fetchCoreMetadata();
  }, [fetchCoreMetadata]);

  useEffect(() => {
    fetchForecastAndHistory();
  }, [fetchForecastAndHistory]);

  useEffect(() => {
    fetchBacktestAndTrends();
  }, [fetchBacktestAndTrends]);

  const handleDatasetUpdated = async (newSummary: DatasetSummaryResponse) => {
    setDatasetSummary(newSummary);
    setSelectedProvision('All');
    setSelectedCategory('All');
    setSelectedAcademicYearFilter('All');
    await fetchCoreMetadata();
    await fetchForecastAndHistory();
    await fetchBacktestAndTrends();
  };

  const handleConfigUpdated = async () => {
    await fetchCoreMetadata();
    await fetchForecastAndHistory();
    await fetchBacktestAndTrends();
  };

  const navItems: { id: NavTab; label: string; icon: React.ReactNode }[] = [
    { id: 'overview', label: 'Overview', icon: <LayoutDashboard className="h-4 w-4" /> },
    { id: 'explorer', label: 'Forecast Explorer', icon: <Sliders className="h-4 w-4" /> },
    { id: 'comparison', label: 'Model Comparison', icon: <Scale className="h-4 w-4" /> },
    { id: 'trends', label: 'Trends & Insights', icon: <TrendingUp className="h-4 w-4" /> },
    { id: 'telemetry', label: 'Early-Warning Telemetry', icon: <Activity className="h-4 w-4" /> },
    { id: 'data', label: 'Data Management', icon: <Database className="h-4 w-4" /> },
  ];

  const isSynthetic = datasetSummary?.is_synthetic ?? true;
  const bannerText =
    datasetSummary?.governance_banner ||
    'SYNTHETIC DEMONSTRATION DATA — NOT ACTUAL COLLEGE SAFEGUARDING RECORDS';

  return (
    <div className="min-h-dvh flex flex-col bg-slate-50 text-slate-900">
      {/* Mandatory Prominent Synthetic Data / Governance Banner */}
      <div
        role="status"
        aria-live="polite"
        className={`w-full px-4 py-2 text-center text-xs font-extrabold tracking-wide uppercase border-b flex items-center justify-center gap-2 ${
          isSynthetic
            ? 'bg-amber-400/95 text-slate-950 border-amber-500'
            : 'bg-teal-900 text-teal-50 border-teal-800'
        }`}
      >
        <AlertTriangle className="h-4 w-4 shrink-0" />
        <span>{bannerText}</span>
        <span className="hidden md:inline font-semibold opacity-85">
          • Decision-Support Prototype (Aggregate Counts Only — Never Scores Individual Students)
        </span>
      </div>

      {/* Deep Navy Top Navigation Header */}
      <header className="sticky top-0 z-30 bg-slate-900 text-white shadow-md border-b border-slate-800">
        <div className="mx-auto max-w-[1440px] px-4 sm:px-6">
          <div className="flex flex-col gap-3 py-3.5 lg:flex-row lg:items-center lg:justify-between">
            {/* Brand Identity */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-teal-400 to-blue-600 text-slate-950 shadow-sm">
                  <Shield className="h-6 w-6 stroke-[2.2]" />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-display text-lg font-extrabold tracking-tight text-white">
                      Safeguarding Forecast AI
                    </span>
                    <span className="rounded-md bg-teal-500/20 px-2 py-0.5 text-[11px] font-bold text-teal-300 border border-teal-400/30">
                      South Devon College
                    </span>
                  </div>
                  <p className="text-xs text-slate-300 font-medium">
                    Predictive Safeguarding Demand Intelligence
                  </p>
                </div>
              </div>
            </div>

            {/* Navigation Tabs */}
            <nav aria-label="Main Dashboard Navigation" className="flex flex-wrap items-center gap-1.5">
              {navItems.map((item) => {
                const isActive = activeTab === item.id;
                return (
                  <button
                    key={item.id}
                    onClick={() => setActiveTab(item.id)}
                    className={`inline-flex items-center gap-2 rounded-xl px-3.5 py-2 text-xs sm:text-sm font-semibold transition-all cursor-pointer ${
                      isActive
                        ? 'bg-teal-500 text-slate-950 shadow-xs'
                        : 'text-slate-300 hover:bg-slate-800 hover:text-white'
                    }`}
                  >
                    {item.icon}
                    <span>{item.label}</span>
                  </button>
                );
              })}
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="mx-auto w-full max-w-[1440px] flex-1 px-4 py-6 sm:px-6">
        {errorMsg && (
          <div className="mb-6 rounded-2xl border border-rose-300 bg-rose-50 p-4 text-xs sm:text-sm text-rose-950 flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <AlertTriangle className="h-5 w-5 text-rose-600 shrink-0" />
              <span>{errorMsg}</span>
            </div>
            <button
              onClick={() => {
                fetchCoreMetadata();
                fetchForecastAndHistory();
                fetchBacktestAndTrends();
              }}
              className="rounded-lg bg-rose-700 px-3 py-1.5 text-xs font-semibold text-white hover:bg-rose-800 cursor-pointer"
            >
              Retry
            </button>
          </div>
        )}

        {!datasetSummary || !forecastData || !historicalStats || !modelsAvailability ? (
          <div className="flex h-96 flex-col items-center justify-center gap-3 rounded-2xl border border-slate-200 bg-white p-8">
            <div className="h-8 w-8 animate-spin rounded-full border-4 border-teal-600 border-t-transparent" />
            <p className="text-sm font-semibold text-slate-700">
              Initialising Safeguarding Forecast AI & Statistical/ML Engines...
            </p>
          </div>
        ) : (
          <>
            {activeTab === 'overview' && (
              <OverviewPage
                datasetSummary={datasetSummary}
                forecastData={forecastData}
                historicalStats={historicalStats}
                models={modelsAvailability.models}
                selectedModel={selectedModel}
                onSelectModel={setSelectedModel}
                selectedHorizon={selectedHorizon}
                onSelectHorizon={setSelectedHorizon}
                selectedProvision={selectedProvision}
                onSelectProvision={setSelectedProvision}
                selectedCategory={selectedCategory}
                onSelectCategory={setSelectedCategory}
                selectedPI={selectedPI}
                onSelectPI={setSelectedPI}
                selectedAcademicYearFilter={selectedAcademicYearFilter}
                onSelectAcademicYearFilter={setSelectedAcademicYearFilter}
                isLoading={isLoading}
              />
            )}

            {activeTab === 'explorer' && (
              <ForecastExplorerPage
                datasetSummary={datasetSummary}
                forecastData={forecastData}
                models={modelsAvailability.models}
                selectedModel={selectedModel}
                onSelectModel={setSelectedModel}
                selectedHorizon={selectedHorizon}
                onSelectHorizon={setSelectedHorizon}
                selectedProvision={selectedProvision}
                onSelectProvision={setSelectedProvision}
                selectedCategory={selectedCategory}
                onSelectCategory={setSelectedCategory}
                selectedPI={selectedPI}
                onSelectPI={setSelectedPI}
                scenario={scenario}
                onChangeScenario={setScenario}
                onResetScenario={() => setScenario(DEFAULT_SCENARIO)}
                isLoading={isLoading}
              />
            )}

            {activeTab === 'comparison' && backtestData && (
              <ModelComparisonPage
                datasetSummary={datasetSummary}
                backtestData={backtestData}
                modelsAvailability={modelsAvailability.models}
                selectedProvision={selectedProvision}
                onSelectProvision={setSelectedProvision}
                selectedCategory={selectedCategory}
                onSelectCategory={setSelectedCategory}
                backtestHorizon={backtestHorizon}
                onSelectBacktestHorizon={setBacktestHorizon}
                onSelectActiveModel={setSelectedModel}
                activeModelId={selectedModel}
                isLoading={isLoading}
              />
            )}

            {activeTab === 'trends' && trendsData && (
              <TrendsInsightsPage
                datasetSummary={datasetSummary}
                trendsData={trendsData}
                selectedProvision={selectedProvision}
                onSelectProvision={setSelectedProvision}
                selectedCategory={selectedCategory}
                onSelectCategory={setSelectedCategory}
                isLoading={isLoading}
              />
            )}

            {activeTab === 'telemetry' && (
              <EsotericTelemetryPage
                selectedProvision={selectedProvision}
                onSelectProvision={setSelectedProvision}
              />
            )}

            {activeTab === 'data' && (
              <DataManagementPage
                datasetSummary={datasetSummary}
                modelsAvailability={modelsAvailability}
                recentRuns={recentRuns}
                onDatasetUpdated={handleDatasetUpdated}
                onConfigUpdated={handleConfigUpdated}
              />
            )}
          </>
        )}
      </main>

      {/* Governance & Ethical Safeguarding Footer */}
      <footer className="mt-12 border-t border-slate-200 bg-white py-5 text-xs text-slate-500">
        <div className="mx-auto max-w-[1440px] px-4 sm:px-6 flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div className="space-y-1">
            <div className="font-bold text-slate-800 flex items-center gap-1.5">
              <Lock className="h-3.5 w-3.5 text-teal-600" />
              Safeguarding Forecast AI • South Devon College Principalship Prototype
            </div>
            <p>
              Forecasts aggregate operational service demand only. Never profiles individual students, never predicts individual risk, and never recommends reducing safeguarding support. Formal Information Governance (DPIA) and Designated Safeguarding Lead review is required before importing real college statistics.
            </p>
          </div>
          <div className="shrink-0 text-right font-medium text-slate-600">
            <div>Local Execution Only • Zero Third-Party Cloud Data Transmission</div>
            <div className="text-[11px] text-slate-400">
              Active Dataset: {datasetSummary?.dataset_id || 'synthetic_sdc_default'}
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;

