import React, { useEffect, useState, useCallback, useMemo } from 'react';
import {
  Activity,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  Compass,
  GitBranch,
  Layers,
  Lock,
  ShieldAlert,
  Sparkles,
  Calendar,
} from 'lucide-react';
import { apiClient } from '../lib/api';
import {
  Badge,
  Button,
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '../components/ui/primitives';

interface EsotericTelemetryPageProps {
  selectedProvision: string;
  onSelectProvision: (prov: string) => void;
}

export const EsotericTelemetryPage: React.FC<EsotericTelemetryPageProps> = ({
  selectedProvision,
  onSelectProvision,
}) => {
  const [data, setData] = useState<any | null>(null);
  const [selectedDate, setSelectedDate] = useState<string | null>(null);
  const [retroMode, setRetroMode] = useState<string>('STAGED_CORE_4');
  const [lagBufferDays, setLagBufferDays] = useState<number>(0);
  const [showResidualOverlay, setShowResidualOverlay] = useState<boolean>(true);
  const [isAuditOpen, setIsAuditOpen] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Unified interactive hover states
  const [hoveredSparkIdx, setHoveredSparkIdx] = useState<number | null>(null);
  const [hoveredHawkesIdx, setHoveredHawkesIdx] = useState<number | null>(null);
  const [hoveredSheafNode, setHoveredSheafNode] = useState<number | null>(null);
  const [hoveredOverlayIdx, setHoveredOverlayIdx] = useState<number | null>(null);

  const loadTelemetry = useCallback(
    async (
      targetDate: string | null,
      provision: string,
      mode: string,
      lagDays: number
    ) => {
      setIsLoading(true);
      setError(null);
      try {
        const res = await apiClient.getEsotericTelemetry(
          targetDate || undefined,
          provision,
          mode,
          lagDays
        );
        setData(res);
        if (!targetDate && res?.telemetry?.selected_date) {
          setSelectedDate(res.telemetry.selected_date);
        }
      } catch (err: any) {
        setError(err.message || 'Unable to load Mathematical Early-Warning Telemetry.');
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  useEffect(() => {
    loadTelemetry(selectedDate, selectedProvision, retroMode, lagBufferDays);
  }, [selectedDate, selectedProvision, retroMode, lagBufferDays, loadTelemetry]);

  const scrubber = data?.scrubber;
  const telemetry = data?.telemetry;
  const retroOverlay = data?.retrospective_overlay;
  const backfillAudit = data?.backfill_audit;
  const availableDates: string[] = scrubber?.available_dates || [];
  const minStep: number = scrubber?.min_step_index ?? 30;
  const maxStep: number = scrubber?.max_step_index ?? Math.max(30, availableDates.length - 1);

  const activeStepIndex = useMemo(() => {
    if (!telemetry) return maxStep;
    const byDateIdx = availableDates.indexOf(telemetry.selected_date);
    if (byDateIdx >= 0) return byDateIdx;
    return telemetry.selected_step_index ?? maxStep;
  }, [telemetry, availableDates, maxStep]);

  const handleSliderChange = (stepIdx: number) => {
    const clamped = Math.max(minStep, Math.min(maxStep, stepIdx));
    const dt = availableDates[clamped];
    if (dt) {
      setSelectedDate(dt);
    }
  };

  if (!data && isLoading) {
    return (
      <div className="flex h-96 flex-col items-center justify-center gap-3 rounded-2xl border border-slate-200 bg-white p-8">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-teal-600 border-t-transparent" />
        <p className="text-sm font-semibold text-slate-700">
          Computing Causal Hilbert Manifold, Hawkes Branching Tensor &amp; Cellular Sheaf Cohomology...
        </p>
      </div>
    );
  }

  if (error || !telemetry) {
    return (
      <Card className="border-rose-300 bg-rose-50/50">
        <CardContent className="p-6 text-sm text-rose-900 flex items-center justify-between">
          <span>{error || 'Telemetry payload unavailable.'}</span>
          <Button
            size="sm"
            variant="danger"
            onClick={() => loadTelemetry(selectedDate, selectedProvision, retroMode, lagBufferDays)}
          >
            Retry
          </Button>
        </CardContent>
      </Card>
    );
  }

  const hilbert = telemetry.hilbert_clock;
  const hawkes = telemetry.hawkes_cascade;
  const sheaf = telemetry.sheaf_frustration;
  const kpiRibbon: any[] = telemetry.kpi_ribbon || [];
  const aux = telemetry.auxiliary_manifold_metrics || {};

  // Active hover point for Hilbert Clock & 60-Day Sparkline
  const sparkPts: any[] = hilbert.sparkline_60 || [];
  const activeHilbertHover =
    hoveredSparkIdx !== null && sparkPts[hoveredSparkIdx]
      ? sparkPts[hoveredSparkIdx]
      : sparkPts[sparkPts.length - 1];

  // Polar Dial Geometry
  const dialCx = 195;
  const dialCy = 185;
  const dialRadius = 138;
  const dialScale = dialRadius / Math.max(hilbert.dial_max_r || 2.5, 1.0);
  const cometPts: any[] = hilbert.comet_trajectory || [];
  const cometHead = cometPts[cometPts.length - 1];

  // Hawkes Ribbon Geometry
  const hawkesHist: any[] = hawkes.ribbon_series || [];
  const hawkesCone: any[] = hawkes.projection_cone || [];
  const hawkesAll = [...hawkesHist, ...hawkesCone];
  const hawkesCats: any[] = hawkes.categories || [];
  const hx0 = 46;
  const hx1 = 620;
  const hy0 = 26;
  const hy1 = 248;
  const hawkesMaxY = Math.max(
    2.0,
    ...hawkesHist.map((d) => Math.max(d.lambda_total, d.observed_count)),
    ...hawkesCone.map((d) => d.upper_cone)
  ) * 1.16;

  const hXp = (i: number) =>
    hx0 + ((hx1 - hx0) * i) / Math.max(1, hawkesAll.length - 1);
  const hYp = (v: number) => hy1 - ((hy1 - hy0) * v) / hawkesMaxY;

  // Build stacked polygon strings for Hawkes
  const exogPolyPts = [
    `${hx0},${hy1}`,
    ...hawkesAll.map((s, i) => `${hXp(i).toFixed(1)},${hYp(s.exog_baseline).toFixed(1)}`),
    `${hXp(hawkesAll.length - 1).toFixed(1)},${hy1}`,
  ].join(' ');

  const catPolygons: { color: string; label: string; points: string }[] = [];
  let runningLower = hawkesAll.map((s) => Number(s.exog_baseline));
  hawkesCats.forEach((cat, j) => {
    const runningUpper = hawkesAll.map(
      (s, i) => runningLower[i] + Number(s.endog_by_source?.[j] || 0)
    );
    const pts = [
      ...runningUpper.map((u, i) => `${hXp(i).toFixed(1)},${hYp(u).toFixed(1)}`),
      ...runningLower
        .map((l, i) => `${hXp(i).toFixed(1)},${hYp(l).toFixed(1)}`)
        .reverse(),
    ].join(' ');
    catPolygons.push({ color: cat.color, label: cat.label, points: pts });
    runningLower = runningUpper;
  });

  const activeHawkesHover =
    hoveredHawkesIdx !== null && hawkesAll[hoveredHawkesIdx]
      ? hawkesAll[hoveredHawkesIdx]
      : hawkesHist[hawkesHist.length - 1];

  // Sheaf Dirichlet Strip Geometry
  const sheafStrip: any[] = sheaf.energy_timeline_60 || [];
  const sx0 = 48;
  const sx1 = 920;
  const sy0 = 18;
  const sy1 = 96;
  const sheafMaxE =
    Math.max(
      sheaf.current.tau975 * 1.22,
      1.0,
      ...sheafStrip.map((d) => Number(d.dirichlet_energy))
    );
  const sXp = (i: number) =>
    sx0 + ((sx1 - sx0) * i) / Math.max(1, sheafStrip.length - 1);
  const sYp = (v: number) => sy1 - ((sy1 - sy0) * v) / sheafMaxE;

  return (
    <div className="space-y-5">
      {/* Page Header & Provision Filter */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <h1 className="font-display text-xl sm:text-2xl font-extrabold tracking-tight text-slate-900">
              Mathematical Early-Warning Telemetry
            </h1>
            <Badge variant="teal">Strictly Causal (s &gt; t Masked)</Badge>
            <Badge variant="slate">{telemetry.topology_mode}</Badge>
          </div>
          <p className="mt-1 text-xs sm:text-sm text-slate-600">
            Low-cognitive-load pastoral phase clocks, peer contagion branching cascades, and sheaf cohomological contextual masking alerts.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-1.5">
            <label htmlFor="retro-mode-select" className="text-xs font-semibold text-slate-600">
              Rollout Mode:
            </label>
            <select
              id="retro-mode-select"
              data-testid="retro-mode-select"
              value={retroMode}
              onChange={(e) => setRetroMode(e.target.value)}
              className="rounded-xl border border-teal-300 bg-teal-50/70 px-3 py-1.5 text-xs font-bold text-slate-900 shadow-2xs focus:border-teal-600 focus:outline-none"
            >
              <option value="STAGED_CORE_4">STAGED_CORE_4 (Default High-Evidence Stack)</option>
              <option value="TELEMETRY_ONLY">TELEMETRY_ONLY (Zero Forecast Change)</option>
              <option value="FULL_13_GATED">FULL_13_GATED (13-Module 1-SE Quarantine)</option>
            </select>
          </div>

          <div className="flex items-center gap-1.5">
            <label htmlFor="telemetry-provision" className="text-xs font-semibold text-slate-600">
              Cohort Scope:
            </label>
            <select
              id="telemetry-provision"
              value={selectedProvision}
              onChange={(e) => onSelectProvision(e.target.value)}
              className="rounded-xl border border-slate-300 bg-white px-3 py-1.5 text-xs font-semibold text-slate-800 shadow-2xs focus:border-teal-500 focus:outline-none"
            >
              <option value="All">Whole College (All Provisions)</option>
              <option value="EPYP">16–18 Study Programmes &amp; EPYP</option>
              <option value="Adult">19+ Adult Provision</option>
              <option value="Apprenticeships">Apprenticeships</option>
            </select>
          </div>
        </div>
      </div>

      {/* Top 3-Card Summary KPI Ribbon (Immediately visible above the fold) */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3" data-testid="telemetry-kpi-ribbon">
        {kpiRibbon.map((kpi) => (
          <Card
            key={kpi.id}
            className="relative overflow-hidden border-l-4"
            style={{ borderLeftColor: kpi.status_color }}
          >
            <CardContent className="p-4">
              <div className="flex items-center justify-between gap-2">
                <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
                  {kpi.title}
                </span>
                {kpi.is_breach && (
                  <span className="rounded-full bg-red-100 px-2 py-0.5 text-[10px] font-extrabold text-red-700">
                    2σ BREACH
                  </span>
                )}
              </div>
              <div className="mt-1.5 flex items-baseline justify-between gap-2">
                <span
                  className="font-display text-lg font-extrabold tracking-tight"
                  style={{ color: kpi.status_color }}
                >
                  {kpi.primary_value}
                </span>
                <span className="rounded-lg bg-slate-100 px-2 py-0.5 text-xs font-bold text-slate-800">
                  {kpi.secondary_metric}
                </span>
              </div>
              <p className="mt-1 text-xs text-slate-600">{kpi.subtitle}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Shared Interactive Causal Time Scrubber */}
      <Card className="border-slate-200/90 bg-white" data-testid="telemetry-time-scrubber">
        <CardContent className="p-4">
          <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            <div className="flex flex-wrap items-center gap-2.5">
              <div className="flex items-center gap-1.5 text-xs font-bold text-slate-800">
                <Calendar className="h-4 w-4 text-teal-600" />
                <span>Active Evaluation Date (t):</span>
              </div>
              <span className="rounded-lg bg-slate-900 px-2.5 py-1 font-mono text-xs font-bold text-teal-300">
                {telemetry.selected_date}
              </span>
              <span className="text-xs font-medium text-slate-600">
                Acad Day <strong>{hilbert.current.t_acad}</strong>
              </span>
              {hilbert.current.is_holiday_freeze && (
                <Badge variant="amber">
                  Phase Frozen: {hilbert.current.holiday_label || 'Non-Instructional Break'}
                </Badge>
              )}
              {isLoading && (
                <span className="text-[11px] font-semibold text-teal-600 animate-pulse">
                  Updating causal prefix...
                </span>
              )}
            </div>

            <div className="flex items-center gap-1.5">
              <Button
                size="sm"
                variant="outline"
                aria-label="Previous day"
                disabled={activeStepIndex <= minStep}
                onClick={() => handleSliderChange(activeStepIndex - 1)}
              >
                <ChevronLeft className="h-3.5 w-3.5" />
                <span>-1d</span>
              </Button>
              <Button
                size="sm"
                variant="outline"
                aria-label="Next day"
                disabled={activeStepIndex >= maxStep}
                onClick={() => handleSliderChange(activeStepIndex + 1)}
              >
                <span>+1d</span>
                <ChevronRight className="h-3.5 w-3.5" />
              </Button>
              <span className="ml-2 hidden xl:inline text-[11px] text-slate-500">
                RMT Sync {(Number(aux.rmt_sync_ratio || 0) * 100).toFixed(1)}% · Fisher-Rao ds={Number(aux.fisher_rao_velocity || 0).toFixed(3)} · FracDiff d*={aux.fractional_d_star}
              </span>
            </div>
          </div>

          <div className="mt-3">
            <input
              aria-label="Causal Telemetry Time Scrubber"
              type="range"
              min={minStep}
              max={maxStep}
              value={activeStepIndex}
              onChange={(e) => handleSliderChange(Number(e.target.value))}
              className="w-full accent-teal-600 cursor-pointer"
            />
          </div>

          {/* Academic Calendar Landmarks */}
          <div className="mt-2.5 flex flex-wrap items-center gap-1.5">
            <span className="text-[11px] font-semibold text-slate-500 mr-1">
              Academic Landmarks:
            </span>
            {(scrubber?.landmarks || []).map((lm: any) => {
              const isSelected = telemetry.selected_date === lm.date;
              return (
                <button
                  key={lm.label}
                  type="button"
                  onClick={() => setSelectedDate(lm.date)}
                  title={lm.description}
                  className={`rounded-full px-2.5 py-1 text-[11px] font-semibold transition-colors cursor-pointer ${
                    isSelected
                      ? 'bg-teal-600 text-white shadow-2xs'
                      : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                  }`}
                >
                  {lm.label} ({lm.date})
                </button>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* RETROSPECTIVE OVERLAY TOGGLE: FROZEN BASELINE VS. ESOTERIC RESIDUAL ADJUSTMENT */}
      {retroOverlay && (
        <Card
          className="border-slate-200/90 bg-white"
          data-testid="retrospective-overlay-panel"
        >
          <CardContent className="p-4 space-y-3">
            <div className="flex flex-col gap-2.5 lg:flex-row lg:items-center lg:justify-between">
              <div className="flex flex-wrap items-center gap-2.5">
                <Lock className="h-4 w-4 text-teal-700" />
                <span className="text-sm font-extrabold text-slate-900">
                  Post-Hoc Residual Stacking Overlay
                </span>
                <Badge variant={retroOverlay.identity_floor_active ? 'slate' : 'teal'}>
                  {retroOverlay.retro_mode}
                </Badge>
                <span
                  data-testid="retrospective-attribution-badge"
                  className={`rounded-full px-2.5 py-0.5 text-xs font-bold ${
                    retroOverlay.identity_floor_active ||
                    Math.abs(Number(retroOverlay.current_step?.delta_pct || 0)) < 0.05
                      ? 'bg-slate-100 text-slate-700'
                      : Number(retroOverlay.current_step?.delta_pct || 0) > 0
                      ? 'bg-amber-100 text-amber-900'
                      : 'bg-teal-100 text-teal-900'
                  }`}
                >
                  {retroOverlay.current_step?.attribution || '0.0% — Frozen Baseline Retained'}
                </span>
              </div>

              <label
                htmlFor="retro-overlay-checkbox"
                className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs font-bold text-slate-800 cursor-pointer select-none hover:bg-slate-100"
              >
                <input
                  id="retro-overlay-checkbox"
                  data-testid="retrospective-overlay-toggle"
                  type="checkbox"
                  checked={showResidualOverlay}
                  onChange={(e) => setShowResidualOverlay(e.target.checked)}
                  className="h-4 w-4 rounded border-slate-300 accent-teal-600 cursor-pointer"
                />
                <span>Show Frozen Baseline vs. Esoteric Residual Adjustment</span>
              </label>
            </div>

            {showResidualOverlay && (() => {
              const ovPts: any[] = retroOverlay.series || [];
              if (ovPts.length === 0) return null;
              const activeOv =
                hoveredOverlayIdx !== null && ovPts[hoveredOverlayIdx]
                  ? ovPts[hoveredOverlayIdx]
                  : retroOverlay.current_step || ovPts[ovPts.length - 1];

              const ow = 940;
              const oh = 152;
              const ox0 = 46;
              const ox1 = 918;
              const oy0 = 16;
              const oy1 = 128;
              const maxY =
                Math.max(
                  4.0,
                  ...ovPts.map((p) =>
                    Math.max(
                      Number(p.observed_count || 0),
                      Number(p.frozen_baseline_mu || 0),
                      Number(p.adjusted_final_mu || 0),
                      Number(p.upper_95_adjusted || 0) * 0.65
                    )
                  )
                ) * 1.16;

              const ox = (i: number) =>
                ox0 + ((ox1 - ox0) * i) / Math.max(1, ovPts.length - 1);
              const oy = (v: number) => oy1 - ((oy1 - oy0) * Math.min(maxY, Math.max(0, v))) / maxY;

              const frozenLine = ovPts
                .map((p, i) => `${ox(i).toFixed(1)},${oy(p.frozen_baseline_mu).toFixed(1)}`)
                .join(' ');
              const adjLine = ovPts
                .map((p, i) => `${ox(i).toFixed(1)},${oy(p.adjusted_final_mu).toFixed(1)}`)
                .join(' ');
              const deltaBandPoly = [
                ...ovPts.map((p, i) => `${ox(i).toFixed(1)},${oy(p.adjusted_final_mu).toFixed(1)}`),
                ...ovPts
                  .map((p, i) => `${ox(i).toFixed(1)},${oy(p.frozen_baseline_mu).toFixed(1)}`)
                  .reverse(),
              ].join(' ');
              const cutoffIdx = ovPts.findIndex((p) => p.is_projection);
              const splitIdx = cutoffIdx >= 0 ? Math.max(0, cutoffIdx - 1) : ovPts.length - 1;

              return (
                <div
                  className="rounded-xl border border-slate-200 bg-slate-50/50 p-3 space-y-2"
                  data-testid="retrospective-overlay-chart"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-700">
                    <div className="flex flex-wrap items-center gap-3">
                      <span className="font-mono font-bold text-slate-900">{activeOv.date}</span>
                      <span>
                        Frozen Baseline μ_base:{' '}
                        <strong>{Number(activeOv.frozen_baseline_mu).toFixed(2)}</strong>
                      </span>
                      <span>
                        Post-Hoc Adjusted μ_final:{' '}
                        <strong className="text-teal-700">
                          {Number(activeOv.adjusted_final_mu).toFixed(2)}
                        </strong>
                      </span>
                      <span>
                        95% PI: [<strong>{activeOv.lower_95_adjusted}</strong>,{' '}
                        <strong>{activeOv.upper_95_adjusted}</strong>]
                      </span>
                    </div>
                    <span className="font-semibold text-slate-800">{activeOv.attribution}</span>
                  </div>

                  <svg
                    viewBox={`0 0 ${ow} ${oh}`}
                    className="w-full select-none"
                    onMouseLeave={() => setHoveredOverlayIdx(null)}
                  >
                    {/* Translucent Residual Delta Band between Frozen Baseline and Adjusted Expectation */}
                    <polygon
                      points={deltaBandPoly}
                      fill={retroOverlay.identity_floor_active ? '#94A3B8' : '#0D9488'}
                      fillOpacity={retroOverlay.identity_floor_active ? '0.08' : '0.24'}
                    />

                    {/* Causal Cutoff Vertical Marker */}
                    <line
                      x1={ox(splitIdx)}
                      y1={oy0}
                      x2={ox(splitIdx)}
                      y2={oy1}
                      stroke="#0F172A"
                      strokeWidth="1.2"
                      strokeDasharray="4,3"
                    />

                    {/* Frozen Baseline Curve (100% Immutable SHA-256) */}
                    <polyline
                      fill="none"
                      stroke="#475569"
                      strokeWidth="1.9"
                      strokeDasharray="5,3"
                      points={frozenLine}
                    />

                    {/* Post-Hoc Esoteric Adjusted Curve */}
                    <polyline
                      fill="none"
                      stroke="#0D9488"
                      strokeWidth="2.3"
                      points={adjLine}
                    />

                    {/* Observed Historical Counts ( strictly <= t ) */}
                    {ovPts.map((p, i) => {
                      if (p.is_projection || p.observed_count === null) return null;
                      return (
                        <circle
                          key={`ov-obs-${p.date}`}
                          cx={ox(i)}
                          cy={oy(p.observed_count)}
                          r="2.4"
                          fill="#0F172A"
                        />
                      );
                    })}

                    {/* Hover Columns */}
                    {ovPts.map((p, i) => {
                      const colW = (ox1 - ox0) / Math.max(1, ovPts.length);
                      return (
                        <rect
                          key={`ov-hov-${i}`}
                          x={ox(i) - colW * 0.5}
                          y={oy0}
                          width={colW}
                          height={oy1 - oy0}
                          fill="transparent"
                          onMouseEnter={() => setHoveredOverlayIdx(i)}
                          className="cursor-crosshair"
                        />
                      );
                    })}

                    <line x1={ox0} y1={oy1} x2={ox1} y2={oy1} stroke="#CBD5E1" strokeWidth="1" />
                    <text x={ox0} y={144} fontSize="9.5" fill="#64748B">
                      {ovPts[0]?.date}
                    </text>
                    <text
                      x={ox(splitIdx) - 7}
                      y={oy0 + 9}
                      textAnchor="end"
                      fontSize="9.5"
                      fontWeight="700"
                      fill="#0F172A"
                    >
                      Cutoff t = {ovPts[splitIdx]?.date}
                    </text>
                    <text x={ox1} y={144} textAnchor="end" fontSize="9.5" fill="#0F766E">
                      +5d Causal Horizon ({ovPts[ovPts.length - 1]?.date})
                    </text>
                  </svg>

                  <div className="flex flex-wrap items-center gap-4 text-[11px] text-slate-600">
                    <span className="inline-flex items-center gap-1.5">
                      <span className="h-0.5 w-4 border-b-2 border-dashed border-slate-600 inline-block" />{' '}
                      Frozen Baseline μ_base (SHA-256 Locked)
                    </span>
                    <span className="inline-flex items-center gap-1.5">
                      <span className="h-0.5 w-4 bg-teal-600 inline-block" /> Post-Hoc Esoteric Adjusted μ_final
                    </span>
                    <span className="inline-flex items-center gap-1.5">
                      <span className="h-2.5 w-3 rounded-xs bg-teal-600/25 inline-block" /> Esoteric Residual Delta Band
                    </span>
                    <span className="inline-flex items-center gap-1.5">
                      <span className="h-2 w-2 rounded-full bg-slate-900 inline-block" /> Observed Daily Count (≤ t)
                    </span>
                  </div>
                </div>
              );
            })()}
          </CardContent>
        </Card>
      )}

      {/* VISUAL SPEC 1: HILBERT STRESS-CYCLE PHASE CLOCK */}
      <Card className="scroll-mt-24" data-testid="hilbert-phase-clock-panel">
        <CardHeader className="pb-2">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <Compass className="h-5 w-5 text-indigo-600" />
              <CardTitle>
                1. Hilbert Stress-Cycle Phase Clock (Warped 190-Day Instructional Manifold)
              </CardTitle>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="slate">Module f01 · Analytic Signal z(t) = r(t)e^iφ(t)</Badge>
            </div>
          </div>
          <CardDescription>
            Polar Argand trajectory normalized by training-fold MAD (1.0σ = normal termly oscillation; 2.0σ dashed crimson = acute amplitude breach). Hollow diamonds mark weekend/holiday phase freezes.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-12 items-center">
            {/* Left: 2D Polar Argand Dial */}
            <div className="lg:col-span-5">
              <svg
                viewBox="0 0 390 370"
                className="w-full max-w-[420px] mx-auto select-none"
                role="img"
                aria-label="Hilbert Stress-Cycle Polar Argand Dial"
              >
                <defs>
                  <marker
                    id="omega-arrow"
                    markerWidth="8"
                    markerHeight="8"
                    refX="6"
                    refY="4"
                    orient="auto"
                  >
                    <path d="M0,1 L7,4 L0,7 Z" fill={hilbert.current.quadrant.color} />
                  </marker>
                </defs>

                {/* Four Pastoral Quadrant Tints */}
                <path
                  d={`M ${dialCx} ${dialCy} L ${dialCx + dialRadius} ${dialCy} A ${dialRadius} ${dialRadius} 0 0 0 ${dialCx} ${dialCy - dialRadius} Z`}
                  fill="rgba(217, 119, 6, 0.11)"
                />
                <path
                  d={`M ${dialCx} ${dialCy} L ${dialCx} ${dialCy - dialRadius} A ${dialRadius} ${dialRadius} 0 0 0 ${dialCx - dialRadius} ${dialCy} Z`}
                  fill="rgba(220, 38, 38, 0.09)"
                />
                <path
                  d={`M ${dialCx} ${dialCy} L ${dialCx - dialRadius} ${dialCy} A ${dialRadius} ${dialRadius} 0 0 0 ${dialCx} ${dialCy + dialRadius} Z`}
                  fill="rgba(13, 148, 136, 0.11)"
                />
                <path
                  d={`M ${dialCx} ${dialCy} L ${dialCx} ${dialCy + dialRadius} A ${dialRadius} ${dialRadius} 0 0 0 ${dialCx + dialRadius} ${dialCy} Z`}
                  fill="rgba(79, 70, 229, 0.11)"
                />

                {/* Crosshairs & Concentric Rings */}
                <line
                  x1={dialCx - dialRadius}
                  y1={dialCy}
                  x2={dialCx + dialRadius}
                  y2={dialCy}
                  stroke="#CBD5E1"
                  strokeWidth="1"
                />
                <line
                  x1={dialCx}
                  y1={dialCy - dialRadius}
                  x2={dialCx}
                  y2={dialCy + dialRadius}
                  stroke="#CBD5E1"
                  strokeWidth="1"
                />
                <circle
                  cx={dialCx}
                  cy={dialCy}
                  r={dialRadius}
                  fill="none"
                  stroke="#CBD5E1"
                  strokeWidth="1.1"
                />
                <circle
                  cx={dialCx}
                  cy={dialCy}
                  r={1.0 * dialScale}
                  fill="none"
                  stroke="#475569"
                  strokeWidth="1.4"
                />
                <circle
                  cx={dialCx}
                  cy={dialCy}
                  r={2.0 * dialScale}
                  fill="none"
                  stroke="#DC2626"
                  strokeWidth="1.6"
                  strokeDasharray="5,4"
                />

                {/* Non-overlapping Quadrant Corner Headers */}
                <text
                  x={dialCx + dialRadius - 4}
                  y={22}
                  textAnchor="end"
                  fontSize="10.5"
                  fontWeight="700"
                  fill="#B45309"
                >
                  QI · Escalating Pressure
                </text>
                <text
                  x={dialCx - dialRadius + 4}
                  y={22}
                  textAnchor="start"
                  fontSize="10.5"
                  fontWeight="700"
                  fill="#B91C1C"
                >
                  QII · Peak Load &amp; Crest
                </text>
                <text
                  x={dialCx - dialRadius + 4}
                  y={354}
                  textAnchor="start"
                  fontSize="10.5"
                  fontWeight="700"
                  fill="#0F766E"
                >
                  QIII · Decompression
                </text>
                <text
                  x={dialCx + dialRadius - 4}
                  y={354}
                  textAnchor="end"
                  fontSize="10.5"
                  fontWeight="700"
                  fill="#4338CA"
                >
                  QIV · Pre-Surge Incubation
                </text>

                {/* 20-Instructional-Day Comet Segments */}
                {cometPts.map((pt, i) => {
                  if (i === 0) return null;
                  const prev = cometPts[i - 1];
                  return (
                    <line
                      key={`seg-${pt.step_index}`}
                      x1={dialCx + prev.x_unit * dialScale}
                      y1={dialCy - prev.y_unit * dialScale}
                      x2={dialCx + pt.x_unit * dialScale}
                      y2={dialCy - pt.y_unit * dialScale}
                      stroke={pt.quadrant_color}
                      strokeWidth={pt.stroke_width}
                      strokeOpacity={pt.opacity}
                      strokeLinecap="round"
                    />
                  );
                })}

                {/* Holiday Phase-Freeze Hollow Diamond Glyphs */}
                {cometPts.map((pt) => {
                  if (!pt.is_holiday_freeze) return null;
                  const px = dialCx + pt.x_unit * dialScale;
                  const py = dialCy - pt.y_unit * dialScale;
                  const d = 4.6;
                  return (
                    <polygon
                      key={`hol-${pt.step_index}`}
                      points={`${px},${py - d} ${px + d},${py} ${px},${py + d} ${px - d},${py}`}
                      fill="#FFFFFF"
                      stroke="#475569"
                      strokeWidth="1.5"
                    >
                      <title>
                        {pt.date} · {pt.holiday_label || 'Holiday Freeze'} (φ={pt.phi_deg}°)
                      </title>
                    </polygon>
                  );
                })}

                {/* Current State Beacon & Tangential Velocity Vector */}
                {cometHead && (
                  <g>
                    {(() => {
                      const hx = dialCx + cometHead.x_unit * dialScale;
                      const hy = dialCy - cometHead.y_unit * dialScale;
                      const tv = hilbert.current.tangent_vector;
                      const tx = hx + tv.dx_unit * dialScale;
                      const ty = hy - tv.dy_unit * dialScale;
                      const bCol = hilbert.current.is_breach_2sigma
                        ? '#DC2626'
                        : hilbert.current.quadrant.color;
                      return (
                        <>
                          {Math.hypot(tx - hx, ty - hy) > 3 && (
                            <line
                              x1={hx}
                              y1={hy}
                              x2={tx}
                              y2={ty}
                              stroke={bCol}
                              strokeWidth="2.1"
                              markerEnd="url(#omega-arrow)"
                            />
                          )}
                          <circle cx={hx} cy={hy} r="10" fill={bCol} fillOpacity="0.22" />
                          <circle
                            cx={hx}
                            cy={hy}
                            r="5.5"
                            fill={bCol}
                            stroke="#FFFFFF"
                            strokeWidth="2"
                          />
                        </>
                      );
                    })()}
                  </g>
                )}

                {/* Ring Labels (rendered on top with white halo for 100% legibility) */}
                <text
                  x={dialCx + 5}
                  y={dialCy - 1.0 * dialScale + 12}
                  fontSize="9.5"
                  fontWeight="600"
                  fill="#475569"
                  paintOrder="stroke"
                  stroke="#FFFFFF"
                  strokeWidth="3.5"
                  strokeLinejoin="round"
                >
                  1.0σ Normal
                </text>
                <text
                  x={dialCx + 5}
                  y={dialCy - 2.0 * dialScale - 4}
                  fontSize="9.5"
                  fontWeight="700"
                  fill="#DC2626"
                  paintOrder="stroke"
                  stroke="#FFFFFF"
                  strokeWidth="3.5"
                  strokeLinejoin="round"
                >
                  2.0σ Breach
                </text>
              </svg>
            </div>

            {/* Right: Companion 60-Step Instructional Sparkline & Unified Hover Readout */}
            <div className="lg:col-span-7 space-y-3">
              {activeHilbertHover && (
                <div className="grid grid-cols-2 gap-2.5 rounded-xl border border-slate-200 bg-slate-50/90 p-3 sm:grid-cols-5 text-xs">
                  <div>
                    <div className="text-[10px] font-bold uppercase text-slate-500">Calendar Date</div>
                    <div className="font-mono font-bold text-slate-900">{activeHilbertHover.date}</div>
                  </div>
                  <div>
                    <div className="text-[10px] font-bold uppercase text-slate-500">Inst. Day t_acad</div>
                    <div className="font-bold text-slate-900">{activeHilbertHover.t_acad}</div>
                  </div>
                  <div>
                    <div className="text-[10px] font-bold uppercase text-slate-500">Phase Angle φ_t</div>
                    <div
                      className="font-bold"
                      style={{ color: activeHilbertHover.quadrant_color }}
                    >
                      {activeHilbertHover.phi_deg > 0 ? `+${activeHilbertHover.phi_deg}` : activeHilbertHover.phi_deg}° ({activeHilbertHover.quadrant_id})
                    </div>
                  </div>
                  <div>
                    <div className="text-[10px] font-bold uppercase text-slate-500">Amplitude r_t</div>
                    <div className="font-bold text-slate-900">{activeHilbertHover.r_norm}σ</div>
                  </div>
                  <div className="col-span-2 sm:col-span-1">
                    <div className="text-[10px] font-bold uppercase text-slate-500">Est. Days to Crest</div>
                    <div className="font-extrabold text-indigo-700">
                      {activeHilbertHover.days_to_peak} inst.d
                    </div>
                  </div>
                </div>
              )}

              {/* 60-Day Companion Sparkline SVG */}
              <div className="rounded-xl border border-slate-200 bg-white p-3">
                <div className="mb-1.5 flex items-center justify-between text-[11px] text-slate-500">
                  <span className="font-semibold text-slate-700">
                    Trailing 60-Day Instructional Trajectory (Observed Counts vs. Warped Fourier Baseline)
                  </span>
                  <span>Hover strip to inspect historical phase state</span>
                </div>
                {(() => {
                  const sw = 540;
                  const sh = 175;
                  const px0 = 34;
                  const px1 = 526;
                  const py0 = 14;
                  const py1 = 148;
                  const maxC = Math.max(
                    4,
                    ...sparkPts.map((p) => Math.max(p.observed_count, p.warped_baseline))
                  ) * 1.15;
                  const sx = (i: number) =>
                    px0 + ((px1 - px0) * i) / Math.max(1, sparkPts.length - 1);
                  const sy = (v: number) => py1 - ((py1 - py0) * v) / maxC;
                  const colW = (px1 - px0) / Math.max(1, sparkPts.length);

                  const obsLine = sparkPts
                    .map((p, i) => `${sx(i).toFixed(1)},${sy(p.observed_count).toFixed(1)}`)
                    .join(' ');
                  const baseLine = sparkPts
                    .map((p, i) => `${sx(i).toFixed(1)},${sy(p.warped_baseline).toFixed(1)}`)
                    .join(' ');

                  return (
                    <svg
                      viewBox={`0 0 ${sw} ${sh}`}
                      className="w-full"
                      onMouseLeave={() => setHoveredSparkIdx(null)}
                    >
                      {/* Shaded Phase Quadrant Background Bands */}
                      {sparkPts.map((p, i) => (
                        <rect
                          key={`band-${p.step_index}`}
                          x={sx(i) - colW * 0.5}
                          y={py0}
                          width={colW + 0.5}
                          height={py1 - py0}
                          fill={p.quadrant_color}
                          fillOpacity={p.is_holiday_freeze ? 0.04 : 0.11}
                          onMouseEnter={() => setHoveredSparkIdx(i)}
                          className="cursor-crosshair"
                        />
                      ))}

                      <line x1={px0} y1={py1} x2={px1} y2={py1} stroke="#CBD5E1" strokeWidth="1" />
                      <polyline
                        fill="none"
                        stroke="#475569"
                        strokeWidth="1.8"
                        strokeDasharray="4,3"
                        points={baseLine}
                        pointerEvents="none"
                      />
                      <polyline
                        fill="none"
                        stroke="#0F172A"
                        strokeWidth="2"
                        points={obsLine}
                        pointerEvents="none"
                      />

                      {hoveredSparkIdx !== null && sparkPts[hoveredSparkIdx] && (
                        <line
                          x1={sx(hoveredSparkIdx)}
                          y1={py0}
                          x2={sx(hoveredSparkIdx)}
                          y2={py1}
                          stroke="#0F172A"
                          strokeWidth="1.3"
                          pointerEvents="none"
                        />
                      )}

                      <text x={px0} y={166} fontSize="9.5" fill="#64748B">
                        {sparkPts[0]?.date}
                      </text>
                      <text x={px1} y={166} textAnchor="end" fontSize="9.5" fontWeight="600" fill="#0F172A">
                        {sparkPts[sparkPts.length - 1]?.date} (t)
                      </text>
                    </svg>
                  );
                })()}

                {/* Legend */}
                <div className="mt-1 flex flex-wrap items-center gap-4 text-[11px] text-slate-600">
                  <span className="inline-flex items-center gap-1.5">
                    <span className="h-0.5 w-4 bg-slate-900 inline-block" /> Observed Daily Count
                  </span>
                  <span className="inline-flex items-center gap-1.5">
                    <span className="h-0.5 w-4 border-b-2 border-dashed border-slate-500 inline-block" /> Warped Fourier Baseline
                  </span>
                  <span className="inline-flex items-center gap-1.5">
                    <span className="h-2.5 w-2.5 rotate-45 border border-slate-600 bg-white inline-block" /> Holiday Freeze
                  </span>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* VISUAL SPEC 2: HAWKES CONTAGION CASCADE & BRANCHING MONITOR */}
      <Card className="scroll-mt-24" data-testid="hawkes-cascade-panel">
        <CardHeader className="pb-2">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <GitBranch className="h-5 w-5 text-teal-600" />
              <CardTitle>
                2. Hawkes Contagion Cascade &amp; Subcritical Branching Monitor
              </CardTitle>
            </div>
            <Badge variant="slate">
              Module f03 · ρ(Γ) = {hawkes.current.spectral_radius.toFixed(2)} ({hawkes.current.regime.label})
            </Badge>
          </div>
          <CardDescription>
            Separates routine seasonal demand (muted slate) from self- and cross-exciting category cascades, with a 5-day causal zero-shock relaxation cone (t, t+5].
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-12 items-start">
            {/* Left Panel (~65% -> 8 cols): Stacked Exogenous vs Endogenous Intensity Ribbon */}
            <div className="lg:col-span-8 space-y-2.5">
              {activeHawkesHover && (
                <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs">
                  <div className="flex items-center gap-2">
                    <span className="font-mono font-bold text-slate-900">
                      {activeHawkesHover.date}
                    </span>
                    {activeHawkesHover.is_projection ? (
                      <Badge variant="purple">Causal Decay Projection (+{activeHawkesHover.horizon_step}d)</Badge>
                    ) : (
                      <Badge variant="slate">Observed Step</Badge>
                    )}
                  </div>
                  <div className="flex flex-wrap items-center gap-3 text-slate-700">
                    <span>
                      Baseline μ: <strong>{activeHawkesHover.exog_baseline}</strong>
                    </span>
                    <span>
                      Contagion ΣM: <strong>{activeHawkesHover.endog_total}</strong>
                    </span>
                    <span>
                      Total λ(t): <strong>{activeHawkesHover.lambda_total}</strong>
                    </span>
                    {!activeHawkesHover.is_projection && (
                      <span>
                        Observed: <strong>{activeHawkesHover.observed_count}</strong> (
                        {(activeHawkesHover.endog_share * 100).toFixed(1)}% Endog)
                      </span>
                    )}
                  </div>
                </div>
              )}

              <div className="rounded-xl border border-slate-200 bg-white p-3">
                <svg
                  viewBox="0 0 645 280"
                  className="w-full select-none"
                  onMouseLeave={() => setHoveredHawkesIdx(null)}
                >
                  <defs>
                    <pattern
                      id="cone-hatch"
                      width="7"
                      height="7"
                      patternTransform="rotate(45 0 0)"
                      patternUnits="userSpaceOnUse"
                    >
                      <line
                        x1="0"
                        y1="0"
                        x2="0"
                        y2="7"
                        stroke="#475569"
                        strokeWidth="1.2"
                        strokeOpacity="0.32"
                      />
                    </pattern>
                  </defs>

                  {/* Routine Seasonal / Calendar Baseline Layer */}
                  <polygon points={exogPolyPts} fill="#475569" fillOpacity="0.38" />

                  {/* Stacked Category Endogenous Excitation Ribbons */}
                  {catPolygons.map((cp) => (
                    <polygon
                      key={cp.label}
                      points={cp.points}
                      fill={cp.color}
                      fillOpacity="0.62"
                    />
                  ))}

                  {/* 5-Step Causal Decay Projection Cone (t, t+5] */}
                  {hawkesCone.length > 0 && hawkesHist.length > 0 && (
                    <g>
                      {(() => {
                        const splitX = hXp(hawkesHist.length - 1);
                        const coneBandPts = [
                          `${splitX.toFixed(1)},${hYp(hawkesHist[hawkesHist.length - 1].lambda_total).toFixed(1)}`,
                          ...hawkesCone.map(
                            (c, k) =>
                              `${hXp(hawkesHist.length + k).toFixed(1)},${hYp(c.upper_cone).toFixed(1)}`
                          ),
                          ...hawkesCone
                            .map(
                              (c, k) =>
                                `${hXp(hawkesHist.length + k).toFixed(1)},${hYp(c.lower_cone).toFixed(1)}`
                            )
                            .reverse(),
                          `${splitX.toFixed(1)},${hYp(hawkesHist[hawkesHist.length - 1].lambda_total).toFixed(1)}`,
                        ].join(' ');
                        return (
                          <>
                            <polygon points={coneBandPts} fill="url(#cone-hatch)" stroke="#64748B" strokeWidth="0.8" strokeDasharray="3,2" />
                            <line
                              x1={splitX}
                              y1={hy0}
                              x2={splitX}
                              y2={hy1}
                              stroke="#0F172A"
                              strokeWidth="1.4"
                              strokeDasharray="4,3"
                            />
                            <text
                              x={hx1}
                              y={15}
                              textAnchor="end"
                              fontSize="9.5"
                              fontWeight="700"
                              fill="#334155"
                              paintOrder="stroke"
                              stroke="#FFFFFF"
                              strokeWidth="3"
                              strokeLinejoin="round"
                            >
                              (t, t+5] Decay Cone
                            </text>
                          </>
                        );
                      })()}
                    </g>
                  )}

                  {/* Observed Count Stems & >35% Endogenous Contagion Shock Pins */}
                  {hawkesHist.map((r, i) => {
                    const xi = hXp(i);
                    const yi = hYp(r.observed_count);
                    return (
                      <g key={`stem-${r.step_index}`}>
                        <line
                          x1={xi}
                          y1={hy1}
                          x2={xi}
                          y2={yi}
                          stroke="#1E293B"
                          strokeWidth="1"
                          strokeOpacity="0.38"
                        />
                        <circle cx={xi} cy={yi} r="2.3" fill="#0F172A" />
                        {r.is_shock_callout && (
                          <g>
                            <line
                              x1={xi}
                              y1={hy0 + 6}
                              x2={xi}
                              y2={Math.max(hy0 + 10, yi - 4)}
                              stroke="#D97706"
                              strokeWidth="1.2"
                              strokeDasharray="2,2"
                            />
                            <circle cx={xi} cy={hy0 + 6} r="3.2" fill="#D97706" />
                          </g>
                        )}
                      </g>
                    );
                  })}

                  {/* Invisible Hover Columns */}
                  {hawkesAll.map((s, i) => {
                    const colW = (hx1 - hx0) / Math.max(1, hawkesAll.length);
                    return (
                      <rect
                        key={`hover-h-${i}`}
                        x={hXp(i) - colW * 0.5}
                        y={hy0}
                        width={colW}
                        height={hy1 - hy0}
                        fill="transparent"
                        onMouseEnter={() => setHoveredHawkesIdx(i)}
                        className="cursor-crosshair"
                      />
                    );
                  })}

                  <line x1={hx0} y1={hy1} x2={hx1} y2={hy1} stroke="#94A3B8" strokeWidth="1.1" />
                  <text x={hx0} y={268} fontSize="9.5" fill="#64748B">
                    {hawkesHist[0]?.date}
                  </text>
                  <text
                    x={hXp(Math.max(0, hawkesHist.length - 1))}
                    y={268}
                    textAnchor="middle"
                    fontSize="9.5"
                    fontWeight="700"
                    fill="#0F172A"
                  >
                    t = {hawkesHist[hawkesHist.length - 1]?.date}
                  </text>
                </svg>

                {/* Ribbon Legend */}
                <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[11px] text-slate-700">
                  <span className="inline-flex items-center gap-1.5">
                    <span className="h-2.5 w-2.5 rounded-xs bg-[#475569]/60 inline-block" /> Routine Seasonal / Calendar Baseline
                  </span>
                  {hawkesCats.map((c) => (
                    <span key={c.id} className="inline-flex items-center gap-1.5">
                      <span
                        className="h-2.5 w-2.5 rounded-xs inline-block"
                        style={{ backgroundColor: c.color }}
                      />
                      Contagion Load from {c.label}
                    </span>
                  ))}
                  <span className="inline-flex items-center gap-1.5">
                    <span className="h-2.5 w-2.5 rounded-full bg-amber-600 inline-block" /> &gt;35% Endogenous Shock Pin
                  </span>
                </div>
              </div>
            </div>

            {/* Right Panel (~35% -> 4 cols): Cross-Category Directed Contagion Network & Criticality Gauge */}
            <div className="lg:col-span-4 space-y-3">
              <div className="rounded-xl border border-slate-200 bg-white p-3">
                <div className="text-xs font-bold text-slate-800 mb-1">
                  Cross-Category Directed Contagion Graph (Γ_cj ≥ 0.03)
                </div>
                {hawkes.is_univariate_fallback ? (
                  <div className="py-6 text-center text-xs text-slate-600">
                    Univariate Aggregate Self-Excitation: Γ ={' '}
                    <strong>{hawkes.network.nodes[0]?.self_branching}</strong> (τ_1/2 ={' '}
                    {hawkes.network.nodes[0]?.self_half_life_days}d)
                  </div>
                ) : (
                  <>
                    <svg viewBox="0 0 320 225" className="w-full select-none">
                      <defs>
                        <marker
                          id="hawkes-edge-arrow"
                          markerWidth="7"
                          markerHeight="7"
                          refX="6"
                          refY="3.5"
                          orient="auto"
                        >
                          <path d="M0,1 L6,3.5 L0,6 Z" fill="#64748B" />
                        </marker>
                      </defs>
                      {(() => {
                        const ncx = 160;
                        const ncy = 114;
                        const nr = 72;
                        const nodes = hawkes.network.nodes || [];
                        const crossEdges = (hawkes.network.edges || []).filter(
                          (e: any) => !e.is_self_loop
                        );
                        return (
                          <>
                            {crossEdges.slice(0, 8).map((edge: any, idx: number) => {
                              const u = nodes[edge.source_idx];
                              const v = nodes[edge.target_idx];
                              if (!u || !v) return null;
                              const x1 = ncx + u.x_unit * nr;
                              const y1 = ncy + u.y_unit * nr;
                              const x2 = ncx + v.x_unit * nr;
                              const y2 = ncy + v.y_unit * nr;
                              const mx = 0.5 * (x1 + x2) - 0.18 * (y2 - y1);
                              const my = 0.5 * (y1 + y2) + 0.18 * (x2 - x1);
                              return (
                                <g key={`he-${idx}`}>
                                  <path
                                    d={`M ${x1.toFixed(1)} ${y1.toFixed(1)} Q ${mx.toFixed(1)} ${my.toFixed(1)} ${x2.toFixed(1)} ${y2.toFixed(1)}`}
                                    fill="none"
                                    stroke={edge.color}
                                    strokeWidth={edge.stroke_width}
                                    strokeOpacity="0.65"
                                    markerEnd="url(#hawkes-edge-arrow)"
                                  >
                                    <title>
                                      {edge.source_label} → {edge.target_label}: Γ={edge.branching_ratio}, τ_1/2={edge.half_life_days}d
                                    </title>
                                  </path>
                                  {idx === 0 && (
                                    <text
                                      x={mx}
                                      y={my}
                                      textAnchor="middle"
                                      fontSize="8.5"
                                      fontWeight="700"
                                      fill="#1E293B"
                                      paintOrder="stroke"
                                      stroke="#FFFFFF"
                                      strokeWidth="3.2"
                                      strokeLinejoin="round"
                                    >
                                      τ½={edge.half_life_days}d
                                    </text>
                                  )}
                                </g>
                              );
                            })}

                            {nodes.map((nd: any) => {
                              const nx = ncx + nd.x_unit * nr;
                              const ny = ncy + nd.y_unit * nr;
                              const labelY = nd.y_unit < -0.2 ? ny - 16 : ny + 22;
                              const shortLbl = nd.label.split('&')[0].trim();
                              return (
                                <g key={nd.id}>
                                  <circle
                                    cx={nx}
                                    cy={ny}
                                    r="11.5"
                                    fill={nd.color}
                                    stroke="#FFFFFF"
                                    strokeWidth="2"
                                  />
                                  <text
                                    x={nx}
                                    y={labelY}
                                    textAnchor="middle"
                                    fontSize="9.5"
                                    fontWeight="700"
                                    fill="#1E293B"
                                  >
                                    {shortLbl}
                                  </text>
                                </g>
                              );
                            })}
                          </>
                        );
                      })()}
                    </svg>

                    {/* Top 3 Branching Pathways List */}
                    <div className="mt-1 space-y-1 border-t border-slate-100 pt-2">
                      {(hawkes.network.edges || []).slice(0, 3).map((e: any, i: number) => (
                        <div
                          key={`pathway-${i}`}
                          className="flex items-center justify-between text-[11px] text-slate-600"
                        >
                          <span className="truncate max-w-[195px]">
                            <strong>{e.source_label.split(' ')[0]}</strong> →{' '}
                            {e.target_label.split(' ')[0]}
                          </span>
                          <span className="font-mono font-semibold text-slate-800">
                            Γ={e.branching_ratio.toFixed(2)} · τ½={e.half_life_days}d
                          </span>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </div>

              {/* Spectral Criticality Bullet Gauge */}
              <div className="rounded-xl border border-slate-200 bg-white p-3.5">
                <div className="flex items-center justify-between text-xs font-bold text-slate-800">
                  <span>Spectral Criticality ρ(Γ)</span>
                  <span style={{ color: hawkes.current.regime.color }}>
                    {hawkes.current.spectral_radius.toFixed(2)} / 1.00
                  </span>
                </div>
                <div className="relative mt-2.5 h-3.5 w-full overflow-hidden rounded-full flex">
                  <div
                    className="h-full bg-[#0D9488]/30"
                    style={{ width: `${(0.6 / 1.2) * 100}%` }}
                    title="< 0.60 Damped / Isolated"
                  />
                  <div
                    className="h-full bg-[#D97706]/35"
                    style={{ width: `${(0.25 / 1.2) * 100}%` }}
                    title="0.60 - 0.85 Amplifying Contagion"
                  />
                  <div
                    className="h-full bg-[#DC2626]/35"
                    style={{ width: `${(0.35 / 1.2) * 100}%` }}
                    title="0.85 - 1.20 Near-Critical Cascade Alert"
                  />
                  <div
                    className="absolute top-0 bottom-0 w-1 bg-slate-950 shadow-xs"
                    style={{
                      left: `${Math.min(98, Math.max(1, (hawkes.current.spectral_radius / 1.2) * 100))}%`,
                    }}
                  />
                </div>
                <div className="mt-1 flex justify-between text-[10px] font-medium text-slate-500">
                  <span>0.0 Damped</span>
                  <span>0.60 Amplifying</span>
                  <span>0.85 Critical</span>
                  <span>1.2</span>
                </div>
                <p className="mt-2 text-[11px] text-slate-600">{hawkes.current.regime.summary}</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* VISUAL SPEC 3: SHEAF FRUSTRATION & CONTEXTUAL MASKING ALERT BOARD */}
      <Card className="scroll-mt-24" data-testid="sheaf-frustration-panel">
        <CardHeader className="pb-2">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <Layers className="h-5 w-5 text-amber-600" />
              <CardTitle>
                3. Sheaf Frustration &amp; Contextual Masking Pastoral Triage Board
              </CardTitle>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="teal">{sheaf.topology_mode}</Badge>
              <Badge variant="slate">Module f08 · L_F = δ^T δ</Badge>
            </div>
          </div>
          <CardDescription>
            Detects cohomological disagreement across connected cohorts when high surface attendance masks sharp drops in workshop engagement or rising low-level pastoral friction ahead of referral crests.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          {/* Top Strip: Rolling 60-Day Global Sheaf Dirichlet Energy Timeline */}
          <div className="rounded-xl border border-slate-200 bg-slate-50/70 p-3">
            <div className="mb-1 flex flex-wrap items-center justify-between gap-2 text-xs">
              <span className="font-bold text-slate-800">
                Global Sheaf Dirichlet Energy E_F(x_τ) = x_τ^T L_F x_τ (Trailing 60 Steps)
              </span>
              <div className="flex items-center gap-4 text-[11px]">
                <span className="text-amber-700 font-semibold">
                  — — 90th Pct Warning ({sheaf.current.tau90.toFixed(2)})
                </span>
                <span className="text-red-700 font-semibold">
                  — — 97.5th Pct Crimson Alert ({sheaf.current.tau975.toFixed(2)})
                </span>
              </div>
            </div>
            <svg viewBox="0 0 940 114" className="w-full select-none">
              <line
                x1={sx0}
                y1={sYp(sheaf.current.tau90)}
                x2={sx1}
                y2={sYp(sheaf.current.tau90)}
                stroke="#D97706"
                strokeWidth="1.2"
                strokeDasharray="5,4"
              />
              <line
                x1={sx0}
                y1={sYp(sheaf.current.tau975)}
                x2={sx1}
                y2={sYp(sheaf.current.tau975)}
                stroke="#DC2626"
                strokeWidth="1.3"
                strokeDasharray="5,3"
              />
              <polyline
                fill="none"
                stroke="#4F46E5"
                strokeWidth="2.2"
                points={sheafStrip
                  .map((p, i) => `${sXp(i).toFixed(1)},${sYp(p.dirichlet_energy).toFixed(1)}`)
                  .join(' ')}
              />
              {sheafStrip.map((p, i) => {
                if (p.status === 'calm' && i < sheafStrip.length - 1) return null;
                const col =
                  p.status === 'breach'
                    ? '#DC2626'
                    : p.status === 'warning'
                    ? '#D97706'
                    : '#4F46E5';
                return (
                  <circle
                    key={`se-${p.step_index}`}
                    cx={sXp(i)}
                    cy={sYp(p.dirichlet_energy)}
                    r={i === sheafStrip.length - 1 ? 4.2 : 2.6}
                    fill={col}
                  >
                    <title>
                      {p.date}: E_F = {p.dirichlet_energy} ({p.energy_z}z)
                    </title>
                  </circle>
                );
              })}
              <line x1={sx0} y1={sy1} x2={sx1} y2={sy1} stroke="#CBD5E1" strokeWidth="1" />
              <text x={sx0} y={110} fontSize="9.5" fill="#64748B">
                {sheafStrip[0]?.date}
              </text>
              <text x={sx1} y={110} textAnchor="end" fontSize="9.5" fontWeight="700" fill="#0F172A">
                t = {sheafStrip[sheafStrip.length - 1]?.date} (E_F = {sheaf.current.dirichlet_energy.toFixed(2)})
              </text>
            </svg>
          </div>

          {/* Bottom Split: Left (~55%) Fixed Cohort Topology + Right (~45%) Ranked Contextual Masking Cards */}
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-12 items-start">
            {/* Left (7 cols): Cohort Sheaf Frustration Topology */}
            <div className="lg:col-span-7 rounded-xl border border-slate-200 bg-white p-4">
              <div className="mb-2 flex items-center justify-between">
                <span className="text-xs font-bold text-slate-800">
                  Fixed-Coordinate {sheaf.topology_mode}
                </span>
                <span className="text-[11px] text-slate-500">
                  Frustrated edges (f_e ≥ τ_90) highlighted with z-scores
                </span>
              </div>

              <svg viewBox="0 0 480 340" className="w-full max-w-[520px] mx-auto select-none">
                {(() => {
                  const gcx = 240;
                  const gcy = 168;
                  const gr = 108;
                  const gNodes: any[] = sheaf.graph.nodes || [];
                  const gEdges: any[] = sheaf.graph.edges || [];
                  return (
                    <>
                      {/* Render quiet edges first, frustrated edges on top */}
                      {[...gEdges]
                        .sort((a, b) => Number(a.is_frustrated) - Number(b.is_frustrated))
                        .map((edge) => {
                          const u = gNodes[edge.u];
                          const v = gNodes[edge.v];
                          if (!u || !v) return null;
                          const x1 = gcx + u.x_unit * gr;
                          const y1 = gcy + u.y_unit * gr;
                          const x2 = gcx + v.x_unit * gr;
                          const y2 = gcy + v.y_unit * gr;
                          const mx = 0.5 * (x1 + x2);
                          const my = 0.5 * (y1 + y2);
                          const px = 0.68 * mx + 0.32 * gcx;
                          const py = 0.68 * my + 0.32 * gcy;
                          return (
                            <g key={`se-edge-${edge.edge_index}`}>
                              <line
                                x1={x1}
                                y1={y1}
                                x2={x2}
                                y2={y2}
                                stroke={edge.color}
                                strokeWidth={edge.stroke_width}
                                strokeOpacity={edge.is_frustrated ? 0.9 : 0.25}
                              >
                                <title>
                                  {edge.u_label} ↔ {edge.v_label}: f_e={edge.frustration_norm} ({edge.frustration_z}z)
                                </title>
                              </line>
                              {edge.is_frustrated && (
                                <g>
                                  <rect
                                    x={px - 19}
                                    y={py - 8.5}
                                    width="38"
                                    height="17"
                                    rx="8.5"
                                    fill={edge.color}
                                  />
                                  <text
                                    x={px}
                                    y={py + 3}
                                    textAnchor="middle"
                                    fontSize="9"
                                    fontWeight="700"
                                    fill="#FFFFFF"
                                  >
                                    {edge.frustration_z > 0
                                      ? `+${edge.frustration_z.toFixed(1)}z`
                                      : `${edge.frustration_z.toFixed(1)}z`}
                                  </text>
                                </g>
                              )}
                            </g>
                          );
                        })}

                      {/* Cohort Vertices */}
                      {gNodes.map((nd) => {
                        const nx = gcx + nd.x_unit * gr;
                        const ny = gcy + nd.y_unit * gr;
                        const isHov = hoveredSheafNode === nd.index;
                        const labelOffset =
                          nd.y_unit < -0.65 ? -(nd.radius_px + 10) : nd.radius_px + 15;
                        return (
                          <g
                            key={nd.id}
                            onMouseEnter={() => setHoveredSheafNode(nd.index)}
                            onMouseLeave={() => setHoveredSheafNode(null)}
                            className="cursor-pointer"
                          >
                            {nd.is_anomalous && (
                              <circle
                                cx={nx}
                                cy={ny}
                                r={nd.radius_px + 5}
                                fill={nd.fill_color}
                                fillOpacity="0.18"
                              />
                            )}
                            <circle
                              cx={nx}
                              cy={ny}
                              r={nd.radius_px}
                              fill={nd.fill_color}
                              stroke={isHov ? '#0F172A' : '#FFFFFF'}
                              strokeWidth={isHov ? 3 : 2}
                            />
                            <text
                              x={nx}
                              y={ny + 3.5}
                              textAnchor="middle"
                              fontSize="9.5"
                              fontWeight="700"
                              fill="#FFFFFF"
                            >
                              {nd.frustration_z.toFixed(1)}σ
                            </text>
                            <text
                              x={nx}
                              y={ny + labelOffset}
                              textAnchor="middle"
                              fontSize="10.5"
                              fontWeight="700"
                              fill="#0F172A"
                            >
                              {nd.label}
                            </text>
                          </g>
                        );
                      })}
                    </>
                  );
                })()}
              </svg>

              <div className="mt-2 flex flex-wrap items-center justify-center gap-4 text-[11px] text-slate-600">
                <span className="inline-flex items-center gap-1.5">
                  <span className="h-2.5 w-2.5 rounded-full bg-[#475569] inline-block" /> Globally Consistent
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <span className="h-2.5 w-2.5 rounded-full bg-[#D97706] inline-block" /> Elevated Divergence
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <span className="h-2.5 w-2.5 rounded-full bg-[#DC2626] inline-block" /> &gt;2σ Contextual Masking Breach
                </span>
              </div>
            </div>

            {/* Right (5 cols): Ranked Top-3 Contextual Masking Pastoral Triage Cards */}
            <div className="lg:col-span-5 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-600">
                  Ranked Contextual Masking Triage Cards (Top 3)
                </span>
              </div>

              {(sheaf.triage_cards || []).map((card: any) => (
                <div
                  key={card.cohort_id}
                  className="rounded-xl border border-slate-200 bg-white p-3.5 shadow-2xs border-l-4"
                  style={{ borderLeftColor: card.accent_color }}
                >
                  <div className="flex flex-wrap items-center justify-between gap-1.5">
                    <span className="font-display text-sm font-extrabold text-slate-900">
                      #{card.rank} · {card.cohort_label}
                    </span>
                    <span
                      className="rounded-full px-2 py-0.5 text-[10px] font-bold text-white"
                      style={{ backgroundColor: card.accent_color }}
                    >
                      {card.header_badge}
                    </span>
                  </div>

                  <p className="mt-1.5 text-xs text-slate-700 leading-relaxed">
                    {card.explanation}
                  </p>

                  {/* Micro Diverging Bar Chart: Observed vs. Sheaf-Expected */}
                  <div className="mt-2.5 space-y-1.5 border-t border-slate-100 pt-2">
                    <div className="flex items-center justify-between text-[10px] font-semibold text-slate-500">
                      <span>Stalk Indicator (SD)</span>
                      <span>Observed (Solid) vs. Sheaf-Expected (Slate)</span>
                    </div>
                    {(card.diverging_bars || []).map((bar: any) => {
                      const obsPct = Math.min(48, (Math.abs(bar.observed_sd) / 3.5) * 48);
                      const expPct = Math.min(48, (Math.abs(bar.sheaf_expected_sd) / 3.5) * 48);
                      const obsLeft = bar.observed_sd >= 0 ? 50 : 50 - obsPct;
                      const expLeft = bar.sheaf_expected_sd >= 0 ? 50 : 50 - expPct;

                      return (
                        <div key={bar.attribute_id} className="space-y-0.5">
                          <div className="flex justify-between text-[10.5px] text-slate-700">
                            <span className="font-medium">{bar.attribute_label}</span>
                            <span className="font-mono text-[10px]">
                              Obs {bar.observed_sd > 0 ? `+${bar.observed_sd}` : bar.observed_sd} vs Exp{' '}
                              {bar.sheaf_expected_sd > 0
                                ? `+${bar.sheaf_expected_sd}`
                                : bar.sheaf_expected_sd}
                            </span>
                          </div>
                          <div className="relative h-3 w-full rounded-xs bg-slate-100 overflow-hidden">
                            <div className="absolute left-1/2 top-0 bottom-0 w-px bg-slate-400 z-10" />
                            {/* Sheaf-Expected bar */}
                            <div
                              className="absolute top-0.5 h-1 rounded-xs bg-slate-400"
                              style={{ left: `${expLeft}%`, width: `${Math.max(1.5, expPct)}%` }}
                            />
                            {/* Observed bar */}
                            <div
                              className="absolute bottom-0.5 h-1 rounded-xs"
                              style={{
                                left: `${obsLeft}%`,
                                width: `${Math.max(1.5, obsPct)}%`,
                                backgroundColor: card.accent_color,
                              }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* COLLAPSED BACKFILL AUDIT ACCORDION (Bottom Governance & Causality Ledger) */}
      {backfillAudit && (
        <Card
          className="border-slate-200/90 bg-white"
          data-testid="backfill-audit-accordion"
        >
          <button
            type="button"
            data-testid="backfill-audit-toggle"
            aria-expanded={isAuditOpen}
            onClick={() => setIsAuditOpen((prev) => !prev)}
            className="flex w-full items-center justify-between gap-3 px-5 py-3.5 text-left hover:bg-slate-50/80 transition-colors cursor-pointer rounded-2xl"
          >
            <div className="flex flex-wrap items-center gap-2.5">
              <Lock className="h-4 w-4 text-slate-700" />
              <span className="font-display text-sm font-extrabold text-slate-900">
                Retrospective Backfill, Quarantine Gate &amp; Vintage Causality Audit
              </span>
              <Badge variant="teal">
                {backfillAudit.rollout_badge?.retro_mode || retroMode}
              </Badge>
              <Badge variant="slate">
                SHA-256 Baseline Verified
              </Badge>
              <span className="text-xs text-slate-500">
                Last Sept Reset:{' '}
                <strong>{backfillAudit.vintage_and_topology?.last_september_reset_date}</strong> (
                {backfillAudit.vintage_and_topology?.active_academic_year})
              </span>
            </div>
            <div className="flex items-center gap-1 text-xs font-semibold text-slate-600">
              <span>{isAuditOpen ? 'Collapse Audit' : 'Expand Audit'}</span>
              {isAuditOpen ? (
                <ChevronUp className="h-4 w-4" />
              ) : (
                <ChevronDown className="h-4 w-4" />
              )}
            </div>
          </button>

          {isAuditOpen && (
            <CardContent
              className="border-t border-slate-200 pt-4 pb-5 space-y-4"
              data-testid="backfill-audit-content"
            >
              <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
                {/* 1. Rollout Mode & Cryptographic Frozen Baseline Proof */}
                <div className="rounded-xl border border-slate-200 bg-slate-50/70 p-3.5 space-y-2 text-xs">
                  <div className="font-bold uppercase tracking-wider text-[11px] text-slate-500">
                    1. Rollout Mode &amp; Immutable Baseline Hash
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-slate-900">
                      {backfillAudit.rollout_badge?.retro_mode}
                    </span>
                    <Badge variant="teal">Zero Core Mutation</Badge>
                  </div>
                  <p className="text-slate-600 leading-relaxed">
                    {backfillAudit.rollout_badge?.description}
                  </p>
                  <div className="space-y-1 border-t border-slate-200/80 pt-2 font-mono text-[10.5px] text-slate-700">
                    <div>
                      Model SHA-256:{' '}
                      <span title={backfillAudit.rollout_badge?.baseline_model_sha256}>
                        {String(backfillAudit.rollout_badge?.baseline_model_sha256 || '').slice(0, 18)}…
                      </span>
                    </div>
                    <div>
                      OOS Pred SHA-256:{' '}
                      <span title={backfillAudit.rollout_badge?.baseline_prediction_sha256}>
                        {String(backfillAudit.rollout_badge?.baseline_prediction_sha256 || '').slice(0, 18)}…
                      </span>
                    </div>
                  </div>
                </div>

                {/* 2. Admitted vs. Quarantined Residual Modules per Academic Year */}
                <div className="rounded-xl border border-slate-200 bg-slate-50/70 p-3.5 space-y-2 text-xs">
                  <div className="font-bold uppercase tracking-wider text-[11px] text-slate-500">
                    2. Admitted vs. Quarantined Modules by Academic Year
                  </div>
                  <div className="space-y-2">
                    {(backfillAudit.admitted_vs_quarantined_by_year || []).map((yr: any) => (
                      <div
                        key={yr.academic_year}
                        className="rounded-lg border border-slate-200 bg-white p-2.5 space-y-1"
                      >
                        <div className="flex items-center justify-between font-bold text-slate-900">
                          <span>AY {yr.academic_year}</span>
                          <span className="text-[10.5px] font-normal text-slate-500">
                            VIF ≤ {yr.max_vif} · Quarantined: {yr.quarantined_count}
                          </span>
                        </div>
                        {yr.admitted_modules?.length > 0 ? (
                          <div className="flex flex-wrap gap-1">
                            {yr.admitted_labels.map((lbl: string) => (
                              <span
                                key={lbl}
                                className="rounded-md bg-teal-50 px-1.5 py-0.5 text-[10.5px] font-semibold text-teal-800 border border-teal-200"
                              >
                                Admitted: {lbl}
                              </span>
                            ))}
                          </div>
                        ) : (
                          <div className="text-[11px] font-semibold text-slate-600">
                            Identity Floor Active (0 modules admitted — Frozen Baseline verbatim)
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                {/* 3. Vintage Cutoff, September Topology Reset & Reporting-Lag Sensitivity */}
                <div className="rounded-xl border border-slate-200 bg-slate-50/70 p-3.5 space-y-2 text-xs">
                  <div className="font-bold uppercase tracking-wider text-[11px] text-slate-500">
                    3. Vintage Cutoff &amp; September Topology Reset
                  </div>
                  <div className="grid grid-cols-2 gap-1.5 text-[11px]">
                    <div>
                      <span className="text-slate-500">Vintage Cutoff:</span>{' '}
                      <strong className="font-mono text-slate-900">
                        {backfillAudit.vintage_and_topology?.vintage_cutoff_date}
                      </strong>
                    </div>
                    <div>
                      <span className="text-slate-500">Last Sept Reset:</span>{' '}
                      <strong className="font-mono text-slate-900">
                        {backfillAudit.vintage_and_topology?.last_september_reset_date}
                      </strong>
                    </div>
                    <div>
                      <span className="text-slate-500">Acad Year Week:</span>{' '}
                      <strong>Wk {backfillAudit.vintage_and_topology?.weeks_into_academic_year}</strong>
                    </div>
                    <div>
                      <span className="text-slate-500">Prior Warmup w:</span>{' '}
                      <strong>
                        {backfillAudit.vintage_and_topology?.warmup_prior_active
                          ? `${backfillAudit.vintage_and_topology?.warmup_prior_weight} (Active)`
                          : '0.0 (Pure AY)'}
                      </strong>
                    </div>
                  </div>

                  {/* Configurable Reporting-Lag Sensitivity Buffer (0d / 2d / 5d) */}
                  <div className="border-t border-slate-200/80 pt-2 space-y-1.5">
                    <div className="flex items-center justify-between">
                      <span className="text-[11px] font-semibold text-slate-700">
                        Reporting-Lag Buffer Sensitivity:
                      </span>
                      <div className="flex items-center gap-1">
                        {[0, 2, 5].map((bDay) => (
                          <button
                            key={bDay}
                            type="button"
                            onClick={() => setLagBufferDays(bDay)}
                            className={`rounded-md px-2 py-0.5 text-[10.5px] font-bold cursor-pointer ${
                              lagBufferDays === bDay
                                ? 'bg-slate-900 text-white'
                                : 'bg-white border border-slate-300 text-slate-700 hover:bg-slate-100'
                            }`}
                          >
                            {bDay}d
                          </button>
                        ))}
                      </div>
                    </div>
                    <div className="space-y-0.5 text-[10.5px] text-slate-600 font-mono">
                      {(backfillAudit.reporting_lag_sensitivity || []).map((row: any) => (
                        <div key={row.lag_buffer_days} className="flex justify-between">
                          <span>Lag B={row.lag_buffer_days}d ({row.hilbert_quadrant}):</span>
                          <span>
                            Crest {row.days_to_crest}d · λ={row.hawkes_lambda_total} · E_F={row.sheaf_dirichlet_energy}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          )}
        </Card>
      )}
    </div>
  );
};

export default EsotericTelemetryPage;
