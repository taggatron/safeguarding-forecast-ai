import React from 'react';
import {
  ComposedChart,
  Line,
  BarChart,
  Bar,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import {
  TrendingUp,
  BarChart3,
  AlertCircle,
  BookOpen,
  Calendar,
  Layers,
  ShieldAlert,
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
  TrendsInsightsResponse,
} from '../types/api';

interface TrendsInsightsPageProps {
  datasetSummary: DatasetSummaryResponse;
  trendsData: TrendsInsightsResponse;
  selectedProvision: string;
  onSelectProvision: (p: string) => void;
  selectedCategory: string;
  onSelectCategory: (c: string) => void;
  isLoading: boolean;
}

const PALETTE = ['#0f172a', '#0d9488', '#2563eb', '#7c3aed', '#f59e0b', '#64748b'];

export const TrendsInsightsPage: React.FC<TrendsInsightsPageProps> = ({
  datasetSummary,
  trendsData,
  selectedProvision,
  onSelectProvision,
  selectedCategory,
  onSelectCategory,
  isLoading,
}) => {
  const topFeatures = trendsData.feature_importances.slice(0, 10);

  return (
    <div className="space-y-6">
      {/* Header & Filter Bar */}
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between bg-white p-5 rounded-2xl border border-slate-200/80 shadow-xs">
        <div>
          <div className="flex items-center gap-2.5">
            <h1 className="fluid-title font-display font-extrabold text-slate-900">
              Historical Trends, Seasonality & Demand Drivers
            </h1>
            {isLoading && (
              <Badge variant="teal" className="animate-pulse">
                Refreshing...
              </Badge>
            )}
          </div>
          <p className="mt-1 text-sm text-slate-600">
            Multi-year structural patterns, rolling averages, cohort-normalised rates per 1,000 students, and LightGBM statistical associations.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2.5">
          <div>
            <label htmlFor="trends-prov" className="block text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-1">
              Provision Filter
            </label>
            <select
              id="trends-prov"
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
            <label htmlFor="trends-cat" className="block text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-1">
              Category Filter
            </label>
            <select
              id="trends-cat"
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
        </div>
      </div>

      {/* Key Empirical Findings Banner */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        {trendsData.key_findings.map((finding, idx) => (
          <Card key={idx} className="bg-white border-l-4 border-l-teal-600">
            <CardContent className="p-4 text-xs sm:text-sm text-slate-700 leading-relaxed">
              {finding}
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Row 1: Monthly Series with 3m & 6m Rolling Averages + Concerns per 1,000 Students */}
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2 dashboard-section-contain">
        <Card>
          <CardHeader>
            <CardTitle>Monthly Safeguarding Volume & Rolling Averages (3M / 6M)</CardTitle>
            <CardDescription>
              Raw monthly recorded concerns alongside causal 3-month and 6-month moving averages smoothing short-term fluctuations.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[310px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={trendsData.monthly_series} margin={{ top: 10, right: 16, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                  <XAxis dataKey="period_label" tick={{ fontSize: 11, fill: '#475569' }} minTickGap={24} />
                  <YAxis tick={{ fontSize: 11, fill: '#475569' }} axisLine={false} tickLine={false} />
                  <Tooltip />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Bar
                    dataKey="concern_count"
                    name="Monthly Recorded Concerns"
                    fill="#cbd5e1"
                    radius={[3, 3, 0, 0]}
                  />
                  <Line
                    type="monotone"
                    dataKey="rolling_3m_avg"
                    name="3-Month Rolling Average"
                    stroke="#0d9488"
                    strokeWidth={2.5}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="rolling_6m_avg"
                    name="6-Month Rolling Average"
                    stroke="#0f172a"
                    strokeWidth={2.5}
                    dot={false}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Recorded Concerns per 1,000 Enrolled Students</CardTitle>
            <CardDescription>
              Cohort-normalised monthly safeguarding rate per 1,000 enrolled students alongside aggregate attendance rate (%).
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[310px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={trendsData.monthly_series} margin={{ top: 10, right: 16, left: 0, bottom: 5 }}>
                  <defs>
                    <linearGradient id="rateGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#2563eb" stopOpacity={0.22} />
                      <stop offset="95%" stopColor="#2563eb" stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                  <XAxis dataKey="period_label" tick={{ fontSize: 11, fill: '#475569' }} minTickGap={24} />
                  <YAxis yAxisId="left" tick={{ fontSize: 11, fill: '#2563eb' }} axisLine={false} tickLine={false} />
                  <YAxis
                    yAxisId="right"
                    orientation="right"
                    domain={[75, 100]}
                    tick={{ fontSize: 11, fill: '#7c3aed' }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <Tooltip />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Area
                    yAxisId="left"
                    type="monotone"
                    dataKey="concerns_per_1000"
                    name="Concerns per 1,000 Students"
                    stroke="#2563eb"
                    strokeWidth={2.2}
                    fill="url(#rateGrad)"
                  />
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="attendance_rate"
                    name="Attendance Rate (%)"
                    stroke="#7c3aed"
                    strokeWidth={2}
                    strokeDasharray="4 4"
                    dot={false}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Row 2: Seasonal Profile & Multi-Year Provision / Category Trends */}
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-3 dashboard-section-contain">
        {/* Seasonal Profile across Academic Year */}
        <Card>
          <CardHeader>
            <CardTitle>Academic-Year Seasonal Profile (Sep–Aug)</CardTitle>
            <CardDescription>
              Average monthly concerns and daily rate per scheduled teaching day across the FE academic calendar.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[280px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={trendsData.seasonal_profile} margin={{ top: 10, right: 12, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                  <XAxis dataKey="month_name" tick={{ fontSize: 11, fill: '#475569' }} />
                  <YAxis yAxisId="left" tick={{ fontSize: 11, fill: '#475569' }} axisLine={false} tickLine={false} />
                  <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11, fill: '#0d9488' }} axisLine={false} tickLine={false} />
                  <Tooltip />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Bar
                    yAxisId="left"
                    dataKey="avg_concerns"
                    name="Avg Monthly Concerns"
                    fill="#0f172a"
                    radius={[5, 5, 0, 0]}
                  />
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="avg_concerns_per_teaching_day"
                    name="Concerns / Teaching Day"
                    stroke="#0d9488"
                    strokeWidth={2.5}
                    dot={{ r: 3, fill: '#0d9488' }}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Provision Multi-Year Demand */}
        <Card>
          <CardHeader>
            <CardTitle>Provision-Level Demand by Academic Year</CardTitle>
            <CardDescription>
              Annual safeguarding concern volume across EPYP, Adult, and Apprenticeships.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[280px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={trendsData.provision_trends} margin={{ top: 10, right: 12, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                  <XAxis dataKey="academic_year" tick={{ fontSize: 11, fill: '#475569' }} />
                  <YAxis tick={{ fontSize: 11, fill: '#475569' }} axisLine={false} tickLine={false} />
                  <Tooltip />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  {datasetSummary.provisions.map((p, idx) => (
                    <Bar
                      key={p}
                      dataKey={p}
                      stackId="prov"
                      fill={PALETTE[idx % PALETTE.length]}
                      radius={idx === datasetSummary.provisions.length - 1 ? [5, 5, 0, 0] : [0, 0, 0, 0]}
                    />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Category Multi-Year Changes */}
        <Card>
          <CardHeader>
            <CardTitle>Broad Safeguarding Category Trajectories</CardTitle>
            <CardDescription>
              Multi-year shifts across broad recorded safeguarding categories.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[280px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={trendsData.category_trends} margin={{ top: 10, right: 12, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                  <XAxis dataKey="academic_year" tick={{ fontSize: 11, fill: '#475569' }} />
                  <YAxis tick={{ fontSize: 11, fill: '#475569' }} axisLine={false} tickLine={false} />
                  <Tooltip />
                  <Legend wrapperStyle={{ fontSize: 10 }} />
                  {datasetSummary.categories.map((cat, idx) => (
                    <Area
                      key={cat}
                      type="monotone"
                      dataKey={cat}
                      stackId="cat"
                      stroke={PALETTE[idx % PALETTE.length]}
                      fill={PALETTE[idx % PALETTE.length]}
                      fillOpacity={0.75}
                    />
                  ))}
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Factors Associated With Demand (LightGBM Feature Importance) */}
      <Card className="dashboard-section-contain">
        <CardHeader className="border-b border-slate-100 pb-4">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <CardTitle className="text-lg flex items-center gap-2">
                <Sparkles className="h-5 w-5 text-teal-600" />
                Factors Associated With Demand (LightGBM + MLForecast Feature Importance)
              </CardTitle>
              <CardDescription className="mt-1">
                Relative gain contribution of causal time-series lags, rolling averages, academic calendar exposure, and cohort indicators in the LightGBM model.
              </CardDescription>
            </div>
            <Badge variant="amber" className="self-start sm:self-center">
              Statistical Associations Only
            </Badge>
          </div>
        </CardHeader>

        <CardContent className="pt-5 space-y-5">
          {/* Mandatory Disclaimers */}
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <div className="rounded-xl border border-amber-300 bg-amber-50 p-4 text-xs sm:text-sm text-amber-950 flex items-start gap-3">
              <AlertCircle className="h-5 w-5 text-amber-700 shrink-0 mt-0.5" />
              <div>
                <div className="font-bold text-amber-950">
                  {trendsData.causation_disclaimer}
                </div>
                <p className="mt-1 text-xs text-amber-900 leading-relaxed">
                  Feature importance measures how strongly each historical or calendar variable helped the gradient boosted trees partition monthly count variance during training. It does not establish causal mechanisms.
                </p>
              </div>
            </div>

            <div className="rounded-xl border border-blue-200 bg-blue-50/70 p-4 text-xs sm:text-sm text-blue-950 flex items-start gap-3">
              <BookOpen className="h-5 w-5 text-blue-700 shrink-0 mt-0.5" />
              <div>
                <div className="font-bold text-blue-950">
                  Recording & Reporting Practice Context
                </div>
                <p className="mt-1 text-xs text-blue-900 leading-relaxed">
                  {trendsData.reporting_practice_note}
                </p>
              </div>
            </div>
          </div>

          {/* Chart + Detailed Feature Explanations */}
          <div className="grid grid-cols-1 gap-6 xl:grid-cols-12">
            <div className="xl:col-span-5 h-[360px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={topFeatures}
                  layout="vertical"
                  margin={{ top: 5, right: 20, left: 25, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" horizontal={false} />
                  <XAxis
                    type="number"
                    tick={{ fontSize: 11, fill: '#475569' }}
                    unit="%"
                  />
                  <YAxis
                    type="category"
                    dataKey="feature_label"
                    width={175}
                    tick={{ fontSize: 11, fill: '#0f172a', fontWeight: 600 }}
                  />
                  <Tooltip
                    formatter={(val: any) => [`${val}% of model gain`, 'Relative Importance']}
                  />
                  <Bar dataKey="importance_pct" name="Relative Importance (%)" radius={[0, 6, 6, 0]} barSize={20}>
                    {topFeatures.map((entry, idx) => (
                      <Cell
                        key={entry.feature_key}
                        fill={idx === 0 ? '#0f172a' : idx < 3 ? '#0d9488' : '#2563eb'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="xl:col-span-7">
              <div
                tabIndex={0}
                role="region"
                aria-labelledby="feature-importance-caption"
                className="table-wrapper rounded-xl border border-slate-200"
              >
                <table className="responsive-stack text-left text-xs">
                  <caption id="feature-importance-caption" className="sr-only">
                    LightGBM feature importance and statistical association explanations
                  </caption>
                  <thead className="bg-slate-100 text-slate-700 font-semibold border-b border-slate-200">
                    <tr>
                      <th className="py-2.5 px-3">Predictor Feature</th>
                      <th className="py-2.5 px-3">Domain Category</th>
                      <th className="py-2.5 px-3 text-right">Importance</th>
                      <th className="py-2.5 px-3">Association Direction</th>
                      <th className="py-2.5 px-3">Operational Explanation</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {trendsData.feature_importances.map((f) => (
                      <tr key={f.feature_key} className="hover:bg-slate-50/80">
                        <th className="py-2.5 px-3 font-bold text-slate-900">
                          {f.feature_label}
                        </th>
                        <td data-label="Category" className="py-2.5 px-3">
                          <Badge variant="slate">{f.category}</Badge>
                        </td>
                        <td data-label="Importance" className="py-2.5 px-3 text-right font-bold text-teal-800">
                          {f.importance_pct.toFixed(1)}%
                        </td>
                        <td data-label="Direction" className="py-2.5 px-3 font-medium text-slate-700">
                          {f.direction_hint}
                        </td>
                        <td data-label="Explanation" className="py-2.5 px-3 text-slate-600 leading-relaxed">
                          {f.explanation}
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

