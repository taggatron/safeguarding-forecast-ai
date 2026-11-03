import React, { useState, useRef } from 'react';
import {
  Upload,
  Download,
  Database,
  ShieldCheck,
  AlertTriangle,
  XCircle,
  CheckCircle2,
  RefreshCw,
  FileSpreadsheet,
  Sliders,
  Lock,
  History,
  Sparkles,
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
import { apiClient } from '../lib/api';
import type {
  DatasetSummaryResponse,
  DataValidationReport,
  ModelsAvailabilityResponse,
} from '../types/api';

interface DataManagementPageProps {
  datasetSummary: DatasetSummaryResponse;
  modelsAvailability: ModelsAvailabilityResponse;
  recentRuns: any[];
  onDatasetUpdated: (newSummary: DatasetSummaryResponse) => void;
  onConfigUpdated: () => void;
}

const CANONICAL_FIELDS = [
  { key: 'period_start', label: 'Period Start Date (Required)', required: true },
  { key: 'concern_count', label: 'Concern Count (Required)', required: true },
  { key: 'provision', label: 'Provision Type', required: false },
  { key: 'safeguarding_category', label: 'Safeguarding Category', required: false },
  { key: 'enrolled_students', label: 'Enrolled Students Headcount', required: false },
  { key: 'teaching_days', label: 'Scheduled Teaching Days', required: false },
  { key: 'attendance_rate', label: 'Aggregate Attendance Rate (%)', required: false },
  { key: 'reporting_change_flag', label: 'Reporting Change Flag (0/1)', required: false },
];

export const DataManagementPage: React.FC<DataManagementPageProps> = ({
  datasetSummary,
  modelsAvailability,
  recentRuns,
  onDatasetUpdated,
  onConfigUpdated,
}) => {
  // Synthetic generator state
  const [seed, setSeed] = useState<number>(datasetSummary.seed || 42);
  const [numYears, setNumYears] = useState<number>(datasetSummary.full_academic_years || 6);
  const [baseVolume, setBaseVolume] = useState<number>(1450);
  const [ayStartMonth, setAyStartMonth] = useState<number>(datasetSummary.academic_year_start_month || 9);
  const [includeRepChange, setIncludeRepChange] = useState<boolean>(true);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);

  // CSV Upload & Validation state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [datasetCustomName, setDatasetCustomName] = useState<string>('');
  const [validationReport, setValidationReport] = useState<DataValidationReport | null>(null);
  const [columnMapping, setColumnMapping] = useState<Record<string, string>>({});
  const [isValidating, setIsValidating] = useState<boolean>(false);
  const [isImporting, setIsImporting] = useState<boolean>(false);
  const [importSuccessMsg, setImportSuccessMsg] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleRegenerateSynthetic = async () => {
    setIsGenerating(true);
    try {
      const updated = await apiClient.regenerateSyntheticDataset({
        seed,
        num_academic_years: numYears,
        start_academic_year: 2026 - numYears,
        academic_year_start_month: ayStartMonth,
        base_annual_volume: baseVolume,
        include_reporting_change: includeRepChange,
      });
      onDatasetUpdated(updated);
    } catch (err: any) {
      alert(`Error generating synthetic data: ${err.message}`);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleSwitchDataset = async (dsId: string) => {
    try {
      const updated = await apiClient.switchDataset(dsId);
      onDatasetUpdated(updated);
    } catch (err: any) {
      alert(`Failed to switch dataset: ${err.message}`);
    }
  };

  const runCsvValidation = async (file: File, customMap?: Record<string, string>) => {
    setIsValidating(true);
    setImportSuccessMsg(null);
    try {
      const report = await apiClient.validateCsv(file, customMap);
      setValidationReport(report);
      if (!customMap) {
        setColumnMapping(report.suggested_mapping || {});
      }
    } catch (err: any) {
      alert(`Validation request failed: ${err.message}`);
    } finally {
      setIsValidating(false);
    }
  };

  const handleFileChosen = (file: File) => {
    setSelectedFile(file);
    setDatasetCustomName(file.name.replace(/\.csv$/i, ''));
    runCsvValidation(file);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChosen(e.dataTransfer.files[0]);
    }
  };

  const handleTestPiiGuardrail = () => {
    const piiCsv = [
      'period_start,provision,safeguarding_category,concern_count,enrolled_students,student_name,case_notes',
      '2025-09-01,EPYP,Mental health and wellbeing concerns,45,3400,Alex Example,Confidential individual student disclosure text',
    ].join('\n');
    const blob = new Blob([piiCsv], { type: 'text/csv' });
    const testFile = new File([blob], 'unsafe_personal_data_test.csv', { type: 'text/csv' });
    handleFileChosen(testFile);
  };

  const handleLoadValidSampleIntoValidator = async () => {
    let res = await fetch('/api/dataset/sample-csv');
    if (!res.ok || (res.headers.get('content-type') || '').includes('text/html')) {
      res = await fetch('/south_devon_college_safeguarding_template.csv');
    }
    const text = await res.text();
    const blob = new Blob([text], { type: 'text/csv' });
    const sampleFile = new File([blob], 'south_devon_college_aggregated_sample.csv', {
      type: 'text/csv',
    });
    handleFileChosen(sampleFile);
  };

  const handleExecuteImport = async () => {
    if (!selectedFile || !validationReport?.can_import) return;
    setIsImporting(true);
    try {
      const updated = await apiClient.importCsv(selectedFile, datasetCustomName, columnMapping);
      onDatasetUpdated(updated);
      setImportSuccessMsg(
        `Successfully imported '${updated.name}' (${updated.total_months} months, ${updated.total_recorded_concerns.toLocaleString()} total concerns) into an isolated dataset slot.`
      );
    } catch (err: any) {
      alert(`Import failed: ${err.message}`);
    } finally {
      setIsImporting(false);
    }
  };

  const handleUpdateGovernance = async (updates: {
    governance_mode?: string;
    timesfm_license_acknowledged?: boolean;
    small_group_suppression_enabled?: boolean;
    academic_year_start_month?: number;
  }) => {
    try {
      await apiClient.updateConfiguration(updates);
      onConfigUpdated();
    } catch (err: any) {
      alert(`Configuration update failed: ${err.message}`);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between bg-white p-5 rounded-2xl border border-slate-200/80 shadow-xs">
        <div>
          <div className="flex items-center gap-2.5">
            <h1 className="fluid-title font-display font-extrabold text-slate-900">
              Data Management, CSV Import & Model Governance
            </h1>
            <Badge variant={datasetSummary.is_synthetic ? 'amber' : 'teal'}>
              Active: {datasetSummary.is_synthetic ? 'Synthetic Demo' : 'Imported Dataset'}
            </Badge>
          </div>
          <p className="mt-1 text-sm text-slate-600">
            Manage synthetic demonstration datasets, validate and import aggregated college CSV statistics, and configure model licensing guardrails.
          </p>
        </div>

        <a
          href="/south_devon_college_safeguarding_template.csv"
          download="south_devon_college_safeguarding_template.csv"
          className="inline-flex items-center gap-2 rounded-xl bg-teal-600 px-4 py-2.5 text-xs sm:text-sm font-semibold text-white hover:bg-teal-700 transition-colors shadow-xs"
        >
          <Download className="h-4 w-4" />
          Download Example CSV Template
        </a>
      </div>

      {/* Active Dataset Summary & Dataset Isolation Switcher */}
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-12">
        <Card className="xl:col-span-7">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="h-4 w-4 text-teal-600" />
              Active Dataset Coverage & Isolated Dataset Slots
            </CardTitle>
            <CardDescription>
              Synthetic demonstration datasets and imported CSV datasets are stored in strictly separated slots and can never be accidentally combined.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <div className="rounded-xl bg-slate-50 p-3 border border-slate-200/80">
                <span className="text-[11px] font-semibold uppercase text-slate-500">Date Range</span>
                <div className="mt-1 font-bold text-xs sm:text-sm text-slate-900">
                  {datasetSummary.date_range_start} → {datasetSummary.date_range_end}
                </div>
              </div>
              <div className="rounded-xl bg-slate-50 p-3 border border-slate-200/80">
                <span className="text-[11px] font-semibold uppercase text-slate-500">Coverage</span>
                <div className="mt-1 font-bold text-xs sm:text-sm text-slate-900">
                  {datasetSummary.total_months} Months ({datasetSummary.full_academic_years} AYs)
                </div>
              </div>
              <div className="rounded-xl bg-slate-50 p-3 border border-slate-200/80">
                <span className="text-[11px] font-semibold uppercase text-slate-500">Total Concerns</span>
                <div className="mt-1 font-bold text-xs sm:text-sm text-teal-800">
                  {datasetSummary.total_recorded_concerns.toLocaleString()}
                </div>
              </div>
              <div className="rounded-xl bg-slate-50 p-3 border border-slate-200/80">
                <span className="text-[11px] font-semibold uppercase text-slate-500">Latest Enrolment</span>
                <div className="mt-1 font-bold text-xs sm:text-sm text-slate-900">
                  {datasetSummary.latest_enrolled_students.toLocaleString()} learners
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <span className="text-xs font-bold text-slate-700">
                Available Isolated Datasets (Click to Switch Active Dataset):
              </span>
              <div className="space-y-2">
                {datasetSummary.available_datasets.map((ds) => {
                  const isActive = ds.dataset_id === datasetSummary.dataset_id;
                  return (
                    <div
                      key={ds.dataset_id}
                      className={`flex flex-col sm:flex-row sm:items-center justify-between gap-2 rounded-xl border p-3 transition-all ${
                        isActive
                          ? 'border-teal-500 bg-teal-50/40'
                          : 'border-slate-200 bg-white hover:bg-slate-50'
                      }`}
                    >
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="font-bold text-xs sm:text-sm text-slate-900">
                            {ds.name}
                          </span>
                          <Badge variant={ds.is_synthetic ? 'amber' : 'teal'}>
                            {ds.is_synthetic ? 'Synthetic Demo' : 'Imported CSV'}
                          </Badge>
                          {isActive && <Badge variant="emerald">Active</Badge>}
                        </div>
                        <div className="text-[11px] text-slate-500 mt-0.5">
                          ID: <code className="font-mono">{ds.dataset_id}</code> • {ds.row_count.toLocaleString()} records • {ds.total_concerns.toLocaleString()} concerns
                          {ds.seed != null ? ` • Seed: ${ds.seed}` : ''}
                        </div>
                      </div>
                      {!isActive && (
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleSwitchDataset(ds.dataset_id)}
                        >
                          Activate Dataset
                        </Button>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Synthetic Demo Data Generator Controls */}
        <Card className="xl:col-span-5">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-amber-600" />
              Synthetic Demonstration Data Generator
            </CardTitle>
            <CardDescription>
              Reproducibly generate fictional multi-year FE college safeguarding data with controlled random seed, seasonality, and cohort growth.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3.5">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label htmlFor="synth-seed" className="block text-xs font-semibold text-slate-700 mb-1">
                  Random Seed (Reproducible)
                </label>
                <input
                  id="synth-seed"
                  type="number"
                  value={seed}
                  onChange={(e) => setSeed(Number(e.target.value))}
                  className="w-full rounded-xl border border-slate-300 px-3 py-1.5 text-xs font-semibold text-slate-900"
                />
              </div>

              <div>
                <label htmlFor="synth-years" className="block text-xs font-semibold text-slate-700 mb-1">
                  Academic Years (3–10)
                </label>
                <select
                  id="synth-years"
                  value={numYears}
                  onChange={(e) => setNumYears(Number(e.target.value))}
                  className="w-full rounded-xl border border-slate-300 bg-slate-50 px-3 py-1.5 text-xs font-semibold text-slate-900"
                >
                  {[3, 4, 5, 6, 7, 8].map((y) => (
                    <option key={y} value={y}>
                      {y} Academic Years ({y * 12} Months)
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label htmlFor="synth-vol" className="block text-xs font-semibold text-slate-700 mb-1">
                  Base Annual Volume
                </label>
                <input
                  id="synth-vol"
                  type="number"
                  step={50}
                  min={400}
                  max={5000}
                  value={baseVolume}
                  onChange={(e) => setBaseVolume(Number(e.target.value))}
                  className="w-full rounded-xl border border-slate-300 px-3 py-1.5 text-xs font-semibold text-slate-900"
                />
              </div>

              <div>
                <label htmlFor="synth-ay-month" className="block text-xs font-semibold text-slate-700 mb-1">
                  Academic Year Start Month
                </label>
                <select
                  id="synth-ay-month"
                  value={ayStartMonth}
                  onChange={(e) => setAyStartMonth(Number(e.target.value))}
                  className="w-full rounded-xl border border-slate-300 bg-slate-50 px-3 py-1.5 text-xs font-semibold text-slate-900"
                >
                  <option value={9}>September – August (Default FE)</option>
                  <option value={8}>August – July</option>
                  <option value={1}>January – December (Calendar)</option>
                </select>
              </div>
            </div>

            <label className="flex items-center gap-2 text-xs text-slate-700 cursor-pointer pt-1">
              <input
                type="checkbox"
                checked={includeRepChange}
                onChange={(e) => setIncludeRepChange(e.target.checked)}
                className="rounded border-slate-300"
              />
              <span>Include Sep 2023 reporting system & mandatory staff training step-change</span>
            </label>

            <Button
              variant="primary"
              size="md"
              className="w-full"
              disabled={isGenerating}
              onClick={handleRegenerateSynthetic}
            >
              <RefreshCw className={`h-4 w-4 ${isGenerating ? 'animate-spin' : ''}`} />
              {isGenerating
                ? 'Generating Synthetic Dataset...'
                : 'Regenerate & Activate Synthetic Dataset'}
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* CSV Drag-and-Drop Import, Column Mapping, and PII Guardrail Validation */}
      <Card>
        <CardHeader>
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <FileSpreadsheet className="h-5 w-5 text-teal-600" />
                Aggregated Safeguarding CSV Import & Validation
              </CardTitle>
              <CardDescription>
                Upload monthly aggregated safeguarding counts. Automatically detects columns, validates missing months and duplicates, and strictly rejects student-level personal data (PII) or case narratives.
              </CardDescription>
            </div>
            <div className="flex flex-wrap gap-2">
              <Button variant="outline" size="sm" onClick={handleLoadValidSampleIntoValidator}>
                Load Sample CSV into Validator
              </Button>
              <Button variant="outline" size="sm" onClick={handleTestPiiGuardrail}>
                <ShieldCheck className="h-3.5 w-3.5 text-rose-600" />
                Test PII Rejection Guardrail
              </Button>
            </div>
          </div>
        </CardHeader>

        <CardContent className="space-y-5">
          {/* Drag and drop zone */}
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragActive(true);
            }}
            onDragLeave={() => setDragActive(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`flex flex-col items-center justify-center rounded-2xl border-2 border-dashed p-6 text-center transition-all cursor-pointer ${
              dragActive
                ? 'border-teal-500 bg-teal-50/50'
                : 'border-slate-300 bg-slate-50/60 hover:border-teal-500 hover:bg-slate-50'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv,text/csv"
              className="hidden"
              onChange={(e) => {
                if (e.target.files && e.target.files[0]) {
                  handleFileChosen(e.target.files[0]);
                }
              }}
            />
            <Upload className="h-8 w-8 text-teal-600 mb-2" />
            <p className="text-sm font-bold text-slate-800">
              {selectedFile
                ? `Selected File: ${selectedFile.name}`
                : 'Drag and drop your aggregated safeguarding CSV here, or click to browse'}
            </p>
            <p className="mt-1 text-xs text-slate-500 max-w-2xl">
              Recommended Schema: <code className="font-mono text-[11px] bg-white px-1.5 py-0.5 rounded border border-slate-200">period_start,provision,safeguarding_category,concern_count,enrolled_students,teaching_days,attendance_rate,reporting_change_flag</code>
            </p>
          </div>

          {importSuccessMsg && (
            <div className="rounded-xl border border-emerald-300 bg-emerald-50 p-4 flex items-center gap-3 text-xs sm:text-sm text-emerald-950">
              <CheckCircle2 className="h-5 w-5 text-emerald-600 shrink-0" />
              <div className="font-medium">{importSuccessMsg}</div>
            </div>
          )}

          {/* Validation Report & Manual Column Mapping */}
          {validationReport && (
            <div className="space-y-4 rounded-2xl border border-slate-200 bg-slate-50/50 p-4">
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 pb-3">
                <div className="flex items-center gap-2.5">
                  {validationReport.can_import ? (
                    <Badge variant="emerald">
                      <CheckCircle2 className="h-3.5 w-3.5" /> Validation Passed — Ready to Import
                    </Badge>
                  ) : (
                    <Badge variant="rose">
                      <XCircle className="h-3.5 w-3.5" /> Validation Blocked — Action Required
                    </Badge>
                  )}
                  <span className="text-xs font-semibold text-slate-700">
                    {validationReport.row_count} rows • {validationReport.total_months} months ({validationReport.full_academic_years} full academic years)
                  </span>
                </div>

                {validationReport.can_import && (
                  <div className="flex items-center gap-2">
                    <input
                      type="text"
                      placeholder="Dataset label..."
                      value={datasetCustomName}
                      onChange={(e) => setDatasetCustomName(e.target.value)}
                      className="rounded-xl border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-slate-900"
                    />
                    <Button
                      variant="secondary"
                      size="md"
                      disabled={isImporting}
                      onClick={handleExecuteImport}
                    >
                      {isImporting ? 'Importing...' : 'Import as Active Dataset'}
                    </Button>
                  </div>
                )}
              </div>

              {/* Issues List */}
              {validationReport.issues.length > 0 && (
                <div className="space-y-2">
                  {validationReport.issues.map((iss, idx) => (
                    <div
                      key={idx}
                      className={`flex items-start gap-2.5 rounded-xl p-3 text-xs border ${
                        iss.severity === 'error'
                          ? 'bg-rose-50 border-rose-200 text-rose-950'
                          : iss.severity === 'warning'
                          ? 'bg-amber-50 border-amber-200 text-amber-950'
                          : 'bg-blue-50 border-blue-200 text-blue-950'
                      }`}
                    >
                      {iss.severity === 'error' ? (
                        <XCircle className="h-4 w-4 text-rose-600 shrink-0 mt-0.5" />
                      ) : (
                        <AlertTriangle className="h-4 w-4 text-amber-600 shrink-0 mt-0.5" />
                      )}
                      <div>
                        <span className="font-bold uppercase mr-1.5">[{iss.code}]:</span>
                        <span>{iss.message}</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* Column Mapping Controls */}
              {validationReport.detected_columns.length > 0 && (
                <div className="bg-white rounded-xl p-4 border border-slate-200">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-xs font-bold text-slate-900">
                      Column Mapping (Auto-Recognised & Manually Configurable)
                    </span>
                    {selectedFile && (
                      <Button
                        variant="outline"
                        size="sm"
                        disabled={isValidating}
                        onClick={() => runCsvValidation(selectedFile, columnMapping)}
                      >
                        Re-Validate with Mapping
                      </Button>
                    )}
                  </div>
                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
                    {CANONICAL_FIELDS.map((field) => (
                      <div key={field.key}>
                        <label className="block text-[11px] font-semibold text-slate-600 mb-1">
                          {field.label}
                        </label>
                        <select
                          value={columnMapping[field.key] || ''}
                          onChange={(e) =>
                            setColumnMapping((prev) => ({
                              ...prev,
                              [field.key]: e.target.value,
                            }))
                          }
                          className="w-full rounded-lg border border-slate-300 bg-slate-50 px-2.5 py-1.5 text-xs text-slate-800"
                        >
                          <option value="">-- Not Mapped --</option>
                          {validationReport.detected_columns.map((col) => (
                            <option key={col} value={col}>
                              {col}
                            </option>
                          ))}
                        </select>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Preview Rows */}
              {validationReport.preview_rows.length > 0 && (
                <div>
                  <span className="block text-xs font-bold text-slate-700 mb-2">
                    Validated Data Preview (First {validationReport.preview_rows.length} Rows):
                  </span>
                  <div className="table-wrapper rounded-xl border border-slate-200 bg-white">
                    <table className="text-left text-xs">
                      <thead className="bg-slate-100 text-slate-700 font-semibold border-b border-slate-200">
                        <tr>
                          {Object.keys(validationReport.preview_rows[0]).map((col) => (
                            <th key={col} className="py-2 px-3">
                              {col}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {validationReport.preview_rows.map((r, rIdx) => (
                          <tr key={rIdx}>
                            {Object.values(r).map((val: any, cIdx) => (
                              <td key={cIdx} className="py-1.5 px-3 text-slate-700">
                                {String(val)}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Model Governance, Licensing & Small-Group Disclosure Controls */}
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-12">
        <Card className="xl:col-span-6">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <ShieldCheck className="h-5 w-5 text-teal-600" />
              Safeguarding Governance & TimesFM 3.0 Licensing Controls
            </CardTitle>
            <CardDescription>
              Configure deployment environment mode, TimesFM 3.0 non-commercial license gate, and small-group count suppression.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Environment Mode */}
            <div className="rounded-xl border border-slate-200 p-3.5 bg-slate-50/60 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-slate-900">
                  Platform Governance Environment Mode
                </span>
                <Badge
                  variant={
                    modelsAvailability.governance_mode === 'research_prototype'
                      ? 'purple'
                      : 'emerald'
                  }
                >
                  {modelsAvailability.governance_mode === 'research_prototype'
                    ? 'Research / Prototype Mode'
                    : 'Operational Production Mode'}
                </Badge>
              </div>
              <p className="text-xs text-slate-600 leading-relaxed">
                Downloaded Google TimesFM 3.0 weights are governed by <code className="font-mono text-[11px]">timesfm-non-commercial-license-v1.0</code>. Switching to <strong>Operational Production Mode</strong> automatically disables local TimesFM 3.0 weights while keeping Statistical Count GLM and LightGBM + MLForecast active.
              </p>
              <div className="flex gap-2 pt-1">
                <Button
                  variant={
                    modelsAvailability.governance_mode === 'research_prototype'
                      ? 'primary'
                      : 'outline'
                  }
                  size="sm"
                  onClick={() =>
                    handleUpdateGovernance({ governance_mode: 'research_prototype' })
                  }
                >
                  Research / Prototype Mode
                </Button>
                <Button
                  variant={
                    modelsAvailability.governance_mode === 'operational_production'
                      ? 'primary'
                      : 'outline'
                  }
                  size="sm"
                  onClick={() =>
                    handleUpdateGovernance({ governance_mode: 'operational_production' })
                  }
                >
                  Operational Production Mode
                </Button>
              </div>
            </div>

            {/* TimesFM License Acknowledgement */}
            <label className="flex items-start gap-2.5 rounded-xl border border-slate-200 p-3.5 text-xs text-slate-700 cursor-pointer">
              <input
                type="checkbox"
                checked={modelsAvailability.timesfm_license_acknowledged}
                onChange={(e) =>
                  handleUpdateGovernance({
                    timesfm_license_acknowledged: e.target.checked,
                  })
                }
                className="mt-0.5 rounded border-slate-300"
              />
              <div>
                <div className="font-bold text-slate-900">
                  Acknowledge Google TimesFM 3.0 Non-Commercial License Terms
                </div>
                <p className="mt-0.5 text-slate-500">
                  Unchecking this immediately sets TimesFM 3.0 status to &ldquo;Not available&rdquo; across the platform to demonstrate fallback resilience.
                </p>
              </div>
            </label>

            {/* Small-Group Disclosure Protection */}
            <label className="flex items-start gap-2.5 rounded-xl border border-slate-200 p-3.5 text-xs text-slate-700 cursor-pointer">
              <input
                type="checkbox"
                checked={datasetSummary.small_group_suppression_enabled}
                onChange={(e) =>
                  handleUpdateGovernance({
                    small_group_suppression_enabled: e.target.checked,
                  })
                }
                className="mt-0.5 rounded border-slate-300"
              />
              <div>
                <div className="font-bold text-slate-900">
                  Enable Small-Group Disclosure Protection (&lt; 5 Concerns Suppression)
                </div>
                <p className="mt-0.5 text-slate-500">
                  Suppresses historical category breakdowns with fewer than 5 recorded concerns to prevent indirect identification in small learner cohorts.
                </p>
              </div>
            </label>
          </CardContent>
        </Card>

        {/* Model Availability & Reproducible Audit Log */}
        <Card className="xl:col-span-6">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <History className="h-5 w-5 text-blue-600" />
              Engine Status & Reproducible Forecast Run Audit Log
            </CardTitle>
            <CardDescription>
              Live status of all registered forecasting engines and recent reproducible forecast executions stored in SQLite.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              {modelsAvailability.models.map((m) => (
                <div
                  key={m.model_id}
                  className="flex items-center justify-between rounded-xl border border-slate-200 p-2.5 text-xs"
                >
                  <div>
                    <div className="font-bold text-slate-900">{m.name}</div>
                    <div className="text-[11px] text-slate-500">{m.backend_engine}</div>
                  </div>
                  <Badge variant={m.available ? 'emerald' : 'slate'}>
                    {m.status_label}
                  </Badge>
                </div>
              ))}
            </div>

            {recentRuns.length > 0 && (
              <div>
                <span className="block text-xs font-bold text-slate-700 mb-1.5">
                  Recent Reproducible Model Runs ({recentRuns.length}):
                </span>
                <div className="max-h-40 overflow-auto rounded-xl border border-slate-200">
                  <table className="w-full text-left text-[11px]">
                    <thead className="bg-slate-100 text-slate-700 font-semibold">
                      <tr>
                        <th className="py-1.5 px-2.5">Run ID</th>
                        <th className="py-1.5 px-2.5">Model</th>
                        <th className="py-1.5 px-2.5">Scope</th>
                        <th className="py-1.5 px-2.5 text-right">Total</th>
                        <th className="py-1.5 px-2.5 text-right">Runtime</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {recentRuns.slice(0, 8).map((r) => (
                        <tr key={r.run_id}>
                          <td className="py-1.5 px-2.5 font-mono text-slate-600">{r.run_id}</td>
                          <td className="py-1.5 px-2.5 font-medium text-slate-800">{r.model_id}</td>
                          <td className="py-1.5 px-2.5 text-slate-600">
                            {r.provision} / {r.horizon}m
                          </td>
                          <td className="py-1.5 px-2.5 text-right font-bold text-teal-800">
                            {r.annual_total}
                          </td>
                          <td className="py-1.5 px-2.5 text-right text-slate-500">
                            {r.runtime_ms}ms
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

