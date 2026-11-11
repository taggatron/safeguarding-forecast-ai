import { vercelFallbackEngine } from './vercelFallbackEngine';
import esotericTelemetrySnapshot from './esotericTelemetrySnapshot.json';
import type {
  BacktestResponse,
  DatasetSummaryResponse,
  DataValidationReport,
  ForecastRequest,
  ForecastResponse,
  HistoricalStatsResponse,
  ModelsAvailabilityResponse,
  TrendsInsightsResponse,
} from '../types/api';

const API_BASE = '/api';

async function fetchWithFallback<T>(
  input: RequestInfo | URL,
  init: RequestInit | undefined,
  fallbackFn: () => T | Promise<T>
): Promise<T> {
  try {
    const res = await fetch(input, init);
    const contentType = res.headers.get('content-type') || '';
    if (!res.ok) {
      // If running on static Vercel deployment where /api/* returns 404 or HTML, use fallback engine
      if (res.status === 404 || res.status === 405 || contentType.includes('text/html')) {
        return await fallbackFn();
      }
      let errText = `HTTP ${res.status}`;
      try {
        const data = await res.json();
        errText = data.detail || JSON.stringify(data);
      } catch {
        errText = await res.text();
      }
      throw new Error(errText);
    }
    if (!contentType.includes('application/json')) {
      return await fallbackFn();
    }
    return (await res.json()) as T;
  } catch (err: any) {
    if (err instanceof TypeError || String(err.message).includes('Failed to fetch')) {
      return await fallbackFn();
    }
    throw err;
  }
}

export const apiClient = {
  getDatasetSummary(): Promise<DatasetSummaryResponse> {
    return fetchWithFallback(
      `${API_BASE}/dataset/summary`,
      undefined,
      () => vercelFallbackEngine.getDatasetSummary()
    );
  },

  getModelsAvailability(): Promise<ModelsAvailabilityResponse> {
    return fetchWithFallback(
      `${API_BASE}/models/availability`,
      undefined,
      () => vercelFallbackEngine.getModelsAvailability()
    );
  },

  getHistoricalStats(
    academicYear = 'All',
    provision = 'All',
    safeguardingCategory = 'All'
  ): Promise<HistoricalStatsResponse> {
    const params = new URLSearchParams({
      academic_year: academicYear,
      provision,
      safeguarding_category: safeguardingCategory,
    });
    return fetchWithFallback(
      `${API_BASE}/historical?${params.toString()}`,
      undefined,
      () => vercelFallbackEngine.getHistoricalStats(academicYear, provision, safeguardingCategory)
    );
  },

  generateForecast(req: ForecastRequest): Promise<ForecastResponse> {
    return fetchWithFallback(
      `${API_BASE}/forecast`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req),
      },
      () => vercelFallbackEngine.generateForecast(req)
    );
  },

  getBacktestComparison(
    provision = 'All',
    safeguardingCategory = 'All',
    horizon = 12
  ): Promise<BacktestResponse> {
    const params = new URLSearchParams({
      provision,
      safeguarding_category: safeguardingCategory,
      horizon: String(horizon),
    });
    return fetchWithFallback(
      `${API_BASE}/models/backtest?${params.toString()}`,
      undefined,
      () => vercelFallbackEngine.getBacktestComparison(provision, safeguardingCategory, horizon)
    );
  },

  getTrendsInsights(
    provision = 'All',
    safeguardingCategory = 'All'
  ): Promise<TrendsInsightsResponse> {
    const params = new URLSearchParams({
      provision,
      safeguarding_category: safeguardingCategory,
    });
    return fetchWithFallback(
      `${API_BASE}/trends?${params.toString()}`,
      undefined,
      () => vercelFallbackEngine.getTrendsInsights(provision, safeguardingCategory)
    );
  },

  regenerateSyntheticDataset(payload: {
    seed: number;
    num_academic_years: number;
    start_academic_year: number;
    academic_year_start_month: number;
    base_annual_volume: number;
    include_reporting_change: boolean;
    provisions?: string[];
    categories?: string[];
  }): Promise<DatasetSummaryResponse> {
    return fetchWithFallback(
      `${API_BASE}/dataset/synthetic`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      },
      () => vercelFallbackEngine.regenerateSyntheticDataset(payload)
    );
  },

  switchDataset(datasetId: string): Promise<DatasetSummaryResponse> {
    return fetchWithFallback(
      `${API_BASE}/dataset/switch`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_id: datasetId }),
      },
      () => vercelFallbackEngine.switchDataset(datasetId)
    );
  },

  updateConfiguration(payload: {
    academic_year_start_month?: number;
    governance_mode?: string;
    timesfm_license_acknowledged?: boolean;
    small_group_suppression_enabled?: boolean;
  }): Promise<any> {
    return fetchWithFallback(
      `${API_BASE}/config/update`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      },
      () => vercelFallbackEngine.updateConfiguration(payload)
    );
  },

  validateCsv(
    file: File,
    columnMapping?: Record<string, string>
  ): Promise<DataValidationReport> {
    const formData = new FormData();
    formData.append('file', file);
    if (columnMapping && Object.keys(columnMapping).length > 0) {
      formData.append('column_mapping_json', JSON.stringify(columnMapping));
    }
    return fetchWithFallback(
      `${API_BASE}/dataset/validate-csv`,
      {
        method: 'POST',
        body: formData,
      },
      () => vercelFallbackEngine.validateCsv(file, columnMapping)
    );
  },

  importCsv(
    file: File,
    datasetName?: string,
    columnMapping?: Record<string, string>
  ): Promise<DatasetSummaryResponse> {
    const formData = new FormData();
    formData.append('file', file);
    if (datasetName) formData.append('dataset_name', datasetName);
    if (columnMapping && Object.keys(columnMapping).length > 0) {
      formData.append('column_mapping_json', JSON.stringify(columnMapping));
    }
    return fetchWithFallback(
      `${API_BASE}/dataset/import-csv`,
      {
        method: 'POST',
        body: formData,
      },
      () => vercelFallbackEngine.importCsv(file, datasetName, columnMapping)
    );
  },

  getRecentForecastRuns(limit = 15): Promise<{ runs: any[] }> {
    return fetchWithFallback(
      `${API_BASE}/models/runs?limit=${limit}`,
      undefined,
      () => vercelFallbackEngine.getRecentForecastRuns()
    );
  },

  getEsotericTelemetry(targetDate?: string, provision = 'All'): Promise<any> {
    const params = new URLSearchParams({ provision });
    if (targetDate) {
      params.set('t', targetDate);
    }
    return fetchWithFallback(
      `${API_BASE}/esoteric-telemetry?${params.toString()}`,
      undefined,
      () => esotericTelemetrySnapshot
    );
  },
};
