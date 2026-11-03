# Safeguarding Forecast AI
**Predictive Safeguarding Demand Intelligence — South Devon College**

> **IMPORTANT GOVERNANCE & DATA STATUS NOTICE**
> `SYNTHETIC DEMONSTRATION DATA — NOT ACTUAL COLLEGE SAFEGUARDING RECORDS`
>
> This platform is an **operational service-demand forecasting prototype** built for the Principalship and Safeguarding Leadership Team at **South Devon College**. It forecasts the **aggregate number of safeguarding concerns** likely to be recorded during future periods to support workload planning, pastoral staffing, and resource allocation.
> - It **never** assigns risk scores to individual students.
> - It **never** predicts which individual students will experience harm or abuse.
> - It **never** processes student names, identifiers, dates of birth, or case narratives.
> - It **never** recommends reducing statutory safeguarding support on the basis of low forecast demand.

---

## 1. Quickstart & Running Locally

### Prerequisites
- **Python 3.11+** (configured in `backend/.venv`)
- **Node.js 20+** and **npm**

### Start the Backend API (FastAPI on port `8000`)
```bash
cd backend
./.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
- Interactive OpenAPI Docs: `http://127.0.0.1:8000/docs`
- Health Endpoint: `http://127.0.0.1:8000/api/health`

### Start the Frontend Dashboard (React + Vite on port `5173`)
```bash
cd frontend
npm run dev
```
- Open `http://localhost:5173` in your browser.

### Run the Automated Test Suite
```bash
PYTHONPATH=backend backend/.venv/bin/pytest backend/tests/test_safeguarding_forecast.py -v
```

---

## 2. How the Forecasting Models Work

### Model A1: Seasonal Naive Baseline (`seasonal_naive`)
- **Engine:** Nixtla `StatsForecast` (`SeasonalNaive(season_length=12)`).
- **Methodology:** Predicts each future calendar month as the observed count from the corresponding calendar month in the previous academic year ($y_{t+h} = y_{t+h-12}$). Prediction intervals (80% and 95%) are calibrated from empirical year-over-year seasonal differences.
- **Role:** Provides a transparent benchmark against which statistical and machine-learning models are evaluated.

### Model A2: Statistical Count Model (`statistical_count`)
- **Engine:** `statsmodels` Generalized Linear Model (`GLM`) with a log link.
- **Methodology:** Models non-negative monthly count data using calendar-month seasonal indicators (or Fourier harmonics on shorter series), scheduled monthly `teaching_days`, secular trend, `reporting_change_flag`, and a $\log(\text{enrolled\_students})$ exposure offset. Automatically tests for overdispersion via the Pearson $\chi^2 / \text{df}$ ratio and fits a **Negative Binomial GLM** when count variance exceeds the Poisson mean ($\text{Var}(Y) = \mu + \alpha \mu^2$).

### Model B: LightGBM + Nixtla MLForecast (`lightgbm_mlforecast`)
- **Engine:** `lightgbm.LGBMRegressor` orchestrated by Nixtla `MLForecast`.
- **Methodology:** Combines a strictly causal historical month-of-year exposure anchor with recursive lag features (`lag1`, `lag2`, `lag3`, `lag6`, `lag12`), causal rolling 3-month and 6-month averages (`RollingMean(3)`, `RollingMean(6)`), cyclical calendar encodings (`month_sin`, `month_cos`, `academic_term_code`), `teaching_days`, `enrolled_students`, strictly lagged `lag12_concerns_per_1000`, aggregate `attendance_rate`, and `reporting_change_flag`.
- **Zero Data Leakage:** All lags and rolling transforms are computed strictly from observations prior to the forecast origin.

### Model C: Google TimesFM 3.0 (`timesfm_3`)
- **Engine:** Official Google Research `timesfm3` (`google/timesfm-3.0-pytorch`, 330M parameter decoder-only time-series foundation model) running locally via native Apple Silicon MLX (`timesfm3.mlx.TimesFM3Forecaster`) with PyTorch (`timesfm3.TimesFM3Evaluator`) fallback.
- **Methodology:** Supports both zero-shot univariate forecasting and multivariate/covariate forecasting (`past_only_covariates` and `past_future_covariates`) with native decile quantile outputs (`0.1` to `0.9`).
- **Licensing & Governance Gate:** Downloaded TimesFM 3.0 weights are distributed under `timesfm-non-commercial-license-v1.0` (restricted to non-commercial, non-production research/evaluation). In this platform, TimesFM 3.0 is gated behind **Research / Prototype Mode** and explicit license acknowledgement. Switching the platform to **Operational Production Mode** automatically disables local TimesFM 3.0 weights while keeping `Statistical Count GLM`, `LightGBM + MLForecast`, and `Seasonal Naive` active.

### Statistically Calibrated Annual Prediction Intervals
Rather than naively summing monthly lower and upper bounds (which falsely assumes perfect correlation $\rho = 1.0$ across all 12 months and produces overly wide intervals), the platform computes the multi-month cumulative variance accounting for cross-month error covariance:
$$\text{Var}\left(\sum_{h=1}^H e_h\right) = \sum_{h=1}^H \sigma_h^2 + 2 \sum_{1 \le i < j \le H} \rho^{|j-i|} \sigma_i \sigma_j$$

---

## 3. Datasets Required for Eventual College Operational Use

Before importing real South Devon College statistics, formal **Information Governance (DPIA)** and **Designated Safeguarding Lead (DSL)** approval must be completed. Only **anonymised, monthly aggregated counts** are permitted.

### Recommended CSV Schema (`sample_safeguarding_data.csv` included)
| Column | Required | Type | Description |
| :--- | :---: | :---: | :--- |
| `period_start` | **Yes** | `YYYY-MM-01` | First day of the reporting calendar month |
| `provision` | Recommended | `String` | Provision cohort (`EPYP`, `Adult`, `Apprenticeships`) |
| `safeguarding_category` | Recommended | `String` | Broad safeguarding category (e.g., `Mental health and wellbeing concerns`) |
| `concern_count` | **Yes** | `Integer >= 0` | Monthly total of recorded safeguarding concerns |
| `enrolled_students` | Recommended | `Integer > 0` | Active enrolled learner headcount for the provision in that month |
| `teaching_days` | Optional | `Integer (0–31)` | Scheduled college teaching days in that calendar month |
| `attendance_rate` | Optional | `Float (0–100)` | Aggregate monthly attendance percentage for the provision |
| `reporting_change_flag` | Optional | `Integer (0 or 1)` | `1` after major recording-system or mandatory reporting changes, else `0` |

### Automatic PII & Personal-Data Rejection
The CSV validator automatically scans column names and cell values and **rejects** any upload containing student identifiers (`student_id`, `uln`), names (`student_name`, `surname`), dates of birth (`dob`), contact details, or free-text case narratives (`case_notes`, `description`).

