# Safeguarding Forecast AI — Vercel Deployment & Technical Audit: Feature Bloat vs. Sample Size

**Date:** 12 November 2026  
**Repository:** `Safeguarding_Forecast_AI` (`main` @ `92510fb`)  
**Production URL:** https://safeguarding-forecast-ai.vercel.app  

---

## 1. Vercel Production Deployment Summary

| Field | Value |
| :--- | :--- |
| **Production Alias** | https://safeguarding-forecast-ai.vercel.app |
| **Deployment URL** | https://safeguarding-forecast-23xdjvk4i-daniel-taggs-projects.vercel.app |
| **Deployment ID** | `dpl_99WQ6CuKtmsXXmX1nJFgnTLV3eZq` |
| **Target Environment** | `production` |
| **Status** | `READY` (HTTP `200` verified) |
| **Commit** | `92510fb` (`BIG refactor: add type hints and improve path resolution in frozen adapter and residual stacker`) |
| **Framework & Output** | `vite` (`frontend/dist`, 2,019 modules transformed) |
| **Build Duration** | `35s` |
| **Post-Deploy Observability** | `0` build/edge errors; static SPA served via Vercel Edge CDN with client-side snapshot fallback (`frontend/src/lib/api.ts`, `esotericTelemetrySnapshot.json`); no server-side log drains configured |

---

## 2. Criterion Under Audit

> **Feature Bloat vs. Sample Size:** *On weekly academic series ($\approx 190$ teaching days or $39$ active weeks per year), stacking all 13 frameworks unregularized will degrade out-of-sample error via variance inflation. The empirical literature consistently shows these features work best when paired with strict causal windowing, group-sparse pruning, and walk-forward ablation against a strong LightGBM or Negative Binomial baseline.*

---

## 3. Executive Verdict

- **Core Operational Forecasting (`backend/app/services/forecast_service.py`):** **0% susceptible.** Completely isolated from `esoteric_math`; core baseline weights and out-of-sample predictions are cryptographically locked via IEEE-754 SHA-256 digests (`FrozenBaselineAdapter`).
- **Live Telemetry & Default Retrospective Pipeline (`STAGED_CORE_4` in `residual_stacker.py`):** **Very low susceptibility.** Restricts eligibility to 4 high-evidence modules (`f01, f02, f03, f07`), freezes baseline predictions as a fixed log-offset (`init_score = log(mu_base)`), applies two-stage block-permuted BorutaSHAP + Greedy 1-SE module quarantining, and falls back to exact baseline identity (`atol = 1e-12`) when noise dominates.
- **Opt-In Full-Stack Gated Mode (`FULL_13_GATED` in `residual_stacker.py`):** **Low-to-moderate susceptibility at small sample horizons ($N \le 78$ weeks).** Successfully quarantines all 13 modules at Folds `104` and `130`, but exhibits a small-sample multiple-testing false positive (`f05` Ollivier-Ricci) at Fold `78` on `NB2` and `ZINB`, slightly degrading `CRPS` (`5.7751` $\rightarrow$ `5.8020`) and `Surge MAE` (`11.109` $\rightarrow$ `11.365`).
- **Legacy Direct-Refit Conditioning Pipeline (`esoteric_math/conditioning.py` + `ablation_runner.py`):** **Moderate-to-high susceptibility (empirically confirmed variance inflation).** Allowing up to 18 esoteric features alongside 12 base predictors (`30` total features on $N = 78..130$ rows) and refitting baselines from scratch degrades departmental hierarchical `WRMSSE` across all three model families (`+6.0%` on `NB2`, `+3.2%` on `ZINB`, and `+17.1%` on `TweedieLightGBM`).

---

## 4. Dimensionality vs. Effective Sample Size ($p$ vs. $N_{\text{eff}}$)

Across all 13 mathematical modules (`f01`–`f13` in `esoteric_math/features/__init__.py`), `EsotericFeatureSuite` generates **~135–145 candidate esoteric columns** on top of the **12 domain baseline covariates** (`DEFAULT_BENCHMARK_FEATURES` in `backend/app/models/safeguarding_benchmark.py`):

- **Temporal, Spectral & Contagion (`f01`–`f03`):** Dual-clock calendar/190-day warped Fourier harmonics & causal Hilbert phase (`f01`, ~20 cols), boundary-causal Haar/DB4 MODWT & Morlet wavelets (`f02`, ~21 cols), multivariate Hawkes self-/cross-exciting contagion (`f03`, ~12 cols).
- **Topological, Geometric & Operator (`f04`–`f10`):** Takens persistent homology (`f04`, 9 cols), Graph Spectral GFT & Ollivier-Ricci curvature (`f05`, 7 cols), Koopman DMD & Tropical max-plus algebra (`f06`, 7 cols), depth-3 Rough Path Log-Signatures & Lévy area (`f07`, ~17 cols), Cellular Sheaf cohomology (`f08`, 8 cols), $p$-adic Vladimirov & Poincaré ball geometry (`f09`, 6 cols), Sinkhorn & Sliced Wasserstein Optimal Transport (`f10`, 7 cols).
- **Spectral Random Matrix, Manifold & Criticality (`f11`–`f13`):** Marchenko-Pastur & Tracy-Widom RMT (`f11`, 6 cols), Negative Binomial Fisher-Rao information geometry (`f12`, 8 cols), Fractional differencing & Sandpile SOC (`f13`, 7 cols).

On the 3-academic-year weekly benchmark (`156` calendar weeks, expanding walk-forward origins at weeks `78`, `104`, and `130` with `20` holiday/closure/low-activity weeks per 52-week year), Fold 1 (`origin = 78`) contains only **$N = 78$ total rows** and **$N_{\text{eff}} \approx 48\text{–}58$ active instructional weeks**. Unregularized 13-framework stacking therefore operates deep inside an overparameterized $p \approx 3\,N_{\text{eff}}$ regime where variance inflation is guaranteed without aggressive group-sparse pruning.

---

## 5. Pipeline-by-Pipeline Empirical Walk-Forward Evidence

### 5.1 Architecture Overview Across Repository Paths

| Pipeline Path | Active Scope | Susceptibility | Mechanism & Empirical Behaviour |
| :--- | :--- | :--- | :--- |
| **Core Monthly Forecasts** (`forecast_service.py`) | `/api/forecast` & primary UI tabs | **None (0%)** | Zero coupling to `esoteric_math`; `FrozenBaselineAdapter` verifies SHA-256 model & prediction immutability. |
| **Post-Hoc Residual Stacker (`STAGED_CORE_4`)** (`residual_stacker.py`) | **Default live mode** (`/api/esoteric-telemetry`) | **Very Low** | Restricts pool to `f01, f02, f03, f07`; quarantines 100% of noise on `NB2`/`ZINB` and admits `f03` on `TweedieLightGBM`. |
| **Post-Hoc Residual Stacker (`FULL_13_GATED`)** (`residual_stacker.py`) | Opt-in 13-module gated mode | **Low–Moderate** (small $N$) | Screens all 13 modules via Block-BorutaSHAP + 1-SE gate; clean at $N \ge 104$, minor `f05` false positive at $N = 78$. |
| **Direct Refit Funnels** (`conditioning.py` + `ablation_runner.py`) | Standalone ablation benchmark | **Moderate–High** | Refits `NB2`, `ZINB`, and `TweedieLightGBM` jointly with up to 18 esoteric cols; suffers hierarchical `WRMSSE` inflation. |

### 5.2 Empirical Proof of Variance Inflation in Joint Refitting (`artifacts/esoteric_ablation_scoreboard.json`)

When baselines are refitted jointly across all 13 modules via `GLMConditioningFunnel` and `GBDTConditioningFunnel` (`folds = [78, 104, 130]`, `78` out-of-sample weeks), curriculum-area `WRMSSE` and `WSPL` degrade across every model family—even though individual modules improve error when evaluated in single-module ablations (e.g., `+f06` reduces `NB2` `RMSE` from `12.552` $\rightarrow$ `10.879` and `CRPS` from `5.9991` $\rightarrow$ `5.3910`; `+f01` reduces `TweedieLightGBM` `CRPS` from `5.9768` $\rightarrow$ `5.7523`):

| Contender (`ablation_runner.py`) | `WRMSSE` | `RMSSE` | `RMSLE` | `RMSE` | `Surge MAE` | `NB Dev` | `WSPL` | `CRPS` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `control_nb2_glm` (Locked Baseline) | **0.5046** | **0.4573** | **0.6022** | 12.552 | 12.968 | 2.3618 | **0.1014** | 5.9991 |
| `esoteric_nb2_glm` (Joint 13-Module Funnel) | 0.5351 | 0.4584 | 0.6102 | **12.541** | **12.937** | **2.3296** | 0.1081 | **5.9881** |
| `control_zinb` (Locked Baseline) | **0.4951** | **0.4517** | **0.6010** | **12.390** | **12.659** | **2.3480** | **0.0992** | **6.0687** |
| `esoteric_zinb` (Joint 13-Module Funnel) | 0.5109 | 0.4642 | 0.6092 | 12.692 | 13.256 | 2.3551 | 0.1055 | 6.2197 |
| `control_tweedie_lgbm` (Locked Baseline) | **0.4601** | **0.3957** | 0.8613 | **10.905** | **11.869** | 3.1496 | **0.1031** | 5.9768 |
| `esoteric_tweedie_lgbm` (Joint 13-Module Funnel) | 0.5389 | 0.4307 | **0.6246** | 11.803 | 12.455 | **2.4313** | 0.1225 | **5.8832** |

### 5.3 Mitigation in Post-Hoc Residual Quarantine (`artifacts/retrospective_5way_scoreboard.json`)

Replacing joint refitting with frozen-baseline post-hoc offset stacking (`PostHocResidualStacker`) eliminates the catastrophic variance inflation above, while highlighting why `STAGED_CORE_4` outperforms `FULL_13_GATED` at small $N$:

| Model Family & Rollout Arm | Admitted Modules (`F78 / F104 / F130`) | `WRMSSE` | `RMSLE` | `RMSE` | `Surge MAE` | `NB Dev` | `CRPS` | `PICP 95%` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NB2 GLM** — `1_frozen_baseline` | `— / — / —` | **0.4888** | 0.5662 | 12.787 | **11.109** | **1.8404** | **5.7751** | **94.87%** |
| **NB2 GLM** — `3_staged_core_4_stack` | `[] / [] / []` | **0.4888** | 0.5662 | 12.787 | **11.109** | **1.8404** | **5.7751** | **94.87%** |
| **NB2 GLM** — `4_full_13_gated_stack` | `[f05] / [] / []` | **0.4888** | **0.5609** | **12.784** | 11.365 | 1.8510 | 5.8020 | 92.31% |
| **ZINB** — `1_frozen_baseline` | `— / — / —` | **0.5089** | 0.5664 | 12.616 | **10.396** | **1.8442** | **5.7304** | **93.59%** |
| **ZINB** — `3_staged_core_4_stack` | `[] / [] / []` | **0.5089** | 0.5664 | 12.616 | **10.396** | **1.8442** | **5.7304** | **93.59%** |
| **ZINB** — `4_full_13_gated_stack` | `[f05] / [] / []` | **0.5089** | **0.5627** | **12.603** | 10.614 | 1.8545 | 5.7509 | **93.59%** |
| **Tweedie LGBM** — `1_frozen_baseline` | `— / — / —` | **0.5225** | 0.8286 | 12.879 | 12.485 | 3.1617 | 6.8536 | 75.64% |
| **Tweedie LGBM** — `3_staged_core_4_stack` | `[f03] / [] / []` | 0.5235 | **0.7272** | **12.508** | **12.265** | **2.5745** | **6.4195** | **78.21%** |
| **Tweedie LGBM** — `4_full_13_gated_stack` | `[f03] / [] / []` | 0.5235 | **0.7272** | **12.508** | **12.265** | **2.5745** | **6.4195** | **78.21%** |

---

## 6. Alignment Against Literature Best Practices

1. **Strict Causal Windowing (`Strong — Verified to 1e-12`):**
   - `AdaptiveSchemaBridge` (`esoteric_math/schema_bridge.py`) enforces one-step causal lag shifting (`_causal_shift_1d`) across count, category, and cohort stalk channels in `"forecast"` mode and fits all scalers/graphs strictly inside `[:train_cutoff]`.
   - `VintageSnapshotFilter` and `AcademicYearTopologyManager` (`esoteric_math/retrospective/vintage_backfill.py`) block late-logged disclosures (`logged_date > t`) and reset Tier C cohort graphs (`f05`, `f08`, `f09`) at every September academic rollover with a 4-week exponentially decayed warmup prior (`w_prior == 0.0` from Week 5 onward).
   - Verified under $+10^6$ future-corruption and backdated-disclosure test suites (`tests/test_causal_bridge_non_leakage.py`, `tests/test_vintage_and_september_reset.py`).
2. **Group-Sparse Pruning (`Strong in residual_stacker.py / Weak in conditioning.py`):**
   - `PostHocResidualStacker` combines feature-level 4-period circular block-permuted `BlockBorutaSHAPGate` (`>= 18/25` shadow wins) with module-group-level `GreedyOneSEModuleSelector` (`1.15 * SE` threshold across 3 chronological sub-splits) and an exact zero-harm identity floor (`identity_floor_active_ = True`).
   - Admitted GLM offset heads enforce $p_{\max} = \min(6, \lfloor N_{\text{eff}}/20 \rfloor)$, $\text{VIF} \le 5.0$, smooth-$L_1 + L_2$ penalties (`l1=0.25, l2=6.0`), coefficient box bounds `[-0.45, 0.45]`, and a structural lock on ZINB zero-inflation $\pi_{\text{zero}}(t)$. Admitted GBDT heads enforce `max_depth=2, num_leaves=4, min_child_samples=max(12, 0.08*N_eff), colsample_bytree=0.5, reg_alpha=2.0, reg_lambda=10.0`.
3. **Walk-Forward Ablation Against NB2 & LightGBM Baselines (`Strong`):**
   - Benchmarked across multi-fold expanding windows (`origins = [78, 104, 130]`, `horizon = 26w`) at both whole-college and departmental grains against exposure-offset `NB2GovernanceGLM`, `ZINBSparseRegimeModel`, and `TweedieLightGBMModel` (`init_score = log(active_enrolments) + log_base_rate_`).

---

## 7. Four Residual Code-Level Vulnerabilities & Recommended Hardening

1. **Multiple-Testing Selection Bias in `FULL_13_GATED` at Small $N$ (`residual_stacker.py`, lines 398–488):**
   - *Issue:* On Fold `78`, `GreedyOneSEModuleSelector` uses $K_{\text{inner}} = 3$ expanding splits with `val_len = round(0.15 * 78) = 12` calendar weeks (~8–9 active weeks). With $df = 2$ across 3 splits, the sample standard error `se_d` has high downward variance and `all(diffs > 0)` occurs by chance at $(1/2)^3 = 12.5\%$ per surviving module—allowing `f05` to slip through on `NB2`/`ZINB` at $N=78$.
   - *Fix:* Scale the 1-SE multiplier by the log number of candidate modules (e.g., Bonferroni/ Romano-Wolf multiplicity factor $1.15 \sqrt{1 + \ln M_{\text{cand}}}$) or require $N_{\text{eff}} \ge 100$ before unlocking `FULL_13_GATED` beyond `STAGED_CORE_4`.
2. **Weekly `N_eff` Undercounts Holiday/Low-Activity Closures (`residual_stacker.py`, lines 91–112):**
   - *Issue:* `compute_effective_temporal_sample_size` filters non-instructional rows only when `"is_instructional_day"` is present. Weekly panels (`cw_df` in `backend/app/features.py`) encode the 20 closure/reading/holiday weeks per year via `"is_low_activity_week == 1"`, causing weekly runs to treat $N_{\text{eff}} = 52\text{/yr}$ instead of $32\text{–}39\text{/yr}$ and over-allocating `p_max = floor(N_eff / 20)`.
   - *Fix:* Exclude rows where `is_low_activity_week > 0` inside `compute_effective_temporal_sample_size` when `is_instructional_day` is absent.
3. **Un-Gated In-Sample Dispersion Fit in `FisherRaoResidualCalibrator` (`residual_stacker.py`, lines 806–880):**
   - *Issue:* Unlike `GBDTInitScoreResidualHead` (which deactivates itself if a 20% chronological validation split fails to reduce NB deviance), `FisherRaoResidualCalibrator` fits 12 depth-2 natural-gradient trees on up to 8 columns across 100% of training rows with no out-of-time CRPS/log-likelihood gate—causing the `94.87%` $\rightarrow$ `92.31%` `PICP_95` narrowing on Fold `78`.
   - *Fix:* Gate `FisherRaoResidualCalibrator.is_active_` on strict inner-holdout NB log-likelihood / CRPS improvement.
4. **In-Sample `mu_base_tr` Residual Optimism & Legacy `conditioning.py` Caps (`residual_stacker.py`, lines 995–999; `conditioning.py`, lines 279–401):**
   - *Issue:* `PostHocResidualStacker.fit` computes baseline residuals from in-sample `frozen_adapter.predict(train_df)` rather than expanding out-of-fold predictions (compressing GBDT training residuals), while `GBDTConditioningFunnel` in `conditioning.py` still defaults to `max_esoteric_cols = 18` with unblocked row permutation (`delta >= -1e-4`).
   - *Fix:* Supply out-of-fold baseline `mu_base` predictions when stacking onto non-parametric GBDT baselines, and deprecate or route `conditioning.py` through `PostHocResidualStacker`.

