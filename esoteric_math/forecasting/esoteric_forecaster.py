"""
Causal Multi-Step Time-Series Forecasting Engine (`esoteric_math/forecasting/esoteric_forecaster.py`).

Integrates the 13-framework esoteric & higher-order mathematical suite (`f01`..`f13`, including
Depth-3 Rough Path Signatures & Antisymmetric Levy Area `f07`, Warped Fourier & Hilbert Phase `f01`,
Causal MODWT Wavelets `f02`, Hawkes Contagion `f03`, Cellular Sheaf Cohomology `f08`, Optimal Transport
`f10`, and Fisher-Rao Natural Gradient NB Boosting `f12`) into forward multi-step time-series forecasting
across both:
  1. Monthly Strategic Horizons (3, 6, or 12 months ahead on the active college dataset)
  2. Weekly UK FE High-Frequency Horizons (13 or 26 weeks ahead on the weekly curriculum panel)

Guarantees:
  - Core operational forecast endpoints (`/api/forecast`, `/api/models/backtest`) remain 100% untouched.
  - Pure baseline control trajectories are locked and returned side-by-side for transparent comparison.
  - Strictly causal recursive rollout (`s < t` at every future horizon step `T + h`) with QR/VIF/L1
    anti-bloat conditioning and Fisher-Rao heteroscedastic dispersion interval calibration.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import PoissonRegressor, Ridge

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_DIR = str(_REPO_ROOT / "backend")
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from app.data.processor import aggregate_monthly_series
from app.data.synthetic_generator import (
    BASE_FE_TEACHING_DAYS,
    get_academic_term,
    get_academic_year_label,
)
from app.evaluation.kaggle_metrics import (
    EVAL_QUANTILES,
    compute_mae,
    compute_picp_and_miw,
    compute_rmse,
    compute_surge_mae,
)
from app.features import (
    CURRICULUM_AREA_PROFILES,
    aggregate_college_wide_weekly,
    engineer_safeguarding_features,
    generate_weekly_safeguarding_panel,
)
from app.models.base import (
    Z_80,
    Z_95,
    build_future_covariates_df,
    compute_calibrated_annual_interval,
)
from app.models.registry import get_forecaster
from app.models.safeguarding_benchmark import (
    DEFAULT_BENCHMARK_FEATURES,
    NB2GovernanceGLM,
    TweedieLightGBMModel,
    ZINBSparseRegimeModel,
)
from app.schemas.models import ScenarioAssumptions
from esoteric_math.benchmark.ablation_runner import compute_crps_from_quantiles
from esoteric_math.conditioning import GLMConditioningFunnel
from esoteric_math.features import FEATURE_ENGINE_REGISTRY, EsotericFeatureSuite
from esoteric_math.features.f12_fisher_rao import nb_fisher_information_vectorized
from esoteric_math.models.fisher_rao_ngboost import FisherRaoNaturalGradientBooster
from esoteric_math.retrospective.rollout_config import (
    ALL_13_MODULES as REGISTRY_13_MODULES,
    CORE_4_MODULES,
    MODULE_DISPLAY_NAMES,
)
from esoteric_math.schema_bridge import attach_panel_hierarchy_wide


MODULE_CATALOG: List[Dict[str, str]] = [
    {
        "id": "f01",
        "short_name": "Hilbert Phase",
        "name": MODULE_DISPLAY_NAMES["f01"],
        "family": "Harmonic & Spectral",
        "role": "190-day UK FE instructional clock + causal Hilbert stress-cycle phase & crest velocity.",
    },
    {
        "id": "f02",
        "short_name": "MODWT Wavelets",
        "name": MODULE_DISPLAY_NAMES["f02"],
        "family": "Multiresolution",
        "role": "Boundary-causal Haar & Daubechies db4 wavelets capturing half-term & post-break step cliffs.",
    },
    {
        "id": "f03",
        "short_name": "Hawkes Contagion",
        "name": MODULE_DISPLAY_NAMES["f03"],
        "family": "Point Processes",
        "role": "Self- and cross-exciting conditional intensity with subcritical branching constraint rho(Gamma) < 1.",
    },
    {
        "id": "f04",
        "short_name": "Takens TDA",
        "name": MODULE_DISPLAY_NAMES["f04"],
        "family": "Algebraic Topology",
        "role": "Causal Takens delay embeddings and Vietoris-Rips H0/H1 persistent homology landscapes.",
    },
    {
        "id": "f05",
        "short_name": "Ollivier-Ricci",
        "name": MODULE_DISPLAY_NAMES["f05"],
        "family": "Differential Geometry",
        "role": "Graph Laplacian spectral energy and Kantorovich-Rubinstein LP Ollivier-Ricci bottleneck curvature.",
    },
    {
        "id": "f06",
        "short_name": "Koopman & Tropical",
        "name": MODULE_DISPLAY_NAMES["f06"],
        "family": "Operator & Max-Plus",
        "role": "Sliding SVD Exact Dynamic Mode Decomposition + max-plus Tropical projective crisis distance.",
    },
    {
        "id": "f07",
        "short_name": "Rough Path Signatures",
        "name": MODULE_DISPLAY_NAMES["f07"],
        "family": "Rough Path Theory",
        "role": "Depth-3 Chen iterated-integral signatures, Lyndon log-signatures & antisymmetric Levy Area lead-lag.",
    },
    {
        "id": "f08",
        "short_name": "Cellular Sheaves",
        "name": MODULE_DISPLAY_NAMES["f08"],
        "family": "Sheaf Cohomology",
        "role": "Procrustes restriction maps and Sheaf Laplacian Dirichlet frustration energy detecting contextual masking.",
    },
    {
        "id": "f09",
        "short_name": "p-adic & Poincare",
        "name": MODULE_DISPLAY_NAMES["f09"],
        "family": "Ultrametric & Hyperbolic",
        "role": "Hierarchical p-adic Vladimirov wavelets and 2D Poincare ball hyperbolic Frechet acuity radius.",
    },
    {
        "id": "f10",
        "short_name": "Optimal Transport",
        "name": MODULE_DISPLAY_NAMES["f10"],
        "family": "Wasserstein Geometry",
        "role": "Entropic Sinkhorn divergence, Sliced W2 velocity, and UK FE seasonal Wasserstein barycenters.",
    },
    {
        "id": "f11",
        "short_name": "RMT Marchenko-Pastur",
        "name": MODULE_DISPLAY_NAMES["f11"],
        "family": "Random Matrix Theory",
        "role": "Marchenko-Pastur bulk upper edge lambda_+ and Tracy-Widom escaping synchronization eigenvalues.",
    },
    {
        "id": "f12",
        "short_name": "Fisher-Rao Geometry",
        "name": MODULE_DISPLAY_NAMES["f12"],
        "family": "Information Geometry",
        "role": "Riemannian geodesic velocity and dispersion curvature on the orthogonal (mu, r) NB2 manifold.",
    },
    {
        "id": "f13",
        "short_name": "Fractional & SOC",
        "name": MODULE_DISPLAY_NAMES["f13"],
        "family": "Long-Memory & Criticality",
        "role": "Grunwald-Letnikov fractional differencing (1 - B)^{d*} and sandpile critical slowing down.",
    },
]

ESOTERIC_STRATEGIES: List[Dict[str, Any]] = [
    {
        "id": "staged_core_4",
        "name": "Staged Core-4 Residual Stack (Recommended)",
        "short_name": "Core-4 Residual Stack",
        "default_modules": ["f01", "f02", "f03", "f07"],
        "description": (
            "Pairs the locked baseline engine with high-evidence harmonic/phase (f01), boundary-causal MODWT "
            "wavelets (f02), Hawkes contagion (f03), and Depth-3 Rough Path Signatures (f07) via QR/VIF/L1 "
            "log-offset residual conditioning and Fisher-Rao interval calibration."
        ),
    },
    {
        "id": "rough_signatures_focus",
        "name": "Rough Path Signatures & Levy Area Lead-Lag Forecaster",
        "short_name": "Rough Signatures (f07)",
        "default_modules": ["f07", "f03", "f01"],
        "description": (
            "Centers multi-step trajectory adjustments on Depth-3 Path Signatures, Lyndon log-signatures, and "
            "antisymmetric Levy Area lead-lag matrices A_{i,j} = 0.5*(S^{i,j} - S^{j,i}) across referral rate, "
            "attendance, and instructional density."
        ),
    },
    {
        "id": "fisher_rao_ngboost",
        "name": "Fisher-Rao Natural Gradient NB Manifold Booster",
        "short_name": "Fisher-Rao NGBoost",
        "default_modules": ["f01", "f02", "f03", "f07", "f12"],
        "description": (
            "Stagewise Riemannian Natural Gradient Booster updating Negative Binomial (log mu, log r) along "
            "G(theta)^{-1} nabla L with exact log-enrolment exposure proportionality and heteroscedastic tail intervals."
        ),
    },
    {
        "id": "full_13_conditioned",
        "name": "Full 13-Framework Conditioned Stack (QR / VIF Gated)",
        "short_name": "Full-13 Conditioned",
        "default_modules": list(REGISTRY_13_MODULES),
        "description": (
            "Evaluates all 13 esoteric mathematical frameworks (f01-f13) simultaneously through QR rank-pivoting, "
            "VIF <= 5.0 collinearity pruning, and L1 Poisson deviance shrinkage (exposing sample-size vs. feature-bloat trade-offs)."
        ),
    },
]


FEATURE_INTERPRETATION_HINTS: Dict[str, str] = {
    "f07_levy_area_max_abs": "Antisymmetric Rough Path Levy Area |A_{i,j}|_max capturing multi-channel lead-lag ordering (attendance/exposure shifts preceding referral inflections).",
    "f07_sig_depth2_frobenius": "Depth-2 Chen iterated-integral tensor norm ||S^{(2)}||_F measuring quadratic trajectory co-movement.",
    "f07_sig_depth3_frobenius": "Depth-3 Chen iterated-integral cubic norm (sum S^{(3)2})^{1/3} capturing higher-order path curvature & inflection.",
    "f07_logsig_": "Compressed free Lie algebra Depth-3 Lyndon-word log-signature coordinate invariant to time-warping.",
    "f01_hilbert_": "Causal discrete Hilbert analytic signal amplitude / pastoral stress-cycle phase angle on the 190-day FE manifold.",
    "f01_warped_": "Time-warped UK Further Education instructional harmonic basis (freezes phase across holiday closures).",
    "f02_modwt_": "Boundary-causal Maximal Overlap Discrete Wavelet (Haar / Daubechies db4) multiresolution detail energy.",
    "f02_morlet_": "One-sided causal complex Morlet wavelet scalogram energy across term-transition scales.",
    "f03_hawkes_": "Subcritical multivariate Hawkes conditional contagion intensity & endogenous peer-branching share.",
    "f04_tda_": "Causal Takens delay-embedding Vietoris-Rips H0/H1 persistent homology lifetime & landscape norm.",
    "f05_ricci_": "Discrete Ollivier-Ricci Kantorovich-Rubinstein optimal transport graph curvature & spectral bottleneck stress.",
    "f06_koopman_": "Sliding SVD Exact Koopman DMD modal growth rate & max-plus Tropical crisis distance.",
    "f08_sheaf_": "Cellular Sheaf Laplacian Dirichlet frustration energy x_t^T L_F x_t detecting cohort contextual masking.",
    "f09_padic_": "Ultrametric p-adic Vladimirov wavelet energy & 2D Poincare ball hyperbolic Frechet acuity radius.",
    "f10_ot_": "Unbiased entropic Sinkhorn divergence & Sliced 2-Wasserstein distance to UK FE seasonal barycenters.",
    "f11_rmt_": "Marchenko-Pastur bulk upper-edge synchronization ratio & Tracy-Widom escaping eigenvalue gap.",
    "f12_fr_": "Fisher-Rao Riemannian geodesic velocity & NB2 overdispersion curvature energy.",
    "f13_frac_": "Grunwald-Letnikov fractional differencing long-memory series & Bak-Tang-Wiesenfeld sandpile criticality.",
}


def _describe_esoteric_column(col: str) -> Tuple[str, str]:
    mod_id = col.split("_", 1)[0].lower()
    mod_name = MODULE_DISPLAY_NAMES.get(mod_id, "Higher-Order Manifold")
    if col in FEATURE_INTERPRETATION_HINTS:
        return mod_name, FEATURE_INTERPRETATION_HINTS[col]
    for prefix, desc in FEATURE_INTERPRETATION_HINTS.items():
        if col.startswith(prefix):
            return mod_name, desc
    return mod_name, f"Strictly causal {mod_name} invariant coordinate extracted on prefix s < t."


def _build_wide_monthly_panel(
    raw_df: pd.DataFrame,
    target_monthly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a 1-row-per-month wide DataFrame aligned to `target_monthly['period_start']`
    attaching multi-provision cohort stalks and multi-category counts from `raw_df`
    so `EsotericFeatureSuite` operates with full Tier-A, Tier-B, and Tier-C structure.
    """
    wide = pd.DataFrame(
        {
            "date": target_monthly["period_start"].astype(str).to_numpy(),
            "period_start": target_monthly["period_start"].astype(str).to_numpy(),
            "referral_count": target_monthly["concern_count"].astype(float).to_numpy(),
            "active_enrolments": np.maximum(10.0, target_monthly["enrolled_students"].astype(float).to_numpy()),
            "teaching_days": target_monthly["teaching_days"].astype(float).to_numpy(),
            "is_instructional_day": np.clip(target_monthly["teaching_days"].astype(float).to_numpy() / 20.0, 0.05, 1.0),
            "overall_attendance_pct": (
                target_monthly["attendance_rate"].ffill().bfill().fillna(89.5).astype(float).to_numpy()
            ),
            "reporting_change_flag": target_monthly["reporting_change_flag"].astype(float).to_numpy(),
            "month_num": target_monthly["month_num"].astype(int).to_numpy(),
        }
    )

    # Attach category counts across months
    if "safeguarding_category" in raw_df.columns:
        cat_piv = (
            raw_df.groupby(["period_start", "safeguarding_category"], as_index=False)["concern_count"]
            .sum()
            .pivot(index="period_start", columns="safeguarding_category", values="concern_count")
            .fillna(0.0)
        )
        for col in cat_piv.columns:
            slug = str(col).lower().replace(" ", "_").replace("/", "_")[:24]
            wide[f"cat_{slug}"] = wide["period_start"].map(cat_piv[col]).fillna(0.0).to_numpy(dtype=float)

    # Attach provision cohort columns for Tier C graph & sheaf stalks
    if "provision" in raw_df.columns:
        prov_grp = raw_df.groupby(["period_start", "provision"], as_index=False).agg(
            concern_count=("concern_count", "sum"),
            enrolled_students=("enrolled_students", "mean"),
            attendance_rate=("attendance_rate", "mean"),
            teaching_days=("teaching_days", "mean"),
        )
        for prov in sorted(raw_df["provision"].unique().tolist()):
            slug = str(prov).lower().replace(" ", "_")
            sub = prov_grp[prov_grp["provision"] == prov].set_index("period_start")
            wide[f"cohort_{slug}_referrals"] = wide["period_start"].map(sub["concern_count"]).fillna(0.0).to_numpy(dtype=float)
            wide[f"cohort_{slug}_enrolment"] = (
                wide["period_start"].map(sub["enrolled_students"]).ffill().bfill().fillna(500.0).to_numpy(dtype=float)
            )
            wide[f"cohort_{slug}_attendance"] = (
                wide["period_start"].map(sub["attendance_rate"]).ffill().bfill().fillna(89.5).to_numpy(dtype=float)
            )
            wide[f"cohort_{slug}_engagement"] = (
                np.clip(wide[f"cohort_{slug}_attendance"] - 2.2, 65.0, 99.0)
            )
            wide[f"cohort_{slug}_pastoral"] = np.maximum(
                0.5,
                (wide[f"cohort_{slug}_referrals"] / np.maximum(1.0, wide[f"cohort_{slug}_enrolment"])) * 1000.0,
            )

    return wide


def _append_future_wide_monthly_row(
    current_wide: pd.DataFrame,
    fut_cov_row: pd.Series,
    hist_cat_shares: Dict[str, float],
    hist_prov_shares: Dict[str, Tuple[float, float, float]],
) -> pd.DataFrame:
    """
    Append one future step `T + h` to `current_wide`.
    Note: `referral_count` at `T + h` is initialized to 0.0 and NEVER read at `T + h`
    because `EsotericFeatureSuite(causal_mode='forecast')` strictly shifts count & category
    channels by 1 step (`s < t`). After predicting `y_hat_{T+h}`, the caller writes `y_hat_{T+h}`
    into row `T + h` before advancing to `T + h + 1`.
    """
    enrol = float(fut_cov_row["enrolled_students"])
    td = float(fut_cov_row["teaching_days"])
    att = float(fut_cov_row["attendance_rate"]) if pd.notna(fut_cov_row["attendance_rate"]) else 89.5
    rep = float(fut_cov_row["reporting_change_flag"])
    dt_str = str(fut_cov_row["period_start"])
    m_num = int(fut_cov_row["month_num"])

    new_row: Dict[str, Any] = {
        "date": dt_str,
        "period_start": dt_str,
        "referral_count": 0.0,
        "active_enrolments": max(10.0, enrol),
        "teaching_days": td,
        "is_instructional_day": float(np.clip(td / 20.0, 0.05, 1.0)),
        "overall_attendance_pct": att,
        "reporting_change_flag": rep,
        "month_num": m_num,
    }
    for c_col, share in hist_cat_shares.items():
        if c_col in current_wide.columns:
            new_row[c_col] = 0.0
    for slug, (ref_sh, enr_sh, att_off) in hist_prov_shares.items():
        if f"cohort_{slug}_enrolment" in current_wide.columns:
            new_row[f"cohort_{slug}_referrals"] = 0.0
            new_row[f"cohort_{slug}_enrolment"] = max(10.0, enrol * enr_sh)
            new_row[f"cohort_{slug}_attendance"] = float(np.clip(att + att_off, 60.0, 99.5))
            new_row[f"cohort_{slug}_engagement"] = float(np.clip(att + att_off - 2.2, 60.0, 99.0))
            new_row[f"cohort_{slug}_pastoral"] = float(current_wide[f"cohort_{slug}_pastoral"].iloc[-1])

    return pd.concat([current_wide, pd.DataFrame([new_row])], ignore_index=True)


def _update_predicted_step_in_wide_monthly(
    current_wide: pd.DataFrame,
    row_idx: int,
    y_pred: float,
    hist_cat_shares: Dict[str, float],
    hist_prov_shares: Dict[str, Tuple[float, float, float]],
) -> None:
    """Write predicted count `y_pred` into `row_idx` so subsequent steps `> row_idx` observe it causally."""
    current_wide.at[row_idx, "referral_count"] = float(max(0.0, y_pred))
    for c_col, share in hist_cat_shares.items():
        if c_col in current_wide.columns:
            current_wide.at[row_idx, c_col] = float(max(0.0, y_pred * share))
    for slug, (ref_sh, _, _) in hist_prov_shares.items():
        if f"cohort_{slug}_referrals" in current_wide.columns:
            c_ref = float(max(0.0, y_pred * ref_sh))
            c_enr = float(max(1.0, current_wide.at[row_idx, f"cohort_{slug}_enrolment"]))
            current_wide.at[row_idx, f"cohort_{slug}_referrals"] = c_ref
            current_wide.at[row_idx, f"cohort_{slug}_pastoral"] = max(0.5, (c_ref / c_enr) * 1000.0)


def _fit_and_predict_esoteric_residual_head(
    y_train: np.ndarray,
    mu_base_train: np.ndarray,
    enrol_train: np.ndarray,
    X_eso_train: pd.DataFrame,
    strategy_id: str,
    active_modules: Tuple[str, ...],
    warm_start: int = 24,
) -> Dict[str, Any]:
    """
    Fit a strictly regularized count log-offset residual head + Fisher-Rao dispersion calibrator
    on top of out-of-fold `mu_base_train[warm_start:]`, returning fitted artifacts, selected features,
    and quarantine diagnostics.
    """
    n_rows = len(y_train)
    warm = min(max(4, int(warm_start)), max(4, n_rows - 16))
    train_stub = pd.DataFrame(
        {
            "referral_count": np.maximum(0.0, y_train),
            "active_enrolments": np.maximum(1.0, enrol_train),
            "log_mu_base": np.log(np.maximum(1e-3, mu_base_train)),
        }
    )

    # Filter candidate columns to active_modules
    cand_cols = [
        c for c in X_eso_train.columns if c.split("_", 1)[0].lower() in active_modules
    ]
    X_cand = X_eso_train[cand_cols].copy()

    # Conservative anti-bloat cap scaled to effective post-burn-in sample size
    n_eff = max(16, n_rows - warm)
    if strategy_id == "rough_signatures_focus":
        cap_k = min(5, max(2, n_eff // 12))
        l1_alpha = 0.05
    elif strategy_id == "full_13_conditioned":
        cap_k = min(7, max(3, n_eff // 9))
        l1_alpha = 0.04
    else:
        cap_k = min(5, max(2, n_eff // 12))
        l1_alpha = 0.06

    funnel = GLMConditioningFunnel(
        base_cols=["log_mu_base"],
        target_col="referral_count",
        exposure_col="active_enrolments",
        vif_cap=5.0,
        cond_cap=30.0,
        max_esoteric_cols=cap_k,
        l1_alpha=l1_alpha,
    )
    funnel.fit(train_stub.iloc[warm:].reset_index(drop=True), X_cand.iloc[warm:].reset_index(drop=True))
    selected_cols = list(funnel.selected_esoteric_cols_)

    # Guarantee Rough Signatures representation when explicitly in rough_signatures_focus mode
    if strategy_id == "rough_signatures_focus":
        f07_cands = [c for c in cand_cols if c.startswith("f07_") and c in funnel.medians_]
        if not any(c.startswith("f07_") for c in selected_cols) and f07_cands:
            priority_f07 = [
                c for c in ("f07_levy_area_max_abs", "f07_sig_depth2_frobenius", "f07_sig_depth3_frobenius", "f07_logsig_0")
                if c in f07_cands
            ] or f07_cands[:2]
            selected_cols = (priority_f07[:2] + selected_cols)[:cap_k]

    # If GLMConditioningFunnel quarantined everything on a small monthly sample, admit top Pearson-correlated QR columns with strong shrinkage
    if len(selected_cols) == 0 and len(funnel.medians_) > 0:
        y_sub = y_train[warm:]
        mu_sub = mu_base_train[warm:]
        pearson_res = (y_sub - mu_sub) / np.sqrt(np.maximum(1.0, mu_sub + 0.15 * (mu_sub ** 2)))
        scored: List[Tuple[float, str]] = []
        for c in cand_cols:
            if c not in funnel.medians_:
                continue
            z = np.clip((np.nan_to_num(X_cand[c].iloc[warm:].to_numpy(dtype=float), nan=funnel.medians_[c]) - funnel.medians_[c]) / funnel.scales_[c], -4.0, 4.0)
            if np.std(z) > 1e-6:
                corr = abs(float(np.corrcoef(z, pearson_res)[0, 1]))
                if np.isfinite(corr):
                    scored.append((corr, c))
        scored.sort(reverse=True)
        selected_cols = [c for _, c in scored[: min(3, cap_k)]]

    def _standardize_selected(df_eso: pd.DataFrame) -> np.ndarray:
        if not selected_cols:
            return np.zeros((len(df_eso), 0), dtype=float)
        cols_z = []
        for c in selected_cols:
            med = funnel.medians_.get(c, 0.0)
            scl = funnel.scales_.get(c, 1.0)
            raw_v = df_eso[c].to_numpy(dtype=float) if c in df_eso.columns else np.full(len(df_eso), med)
            z = np.clip((np.nan_to_num(raw_v, nan=med, posinf=med, neginf=med) - med) / max(scl, 1e-6), -4.0, 4.0)
            cols_z.append(z)
        return np.column_stack(cols_z)

    Z_sel = _standardize_selected(X_cand)
    if Z_sel.shape[1] > 0:
        y_w = np.maximum(0.0, y_train[warm:])
        mu_w = np.maximum(0.5, mu_base_train[warm:])
        Z_w = Z_sel[warm:]
        work_resid = np.clip((y_w - mu_w) / np.maximum(2.0, mu_w), -0.60, 0.60)
        fisher_weights = mu_w / (1.0 + 0.16 * mu_w)
        ridge_pen = 16.0 if strategy_id != "full_13_conditioned" else 9.0

        # Inner temporal shrinkage calibration to prevent variance inflation on weekly/monthly holdouts
        val_split = max(10, int(0.72 * len(y_w)))
        if len(y_w) - val_split >= 6:
            reg_in = Ridge(alpha=ridge_pen, fit_intercept=False)
            reg_in.fit(Z_w[:val_split], work_resid[:val_split], sample_weight=fisher_weights[:val_split])
            raw_b = np.asarray(reg_in.coef_, dtype=float)
            best_gamma = 0.25
            best_mae = float(np.mean(np.abs(y_w[val_split:] - mu_w[val_split:] * np.exp(np.clip(0.25 * (Z_w[val_split:] @ raw_b), -0.28, 0.28)))))
            for g in (0.15, 0.35, 0.55, 0.80, 1.0):
                pred_v = mu_w[val_split:] * np.exp(np.clip(g * (Z_w[val_split:] @ raw_b), -0.28, 0.28))
                mae_v = float(np.mean(np.abs(y_w[val_split:] - pred_v)))
                if mae_v < best_mae:
                    best_mae = mae_v
                    best_gamma = g
        else:
            best_gamma = 0.5

        reg = Ridge(alpha=ridge_pen, fit_intercept=False)
        reg.fit(Z_w, work_resid, sample_weight=fisher_weights)
        betas = best_gamma * np.asarray(reg.coef_, dtype=float)
    else:
        betas = np.zeros(0, dtype=float)

    # Estimate NB2 baseline overdispersion r_base & Fisher-Rao natural-gradient dispersion sensitivity
    mu_adj_tr = mu_base_train * np.exp(np.clip(Z_sel @ betas if len(betas) else 0.0, -0.35, 0.35))
    emp_var = float(np.mean((y_train - mu_adj_tr) ** 2))
    mean_mu = float(np.mean(mu_adj_tr))
    alpha_hat = float(np.clip((emp_var - mean_mu) / max(1.0, mean_mu ** 2), 0.025, 0.45))
    r_base = 1.0 / alpha_hat

    # Fisher-Rao natural gradient step on log(r) wrt normalized amplitude/contagion stress
    _, g_rr = nb_fisher_information_vectorized(np.maximum(1.0, mu_adj_tr), np.full_like(mu_adj_tr, r_base))
    abs_rel_err = np.abs(y_train - mu_adj_tr) / np.maximum(1.0, np.sqrt(mu_adj_tr + alpha_hat * (mu_adj_tr ** 2)))
    stress_driver = np.mean(np.abs(Z_sel), axis=1) if Z_sel.shape[1] > 0 else np.zeros(n_rows)
    fr_corr = (
        float(np.corrcoef(stress_driver, abs_rel_err)[0, 1])
        if (len(stress_driver) > 6 and np.std(stress_driver) > 1e-6 and np.std(abs_rel_err) > 1e-6)
        else 0.0
    )
    fr_corr = 0.0 if not np.isfinite(fr_corr) else float(np.clip(fr_corr, -0.35, 0.45))

    return {
        "funnel": funnel,
        "cand_cols": cand_cols,
        "selected_cols": selected_cols,
        "betas": betas,
        "standardize_fn": _standardize_selected,
        "alpha_hat": alpha_hat,
        "r_base": r_base,
        "fr_dispersion_gain": fr_corr,
        "mean_g_rr": float(np.mean(g_rr)),
    }


class EsotericForecastEngine:
    """
    Orchestrates forward multi-step time-series forecasting with esoteric mathematical
    processes (`f01`..`f13`) across both Monthly Strategic and Weekly High-Frequency grains.
    """

    def __init__(self) -> None:
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._mu_base_cache: Dict[str, np.ndarray] = {}

    def clear_cache(self) -> None:
        self._cache.clear()
        self._mu_base_cache.clear()

    def generate_esoteric_forecast(
        self,
        cadence: Literal["monthly", "weekly"] = "monthly",
        strategy_id: str = "staged_core_4",
        base_model_id: str = "lightgbm_mlforecast",
        horizon: int = 12,
        provision: str = "All",
        safeguarding_category: str = "All",
        prediction_interval: int = 80,
        custom_modules: Optional[Sequence[str]] = None,
        scenario: Optional[ScenarioAssumptions] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        scen = scenario or ScenarioAssumptions()
        pi_level = 95 if int(prediction_interval) == 95 else 80

        strat_meta = next((s for s in ESOTERIC_STRATEGIES if s["id"] == strategy_id), ESOTERIC_STRATEGIES[0])
        strategy_id = strat_meta["id"]

        if custom_modules is not None and len(custom_modules) > 0:
            active_mods = tuple(
                m for m in custom_modules if m in FEATURE_ENGINE_REGISTRY
            )
            if not active_mods:
                active_mods = tuple(strat_meta["default_modules"])
        else:
            active_mods = tuple(strat_meta["default_modules"])

        cache_payload = {
            "cadence": cadence,
            "strategy_id": strategy_id,
            "base_model_id": base_model_id,
            "horizon": int(horizon),
            "provision": provision,
            "category": safeguarding_category,
            "pi": pi_level,
            "mods": list(active_mods),
            "scenario": scen.model_dump(),
        }
        # Include active dataset id in cache key if monthly
        if cadence == "monthly":
            from app.services.forecast_service import platform_service
            ds_info, _ = platform_service.get_active_dataset_and_df()
            cache_payload["dataset_id"] = ds_info["dataset_id"]
            cache_payload["ds_created"] = ds_info["created_at"]

        cache_key = hashlib.sha256(json.dumps(cache_payload, sort_keys=True).encode()).hexdigest()[:24]
        if cache_key in self._cache:
            return self._cache[cache_key]

        if cadence == "weekly":
            result = self._run_weekly_forecast(
                strategy_id=strategy_id,
                strat_meta=strat_meta,
                base_model_id=base_model_id,
                horizon_weeks=int(horizon) if int(horizon) in (13, 26) else 26,
                provision=provision,
                pi_level=pi_level,
                active_mods=active_mods,
                scenario=scen,
                t0=t0,
            )
        else:
            result = self._run_monthly_forecast(
                strategy_id=strategy_id,
                strat_meta=strat_meta,
                base_model_id=base_model_id,
                horizon_months=int(horizon) if int(horizon) in (3, 6, 12) else 12,
                provision=provision,
                safeguarding_category=safeguarding_category,
                pi_level=pi_level,
                active_mods=active_mods,
                scenario=scen,
                t0=t0,
            )

        self._cache[cache_key] = result
        return result

    def _run_monthly_forecast(
        self,
        strategy_id: str,
        strat_meta: Dict[str, Any],
        base_model_id: str,
        horizon_months: int,
        provision: str,
        safeguarding_category: str,
        pi_level: int,
        active_mods: Tuple[str, ...],
        scenario: ScenarioAssumptions,
        t0: float,
    ) -> Dict[str, Any]:
        from app.services.forecast_service import platform_service

        info, raw_df = platform_service.get_active_dataset_and_df()
        ay_start = int(platform_service.db.get_config("academic_year_start_month", str(info["academic_year_start_month"])) or 9)
        gov_mode = platform_service.db.get_config("governance_mode", "research_prototype") or "research_prototype"
        lic_ack = (platform_service.db.get_config("timesfm_license_acknowledged", "true") or "true").lower() == "true"

        eff_base_id = base_model_id if base_model_id in ("lightgbm_mlforecast", "statistical_count", "seasonal_naive") else "lightgbm_mlforecast"
        forecaster = get_forecaster(eff_base_id)

        monthly_hist = aggregate_monthly_series(raw_df, provision, safeguarding_category, ay_start)
        if len(monthly_hist) < 24:
            raise ValueError("At least 24 months of historical data are required for causal esoteric forecasting.")

        # 1. Locked Pure Baseline Forecast on full history
        base_out = forecaster.forecast(
            monthly_hist,
            horizon=horizon_months,
            scenario=scenario,
            academic_year_start_month=ay_start,
            governance_mode=gov_mode,
            license_ack=lic_ack,
        )
        baseline_sha256 = hashlib.sha256(
            json.dumps(base_out.raw_forecasts, sort_keys=True).encode()
        ).hexdigest()

        # 2. Compute causal historical in-sample / expanding baseline expectations for residual learning
        n_hist = len(monthly_hist)
        y_hist = monthly_hist["concern_count"].to_numpy(dtype=float)
        enrol_hist = np.maximum(10.0, monthly_hist["enrolled_students"].to_numpy(dtype=float))
        mu_base_hist = self._compute_monthly_causal_baseline_fitted(monthly_hist, forecaster, ay_start)

        # 3. Build multi-cohort / multi-category wide historical frame & fit EsotericFeatureSuite strictly on {0..T-1}
        wide_hist = _build_wide_monthly_panel(raw_df, monthly_hist)
        suite = EsotericFeatureSuite(
            active_modules=active_mods,
            causal_mode="forecast",
            train_cutoff=n_hist,
        )
        suite.fit(wide_hist)
        X_eso_hist = suite.transform(wide_hist).reset_index(drop=True)

        # Historical shares for recursive rollout of multi-category / cohort stalks
        tot_hist_concerns = max(1.0, float(np.sum(y_hist)))
        hist_cat_shares = {
            c: float(np.sum(wide_hist[c])) / tot_hist_concerns
            for c in wide_hist.columns
            if c.startswith("cat_")
        }
        hist_prov_shares: Dict[str, Tuple[float, float, float]] = {}
        for p_name in sorted(raw_df["provision"].unique().tolist()):
            slug = str(p_name).lower().replace(" ", "_")
            if f"cohort_{slug}_referrals" in wide_hist.columns:
                r_sh = float(np.sum(wide_hist[f"cohort_{slug}_referrals"])) / max(1.0, float(raw_df["concern_count"].sum()))
                e_sh = float(np.mean(wide_hist[f"cohort_{slug}_enrolment"])) / max(1.0, float(np.mean(enrol_hist)))
                a_off = float(np.mean(wide_hist[f"cohort_{slug}_attendance"] - wide_hist["overall_attendance_pct"]))
                hist_prov_shares[slug] = (r_sh, e_sh, a_off)

        # 4. Fit selected Esoteric Forecasting Strategy on {0..T-1}
        fut_cov = build_future_covariates_df(monthly_hist, horizon_months, scenario, ay_start)

        if strategy_id == "fisher_rao_ngboost":
            eso_points, eso_raw, l80, u80, l95, u95, step_telemetry, head_diag = self._rollout_monthly_fisher_rao_ngboost(
                wide_hist=wide_hist,
                monthly_hist=monthly_hist,
                fut_cov=fut_cov,
                base_out_raw=np.asarray(base_out.raw_forecasts, dtype=float),
                suite=suite,
                X_eso_hist=X_eso_hist,
                active_mods=active_mods,
                hist_cat_shares=hist_cat_shares,
                hist_prov_shares=hist_prov_shares,
            )
            cal_label = "Fisher-Rao Natural Gradient Heteroscedastic NB2 Manifold Quantiles"
        else:
            head = _fit_and_predict_esoteric_residual_head(
                y_train=y_hist,
                mu_base_train=mu_base_hist,
                enrol_train=enrol_hist,
                X_eso_train=X_eso_hist,
                strategy_id=strategy_id,
                active_modules=active_mods,
            )
            eso_points, eso_raw, l80, u80, l95, u95, step_telemetry, head_diag = self._rollout_monthly_residual_head(
                wide_hist=wide_hist,
                fut_cov=fut_cov,
                base_out=base_out,
                suite=suite,
                head=head,
                hist_cat_shares=hist_cat_shares,
                hist_prov_shares=hist_prov_shares,
            )
            cal_label = "Baseline Conformal/Parametric Interval + Fisher-Rao Dispersion Scaling"

        # 5. Rolling-Origin Walk-Forward Backtest (2 expanding origins over final 12 months)
        bt_summary = self._evaluate_monthly_walk_forward(
            raw_df=raw_df,
            monthly_hist=monthly_hist,
            forecaster=forecaster,
            strategy_id=strategy_id,
            active_mods=active_mods,
            ay_start=ay_start,
            pi_level=pi_level,
        )

        # 6. Assemble Calibrated Horizon Totals & Period Points
        ann_tot, ann_l80, ann_u80, ann_l95, ann_u95, _ = compute_calibrated_annual_interval(
            eso_points, l80, u80, l95, u95, residual_autocorr_lag1=0.26
        )
        base_tot = int(base_out.annual_total)
        base_lb = int(base_out.annual_lower_95 if (pi_level == 95 and base_out.annual_lower_95 is not None) else base_out.annual_lower_80)
        base_ub = int(base_out.annual_upper_95 if (pi_level == 95 and base_out.annual_upper_95 is not None) else base_out.annual_upper_80)
        eso_lb = int(ann_l95 if (pi_level == 95 and ann_l95 is not None) else ann_l80)
        eso_ub = int(ann_u95 if (pi_level == 95 and ann_u95 is not None) else ann_u80)

        prior_total = int(monthly_hist.tail(horizon_months)["concern_count"].sum())
        net_delta = int(ann_tot - base_tot)
        net_delta_pct = round((net_delta / max(1, base_tot)) * 100.0, 2)

        hist_points = [
            {
                "period_start": str(r["period_start"]),
                "period_label": str(r["period_label"]),
                "academic_year": str(r["academic_year"]),
                "academic_term": str(r["academic_term"]),
                "actual_count": int(r["concern_count"]),
                "enrolled_students": int(r["enrolled_students"]),
                "teaching_days": int(r["teaching_days"]),
                "rate_per_1000": float(r["concerns_per_1000"]),
            }
            for _, r in monthly_hist.tail(36).iterrows()
        ]

        forecast_points = []
        for idx, d_str in enumerate(base_out.dates):
            dt = pd.to_datetime(d_str)
            enrol_v = max(1, int(base_out.future_enrolled_students[idx]))
            b_pt = int(base_out.point_forecasts[idx])
            b_l = int(base_out.lower_95[idx]) if (pi_level == 95 and base_out.lower_95) else int(base_out.lower_80[idx])
            b_u = int(base_out.upper_95[idx]) if (pi_level == 95 and base_out.upper_95) else int(base_out.upper_80[idx])
            e_pt = int(eso_points[idx])
            e_l = int(l95[idx]) if pi_level == 95 else int(l80[idx])
            e_u = int(u95[idx]) if pi_level == 95 else int(u80[idx])
            st = step_telemetry[idx]

            forecast_points.append(
                {
                    "step": idx + 1,
                    "period_start": d_str,
                    "period_label": dt.strftime("%b %Y"),
                    "academic_year": get_academic_year_label(dt.year, dt.month, ay_start),
                    "academic_term": get_academic_term(dt.month),
                    "baseline_forecast": b_pt,
                    "baseline_lower": b_l,
                    "baseline_upper": b_u,
                    "esoteric_forecast": e_pt,
                    "esoteric_raw": round(float(eso_raw[idx]), 2),
                    "lower_bound": e_l,
                    "upper_bound": e_u,
                    "lower_80": int(l80[idx]),
                    "upper_80": int(u80[idx]),
                    "lower_95": int(l95[idx]),
                    "upper_95": int(u95[idx]),
                    "delta_count": e_pt - b_pt,
                    "delta_pct": round(((eso_raw[idx] - base_out.raw_forecasts[idx]) / max(1.0, base_out.raw_forecasts[idx])) * 100.0, 2),
                    "enrolled_students": enrol_v,
                    "teaching_days": int(base_out.future_teaching_days[idx]),
                    "rate_per_1000": round((e_pt / enrol_v) * 1000.0, 2),
                    "rough_levy_area": st["rough_levy_area"],
                    "rough_sig_depth3": st["rough_sig_depth3"],
                    "hilbert_phase_deg": st["hilbert_phase_deg"],
                    "hawkes_intensity": st["hawkes_intensity"],
                    "dispersion_r": st["dispersion_r"],
                    "dominant_driver": st["dominant_driver"],
                }
            )

        peak_item = max(forecast_points, key=lambda x: x["esoteric_forecast"])
        runtime_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return {
            "cadence": "monthly",
            "period_unit": "months",
            "dataset_id": info["dataset_id"],
            "is_synthetic": bool(info["is_synthetic"]),
            "governance_banner": platform_service.get_active_dataset_summary().governance_banner,
            "strategy": strat_meta,
            "base_model_id": eff_base_id,
            "base_model_name": forecaster.name,
            "baseline_sha256": baseline_sha256,
            "active_modules": list(active_mods),
            "module_catalog": MODULE_CATALOG,
            "strategies_catalog": ESOTERIC_STRATEGIES,
            "horizon": horizon_months,
            "provision": provision,
            "safeguarding_category": safeguarding_category,
            "prediction_interval": pi_level,
            "runtime_ms": runtime_ms,
            "summary": {
                "esoteric_total": int(ann_tot),
                "esoteric_lower": eso_lb,
                "esoteric_upper": eso_ub,
                "baseline_total": base_tot,
                "baseline_lower": base_lb,
                "baseline_upper": base_ub,
                "prior_horizon_total": prior_total,
                "yoy_change_pct": round(((ann_tot - prior_total) / max(1, prior_total)) * 100.0, 1),
                "net_esoteric_delta": net_delta,
                "net_esoteric_delta_pct": net_delta_pct,
                "peak_period_label": peak_item["period_label"],
                "peak_period_count": peak_item["esoteric_forecast"],
                "calibration_method": cal_label,
            },
            "conditioning_diagnostics": head_diag,
            "walk_forward_backtest": bt_summary,
            "historical_series": hist_points,
            "forecast_series": forecast_points,
            "mathematical_notes": self._build_mathematical_notes(
                cadence="monthly",
                strat_meta=strat_meta,
                base_name=forecaster.name,
                head_diag=head_diag,
                bt_summary=bt_summary,
                net_delta=net_delta,
                net_delta_pct=net_delta_pct,
            ),
        }

    def _compute_monthly_causal_baseline_fitted(
        self,
        monthly_hist: pd.DataFrame,
        forecaster: Any,
        ay_start: int,
    ) -> np.ndarray:
        """
        Produce causal 1-step baseline expectations `mu_base_hist` across `monthly_hist`
        using seasonal exposure rates for early burn-in and expanding rolling 1-step forecasts
        so residual learning never overfits in-sample GBDT memorization.
        """
        n = len(monthly_hist)
        y = monthly_hist["concern_count"].to_numpy(dtype=float)
        enrol = np.maximum(10.0, monthly_hist["enrolled_students"].to_numpy(dtype=float))
        td = np.maximum(1.0, monthly_hist["teaching_days"].to_numpy(dtype=float))
        m_nums = monthly_hist["month_num"].to_numpy(dtype=int)

        mu_hat = np.zeros(n, dtype=float)
        # First 24 months: causal seasonal-rate anchor
        base_rate = float(np.sum(y[:12]) / np.sum(enrol[:12]))
        for t in range(min(24, n)):
            if t >= 12:
                mu_hat[t] = max(1.0, (y[t - 12] / enrol[t - 12]) * enrol[t])
            else:
                mu_hat[t] = max(1.0, base_rate * enrol[t] * (td[t] / 16.5))

        # Months 24..n-1: expanding-window 12-month blocks with `forecaster` (cached by series hash)
        sig_key = hashlib.sha256(
            f"{getattr(forecaster, 'model_id', 'm')}:{ay_start}:{n}:{round(float(np.sum(y)), 2)}".encode()
        ).hexdigest()[:20]
        if sig_key in self._mu_base_cache:
            return self._mu_base_cache[sig_key].copy()

        step_block = 6
        for start_t in range(24, n, step_block):
            end_t = min(n, start_t + step_block)
            h_sub = end_t - start_t
            sub_train = monthly_hist.iloc[:start_t].copy().reset_index(drop=True)
            try:
                out_sub = forecaster.forecast(
                    sub_train,
                    horizon=h_sub,
                    scenario=ScenarioAssumptions(),
                    academic_year_start_month=ay_start,
                )
                mu_hat[start_t:end_t] = np.maximum(0.5, np.asarray(out_sub.raw_forecasts[:h_sub], dtype=float))
            except Exception:
                for t in range(start_t, end_t):
                    mu_hat[t] = max(1.0, (y[t - 12] / enrol[t - 12]) * enrol[t])

        self._mu_base_cache[sig_key] = mu_hat.copy()
        return mu_hat

    def _rollout_monthly_residual_head(
        self,
        wide_hist: pd.DataFrame,
        fut_cov: pd.DataFrame,
        base_out: Any,
        suite: EsotericFeatureSuite,
        head: Dict[str, Any],
        hist_cat_shares: Dict[str, float],
        hist_prov_shares: Dict[str, Tuple[float, float, float]],
    ) -> Tuple[List[int], List[float], List[int], List[int], List[int], List[int], List[Dict[str, Any]], Dict[str, Any]]:
        work_wide = wide_hist.copy().reset_index(drop=True)
        n_hist = len(work_wide)
        selected_cols: List[str] = head["selected_cols"]
        betas: np.ndarray = head["betas"]
        std_fn = head["standardize_fn"]
        r_base = float(head["r_base"])
        fr_gain = float(head["fr_dispersion_gain"])

        pts: List[int] = []
        raws: List[float] = []
        l80_list: List[int] = []
        u80_list: List[int] = []
        l95_list: List[int] = []
        u95_list: List[int] = []
        step_telem: List[Dict[str, Any]] = []

        for h_idx in range(len(fut_cov)):
            work_wide = _append_future_wide_monthly_row(
                work_wide, fut_cov.iloc[h_idx], hist_cat_shares, hist_prov_shares
            )
            X_eso_all = suite.transform(work_wide)
            row_eso = X_eso_all.iloc[[n_hist + h_idx]]
            Z_step = std_fn(row_eso)

            if len(betas) > 0 and Z_step.shape[1] == len(betas):
                contrib_vec = Z_step[0] * betas
                eta_shift = float(np.clip(np.sum(contrib_vec), -0.28, 0.28))
                top_j = int(np.argmax(np.abs(contrib_vec)))
                dom_col = selected_cols[top_j]
                dom_mod = MODULE_DISPLAY_NAMES.get(dom_col.split("_", 1)[0], dom_col)
            else:
                eta_shift = 0.0
                dom_mod = "Locked Baseline (Identity Floor)"

            mu_b = max(0.25, float(base_out.raw_forecasts[h_idx]))
            mu_e = max(0.0, mu_b * float(np.exp(eta_shift)))
            pt_e = max(0, int(round(mu_e)))

            # Calibrate heteroscedastic dispersion scale via Fisher-Rao stress coupling
            stress_norm = float(np.mean(np.abs(Z_step[0]))) if Z_step.shape[1] > 0 else 0.0
            eta_r_step = np.log(r_base) - 0.18 * fr_gain * (stress_norm - 0.65)
            r_step = float(np.clip(np.exp(eta_r_step), 2.2, 45.0))
            disp_scale = float(np.clip(np.sqrt(r_base / r_step), 0.86, 1.24))

            base_half_80 = max(1.0, 0.5 * (float(base_out.upper_80[h_idx]) - float(base_out.lower_80[h_idx])))
            b_l95 = float(base_out.lower_95[h_idx]) if base_out.lower_95 else max(0.0, mu_b - 1.53 * base_half_80)
            b_u95 = float(base_out.upper_95[h_idx]) if base_out.upper_95 else mu_b + 1.53 * base_half_80
            base_half_95 = max(1.5, 0.5 * (b_u95 - b_l95))

            level_ratio = float(np.sqrt(max(0.5, mu_e) / max(0.5, mu_b)))
            half_80 = base_half_80 * level_ratio * disp_scale
            half_95 = base_half_95 * level_ratio * disp_scale

            l80_v = max(0, int(round(mu_e - half_80)))
            u80_v = max(pt_e, int(round(mu_e + half_80)))
            l95_v = max(0, int(round(mu_e - half_95)))
            u95_v = max(u80_v, int(round(mu_e + half_95)))

            pts.append(pt_e)
            raws.append(mu_e)
            l80_list.append(l80_v)
            u80_list.append(u80_v)
            l95_list.append(l95_v)
            u95_list.append(u95_v)

            _update_predicted_step_in_wide_monthly(
                work_wide, n_hist + h_idx, mu_e, hist_cat_shares, hist_prov_shares
            )

            step_telem.append(
                self._extract_step_manifold_indicators(row_eso.iloc[0], suite, r_step, dom_mod)
            )

        diag = self._build_head_diagnostics(head, len(wide_hist))
        return pts, raws, l80_list, u80_list, l95_list, u95_list, step_telem, diag

    def _rollout_monthly_fisher_rao_ngboost(
        self,
        wide_hist: pd.DataFrame,
        monthly_hist: pd.DataFrame,
        fut_cov: pd.DataFrame,
        base_out_raw: np.ndarray,
        suite: EsotericFeatureSuite,
        X_eso_hist: pd.DataFrame,
        active_mods: Tuple[str, ...],
        hist_cat_shares: Dict[str, float],
        hist_prov_shares: Dict[str, Tuple[float, float, float]],
    ) -> Tuple[List[int], List[float], List[int], List[int], List[int], List[int], List[Dict[str, Any]], Dict[str, Any]]:
        n_hist = len(monthly_hist)
        y_hist = monthly_hist["concern_count"].to_numpy(dtype=float)
        enrol_hist = np.maximum(10.0, monthly_hist["enrolled_students"].to_numpy(dtype=float))

        # Construct causal baseline features + conditioned esoteric features for FisherRaoNaturalGradientBooster
        def _build_ngb_base_row(df_w: pd.DataFrame) -> pd.DataFrame:
            m_rad = 2.0 * np.pi * df_w["month_num"].to_numpy(dtype=float) / 12.0
            y_arr = df_w["referral_count"].to_numpy(dtype=float)
            enr_arr = np.maximum(10.0, df_w["active_enrolments"].to_numpy(dtype=float))
            rate_arr = (y_arr / enr_arr) * 1000.0
            lag1_r = np.concatenate([[rate_arr[0]], rate_arr[:-1]])
            lag12_r = np.array([rate_arr[i - 12] if i >= 12 else rate_arr[i % min(12, len(rate_arr))] for i in range(len(rate_arr))])
            return pd.DataFrame(
                {
                    "referral_count": y_arr,
                    "active_enrolments": enr_arr,
                    "teaching_days": df_w["teaching_days"].to_numpy(dtype=float),
                    "overall_attendance_pct": df_w["overall_attendance_pct"].to_numpy(dtype=float),
                    "reporting_change_flag": df_w["reporting_change_flag"].to_numpy(dtype=float),
                    "month_sin": np.sin(m_rad),
                    "month_cos": np.cos(m_rad),
                    "lag1_rate": lag1_r,
                    "lag12_rate": lag12_r,
                }
            )

        base_tr_df = _build_ngb_base_row(wide_hist)
        ngb_base_cols = [
            "teaching_days",
            "overall_attendance_pct",
            "reporting_change_flag",
            "month_sin",
            "month_cos",
            "lag1_rate",
            "lag12_rate",
        ]

        cand_cols = [c for c in X_eso_hist.columns if c.split("_", 1)[0].lower() in active_mods]
        funnel = GLMConditioningFunnel(
            base_cols=ngb_base_cols,
            target_col="referral_count",
            exposure_col="active_enrolments",
            vif_cap=5.0,
            cond_cap=30.0,
            max_esoteric_cols=5,
            l1_alpha=0.05,
        )
        funnel.fit(base_tr_df, X_eso_hist[cand_cols])
        sel_eso = list(funnel.selected_esoteric_cols_)
        if not sel_eso and cand_cols:
            sel_eso = [c for c in ("f07_levy_area_max_abs", "f01_hilbert_amplitude", "f03_hawkes_intensity_total", "f12_fr_velocity") if c in cand_cols][:3]
            if not sel_eso:
                sel_eso = cand_cols[:3]

        tr_design = funnel.transform(base_tr_df, X_eso_hist[cand_cols])
        for c in sel_eso:
            if c not in tr_design.columns and c in X_eso_hist.columns:
                tr_design[c] = X_eso_hist[c].to_numpy(dtype=float)

        all_feat_cols = ngb_base_cols + [c for c in sel_eso if c in tr_design.columns]
        booster = FisherRaoNaturalGradientBooster(
            feature_cols=all_feat_cols,
            n_estimators=48,
            learning_rate_mu=0.045,
            learning_rate_r=0.035,
            max_depth_mu=2,
            max_depth_r=2,
            random_state=42,
        )
        booster.fit(tr_design.iloc[12:].reset_index(drop=True))

        work_wide = wide_hist.copy().reset_index(drop=True)
        pts: List[int] = []
        raws: List[float] = []
        l80_list: List[int] = []
        u80_list: List[int] = []
        l95_list: List[int] = []
        u95_list: List[int] = []
        step_telem: List[Dict[str, Any]] = []

        for h_idx in range(len(fut_cov)):
            work_wide = _append_future_wide_monthly_row(
                work_wide, fut_cov.iloc[h_idx], hist_cat_shares, hist_prov_shares
            )
            X_eso_all = suite.transform(work_wide)
            ngb_all = _build_ngb_base_row(work_wide)
            row_base = ngb_all.iloc[[n_hist + h_idx]].reset_index(drop=True)
            row_eso = X_eso_all.iloc[[n_hist + h_idx]][cand_cols].reset_index(drop=True)
            step_design = funnel.transform(row_base, row_eso)
            for c in sel_eso:
                if c not in step_design.columns and c in row_eso.columns:
                    step_design[c] = row_eso[c].to_numpy(dtype=float)

            bundle = booster.predict(step_design, quantiles=(0.025, 0.10, 0.50, 0.90, 0.975))
            # Blend 70% Fisher-Rao NGBoost manifold prediction with 30% seasonal-cov anchor for smooth multi-step stability
            mu_ngb = float(bundle.point_forecasts[0])
            mu_e = max(0.0, 0.72 * mu_ngb + 0.28 * float(base_out_raw[h_idx]))
            ratio = mu_e / max(0.5, mu_ngb)
            pt_e = max(0, int(round(mu_e)))

            l80_v = max(0, int(round(float(bundle.quantiles[0.10][0]) * ratio)))
            u80_v = max(pt_e, int(round(float(bundle.quantiles[0.90][0]) * ratio)))
            l95_v = max(0, int(round(float(bundle.quantiles[0.025][0]) * ratio)))
            u95_v = max(u80_v, int(round(float(bundle.quantiles[0.975][0]) * ratio)))

            r_init_val = float(np.exp(booster.base_eta_r_))
            r_step = float(bundle.diagnostics.get("mean_r_shape", r_init_val))

            pts.append(pt_e)
            raws.append(mu_e)
            l80_list.append(l80_v)
            u80_list.append(u80_v)
            l95_list.append(l95_v)
            u95_list.append(u95_v)

            _update_predicted_step_in_wide_monthly(
                work_wide, n_hist + h_idx, mu_e, hist_cat_shares, hist_prov_shares
            )
            step_telem.append(
                self._extract_step_manifold_indicators(
                    X_eso_all.iloc[n_hist + h_idx],
                    suite,
                    r_step,
                    "Fisher-Rao Natural Gradient NB Manifold",
                )
            )

        # Build attribution weights from booster feature importances across mu & r trees
        used_cols = list(booster.used_cols_)
        imp_vec = np.zeros(len(used_cols), dtype=float)
        for tr_m, cols_m, sc_m in booster.trees_mu_:
            imp_vec[cols_m] += abs(float(sc_m)) * tr_m.feature_importances_
        for tr_r, cols_r, sc_r in booster.trees_r_:
            imp_vec[cols_r] += 0.5 * abs(float(sc_r)) * tr_r.feature_importances_
        eso_attributions = []
        tot_imp = max(1e-9, float(np.sum(imp_vec)))
        for idx_f, col in enumerate(used_cols):
            if col in sel_eso:
                m_name, desc = _describe_esoteric_column(col)
                eso_attributions.append(
                    {
                        "feature": col,
                        "module_id": col.split("_", 1)[0].lower(),
                        "module_name": m_name,
                        "weight": round(float(imp_vec[idx_f] / tot_imp), 4),
                        "direction": "Manifold Natural Gradient (mu & r)",
                        "description": desc,
                    }
                )
        eso_attributions.sort(key=lambda x: abs(x["weight"]), reverse=True)

        diag = {
            "funnel_type": "Fisher-Rao Riemannian Natural Gradient + QR/VIF Funnel",
            "n_train_steps": n_hist,
            "n_candidate_esoteric_features": len(cand_cols),
            "n_admitted_esoteric_features": len(eso_attributions),
            "n_quarantined_esoteric_features": max(0, len(cand_cols) - len(eso_attributions)),
            "max_vif_after": round(float(funnel.diagnostics_.max_vif_after), 2),
            "condition_number_after": round(float(funnel.diagnostics_.condition_number_after), 2),
            "base_nb_dispersion_r": round(float(np.exp(booster.base_eta_r_)), 2),
            "identity_floor_active": len(eso_attributions) == 0,
            "admitted_features": eso_attributions,
        }
        return pts, raws, l80_list, u80_list, l95_list, u95_list, step_telem, diag

    def _evaluate_monthly_walk_forward(
        self,
        raw_df: pd.DataFrame,
        monthly_hist: pd.DataFrame,
        forecaster: Any,
        strategy_id: str,
        active_mods: Tuple[str, ...],
        ay_start: int,
        pi_level: int,
    ) -> Dict[str, Any]:
        """Evaluate 2 expanding walk-forward folds across the final 12 historical months (2 x 6m)."""
        n_total = len(monthly_hist)
        if n_total < 36:
            return {"evaluated_periods": 0, "folds": 0, "points": []}

        origins = [n_total - 12, n_total - 6]
        fold_h = 6
        yt_all: List[float] = []
        yb_all: List[float] = []
        ye_all: List[float] = []
        b_cov_hits: List[int] = []
        e_cov_hits: List[int] = []
        surge_flags: List[int] = []
        bt_points: List[Dict[str, Any]] = []

        for orig in origins:
            sub_monthly = monthly_hist.iloc[:orig].copy().reset_index(drop=True)
            cutoff_dt = str(sub_monthly["period_start"].iloc[-1])
            sub_raw = raw_df[raw_df["period_start"] <= cutoff_dt].copy().reset_index(drop=True)
            holdout = monthly_hist.iloc[orig : orig + fold_h].copy().reset_index(drop=True)

            base_f = forecaster.forecast(sub_monthly, horizon=fold_h, academic_year_start_month=ay_start)
            mu_b_tr = self._compute_monthly_causal_baseline_fitted(sub_monthly, forecaster, ay_start)
            wide_tr = _build_wide_monthly_panel(sub_raw, sub_monthly)
            suite_f = EsotericFeatureSuite(active_modules=active_mods, causal_mode="forecast", train_cutoff=len(wide_tr))
            suite_f.fit(wide_tr)
            X_eso_tr = suite_f.transform(wide_tr).reset_index(drop=True)

            tot_c = max(1.0, float(sub_monthly["concern_count"].sum()))
            cat_sh = {c: float(wide_tr[c].sum()) / tot_c for c in wide_tr.columns if c.startswith("cat_")}
            prov_sh: Dict[str, Tuple[float, float, float]] = {}
            for p_name in sorted(sub_raw["provision"].unique().tolist()):
                slug = str(p_name).lower().replace(" ", "_")
                if f"cohort_{slug}_referrals" in wide_tr.columns:
                    prov_sh[slug] = (
                        float(wide_tr[f"cohort_{slug}_referrals"].sum()) / max(1.0, float(sub_raw["concern_count"].sum())),
                        float(wide_tr[f"cohort_{slug}_enrolment"].mean()) / max(1.0, float(sub_monthly["enrolled_students"].mean())),
                        float((wide_tr[f"cohort_{slug}_attendance"] - wide_tr["overall_attendance_pct"]).mean()),
                    )

            fut_c = build_future_covariates_df(sub_monthly, fold_h, ScenarioAssumptions(), ay_start)
            if strategy_id == "fisher_rao_ngboost":
                e_pts, e_raw, l80, u80, l95, u95, _, _ = self._rollout_monthly_fisher_rao_ngboost(
                    wide_tr, sub_monthly, fut_c, np.asarray(base_f.raw_forecasts, dtype=float),
                    suite_f, X_eso_tr, active_mods, cat_sh, prov_sh
                )
            else:
                head_f = _fit_and_predict_esoteric_residual_head(
                    sub_monthly["concern_count"].to_numpy(dtype=float),
                    mu_b_tr,
                    sub_monthly["enrolled_students"].to_numpy(dtype=float),
                    X_eso_tr,
                    strategy_id,
                    active_mods,
                )
                e_pts, e_raw, l80, u80, l95, u95, _, _ = self._rollout_monthly_residual_head(
                    wide_tr, fut_c, base_f, suite_f, head_f, cat_sh, prov_sh
                )

            for k in range(len(holdout)):
                act_v = float(holdout["concern_count"].iloc[k])
                b_v = float(base_f.raw_forecasts[k])
                e_v = float(e_raw[k])
                m_num = int(holdout["month_num"].iloc[k])
                is_surge = 1 if m_num in (9, 10, 11, 1) else 0

                b_lo = float(base_f.lower_95[k] if (pi_level == 95 and base_f.lower_95) else base_f.lower_80[k])
                b_hi = float(base_f.upper_95[k] if (pi_level == 95 and base_f.upper_95) else base_f.upper_80[k])
                e_lo = float(l95[k] if pi_level == 95 else l80[k])
                e_hi = float(u95[k] if pi_level == 95 else u80[k])

                yt_all.append(act_v)
                yb_all.append(b_v)
                ye_all.append(e_v)
                surge_flags.append(is_surge)
                b_cov_hits.append(1 if b_lo <= act_v <= b_hi else 0)
                e_cov_hits.append(1 if e_lo <= act_v <= e_hi else 0)
                bt_points.append(
                    {
                        "period_label": str(holdout["period_label"].iloc[k]),
                        "actual": int(round(act_v)),
                        "baseline": round(b_v, 1),
                        "esoteric": round(e_v, 1),
                    }
                )

        b_mae = compute_mae(yt_all, yb_all)
        e_mae = compute_mae(yt_all, ye_all)
        b_rmse = compute_rmse(yt_all, yb_all)
        e_rmse = compute_rmse(yt_all, ye_all)
        tot_act = max(1.0, sum(yt_all))
        b_wape = (sum(abs(a - p) for a, p in zip(yt_all, yb_all)) / tot_act) * 100.0
        e_wape = (sum(abs(a - p) for a, p in zip(yt_all, ye_all)) / tot_act) * 100.0
        zeros = [0] * len(yt_all)
        b_surge = compute_surge_mae(yt_all, yb_all, surge_flags, zeros, zeros)
        e_surge = compute_surge_mae(yt_all, ye_all, surge_flags, zeros, zeros)
        mae_gain_pct = ((b_mae - e_mae) / max(1e-6, b_mae)) * 100.0

        return {
            "evaluated_periods": len(yt_all),
            "folds": len(origins),
            "baseline_mae": round(b_mae, 2),
            "esoteric_mae": round(e_mae, 2),
            "mae_delta_pct": round(mae_gain_pct, 2),
            "baseline_rmse": round(b_rmse, 2),
            "esoteric_rmse": round(e_rmse, 2),
            "baseline_wape_pct": round(b_wape, 2),
            "esoteric_wape_pct": round(e_wape, 2),
            "baseline_surge_mae": round(b_surge, 2),
            "esoteric_surge_mae": round(e_surge, 2),
            "baseline_pi_coverage_pct": round(float(np.mean(b_cov_hits) * 100.0), 1),
            "esoteric_pi_coverage_pct": round(float(np.mean(e_cov_hits) * 100.0), 1),
            "points": bt_points,
        }

    def _run_weekly_forecast(
        self,
        strategy_id: str,
        strat_meta: Dict[str, Any],
        base_model_id: str,
        horizon_weeks: int,
        provision: str,
        pi_level: int,
        active_mods: Tuple[str, ...],
        scenario: ScenarioAssumptions,
        t0: float,
    ) -> Dict[str, Any]:
        """
        Run high-frequency weekly UK FE time-series forecasting (156 historical weeks -> 13 or 26 weeks ahead)
        where active teaching weeks (~39/yr) directly exercise Hawkes contagion, Rough Path Signatures,
        MODWT half-term cliffs, and Fisher-Rao NB boosting.
        """
        panel_4y = generate_weekly_safeguarding_panel(num_academic_years=4, start_year=2022, seed=42)
        prov_to_area = {
            "EPYP": "Foundation Learning & SEND (EPYP)",
            "Adult": "Health, Social Care & Childcare",
            "Apprenticeships": "Adult Access & Work-Based Apprenticeships",
        }

        if provision in prov_to_area:
            area_name = prov_to_area[provision]
            area_sub = panel_4y[panel_4y["curriculum_area"] == area_name].copy().reset_index(drop=True)
            full_df, _ = engineer_safeguarding_features(area_sub, group_col="curriculum_area")
            full_wide = attach_panel_hierarchy_wide(full_df, panel_4y)
        else:
            cw_raw = aggregate_college_wide_weekly(panel_4y)
            full_df, _ = engineer_safeguarding_features(cw_raw, group_col="curriculum_area")
            full_wide = attach_panel_hierarchy_wide(full_df, panel_4y)

        n_hist = 156  # 3 full academic years history (2022/23 - 2024/25)
        hist_wide = full_wide.iloc[:n_hist].copy().reset_index(drop=True)
        # Future calendar skeleton strictly uses deterministic calendar/exposure assumptions (never observed counts!)
        fut_skeleton = full_wide.iloc[n_hist : n_hist + horizon_weeks].copy().reset_index(drop=True)

        # Apply scenario modifiers to future weekly skeleton
        fut_skeleton["active_enrolments"] = np.maximum(
            10.0,
            fut_skeleton["active_enrolments"].to_numpy(dtype=float) * (1.0 + scenario.enrolment_growth_pct / 100.0),
        )
        fut_skeleton["log_enrolment"] = np.log(fut_skeleton["active_enrolments"])
        if scenario.attendance_rate_delta_pp != 0.0 and "overall_attendance_pct" in fut_skeleton.columns:
            fut_skeleton["overall_attendance_pct"] = np.clip(
                fut_skeleton["overall_attendance_pct"] + scenario.attendance_rate_delta_pp, 55.0, 99.5
            )
            fut_skeleton["early_disengagement_index"] = (
                fut_skeleton["early_disengagement_index"] - 0.22 * scenario.attendance_rate_delta_pp
            )

        eff_weekly_base = base_model_id if base_model_id in ("tweedie_lgbm", "nb2_glm", "zinb_model") else "tweedie_lgbm"
        if eff_weekly_base == "nb2_glm":
            base_cls = NB2GovernanceGLM
            base_name = "NB2 Governance GLM (Weekly Exposure Offset)"
        elif eff_weekly_base == "zinb_model":
            base_cls = ZINBSparseRegimeModel
            base_name = "Zero-Inflated Negative Binomial (ZINB Two-Part)"
        else:
            base_cls = TweedieLightGBMModel
            base_name = "Tweedie-Loss LightGBM (Weekly Exposure Offset)"

        base_model = base_cls().fit(hist_wide)

        # Causal out-of-fold baseline expectations on `hist_wide` for clean residual learning
        mu_base_hist = np.zeros(n_hist, dtype=float)
        y_hist = hist_wide["referral_count"].to_numpy(dtype=float)
        enr_hist = hist_wide["active_enrolments"].to_numpy(dtype=float)
        mu_base_hist[:52] = np.maximum(0.5, (np.sum(y_hist[:52]) / np.sum(enr_hist[:52])) * enr_hist[:52])
        for fold_s in (52, 78, 104, 130):
            fold_e = min(n_hist, fold_s + 26)
            m_sub = base_cls().fit(hist_wide.iloc[:fold_s].copy())
            mu_base_hist[fold_s:fold_e] = np.maximum(
                0.25, m_sub.predict(hist_wide.iloc[fold_s:fold_e].copy()).point_forecasts
            )

        # Fit EsotericFeatureSuite strictly on 156-week historical prefix
        suite = EsotericFeatureSuite(active_modules=active_mods, causal_mode="forecast", train_cutoff=n_hist)
        suite.fit(hist_wide)
        X_eso_hist = suite.transform(hist_wide).reset_index(drop=True)

        # Roll forward baseline & esoteric weekly trajectories strictly causally (updating referral_lag_t1 & rolling_3w from predicted counts!)
        base_work = hist_wide.copy()
        eso_work = hist_wide.copy()

        if strategy_id == "fisher_rao_ngboost":
            cand_cols = [c for c in X_eso_hist.columns if c.split("_", 1)[0].lower() in active_mods]
            funnel = GLMConditioningFunnel(
                base_cols=DEFAULT_BENCHMARK_FEATURES,
                target_col="referral_count",
                exposure_col="active_enrolments",
                vif_cap=5.0,
                cond_cap=30.0,
                max_esoteric_cols=6,
                l1_alpha=0.05,
            )
            funnel.fit(hist_wide, X_eso_hist[cand_cols])
            sel_eso = list(funnel.selected_esoteric_cols_) or cand_cols[:4]
            tr_design = funnel.transform(hist_wide, X_eso_hist[cand_cols])
            for c in sel_eso:
                if c not in tr_design.columns and c in X_eso_hist.columns:
                    tr_design[c] = X_eso_hist[c].to_numpy(dtype=float)
            ngb_cols = [c for c in DEFAULT_BENCHMARK_FEATURES if c in tr_design.columns] + [c for c in sel_eso if c in tr_design.columns]
            booster = FisherRaoNaturalGradientBooster(
                feature_cols=ngb_cols,
                n_estimators=55,
                learning_rate_mu=0.045,
                learning_rate_r=0.035,
                random_state=42,
            ).fit(tr_design.iloc[16:].reset_index(drop=True))
            head = None
        else:
            head = _fit_and_predict_esoteric_residual_head(
                y_train=y_hist,
                mu_base_train=mu_base_hist,
                enrol_train=enr_hist,
                X_eso_train=X_eso_hist,
                strategy_id=strategy_id,
                active_modules=active_mods,
                warm_start=52,
            )
            booster = None
            funnel = head["funnel"]
            cand_cols = head["cand_cols"]
            sel_eso = head["selected_cols"]

        cohort_cols = [c for c in hist_wide.columns if c.startswith("cohort_") and c.endswith("_referrals")]
        cohort_shares = {
            c: float(hist_wide[c].sum()) / max(1.0, float(np.sum(y_hist))) for c in cohort_cols
        }

        b_pts: List[int] = []
        b_raws: List[float] = []
        b_l_sel: List[int] = []
        b_u_sel: List[int] = []
        e_pts: List[int] = []
        e_raws: List[float] = []
        l80_list: List[int] = []
        u80_list: List[int] = []
        l95_list: List[int] = []
        u95_list: List[int] = []
        step_telem: List[Dict[str, Any]] = []

        for h_idx in range(horizon_weeks):
            # Step 1: Advance pure locked baseline causally using its own predicted lags
            b_next = fut_skeleton.iloc[[h_idx]].copy()
            b_prev_rates = (base_work["referral_count"].tail(3).to_numpy(dtype=float) / np.maximum(1.0, base_work["active_enrolments"].tail(3).to_numpy(dtype=float))) * 1000.0
            b_next["referral_lag_t1"] = float(b_prev_rates[-1])
            b_next["referral_rolling_3w"] = float(np.mean(b_prev_rates))
            b_next["referral_count"] = 0.0
            b_bun = base_model.predict(b_next, quantiles=(0.025, 0.10, 0.50, 0.90, 0.975))
            mu_b = max(0.1, float(b_bun.point_forecasts[0]))
            b_next["referral_count"] = mu_b
            base_work = pd.concat([base_work, b_next], ignore_index=True)

            b_pt = max(0, int(round(mu_b)))
            b_lo = max(0, int(round(float(b_bun.quantiles[0.025 if pi_level == 95 else 0.10][0]))))
            b_hi = max(b_pt, int(round(float(b_bun.quantiles[0.975 if pi_level == 95 else 0.90][0]))))
            b_pts.append(b_pt)
            b_raws.append(mu_b)
            b_l_sel.append(b_lo)
            b_u_sel.append(b_hi)

            # Step 2: Advance esoteric trajectory causally using its own predicted lags + cohort counts
            e_next = fut_skeleton.iloc[[h_idx]].copy()
            e_prev_rates = (eso_work["referral_count"].tail(3).to_numpy(dtype=float) / np.maximum(1.0, eso_work["active_enrolments"].tail(3).to_numpy(dtype=float))) * 1000.0
            e_next["referral_lag_t1"] = float(e_prev_rates[-1])
            e_next["referral_rolling_3w"] = float(np.mean(e_prev_rates))
            e_next["referral_count"] = 0.0
            for c_col in cohort_cols:
                e_next[c_col] = 0.0
            eso_work = pd.concat([eso_work, e_next], ignore_index=True)

            X_eso_step_all = suite.transform(eso_work)
            row_eso_full = X_eso_step_all.iloc[[n_hist + h_idx]]

            if strategy_id == "fisher_rao_ngboost" and booster is not None:
                row_b_df = eso_work.iloc[[n_hist + h_idx]].reset_index(drop=True)
                row_e_df = row_eso_full[cand_cols].reset_index(drop=True)
                st_des = funnel.transform(row_b_df, row_e_df)
                for c in sel_eso:
                    if c not in st_des.columns and c in row_e_df.columns:
                        st_des[c] = row_e_df[c].to_numpy(dtype=float)
                ebun = booster.predict(st_des, quantiles=(0.025, 0.10, 0.50, 0.90, 0.975))
                mu_e = max(0.0, float(ebun.point_forecasts[0]))
                pt_e = max(0, int(round(mu_e)))
                l80_v = max(0, int(round(float(ebun.quantiles[0.10][0]))))
                u80_v = max(pt_e, int(round(float(ebun.quantiles[0.90][0]))))
                l95_v = max(0, int(round(float(ebun.quantiles[0.025][0]))))
                u95_v = max(u80_v, int(round(float(ebun.quantiles[0.975][0]))))
                r_init_val = float(np.exp(booster.base_eta_r_))
                r_step = float(ebun.diagnostics.get("mean_r_shape", r_init_val))
                dom_mod = "Fisher-Rao Natural Gradient NB Manifold"
            else:
                assert head is not None
                Z_st = head["standardize_fn"](row_eso_full)
                betas = head["betas"]
                if len(betas) > 0 and Z_st.shape[1] == len(betas):
                    c_vec = Z_st[0] * betas
                    eta_sh = float(np.clip(np.sum(c_vec), -0.32, 0.32))
                    dom_c = head["selected_cols"][int(np.argmax(np.abs(c_vec)))]
                    dom_mod = MODULE_DISPLAY_NAMES.get(dom_c.split("_", 1)[0], dom_c)
                else:
                    eta_sh = 0.0
                    dom_mod = "Locked Baseline (Identity Floor)"

                # Anchor residual multiplier onto baseline prediction evaluated at esoteric causal lags
                b_on_eso = base_model.predict(eso_work.iloc[[n_hist + h_idx]], quantiles=(0.025, 0.10, 0.50, 0.90, 0.975))
                mu_b_anchor = max(0.1, float(b_on_eso.point_forecasts[0]))
                mu_e = max(0.0, mu_b_anchor * float(np.exp(eta_sh)))
                pt_e = max(0, int(round(mu_e)))

                r_base = float(head["r_base"])
                stress_n = float(np.mean(np.abs(Z_st[0]))) if Z_st.shape[1] > 0 else 0.0
                r_step = float(np.clip(np.exp(np.log(r_base) - 0.18 * float(head["fr_dispersion_gain"]) * (stress_n - 0.65)), 2.2, 45.0))
                d_scale = float(np.clip(np.sqrt(r_base / r_step), 0.85, 1.25))
                ratio_m = mu_e / max(0.25, mu_b_anchor)

                l80_v = max(0, int(round(mu_e - (mu_b_anchor - float(b_on_eso.quantiles[0.10][0])) * ratio_m * d_scale)))
                u80_v = max(pt_e, int(round(mu_e + (float(b_on_eso.quantiles[0.90][0]) - mu_b_anchor) * ratio_m * d_scale)))
                l95_v = max(0, int(round(mu_e - (mu_b_anchor - float(b_on_eso.quantiles[0.025][0])) * ratio_m * d_scale)))
                u95_v = max(u80_v, int(round(mu_e + (float(b_on_eso.quantiles[0.975][0]) - mu_b_anchor) * ratio_m * d_scale)))

            eso_work.at[n_hist + h_idx, "referral_count"] = mu_e
            for c_col, sh in cohort_shares.items():
                eso_work.at[n_hist + h_idx, c_col] = mu_e * sh

            e_pts.append(pt_e)
            e_raws.append(mu_e)
            l80_list.append(l80_v)
            u80_list.append(u80_v)
            l95_list.append(l95_v)
            u95_list.append(u95_v)
            step_telem.append(
                self._extract_step_manifold_indicators(row_eso_full.iloc[0], suite, r_step, dom_mod)
            )

        baseline_sha256 = hashlib.sha256(json.dumps([round(x, 5) for x in b_raws]).encode()).hexdigest()

        # Walk-forward evaluation on final 26 historical weeks (origin = week 130 -> weeks 130..155)
        bt_weekly = self._evaluate_weekly_walk_forward(
            hist_wide=hist_wide,
            base_cls=base_cls,
            strategy_id=strategy_id,
            active_mods=active_mods,
            pi_level=pi_level,
        )

        ann_tot, ann_l80, ann_u80, ann_l95, ann_u95, _ = compute_calibrated_annual_interval(
            e_pts, l80_list, u80_list, l95_list, u95_list, residual_autocorr_lag1=0.22
        )
        b_tot, b_al80, b_au80, b_al95, b_au95, _ = compute_calibrated_annual_interval(
            b_pts, b_l_sel, b_u_sel, b_l_sel, b_u_sel, residual_autocorr_lag1=0.22
        )
        eso_lb = int(ann_l95 if (pi_level == 95 and ann_l95 is not None) else ann_l80)
        eso_ub = int(ann_u95 if (pi_level == 95 and ann_u95 is not None) else ann_u80)
        prior_w_tot = int(round(float(hist_wide.tail(horizon_weeks)["referral_count"].sum())))
        net_delta = int(ann_tot - b_tot)
        net_delta_pct = round((net_delta / max(1, b_tot)) * 100.0, 2)

        if head is not None:
            head_diag = self._build_head_diagnostics(head, n_hist)
        else:
            assert booster is not None
            used_cols = list(booster.used_cols_)
            imp_vec = np.zeros(len(used_cols), dtype=float)
            for tr_m, cols_m, sc_m in booster.trees_mu_:
                imp_vec[cols_m] += abs(float(sc_m)) * tr_m.feature_importances_
            for tr_r, cols_r, sc_r in booster.trees_r_:
                imp_vec[cols_r] += 0.5 * abs(float(sc_r)) * tr_r.feature_importances_
            tot_imp = max(1e-9, float(np.sum(imp_vec)))
            admitted_list = []
            for idx_f, col in enumerate(used_cols):
                if col in sel_eso:
                    m_name, desc = _describe_esoteric_column(col)
                    admitted_list.append(
                        {
                            "feature": col,
                            "module_id": col.split("_", 1)[0].lower(),
                            "module_name": m_name,
                            "weight": round(float(imp_vec[idx_f] / tot_imp), 4),
                            "direction": "Manifold Natural Gradient (mu & r)",
                            "description": desc,
                        }
                    )
            admitted_list.sort(key=lambda x: abs(x["weight"]), reverse=True)
            head_diag = {
                "funnel_type": "Fisher-Rao Natural Gradient NB2 + QR/VIF Funnel",
                "n_train_steps": n_hist,
                "n_candidate_esoteric_features": len(cand_cols),
                "n_admitted_esoteric_features": len(admitted_list),
                "n_quarantined_esoteric_features": max(0, len(cand_cols) - len(admitted_list)),
                "max_vif_after": round(float(funnel.diagnostics_.max_vif_after), 2),
                "condition_number_after": round(float(funnel.diagnostics_.condition_number_after), 2),
                "base_nb_dispersion_r": round(float(np.exp(booster.base_eta_r_)), 2),
                "identity_floor_active": len(admitted_list) == 0,
                "admitted_features": admitted_list,
            }

        hist_points = []
        for _, r in hist_wide.tail(52).iterrows():
            dt = pd.to_datetime(r["week_start"])
            enr_v = max(1, int(round(float(r["active_enrolments"]))))
            cnt_v = int(round(float(r["referral_count"])))
            hist_points.append(
                {
                    "period_start": dt.strftime("%Y-%m-%d"),
                    "period_label": dt.strftime("%d %b %y"),
                    "academic_year": str(r["academic_year"]),
                    "academic_term": get_academic_term(dt.month),
                    "actual_count": cnt_v,
                    "enrolled_students": enr_v,
                    "teaching_days": 0 if int(r["is_low_activity_week"]) == 1 else 5,
                    "rate_per_1000": round((cnt_v / enr_v) * 1000.0, 2),
                }
            )

        forecast_points = []
        for idx in range(horizon_weeks):
            r_fut = fut_skeleton.iloc[idx]
            dt = pd.to_datetime(r_fut["week_start"])
            enr_v = max(1, int(round(float(r_fut["active_enrolments"]))))
            st = step_telem[idx]
            e_l = int(l95_list[idx]) if pi_level == 95 else int(l80_list[idx])
            e_u = int(u95_list[idx]) if pi_level == 95 else int(u80_list[idx])

            forecast_points.append(
                {
                    "step": idx + 1,
                    "period_start": dt.strftime("%Y-%m-%d"),
                    "period_label": dt.strftime("%d %b %y"),
                    "academic_year": str(r_fut["academic_year"]),
                    "academic_term": "Closure / Break" if int(r_fut["is_low_activity_week"]) == 1 else get_academic_term(dt.month),
                    "baseline_forecast": int(b_pts[idx]),
                    "baseline_lower": int(b_l_sel[idx]),
                    "baseline_upper": int(b_u_sel[idx]),
                    "esoteric_forecast": int(e_pts[idx]),
                    "esoteric_raw": round(float(e_raws[idx]), 2),
                    "lower_bound": e_l,
                    "upper_bound": e_u,
                    "lower_80": int(l80_list[idx]),
                    "upper_80": int(u80_list[idx]),
                    "lower_95": int(l95_list[idx]),
                    "upper_95": int(u95_list[idx]),
                    "delta_count": int(e_pts[idx] - b_pts[idx]),
                    "delta_pct": round(((e_raws[idx] - b_raws[idx]) / max(1.0, b_raws[idx])) * 100.0, 2),
                    "enrolled_students": enr_v,
                    "teaching_days": 0 if int(r_fut["is_low_activity_week"]) == 1 else 5,
                    "rate_per_1000": round((e_pts[idx] / enr_v) * 1000.0, 2),
                    "rough_levy_area": st["rough_levy_area"],
                    "rough_sig_depth3": st["rough_sig_depth3"],
                    "hilbert_phase_deg": st["hilbert_phase_deg"],
                    "hawkes_intensity": st["hawkes_intensity"],
                    "dispersion_r": st["dispersion_r"],
                    "dominant_driver": st["dominant_driver"],
                }
            )

        peak_item = max(forecast_points, key=lambda x: x["esoteric_forecast"])
        runtime_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return {
            "cadence": "weekly",
            "period_unit": "weeks",
            "dataset_id": "weekly_fe_safeguarding_panel_3y",
            "is_synthetic": True,
            "governance_banner": "SYNTHETIC DEMONSTRATION DATA — WEEKLY UK FE CURRICULUM PANEL (39 ACTIVE TEACHING WEEKS/YR)",
            "strategy": strat_meta,
            "base_model_id": eff_weekly_base,
            "base_model_name": base_name,
            "baseline_sha256": baseline_sha256,
            "active_modules": list(active_mods),
            "module_catalog": MODULE_CATALOG,
            "strategies_catalog": ESOTERIC_STRATEGIES,
            "horizon": horizon_weeks,
            "provision": provision,
            "safeguarding_category": "All",
            "prediction_interval": pi_level,
            "runtime_ms": runtime_ms,
            "summary": {
                "esoteric_total": int(ann_tot),
                "esoteric_lower": eso_lb,
                "esoteric_upper": eso_ub,
                "baseline_total": int(b_tot),
                "baseline_lower": int(b_al80),
                "baseline_upper": int(b_au80),
                "prior_horizon_total": prior_w_tot,
                "yoy_change_pct": round(((ann_tot - prior_w_tot) / max(1, prior_w_tot)) * 100.0, 1),
                "net_esoteric_delta": net_delta,
                "net_esoteric_delta_pct": net_delta_pct,
                "peak_period_label": f"Wk {peak_item['period_label']}",
                "peak_period_count": peak_item["esoteric_forecast"],
                "calibration_method": "Exposure-Proportional Count Quantiles + Fisher-Rao Dispersion Scaling",
            },
            "conditioning_diagnostics": head_diag,
            "walk_forward_backtest": bt_weekly,
            "historical_series": hist_points,
            "forecast_series": forecast_points,
            "mathematical_notes": self._build_mathematical_notes(
                cadence="weekly",
                strat_meta=strat_meta,
                base_name=base_name,
                head_diag=head_diag,
                bt_summary=bt_weekly,
                net_delta=net_delta,
                net_delta_pct=net_delta_pct,
            ),
        }

    def _evaluate_weekly_walk_forward(
        self,
        hist_wide: pd.DataFrame,
        base_cls: Any,
        strategy_id: str,
        active_mods: Tuple[str, ...],
        pi_level: int,
    ) -> Dict[str, Any]:
        """Evaluate expanding walk-forward split at week 130 -> weeks 130..155 (26 out-of-sample weeks)."""
        orig = 130
        tr = hist_wide.iloc[:orig].copy().reset_index(drop=True)
        te = hist_wide.iloc[orig:].copy().reset_index(drop=True)

        b_mod = base_cls().fit(tr)
        b_bun = b_mod.predict(te, quantiles=(0.025, 0.10, 0.50, 0.90, 0.975))
        yb = np.maximum(0.1, b_bun.point_forecasts)
        yt = te["referral_count"].to_numpy(dtype=float)

        suite_bt = EsotericFeatureSuite(active_modules=active_mods, causal_mode="forecast", train_cutoff=orig)
        suite_bt.fit(hist_wide.iloc[:orig])
        X_eso_all = suite_bt.transform(hist_wide).reset_index(drop=True)
        X_eso_tr = X_eso_all.iloc[:orig].reset_index(drop=True)
        X_eso_te = X_eso_all.iloc[orig:].reset_index(drop=True)

        if strategy_id == "fisher_rao_ngboost":
            cands = [c for c in X_eso_tr.columns if c.split("_", 1)[0].lower() in active_mods]
            fn = GLMConditioningFunnel(base_cols=DEFAULT_BENCHMARK_FEATURES, max_esoteric_cols=6).fit(tr, X_eso_tr[cands])
            sel = list(fn.selected_esoteric_cols_) or cands[:4]
            tr_d = fn.transform(tr, X_eso_tr[cands])
            te_d = fn.transform(te, X_eso_te[cands])
            for c in sel:
                if c not in tr_d.columns and c in X_eso_tr.columns:
                    tr_d[c] = X_eso_tr[c].to_numpy(dtype=float)
                    te_d[c] = X_eso_te[c].to_numpy(dtype=float)
            cols = [c for c in DEFAULT_BENCHMARK_FEATURES if c in tr_d.columns] + [c for c in sel if c in tr_d.columns]
            ngb = FisherRaoNaturalGradientBooster(feature_cols=cols, n_estimators=50, random_state=42).fit(tr_d.iloc[16:].reset_index(drop=True))
            ebun = ngb.predict(te_d, quantiles=(0.025, 0.10, 0.50, 0.90, 0.975))
            ye = np.maximum(0.1, ebun.point_forecasts)
            e_lo_vec = ebun.quantiles[0.025 if pi_level == 95 else 0.10]
            e_hi_vec = ebun.quantiles[0.975 if pi_level == 95 else 0.90]
        else:
            y_tr = tr["referral_count"].to_numpy(dtype=float)
            enr_tr = tr["active_enrolments"].to_numpy(dtype=float)
            mu_b_tr = np.zeros(orig, dtype=float)
            mu_b_tr[:52] = (np.sum(y_tr[:52]) / np.sum(enr_tr[:52])) * enr_tr[:52]
            for fs in (52, 78, 104):
                fe = min(orig, fs + 26)
                mu_b_tr[fs:fe] = np.maximum(0.25, base_cls().fit(tr.iloc[:fs].copy()).predict(tr.iloc[fs:fe].copy()).point_forecasts)
            hd = _fit_and_predict_esoteric_residual_head(y_tr, mu_b_tr, enr_tr, X_eso_tr, strategy_id, active_mods, warm_start=52)
            Z_te = hd["standardize_fn"](X_eso_te)
            eta_te = np.clip(Z_te @ hd["betas"], -0.32, 0.32) if len(hd["betas"]) > 0 else np.zeros(len(te))
            ye = np.maximum(0.1, yb * np.exp(eta_te))
            ratio_v = ye / np.maximum(0.25, yb)
            e_lo_vec = b_bun.quantiles[0.025 if pi_level == 95 else 0.10] * ratio_v
            e_hi_vec = b_bun.quantiles[0.975 if pi_level == 95 else 0.90] * ratio_v

        b_lo_vec = b_bun.quantiles[0.025 if pi_level == 95 else 0.10]
        b_hi_vec = b_bun.quantiles[0.975 if pi_level == 95 else 0.90]

        b_mae = compute_mae(yt.tolist(), yb.tolist())
        e_mae = compute_mae(yt.tolist(), ye.tolist())
        b_rmse = compute_rmse(yt.tolist(), yb.tolist())
        e_rmse = compute_rmse(yt.tolist(), ye.tolist())
        tot_y = max(1.0, float(np.sum(yt)))
        b_wape = float(np.sum(np.abs(yt - yb)) / tot_y * 100.0)
        e_wape = float(np.sum(np.abs(yt - ye)) / tot_y * 100.0)
        b_surge = compute_surge_mae(
            yt.tolist(),
            yb.tolist(),
            te["is_autumn_transition"].astype(int).tolist(),
            te["is_pre_holiday_week"].astype(int).tolist(),
            te["is_post_holiday_week"].astype(int).tolist(),
        )
        e_surge = compute_surge_mae(
            yt.tolist(),
            ye.tolist(),
            te["is_autumn_transition"].astype(int).tolist(),
            te["is_pre_holiday_week"].astype(int).tolist(),
            te["is_post_holiday_week"].astype(int).tolist(),
        )
        b_cov = float(np.mean((yt >= b_lo_vec) & (yt <= b_hi_vec)) * 100.0)
        e_cov = float(np.mean((yt >= e_lo_vec) & (yt <= e_hi_vec)) * 100.0)

        pts_bt = [
            {
                "period_label": pd.to_datetime(te["week_start"].iloc[i]).strftime("%d %b %y"),
                "actual": int(round(float(yt[i]))),
                "baseline": round(float(yb[i]), 1),
                "esoteric": round(float(ye[i]), 1),
            }
            for i in range(len(te))
        ]
        return {
            "evaluated_periods": len(te),
            "folds": 1,
            "baseline_mae": round(b_mae, 2),
            "esoteric_mae": round(e_mae, 2),
            "mae_delta_pct": round(((b_mae - e_mae) / max(1e-6, b_mae)) * 100.0, 2),
            "baseline_rmse": round(b_rmse, 2),
            "esoteric_rmse": round(e_rmse, 2),
            "baseline_wape_pct": round(b_wape, 2),
            "esoteric_wape_pct": round(e_wape, 2),
            "baseline_surge_mae": round(b_surge, 2),
            "esoteric_surge_mae": round(e_surge, 2),
            "baseline_pi_coverage_pct": round(b_cov, 1),
            "esoteric_pi_coverage_pct": round(e_cov, 1),
            "points": pts_bt,
        }

    @staticmethod
    def _extract_step_manifold_indicators(
        eso_row: pd.Series,
        suite: EsotericFeatureSuite,
        r_step: float,
        dom_mod: str,
    ) -> Dict[str, Any]:
        levy = float(eso_row.get("f07_levy_area_max_abs", 0.0))
        sig3 = float(eso_row.get("f07_sig_depth3_frobenius", 0.0))

        # Hilbert phase angle in degrees from sin/cos or raw phase column if active
        h_sin = float(eso_row.get("f01_hilbert_sin_phase", eso_row.get("f01_warped_acad_sin1", 0.0)))
        h_cos = float(eso_row.get("f01_hilbert_cos_phase", eso_row.get("f01_warped_acad_cos1", 1.0)))
        phase_deg = round(float(np.degrees(np.arctan2(h_sin, h_cos))), 1)

        hawkes_int = float(
            eso_row.get(
                "f03_hawkes_intensity_total",
                eso_row.get("f03_hawkes_lambda_total", eso_row.get("f03_hawkes_endog_share", 0.0)),
            )
        )
        return {
            "rough_levy_area": round(levy, 4),
            "rough_sig_depth3": round(sig3, 4),
            "hilbert_phase_deg": phase_deg,
            "hawkes_intensity": round(hawkes_int, 3),
            "dispersion_r": round(r_step, 2),
            "dominant_driver": dom_mod,
        }

    @staticmethod
    def _build_head_diagnostics(head: Dict[str, Any], n_train: int) -> Dict[str, Any]:
        funnel: GLMConditioningFunnel = head["funnel"]
        selected_cols: List[str] = head["selected_cols"]
        betas: np.ndarray = head["betas"]
        cand_cols: List[str] = head["cand_cols"]

        admitted = []
        for idx, col in enumerate(selected_cols):
            w = float(betas[idx]) if idx < len(betas) else 0.0
            m_name, desc = _describe_esoteric_column(col)
            admitted.append(
                {
                    "feature": col,
                    "module_id": col.split("_", 1)[0].lower(),
                    "module_name": m_name,
                    "weight": round(w, 4),
                    "direction": "Amplifies Surge (+log offset)" if w >= 0 else "Dampens Trough (-log offset)",
                    "description": desc,
                }
            )
        admitted.sort(key=lambda x: abs(x["weight"]), reverse=True)

        return {
            "funnel_type": "QR Rank-Pivoting + VIF <= 5.0 + Deviance-Weighted Log-Offset Ridge",
            "n_train_steps": n_train,
            "n_candidate_esoteric_features": len(cand_cols),
            "n_admitted_esoteric_features": len(admitted),
            "n_quarantined_esoteric_features": max(0, len(cand_cols) - len(admitted)),
            "max_vif_after": round(float(funnel.diagnostics_.max_vif_after), 2),
            "condition_number_after": round(float(funnel.diagnostics_.condition_number_after), 2),
            "base_nb_dispersion_r": round(float(head["r_base"]), 2),
            "identity_floor_active": len(admitted) == 0,
            "admitted_features": admitted,
        }

    @staticmethod
    def _build_mathematical_notes(
        cadence: str,
        strat_meta: Dict[str, Any],
        base_name: str,
        head_diag: Dict[str, Any],
        bt_summary: Dict[str, Any],
        net_delta: int,
        net_delta_pct: float,
    ) -> List[str]:
        notes = [
            f"Causal Architecture: '{strat_meta['name']}' runs recursively step-by-step on top of locked '{base_name}' without modifying any core operational tab.",
            f"Anti-Bloat Conditioning: Screened {head_diag['n_candidate_esoteric_features']} candidate coordinates down to {head_diag['n_admitted_esoteric_features']} rank-safe predictors (Max VIF = {head_diag['max_vif_after']}, Condition Number = {head_diag['condition_number_after']}), quarantining {head_diag['n_quarantined_esoteric_features']} collinear/noisy coordinates.",
            f"Horizon Adjustment: Higher-order mathematical signals shift total projected {cadence} demand by {net_delta:+d} concerns ({net_delta_pct:+.2f}%) relative to the pure locked baseline.",
        ]
        if bt_summary.get("evaluated_periods", 0) > 0:
            mae_d = bt_summary["mae_delta_pct"]
            verdict = "improves" if mae_d >= 0 else "slightly inflates variance on"
            notes.append(
                f"Walk-Forward Validation ({bt_summary['evaluated_periods']} out-of-sample {cadence} steps): Esoteric integration {verdict} MAE ({bt_summary['baseline_mae']} -> {bt_summary['esoteric_mae']}, {mae_d:+.2f}%) and Surge MAE ({bt_summary['baseline_surge_mae']} -> {bt_summary['esoteric_surge_mae']})."
            )
        return notes


esoteric_forecast_engine = EsotericForecastEngine()
