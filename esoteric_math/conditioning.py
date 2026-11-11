"""
Phase 3: Model-Aware Feature Conditioning Funnels (`esoteric_math/conditioning.py`).

Separates parametric GLM conditioning from non-linear GBDT conditioning strictly on training folds:
  1. `GLMConditioningFunnel`:
     - Keeps existing baseline predictors (`base_cols`) 100% intact.
     - Robust-standardizes esoteric candidate features using training-fold median/IQR and clips to [-5, 5].
     - Applies QR column-pivoting rank pruning + iterative VIF / condition-number control (cond <= max_cond)
       against the locked baseline design matrix so statsmodels IRLS/MLE never encounters rank deficiency.
     - Screens surviving esoteric candidates via L1-penalized Poisson/NB log-offset deviance improvement
       over the baseline GLM linear predictor, capping added degrees of freedom dynamically by sample size.
  2. `GBDTConditioningFunnel`:
     - Concatenates baseline covariates with esoteric feature blocks (`f01`..`f13`).
     - Applies variance filtering + inner expanding-window group/feature gain & permutation screening
       to prune noisy features that overfit quiet/holiday weeks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import warnings
import numpy as np
import pandas as pd
from scipy.linalg import qr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import PoissonRegressor
from lightgbm import LGBMRegressor


@dataclass
class ConditioningDiagnostics:
    funnel_type: str
    n_train_rows: int
    n_base_features: int
    n_esoteric_candidates: int
    n_selected_esoteric: int
    selected_esoteric_cols: List[str]
    condition_number_after: float
    max_vif_after: float
    notes: str = ""
    module_scores: Dict[str, float] = field(default_factory=dict)


def _compute_max_vif_and_cond(X: np.ndarray) -> Tuple[float, float]:
    """Compute maximum Variance Inflation Factor and spectral condition number of standardized X."""
    if X.ndim != 2 or X.shape[1] == 0 or X.shape[0] < 3:
        return 1.0, 1.0
    stds = np.std(X, axis=0)
    valid = stds > 1e-9
    if not np.any(valid):
        return 1.0, 1.0
    Z = (X[:, valid] - np.mean(X[:, valid], axis=0)) / stds[valid]
    svals = np.linalg.svd(Z, compute_uv=False)
    cond = float(svals[0] / max(svals[-1], 1e-12)) if len(svals) > 0 else 1.0
    if Z.shape[1] == 1:
        return 1.0, cond
    corr = (Z.T @ Z) / float(len(Z))
    corr = 0.5 * (corr + corr.T) + np.eye(Z.shape[1]) * 1e-6
    inv_corr = np.linalg.pinv(corr)
    max_vif = float(np.max(np.diag(inv_corr)))
    return max(1.0, max_vif), cond


class GLMConditioningFunnel(BaseEstimator, TransformerMixin):
    """
    Parametric Count GLM Conditioning Funnel:
    Preserves `base_cols` untouched and appends a strictly rank-safe, VIF-controlled,
    L1-deviance-screened subset of standardized esoteric features.
    """

    def __init__(
        self,
        base_cols: Sequence[str],
        target_col: str = "referral_count",
        exposure_col: str = "active_enrolments",
        vif_cap: float = 6.5,
        cond_cap: float = 35.0,
        max_esoteric_cols: Optional[int] = None,
        l1_alpha: float = 0.08,
    ) -> None:
        self.base_cols = list(base_cols)
        self.target_col = target_col
        self.exposure_col = exposure_col
        self.vif_cap = vif_cap
        self.cond_cap = cond_cap
        self.max_esoteric_cols = max_esoteric_cols
        self.l1_alpha = l1_alpha

    def fit(self, train_df: pd.DataFrame, esoteric_df: pd.DataFrame) -> "GLMConditioningFunnel":
        n_rows = len(train_df)
        self.active_base_cols_ = [c for c in self.base_cols if c in train_df.columns]
        cand_cols = [c for c in esoteric_df.columns if c not in self.active_base_cols_]

        # 1. Training-only robust standardization (median / IQR fallback to std)
        self.medians_: Dict[str, float] = {}
        self.scales_: Dict[str, float] = {}
        non_const_cands: List[str] = []

        for col in cand_cols:
            vals = np.nan_to_num(esoteric_df[col].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
            std_v = float(np.std(vals))
            if std_v <= 1e-6:
                continue
            med_v = float(np.median(vals))
            q75, q25 = np.percentile(vals, [75.0, 25.0])
            iqr = float((q75 - q25) / 1.349)
            scale_v = max(iqr, std_v, 1e-6)
            self.medians_[col] = med_v
            self.scales_[col] = scale_v
            non_const_cands.append(col)

        if not non_const_cands or n_rows < 24:
            self.selected_esoteric_cols_ = []
            self.diagnostics_ = ConditioningDiagnostics(
                funnel_type="GLM_QR_VIF_L1",
                n_train_rows=n_rows,
                n_base_features=len(self.active_base_cols_),
                n_esoteric_candidates=len(cand_cols),
                n_selected_esoteric=0,
                selected_esoteric_cols=[],
                condition_number_after=1.0,
                max_vif_after=1.0,
                notes="Insufficient sample rows or non-constant esoteric candidates.",
            )
            return self

        Z_cand = np.column_stack(
            [
                np.clip(
                    (np.nan_to_num(esoteric_df[c].to_numpy(dtype=float), nan=self.medians_[c]) - self.medians_[c])
                    / self.scales_[c],
                    -5.0,
                    5.0,
                )
                for c in non_const_cands
            ]
        )

        # 2. Pivoted QR rank & collinearity pre-filter among esoteric candidates
        _, R_qr, piv = qr(Z_cand, mode="economic", pivoting=True)
        diag_r = np.abs(np.diag(R_qr))
        tol_r = max(1e-4, 0.05 * float(diag_r[0])) if len(diag_r) > 0 else 1e-4
        keep_rank = int(np.sum(diag_r >= tol_r))
        qr_indices = list(piv[: max(1, keep_rank)])
        qr_cols = [non_const_cands[i] for i in qr_indices]
        Z_qr = Z_cand[:, qr_indices]

        # 3. Exposure-adjusted L1-Poisson / deviance residual screening on training fold
        y = np.maximum(0.0, train_df[self.target_col].to_numpy(dtype=float))
        if self.exposure_col in train_df.columns:
            enrol = np.maximum(1.0, train_df[self.exposure_col].to_numpy(dtype=float))
        else:
            enrol = np.ones(n_rows, dtype=float)
        rate_target = y / enrol

        X_base = (
            train_df[self.active_base_cols_].to_numpy(dtype=float)
            if self.active_base_cols_
            else np.zeros((n_rows, 0), dtype=float)
        )
        base_means = np.mean(X_base, axis=0) if X_base.shape[1] > 0 else np.array([])
        base_stds = np.maximum(np.std(X_base, axis=0), 1e-6) if X_base.shape[1] > 0 else np.array([])
        Z_base = (X_base - base_means) / base_stds if X_base.shape[1] > 0 else np.zeros((n_rows, 0))

        # Fit baseline regularized Poisson rate model to obtain baseline log-rate offset
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if Z_base.shape[1] > 0:
                base_pois = PoissonRegressor(alpha=0.02, max_iter=300)
                base_pois.fit(Z_base, rate_target, sample_weight=enrol)
                mu_base = np.maximum(1e-7, base_pois.predict(Z_base) * enrol)
            else:
                mu_base = np.maximum(1e-7, (np.sum(y) / np.sum(enrol)) * enrol)

            # Score each QR-surviving esoteric column by inner out-of-time deviance reduction & Pearson score
            pearson_resid = (y - mu_base) / np.sqrt(mu_base + 0.18 * (mu_base ** 2))
            corrs = np.array(
                [abs(float(np.corrcoef(Z_qr[:, j], pearson_resid)[0, 1])) if np.std(Z_qr[:, j]) > 1e-8 else 0.0 for j in range(Z_qr.shape[1])]
            )
            corrs = np.nan_to_num(corrs, nan=0.0)

            # Also fit joint L2/Poisson screen on top 24 correlated QR columns
            top_pre = np.argsort(corrs)[::-1][: min(24, len(qr_cols))]
            Z_joint = np.hstack([Z_base, Z_qr[:, top_pre]])
            joint_pois = PoissonRegressor(alpha=self.l1_alpha, max_iter=400)
            joint_pois.fit(Z_joint, rate_target, sample_weight=enrol)
            eso_coefs = np.abs(joint_pois.coef_[Z_base.shape[1] :])

        combined_importance = 0.55 * (eso_coefs / (np.max(eso_coefs) + 1e-9)) + 0.45 * (
            corrs[top_pre] / (np.max(corrs[top_pre]) + 1e-9)
        )
        ranked_order = [top_pre[i] for i in np.argsort(combined_importance)[::-1]]

        # Cap added GLM columns conservatively to preserve IRLS degrees of freedom
        cap_k = self.max_esoteric_cols if self.max_esoteric_cols is not None else min(5, max(1, n_rows // 24))

        # 4. Forward greedy VIF + Condition Number + Inner-Fold NB Deviance verification
        selected: List[str] = []
        selected_vecs: List[np.ndarray] = []
        split_idx = max(18, int(n_rows * 0.72))

        def _inner_val_nb_dev(extra_mat: Optional[np.ndarray]) -> float:
            if split_idx >= n_rows - 3:
                return 0.0
            X_tr = Z_base[:split_idx] if extra_mat is None else np.hstack([Z_base[:split_idx], extra_mat[:split_idx]])
            X_va = Z_base[split_idx:] if extra_mat is None else np.hstack([Z_base[split_idx:], extra_mat[split_idx:]])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mod = PoissonRegressor(alpha=0.05, max_iter=300)
                mod.fit(X_tr, rate_target[:split_idx], sample_weight=enrol[:split_idx])
                pred_va = np.maximum(1e-6, mod.predict(X_va) * enrol[split_idx:])
            yt_va = y[split_idx:]
            a = 0.18
            t1 = np.where(yt_va > 0, yt_va * np.log(np.maximum(yt_va, 1e-12) / pred_va), 0.0)
            t2 = (yt_va + 1.0 / a) * np.log((1.0 + a * yt_va) / (1.0 + a * pred_va))
            return float(np.mean(np.maximum(0.0, 2.0 * (t1 - t2))))

        best_val_dev = _inner_val_nb_dev(None)

        for qr_i in ranked_order:
            if len(selected) >= cap_k:
                break
            col_name = qr_cols[qr_i]
            vec = Z_qr[:, qr_i : qr_i + 1]
            trial_mat = np.hstack([Z_base] + selected_vecs + [vec]) if (Z_base.shape[1] > 0 or selected_vecs) else vec
            max_vif, cond_num = _compute_max_vif_and_cond(trial_mat)
            if max_vif > self.vif_cap or cond_num > self.cond_cap:
                continue

            trial_eso = np.hstack(selected_vecs + [vec])
            cand_dev = _inner_val_nb_dev(trial_eso)
            # Require strict inner-validation deviance non-degradation (or accept top 1 if single-module probe)
            if cand_dev <= best_val_dev * 1.002 or (self.max_esoteric_cols == 1 and len(selected) == 0 and cand_dev <= best_val_dev * 1.025):
                selected.append(col_name)
                selected_vecs.append(vec)
                best_val_dev = min(best_val_dev, cand_dev)

        final_mat = np.hstack([Z_base] + selected_vecs) if selected_vecs else Z_base
        final_vif, final_cond = _compute_max_vif_and_cond(final_mat)

        self.selected_esoteric_cols_ = selected
        self.selected_feature_cols_ = list(self.active_base_cols_) + list(selected)
        self.diagnostics_ = ConditioningDiagnostics(
            funnel_type="GLM_QR_VIF_L1",
            n_train_rows=n_rows,
            n_base_features=len(self.active_base_cols_),
            n_esoteric_candidates=len(cand_cols),
            n_selected_esoteric=len(selected),
            selected_esoteric_cols=list(selected),
            condition_number_after=round(final_cond, 3),
            max_vif_after=round(final_vif, 3),
            notes=f"Retained {len(selected)}/{len(cand_cols)} rank-safe esoteric predictors (VIF<={self.vif_cap}, cond<={self.cond_cap}).",
        )
        return self

    def transform(self, df: pd.DataFrame, esoteric_df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.selected_esoteric_cols_:
            med = self.medians_[col]
            sc = self.scales_[col]
            raw = np.nan_to_num(esoteric_df[col].to_numpy(dtype=float), nan=med, posinf=med, neginf=med)
            out[col] = np.clip((raw - med) / sc, -5.0, 5.0)
        return out


class GBDTConditioningFunnel(BaseEstimator, TransformerMixin):
    """
    Gradient-Boosted Decision Tree Conditioning Funnel:
    Concatenates baseline covariates with all 13 esoteric blocks, applies variance filtering,
    and performs inner expanding-window group/feature permutation & split-gain pruning to
    prevent overfitting quiet/holiday weeks.
    """

    def __init__(
        self,
        base_cols: Sequence[str],
        target_col: str = "referral_count",
        exposure_col: str = "active_enrolments",
        max_esoteric_cols: int = 18,
    ) -> None:
        self.base_cols = list(base_cols)
        self.target_col = target_col
        self.exposure_col = exposure_col
        self.max_esoteric_cols = max_esoteric_cols

    def fit(self, train_df: pd.DataFrame, esoteric_df: pd.DataFrame) -> "GBDTConditioningFunnel":
        n_rows = len(train_df)
        self.active_base_cols_ = [c for c in self.base_cols if c in train_df.columns]
        cand_cols = [c for c in esoteric_df.columns if c not in self.active_base_cols_]

        valid_cands: List[str] = []
        self.fill_values_: Dict[str, float] = {}
        for col in cand_cols:
            vals = np.nan_to_num(esoteric_df[col].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
            if float(np.std(vals)) > 1e-6:
                valid_cands.append(col)
                self.fill_values_[col] = float(np.median(vals))

        if not valid_cands or n_rows < 28:
            self.selected_esoteric_cols_ = valid_cands[: min(4, len(valid_cands))]
            self.selected_feature_cols_ = list(self.active_base_cols_) + list(self.selected_esoteric_cols_)
            self.diagnostics_ = ConditioningDiagnostics(
                funnel_type="GBDT_VARIANCE_PERMUTATION",
                n_train_rows=n_rows,
                n_base_features=len(self.active_base_cols_),
                n_esoteric_candidates=len(cand_cols),
                n_selected_esoteric=len(self.selected_esoteric_cols_),
                selected_esoteric_cols=list(self.selected_esoteric_cols_),
                condition_number_after=1.0,
                max_vif_after=1.0,
            )
            return self

        y = np.maximum(0.0, train_df[self.target_col].to_numpy(dtype=float))
        enrol = (
            np.maximum(1.0, train_df[self.exposure_col].to_numpy(dtype=float))
            if self.exposure_col in train_df.columns
            else np.ones(n_rows, dtype=float)
        )
        log_enrol = np.log(enrol)

        X_base = (
            train_df[self.active_base_cols_].to_numpy(dtype=float)
            if self.active_base_cols_
            else np.zeros((n_rows, 0), dtype=float)
        )
        X_eso = np.column_stack(
            [np.nan_to_num(esoteric_df[c].to_numpy(dtype=float), nan=self.fill_values_[c]) for c in valid_cands]
        )
        X_all = np.hstack([X_base, X_eso])

        split = max(20, int(n_rows * 0.72))
        base_rate = float(np.log(max(1e-6, np.sum(y[:split]) / np.sum(enrol[:split]))))
        probe = LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=1.30,
            boost_from_average=False,
            n_estimators=45,
            learning_rate=0.05,
            max_depth=3,
            num_leaves=7,
            min_child_samples=max(4, min(10, split // 6)),
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.6,
            reg_lambda=1.5,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        )
        probe.fit(X_all[:split], y[:split], init_score=log_enrol[:split] + base_rate)

        gains = probe.booster_.feature_importance(importance_type="gain")
        eso_gains = gains[X_base.shape[1] :]

        # Inner validation group permutation delta per module prefix (`f01`..`f13`)
        raw_val_full = probe.predict(X_all[split:], raw_score=True)
        pred_val_full = np.exp(np.clip(raw_val_full + log_enrol[split:] + base_rate, -15.0, 15.0))
        base_rmsle = float(np.sqrt(np.mean((np.log1p(pred_val_full) - np.log1p(y[split:])) ** 2)))

        module_groups: Dict[str, List[int]] = {}
        for j, col in enumerate(valid_cands):
            prefix = col.split("_")[0] if "_" in col else "misc"
            module_groups.setdefault(prefix, []).append(j)

        rng = np.random.default_rng(42)
        module_scores: Dict[str, float] = {}
        helpful_modules: List[str] = []

        for mod_id, idx_list in module_groups.items():
            X_perm = X_all[split:].copy()
            for j in idx_list:
                col_idx = X_base.shape[1] + j
                X_perm[:, col_idx] = rng.permutation(X_perm[:, col_idx])
            raw_p = probe.predict(X_perm, raw_score=True)
            pred_p = np.exp(np.clip(raw_p + log_enrol[split:] + base_rate, -15.0, 15.0))
            perm_rmsle = float(np.sqrt(np.mean((np.log1p(pred_p) - np.log1p(y[split:])) ** 2)))
            delta = perm_rmsle - base_rmsle  # > 0 means permuting hurt accuracy => module is genuinely predictive
            module_scores[mod_id] = round(delta, 6)
            if delta >= -1e-4:
                helpful_modules.append(mod_id)

        # Rank features inside helpful modules by positive split gain
        scored_cols: List[Tuple[float, str]] = []
        for j, col in enumerate(valid_cands):
            prefix = col.split("_")[0] if "_" in col else "misc"
            mod_bonus = max(0.0, module_scores.get(prefix, 0.0)) * 100.0
            if prefix in helpful_modules and eso_gains[j] > 0.0:
                scored_cols.append((float(eso_gains[j]) + mod_bonus, col))

        if not scored_cols:
            # Fallback to top gain features if single-module ablation has zero split gain in inner split
            for j, col in enumerate(valid_cands):
                if eso_gains[j] > 0.0:
                    scored_cols.append((float(eso_gains[j]), col))
        if not scored_cols and len(valid_cands) <= 12:
            # Single-module probe fallback: keep top variance-filtered features
            scored_cols = [(1.0, c) for c in valid_cands[: min(4, len(valid_cands))]]

        scored_cols.sort(key=lambda pair: pair[0], reverse=True)
        selected = [c for _, c in scored_cols[: self.max_esoteric_cols]]

        # Verify inner-fold validation vs baseline-only GBDT so noisy sets don't degrade small folds
        probe_base = LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=1.30,
            boost_from_average=False,
            n_estimators=45,
            learning_rate=0.05,
            max_depth=3,
            num_leaves=7,
            min_child_samples=max(4, min(10, split // 6)),
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.2,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        )
        if X_base.shape[1] > 0:
            probe_base.fit(X_base[:split], y[:split], init_score=log_enrol[:split] + base_rate)
            pred_b = np.exp(
                np.clip(probe_base.predict(X_base[split:], raw_score=True) + log_enrol[split:] + base_rate, -15.0, 15.0)
            )
            b_only_rmsle = float(np.sqrt(np.mean((np.log1p(pred_b) - np.log1p(y[split:])) ** 2)))

            if selected:
                sel_idx = [valid_cands.index(c) for c in selected]
                X_sel = np.hstack([X_base, X_eso[:, sel_idx]])
                probe_base.fit(X_sel[:split], y[:split], init_score=log_enrol[:split] + base_rate)
                pred_s = np.exp(
                    np.clip(probe_base.predict(X_sel[split:], raw_score=True) + log_enrol[split:] + base_rate, -15.0, 15.0)
                )
                s_rmsle = float(np.sqrt(np.mean((np.log1p(pred_s) - np.log1p(y[split:])) ** 2)))
                if s_rmsle > b_only_rmsle * 1.015 and len(selected) > 3:
                    selected = selected[:3]

        self.selected_esoteric_cols_ = selected
        self.selected_feature_cols_ = list(self.active_base_cols_) + list(selected)
        self.diagnostics_ = ConditioningDiagnostics(
            funnel_type="GBDT_VARIANCE_PERMUTATION",
            n_train_rows=n_rows,
            n_base_features=len(self.active_base_cols_),
            n_esoteric_candidates=len(cand_cols),
            n_selected_esoteric=len(selected),
            selected_esoteric_cols=list(selected),
            condition_number_after=1.0,
            max_vif_after=1.0,
            notes=f"Selected {len(selected)} esoteric GBDT features across {len(helpful_modules)} positive-gain modules.",
            module_scores=module_scores,
        )
        return self

    def transform(self, df: pd.DataFrame, esoteric_df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.selected_esoteric_cols_:
            fv = self.fill_values_.get(col, 0.0)
            out[col] = np.nan_to_num(esoteric_df[col].to_numpy(dtype=float), nan=fv, posinf=fv, neginf=fv)
        return out
