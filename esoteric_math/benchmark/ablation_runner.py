"""
Phase 5: Walk-Forward Scoreboard Ablation Runner (`esoteric_math/benchmark/ablation_runner.py`).

Benchmarks across the app's native expanding-window splits (`fold_origins_weeks = (78, 104, 130)`,
`horizon_weeks = 26`) and native curriculum-area hierarchy:
  1. Untouched Control Baseline Models (`legacy_ml_baseline`, `nb2_glm`, `zinb_model`, `tweedie_lgbm`)
  2. Single-Module Esoteric Additions (`+f01` .. `+f13` evaluated individually on NB2 GLM & Tweedie LightGBM)
  3. Conditioned Full Esoteric Existing Models (`nb2_glm_esoteric`, `zinb_model_esoteric`, `tweedie_lgbm_esoteric`)
  4. Fisher-Rao Natural Gradient Booster (`fisher_rao_ngboost_esoteric`)

Evaluates all native metrics + CRPS:
  - WRMSSE, RMSSE, RMSLE, RMSE, MAE, Surge MAE
  - Pinball Loss (0.025, 0.50, 0.90, 0.975), SPL, WSPL, PICP_95, MIW_95
  - NB Deviance, Tweedie Deviance (p=1.3), and Continuous Ranked Probability Score (CRPS)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence
import numpy as np
import pandas as pd
from scipy import stats

from app.evaluation.kaggle_metrics import (
    EVAL_QUANTILES,
    compute_mae,
    compute_mean_nb_deviance,
    compute_mean_tweedie_deviance_safe,
    compute_picp_and_miw,
    compute_pinball_loss,
    compute_rmsle,
    compute_rmse,
    compute_rmsse,
    compute_spl,
    compute_surge_mae,
    compute_wrmsse,
    compute_wspl,
)
from app.features import (
    aggregate_college_wide_weekly,
    engineer_safeguarding_features,
    generate_weekly_safeguarding_panel,
)
from app.models.safeguarding_benchmark import (
    LegacyMLBaselineModel,
    ModelPredictionBundle,
    NB2GovernanceGLM,
    TweedieLightGBMModel,
    ZINBSparseRegimeModel,
)
from esoteric_math.conditioning import GBDTConditioningFunnel, GLMConditioningFunnel
from esoteric_math.features import FEATURE_ENGINE_REGISTRY, EsotericFeatureSuite
from esoteric_math.models.fisher_rao_ngboost import FisherRaoNaturalGradientBooster
from esoteric_math.schema_bridge import attach_panel_hierarchy_wide


def compute_crps_from_quantiles(
    y_true: Sequence[float],
    point_mu: Sequence[float],
    quantiles_map: Mapping[float, Sequence[float]],
    alpha_disp: float = 0.18,
) -> float:
    """
    Compute Continuous Ranked Probability Score (CRPS) via dense 19-quantile pinball
    integration: CRPS(F, y) = 2 * int_0^1 L_tau(y, F^{-1}(tau)) d_tau.
    Interpolates available predictive quantiles with the calibrated NB2 tail curve.
    """
    yt = np.maximum(0.0, np.asarray(y_true, dtype=float))
    mu = np.maximum(1e-5, np.asarray(point_mu, dtype=float))
    taus_dense = np.linspace(0.05, 0.95, 19)

    # Calibrate local dispersion scale from predicted Q975 - Q50 spread where available
    if 0.975 in quantiles_map and 0.50 in quantiles_map:
        q50 = np.asarray(quantiles_map[0.50], dtype=float)
        q975 = np.asarray(quantiles_map[0.975], dtype=float)
        spread = np.maximum(0.5, (q975 - q50) / 1.96)
        var_eff = np.maximum(mu + 1e-4, spread ** 2)
        alpha_vec = np.clip((var_eff - mu) / np.maximum(mu ** 2, 1e-4), 0.02, 1.5)
    else:
        alpha_vec = np.full_like(mu, max(0.02, float(alpha_disp)))

    n_param = 1.0 / alpha_vec
    p_param = n_param / (n_param + mu)

    pinball_losses: List[float] = []
    for tau in taus_dense:
        if float(tau) in quantiles_map:
            q_tau = np.asarray(quantiles_map[float(tau)], dtype=float)
        else:
            q_tau = np.maximum(0.0, stats.nbinom.ppf(float(tau), n_param, p_param))
            q_tau = np.where(np.isfinite(q_tau), q_tau, mu)
        diff = yt - q_tau
        loss_t = np.mean(np.maximum(tau * diff, (tau - 1.0) * diff))
        pinball_losses.append(float(loss_t))

    return float(2.0 * np.mean(pinball_losses))


def _assemble_scoreboard_row(
    variant_id: str,
    variant_label: str,
    category_group: str,
    yt: List[float],
    yp: List[float],
    q_map: Dict[float, List[float]],
    autumn_flags: List[int],
    pre_hol_flags: List[int],
    post_hol_flags: List[int],
    rmsse_folds: List[float],
    spl_folds: List[float],
    dept_rmsse: Dict[str, float],
    dept_spl: Dict[str, float],
    dept_weights: Dict[str, float],
    alpha_ref: float,
) -> Dict[str, Any]:
    picp_95, miw_95 = compute_picp_and_miw(yt, q_map[0.025], q_map[0.975])
    wrmsse_v = compute_wrmsse(dept_rmsse, dept_weights) if dept_rmsse else float(np.mean(rmsse_folds))
    wspl_v = compute_wspl(dept_spl, dept_weights) if dept_spl else float(np.mean(spl_folds))
    crps_v = compute_crps_from_quantiles(yt, yp, q_map, alpha_disp=alpha_ref)

    return {
        "variant_id": variant_id,
        "variant_label": variant_label,
        "group": category_group,
        "wrmsse": round(wrmsse_v, 4),
        "rmsse": round(float(np.mean(rmsse_folds)), 4),
        "rmsle": round(compute_rmsle(yt, yp), 4),
        "rmse": round(compute_rmse(yt, yp), 3),
        "mae": round(compute_mae(yt, yp), 3),
        "surge_mae": round(compute_surge_mae(yt, yp, autumn_flags, pre_hol_flags, post_hol_flags), 3),
        "nb_deviance": round(compute_mean_nb_deviance(yt, yp, alpha=alpha_ref), 4),
        "tweedie_deviance": round(compute_mean_tweedie_deviance_safe(yt, yp, power=1.30), 4),
        "pinball_500": round(compute_pinball_loss(yt, q_map[0.50], 0.50), 4),
        "pinball_900": round(compute_pinball_loss(yt, q_map[0.90], 0.90), 4),
        "pinball_975": round(compute_pinball_loss(yt, q_map[0.975], 0.975), 4),
        "spl": round(float(np.mean(spl_folds)), 4),
        "wspl": round(wspl_v, 4),
        "crps": round(crps_v, 4),
        "picp_95": round(picp_95, 2),
        "miw_95": round(miw_95, 2),
    }


def run_walk_forward_ablation(
    fold_origins_weeks: Sequence[int] = (78, 104, 130),
    horizon_weeks: int = 26,
    output_json_path: Optional[str | Path] = "artifacts/esoteric_ablation_scoreboard.json",
) -> Dict[str, Any]:
    """
    Execute the complete Phase 5 walk-forward ablation benchmark across:
      - Control Baseline Models (untouched)
      - +f01 .. +f13 Single-Module Esoteric Additions
      - Conditioned Full Esoteric Models (NB2, ZINB, Tweedie LightGBM)
      - Fisher-Rao Natural Gradient Booster
    """
    panel_df = generate_weekly_safeguarding_panel(num_academic_years=3, start_year=2022, seed=42)
    cw_raw = aggregate_college_wide_weekly(panel_df)
    cw_df, vif_diag = engineer_safeguarding_features(cw_raw, group_col="curriculum_area")
    dept_df, _ = engineer_safeguarding_features(panel_df, group_col="curriculum_area")
    cw_wide = attach_panel_hierarchy_wide(cw_df, panel_df)

    base_cols = vif_diag.selected_feature_cols
    valid_origins = [o for o in fold_origins_weeks if 30 <= o <= (len(cw_df) - horizon_weeks)]
    final_origin = valid_origins[-1]

    # Precompute causal esoteric feature matrices per fold origin (strictly fitted on :origin!)
    fold_eso_frames: Dict[int, pd.DataFrame] = {}
    for origin in valid_origins:
        suite_fold = EsotericFeatureSuite(
            causal_mode="forecast",
            train_cutoff=origin,
        )
        suite_fold.fit(cw_wide.iloc[:origin])
        fold_eso_frames[origin] = suite_fold.transform(cw_wide.iloc[: origin + horizon_weeks]).reset_index(drop=True)

    # Also precompute departmental causal esoteric frames for the final hierarchical fold (WRMSSE & WSPL)
    dept_splits: Dict[str, Dict[str, pd.DataFrame]] = {}
    dept_weights: Dict[str, float] = {}
    for area_name, g_area in dept_df.groupby("curriculum_area"):
        if area_name == "College-Wide":
            continue
        g_sorted = g_area.sort_values("week_start").reset_index(drop=True)
        if len(g_sorted) < final_origin + horizon_weeks:
            continue
        tr_a = g_sorted.iloc[:final_origin].copy().reset_index(drop=True)
        te_a = g_sorted.iloc[final_origin : final_origin + horizon_weeks].copy().reset_index(drop=True)
        dept_weights[str(area_name)] = float(tr_a["active_enrolments"].mean())

        suite_a = EsotericFeatureSuite(causal_mode="forecast", train_cutoff=final_origin)
        suite_a.fit(tr_a)
        eso_a = suite_a.transform(g_sorted.iloc[: final_origin + horizon_weeks]).reset_index(drop=True)
        dept_splits[str(area_name)] = {
            "tr": tr_a,
            "te": te_a,
            "eso_tr": eso_a.iloc[:final_origin].reset_index(drop=True),
            "eso_te": eso_a.iloc[final_origin : final_origin + horizon_weeks].reset_index(drop=True),
        }

    # Define main contender specifications
    contender_ids = [
        "control_legacy_ml",
        "control_nb2_glm",
        "control_zinb",
        "control_tweedie_lgbm",
        "esoteric_nb2_glm",
        "esoteric_zinb",
        "esoteric_tweedie_lgbm",
        "fisher_rao_ngboost",
    ]
    contender_labels = {
        "control_legacy_ml": "Control: Legacy ML Baseline (Unconstrained L2 GBDT)",
        "control_nb2_glm": "Control: NB2 Governance GLM (Locked Baseline)",
        "control_zinb": "Control: ZINB Two-Part Regime (Locked Baseline)",
        "control_tweedie_lgbm": "Control: Tweedie-Loss LightGBM (Locked Baseline)",
        "esoteric_nb2_glm": "Augmented: Conditioned Full-Esoteric NB2 GLM",
        "esoteric_zinb": "Augmented: Conditioned Full-Esoteric ZINB",
        "esoteric_tweedie_lgbm": "Augmented: Conditioned Full-Esoteric Tweedie LightGBM",
        "fisher_rao_ngboost": "Contender: Fisher-Rao Natural Gradient NB Booster",
    }
    contender_groups = {
        "control_legacy_ml": "Control",
        "control_nb2_glm": "Control",
        "control_zinb": "Control",
        "control_tweedie_lgbm": "Control",
        "esoteric_nb2_glm": "Conditioned Full Esoteric",
        "esoteric_zinb": "Conditioned Full Esoteric",
        "esoteric_tweedie_lgbm": "Conditioned Full Esoteric",
        "fisher_rao_ngboost": "Natural Gradient Manifold Contender",
    }

    yt_acc: Dict[str, List[float]] = {k: [] for k in contender_ids}
    yp_acc: Dict[str, List[float]] = {k: [] for k in contender_ids}
    q_acc: Dict[str, Dict[float, List[float]]] = {
        k: {q: [] for q in EVAL_QUANTILES} for k in contender_ids
    }
    rmsse_acc: Dict[str, List[float]] = {k: [] for k in contender_ids}
    spl_acc: Dict[str, List[float]] = {k: [] for k in contender_ids}
    autumn_all: List[int] = []
    pre_hol_all: List[int] = []
    post_hol_all: List[int] = []

    alpha_ref = 0.18
    conditioning_logs: Dict[int, Dict[str, Any]] = {}

    for origin in valid_origins:
        tr_df = cw_df.iloc[:origin].copy().reset_index(drop=True)
        te_df = cw_df.iloc[origin : origin + horizon_weeks].copy().reset_index(drop=True)
        eso_full = fold_eso_frames[origin]
        eso_tr = eso_full.iloc[:origin].reset_index(drop=True)
        eso_te = eso_full.iloc[origin : origin + horizon_weeks].reset_index(drop=True)

        y_tr = tr_df["referral_count"].to_numpy(dtype=float)
        y_te = te_df["referral_count"].to_numpy(dtype=float)
        autumn_all.extend(te_df["is_autumn_transition"].astype(int).tolist())
        pre_hol_all.extend(te_df["is_pre_holiday_week"].astype(int).tolist())
        post_hol_all.extend(te_df["is_post_holiday_week"].astype(int).tolist())

        # Fit conditioning funnels strictly on training slice
        glm_funnel = GLMConditioningFunnel(base_cols=base_cols).fit(tr_df, eso_tr)
        gbdt_funnel = GBDTConditioningFunnel(base_cols=base_cols).fit(tr_df, eso_tr)
        conditioning_logs[origin] = {
            "glm_selected": glm_funnel.selected_esoteric_cols_,
            "glm_max_vif": glm_funnel.diagnostics_.max_vif_after,
            "glm_cond": glm_funnel.diagnostics_.condition_number_after,
            "gbdt_selected": gbdt_funnel.selected_esoteric_cols_,
        }

        tr_glm = glm_funnel.transform(tr_df, eso_tr)
        te_glm = glm_funnel.transform(te_df, eso_te)
        tr_gbdt = gbdt_funnel.transform(tr_df, eso_tr)
        te_gbdt = gbdt_funnel.transform(te_df, eso_te)

        # 1. Control models (untouched)
        c_leg = LegacyMLBaselineModel(feature_cols=base_cols).fit(tr_df)
        c_nb2 = NB2GovernanceGLM(feature_cols=base_cols).fit(tr_df)
        c_zinb = ZINBSparseRegimeModel(feature_cols=base_cols).fit(tr_df)
        c_tw = TweedieLightGBMModel(feature_cols=base_cols).fit(tr_df)
        alpha_ref = float(c_nb2.alpha_)

        # 2. Conditioned Full Esoteric Existing Models + Fisher-Rao Natural Gradient Booster
        e_nb2 = NB2GovernanceGLM(feature_cols=glm_funnel.selected_feature_cols_).fit(tr_glm)
        e_zinb = ZINBSparseRegimeModel(feature_cols=glm_funnel.selected_feature_cols_).fit(tr_glm)
        e_tw = TweedieLightGBMModel(feature_cols=gbdt_funnel.selected_feature_cols_).fit(tr_gbdt)
        f_ngb = FisherRaoNaturalGradientBooster(feature_cols=gbdt_funnel.selected_feature_cols_).fit(tr_gbdt)

        fold_preds: Dict[str, ModelPredictionBundle] = {
            "control_legacy_ml": c_leg.predict(te_df),
            "control_nb2_glm": c_nb2.predict(te_df),
            "control_zinb": c_zinb.predict(te_df),
            "control_tweedie_lgbm": c_tw.predict(te_df),
            "esoteric_nb2_glm": e_nb2.predict(te_glm),
            "esoteric_zinb": e_zinb.predict(te_glm),
            "esoteric_tweedie_lgbm": e_tw.predict(te_gbdt),
            "fisher_rao_ngboost": f_ngb.predict(te_gbdt),
        }

        for cid, bnd in fold_preds.items():
            yt_acc[cid].extend(y_te.tolist())
            yp_acc[cid].extend(bnd.point_forecasts.tolist())
            for q in EVAL_QUANTILES:
                q_acc[cid][q].extend(bnd.quantiles[q].tolist())
            rmsse_acc[cid].append(compute_rmsse(y_tr, y_te, bnd.point_forecasts))
            spl_acc[cid].append(compute_spl(y_tr, y_te, bnd.quantiles))

    # Hierarchical curriculum-area evaluation for WRMSSE & WSPL on final fold
    dept_rmsse_map: Dict[str, Dict[str, float]] = {k: {} for k in contender_ids}
    dept_spl_map: Dict[str, Dict[str, float]] = {k: {} for k in contender_ids}

    for area_name, sp in dept_splits.items():
        tr_a, te_a, eso_tr_a, eso_te_a = sp["tr"], sp["te"], sp["eso_tr"], sp["eso_te"]
        y_tr_a = tr_a["referral_count"].to_numpy(dtype=float)
        y_te_a = te_a["referral_count"].to_numpy(dtype=float)

        gf_a = GLMConditioningFunnel(base_cols=base_cols).fit(tr_a, eso_tr_a)
        bf_a = GBDTConditioningFunnel(base_cols=base_cols).fit(tr_a, eso_tr_a)
        tr_a_glm, te_a_glm = gf_a.transform(tr_a, eso_tr_a), gf_a.transform(te_a, eso_te_a)
        tr_a_gbdt, te_a_gbdt = bf_a.transform(tr_a, eso_tr_a), bf_a.transform(te_a, eso_te_a)

        area_preds: Dict[str, ModelPredictionBundle] = {
            "control_legacy_ml": LegacyMLBaselineModel(feature_cols=base_cols).fit(tr_a).predict(te_a),
            "control_nb2_glm": NB2GovernanceGLM(feature_cols=base_cols).fit(tr_a).predict(te_a),
            "control_zinb": ZINBSparseRegimeModel(feature_cols=base_cols).fit(tr_a).predict(te_a),
            "control_tweedie_lgbm": TweedieLightGBMModel(feature_cols=base_cols, tune_power=False).fit(tr_a).predict(te_a),
            "esoteric_nb2_glm": NB2GovernanceGLM(feature_cols=gf_a.selected_feature_cols_).fit(tr_a_glm).predict(te_a_glm),
            "esoteric_zinb": ZINBSparseRegimeModel(feature_cols=gf_a.selected_feature_cols_).fit(tr_a_glm).predict(te_a_glm),
            "esoteric_tweedie_lgbm": TweedieLightGBMModel(feature_cols=bf_a.selected_feature_cols_, tune_power=False).fit(tr_a_gbdt).predict(te_a_gbdt),
            "fisher_rao_ngboost": FisherRaoNaturalGradientBooster(feature_cols=bf_a.selected_feature_cols_).fit(tr_a_gbdt).predict(te_a_gbdt),
        }
        for cid, b_a in area_preds.items():
            dept_rmsse_map[cid][area_name] = compute_rmsse(y_tr_a, y_te_a, b_a.point_forecasts)
            dept_spl_map[cid][area_name] = compute_spl(y_tr_a, y_te_a, b_a.quantiles)

    main_scoreboard: List[Dict[str, Any]] = []
    for cid in contender_ids:
        main_scoreboard.append(
            _assemble_scoreboard_row(
                variant_id=cid,
                variant_label=contender_labels[cid],
                category_group=contender_groups[cid],
                yt=yt_acc[cid],
                yp=yp_acc[cid],
                q_map=q_acc[cid],
                autumn_flags=autumn_all,
                pre_hol_flags=pre_hol_all,
                post_hol_flags=post_hol_all,
                rmsse_folds=rmsse_acc[cid],
                spl_folds=spl_acc[cid],
                dept_rmsse=dept_rmsse_map[cid],
                dept_spl=dept_spl_map[cid],
                dept_weights=dept_weights,
                alpha_ref=alpha_ref,
            )
        )

    # Single-module `+f01` .. `+f13` walk-forward additions (evaluated on NB2 GLM and Tweedie LightGBM)
    single_module_rows: List[Dict[str, Any]] = []
    ctrl_nb2_row = next(r for r in main_scoreboard if r["variant_id"] == "control_nb2_glm")
    ctrl_tw_row = next(r for r in main_scoreboard if r["variant_id"] == "control_tweedie_lgbm")

    for mod_id, mod_cls in FEATURE_ENGINE_REGISTRY.items():
        yt_m: List[float] = []
        yp_nb2_m: List[float] = []
        yp_tw_m: List[float] = []
        q_nb2_m: Dict[float, List[float]] = {q: [] for q in EVAL_QUANTILES}
        q_tw_m: Dict[float, List[float]] = {q: [] for q in EVAL_QUANTILES}
        rmsse_nb2_m: List[float] = []
        rmsse_tw_m: List[float] = []
        spl_nb2_m: List[float] = []
        spl_tw_m: List[float] = []

        for origin in valid_origins:
            tr_df = cw_df.iloc[:origin].copy().reset_index(drop=True)
            te_df = cw_df.iloc[origin : origin + horizon_weeks].copy().reset_index(drop=True)
            eso_full = fold_eso_frames[origin]
            mod_cols = [c for c in eso_full.columns if c.startswith(f"{mod_id}_")]
            eso_tr_m = eso_full[mod_cols].iloc[:origin].reset_index(drop=True)
            eso_te_m = eso_full[mod_cols].iloc[origin : origin + horizon_weeks].reset_index(drop=True)

            gf_m = GLMConditioningFunnel(base_cols=base_cols, max_esoteric_cols=1).fit(tr_df, eso_tr_m)
            bf_m = GBDTConditioningFunnel(base_cols=base_cols, max_esoteric_cols=4).fit(tr_df, eso_tr_m)

            p_nb2 = NB2GovernanceGLM(feature_cols=gf_m.selected_feature_cols_).fit(
                gf_m.transform(tr_df, eso_tr_m)
            ).predict(gf_m.transform(te_df, eso_te_m))
            p_tw = TweedieLightGBMModel(feature_cols=bf_m.selected_feature_cols_, tune_power=False).fit(
                bf_m.transform(tr_df, eso_tr_m)
            ).predict(bf_m.transform(te_df, eso_te_m))

            y_tr = tr_df["referral_count"].to_numpy(dtype=float)
            y_te = te_df["referral_count"].to_numpy(dtype=float)
            yt_m.extend(y_te.tolist())
            yp_nb2_m.extend(p_nb2.point_forecasts.tolist())
            yp_tw_m.extend(p_tw.point_forecasts.tolist())
            for q in EVAL_QUANTILES:
                q_nb2_m[q].extend(p_nb2.quantiles[q].tolist())
                q_tw_m[q].extend(p_tw.quantiles[q].tolist())
            rmsse_nb2_m.append(compute_rmsse(y_tr, y_te, p_nb2.point_forecasts))
            rmsse_tw_m.append(compute_rmsse(y_tr, y_te, p_tw.point_forecasts))
            spl_nb2_m.append(compute_spl(y_tr, y_te, p_nb2.quantiles))
            spl_tw_m.append(compute_spl(y_tr, y_te, p_tw.quantiles))

        nb2_rmsle = compute_rmsle(yt_m, yp_nb2_m)
        tw_rmsle = compute_rmsle(yt_m, yp_tw_m)
        nb2_crps = compute_crps_from_quantiles(yt_m, yp_nb2_m, q_nb2_m, alpha_disp=alpha_ref)
        tw_crps = compute_crps_from_quantiles(yt_m, yp_tw_m, q_tw_m, alpha_disp=alpha_ref)

        single_module_rows.append(
            {
                "module_id": mod_id,
                "module_name": mod_cls.module_name,
                "nb2_rmsse": round(float(np.mean(rmsse_nb2_m)), 4),
                "nb2_rmsle": round(nb2_rmsle, 4),
                "nb2_rmse": round(compute_rmse(yt_m, yp_nb2_m), 3),
                "nb2_nb_deviance": round(compute_mean_nb_deviance(yt_m, yp_nb2_m, alpha=alpha_ref), 4),
                "nb2_crps": round(nb2_crps, 4),
                "nb2_rmsle_delta": round(nb2_rmsle - ctrl_nb2_row["rmsle"], 4),
                "tweedie_rmsse": round(float(np.mean(rmsse_tw_m)), 4),
                "tweedie_rmsle": round(tw_rmsle, 4),
                "tweedie_rmse": round(compute_rmse(yt_m, yp_tw_m), 3),
                "tweedie_spl": round(float(np.mean(spl_tw_m)), 4),
                "tweedie_crps": round(tw_crps, 4),
                "tweedie_rmsle_delta": round(tw_rmsle - ctrl_tw_row["rmsle"], 4),
            }
        )

    report = {
        "expanding_window_folds": list(valid_origins),
        "horizon_weeks": horizon_weeks,
        "main_scoreboard": main_scoreboard,
        "single_module_ablations": single_module_rows,
        "conditioning_diagnostics_by_fold": {str(k): v for k, v in conditioning_logs.items()},
    }

    if output_json_path is not None:
        out_p = Path(output_json_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


if __name__ == "__main__":
    res = run_walk_forward_ablation()
    print(json.dumps(res["main_scoreboard"], indent=2))
