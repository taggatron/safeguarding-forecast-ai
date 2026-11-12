"""
Phase 5 Mandatory Verification Gate 1: Frozen Baseline Immutability & Zero-Harm Identity Floor
(`tests/test_frozen_baseline_immutability.py`).

Verifies:
  1. Cryptographic IEEE-754 SHA-256 digests of all 4 core baseline estimators
     (`LegacyMLBaselineModel`, `NB2GovernanceGLM`, `ZINBSparseRegimeModel`, `TweedieLightGBMModel`)
     and their out-of-sample `ModelPredictionBundle` outputs match pre-patch bit-for-bit
     after running `PostHocResidualStacker` across all 3 `RolloutMode`s.
  2. Calling `.fit()` on `FrozenBaselineAdapter` or mutating an underlying estimator raises
     `FrozenBaselineMutationError`.
  3. `RETRO_MODE="TELEMETRY_ONLY"` reproduces frozen baseline point, quantile, and interval
     predictions to `atol=1e-12`.
  4. Pure-noise esoteric inputs trigger the anti-bloat quarantine gate (`BlockBorutaSHAPGate` +
     `GreedyOneSEModuleSelector`) and fall back to exact frozen baseline identity (`atol=1e-12`).
  5. `ZINBSparseRegimeModel` structural zero-inflation probability `pi_zero(t)` remains locked
     bit-for-bit (`atol=1e-12`) even when active count residual offsets are applied.
"""
from __future__ import annotations

import copy
import numpy as np
import pandas as pd
import pytest

from app.evaluation.kaggle_metrics import EVAL_QUANTILES
from app.features import (
    aggregate_college_wide_weekly,
    engineer_safeguarding_features,
    generate_weekly_safeguarding_panel,
)
from app.models.safeguarding_benchmark import (
    LegacyMLBaselineModel,
    NB2GovernanceGLM,
    TweedieLightGBMModel,
    ZINBSparseRegimeModel,
)
from esoteric_math.features import EsotericFeatureSuite
from esoteric_math.retrospective import (
    FrozenBaselineAdapter,
    FrozenBaselineMutationError,
    PostHocResidualStacker,
    RolloutConfig,
    RolloutMode,
    compute_model_state_sha256,
    compute_prediction_bundle_sha256,
)
from esoteric_math.schema_bridge import attach_panel_hierarchy_wide


@pytest.fixture(scope="module")
def weekly_split_fixture():
    panel_df = generate_weekly_safeguarding_panel(num_academic_years=3, start_year=2022, seed=42)
    cw_raw = aggregate_college_wide_weekly(panel_df)
    cw_df, vif_diag = engineer_safeguarding_features(cw_raw, group_col="curriculum_area")
    cw_wide = attach_panel_hierarchy_wide(cw_df, panel_df)

    origin = 104
    horizon = 26
    tr_df = cw_df.iloc[:origin].copy().reset_index(drop=True)
    te_df = cw_df.iloc[origin : origin + horizon].copy().reset_index(drop=True)

    suite = EsotericFeatureSuite(causal_mode="forecast", train_cutoff=origin)
    suite.fit(cw_wide.iloc[:origin])
    eso_all = suite.transform(cw_wide.iloc[: origin + horizon]).reset_index(drop=True)
    eso_tr = eso_all.iloc[:origin].reset_index(drop=True)
    eso_te = eso_all.iloc[origin : origin + horizon].reset_index(drop=True)

    return {
        "base_cols": vif_diag.selected_feature_cols,
        "tr_df": tr_df,
        "te_df": te_df,
        "eso_tr": eso_tr,
        "eso_te": eso_te,
    }


def test_all_four_core_baselines_sha256_bit_for_bit_immutable_and_telemetry_only_exact(
    weekly_split_fixture,
) -> None:
    base_cols = weekly_split_fixture["base_cols"]
    tr_df = weekly_split_fixture["tr_df"]
    te_df = weekly_split_fixture["te_df"]
    eso_tr = weekly_split_fixture["eso_tr"]
    eso_te = weekly_split_fixture["eso_te"]

    estimators = [
        LegacyMLBaselineModel(feature_cols=base_cols).fit(tr_df),
        NB2GovernanceGLM(feature_cols=base_cols).fit(tr_df),
        ZINBSparseRegimeModel(feature_cols=base_cols).fit(tr_df),
        TweedieLightGBMModel(feature_cols=base_cols).fit(tr_df),
    ]

    for est in estimators:
        pre_model_sha = compute_model_state_sha256(est)
        pre_bundle = est.predict(te_df, quantiles=EVAL_QUANTILES)
        pre_pred_sha = compute_prediction_bundle_sha256(pre_bundle)

        frozen = FrozenBaselineAdapter(est)

        # Calling .fit() on FrozenBaselineAdapter must be strictly blocked
        with pytest.raises(FrozenBaselineMutationError):
            frozen.fit(tr_df)

        # Run PostHocResidualStacker across all 3 RolloutModes
        for mode in (RolloutMode.TELEMETRY_ONLY, RolloutMode.STAGED_CORE_4, RolloutMode.FULL_13_GATED):
            cfg = RolloutConfig.from_mode(mode)
            stacker = PostHocResidualStacker(frozen, config=cfg, random_state=42).fit(tr_df, eso_tr)
            out_bundle = stacker.predict(te_df, eso_te, quantiles=EVAL_QUANTILES, tag="oos_holdout")

            if mode == RolloutMode.TELEMETRY_ONLY:
                assert stacker.identity_floor_active_ is True
                assert np.allclose(out_bundle.point_forecasts, pre_bundle.point_forecasts, atol=1e-12, rtol=0.0)
                for tau in EVAL_QUANTILES:
                    assert np.allclose(out_bundle.quantiles[tau], pre_bundle.quantiles[tau], atol=1e-12, rtol=0.0)
                assert compute_prediction_bundle_sha256(out_bundle) == pre_pred_sha

        # Re-verify original estimator state & OOS predictions match pre-patch SHA-256 bit-for-bit
        post_model_sha = compute_model_state_sha256(est)
        post_pred_sha = compute_prediction_bundle_sha256(est.predict(te_df, quantiles=EVAL_QUANTILES))
        assert post_model_sha == pre_model_sha
        assert post_pred_sha == pre_pred_sha
        chk = frozen.verify_all_checksums()
        assert chk["immutable_verified"] is True
        assert chk["model_state_sha256"] == pre_model_sha
        assert chk["verified_prediction_hashes"]["oos_holdout"] == pre_pred_sha


def test_tampering_detection_raises_frozen_baseline_mutation_error(weekly_split_fixture) -> None:
    base_cols = weekly_split_fixture["base_cols"]
    tr_df = weekly_split_fixture["tr_df"]
    nb2 = NB2GovernanceGLM(feature_cols=base_cols).fit(tr_df)
    nb2_clone = copy.deepcopy(nb2)
    frozen = FrozenBaselineAdapter(nb2_clone)

    # Deliberately tamper with dispersion scalar on the original reference
    nb2_clone.alpha_ = float(nb2_clone.alpha_) + 0.05
    with pytest.raises(FrozenBaselineMutationError):
        frozen.verify_model_immutability()


def test_pure_noise_esoteric_matrix_quarantined_to_exact_identity_floor(weekly_split_fixture) -> None:
    base_cols = weekly_split_fixture["base_cols"]
    tr_df = weekly_split_fixture["tr_df"]
    te_df = weekly_split_fixture["te_df"]

    rng = np.random.default_rng(2026)
    n_tr, n_te = len(tr_df), len(te_df)
    noise_cols = [f"f{(i % 13) + 1:02d}_pure_noise_{i}" for i in range(26)]
    # Autocorrelated AR(1) Gaussian noise with zero relationship to safeguarding counts
    ar_tr = np.zeros((n_tr, len(noise_cols)), dtype=float)
    ar_te = np.zeros((n_te, len(noise_cols)), dtype=float)
    for j in range(len(noise_cols)):
        eps_tr = rng.normal(0.0, 1.0, size=n_tr)
        eps_te = rng.normal(0.0, 1.0, size=n_te)
        for t in range(1, n_tr):
            ar_tr[t, j] = 0.65 * ar_tr[t - 1, j] + eps_tr[t]
        for t in range(1, n_te):
            ar_te[t, j] = 0.65 * ar_te[t - 1, j] + eps_te[t]

    X_noise_tr = pd.DataFrame(ar_tr, columns=noise_cols)
    X_noise_te = pd.DataFrame(ar_te, columns=noise_cols)

    for est in (
        NB2GovernanceGLM(feature_cols=base_cols).fit(tr_df),
        TweedieLightGBMModel(feature_cols=base_cols).fit(tr_df),
    ):
        frozen = FrozenBaselineAdapter(est)
        base_out = frozen.predict(te_df, quantiles=EVAL_QUANTILES)

        for mode in (RolloutMode.STAGED_CORE_4, RolloutMode.FULL_13_GATED):
            stacker = PostHocResidualStacker(
                frozen,
                config=RolloutConfig.from_mode(mode),
                random_state=99,
            ).fit(tr_df, X_noise_tr)
            adj_out = stacker.predict(te_df, X_noise_te, quantiles=EVAL_QUANTILES)

            assert stacker.identity_floor_active_ is True
            assert stacker.audit_ is not None and len(stacker.audit_.admitted_modules) == 0
            assert np.allclose(adj_out.point_forecasts, base_out.point_forecasts, atol=1e-12, rtol=0.0)
            for tau in EVAL_QUANTILES:
                assert np.allclose(adj_out.quantiles[tau], base_out.quantiles[tau], atol=1e-12, rtol=0.0)


def test_zinb_structural_zero_inflation_locked_under_active_residual_offset(
    weekly_split_fixture,
) -> None:
    """
    Construct a synthetic planted esoteric signal orthogonal to baseline predictors so
    `PostHocResidualStacker` activates (`identity_floor_active_ == False`), and verify:
      - ZINB structural zero probability `pi_zero(t)` is 100% untouched (`atol=1e-12`)
      - Adjusted mean equals `(1 - pi_zero(t)) * mu_count_base(t) * exp(delta_eta(t))`.
    """
    base_cols = weekly_split_fixture["base_cols"]
    tr_df = weekly_split_fixture["tr_df"].copy()
    te_df = weekly_split_fixture["te_df"].copy()

    rng = np.random.default_rng(7)
    s_tr = np.sin(np.linspace(0, 6.0 * np.pi, len(tr_df))) + 0.15 * rng.normal(size=len(tr_df))
    s_te = np.sin(np.linspace(6.0 * np.pi, 7.5 * np.pi, len(te_df)))

    # Fit ZINB on baseline tr_df first, then inject planted multiplicative wave into target counts
    zinb_raw = ZINBSparseRegimeModel(feature_cols=base_cols).fit(tr_df)
    frozen_zinb = FrozenBaselineAdapter(zinb_raw)
    pi_before, mu_c_before, alpha_before = frozen_zinb.extract_zinb_structural_components(te_df)

    tr_planted = tr_df.copy()
    base_mu_tr = frozen_zinb.predict(tr_planted).point_forecasts
    tr_planted["referral_count"] = np.round(np.maximum(0.0, base_mu_tr * np.exp(0.42 * s_tr)))

    X_sig_tr = pd.DataFrame(
        {
            "f01_planted_harmonic": s_tr,
            "f03_planted_cascade": np.roll(s_tr, 1),
        }
    )
    X_sig_te = pd.DataFrame(
        {
            "f01_planted_harmonic": s_te,
            "f03_planted_cascade": np.roll(s_te, 1),
        }
    )

    stacker = PostHocResidualStacker(
        frozen_zinb,
        config=RolloutConfig.from_mode(RolloutMode.STAGED_CORE_4, enable_fisher_rao_calibrator=False),
        random_state=42,
    ).fit(tr_planted, X_sig_tr)
    assert stacker.identity_floor_active_ is False

    adj_bundle = stacker.predict(te_df, X_sig_te)
    pi_after, mu_c_after, alpha_after = frozen_zinb.extract_zinb_structural_components(te_df)

    # Structural zero probability and base conditional count must be 100% unchanged!
    assert np.allclose(pi_after, pi_before, atol=1e-12, rtol=0.0)
    assert np.allclose(mu_c_after, mu_c_before, atol=1e-12, rtol=0.0)
    assert abs(alpha_after - alpha_before) <= 1e-12

    delta_eta = np.asarray(adj_bundle.diagnostics["retrospective"]["delta_eta"], dtype=float)
    expected_mu_final = (1.0 - pi_before) * mu_c_before * np.exp(delta_eta)
    assert np.allclose(adj_bundle.point_forecasts, expected_mu_final, atol=1e-4)
    frozen_zinb.verify_all_checksums()

