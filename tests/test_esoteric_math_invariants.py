"""
Phase 5 Verification Gate 2: Algebraic, Geometric & Spectral Invariants
(`tests/test_esoteric_math_invariants.py`).

Verifies:
  1. p-adic Ultrametric Strong Triangle Inequality: d_p(x, z) <= max(d_p(x, y), d_p(y, z))
  2. Depth-3 Path Signature, Levy Area & Log-Signature Monotone Time-Warp Invariance
  3. Cellular Sheaf Laplacian Positive Semi-Definiteness (x^T L_F x >= -1e-10) & Symmetry
  4. Marchenko-Pastur i.i.d. Spectral Bounds
  5. Subcritical Hawkes Intensity Positivity (lambda(t) > 0) & Spectral Radius (rho(Gamma) < 1)
  6. Fisher-Rao Negative Binomial Metric Positivity (g_mumu > 0, g_rr > 0) & Natural Gradient
     Booster Exact 2x Exposure Proportionality + Quantile Monotonicity
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.features import (
    aggregate_college_wide_weekly,
    engineer_safeguarding_features,
    generate_weekly_safeguarding_panel,
)
from esoteric_math.features.f03_hawkes_process import HawkesContagionTransformer
from esoteric_math.features.f07_rough_signatures import (
    compute_levy_area_matrix,
    compute_log_signature_depth3,
    compute_path_signature_depth3,
)
from esoteric_math.features.f08_cellular_sheaves import CellularSheafCohomologyTransformer
from esoteric_math.features.f09_padic_poincare import build_padic_ultrametric_matrix, poincare_distance
from esoteric_math.features.f11_rmt_marchenko import analyze_marchenko_pastur_window
from esoteric_math.features.f12_fisher_rao import nb_fisher_information_diagonal
from esoteric_math.models.fisher_rao_ngboost import FisherRaoNaturalGradientBooster
from esoteric_math.schema_bridge import build_instructional_day_telemetry_frame


def test_padic_ultrametric_strong_triangle_inequality_and_poincare_metric() -> None:
    rng = np.random.default_rng(123)
    profiles = rng.normal(size=(9, 5))
    U = build_padic_ultrametric_matrix(profiles, p=2)

    assert U.shape == (9, 9)
    assert np.allclose(U, U.T, atol=1e-14)
    assert np.allclose(np.diag(U), 0.0, atol=1e-14)
    assert np.all(U >= 0.0)

    # Strong Triangle Inequality: U[i, k] <= max(U[i, j], U[j, k]) for all triples (i, j, k)
    for i in range(9):
        for j in range(9):
            for k in range(9):
                rhs = max(float(U[i, j]), float(U[j, k]))
                assert float(U[i, k]) <= rhs + 1e-14, (
                    f"Strong triangle inequality violated for ({i},{j},{k}): {U[i,k]} > {rhs}"
                )

    # Poincare ball geodesic metric symmetry & positivity
    z1 = np.array([0.25, -0.30])
    z2 = np.array([-0.40, 0.15])
    assert poincare_distance(z1, z1) == 0.0
    assert poincare_distance(z1, z2) > 0.0
    assert np.isclose(poincare_distance(z1, z2), poincare_distance(z2, z1), atol=1e-12)


def test_depth3_path_signature_time_reparameterization_invariance() -> None:
    rng = np.random.default_rng(99)
    ctrl_pts = np.cumsum(rng.normal(scale=0.6, size=(12, 3)), axis=0)

    # Construct nonlinear strictly increasing time warp psi(u) = u^2.3 preserving exact path corners
    t_orig = np.linspace(0.0, 1.0, len(ctrl_pts))
    warped_times = np.unique(np.concatenate([t_orig, np.linspace(0.0, 1.0, 145) ** 2.3]))
    warped_pts = np.column_stack([np.interp(warped_times, t_orig, ctrl_pts[:, d]) for d in range(3)])

    s1_a, s2_a, s3_a = compute_path_signature_depth3(ctrl_pts)
    s1_b, s2_b, s3_b = compute_path_signature_depth3(warped_pts)

    assert np.allclose(s1_a, s1_b, atol=1e-11)
    assert np.allclose(s2_a, s2_b, atol=1e-11)
    assert np.allclose(s3_a, s3_b, atol=1e-11)

    levy_a = compute_levy_area_matrix(s2_a)
    levy_b = compute_levy_area_matrix(s2_b)
    assert np.allclose(levy_a, -levy_a.T, atol=1e-14)
    assert np.allclose(levy_a, levy_b, atol=1e-11)

    logsig_a = compute_log_signature_depth3(ctrl_pts)
    logsig_b = compute_log_signature_depth3(warped_pts)
    assert np.allclose(logsig_a, logsig_b, atol=1e-11)


def test_cellular_sheaf_laplacian_positive_semidefinite() -> None:
    df = build_instructional_day_telemetry_frame(num_days=90, seed=42)
    sheaf = CellularSheafCohomologyTransformer(causal_mode="telemetry", train_cutoff=60)
    sheaf.fit_transform(df)

    L_F = sheaf.last_diagnostics_["sheaf_laplacian"]
    assert np.allclose(L_F, L_F.T, atol=1e-12)

    eigvals = np.linalg.eigvalsh(L_F)
    assert float(np.min(eigvals)) >= -1e-10

    rng = np.random.default_rng(77)
    for _ in range(50):
        x = rng.normal(size=L_F.shape[0])
        energy = float(x.T @ L_F @ x)
        assert energy >= -1e-10, f"Negative Sheaf Dirichlet energy encountered: {energy}"


def test_marchenko_pastur_iid_gaussian_bounds() -> None:
    rng = np.random.default_rng(2026)
    W, N = 600, 12
    iid_noise = rng.normal(0.0, 1.0, size=(W, N))
    spec = analyze_marchenko_pastur_window(iid_noise)

    q = N / W
    expected_plus = (1.0 + np.sqrt(q)) ** 2
    expected_minus = (1.0 - np.sqrt(q)) ** 2
    assert np.isclose(spec["lambda_plus"], expected_plus, atol=1e-12)
    assert np.isclose(spec["lambda_minus"], expected_minus, atol=1e-12)

    # For large W=600, N=12 i.i.d. Gaussian noise, eigenvalues lie well inside [lambda_- - 0.12, lambda_+ + 0.12]
    eigs = spec["eigenvalues"]
    assert float(np.max(eigs)) <= expected_plus + 0.12
    assert float(np.min(eigs)) >= expected_minus - 0.12


def test_subcritical_hawkes_intensity_positivity_and_spectral_radius() -> None:
    df = build_instructional_day_telemetry_frame(num_days=100, seed=42)
    hawkes = HawkesContagionTransformer(causal_mode="telemetry", train_cutoff=65)
    out = hawkes.fit_transform(df)
    diag = hawkes.last_diagnostics_

    assert 0.0 <= diag["spectral_radius"] < 1.0
    assert np.all(diag["lambda_t"] > 0.0)
    assert np.all(diag["exog_total"] > 0.0)
    assert np.all(diag["gamma_matrix"] >= 0.0)
    assert np.all(np.isfinite(out.to_numpy(dtype=float)))


def test_fisher_rao_metric_positivity_and_ngboost_contract() -> None:
    for mu in (0.15, 1.0, 8.5, 45.0, 220.0):
        for r in (0.4, 2.0, 8.0, 35.0):
            g_mu, g_r = nb_fisher_information_diagonal(mu, r)
            assert np.isfinite(g_mu) and g_mu > 0.0
            assert np.isfinite(g_r) and g_r > 0.0
            assert np.isclose(g_mu, r / (mu * (r + mu)), rtol=1e-10)

    panel = generate_weekly_safeguarding_panel(num_academic_years=3, seed=42)
    cw_df, vif = engineer_safeguarding_features(aggregate_college_wide_weekly(panel))
    tr = cw_df.iloc[:104].copy().reset_index(drop=True)
    te = cw_df.iloc[104:130].copy().reset_index(drop=True)

    booster = FisherRaoNaturalGradientBooster(feature_cols=vif.selected_feature_cols).fit(tr)
    pred_1x = booster.predict(te)

    te_2x = te.copy()
    te_2x["active_enrolments"] = te_2x["active_enrolments"] * 2
    pred_2x = booster.predict(te_2x)

    assert np.all(pred_1x.point_forecasts > 0.0)
    assert np.allclose(pred_2x.point_forecasts, 2.0 * pred_1x.point_forecasts, rtol=1e-10)
    assert np.all(pred_1x.quantiles[0.025] <= pred_1x.quantiles[0.50])
    assert np.all(pred_1x.quantiles[0.50] <= pred_1x.quantiles[0.90])
    assert np.all(pred_1x.quantiles[0.90] <= pred_1x.quantiles[0.975])
