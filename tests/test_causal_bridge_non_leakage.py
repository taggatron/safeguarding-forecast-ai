"""
Phase 5 Verification Gate 1: Strict Causal Anti-Leakage Suite (`tests/test_causal_bridge_non_leakage.py`).

Corrupts ALL targets and covariates for s > t_0 with +10^6 spikes and asserts
`np.allclose` to `1e-12` across:
  1. All 13 feature matrix blocks (`f01` .. `f13`) for all rows s <= t_0 (in both
     `"telemetry"` and `"forecast"` causal modes).
  2. Every numerical array, coordinate, and scalar inside `telemetry_payload.py`
     JSON outputs driving the Hilbert Clock, Hawkes Cascade, and Sheaf Frustration UI.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
import pytest

from esoteric_math.features import FEATURE_ENGINE_REGISTRY, EsotericFeatureSuite
from esoteric_math.schema_bridge import AdaptiveSchemaBridge, build_instructional_day_telemetry_frame
from esoteric_math.viz.telemetry_payload import extract_telemetry_payload


def _corrupt_future_observations(df: pd.DataFrame, t0: int, spike: float = 1e6) -> pd.DataFrame:
    """Spike every target, category, cohort, and exogenous numeric column strictly after t0."""
    corrupted = df.copy()
    numeric_cols = [
        c
        for c in corrupted.columns
        if pd.api.types.is_numeric_dtype(corrupted[c])
        and c not in ("is_instructional_day",)
    ]
    for col in numeric_cols:
        corrupted[col] = corrupted[col].astype(float)
        corrupted.loc[corrupted.index > t0, col] = corrupted.loc[corrupted.index > t0, col] + spike
    return corrupted


def _assert_nested_json_allclose(obj_a: Any, obj_b: Any, atol: float = 1e-12, path: str = "root") -> None:
    """Recursively assert exact `1e-12` equality across nested JSON dicts/lists/floats."""
    if isinstance(obj_a, dict):
        assert isinstance(obj_b, dict), f"Type mismatch at {path}"
        assert set(obj_a.keys()) == set(obj_b.keys()), f"Key mismatch at {path}"
        for k in obj_a:
            _assert_nested_json_allclose(obj_a[k], obj_b[k], atol=atol, path=f"{path}.{k}")
    elif isinstance(obj_a, (list, tuple)):
        assert len(obj_a) == len(obj_b), f"Length mismatch at {path}: {len(obj_a)} vs {len(obj_b)}"
        for i, (va, vb) in enumerate(zip(obj_a, obj_b)):
            _assert_nested_json_allclose(va, vb, atol=atol, path=f"{path}[{i}]")
    elif isinstance(obj_a, (int, float, np.number)) and not isinstance(obj_a, bool):
        assert np.allclose(float(obj_a), float(obj_b), atol=atol, rtol=0.0), (
            f"Causal leakage at {path}: clean={obj_a} vs future_corrupted={obj_b}"
        )
    else:
        assert obj_a == obj_b, f"Value mismatch at {path}: {obj_a!r} vs {obj_b!r}"


@pytest.mark.parametrize("causal_mode", ["telemetry", "forecast"])
def test_all_13_feature_modules_strictly_causal_under_million_spike(causal_mode: str) -> None:
    """
    Corrupt all targets and covariates for s > t0 with +10^6 spikes and assert
    np.allclose(..., atol=1e-12, rtol=0.0) for all s <= t0 across all 13 modules.
    """
    df_clean = build_instructional_day_telemetry_frame(num_days=96, seed=42)
    train_cutoff = 52
    t0 = 72
    df_corrupted = _corrupt_future_observations(df_clean, t0=t0, spike=1e6)

    suite_clean = EsotericFeatureSuite(causal_mode=causal_mode, train_cutoff=train_cutoff)
    feats_clean = suite_clean.fit_transform(df_clean)

    suite_corrupt = EsotericFeatureSuite(causal_mode=causal_mode, train_cutoff=train_cutoff)
    feats_corrupt = suite_corrupt.fit_transform(df_corrupted)

    assert set(suite_clean.module_columns_.keys()) == set(FEATURE_ENGINE_REGISTRY.keys())
    assert len(suite_clean.module_columns_) == 13

    for mod_id, cols in suite_clean.module_columns_.items():
        block_clean = feats_clean.loc[:t0, cols].to_numpy(dtype=float)
        block_corrupt = feats_corrupt.loc[:t0, cols].to_numpy(dtype=float)
        max_diff = float(np.max(np.abs(block_clean - block_corrupt)))
        assert np.allclose(block_clean, block_corrupt, atol=1e-12, rtol=0.0), (
            f"Module {mod_id} ({causal_mode}) leaked future s > {t0}: max_abs_diff={max_diff}"
        )


def test_schema_bridge_latent_regime_fallback_is_causal() -> None:
    """Verify Tier B/C univariate & flat-total latent graph synthesis has zero future leakage."""
    dates = pd.date_range("2024-09-02", periods=80, freq="D")
    rng = np.random.default_rng(7)
    flat_df = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "referral_count": rng.poisson(6.0, size=80),
            "active_enrolments": np.full(80, 6800),
        }
    )
    t0 = 58
    flat_corrupt = _corrupt_future_observations(flat_df, t0=t0, spike=1e6)

    b_clean = AdaptiveSchemaBridge(causal_mode="telemetry", train_cutoff=44).fit(flat_df).transform(flat_df)
    b_corrupt = AdaptiveSchemaBridge(causal_mode="telemetry", train_cutoff=44).fit(flat_corrupt).transform(flat_corrupt)

    assert b_clean.tier_c.topology_mode == "Latent Feature/Regime Graph"
    assert np.allclose(b_clean.tier_a.y_rate_per_1000[: t0 + 1], b_corrupt.tier_a.y_rate_per_1000[: t0 + 1], atol=1e-12, rtol=0.0)
    assert np.allclose(b_clean.tier_b.covariate_matrix[: t0 + 1], b_corrupt.tier_b.covariate_matrix[: t0 + 1], atol=1e-12, rtol=0.0)
    assert np.allclose(b_clean.tier_c.stalks[: t0 + 1], b_corrupt.tier_c.stalks[: t0 + 1], atol=1e-12, rtol=0.0)


def test_telemetry_payload_ui_json_strictly_causal_under_million_spike() -> None:
    """
    Assert that the complete `telemetry_payload.py` JSON output driving the Hilbert Clock,
    Hawkes Cascade (including its (t, t+5] projection cone), and Sheaf Frustration UI
    at t <= t0 is invariant to 1e-12 when s > t0 is spiked by +10^6.
    """
    df_clean = build_instructional_day_telemetry_frame(num_days=110, seed=42)
    train_cutoff = 55
    t0 = 80
    df_corrupted = _corrupt_future_observations(df_clean, t0=t0, spike=1e6)

    for probe_t in (60, t0):
        payload_clean = extract_telemetry_payload(df_clean, t_index=probe_t, train_cutoff=train_cutoff)
        payload_corrupt = extract_telemetry_payload(df_corrupted, t_index=probe_t, train_cutoff=train_cutoff)
        _assert_nested_json_allclose(payload_clean, payload_corrupt, atol=1e-12, path=f"payload[t={probe_t}]")
