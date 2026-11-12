"""
Phase 5 Verification Gate 3: Telemetry Payload & Figure Builder Edge-Case Suite
(`tests/test_viz_builders.py`).

Asserts all three visual spec builders (`hilbert_clock.py`, `hawkes_cascade.py`,
`sheaf_frustration.py`), `telemetry_payload.py`, and standalone SVG/HTML renderers
execute cleanly across edge cases:
  1. Zero-referral holiday blocks (`referral_count == 0` stretches)
  2. Single-category univariate fallback (`C == 1`)
  3. Flat institutional totals triggering `"Latent Feature/Regime Graph"` fallback
with valid finite coordinates (`r >= 0`, `phi in [-pi, pi]`, `0 <= rho < 1.5`,
non-negative edge frustration).
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from esoteric_math.schema_bridge import build_instructional_day_telemetry_frame
from esoteric_math.viz import (
    extract_telemetry_payload,
    render_hawkes_cascade_svg,
    render_hilbert_clock_svg,
    render_sheaf_frustration_svg,
)


def _validate_telemetry_invariants(payload: dict) -> None:
    # Must be strictly JSON-serializable without NaNs/Infs
    dumped = json.dumps(payload, allow_nan=False)
    assert len(dumped) > 200

    assert len(payload["kpi_ribbon"]) == 3

    # 1. Hilbert Clock bounds
    hc = payload["hilbert_clock"]
    cur_h = hc["current"]
    assert np.isfinite(cur_h["r_norm"]) and cur_h["r_norm"] >= 0.0
    assert -np.pi - 1e-9 <= cur_h["phi_rad"] <= np.pi + 1e-9
    assert cur_h["days_to_peak"] >= 0.0
    for pt in hc["comet_trajectory"]:
        assert pt["r_norm"] >= 0.0
        assert -180.001 <= pt["phi_deg"] <= 180.001
        assert 0.14 <= pt["opacity"] <= 1.01

    # 2. Hawkes Cascade bounds
    hk = payload["hawkes_cascade"]
    rho = hk["current"]["spectral_radius"]
    assert np.isfinite(rho) and 0.0 <= rho < 1.5
    assert 0.0 <= hk["current"]["endog_share"] <= 1.0
    assert len(hk["projection_cone"]) == 5
    for e in hk["network"]["edges"]:
        assert e["branching_ratio"] >= 0.03
        assert e["half_life_days"] > 0.0

    # 3. Sheaf Frustration bounds
    sf = payload["sheaf_frustration"]
    assert sf["current"]["dirichlet_energy"] >= 0.0
    for edge in sf["graph"]["edges"]:
        assert np.isfinite(edge["frustration_norm"]) and edge["frustration_norm"] >= 0.0
    assert 1 <= len(sf["triage_cards"]) <= 3

    # 4. Valid standalone SVG strings (must parse cleanly as strict XML)
    import xml.etree.ElementTree as ET

    for svg_str in (
        render_hilbert_clock_svg(hc),
        render_hawkes_cascade_svg(hk),
        render_sheaf_frustration_svg(sf),
    ):
        assert svg_str.startswith("<svg") and svg_str.strip().endswith("</svg>")
        root = ET.fromstring(svg_str)
        assert root.tag.endswith("svg")


def test_full_telemetry_builders_and_zero_referral_holiday_blocks() -> None:
    df = build_instructional_day_telemetry_frame(num_days=95, seed=42)
    # Inject a strict 10-day zero-referral holiday closure block
    df.loc[45:54, "referral_count"] = 0
    df.loc[45:54, "is_instructional_day"] = 0
    df.loc[45:54, "holiday_label"] = "Summer Closure Block"
    for c in [col for col in df.columns if col.startswith("cat_") or col.endswith("_referrals")]:
        df.loc[45:54, c] = 0

    # Evaluate right inside the zero-referral holiday block (t=52) and post-return (t=62)
    payload_hol = extract_telemetry_payload(df, t_index=52, train_cutoff=40)
    _validate_telemetry_invariants(payload_hol)
    assert payload_hol["hilbert_clock"]["current"]["is_holiday_freeze"] is True

    payload_ret = extract_telemetry_payload(df, t_index=62, train_cutoff=40)
    _validate_telemetry_invariants(payload_ret)
    assert payload_ret["topology_mode"] == "Cohort Hierarchy Graph"


def test_single_category_and_flat_latent_graph_fallback() -> None:
    """
    Verify purely univariate flat institutional counts (no categories, no cohorts, no covariates)
    trigger single-category Hawkes fallback AND `"Latent Feature/Regime Graph"` sheaf topology cleanly.
    """
    dates = pd.date_range("2024-09-02", periods=75, freq="D")
    rng = np.random.default_rng(11)
    flat_df = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "referral_count": rng.poisson(5.0, size=75),
        }
    )

    payload = extract_telemetry_payload(flat_df, t_index=70, train_cutoff=45)
    _validate_telemetry_invariants(payload)
    assert payload["topology_mode"] == "Latent Feature/Regime Graph"
    assert payload["sheaf_frustration"]["topology_mode"] == "Latent Feature/Regime Graph"
    assert payload["hawkes_cascade"]["is_univariate_fallback"] is True


def test_fastapi_esoteric_telemetry_endpoint_and_holiday_freeze() -> None:
    """Verify non-breaking FastAPI route `/api/esoteric-telemetry` serves valid causal payloads."""
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    resp_latest = client.get("/api/esoteric-telemetry", params={"provision": "All"})
    assert resp_latest.status_code == 200
    body = resp_latest.json()
    assert "scrubber" in body and "telemetry" in body
    _validate_telemetry_invariants(body["telemetry"])

    # Verify Autumn Half-Term Freeze landmark sets is_holiday_freeze=True
    resp_freeze = client.get("/api/esoteric-telemetry", params={"t": "2024-10-29", "provision": "All"})
    assert resp_freeze.status_code == 200
    body_freeze = resp_freeze.json()
    assert body_freeze["telemetry"]["selected_date"] == "2024-10-29"
    assert body_freeze["telemetry"]["hilbert_clock"]["current"]["is_holiday_freeze"] is True
    _validate_telemetry_invariants(body_freeze["telemetry"])

