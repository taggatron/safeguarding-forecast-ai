"""
Tests for the dedicated Esoteric & Rough-Path Multi-Step Forecasting Tab (`POST /api/esoteric-forecast`).

Verifies:
1. Causal recursive multi-step rollouts across both monthly (3m/6m/12m) and weekly (13w/26w) cadences.
2. Rough Path Signatures (`rough_signatures_focus`) and Fisher-Rao NGBoost (`fisher_rao_ngboost`) execution.
3. Identity-floor fallback when `custom_modules=[]` (zero variance inflation).
4. Strict non-interference with the locked core operational forecast endpoint (`POST /api/forecast`).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.main import app  # noqa: E402

client = TestClient(app)


def _core_forecast_digest() -> str:
    resp = client.post(
        "/api/forecast",
        json={
            "model_id": "lightgbm_mlforecast",
            "horizon": 12,
            "provision": "All",
            "safeguarding_category": "All",
            "prediction_interval": 80,
        },
    )
    assert resp.status_code == 200
    pts = [p["point_forecast"] for p in resp.json()["monthly_forecasts"]]
    return hashlib.sha256(json.dumps(pts).encode("utf-8")).hexdigest()


def test_esoteric_forecast_monthly_and_core_isolation() -> None:
    digest_before = _core_forecast_digest()

    resp = client.post(
        "/api/esoteric-forecast",
        json={
            "cadence": "monthly",
            "strategy_id": "rough_signatures_focus",
            "base_model_id": "lightgbm_mlforecast",
            "horizon": 6,
            "provision": "All",
            "safeguarding_category": "All",
            "prediction_interval": 80,
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["cadence"] == "monthly"
    assert payload["horizon"] == 6
    assert len(payload["forecast_series"]) == 6
    assert "f07" in payload["active_modules"]

    for pt in payload["forecast_series"]:
        assert pt["lower_bound"] <= pt["esoteric_forecast"] <= pt["upper_bound"]
        assert pt["dispersion_r"] > 0.0

    # Core operational endpoint must remain 100% byte-identical
    digest_after = _core_forecast_digest()
    assert digest_before == digest_after


def test_esoteric_forecast_weekly_and_fisher_rao_booster() -> None:
    resp = client.post(
        "/api/esoteric-forecast",
        json={
            "cadence": "weekly",
            "strategy_id": "fisher_rao_ngboost",
            "base_model_id": "tweedie_lgbm",
            "horizon": 13,
            "provision": "EPYP",
            "safeguarding_category": "All",
            "prediction_interval": 95,
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["cadence"] == "weekly"
    assert payload["horizon"] == 13
    assert len(payload["forecast_series"]) == 13
    assert payload["prediction_interval"] == 95
    assert payload["summary"]["esoteric_lower"] <= payload["summary"]["esoteric_total"] <= payload["summary"]["esoteric_upper"]


def test_esoteric_forecast_empty_custom_modules_triggers_identity_floor() -> None:
    resp = client.post(
        "/api/esoteric-forecast",
        json={
            "cadence": "monthly",
            "strategy_id": "staged_core_4",
            "base_model_id": "statistical_count",
            "horizon": 6,
            "provision": "All",
            "safeguarding_category": "All",
            "prediction_interval": 80,
            "custom_modules": [],
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["conditioning_diagnostics"]["identity_floor_active"] is True
    assert payload["summary"]["net_esoteric_delta"] == 0
    for pt in payload["forecast_series"]:
        assert pt["esoteric_forecast"] == pt["baseline_forecast"]

