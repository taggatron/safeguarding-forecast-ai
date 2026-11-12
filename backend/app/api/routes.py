"""FastAPI route definitions for Safeguarding Forecast AI."""
from datetime import datetime, timezone
import json
from typing import Any, Dict, Optional
import uuid
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from app.core.database import IMPORTED_BANNER, SYNTHETIC_BANNER
from app.data.processor import validate_and_parse_csv
from app.data.synthetic_generator import generate_synthetic_safeguarding_dataset
from app.schemas.models import (
    BacktestResponse,
    DatasetSummaryResponse,
    DataValidationReport,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    HistoricalStatsResponse,
    ModelsAvailabilityResponse,
    SyntheticGeneratorConfig,
    TrendsInsightsResponse,
)
from app.services.forecast_service import platform_service

router = APIRouter(prefix="/api")


class ConfigUpdateRequest(BaseModel):
    academic_year_start_month: Optional[int] = None
    governance_mode: Optional[str] = None
    timesfm_license_acknowledged: Optional[bool] = None
    small_group_suppression_enabled: Optional[bool] = None


class SwitchDatasetRequest(BaseModel):
    dataset_id: str


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    info, _ = platform_service.get_active_dataset_and_df()
    is_synth = bool(info["is_synthetic"])
    ay_start = int(
        platform_service.db.get_config("academic_year_start_month", str(info["academic_year_start_month"])) or 9
    )
    return HealthResponse(
        status="healthy",
        application="Safeguarding Forecast AI",
        subtitle="Predictive Safeguarding Demand Intelligence",
        organisation="South Devon College",
        active_dataset_id=info["dataset_id"],
        is_synthetic=is_synth,
        governance_banner=SYNTHETIC_BANNER if is_synth else IMPORTED_BANNER,
        academic_year_start_month=ay_start,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/dataset/summary", response_model=DatasetSummaryResponse)
def get_dataset_summary() -> DatasetSummaryResponse:
    return platform_service.get_active_dataset_summary()


@router.post("/dataset/synthetic", response_model=DatasetSummaryResponse)
def generate_synthetic_data(config: SyntheticGeneratorConfig) -> DatasetSummaryResponse:
    return platform_service.regenerate_synthetic_dataset(config)


@router.post("/dataset/switch", response_model=DatasetSummaryResponse)
def switch_active_dataset(req: SwitchDatasetRequest) -> DatasetSummaryResponse:
    if not platform_service.db.dataset_exists(req.dataset_id):
        raise HTTPException(status_code=404, detail=f"Dataset '{req.dataset_id}' not found.")
    platform_service.db.set_config("active_dataset_id", req.dataset_id)
    platform_service.clear_caches()
    return platform_service.get_active_dataset_summary()


@router.post("/config/update")
def update_configuration(req: ConfigUpdateRequest) -> Dict[str, Any]:
    if req.academic_year_start_month is not None:
        if not (1 <= req.academic_year_start_month <= 12):
            raise HTTPException(status_code=400, detail="academic_year_start_month must be between 1 and 12.")
        platform_service.db.set_config("academic_year_start_month", str(req.academic_year_start_month))
    if req.governance_mode is not None:
        if req.governance_mode not in ("research_prototype", "operational_production"):
            raise HTTPException(
                status_code=400,
                detail="governance_mode must be 'research_prototype' or 'operational_production'.",
            )
        platform_service.db.set_config("governance_mode", req.governance_mode)
    if req.timesfm_license_acknowledged is not None:
        platform_service.db.set_config(
            "timesfm_license_acknowledged", "true" if req.timesfm_license_acknowledged else "false"
        )
    if req.small_group_suppression_enabled is not None:
        platform_service.db.set_config(
            "small_group_suppression_enabled", "true" if req.small_group_suppression_enabled else "false"
        )
    platform_service.clear_caches()
    return {
        "status": "updated",
        "configs": platform_service.db.get_all_configs(),
        "models_availability": platform_service.get_models_availability(),
    }


@router.post("/dataset/validate-csv", response_model=DataValidationReport)
async def validate_csv_upload(
    file: UploadFile = File(...),
    column_mapping_json: Optional[str] = Form(default=None),
) -> DataValidationReport:
    content = await file.read()
    mapping = json.loads(column_mapping_json) if column_mapping_json else None
    ay_start = int(platform_service.db.get_config("academic_year_start_month", "9") or 9)
    report, _ = validate_and_parse_csv(content, column_mapping=mapping, academic_year_start_month=ay_start)
    return report


@router.post("/dataset/import-csv", response_model=DatasetSummaryResponse)
async def import_csv_upload(
    file: UploadFile = File(...),
    dataset_name: Optional[str] = Form(default=None),
    column_mapping_json: Optional[str] = Form(default=None),
) -> DatasetSummaryResponse:
    content = await file.read()
    mapping = json.loads(column_mapping_json) if column_mapping_json else None
    ay_start = int(platform_service.db.get_config("academic_year_start_month", "9") or 9)
    report, norm_df = validate_and_parse_csv(content, column_mapping=mapping, academic_year_start_month=ay_start)
    if not report.can_import or norm_df is None:
        err_msgs = "; ".join(i.message for i in report.issues if i.severity == "error")
        raise HTTPException(
            status_code=400,
            detail=f"CSV failed validation and cannot be imported: {err_msgs or 'Insufficient valid data.'}",
        )

    # Save into a separate imported dataset slot (never combining with synthetic demo data)
    imported_id = f"imported_{uuid.uuid4().hex[:8]}"
    clean_name = (dataset_name or file.filename or "Imported College Dataset").strip()
    platform_service.db.save_dataset(
        dataset_id=imported_id,
        name=f"Imported: {clean_name}",
        df=norm_df,
        is_synthetic=False,
        seed=None,
        academic_year_start_month=ay_start,
        metadata={"filename": file.filename, "row_count": len(norm_df)},
    )
    platform_service.db.set_config("active_dataset_id", imported_id)
    platform_service.clear_caches()
    return platform_service.get_active_dataset_summary()


@router.get("/dataset/sample-csv", response_class=PlainTextResponse)
def download_sample_csv_template() -> PlainTextResponse:
    """Generate a downloadable 4-academic-year aggregated sample CSV template following the recommended schema."""
    sample_cfg = SyntheticGeneratorConfig(
        seed=101,
        num_academic_years=4,
        start_academic_year=2022,
        academic_year_start_month=9,
    )
    df = generate_synthetic_safeguarding_dataset(sample_cfg)
    cols = [
        "period_start",
        "provision",
        "safeguarding_category",
        "concern_count",
        "enrolled_students",
        "teaching_days",
        "attendance_rate",
        "reporting_change_flag",
    ]
    csv_str = df[cols].to_csv(index=False)
    return PlainTextResponse(
        content=csv_str,
        media_type="text/csv",
        headers={
            "Content-Disposition": 'attachment; filename="south_devon_college_safeguarding_template.csv"'
        },
    )


@router.get("/historical", response_model=HistoricalStatsResponse)
def get_historical_statistics(
    academic_year: str = Query(default="All"),
    provision: str = Query(default="All"),
    safeguarding_category: str = Query(default="All"),
) -> HistoricalStatsResponse:
    return platform_service.get_historical_stats(
        academic_year=academic_year,
        provision=provision,
        safeguarding_category=safeguarding_category,
    )


@router.post("/forecast", response_model=ForecastResponse)
def create_forecast(req: ForecastRequest) -> ForecastResponse:
    try:
        return platform_service.generate_forecast(req)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Forecasting engine error: {exc}") from exc


@router.get("/models/availability", response_model=ModelsAvailabilityResponse)
def get_models_availability() -> ModelsAvailabilityResponse:
    return platform_service.get_models_availability()


@router.get("/models/backtest", response_model=BacktestResponse)
def get_models_backtest(
    provision: str = Query(default="All"),
    safeguarding_category: str = Query(default="All"),
    horizon: int = Query(default=12, ge=3, le=12),
) -> BacktestResponse:
    return platform_service.get_backtest_comparison(
        provision=provision,
        safeguarding_category=safeguarding_category,
        horizon=horizon,
    )


@router.get("/trends", response_model=TrendsInsightsResponse)
def get_trends_and_insights(
    provision: str = Query(default="All"),
    safeguarding_category: str = Query(default="All"),
) -> TrendsInsightsResponse:
    return platform_service.get_trends_and_insights(
        provision=provision,
        safeguarding_category=safeguarding_category,
    )


@router.get("/models/runs")
def get_recent_forecast_runs(limit: int = Query(default=20, ge=1, le=100)) -> Dict[str, Any]:
    return {"runs": platform_service.db.list_forecast_runs(limit=limit)}


@router.get("/benchmark/count-models")
def get_count_models_benchmark(
    horizon_weeks: int = Query(default=26, ge=4, le=52),
) -> Dict[str, Any]:
    """Run and return the NB2 GLM vs. ZINB vs. Tweedie LightGBM count benchmark & DSL governance report."""
    from app.models.safeguarding_benchmark import run_expanding_window_benchmark

    return run_expanding_window_benchmark(horizon_weeks=horizon_weeks)


@router.get("/esoteric-telemetry")
def get_esoteric_telemetry(
    t: Optional[str] = Query(default=None, description="Causal evaluation ISO date YYYY-MM-DD (masks s > t)"),
    step_index: Optional[int] = Query(default=None, ge=10, description="Optional causal step index"),
    provision: str = Query(default="All", description="Provision filter (All, EPYP, Adult, Apprenticeships)"),
    retro_mode: str = Query(
        default="STAGED_CORE_4",
        description="Retrospective rollout mode: TELEMETRY_ONLY, STAGED_CORE_4, or FULL_13_GATED",
    ),
    lag_buffer_days: int = Query(
        default=0,
        ge=0,
        le=14,
        description="Single-timestamp reporting-lag sensitivity buffer in days (e.g. 0, 2, 5)",
    ),
) -> Dict[str, Any]:
    """
    Serve strictly causal Mathematical Early-Warning Telemetry & Retrospective Residual Overlay JSON
    (Hilbert Stress-Cycle Phase Clock, Hawkes Contagion Cascade, Sheaf Frustration Triage,
    Frozen Baseline vs. Esoteric Residual Adjustment Overlay, and Backfill Audit Accordion).
    """
    from pathlib import Path
    import sys

    repo_root = str(Path(__file__).resolve().parent.parent.parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from esoteric_math.retrospective.replay_telemetry_bus import replay_telemetry_bus

    return replay_telemetry_bus.get_replay_response(
        t_date=t,
        t_index=step_index,
        provision=provision,
        retro_mode=retro_mode,
        lag_buffer_days=lag_buffer_days,
    )


class EsotericForecastRequest(BaseModel):
    cadence: str = "monthly"
    strategy_id: str = "staged_core_4"
    base_model_id: str = "lightgbm_mlforecast"
    horizon: int = 12
    provision: str = "All"
    safeguarding_category: str = "All"
    prediction_interval: int = 80
    custom_modules: Optional[list[str]] = None
    scenario: Optional[Dict[str, Any]] = None


@router.post("/esoteric-forecast")
def create_esoteric_forecast(req: EsotericForecastRequest) -> Dict[str, Any]:
    """
    Generate causal multi-step time-series forecasts integrating the 13-framework esoteric
    mathematical suite (Rough Path Signatures, Warped Hilbert Phase, Causal MODWT Wavelets,
    Hawkes Contagion, Cellular Sheaves, Optimal Transport, Fisher-Rao NGBoost, etc.)
    alongside locked pure-baseline control trajectories and walk-forward validation.
    """
    from pathlib import Path
    import sys

    from app.schemas.models import ScenarioAssumptions

    repo_root = str(Path(__file__).resolve().parent.parent.parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from esoteric_math.forecasting import esoteric_forecast_engine

    scen = ScenarioAssumptions(**req.scenario) if req.scenario else ScenarioAssumptions()
    cad: Any = "weekly" if str(req.cadence).lower() == "weekly" else "monthly"

    try:
        return esoteric_forecast_engine.generate_esoteric_forecast(
            cadence=cad,
            strategy_id=req.strategy_id,
            base_model_id=req.base_model_id,
            horizon=req.horizon,
            provision=req.provision,
            safeguarding_category=req.safeguarding_category,
            prediction_interval=req.prediction_interval,
            custom_modules=req.custom_modules,
            scenario=scen,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Esoteric forecasting engine error: {exc}") from exc




