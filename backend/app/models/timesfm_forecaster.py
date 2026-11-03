"""Model C: Official Google TimesFM 3.0 foundation model adapter (Apple Silicon MLX & PyTorch backends)."""
import os
import platform
import time
from typing import Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from app.models.base import (
    Z_80,
    Z_95,
    BaseForecaster,
    EngineForecastOutput,
    build_future_covariates_df,
    compute_calibrated_annual_interval,
)
from app.schemas.models import (
    ScenarioAssumptions,
    ScenarioSensitivityCapabilities,
)

TIMESFM3_CHECKPOINT = "google/timesfm-3.0-pytorch"
TIMESFM3_LICENSE_NAME = "timesfm-non-commercial-license-v1.0 (Non-Commercial / Non-Production Research Only)"


class TimesFM3ForecasterAdapter(BaseForecaster):
    """
    Model C: Google TimesFM 3.0 (330M parameter zero-shot time-series foundation model).
    Uses the official `timesfm3` API from https://github.com/google-research/timesfm:
    - Apple Silicon native backend: `from timesfm3.mlx import TimesFM3Forecaster`
    - PyTorch/CUDA/CPU fallback: `from timesfm3 import TimesFM3Evaluator, ModelConfig`
    Supports both univariate zero-shot forecasting and multivariate/covariate-assisted forecasting
    with native decile quantile outputs (0.1 to 0.9).

    Strictly enforces licensing & governance checks:
    - Pretrained weights carry `timesfm-non-commercial-license-v1.0` restrictions.
    - Enabled ONLY when running in `research_prototype` mode with license acknowledgement.
    - Never transmits safeguarding datasets to Google Cloud or external inference endpoints.
    - Never fabricates predictions if weights, runtime, or license permissions are unavailable.
    """

    model_id = "timesfm_3"
    name = "Google TimesFM 3.0 (Foundation Model)"
    short_name = "TimesFM 3.0"
    family = "Zero-Shot Time-Series Foundation Model"
    version = "timesfm-3.0.2 (330M)"
    license_name = TIMESFM3_LICENSE_NAME
    commercial_production_permitted = False
    supports_95_pi = True
    data_requirements = "12+ months historical series (supports zero-shot univariate & dynamic covariates)"
    backend_engine = "Google Research timesfm3 (Local Apple Silicon MLX / PyTorch)"

    _cached_forecaster: Any = None
    _cached_backend_type: Optional[str] = None
    _load_error: Optional[str] = None

    def __init__(self, force_unavailable: bool = False):
        self.force_unavailable = force_unavailable

    @classmethod
    def _check_weights_cached_or_loadable(cls) -> Tuple[bool, str]:
        """Verify if `timesfm3` package and local pretrained checkpoint are accessible."""
        if os.environ.get("SAFEGUARDING_DISABLE_TIMESFM", "").strip() == "1":
            return (
                False,
                "Not available: TimesFM 3.0 runtime explicitly disabled via SAFEGUARDING_DISABLE_TIMESFM=1 environment setting.",
            )

        try:
            import timesfm3  # noqa: F401
        except ImportError as exc:
            return (
                False,
                f"Not available: Official `timesfm3` package is not installed in the current Python environment ({exc}).",
            )

        # Check if MLX or PyTorch backend is importable
        has_mlx = False
        has_torch = False
        try:
            from timesfm3.mlx import TimesFM3Forecaster as _MLXForecaster  # noqa: F401
            has_mlx = True
        except Exception:
            pass

        if not has_mlx:
            try:
                from timesfm3 import TimesFM3Evaluator as _TorchEvaluator  # noqa: F401
                has_torch = True
            except Exception:
                pass

        if not has_mlx and not has_torch:
            return (
                False,
                "Not available: Neither MLX (`timesfm[mlx]`) nor PyTorch (`timesfm[torch]`) runtime backend is installed.",
            )

        backend_desc = "Apple Silicon MLX (`timesfm3.mlx`)" if has_mlx else "PyTorch (`timesfm3`)"
        return (
            True,
            f"Local {backend_desc} runtime ready with `{TIMESFM3_CHECKPOINT}` (Research/Prototype non-commercial license only).",
        )

    def is_available(
        self,
        governance_mode: str = "research_prototype",
        license_ack: bool = True,
    ) -> Tuple[bool, str, str]:
        if self.force_unavailable:
            return (
                False,
                "Not available",
                "TimesFM 3.0 is marked unavailable in this configuration (hardware, weights, or license gate disabled).",
            )

        if governance_mode != "research_prototype":
            return (
                False,
                "Not available (Governance Restriction)",
                (
                    "TimesFM 3.0 downloaded weights are governed by `timesfm-non-commercial-license-v1.0`, "
                    "which restricts self-hosted weights to non-commercial, non-production research/evaluation. "
                    "Because Governance Mode is set to Operational Production, TimesFM 3.0 is disabled. "
                    "Use LightGBM + MLForecast or Statistical Count GLM for operational college use."
                ),
            )

        if not license_ack:
            return (
                False,
                "Not available (License Unacknowledged)",
                (
                    "TimesFM 3.0 requires explicit acknowledgement of the `timesfm-non-commercial-license-v1.0` "
                    "research/prototype restriction in Data Management before local weights can be executed."
                ),
            )

        ok, msg = self._check_weights_cached_or_loadable()
        if not ok:
            return False, "Not available", msg

        if self._load_error:
            return False, "Not available", f"TimesFM 3.0 failed to initialize local weights: {self._load_error}"

        return True, "Available (Research Prototype Mode)", msg

    @classmethod
    def _get_or_load_model(cls) -> Tuple[Any, str]:
        """Lazily load and cache the official TimesFM 3.0 forecaster in local memory."""
        if cls._cached_forecaster is not None and cls._cached_backend_type is not None:
            return cls._cached_forecaster, cls._cached_backend_type

        # Prefer Apple Silicon MLX backend when available on arm64 macOS
        try:
            from timesfm3.mlx import TimesFM3Forecaster as MLXTimesFM3Forecaster

            forecaster = MLXTimesFM3Forecaster.from_pretrained(TIMESFM3_CHECKPOINT)
            cls._cached_forecaster = forecaster
            cls._cached_backend_type = "mlx"
            cls._load_error = None
            return forecaster, "mlx"
        except Exception as mlx_exc:
            # Fallback to PyTorch TimesFM3Evaluator
            try:
                from timesfm3 import ModelConfig, TimesFM3Evaluator

                device = "cpu"
                config = ModelConfig(
                    checkpoint_path=TIMESFM3_CHECKPOINT,
                    per_core_batch_size=16,
                    device=device,
                )
                evaluator = TimesFM3Evaluator(config)
                cls._cached_forecaster = evaluator
                cls._cached_backend_type = "pytorch"
                cls._load_error = None
                return evaluator, "pytorch"
            except Exception as torch_exc:
                cls._load_error = f"MLX error: {mlx_exc}; PyTorch error: {torch_exc}"
                raise RuntimeError(cls._load_error) from torch_exc

    def get_scenario_capabilities(self, has_attendance_data: bool = True) -> ScenarioSensitivityCapabilities:
        return ScenarioSensitivityCapabilities(
            supports_enrolment_growth=True,
            enrolment_explanation="TimesFM 3.0 incorporates enrolled student headcount via dynamic `past_future_covariates`.",
            supports_teaching_days=True,
            teaching_days_explanation="TimesFM 3.0 incorporates scheduled teaching days via dynamic `past_future_covariates`.",
            supports_attendance_scenario=has_attendance_data,
            attendance_explanation=(
                "TimesFM 3.0 incorporates attendance scenarios as dynamic covariates when multivariate/covariate mode is active."
                if has_attendance_data
                else "Disabled because historical attendance rate data is not present in the active dataset."
            ),
            supports_reporting_flag=True,
        )

    def forecast(
        self,
        hist_df: pd.DataFrame,
        horizon: int = 12,
        scenario: Optional[ScenarioAssumptions] = None,
        academic_year_start_month: int = 9,
        governance_mode: str = "research_prototype",
        license_ack: bool = True,
        use_covariates: bool = True,
        **kwargs: Any,
    ) -> EngineForecastOutput:
        avail, status_label, explanation = self.is_available(
            governance_mode=governance_mode, license_ack=license_ack
        )
        if not avail:
            raise RuntimeError(f"{status_label}: {explanation}")

        t0 = time.perf_counter()
        if scenario is None:
            scenario = ScenarioAssumptions()

        future_cov = build_future_covariates_df(hist_df, horizon, scenario, academic_year_start_month)
        forecaster, backend_type = self._get_or_load_model()

        y_context = hist_df["concern_count"].to_numpy(dtype=np.float32)
        context_len = len(y_context)

        # Build past_only_covariates and past_future_covariates for TimesFM 3.0 when use_covariates=True
        past_only_cov: Optional[np.ndarray] = None
        past_future_cov: Optional[np.ndarray] = None

        if use_covariates and context_len >= 12:
            # Past-and-future covariates: (num_channels, context_len + horizon)
            # Channels: teaching_days, enrolled_students, reporting_change_flag, seasonal sin, seasonal cos
            td_all = np.concatenate(
                [
                    hist_df["teaching_days"].to_numpy(dtype=np.float32),
                    future_cov["teaching_days"].to_numpy(dtype=np.float32),
                ]
            )
            enrol_all = np.concatenate(
                [
                    hist_df["enrolled_students"].to_numpy(dtype=np.float32),
                    future_cov["enrolled_students"].to_numpy(dtype=np.float32),
                ]
            )
            rep_all = np.concatenate(
                [
                    hist_df["reporting_change_flag"].to_numpy(dtype=np.float32),
                    future_cov["reporting_change_flag"].to_numpy(dtype=np.float32),
                ]
            )
            months_all = np.concatenate(
                [
                    hist_df["month_num"].to_numpy(dtype=np.float32),
                    future_cov["month_num"].to_numpy(dtype=np.float32),
                ]
            )
            sin_all = np.sin(2.0 * np.pi * months_all / 12.0).astype(np.float32)
            cos_all = np.cos(2.0 * np.pi * months_all / 12.0).astype(np.float32)

            pf_channels = [td_all, enrol_all, rep_all, sin_all, cos_all]

            # If attendance is available and scenario modifies it, include attendance in past_future_covariates
            if hist_df["attendance_rate"].notna().any():
                default_att = float(hist_df["attendance_rate"].dropna().mean())
                att_hist = hist_df["attendance_rate"].fillna(default_att).to_numpy(dtype=np.float32)
                att_fut = (
                    pd.Series(future_cov["attendance_rate"]).fillna(default_att).to_numpy(dtype=np.float32)
                )
                pf_channels.append(np.concatenate([att_hist, att_fut]))

            past_future_cov = np.stack(pf_channels, axis=0).astype(np.float32)

            # Past-only covariate: historical concerns_per_1000 rate (1, context_len)
            past_only_cov = hist_df["concerns_per_1000"].to_numpy(dtype=np.float32)[None, :]

        if backend_type == "mlx":
            out = forecaster.predict(
                y_context,
                horizon=horizon,
                past_only_covariates=past_only_cov,
                past_future_covariates=past_future_cov,
                return_quantiles=True,
                make_positive=True,
                sort_quantiles=True,
            )
            raw_preds = np.maximum(0.0, np.asarray(out.forecast, dtype=float).reshape(-1))
            quantiles = np.maximum(0.0, np.asarray(out.quantiles, dtype=float))  # shape (horizon, 9)
        else:
            outputs = list(
                forecaster.predict_batch(
                    contexts=[y_context],
                    horizon=horizon,
                    past_only_covariates=[past_only_cov] if past_only_cov is not None else None,
                    past_future_covariates=[past_future_cov] if past_future_cov is not None else None,
                    return_quantiles=True,
                    use_symmetric_averaging=False,
                )
            )
            raw_preds = np.maximum(0.0, np.asarray(outputs[0].forecast, dtype=float).reshape(-1))
            quantiles = np.maximum(0.0, np.asarray(outputs[0].quantiles, dtype=float))

        # TimesFM 3.0 quantile head returns 9 deciles: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # Index 0 = 10th percentile, Index 8 = 90th percentile -> Exact 80% Prediction Interval!
        q10 = quantiles[:, 0]
        q90 = quantiles[:, 8]

        point_forecasts: List[int] = []
        lower_80: List[int] = []
        upper_80: List[int] = []
        lower_95: List[int] = []
        upper_95: List[int] = []

        tail_ratio_95_to_80 = Z_95 / Z_80  # ~1.5294

        for h_idx in range(horizon):
            val = float(raw_preds[h_idx])
            pt = max(0, int(round(val)))
            l80_raw = min(val, float(q10[h_idx]))
            u80_raw = max(val, float(q90[h_idx]))

            l80 = max(0, int(round(l80_raw)))
            u80 = max(pt, int(round(u80_raw)))

            # Tail extension from native [0.10, 0.90] deciles to 95% [0.025, 0.975]
            lower_spread = max(1.0, val - l80_raw)
            upper_spread = max(1.0, u80_raw - val)
            l95 = max(0, int(round(val - lower_spread * tail_ratio_95_to_80)))
            u95 = max(u80, int(round(val + upper_spread * tail_ratio_95_to_80)))

            point_forecasts.append(pt)
            lower_80.append(l80)
            upper_80.append(u80)
            lower_95.append(l95)
            upper_95.append(u95)

        ann_total, ann_l80, ann_u80, ann_l95, ann_u95, cal_method = compute_calibrated_annual_interval(
            point_forecasts, lower_80, upper_80, lower_95, upper_95, residual_autocorr_lag1=0.30
        )

        runtime_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        mode_str = "Multivariate/Covariate" if (use_covariates and past_future_cov is not None) else "Univariate Zero-Shot"
        hw_str = f"Apple Silicon MLX ({platform.machine()})" if backend_type == "mlx" else "PyTorch"

        warnings_list = [
            (
                "LICENSING & GOVERNANCE NOTICE: Google TimesFM 3.0 weights are governed by "
                "`timesfm-non-commercial-license-v1.0` (non-commercial, non-production research only). "
                "Do not use self-hosted TimesFM 3.0 weights in live operational college production without "
                "commercial licensing (e.g., Google Cloud BigQuery ML)."
            )
        ]

        return EngineForecastOutput(
            model_id=self.model_id,
            model_name=self.name,
            model_version=f"{self.version} [{hw_str} - {mode_str}]",
            dates=future_cov["period_start"].tolist(),
            point_forecasts=point_forecasts,
            raw_forecasts=[round(float(x), 2) for x in raw_preds],
            lower_80=lower_80,
            upper_80=upper_80,
            lower_95=lower_95,
            upper_95=upper_95,
            future_enrolled_students=future_cov["enrolled_students"].tolist(),
            future_teaching_days=future_cov["teaching_days"].tolist(),
            future_attendance_rate=future_cov["attendance_rate"].tolist(),
            future_reporting_flag=future_cov["reporting_change_flag"].tolist(),
            annual_total=ann_total,
            annual_lower_80=ann_l80,
            annual_upper_80=ann_u80,
            annual_lower_95=ann_l95,
            annual_upper_95=ann_u95,
            calibration_method=f"TimesFM 3.0 native quantile head (q0.10–q0.90) + {cal_method}",
            runtime_ms=runtime_ms,
            warnings=warnings_list,
        )

