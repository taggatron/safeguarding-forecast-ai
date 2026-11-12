"""
Retrospective Delta-Patch Subpackage (`esoteric_math/retrospective/`).

Provides:
  - `RolloutMode` & `RolloutConfig` (`TELEMETRY_ONLY`, `STAGED_CORE_4`, `FULL_13_GATED`)
  - `FrozenBaselineAdapter` & SHA-256 bit-for-bit immutability verifiers
  - `PostHocResidualStacker`, `BlockBorutaSHAPGate`, `GreedyOneSEModuleSelector`,
    `GLMResidualOffsetHead`, `GBDTInitScoreResidualHead`, `FisherRaoResidualCalibrator`
  - `VintageSnapshotFilter`, `AcademicYearTopologyManager`, `ExpandingReplayRunner`,
    `run_five_way_retrospective_benchmark`, `run_reporting_lag_sensitivity_audit`
  - `ReplayTelemetryBus` & `extract_vintage_replay_payload`
"""
from __future__ import annotations

from esoteric_math.retrospective.frozen_adapter import (
    FrozenBaselineAdapter,
    FrozenBaselineMutationError,
    clone_prediction_bundle,
    compute_model_state_sha256,
    compute_prediction_bundle_sha256,
)
from esoteric_math.retrospective.replay_telemetry_bus import (
    ReplayTelemetryBus,
    extract_vintage_replay_payload,
    replay_telemetry_bus,
)
from esoteric_math.retrospective.residual_stacker import (
    BlockBorutaSHAPGate,
    FisherRaoResidualCalibrator,
    GBDTInitScoreResidualHead,
    GLMResidualOffsetHead,
    GreedyOneSEModuleSelector,
    PostHocResidualStacker,
    ResidualGateAudit,
    compute_effective_temporal_sample_size,
)
from esoteric_math.retrospective.rollout_config import (
    ALL_13_MODULES,
    CORE_4_MODULES,
    MODULE_DISPLAY_NAMES,
    RolloutConfig,
    RolloutMode,
    parse_rollout_cli_args,
)
from esoteric_math.retrospective.vintage_backfill import (
    AcademicYearTopologyManager,
    ExpandingReplayRunner,
    SeptemberTopologyState,
    VintageAuditMetadata,
    VintageSnapshotFilter,
    fit_vintage_esoteric_suite,
    fit_vintage_telemetry_models,
    run_five_way_retrospective_benchmark,
    run_reporting_lag_sensitivity_audit,
)

__all__ = [
    "ALL_13_MODULES",
    "CORE_4_MODULES",
    "MODULE_DISPLAY_NAMES",
    "RolloutConfig",
    "RolloutMode",
    "parse_rollout_cli_args",
    "FrozenBaselineAdapter",
    "FrozenBaselineMutationError",
    "clone_prediction_bundle",
    "compute_model_state_sha256",
    "compute_prediction_bundle_sha256",
    "BlockBorutaSHAPGate",
    "FisherRaoResidualCalibrator",
    "GBDTInitScoreResidualHead",
    "GLMResidualOffsetHead",
    "GreedyOneSEModuleSelector",
    "PostHocResidualStacker",
    "ResidualGateAudit",
    "compute_effective_temporal_sample_size",
    "AcademicYearTopologyManager",
    "ExpandingReplayRunner",
    "SeptemberTopologyState",
    "VintageAuditMetadata",
    "VintageSnapshotFilter",
    "fit_vintage_esoteric_suite",
    "fit_vintage_telemetry_models",
    "run_five_way_retrospective_benchmark",
    "run_reporting_lag_sensitivity_audit",
    "ReplayTelemetryBus",
    "extract_vintage_replay_payload",
    "replay_telemetry_bus",
]

