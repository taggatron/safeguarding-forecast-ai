"""
Staged Rollout Controller & Pluggable Runtime Modes (`esoteric_math/retrospective/rollout_config.py`).

Supports incremental, risk-managed adoption of esoteric mathematical features on top of
100% immutable baseline models via `RETRO_MODE`:
  1. `TELEMETRY_ONLY`: Zero forecast modification (`y_hat_final == y_hat_frozen_base`);
     computes and mounts only Hilbert Phase Clocks, Hawkes Contagion Cascades, and Sheaf Alerts.
  2. `STAGED_CORE_4`: Default high-evidence post-hoc residual stacking using `f01`, `f02`, `f03`, `f07`
     (with optional per-module toggles for `f04`, `f05`, `f06`, `f08..f13`).
  3. `FULL_13_GATED`: Evaluates all 13 mathematical modules against baseline residuals under
     strict 4-period Block-BorutaSHAP and Greedy 1-SE quarantine gates.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
import os
from typing import Dict, List, Optional, Sequence, Tuple


class RolloutMode(str, Enum):
    """Switchable retrospective rollout modes."""

    TELEMETRY_ONLY = "TELEMETRY_ONLY"
    STAGED_CORE_4 = "STAGED_CORE_4"
    FULL_13_GATED = "FULL_13_GATED"

    @classmethod
    def coerce(cls, value: Optional[str | "RolloutMode"]) -> "RolloutMode":
        if isinstance(value, cls):
            return value
        if value is None or str(value).strip() == "":
            env_val = os.environ.get("RETRO_MODE", cls.STAGED_CORE_4.value).strip().upper()
            return cls(env_val) if env_val in cls._value2member_map_ else cls.STAGED_CORE_4
        norm = str(value).strip().upper().replace("-", "_")
        if norm not in cls._value2member_map_:
            raise ValueError(
                f"Invalid RETRO_MODE '{value}'. Expected one of {[m.value for m in cls]}."
            )
        return cls(norm)


CORE_4_MODULES: Tuple[str, ...] = ("f01", "f02", "f03", "f07")

ALL_13_MODULES: Tuple[str, ...] = (
    "f01",
    "f02",
    "f03",
    "f04",
    "f05",
    "f06",
    "f07",
    "f08",
    "f09",
    "f10",
    "f11",
    "f12",
    "f13",
)

TELEMETRY_UI_MODULES: Tuple[str, ...] = ("f01", "f03", "f08", "f11", "f12", "f13")

MODULE_DISPLAY_NAMES: Dict[str, str] = {
    "f01": "Warped Fourier & Hilbert Phase",
    "f02": "Boundary-Causal MODWT Wavelets",
    "f03": "Hawkes Contagion Intensity",
    "f04": "Takens TDA Persistent Homology",
    "f05": "Graph Spectral & Ollivier-Ricci",
    "f06": "Koopman DMD & Tropical Algebra",
    "f07": "Depth-3 Rough Path Signatures",
    "f08": "Cellular Sheaf Cohomology",
    "f09": "p-adic & Poincare Hyperbolic",
    "f10": "Optimal Transport & Barycenters",
    "f11": "RMT Marchenko-Pastur & TW1",
    "f12": "Fisher-Rao NB Info Geometry",
    "f13": "Fractional Diff & Sandpile SOC",
}


@dataclass(frozen=True)
class RolloutConfig:
    """Immutable runtime configuration for retrospective residual stacking and vintage backfill."""

    mode: RolloutMode = RolloutMode.STAGED_CORE_4
    extra_staged_modules: Tuple[str, ...] = ()
    disabled_modules: Tuple[str, ...] = ()
    enable_fisher_rao_calibrator: bool = True
    boruta_n_trials: int = 25
    boruta_min_hits: int = 18
    boruta_block_size: int = 4
    boruta_alpha: float = 0.05
    one_se_inner_splits: int = 3
    glm_max_vif: float = 5.0
    min_burn_in_weeks: int = 52
    min_burn_in_days: int = 190
    lag_buffer_days_audit: Tuple[int, ...] = (0, 2, 5)

    def active_candidate_modules(self) -> Tuple[str, ...]:
        """Return ordered tuple of esoteric module IDs eligible for residual stacking."""
        if self.mode == RolloutMode.TELEMETRY_ONLY:
            return ()

        disabled = set(self.disabled_modules)
        if self.mode == RolloutMode.STAGED_CORE_4:
            raw: List[str] = list(CORE_4_MODULES) + [
                m for m in self.extra_staged_modules if m in ALL_13_MODULES and m not in CORE_4_MODULES
            ]
        else:
            raw = list(ALL_13_MODULES)

        seen = set()
        ordered: List[str] = []
        for mod_id in raw:
            if mod_id in ALL_13_MODULES and mod_id not in disabled and mod_id not in seen:
                seen.add(mod_id)
                ordered.append(mod_id)
        return tuple(ordered)

    @classmethod
    def from_mode(
        cls,
        mode: Optional[str | RolloutMode] = None,
        extra_staged_modules: Optional[Sequence[str]] = None,
        disabled_modules: Optional[Sequence[str]] = None,
        enable_fisher_rao_calibrator: bool = True,
    ) -> "RolloutConfig":
        coerced = RolloutMode.coerce(mode)
        extra_env = os.environ.get("RETRO_EXTRA_MODULES", "").strip()
        env_mods = [m.strip().lower() for m in extra_env.split(",") if m.strip()] if extra_env else []
        merged_extra = tuple(dict.fromkeys(list(extra_staged_modules or ()) + env_mods))
        merged_disabled = tuple(dict.fromkeys(disabled_modules or ()))
        return cls(
            mode=coerced,
            extra_staged_modules=merged_extra,
            disabled_modules=merged_disabled,
            enable_fisher_rao_calibrator=enable_fisher_rao_calibrator,
        )

    def to_badge_dict(self) -> Dict[str, object]:
        active = self.active_candidate_modules()
        descriptions = {
            RolloutMode.TELEMETRY_ONLY: "Zero forecast modification (exact frozen baseline identity); causal UI telemetry active.",
            RolloutMode.STAGED_CORE_4: "Post-hoc residual stacking restricted to Core-4 high-evidence modules (f01, f02, f03, f07).",
            RolloutMode.FULL_13_GATED: "All 13 mathematical modules evaluated under Block-BorutaSHAP + Greedy 1-SE residual quarantine.",
        }
        return {
            "retro_mode": self.mode.value,
            "description": descriptions[self.mode],
            "eligible_modules": list(active),
            "eligible_module_labels": [f"{m}: {MODULE_DISPLAY_NAMES[m]}" for m in active],
            "fisher_rao_calibrator_enabled": bool(
                self.enable_fisher_rao_calibrator and self.mode != RolloutMode.TELEMETRY_ONLY
            ),
        }


def parse_rollout_cli_args(argv: Optional[Sequence[str]] = None) -> RolloutConfig:
    """Parse CLI flags `--retro-mode`, `--extra-module`, and `--disable-fisher-rao`."""
    parser = argparse.ArgumentParser(description="Retrospective Esoteric Rollout & Backfill Controller")
    parser.add_argument(
        "--retro-mode",
        type=str,
        default=os.environ.get("RETRO_MODE", RolloutMode.STAGED_CORE_4.value),
        choices=[m.value for m in RolloutMode],
        help="Rollout mode: TELEMETRY_ONLY, STAGED_CORE_4, or FULL_13_GATED",
    )
    parser.add_argument(
        "--extra-module",
        action="append",
        default=[],
        help="Optional extra module ID (e.g. f06, f08) to toggle onto STAGED_CORE_4",
    )
    parser.add_argument(
        "--disable-module",
        action="append",
        default=[],
        help="Optional module ID to quarantine manually",
    )
    parser.add_argument(
        "--disable-fisher-rao",
        action="store_true",
        help="Disable post-hoc Fisher-Rao interval calibrator",
    )
    args, _ = parser.parse_known_args(argv)
    return RolloutConfig.from_mode(
        mode=args.retro_mode,
        extra_staged_modules=args.extra_module,
        disabled_modules=args.disable_module,
        enable_fisher_rao_calibrator=not args.disable_fisher_rao,
    )

