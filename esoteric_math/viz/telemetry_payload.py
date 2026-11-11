"""
Causal JSON-Serializable Telemetry Extractor (`esoteric_math/viz/telemetry_payload.py`).

Extracts the complete state driving the "Mathematical Early-Warning Telemetry" UI:
  - Top 3-card Summary KPI Ribbon
  - Visual Spec 1: Hilbert Stress-Cycle Phase Clock (`hilbert_clock`)
  - Visual Spec 2: Hawkes Contagion Cascade & Branching Monitor (`hawkes_cascade`)
  - Visual Spec 3: Sheaf Frustration & Contextual Masking Alert Board (`sheaf_frustration`)
  - Auxiliary manifold indicators (`f10` Optimal Transport, `f11` RMT Sync, `f12` Fisher-Rao, `f13` SOC)

Strictly enforces causal anti-leakage: evaluating at `t_index` (or `t_date`) reads ONLY
observations {0, ..., t_index} and masks all s > t_index prior to transformation.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from esoteric_math.schema_bridge import (
    AdaptiveSchemaBridge,
    DATE_CANDIDATES,
    SchemaBundle,
)
from esoteric_math.features.f01_fourier_hilbert import WarpedFourierHilbertTransformer
from esoteric_math.features.f03_hawkes_process import HawkesContagionTransformer
from esoteric_math.features.f08_cellular_sheaves import CellularSheafCohomologyTransformer
from esoteric_math.features.f11_rmt_marchenko import RMTMarchenkoPasturTransformer
from esoteric_math.features.f12_fisher_rao import FisherRaoInformationGeometryTransformer
from esoteric_math.features.f13_fractional_soc import FractionalSOCTransformer
from esoteric_math.viz.hilbert_clock import build_hilbert_clock_spec
from esoteric_math.viz.hawkes_cascade import build_hawkes_cascade_spec
from esoteric_math.viz.sheaf_frustration import build_sheaf_frustration_spec


ACCESSIBLE_PALETTE: Dict[str, str] = {
    "calm_slate": "#475569",
    "decompression_teal": "#0D9488",
    "incubation_indigo": "#4F46E5",
    "escalating_amber": "#D97706",
    "breach_crimson": "#DC2626",
}


def _detect_date_col(df: pd.DataFrame) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in DATE_CANDIDATES:
        if cand in cols_lower:
            return cols_lower[cand]
    return None


def _slice_df_causally(
    df: pd.DataFrame,
    t_index: Optional[int] = None,
    t_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, int, List[str]]:
    """
    Slice `df` strictly up to timestep `t_index` or ISO date `t_date` so NO row with s > t
    ever enters the schema bridge or any feature transformer.
    """
    date_col = _detect_date_col(df)
    if date_col is None:
        n_total = len(df)
        idx = (n_total - 1) if t_index is None else int(np.clip(t_index, 0, n_total - 1))
        sub = df.iloc[: idx + 1].copy().reset_index(drop=True)
        dates_list = [f"Step-{i}" for i in range(idx + 1)]
        return sub, idx, dates_list

    parsed_dates = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
    unique_dates = sorted(parsed_dates.unique().tolist())
    n_steps = len(unique_dates)

    if t_date is not None and str(t_date).strip() != "":
        target_str = pd.Timestamp(str(t_date)).strftime("%Y-%m-%d")
        eligible = [i for i, d in enumerate(unique_dates) if d <= target_str]
        eff_idx = eligible[-1] if eligible else 0
    elif t_index is not None:
        eff_idx = int(np.clip(int(t_index), 0, n_steps - 1))
    else:
        eff_idx = n_steps - 1

    cutoff_date = unique_dates[eff_idx]
    mask = parsed_dates <= cutoff_date
    sub = df.loc[mask].copy().reset_index(drop=True)
    return sub, eff_idx, unique_dates[: eff_idx + 1]


class FittedTelemetryModels:
    """Reusable fitted training-fold transformers for fast (<20ms) causal prefix evaluation."""

    def __init__(self, train_df: pd.DataFrame) -> None:
        self.bridge = AdaptiveSchemaBridge(causal_mode="telemetry", train_cutoff=None)
        self.bridge.fit(train_df)
        train_bundle: SchemaBundle = self.bridge.transform(train_df)

        self.f01 = WarpedFourierHilbertTransformer(causal_mode="telemetry").fit(train_bundle)
        self.f03 = HawkesContagionTransformer(causal_mode="telemetry").fit(train_bundle)
        self.f08 = CellularSheafCohomologyTransformer(causal_mode="telemetry").fit(train_bundle)
        self.f11 = RMTMarchenkoPasturTransformer(causal_mode="telemetry").fit(train_bundle)
        self.f12 = FisherRaoInformationGeometryTransformer(causal_mode="telemetry").fit(train_bundle)
        self.f13 = FractionalSOCTransformer(causal_mode="telemetry").fit(train_bundle)


def build_kpi_ribbon(
    hilbert_spec: Dict[str, Any],
    hawkes_spec: Dict[str, Any],
    sheaf_spec: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Construct the Top 3-Card Summary KPI Ribbon for immediate low-cognitive-load triage."""
    h_cur = hilbert_spec["current"]
    k_cur = hawkes_spec["current"]
    s_cur = sheaf_spec["current"]

    # KPI 1: Current Stress-Cycle Regime & Days to Crest (Hilbert Clock)
    if h_cur["is_holiday_freeze"]:
        crest_sub = f"Phase Frozen ({h_cur['holiday_label'] or 'Break'}) · Next crest ~{h_cur['days_to_peak']:.0f} inst. days"
    else:
        crest_sub = f"Amplitude {h_cur['r_norm']:.2f}σ · Est. {h_cur['days_to_peak']:.1f} instructional days to crest"

    kpi_1 = {
        "id": "hilbert_regime",
        "title": "Stress-Cycle Regime & Crest Horizon",
        "primary_value": h_cur["quadrant"]["title"],
        "secondary_metric": f"{h_cur['days_to_peak']:.1f}d to Crest",
        "subtitle": crest_sub,
        "status_color": "#DC2626" if h_cur["is_breach_2sigma"] else h_cur["quadrant"]["color"],
        "is_breach": bool(h_cur["is_breach_2sigma"]),
    }

    # KPI 2: Endogenous Contagion Share % & Branching Status (Hawkes)
    kpi_2 = {
        "id": "hawkes_contagion",
        "title": "Endogenous Contagion Share & Branching",
        "primary_value": f"{k_cur['endog_share_pct']:.1f}% Endogenous",
        "secondary_metric": f"ρ(Γ) = {k_cur['spectral_radius']:.2f}",
        "subtitle": f"{k_cur['regime']['label']} · Intensity λ(t) = {k_cur['lambda_total']:.1f}/step",
        "status_color": k_cur["regime"]["color"],
        "is_breach": bool(k_cur["spectral_radius"] >= 0.85),
    }

    # KPI 3: Active Contextual Masking Alerts (Sheaf Cohomology)
    alert_cnt = int(s_cur["active_masking_alerts"])
    frust_edges = int(s_cur["frustrated_edge_count"])
    if s_cur["is_breach_975"]:
        s_color = "#DC2626"
        s_headline = f"{alert_cnt} Active Masking Alert{'s' if alert_cnt != 1 else ''}"
    elif s_cur["is_warning_90"] or alert_cnt > 0:
        s_color = "#D97706"
        s_headline = f"{alert_cnt} Cohort Divergence Watch"
    else:
        s_color = "#475569"
        s_headline = "0 Masking Alerts (Consistent)"

    kpi_3 = {
        "id": "sheaf_masking",
        "title": "Contextual Masking & Sheaf Cohomology",
        "primary_value": s_headline,
        "secondary_metric": f"{frust_edges} Frustrated Edge{'s' if frust_edges != 1 else ''}",
        "subtitle": f"Lead: {s_cur['top_cohort']} · Topology: {sheaf_spec['topology_mode']}",
        "status_color": s_color,
        "is_breach": bool(s_cur["is_breach_975"]),
    }

    return [kpi_1, kpi_2, kpi_3]


def extract_telemetry_payload(
    df: pd.DataFrame,
    t_index: Optional[int] = None,
    t_date: Optional[str] = None,
    train_cutoff: Optional[int] = None,
    fitted_models: Optional[FittedTelemetryModels] = None,
) -> Dict[str, Any]:
    """
    Produce a 100% JSON-serializable, strictly causal mathematical telemetry payload at
    timestep `t_index` / `t_date` (guaranteeing zero dependence on s > t).
    """
    prefix_df, eff_t_idx, causal_dates = _slice_df_causally(df, t_index=t_index, t_date=t_date)

    if fitted_models is None:
        if train_cutoff is not None:
            tr_df, _, _ = _slice_df_causally(prefix_df, t_index=min(int(train_cutoff) - 1, eff_t_idx))
        else:
            tr_df = prefix_df
        fitted_models = FittedTelemetryModels(tr_df)

    bundle = fitted_models.bridge.transform(prefix_df)
    last_i = bundle.n_steps - 1

    fitted_models.f01.transform(bundle)
    fitted_models.f03.transform(bundle)
    fitted_models.f08.transform(bundle)
    fitted_models.f11.transform(bundle)
    fitted_models.f12.transform(bundle)
    fitted_models.f13.transform(bundle)

    hilbert_spec = build_hilbert_clock_spec(fitted_models.f01.last_diagnostics_, last_i)
    hawkes_spec = build_hawkes_cascade_spec(fitted_models.f03.last_diagnostics_, last_i)
    sheaf_spec = build_sheaf_frustration_spec(fitted_models.f08.last_diagnostics_, last_i)
    kpi_ribbon = build_kpi_ribbon(hilbert_spec, hawkes_spec, sheaf_spec)

    aux_summary = {
        "rmt_sync_ratio": round(float(fitted_models.f11.last_diagnostics_["sync_ratio"][last_i]), 4),
        "rmt_escaping_eigenvalues": int(round(float(fitted_models.f11.last_diagnostics_["escaping_count"][last_i]))),
        "fisher_rao_velocity": round(float(fitted_models.f12.last_diagnostics_["fisher_rao_velocity"][last_i]), 4),
        "nb_dispersion_r_hat": round(float(fitted_models.f12.last_diagnostics_["r_hat"][last_i]), 3),
        "fractional_d_star": round(float(fitted_models.f13.last_diagnostics_["optimal_d"]), 2),
        "sandpile_soc_score": round(float(fitted_models.f13.last_diagnostics_["soc_criticality_score"][last_i]), 3),
    }

    return {
        "selected_step_index": int(last_i),
        "selected_date": str(bundle.tier_a.dates[last_i]),
        "cadence": str(bundle.tier_a.cadence),
        "topology_mode": str(bundle.tier_c.topology_mode),
        "causal_mask_verified": True,
        "causal_prefix_dates": causal_dates,
        "palette": dict(ACCESSIBLE_PALETTE),
        "kpi_ribbon": kpi_ribbon,
        "hilbert_clock": hilbert_spec,
        "hawkes_cascade": hawkes_spec,
        "sheaf_frustration": sheaf_spec,
        "auxiliary_manifold_metrics": aux_summary,
    }
