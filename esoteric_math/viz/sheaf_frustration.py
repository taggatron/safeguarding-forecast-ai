"""
Visual Spec 3: Sheaf Frustration & Contextual Masking Alert Board (`esoteric_math/viz/sheaf_frustration.py`).

Three-Part Pastoral Triage View:
  1. Top Strip: Rolling 60-step Global Sheaf Dirichlet Energy E_F(x_tau) = x_tau^T L_F x_tau
     against training-fold 90th-percentile warning (tau_90) and 97.5th-percentile crimson alert (tau_97.5).
  2. Left Panel (55% width): Deterministic fixed-coordinate Cohort / Cluster Sheaf Frustration Topology
     badged as either "Cohort Hierarchy Graph" or "Latent Feature/Regime Graph", with node fill encoding
     local non-harmonic residual norm and frustrated edges (f_e >= tau_90) highlighted with z-scores.
  3. Right Panel (45% width): Ranked Top-3 "Contextual Masking" Pastoral Triage Cards with plain-English
     explanations and Observed vs. Sheaf-Expected micro diverging bar charts.
"""
from __future__ import annotations

from typing import Any, Dict, List
import numpy as np


STALK_FRIENDLY_LABELS = {
    "attendance": "Surface Attendance Stability",
    "engagement": "Workshop / Maths-Eng Engagement",
    "pastoral": "Low-Level Pastoral / Triage Load",
    "referral_pressure": "Cohort Referral Acuity",
    "rate_level": "Incidence Rate Level",
    "momentum": "Short-Horizon Momentum",
    "volatility": "Local Dispersion Volatility",
}


def _friendly_stalk_name(raw: str) -> str:
    low = raw.lower().strip()
    for k, label in STALK_FRIENDLY_LABELS.items():
        if k in low:
            return label
    return raw.replace("_", " ").title()


def _node_severity_color(z_val: float, norm_val: float) -> str:
    if z_val >= 2.0 or norm_val >= 2.1:
        return "#DC2626"  # Crimson 2-sigma breach
    if z_val >= 1.1 or norm_val >= 1.35:
        return "#D97706"  # Amber elevated divergence
    return "#475569"      # Muted slate consistent


def _synthesize_masking_explanation(
    cohort_label: str,
    dim_names: List[str],
    obs_vec: np.ndarray,
    exp_vec: np.ndarray,
    z_score: float,
) -> str:
    diffs = obs_vec - exp_vec
    if len(diffs) == 0:
        return f"{cohort_label} remains aligned with connected peer cohorts across tracked indicators."

    if np.max(np.abs(diffs)) < 0.45 and z_score < 1.0:
        return (
            f"All tracked stalk indicators in {cohort_label} remain within ±0.5 SD of "
            f"sheaf-consistent peer cohort expectations."
        )

    # Check classic contextual masking pattern: positive attendance (+dim 0) paired with divergence in engagement/pastoral
    if len(diffs) >= 3 and diffs[0] >= 0.35 and (diffs[1] <= -0.55 or diffs[2] >= 0.55):
        sec_idx = 1 if abs(diffs[1]) >= abs(diffs[2]) else 2
        sec_name = _friendly_stalk_name(dim_names[sec_idx]).lower()
        return (
            f"High surface attendance ({diffs[0]:+.1f} SD above peer consensus) is masking sharp "
            f"divergence in {sec_name} ({diffs[sec_idx]:+.1f} SD) relative to connected peer cohorts."
        )

    order = np.argsort(np.abs(diffs))[::-1]
    i0 = int(order[0])
    d0_name = _friendly_stalk_name(dim_names[i0]).lower()
    if len(order) > 1:
        i1 = int(order[1])
        d1_name = _friendly_stalk_name(dim_names[i1]).lower()
        return (
            f"Cohomological divergence driven by {d0_name} ({diffs[i0]:+.1f} SD vs. sheaf consensus) "
            f"alongside {d1_name} ({diffs[i1]:+.1f} SD) across relational cohort boundaries."
        )
    return f"Local stalk discrepancy in {d0_name} ({diffs[i0]:+.1f} SD) relative to connected cohorts."


def build_sheaf_frustration_spec(f08_diag: Dict[str, Any], t_idx: int) -> Dict[str, Any]:
    """
    Build strictly causal Sheaf Frustration & Contextual Masking Triage Board payload at `t_idx`
    (reading ONLY indices 0 .. t_idx).
    """
    end = int(t_idx) + 1
    dates = [str(d) for d in f08_diag["dates"][:end]]
    topology_mode = str(f08_diag["topology_mode"])
    node_ids = [str(x) for x in f08_diag["node_ids"]]
    node_labels = [str(x) for x in f08_diag["node_labels"]]
    node_weights = np.maximum(0.02, np.asarray(f08_diag["node_weights"], dtype=float))
    node_weights = node_weights / np.sum(node_weights)
    V = len(node_ids)

    dim_names = [str(d) for d in f08_diag["stalk_dim_names"]]
    obs_stalks_t = np.asarray(f08_diag["observed_stalks"][t_idx], dtype=float)   # (V, d)
    harm_stalks_t = np.asarray(f08_diag["harmonic_stalks"][t_idx], dtype=float)  # (V, d)
    res_stalks_t = np.asarray(f08_diag["residual_stalks"][t_idx], dtype=float)   # (V, d)

    vr_norm_t = np.maximum(0.0, np.asarray(f08_diag["vertex_residual_norm"][t_idx], dtype=float))
    vr_z_t = np.asarray(f08_diag["vertex_residual_z"][t_idx], dtype=float)

    ge_arr = np.maximum(0.0, np.asarray(f08_diag["global_energy"][:end], dtype=float))
    ge_z_arr = np.asarray(f08_diag["global_energy_z"][:end], dtype=float)
    tau90 = float(max(1e-6, f08_diag["energy_tau90"]))
    tau975 = float(max(tau90 * 1.05, f08_diag["energy_tau975"]))

    # 1. Top Strip: Trailing 60-step Global Sheaf Dirichlet Energy Timeline
    strip_start = max(0, end - 60)
    energy_timeline: List[Dict[str, Any]] = []
    for idx in range(strip_start, end):
        ev = float(ge_arr[idx])
        status = "breach" if ev >= tau975 else ("warning" if ev >= tau90 else "calm")
        energy_timeline.append(
            {
                "step_index": int(idx),
                "date": dates[idx],
                "dirichlet_energy": round(ev, 4),
                "energy_z": round(float(ge_z_arr[idx]), 3),
                "status": status,
            }
        )

    # 2. Deterministic fixed-coordinate radial/hierarchical graph layout (never jumps on time-scrub)
    nodes_spec: List[Dict[str, Any]] = []
    for v in range(V):
        ang = -0.5 * np.pi + (2.0 * np.pi * float(v) / float(max(V, 1)))
        rx = 0.0 if V == 1 else float(np.cos(ang))
        ry = 0.0 if V == 1 else float(np.sin(ang))
        z_v = float(vr_z_t[v])
        n_v = float(vr_norm_t[v])
        col_v = _node_severity_color(z_v, n_v)
        rad_px = float(np.clip(14.0 + 34.0 * node_weights[v], 15.0, 28.0))
        nodes_spec.append(
            {
                "index": v,
                "id": node_ids[v],
                "label": node_labels[v],
                "weight": round(float(node_weights[v]), 4),
                "radius_px": round(rad_px, 2),
                "x_unit": round(rx, 4),
                "y_unit": round(ry, 4),
                "residual_norm": round(n_v, 4),
                "frustration_z": round(z_v, 2),
                "fill_color": col_v,
                "is_anomalous": bool(z_v >= 1.1 or n_v >= 1.35),
            }
        )

    raw_edges = list(f08_diag["edges"])
    ef_t = np.maximum(0.0, np.asarray(f08_diag["edge_frustration"][t_idx], dtype=float)) if len(raw_edges) > 0 else np.zeros(0)
    ef_z_t = np.asarray(f08_diag["edge_frustration_z"][t_idx], dtype=float) if len(raw_edges) > 0 else np.zeros(0)
    edge_tau90 = np.asarray(f08_diag["edge_tau90"], dtype=float) if len(raw_edges) > 0 else np.zeros(0)

    edges_spec: List[Dict[str, Any]] = []
    frustrated_count = 0
    for e_i, (u, v, w) in enumerate(raw_edges):
        fe_val = float(ef_t[e_i])
        ez_val = float(ef_z_t[e_i])
        thr_val = float(edge_tau90[e_i])
        is_frust = bool(fe_val >= thr_val and fe_val > 1e-4)
        if is_frust:
            frustrated_count += 1
            sw = float(np.clip(3.0 + 0.85 * max(0.0, ez_val - 1.0), 3.0, 6.0))
            ecol = "#DC2626" if ez_val >= 2.0 else "#D97706"
        else:
            sw = 1.0
            ecol = "#94A3B8"

        edges_spec.append(
            {
                "edge_index": e_i,
                "u": int(u),
                "v": int(v),
                "u_label": node_labels[int(u)],
                "v_label": node_labels[int(v)],
                "weight": round(float(w), 4),
                "frustration_norm": round(fe_val, 4),
                "frustration_z": round(ez_val, 2),
                "tau90": round(thr_val, 4),
                "is_frustrated": is_frust,
                "stroke_width": round(sw, 2),
                "color": ecol,
            }
        )

    # 3. Ranked Top-3 "Contextual Masking" Pastoral Triage Cards
    # Combine vertex non-harmonic z-score with incident max edge z-score for sharp ranking
    vertex_rank_score = np.zeros(V, dtype=float)
    for v in range(V):
        inc_z = [e["frustration_z"] for e in edges_spec if e["u"] == v or e["v"] == v]
        max_inc = max(inc_z) if inc_z else 0.0
        vertex_rank_score[v] = max(float(vr_z_t[v]), 0.75 * max_inc + 0.35 * float(vr_norm_t[v]))

    top_vertices = np.argsort(vertex_rank_score)[::-1][: min(3, V)]
    triage_cards: List[Dict[str, Any]] = []

    for rank_i, v_idx in enumerate(top_vertices):
        v = int(v_idx)
        score_sig = float(max(vr_z_t[v], vertex_rank_score[v]))
        obs_v = obs_stalks_t[v]
        exp_v = harm_stalks_t[v]
        res_v = res_stalks_t[v]

        is_masking = bool(
            (len(res_v) >= 3 and res_v[0] >= 0.35 and (res_v[1] <= -0.55 or res_v[2] >= 0.55))
            or score_sig >= 2.0
        )
        if score_sig >= 2.0:
            badge_text = f"{score_sig:.1f}σ Frustration — Contextual Masking Detected"
            severity = "critical"
            accent = "#DC2626"
        elif score_sig >= 1.1 or is_masking:
            badge_text = f"{max(score_sig, 1.2):.1f}σ Divergence — Pre-Surge Watch"
            severity = "warning"
            accent = "#D97706"
        else:
            badge_text = f"{max(0.1, score_sig):.1f}σ Residual — Globally Consistent"
            severity = "calm"
            accent = "#475569"

        bars: List[Dict[str, Any]] = []
        for d_i, d_name in enumerate(dim_names):
            ov = float(np.clip(obs_v[d_i], -3.5, 3.5))
            ev = float(np.clip(exp_v[d_i], -3.5, 3.5))
            bars.append(
                {
                    "attribute_id": d_name,
                    "attribute_label": _friendly_stalk_name(d_name),
                    "observed_sd": round(ov, 2),
                    "sheaf_expected_sd": round(ev, 2),
                    "discrepancy_sd": round(ov - ev, 2),
                }
            )

        triage_cards.append(
            {
                "rank": rank_i + 1,
                "vertex_index": v,
                "cohort_id": node_ids[v],
                "cohort_label": node_labels[v],
                "frustration_sigma": round(score_sig, 2),
                "header_badge": badge_text,
                "severity": severity,
                "accent_color": accent,
                "explanation": _synthesize_masking_explanation(node_labels[v], dim_names, obs_v, exp_v, score_sig),
                "diverging_bars": bars,
            }
        )

    cur_energy = float(ge_arr[-1])
    active_alert_count = int(sum(1 for c in triage_cards if c["severity"] in ("critical", "warning")))

    return {
        "topology_mode": topology_mode,
        "current": {
            "date": dates[-1],
            "dirichlet_energy": round(cur_energy, 4),
            "energy_z": round(float(ge_z_arr[-1]), 2),
            "tau90": round(tau90, 4),
            "tau975": round(tau975, 4),
            "is_breach_975": bool(cur_energy >= tau975),
            "is_warning_90": bool(cur_energy >= tau90),
            "frustrated_edge_count": frustrated_count,
            "active_masking_alerts": active_alert_count,
            "top_cohort": triage_cards[0]["cohort_label"] if triage_cards else "None",
        },
        "energy_timeline_60": energy_timeline,
        "graph": {
            "nodes": nodes_spec,
            "edges": edges_spec,
        },
        "triage_cards": triage_cards,
    }


def render_sheaf_frustration_svg(spec: Dict[str, Any], width: int = 860, height: int = 400) -> str:
    """Render standalone SVG for the Sheaf Frustration & Contextual Masking Alert Board."""
    strip = spec["energy_timeline_60"]
    tau90 = float(spec["current"]["tau90"])
    tau975 = float(spec["current"]["tau975"])
    max_e = max([tau975 * 1.25] + [float(p["dirichlet_energy"]) for p in strip] + [1.0])

    sx0, sx1, sy0, sy1 = 52.0, 825.0, 24.0, 105.0

    def _sxp(i: int) -> float:
        return sx0 + (sx1 - sx0) * (float(i) / float(max(len(strip) - 1, 1)))

    def _syp(val: float) -> float:
        return sy1 - (sy1 - sy0) * (float(val) / max_e)

    line_pts = " ".join(f"{_sxp(i):.1f},{_syp(p['dirichlet_energy']):.1f}" for i, p in enumerate(strip))

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Sheaf Frustration and Contextual Masking Alert Board">',
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="12" fill="#FFFFFF" />',
        f'<line x1="{sx0}" y1="{_syp(tau90):.1f}" x2="{sx1}" y2="{_syp(tau90):.1f}" stroke="#D97706" stroke-width="1.2" stroke-dasharray="4,3" />',
        f'<line x1="{sx0}" y1="{_syp(tau975):.1f}" x2="{sx1}" y2="{_syp(tau975):.1f}" stroke="#DC2626" stroke-width="1.3" stroke-dasharray="5,3" />',
        f'<polyline fill="none" stroke="#4F46E5" stroke-width="2.2" points="{line_pts}" />',
    ]

    # Left topology graph
    gcx, gcy, gr = 235.0, 258.0, 96.0
    nodes = spec["graph"]["nodes"]
    for edge in spec["graph"]["edges"]:
        u, v = nodes[edge["u"]], nodes[edge["v"]]
        x1, y1 = gcx + u["x_unit"] * gr, gcy + u["y_unit"] * gr
        x2, y2 = gcx + v["x_unit"] * gr, gcy + v["y_unit"] * gr
        op = "0.88" if edge["is_frustrated"] else "0.28"
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{edge["color"]}" stroke-width="{edge["stroke_width"]}" stroke-opacity="{op}" />'
        )
        if edge["is_frustrated"]:
            mx, my = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            parts.append(f'<rect x="{mx - 18:.1f}" y="{my - 9:.1f}" width="36" height="17" rx="8" fill="{edge["color"]}" />')
            parts.append(
                f'<text x="{mx:.1f}" y="{my + 3:.1f}" text-anchor="middle" font-family="Inter,sans-serif" font-size="9.5" font-weight="700" fill="#FFFFFF">{edge["frustration_z"]:+.1f}z</text>'
            )

    for nd in nodes:
        nx, ny = gcx + nd["x_unit"] * gr, gcy + nd["y_unit"] * gr
        parts.append(
            f'<circle cx="{nx:.1f}" cy="{ny:.1f}" r="{nd["radius_px"]:.1f}" fill="{nd["fill_color"]}" stroke="#FFFFFF" stroke-width="2" />'
        )
        parts.append(
            f'<text x="{nx:.1f}" y="{ny + nd["radius_px"] + 13:.1f}" text-anchor="middle" font-family="Inter,sans-serif" font-size="10" font-weight="600" fill="#1E293B">{nd["label"]}</text>'
        )

    # Right triage cards summary inside SVG
    cy_card = 148.0
    for card in spec["triage_cards"]:
        parts.extend(
            [
                f'<rect x="460" y="{cy_card:.1f}" width="372" height="74" rx="8" fill="#F8FAFC" stroke="{card["accent_color"]}" stroke-width="1.4" />',
                f'<text x="474" y="{cy_card + 21:.1f}" font-family="Plus Jakarta Sans,sans-serif" font-size="11.5" font-weight="700" fill="{card["accent_color"]}">{card["cohort_label"]} — {card["header_badge"]}</text>',
                f'<text x="474" y="{cy_card + 44:.1f}" font-family="Inter,sans-serif" font-size="10" fill="#334155">{card["explanation"][:68]}...</text>',
            ]
        )
        cy_card += 82.0

    parts.append('</svg>')
    return "\n".join(parts)
