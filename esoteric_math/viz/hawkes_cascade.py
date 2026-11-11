"""
Visual Spec 2: Hawkes Contagion Cascade & Branching Monitor (`esoteric_math/viz/hawkes_cascade.py`).

Synchronized Two-Column Contagion View:
  - Left Panel (65% width): Stacked Exogenous Baseline (#475569 slate) + Category Endogenous
    Excitation Ribbons over trailing 45 instructional steps up to t, PLUS 5-step causal
    Hawkes exponential decay projection cone (t, t+5] assuming zero new exogenous shocks,
    overlaid with observed count stems and >0.35 endogenous-share shock callout pins.
  - Right Panel (35% width): Directed Weighted Cross-Category Contagion Network (hiding
    negligible edges Gamma_{cj} < 0.03, labeled with decay half-life tau_{1/2} in days;
    or annotated self-excitation card when C == 1) + Horizontal Spectral Criticality Gauge
    for rho(Gamma) in [0, 1.2].
"""
from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
import pandas as pd


CATEGORY_PALETTE = [
    "#4F46E5",  # Indigo
    "#0D9488",  # Teal
    "#D97706",  # Amber
    "#7C3AED",  # Violet
    "#0284C7",  # Sky
    "#EA580C",  # Deep Orange
]

CATEGORY_DISPLAY = {
    "mental_health": "Mental Health & Wellbeing",
    "abuse_neglect": "Abuse & Neglect",
    "online_safety": "Online Safety & Peer",
    "exploitation": "Exploitation & Missing",
    "other_concerns": "Other Welfare Concerns",
    "aggregate_referrals": "Aggregate Referrals",
}


def _pretty_category_label(raw: str) -> str:
    clean = raw.replace("cat_", "").strip()
    if clean in CATEGORY_DISPLAY:
        return CATEGORY_DISPLAY[clean]
    return clean.replace("_", " ").title()


def classify_branching_regime(rho: float) -> Dict[str, str]:
    if rho < 0.60:
        return {
            "zone": "damped",
            "label": "Damped / Isolated",
            "color": "#0D9488",
            "summary": "Incidents remain largely isolated; secondary contagion decays rapidly.",
        }
    if rho < 0.85:
        return {
            "zone": "amplifying",
            "label": "Amplifying Contagion",
            "color": "#D97706",
            "summary": "Moderate peer/cross-category clustering active; monitor secondary disclosures.",
        }
    return {
        "zone": "critical",
        "label": "Near-Critical Cascade Alert",
        "color": "#DC2626",
        "summary": "High endogenous branching ratio; proactive group pastoral intervention advised.",
    }


def build_hawkes_cascade_spec(f03_diag: Dict[str, Any], t_idx: int) -> Dict[str, Any]:
    """
    Build strictly causal Hawkes Contagion Cascade & Branching Monitor payload at `t_idx`
    (reading ONLY indices 0 .. t_idx).
    """
    end = int(t_idx) + 1
    dates = [str(d) for d in f03_diag["dates"][:end]]
    cat_raw = [str(c) for c in f03_diag["category_names"]]
    cat_labels = [_pretty_category_label(c) for c in cat_raw]
    C = len(cat_raw)

    exog_arr = np.maximum(0.0, np.asarray(f03_diag["exog_total"][:end], dtype=float))
    endog_src_arr = np.maximum(0.0, np.asarray(f03_diag["endog_by_source"][:end], dtype=float))
    lam_tot_arr = np.maximum(1e-6, np.asarray(f03_diag["lambda_total"][:end], dtype=float))
    share_arr = np.clip(np.asarray(f03_diag["endog_share"][:end], dtype=float), 0.0, 0.999)
    obs_tot_arr = np.maximum(0.0, np.asarray(f03_diag["observed_total_counts"][:end], dtype=float))

    # Trailing 45 steps up to t_idx
    win_start = max(0, end - 45)
    ribbon_series: List[Dict[str, Any]] = []
    callout_pins: List[Dict[str, Any]] = []

    for idx in range(win_start, end):
        src_vals = [round(float(endog_src_arr[idx, j]), 4) for j in range(C)]
        tot_e = float(np.sum(src_vals))
        sh = float(share_arr[idx])
        is_shock_pin = bool(sh > 0.35 and tot_e > 0.25)
        entry = {
            "step_index": int(idx),
            "date": dates[idx],
            "is_projection": False,
            "exog_baseline": round(float(exog_arr[idx]), 4),
            "endog_by_source": src_vals,
            "endog_total": round(tot_e, 4),
            "lambda_total": round(float(lam_tot_arr[idx]), 4),
            "observed_count": round(float(obs_tot_arr[idx]), 2),
            "endog_share": round(sh, 4),
            "is_shock_callout": is_shock_pin,
        }
        ribbon_series.append(entry)
        if is_shock_pin:
            callout_pins.append(
                {
                    "date": dates[idx],
                    "endog_share_pct": round(sh * 100.0, 1),
                    "lambda_total": round(float(lam_tot_arr[idx]), 2),
                    "observed_count": round(float(obs_tot_arr[idx]), 1),
                }
            )

    # 5-step strictly causal decay projection cone (t+1 .. t+5) assuming ZERO new exogenous shocks
    mem_anchor = np.maximum(0.0, np.asarray(f03_diag["memory_cone_anchor"][t_idx], dtype=float))  # (C, C)
    beta_mat = np.maximum(1e-4, np.asarray(f03_diag["beta_matrix"], dtype=float))
    cadence_days = float(max(1.0, f03_diag.get("cadence_days", 1.0)))
    last_dt = pd.Timestamp(dates[-1])
    # Anchor routine baseline on trailing active median up to t_idx
    recent_exog = float(np.median(exog_arr[max(0, end - 10) : end]))

    projection_cone: List[Dict[str, Any]] = []
    for h in range(1, 6):
        proj_dt = last_dt + pd.Timedelta(days=int(round(h * cadence_days)))
        mem_h = mem_anchor * np.exp(-beta_mat * float(h))
        src_h = [round(float(np.sum(mem_h[:, j])), 4) for j in range(C)]
        endog_h = float(np.sum(src_h))
        lam_h = recent_exog + endog_h
        uncert_band = round(float(1.28 * np.sqrt(max(lam_h, 0.25))), 3)
        projection_cone.append(
            {
                "horizon_step": h,
                "date": proj_dt.strftime("%Y-%m-%d"),
                "is_projection": True,
                "exog_baseline": round(recent_exog, 4),
                "endog_by_source": src_h,
                "endog_total": round(endog_h, 4),
                "lambda_total": round(lam_h, 4),
                "upper_cone": round(lam_h + uncert_band, 4),
                "lower_cone": round(max(0.0, lam_h - uncert_band), 4),
            }
        )

    # Cross-category directed branching graph (hiding negligible Gamma_{cj} < 0.03)
    gamma_mat = np.maximum(0.0, np.asarray(f03_diag["gamma_matrix"], dtype=float))
    hl_mat = np.maximum(0.05, np.asarray(f03_diag["half_life_days"], dtype=float))
    rho = float(np.clip(f03_diag["spectral_radius"], 0.0, 1.49))
    regime = classify_branching_regime(rho)

    nodes: List[Dict[str, Any]] = []
    for i in range(C):
        ang = -0.5 * np.pi + (2.0 * np.pi * float(i) / float(max(C, 1)))
        rx = 0.0 if C == 1 else float(np.cos(ang))
        ry = 0.0 if C == 1 else float(np.sin(ang))
        nodes.append(
            {
                "index": i,
                "id": cat_raw[i],
                "label": cat_labels[i],
                "color": CATEGORY_PALETTE[i % len(CATEGORY_PALETTE)],
                "x_unit": round(rx, 4),
                "y_unit": round(ry, 4),
                "self_branching": round(float(gamma_mat[i, i]), 4),
                "self_half_life_days": round(float(hl_mat[i, i]), 2),
                "active_source_load": round(float(endog_src_arr[-1, i]), 3),
            }
        )

    edges: List[Dict[str, Any]] = []
    for j in range(C):
        for c in range(C):
            g_val = float(gamma_mat[c, j])
            if g_val < 0.03:
                continue
            edges.append(
                {
                    "source_idx": j,
                    "target_idx": c,
                    "source_label": cat_labels[j],
                    "target_label": cat_labels[c],
                    "is_self_loop": bool(j == c),
                    "branching_ratio": round(g_val, 4),
                    "half_life_days": round(float(hl_mat[c, j]), 2),
                    "stroke_width": round(float(np.clip(1.2 + 9.5 * g_val, 1.2, 6.0)), 2),
                    "color": CATEGORY_PALETTE[j % len(CATEGORY_PALETTE)],
                }
            )
    edges.sort(key=lambda e: e["branching_ratio"], reverse=True)

    cur_share = float(share_arr[-1])
    return {
        "current": {
            "date": dates[-1],
            "exog_baseline": round(float(exog_arr[-1]), 3),
            "endog_total": round(float(np.sum(endog_src_arr[-1])), 3),
            "lambda_total": round(float(lam_tot_arr[-1]), 3),
            "observed_count": round(float(obs_tot_arr[-1]), 1),
            "endog_share": round(cur_share, 4),
            "endog_share_pct": round(cur_share * 100.0, 1),
            "spectral_radius": round(rho, 4),
            "regime": regime,
        },
        "categories": [
            {"index": i, "id": cat_raw[i], "label": cat_labels[i], "color": CATEGORY_PALETTE[i % len(CATEGORY_PALETTE)]}
            for i in range(C)
        ],
        "is_univariate_fallback": bool(C == 1),
        "ribbon_series": ribbon_series,
        "projection_cone": projection_cone,
        "shock_callouts": callout_pins[-5:],
        "network": {
            "nodes": nodes,
            "edges": edges,
            "min_edge_threshold": 0.03,
        },
        "criticality_gauge": {
            "value": round(rho, 4),
            "min": 0.0,
            "max": 1.2,
            "zones": [
                {"from": 0.0, "to": 0.60, "label": "Damped / Isolated", "color": "#0D9488"},
                {"from": 0.60, "to": 0.85, "label": "Amplifying Contagion", "color": "#D97706"},
                {"from": 0.85, "to": 1.20, "label": "Near-Critical Cascade Alert", "color": "#DC2626"},
            ],
        },
    }


def render_hawkes_cascade_svg(spec: Dict[str, Any], width: int = 860, height: int = 380) -> str:
    """Render standalone synchronized two-column SVG for Hawkes Contagion Cascade & Branching Monitor."""
    hist = spec["ribbon_series"]
    cone = spec["projection_cone"]
    cats = spec["categories"]
    C = len(cats)

    left_x0, left_x1 = 48.0, 525.0
    plot_y0, plot_y1 = 38.0, 295.0
    total_pts = max(len(hist) + len(cone), 2)

    max_y = 1.0
    for r in hist:
        max_y = max(max_y, float(r["lambda_total"]), float(r["observed_count"]))
    for c_pt in cone:
        max_y = max(max_y, float(c_pt["upper_cone"]))
    max_y *= 1.15

    def _xp(i: int) -> float:
        return left_x0 + (left_x1 - left_x0) * (float(i) / float(max(total_pts - 1, 1)))

    def _yp(val: float) -> float:
        return plot_y1 - (plot_y1 - plot_y0) * (float(val) / max_y)

    all_steps = hist + cone
    exog_poly = (
        [(left_x0, plot_y1)]
        + [(_xp(i), _yp(s["exog_baseline"])) for i, s in enumerate(all_steps)]
        + [(_xp(len(all_steps) - 1), plot_y1)]
    )
    exog_pts_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in exog_poly)

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Hawkes Contagion Cascade and Branching Monitor">',
        '<defs>',
        '<pattern id="hawkes-hatch" width="7" height="7" patternTransform="rotate(45 0 0)" patternUnits="userSpaceOnUse">',
        '<line x1="0" y1="0" x2="0" y2="7" stroke="#64748B" stroke-width="1.2" stroke-opacity="0.35" />',
        '</pattern>',
        '</defs>',
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="12" fill="#FFFFFF" />',
        f'<polygon points="{exog_pts_str}" fill="#475569" fill-opacity="0.38" />',
    ]

    # Stacked category endogenous ribbons
    cum_lower = [float(s["exog_baseline"]) for s in all_steps]
    for j in range(C):
        cum_upper = [cum_lower[i] + float(all_steps[i]["endog_by_source"][j]) for i in range(len(all_steps))]
        poly = [(_xp(i), _yp(cum_upper[i])) for i in range(len(all_steps))] + [
            (_xp(i), _yp(cum_lower[i])) for i in range(len(all_steps) - 1, -1, -1)
        ]
        p_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in poly)
        parts.append(f'<polygon points="{p_str}" fill="{cats[j]["color"]}" fill-opacity="0.58" />')
        cum_lower = cum_upper

    # Hatched projection cone overlay for (t, t+5]
    if cone and hist:
        cone_start_i = len(hist) - 1
        cone_x0 = _xp(cone_start_i)
        parts.append(
            f'<rect x="{cone_x0:.1f}" y="{plot_y0:.1f}" width="{left_x1 - cone_x0:.1f}" '
            f'height="{plot_y1 - plot_y0:.1f}" fill="url(#hawkes-hatch)" />'
        )
        parts.append(
            f'<line x1="{cone_x0:.1f}" y1="{plot_y0:.1f}" x2="{cone_x0:.1f}" y2="{plot_y1:.1f}" '
            f'stroke="#0F172A" stroke-width="1.2" stroke-dasharray="4,3" />'
        )

    # Observed referral count stems (strictly <= t)
    for i, r in enumerate(hist):
        xi = _xp(i)
        yi = _yp(float(r["observed_count"]))
        parts.append(f'<line x1="{xi:.1f}" y1="{plot_y1:.1f}" x2="{xi:.1f}" y2="{yi:.1f}" stroke="#334155" stroke-width="1" stroke-opacity="0.4" />')
        parts.append(f'<circle cx="{xi:.1f}" cy="{yi:.1f}" r="2.3" fill="#0F172A" />')
        if r["is_shock_callout"]:
            parts.append(f'<line x1="{xi:.1f}" y1="{plot_y0 + 8:.1f}" x2="{xi:.1f}" y2="{yi - 4:.1f}" stroke="#D97706" stroke-width="1.2" stroke-dasharray="2,2" />')
            parts.append(f'<circle cx="{xi:.1f}" cy="{plot_y0 + 8:.1f}" r="3" fill="#D97706" />')

    # Right panel: Network + Criticality Bullet Gauge
    net_cx, net_cy, net_r = 700.0, 145.0, 74.0
    nodes = spec["network"]["nodes"]
    for edge in spec["network"]["edges"]:
        if edge["is_self_loop"]:
            continue
        u = nodes[edge["source_idx"]]
        v = nodes[edge["target_idx"]]
        x1, y1 = net_cx + u["x_unit"] * net_r, net_cy + u["y_unit"] * net_r
        x2, y2 = net_cx + v["x_unit"] * net_r, net_cy + v["y_unit"] * net_r
        mx, my = 0.5 * (x1 + x2) - 0.16 * (y2 - y1), 0.5 * (y1 + y2) + 0.16 * (x2 - x1)
        parts.append(
            f'<path d="M {x1:.1f} {y1:.1f} Q {mx:.1f} {my:.1f} {x2:.1f} {y2:.1f}" '
            f'fill="none" stroke="{edge["color"]}" stroke-width="{edge["stroke_width"]}" stroke-opacity="0.65" />'
        )
    for nd in nodes:
        nx, ny = net_cx + nd["x_unit"] * net_r, net_cy + nd["y_unit"] * net_r
        parts.append(f'<circle cx="{nx:.1f}" cy="{ny:.1f}" r="11" fill="{nd["color"]}" stroke="#FFFFFF" stroke-width="2" />')

    # Bullet gauge [0, 1.2]
    gx0, gw, gy = 575.0, 250.0, 315.0
    rho_v = min(1.2, max(0.0, float(spec["criticality_gauge"]["value"])))
    marker_x = gx0 + gw * (rho_v / 1.2)
    parts.extend(
        [
            f'<rect x="{gx0:.1f}" y="{gy:.1f}" width="{gw * (0.60 / 1.2):.1f}" height="12" rx="3" fill="#0D9488" fill-opacity="0.28" />',
            f'<rect x="{gx0 + gw * (0.60 / 1.2):.1f}" y="{gy:.1f}" width="{gw * (0.25 / 1.2):.1f}" height="12" fill="#D97706" fill-opacity="0.28" />',
            f'<rect x="{gx0 + gw * (0.85 / 1.2):.1f}" y="{gy:.1f}" width="{gw * (0.35 / 1.2):.1f}" height="12" rx="3" fill="#DC2626" fill-opacity="0.28" />',
            f'<line x1="{marker_x:.1f}" y1="{gy - 4:.1f}" x2="{marker_x:.1f}" y2="{gy + 16:.1f}" stroke="#0F172A" stroke-width="3" />',
            f'<text x="{gx0:.1f}" y="{gy - 8:.1f}" font-family="Inter,sans-serif" font-size="11" font-weight="600" fill="#0F172A">Spectral Radius ρ(Γ) = {rho_v:.2f} — {spec["current"]["regime"]["label"]}</text>',
            '</svg>',
        ]
    )
    return "\n".join(parts)
