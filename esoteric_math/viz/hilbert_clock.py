"""
Visual Spec 1: Hilbert Stress-Cycle Phase Clock (`esoteric_math/viz/hilbert_clock.py`).

Builds JSON-serializable polar Argand geometry and standalone SVG rendering for the
analytic signal z(tau) = r(tau) * exp(i * phi(tau)):
  - Four annotated pastoral quadrants (Q1 Escalating Pressure, Q2 Peak Load & Crest,
    Q3 Post-Peak / Holiday Decompression, Q4 Pre-Surge Incubation)
  - Concentric reference rings at r = 1.0 sigma (Normal Termly Rhythm) and r = 2.0 sigma
    (Acute Amplitude Breach, crimson dashed)
  - 20-instructional-day comet trajectory (opacity 0.15 -> 1.0) with tangential velocity arrow
  - Hollow diamond holiday phase-freeze glyphs along trajectory
  - Companion 60-step instructional sparkline with unified hover fields including
    estimated instructional days to next peak Delta_hat t_peak.
"""
from __future__ import annotations

from typing import Any, Dict, List
import numpy as np


PALETTE = {
    "slate": "#475569",
    "teal": "#0D9488",
    "indigo": "#4F46E5",
    "amber": "#D97706",
    "crimson": "#DC2626",
}


def classify_pastoral_quadrant(phi_rad: float) -> Dict[str, str]:
    """Map phase angle phi in (-pi, pi] to its pastoral quadrant metadata."""
    deg = float(np.degrees(phi_rad))
    if 0.0 <= deg < 90.0:
        return {
            "id": "Q1",
            "title": "Escalating Pressure",
            "short": "Escalating (QI)",
            "color": PALETTE["amber"],
            "tint": "rgba(217, 119, 6, 0.10)",
            "summary": "Referrals above seasonal baseline and accelerating upward toward a termly crest.",
        }
    if 90.0 <= deg <= 180.0:
        return {
            "id": "Q2",
            "title": "Peak Load & Crest",
            "short": "Peak Crest (QII)",
            "color": PALETTE["crimson"],
            "tint": "rgba(220, 38, 38, 0.09)",
            "summary": "Institutional stress cresting; pastoral triage load at local maximum and inflecting.",
        }
    if -180.0 <= deg < -90.0:
        return {
            "id": "Q3",
            "title": "Post-Peak / Holiday Decompression",
            "short": "Decompression (QIII)",
            "color": PALETTE["teal"],
            "tint": "rgba(13, 148, 136, 0.10)",
            "summary": "Volume below seasonal baseline and relaxing following a termly crest or break.",
        }
    return {
        "id": "Q4",
        "title": "Pre-Surge Incubation",
        "short": "Incubation (QIV)",
        "color": PALETTE["indigo"],
        "tint": "rgba(79, 70, 229, 0.10)",
        "summary": "Surface counts remain quiet, while positive Hilbert phase velocity warns the next wave is winding up.",
    }


def build_hilbert_clock_spec(f01_diag: Dict[str, Any], t_idx: int) -> Dict[str, Any]:
    """
    Extract strictly causal Hilbert Phase Clock visual payload at index `t_idx`
    (reading ONLY indices 0 .. t_idx).
    """
    end = int(t_idx) + 1
    dates = [str(d) for d in f01_diag["dates"][:end]]
    r_arr = np.maximum(0.0, np.asarray(f01_diag["r_norm"][:end], dtype=float))
    phi_arr = np.asarray(f01_diag["phi"][:end], dtype=float)
    # Wrap strictly into [-pi, pi]
    phi_arr = (phi_arr + np.pi) % (2.0 * np.pi) - np.pi
    omega_arr = np.asarray(f01_diag["omega"][:end], dtype=float)
    omega_bar_arr = np.asarray(f01_diag["omega_bar"][:end], dtype=float)
    dtp_arr = np.maximum(0.0, np.asarray(f01_diag["days_to_peak"][:end], dtype=float))
    freeze_arr = np.asarray(f01_diag["is_holiday_freeze"][:end], dtype=int)
    inst_cum_arr = np.asarray(f01_diag["instructional_days_cum"][:end], dtype=float)
    obs_arr = np.maximum(0.0, np.asarray(f01_diag["observed_counts"][:end], dtype=float))
    base_arr = np.maximum(0.0, np.asarray(f01_diag["baseline_counts"][:end], dtype=float))
    hol_labels = [str(h) for h in f01_diag["holiday_labels"][:end]]

    # Select comet window spanning up to 20 active instructional steps up to t_idx (including intervening freezes)
    active_seen = 0
    comet_start = max(0, end - 1)
    for idx in range(end - 1, -1, -1):
        comet_start = idx
        if freeze_arr[idx] == 0:
            active_seen += 1
        if active_seen >= 20:
            break
    # Cap total comet points at 34 so weekend/half-term diamonds stay crisp
    comet_start = max(comet_start, max(0, end - 34))

    dial_max_r = float(max(2.5, min(4.0, np.ceil(max(float(np.max(r_arr[comet_start:end])), 2.1) * 2.0) / 2.0)))

    comet_len = max(1, end - comet_start)
    comet_points: List[Dict[str, Any]] = []
    for k, idx in enumerate(range(comet_start, end)):
        r_v = float(min(r_arr[idx], dial_max_r * 0.98))
        phi_v = float(phi_arr[idx])
        frac = float(k) / float(max(comet_len - 1, 1))
        opacity = round(0.15 + 0.85 * frac, 4)
        stroke_w = round(1.3 + 2.9 * frac, 2)
        q_meta = classify_pastoral_quadrant(phi_v)
        comet_points.append(
            {
                "step_index": int(idx),
                "date": dates[idx],
                "t_acad": round(float(inst_cum_arr[idx]), 1),
                "r_norm": round(float(r_arr[idx]), 4),
                "r_plot": round(r_v, 4),
                "phi_rad": round(phi_v, 5),
                "phi_deg": round(float(np.degrees(phi_v)), 2),
                "x_unit": round(float(r_v * np.cos(phi_v)), 4),
                "y_unit": round(float(r_v * np.sin(phi_v)), 4),
                "omega_rad": round(float(omega_arr[idx]), 4),
                "days_to_peak": round(float(dtp_arr[idx]), 1),
                "is_holiday_freeze": bool(freeze_arr[idx] > 0),
                "holiday_label": hol_labels[idx],
                "opacity": opacity,
                "stroke_width": stroke_w,
                "quadrant_id": q_meta["id"],
                "quadrant_title": q_meta["title"],
                "quadrant_color": q_meta["color"],
            }
        )

    # Companion 60-step sparkline up to t_idx
    spark_start = max(0, end - 60)
    sparkline: List[Dict[str, Any]] = []
    for idx in range(spark_start, end):
        phi_v = float(phi_arr[idx])
        q_meta = classify_pastoral_quadrant(phi_v)
        sparkline.append(
            {
                "step_index": int(idx),
                "date": dates[idx],
                "t_acad": round(float(inst_cum_arr[idx]), 1),
                "observed_count": round(float(obs_arr[idx]), 2),
                "warped_baseline": round(float(base_arr[idx]), 2),
                "r_norm": round(float(r_arr[idx]), 3),
                "phi_deg": round(float(np.degrees(phi_v)), 1),
                "days_to_peak": round(float(dtp_arr[idx]), 1),
                "quadrant_id": q_meta["id"],
                "quadrant_title": q_meta["title"],
                "quadrant_color": q_meta["color"],
                "is_holiday_freeze": bool(freeze_arr[idx] > 0),
                "holiday_label": hol_labels[idx],
            }
        )

    cur_r = float(r_arr[-1])
    cur_phi = float(phi_arr[-1])
    cur_omega = float(omega_bar_arr[-1] if freeze_arr[-1] == 0 else 0.0)
    cur_q = classify_pastoral_quadrant(cur_phi)
    # Unit tangent in standard math Argand plane: d/dphi (cos phi, sin phi) = (-sin phi, cos phi)
    tan_speed = float(np.clip(cur_omega * 2.2, 0.12 if cur_omega > 0 else 0.0, 0.65))
    tan_dx = float(-np.sin(cur_phi) * tan_speed)
    tan_dy = float(np.cos(cur_phi) * tan_speed)

    return {
        "current": {
            "date": dates[-1],
            "t_acad": round(float(inst_cum_arr[-1]), 1),
            "r_norm": round(cur_r, 3),
            "phi_rad": round(cur_phi, 5),
            "phi_deg": round(float(np.degrees(cur_phi)), 1),
            "omega_inst_rad": round(cur_omega, 4),
            "days_to_peak": round(float(dtp_arr[-1]), 1),
            "is_breach_2sigma": bool(cur_r >= 2.0),
            "is_holiday_freeze": bool(freeze_arr[-1] > 0),
            "holiday_label": hol_labels[-1],
            "quadrant": cur_q,
            "tangent_vector": {
                "dx_unit": round(tan_dx, 4),
                "dy_unit": round(tan_dy, 4),
            },
        },
        "dial_max_r": dial_max_r,
        "rings": [
            {"r": 1.0, "label": "Normal Termly Rhythm (1.0σ)", "style": "solid", "color": PALETTE["slate"]},
            {"r": 2.0, "label": "Acute Amplitude Breach (2.0σ)", "style": "dashed", "color": PALETTE["crimson"]},
        ],
        "quadrants": [
            classify_pastoral_quadrant(np.radians(45.0)),
            classify_pastoral_quadrant(np.radians(135.0)),
            classify_pastoral_quadrant(np.radians(-135.0)),
            classify_pastoral_quadrant(np.radians(-45.0)),
        ],
        "comet_trajectory": comet_points,
        "sparkline_60": sparkline,
    }


def render_hilbert_clock_svg(spec: Dict[str, Any], width: int = 560, height: int = 390) -> str:
    """Render standalone crisp SVG for the Hilbert Stress-Cycle Phase Clock."""
    import html as _html

    cx, cy = 205.0, 195.0
    radius_px = 144.0
    max_r = float(spec["dial_max_r"])
    scale = radius_px / max(max_r, 1e-6)

    r1 = 1.0 * scale
    r2 = 2.0 * scale

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" style="max-width:680px;height:auto;display:block;margin:0 auto;" role="img" aria-label="Hilbert Stress-Cycle Phase Clock">',
        '<defs>',
        '<marker id="hilbert-vel-arrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">',
        f'<path d="M0,1 L7,4 L0,7 Z" fill="{spec["current"]["quadrant"]["color"]}" />',
        '</marker>',
        '</defs>',
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="12" fill="#FFFFFF" />',
        # Four quadrant wedges
        f'<path d="M {cx} {cy} L {cx + radius_px} {cy} A {radius_px} {radius_px} 0 0 0 {cx} {cy - radius_px} Z" fill="rgba(217,119,6,0.10)" />',
        f'<path d="M {cx} {cy} L {cx} {cy - radius_px} A {radius_px} {radius_px} 0 0 0 {cx - radius_px} {cy} Z" fill="rgba(220,38,38,0.09)" />',
        f'<path d="M {cx} {cy} L {cx - radius_px} {cy} A {radius_px} {radius_px} 0 0 0 {cx} {cy + radius_px} Z" fill="rgba(13,148,136,0.10)" />',
        f'<path d="M {cx} {cy} L {cx} {cy + radius_px} A {radius_px} {radius_px} 0 0 0 {cx + radius_px} {cy} Z" fill="rgba(79,70,229,0.10)" />',
        # Axes & reference rings
        f'<line x1="{cx - radius_px}" y1="{cy}" x2="{cx + radius_px}" y2="{cy}" stroke="#CBD5E1" stroke-width="1" />',
        f'<line x1="{cx}" y1="{cy - radius_px}" x2="{cx}" y2="{cy + radius_px}" stroke="#CBD5E1" stroke-width="1" />',
        f'<circle cx="{cx}" cy="{cy}" r="{radius_px:.1f}" fill="none" stroke="#E2E8F0" stroke-width="1" />',
        f'<circle cx="{cx}" cy="{cy}" r="{r1:.1f}" fill="none" stroke="#475569" stroke-width="1.4" />',
        f'<circle cx="{cx}" cy="{cy}" r="{r2:.1f}" fill="none" stroke="#DC2626" stroke-width="1.6" stroke-dasharray="5,4" />',
        # Quadrant labels
        f'<text x="{cx + radius_px - 6}" y="28" text-anchor="end" font-family="Inter,sans-serif" font-size="10.5" font-weight="700" fill="#B45309">QI · Escalating</text>',
        f'<text x="{cx - radius_px + 6}" y="28" text-anchor="start" font-family="Inter,sans-serif" font-size="10.5" font-weight="700" fill="#B91C1C">QII · Peak Crest</text>',
        f'<text x="{cx - radius_px + 6}" y="{height - 18}" text-anchor="start" font-family="Inter,sans-serif" font-size="10.5" font-weight="700" fill="#0F766E">QIII · Decompression</text>',
        f'<text x="{cx + radius_px - 6}" y="{height - 18}" text-anchor="end" font-family="Inter,sans-serif" font-size="10.5" font-weight="700" fill="#4338CA">QIV · Incubation</text>',
    ]

    pts = spec["comet_trajectory"]
    for i in range(1, len(pts)):
        p0, p1 = pts[i - 1], pts[i]
        x0, y0 = cx + p0["x_unit"] * scale, cy - p0["y_unit"] * scale
        x1, y1 = cx + p1["x_unit"] * scale, cy - p1["y_unit"] * scale
        parts.append(
            f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" '
            f'stroke="{p1["quadrant_color"]}" stroke-width="{p1["stroke_width"]}" '
            f'stroke-opacity="{p1["opacity"]}" stroke-linecap="round" />'
        )

    for p in pts:
        px, py = cx + p["x_unit"] * scale, cy - p["y_unit"] * scale
        if p["is_holiday_freeze"]:
            d_sz = 4.5
            parts.append(
                f'<polygon points="{px:.2f},{py - d_sz:.2f} {px + d_sz:.2f},{py:.2f} '
                f'{px:.2f},{py + d_sz:.2f} {px - d_sz:.2f},{py:.2f}" '
                f'fill="#FFFFFF" stroke="#475569" stroke-width="1.5" />'
            )

    if pts:
        head = pts[-1]
        hx, hy = cx + head["x_unit"] * scale, cy - head["y_unit"] * scale
        tv = spec["current"]["tangent_vector"]
        tx = hx + tv["dx_unit"] * scale
        ty = hy - tv["dy_unit"] * scale
        if abs(tx - hx) + abs(ty - hy) > 3.0:
            parts.append(
                f'<line x1="{hx:.2f}" y1="{hy:.2f}" x2="{tx:.2f}" y2="{ty:.2f}" '
                f'stroke="{spec["current"]["quadrant"]["color"]}" stroke-width="2" marker-end="url(#hilbert-vel-arrow)" />'
            )
        beacon_col = "#DC2626" if spec["current"]["is_breach_2sigma"] else spec["current"]["quadrant"]["color"]
        parts.append(f'<circle cx="{hx:.2f}" cy="{hy:.2f}" r="9" fill="{beacon_col}" fill-opacity="0.22" />')
        parts.append(f'<circle cx="{hx:.2f}" cy="{hy:.2f}" r="5.2" fill="{beacon_col}" stroke="#FFFFFF" stroke-width="1.8" />')

    # Ring Labels (rendered above comet with crisp white halo)
    parts.append(
        f'<text x="{cx + 5}" y="{cy - r1 + 12:.1f}" font-family="Inter,sans-serif" font-size="9.5" font-weight="600" fill="#475569" paint-order="stroke" stroke="#FFFFFF" stroke-width="3.2" stroke-linejoin="round">1.0σ Normal</text>'
    )
    parts.append(
        f'<text x="{cx + 5}" y="{cy - r2 - 4:.1f}" font-family="Inter,sans-serif" font-size="9.5" font-weight="700" fill="#DC2626" paint-order="stroke" stroke="#FFFFFF" stroke-width="3.2" stroke-linejoin="round">2.0σ Breach</text>'
    )

    # Right summary callout column inside SVG
    cur = spec["current"]
    q_title_esc = _html.escape(str(cur["quadrant"]["title"]))
    parts.extend(
        [
            f'<text x="372" y="96" font-family="Plus Jakarta Sans,sans-serif" font-size="10.5" font-weight="700" fill="#64748B">ACTIVE PHASE STATE</text>',
            f'<text x="372" y="120" font-family="Inter,sans-serif" font-size="13" font-weight="700" fill="{cur["quadrant"]["color"]}">{q_title_esc}</text>',
            f'<text x="372" y="146" font-family="Inter,sans-serif" font-size="11.5" fill="#334155">Amplitude r(t): {cur["r_norm"]:.2f}σ</text>',
            f'<text x="372" y="168" font-family="Inter,sans-serif" font-size="11.5" fill="#334155">Phase Angle φ(t): {cur["phi_deg"]:+.1f}°</text>',
            f'<text x="372" y="190" font-family="Inter,sans-serif" font-size="11.5" fill="#334155">Acad Day t_acad: {cur["t_acad"]:.0f}</text>',
            f'<text x="372" y="216" font-family="Inter,sans-serif" font-size="12" font-weight="700" fill="#4F46E5">Est. Crest: {cur["days_to_peak"]:.1f} inst.d</text>',
            '</svg>',
        ]
    )
    return "\n".join(parts)
