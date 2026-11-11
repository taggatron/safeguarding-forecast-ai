"""
Stack-Native Telemetry Mount & Standalone Interactive HTML Exporter (`esoteric_math/viz/host_mount.py`).

Supports:
  1. FastAPI Web Backend (`GET /api/esoteric-telemetry`): Caches training-fold fitted transformers
     and serves strictly causal `telemetry_payload.py` JSON for any requested date `t` and provision,
     plus calendar scrubber landmarks (Autumn Induction Spike, Half-Term Boundary, January Return, etc.).
  2. CLI / Static Pipeline (`export_standalone_telemetry_html`): Generates a standalone single-file
     interactive HTML5 + SVG + JS dashboard at `artifacts/esoteric_telemetry_dashboard.html` with
     embedded causal time-scrubbing plus SVG snapshots.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from esoteric_math.schema_bridge import (
     _infer_uk_fe_holiday_for_date,
    build_instructional_day_telemetry_frame,
)
from esoteric_math.viz.hilbert_clock import render_hilbert_clock_svg
from esoteric_math.viz.hawkes_cascade import render_hawkes_cascade_svg
from esoteric_math.viz.sheaf_frustration import render_sheaf_frustration_svg
from esoteric_math.viz.telemetry_payload import (
    FittedTelemetryModels,
    extract_telemetry_payload,
)


class TelemetryHostAdapter:
    """FastAPI / In-Memory cached adapter for the Mathematical Early-Warning Telemetry view."""

    def __init__(self, num_days: int = 380, train_cutoff: int = 240, seed: int = 42) -> None:
        self.num_days = num_days
        self.train_cutoff = train_cutoff
        self.seed = seed
        self._cache: Dict[str, Tuple[pd.DataFrame, FittedTelemetryModels, Dict[str, Any]]] = {}

    def _get_or_build_provision(self, provision: str) -> Tuple[pd.DataFrame, FittedTelemetryModels, Dict[str, Any]]:
        prov_key = provision if provision in ("All", "EPYP", "Adult", "Apprenticeships") else "All"
        if prov_key not in self._cache:
            df = build_instructional_day_telemetry_frame(
                num_days=self.num_days,
                start_date="2024-09-02",
                seed=self.seed,
                provision_filter=prov_key,
            )
            train_slice = df.iloc[: self.train_cutoff].copy().reset_index(drop=True)
            fitted = FittedTelemetryModels(train_slice)
            scrubber = self._build_calendar_scrubber_catalog(df["date"].tolist())
            self._cache[prov_key] = (df, fitted, scrubber)
        return self._cache[prov_key]

    @staticmethod
    def _build_calendar_scrubber_catalog(all_dates: List[str]) -> Dict[str, Any]:
        """
        Build timeline scrubber metadata purely from deterministic UK FE calendar dates
        (never leaking future referral counts).
        """
        landmarks: List[Dict[str, Any]] = []
        target_specs = [
            ("2024-09-24", "Autumn Induction Spike", "Early-Autumn enrolment & school file handover surge"),
            ("2024-10-18", "Pre-Half-Term Masking", "Cohomological contextual masking ahead of October break"),
            ("2024-10-29", "Autumn Half-Term Freeze", "Non-instructional break (Hilbert phase clock frozen)"),
            ("2024-11-04", "Post-Half-Term Return", "Accumulated holiday disclosure step-jump on Monday return"),
            ("2025-01-15", "January Return Wave", "Post-winter termly incubation & peer cascade monitoring"),
            ("2025-05-14", "May Exam Crunch", "GCSE resit & vocational assessment pressure window"),
        ]
        for target_dt, title, desc in target_specs:
            eligible = [i for i, d in enumerate(all_dates) if d <= target_dt]
            if eligible:
                idx = eligible[-1]
                landmarks.append(
                    {
                        "step_index": int(idx),
                        "date": all_dates[idx],
                        "label": title,
                        "description": desc,
                    }
                )
        if all_dates:
            landmarks.append(
                {
                    "step_index": len(all_dates) - 1,
                    "date": all_dates[-1],
                    "label": "Latest Validation Step",
                    "description": "Most recent causal validation observation",
                }
            )

        holiday_flags = []
        for d in all_dates:
            frac, lbl = _infer_uk_fe_holiday_for_date(pd.Timestamp(d), "daily")
            holiday_flags.append({"date": d, "is_instructional": bool(frac > 0.5), "holiday_label": lbl})

        return {
            "available_dates": all_dates,
            "min_step_index": 30,
            "max_step_index": len(all_dates) - 1,
            "default_date": all_dates[-1] if all_dates else "",
            "landmarks": landmarks,
            "calendar_strip": holiday_flags,
        }

    def get_telemetry_response(
        self,
        t_date: Optional[str] = None,
        t_index: Optional[int] = None,
        provision: str = "All",
    ) -> Dict[str, Any]:
        df, fitted, scrubber = self._get_or_build_provision(provision)
        if t_date is None and t_index is None:
            t_index = len(df) - 1
        elif t_index is not None:
            t_index = max(10, min(int(t_index), len(df) - 1))

        payload = extract_telemetry_payload(
            df,
            t_index=t_index,
            t_date=t_date,
            train_cutoff=self.train_cutoff,
            fitted_models=fitted,
        )
        return {
            "governance_banner": "SYNTHETIC DEMONSTRATION DATA — NOT ACTUAL COLLEGE SAFEGUARDING RECORDS",
            "provision": provision,
            "scrubber": scrubber,
            "telemetry": payload,
        }


telemetry_host_adapter = TelemetryHostAdapter()


def export_standalone_telemetry_html(
    output_html_path: str | Path = "artifacts/esoteric_telemetry_dashboard.html",
    provision: str = "All",
) -> Dict[str, str]:
    """
    Generate a standalone single-file interactive HTML5 + SVG + JS telemetry dashboard
    (`artifacts/esoteric_telemetry_dashboard.html`) with embedded causal time-scrubbing
    across sampled timeline steps + landmark dates, plus standalone SVG snapshots.
    """
    out_path = Path(output_html_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df, fitted, scrubber = telemetry_host_adapter._get_or_build_provision(provision)
    all_dates = scrubber["available_dates"]

    # Precompute dense interactive scrubber frames (every 3 days + all landmarks + latest 25 days daily)
    landmark_indices = {int(lm["step_index"]) for lm in scrubber["landmarks"]}
    sampled_indices = sorted(
        set(range(30, len(all_dates), 3))
        | landmark_indices
        | set(range(max(30, len(all_dates) - 25), len(all_dates)))
    )

    frames: List[Dict[str, Any]] = []
    for idx in sampled_indices:
        p = extract_telemetry_payload(df, t_index=idx, train_cutoff=telemetry_host_adapter.train_cutoff, fitted_models=fitted)
        frames.append(
            {
                "step_index": int(idx),
                "date": p["selected_date"],
                "kpi_ribbon": p["kpi_ribbon"],
                "hilbert_svg": render_hilbert_clock_svg(p["hilbert_clock"]),
                "hawkes_svg": render_hawkes_cascade_svg(p["hawkes_cascade"]),
                "sheaf_svg": render_sheaf_frustration_svg(p["sheaf_frustration"]),
                "sheaf_cards": p["sheaf_frustration"]["triage_cards"],
                "topology_mode": p["topology_mode"],
                "auxiliary": p["auxiliary_manifold_metrics"],
            }
        )

    # Also write latest SVG snapshots next to HTML
    latest = extract_telemetry_payload(df, t_index=len(all_dates) - 1, train_cutoff=telemetry_host_adapter.train_cutoff, fitted_models=fitted)
    svg_dir = out_path.parent
    hilbert_svg_path = svg_dir / "hilbert_phase_clock.svg"
    hawkes_svg_path = svg_dir / "hawkes_contagion_cascade.svg"
    sheaf_svg_path = svg_dir / "sheaf_frustration_board.svg"

    hilbert_svg_path.write_text(render_hilbert_clock_svg(latest["hilbert_clock"]), encoding="utf-8")
    hawkes_svg_path.write_text(render_hawkes_cascade_svg(latest["hawkes_cascade"]), encoding="utf-8")
    sheaf_svg_path.write_text(render_sheaf_frustration_svg(latest["sheaf_frustration"]), encoding="utf-8")

    embedded_json = json.dumps({"landmarks": scrubber["landmarks"], "frames": frames})

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Safeguarding Forecast AI — Mathematical Early-Warning Telemetry</title>
  <style>
    :root {{
      --slate: #475569;
      --teal: #0D9488;
      --indigo: #4F46E5;
      --amber: #D97706;
      --crimson: #DC2626;
      --bg: #F8FAFC;
      --navy: #0F172A;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; background: var(--bg); color: #0F172A; line-height: 1.5; }}
    .banner {{ background: #FEF3C7; border-bottom: 1px solid #FDE68A; color: #78350F; font-size: 12px; font-weight: 600; padding: 6px 24px; text-align: center; }}
    header {{ background: var(--navy); color: #FFF; padding: 18px 28px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; }}
    header h1 {{ font-size: 19px; font-weight: 700; letter-spacing: -0.01em; }}
    header p {{ font-size: 12.5px; color: #94A3B8; }}
    .container {{ max-width: 1360px; margin: 0 auto; padding: 20px 24px 48px; display: flex; flex-direction: column; gap: 18px; }}
    .scrubber-card {{ background: #FFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 14px 18px; box-shadow: 0 1px 2px rgba(15,23,42,0.04); }}
    .scrubber-top {{ display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 10px; }}
    .landmarks {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }}
    .lm-btn {{ background: #F1F5F9; border: 1px solid #CBD5E1; border-radius: 999px; padding: 4px 11px; font-size: 11.5px; font-weight: 600; color: #334155; cursor: pointer; transition: all 0.15s; }}
    .lm-btn:hover, .lm-btn.active {{ background: #0D9488; border-color: #0D9488; color: #FFF; }}
    input[type=range] {{ width: 100%; accent-color: #0D9488; cursor: pointer; }}
    .kpi-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }}
    .kpi-card {{ background: #FFF; border: 1px solid #E2E8F0; border-left: 5px solid var(--slate); border-radius: 12px; padding: 14px 16px; }}
    .kpi-title {{ font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; color: #64748B; }}
    .kpi-val {{ font-size: 19px; font-weight: 700; margin: 4px 0 2px; color: #0F172A; display: flex; justify-content: space-between; align-items: baseline; gap: 8px; }}
    .kpi-sub {{ font-size: 12px; color: #475569; }}
    .panel {{ background: #FFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 18px; }}
    .panel-hdr {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 10px; flex-wrap: wrap; gap: 8px; }}
    .panel-hdr h2 {{ font-size: 15px; font-weight: 700; color: #0F172A; }}
    .badge {{ font-size: 11px; font-weight: 600; padding: 3px 9px; border-radius: 999px; background: #F1F5F9; color: #334155; }}
    @media (max-width: 920px) {{ .kpi-grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="banner">SYNTHETIC DEMONSTRATION DATA — STRICTLY CAUSAL TELEMETRY (s &gt; t MASKED) — NO INDIVIDUAL LEARNER RECORDS</div>
  <header>
    <div>
      <h1>Safeguarding Forecast AI · Mathematical Early-Warning Telemetry</h1>
      <p>Hilbert Stress-Cycle Phase Clocks · Hawkes Contagion Cascades · Cellular Sheaf Frustration Triage</p>
    </div>
    <div id="causal-badge" style="background:#1E293B;border:1px solid #334155;padding:6px 12px;border-radius:8px;font-size:12px;color:#5EEAD4;font-weight:600;">
      Active Horizon: Loading...
    </div>
  </header>
  <main class="container">
    <section class="scrubber-card">
      <div class="scrubber-top">
        <div>
          <strong style="font-size:13.5px;">Shared Causal Time Scrubber</strong>
          <span id="date-readout" style="margin-left:10px;font-size:13px;font-weight:700;color:#0D9488;"></span>
        </div>
        <span style="font-size:11.5px;color:#64748B;">Future observations s &gt; t are strictly masked at every scrubber position</span>
      </div>
      <input id="time-slider" type="range" min="0" max="1" value="1" />
      <div class="landmarks" id="landmarks-bar"></div>
    </section>

    <section class="kpi-grid" id="kpi-ribbon"></section>

    <section class="panel">
      <div class="panel-hdr">
        <h2>1. Hilbert Stress-Cycle Phase Clock (Warped 190-Day Instructional Manifold)</h2>
        <span class="badge">Module f01 · Analytic Signal z(t) = r(t)e^{{iφ(t)}}</span>
      </div>
      <div id="hilbert-mount"></div>
    </section>

    <section class="panel">
      <div class="panel-hdr">
        <h2>2. Hawkes Self- &amp; Cross-Exciting Contagion Cascade &amp; Criticality Monitor</h2>
        <span class="badge">Module f03 · Subcritical Branching Matrix Γ = α / β</span>
      </div>
      <div id="hawkes-mount"></div>
    </section>

    <section class="panel">
      <div class="panel-hdr">
        <h2>3. Cellular Sheaf Cohomology &amp; Contextual Masking Triage Board</h2>
        <span class="badge" id="topology-badge">Module f08 · Sheaf Laplacian L_F = δ^T δ</span>
      </div>
      <div id="sheaf-mount"></div>
    </section>
  </main>

  <script>
    const DATA = {embedded_json};
    const frames = DATA.frames;
    const slider = document.getElementById('time-slider');
    slider.max = String(frames.length - 1);
    slider.value = String(frames.length - 1);

    function renderFrame(idx) {{
      const f = frames[idx];
      if (!f) return;
      document.getElementById('date-readout').textContent = `${{f.date}} (Step ${{f.step_index}})`;
      document.getElementById('causal-badge').textContent = `Causal Cutoff t = ${{f.date}} · RMT Sync = ${{(f.auxiliary.rmt_sync_ratio * 100).toFixed(1)}}% · d* = ${{f.auxiliary.fractional_d_star}}`;
      document.getElementById('topology-badge').textContent = `${{f.topology_mode}} · Module f08`;

      const kpiHtml = f.kpi_ribbon.map(k => `
        <div class="kpi-card" style="border-left-color:${{k.status_color}}">
          <div class="kpi-title">${{k.title}}</div>
          <div class="kpi-val">
            <span style="color:${{k.status_color}}">${{k.primary_value}}</span>
            <span style="font-size:12.5px;color:#334155;font-weight:600;">${{k.secondary_metric}}</span>
          </div>
          <div class="kpi-sub">${{k.subtitle}}</div>
        </div>
      `).join('');
      document.getElementById('kpi-ribbon').innerHTML = kpiHtml;
      document.getElementById('hilbert-mount').innerHTML = f.hilbert_svg;
      document.getElementById('hawkes-mount').innerHTML = f.hawkes_svg;
      document.getElementById('sheaf-mount').innerHTML = f.sheaf_svg;

      document.querySelectorAll('.lm-btn').forEach(btn => {{
        btn.classList.toggle('active', Number(btn.dataset.frameIdx) === idx);
      }});
    }}

    const lmBar = document.getElementById('landmarks-bar');
    DATA.landmarks.forEach(lm => {{
      let bestIdx = 0;
      let bestDist = 1e9;
      frames.forEach((fr, i) => {{
        const d = Math.abs(fr.step_index - lm.step_index);
        if (d < bestDist) {{ bestDist = d; bestIdx = i; }}
      }});
      const btn = document.createElement('button');
      btn.className = 'lm-btn';
      btn.dataset.frameIdx = String(bestIdx);
      btn.textContent = `${{lm.label}} (${{frames[bestIdx].date}})`;
      btn.title = lm.description;
      btn.onclick = () => {{ slider.value = String(bestIdx); renderFrame(bestIdx); }};
      lmBar.appendChild(btn);
    }});

    slider.addEventListener('input', (e) => renderFrame(Number(e.target.value)));
    renderFrame(frames.length - 1);
  </script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    return {
        "dashboard_html": str(out_path),
        "hilbert_svg": str(hilbert_svg_path),
        "hawkes_svg": str(hawkes_svg_path),
        "sheaf_svg": str(sheaf_svg_path),
    }
