"""
Module 01: Warped-Time Fourier & Causal Hilbert Phase (`esoteric_math/features/f01_fourier_hilbert.py`).

Implements:
- Dual harmonic clocks: Calendar Year (T = 365.25d) and UK FE Instructional-Day Manifold
  (T_acad = 190 teaching days, freezing phase across weekends and holiday dead zones).
- Trailing causal FFT amplitude thresholding + IFFT denoised baseline x_hat_base(t).
- Strictly causal discrete Hilbert Transform z(t) on band-pass residual x_tilde(t),
  extracting normalized stress amplitude r(t), pastoral Argand phase angle phi(t) in (-pi, pi],
  instantaneous angular velocity omega(t), and estimated instructional days to next crest.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import pandas as pd

from esoteric_math.features.base import CausalBaseTransformer
from esoteric_math.schema_bridge import SchemaBundle, _causal_ewma_1d


def _wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle(s) into (-pi, pi]."""
    wrapped = (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi
    wrapped = np.where(wrapped <= -np.pi, wrapped + 2.0 * np.pi, wrapped)
    if np.ndim(angle) == 0:
        return float(wrapped)
    return wrapped


def _causal_fft_denoise_step(window: np.ndarray, threshold_ratio: float = 0.25) -> float:
    """
    Compute denoised baseline value at the right endpoint of `window` (which contains
    strictly past/current observations <= t) via symmetric causal reflection + rFFT thresholding.
    """
    m = len(window)
    if m <= 2:
        return float(np.mean(window))
    # Even mirror reflection strictly of past samples [w_0..w_{m-1}, w_{m-2}..w_1] avoids boundary jump
    sym = np.concatenate([window, window[-2:0:-1]]) if m > 2 else np.concatenate([window, window[::-1]])
    spec = np.fft.rfft(sym)
    amps = np.abs(spec)
    if len(amps) > 2:
        max_ac = float(np.max(amps[1:]))
        cutoff = threshold_ratio * max_ac
        # Always preserve DC + fundamental lowest trend/seasonal frequency bin
        keep = np.zeros_like(amps, dtype=bool)
        keep[0] = True
        keep[1] = True
        if len(amps) > 2:
            keep[2] = amps[2] >= 0.5 * cutoff
        keep = keep | (amps >= cutoff)
        # Suppress high-frequency jitter in upper half of spectrum
        high_start = max(3, len(amps) // 3)
        keep[high_start:] = False
        spec_filtered = np.where(keep, spec, 0.0)
    else:
        spec_filtered = spec
    recon = np.fft.irfft(spec_filtered, n=len(sym))
    return float(recon[m - 1])


def _causal_hilbert_quadrature_step(resid_window: np.ndarray) -> Tuple[float, float]:
    """
    Strictly causal discrete Hilbert Transform at the endpoint of `resid_window` (<= t).
    Uses regularized AR(2) forward wave continuation derived EXCLUSIVELY from `resid_window`
    followed by discrete Marple analytic FFT at index `len(resid_window) - 1`.
    For any pure sinusoid u_k = A sin(w k + phi), AR(2) continuation is exact to machine precision.
    Returns (re_x_tilde, im_hilbert).
    """
    L = len(resid_window)
    re_val = float(resid_window[-1])
    if L < 4 or float(np.std(resid_window)) < 1e-10:
        return re_val, 0.0

    centered = resid_window - float(np.mean(resid_window))
    re_c = float(centered[-1])

    # Fit causal damped AR(2): u_k = a1 * u_{k-1} + a2 * u_{k-2} strictly inside `centered`
    y_ar = centered[2:]
    X_ar = np.column_stack([centered[1:-1], centered[:-2]])
    gram = X_ar.T @ X_ar + 1e-3 * float(np.dot(centered, centered) / L + 1e-6) * np.eye(2)
    rhs = X_ar.T @ y_ar
    a_vec = np.linalg.solve(gram, rhs)

    # Stabilize characteristic roots inside unit circle (radius <= 0.96) so virtual tail never blows up
    a1, a2 = float(a_vec[0]), float(a_vec[1])
    a2 = float(np.clip(a2, -0.92, 0.92))
    max_a1 = 1.90 * np.sqrt(max(0.04, abs(a2))) if a2 < 0 else (0.95 - a2)
    a1 = float(np.clip(a1, -max(0.95, max_a1), max(0.95, max_a1)))

    n_ext = min(12, max(6, L // 2))
    ext = np.empty(L + n_ext, dtype=float)
    ext[:L] = centered
    for k in range(L, L + n_ext):
        ext[k] = 0.94 * (a1 * ext[k - 1] + a2 * ext[k - 2])

    # Taper both ends smoothly to zero DC wrap-around and compute discrete analytic signal
    n_tot = len(ext)
    F = np.fft.fft(ext)
    h_mult = np.zeros(n_tot, dtype=float)
    h_mult[0] = 1.0
    if n_tot % 2 == 0:
        h_mult[1 : n_tot // 2] = 2.0
        h_mult[n_tot // 2] = 1.0
    else:
        h_mult[1 : (n_tot + 1) // 2] = 2.0
    z_analytic = np.fft.ifft(F * h_mult)
    im_hilb = float(np.imag(z_analytic[L - 1]))
    return re_c, im_hilb


class WarpedFourierHilbertTransformer(CausalBaseTransformer):
    """
    Module 01: Dual-Clock Warped Academic Fourier + Causal Hilbert Analytic Phase Engine.
    """

    module_id = "f01"
    module_name = "Warped Academic Fourier & Hilbert Phase"

    def __init__(
        self,
        n_harmonics: int = 3,
        fft_window: int = 28,
        hilbert_window: int = 20,
        fft_threshold_ratio: float = 0.25,
        causal_mode: Literal["telemetry", "forecast"] = "telemetry",
        train_cutoff: Optional[int] = None,
        return_dataframe: bool = True,
    ) -> None:
        super().__init__(
            causal_mode=causal_mode,
            train_cutoff=train_cutoff,
            return_dataframe=return_dataframe,
        )
        self.n_harmonics = n_harmonics
        self.fft_window = fft_window
        self.hilbert_window = hilbert_window
        self.fft_threshold_ratio = fft_threshold_ratio

    def _compute_raw_hilbert_states(self, bundle: SchemaBundle) -> Dict[str, np.ndarray]:
        ta = bundle.tier_a
        n = bundle.n_steps
        x = ta.y_rate_per_1000.astype(float)
        x_counts = ta.y_count.astype(float)
        enrol_scale = np.maximum(1.0, ta.exposure) / 1000.0

        w_fft = max(8, int(self.fft_window))
        w_hilb = max(8, int(self.hilbert_window))

        baseline_rate = np.zeros(n, dtype=float)
        resid_rate = np.zeros(n, dtype=float)
        for t in range(n):
            # Left-pad causally with x[0] when t + 1 < w_fft
            raw_slice = x[max(0, t - w_fft + 1) : t + 1]
            if len(raw_slice) < w_fft:
                pad = np.full(w_fft - len(raw_slice), float(raw_slice[0]), dtype=float)
                win = np.concatenate([pad, raw_slice])
            else:
                win = raw_slice
            b_t = _causal_fft_denoise_step(win, threshold_ratio=self.fft_threshold_ratio)
            baseline_rate[t] = max(0.0, b_t)
            resid_rate[t] = x[t] - baseline_rate[t]

        # Smooth band-pass residual slightly with causal EWMA so single-day Poisson shot noise doesn't flip phase
        smooth_resid = _causal_ewma_1d(resid_rate, alpha=0.45)

        re_vec = np.zeros(n, dtype=float)
        im_vec = np.zeros(n, dtype=float)
        raw_amp = np.zeros(n, dtype=float)
        phi_std = np.zeros(n, dtype=float)       # atan2(Im z, Re z) for z = x_tilde + i H{x_tilde}
        phi_pastoral = np.zeros(n, dtype=float)  # Pastoral Argand dial angle: Peak Crest at +90 deg (pi/2)
        omega = np.zeros(n, dtype=float)
        is_freeze = (ta.is_instructional <= 0.15)

        prev_phi_p = -0.25 * np.pi
        for t in range(n):
            r_slice = smooth_resid[max(0, t - w_hilb + 1) : t + 1]
            if len(r_slice) < 6:
                pad = np.full(6 - len(r_slice), float(r_slice[0]), dtype=float)
                r_win = np.concatenate([pad, r_slice])
            else:
                r_win = r_slice

            re_c, im_h = _causal_hilbert_quadrature_step(r_win)
            re_vec[t] = re_c
            im_vec[t] = im_h
            amp_t = float(np.hypot(re_c, im_h))
            raw_amp[t] = amp_t

            phi_std[t] = float(_wrap_angle(np.arctan2(im_h, re_c))) if amp_t > 1e-9 else 0.0

            if is_freeze[t] and t > 0:
                # Freeze instructional stress-cycle phase across weekends & holiday closures
                phi_pastoral[t] = prev_phi_p
                omega[t] = 0.0
                raw_amp[t] = raw_amp[t - 1] * 0.96
            else:
                # Pastoral dial coordinates: Re_dial = -H{x_tilde} (velocity lead), Im_dial = x_tilde (elevation)
                # => Quadrant I [0, 90): above baseline & rising toward crest at +90 deg (pi/2)
                # => Quadrant II [90, 180]: cresting & inflecting downward
                # => Quadrant III [-180, -90): post-peak decompression below baseline
                # => Quadrant IV [-90, 0): below baseline but winding upward toward surge
                if amp_t > 1e-8:
                    p_ang = float(_wrap_angle(np.arctan2(re_c, -im_h)))
                else:
                    p_ang = prev_phi_p
                d_phi = float(_wrap_angle(p_ang - prev_phi_p)) if t > 0 else 0.0
                phi_pastoral[t] = p_ang
                omega[t] = d_phi
                prev_phi_p = p_ang

        # Smoothed positive instructional angular speed (radians per instructional day)
        inst_step = np.maximum(0.2, ta.instructional_days_step)
        omega_per_inst_day = np.where(is_freeze, 0.0, omega / inst_step)
        default_omega = 2.0 * np.pi / 38.0  # ~38 instructional days per half-term wave
        omega_bar = np.zeros(n, dtype=float)
        cur_w = default_omega
        for t in range(n):
            if not is_freeze[t] and omega_per_inst_day[t] > 0.005:
                cur_w = 0.80 * cur_w + 0.20 * float(np.clip(omega_per_inst_day[t], 0.02, 0.85))
            omega_bar[t] = max(0.03, cur_w)

        # Estimated instructional days to next crest at phi = +pi/2
        days_to_peak = np.zeros(n, dtype=float)
        for t in range(n):
            phi_t = phi_pastoral[t]
            if phi_t <= 0.5 * np.pi:
                delta_ang = 0.5 * np.pi - phi_t
            else:
                delta_ang = (2.5 * np.pi) - phi_t
            days_to_peak[t] = float(np.clip(max(0.0, delta_ang / max(omega_bar[t], 1e-3)), 0.0, 60.0))

        baseline_counts = baseline_rate * enrol_scale
        return {
            "baseline_rate": baseline_rate,
            "baseline_counts": baseline_counts,
            "resid_rate": smooth_resid,
            "re_z": re_vec,
            "im_z": im_vec,
            "raw_amp": raw_amp,
            "phi_std": phi_std,
            "phi": phi_pastoral,
            "omega": omega,
            "omega_bar": omega_bar,
            "days_to_peak": days_to_peak,
            "is_holiday_freeze": is_freeze.astype(int),
            "observed_counts": x_counts,
        }

    def fit(self, X: Union[pd.DataFrame, SchemaBundle], y: Any = None) -> "WarpedFourierHilbertTransformer":
        bundle = self._ensure_bundle_fit(X)
        states = self._compute_raw_hilbert_states(bundle)
        active_amps = states["raw_amp"][states["is_holiday_freeze"] == 0]
        if len(active_amps) == 0:
            active_amps = states["raw_amp"]
        med_a = float(np.median(active_amps)) if len(active_amps) > 0 else 1.0
        mad_a = float(np.median(np.abs(active_amps - med_a))) if len(active_amps) > 0 else 0.5
        # Scale so r_norm = 1.0 represents typical training oscillation amplitude
        self.amp_scale_ = max(0.15, med_a + 1.4826 * mad_a * 0.35)
        self.is_fitted_ = True
        return self

    def transform(self, X: Union[pd.DataFrame, SchemaBundle]) -> Union[pd.DataFrame, np.ndarray]:
        if not getattr(self, "is_fitted_", False):
            self.fit(X)
        bundle = self._ensure_bundle_transform(X)
        ta = bundle.tier_a
        states = self._compute_raw_hilbert_states(bundle)

        r_norm = states["raw_amp"] / max(1e-6, self.amp_scale_)
        phi = states["phi"]

        cols: List[np.ndarray] = []
        names: List[str] = []

        # Dual harmonic clocks (Calendar 365.25d vs Warped Academic 190 instructional days)
        for k in range(1, self.n_harmonics + 1):
            cols.append(np.sin(2.0 * np.pi * k * ta.calendar_cycles))
            cols.append(np.cos(2.0 * np.pi * k * ta.calendar_cycles))
            names.extend([f"f01_cal_sin_k{k}", f"f01_cal_cos_k{k}"])

        for k in range(1, self.n_harmonics + 1):
            cols.append(np.sin(2.0 * np.pi * k * ta.warped_acad_cycles))
            cols.append(np.cos(2.0 * np.pi * k * ta.warped_acad_cycles))
            names.extend([f"f01_acad_warped_sin_k{k}", f"f01_acad_warped_cos_k{k}"])

        cols.extend(
            [
                states["baseline_rate"],
                states["resid_rate"],
                r_norm,
                np.sin(phi),
                np.cos(phi),
                phi,
                states["omega"],
                states["days_to_peak"],
            ]
        )
        names.extend(
            [
                "f01_fft_denoised_baseline_rate",
                "f01_bandpass_residual_rate",
                "f01_hilbert_stress_amp_norm",
                "f01_hilbert_phase_sin",
                "f01_hilbert_phase_cos",
                "f01_hilbert_phase_rad",
                "f01_hilbert_angular_velocity",
                "f01_est_inst_days_to_peak",
            ]
        )

        self.last_diagnostics_ = {
            **states,
            "r_norm": r_norm,
            "amp_scale": self.amp_scale_,
            "dates": ta.dates,
            "instructional_days_cum": ta.instructional_days_cum,
            "holiday_labels": ta.holiday_labels,
        }

        feat_mat = np.column_stack(cols)
        return self._format_output(feat_mat, names, bundle, X)
