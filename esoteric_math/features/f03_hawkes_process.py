"""
Module 03: Self- & Cross-Exciting Hawkes Contagion Engine (`esoteric_math/features/f03_hawkes_process.py`).

Fits univariate or C-category conditional intensity:
    lambda_c(t) = mu_c(t) + sum_{j=1}^C sum_{t_k^{(j)} < t} alpha_{cj} * exp(-beta_{cj} * (t - t_k^{(j)}))
on training splits via L-BFGS-B enforcing strict subcritical branching rho(Gamma) < 1
where Gamma_{cj} = alpha_{cj} / beta_{cj}.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from esoteric_math.features.base import CausalBaseTransformer
from esoteric_math.schema_bridge import SchemaBundle


def _spectral_radius(mat: np.ndarray) -> float:
    if mat.size == 0:
        return 0.0
    eigvals = np.linalg.eigvals(mat)
    return float(np.max(np.abs(eigvals)))


def _enforce_subcritical_gamma(
    alpha: np.ndarray,
    beta: np.ndarray,
    max_rho: float = 0.90,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Project alpha >= 0 so Gamma = alpha / beta satisfies spectral radius rho(Gamma) <= max_rho < 1."""
    alpha_pos = np.maximum(0.0, alpha)
    beta_pos = np.maximum(0.05, beta)
    gamma = alpha_pos / beta_pos
    rho = _spectral_radius(gamma)
    if rho > max_rho and rho > 1e-12:
        scale = max_rho / rho
        alpha_pos = alpha_pos * scale
        gamma = alpha_pos / beta_pos
        rho = _spectral_radius(gamma)
    return alpha_pos, gamma, rho


def _compute_hawkes_recursion(
    counts: np.ndarray,
    seasonal_scale: np.ndarray,
    mu_base: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    dt_steps: np.ndarray,
    endog_gate: Optional[np.ndarray] = None,
    shock_floor_ratio: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Strictly causal discrete-time Hawkes excitation recursion over T steps and C categories.
    At step t:
      - `mem_pre[t, c, j]` sums shocks strictly from s < t: sum_{s < t} alpha_{cj} * y_j(s) * exp(-beta_{cj}*(tau_t - tau_s))
      - `mem_post[t, c, j]` updates after observing step t (reading <= t, used for forward projection cone (t, t+5]!)
    """
    T, C = counts.shape
    mu_t = np.maximum(1e-4, seasonal_scale[:, None] * mu_base[None, :])
    gate = np.ones((T, 1, 1), dtype=float) if endog_gate is None else np.asarray(endog_gate, dtype=float).reshape(T, 1, 1)
    mem_pre = np.zeros((T, C, C), dtype=float)
    mem_post = np.zeros((T, C, C), dtype=float)
    cur_mem = np.zeros((C, C), dtype=float)

    for t in range(T):
        dt = max(0.2, float(dt_steps[t]))
        decay = np.exp(-beta * dt)
        cur_mem = cur_mem * decay
        mem_pre[t] = cur_mem * gate[t]
        # Incorporate step t count/excess shock into post-step memory (still strictly <= t!)
        shock_t = np.maximum(0.0, counts[t] - shock_floor_ratio * mu_t[t])
        cur_mem = cur_mem + alpha * shock_t[None, :]
        mem_post[t] = cur_mem * gate[t]

    endog_sum = np.sum(mem_pre, axis=2)
    lam_t = np.maximum(1e-4, mu_t + endog_sum)
    return mu_t, mem_pre, mem_post, lam_t


class HawkesContagionTransformer(CausalBaseTransformer):
    """
    Module 03: Multivariate / Univariate Subcritical Hawkes Contagion Intensity Transformer.
    """

    module_id = "f03"
    module_name = "Hawkes Self- & Cross-Exciting Contagion"

    def __init__(
        self,
        max_rho: float = 0.88,
        max_iter: int = 60,
        causal_mode: Literal["telemetry", "forecast"] = "telemetry",
        train_cutoff: Optional[int] = None,
        return_dataframe: bool = True,
    ) -> None:
        super().__init__(
            causal_mode=causal_mode,
            train_cutoff=train_cutoff,
            return_dataframe=return_dataframe,
        )
        self.max_rho = max_rho
        self.max_iter = max_iter

    @staticmethod
    def _build_seasonal_scale(bundle: SchemaBundle, base_enrol: float) -> np.ndarray:
        ta = bundle.tier_a
        enrol_ratio = np.maximum(1.0, ta.exposure) / max(1.0, base_enrol)
        # Modulate routine calendar baseline by instructional activity & harmonic academic calendar
        acad_wave = 1.0 + 0.18 * np.sin(2.0 * np.pi * ta.warped_acad_cycles - 0.3)
        inst_mod = np.where(ta.is_instructional > 0.15, np.maximum(0.35, ta.is_instructional) * acad_wave, 0.12)
        return np.maximum(0.05, enrol_ratio * inst_mod)

    def fit(self, X: Union[pd.DataFrame, SchemaBundle], y: Any = None) -> "HawkesContagionTransformer":
        bundle = self._ensure_bundle_fit(X)
        ta, tb = bundle.tier_a, bundle.tier_b
        # Fit on observed counts within the training slice
        counts_tr = np.maximum(0.0, tb.raw_category_counts.astype(float))
        T_tr, C = counts_tr.shape
        self.n_categories_ = C
        self.category_names_ = list(tb.category_names)
        self.base_enrol_ = float(np.mean(ta.exposure)) if len(ta.exposure) > 0 else 1000.0

        seas_scale = self._build_seasonal_scale(bundle, self.base_enrol_)
        dt_steps = np.ones(T_tr, dtype=float)
        is_daily = (ta.cadence == "daily")
        endog_gate_tr = (
            np.where(ta.is_instructional > 0.15, np.maximum(0.45, ta.is_instructional), 0.08)
            if is_daily
            else None
        )
        shock_floor = 0.35 if is_daily else 0.0
        self.shock_floor_ = shock_floor

        mean_counts = np.maximum(0.05, np.mean(counts_tr, axis=0))
        mean_seas = max(0.1, float(np.mean(seas_scale)))

        # Parameter vector theta: [log_mu (C), log_alpha (C*C), log_beta (C*C)]
        mu_init = np.maximum(0.03, 0.68 * mean_counts / mean_seas)
        beta_init = np.full((C, C), 0.35, dtype=float)
        gamma_init = np.full((C, C), 0.22 / max(1, C), dtype=float)
        np.fill_diagonal(gamma_init, min(0.26, 0.40 / max(1, np.sqrt(C))))
        alpha_init = gamma_init * beta_init

        theta0 = np.concatenate(
            [
                np.log(mu_init),
                np.log(np.maximum(1e-4, alpha_init)).reshape(-1),
                np.log(beta_init).reshape(-1),
            ]
        )

        bounds = (
            [(-6.0, 8.0)] * C
            + [(-8.0, 1.5)] * (C * C)
            + [(np.log(0.08), np.log(2.5))] * (C * C)
        )

        def _neg_loglik(theta: np.ndarray) -> float:
            mu_b = np.exp(theta[:C])
            alp = np.exp(theta[C : C + C * C]).reshape(C, C)
            bet = np.exp(theta[C + C * C :]).reshape(C, C)
            gam = alp / bet
            rho = _spectral_radius(gam)

            _, _, _, lam = _compute_hawkes_recursion(
                counts_tr,
                seas_scale,
                mu_b,
                alp,
                bet,
                dt_steps,
                endog_gate=endog_gate_tr,
                shock_floor_ratio=shock_floor,
            )
            # Point-process / Poisson conditional log-likelihood + soft spectral barrier
            nll = float(np.sum(lam - counts_tr * np.log(np.maximum(1e-8, lam)))) / max(1, T_tr)
            reg = 0.015 * float(np.sum(gam**2)) + 0.005 * float(np.sum(( np.log(bet / 0.35) ) ** 2))
            if rho > self.max_rho:
                reg += 50.0 * ((rho - self.max_rho) ** 2)
            return nll + reg

        if T_tr >= 8 and float(np.sum(counts_tr)) > 0:
            opt = minimize(
                _neg_loglik,
                theta0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": self.max_iter, "ftol": 1e-6},
            )
            theta_hat = np.asarray(opt.x, dtype=float)
        else:
            theta_hat = theta0

        self.mu_base_ = np.exp(theta_hat[:C])
        raw_alpha = np.exp(theta_hat[C : C + C * C]).reshape(C, C)
        self.beta_ = np.exp(theta_hat[C + C * C :]).reshape(C, C)
        self.alpha_, self.gamma_, self.spectral_radius_ = _enforce_subcritical_gamma(
            raw_alpha, self.beta_, max_rho=self.max_rho
        )

        # Convert decay rates into instructional-day half-lives
        cadence_days = 1.0 if ta.cadence == "daily" else (5.0 if ta.cadence == "weekly" else 18.0)
        self.half_life_days_ = (np.log(2.0) / np.maximum(1e-4, self.beta_)) * cadence_days
        self.cadence_days_ = cadence_days
        self.is_fitted_ = True
        return self

    def transform(self, X: Union[pd.DataFrame, SchemaBundle]) -> Union[pd.DataFrame, np.ndarray]:
        if not getattr(self, "is_fitted_", False):
            self.fit(X)
        bundle = self._ensure_bundle_transform(X)
        ta, tb = bundle.tier_a, bundle.tier_b
        n = bundle.n_steps
        C = self.n_categories_

        # In telemetry mode: `tb.raw_category_counts` has counts up to t, and `mem_pre[t]` uses strictly s < t,
        # while for step t's immediate intensity including prior days we also record `mem_post` for cone (t, t+5].
        # In forecast mode: `tb.category_counts` is already shifted by 1, so we use `mem_post` on shifted counts
        # (which equals `mem_pre` on raw counts, i.e., strictly reading {0, ..., t-1}!).
        seas_scale = self._build_seasonal_scale(bundle, self.base_enrol_)
        dt_steps = np.ones(n, dtype=float)
        is_daily = (ta.cadence == "daily")
        endog_gate = (
            np.where(ta.is_instructional > 0.15, np.maximum(0.45, ta.is_instructional), 0.08)
            if is_daily
            else None
        )
        shock_floor = float(getattr(self, "shock_floor_", 0.35 if is_daily else 0.0))

        if bundle.causal_mode == "forecast":
            mu_t, _, mem_active, _ = _compute_hawkes_recursion(
                tb.category_counts,
                seas_scale,
                self.mu_base_,
                self.alpha_,
                self.beta_,
                dt_steps,
                endog_gate=endog_gate,
                shock_floor_ratio=shock_floor,
            )
            endog_by_target = np.sum(mem_active, axis=2)
            lam_t = np.maximum(1e-4, mu_t + endog_by_target)
            mem_cone_anchor = mem_active
        else:
            mu_t, mem_pre, mem_post, lam_t = _compute_hawkes_recursion(
                tb.raw_category_counts,
                seas_scale,
                self.mu_base_,
                self.alpha_,
                self.beta_,
                dt_steps,
                endog_gate=endog_gate,
                shock_floor_ratio=shock_floor,
            )
            mem_active = mem_pre
            endog_by_target = np.sum(mem_pre, axis=2)
            mem_cone_anchor = mem_post

        tot_mu = np.sum(mu_t, axis=1)
        tot_endog = np.sum(endog_by_target, axis=1)
        tot_lam = np.maximum(1e-6, tot_mu + tot_endog)
        endog_share = np.clip(tot_endog / tot_lam, 0.0, 0.999)

        # Excitation by source category j: sum_c M_{cj}(t)
        endog_by_source = np.sum(mem_active, axis=1)  # (T, C)

        cols: List[np.ndarray] = [
            tot_mu,
            tot_endog,
            tot_lam,
            endog_share,
            np.full(n, self.spectral_radius_, dtype=float),
        ]
        names: List[str] = [
            "f03_hawkes_exog_baseline_total",
            "f03_hawkes_endog_memory_total",
            "f03_hawkes_total_intensity",
            "f03_hawkes_endog_share",
            "f03_hawkes_spectral_radius",
        ]

        for j in range(C):
            cols.append(endog_by_source[:, j])
            names.append(f"f03_hawkes_source_memory_c{j}")

        mean_hl = float(np.mean(self.half_life_days_))
        cols.append(np.full(n, mean_hl, dtype=float))
        names.append("f03_hawkes_mean_halflife_days")

        self.last_diagnostics_ = {
            "dates": ta.dates,
            "category_names": list(self.category_names_),
            "observed_category_counts": tb.raw_category_counts.copy(),
            "observed_total_counts": ta.y_raw_observed.copy(),
            "mu_t": mu_t,
            "exog_total": tot_mu,
            "memory_pre": mem_active,
            "memory_cone_anchor": mem_cone_anchor,
            "endog_by_source": endog_by_source,
            "lambda_t": lam_t,
            "lambda_total": tot_lam,
            "endog_share": endog_share,
            "alpha_matrix": self.alpha_.copy(),
            "beta_matrix": self.beta_.copy(),
            "gamma_matrix": self.gamma_.copy(),
            "half_life_days": self.half_life_days_.copy(),
            "spectral_radius": float(self.spectral_radius_),
            "cadence_days": float(self.cadence_days_),
        }
        return self._format_output(np.column_stack(cols), names, bundle, X)
