"""
Module 08: Cellular Sheaf Cohomology & Contextual Masking Frustration
(`esoteric_math/features/f08_cellular_sheaves.py`).

Attaches multi-attribute behavioral/contextual stalks x_v(t) in R^{d_v} to vertices of G=(V, E),
learns linear restriction maps F_{v <= e} on calm training periods via regularized Procrustes
alignment, and assembles coboundary delta and Sheaf Laplacian L_F = delta^T delta (guaranteed PSD).
Outputs global Sheaf Dirichlet Energy E_F(x_t) = x_t^T L_F x_t, per-edge coboundary frustration
f_e(t) = ||F_{u <= e} x_u(t) - F_{v <= e} x_v(t)||_2, harmonic consensus projections P_{ker(L_F)} x_t,
and signed stalk discrepancies detecting "Contextual Masking".
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import pandas as pd

from esoteric_math.features.base import CausalBaseTransformer
from esoteric_math.schema_bridge import SchemaBundle


def learn_procrustes_sheaf_laplacian(
    calm_stalks: np.ndarray,
    adjacency: np.ndarray,
    reg_identity: float = 0.20,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[Tuple[int, int, float, np.ndarray, np.ndarray]],
]:
    """
    Learn cellular sheaf restriction maps on calm training stalks `calm_stalks` of shape (T_calm, V, d)
    over undirected weighted graph `adjacency` (V, V).
    Returns:
      - `coboundary` delta of shape (|E|*d, V*d)
      - `sheaf_laplacian` L_F = delta^T delta of shape (V*d, V*d) (exact symmetric PSD)
      - `harmonic_proj` P_{ker(L_F)} of shape (V*d, V*d) projecting onto the lowest d consensus modes
      - `edge_specs`: list of (u, v, weight, F_u, F_v)
    """
    T_c, v_cnt, d_dim = calm_stalks.shape
    A = np.maximum(0.0, 0.5 * (adjacency + adjacency.T)).copy()
    np.fill_diagonal(A, 0.0)

    edge_specs: List[Tuple[int, int, float, np.ndarray, np.ndarray]] = []
    for u in range(v_cnt):
        for v in range(u + 1, v_cnt):
            w_e = float(A[u, v])
            if w_e <= 1e-8:
                continue
            sqrt_w = float(np.sqrt(w_e))
            if T_c >= 3:
                Xu = calm_stalks[:, u, :]  # (T_c, d)
                Xv = calm_stalks[:, v, :]  # (T_c, d)
                # Regularized orthogonal Procrustes alignment: Xu @ R ≈ Xv
                cross = (Xu.T @ Xv) / max(1, T_c) + reg_identity * np.eye(d_dim, dtype=float)
                U_svd, _, Vt_svd = np.linalg.svd(cross, full_matrices=False)
                R_uv = U_svd @ Vt_svd
                # Keep det(R_uv) = +1 so stalks preserve orientation across peer cohorts
                if np.linalg.det(R_uv) < 0:
                    U_svd[:, -1] *= -1.0
                    R_uv = U_svd @ Vt_svd
            else:
                R_uv = np.eye(d_dim, dtype=float)

            F_u = sqrt_w * R_uv.T
            F_v = sqrt_w * np.eye(d_dim, dtype=float)
            edge_specs.append((u, v, w_e, F_u, F_v))

    n_edges = len(edge_specs)
    total_v_dim = v_cnt * d_dim
    if n_edges == 0:
        delta = np.zeros((0, total_v_dim), dtype=float)
        L_F = np.zeros((total_v_dim, total_v_dim), dtype=float)
        P_harm = np.eye(total_v_dim, dtype=float)
        return delta, L_F, P_harm, edge_specs

    delta = np.zeros((n_edges * d_dim, total_v_dim), dtype=float)
    for e_idx, (u, v, _, F_u, F_v) in enumerate(edge_specs):
        r0, r1 = e_idx * d_dim, (e_idx + 1) * d_dim
        delta[r0:r1, u * d_dim : (u + 1) * d_dim] = F_u
        delta[r0:r1, v * d_dim : (v + 1) * d_dim] = -F_v

    L_F = delta.T @ delta
    L_F = 0.5 * (L_F + L_F.T)

    # Harmonic subspace H^0(G; F): span of the d lowest-energy global consensus eigenvectors of L_F
    eigvals, eigvecs = np.linalg.eigh(L_F)
    k_harm = min(d_dim, total_v_dim)
    U_harm = eigvecs[:, :k_harm]
    P_harm = U_harm @ U_harm.T
    return delta, L_F, P_harm, edge_specs


class CellularSheafCohomologyTransformer(CausalBaseTransformer):
    """
    Module 08: Cellular Sheaf Laplacian & Cohomological Frustration Engine.
    """

    module_id = "f08"
    module_name = "Cellular Sheaf Cohomology & Frustration"

    def __init__(
        self,
        calm_percentile: float = 75.0,
        reg_identity: float = 0.25,
        causal_mode: Literal["telemetry", "forecast"] = "telemetry",
        train_cutoff: Optional[int] = None,
        return_dataframe: bool = True,
    ) -> None:
        super().__init__(
            causal_mode=causal_mode,
            train_cutoff=train_cutoff,
            return_dataframe=return_dataframe,
        )
        self.calm_percentile = calm_percentile
        self.reg_identity = reg_identity

    def _evaluate_sheaf_states(self, stalks: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Evaluate global Dirichlet energy, edge frustrations, harmonic consensus projections,
        and per-vertex non-harmonic residuals strictly pointwise at each t (zero future lookahead).
        """
        T, v_cnt, d_dim = stalks.shape
        x_flat = stalks.reshape(T, v_cnt * d_dim)

        # Coboundary residual vector delta * x_t of shape (T, |E| * d)
        coboundary_vec = x_flat @ self.delta_.T
        n_edges = len(self.edge_specs_)
        if n_edges > 0:
            coboundary_by_edge = coboundary_vec.reshape(T, n_edges, d_dim)
            edge_frustration = np.linalg.norm(coboundary_by_edge, axis=2)  # (T, |E|)
            global_energy = np.sum(edge_frustration**2, axis=1)           # exact x_t^T L_F x_t >= 0
        else:
            coboundary_by_edge = np.zeros((T, 0, d_dim), dtype=float)
            edge_frustration = np.zeros((T, 0), dtype=float)
            global_energy = np.zeros(T, dtype=float)

        harm_flat = x_flat @ self.harmonic_proj_
        harm_stalks = harm_flat.reshape(T, v_cnt, d_dim)
        resid_stalks = stalks - harm_stalks  # Signed contextual masking discrepancy x_v(t) - (P_{ker L_F} x_t)_v
        vertex_resid_norm = np.linalg.norm(resid_stalks, axis=2)  # (T, V)

        return {
            "global_energy": global_energy,
            "edge_frustration": edge_frustration,
            "coboundary_by_edge": coboundary_by_edge,
            "harmonic_stalks": harm_stalks,
            "residual_stalks": resid_stalks,
            "vertex_residual_norm": vertex_resid_norm,
        }

    def fit(self, X: Union[pd.DataFrame, SchemaBundle], y: Any = None) -> "CellularSheafCohomologyTransformer":
        bundle = self._ensure_bundle_fit(X)
        stalks_tr = bundle.tier_c.stalks
        rates_tr = bundle.tier_a.y_rate_per_1000

        if len(rates_tr) >= 6:
            cutoff_r = float(np.percentile(rates_tr, self.calm_percentile))
            calm_mask = rates_tr <= cutoff_r
            calm_stalks = stalks_tr[calm_mask] if int(np.sum(calm_mask)) >= 3 else stalks_tr
        else:
            calm_stalks = stalks_tr

        self.delta_, self.sheaf_laplacian_, self.harmonic_proj_, self.edge_specs_ = (
            learn_procrustes_sheaf_laplacian(
                calm_stalks, bundle.tier_c.adjacency, reg_identity=self.reg_identity
            )
        )

        tr_states = self._evaluate_sheaf_states(stalks_tr)
        ge = tr_states["global_energy"]
        self.energy_median_ = float(np.median(ge)) if len(ge) > 0 else 1.0
        self.energy_std_ = max(0.15, float(np.std(ge))) if len(ge) > 0 else 1.0
        self.energy_tau90_ = float(np.percentile(ge, 90.0)) if len(ge) > 0 else 2.0
        self.energy_tau975_ = float(np.percentile(ge, 97.5)) if len(ge) > 0 else 3.5

        ef = tr_states["edge_frustration"]
        if ef.size > 0:
            self.edge_mean_ = np.mean(ef, axis=0)
            self.edge_std_ = np.maximum(0.10, np.std(ef, axis=0))
            self.edge_tau90_ = np.percentile(ef, 90.0, axis=0)
        else:
            self.edge_mean_ = np.zeros(0, dtype=float)
            self.edge_std_ = np.ones(0, dtype=float)
            self.edge_tau90_ = np.zeros(0, dtype=float)

        vr = tr_states["vertex_residual_norm"]
        self.vertex_mean_ = np.mean(vr, axis=0) if vr.size > 0 else np.zeros(stalks_tr.shape[1])
        self.vertex_std_ = np.maximum(0.10, np.std(vr, axis=0)) if vr.size > 0 else np.ones(stalks_tr.shape[1])

        self.is_fitted_ = True
        return self

    def transform(self, X: Union[pd.DataFrame, SchemaBundle]) -> Union[pd.DataFrame, np.ndarray]:
        if not getattr(self, "is_fitted_", False):
            self.fit(X)
        bundle = self._ensure_bundle_transform(X)
        tc = bundle.tier_c
        states = self._evaluate_sheaf_states(tc.stalks)

        ge = states["global_energy"]
        ge_z = (ge - self.energy_median_) / self.energy_std_
        ef = states["edge_frustration"]
        vr = states["vertex_residual_norm"]

        if ef.shape[1] > 0:
            ef_z = (ef - self.edge_mean_[None, :]) / self.edge_std_[None, :]
            max_edge_f = np.max(ef, axis=1)
            max_edge_z = np.max(ef_z, axis=1)
            frustrated_edge_cnt = np.sum(ef >= self.edge_tau90_[None, :], axis=1).astype(float)
        else:
            ef_z = np.zeros_like(ef)
            max_edge_f = np.zeros(bundle.n_steps, dtype=float)
            max_edge_z = np.zeros(bundle.n_steps, dtype=float)
            frustrated_edge_cnt = np.zeros(bundle.n_steps, dtype=float)

        vr_z = (vr - self.vertex_mean_[None, :]) / self.vertex_std_[None, :]
        max_vertex_resid = np.max(vr, axis=1) if vr.shape[1] > 0 else np.zeros(bundle.n_steps)
        harm_energy = np.sum(states["harmonic_stalks"] ** 2, axis=(1, 2))

        # Contextual masking score: positive attendance stalk discrepancy (+dim 0) paired with negative pastoral/engagement (-dim 1,2)
        res_s = states["residual_stalks"]  # (T, V, d)
        if res_s.shape[2] >= 3:
            masking_div = np.max(
                np.maximum(0.0, res_s[:, :, 0]) * np.maximum(0.0, -0.5 * (res_s[:, :, 1] + res_s[:, :, 2])),
                axis=1,
            )
        else:
            masking_div = max_vertex_resid

        feats = np.column_stack(
            [
                ge,
                ge_z,
                max_edge_f,
                max_edge_z,
                frustrated_edge_cnt,
                max_vertex_resid,
                harm_energy,
                masking_div,
            ]
        )
        names = [
            "f08_sheaf_dirichlet_energy",
            "f08_sheaf_energy_zscore",
            "f08_sheaf_max_edge_frustration",
            "f08_sheaf_max_edge_zscore",
            "f08_sheaf_frustrated_edge_count",
            "f08_sheaf_max_vertex_residual",
            "f08_sheaf_harmonic_consensus_norm2",
            "f08_sheaf_contextual_masking_index",
        ]

        self.last_diagnostics_ = {
            "dates": bundle.tier_a.dates,
            "topology_mode": tc.topology_mode,
            "node_ids": list(tc.node_ids),
            "node_labels": list(tc.node_labels),
            "node_weights": tc.node_weights.copy(),
            "stalk_dim_names": list(tc.stalk_dim_names),
            "observed_stalks": tc.stalks.copy(),
            "harmonic_stalks": states["harmonic_stalks"],
            "residual_stalks": states["residual_stalks"],
            "vertex_residual_norm": vr,
            "vertex_residual_z": vr_z,
            "global_energy": ge,
            "global_energy_z": ge_z,
            "energy_tau90": float(self.energy_tau90_),
            "energy_tau975": float(self.energy_tau975_),
            "edges": [(u, v, float(w)) for u, v, w, _, _ in self.edge_specs_],
            "edge_frustration": ef,
            "edge_frustration_z": ef_z,
            "edge_tau90": self.edge_tau90_.copy(),
            "sheaf_laplacian": self.sheaf_laplacian_.copy(),
        }
        return self._format_output(feats, names, bundle, X)
