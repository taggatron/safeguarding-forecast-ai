"""
Module 05: Graph Spectral Analysis & Discrete Ollivier-Ricci Curvature
(`esoteric_math/features/f05_graph_ricci.py`).

Implements:
- Normalized Graph Laplacian L = I - D^{-1/2} A D^{-1/2} Graph Fourier Transform (GFT),
  low-frequency spectral energy ratio, Fiedler projection, and Graph Dirichlet energy.
- Exact Discrete Ollivier-Ricci Curvature kappa(u, v) = 1 - W_1(m_u^alpha, m_v^alpha) / d_G(u, v)
  via Kantorovich-Rubinstein optimal transport linear programming (`scipy.optimize.linprog`),
  outputting negatively curved bottleneck stress scores.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse.csgraph import shortest_path

from esoteric_math.features.base import CausalBaseTransformer
from esoteric_math.schema_bridge import SchemaBundle


def compute_normalized_laplacian(adjacency: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute symmetric normalized Graph Laplacian L = I - D^{-1/2} A D^{-1/2}
    and its sorted eigendecomposition (eigvals, eigvecs).
    """
    A = np.maximum(0.0, 0.5 * (adjacency + adjacency.T)).copy()
    np.fill_diagonal(A, 0.0)
    v_cnt = A.shape[0]
    if v_cnt == 0:
        return np.zeros((0, 0)), np.zeros(0), np.zeros((0, 0))
    deg = np.sum(A, axis=1)
    inv_sqrt_deg = np.where(deg > 1e-12, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = np.diag(inv_sqrt_deg)
    L_norm = np.eye(v_cnt, dtype=float) - (D_inv_sqrt @ A @ D_inv_sqrt)
    L_norm = 0.5 * (L_norm + L_norm.T)
    eigvals, eigvecs = np.linalg.eigh(L_norm)
    eigvals = np.maximum(0.0, eigvals)
    return L_norm, eigvals, eigvecs


def _earth_movers_w1(mu: np.ndarray, nu: np.ndarray, dist_mat: np.ndarray) -> float:
    """
    Exact 1-Wasserstein distance W_1(mu, nu) on finite metric space (V, dist_mat)
    solved via linear programming (`scipy.optimize.linprog` HiGHS solver).
    """
    v_cnt = len(mu)
    if v_cnt <= 1 or np.allclose(mu, nu, atol=1e-12):
        return 0.0

    c = dist_mat.reshape(-1)
    # Row sum constraints: sum_j Pi_{i,j} = mu_i; Col sum constraints: sum_i Pi_{i,j} = nu_j
    n_vars = v_cnt * v_cnt
    A_eq = np.zeros((2 * v_cnt - 1, n_vars), dtype=float)
    b_eq = np.zeros(2 * v_cnt - 1, dtype=float)

    for i in range(v_cnt):
        A_eq[i, i * v_cnt : (i + 1) * v_cnt] = 1.0
        b_eq[i] = float(mu[i])
    for j in range(v_cnt - 1):
        A_eq[v_cnt + j, j::v_cnt] = 1.0
        b_eq[v_cnt + j] = float(nu[j])

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0.0, 1.0), method="highs")
    if res.success and np.isfinite(res.fun):
        return float(max(0.0, res.fun))
    return float(np.sum(np.abs(mu - nu)) * 0.5 * np.max(dist_mat))


def compute_ollivier_ricci_curvatures(
    adjacency: np.ndarray,
    laziness_alpha: float = 0.5,
) -> Tuple[List[Tuple[int, int, float]], np.ndarray]:
    """
    Compute discrete Ollivier-Ricci curvature kappa(u, v) = 1 - W_1(m_u^alpha, m_v^alpha) / d_G(u, v)
    for all edges (u, v) with A_{u,v} > 0.
    """
    A = np.maximum(0.0, 0.5 * (adjacency + adjacency.T)).copy()
    np.fill_diagonal(A, 0.0)
    v_cnt = A.shape[0]
    if v_cnt < 2:
        return [], np.zeros((v_cnt, v_cnt), dtype=float)

    # Edge length metric d_e(u, v) = 1 / A_{u,v} where connected
    length_graph = np.where(A > 1e-8, 1.0 / np.maximum(1e-4, A), np.inf)
    np.fill_diagonal(length_graph, 0.0)
    d_G = shortest_path(length_graph, directed=False, unweighted=False)
    finite_max = float(np.max(d_G[np.isfinite(d_G)])) if np.any(np.isfinite(d_G)) else 1.0
    d_G = np.where(np.isfinite(d_G), d_G, 2.0 * max(1.0, finite_max))

    # Lazy random-walk probability measures m_u^alpha
    measures = np.zeros((v_cnt, v_cnt), dtype=float)
    for u in range(v_cnt):
        deg_u = float(np.sum(A[u]))
        if deg_u > 1e-12:
            measures[u] = (1.0 - laziness_alpha) * (A[u] / deg_u)
            measures[u, u] += laziness_alpha
        else:
            measures[u, u] = 1.0

    edge_curvatures: List[Tuple[int, int, float]] = []
    kappa_mat = np.zeros((v_cnt, v_cnt), dtype=float)
    for u in range(v_cnt):
        for v in range(u + 1, v_cnt):
            if A[u, v] > 1e-8 and d_G[u, v] > 1e-12:
                w1 = _earth_movers_w1(measures[u], measures[v], d_G)
                kap = 1.0 - (w1 / float(d_G[u, v]))
                edge_curvatures.append((u, v, float(kap)))
                kappa_mat[u, v] = float(kap)
                kappa_mat[v, u] = float(kap)

    return edge_curvatures, kappa_mat


class GraphSpectralRicciTransformer(CausalBaseTransformer):
    """
    Module 05: Normalized Graph Laplacian GFT & Discrete Ollivier-Ricci Curvature Transformer.
    """

    module_id = "f05"
    module_name = "Graph Spectral GFT & Ollivier-Ricci Curvature"

    def __init__(
        self,
        laziness_alpha: float = 0.5,
        causal_mode: Literal["telemetry", "forecast"] = "telemetry",
        train_cutoff: Optional[int] = None,
        return_dataframe: bool = True,
    ) -> None:
        super().__init__(
            causal_mode=causal_mode,
            train_cutoff=train_cutoff,
            return_dataframe=return_dataframe,
        )
        self.laziness_alpha = laziness_alpha

    def fit(self, X: Union[pd.DataFrame, SchemaBundle], y: Any = None) -> "GraphSpectralRicciTransformer":
        bundle = self._ensure_bundle_fit(X)
        A = bundle.tier_c.adjacency
        self.L_norm_, self.eigvals_, self.eigvecs_ = compute_normalized_laplacian(A)
        self.edge_curvatures_, self.kappa_mat_ = compute_ollivier_ricci_curvatures(
            A, laziness_alpha=self.laziness_alpha
        )
        self.min_kappa_ = (
            min(k for _, _, k in self.edge_curvatures_) if self.edge_curvatures_ else 0.0
        )
        self.neg_kappa_count_ = sum(1 for _, _, k in self.edge_curvatures_ if k < 0.0)
        self.is_fitted_ = True
        return self

    def transform(self, X: Union[pd.DataFrame, SchemaBundle]) -> Union[pd.DataFrame, np.ndarray]:
        if not getattr(self, "is_fitted_", False):
            self.fit(X)
        bundle = self._ensure_bundle_transform(X)
        n = bundle.n_steps
        S = bundle.tier_c.node_scalar_stress  # (T, V)
        v_cnt = S.shape[1]

        U = self.eigvecs_
        L_mat = self.L_norm_
        s_hat = S @ U  # Graph Fourier Transform (T, V)
        total_energy = np.sum(s_hat**2, axis=1)
        low_cutoff = max(1, (v_cnt + 1) // 2)
        low_energy = np.sum(s_hat[:, :low_cutoff] ** 2, axis=1)
        low_freq_ratio = np.where(total_energy > 1e-10, low_energy / np.maximum(1e-10, total_energy), 1.0)

        fiedler_proj = np.abs(s_hat[:, 1]) if v_cnt >= 2 else np.zeros(n, dtype=float)
        high_freq_energy = np.sum(s_hat[:, low_cutoff:] ** 2, axis=1) if v_cnt > low_cutoff else np.zeros(n)
        dirichlet_energy = np.einsum("ti,ij,tj->t", S, L_mat, S)

        # Bottleneck-weighted stress across negatively curved (or lowest-curvature bridge) edges
        bottleneck_stress = np.zeros(n, dtype=float)
        bridge_gradient = np.zeros(n, dtype=float)
        if self.edge_curvatures_:
            kappas = np.array([k for _, _, k in self.edge_curvatures_], dtype=float)
            # Negative curvature weight max(0, -kappa); if graph is positively curved everywhere, weight weakest bridges
            neg_weights = np.maximum(0.0, -kappas)
            if np.sum(neg_weights) < 1e-9:
                neg_weights = np.maximum(0.0, float(np.median(kappas)) - kappas + 0.05)
            for idx_e, (u, v, _) in enumerate(self.edge_curvatures_):
                w_b = float(neg_weights[idx_e])
                if w_b > 0:
                    bottleneck_stress += w_b * (np.abs(S[:, u]) + np.abs(S[:, v]))
                    bridge_gradient += w_b * np.abs(S[:, u] - S[:, v])

        feats = np.column_stack(
            [
                low_freq_ratio,
                fiedler_proj,
                high_freq_energy,
                dirichlet_energy,
                bottleneck_stress,
                bridge_gradient,
                np.full(n, float(self.min_kappa_), dtype=float),
            ]
        )
        names = [
            "f05_gft_low_freq_energy_ratio",
            "f05_fiedler_mode_projection",
            "f05_gft_high_freq_energy",
            "f05_graph_dirichlet_energy",
            "f05_ricci_bottleneck_stress",
            "f05_ricci_bridge_gradient",
            "f05_min_ollivier_ricci_kappa",
        ]

        self.last_diagnostics_ = {
            "laplacian_eigenvalues": self.eigvals_.copy(),
            "edge_curvatures": list(self.edge_curvatures_),
            "topology_mode": bundle.tier_c.topology_mode,
        }
        return self._format_output(feats, names, bundle, X)
