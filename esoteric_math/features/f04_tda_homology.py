"""
Module 04: Topological Data Analysis — Causal Takens Embedding & Vietoris-Rips Persistence
(`esoteric_math/features/f04_tda_homology.py`).

Computes strictly causal Takens delay-coordinate point clouds on {0, ..., t} plus multivariate
Tier B state clouds, extracting exact H0 (MST Union-Find) and H1 (Vietoris-Rips cycle minimax
filtration) persistence diagrams, Persistent Entropy, max persistence lifetime, Betti integrals,
and first Persistence Landscape L2-norm.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from esoteric_math.features.base import CausalBaseTransformer
from esoteric_math.schema_bridge import SchemaBundle


def compute_vr_persistence_h0_h1(point_cloud: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Vietoris-Rips H0 and H1 persistence intervals [(birth, death), ...] on a small
    point cloud X of shape (M, D) using pure NumPy/SciPy:
      - H0: Exact Vietoris-Rips 0-dimensional persistence via Kruskal Minimum Spanning Tree
            (all components born at 0.0, dying at MST merge edge weights).
      - H1: 1-cycle persistence from non-MST edges (born at edge length b_e when cycle closes,
            dying at minimax 2-simplex triangle fill diameter max(d(u, w), d(v, w)) across third vertices w).
    """
    m = len(point_cloud)
    if m < 2:
        return np.zeros((0, 2), dtype=float), np.zeros((0, 2), dtype=float)

    dists = squareform(pdist(point_cloud, metric="euclidean"))
    if np.max(dists) < 1e-12:
        return np.zeros((0, 2), dtype=float), np.zeros((0, 2), dtype=float)

    # Collect upper-triangle edges sorted by length
    u_idx, v_idx = np.triu_indices(m, k=1)
    weights = dists[u_idx, v_idx]
    order = np.argsort(weights, kind="mergesort")

    parent = np.arange(m, dtype=int)

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    h0_intervals: List[Tuple[float, float]] = []
    non_mst_edges: List[Tuple[int, int, float]] = []

    for e_i in order:
        u, v, w = int(u_idx[e_i]), int(v_idx[e_i]), float(weights[e_i])
        ru, rv = _find(u), _find(v)
        if ru != rv:
            parent[ru] = rv
            if w > 1e-12:
                h0_intervals.append((0.0, w))
        else:
            non_mst_edges.append((u, v, w))

    # H1 1-cycles: check candidate chordless cycles where birth < triangle fill cap
    h1_intervals: List[Tuple[float, float]] = []
    # Adjacency in MST + earlier edges to measure shortest path / triangle cap
    for u, v, birth in non_mst_edges[: min(40, len(non_mst_edges))]:
        # Minimum scale at which a 2-simplex [u, v, w] caps edge (u, v) directly, or 2-step bridge
        other_max = np.maximum(dists[u, :], dists[v, :])
        other_max[u] = np.inf
        other_max[v] = np.inf
        death = float(np.min(other_max))
        if np.isfinite(death) and death > birth + 1e-9:
            h1_intervals.append((birth, death))

    h0_arr = np.array(h0_intervals, dtype=float) if h0_intervals else np.zeros((0, 2), dtype=float)
    h1_arr = np.array(h1_intervals, dtype=float) if h1_intervals else np.zeros((0, 2), dtype=float)
    return h0_arr, h1_arr


def _persistence_summaries(intervals: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Extract (persistent_entropy, max_lifetime, betti_integral, landscape_1_l2_norm)
    from persistence diagram `intervals` of shape (K, 2).
    """
    if len(intervals) == 0:
        return 0.0, 0.0, 0.0, 0.0

    lifetimes = np.maximum(0.0, intervals[:, 1] - intervals[:, 0])
    pos = lifetimes[lifetimes > 1e-12]
    if len(pos) == 0:
        return 0.0, 0.0, 0.0, 0.0

    L_sum = float(np.sum(pos))
    probs = pos / L_sum
    entropy = float(-np.sum(probs * np.log(probs)))
    max_life = float(np.max(pos))
    betti_int = L_sum

    # Exact evaluation of first Persistence Landscape lambda_1(s) = max_i max(0, min(s - b_i, d_i - s)) on grid
    b_min = float(np.min(intervals[:, 0]))
    d_max = float(np.max(intervals[:, 1]))
    if d_max <= b_min + 1e-12:
        return entropy, max_life, betti_int, 0.0

    grid = np.linspace(b_min, d_max, 32)
    tents = np.maximum(0.0, np.minimum(grid[None, :] - intervals[:, 0:1], intervals[:, 1:2] - grid[None, :]))
    lambda_1 = np.max(tents, axis=0)
    l2_norm = float(np.sqrt(np.trapezoid(lambda_1**2, grid)))
    return entropy, max_life, betti_int, l2_norm


class TakensPersistenceTDATransformer(CausalBaseTransformer):
    """
    Module 04: Causal Takens' Delay Embedding + H0/H1 Vietoris-Rips Persistence Transformer.
    """

    module_id = "f04"
    module_name = "Takens Delay Embedding & Persistent Homology"

    def __init__(
        self,
        embedding_dim: int = 3,
        delay: int = 2,
        cloud_window: int = 16,
        causal_mode: Literal["telemetry", "forecast"] = "telemetry",
        train_cutoff: Optional[int] = None,
        return_dataframe: bool = True,
    ) -> None:
        super().__init__(
            causal_mode=causal_mode,
            train_cutoff=train_cutoff,
            return_dataframe=return_dataframe,
        )
        self.embedding_dim = embedding_dim
        self.delay = delay
        self.cloud_window = cloud_window

    def fit(self, X: Union[pd.DataFrame, SchemaBundle], y: Any = None) -> "TakensPersistenceTDATransformer":
        bundle = self._ensure_bundle_fit(X)
        r = bundle.tier_a.y_rate_per_1000
        self.rate_mean_ = float(np.mean(r)) if len(r) > 0 else 0.0
        self.rate_std_ = max(1e-4, float(np.std(r))) if len(r) > 0 else 1.0
        self.is_fitted_ = True
        return self

    def transform(self, X: Union[pd.DataFrame, SchemaBundle]) -> Union[pd.DataFrame, np.ndarray]:
        if not getattr(self, "is_fitted_", False):
            self.fit(X)
        bundle = self._ensure_bundle_transform(X)
        n = bundle.n_steps
        r_norm = (bundle.tier_a.y_rate_per_1000 - self.rate_mean_) / self.rate_std_
        cov_mat = bundle.tier_b.covariate_matrix

        m_dim = max(2, int(self.embedding_dim))
        tau = max(1, int(self.delay))
        w_pts = max(5, int(self.cloud_window))

        feats = np.zeros((n, 9), dtype=float)
        names = [
            "f04_tda_h0_entropy",
            "f04_tda_h0_max_lifetime",
            "f04_tda_h0_betti_integral",
            "f04_tda_h0_landscape_l2",
            "f04_tda_h1_entropy",
            "f04_tda_h1_max_lifetime",
            "f04_tda_h1_betti_integral",
            "f04_tda_h1_landscape_l2",
            "f04_tda_multivar_h1_max_lifetime",
        ]

        for t in range(n):
            # Build strictly causal Takens delay vectors at s in [max(0, t - w_pts + 1) .. t]
            s_indices = range(max(0, t - w_pts + 1), t + 1)
            delay_cloud = []
            for s in s_indices:
                vec = [float(r_norm[max(0, s - k * tau)]) for k in range(m_dim)]
                delay_cloud.append(vec)
            cloud_arr = np.asarray(delay_cloud, dtype=float)

            h0_d, h1_d = compute_vr_persistence_h0_h1(cloud_arr)
            e0, m0, b0, l0 = _persistence_summaries(h0_d)
            e1, m1, b1, l1 = _persistence_summaries(h1_d)

            # Multivariate Tier B state trajectory cloud up to t
            mv_slice = cov_mat[max(0, t - w_pts + 1) : t + 1, : min(4, cov_mat.shape[1])]
            _, h1_mv = compute_vr_persistence_h0_h1(mv_slice)
            _, m1_mv, _, _ = _persistence_summaries(h1_mv)

            feats[t] = [e0, m0, b0, l0, e1, m1, b1, l1, m1_mv]

        self.last_diagnostics_ = {"feature_names": names}
        return self._format_output(feats, names, bundle, X)
