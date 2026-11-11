"""
Module 07: Rough Path Signatures, Log-Signatures & Antisymmetric Levy Area
(`esoteric_math/features/f07_rough_signatures.py`).

Implements exact piecewise-linear Path Signatures up to depth 3 via Chen's Identity
(strictly invariant under monotone time reparameterizations), antisymmetric Levy Area
lead-lag matrices 1/2 (S^{(i,j)} - S^{(j,i)}), and compressed depth-3 Log-Signatures
via the free Lie algebra tensor logarithm.
"""
from __future__ import annotations

from typing import Any, List, Literal, Optional, Tuple, Union
import numpy as np
import pandas as pd

from esoteric_math.features.base import CausalBaseTransformer
from esoteric_math.schema_bridge import SchemaBundle


def compute_path_signature_depth3(path: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute exact Depth-1, Depth-2, and Depth-3 Path Signature tensors (S1, S2, S3)
    for a piecewise-linear path `path` of shape (M, D) via Chen's Identity:
        S(X_{0, t}) = S(X_{0, s}) (x) exp_tensor(X_t - X_s).
    Invariant to any monotone reparameterization / segment subdivision to machine precision.
    """
    pts = np.asarray(path, dtype=float)
    if pts.ndim == 1:
        pts = pts[:, None]
    m_len, d_dim = pts.shape

    s1 = np.zeros(d_dim, dtype=float)
    s2 = np.zeros((d_dim, d_dim), dtype=float)
    s3 = np.zeros((d_dim, d_dim, d_dim), dtype=float)

    if m_len < 2:
        return s1, s2, s3

    diffs = np.diff(pts, axis=0)
    for delta in diffs:
        e1 = delta
        e2 = 0.5 * np.outer(delta, delta)
        e3 = (1.0 / 6.0) * np.einsum("i,j,k->ijk", delta, delta, delta)

        # Associative Chen tensor product up to degree 3
        s3 = s3 + e3 + np.einsum("ij,k->ijk", s2, e1) + np.einsum("i,jk->ijk", s1, e2)
        s2 = s2 + e2 + np.outer(s1, e1)
        s1 = s1 + e1

    return s1, s2, s3


def compute_levy_area_matrix(s2: np.ndarray) -> np.ndarray:
    """Antisymmetric Levy Area lead-lag matrix A_{i,j} = 0.5 * (S^{(i,j)} - S^{(j,i)})."""
    return 0.5 * (s2 - s2.T)


def compute_log_signature_depth3(path: np.ndarray) -> np.ndarray:
    """
    Compute compressed Depth-3 Log-Signature coordinates via the tensor logarithm:
        log_tensor(1 + X) = X - 1/2 X^{(x)2} + 1/3 X^{(x)3}
    extracting:
      - Depth 1: L^{(1)}_i (d coordinates)
      - Depth 2: Strict upper-triangular Levy areas L^{(2)}_{i < j} (d*(d-1)/2 coordinates)
      - Depth 3: Lyndon-word Lie brackets [e_i, [e_j, e_k]] components from L^{(3)}
    """
    s1, s2, s3 = compute_path_signature_depth3(path)
    d_dim = len(s1)

    l1 = s1
    l2 = 0.5 * (s2 - s2.T)
    l3_tensor = (
        s3
        - 0.5 * (np.einsum("i,jk->ijk", s1, s2) + np.einsum("ij,k->ijk", s2, s1))
        + (1.0 / 3.0) * np.einsum("i,j,k->ijk", s1, s1, s1)
    )

    coords: List[float] = list(l1.tolist())
    # Upper-triangle Levy areas (i < j)
    for i in range(d_dim):
        for j in range(i + 1, d_dim):
            coords.append(float(l2[i, j]))

    # Independent Lyndon words of length 3: (i, j, k) with i < j and i < k (or i == j < k, i < j == k)
    for i in range(d_dim):
        for j in range(i + 1, d_dim):
            coords.append(float(l3_tensor[i, i, j]))
            coords.append(float(l3_tensor[i, j, j]))
            for k in range(i + 1, d_dim):
                if k != j:
                    coords.append(float(l3_tensor[i, j, k]))

    return np.asarray(coords, dtype=float)


class RoughPathSignaturesTransformer(CausalBaseTransformer):
    """
    Module 07: Depth-3 Rough Path Signatures, Levy Area Lead-Lag & Log-Signatures.
    """

    module_id = "f07"
    module_name = "Depth-3 Rough Path Signatures & Levy Area"

    def __init__(
        self,
        path_window: int = 12,
        n_channels: int = 3,
        causal_mode: Literal["telemetry", "forecast"] = "telemetry",
        train_cutoff: Optional[int] = None,
        return_dataframe: bool = True,
    ) -> None:
        super().__init__(
            causal_mode=causal_mode,
            train_cutoff=train_cutoff,
            return_dataframe=return_dataframe,
        )
        self.path_window = path_window
        self.n_channels = n_channels

    def _build_channels(self, bundle: SchemaBundle) -> np.ndarray:
        r_z = (bundle.tier_a.y_rate_per_1000 - self.rate_mean_) / self.rate_std_
        k_cov = max(1, int(self.n_channels) - 1)
        cov = bundle.tier_b.covariate_matrix[:, : min(k_cov, bundle.tier_b.covariate_matrix.shape[1])]
        mat = np.column_stack([r_z, cov])
        # Pad with zeros if fewer than n_channels exist
        if mat.shape[1] < self.n_channels:
            pad = np.zeros((len(mat), self.n_channels - mat.shape[1]), dtype=float)
            mat = np.column_stack([mat, pad])
        return np.clip(mat[:, : self.n_channels], -12.0, 12.0)

    def fit(self, X: Union[pd.DataFrame, SchemaBundle], y: Any = None) -> "RoughPathSignaturesTransformer":
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
        channels = self._build_channels(bundle)
        w_len = max(3, int(self.path_window))

        sample_logsig = compute_log_signature_depth3(np.zeros((2, self.n_channels)))
        n_logsig = len(sample_logsig)

        names: List[str] = [f"f07_logsig_{k}" for k in range(n_logsig)]
        names.extend(["f07_sig_depth2_frobenius", "f07_sig_depth3_frobenius", "f07_levy_area_max_abs"])

        feats = np.zeros((n, len(names)), dtype=float)
        for t in range(n):
            sub_path = channels[max(0, t - w_len + 1) : t + 1]
            _, s2, s3 = compute_path_signature_depth3(sub_path)
            levy = compute_levy_area_matrix(s2)
            logsig = compute_log_signature_depth3(sub_path)

            feats[t, :n_logsig] = logsig
            feats[t, n_logsig] = float(np.linalg.norm(s2))
            feats[t, n_logsig + 1] = float(np.cbrt(np.sum(s3**2)))
            feats[t, n_logsig + 2] = float(np.max(np.abs(levy))) if levy.size > 0 else 0.0

        self.last_diagnostics_ = {"n_channels": self.n_channels, "logsig_dim": n_logsig}
        return self._format_output(feats, names, bundle, X)
