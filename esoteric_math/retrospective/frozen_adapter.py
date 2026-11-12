"""
Read-Only Frozen Baseline Wrapper & Cryptographic SHA-256 Verifier
(`esoteric_math/retrospective/frozen_adapter.py`).

Guarantees 100% Zero Core Model Mutation:
  - Never calls `.fit()` on or mutates any attribute of the wrapped baseline estimator.
  - Records canonical IEEE-754 SHA-256 digests of both the fitted model parameters/trees
    and out-of-sample `ModelPredictionBundle` outputs (`point_forecasts` + all quantile curves).
  - Exposes read-only ZINB structural zero probabilities `pi_zero(t)` vs. active conditional
    count expectations `mu_count(t)` so post-hoc offsets lock `pi_zero(t)` bit-for-bit.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Dict, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from app.evaluation.kaggle_metrics import EVAL_QUANTILES
from app.models.safeguarding_benchmark import ModelPredictionBundle


class FrozenBaselineMutationError(RuntimeError):
    """Raised if a wrapped baseline estimator or its predictions drift by even a single bit."""


def _update_hash_with_array(hasher: "hashlib._Hash", arr: Any) -> None:
    a = np.ascontiguousarray(np.asarray(arr, dtype="<f8"))
    hasher.update(str(a.shape).encode("utf-8"))
    hasher.update(a.tobytes())


def compute_model_state_sha256(estimator: Any) -> str:
    """
    Compute a deterministic cryptographic SHA-256 digest of a fitted baseline estimator's
    internal weights, dispersion scalars, tree structures, and quantile calibration state.
    """
    h = hashlib.sha256()
    cls_name = f"{type(estimator).__module__}.{type(estimator).__qualname__}"
    h.update(cls_name.encode("utf-8"))

    # 1. NB2GovernanceGLM state
    if hasattr(estimator, "fit_res_") and getattr(estimator, "fit_res_", None) is not None:
        h.update(b"|nb2_params|")
        _update_hash_with_array(h, estimator.fit_res_.params)
        _update_hash_with_array(h, estimator.fit_res_.bse)
        _update_hash_with_array(h, [float(getattr(estimator, "alpha_", 0.0))])
        h.update(json.dumps(list(getattr(estimator, "used_cols_", []))).encode("utf-8"))

    # 2. ZINBSparseRegimeModel state
    if hasattr(estimator, "infl_params_") and hasattr(estimator, "count_params_"):
        h.update(b"|zinb_params|")
        _update_hash_with_array(h, getattr(estimator, "infl_params_", []))
        _update_hash_with_array(h, getattr(estimator, "count_params_", []))
        _update_hash_with_array(h, [float(getattr(estimator, "alpha_", 0.0))])
        h.update(
            json.dumps(
                {
                    "used_cols": list(getattr(estimator, "used_cols_", [])),
                    "used_infl_cols": list(getattr(estimator, "used_infl_cols_", [])),
                    "cohort_log_enrol": dict(sorted(getattr(estimator, "cohort_baseline_log_enrol_", {}).items())),
                },
                sort_keys=True,
            ).encode("utf-8")
        )

    # 3. LightGBM (TweedieLightGBMModel / LegacyMLBaselineModel) state
    lgbm_mod = getattr(estimator, "model_", None)
    if lgbm_mod is not None and hasattr(lgbm_mod, "booster_"):
        h.update(b"|lgbm_booster|")
        h.update(lgbm_mod.booster_.model_to_string().encode("utf-8"))
        for scalar_attr in ("best_power_", "log_base_rate_", "alpha_equiv_"):
            if hasattr(estimator, scalar_attr):
                h.update(scalar_attr.encode("utf-8"))
                _update_hash_with_array(h, [float(getattr(estimator, scalar_attr))])
        for map_attr in ("conformal_log_quantiles_", "resid_quantiles_"):
            if hasattr(estimator, map_attr):
                q_dict = getattr(estimator, map_attr) or {}
                for k in sorted(q_dict.keys()):
                    _update_hash_with_array(h, [float(k), float(q_dict[k])])

    # 4. Generic sklearn / custom estimator fallback
    if hasattr(estimator, "coef_"):
        h.update(b"|sklearn_coef|")
        _update_hash_with_array(h, getattr(estimator, "coef_"))
    if hasattr(estimator, "intercept_"):
        _update_hash_with_array(h, getattr(estimator, "intercept_"))

    return h.hexdigest()


def compute_prediction_bundle_sha256(bundle: ModelPredictionBundle) -> str:
    """
    Compute canonical IEEE-754 SHA-256 digest of a `ModelPredictionBundle`
    (`point_forecasts` + all quantile vectors).
    """
    h = hashlib.sha256()
    h.update(str(bundle.model_id).encode("utf-8"))
    h.update(b"|point|")
    _update_hash_with_array(h, bundle.point_forecasts)
    for tau in sorted(bundle.quantiles.keys()):
        h.update(f"|q:{tau:.6f}|".encode("utf-8"))
        _update_hash_with_array(h, bundle.quantiles[tau])
    return h.hexdigest()


def clone_prediction_bundle(bundle: ModelPredictionBundle) -> ModelPredictionBundle:
    """Return a deep defensive copy of a `ModelPredictionBundle`."""
    return ModelPredictionBundle(
        model_id=str(bundle.model_id),
        model_name=str(bundle.model_name),
        point_forecasts=np.asarray(bundle.point_forecasts, dtype=float).copy(),
        quantiles={float(tau): np.asarray(q, dtype=float).copy() for tau, q in bundle.quantiles.items()},
        aic=bundle.aic,
        bic=bundle.bic,
        log_likelihood=bundle.log_likelihood,
        alpha_dispersion=bundle.alpha_dispersion,
        diagnostics=copy.deepcopy(bundle.diagnostics),
    )


class FrozenBaselineAdapter:
    """
    Strictly read-only wrapper around an already-fitted baseline estimator.
    Blocks `.fit()` mutation, isolates predictions via deep copies, and verifies SHA-256
    digests before and after post-hoc residual stacking.
    """

    def __init__(self, fitted_estimator: Any, label: Optional[str] = None) -> None:
        self._original_ref = fitted_estimator
        self._frozen_copy = copy.deepcopy(fitted_estimator)
        self.model_id: str = str(getattr(fitted_estimator, "model_id", type(fitted_estimator).__name__))
        self.model_name: str = label or str(getattr(fitted_estimator, "model_name", self.model_id))
        self.initial_model_state_sha256_: str = compute_model_state_sha256(self._original_ref)
        snapshot_hash = compute_model_state_sha256(self._frozen_copy)
        if snapshot_hash != self.initial_model_state_sha256_:
            raise FrozenBaselineMutationError(
                f"Frozen copy digest mismatch on initialization for {self.model_id}."
            )
        self.prediction_hashes_: Dict[str, str] = {}
        self._reference_inputs: Dict[str, Tuple[pd.DataFrame, Tuple[float, ...]]] = {}

    @property
    def model_family(self) -> str:
        if hasattr(self._frozen_copy, "infl_params_") and hasattr(self._frozen_copy, "count_params_"):
            return "glm_zinb"
        if hasattr(self._frozen_copy, "fit_res_"):
            return "glm_nb2"
        if hasattr(self._frozen_copy, "best_power_") and hasattr(self._frozen_copy, "conformal_log_quantiles_"):
            return "gbdt_tweedie"
        if hasattr(self._frozen_copy, "resid_quantiles_"):
            return "gbdt_l2"
        return "generic"

    @property
    def alpha_dispersion(self) -> float:
        if hasattr(self._frozen_copy, "alpha_"):
            return float(max(1e-4, getattr(self._frozen_copy, "alpha_")))
        if hasattr(self._frozen_copy, "alpha_equiv_"):
            return float(max(1e-4, getattr(self._frozen_copy, "alpha_equiv_")))
        return 0.18

    def fit(self, *args: Any, **kwargs: Any) -> None:
        """Explicitly forbidden: frozen baseline models are 100% immutable."""
        raise FrozenBaselineMutationError(
            f"Illegal attempt to call .fit() on FrozenBaselineAdapter('{self.model_id}'). "
            "Core baseline estimators are strictly immutable."
        )

    def predict(
        self,
        df: pd.DataFrame,
        quantiles: Sequence[float] = EVAL_QUANTILES,
        tag: Optional[str] = None,
    ) -> ModelPredictionBundle:
        """
        Evaluate the frozen baseline read-only on `df`, record SHA-256 prediction digest
        when `tag` is supplied, and verify zero mutation of the original estimator.
        """
        self.verify_model_immutability()
        raw_out = self._frozen_copy.predict(df, quantiles=quantiles)
        if not isinstance(raw_out, ModelPredictionBundle):
            pts = np.maximum(0.0, np.asarray(raw_out, dtype=float))
            raw_out = ModelPredictionBundle(
                model_id=self.model_id,
                model_name=self.model_name,
                point_forecasts=pts,
                quantiles={float(tau): pts.copy() for tau in quantiles},
                alpha_dispersion=self.alpha_dispersion,
            )

        bundle = clone_prediction_bundle(raw_out)
        digest = compute_prediction_bundle_sha256(bundle)
        if tag is not None:
            if tag in self.prediction_hashes_ and self.prediction_hashes_[tag] != digest:
                raise FrozenBaselineMutationError(
                    f"Frozen baseline prediction SHA-256 drifted for tag='{tag}' on {self.model_id}: "
                    f"expected {self.prediction_hashes_[tag]}, got {digest}"
                )
            self.prediction_hashes_[tag] = digest
            if tag not in self._reference_inputs:
                self._reference_inputs[tag] = (df.copy(), tuple(float(q) for q in quantiles))

        self.verify_model_immutability()
        return bundle

    def extract_zinb_structural_components(
        self,
        df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        For frozen `ZINBSparseRegimeModel`, extract read-only:
          - `pi_zero`: frozen structural zero-inflation probability `hat{pi}_base(t)`
          - `mu_count`: frozen active conditional count expectation `hat{mu}_{count,base}(t)`
          - `alpha`: frozen NB2 overdispersion scalar
        without mutating a single attribute.
        """
        if self.model_family != "glm_zinb":
            raise ValueError(f"Estimator {self.model_id} is not a ZINB model.")
        X_count, X_infl, offset = self._frozen_copy._build_matrices(df, is_train=False)
        logit_pi = np.clip(X_infl @ self._frozen_copy.infl_params_, -15.0, 15.0)
        pi_zero = 1.0 / (1.0 + np.exp(-logit_pi))
        mu_count = np.exp(np.clip(offset + X_count @ self._frozen_copy.count_params_, -15.0, 15.0))
        return pi_zero.copy(), mu_count.copy(), float(self._frozen_copy.alpha_)

    def verify_model_immutability(self) -> str:
        """Verify both the original estimator reference and internal frozen copy match SHA-256 bit-for-bit."""
        live_orig_sha = compute_model_state_sha256(self._original_ref)
        live_copy_sha = compute_model_state_sha256(self._frozen_copy)
        if live_orig_sha != self.initial_model_state_sha256_:
            raise FrozenBaselineMutationError(
                f"Original baseline estimator '{self.model_id}' state mutated! "
                f"Expected SHA-256 {self.initial_model_state_sha256_}, got {live_orig_sha}."
            )
        if live_copy_sha != self.initial_model_state_sha256_:
            raise FrozenBaselineMutationError(
                f"Frozen copy '{self.model_id}' state mutated! "
                f"Expected SHA-256 {self.initial_model_state_sha256_}, got {live_copy_sha}."
            )
        return live_orig_sha

    def verify_all_checksums(self) -> Dict[str, Any]:
        """Re-evaluate all tagged prediction inputs and verify 100% bit-for-bit SHA-256 reproducibility."""
        model_sha = self.verify_model_immutability()
        verified_tags: Dict[str, str] = {}
        for tag, (ref_df, ref_q) in self._reference_inputs.items():
            re_bundle = self._frozen_copy.predict(ref_df, quantiles=ref_q)
            re_sha = compute_prediction_bundle_sha256(re_bundle)
            expected_sha = self.prediction_hashes_[tag]
            if re_sha != expected_sha:
                raise FrozenBaselineMutationError(
                    f"Prediction SHA-256 verification failed for '{self.model_id}' [tag={tag}]: "
                    f"{re_sha} != {expected_sha}"
                )
            verified_tags[tag] = re_sha
        return {
            "model_id": self.model_id,
            "model_state_sha256": model_sha,
            "verified_prediction_hashes": verified_tags,
            "immutable_verified": True,
        }

