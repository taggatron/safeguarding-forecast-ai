"""
Base causal sklearn-compatible transformer infrastructure (`esoteric_math/features/base.py`).
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from esoteric_math.schema_bridge import AdaptiveSchemaBridge, SchemaBundle


class CausalBaseTransformer(BaseEstimator, TransformerMixin):
    """
    Base class for all 13 esoteric mathematical transformers.
    Guarantees:
      - Full `sklearn` BaseEstimator + TransformerMixin contract (`fit`, `transform`, `fit_transform`, `get_feature_names_out`)
      - Accepts either a raw `pd.DataFrame` or a pre-extracted `SchemaBundle`
      - Enforces strict causal non-leakage (index t reads only {0, ..., t})
      - Stores structured causal diagnostic payloads in `self.last_diagnostics_` for telemetry views
    """

    module_id: str = "f00"
    module_name: str = "Base Causal Transformer"

    def __init__(
        self,
        causal_mode: Literal["telemetry", "forecast"] = "telemetry",
        train_cutoff: Optional[int] = None,
        return_dataframe: bool = True,
    ) -> None:
        self.causal_mode = causal_mode
        self.train_cutoff = train_cutoff
        self.return_dataframe = return_dataframe

    def _ensure_bundle_fit(self, X: Union[pd.DataFrame, SchemaBundle]) -> SchemaBundle:
        if isinstance(X, SchemaBundle):
            self.bridge_ = None
            return X
        self.bridge_ = AdaptiveSchemaBridge(
            causal_mode=self.causal_mode,
            train_cutoff=self.train_cutoff,
        )
        self.bridge_.fit(X)
        train_slice = X.iloc[: self.train_cutoff] if self.train_cutoff is not None else X
        return self.bridge_.transform(train_slice)

    def _ensure_bundle_transform(self, X: Union[pd.DataFrame, SchemaBundle]) -> SchemaBundle:
        if isinstance(X, SchemaBundle):
            return X
        if getattr(self, "bridge_", None) is None:
            self.bridge_ = AdaptiveSchemaBridge(
                causal_mode=self.causal_mode,
                train_cutoff=self.train_cutoff,
            ).fit(X)
        return self.bridge_.transform(X)

    def _format_output(
        self,
        feats: np.ndarray,
        col_names: List[str],
        bundle: SchemaBundle,
        X_orig: Union[pd.DataFrame, SchemaBundle],
    ) -> Union[pd.DataFrame, np.ndarray]:
        clean_feats = np.nan_to_num(feats, nan=0.0, posinf=50.0, neginf=-50.0)
        if bundle.row_to_step_idx is not None and isinstance(X_orig, pd.DataFrame) and len(X_orig) != len(clean_feats):
            clean_feats = clean_feats[bundle.row_to_step_idx]

        self.feature_names_out_ = list(col_names)
        if self.return_dataframe:
            idx = X_orig.index if isinstance(X_orig, pd.DataFrame) and len(X_orig) == len(clean_feats) else None
            return pd.DataFrame(clean_feats, columns=col_names, index=idx)
        return clean_feats

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(getattr(self, "feature_names_out_", []), dtype=object)
