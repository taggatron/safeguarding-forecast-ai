"""
Modular 13-Framework Esoteric Mathematical Feature Engineering Suite (`esoteric_math/features/`).
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Type, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from esoteric_math.schema_bridge import AdaptiveSchemaBridge, SchemaBundle
from esoteric_math.features.base import CausalBaseTransformer
from esoteric_math.features.f01_fourier_hilbert import WarpedFourierHilbertTransformer
from esoteric_math.features.f02_wavelets_modwt import CausalMODWTWaveletTransformer
from esoteric_math.features.f03_hawkes_process import HawkesContagionTransformer
from esoteric_math.features.f04_tda_homology import TakensPersistenceTDATransformer
from esoteric_math.features.f05_graph_ricci import GraphSpectralRicciTransformer
from esoteric_math.features.f06_koopman_tropical import KoopmanTropicalTransformer
from esoteric_math.features.f07_rough_signatures import RoughPathSignaturesTransformer
from esoteric_math.features.f08_cellular_sheaves import CellularSheafCohomologyTransformer
from esoteric_math.features.f09_padic_poincare import PadicPoincareTransformer
from esoteric_math.features.f10_optimal_transport import OptimalTransportTransformer
from esoteric_math.features.f11_rmt_marchenko import RMTMarchenkoPasturTransformer
from esoteric_math.features.f12_fisher_rao import FisherRaoInformationGeometryTransformer
from esoteric_math.features.f13_fractional_soc import FractionalSOCTransformer


FEATURE_ENGINE_REGISTRY: Dict[str, Type[CausalBaseTransformer]] = {
    "f01": WarpedFourierHilbertTransformer,
    "f02": CausalMODWTWaveletTransformer,
    "f03": HawkesContagionTransformer,
    "f04": TakensPersistenceTDATransformer,
    "f05": GraphSpectralRicciTransformer,
    "f06": KoopmanTropicalTransformer,
    "f07": RoughPathSignaturesTransformer,
    "f08": CellularSheafCohomologyTransformer,
    "f09": PadicPoincareTransformer,
    "f10": OptimalTransportTransformer,
    "f11": RMTMarchenkoPasturTransformer,
    "f12": FisherRaoInformationGeometryTransformer,
    "f13": FractionalSOCTransformer,
}


class EsotericFeatureSuite(BaseEstimator, TransformerMixin):
    """
    Unified causal orchestrator running a shared `AdaptiveSchemaBridge` once and executing
    any selected subset (default: all 13) of `f01`..`f13` feature transformers.
    """

    def __init__(
        self,
        active_modules: Optional[Sequence[str]] = None,
        causal_mode: Literal["telemetry", "forecast"] = "telemetry",
        train_cutoff: Optional[int] = None,
    ) -> None:
        self.active_modules = tuple(active_modules) if active_modules is not None else tuple(FEATURE_ENGINE_REGISTRY.keys())
        self.causal_mode = causal_mode
        self.train_cutoff = train_cutoff

    def fit(self, X: pd.DataFrame, y: Any = None) -> "EsotericFeatureSuite":
        self.bridge_ = AdaptiveSchemaBridge(
            causal_mode=self.causal_mode,
            train_cutoff=self.train_cutoff,
        )
        self.bridge_.fit(X)
        train_slice = X.iloc[: self.train_cutoff] if self.train_cutoff is not None else X
        train_bundle = self.bridge_.transform(train_slice)

        self.engines_: Dict[str, CausalBaseTransformer] = {}
        for mod_id in self.active_modules:
            cls = FEATURE_ENGINE_REGISTRY[mod_id]
            eng = cls(causal_mode=self.causal_mode, train_cutoff=self.train_cutoff, return_dataframe=True)
            eng.fit(train_bundle)
            self.engines_[mod_id] = eng
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        bundle = self.bridge_.transform(X)
        blocks: List[pd.DataFrame] = []
        self.module_columns_: Dict[str, List[str]] = {}
        self.diagnostics_: Dict[str, Dict[str, Any]] = {}

        for mod_id in self.active_modules:
            eng = self.engines_[mod_id]
            df_block = eng.transform(bundle)
            assert isinstance(df_block, pd.DataFrame)
            if bundle.row_to_step_idx is not None and len(df_block) != len(X):
                df_block = df_block.iloc[bundle.row_to_step_idx].reset_index(drop=True)
            df_block.index = X.index
            blocks.append(df_block)
            self.module_columns_[mod_id] = list(df_block.columns)
            self.diagnostics_[mod_id] = getattr(eng, "last_diagnostics_", {})

        out = pd.concat(blocks, axis=1)
        self.feature_names_out_ = list(out.columns)
        self.last_bundle_ = bundle
        return out

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(getattr(self, "feature_names_out_", []), dtype=object)


__all__ = [
    "CausalBaseTransformer",
    "WarpedFourierHilbertTransformer",
    "CausalMODWTWaveletTransformer",
    "HawkesContagionTransformer",
    "TakensPersistenceTDATransformer",
    "GraphSpectralRicciTransformer",
    "KoopmanTropicalTransformer",
    "RoughPathSignaturesTransformer",
    "CellularSheafCohomologyTransformer",
    "PadicPoincareTransformer",
    "OptimalTransportTransformer",
    "RMTMarchenkoPasturTransformer",
    "FisherRaoInformationGeometryTransformer",
    "FractionalSOCTransformer",
    "FEATURE_ENGINE_REGISTRY",
    "EsotericFeatureSuite",
]
