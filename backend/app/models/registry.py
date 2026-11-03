"""Central registry for all forecasting models in Safeguarding Forecast AI."""
from typing import Dict, List
from app.models.base import AmazonChronos2AdapterStub, BaseForecaster
from app.models.lightgbm_forecaster import LightGBMMLForecaster
from app.models.statistical import SeasonalNaiveForecaster, StatisticalCountForecaster
from app.models.timesfm_forecaster import TimesFM3ForecasterAdapter

MODEL_REGISTRY: Dict[str, BaseForecaster] = {
    "seasonal_naive": SeasonalNaiveForecaster(),
    "statistical_count": StatisticalCountForecaster(),
    "lightgbm_mlforecast": LightGBMMLForecaster(),
    "timesfm_3": TimesFM3ForecasterAdapter(),
}

EXTENSIBLE_ADAPTERS: List[BaseForecaster] = [
    AmazonChronos2AdapterStub(),
]


def get_forecaster(model_id: str) -> BaseForecaster:
    if model_id not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model_id '{model_id}'. Available models: {', '.join(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_id]

