"""
Audio quality evaluation metrics module.
"""

import importlib
from enum import Enum
from typing import TYPE_CHECKING, Type

from .base import (
    BaseMetricCalculator,
    MetricCalculationError,
    ModelConfig,
    ModelLoadError,
)

if TYPE_CHECKING:
    from .ffe import FFECalculator
    from .mcd import MCDCalculator
    from .sim import SIMCalculator
    from .utmos import UTMOSCalculator
    from .wer import WERCalculator


class MetricType(Enum):
    """Supported metric types."""

    UTMOS = "utmos"
    WER = "wer"
    SIM = "sim"
    FFE = "ffe"
    MCD = "mcd"


_CALCULATOR_MAPPING = {
    MetricType.UTMOS: ("utmos", "UTMOSCalculator"),
    MetricType.WER: ("wer", "WERCalculator"),
    MetricType.SIM: ("sim", "SIMCalculator"),
    MetricType.FFE: ("ffe", "FFECalculator"),
    MetricType.MCD: ("mcd", "MCDCalculator"),
}


def _get_calculator_class(metric_type: MetricType) -> Type[BaseMetricCalculator]:
    module_name, class_name = _CALCULATOR_MAPPING[metric_type]
    module = importlib.import_module(f".{module_name}", package="spk_incon.metrics")
    return getattr(module, class_name)


def create_calculator(
    metric_type: MetricType, config: ModelConfig
) -> BaseMetricCalculator:
    """
    Factory function to create metric calculators.

    Args:
        metric_type: Type of metric calculator
        config: Model configuration

    Returns:
        Metric calculator instance

    Raises:
        ValueError: If metric type is not supported
    """
    try:
        calculator_class = _get_calculator_class(metric_type)
        return calculator_class(config)
    except (KeyError, ImportError, AttributeError) as e:
        available_metrics = get_available_metrics()
        raise ValueError(
            f"Failed to create calculator for {metric_type}. "
            f"Available metrics: {available_metrics}"
        ) from e


def get_available_metrics() -> list[MetricType]:
    """Get list of available metric types."""
    return list(MetricType)


__all__ = [
    "BaseMetricCalculator",
    "ModelConfig",
    "MetricCalculationError",
    "ModelLoadError",
    "MetricType",
    "create_calculator",
    "get_available_metrics",
]
