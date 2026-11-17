"""Core explainer modules for robust multi-proxy attribution."""

from .robust_proxy import (
    aggregate_local_median,
    aggregate_proxy_attributions,
    compatible_target_sign,
    compatible_target_intervals,
    global_importance,
    scale_compatible_target_intervals,
)

__all__ = (
    "aggregate_local_median",
    "aggregate_proxy_attributions",
    "compatible_target_sign",
    "compatible_target_intervals",
    "global_importance",
    "scale_compatible_target_intervals",
)
