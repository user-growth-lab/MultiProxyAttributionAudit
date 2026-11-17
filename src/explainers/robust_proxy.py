"""Robust aggregation of local attributions from multiple proxy models.

The order of operations is intentional: aggregate signed *local* attribution
vectors across proxies first, then summarize their global magnitudes.  Taking
absolute values before aggregation discards direction and targets a different
estimand.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _validate_attributions(attributions: np.ndarray) -> np.ndarray:
    values = np.asarray(attributions, dtype=float)
    if values.ndim != 3:
        raise ValueError(
            "attributions must have shape (n_proxies, n_instances, n_features)"
        )
    if values.shape[0] < 1 or values.shape[1] < 1 or values.shape[2] < 1:
        raise ValueError("all attribution dimensions must be non-empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("attributions must contain only finite values")
    return values


def aggregate_local_median(
    attributions: np.ndarray,
    scales: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return the coordinate-wise median of signed local attributions.

    Parameters
    ----------
    attributions:
        Tensor with shape ``(K, n, d)`` for ``K`` proxy models.
    scales:
        Optional positive vector with shape ``(K,)``. Proxy ``k`` is divided
        by ``scales[k]`` before aggregation. This makes cross-proxy magnitude
        claims explicit; without calibration, the estimator is still valid for
        the supplied attribution scale but not invariant to proxy rescaling.

    Notes
    -----
    Robust exact recovery requires a strict clean majority in every coordinate.
    With an even number of proxies, NumPy's median averages the two central
    values; in particular, one clean and one arbitrary proxy (``K=2``) does not
    identify the clean attribution.
    """

    values = _validate_attributions(attributions)
    if scales is not None:
        scale = np.asarray(scales, dtype=float)
        if scale.shape != (values.shape[0],):
            raise ValueError("scales must have shape (n_proxies,)")
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("scales must be finite and strictly positive")
        values = values / scale[:, None, None]
    return np.median(values, axis=0)


def global_importance(local_attributions: np.ndarray) -> np.ndarray:
    """Summarize local attributions as mean absolute importance per feature."""

    local = np.asarray(local_attributions, dtype=float)
    if local.ndim != 2:
        raise ValueError("local_attributions must have shape (n_instances, n_features)")
    if local.shape[0] < 1 or local.shape[1] < 1:
        raise ValueError("both local attribution dimensions must be non-empty")
    if not np.all(np.isfinite(local)):
        raise ValueError("local_attributions must contain only finite values")
    return np.mean(np.abs(local), axis=0)


def compatible_target_intervals(
    values: np.ndarray,
    contamination_budget: int,
    clean_tolerance: float = 0.0,
    target_bounds: Optional[Tuple[float, float]] = None,
) -> Tuple[Tuple[float, float], ...]:
    """Return targets compatible with a stated contamination/error budget.

    A target ``t`` is compatible when at least ``K - contamination_budget``
    proxy values fall in ``[t-clean_tolerance, t+clean_tolerance]``.  The result
    is represented exactly as a tuple of disjoint closed intervals.  It may be
    disconnected at or beyond the 50% contamination boundary. When
    ``target_bounds=(L, U)`` is supplied, the set is intersected with the known
    target domain ``[L, U]``; when every proxy may be contaminated, the result
    is that declared domain (or the whole real line when no bounds are given).
    """

    proxy_values = np.asarray(values, dtype=float)
    if proxy_values.ndim != 1 or proxy_values.size < 1:
        raise ValueError("values must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(proxy_values)):
        raise ValueError("values must contain only finite numbers")
    if not isinstance(contamination_budget, (int, np.integer)):
        raise ValueError("contamination_budget must be an integer")
    budget = int(contamination_budget)
    if budget < 0 or budget > proxy_values.size:
        raise ValueError("contamination_budget must lie in [0, n_proxies]")
    tolerance = float(clean_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("clean_tolerance must be finite and non-negative")

    bounds: Optional[Tuple[float, float]] = None
    if target_bounds is not None:
        if len(target_bounds) != 2:
            raise ValueError("target_bounds must contain exactly (lower, upper)")
        lower, upper = (float(target_bounds[0]), float(target_bounds[1]))
        if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
            raise ValueError("target_bounds must be finite with lower <= upper")
        bounds = (lower, upper)

    def restrict(
        intervals: Tuple[Tuple[float, float], ...],
    ) -> Tuple[Tuple[float, float], ...]:
        if bounds is None:
            return intervals
        lower, upper = bounds
        clipped = []
        for start, end in intervals:
            clipped_start = max(start, lower)
            clipped_end = min(end, upper)
            if clipped_start <= clipped_end:
                clipped.append((clipped_start, clipped_end))
        return tuple(clipped)

    required = proxy_values.size - budget
    if required == 0:
        return restrict(((-np.inf, np.inf),))

    # Each observation contributes a closed feasible interval for t. Sweep all
    # endpoints, processing starts before ends so point intersections are kept.
    events: dict[float, list[int]] = {}
    for value in proxy_values:
        events.setdefault(float(value - tolerance), [0, 0])[0] += 1
        events.setdefault(float(value + tolerance), [0, 0])[1] += 1

    coverage = 0
    interval_start: Optional[float] = None
    compatible: list[Tuple[float, float]] = []
    for location in sorted(events):
        starts, ends = events[location]
        coverage_at_endpoint = coverage + starts
        if interval_start is None and coverage_at_endpoint >= required:
            interval_start = location
        coverage_after_endpoint = coverage_at_endpoint - ends
        if interval_start is not None and coverage_after_endpoint < required:
            compatible.append((interval_start, location))
            interval_start = None
        coverage = coverage_after_endpoint
    return restrict(tuple(compatible))


def compatible_target_sign(
    values: np.ndarray,
    contamination_budget: int,
    clean_tolerance: float = 0.0,
    target_bounds: Optional[Tuple[float, float]] = None,
) -> Optional[int]:
    """Return the sign shared by every compatible target, or abstain.

    Returns ``1`` when the full compatible set is strictly positive, ``-1``
    when it is strictly negative, ``0`` when compatible targets include zero
    or span both signs, and ``None`` when the declared budget is incompatible
    with the observations. A nonempty set is a sensitivity statement for the
    supplied budget, not a certificate that the budget is valid.
    """

    intervals = compatible_target_intervals(
        values,
        contamination_budget=contamination_budget,
        clean_tolerance=clean_tolerance,
        target_bounds=target_bounds,
    )
    if not intervals:
        return None
    if intervals[0][0] > 0:
        return 1
    if intervals[-1][1] < 0:
        return -1
    return 0


def scale_compatible_target_intervals(
    raw_values: np.ndarray,
    scale_bounds: np.ndarray,
    contamination_budget: int,
    clean_tolerance: float = 0.0,
    target_bounds: Optional[Tuple[float, float]] = None,
) -> Tuple[Tuple[float, float], ...]:
    """Return targets compatible with externally supplied scale intervals.

    For one fixed instance/coordinate, ``scale_bounds[k] = (lower_k, upper_k)``
    must contain proxy ``k``'s
    positive measurement scale.  A target is compatible when at least
    ``K-contamination_budget`` proxies admit some scale inside their interval
    whose normalized attribution is within ``clean_tolerance`` of that target.
    The result is the exact union of closed intervals satisfying this joint
    scale/contamination/error declaration; nonemptiness does not validate the
    declaration. Because proxy scales are global, jointly enforcing one scale
    vector across several instance/coordinates can only shrink this pointwise
    set.
    """

    values = np.asarray(raw_values, dtype=float)
    intervals = np.asarray(scale_bounds, dtype=float)
    if values.ndim != 1 or values.size < 1:
        raise ValueError("raw_values must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("raw_values must contain only finite numbers")
    if intervals.shape != (values.size, 2):
        raise ValueError("scale_bounds must have shape (n_proxies, 2)")
    if (
        not np.all(np.isfinite(intervals))
        or np.any(intervals[:, 0] <= 0)
        or np.any(intervals[:, 1] < intervals[:, 0])
    ):
        raise ValueError("scale bounds must be finite, positive, and ordered")

    normalized_endpoints = np.stack(
        [values / intervals[:, 0], values / intervals[:, 1]], axis=1
    )
    tolerance = float(clean_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("clean_tolerance must be finite and non-negative")
    feasible = np.column_stack(
        [
            np.min(normalized_endpoints, axis=1) - tolerance,
            np.max(normalized_endpoints, axis=1) + tolerance,
        ]
    )

    if not isinstance(contamination_budget, (int, np.integer)):
        raise ValueError("contamination_budget must be an integer")
    budget = int(contamination_budget)
    if budget < 0 or budget > values.size:
        raise ValueError("contamination_budget must lie in [0, n_proxies]")

    bounds: Optional[Tuple[float, float]] = None
    if target_bounds is not None:
        if len(target_bounds) != 2:
            raise ValueError("target_bounds must contain exactly (lower, upper)")
        lower, upper = (float(target_bounds[0]), float(target_bounds[1]))
        if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
            raise ValueError("target_bounds must be finite with lower <= upper")
        bounds = (lower, upper)

    required = values.size - budget
    if required == 0:
        return ((-np.inf, np.inf),) if bounds is None else (bounds,)

    events: dict[float, list[int]] = {}
    for start, end in feasible:
        events.setdefault(float(start), [0, 0])[0] += 1
        events.setdefault(float(end), [0, 0])[1] += 1

    coverage = 0
    interval_start: Optional[float] = None
    compatible: list[Tuple[float, float]] = []
    for location in sorted(events):
        starts, ends = events[location]
        coverage_at_endpoint = coverage + starts
        if interval_start is None and coverage_at_endpoint >= required:
            interval_start = location
        coverage_after_endpoint = coverage_at_endpoint - ends
        if interval_start is not None and coverage_after_endpoint < required:
            compatible.append((interval_start, location))
            interval_start = None
        coverage = coverage_after_endpoint

    if bounds is None:
        return tuple(compatible)
    lower, upper = bounds
    return tuple(
        (max(start, lower), min(end, upper))
        for start, end in compatible
        if max(start, lower) <= min(end, upper)
    )


def aggregate_proxy_attributions(
    attributions: np.ndarray,
    scales: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(local_consensus, global_importance)`` for proxy attributions."""

    local = aggregate_local_median(attributions, scales=scales)
    return local, global_importance(local)
