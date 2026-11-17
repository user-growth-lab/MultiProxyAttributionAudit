import numpy as np
import pytest

from src.explainers.robust_proxy import (
    aggregate_local_median,
    aggregate_proxy_attributions,
    compatible_target_sign,
    compatible_target_intervals,
    global_importance,
    scale_compatible_target_intervals,
)


def test_strict_majority_recovers_clean_local_attributions():
    clean = np.array([[1.0, -2.0], [0.5, 3.0]])
    values = np.stack([clean, clean, np.full_like(clean, 1_000.0)])
    np.testing.assert_allclose(aggregate_local_median(values), clean)


def test_even_k_strict_majority_recovers_clean_local_attributions():
    clean = np.array([[1.0, -2.0]])
    values = np.stack([clean, clean, clean, np.array([[999.0, -999.0]])])
    np.testing.assert_allclose(aggregate_local_median(values), clean)


def test_clean_proxy_set_may_vary_by_coordinate():
    clean = np.array([[1.0, -2.0, 3.0]])
    values = np.stack(
        [
            np.array([[99.0, -2.0, 3.0]]),
            np.array([[1.0, 99.0, 3.0]]),
            np.array([[1.0, -2.0, 99.0]]),
        ]
    )
    np.testing.assert_allclose(aggregate_local_median(values), clean)


def test_scale_calibration_recovers_common_units():
    clean = np.array([[1.0, -2.0]])
    values = np.stack([clean, 10.0 * clean, np.array([[999.0, 999.0]])])
    calibrated = aggregate_local_median(values, scales=np.array([1.0, 10.0, 1.0]))
    np.testing.assert_allclose(calibrated, clean)


def test_local_aggregation_precedes_global_absolute_value():
    values = np.array([[[1.0]], [[-1.0]], [[0.0]]])
    local, importance = aggregate_proxy_attributions(values)
    np.testing.assert_allclose(local, [[0.0]])
    np.testing.assert_allclose(importance, [0.0])
    assert np.median(np.mean(np.abs(values), axis=1), axis=0)[0] == 1.0


def test_two_proxies_do_not_recover_one_clean_value():
    clean = np.array([[1.0]])
    contaminated = np.array([[9.0]])
    result = aggregate_local_median(np.stack([clean, contaminated]))
    np.testing.assert_allclose(result, [[5.0]])


def test_even_k_half_contamination_is_not_identified():
    clean = np.array([[1.0]])
    contaminated = np.array([[9.0]])
    values = np.stack([clean, clean, contaminated, contaminated])
    result = aggregate_local_median(values)
    np.testing.assert_allclose(result, [[5.0]])


def test_compatibility_set_is_singleton_below_boundary():
    intervals = compatible_target_intervals([1.0, 1.0, 9.0], contamination_budget=1)
    assert intervals == ((1.0, 1.0),)


def test_compatibility_set_can_be_disconnected_at_boundary():
    intervals = compatible_target_intervals(
        [0.0, 0.0, 10.0, 10.0], contamination_budget=2
    )
    assert intervals == ((0.0, 0.0), (10.0, 10.0))


def test_compatible_target_sign_distinguishes_decision_abstention_and_falsification():
    assert compatible_target_sign(np.array([1.0, 1.1, 0.9]), 0, 0.11) == 1
    assert compatible_target_sign(np.array([-1.0, -1.1, -0.9]), 0, 0.11) == -1
    assert compatible_target_sign(np.array([-1.0, 0.0, 1.0]), 2, 0.0) == 0
    assert compatible_target_sign(np.array([-1.0, 0.0, 1.0]), 0, 0.1) is None


def test_compatibility_set_handles_tolerance_and_unrestricted_budget():
    intervals = compatible_target_intervals(
        [0.9, 1.0, 1.1, 9.0, 10.0], contamination_budget=2, clean_tolerance=0.1
    )
    np.testing.assert_allclose(intervals, ((1.0, 1.0),), atol=1e-12)
    assert compatible_target_intervals([1.0, 2.0], 2) == ((-np.inf, np.inf),)


def test_compatibility_set_respects_declared_target_domain():
    assert compatible_target_intervals(
        [2.0, 2.0, 3.0], 1, target_bounds=(-1.0, 1.0)
    ) == ()
    assert compatible_target_intervals(
        [1.0, 2.0], 2, target_bounds=(-1.0, 1.0)
    ) == ((-1.0, 1.0),)
    assert compatible_target_sign(
        np.array([2.0, 2.0, 3.0]), 1, target_bounds=(-1.0, 1.0)
    ) is None


def test_scale_compatible_set_reduces_to_point_calibration():
    intervals = scale_compatible_target_intervals(
        raw_values=np.array([2.0, 4.0, 18.0]),
        scale_bounds=np.array([[2.0, 2.0], [4.0, 4.0], [2.0, 2.0]]),
        contamination_budget=1,
    )
    assert intervals == ((1.0, 1.0),)


def test_scale_compatible_set_propagates_interval_uncertainty():
    intervals = scale_compatible_target_intervals(
        raw_values=np.array([2.0, 3.0, 100.0]),
        scale_bounds=np.array([[1.0, 2.0], [2.0, 3.0], [1.0, 1.0]]),
        contamination_budget=1,
        target_bounds=(-1.0, 1.5),
    )
    assert intervals == ((1.0, 1.5),)


def test_scale_compatible_set_validates_inputs():
    with pytest.raises(ValueError):
        scale_compatible_target_intervals([1.0], [[0.0, 1.0]], 0)
    with pytest.raises(ValueError):
        scale_compatible_target_intervals([1.0], [[2.0, 1.0]], 0)
    with pytest.raises(ValueError):
        scale_compatible_target_intervals([1.0], [[1.0, 2.0]], 2)


def test_scale_compatible_set_handles_negative_touching_empty_and_full_budget():
    assert scale_compatible_target_intervals(
        [-2.0, -1.0, 100.0], [[1.0, 2.0], [0.5, 1.0], [1.0, 1.0]], 1
    ) == ((-2.0, -1.0),)
    assert scale_compatible_target_intervals(
        [1.0, 2.0, 100.0], [[1.0, 1.0], [1.0, 2.0], [1.0, 1.0]], 1
    ) == ((1.0, 1.0),)
    assert scale_compatible_target_intervals(
        [1.0, 3.0], [[1.0, 1.0], [1.0, 1.0]], 0, clean_tolerance=0.1
    ) == ()
    assert scale_compatible_target_intervals(
        [1.0, 3.0], [[1.0, 2.0], [1.0, 2.0]], 2, target_bounds=(-1.0, 1.0)
    ) == ((-1.0, 1.0),)


def test_scale_compatible_diameter_bound_below_boundary():
    rng = np.random.default_rng(20260712)
    for _ in range(100):
        raw = rng.uniform(-2.0, 2.0, size=5)
        lower = rng.uniform(0.5, 1.5, size=5)
        upper = lower + rng.uniform(0.0, 0.8, size=5)
        tolerance = 0.05
        widths = np.sort(
            np.abs(raw) * (1.0 / lower - 1.0 / upper) + 2 * tolerance
        )
        for budget in (0, 1, 2):
            intervals = scale_compatible_target_intervals(
                raw, np.column_stack([lower, upper]), budget, tolerance
            )
            if intervals:
                diameter = intervals[-1][1] - intervals[0][0]
                assert diameter <= widths[2 * budget] + 1e-12


def test_global_importance_and_validation():
    np.testing.assert_allclose(global_importance([[1.0, -2.0], [-3.0, 4.0]]), [2.0, 3.0])
    with pytest.raises(ValueError):
        aggregate_local_median(np.ones((2, 3)))
    with pytest.raises(ValueError):
        aggregate_local_median(np.ones((2, 3, 1)), scales=np.array([1.0, 0.0]))
