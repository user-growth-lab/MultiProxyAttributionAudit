from pathlib import Path

import numpy as np
import pytest

from src.experiments.run_sensitivity_revised import (
    RunConfig,
    require_explicit_overwrite,
    run_experiment,
    scale_factors,
    summarize_rows,
    validate_results,
)
from src.explainers.robust_proxy import aggregate_local_median


def _small_config(output_dir: Path) -> RunConfig:
    return RunConfig(
        profile="quick",
        seeds=(0,),
        n_eval=12,
        n_anchor=12,
        d=4,
        driver_indices=(0, 1),
        driver_coefficients=(0.12, -0.10),
        bias_index=2,
        K=5,
        contaminated=2,
        contamination_strength=1.0,
        baseline_clean_tolerance=0.01,
        clean_bias_fractions=(0.0, 0.35),
        scale_severities=(0.0, 0.5),
        clean_tolerances=(0.0, 0.01),
        anchor_noise_std=0.02,
        output_dir=str(output_dir),
        overwrite=False,
    )


def test_positive_scales_preserve_sign_but_not_raw_magnitude():
    truth = np.array([[[1.0, -2.0]], [[1.0, -2.0]], [[1.0, -2.0]],
                      [[1.0, -2.0]], [[1.0, -2.0]]])
    scales = scale_factors(0.5, 5)
    observed = truth * scales[:, None, None]
    raw = aggregate_local_median(observed)
    normalized = aggregate_local_median(observed, scales=scales)
    np.testing.assert_allclose(raw, np.median(scales) * truth[0])
    np.testing.assert_array_equal(np.sign(raw), np.sign(truth[0]))
    np.testing.assert_allclose(normalized, truth[0])


def test_small_suite_covers_all_families_and_validates(tmp_path):
    cfg = _small_config(tmp_path)
    rows, _ = run_experiment(cfg)
    summaries = summarize_rows(rows)
    validate_results(cfg, rows, summaries)
    assert {row["family"] for row in rows} == {
        "contamination_direction",
        "support_overlap",
        "positive_scale",
        "approximate_cleanliness",
        "compatibility_misspecification",
    }
    misspecified = [
        row for row in rows if row["family"] == "compatibility_misspecification"
    ]
    assert misspecified
    assert max(row["compatibility_empty_rate"] for row in misspecified) > 0.0


def test_overwrite_guard_requires_explicit_flag(tmp_path):
    cfg = _small_config(tmp_path)
    (tmp_path / "gamma_sensitivity_revised_raw.csv").write_text("existing")
    with pytest.raises(FileExistsError):
        require_explicit_overwrite(cfg)
