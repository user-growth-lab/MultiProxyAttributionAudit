from pathlib import Path

import numpy as np
import pytest

from src.experiments.analyze_gamma_paired import (
    DEFAULT_CONFIRMATION,
    DEFAULT_DISCOVERY,
    EXPECTED_DGPS,
    EXPECTED_METHODS,
    SECONDARY_RPCA_METRICS,
    TABLE_METRICS,
    complete_method_table,
    median_pairwise_analysis,
    median_rpca_secondary_analysis,
    output_paths,
    require_fresh_outputs,
    runtime_environment as paired_runtime_environment,
    select_focal_rows,
    validate_frozen_protocol,
)
from src.experiments.run_gamma_revised import (
    _build_parser,
    resolve_config,
    runtime_environment,
)


def _records(seeds=(0, 1)):
    rows = []
    for dgp in EXPECTED_DGPS:
        for seed in seeds:
            for method_index, method in enumerate(EXPECTED_METHODS):
                common = {
                    "seed": seed,
                    "dgp": dgp,
                    "bias_multiplier": 1.0,
                    "K": 5,
                    "contaminated": 2,
                    "aggregator": method,
                    "n_train": 6000,
                    "n_eval": 200,
                    "d": 8,
                    "driver_indices": "0,1",
                    "bias_index": 2,
                    "clean": 3,
                    "latent_driver_coefficients": (
                        "0.12,-0.1" if dgp == "additive_probability" else "1.2,-1"
                    ),
                    "base_bias_coefficient": (
                        0.18 if dgp == "additive_probability" else 1.5
                    ),
                    "bias_coefficient": (
                        0.18 if dgp == "additive_probability" else 1.5
                    ),
                    "probability_epsilon": 0.01,
                    "trim_fraction": 0.20,
                    "rpca_enabled": True,
                    "shapley_reference": "zero_population_mean",
                }
                for metric in TABLE_METRICS:
                    common[metric] = 0.01 * (method_index + 1) + 0.001 * seed
                # Give the primary RPCA contrast a simple exact value.
                if method == "median":
                    common["bias_leakage"] = 0.1 + 0.001 * seed
                elif method == "rpca":
                    common["bias_leakage"] = 0.2 + 0.001 * seed
                rows.append(common)
    return rows


def test_complete_method_and_pairwise_tables_are_seed_paired():
    focal = select_focal_rows(_records(), expected_seeds=(0, 1), cohort="test")
    methods = complete_method_table(focal, cohort="test")
    paired_seed, paired_summary = median_pairwise_analysis(focal, cohort="test")
    secondary = median_rpca_secondary_analysis(focal, cohort="test")

    assert len(methods) == 2 * 11
    assert methods["seed_count"].eq(2).all()
    assert len(paired_seed) == 2 * 2 * 10
    assert len(paired_summary) == 2 * 10
    assert len(secondary) == 2 * len(SECONDARY_RPCA_METRICS)
    assert secondary["status"].eq("post_primary_denominator_diagnostic").all()

    rpca = paired_summary[paired_summary["comparator"].eq("rpca")]
    np.testing.assert_allclose(rpca["mean"], -0.1)
    np.testing.assert_allclose(rpca["ci95_low"], -0.1)
    np.testing.assert_allclose(rpca["ci95_high"], -0.1)
    assert rpca["median_win_count"].eq(2).all()
    assert rpca["median_loss_count"].eq(0).all()


def test_paired_defaults_use_packaged_inputs_and_portable_metadata():
    project_root = DEFAULT_DISCOVERY.parents[2]
    assert DEFAULT_DISCOVERY.relative_to(project_root).as_posix() == (
        "registered_results/synthetic_discovery/gamma_revised_raw.csv"
    )
    assert DEFAULT_CONFIRMATION.relative_to(project_root).as_posix() == (
        "registered_results/paired_confirmation/confirmatory_run/"
        "gamma_revised_raw.csv"
    )
    environment = paired_runtime_environment()
    assert set(environment) == {"python", "packages"}
    assert set(environment["packages"]) == {"numpy", "pandas", "scipy"}


def test_focal_validation_rejects_incomplete_method_grid():
    records = _records()
    records.pop()
    with pytest.raises(ValueError, match="method set"):
        select_focal_rows(records, expected_seeds=(0, 1), cohort="test")


def test_frozen_protocol_rejects_pooling_or_seed_drift():
    protocol = {
        "analysis": {
            "unit_of_replication": "synthetic_seed",
            "discovery_seeds": list(range(10)),
            "confirmation_seeds": list(range(10, 30)),
            "K": 5,
            "contaminated": 2,
            "bias_multiplier": 1.0,
            "primary_metric": "bias_leakage",
            "contrast": "median_minus_comparator",
            "confidence_level": 0.95,
            "interval": "paired_two_sided_student_t_across_seeds",
            "pool_cohorts": False,
        }
    }
    validate_frozen_protocol(protocol)
    protocol["analysis"]["pool_cohorts"] = True
    with pytest.raises(ValueError, match="Frozen protocol"):
        validate_frozen_protocol(protocol)


def test_paired_analysis_never_overwrites_existing_results(tmp_path: Path):
    paths = output_paths(tmp_path)
    next(iter(paths.values())).write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        require_fresh_outputs(paths)


def test_core_runner_can_limit_confirmation_to_one_contamination_count(tmp_path: Path):
    args = _build_parser().parse_args(
        [
            "--profile",
            "paper",
            "--seeds",
            "10,11",
            "--ks",
            "5",
            "--contamination-counts",
            "2",
            "--bias-multipliers",
            "1",
            "--output-dir",
            str(tmp_path),
            "--no-plot",
        ]
    )
    cfg, _ = resolve_config(args)
    assert cfg.seeds == (10, 11)
    assert cfg.ks == (5,)
    assert cfg.contamination_counts == (2,)
    assert cfg.bias_multipliers == (1.0,)


def test_core_runtime_metadata_never_serializes_python_home_path():
    environment = runtime_environment()
    payload = str(environment)
    assert "/" + "Users" + "/" not in payload
    assert set(environment) == {"python", "packages", "threads"}
    assert set(environment["python"]) == {"version", "implementation"}
