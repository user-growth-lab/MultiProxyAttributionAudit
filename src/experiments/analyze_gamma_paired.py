"""Frozen paired analysis for the focal controlled benchmark.

The script deliberately keeps the original discovery seeds (0--9) separate
from the unseen confirmation seeds (10--29).  At K=5, m=2 and bias multiplier
1, it produces (i) a complete eleven-method table and (ii) seed-paired
median-minus-comparator bias-leakage contrasts.  Synthetic seed, rather than
evaluation user, is the unit of replication.

The analysis is fail-closed: it rejects incomplete method grids, unexpected
seeds, duplicate rows, configuration drift, non-finite metrics, a mismatched
frozen protocol, and any attempt to replace an existing output.

Example
-------
Run after the confirmation experiment has been generated::

    python3 -m src.experiments.analyze_gamma_paired
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DISCOVERY = (
    PROJECT_ROOT
    / "registered_results"
    / "synthetic_discovery"
    / "gamma_revised_raw.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "reproduced_results" / "paired_analysis"
)
DEFAULT_CONFIRMATION = (
    PROJECT_ROOT
    / "registered_results"
    / "paired_confirmation"
    / "confirmatory_run"
    / "gamma_revised_raw.csv"
)
DEFAULT_PROTOCOL = (
    PROJECT_ROOT
    / "registered_results"
    / "paired_confirmation"
    / "protocol_frozen.json"
)

DISCOVERY_SEEDS = tuple(range(10))
CONFIRMATION_SEEDS = tuple(range(10, 30))
EXPECTED_DGPS = ("additive_probability", "logistic_link_stress")
EXPECTED_METHODS = (
    "single_proxy",
    "mean",
    "scaled_huber",
    "trimmed_mean",
    "winsorized_mean",
    "median_of_means",
    "median",
    "rank1_svd",
    "rpca",
    "label_first_majority",
    "oracle_clean",
)
TABLE_METRICS = (
    "attribution_l1_error",
    "attribution_linf_error",
    "local_normalized_l1",
    "local_normalized_linf",
    "global_normalized_l1",
    "support_precision",
    "support_recall",
    "spearman",
    "rank_correlation",
    "driver_sign_accuracy",
    "bias_leakage",
    "bias_global_attribution",
    "driver_global_attribution",
    "runtime_seconds",
)
PRIMARY_METRIC = "bias_leakage"
SECONDARY_RPCA_METRICS = (
    "bias_global_attribution",
    "driver_global_attribution",
    "attribution_l1_error",
)
FOCAL_K = 5
FOCAL_CONTAMINATED = 2
FOCAL_BIAS_MULTIPLIER = 1.0
CONFIDENCE_LEVEL = 0.95
TIE_TOLERANCE = 1e-12


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the frozen K=5, m=2, lambda=1 median comparison without "
            "pooling discovery and confirmation seeds."
        )
    )
    parser.add_argument("--discovery", default=str(DEFAULT_DISCOVERY))
    parser.add_argument("--confirmation", default=str(DEFAULT_CONFIRMATION))
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace only outputs managed by this analyzer.",
    )
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_environment() -> dict[str, Any]:
    """Record portable interpreter and numerical-library versions."""

    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
    }


def _artifact_safe_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return f"<external-input>/{resolved.name}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            _json_safe(payload),
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path).to_dict(orient="records")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"Expected a JSON list of records in {path}.")
    return payload


def validate_frozen_protocol(protocol: Mapping[str, Any]) -> None:
    expected = {
        "unit_of_replication": "synthetic_seed",
        "discovery_seeds": list(DISCOVERY_SEEDS),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "K": FOCAL_K,
        "contaminated": FOCAL_CONTAMINATED,
        "bias_multiplier": FOCAL_BIAS_MULTIPLIER,
        "primary_metric": PRIMARY_METRIC,
        "contrast": "median_minus_comparator",
        "confidence_level": CONFIDENCE_LEVEL,
        "interval": "paired_two_sided_student_t_across_seeds",
        "pool_cohorts": False,
    }
    registered = protocol.get("analysis", {})
    mismatches = {
        key: {"expected": value, "observed": registered.get(key)}
        for key, value in expected.items()
        if registered.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Frozen protocol does not match the checked analysis: {mismatches}")


def _require_constant(frame: pd.DataFrame, column: str, expected: Any) -> None:
    if column not in frame:
        raise ValueError(f"Missing frozen-configuration column {column!r}.")
    values = frame[column]
    if isinstance(expected, float):
        matches = np.isclose(values.to_numpy(dtype=float), expected, rtol=0.0, atol=1e-12)
    else:
        matches = values.eq(expected).to_numpy(dtype=bool)
    if not bool(np.all(matches)):
        observed = sorted(str(value) for value in values.drop_duplicates().tolist())
        raise ValueError(
            f"Configuration drift in {column!r}: expected {expected!r}, observed {observed}."
        )


def select_focal_rows(
    records: Sequence[Mapping[str, Any]],
    expected_seeds: Sequence[int],
    cohort: str,
) -> pd.DataFrame:
    """Select and validate one cohort at the frozen focal configuration."""

    frame = pd.DataFrame(records)
    required = {
        "seed",
        "dgp",
        "bias_multiplier",
        "K",
        "contaminated",
        "aggregator",
        "n_train",
        "n_eval",
        "d",
        "driver_indices",
        "bias_index",
        "clean",
        "latent_driver_coefficients",
        "base_bias_coefficient",
        "bias_coefficient",
        "probability_epsilon",
        "trim_fraction",
        "rpca_enabled",
        "shapley_reference",
        *TABLE_METRICS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{cohort}: input is missing columns {missing}.")

    focal = frame[
        frame["K"].eq(FOCAL_K)
        & frame["contaminated"].eq(FOCAL_CONTAMINATED)
        & np.isclose(
            frame["bias_multiplier"].to_numpy(dtype=float),
            FOCAL_BIAS_MULTIPLIER,
            rtol=0.0,
            atol=1e-12,
        )
    ].copy()
    if focal.empty:
        raise ValueError(f"{cohort}: no rows match the frozen focal configuration.")

    expected_seed_set = {int(seed) for seed in expected_seeds}
    focal = focal[focal["seed"].isin(expected_seed_set)].copy()
    observed_seeds = {int(seed) for seed in focal["seed"].unique()}
    if observed_seeds != expected_seed_set:
        raise ValueError(
            f"{cohort}: expected seeds {sorted(expected_seed_set)}, "
            f"observed {sorted(observed_seeds)}."
        )

    observed_dgps = set(focal["dgp"].unique())
    if observed_dgps != set(EXPECTED_DGPS):
        raise ValueError(
            f"{cohort}: expected DGPs {list(EXPECTED_DGPS)}, "
            f"observed {sorted(observed_dgps)}."
        )

    duplicate_key = ["seed", "dgp", "aggregator"]
    if focal.duplicated(duplicate_key).any():
        duplicate_rows = focal.loc[focal.duplicated(duplicate_key, keep=False), duplicate_key]
        raise ValueError(
            f"{cohort}: duplicate seed/DGP/method rows: "
            f"{duplicate_rows.to_dict(orient='records')[:5]}."
        )

    expected_methods = set(EXPECTED_METHODS)
    for (seed, dgp), group in focal.groupby(["seed", "dgp"], sort=True):
        methods = set(group["aggregator"].unique())
        if methods != expected_methods:
            raise ValueError(
                f"{cohort}: seed={seed}, dgp={dgp} has method set "
                f"{sorted(methods)}, expected {sorted(expected_methods)}."
            )

    expected_rows = len(expected_seed_set) * len(EXPECTED_DGPS) * len(EXPECTED_METHODS)
    if len(focal) != expected_rows:
        raise ValueError(f"{cohort}: expected {expected_rows} focal rows, observed {len(focal)}.")

    _require_constant(focal, "K", FOCAL_K)
    _require_constant(focal, "contaminated", FOCAL_CONTAMINATED)
    _require_constant(focal, "bias_multiplier", FOCAL_BIAS_MULTIPLIER)
    _require_constant(focal, "n_train", 6000)
    _require_constant(focal, "n_eval", 200)
    _require_constant(focal, "d", 8)
    _require_constant(focal, "driver_indices", "0,1")
    _require_constant(focal, "bias_index", 2)
    _require_constant(focal, "clean", 3)
    _require_constant(focal, "probability_epsilon", 0.01)
    _require_constant(focal, "trim_fraction", 0.20)
    _require_constant(focal, "rpca_enabled", True)
    _require_constant(focal, "shapley_reference", "zero_population_mean")
    dgp_configuration = {
        "additive_probability": {
            "latent_driver_coefficients": "0.12,-0.1",
            "base_bias_coefficient": 0.18,
            "bias_coefficient": 0.18,
        },
        "logistic_link_stress": {
            "latent_driver_coefficients": "1.2,-1",
            "base_bias_coefficient": 1.5,
            "bias_coefficient": 1.5,
        },
    }
    for dgp, expected in dgp_configuration.items():
        dgp_rows = focal[focal["dgp"].eq(dgp)]
        for column, value in expected.items():
            _require_constant(dgp_rows, column, value)

    numeric = focal.loc[:, TABLE_METRICS].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{cohort}: focal metrics contain non-finite values.")
    return focal.sort_values(["dgp", "seed", "aggregator"]).reset_index(drop=True)


def mean_student_t_interval(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 2 or not np.isfinite(array).all():
        raise ValueError("A finite one-dimensional sample of at least two seeds is required.")
    count = int(array.size)
    mean = float(np.mean(array))
    standard_deviation = float(np.std(array, ddof=1))
    standard_error = standard_deviation / math.sqrt(count)
    alpha = 1.0 - CONFIDENCE_LEVEL
    critical = float(stats.t.ppf(1.0 - alpha / 2.0, df=count - 1))
    return {
        "n_seeds": count,
        "mean": mean,
        "std": standard_deviation,
        "se": standard_error,
        "ci95_low": mean - critical * standard_error,
        "ci95_high": mean + critical * standard_error,
    }


def complete_method_table(focal: pd.DataFrame, cohort: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dgp in EXPECTED_DGPS:
        for method in EXPECTED_METHODS:
            group = focal[focal["dgp"].eq(dgp) & focal["aggregator"].eq(method)]
            row: dict[str, Any] = {
                "cohort": cohort,
                "dgp": dgp,
                "aggregator": method,
                "seed_count": int(group["seed"].nunique()),
            }
            for metric in TABLE_METRICS:
                summary = mean_student_t_interval(group[metric].to_numpy(dtype=float))
                for statistic, value in summary.items():
                    if statistic == "n_seeds":
                        continue
                    row[f"{metric}_{statistic}"] = value
            rows.append(row)
    return pd.DataFrame(rows)


def median_pairwise_analysis(
    focal: pd.DataFrame, cohort: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    comparators = [method for method in EXPECTED_METHODS if method != "median"]
    for dgp in EXPECTED_DGPS:
        dgp_frame = focal[focal["dgp"].eq(dgp)]
        pivot = dgp_frame.pivot(
            index="seed", columns="aggregator", values=PRIMARY_METRIC
        ).sort_index()
        median_values = pivot["median"]
        for comparator in comparators:
            comparator_values = pivot[comparator]
            differences = median_values - comparator_values
            for seed, difference in differences.items():
                value = float(difference)
                if value < -TIE_TOLERANCE:
                    outcome = "median_win"
                elif value > TIE_TOLERANCE:
                    outcome = "median_loss"
                else:
                    outcome = "tie"
                seed_rows.append(
                    {
                        "cohort": cohort,
                        "dgp": dgp,
                        "seed": int(seed),
                        "metric": PRIMARY_METRIC,
                        "comparator": comparator,
                        "median_value": float(median_values.loc[seed]),
                        "comparator_value": float(comparator_values.loc[seed]),
                        "median_minus_comparator": value,
                        "outcome_lower_is_better": outcome,
                    }
                )
            summary = mean_student_t_interval(differences.to_numpy(dtype=float))
            summary_rows.append(
                {
                    "cohort": cohort,
                    "dgp": dgp,
                    "metric": PRIMARY_METRIC,
                    "contrast": "median_minus_comparator",
                    "comparator": comparator,
                    **summary,
                    "median_win_count": int(
                        np.sum(differences.to_numpy(dtype=float) < -TIE_TOLERANCE)
                    ),
                    "median_loss_count": int(
                        np.sum(differences.to_numpy(dtype=float) > TIE_TOLERANCE)
                    ),
                    "tie_count": int(
                        np.sum(np.abs(differences.to_numpy(dtype=float)) <= TIE_TOLERANCE)
                    ),
                }
            )
    return pd.DataFrame(seed_rows), pd.DataFrame(summary_rows)


def median_rpca_secondary_analysis(focal: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """Decompose the primary ratio after confirmation as a labeled diagnostic.

    These contrasts were not part of the frozen primary protocol. They report
    median-minus-RPCA paired differences for the absolute bias coordinate, mean
    driver magnitude, and total attribution L1 error so that a leakage-ratio
    difference cannot be mistaken for smaller absolute bias attribution.
    """

    rows: list[dict[str, Any]] = []
    for dgp in EXPECTED_DGPS:
        dgp_frame = focal[focal["dgp"].eq(dgp)]
        for metric in SECONDARY_RPCA_METRICS:
            pivot = dgp_frame.pivot(
                index="seed", columns="aggregator", values=metric
            ).sort_index()
            median_values = pivot["median"].to_numpy(dtype=float)
            rpca_values = pivot["rpca"].to_numpy(dtype=float)
            summary = mean_student_t_interval(median_values - rpca_values)
            rows.append(
                {
                    "cohort": cohort,
                    "status": "post_primary_denominator_diagnostic",
                    "dgp": dgp,
                    "metric": metric,
                    "contrast": "median_minus_rpca",
                    "median_mean": float(np.mean(median_values)),
                    "rpca_mean": float(np.mean(rpca_values)),
                    **summary,
                }
            )
    return pd.DataFrame(rows)


def output_paths(output_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for cohort in ("discovery", "confirmation"):
        paths[f"{cohort}_methods_csv"] = output_dir / f"{cohort}_method_table.csv"
        paths[f"{cohort}_methods_json"] = output_dir / f"{cohort}_method_table.json"
        paths[f"{cohort}_paired_seed_csv"] = (
            output_dir / f"{cohort}_median_pairwise_seed.csv"
        )
        paths[f"{cohort}_paired_seed_json"] = (
            output_dir / f"{cohort}_median_pairwise_seed.json"
        )
        paths[f"{cohort}_paired_summary_csv"] = (
            output_dir / f"{cohort}_median_pairwise_summary.csv"
        )
        paths[f"{cohort}_paired_summary_json"] = (
            output_dir / f"{cohort}_median_pairwise_summary.json"
        )
        paths[f"{cohort}_secondary_csv"] = (
            output_dir / f"{cohort}_median_rpca_secondary.csv"
        )
        paths[f"{cohort}_secondary_json"] = (
            output_dir / f"{cohort}_median_rpca_secondary.json"
        )
    paths["metadata"] = output_dir / "paired_analysis_metadata.json"
    return paths


def require_fresh_outputs(
    paths: Mapping[str, Path], *, overwrite: bool = False
) -> None:
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        if overwrite:
            for path in existing:
                path.unlink()
            return
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            "Refusing to replace an existing paired-analysis result: " + formatted
        )


def _primary_result(summary: pd.DataFrame) -> list[dict[str, Any]]:
    selected = summary[summary["comparator"].eq("rpca")].copy()
    columns = [
        "cohort",
        "dgp",
        "metric",
        "contrast",
        "comparator",
        "n_seeds",
        "mean",
        "ci95_low",
        "ci95_high",
        "median_win_count",
        "median_loss_count",
        "tie_count",
    ]
    return selected.loc[:, columns].to_dict(orient="records")


def run_analysis(
    discovery_path: Path,
    confirmation_path: Path,
    protocol_path: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> dict[str, Path]:
    with protocol_path.open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    if not isinstance(protocol, dict):
        raise ValueError("The frozen protocol must be a JSON object.")
    validate_frozen_protocol(protocol)

    paths = output_paths(output_dir)
    require_fresh_outputs(paths, overwrite=overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)

    cohorts = {
        "discovery": (
            discovery_path,
            DISCOVERY_SEEDS,
            load_records(discovery_path),
        ),
        "confirmation": (
            confirmation_path,
            CONFIRMATION_SEEDS,
            load_records(confirmation_path),
        ),
    }
    cohort_artifacts: dict[str, dict[str, Any]] = {}
    primary_results: dict[str, list[dict[str, Any]]] = {}
    written_nonmetadata: list[Path] = []
    for cohort, (input_path, seeds, records) in cohorts.items():
        focal = select_focal_rows(records, seeds, cohort)
        method_table = complete_method_table(focal, cohort)
        paired_seed, paired_summary = median_pairwise_analysis(focal, cohort)
        secondary = median_rpca_secondary_analysis(focal, cohort)

        expected_method_rows = len(EXPECTED_DGPS) * len(EXPECTED_METHODS)
        expected_pair_seed_rows = (
            len(seeds) * len(EXPECTED_DGPS) * (len(EXPECTED_METHODS) - 1)
        )
        expected_pair_summary_rows = len(EXPECTED_DGPS) * (len(EXPECTED_METHODS) - 1)
        if len(method_table) != expected_method_rows:
            raise AssertionError("The complete eleven-method table is incomplete.")
        if len(paired_seed) != expected_pair_seed_rows:
            raise AssertionError("The seed-paired table is incomplete.")
        if len(paired_summary) != expected_pair_summary_rows:
            raise AssertionError("The paired-summary table is incomplete.")
        if len(secondary) != len(EXPECTED_DGPS) * len(SECONDARY_RPCA_METRICS):
            raise AssertionError("The secondary median--RPCA diagnostic is incomplete.")

        payloads = {
            paths[f"{cohort}_methods_csv"]: method_table,
            paths[f"{cohort}_paired_seed_csv"]: paired_seed,
            paths[f"{cohort}_paired_summary_csv"]: paired_summary,
            paths[f"{cohort}_secondary_csv"]: secondary,
        }
        for path, table in payloads.items():
            table.to_csv(path, index=False)
            written_nonmetadata.append(path)
        json_payloads = {
            paths[f"{cohort}_methods_json"]: method_table.to_dict(orient="records"),
            paths[f"{cohort}_paired_seed_json"]: paired_seed.to_dict(orient="records"),
            paths[f"{cohort}_paired_summary_json"]: paired_summary.to_dict(
                orient="records"
            ),
            paths[f"{cohort}_secondary_json"]: secondary.to_dict(orient="records"),
        }
        for path, payload in json_payloads.items():
            _write_json(path, payload)
            written_nonmetadata.append(path)

        cohort_artifacts[cohort] = {
            "input": _artifact_safe_path(input_path),
            "input_sha256": sha256_file(input_path),
            "expected_seeds": list(seeds),
            "focal_row_count": int(len(focal)),
            "method_table_row_count": int(len(method_table)),
            "paired_seed_row_count": int(len(paired_seed)),
            "paired_summary_row_count": int(len(paired_summary)),
            "secondary_diagnostic_row_count": int(len(secondary)),
        }
        primary_results[cohort] = _primary_result(paired_summary)

    script_path = Path(__file__).resolve()
    runner_path = PROJECT_ROOT / "src" / "experiments" / "run_gamma_revised.py"
    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed_frozen_paired_analysis",
        "cohorts_are_pooled": False,
        "unit_of_replication": "synthetic_seed",
        "evaluation_users_are_not_replicates": True,
        "analysis": protocol["analysis"],
        "expected_dgps": list(EXPECTED_DGPS),
        "expected_methods": list(EXPECTED_METHODS),
        "table_metrics": list(TABLE_METRICS),
        "cohorts": cohort_artifacts,
        "primary_median_minus_rpca": primary_results,
        "secondary_analysis_status": (
            "post-primary denominator diagnostic; not part of the frozen primary contrast"
        ),
        "secondary_median_rpca_metrics": list(SECONDARY_RPCA_METRICS),
        "source_hashes": {
            "frozen_protocol_sha256": sha256_file(protocol_path),
            "analysis_script_sha256": sha256_file(script_path),
            "experiment_runner_sha256": sha256_file(runner_path),
        },
        "output_hashes": {
            path.name: sha256_file(path) for path in sorted(written_nonmetadata)
        },
        "environment": runtime_environment(),
        "interpretation_guardrail": (
            "The focal contrast does not establish method dominance across the full grid."
        ),
    }
    _write_json(paths["metadata"], metadata)
    return paths


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    paths = run_analysis(
        discovery_path=Path(args.discovery).expanduser(),
        confirmation_path=Path(args.confirmation).expanduser(),
        protocol_path=Path(args.protocol).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
        overwrite=bool(args.overwrite),
    )
    print("Frozen paired analysis completed without pooling cohorts.")
    for name, path in paths.items():
        print(f"  {name:32s}: {path}")


if __name__ == "__main__":
    main()
