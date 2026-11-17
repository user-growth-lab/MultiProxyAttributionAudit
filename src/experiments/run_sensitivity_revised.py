"""Reproducible attribution-tensor sensitivity suite.

This attribution-level runner complements :mod:`run_gamma_revised` without
fitting or refitting response models.  It starts from a shared local-attribution
truth tensor for every seed and isolates four assumptions one at a time:

* same-sign versus mixed-sign contamination;
* contamination support disjoint from versus overlapping a true driver;
* positive proxy-scale perturbations, with raw, known-scale, and independent
  anchor-calibrated medians;
* approximately clean proxies with bounded nonzero local-attribution error.

In every case, signed local vectors are aggregated proxy-wise for each
instance before global importance is computed.  Raw seed rows, Student-t
summaries, a resolved-configuration snapshot, and runtime metadata are saved.

Examples
--------
Quick profile::

    python3 -m src.experiments.run_sensitivity_revised --profile quick

Paper profile in a separate output directory::

    python3 -m src.experiments.run_sensitivity_revised --profile paper \
        --output-dir reproduced_results/sensitivity
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy
import sklearn
import yaml
from scipy import stats

from src.explainers.robust_proxy import (
    aggregate_local_median,
    compatible_target_sign,
    compatible_target_intervals,
    global_importance,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "gamma.yaml"

BUILTIN_PROFILES: dict[str, dict[str, Any]] = {
    "quick": {
        "seeds": [0, 1, 2],
        "n_eval": 128,
        "n_anchor": 64,
        "scale_severities": [0.0, 0.5],
        "clean_tolerances": [0.0, 0.01, 0.04],
    },
    "paper": {
        "seeds": list(range(10)),
        "n_eval": 512,
        "n_anchor": 256,
        "scale_severities": [0.0, 0.125, 0.25, 0.5],
        "clean_tolerances": [0.0, 0.0025, 0.005, 0.01, 0.02, 0.04],
    },
}

GENERAL_METRICS = [
    "local_normalized_l1",
    "local_linf_error",
    "global_relative_l1",
    "magnitude_ratio",
    "support_precision",
    "support_recall",
    "driver_sign_accuracy",
    "non_driver_leakage",
    "aggregation_seconds",
]

OPTIONAL_METRICS = [
    "scale_estimation_relative_mae",
    "compatibility_coverage",
    "compatibility_nonempty_rate",
    "compatibility_empty_rate",
    "compatibility_diameter_mean",
    "compatibility_diameter_max",
    "compatibility_component_count_mean",
    "compatibility_diameter_bound_satisfied",
    "compatibility_direction_decision_rate",
    "compatibility_direction_correct_rate",
    "compatibility_false_direction_rate",
    "compatibility_abstention_rate",
    "median_error_bound_coverage",
    "median_max_error_over_epsilon",
]


@dataclass(frozen=True)
class RunConfig:
    """Fully resolved sensitivity-suite configuration."""

    profile: str
    seeds: tuple[int, ...]
    n_eval: int
    n_anchor: int
    d: int
    driver_indices: tuple[int, ...]
    driver_coefficients: tuple[float, ...]
    bias_index: int
    K: int
    contaminated: int
    contamination_strength: float
    baseline_clean_tolerance: float
    clean_bias_fractions: tuple[float, ...]
    scale_severities: tuple[float, ...]
    clean_tolerances: tuple[float, ...]
    anchor_noise_std: float
    output_dir: str
    overwrite: bool


def _parse_csv(value: str | Sequence[Any], cast: type) -> tuple[Any, ...]:
    fields = [item.strip() for item in value.split(",")] if isinstance(value, str) else value
    return tuple(cast(item) for item in fields if str(item).strip())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--profile", choices=sorted(BUILTIN_PROFILES), default="quick")
    parser.add_argument("--seeds", help="Comma-separated seeds overriding the profile.")
    parser.add_argument("--n-eval", type=int)
    parser.add_argument("--n-anchor", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--contaminated", type=int)
    parser.add_argument("--scale-severities", help="Comma-separated nonnegative values.")
    parser.add_argument("--clean-tolerances", help="Comma-separated epsilon values.")
    parser.add_argument(
        "--output-dir", default="reproduced_results/sensitivity"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace existing sensitivity-suite outputs.",
    )
    return parser


def _artifact_safe_path(path: Path) -> str:
    """Serialize paths without exposing an author's absolute workspace."""

    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return f"<external-output>/{resolved.name}"


def resolve_config(args: argparse.Namespace) -> tuple[RunConfig, dict[str, Any]]:
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        yaml_snapshot = yaml.safe_load(handle) or {}
    profile = dict(BUILTIN_PROFILES[args.profile])
    sensitivity = yaml_snapshot.get("sensitivity", {}) or {}
    profile.update((sensitivity.get("profiles", {}) or {}).get(args.profile, {}) or {})
    experiment = yaml_snapshot.get("experiment", {}) or {}
    seeds = _parse_csv(args.seeds, int) if args.seeds else tuple(profile["seeds"])
    scale_severities = (
        _parse_csv(args.scale_severities, float)
        if args.scale_severities
        else tuple(profile["scale_severities"])
    )
    tolerances = (
        _parse_csv(args.clean_tolerances, float)
        if args.clean_tolerances
        else tuple(profile["clean_tolerances"])
    )
    output_path = Path(args.output_dir).expanduser()
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    cfg = RunConfig(
        profile=args.profile,
        seeds=seeds,
        n_eval=int(args.n_eval or profile["n_eval"]),
        n_anchor=int(args.n_anchor or profile["n_anchor"]),
        d=int(experiment.get("d", 8)),
        driver_indices=tuple(int(v) for v in experiment.get("driver_indices", [0, 1])),
        driver_coefficients=tuple(
            float(v)
            for v in experiment.get("additive_driver_coefficients", [0.12, -0.10])
        ),
        bias_index=int(experiment.get("bias_index", 2)),
        K=int(args.k if args.k is not None else sensitivity.get("K", 5)),
        contaminated=int(
            args.contaminated
            if args.contaminated is not None
            else sensitivity.get("contaminated", 2)
        ),
        contamination_strength=float(
            sensitivity.get("contamination_strength", 1.0)
        ),
        baseline_clean_tolerance=float(
            sensitivity.get("baseline_clean_tolerance", 0.01)
        ),
        clean_bias_fractions=tuple(
            float(value)
            for value in sensitivity.get("clean_bias_fractions", [0.0, 0.35])
        ),
        scale_severities=scale_severities,
        clean_tolerances=tolerances,
        anchor_noise_std=float(sensitivity.get("anchor_noise_std", 0.02)),
        output_dir=str(output_path),
        overwrite=bool(args.overwrite),
    )
    validate_config(cfg)
    return cfg, yaml_snapshot


def validate_config(cfg: RunConfig) -> None:
    if not cfg.seeds:
        raise ValueError("At least one seed is required.")
    if cfg.n_eval < 2 or cfg.n_anchor < 2:
        raise ValueError("n_eval and n_anchor must both be at least 2.")
    if cfg.K < 3 or not 0 <= cfg.contaminated < cfg.K:
        raise ValueError("Require K >= 3 and contaminated in [0, K).")
    if 2 * cfg.contaminated >= cfg.K:
        raise ValueError("This suite is registered below the strict-majority boundary.")
    if len(cfg.driver_indices) != len(cfg.driver_coefficients):
        raise ValueError("driver coefficients must match driver indices.")
    if min((*cfg.driver_indices, cfg.bias_index)) < 0 or max(
        (*cfg.driver_indices, cfg.bias_index)
    ) >= cfg.d:
        raise ValueError("driver and bias indices must fall inside [0, d).")
    if cfg.bias_index in cfg.driver_indices:
        raise ValueError("bias_index must be disjoint from the registered driver set.")
    if any(value < 0 for value in (*cfg.scale_severities, *cfg.clean_tolerances)):
        raise ValueError("scale severities and clean tolerances must be nonnegative.")
    if max(cfg.scale_severities, default=0.0) >= 1.25:
        raise ValueError("scale severity must stay below 1.25 so all scales are positive.")


def _rng(seed: int, *parts: Any) -> np.random.Generator:
    payload = "|".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    derived = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(derived)


def coefficient_vector(cfg: RunConfig) -> np.ndarray:
    beta = np.zeros(cfg.d, dtype=float)
    beta[np.asarray(cfg.driver_indices, dtype=int)] = np.asarray(
        cfg.driver_coefficients, dtype=float
    )
    return beta


def local_truth(x: np.ndarray, cfg: RunConfig) -> np.ndarray:
    """Linear zero-reference Shapley tensor used solely as controlled truth."""

    return x * coefficient_vector(cfg)[None, :]


def scale_factors(severity: float, K: int) -> np.ndarray:
    """Return deterministic positive, asymmetric scale perturbations.

    For K=5 the factors are ``1 + severity * [-.8, -.4, .5, 1, 1.6]``;
    hence the raw median preserves support/sign but has scale ``1+.5*severity``.
    Other odd K values use linearly interpolated quantiles of the same template.
    """

    template = np.array([-0.8, -0.4, 0.5, 1.0, 1.6], dtype=float)
    quantiles = np.linspace(0.0, 1.0, K)
    offsets = np.interp(quantiles, np.linspace(0.0, 1.0, template.size), template)
    values = 1.0 + float(severity) * offsets
    if np.any(values <= 0):
        raise ValueError("The requested scale severity produced a nonpositive scale.")
    return values


def estimate_scales_from_independent_anchor(
    scales: np.ndarray, cfg: RunConfig, seed: int, severity: float
) -> np.ndarray:
    """Estimate positive slopes from an independent, externally anchored tensor."""

    rng = _rng(seed, "scale-anchor", severity)
    anchor_x = rng.uniform(-1.0, 1.0, size=(cfg.n_anchor, cfg.d))
    anchor_truth = local_truth(anchor_x, cfg)
    active = np.zeros_like(anchor_truth, dtype=bool)
    active[:, np.asarray(cfg.driver_indices, dtype=int)] = True
    denominator = float(np.sum(anchor_truth[active] ** 2))
    estimates = []
    for proxy_id, scale in enumerate(scales):
        noise = _rng(seed, "anchor-noise", severity, proxy_id).normal(
            0.0, cfg.anchor_noise_std, size=anchor_truth.shape
        )
        observed = scale * anchor_truth + noise
        slope = float(np.sum(observed[active] * anchor_truth[active]) / denominator)
        estimates.append(max(slope, 1e-8))
    return np.asarray(estimates, dtype=float)


def evaluate_estimate(
    estimate: np.ndarray, truth: np.ndarray, cfg: RunConfig, runtime: float
) -> dict[str, float]:
    error = estimate - truth
    truth_l1 = float(np.sum(np.abs(truth)))
    truth_global = global_importance(truth)
    estimate_global = global_importance(estimate)
    global_denominator = float(np.sum(truth_global))
    threshold = max(1e-12, 0.05 * float(np.max(truth_global)))
    true_support = truth_global > threshold
    estimated_support = estimate_global > threshold
    true_positives = int(np.sum(true_support & estimated_support))
    precision = true_positives / max(1, int(np.sum(estimated_support)))
    recall = true_positives / max(1, int(np.sum(true_support)))
    active = np.zeros_like(truth, dtype=bool)
    active[:, np.asarray(cfg.driver_indices, dtype=int)] = True
    sign_mask = active & (np.abs(truth) > 1e-12)
    sign_accuracy = float(np.mean(np.sign(estimate[sign_mask]) == np.sign(truth[sign_mask])))
    non_driver = np.ones(cfg.d, dtype=bool)
    non_driver[np.asarray(cfg.driver_indices, dtype=int)] = False
    driver_level = float(np.mean(estimate_global[np.asarray(cfg.driver_indices)]))
    leakage = float(np.max(estimate_global[non_driver]) / max(driver_level, 1e-12))
    return {
        "local_normalized_l1": float(np.sum(np.abs(error)) / max(truth_l1, 1e-12)),
        "local_linf_error": float(np.max(np.abs(error))),
        "global_relative_l1": float(
            np.sum(np.abs(estimate_global - truth_global)) / max(global_denominator, 1e-12)
        ),
        "magnitude_ratio": float(np.sum(estimate_global) / max(global_denominator, 1e-12)),
        "support_precision": float(precision),
        "support_recall": float(recall),
        "driver_sign_accuracy": sign_accuracy,
        "non_driver_leakage": leakage,
        "aggregation_seconds": float(runtime),
    }


def compatibility_diagnostics(
    tensor: np.ndarray,
    truth: np.ndarray,
    estimate: np.ndarray,
    contamination_budget: int,
    epsilon: float,
    target_bounds: tuple[float, float] = (-1.0, 1.0),
) -> dict[str, float]:
    covered: list[bool] = []
    nonempty: list[bool] = []
    diameters: list[float] = []
    components: list[int] = []
    direction_decisions: list[bool] = []
    direction_correct: list[bool] = []
    direction_false: list[bool] = []
    direction_abstain: list[bool] = []
    for row in range(truth.shape[0]):
        for feature in range(truth.shape[1]):
            intervals = compatible_target_intervals(
                tensor[:, row, feature], contamination_budget, epsilon,
                target_bounds=target_bounds,
            )
            target = float(truth[row, feature])
            is_nonempty = bool(intervals)
            nonempty.append(is_nonempty)
            covered.append(
                any(low - 1e-12 <= target <= high + 1e-12 for low, high in intervals)
            )
            if is_nonempty:
                diameters.append(float(intervals[-1][1] - intervals[0][0]))
            components.append(len(intervals))
            if abs(target) > 1e-12:
                decision = compatible_target_sign(
                    tensor[:, row, feature], contamination_budget, epsilon,
                    target_bounds=target_bounds,
                )
                decided = decision in {-1, 1}
                correct = bool(decided and decision == int(np.sign(target)))
                direction_decisions.append(decided)
                direction_correct.append(correct)
                direction_false.append(bool(decided and not correct))
                direction_abstain.append(not decided)
    absolute_error = np.abs(estimate - truth)
    if epsilon == 0.0:
        normalized_max_error = 0.0 if float(np.max(absolute_error)) <= 1e-12 else 1e12
    else:
        normalized_max_error = float(np.max(absolute_error) / epsilon)
    diameter_max = float(np.max(diameters)) if diameters else 0.0
    diameter_mean = float(np.mean(diameters)) if diameters else 0.0
    return {
        "compatibility_coverage": float(np.mean(covered)),
        "compatibility_nonempty_rate": float(np.mean(nonempty)),
        "compatibility_empty_rate": float(1.0 - np.mean(nonempty)),
        "compatibility_diameter_mean": diameter_mean,
        "compatibility_diameter_max": diameter_max,
        "compatibility_component_count_mean": float(np.mean(components)),
        "compatibility_diameter_bound_satisfied": float(
            diameter_max <= 2.0 * epsilon + 1e-10
        ),
        "compatibility_direction_decision_rate": float(np.mean(direction_decisions)),
        "compatibility_direction_correct_rate": float(np.mean(direction_correct)),
        "compatibility_false_direction_rate": float(np.mean(direction_false)),
        "compatibility_abstention_rate": float(np.mean(direction_abstain)),
        "median_error_bound_coverage": float(
            np.mean(absolute_error <= epsilon + 1e-12)
        ),
        "median_max_error_over_epsilon": normalized_max_error,
    }


def _aggregate(tensor: np.ndarray, estimator: str, scales: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    if estimator in {"local_median", "raw_median"}:
        estimate = aggregate_local_median(tensor)
    elif estimator == "local_mean":
        estimate = np.mean(tensor, axis=0)
    elif estimator in {
        "known_scale_normalized_median",
        "independent_anchor_normalized_median",
    }:
        if scales is None:
            raise ValueError(f"{estimator} requires positive scales.")
        estimate = aggregate_local_median(tensor, scales=scales)
    else:
        raise ValueError(f"Unknown estimator: {estimator}")
    return estimate, time.perf_counter() - start


def _base_row(
    cfg: RunConfig,
    seed: int,
    family: str,
    setting: str,
    estimator: str,
    polluted_ids: np.ndarray,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "profile": cfg.profile,
        "family": family,
        "setting": setting,
        "estimator": estimator,
        "setting_parameters": json.dumps(parameters, sort_keys=True, separators=(",", ":")),
        "K": cfg.K,
        "contaminated": cfg.contaminated if family != "positive_scale" else 0,
        "strict_clean_majority": True,
        "n_eval": cfg.n_eval,
        "n_anchor": cfg.n_anchor,
        "d": cfg.d,
        "polluted_proxy_ids": ",".join(str(int(v)) for v in polluted_ids),
    }


def _add_estimator_row(
    rows: list[dict[str, Any]],
    cfg: RunConfig,
    seed: int,
    family: str,
    setting: str,
    estimator: str,
    polluted_ids: np.ndarray,
    parameters: Mapping[str, Any],
    tensor: np.ndarray,
    truth: np.ndarray,
    scales: np.ndarray | None = None,
    scale_error: float | None = None,
    compatibility_epsilon: float | None = None,
    compatibility_budget: int | None = None,
) -> None:
    estimate, runtime = _aggregate(tensor, estimator, scales=scales)
    row = _base_row(cfg, seed, family, setting, estimator, polluted_ids, parameters)
    row.update(evaluate_estimate(estimate, truth, cfg, runtime))
    row["scale_estimation_relative_mae"] = scale_error
    for metric in OPTIONAL_METRICS:
        if metric != "scale_estimation_relative_mae":
            row[metric] = None
    if compatibility_epsilon is not None:
        row.update(
            compatibility_diagnostics(
                tensor,
                truth,
                estimate,
                cfg.contaminated
                if compatibility_budget is None
                else compatibility_budget,
                compatibility_epsilon,
            )
        )
    rows.append(row)


def run_experiment(cfg: RunConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    start = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for seed in cfg.seeds:
        x = _rng(seed, "shared-evaluation-design").uniform(
            -1.0, 1.0, size=(cfg.n_eval, cfg.d)
        )
        truth = local_truth(x, cfg)
        polluted_ids = np.sort(
            _rng(seed, "polluted-identities").choice(
                cfg.K, size=cfg.contaminated, replace=False
            )
        )
        clean_mask = np.ones(cfg.K, dtype=bool)
        clean_mask[polluted_ids] = False
        baseline_tensor = np.repeat(truth[None, :, :], cfg.K, axis=0)
        for proxy_id in range(cfg.K):
            baseline_tensor[proxy_id] += _rng(seed, "baseline-clean-error", proxy_id).uniform(
                -cfg.baseline_clean_tolerance,
                cfg.baseline_clean_tolerance,
                size=truth.shape,
            )

        # 1. Contamination directions on the same inactive support.
        direction_settings = {
            "same_positive": np.ones(cfg.contaminated),
            "same_negative": -np.ones(cfg.contaminated),
            "mixed_sign": np.where(np.arange(cfg.contaminated) % 2 == 0, 1.0, -1.0),
        }
        contamination_shape = cfg.contamination_strength * (
            1.0 + 0.5 * np.abs(x[:, cfg.bias_index])
        )
        for setting, directions in direction_settings.items():
            tensor = baseline_tensor.copy()
            for rank, proxy_id in enumerate(polluted_ids):
                tensor[proxy_id, :, cfg.bias_index] += directions[rank] * contamination_shape
            parameters = {
                "support": "inactive_bias_feature",
                "directions": [float(v) for v in directions],
                "clean_error_bound": cfg.baseline_clean_tolerance,
            }
            for estimator in ("local_median", "local_mean"):
                _add_estimator_row(
                    rows, cfg, seed, "contamination_direction", setting, estimator,
                    polluted_ids, parameters, tensor, truth
                )

        # 2. Disjoint versus driver-overlapping contamination support.
        overlap_settings = {
            "disjoint_inactive_support": cfg.bias_index,
            "driver_overlap_opposing": cfg.driver_indices[0],
        }
        for setting, feature in overlap_settings.items():
            tensor = baseline_tensor.copy()
            if feature == cfg.bias_index:
                signal = cfg.contamination_strength * (1.0 + 0.5 * np.abs(x[:, feature]))
            else:
                # Deliberately oppose the true local driver sign to expose sign risk.
                signal = -np.sign(truth[:, feature]) * cfg.contamination_strength * (
                    1.0 + 0.5 * np.abs(x[:, feature])
                )
            for proxy_id in polluted_ids:
                tensor[proxy_id, :, feature] += signal
            parameters = {
                "contamination_feature": int(feature),
                "overlaps_true_driver": bool(feature in cfg.driver_indices),
                "direction": "opposes_local_driver" if feature in cfg.driver_indices else "positive",
                "clean_error_bound": cfg.baseline_clean_tolerance,
            }
            for estimator in ("local_median", "local_mean"):
                _add_estimator_row(
                    rows, cfg, seed, "support_overlap", setting, estimator,
                    polluted_ids, parameters, tensor, truth
                )

        # 3. Positive scale perturbation and two explicit calibration regimes.
        no_pollution = np.asarray([], dtype=int)
        for severity in cfg.scale_severities:
            true_scales = scale_factors(severity, cfg.K)
            tensor = true_scales[:, None, None] * truth[None, :, :]
            estimated_scales = estimate_scales_from_independent_anchor(
                true_scales, cfg, seed, severity
            )
            relative_scale_error = float(
                np.mean(np.abs(estimated_scales - true_scales) / true_scales)
            )
            setting = f"severity_{severity:g}"
            parameters = {
                "severity": float(severity),
                "true_scales": [float(v) for v in true_scales],
                "median_true_scale": float(np.median(true_scales)),
                "anchor_noise_std": cfg.anchor_noise_std,
                "anchor_is_independent": True,
            }
            _add_estimator_row(
                rows, cfg, seed, "positive_scale", setting, "raw_median",
                no_pollution, parameters, tensor, truth
            )
            _add_estimator_row(
                rows, cfg, seed, "positive_scale", setting,
                "known_scale_normalized_median", no_pollution, parameters,
                tensor, truth, scales=true_scales, scale_error=0.0
            )
            _add_estimator_row(
                rows, cfg, seed, "positive_scale", setting,
                "independent_anchor_normalized_median", no_pollution, parameters,
                tensor, truth, scales=estimated_scales,
                scale_error=relative_scale_error
            )

        # 4. Approximate cleanliness and the compatibility-set diagnostic.
        for epsilon in cfg.clean_tolerances:
            for bias_fraction in cfg.clean_bias_fractions:
                tensor = np.repeat(truth[None, :, :], cfg.K, axis=0)
                shared_pattern = np.where(np.abs(truth) > 1e-12, np.sign(truth), 1.0)
                for proxy_id in range(cfg.K):
                    if clean_mask[proxy_id]:
                        random_part = _rng(
                            seed, "approx-clean-error", epsilon, bias_fraction, proxy_id
                        ).uniform(-1.0, 1.0, size=truth.shape)
                        perturbation = epsilon * (
                            bias_fraction * shared_pattern
                            + (1.0 - bias_fraction) * random_part
                        )
                        if float(np.max(np.abs(perturbation))) > epsilon + 1e-12:
                            raise AssertionError("Clean perturbation escaped its epsilon bound.")
                        tensor[proxy_id] += perturbation
                    else:
                        rank = int(np.flatnonzero(polluted_ids == proxy_id)[0])
                        direction = 1.0 if rank % 2 == 0 else -1.0
                        tensor[proxy_id] += direction * cfg.contamination_strength * (
                            1.2 + 0.2 * np.abs(x)
                        )
                setting = f"epsilon_{epsilon:g}_bias_{bias_fraction:g}"
                parameters = {
                    "epsilon": float(epsilon),
                    "clean_bias_fraction": float(bias_fraction),
                    "clean_error_is_nonzero_when_epsilon_positive": True,
                    "compatibility_contamination_budget": cfg.contaminated,
                }
                _add_estimator_row(
                    rows, cfg, seed, "approximate_cleanliness", setting,
                    "local_median", polluted_ids, parameters, tensor, truth,
                    compatibility_epsilon=float(epsilon)
                )
                if epsilon > 0.0:
                    declared_settings = (
                        ("understate_r", max(0, cfg.contaminated - 1), float(epsilon)),
                        ("understate_epsilon", cfg.contaminated, float(0.5 * epsilon)),
                        (
                            "understate_both",
                            max(0, cfg.contaminated - 1),
                            float(0.5 * epsilon),
                        ),
                    )
                    for declared_name, declared_r, declared_epsilon in declared_settings:
                        declared_parameters = dict(parameters)
                        declared_parameters.update(
                            {
                                "analysis_role": "compatibility_misspecification_stress",
                                "true_clean_error_bound": float(epsilon),
                                "declared_contamination_budget": int(declared_r),
                                "declared_clean_tolerance": float(declared_epsilon),
                            }
                        )
                        _add_estimator_row(
                            rows,
                            cfg,
                            seed,
                            "compatibility_misspecification",
                            f"{setting}_{declared_name}",
                            "local_median",
                            polluted_ids,
                            declared_parameters,
                            tensor,
                            truth,
                            compatibility_epsilon=declared_epsilon,
                            compatibility_budget=declared_r,
                        )

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - start,
        "estimand": "signed local zero-reference attribution tensor",
        "aggregation_order": (
            "aggregate signed local attributions across proxies for each instance; "
            "then compute mean_instance(abs(local consensus))"
        ),
        "scope": (
            "controlled attribution-level sensitivity analysis; no response-model fitting "
            "and no causal interpretation"
        ),
        "confidence_intervals": (
            "two-sided 95% Student-t intervals over seed-level results; instances are "
            "not treated as independent replicates"
        ),
        "scale_interpretation": (
            "positive raw rescaling preserves exact support and sign in the noiseless scale "
            "scenario, but magnitude is not comparable without external calibration"
        ),
        "compatibility_definition": (
            "targets in [-1,1] supported within epsilon by at least "
            "K-contaminated proxy values"
        ),
        "additional_analysis": (
            "misspecified-budget falsification and sign-abstention diagnostics were "
            "specified after the primary sensitivity analysis was completed and are "
            "reported separately"
        ),
        "environment": runtime_environment(),
    }
    return rows, metadata


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    summaries: list[dict[str, Any]] = []
    group_columns = ["family", "setting", "estimator"]
    for group_key, group in frame.groupby(group_columns, sort=True):
        row: dict[str, Any] = dict(zip(group_columns, group_key))
        row["setting_parameters"] = group["setting_parameters"].iloc[0]
        row["seed_count"] = int(group["seed"].nunique())
        for metric in GENERAL_METRICS + OPTIONAL_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            row[f"{metric}_n"] = int(values.size)
            if values.size == 0:
                for suffix in ("mean", "std", "se", "ci95_low", "ci95_high"):
                    row[f"{metric}_{suffix}"] = None
                continue
            mean = float(np.mean(values))
            if values.size >= 2:
                std = float(np.std(values, ddof=1))
                se = std / math.sqrt(values.size)
                critical = float(stats.t.ppf(0.975, df=values.size - 1))
                low, high = mean - critical * se, mean + critical * se
            else:
                std = se = low = high = None
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_se"] = se
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        summaries.append(row)
    return summaries


def validate_results(
    cfg: RunConfig,
    rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
) -> None:
    raw = pd.DataFrame(rows)
    summary = pd.DataFrame(summaries)
    if raw.empty or summary.empty:
        raise AssertionError("Sensitivity suite produced an empty table.")
    key = ["seed", "family", "setting", "estimator"]
    if raw.duplicated(key).any():
        raise AssertionError("Duplicate seed/family/setting/estimator rows detected.")
    if not np.isfinite(raw[GENERAL_METRICS].to_numpy(dtype=float)).all():
        raise AssertionError("A general metric is missing or non-finite.")
    if not summary["seed_count"].eq(len(cfg.seeds)).all():
        raise AssertionError("At least one summary group is missing a registered seed.")
    approximate = raw[raw["family"].eq("approximate_cleanliness")]
    for metric in (
        "compatibility_coverage",
        "compatibility_diameter_max",
        "compatibility_diameter_bound_satisfied",
        "median_error_bound_coverage",
    ):
        if not np.isfinite(approximate[metric].to_numpy(dtype=float)).all():
            raise AssertionError(f"Approximate-cleanliness metric {metric} is non-finite.")
    if not approximate["compatibility_coverage"].eq(1.0).all():
        raise AssertionError("The true target escaped a registered compatibility set.")
    if not approximate["compatibility_diameter_bound_satisfied"].eq(1.0).all():
        raise AssertionError("A compatibility-set diameter exceeded 2 epsilon.")
    if not approximate["median_error_bound_coverage"].eq(1.0).all():
        raise AssertionError("The local median escaped its epsilon error bound.")
    misspecified = raw[raw["family"].eq("compatibility_misspecification")]
    if not misspecified.empty:
        diagnostic_metrics = (
            "compatibility_nonempty_rate",
            "compatibility_empty_rate",
            "compatibility_direction_decision_rate",
            "compatibility_direction_correct_rate",
            "compatibility_false_direction_rate",
            "compatibility_abstention_rate",
        )
        if not np.isfinite(
            misspecified[list(diagnostic_metrics)].to_numpy(dtype=float)
        ).all():
            raise AssertionError("A misspecified-budget diagnostic is non-finite.")
        if float(misspecified["compatibility_empty_rate"].max()) <= 0.0:
            raise AssertionError(
                "Registered misspecification settings never falsified a declared budget."
            )
    known = raw[
        raw["family"].eq("positive_scale")
        & raw["estimator"].eq("known_scale_normalized_median")
    ]
    if float(known["local_linf_error"].max()) > 1e-12:
        raise AssertionError("Known-scale normalization failed to recover the truth.")
    scale_rows = raw[raw["family"].eq("positive_scale")]
    if not scale_rows["support_precision"].eq(1.0).all() or not scale_rows[
        "support_recall"
    ].eq(1.0).all():
        raise AssertionError("Positive scale perturbation unexpectedly changed support.")
    if not scale_rows["driver_sign_accuracy"].eq(1.0).all():
        raise AssertionError("Positive scale perturbation unexpectedly changed driver signs.")


def runtime_environment() -> dict[str, Any]:
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "pyyaml": yaml.__version__,
        },
    }


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


def output_paths(cfg: RunConfig) -> dict[str, Path]:
    output_dir = Path(cfg.output_dir)
    prefix = output_dir / "gamma_sensitivity_revised"
    return {
        "raw_csv": Path(f"{prefix}_raw.csv"),
        "raw_json": Path(f"{prefix}_raw.json"),
        "summary_csv": Path(f"{prefix}_summary.csv"),
        "summary_json": Path(f"{prefix}_summary.json"),
        "metadata_json": Path(f"{prefix}_metadata.json"),
    }


def require_explicit_overwrite(cfg: RunConfig) -> None:
    existing = [path for path in output_paths(cfg).values() if path.exists()]
    if existing and not cfg.overwrite:
        joined = "\n    ".join(str(path) for path in existing)
        raise FileExistsError(
            "Refusing to replace registered sensitivity outputs. Audit them, then "
            f"re-run with --overwrite:\n    {joined}"
        )


def save_outputs(
    cfg: RunConfig,
    yaml_snapshot: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> dict[str, Path]:
    paths = output_paths(cfg)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(paths["raw_csv"], index=False)
    pd.DataFrame(summaries).to_csv(paths["summary_csv"], index=False)
    with paths["raw_json"].open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(list(rows)), handle, indent=2, ensure_ascii=False, allow_nan=False)
    with paths["summary_json"].open("w", encoding="utf-8") as handle:
        json.dump(
            _json_safe(list(summaries)), handle, indent=2, ensure_ascii=False, allow_nan=False
        )
    resolved = asdict(cfg)
    resolved["output_dir"] = _artifact_safe_path(Path(cfg.output_dir))
    config_blob = json.dumps(_json_safe(resolved), sort_keys=True).encode("utf-8")
    full_metadata = dict(metadata)
    full_metadata.update(
        {
            "resolved_config": resolved,
            "config_sha256": hashlib.sha256(config_blob).hexdigest(),
            "yaml_snapshot": yaml_snapshot,
            "raw_row_count": len(rows),
            "summary_row_count": len(summaries),
            "output_files": {
                key: _artifact_safe_path(path) for key, path in paths.items()
            },
        }
    )
    with paths["metadata_json"].open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(full_metadata), handle, indent=2, ensure_ascii=False, allow_nan=False)
    return paths


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    cfg, yaml_snapshot = resolve_config(args)
    require_explicit_overwrite(cfg)
    print("Revised attribution-tensor sensitivity suite")
    print(
        f"  profile={cfg.profile} seeds={list(cfg.seeds)} K={cfg.K} "
        f"m={cfg.contaminated} n_eval={cfg.n_eval} n_anchor={cfg.n_anchor}"
    )
    print("  order=signed local aggregation -> mean(abs(local consensus))")
    rows, metadata = run_experiment(cfg)
    summaries = summarize_rows(rows)
    validate_results(cfg, rows, summaries)
    paths = save_outputs(cfg, yaml_snapshot, rows, summaries, metadata)
    print(f"Completed in {metadata['elapsed_seconds']:.2f} seconds")
    for label, path in paths.items():
        print(f"  {label:13s}: {path}")


if __name__ == "__main__":
    main()
