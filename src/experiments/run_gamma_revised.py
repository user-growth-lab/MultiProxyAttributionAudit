"""Controlled benchmark for robust multi-proxy latent attribution.

This module is deliberately self-contained and implements three design rules:

1. The well-specified DGP is *exactly additive on the probability scale* and
   is analytically checked to stay in ``[epsilon, 1 - epsilon]``.
2. Proxy attributions are aggregated for every evaluation user first; global
   importance is only then computed as ``mean_x(abs(local_consensus(x)))``.
3. Every contamination count is scanned for K in {3, 4, 5, 7}, with raw
   seed-level results and Student-t confidence intervals.

The experiment is end-to-end: it samples binary proxy responses, fits one
response model per proxy, and explains the fitted response surface.  The
additive DGP uses a linear-probability model, for which baseline Shapley values
are exact.  The stress-test DGP applies a logistic link and uses exact Shapley
enumeration about the zero (population-mean) reference point. The latter is a
controlled nonlinear, signal-dependent contamination stress, not a causal
claim or a violation of the arbitrary attribution-contamination theorem.

Examples
--------
Quick grid (uses ``configs/gamma.yaml`` when present)::

    python3 -m src.experiments.run_gamma_revised --profile quick

Paper grid::

    python3 -m src.experiments.run_gamma_revised --profile paper

Minimal smoke test::

    python3 -m src.experiments.run_gamma_revised \
        --seeds 0 --ks 3 --n-train 300 --n-eval 12 \
        --bias-multipliers 1 --output-dir /tmp/gamma_revised_smoke
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "wsdm27-matplotlib")
)

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import yaml
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression

from src.explainers.robust_proxy import aggregate_local_median

# Some numerical backends can leave harmless floating-point status flags after
# finite matrix multiplications. The runner performs explicit finite-value
# checks at every model boundary.
np.seterr(all="ignore")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "gamma.yaml"


BUILTIN_PROFILES: dict[str, dict[str, Any]] = {
    "quick": {
        "seeds": [0, 1, 2],
        "ks": [3, 4, 5, 7],
        "n_train": 1500,
        "n_eval": 50,
        "bias_multipliers": [1.0],
        "trim_fraction": 0.20,
        "rpca": True,
    },
    "paper": {
        "seeds": list(range(10)),
        "ks": [3, 4, 5, 7],
        "n_train": 6000,
        "n_eval": 200,
        "bias_multipliers": [0.5, 1.0, 1.5],
        "trim_fraction": 0.20,
        "rpca": True,
    },
}


BUILTIN_EXPERIMENT: dict[str, Any] = {
    "d": 8,
    "driver_indices": [0, 1],
    "bias_index": 2,
    "additive_driver_coefficients": [0.12, -0.10],
    "additive_bias_coefficient": 0.18,
    "logistic_driver_coefficients": [1.20, -1.00],
    "logistic_bias_coefficient": 1.50,
    "probability_epsilon": 0.01,
}


METRIC_NAMES = [
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
]

RUNTIME_METRIC_NAMES = [
    "runtime_seconds",
    "aggregation_runtime_seconds",
    "proxy_training_seconds",
    "proxy_attribution_seconds",
    "proxy_fit_explain_seconds",
    "scenario_runtime_seconds",
]


@dataclass(frozen=True)
class RunConfig:
    """Fully resolved experiment configuration."""

    profile: str
    seeds: tuple[int, ...]
    ks: tuple[int, ...]
    contamination_counts: tuple[int, ...] | None
    n_train: int
    n_eval: int
    bias_multipliers: tuple[float, ...]
    trim_fraction: float
    use_rpca: bool
    d: int
    driver_indices: tuple[int, ...]
    bias_index: int
    additive_driver_coefficients: tuple[float, ...]
    additive_bias_coefficient: float
    logistic_driver_coefficients: tuple[float, ...]
    logistic_bias_coefficient: float
    probability_epsilon: float
    output_dir: str
    save_raw_csv: bool
    save_raw_json: bool
    save_summary_csv: bool
    save_summary_json: bool
    make_plot: bool
    overwrite: bool


def _parse_csv(value: str | Sequence[Any], cast: type) -> list[Any]:
    if isinstance(value, str):
        fields = [item.strip() for item in value.split(",") if item.strip()]
    else:
        fields = list(value)
    return [cast(item) for item in fields]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        warnings.warn(
            f"Configuration file {path} does not exist; using built-in defaults.",
            stacklevel=2,
        )
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping at the root of {path}.")
    return loaded


def _first(mapping: Mapping[str, Any], keys: Iterable[str], default: Any) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _resolve_output_dir(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _artifact_safe_path(path: Path) -> str:
    """Serialize paths without leaking an author's absolute workspace."""

    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return f"<external-output>/{resolved.name}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the theory-aligned additive DGP and logistic-link stress test "
            "for local-first multi-proxy attribution aggregation."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--profile", choices=sorted(BUILTIN_PROFILES), default=None)
    parser.add_argument("--seeds", help="Comma-separated random seeds.")
    parser.add_argument("--ks", help="Comma-separated proxy counts.")
    parser.add_argument(
        "--contamination-counts",
        help=(
            "Optional comma-separated contamination counts to run for every K. "
            "The default remains the complete integer grid 0,...,K."
        ),
    )
    parser.add_argument("--n-train", type=int)
    parser.add_argument("--n-eval", type=int)
    parser.add_argument("--d", type=int)
    parser.add_argument(
        "--bias-multipliers", help="Comma-separated multipliers of the base bias strength."
    )
    parser.add_argument("--trim-fraction", type=float)
    parser.add_argument("--attr-noise", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir")
    rpca_group = parser.add_mutually_exclusive_group()
    rpca_group.add_argument("--enable-rpca", action="store_true")
    rpca_group.add_argument("--skip-rpca", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace existing gamma_revised_* outputs.",
    )
    return parser


def resolve_config(args: argparse.Namespace) -> tuple[RunConfig, dict[str, Any]]:
    """Resolve built-ins, YAML profile, and finally explicit CLI overrides."""

    yaml_path = Path(args.config).expanduser()
    if not yaml_path.is_absolute():
        yaml_path = PROJECT_ROOT / yaml_path
    raw_yaml = _load_yaml(yaml_path)

    profile_name = args.profile or raw_yaml.get("default_profile", "quick")
    if profile_name not in BUILTIN_PROFILES:
        raise ValueError(f"Unknown profile {profile_name!r}; choose quick or paper.")
    profile = dict(BUILTIN_PROFILES[profile_name])
    profile.update((raw_yaml.get("profiles", {}) or {}).get(profile_name, {}) or {})
    experiment = dict(BUILTIN_EXPERIMENT)
    experiment.update(raw_yaml.get("experiment", {}) or {})

    seeds_value = args.seeds if args.seeds is not None else _first(profile, ["seeds"], [0])
    ks_value = args.ks if args.ks is not None else _first(profile, ["ks", "K"], [3, 4, 5, 7])
    bias_value = (
        args.bias_multipliers
        if args.bias_multipliers is not None
        else _first(profile, ["bias_multipliers", "bias_strengths"], [1.0])
    )
    output_block = raw_yaml.get("output", {}) or {}
    output_value = args.output_dir or output_block.get(
        "dir", "reproduced_results/synthetic_discovery"
    )

    cfg = RunConfig(
        profile=profile_name,
        seeds=tuple(_parse_csv(seeds_value, int)),
        ks=tuple(_parse_csv(ks_value, int)),
        contamination_counts=(
            None
            if args.contamination_counts is None
            else tuple(_parse_csv(args.contamination_counts, int))
        ),
        n_train=int(args.n_train or _first(profile, ["n_train"], 1500)),
        n_eval=int(args.n_eval or _first(profile, ["n_eval", "eval_users"], 50)),
        bias_multipliers=tuple(_parse_csv(bias_value, float)),
        trim_fraction=float(
            args.trim_fraction
            if args.trim_fraction is not None
            else _first(profile, ["trim_fraction"], 0.20)
        ),
        use_rpca=(
            True
            if args.enable_rpca
            else bool(_first(profile, ["rpca", "use_rpca"], True))
            and not args.skip_rpca
        ),
        d=int(args.d or experiment["d"]),
        driver_indices=tuple(int(v) for v in experiment["driver_indices"]),
        bias_index=int(experiment["bias_index"]),
        additive_driver_coefficients=tuple(
            float(v) for v in experiment["additive_driver_coefficients"]
        ),
        additive_bias_coefficient=float(experiment["additive_bias_coefficient"]),
        logistic_driver_coefficients=tuple(
            float(v) for v in experiment["logistic_driver_coefficients"]
        ),
        logistic_bias_coefficient=float(experiment["logistic_bias_coefficient"]),
        probability_epsilon=float(experiment["probability_epsilon"]),
        output_dir=str(_resolve_output_dir(str(output_value))),
        save_raw_csv=bool(output_block.get("save_raw_csv", True)),
        save_raw_json=bool(output_block.get("save_raw_json", True)),
        save_summary_csv=bool(output_block.get("save_summary_csv", True)),
        save_summary_json=bool(output_block.get("save_summary_json", True)),
        make_plot=bool(output_block.get("save_figure", True)) and not args.no_plot,
        overwrite=bool(args.overwrite),
    )
    validate_config(cfg)
    if args.attr_noise not in (None, 0.0):
        warnings.warn(
            "--attr-noise is deprecated and ignored: model/label sampling now supplies "
            "end-to-end estimation noise.",
            stacklevel=2,
        )
    return cfg, raw_yaml


def validate_config(cfg: RunConfig) -> None:
    if not cfg.seeds:
        raise ValueError("At least one seed is required.")
    if not cfg.ks or any(k < 2 for k in cfg.ks):
        raise ValueError("Every K must be at least 2.")
    if cfg.contamination_counts is not None:
        if not cfg.contamination_counts:
            raise ValueError("contamination_counts cannot be empty when provided.")
        if len(set(cfg.contamination_counts)) != len(cfg.contamination_counts):
            raise ValueError("contamination_counts cannot contain duplicates.")
        if any(count < 0 for count in cfg.contamination_counts):
            raise ValueError("contamination_counts must be non-negative.")
        invalid = [
            (k, count)
            for k in cfg.ks
            for count in cfg.contamination_counts
            if count > k
        ]
        if invalid:
            raise ValueError(
                "Every contamination count must be no greater than every selected K; "
                f"invalid pairs: {invalid}."
            )
    if cfg.n_train < 50 or cfg.n_eval < 2:
        raise ValueError("n_train must be >= 50 and n_eval must be >= 2.")
    if not 0 <= cfg.trim_fraction < 0.5:
        raise ValueError("trim_fraction must lie in [0, 0.5).")
    if len(cfg.driver_indices) != len(cfg.additive_driver_coefficients):
        raise ValueError("additive_driver_coefficients must match driver_indices.")
    if len(cfg.driver_indices) != len(cfg.logistic_driver_coefficients):
        raise ValueError("logistic_driver_coefficients must match driver_indices.")
    used = set(cfg.driver_indices) | {cfg.bias_index}
    if min(used) < 0 or max(used) >= cfg.d or len(used) != len(cfg.driver_indices) + 1:
        raise ValueError("Driver and bias indices must be distinct and fall inside [0, d).")
    if not 0 < cfg.probability_epsilon < 0.5:
        raise ValueError("probability_epsilon must lie in (0, 0.5).")
    if any(multiplier < 0 for multiplier in cfg.bias_multipliers):
        raise ValueError("bias_multipliers must be non-negative.")

    # X_j is supported on [-1, 1].  This is an analytic, not empirical, bound.
    largest_bias = max(cfg.bias_multipliers, default=0.0) * abs(
        cfg.additive_bias_coefficient
    )
    max_deviation = sum(abs(v) for v in cfg.additive_driver_coefficients) + largest_bias
    admissible = 0.5 - cfg.probability_epsilon
    if max_deviation > admissible + 1e-12:
        raise ValueError(
            "The additive DGP is not bounded as requested: "
            f"sum(|driver coefficients|) + max(|bias|) = {max_deviation:.6f} "
            f"> 0.5 - epsilon = {admissible:.6f}."
        )


def sigmoid(z: np.ndarray) -> np.ndarray:
    positive = z >= 0
    output = np.empty_like(z, dtype=float)
    output[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    output[~positive] = exp_z / (1.0 + exp_z)
    return output


def coefficient_vector(
    d: int, indices: Sequence[int], coefficients: Sequence[float]
) -> np.ndarray:
    beta = np.zeros(d, dtype=float)
    beta[np.asarray(indices, dtype=int)] = np.asarray(coefficients, dtype=float)
    return beta


def exact_zero_reference_logistic_shapley(
    X: np.ndarray, coefficients: np.ndarray, intercept: float
) -> np.ndarray:
    """Exact baseline Shapley values for ``sigmoid(intercept + x @ beta)``.

    Missing coordinates are set to the zero reference.  Since the synthetic
    covariates are symmetric Uniform[-1, 1], zero is also their population
    mean.  Enumeration is practical for the registered d=8 experiment.
    """

    n, d = X.shape
    if coefficients.shape != (d,):
        raise ValueError("Coefficient vector has the wrong dimension.")
    if d > 14:
        raise ValueError(
            "Exact Shapley enumeration is intentionally limited to d <= 14; "
            "use the registered d=8 grid for this core experiment."
        )
    phi = np.zeros((n, d), dtype=float)
    all_indices = tuple(range(d))
    for feature in all_indices:
        others = tuple(j for j in all_indices if j != feature)
        feature_term = X[:, feature] * coefficients[feature]
        for size in range(d):
            weight = 1.0 / (d * math.comb(d - 1, size))
            for subset in itertools.combinations(others, size):
                if subset:
                    subset_idx = np.asarray(subset, dtype=int)
                    z_without = intercept + X[:, subset_idx] @ coefficients[subset_idx]
                else:
                    z_without = np.full(n, intercept, dtype=float)
                phi[:, feature] += weight * (
                    sigmoid(z_without + feature_term) - sigmoid(z_without)
                )
    return phi


def response_probability(
    X: np.ndarray,
    dgp: str,
    latent_beta: np.ndarray,
    bias_index: int,
    bias_coefficient: float,
) -> np.ndarray:
    linear = X @ latent_beta + bias_coefficient * X[:, bias_index]
    if not np.all(np.isfinite(linear)):
        raise FloatingPointError("The proxy response surface produced non-finite logits.")
    if dgp == "additive_probability":
        return 0.5 + linear
    if dgp == "logistic_link_stress":
        return sigmoid(linear)
    raise ValueError(f"Unknown DGP: {dgp}")


def fit_and_explain_proxy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    p_eval: np.ndarray,
    dgp: str,
    seed: int,
) -> tuple[np.ndarray, float, float, float]:
    """Fit a proxy model and return local attributions plus response RMSE."""

    fit_start = time.perf_counter()
    if dgp == "additive_probability":
        model = LinearRegression().fit(X_train, y_train)
        fit_seconds = time.perf_counter() - fit_start
        coefficients = np.asarray(model.coef_, dtype=float)
        explain_start = time.perf_counter()
        local_phi = X_eval * coefficients[None, :]
        prediction = np.asarray(model.predict(X_eval), dtype=float)
    elif dgp == "logistic_link_stress":
        model = LogisticRegression(
            C=1000.0,
            solver="lbfgs",
            max_iter=500,
            random_state=seed,
        ).fit(X_train, y_train)
        fit_seconds = time.perf_counter() - fit_start
        coefficients = np.asarray(model.coef_[0], dtype=float)
        intercept = float(model.intercept_[0])
        explain_start = time.perf_counter()
        local_phi = exact_zero_reference_logistic_shapley(
            X_eval, coefficients, intercept
        )
        prediction = np.asarray(model.predict_proba(X_eval)[:, 1], dtype=float)
    else:
        raise ValueError(f"Unknown DGP: {dgp}")
    explain_seconds = time.perf_counter() - explain_start
    probability_rmse = float(np.sqrt(np.mean((prediction - p_eval) ** 2)))
    if not np.all(np.isfinite(local_phi)) or not np.isfinite(probability_rmse):
        raise FloatingPointError("Proxy fitting or explanation produced non-finite values.")
    return local_phi, probability_rmse, fit_seconds, explain_seconds


def scaled_huber_location(
    proxy_phi: np.ndarray, tuning: float = 1.345, max_iter: int = 60
) -> np.ndarray:
    """Coordinatewise Huber location with a coordinatewise MAD scale."""

    center = np.median(proxy_phi, axis=0)
    mad = np.median(np.abs(proxy_phi - center[None, :, :]), axis=0)
    scale = 1.4826 * mad
    # A zero MAD is common with small K.  The mean absolute deviation is a
    # scale fallback, not a fixed threshold tied to the attribution units.
    fallback = np.mean(np.abs(proxy_phi - center[None, :, :]), axis=0)
    scale = np.where(scale > 1e-10, scale, fallback)
    scale = np.maximum(scale, 1e-10)
    threshold = tuning * scale
    for _ in range(max_iter):
        residual = proxy_phi - center[None, :, :]
        absolute = np.abs(residual)
        weights = np.minimum(1.0, threshold[None, :, :] / np.maximum(absolute, 1e-12))
        updated = np.sum(weights * proxy_phi, axis=0) / np.maximum(
            np.sum(weights, axis=0), 1e-12
        )
        if np.max(np.abs(updated - center)) < 1e-9:
            center = updated
            break
        center = updated
    return center


def trimmed_location(proxy_phi: np.ndarray, fraction: float) -> np.ndarray:
    k = proxy_phi.shape[0]
    trim = min(int(math.floor(fraction * k)), (k - 1) // 2)
    ordered = np.sort(proxy_phi, axis=0)
    return ordered.mean(axis=0) if trim == 0 else ordered[trim : k - trim].mean(axis=0)


def winsorized_location(proxy_phi: np.ndarray, fraction: float) -> np.ndarray:
    k = proxy_phi.shape[0]
    trim = min(int(math.floor(fraction * k)), (k - 1) // 2)
    ordered = np.sort(proxy_phi, axis=0)
    if trim == 0:
        return ordered.mean(axis=0)
    winsorized = ordered.copy()
    winsorized[:trim] = ordered[trim]
    winsorized[k - trim :] = ordered[k - trim - 1]
    return winsorized.mean(axis=0)


def median_of_means_location(proxy_phi: np.ndarray) -> np.ndarray:
    """Deterministic two-block median-of-means across randomly permuted proxies.

    Proxy identities are randomly permuted for every seed/scenario before the
    contamination set is selected.  Thus fixed contiguous blocks do not favor
    a particular proxy identity.  For the small registered K values, two
    nonempty blocks avoid singleton-only groups while retaining a genuine MoM
    comparator.
    """

    group_count = min(2, proxy_phi.shape[0])
    group_means = [group.mean(axis=0) for group in np.array_split(proxy_phi, group_count)]
    return np.median(np.stack(group_means, axis=0), axis=0)


def rank1_svd_consensus(proxy_phi: np.ndarray) -> np.ndarray:
    """Rank-one reconstruction of the proxy-by-(user,feature) matrix."""

    k, n, d = proxy_phi.shape
    matrix = proxy_phi.reshape(k, n * d)
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    reconstruction = singular[0] * np.outer(u[:, 0], vt[0])
    return reconstruction.mean(axis=0).reshape(n, d)


def _soft_threshold(matrix: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(matrix) * np.maximum(np.abs(matrix) - threshold, 0.0)


def _singular_threshold(matrix: np.ndarray, threshold: float) -> np.ndarray:
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    retained = singular > threshold
    if not np.any(retained):
        return np.zeros_like(matrix)
    shrunk = singular[retained] - threshold
    return (u[:, retained] * shrunk[None, :]) @ vt[retained]


def robust_pca(
    matrix: np.ndarray, tolerance: float = 1e-6, max_iter: int = 300
) -> tuple[np.ndarray, np.ndarray, int]:
    """Principal-component pursuit via the inexact augmented Lagrangian method."""

    rows, columns = matrix.shape
    lam = 1.0 / math.sqrt(max(rows, columns))
    fro_norm = float(np.linalg.norm(matrix, ord="fro"))
    if fro_norm <= 1e-15:
        return np.zeros_like(matrix), np.zeros_like(matrix), 0
    spectral_norm = float(np.linalg.norm(matrix, ord=2))
    infinity_norm = float(np.max(np.abs(matrix)) / lam)
    dual_norm = max(spectral_norm, infinity_norm, 1e-15)
    dual = matrix / dual_norm
    low_rank = np.zeros_like(matrix)
    sparse = np.zeros_like(matrix)
    mu = 1.25 / max(spectral_norm, 1e-15)
    mu_max = mu * 1e7
    rho = 1.5

    for iteration in range(1, max_iter + 1):
        low_rank = _singular_threshold(matrix - sparse + dual / mu, 1.0 / mu)
        sparse = _soft_threshold(matrix - low_rank + dual / mu, lam / mu)
        residual = matrix - low_rank - sparse
        dual = dual + mu * residual
        mu = min(mu * rho, mu_max)
        if np.linalg.norm(residual, ord="fro") / fro_norm < tolerance:
            return low_rank, sparse, iteration
    return low_rank, sparse, max_iter


def rpca_consensus(proxy_phi: np.ndarray) -> tuple[np.ndarray, int]:
    k, n, d = proxy_phi.shape
    matrix = proxy_phi.reshape(k, n * d)
    low_rank, _, iterations = robust_pca(matrix)
    return low_rank.mean(axis=0).reshape(n, d), iterations


def aggregate_local_attributions(
    proxy_phi: np.ndarray,
    clean_mask: np.ndarray,
    trim_fraction: float,
    use_rpca: bool,
) -> tuple[dict[str, np.ndarray], int | None, dict[str, float]]:
    """Return local (n,d) consensus arrays for every baseline."""

    aggregations: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}

    def timed(name: str, function: Any) -> None:
        start = time.perf_counter()
        aggregations[name] = function()
        timings[name] = time.perf_counter() - start

    timed("single_proxy", lambda: proxy_phi[0].copy())
    timed("mean", lambda: proxy_phi.mean(axis=0))
    timed("scaled_huber", lambda: scaled_huber_location(proxy_phi))
    timed("trimmed_mean", lambda: trimmed_location(proxy_phi, trim_fraction))
    timed("winsorized_mean", lambda: winsorized_location(proxy_phi, trim_fraction))
    timed("median_of_means", lambda: median_of_means_location(proxy_phi))
    timed("median", lambda: aggregate_local_median(proxy_phi))
    timed("rank1_svd", lambda: rank1_svd_consensus(proxy_phi))
    rpca_iterations: int | None = None
    if use_rpca:
        start = time.perf_counter()
        aggregations["rpca"], rpca_iterations = rpca_consensus(proxy_phi)
        timings["rpca"] = time.perf_counter() - start
    if np.any(clean_mask):
        timed("oracle_clean", lambda: proxy_phi[clean_mask].mean(axis=0))
    return aggregations, rpca_iterations, timings


def _safe_spearman(estimate: np.ndarray, truth: np.ndarray) -> float:
    value = stats.spearmanr(estimate, truth).statistic
    return float(value) if np.isfinite(value) else 0.0


def evaluate_local_consensus(
    local_estimate: np.ndarray,
    local_truth: np.ndarray,
    driver_indices: Sequence[int],
    bias_index: int,
) -> dict[str, float]:
    if not np.all(np.isfinite(local_estimate)):
        return {name: float("nan") for name in METRIC_NAMES}
    global_estimate = np.mean(np.abs(local_estimate), axis=0)
    global_truth = np.mean(np.abs(local_truth), axis=0)
    denominator_local = float(np.mean(np.abs(local_truth))) + 1e-12
    denominator_global = float(np.sum(global_truth)) + 1e-12
    drivers = np.asarray(driver_indices, dtype=int)
    top = np.argsort(-global_estimate, kind="stable")[: len(drivers)]
    overlap = len(set(top.tolist()) & set(drivers.tolist()))
    truth_driver = local_truth[:, drivers]
    estimate_driver = local_estimate[:, drivers]
    active = np.abs(truth_driver) > 1e-10
    sign_accuracy = float(
        np.mean(np.sign(estimate_driver[active]) == np.sign(truth_driver[active]))
    )
    driver_magnitude = float(np.mean(global_estimate[drivers]))
    bias_magnitude = float(global_estimate[bias_index])
    return {
        "attribution_l1_error": float(np.mean(np.abs(local_estimate - local_truth))),
        "attribution_linf_error": float(np.max(np.abs(local_estimate - local_truth))),
        "local_normalized_l1": float(
            np.mean(np.abs(local_estimate - local_truth)) / denominator_local
        ),
        "local_normalized_linf": float(
            np.max(np.abs(local_estimate - local_truth))
            / (float(np.max(np.abs(local_truth))) + 1e-12)
        ),
        "global_normalized_l1": float(
            np.sum(np.abs(global_estimate - global_truth)) / denominator_global
        ),
        "support_precision": float(overlap / len(top)),
        "support_recall": float(overlap / len(drivers)),
        "spearman": _safe_spearman(global_estimate, global_truth),
        "rank_correlation": _safe_spearman(global_estimate, global_truth),
        "driver_sign_accuracy": sign_accuracy,
        "bias_leakage": float(bias_magnitude / (driver_magnitude + 1e-12)),
        "bias_global_attribution": bias_magnitude,
        "driver_global_attribution": driver_magnitude,
    }


def _seed_from_parts(*parts: int) -> int:
    sequence = np.random.SeedSequence([int(part) for part in parts])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _latent_truth(
    X_eval: np.ndarray, dgp: str, latent_beta: np.ndarray
) -> np.ndarray:
    if dgp == "additive_probability":
        return X_eval * latent_beta[None, :]
    return exact_zero_reference_logistic_shapley(X_eval, latent_beta, 0.0)


def runtime_environment() -> dict[str, Any]:
    """Capture portable software versions and explicit thread settings."""

    thread_variables = [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]
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
            "matplotlib": matplotlib.__version__,
            "pyyaml": yaml.__version__,
        },
        "threads": {name: os.environ.get(name) for name in thread_variables},
    }


def run_experiment(cfg: RunConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    dgp_specs = {
        "additive_probability": (
            coefficient_vector(
                cfg.d, cfg.driver_indices, cfg.additive_driver_coefficients
            ),
            cfg.additive_bias_coefficient,
        ),
        "logistic_link_stress": (
            coefficient_vector(
                cfg.d, cfg.driver_indices, cfg.logistic_driver_coefficients
            ),
            cfg.logistic_bias_coefficient,
        ),
    }
    total_scenarios = (
        len(cfg.seeds)
        * len(dgp_specs)
        * len(cfg.bias_multipliers)
        * sum(
            k + 1
            if cfg.contamination_counts is None
            else len(cfg.contamination_counts)
            for k in cfg.ks
        )
    )
    completed = 0

    for seed in cfg.seeds:
        covariate_rng = np.random.default_rng(_seed_from_parts(seed, 11))
        X_train = covariate_rng.uniform(-1.0, 1.0, size=(cfg.n_train, cfg.d))
        X_eval = covariate_rng.uniform(-1.0, 1.0, size=(cfg.n_eval, cfg.d))

        for dgp_index, (dgp, (latent_beta, base_bias)) in enumerate(dgp_specs.items()):
            local_truth = _latent_truth(X_eval, dgp, latent_beta)
            for multiplier_index, bias_multiplier in enumerate(cfg.bias_multipliers):
                bias_coefficient = base_bias * bias_multiplier
                for k in cfg.ks:
                    contamination_counts = (
                        range(k + 1)
                        if cfg.contamination_counts is None
                        else cfg.contamination_counts
                    )
                    for contaminated in contamination_counts:
                        scenario_start = time.perf_counter()
                        scenario_seed = _seed_from_parts(
                            seed,
                            101 + dgp_index,
                            1009 + multiplier_index,
                            k,
                            contaminated,
                        )
                        scenario_rng = np.random.default_rng(scenario_seed)
                        permutation = scenario_rng.permutation(k)
                        polluted_ids = np.sort(permutation[:contaminated])
                        clean_mask = np.ones(k, dtype=bool)
                        clean_mask[polluted_ids] = False

                        proxy_attributions: list[np.ndarray] = []
                        proxy_labels: list[np.ndarray] = []
                        model_rmses: list[float] = []
                        proxy_training_seconds = 0.0
                        proxy_attribution_seconds = 0.0
                        probability_min = 1.0
                        probability_max = 0.0
                        fit_explain_start = time.perf_counter()
                        for proxy in range(k):
                            proxy_bias = bias_coefficient if not clean_mask[proxy] else 0.0
                            p_train = response_probability(
                                X_train,
                                dgp,
                                latent_beta,
                                cfg.bias_index,
                                proxy_bias,
                            )
                            p_eval = response_probability(
                                X_eval,
                                dgp,
                                latent_beta,
                                cfg.bias_index,
                                proxy_bias,
                            )
                            probability_min = min(probability_min, float(p_train.min()))
                            probability_max = max(probability_max, float(p_train.max()))
                            if dgp == "additive_probability":
                                epsilon = cfg.probability_epsilon
                                if p_train.min() < epsilon - 1e-10 or p_train.max() > 1 - epsilon + 1e-10:
                                    raise AssertionError(
                                        "Analytically bounded additive probability escaped its registered bounds."
                                    )
                            label_seed = _seed_from_parts(scenario_seed, proxy, 7919)
                            label_rng = np.random.default_rng(label_seed)
                            y_train = label_rng.binomial(1, p_train)
                            proxy_labels.append(y_train)
                            local_phi, rmse, fit_seconds, explain_seconds = fit_and_explain_proxy(
                                X_train,
                                y_train,
                                X_eval,
                                p_eval,
                                dgp,
                                label_seed,
                            )
                            proxy_attributions.append(local_phi)
                            model_rmses.append(rmse)
                            proxy_training_seconds += fit_seconds
                            proxy_attribution_seconds += explain_seconds
                        proxy_fit_explain_seconds = time.perf_counter() - fit_explain_start

                        proxy_phi = np.stack(proxy_attributions, axis=0)
                        aggregations, rpca_iterations, aggregation_timings = aggregate_local_attributions(
                            proxy_phi,
                            clean_mask,
                            cfg.trim_fraction,
                            cfg.use_rpca,
                        )

                        # Label-first comparator: majority-vote the observed proxy
                        # labels, break exact even-K ties with a registered seed,
                        # then fit and explain one response model.
                        label_first_start = time.perf_counter()
                        label_sum = np.sum(np.stack(proxy_labels, axis=0), axis=0)
                        label_first = (label_sum > k / 2).astype(int)
                        ties = label_sum == k / 2
                        if np.any(ties):
                            tie_rng = np.random.default_rng(
                                _seed_from_parts(scenario_seed, 104729)
                            )
                            label_first[ties] = tie_rng.binomial(1, 0.5, size=int(ties.sum()))
                        latent_p_eval = response_probability(
                            X_eval, dgp, latent_beta, cfg.bias_index, 0.0
                        )
                        (
                            label_first_phi,
                            label_first_rmse,
                            label_first_fit_seconds,
                            label_first_explain_seconds,
                        ) = fit_and_explain_proxy(
                            X_train,
                            label_first,
                            X_eval,
                            latent_p_eval,
                            dgp,
                            _seed_from_parts(scenario_seed, 130363),
                        )
                        aggregations["label_first_majority"] = label_first_phi
                        aggregation_timings["label_first_majority"] = (
                            time.perf_counter() - label_first_start
                        )
                        scenario_seconds = time.perf_counter() - scenario_start
                        common = {
                            "seed": seed,
                            "profile": cfg.profile,
                            "n_train": cfg.n_train,
                            "n_eval": cfg.n_eval,
                            "d": cfg.d,
                            "driver_indices": ",".join(str(v) for v in cfg.driver_indices),
                            "latent_driver_coefficients": ",".join(
                                f"{latent_beta[index]:.12g}" for index in cfg.driver_indices
                            ),
                            "bias_index": cfg.bias_index,
                            "base_bias_coefficient": float(base_bias),
                            "probability_epsilon": cfg.probability_epsilon,
                            "trim_fraction": cfg.trim_fraction,
                            "rpca_enabled": cfg.use_rpca,
                            "shapley_reference": "zero_population_mean",
                            "dgp": dgp,
                            "bias_multiplier": float(bias_multiplier),
                            "bias_coefficient": float(bias_coefficient),
                            "K": k,
                            "contaminated": contaminated,
                            "clean": k - contaminated,
                            "contamination_fraction": float(contaminated / k),
                            "strict_clean_majority": bool((k - contaminated) > k / 2),
                            "at_or_beyond_half_contamination": bool(2 * contaminated >= k),
                            "polluted_proxy_ids": ",".join(str(v) for v in polluted_ids.tolist()),
                            "single_proxy_is_contaminated": bool(not clean_mask[0]),
                            "model_probability_rmse": float(np.mean(model_rmses)),
                            "label_first_probability_rmse": label_first_rmse,
                            "label_first_fit_seconds": label_first_fit_seconds,
                            "label_first_attribution_seconds": label_first_explain_seconds,
                            "response_probability_min": probability_min,
                            "response_probability_max": probability_max,
                            "rpca_iterations": rpca_iterations,
                            "proxy_training_seconds": proxy_training_seconds,
                            "proxy_attribution_seconds": proxy_attribution_seconds,
                            "proxy_fit_explain_seconds": proxy_fit_explain_seconds,
                            "scenario_runtime_seconds": scenario_seconds,
                        }
                        for aggregator, local_estimate in aggregations.items():
                            row = dict(common)
                            row["aggregator"] = aggregator
                            row["aggregation_runtime_seconds"] = aggregation_timings[aggregator]
                            row["runtime_seconds"] = (
                                proxy_fit_explain_seconds + aggregation_timings[aggregator]
                            )
                            row.update(
                                evaluate_local_consensus(
                                    local_estimate,
                                    local_truth,
                                    cfg.driver_indices,
                                    cfg.bias_index,
                                )
                            )
                            raw_rows.append(row)
                        completed += 1
                        if completed == 1 or completed % max(1, total_scenarios // 10) == 0:
                            print(
                                f"  progress {completed:>4d}/{total_scenarios} "
                                f"({100 * completed / total_scenarios:5.1f}%)",
                                flush=True,
                            )

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - start,
        "aggregation_order": (
            "aggregate signed local attributions across proxies for each user, "
            "then compute mean_user(abs(local consensus))"
        ),
        "reference_distribution": (
            "degenerate zero reference; zero is the population mean of independent "
            "Uniform[-1,1] covariates"
        ),
        "dgp_claims": {
            "additive_probability": (
                "well-specified additive proxy response; analytically bounded away from 0 and 1"
            ),
            "logistic_link_stress": (
                "nonlinear signal-dependent contamination stress within the theorem's "
                "arbitrary attribution-level contamination class"
            ),
        },
        "confidence_intervals": (
            "two-sided 95% Student-t intervals across seed-level metrics; evaluation users "
            "are not treated as independent replicates"
        ),
        "baselines": sorted({row["aggregator"] for row in raw_rows}),
        "metrics": METRIC_NAMES + RUNTIME_METRIC_NAMES,
        "oracle_clean_availability": (
            "reported only when at least one clean proxy exists; omitted at m=K"
        ),
        "environment": runtime_environment(),
    }
    return raw_rows, metadata


def summarize_rows(raw_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    group_columns = [
        "dgp",
        "bias_multiplier",
        "bias_coefficient",
        "K",
        "contaminated",
        "clean",
        "contamination_fraction",
        "strict_clean_majority",
        "at_or_beyond_half_contamination",
        "aggregator",
    ]
    frame = pd.DataFrame(raw_rows)
    summary_rows: list[dict[str, Any]] = []
    for group_key, group in frame.groupby(group_columns, dropna=False, sort=True):
        row = dict(zip(group_columns, group_key))
        row["seed_count"] = int(group["seed"].nunique())
        row["model_probability_rmse_mean"] = float(group["model_probability_rmse"].mean())
        for metric in METRIC_NAMES + RUNTIME_METRIC_NAMES:
            values = group[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            count = int(values.size)
            row[f"{metric}_n"] = count
            if count == 0:
                mean = standard_deviation = standard_error = ci_low = ci_high = float("nan")
            else:
                mean = float(np.mean(values))
                if count >= 2:
                    standard_deviation = float(np.std(values, ddof=1))
                    standard_error = standard_deviation / math.sqrt(count)
                    critical = float(stats.t.ppf(0.975, df=count - 1))
                    ci_low = mean - critical * standard_error
                    ci_high = mean + critical * standard_error
                else:
                    standard_deviation = standard_error = ci_low = ci_high = float("nan")
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = standard_deviation
            row[f"{metric}_se"] = standard_error
            row[f"{metric}_ci95_low"] = ci_low
            row[f"{metric}_ci95_high"] = ci_high
        summary_rows.append(row)
    return summary_rows


def validate_run_results(
    cfg: RunConfig,
    raw_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Fail fast on incomplete grids, non-finite metrics, or schema drift."""

    raw = pd.DataFrame(raw_rows)
    summary = pd.DataFrame(summary_rows)
    if raw.empty or summary.empty:
        raise AssertionError("The experiment produced an empty raw or summary table.")
    expected_baselines = {
        "single_proxy",
        "mean",
        "scaled_huber",
        "trimmed_mean",
        "winsorized_mean",
        "median_of_means",
        "median",
        "rank1_svd",
        "label_first_majority",
        "oracle_clean",
    }
    if cfg.use_rpca:
        expected_baselines.add("rpca")
    observed_baselines = set(raw["aggregator"].unique())
    if observed_baselines != expected_baselines:
        raise AssertionError(
            f"Baseline grid mismatch: expected {sorted(expected_baselines)}, "
            f"observed {sorted(observed_baselines)}."
        )
    duplicate_columns = ["seed", "dgp", "bias_multiplier", "K", "contaminated", "aggregator"]
    if raw.duplicated(duplicate_columns).any():
        raise AssertionError("Duplicate seed/scenario/aggregator rows were produced.")
    numeric_columns = METRIC_NAMES + RUNTIME_METRIC_NAMES
    if not np.isfinite(raw[numeric_columns].to_numpy(dtype=float)).all():
        raise AssertionError("Non-finite metric or runtime detected in the raw table.")
    invalid_oracle = raw[
        raw["aggregator"].eq("oracle_clean") & raw["contaminated"].eq(raw["K"])
    ]
    if not invalid_oracle.empty:
        raise AssertionError("oracle_clean must be omitted when no clean proxy exists.")
    additive = raw[raw["dgp"].eq("additive_probability")]
    if (
        additive["response_probability_min"].min() < cfg.probability_epsilon - 1e-10
        or additive["response_probability_max"].max()
        > 1.0 - cfg.probability_epsilon + 1e-10
    ):
        raise AssertionError("The additive response escaped its registered probability bounds.")
    if not summary["seed_count"].eq(len(cfg.seeds)).all():
        raise AssertionError("At least one summary cell is missing a registered seed.")
    if len(cfg.seeds) >= 2:
        ci_columns = [f"{metric}_ci95_low" for metric in METRIC_NAMES]
        if not np.isfinite(summary[ci_columns].to_numpy(dtype=float)).all():
            raise AssertionError("A seed-level confidence interval is missing or non-finite.")


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
        json.dump(_json_safe(payload), handle, indent=2, ensure_ascii=False, allow_nan=False)


def configured_output_paths(cfg: RunConfig) -> dict[str, Path]:
    output_dir = Path(cfg.output_dir)
    candidates = {
        "raw_csv": output_dir / "gamma_revised_raw.csv",
        "raw_json": output_dir / "gamma_revised_raw.json",
        "summary_csv": output_dir / "gamma_revised_summary.csv",
        "summary_json": output_dir / "gamma_revised_summary.json",
        "metadata_json": output_dir / "gamma_revised_metadata.json",
        "figure": output_dir / "gamma_revised_main.png",
    }
    enabled = {"metadata_json": candidates["metadata_json"]}
    if cfg.save_raw_csv:
        enabled["raw_csv"] = candidates["raw_csv"]
    if cfg.save_raw_json:
        enabled["raw_json"] = candidates["raw_json"]
    if cfg.save_summary_csv:
        enabled["summary_csv"] = candidates["summary_csv"]
    if cfg.save_summary_json:
        enabled["summary_json"] = candidates["summary_json"]
    if cfg.make_plot:
        enabled["figure"] = candidates["figure"]
    return enabled


def require_explicit_overwrite(cfg: RunConfig) -> None:
    existing = [path for path in configured_output_paths(cfg).values() if path.exists()]
    if existing and not cfg.overwrite:
        formatted = "\n    ".join(str(path) for path in existing)
        raise FileExistsError(
            "Refusing to silently replace existing registered outputs. Re-run with "
            f"--overwrite after auditing them:\n    {formatted}"
        )


def make_main_figure(summary_rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    frame = pd.DataFrame(summary_rows)
    if frame.empty:
        return
    available_strengths = sorted(frame["bias_multiplier"].unique())
    selected_strength = min(available_strengths, key=lambda value: abs(float(value) - 1.0))
    frame = frame[np.isclose(frame["bias_multiplier"], selected_strength)]
    available_ks = sorted(int(value) for value in frame["K"].unique())
    selected_k = min(available_ks, key=lambda value: abs(value - 5))
    frame = frame[frame["K"] == selected_k]
    dgp_order = ["additive_probability", "logistic_link_stress"]
    metrics = ["local_normalized_l1", "bias_leakage"]
    labels = {
        "additive_probability": "Additive bounded-probability DGP",
        "logistic_link_stress": "Logistic-link nonlinear-contamination stress",
        "local_normalized_l1": "Normalized local $L_1$ error",
        "bias_leakage": "Bias/driver attribution ratio",
    }
    aggregators = [
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
    ]
    colors = plt.get_cmap("tab20").colors
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), sharex=True)
    for column, dgp in enumerate(dgp_order):
        dgp_frame = frame[frame["dgp"] == dgp]
        for row_index, metric in enumerate(metrics):
            axis = axes[row_index, column]
            axis.axvspan(0.5, 1.01, color="#eeeeee", alpha=0.8, zorder=0)
            axis.axvline(0.5, color="#555555", linestyle="--", linewidth=1.0)
            for agg_index, aggregator in enumerate(aggregators):
                subset = dgp_frame[dgp_frame["aggregator"] == aggregator]
                if subset.empty:
                    continue
                averaged = subset.sort_values("contamination_fraction")
                x_values = averaged["contamination_fraction"].to_numpy(dtype=float)
                means = averaged[f"{metric}_mean"].to_numpy(dtype=float)
                axis.plot(
                    x_values,
                    means,
                    marker="o",
                    markersize=3.2,
                    linewidth=1.25,
                    color=colors[agg_index % len(colors)],
                    label=aggregator,
                )
                ci_low = averaged[f"{metric}_ci95_low"].to_numpy(dtype=float)
                ci_high = averaged[f"{metric}_ci95_high"].to_numpy(dtype=float)
                finite_ci = np.isfinite(ci_low) & np.isfinite(ci_high)
                if np.any(finite_ci):
                    axis.fill_between(
                        x_values[finite_ci],
                        ci_low[finite_ci],
                        ci_high[finite_ci],
                        color=colors[agg_index % len(colors)],
                        alpha=0.10,
                        linewidth=0,
                    )
            axis.grid(alpha=0.22, linewidth=0.6)
            axis.set_xlim(-0.02, 1.02)
            if column == 0:
                axis.set_ylabel(labels[metric])
            if row_index == 0:
                axis.set_title(labels[dgp])
            if row_index == 1:
                axis.set_xlabel("Fraction of contaminated proxies")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        f"Local-first aggregation across the strict-majority boundary "
        f"(K={selected_k}, bias multiplier={selected_strength:g}; bands are seed-level 95% CIs)",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_outputs(
    cfg: RunConfig,
    raw_yaml: Mapping[str, Any],
    raw_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Path]:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = configured_output_paths(cfg)
    if "raw_csv" in paths:
        pd.DataFrame(raw_rows).to_csv(paths["raw_csv"], index=False)
    if "summary_csv" in paths:
        pd.DataFrame(summary_rows).to_csv(paths["summary_csv"], index=False)
    if "raw_json" in paths:
        _write_json(paths["raw_json"], list(raw_rows))
    if "summary_json" in paths:
        _write_json(paths["summary_json"], list(summary_rows))

    resolved = asdict(cfg)
    resolved["output_dir"] = _artifact_safe_path(Path(cfg.output_dir))
    config_blob = json.dumps(_json_safe(resolved), sort_keys=True).encode("utf-8")
    metadata = dict(metadata)
    metadata.update(
        {
            "resolved_config": resolved,
            "config_sha256": hashlib.sha256(config_blob).hexdigest(),
            "yaml_snapshot": raw_yaml,
            "raw_row_count": len(raw_rows),
            "summary_row_count": len(summary_rows),
        }
    )
    _write_json(paths["metadata_json"], metadata)
    if "figure" in paths:
        make_main_figure(summary_rows, paths["figure"])
    return paths


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    cfg, raw_yaml = resolve_config(args)
    require_explicit_overwrite(cfg)
    print("=" * 78)
    print("Revised gamma core: local-first robust multi-proxy attribution")
    print("=" * 78)
    print(
        f"  profile={cfg.profile} seeds={list(cfg.seeds)} K={list(cfg.ks)} "
        f"n_train={cfg.n_train} n_eval={cfg.n_eval} d={cfg.d}"
    )
    contamination_label = (
        "all integer counts 0,...,K"
        if cfg.contamination_counts is None
        else list(cfg.contamination_counts)
    )
    print(f"  contamination_counts={contamination_label}")
    print(
        f"  bias_multipliers={list(cfg.bias_multipliers)} "
        f"RPCA={'on' if cfg.use_rpca else 'off'}"
    )
    print(
        "  estimator order: signed local proxy attributions -> per-user consensus "
        "-> mean(abs(consensus))"
    )
    raw_rows, metadata = run_experiment(cfg)
    summary_rows = summarize_rows(raw_rows)
    validate_run_results(cfg, raw_rows, summary_rows)
    paths = save_outputs(cfg, raw_yaml, raw_rows, summary_rows, metadata)
    print("\nCompleted")
    print(f"  elapsed: {metadata['elapsed_seconds']:.2f} seconds")
    for label, path in paths.items():
        if label == "figure" and not cfg.make_plot:
            continue
        print(f"  {label:13s}: {path}")


if __name__ == "__main__":
    main()
