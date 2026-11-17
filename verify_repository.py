#!/usr/bin/env python3
"""Verify the anonymous repository using only the Python standard library."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath


EXPECTED_FACTS = {
    "discovery": {
        "raw_rows": 14940,
        "summary_rows": 1494,
        "scenario_count": 138,
        "seeds": list(range(10)),
    },
    "confirmation": {
        "method_rows": 440,
        "paired_rows": 400,
        "seeds": list(range(10, 30)),
    },
    "cohort_separation": {"seed_intersection": [], "disjoint": True},
    "sensitivity": {"raw_rows": 640, "summary_rows": 64},
    "rpca": {
        "discovery": {
            "unique_solves": 1380,
            "iteration_min": 20,
            "iteration_max": 31,
            "iteration_cap": 300,
            "iteration_cap_hits": 0,
        },
        "confirmation": {
            "unique_solves": 40,
            "iteration_min": 26,
            "iteration_max": 30,
            "iteration_cap": 300,
            "iteration_cap_hits": 0,
        },
    },
}


def fail(message: str) -> None:
    raise SystemExit(f"verification failed: {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def rpca_facts(rows: list[dict[str, str]], cohort: str) -> dict:
    rpca = [row for row in rows if row.get("aggregator") == "rpca"]
    keys = {
        (row["seed"], row["dgp"], row["bias_multiplier"], row["K"], row["contaminated"])
        for row in rpca
    }
    if len(keys) != len(rpca):
        fail(f"{cohort} RPCA rows do not represent unique solves")
    iterations = [int(float(row["rpca_iterations"])) for row in rpca]
    return {
        "unique_solves": len(keys),
        "iteration_min": min(iterations),
        "iteration_max": max(iterations),
        "iteration_cap": 300,
        "iteration_cap_hits": sum(value >= 300 for value in iterations),
    }


def verify_manifest(root: Path) -> dict:
    try:
        manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read artifact_manifest.json: {exc}")
    if manifest.get("schema_version") != 1:
        fail("unexpected artifact manifest schema")
    actual = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    if any(path.is_symlink() for path in root.rglob("*")):
        fail("symlinks are not allowed")
    for name in actual:
        pure = PurePosixPath(name)
        if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
            fail(f"unsafe path: {name}")
    if manifest.get("allowed_files") != actual:
        fail("manifest allowlist does not equal the released file closure")
    if manifest.get("file_count") != len(actual):
        fail("manifest file count drifted")
    if manifest.get("scientific_facts") != EXPECTED_FACTS:
        fail("manifest scientific facts drifted")

    payload = manifest.get("payload_sha256")
    if not isinstance(payload, dict):
        fail("payload hashes are missing")
    expected_payload = [
        name for name in actual if name not in {"artifact_manifest.json", "MANIFEST.sha256"}
    ]
    if sorted(payload) != expected_payload:
        fail("payload hash closure drifted")
    for name, expected in payload.items():
        if sha256_file(root / name) != expected:
            fail(f"payload hash mismatch: {name}")

    lines = (root / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines()
    parsed = {}
    for line in lines:
        if "  " not in line:
            fail("malformed MANIFEST.sha256 line")
        digest, name = line.split("  ", 1)
        if name in parsed or len(digest) != 64:
            fail("duplicate or malformed checksum entry")
        parsed[name] = digest
    expected_checksums = [name for name in actual if name != "MANIFEST.sha256"]
    if sorted(parsed) != expected_checksums:
        fail("checksum closure drifted")
    for name, expected in parsed.items():
        if sha256_file(root / name) != expected:
            fail(f"checksum mismatch: {name}")
    return manifest


def verify_scientific_facts(root: Path) -> dict:
    discovery_raw = read_csv(root / "registered_results/synthetic_discovery/gamma_revised_raw.csv")
    discovery_summary = read_csv(root / "registered_results/synthetic_discovery/gamma_revised_summary.csv")
    confirmation = read_csv(root / "registered_results/paired_confirmation/confirmatory_run/gamma_revised_raw.csv")
    paired = read_csv(root / "registered_results/paired_confirmation/confirmation_median_pairwise_seed.csv")
    sensitivity_raw = read_csv(root / "registered_results/sensitivity/gamma_sensitivity_revised_raw.csv")
    sensitivity_summary = read_csv(root / "registered_results/sensitivity/gamma_sensitivity_revised_summary.csv")

    discovery_seeds = sorted({int(row["seed"]) for row in discovery_raw})
    confirmation_seeds = sorted({int(row["seed"]) for row in confirmation})
    scenarios = {
        (row["dgp"], row["bias_multiplier"], row["K"], row["contaminated"])
        for row in discovery_raw
    }
    facts = {
        "discovery": {
            "raw_rows": len(discovery_raw),
            "summary_rows": len(discovery_summary),
            "scenario_count": len(scenarios),
            "seeds": discovery_seeds,
        },
        "confirmation": {
            "method_rows": len(confirmation),
            "paired_rows": len(paired),
            "seeds": confirmation_seeds,
        },
        "cohort_separation": {
            "seed_intersection": sorted(set(discovery_seeds) & set(confirmation_seeds)),
            "disjoint": not bool(set(discovery_seeds) & set(confirmation_seeds)),
        },
        "sensitivity": {
            "raw_rows": len(sensitivity_raw),
            "summary_rows": len(sensitivity_summary),
        },
        "rpca": {
            "discovery": rpca_facts(discovery_raw, "discovery"),
            "confirmation": rpca_facts(confirmation, "confirmation"),
        },
    }
    if facts != EXPECTED_FACTS:
        fail(f"scientific facts drifted: {facts!r}")
    if sorted({int(row["seed"]) for row in paired}) != list(range(10, 30)):
        fail("paired confirmation seeds drifted")
    if {row["cohort"] for row in paired} != {"confirmation"}:
        fail("paired confirmation includes an unexpected cohort")

    reporting = json.loads(
        (root / "aggregate_evidence/amazon/amazon_aggregate_results.json").read_text(encoding="utf-8")
    )
    expected_keys = {
        "schema_version", "artifact_type", "protocol_id", "dataset",
        "evidence_boundary", "definitions", "compatible_coverage_exact",
        "llm_empty_profile_audit", "primary_absolute_levels",
        "point_method_descriptive_inventory",
    }
    if set(reporting) != expected_keys:
        fail("Amazon aggregate field closure drifted")
    if reporting.get("artifact_type") != "amazon_aggregate_results":
        fail("Amazon aggregate artifact type drifted")
    dataset = reporting.get("dataset", {})
    if dataset != {
        "name": "Amazon Reviews 2023",
        "source_data_included": False,
        "source_url": "https://amazon-reviews-2023.github.io/",
        "subset": "CDs_and_Vinyl",
    }:
        fail("Amazon dataset declaration drifted")
    boundary = reporting.get("evidence_boundary", {})
    if boundary != {
        "complete_human_readable_tables_in_technical_appendix": True,
        "observable_comparator_not_latent_ground_truth": True,
        "selected_machine_readable_aggregates": True,
        "user_level_source_or_derived_records_included": False,
    }:
        fail("Amazon evidence boundary drifted")
    if len(reporting.get("compatible_coverage_exact", [])) != 8:
        fail("Amazon compatible-coverage rows drifted")
    if len(reporting.get("llm_empty_profile_audit", [])) != 2:
        fail("Amazon LLM audit rows drifted")
    if len(reporting.get("primary_absolute_levels", [])) != 8:
        fail("Amazon primary-level rows drifted")
    inventory = reporting.get("point_method_descriptive_inventory", {})
    if set(inventory) != {"inference", "selection_rule", "rows"} or len(inventory.get("rows", [])) != 40:
        fail("Amazon point-method inventory drifted")
    return facts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = verify_manifest(root)
    facts = verify_scientific_facts(root)
    print(json.dumps({
        "status": "verified",
        "artifact_type": manifest["artifact_type"],
        "file_count": manifest["file_count"],
        "scientific_facts": facts,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
