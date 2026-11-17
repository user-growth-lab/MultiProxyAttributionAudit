# Anonymous Repository: Multi-Proxy Preference Attribution

This optional repository supports verification of the paper's methods and
evidence. The paper and its technical appendix are self-contained; no paper
claim requires a reviewer to execute code or inspect this repository.

The repository contains:

1. controlled-benchmark source code, tests, public configuration, and frozen
   discovery, confirmation, and sensitivity outputs;
2. one anonymous technical appendix with complete proofs, extended protocols,
   and the full human-readable result tables omitted from the page-limited
   manuscript;
3. selected machine-readable Amazon aggregate results corresponding to those
   tables; and
4. a standard-library verifier for the file closure, hashes, cohort separation,
   result cardinalities, and registered numerical facts.

## Verify the snapshot

From the extracted repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 verify_repository.py
```

The verifier uses only the Python standard library.

## Install and run the controlled tests

The commands below keep environment and cache files outside the repository:

```bash
python3 -m venv /tmp/wsdm27-venv
source /tmp/wsdm27-venv/bin/activate
python3 -m pip install -r requirements.txt
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
```

## Lightweight smoke runs

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m src.experiments.run_gamma_revised \
  --config configs/gamma.yaml --profile quick --seeds 0 --ks 3 \
  --n-train 300 --n-eval 12 --bias-multipliers 1 --no-plot \
  --output-dir /tmp/wsdm27-gamma-smoke

PYTHONDONTWRITEBYTECODE=1 python3 -m src.experiments.run_sensitivity_revised \
  --config configs/gamma.yaml --profile quick --seeds 0 \
  --n-eval 12 --n-anchor 12 --output-dir /tmp/wsdm27-sensitivity-smoke
```

## Registered reruns and paired analysis

The discovery grid, disjoint confirmation cohort, sensitivity suite, and paired
analysis can be rerun without replacing the frozen reference outputs:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m src.experiments.run_gamma_revised \
  --config configs/gamma.yaml --profile paper \
  --output-dir /tmp/wsdm27-discovery-paper

PYTHONDONTWRITEBYTECODE=1 python3 -m src.experiments.run_gamma_revised \
  --config configs/gamma.yaml --profile paper \
  --seeds 10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29 \
  --ks 5 --contamination-counts 2 --bias-multipliers 1 --no-plot \
  --output-dir /tmp/wsdm27-confirmation-paper

PYTHONDONTWRITEBYTECODE=1 python3 -m src.experiments.run_sensitivity_revised \
  --config configs/gamma.yaml --profile paper \
  --output-dir /tmp/wsdm27-sensitivity-paper

PYTHONDONTWRITEBYTECODE=1 python3 -m src.experiments.analyze_gamma_paired \
  --discovery registered_results/synthetic_discovery/gamma_revised_raw.csv \
  --confirmation registered_results/paired_confirmation/confirmatory_run/gamma_revised_raw.csv \
  --protocol registered_results/paired_confirmation/protocol_frozen.json \
  --output-dir /tmp/wsdm27-paired-analysis
```

The archived output bytes are fixed reference artifacts. New numerical reruns
should reproduce the reported conclusions, but exact byte identity across
operating systems and numerical-library builds is not asserted.

## Evidence boundary

The Amazon source dataset is publicly hosted upstream and is not copied into
this review bundle. The repository provides selected aggregate results only;
it contains no raw reviews, user identifiers, prompts, raw model responses,
parsed user profiles, user-level labels or attributions, fitted study state, or
model weights. Consequently, exact end-to-end regeneration of the Amazon
numbers is not supported by this repository.

The future-rated surface is an observable comparator, not latent-preference
ground truth. The LLM views are not assumed clean, and the controlled
benchmark's strict-majority guarantees apply only under their stated premises.

## License

Original code and documentation are released under the MIT License. Third-party
data, models, dependencies, fonts, and conference materials retain their own
terms; see `THIRD_PARTY_NOTICES.md`.
