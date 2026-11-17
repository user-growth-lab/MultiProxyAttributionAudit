# Data Access and Release Boundary

This boundary applies to the MultiProxyAttributionAudit public release. Moving
the approved review snapshot to a named repository does not expand the data or
experimental state included in the package.

Amazon Reviews 2023 is publicly accessible from its upstream project page.
This repository does not claim that the upstream dataset is confidential or
that ordinary access to it requires de-identification.

## Included here

- Synthetic controlled-benchmark code, tests, public configuration, and frozen
  seed-level outputs.
- One prebuilt technical appendix, preserved from the anonymous review version,
  containing complete proofs,
  extended protocols, diagnostics, and human-readable aggregate tables.
- Selected machine-readable Amazon aggregate results used in the paper.

## Not redistributed here

- The upstream Amazon review and metadata objects. Readers can obtain the
  public source independently from `https://amazon-reviews-2023.github.io/`.
- User identifiers or any user-level derivatives created by this study,
  including prompts, raw model responses, parsed profiles, formal roles,
  labels, probabilities, attributions, calibration state, and held-out records.
- Model weights, dependency packages, caches, local paths, repository history,
  internal reviews, approval records, and revision history.
- The main-paper PDF and manuscript LaTeX sources.

Omitting the upstream objects keeps the optional evidence release focused and
avoids re-publishing third-party data under the repository's MIT license. The
user-level study derivatives are also excluded to minimize disclosure risk.
These choices do not prevent a reader from reading every core claim, proof,
protocol, and reported result in the manuscript plus technical appendix.

The controlled benchmark is rerunnable with this package. The selected Amazon
aggregates support inspection of reported results, but the package does not
support exact end-to-end regeneration of the Amazon study. Independently
obtaining the public source data does not supply the omitted study state.
