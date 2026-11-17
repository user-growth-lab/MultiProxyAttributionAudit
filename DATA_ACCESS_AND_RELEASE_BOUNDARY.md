# Data Access and Release Boundary

Amazon Reviews 2023 is publicly accessible from its upstream project page.
This repository does not claim that the upstream dataset is confidential or
that ordinary access to it requires de-identification.

## Included here

- Synthetic controlled-benchmark code, tests, public configuration, and frozen
  seed-level outputs.
- One prebuilt anonymous technical appendix containing complete proofs,
  extended protocols, diagnostics, and human-readable aggregate tables.
- Selected machine-readable Amazon aggregate results used in the paper.

## Not redistributed here

- The upstream Amazon review and metadata objects. Reviewers can obtain the
  public source independently from `https://amazon-reviews-2023.github.io/`.
- User identifiers or any user-level derivatives created by this study,
  including prompts, raw model responses, parsed profiles, formal roles,
  labels, probabilities, attributions, calibration state, and held-out records.
- Model weights, dependency packages, caches, local paths, repository history,
  internal reviews, approval records, and revision history.

Omitting the upstream objects keeps the optional review archive focused and
avoids re-publishing third-party data under the repository's MIT license. The
user-level study derivatives are also excluded to minimize disclosure risk.
These choices do not prevent a reviewer from reading every core claim, proof,
protocol, and reported result in the manuscript plus technical appendix.
