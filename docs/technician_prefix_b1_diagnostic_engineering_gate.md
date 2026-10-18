# B1 prefix-use diagnostic engineering gate

Locked after B1 DEV failure and before full descriptive diagnostic. B1 TEST
remains sealed. This is analysis of opened DEV only, with no new learning.

- Independent DEV recomputation:
  `artifacts/technician_prefix_mechanism_v1_development_independent_audit_20261018T120243Z`:
  source/output hashes, exact 3,072 DEV grid, checkpoint choices, physical
  population pairing, budgets, health and gain formula PASS; gate FAIL.
- Diagnostic protocol:
  `docs/technician_prefix_b1_diagnostic_plan.md`, all three B1 training
  replicas and 32 opened DEV seeds at N4/N8, one-hour hardstop. Smoke uses
  only the first two opened DEV seeds per replica.
- `compileall`/whitespace PASS and five prefix unit tests PASS. Smoke artifact
  `artifacts/technician_prefix_b1_diagnostic_smoke_20261018T120508Z`:
  COMPLETED, 12 paired episodes, 576/1,152 stage-mask checks at N4/N8,
  3/288 and 60/288 full-trajectory states with different same-weight
  mask-only actions. These are technical/descriptive values from six episodes
  per scale, not a selection or inference result.
- Next after commit/push: run full DEV-only diagnostic, then record the
  mechanistic explanation and, if warranted, one bounded follow-up with fresh
  seeds and unchanged failed B1 verdict.
