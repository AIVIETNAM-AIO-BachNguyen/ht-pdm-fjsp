# Prefix mechanism v1 engineering gate

Locked on 2026-10-18 UTC before development FULL. The prospective protocol and
seed registry were written before the engineering smoke. No development or
confirmation outcome has been opened.

- New graph arms: full prefix + mask, mask-only, and no-reservation. Full and
  mask-only have identical 82,819 parameters and shared initialization, teacher
  bank, augmentation, training tapes, feasible masks, remaining-resource
  summaries and public order. Test assertions verify these stage by stage.
- Targeted tests: 4 passed. Full repository suite: **1,034 passed**, one
  third-party tqdm warning, 705.73 seconds. `compileall` and `git diff --check`
  passed.
- Engineering smoke:
  `artifacts/technician_prefix_mechanism_v1_smoke_20261018T065602Z`.
  Manifest COMPLETED, audit PASS, 8 models, 40 episodes, 2,400 scalar physical
  transitions replayed, 96 full-policy decisions with full/mask-only stage
  feasibility equality, 105 nonzero full-prefix stages, zero invalid
  assignments. Runtime 12.82 seconds. Smoke quality is excluded from scientific
  selection and inference.
- Next step after push: development FULL on N4/N8 and its registered gate.
  Confirmation TEST remains sealed until that gate passes. Fresh artifact
  directories only.
