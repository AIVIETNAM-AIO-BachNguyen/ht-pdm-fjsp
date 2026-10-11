# Maintenance decision headroom v1 — engineering verification

Branch: `codex/maintenance-decision-headroom-v1`. Prospective protocol is in
`docs/maintenance_decision_headroom_v1_plan.md`.

## Local gates
- Focused tests: 12 passed in 2.45 seconds.
- Full repository suite: 656 passed, one existing TqdmExperimentalWarning,
  217.13 seconds. No failed tests.
- Actual pinned-checkpoint macOS smoke: `maintenance_decision_headroom_v1_smoke_20261011T102655Z`,
  COMPLETED / PASS, 10.272 seconds.
- Four new states, four live trajectories,192 live transitions; six unique
  roots across states,864 forecast transitions,576 tail transitions and240
  continued-RL decision rows. All root/scenario scalar replays passed.
- Source audit: all85 historical artifact hashes and all10 frozen model
  payloads verified. Smoke evaluates only the first model.
- Fresh physical audit:52179 excluded signatures, two fresh smoke signatures;
  new full physical profiles generated as metadata only for exclusion,
  no new full outcome evaluated. Full and smoke physical signatures disjoint.
- Immutable weights, checkpoint roundtrip, feasible root/future actions,
  original H48+L12 clock, and independent selection/evaluation roles passed.
- Independently reloaded raw arrays to reconstruct EVERY selected index,
  held-out mean and paired MC SE, base/tail split and four cost components.
- Verified all83 output hashes and byte counts; smoke artifacts total
  2888949 bytes, excluding console log.
- Smoke manifest records inherited parent commit and dirty working tree because
  workflow requires verifying code before committing. The exact executable
  runner/configuration hashes below match its saved source snapshot. Final
  full run requires a clean checkout and records its own committed version.

## Verified executable/configuration hashes
- `src/ht_pdm_fjsp/maintenance_decision_headroom.py`: `65ae5651f4a126178272ef2be915f670ef81cc69562ae7a1e1733fd477eaf2a4`
- `src/ht_pdm_fjsp/maintenance_headroom_forks.py`: `e180f7211ae0d26abc815aa815763aa597a9d80c4d5cc4a8612f81e00ff792ba`
- `configs/maintenance_decision_headroom_v1_seed_registry.json`: `e263bbbabf38c4b01d6c1fd45455b8b4fd4bc3ab93e2ce0748e07270c88ae4c2`
- `configs/maintenance_decision_headroom_v1_sources.json`: `9cf42b3c1c5744c89ac478e000c929be1c916abafdd93b745579b97c22950a0d`

## Full handoff scope
No training. Ten frozen policies, eight fresh paired profile identities,
640 live trajectories,2560 fixed states,32 selection scenarios and64 independent
held-out scenarios. Hard forecast cap:26542080 simulator transitions.
Expected artifact scale is tens of MB (approximately20–80MB depending on root
coverage/compression); CPU execution time is not benchmarked on the lab here.
All primary intervals and correction gains await the user-run full experiment.
Mac smoke cannot support any scientific performance claim.

Run directly in the lab terminal following `docs/experiment_workflow.md`.
Dependency: `artifacts/maintenance_timing_diagnostic_v1_full_cpu_20261011T072307Z`.
No SSH, lab execution, or result transfer was performed by the assistant.
