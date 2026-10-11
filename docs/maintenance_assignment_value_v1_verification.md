# Mac verification before lab handoff

Engineering verification on2026-10-11, branch
`codex/maintenance-assignment-value-v1`, parent source commit
`de47c39379f1ae1e8c4d0e64745b90cb6322c550`.

- Entire repository suite:734 passed, one existing TqdmExperimentalWarning,
  224.06seconds. Includes18 new tests for complete PM assignment classes,
  exact CM/machine-set isolation, minimum-cost sign, source detachment,
  machine/technician permutation equivariance, tied decoding, condition-local
  shuffle, all21-root scalar replay at original ticks0/42/47, own-policy
  continuation, budget/seed/physical exclusion, primary grids, empty cluster
  handling, tampered-source rejection, and overwrite refusal.
- Final CLI smoke:
  `artifacts/maintenance_assignment_value_v1_smoke_20261011T141235Z`;
  COMPLETED/PASS,37.0696seconds, CPU/one thread, real H48+tail12 and hidden32.
  Source run `maintenance_assignment_scoring_v1_full_cpu_20261011T131859Z`.
- All3 source checkpoints verified;957 source artifact hashes/byte counts
  verified. Source environment/policy snapshot unchanged. Frozen source PPO
  parameter digest unchanged after labels/training/deployment diagnostics.
- Source trajectories:2 training,2 development,2 test;288 source decisions.
  Four training states, four development states, four assignment-ranking states,
  four common timing states. Two learned heads, four matched optimizer updates,
  six closed-loop test episodes,288 base+72 tail decisions,360 logged decisions.
- All376 forecast futures /21,216 physical transitions scalar-replayed exactly,
  including base/tail costs, components, actions, cutoff and terminal states.
  This covers all smoke label/ranking/timing phases and value-head continuation.
- Head/dataset checkpoint roundtrips, source binding, feasibility/coordination,
  count/forecast-budget audits, timing clock and progress display pass.
- Final smoke256 artifact hashes independently verified,6,235,096 bytes excluding
  manifest and console log. Final handoff commands pass `bash -n`.
- Metadata-only exclusion blocks67,727 previous physical signatures. All198
  new full/smoke/train/dev/test signatures disjoint. Raw and derived RNG audits
  PASS. New FULL outcomes have not been run/opened locally.

No learning-quality or scientific-performance conclusion follows from smoke.
The user must run the fixed full protocol in the lab and transfer results before
analysis. Full artifacts record actual label counts/optimizer updates and
per-phase forecast compute rather than treating simulator forecasts as PPO
training steps. See the plan/source contract and handoff documents.
