# B2 confirmation source guard

Prepared in an isolated worktree while B2 scientific DEV was running. The
confirmation runner now rejects an opened or failed DEV artifact, changed
prospective protocol or registry, or changed package source files in the DEV
snapshot. The runner itself gains only this pre-TEST guard; no learner, policy,
physics, seed, analysis or budget code changes. The guard was exercised against
the live B2 DEV source snapshot and passed without reading DEV outcomes.

Focused regression suite: 7 passed. `compileall` and `git diff --check` passed.
Fresh engineering smoke:
`artifacts/technician_relational_prefix_v2_confirmation_guard_smoke_20261018T131814Z`,
COMPLETED/audit PASS, eight small fits and zero invalid assignments. Smoke
quality is engineering-only. Scientific B2 TEST remains sealed and may run
only if DEV passes its registered gate and the independent DEV audit agrees.

The independent confirmation verifier now checks full/mask-only feasibility,
remaining-resource summaries and zeroed mask-only prefix on **both** arms'
selected-policy trajectories, not only full-policy trajectories. Fresh replay
of the engineering smoke at
`artifacts/technician_relational_prefix_v2_both_trajectory_audit_20261018T132751Z`
PASSed: 384 full states, 384 mask-only states, 2,304 decoder stages,
2,400 scalar physical ticks and zero invalid assignments. This strengthens
trajectory coverage without changing any learner, protocol or scientific
outcome.
