# C1 sealed-confirmation engineering gate

The confirmation runner was implemented in a separate worktree while C1
scientific DEV was still running. It imports the frozen C1 learner and checks
byte-for-byte hashes of the C1 environment, experiment, B2 policy,
distillation and protocol/registry against the DEV artifact before any
scientific TEST. A failed DEV gate or opened DEV TEST is rejected. The only
gate bypass is explicit `--profile smoke`, classified engineering-only.

Thirteen targeted tests pass, including synthetic matched cost/health/latency
inference and rejection of a smoke artifact as scientific DEV.
`compileall` and `git diff --check` pass. Engineering confirmation smoke
`artifacts/technician_diverse_maintenance_confirmation_smoke_20261018T124019Z`
COMPLETED/audit PASS: eight small fits, 48 selected-policy TEST episodes,
2,880 physical ticks, 40 fixed-final/zero-shot secondary episodes, zero
invalid assignments. It froze checkpoint and FCFS reference choices before
opening engineering TEST. Its one-bundle, 384-decision result is not a C1
claim.

The independent read-only verifier was tested on that smoke; final verifier
output is
`artifacts/technician_diverse_maintenance_confirmation_smoke_verifier_20261018T124435Z`.
It PASSed all source/output hashes, independently reconstructed the four
registered physical configurations and request pool, replayed 2,880 scalar
physical transitions and costs, checked 384 checkpoint actions and 336
full/mask-only decoder stages, recomputed the 10-endpoint crossed inference,
and reconciled 40 secondary episodes, all training tapes and teacher/scratch
budgets. It confirmed zero invalid assignments. Scientific C1 confirmation
may run only after the currently running DEV artifact passes its locked gate
and its independent DEV verifier.
