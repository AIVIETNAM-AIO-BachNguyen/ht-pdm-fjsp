# A1 independent confirmation verifier preparation

The read-only `scripts/verify_teacher_transfer_confirmation.py` independently
checks artifact hashes, selected-policy episode grid, checkpoint metadata,
paired physical shock tape, every scalar physical transition, submitted
assignment, teacher/scratch feasibility at every decoder stage, recomputed
cost gains and crossed bootstrap guards. It additionally verifies the
fixed-final and zero-shot secondary episode grid, physical pairing, cost
normalization and reported means. It does not alter A1 training source or
reopen a sealed panel.

Seven targeted tests pass, including a positive/zero-gain synthetic test of
the independent inference guard. `compileall` and `git diff --check` pass.
The verifier passed on corrected A1 engineering smoke at
`artifacts/technician_teacher_transfer_v1_smoke_20261018T073234Z`, producing
fresh output
`artifacts/technician_teacher_transfer_v1_smoke_confirmation_verifier_20261018T123014Z`.
It replayed 720 physical ticks, 192 matched feasibility states/768 stages,
and validated 12 secondary fixed-final/zero-shot episodes, with zero invalid
assignments. The tiny smoke cost gain is engineering-only. The verifier will
run on scientific confirmation only if A1 DEV passes its registered gate.

The verifier was strengthened before any A1 scientific TEST to check matching
feasibility and remaining-resource summaries on both selected-policy
trajectories. Fresh engineering replay
`artifacts/technician_teacher_transfer_v1_both_trajectory_audit_20261018T132934Z`
PASSed: 192 teacher states, 192 scratch states, 1,536 decoder stages,
720 scalar physical ticks and zero invalid assignments. Only the read-only
verifier changed; learner, protocol, seeds and active DEV source stayed frozen.
