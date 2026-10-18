# A1 independent DEV audit preparation

The read-only verifier `scripts/audit_teacher_transfer_development.py` was
implemented while A1 DEV FULL was already running. It does not modify the
experiment package, prospective protocol, seed registry or run artifact.
It checks source/output hashes, exact DEV episode grid and checkpoint
selection, matching initial tensors and training tapes, fixed PPO/bank
budgets, reward/cost reconciliation, learning AUC, fixed-final cost, health
and the registered DEV gate from raw CSV. It refuses a scientific development
artifact whose TEST has been opened.

The verifier passed on the corrected A1 engineering smoke at
`artifacts/technician_teacher_transfer_v1_smoke_20261018T073234Z`; its own
fresh output is
`artifacts/technician_teacher_transfer_v1_smoke_independent_audit_20261018T121922Z`.
The first verifier smoke attempt was preserved as a technical failure:
it read smoke's inferential `summary.json` instead of its
`learning_summary.json`. This was a verifier-input bug, not an A1 training
failure. The corrected verifier uses `learning_summary.json` for both smoke
and development and produced PASS. A1 DEV will be audited after completion;
the independent verifier is not a replacement for confirmation TEST replay.
