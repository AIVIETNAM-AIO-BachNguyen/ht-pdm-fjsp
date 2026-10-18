# C1 independent development audit preparation

The read-only verifier `scripts/audit_diverse_maintenance_development.py`
was added after the C1 scientific development process had started. It does
not edit the locked protocol, registry, experiment package or active run.
It independently checks hashes, exact trained/DEV/heuristic grids, paired
physical tapes, teacher bank and scratch provenance, PPO budgets,
checkpoint choices, request conservation, cost components, learning AUC,
fixed-final cost, DEV-selected FCFS/risk reference, cost gains and health
guards. A development artifact with opened scientific TEST is rejected.

The verifier passed on the final C1 engineering smoke
`artifacts/technician_diverse_maintenance_v1_smoke_20261018T122616Z`;
its own fresh output is
`artifacts/technician_diverse_maintenance_v1_smoke_independent_audit_20261018T122818Z`.
It will be run again on the scientific development artifact after completion.
Full confirmation still requires a separate read-only scalar decision replay
and crossed-bootstrap verifier before any C claim.
