# B4 confirmation opening guard

This branch adds only a confirmation authorization gate and independent
auditors; it does not alter the live B4 DEV learner, physics, student
budgets, seed registry or prospective thresholds.

The confirmation runner requires a completed B4 development artifact with
internal PASS, registered DEV cost/health screen PASS and `test_opened=false`.
It binds the exact protocol and seed registry plus every saved learner and
physics Python module from the DEV source snapshot. The runner file may
differ only to add this gate. It also requires an independent DEV audit
report with `status=PASS`, `verdict=DEV_GATE_PASS`, sealed TEST, exact
DEV artifact path and SHA-256 bindings to the DEV manifest and independent
verifier. The confirmation manifest records those bindings before any
scientific TEST row is generated.

The independent DEV auditor recomputes source/output hashes, complete
training/DEV seed grids, paired physical tapes, checkpoint selection,
selected cost/health and the prospective screen. The independent TEST
verifier replays both full and mask-only trajectories, legal reservation
sets, prefix residual arithmetic, physical transitions and the registered
inference. A smoke artifact can exercise these gates and verifiers but
cannot authorize B4 scientific confirmation. B1/B2/B3 and their TEST
panels remain sealed.
