# Exact capacity witness on two synthetic states

Date: 2026-10-22. Pre-outcome protocol and source commit:
`docs/machine_capacity_exact_witness_plan_vi.md` at `0b00e59`. The manual
run `machine_capacity_exact_witness_20261022T045403097777Z` has terminal
`COMPLETED` manifest, source/protocol hashes and 4,260 enumerated transitions.
The separate request-free physical dynamic-program audit
`machine_capacity_exact_audit_20261022T045431109725Z` is `PASS`: maximum
absolute first-root Q discrepancy is 7.11e-15. Both calculations use the
same physical kernel, so this is an independent decision/request-layer
recalculation, not an independent physics implementation.

The target machine is operating at age 3 with an existing PM request in
both states. The worker is idle and can repair either machine. Machine1 is
healthy with a PM request in A and failed with an automatic CM request in B.
The service durations are 3 ticks for the target and 1 for machine1. Exact
six-tick expected economic cost under optimal matching continuation is:

| First matching | A: healthy competitor | B: failed competitor |
|---|---:|---:|
| Serve target machine0 | **26.00** | 45.00 |
| Serve machine1 | 40.20 | **41.20** |
| WAIT | 40.40 | 53.61 |

The selected first service switches from machine0 to machine1; the strict
runner-up gaps are 14.20 and 3.80 cost units. This passes the locked witness
gate. It shows that the coordinator's service decision can depend on current
competing demand even when the target health is unchanged. It does **not**
show that a machine should delay a free PM request: both states expose all
requests, and the coordinator can choose which to serve or WAIT.

This is one manually constructed N2/K1, H6 state pair with exact Bernoulli
event integration, no sampled physical episode panel and no learned policy.
No distributional gain, training-seed consistency, scaling or MARL necessity
can be inferred. It is separate from the earlier eight-pair, 512-tick
always-request emulation check, which found identical physical/economic
trajectories under forced service choices. The present information contract
still makes a separate requester layer structurally redundant; a genuine
machine-information or operational constraint would need independent
justification before two-policy training.
