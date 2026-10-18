# C1 adapter engineering checkpoint

The preregistered C1 protocol/registry were committed and pushed at
`3b28c56` before the adapter was implemented or run. No C1 scientific TEST
has been opened. The adapter preserves the audited `WaitingEnv` transition,
costs and persistent request contract; it changes only the four locked skill,
capacity and duration matrices and maps request age into the existing graph
observation slot.

Six correctness tests pass: the exact N/K and skill matrices, 60-tick
completion, busy-resource masks, paired physical tape across arms, adapter
request feature, duplicate/unrequested-action rejection, request
conservation and reward/cost accounting. `compileall` and `git diff --check`
also pass.

Fresh lab engineering smoke:
`artifacts/technician_diverse_maintenance_adapter_smoke_20261018T122025Z`.
It completed 16 FCFS/risk episodes on the four locked configurations and two
engineering DEV seeds, all with 60 physical ticks, zero invalid assignments,
eight paired physical tapes and request/cost reconciliation. The measured
FCFS/risk cost/machine means are in `summary.json`; these two smoke seeds
do not select the C1 heuristic or establish a learned-policy outcome. The
full learner, DEV gate, sealed confirmation and independent verifier remain
to be implemented and run.
