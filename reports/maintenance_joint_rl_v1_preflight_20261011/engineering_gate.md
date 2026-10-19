# Joint maintenance RL v1 — engineering gate

## Material Passport

Origin Skill: academic-research-suite / experiment-agent; Origin Mode: inline
plan/run; Date: 2026-10-11; Version: maintenance_joint_rl_v1.
Verification Status: VERIFIED for engineering only. Scientific superiority and
learning on the sealed full panel remain UNVERIFIED until the lab run is complete.

## Scope and gate

Main policy learns both when/which machines to maintain and technician matching,
including waiting. Restricted machine-selection PPO and development-selected
rules are controls. See `maintenance_joint_rl_v1_plan.md` for the frozen protocol.
This is a new heterogeneous prospective distribution; do not compare its raw
costs to the previous favorite-worker benchmark as a controlled improvement.
The user stops the prior confirmation themselves; its incomplete artifacts are
not deleted, resumed or included as final results. Assistant has not contacted
the lab or started a full experiment.

- Repository tests: **615 passed, 3 existing dependency warnings, 204.83 seconds**.
- Final Mac CLI smoke: `maintenance_joint_rl_v1_smoke_20261011T045107Z`; COMPLETED / audit PASS.
- Both PPO arms updated weights; 128 learned decisions, 192 physical training
  ticks, 16 optimizer steps, 32 training episodes, 26 dev episodes, 12 test
  episodes and 72 replayed/logged evaluation ticks, all matching budgets.
- Initial/selected checkpoints reload with identical weights, matching action,
  log-probability and critic output. Initial checkpoint cannot be selected.
- Independent scalar replay reconstructs actions, rewards, requests and tail
  carryover; repeat smoke reproduces decisions, training episodes and summary.
- Complete returns include H+L costs; the PPO sample set contains only H ticks.
- Exact legal action masks, WAIT, same feasible machine subsets, paired initial
  parameters and permutation-consistent matching probabilities pass.
- 800 historical physical signatures excluded; train/dev/test/full/smoke split
  audit and 103,014 distinct derived RNG roles pass. Paired arms deliberately
  share train tapes; controllers deliberately share test tapes.
- 57 artifact hashes checked; smoke size
  3.25 MB.
- Progress bars present in console log. Full test outcome flag remains false.
- Interruption retains FAILED manifest, partial journals and checkpoints.
  Nonempty output directories and dirty-code full runs are rejected.

## Engineering compute and storage estimate

Reserved smoke seeds only, full-size H48/L12 and hidden32. One PPO update per
arm uses 3,072 learned steps, 64 complete episodes and 48 optimizer steps.
Measured 0.771 / 0.787 seconds per update, excluding initial optimizer setup.
Full training extrapolates to 1,246 seconds (~21 minutes) on this Mac.
Four full-length engineering evaluation episodes took 0.066 / 0.067 seconds
with batch4; extrapolated dev+test adds ~8 minutes before checkpoint I/O and
bootstrap. **About 30 minutes on this Mac is an extrapolation, not a full run.**
Ubuntu i7-10700 performance was not measured; plan roughly 1–2 hours initially
and use the printed actual update progress to refine that estimate. No guaranteed
lab ETA, GPU speedup or parallel-worker claim.

Full run: 20 models, 6,144,000 training physical ticks, 6,848 dev episodes,
21,504 test episodes / 1,290,240 test physical ticks. No per-action Monte Carlo
planning loop. CPU and thread1; vectorization batches inference, not OS workers.

Artifact probe extrapolates ~124 MB of principal CSV/checkpoint files; reserve
**150–300 MB** for raw outputs, snapshots, JSON and console logs. Request density
and action-string lengths vary. Full decision journals are locked to first two
profiles and first two shocks per condition for all policies: 10,080 decisions.
Every evaluation tick still receives feasibility and scalar/batch consistency
checks. All episode/request/carryover records are retained.

Reproduce engineering probes from repository root (fresh timestamp directory):

```bash
uv run python scripts/benchmark_maintenance_joint_rl.py
uv run python scripts/probe_maintenance_joint_rl_artifacts.py
```

Benchmark evidence: `maintenance_joint_rl_v1_engineering_benchmark_20261011T044800Z/benchmark.json`.
Artifact evidence: `maintenance_joint_rl_v1_artifact_probe_20261011T045009Z/size_estimate.json`.
Full test outcomes were not opened by either probe; they cannot select a model
or tune a hyperparameter. Source hashes accompany the timing benchmark.
