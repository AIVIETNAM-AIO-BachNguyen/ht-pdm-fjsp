# Completion-Credit PPO v1 negative pilot

The complete pilot does **not** authorize full development or confirmation.
Seven of nineteen point screens fail; scientific tests remain unopened. No
teacher-free success or confirmed mechanism claim follows from this result.

Run: `artifacts/technician_completion_credit_ppo_v1_pilot_20261014T200910Z`,
learning commit `27d3e7f8dc09a914af21d305fd23f79691b32933`.
COMPLETE/audit PASS, 5751.40 seconds (95.86 minutes), 22 randomly initialized fits,
61,440 decisions each (1,351,680 total), 28,160 training episodes, 5,120 paired
training tapes, 1,056 DEV checkpoint episodes, zero invalid assignments, zero
pretraining and zero scientific test episodes. All replicas spent the full budget.

Independent report rehashes all 397 bound outputs and checks 248 current
source/protocol/registry bindings. It independently reconstructs every per-replica
checkpoint selection and selected mean from the complete DEV grid. Report:
`artifacts/technician_completion_credit_ppo_v1_pilot_report_20261015T063545594591Z`.
The PNG was visually checked: readable N4/N32 panels, all replicas retained,
shading explicitly min–max rather than confidence intervals. The report uses an
isolated cached analysis environment (NumPy 2.5.4, Matplotlib 3.11.2); training
runtime remains recorded in the original immutable manifest.

| Setting | Candidate cost/machine | Reference cost/machine | Interpretation |
|---|---:|---:|---|
| N4 vs equally short PPO | 357.7578 | 309.4531 | Cost +15.61%, fails -1% gain margin |
| N4 vs local MAPPO | 357.7578 | 351.9219 | Cost +1.66%, fails positive gain |
| N32 vs deadline | 442.5654 | 271.5527 | Cost +62.98%, fails -2% gain margin |

Other failed screens: N4 availability difference vs PPO -0.092773 (limit -0.01);
N32 availability difference vs deadline -0.144531 (limit -0.01); N32 failures per
machine difference +0.668945 (limit +0.25); N32/N4 cost per machine ratio 1.237053
(limit 1.10). Pending limits pass; N4 failure difference passes. Lower failure
count at N4 does not offset its availability/cost failures.

N32 candidate point gain vs state-credit is 12.56%, vs joint-credit 12.24%, and
vs independent 50.05%. These descriptive ablation gains coexist with failed
absolute cost and health gates. The N32-only pooled proxy cannot establish the
registered N8/N16/N32 effect or statistical significance. The joint-credit
contrast changes both credit and surrogate; it is not a clipping-only ablation.

Both candidate replicas select 30,720 at N4 and 61,440 at N32. All replicas remain
in the means. N32 selected cost/machine is 271.9023 for seed 70400000 but 613.2285
for seed 70400001; availability is 0.653442 and 0.357381 respectively. Do not
report only the good replica. Controls also have large seed dispersion.

Logged prefix-baseline spreads are tiny relative to raw advantage spread.
Actual actor/critic update norms are available and nonzero. These descriptive
observations alone do not identify the cause, establish effective credit
assignment, or prove a gradient-variance reduction. Do not change clipping,
thresholds, budgets or architecture based on an untested causal story.

Next: frozen-checkpoint allocation diagnostic on **all opened pilot DEV** seeds
and replicas, including deadline, under its separate locked diagnostic plan.
No optimizer, new test seeds or changes to learning source. Any later revised
learning design requires a new prospective protocol and disjoint seed panels.

Analysis validation: 37 completion tests pass (11.57s); syntax/diff checks pass.
Frozen diagnostic smoke `..._allocation_diagnostic_smoke_20261015T063557817278Z`
passes 13 episodes and 780 independent physical replays in 3.15s, all original
metrics reconciled, parameters unchanged. The actual full-development prerequisite
was called on this pilot and correctly rejected it: `pilot cost/health screen fails`.
