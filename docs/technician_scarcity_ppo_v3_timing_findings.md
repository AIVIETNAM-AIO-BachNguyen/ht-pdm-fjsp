# Frozen v3 development timing findings

Diagnostic `technician_scarcity_ppo_v3_timing_diagnostic_20261014T151358964289Z`
COMPLETED in54.15s:672 selected N4 episodes (7 methods,3 training replicas,
32 previously opened DEV seeds),40320 independently replayed policy/tail ticks.
All original episode metrics reconcile, initial physical populations are paired,
checkpoint tensors and algorithm identities unchanged. No optimizer, training,
teacher or scientific test. Script/plan commit `b8cbad1` pushed after a7-episode,
420-tick smoke. All716 original input bindings and diagnostic output bindings
verified. This is descriptive evidence on the failed development panel.

`technician_scarcity_ppo_v3_timing_report_20261014T183028422990Z` preserves
the derived risk/availability and job-weighted service fractions, with copied
analysis script and input/output hashes, rechecks frozen physics and the complete
decision grid. Its numerical summary exactly matches the earlier182741 report;
both artifacts are retained. All raw trajectories remain available.

## Observed decision pattern

Numbers below are per episode, averaged over all96 episodes of each method.
At-risk means nonfailed, age>=failure_age-1, not already servicing, and not
included in that tick's executed starts. Counts sum machine-tick opportunities,
not distinct machines. Residual free means a technician remains free after all
chosen starts; all N4 service edges are compatible. It is not a causal label of
preventable failure. First-slot STOP only applies to the three broker arms.

| Method | First STOP with legal edges | Residual-legal ticks | Unserved at-risk | Failures | Failures with residual free technician | Full-restoration fraction of starts |
|---|---:|---:|---:|---:|---:|---:|
| proposed hierarchical opportunity PPO | 0.6354 | 0.6354 | 9.9375 | 4.5313 | 0.3021 | 0.7191% |
| no-price reservation | 0 | 0.8750 | 10.8438 | 4.6146 | 0.5729 | 0.9567% |
| independent | n/a | 2.5313 | 12.1979 | 5.4375 | 2.0417 | 0% |
| flat STOP | 0.4583 | 0.7396 | 5.6667 | 2.4271 | 0.4271 | 0.4684% |
| original centralized PPO | n/a | 0.4688 | 1.4063 | 0.6250 | 0.1771 | 3.2581% |
| local MAPPO | n/a | 4.4375 | 10.0417 | 4.6563 | 1.8646 | 10.1181% |
| full-information MAPPO | n/a | 1.8958 | 6.1979 | 2.6354 | 0.9688 | 7.6048% |

Every candidate unserved at-risk machine-tick has an initially free compatible
technician. Only0.5729 of9.9375 still have one free after executed starts. Likewise,
only0.3021 of4.5313 candidate failures occur with residual free capacity;
4.2292 occur after free technicians have been allocated to other machines.
Thus removing occasional STOP alone does not directly explain most of this
failure gap. Allocation contention and the states reached by earlier choices
need examination. This classification does not assert that another matching
would eliminate failures: other machines also require service and future
trajectories would change.

The candidate almost always selects1-tick,half-restoration edges. PPO uses a
small but higher fraction of longer,full-restoration edges. This could help
shape later demand phases, but local MAPPO chooses an even larger fraction
and still fails more. Therefore “prefer full restoration” is not an established
fix or a justified handcoded policy. Candidate mean start age3.1598 vsPPO3.0234
is another association with delayed service, not an independent causal effect.
Policy-only start counts here differ from earlier episode-level preventive/
corrective totals because those include12 deadline-tail ticks.

## Consequence for the next design

Preserve v3 failure and keep its confirmation panel closed. Focus the next
prospective design on credit for coupled assignments and future technician
occupation/restoration, rather than another STOP bias or price-scale adjustment.
V3 computes pair bids once per physical decision; its prefix changes masks,
prices and STOP, but not the learned pair bid. All assignments receive one
state-baseline joint PPO advantage. These are candidate structural limitations,
not proven causes from this diagnostic.

A possible next version should use one canonical machine-or-WAIT choice per
technician, condition subsequent allocation scores on reservations, and learn
completion-aware continuation values from its own online returns. No teacher,
old model, future-event input, simulator branching or extra physical samples.
To isolate learning credit, train the same prefix-value network on the same
data in both arms, but compare actor advantages using prefix values against
actor advantages using the first state-only value. Inputs, architecture, initial
tensors, optimization steps and physical budgets must match. Include a joint
PPO-ratio control and an uncoordinated control; retain PPO and both MAPPO
information variants at N4. Health and practical cost bounds stay unchanged.

This is a design direction, not a registered experiment or a success claim.
Before implementation/run, lock the exact architecture, surrogate, comparisons,
fresh seeds, stopping rule and statistical family; then tests/smoke/commit/push.
The research goal remains unachieved.
