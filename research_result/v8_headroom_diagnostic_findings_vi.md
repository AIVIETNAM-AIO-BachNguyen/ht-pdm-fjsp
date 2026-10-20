# V8 DEV current-request headroom: actor selection dominates

This exploratory mechanism diagnostic was locked in `v8_headroom_diagnostic_plan_vi.md` before code or results. It is neither a new method nor a TEST result. The runner was committed and pushed as `1cde5dd` before the full run. The first smoke directory records a JSON serialization failure; corrected fresh smoke completed and passed independent replay.

## Provenance and audit

Full run: `research_result/v8_headroom_dev_20261020T052426Z/`; 1,920 complete on-policy episodes (4 cases × 3 V7 learned arms × 5 training seeds × 32 new physical seeds), 720 hash-selected pre-action states, 2,160 same-state learned choices, and 239,040 bounded forecast transitions. Immutable V7 checkpoints supplied the policies. No new training or TEST episodes were used. Independent audit `research_result/v8_headroom_dev_audit_20261020T053100Z/independent_audit.json` passed source/artifact/checkpoint hashes, full trajectory and cost/request replay, hash sampling, all root scores and same-state decisions, and the regret decomposition. Corrected smoke passed identical checks (120 episodes, 180 states).

At each state the reference is the best current matching under a six-tick, 16-scenario, risk-continuation forecast with a common scenario bank. Root regret is forecast cost difference for one choice; it is not closed-loop economic regret or an optimality guarantee.

## Decision mechanism

Each source contributed 240 states, balanced over four cases and five training seeds. Means are bounded-root cost units per sampled state.

| Source on own states | Full regret | Available-action actor gap | Canonical-mask gap | Regret ≥5 / ≥10 states | Risk-rule regret on same states |
|---|---:|---:|---:|---:|---:|
| Shared canonical | 3.355 | 3.307 | 0.047 | 40 / 29 | 1.858 |
| Shared free | 6.703 | 6.703 | 0 | 66 / 50 | 1.901 |
| Flat free | 1.793 | 1.793 | 0 | 30 / 16 | 0.998 |

The canonical mask excluded the forecast-best action in only **3/240** canonical-source states. Its total gap was about 11.3 units versus about 793.7 units of available-action actor gap. A method centered on removing that mask is unsupported by this probe; rare high-impact states or longer horizons remain possible.

Canonical-source mean available-action gaps by training seed 00–04 were **1.112, 1.281, 4.803, 5.270, 4.070**; full gaps were 1.112, 1.281, 4.997, 5.270, 4.113. Shared-free full gaps were 1.055, 1.077, 11.939, 11.056, 8.389. Thus the V7 mask largely rescued weak replicas, while strong structured replicas made similar choices. Flat-free full gaps were more stable (0.868, 1.177, 2.999, 2.714, 1.208), although flat closed-loop cost exceeded the strong structured replicas.

On the **same canonical-source states**, mean regret was 3.355 canonical, 8.187 shared free, 3.004 flat free, 1.858 risk. For the two strong structured seeds these were 1.197, 1.220, 2.293, 1.293; for three weak seeds 4.793, 12.832, 3.477, 2.235. The heuristic often wins the same-state aggregate but not every case/seed.

Canonical mean full root regret by case was 0.662 (interchangeable scale1), 5.713 (reciprocal scale2), 4.446 (reciprocal scale3), and 2.598 (specialist/generalist scale2). Versus planner, canonical actions were exact matches in 153 states, different machine subsets in 51, over-service in 33, assignment only in 3, and under-service in 0. This selected six-tick root taxonomy does not refute earlier full-trajectory under-service observations: visitation, forecast horizon and comparison target differ.

## Episode context and research decision

On the 32 diagnostic physical seeds, case-balanced H+tail economic cost was 919.04 flat, 944.03 canonical and 1000.65 shared free. Canonical cost by training seed was 880.91, 880.73, 996.47, 990.47, 971.57; by case 749.96, 904.09, 1118.33, 1003.73. Its mean H availability by case was 0.603, 0.527, 0.403, 0.512. Mean action-selection time was about 0.025 seconds per episode for both structured arms and 0.019 for flat. These DEV episode numbers are descriptive, not confirmatory planner contrasts or causal translations of root regret.

**Decision:** Do not add an absent-request context encoder alone (the V8 context probe had only four ≥5-unit material flips), and do not center the method on mask removal. Investigate a resource-competition-aware learned choice among feasible current matchings with training-seed robustness. Compare scratch RL against flat PPO, both V7 structured arms, risk matching, bounded planner, and an ablation of the new mechanism. If a risk-rule prior is used, show value from RL beyond the heuristic; a prior alone cannot satisfy the goal.

The 720 states come from different on-policy distributions. Same-state choices do not rank closed-loop policies. The bounded planner has risk continuation, a short horizon and finite stochastic scenarios; it can misrank long-horizon actions. These DEV seeds are opened. No competitive criterion or independent replication has been met; no V8 method has been trained yet.
