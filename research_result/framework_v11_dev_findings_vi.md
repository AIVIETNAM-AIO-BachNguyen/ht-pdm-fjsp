# V11 reference-continuation RL: audited DEV failure from weak policy closure

V11 proposal/protocol/seeds/source/tests were locked, smoke-verified and pushed as `0063de2` before the five-seed full DEV run. The method used a reference risk-policy continuation to estimate the H64 cost of every legal matching, then updated the same free-mask shared-edge RL actor with all-action expected policy gradient. The equal-compute `self_credit` ablation used the same complete physical scenario banks and both computed Q sets, but updated from current-actor continuation Q. Neither arm used planner labels, imitation, a hard WAIT rule, changed objective or inference-time forecast.

Full artifact `research_result/framework_v11_dev_20261020T115151Z/` completed five fresh training seeds×two arms×six updates, **184,320 source steps**, **11,611,168 paired full-H64 forecast transitions** (5,967,664 reference arm and 5,643,504 self arm), 4,480 DEV checkpoint-selection episodes and 1,792 selected/comparator episodes in **2,199.40 seconds**. Independent verifier `verify_framework_v11.py`, artifact `research_result/framework_v11_audit_20261020T122855Z/independent_audit.json`, is **PASS**: source/artifact/V7 comparator hashes, training physical streams and identical first-iteration dual Q banks, 240 independently replayed paired Q banks (347,016 transitions), all 640 selected learned episodes, complete grids, checkpoint selection, bootstrap and prelocked gate. Reward/cost/component/request conservation and zero invalid assignments passed. Raw/checkpoints stay outside Git.

## Locked DEV comparison

All costs are equal-weight means of the four frozen cases with 16 fresh common physical DEV seeds each; lower is better.

| Controller | Cost | Selection time / episode |
|---|---:|---:|
| V11 reference-continuation RL | **1138.76** | 0.0131 s |
| Matched self-continuation RL | 946.45 | 0.0126 s |
| Flat PPO free mask | 924.62 | 0.0203 s |
| Risk matching | 914.97 | 0.00136 s |
| Bounded rollout | **875.84** | 0.684 s |

The prelocked ablation difference (reference minus self) is **+192.31 cost**, hierarchical five-training-seed/16-physical-seed CI95 **[−7.39,+392.15]**. Seedwise differences: **−2.66, +471.83, −51.22, +478.61, +64.97**. Only two seeds favor reference; both strong self-control seeds 01 and 03 collapse. Risk matching is simultaneously cheaper and about 9.6× faster than V11 reference. The prelocked DEV gate is **false**, with one improvement among the three cheapest self-control seeds. V11 is +30.0% above rollout, far from unchanged completion criterion B; criterion A is also unmet. No V11 TEST or independent method replication is justified.

| Case | Reference | Self | Risk | Rollout | Main reference failure |
|---|---:|---:|---:|---:|---|
| Interchangeable scale1 | 1017.65 | 738.15 | 737.50 | **677.62** | H availability 0.351 vs self 0.601; CM queue 86.10 vs 0.20 |
| Reciprocal scale2 | 1103.66 | 927.38 | 836.00 | **757.69** | 29.20 service starts vs self 42.98; CM queue 92.67 vs 11.60 |
| Reciprocal scale3 | 1249.22 | 1128.74 | 1081.12 | **1070.38** | 23.23 starts vs self 31.73; CM queue 98.36 vs 27.41 |
| Specialist/generalist scale2 | 1184.49 | 991.54 | 1005.25 | **997.69** | 26.60 starts vs self 40.86; CM queue 91.15 vs 7.25 |

The reference arm's full-episode availability/service collapse is not infeasibility or an objective accounting error. On seeds **70500001 and 70500003**, DEV selection chose **iteration 0**, the untrained actor, with identical four-case mean **1374.20**; every trained checkpoint was worse by the prelocked selection rule. Seed 70500004 selected iteration5 and still cost 1135.80; seeds00/02 cost 898.50/911.00. Self-credit selected later checkpoints for all five seeds and cost 895.61–1070.82. Both arms completed 300 parameter updates without KL early-stop; mean final KL per update was 0.0011 reference versus 0.0023 self. The run's 4-scenario Q_ref estimates are a materially different training signal, but their use alone did not improve closed-loop dispatch.

## Interpretation and next gate

The V10 first-divergence evidence still shows long-horizon root value under **risk continuation**; V11 now demonstrates that optimizing that value directly can make the learned policy under-serve badly. A plausible failure is **continuation inconsistency**: Q_ref may prefer deferring a root because it assumes risk matching services later, while the learned actor under its own future decisions keeps deferring. Initial seed weakness and finite-bank noise could contribute. This is a hypothesis, not yet proven from action/Q traces. The failed method cannot be repaired by reporting only its two favorable seeds or by treating risk continuation as an optimal teacher.

Next prelock a DEV-only selected-checkpoint action/Q audit on immutable V11 artifacts: measure WAIT with legal service by case/seed/checkpoint and compare Q_ref versus Q_self preference for WAIT on the *same stored training states*, especially seeds01/03 at iteration0 and trained candidates. Then test one self-consistent long-horizon credit method if the mechanism is supported; keep actor learning from scratch, strong baselines and five-seed uncertainty. Preserve the negative V11 result and keep the goal **ACTIVE**.
