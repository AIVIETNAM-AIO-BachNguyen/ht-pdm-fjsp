## Material Passport

- Origin: ARS experiment-agent, independent statistics calculation in shared context.
- Verification: VERIFIED artifact/commit/CI calculations; physical replay comes from runner audit PASS.

# Vòng 2: joint versus economic baseline frontier

Verdict: **JOINT_SETTING_CONFIRMED**. Run `maintenance_joint_frontier_mac_v1_full_cpu_20261012T062329Z`; commit `a769189`. Fresh confirmation256 paired seeds, five methods, all1,280 episodes retained; development32 seeds frozen before test. Complete scalar replay of82,560 physical transitions; forecast replay limited to first2 confirmation seeds as registered. All file hashes and eight simultaneous99.375% intervals independently verified using count-weight bootstrap.

| Method | Cost H+L | H cost | Availability | Pending H |
|---|---:|---:|---:|---:|
| deadline_matching | 469.785 | 377.633 | 71.56% | 0.0000 |
| isolated_h6 | 485.160 | 390.402 | 71.88% | 0.0352 |
| isolated_endpoint | 471.152 | 379.992 | 74.15% | 0.0000 |
| joint_rollout | 444.441 | 359.207 | 73.56% | 0.0000 |
| subset_rollout | 445.254 | 359.953 | 73.53% | 0.0000 |

| Contrast | Estimate | CI99.375% | Gate |
|---|---:|---:|---|
| cost_vs_deadline_matching | 5.3948% | [4.0623; 6.8000]% | True |
| cost_vs_isolated_h6 | 8.3928% | [6.7704; 10.0486]% | True |
| cost_vs_isolated_endpoint | 5.6693% | [4.0276; 7.3134]% | True |
| availability_difference | 2.0040% | [1.4811; 2.5655]% | True |
| pending_difference | 0.0000 | [0.0000; 0.0000] | True |
| absolute_availability | 73.5616% | [73.1954; 73.8698]% | True |
| absolute_pending | 0.0000 | [0.0000; 0.0000] | True |
| interior_cost_gain | 4.8793% | [3.6516; 6.1715]% | True |

Fixed physical profile only; known-model planning, restricted isolated DP is a capacity-relaxed surrogate. No inference of RL learnability, necessity of MARL, global optimality or assignment-isolated causality. Earlier assignment-primary failure stays unchanged. Per-protocol intervals are approximate percentile bootstrap; they do not give a global error guarantee for indefinite adaptive searches. All failed outcomes, controls and natural zero-disagreement landmarks retained.

11 fallacy checks: method-specific aggregation (Simpson/ecological); post-screen fixed profile disclosed (Berkson/regression to mean); all landmarks and ticks included (collider/base rate); complete episodes/losses kept (survivorship); eight locked simultaneous intervals (look elsewhere/forking paths); policy comparison separate from local mechanism/occupancy (causality/reverse causality). **11/11 checked.**
