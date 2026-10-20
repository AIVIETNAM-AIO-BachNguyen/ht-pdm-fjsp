# V10 assignment-contrast RL: audited DEV partial progress, prelocked no-go

V10's method, protocol, seeds, source and tests were locked and pushed at `55878b1` after 37 focused tests and two completed Ubuntu smokes. The full new DEV run `research_result/framework_v10_dev_20261020T072256Z/` used the exact prelocked four cases, threshold requester, H48+tail16 cost, free legal mask, V7 shared-edge actor and five new scratch-training seeds 70400000–04. The method added λ=0.5 equal-machine-set within-assignment return contrast to the all-action expected policy gradient; the λ=0 ablation used the same actor, root bank and optimizer. No hard shortest-duration restriction or planner action label entered training.

## Verification and compute

The full run completed 184,320 source steps, 15,360 selected counterfactual states, 4,799,516 full-horizon forecast transitions (2,398,436 method; 2,401,080 ablation), 4,480 DEV checkpoint-selection episodes and 1,792 selected/comparator episodes in **1,376.80 s**. Training saw 5,626 eligible within-machine-set assignment groups in the method arm versus 5,664 in the ablation's diverged states; both received 600 parameter updates. Independent audit `research_result/framework_v10_audit_20261020T074623Z/independent_audit.json` is **PASS**: source/artifact/comparator hashes, training physical streams and exact first-iteration paired Q banks, 240 independently replayed full-H+tail Q banks (68,564 transitions), all 640 selected RL episodes, complete grids, checkpoint selection, bootstrap, and the stricter DEV gate. Simulator objective, reward, component sum, request conservation and zero invalid assignments passed. Raw checkpoints and traces remain outside Git.

## Locked DEV result

All costs below are mean H48+tail16 physical objective on four equally weighted cases and 16 common fresh DEV physical seeds per case. Lower is better.

| Controller | Equal-case cost | Selection time per episode |
|---|---:|---:|
| V10 assignment credit | **878.22** | 0.0125 s |
| Matched joint-credit RL ablation | 882.25 | 0.0125 s |
| Flat PPO free mask | 905.77 | 0.0203 s |
| V7 shared canonical RL | 968.59 | 0.0281 s |
| Risk matching | 894.61 | 0.00139 s |
| Bounded rollout | **849.34** | 0.691 s |

V10 is **−16.39 cost** versus risk and **−27.54** versus flat PPO, but **+28.88 (3.40%)** versus bounded rollout. Selection is about **55× faster** than rollout, while risk matching is faster than V10. The unchanged competitive goal's planner non-inferiority threshold is ≤857.84 on this DEV panel (1% above rollout); V10 misses it. A 5% improvement over the strongest locked baseline would require ≤806.88, also far away. These DEV planning distances are **not fresh TEST acceptance thresholds**.

The prelocked ablation contrast is assignment credit minus joint credit **−4.03**, hierarchical five-train-seed/16-physical-seed CI95 **[−11.64,+0.46]**. Seedwise deltas were **−3.84, −1.06, −0.52, −14.20, −0.53**: all five point estimates favor V10, including the three cheapest ablation seeds, but only one seed contributes a double-digit gain. The 4.03-unit mean misses the required 10-unit method gain and the interval crosses zero. The prelocked DEV gate is **false** despite risk not dominating cost and the 55× selection-speed ratio. This does not establish a robust assignment-credit advantage. Both arms' better cost versus the earlier V9 DEV result uses different train and physical seed splits and cannot be attributed to the new auxiliary term.

| Case | V10 | Ablation | Risk | Rollout | V10 − rollout |
|---|---:|---:|---:|---:|---:|
| Interchangeable scale1 | 755.40 | 754.67 | 760.31 | **681.94** | +73.46 |
| Reciprocal scale2 | 790.17 | 790.38 | 808.25 | **698.75** | +91.42 |
| Reciprocal scale3 | **975.96** | 990.27 | 1010.50 | 1035.88 | −59.91 |
| Specialist/generalist scale2 | 991.35 | 993.69 | 999.38 | **980.81** | +10.54 |

The assignment term has its largest case gain in reciprocal scale3 (−14.31 versus ablation), but the two cases where rollout most exceeds V10 are interchangeable scale1 and reciprocal scale2. In interchangeable scale1, V10 and rollout start the same mean number of services (**51.56**) and have nearly equal H availability (0.602 vs 0.599), yet rollout has **0.75 H failures versus 3.98** and failure cost 11.25 versus 83.81. In reciprocal scale2, V10 has **4.71 H failures versus rollout 1.88**, failure cost 99.75 versus 28.12, with starts 50.14 versus 51.50. These aggregates indicate a timing, machine priority or assignment sequence difference; they do **not** identify the causal action class. In reciprocal scale3 rollout itself starts fewer services, queues more CM work and costs more than V10, so its superiority is not universal.

## Decision and limits

V10 remains a genuine RL dispatcher with learned full matching actions and a verified simulator/evaluation pipeline. On this fresh DEV panel it is the first structured RL candidate here to beat risk matching on mean cost while remaining far faster than bounded rollout. The **specific within-machine-set credit hypothesis did not pass its matched ablation or competitive DEV gate**. No V10 TEST, independent method replication, N/K scaling or novelty claim is justified. Keep the goal **ACTIVE**.

Next prelock a DEV-only selected-state action/value diagnostic in the two cases with the large rollout advantage, using immutable V10 checkpoints and a separate forecast stream. Classify WAIT/service cardinality, serviced machine set, assignment and six-tick current-information root value on V10 and rollout visited states, with per-seed and tick-bin coverage. Then test a single mechanism that addresses the dominant value gap; if none is supported, report the limit rather than changing the objective or cherry-picking a setting. Strong baselines, fresh confirmation and independent replication remain required for goal completion.
