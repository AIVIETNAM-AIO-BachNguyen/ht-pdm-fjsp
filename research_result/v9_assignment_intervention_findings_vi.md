# V9 DEV paired assignment intervention: long assignments cause most weak reciprocal loss

The protocol `v9_assignment_intervention_plan_vi.md` was locked before code or outcomes. Source, tests and completed Ubuntu smoke were pushed as `d5c004f` before full DEV. The run `research_result/v9_assignment_intervention_dev_20261020T070903Z/` reused only the already-opened V9 DEV physical tapes; it is a mechanism experiment, **not** a new RL method, fresh TEST or confirmation. The treatment keeps the actor's chosen WAIT/service and machine set at each visited state, replacing only its technician assignment by the shortest feasible known-duration assignment for that set. The same selected V9 actor then continues from the treatment state. Control and treatment each use the same physical seed, objective, threshold requester and H48+tail16 environment.

## Audit and scope

Full run completed **640 pairs, 1,280 episodes and 61,440 H decisions** in 32.60 seconds. Independent verifier `verify_v9_assignment_intervention.py` replayed every action from actor logits and legal masks, independently recomputed the assignment rule, checked all episode metrics against environment state and checked every control against immutable V9 `episodes.csv`. Artifact `research_result/v9_assignment_intervention_audit_20261020T071227Z/independent_audit.json` is **PASS**. The treatment changed 2,607 H assignments across both V9 arms (877 all-action, 1,730 chosen-action), saving 22,958 known service-duration ticks at those decisions. These are decision-local duration sums, not episode-time saved. No simulator, checkpoint, cost, seed or baseline was altered.

## Paired full-episode results

Lower objective is better. Each cell averages five selected training seeds and 16 common physical DEV tapes per seed (80 paired episodes/case/arm). Deltas are treatment minus control.

| Case | All-action control → treatment (delta) | Chosen-action control → treatment (delta) |
|---|---:|---:|
| Interchangeable scale1 | 745.94 → 745.94 (0.00) | 749.23 → 749.23 (0.00) |
| Reciprocal scale2 | 936.83 → 789.24 (**−147.59**) | 1103.78 → 810.58 (**−293.20**) |
| Reciprocal scale3 | 1163.53 → 1062.54 (**−100.99**) | 1255.38 → 1069.71 (**−185.66**) |
| Specialist/generalist scale2 | 998.80 → 997.91 (−0.89) | 1005.02 → 996.45 (−8.57) |
| Four-case equal-weight mean | **961.27 → 898.91 (−62.37)** | **1028.35 → 906.49 (−121.86)** |

All-action seedwise equal-weight deltas were **0.00, −0.44, −151.09, −160.72, +0.42** for training seeds 00–04. Chosen-action deltas were 0.00, −168.23, −150.23, −161.45, −129.38. An exploratory two-level resample of five training seeds and 16 physical tapes per case (10,000 draws; seed 70370000) gave an all-action paired 95% interval **[−130.75, +0.08]** and chosen-action **[−166.00, −59.56]**; this is descriptive on opened DEV and the all-action effect remains two weak-replica rescue. No new TEST claim follows.

The prelocked weak reciprocal attribution threshold was met. All-action weak seeds 02–03 had control cost 1194.88 versus strong-seed 00/01/04 control 764.79 in reciprocal scale2; assignment-only treatment reduced weak cost to 825.91, recovering **85.8%** of that weak/strong gap. In reciprocal scale3 the analogous costs were 1339.75 versus 1046.04, with treated weak cost 1087.28, recovering **86.0%**. Strong reciprocal seeds had zero assignment swaps and zero cost change. This gives direct, paired, closed-loop evidence that long assignments are a major cause of V9 weak reciprocal loss under this specific intervention. It does not prove shortest assignment globally optimal or fix machine-set/timing errors.

The weak-seed operational changes are coherent with technician occupancy: in reciprocal scale2 H service starts rose **16.63→34.56**, corrective queue exposure fell **31.88→6.91 ticks**, H availability rose **0.362→0.556**, and H failures fell **13.88→5.06**. In reciprocal scale3 starts rose **12.34→25.34**, CM queue exposure fell **55.91→20.75**, availability rose **0.268→0.423**, and H failures fell **10.09→7.94**. Across all-action seeds, reciprocal scale2 paired changes in cost components were maintenance **+3.95**, unavailability **−98.85**, failure **−52.69** and tail **−9.64**; reciprocal scale3 changes were **+4.58, −84.38, −21.19, −20.98** respectively. Tail is part of full objective and is not an additional additive cost component.

## Interpretation and next gate

The intervention supports **assignment-specific credit** as a V10 mechanism target. But the simple assignment correction is only an oracle-like heuristic overlay on a V9 actor, not learned RL. Its all-action mean **898.91** still exceeds the locked V9 DEV risk matching **863.72** and bounded rollout **857.81**. The stronger improvement in two weak seeds leaves the central goal criterion unmet. Specialist assignment swaps had mixed per-pair effects (all-action 18 better, 47 equal, 15 worse), and the V9 bounded planner itself preferred nonshortest assignments in 3/240 specialist states. A hard shortest-duration mask would discard possible beneficial actions and would not constitute learned assignment competence.

Next, formulate a **conditional assignment-value RL method**: explicitly compare technician assignments within each actor-selected machine set using current-information, full-horizon counterfactual return estimates, while preserving all legal matchings and learned timing/subset choice. Test a scratch-RL ablation on at least five training seeds with equal information/compute, and judge strong and weak seeds separately against flat PPO, risk and bounded rollout. Prelock the V10 method/metrics/seed split/stopping and failure gate before code or run. If learned assignment credit merely reproduces the two weak-seed rescue or stays above risk/rollout, keep the goal ACTIVE and investigate subset and temporal credit rather than claiming a successful framework. Independent replication and fresh TEST remain unopened until a competitive DEV candidate exists.
