# V15 earliest-compatible-release hazard probe — audited screening no-go

## Provenance and scope

After V14's matched-capacity identity-term failure, `v15_release_hazard_proposal_vi.md` proposed an action-conditioned estimate of a PM machine's failure probability before a compatible technician can next start it. The first written formula was prospectively corrected at `483a680`, **before code or outcomes**, to count the current tick whenever an action leaves a machine unserved. The prelocked `v15_release_hazard_probe_protocol_vi.md` set a leave-one-training-seed-out comparison against an age-agnostic release proxy using only immutable V14 `count_calibrated` training Q banks. This is teacher-bank screening, **not** V15 RL training or independent policy evaluation.

Source/tests/verifier were pushed at `cde4b72` after seven focused tests and a 12-state Ubuntu smoke plus independent smoke audit PASS. Full `v15_release_hazard_probe_dev_20261021T020150Z` completed all 7,680 training states (five seeds × six updates × four cases × 64 states), 30 pre-update checkpoints and the fixed nine-alpha grid. Independent `v15_release_hazard_probe_audit_20261021T020216Z/independent_audit.json` is **PASS** for source/artifact/checkpoint hashes, every legal action, both feature formulas, all fixed-alpha actions and Q values, five held-out alpha choices, summaries, 20,000-draw hierarchical bootstrap and gate. No new physical tape, training seed, TEST or policy checkpoint was created.

## Locked screening result

Held-out saved H64 root Q for the age-threshold feature minus the age-agnostic release feature is **−0.269 cost units/state**, hierarchical CI95 **[−0.526,+0.001]**. All five held-out training-seed means are negative (−0.082,−0.155,−0.243,−0.463,−0.401), but the prelocked diagnostic gate required a material mean ≤−1, CI upper <0, at least four negative seeds and ≥1% changed actions. Only the seed-consistency and action-change requirements pass; **the joint gate is FALSE**. The comparison changes 876/7,680 actions (11.4%): 406 WAIT-related, 446 machine-set and 24 assignment changes. Among changed rows 487 improve Q, 386 worsen and three tie; mean changed-row Q difference is −2.36. The feature is therefore active, but its net Q effect is small and uncertain.

| Case | Held-out threshold minus age-agnostic Q |
|---|---:|
| Reciprocal scale2 | −0.107 |
| Interchangeable scale1 | −0.778 |
| Reciprocal scale3 | **+0.431** |
| Specialist/generalist scale2 | −0.622 |

The target interchangeable case benefits more than reciprocal2, while reciprocal3 worsens. Age-threshold and age-agnostic feature ranges exceed 0.05 on 82.6% and 84.3% of states, respectively, but their action-feature rankings disagree materially on only 2.67% of states. The five held-out alpha pairs (threshold/control) are 2/2, 8/4, 16/4, 2/2 and 4/4. Mean saved Q at the base alpha0 action is 699.845; selected threshold/control means are 698.983/699.252. About 1,039 and 1,024 selected adjusted-score top pairs are within 1e−6, so small changes around numerical ties require care. None of these Q-bank observations establishes closed-loop policy value: all Q estimates use four scenarios and the **old V14 count actor** for continuation.

## Decision

The exact release-horizon PM hazard is a nondegenerate representation of current physics, but its incremental advantage over the active, equally informed age-agnostic release feature misses the prelocked material and uncertainty gates. **Do not launch a V15 scratch-RL campaign on this evidence alone.** Retain the formulation and result as negative development evidence. The remaining competitive gap appears to require a more flexible way to evaluate whole post-decision resource states or to improve temporal policy continuation, rather than a single fixed hazard feature. Any new direction needs its own proposal, fair active-capacity ablation and fresh evaluation. The original user goal remains ACTIVE, with no independent replication or new TEST.
