# Frozen V14 historical comparator on V17 final DEV panel — audited

This was planned after V17's audited overall gate failed and is explicitly retrospective opened DEV. Plan `v17_v14_comparator_plan_vi.md` was pushed at `1533e0e`. Initial full comparator `v17_v14_comparator_dev_20261021T032300Z` finished its episodes but failed before summary because the in-memory training seed was an integer whereas the baseline join expected a string; artifact retained. Fix/regression test, fresh smoke/independent smoke audit and push `d096233` preceded the new full run. Final `research_result/v17_v14_comparator_dev_20261021T032416Z` **COMPLETED**; independent `research_result/v17_v14_comparator_audit_20261021T032446Z/independent_audit.json` **PASS**, replaying all 640 frozen V14 actor episodes and verifying checkpoint/source hashes, legal/current-only actions, objective/operational metrics, joins and 20,000-draw uncertainty. No checkpoint was trained, retuned or selected on the V17 panel; no new physical/TEST seed was opened.

| Controller | Same-panel four-case H64 cost | Selection seconds/episode |
|---|---:|---:|
| V14 joint identity `competition` | **879.14** | 0.02194 |
| V14 count-calibrated | 879.61 | 0.02364 |
| V17 rank projection | 898.69 | 0.01407 |
| bounded rollout | **860.45** | 0.65629 |

V17 rank−V14 joint = **+19.55**, descriptive hierarchical CI95 **[+10.05,+29.59]**; versus V14 count +19.08 [9.27,29.41]. The CI independently resamples each method's training seeds and shares final physical samples; training seeds are not paired. V17's supported same-network rank-versus-expected-Q ablation therefore does **not** imply competitiveness against the earlier structured policy. V14 joint is still +2.17% above rollout on this panel, so neither method satisfies the original ≤1% planner margin or ≥5% better-than-strongest route here.

| Case | V14 joint | V17 rank | Rollout |
|---|---:|---:|---:|
| reciprocal scale2 | 805.89 | 801.01 | 707.56 |
| interchangeable scale1 | 784.39 | 785.60 | 685.75 |
| reciprocal scale3 | 944.54 | 999.79 | 1074.38 |
| specialist/generalist scale2 | 981.75 | 1008.35 | 974.12 |

V17 slightly improves V14 on reciprocal2 but is worse in the other three cases; all learned policies still face substantial rollout gaps in interchangeable/reciprocal2. This strengthens the decision to retire V17 as a competitive candidate while retaining its rank-projection credit ablation as a method observation. The next analysis should focus on **which repeated dispatch decisions** create excess PM queue exposure/failures on the same physical tapes, with a prelocked mechanism diagnostic. No V17 replication or TEST; original goal ACTIVE.
