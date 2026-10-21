# V17 full-matching rank projection scratch RL — audited DEV ablation gain, competitive no-go

Protocol `framework_v17_dev_protocol_vi.md`, proposal and disjoint seed registry were pushed at `38cf615` before code/outcomes. Verified campaign source/tests/verifier was pushed at `6232fe3` after 11 focused/regression tests, final Ubuntu smoke `framework_v17_smoke_20261021T022743Z` and independent smoke audit PASS. Unopened conditional replication/one-TEST reservation and a prior-art boundary were pushed while full DEV was still training; no reserved seed/tape was opened. Full run `research_result/framework_v17_dev_20261021T022840Z` **COMPLETED** in 2,853.6 s from source commit `6232fe3`. Independent `research_result/framework_v17_audit_20261021T031626Z/independent_audit.json` **PASS**: source/artifact/checkpoint/V7 hashes, paired initial active capacity 9,025 parameters for all five seeds, physical-role source streams, 240 complete full-H64 common-bank Q replays (191,392 transitions) from 15,360 saved Q rows, all 640 selected learned final episodes, disjoint selection/final grids, full summary/bootstrap/gate. The audit did not independently retrain all 60 actor updates.

## Locked hypothesis and primary DEV outcome

Both arms trained from random weights on five fresh seeds 73000000–04 with the same current 74-D full-matching actor, legal actions, six updates, per-cell state quota, four common-scenario H64 old-policy root Q, optimizer and KL limit. Only the main credit projection differed: full-legal-action relative-Q softmax cross-entropy (`rank_projection`) versus expected-Q advantage (`expected_q`). Checkpoints were selected on physical 73010000–07 and evaluated once on separate 73011000–15. Final equal-four-case H64 economic cost:

| Controller | Mean cost | Mean selection time/episode |
|---|---:|---:|
| rank projection | **898.69** | 0.01407 s |
| same-capacity expected Q | 907.45 | 0.01404 s |
| flat PPO | 911.52 | 0.01940 s |
| risk matching | 926.34 | 0.00133 s |
| bounded rollout | **860.45** | 0.65629 s |

Paired rank−expected cost **−8.76**, hierarchical CI95 **[−13.99,−4.02]**. Training-seed means were −11.25, −8.14, −7.50, −4.77, −12.14; all five negative and all three strongest-control seed means negative. Thus the **credit-objective ablation** passes its material/seed/uncertainty clauses. Candidate is ~46.7× faster than rollout at action selection and risk is cheaper in runtime but higher in cost, so no simple risk cost/runtime dominance. However candidate cost is **4.44% above rollout**, beyond the locked 2% DEV gate and the original goal's ≤1% planner margin. It is also worse in cost than the strongest comparator, so the original ≥5% better route fails. **Overall locked DEV go gate FALSE; no V17 replication or TEST.** The reserved 731xxxxx replication and 732xxxxx TEST blocks are retired unopened for V17.

## Case mechanism and failure examples

| Case | Rank cost | Expected Q cost | Rank−expected | Rollout cost | Rank−rollout |
|---|---:|---:|---:|---:|---:|
| reciprocal scale2 | 801.01 | 804.80 | −3.79 | 707.56 | +93.45 |
| interchangeable scale1 | 785.60 | 787.44 | −1.84 | 685.75 | +99.85 |
| reciprocal scale3 | 999.79 | 1022.94 | −23.15 | 1074.38 | −74.59 |
| specialist/generalist scale2 | 1008.35 | 1014.61 | −6.26 | 974.12 | +34.23 |

The controlled gain is largest in reciprocal3, where rollout is comparatively weak. In the two largest rollout-gap cases, rank actor H48 failures are 5.81/5.86 versus rollout 1.13/2.00; PM queue ticks 17.31/18.44 versus 2.50/5.13, while total service starts are nearly equal (51.18/50.19 versus 51.25/50.44). This is descriptive: the actor is not simply doing fewer services; timing/machine choice and future consequences remain the likely missing mechanism. Seed-level rank−expected in reciprocal2 was −29.94, +2.00, +7.31, −2.94, +4.62, and in interchangeable −2.81, −4.81, −7.50, −4.75, +10.69: target-case improvement is not seed-consistent. A severe reciprocal3 rank episode at train73000001/physical73011005 cost1345 with 11 failures and 55 CM queue ticks; another train73000001/physical73011009 cost1301 with 10 failures and 51 CM queue ticks. Do not infer generalization or physical scalability from this fixed four-machine/two-technician study.

## Compute and limitations

Full run used 184,320 on-policy training steps. Root forecasts were 5,224,040 transitions for rank and 5,206,256 for expected (+0.34% rank), with exactly 7,680 selected Q states per arm; differing legal action counts/continuations account for the small realized compute mismatch. Selection involved 2,240 episodes; final panel 1,792 episodes. Inference latency includes exact full matching scores, and rank/expected inference is effectively equal. Any stronger performance claim needs independent five-seed replication and fresh TEST, neither launched. V17's softmax-Q/KL projection is established RL machinery, as bounded in `v17_prior_art_boundary_vi.md`; this study supports an application-specific credit mechanism but not a novel generic optimizer or the user's full method/competition goal.

Next: on the already opened V17 final physical panel, evaluate frozen V14 count/joint selected checkpoints as additional historical structured comparators, explicitly post hoc, without changing the failed V17 gate. Then prelock a paired trajectory/first-divergence mechanism diagnostic in reciprocal2/interchangeable to distinguish premature service, wrong machine set/assignment, and continuation-Q bias; a new method must address the repeated H64 failure/queue gap rather than retune V17 rank loss on this panel.
