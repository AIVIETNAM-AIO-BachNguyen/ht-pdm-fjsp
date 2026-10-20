# V13 matching-conditioned technician opportunity cost — audited DEV result

V13 source/protocol were pushed at `061c788` before the full run. The fresh five-training-seed DEV campaign `framework_v13_dev_20261020T141536Z` ended `COMPLETED` in 3,757.7 s. Independent audit `framework_v13_audit_20261020T151831Z/independent_audit.json` is **PASS**: source/artifact/comparator hashes, physical/source-policy streams, 240 full-H64 common-bank Q replays (184,636 transitions), 640 selected learned-episode replays, 4,480 selection episodes, 1,792 final evaluations, aggregation and bootstrap. The sealed TEST remains closed. This is exploratory DEV evidence, not goal completion.

## Locked contrast and performance

Both arms trained from scratch with the same 56-dimensional current observation, 3,219 instantiated parameters, legal matching catalogue, six updates, 16 on-policy episodes/case/update, 64 selected states/case/update and four stochastic H64 scenarios per legal root. `competition` applies the learned duration × unserved-machine-demand term; `no_competition` disables it. The latter has only 2,354 initially gradient-active parameters versus 3,219 for the candidate, a material capacity confound. The arms' on-policy trajectories differ; realized forecast transitions were 5,428,616 versus 4,775,876 despite the same per-state scenario and selection budget. The prelocked capacity-matched V14 control is therefore required before attributing gain to matching identity.

| Controller | Four-case mean cost | Selection s/episode |
|---|---:|---:|
| Competition V13 | **857.96** | 0.02192 |
| No competition | 876.13 | 0.01896 |
| Flat PPO | 905.97 | 0.01983 |
| V7 shared canonical | 916.25 | 0.02753 |
| Risk matching | 887.05 | 0.00137 |
| Bounded rollout | **845.70** | 0.65353 |

Primary paired candidate-minus-ablation difference: **−18.17**, hierarchical 95% bootstrap **[−39.42, −2.14]** over five training seeds and 16 shared DEV physical tapes. Seed differences are −29.67, −11.03, −18.66, −9.81 and −21.69; all five improve, including the three strongest ablation seeds. The prelocked V13 DEV promotion gate is **TRUE**: mean gain exceeds 10, candidate is within 2% of rollout, selection is about 29.8× faster, and risk matching does not dominate on cost and speed together.

| Case | Competition | Ablation | Rollout | Candidate−ablation |
|---|---:|---:|---:|---:|
| Interchangeable scale1 | 739.54 | 738.61 | 682.88 | +0.93 |
| Reciprocal scale2 | 762.86 | 773.09 | 706.56 | −10.23 |
| Reciprocal scale3 | 936.84 | 1006.51 | 1033.06 | **−69.68** |
| Specialist/generalist scale2 | 992.60 | 986.31 | 960.31 | +6.29 |

The gain is uneven across cases. In reciprocal scale3, candidate H availability is 0.502 versus 0.455, CM queue 0.875 versus 11.925 ticks and service starts 42.99 versus 38.36; this is the largest operating improvement. Specialist/generalist is a failure case: candidate cost rises 6.29, failures 9.56 versus 8.19 and CM queue 9.78 versus 7.94. Interchangeable also has a small cost loss. Candidate remains 1.45% above rollout on the four-case DEV aggregate and is worse on three of four cases. It misses the goal's default ≥5% improvement route and its ≤1% planner-cost route on this DEV panel; the 10× speed condition alone is insufficient. These are descriptive distances from opened DEV, not a new confirmatory threshold.

A retrospective cost-component decomposition of the already audited `episodes.csv` sharpens the remaining planner-gap question. Candidate-minus-rollout cost is +56.66/+56.30/+32.29 in interchangeable scale1, reciprocal scale2 and specialist/generalist scale2; failure-cost differences are +55.50/+54.38/+39.56, respectively (98%/97%/123% of those net gaps). H48 service starts differ by only 0.00/+0.01/+0.61 per episode, and H48 availability is slightly *higher* for the candidate by +0.002/+0.003/+0.010. This pattern is consistent with poorer preventive timing or machine choice and PM→CM conversion, rather than a simple shortage of total service starts. It is descriptive on selected DEV policies; it does not identify a causal action or establish that a new hazard module would help. A next discriminating probe should compare candidate/planner action timing and subsequent failure events on shared tapes before proposing another RL mechanism.

Training used 184,320 actual environment steps and 10,204,492 combined counterfactual forecast transitions. Both arms used five independent training seeds 72500000–04; selected candidate checkpoints were iterations 4, 6, 4, 5 and 5, with no untrained checkpoint selected. DEV physical seeds were 72510000–15. Source/config/seed hashes and checkpoint hashes are in the immutable run manifest; raw data/checkpoints remain outside Git.

## Prelocked same-checkpoint component intervention

The separately prelocked `v13_component_intervention_protocol_vi.md` toggles only the candidate penalty at inference on all 7,680 saved competition-arm **training** Q states, preserving each checkpoint, observation, legal mask and four-scenario root bank. Run `v13_term_toggle_dev_20261020T151959Z` and independent audit `v13_term_toggle_audit_20261020T152006Z` both completed; audit **PASS** for the full grid, all action/Q differences and bootstrap.

The mean full-minus-zero same-bank root cost is **−0.30**, hierarchical CI95 **[−2.02, +1.25]**; seed means are +0.42, +2.04, −1.13, +0.45, −3.27. The prelocked directional gate (≤−5, ≥4/5 negative seeds, CI upper <0) is **FALSE**. The term changed 3,049/7,680 root actions, mainly WAIT-related (2,342); with the term active there were 3,190 WAIT choices versus 848 when zeroed. Improved/worsened/tied rows were 1,430/1,606/4,644. This rejects a simple claim that the explicit term directly selects lower H64-Q actions on these stored training states. It does not explain away the closed-loop DEV gain: training trajectories and network weights diverged, and four-scenario old-actor Q may miss later effects.

## Decision and limits

V13 passes its registered exploratory DEV gate but **does not satisfy the overall goal**. Do not open TEST or claim novelty now. The capacity-matched V14 control was prospectively prelocked at `f872f20` while V13 was still running; its H1 trigger is now met. Implement and run that identity-specific control with fresh seeds, matching active capacity and compute, then use independent audit before considering any candidate freeze or fresh confirmation. Keep the original goal A/B cost, speed, five-seed, independent-replication and provenance requirements unchanged.

Primary-source boundary: `v13_prior_art_boundary.md` compares [Learning to Dispatch](https://proceedings.nips.cc/paper/2020/file/11958dfee29b6709f48a9ba0387a2431-Paper.pdf) and [HetGPO/AirME](https://cdn.aaai.org/ojs/19839/19839-40-23852-1-2-20220613.pdf). Neither RL dispatch, global maintenance context nor saving resources for future work is a V13 novelty claim. The narrower matching-specific penalty still needs broader prior-art review and mechanism evidence.
