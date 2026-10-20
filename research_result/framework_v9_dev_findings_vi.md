# V9 paired counterfactual action credit: audited DEV, go gate failed

**Verdict:** The prelocked V9 scratch-RL DEV campaign does not advance to fresh confirmatory TEST. All-action paired counterfactual credit lowered cost relative to a compute-matched sampled-action update, but most of that effect rescued two weak training replicas. The resulting actor was still substantially worse than risk matching and bounded rollout on the four-case aggregate. This is a useful negative method result, not goal completion.

## Protocol and independent verification

Proposal `method_v9_proposal_vi.md`, protocol `framework_v9_dev_protocol_vi.md`, registry, source and tests were committed/pushed as `6f9a01f` **before** full DEV. The V9 actor was the same 1,597-parameter V7 shared-edge scorer with free legal-action mask in both arms; only the action-credit update differed. All-action and sampled-action ablation both received the same full-H48+tail16 two-scenario current-actor counterfactual bank on their own on-policy states. No planner/risk teacher labels, imitation, reward shaping, physics or objective change entered training. The fixed requester, simulator and feasibility mask were unchanged. The first engineering smoke failed at a duplicate trace key and is retained; the corrected final smoke and 46 focused tests passed before push.

Full raw run `research_result/framework_v9_dev_20261020T062502Z/` completed within its 3-hour cap, in 1,386 wall seconds. Five independent V9 training seeds × two scratch-RL arms × six iterations used 184,320 source environment steps, 15,360 hash-selected multi-action states and 4,966,404 extra training forecast transitions (2,469,664 all-action; 2,496,740 sampled-action). Both arms sampled 256 states each iteration; no KL early stop. DEV selection examined 4,480 checkpoint episodes (iterations 0–6), then froze one checkpoint per arm/seed before the 1,792-episode common physical-seed comparator grid. Selection and all checkpoint hashes were saved.

Independent audit `research_result/framework_v9_audit_20261020T064820Z/independent_audit.json` **PASS**. It verified source/output and immutable V7 comparator hashes, fixed physical/source-policy streams, independently replayed 240 sampled full-horizon root banks (81,080 transitions) and all 640 selected V9 episodes with cost/health/request metrics, checked complete selection/evaluation grids, and recomputed selection, aggregates and hierarchical bootstrap. All 24 smoke Q records and 16 smoke learned episodes were also independently replayed before full audit. Feasibility came from the legal mask, not a learned constraint.

## Prelocked DEV result

Mean H48+tail16 economic cost, equally weighted over the four cases and training/physical seeds (lower is better):

| Controller | Case-balanced cost | Selection seconds per episode |
|---|---:|---:|
| V9 all-action RL | **961.27** | 0.0121 |
| V9 sampled-action ablation | 1,028.35 | 0.0120 |
| V7 flat PPO | 918.10 | 0.0196 |
| V7 shared canonical PPO | 933.99 | 0.0274 |
| V7 shared free PPO | 990.01 | 0.0260 |
| FCFS | 922.02 | 0.0007 |
| Risk matching | **863.72** | 0.0014 |
| Bounded rollout | **857.81** | 0.6878 |

V9 all-action − sampled-action was **−67.08** cost, paired hierarchical CI95 **[−141.54, −0.47]** on this opened DEV panel (five training seeds, 16 physical seeds, four cases). Training-seed mean deltas were **−0.83, −179.98, −3.02, −1.05, −150.52**: two seeds contribute nearly all the aggregate gain. Thus the CI does not establish a broad design benefit beyond weak-replica rescue. The V9 all-action training-seed costs were **883.00, 891.14, 1,071.41, 1,071.97, 888.84**, while sampled-action costs were 883.83, 1,071.12, 1,074.42, 1,073.02, 1,039.36. Four-case DEV checkpoint selection chose all-action iterations 6, 6, 1, 6, 5 respectively. It did not remove the bad policy mode in seeds 02–03.

Against the strongest locked comparator on this fresh DEV panel, all-action cost was **+103.46** above rollout, descriptive hierarchical CI95 **[+23.38, +186.17]**, about 12.1% worse. It was **+97.55** above risk, CI95 **[+18.47, +178.01]**. V9 was about **56.8× faster** than rollout at decision time, but risk matching was both cheaper and about **8.8× faster** than V9. The prelocked DEV go gate (≥10 cost benefit, ≥4/5 training-seed wins, ≤2% rollout cost gap and ≥10× decision speed) **failed**: the ablation mean, five directional seed wins and speed parts passed, but the component value was concentrated in two seeds and the planner cost gap was far too large. The stronger goal completion criteria A and B plainly fail; no new TEST or independent replication is justified for V9 as-is.

### Per-case economic and operational pattern

| Case | All-action | Sampled action | Flat | Risk | Rollout | All-action H availability | All-action H failures | All-action queued CM ticks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Interchangeable scale1 | 745.94 | 749.23 | 824.89 | 755.50 | 691.06 | 0.60 | 3.95 | 0.50 |
| Reciprocal scale2 | 936.83 | 1,103.78 | 826.81 | 751.06 | 688.19 | 0.50 | 7.99 | 14.00 |
| Reciprocal scale3 | 1,163.53 | 1,255.38 | 988.41 | 953.94 | 1,052.00 | 0.38 | 8.93 | 29.88 |
| Specialist/generalist scale2 | 998.80 | 1,005.02 | 1,032.30 | 994.38 | 1,000.00 | 0.50 | 8.39 | 10.40 |

The severe failure is concentrated in reciprocal cases. All-action began about 41.76 services per episode on reciprocal scale2 and 30.51 on scale3, versus risk's 51.69 and 42.44; queued corrective exposure was 14.00 and 29.88 ticks versus risk's 0.19 and 1.25. Its unavailability cost in those cases was 749.77 and 936.75, versus risk's 613.50 and 765.00; failure cost was 136.31 and 184.50, versus 80.62 and 137.81. This is consistent with insufficient or ill-timed service under some learned policies, but episode aggregates alone cannot identify whether they choose WAIT, the wrong machine subset, or an inferior technician assignment on a particular state. A prelocked action trace is needed before a V10 mechanism claim.

## Interpretation, prior work and limits

[Expected Policy Gradients](https://www.jmlr.org/papers/volume21/18-012/18-012.pdf) already derives a discrete all-action policy gradient; it reports no measurable gain in its Pong study when a learned critic is inaccurate. V9 used simulator Monte Carlo current-policy returns in place of a learned Q critic, with common randomness across joint matching roots, yet still suffered policy-mode instability. [HetGPO/AirME](https://ojs.aaai.org/index.php/ICAPS/article/download/19839/19598/23852) already studies shared-crew maintenance RL and uses a time-step return baseline for stochastic credit; V9's paired same-state action bank differs in estimator and fixed N4/K2 persistent-request setting, not in the general idea of maintenance RL or variance reduction. [Learning to Dispatch](https://proceedings.neurips.cc/paper_files/paper/2020/file/11958dfee29b6709f48a9ba0387a2431-Paper.pdf) and action-value credit methods also predate this work. No novelty or scalable-generalization claim follows from the V9 result.

The counterfactual Q bank used only two scenarios per state and old-policy continuation; its finite-sample rank and distribution shift may limit improvement. The V9 actor remained local-additive, and the six policy-improvement iterations/256 states per iteration were fixed rather than tuned after outcomes. Training CPU/wall and forecast compute were much larger than inference compute; the V7 PPO comparisons were read-only checkpoints with different training budgets and cannot isolate the new update. The DEV panel is opened and exploratory. Next: characterize the reciprocal-case deterministic action pathology on fixed V9 selected on-policy states, then choose a falsifiable RL design that improves both strong and weak seeds rather than adding a heuristic restriction or declaring a conditional ablation success.
