# Conditional V14 replication and single-TEST reservation — unopened

Written while V14 DEV remains RUNNING and before its selected DEV scores are read. This reserves a disjoint validation path **only if** the prelocked V14 identity-control DEV gate passes its independent audit. It does not freeze a candidate, open TEST, authorize a claim, or substitute for a fully specified campaign protocol before execution. If V14 fails its gate, this reservation is retired unopened; the next method must have a newly prelocked plan. The goal's original A/B, five-training-seed, independent-replication and provenance requirements remain unchanged.

## Disjoint seed blocks

| Role | Reserved integer seeds |
|---|---|
| Independent replication scratch training | 72700000–72700004 |
| Replication checkpoint selection DEV | 72710000–72710015 |
| Replication held-out evaluation | 72720000–72720031 |
| Replication forecast anchor | 72730000 |
| Replication bootstrap | 72740000 |
| One candidate confirmation TEST physical panel | 72810000–72810063 |
| Confirmation bootstrap | 72840000 |

All seven blocks were checked against every integer and expanded `start/count` interval in existing `configs/*seed_registry.json`, including historical TEST registries; collisions: zero. They are mutually disjoint and separate from V14's 726xxxxx and V13's 725xxxxx. No physical tape for these blocks has been generated or evaluated. Their exact roles, hash audit and any additional training-policy random streams must be frozen in a source/config registry before a replication run.

## Conditional decision order

1. Finish V14 full DEV, independent main audit and same-checkpoint diagnostic audit. Apply the locked V14 H1/H2 gate without using partial scores or post-hoc case selection. If it fails, keep TEST sealed and pursue the already documented causal resource-commitment question on opened DEV.
2. If V14 passes, freeze its actor formula, source commit, five original training seeds, per-seed selected checkpoint hashes, V14 count-calibrated ablation, comparators, 56-coordinate current information, all legal matchings, inference timing procedure and objective. Write and push a detailed independent replication protocol before training new seeds. Replication repeats both V14 arms from scratch with the same six-update budget and checkpoint-selection rule, but selects checkpoints on 72710000–15 and evaluates them once on 72720000–31. Use the same equal-case paired cost estimand and hierarchical train-seed/physical-seed uncertainty, and independently audit all artifacts. A failed replication does not trigger TEST.
3. Only after replication supports the identity component and a competitive route, freeze one confirmatory TEST protocol and code, then evaluate the frozen original plus replication checkpoints once on 72810000–63. Report all four cases, maintenance/availability/failure/queue metrics, train/inference compute, each training-seed result and failures. Comparator set includes the capacity-matched count control, V13 competition, flat PPO, prior structured policies, FCFS/risk matching and bounded rollout. Use paired uncertainty across training seeds and physical seeds; account for the two alternative A/B routes and multiple comparators explicitly. The original goal requires either at least 5% lower aggregate cost than the strongest locked comparator with supporting uncertainty, or cost within 1% of bounded rollout with ≥10× faster selection and no simpler heuristic dominating both cost and runtime. Do not reinterpret a DEV-only pass or one favorable case as confirmation.

There will be at most one fresh TEST panel for this frozen V14 candidate. Its results cannot be used to tune and reopen more TEST panels for the same method. No SSH, rsync or external service is needed; full replication/TEST may run directly on the authorized Ubuntu lab only after their source, protocol and tests/smoke are pushed, with tqdm, hard timeout and new timestamped `research_result/` directories.
