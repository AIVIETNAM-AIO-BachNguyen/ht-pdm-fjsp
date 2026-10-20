# V12 planner-correction budget curve — audited DEV, direct sparse consultation no-go

Prelock/source push `b4fe07a`, full immutable opened-DEV run `v12_planner_budget_dev_20261020T140034Z` **COMPLETED** in 225.1 s. Independent `verify_v12_planner_budget.py` audit `v12_planner_budget_audit_20261020T140436Z/independent_audit.json` **PASS**: 160 pairs × K∈{0,1,2,4}, all 640 full episodes and 30,720 logged decisions, planner-query/correction/forecast counts, original V12 K0 and prior intervention K1 source metrics, source/checkpoint hashes, operational aggregates and fixed 20,000-draw hierarchical bootstrap. No TEST or new evaluation tapes were opened.

At each tick while correction budget remains, an oracle queries the locked bounded rollout on the actor's *current state*. It executes the planner action only when actor and planner disagree, counts one correction, and reverts permanently to actor when K corrections have been used. A matching planner query still consumes compute. The policy is an oracle diagnostic, **not** a deployable learned RL controller.

| Case | K | Cost | Gap to full rollout | Gap recovered | Planner queries/episode | Actual corrections | Selection s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|
| Interchangeable scale1 | 0 | 738.19 | +60.56 | 0% | 0.00 | 0.00 | 0.012 |
|  | 1 | 711.08 | +33.46 | 44.8% | 23.35 | 0.56 | 0.335 |
|  | 2 | 681.62 | +4.00 | 93.4% | 32.94 | 0.93 | 0.477 |
|  | 4 | **677.62** | 0.00 | 100% | 48.00 | 0.97 | 0.723 |
| Reciprocal scale2 | 0 | 851.46 | +141.09 | 0% | 0.00 | 0.00 | 0.012 |
|  | 1 | 779.94 | +69.56 | 50.7% | 9.44 | 0.85 | 0.164 |
|  | 2 | 754.96 | +44.59 | 68.4% | 22.16 | 1.45 | 0.348 |
|  | 4 | 723.89 | +13.52 | 90.4% | 41.54 | 1.96 | 0.648 |

Locked full rollout cost/selection time was 677.62/0.710 s in interchangeable and 710.38/0.772 s in reciprocal2. K2 interchangeable reaches within 0.59% planner cost, but query time is 0.477 s, far slower than the 0.071 s limit for 10× speed. K4 reaches rollout cost in interchangeable only by querying the planner on **all 48 ticks** on average. In reciprocal2 K4 still costs 1.90% more than rollout and queries 41.54 ticks. K4 seed-level reciprocal2 gaps are +1.38, +38.56, +16.75, +9.50 and +1.38; only two of five seeds are within the 1% cost margin. The hierarchical CI95 for reciprocal2 K4 minus rollout is [−6.30,+45.67], so its apparent closeness is uncertain. Every prelocked K cost-headroom and direct-runtime-headroom gate is **FALSE**.

Operational changes track the cost curve. In interchangeable, H failures fall 3.29→2.15→0.71→0.56 from K0→1→2→4, while mean H availability remains about 0.60. In reciprocal2, H failures fall 5.71→4.29→2.81→2.36, availability rises 0.548→0.578→0.578→0.588, and queued CM ticks fall 7.01→3.01→3.67→2.16. Reciprocal2 service starts rise 46.80→49.41→49.30→50.55. Mean forecast transitions per K4 episode are 10,956/10,280 in the two cases. These are costs paid to *find* corrections; counting only executed planner actions would severely understate compute.

**Decision.** Sparse direct consultation cannot supply the goal's planner-level cost and ≥10× speed on these two target cases. Even an oracle with four corrections misses reciprocal2 cost and queries most ticks. A cheap learned trigger could remove query overhead, but the K4 cost shortfall and seed heterogeneity mean a trigger alone is insufficient. The next major method should make learned long-horizon resource-competition decisions repeatedly, including machine-set and WAIT timing, rather than depend on a handful of planner calls or fixed WAIT duration. Any future planner labels may support training later but cannot be the main contribution under the goal. No RL method has yet met A/B; goal **ACTIVE**.

Reproduce audit: `uv run python research_result/verify_v12_planner_budget.py research_result/v12_planner_budget_dev_20261020T140034Z --output-dir research_result/v12_planner_budget_audit_$(date -u +%Y%m%dT%H%M%SZ)` from repo root. The audit refuses an existing output directory.
