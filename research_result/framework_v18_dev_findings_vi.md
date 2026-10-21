# V18 bounded continuation-disagreement credit — five-seed DEV findings

**Decision: prelocked DEV go gate FALSE. Original research goal remains ACTIVE.** This is development evidence only; no V18 replication or TEST was opened.

## Provenance and experimental contrast

Prospective method/protocol/seed registry were pushed at `3bc0515`; source, tests and independent verifier at `a22ea86` before the full run. The Ubuntu smoke and its independent audit passed; the full repository test suite passed 1,091 tests. Full run: `research_result/framework_v18_dev_20261021T050413Z` (manifest `COMPLETED`, training source commit `a22ea861c693575d69030342a5024d71a38e30eb`, 3,935.5 s). Independent audit: `research_result/framework_v18_audit_20261021T061005Z/independent_audit.json`, `PASS`; 15,360 saved complete dual-Q state rows, 240 stratified source/full-dual-Q bank replays (338,168 forecast transitions), all 640 selected learned and all 640 frozen V14 final episode replays, 2,240 checkpoint-selection episodes, 2,432 final episodes, physical-role/checkpoint/hash/bootstrap/gate checks. Audit sampling does not replay every Q bank; the run saved all banks and original states. Manifest SHA256 `508ae2511ce5e3ac935fd537841907dbb9d9ec4fabc069862bff816ef850f79f`; audit SHA256 `315d17c5ccccc7d494fe0dbb39bb9c47ff2ff8d02e18a327635aea930dab826e`.

Both scratch RL arms use the same 9,025-active-parameter full-matching actor, initial weights, legal actions including WAIT, 56-D current observation/74-D matching features, training and selection budgets. Each selected on-policy state computes all legal roots under the same four failure scenarios with both own-policy and deterministic risk continuation; both arms pay for both banks. `own_rank` projects from own-Q; `bounded_disagreement` adds the prelocked centered, 0.5-weight, ±40-clipped own/risk Q difference before the rank target. Assignment auxiliary, KL and soft service-flow terms are identical. Training seeds `73300000–04`, eight selection-only physical seeds `73310000–07`, sixteen final DEV physical seeds `73311000–15`, four unchanged cases equally weighted. No future tape is used at inference.

## Prelocked primary ablation and competitive gate

| Controller | Equal-case economic cost | Selection s/episode |
|---|---:|---:|
| Bounded disagreement | 876.57 | 0.01597 |
| Own-Q rank control | 883.03 | 0.01575 |
| Frozen V14 joint identity | 865.53 | 0.02394 |
| Frozen V14 count control | 866.04 | 0.02529 |
| V7 flat PPO | 906.43 | 0.02124 |
| V7 shared free / canonical | 1002.16 / 964.32 | 0.02784 / 0.02925 |
| FCFS / risk matching | 938.64 / 897.73 | 0.00120 / 0.00182 |
| Bounded rollout | **857.23** | 0.66037 |

Paired candidate−own-Q cost is **−6.47**, prelocked 20,000-draw hierarchical CI95 **[−17.87,+1.40]**, with four of five training-seed means negative: −6.59, −0.86, **+0.63**, −0.80, −24.72 in registry order. Two of three strongest-control seeds win, but the material ≤−10 and CI-upper<0 gates fail. Candidate is +11.04 above V14 joint and +19.33 (+2.26%) above rollout on this panel. It is 41.4× faster than rollout at action selection; however original criterion B requires ≤1% cost excess, here ≤865.81, while candidate is 876.57. Original criterion A would require ≤814.37 against the strongest locked comparator rollout. No simple risk-matching cost/runtime dominance (risk is cheaper to compute but more costly). The V18 promotion gate also fails candidate≤1.02×rollout and candidate≤V14 joint. Thus the positive point estimate is insufficient for method promotion or goal completion.

## Per-case and operational evidence

| Case | Candidate | Own-Q | V14 joint | Rollout | Candidate−own-Q | Candidate H failures / rollout | Candidate service starts / rollout | Candidate CM queue / rollout |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Interchangeable 1 | 756.58 | 753.39 | 753.74 | **685.62** | +3.19 | 4.00 / 1.00 | 51.60 / 51.62 | 0.10 / 0.75 |
| Reciprocal 2 | 775.49 | 778.92 | 776.69 | **708.50** | −3.43 | 4.24 / 2.06 | 50.51 / 51.00 | 1.41 / 1.25 |
| Reciprocal 3 | 976.14 | 1000.01 | **939.42** | 1038.44 | −23.87 | 6.33 / 8.44 | 40.21 / 38.81 | 7.67 / 18.81 |
| Specialist/generalist 2 | 998.06 | 999.81 | **992.26** | 996.38 | −1.75 | 8.12 / 8.44 | 39.64 / 40.81 | 11.61 / 12.25 |

Reciprocal3 supplies most of the ablation gain, especially training seed `73300004` (candidate−control −100.56 in that case). Its cost still exceeds frozen V14 by 36.72. Interchangeable remains the main gap to rollout: candidate and rollout have nearly identical service-start counts, yet rollout has 3 fewer H failures and far lower failure cost (15.00 versus 85.69); candidate PM queue is 12.80 ticks versus rollout 1.56. These aggregate episode metrics implicate timing/assignment and future failure exposure, but do not establish a causal first action. The prior V17 first-divergence probe motivates a fresh same-panel V18 trajectory diagnostic before another method design.

The candidate and control consumed 10,487,624 and 10,461,496 training forecast transitions, respectively; the slight difference comes from later on-policy trajectories diverging, while budget, root/scenario rule and bank computation are matched. Across 30 updates/arm, maximum final KL was 0.00106/0.00090, below the locked 0.02 limit. The recorded **last-minibatch** service-flow penalty was zero in all updates, while mean old→new expected service increased from 0.7734→0.7808 for candidate and 0.7714→0.7784 for control. This does not prove the penalty was never active in earlier minibatches or that it prevents a future WAIT collapse. No completed run was overwritten; raw Q rows/checkpoints remain in the timestamped directory outside Git.

## Interpretation and next gate

V18's clipped continuation signal changes training root scores but does not yield a stable or competitive closed-loop improvement. No claim of novel multi-policy value estimation, conservative/safe policy improvement, global matching, or stochastic maintenance RL follows; see `v18_primary_prior_art_review_vi.md` and its primary sources. Preserve this negative result and retire V18-specific follow-on replication/TEST until a new method is prelocked. Next, diagnose V18 versus rollout and V14 first divergence on the **already opened V18 DEV** final physical panel, with exact action category, own/risk continuation sign, case/seed/failure linkage and independent replay; then decide whether action timing, assignment, or longer-horizon credit is the discriminating failure. Any V19 training requires a new proposal/protocol, fresh DEV seed registry, tests/smoke, source push and a strong same-capacity ablation. The user's original A/B, independent replication and one frozen fresh TEST requirements remain unchanged.
