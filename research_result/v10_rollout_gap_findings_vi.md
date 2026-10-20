# V10 selected-policy DEV diagnosis: rollout gap is not a frequent six-tick assignment gap

The DEV-only mechanism protocol `v10_rollout_gap_diagnostic_plan_vi.md` was locked before code/outcomes after V10's audited no-go. Runner/tests/Ubuntu smoke were pushed at `75586d0` before full trace. Full artifact `research_result/v10_rollout_gap_dev_20261020T075203Z/` replayed all **640 selected V10 episodes and 30,720 H decisions**, then scored **240 hash-stratified states and 83,424 bounded forecast transitions** from new forecast anchor 70460000. Independent verifier `verify_v10_rollout_gap.py`, artifact `research_result/v10_rollout_gap_audit_20261020T075306Z/independent_audit.json`, is **PASS** for input/source/checkpoint hashes, every trajectory and saved objective, exact state sample, all common-bank 16-scenario six-tick root values, action taxonomy, canonical-mask exclusions and summary. No new TEST, model, cost or simulator change was made.

## Full on-policy action trace

On the two target cases, V10's learned actions are close to the same-state risk rule, and **neither arm ever WAITs when any service matching is legal**:

| Case and V10 arm | H decisions | Legal service opportunities | WAIT with legal service | Exact V10/risk matching | Different machine set | Different technician assignment |
|---|---:|---:|---:|---:|---:|---:|
| Interchangeable scale1, assignment credit | 3,840 | 2,465 | 0 | 3,806 | 34 | 0 |
| Interchangeable scale1, λ0 joint credit | 3,840 | 2,465 | 0 | 3,809 | 31 | 0 |
| Reciprocal scale2, assignment credit | 3,840 | 2,445 | 0 | 3,788 | 50 | 2 |
| Reciprocal scale2, λ0 joint credit | 3,840 | 2,445 | 0 | 3,788 | 50 | 2 |

The earlier V10 closed-loop rollout cost advantage is not a weak-seed artifact: in interchangeable scale1, V10 assignment-credit cost exceeds rollout by **70.06–76.06** for every training seed; in reciprocal scale2 by **79.00–108.12**. On these V10 trajectories the actor largely implements risk matching, while rollout's *closed-loop* decisions produce fewer failures. Agreement with risk is descriptive, not proof risk is optimal.

## Same-state bounded root values

The fixed sample has 30 states per case×arm. Its 16-scenario six-tick planner uses the same current information and risk continuation for every candidate root. Regret here is planner-root estimated cost gap, **not full H48+tail16 episode regret**.

| Source case/arm | Mean six-tick V10 root regret | States ≥5 gap | V10 versus bounded planner action classes |
|---|---:|---:|---|
| Interchangeable, assignment credit | **0.30** | **0/30** | 26 match, 3 service where planner waits, 1 higher cardinality |
| Interchangeable, joint credit | 1.13 | 2/30 | 22 match, 5 service where planner waits, 2 machine subset, 1 higher cardinality |
| Reciprocal2, assignment credit | **1.56** | **1/30** | 26 match, 4 service where planner waits |
| Reciprocal2, joint credit | 0.24 | 0/30 | 26 match, 4 service where planner waits |

For V10 assignment credit, bounded planner's root is **WAIT in 7/60** sampled target-case states where V10 services, but only **1/60** has a ≥5 bounded-root gap (a reciprocal2 state with estimated 40.38 units); there are **no technician-assignment-only disagreements** with planner in these 60 states. The prelocked H1/H2 requirement of frequent ≥5 timing/subset/assignment gaps across four seeds is **not met**. A targeted sixth-tick value/canonical-mask remedy is unsupported by these data. The 16-scenario forecast has sampling variance and risk continuation; small root gaps can coexist with large long-horizon and state-distribution effects.

Across all 240 sampled states, the bounded planner preferred a noncanonical assignment in **3**, all specialist/generalist states. In the specialist case, V10 assignment credit has mean root regret 1.96 with 4/30 gaps ≥5; its most frequent nonmatch is a different machine subset. This continues to caution against treating the V7 shortest-duration mask as feasibility. Reciprocal3 mean root regret is 0.45 for assignment credit despite V10 beating rollout's full episode cost there. Local bounded root values alone do not rank complete policies reliably.

## Decision and limits

This diagnostic favors the prelocked H3: the two large closed-loop rollout advantages are **not explained by frequent material six-tick root disagreements on the sampled V10 states**. It does not prove that long-horizon timing is the cause, because rollout's own state distribution and full H+tail counterfactual values were not examined. No V11 algorithm is selected yet. Next prelock a same-tape rollout-versus-V10 trajectory and long-horizon first-divergence probe, with common physical/forecast randomness, to test whether strategically delaying a currently legal service changes downstream failure and queue outcomes under a consistent continuation. Only then decide whether a planner-bootstrapped RL actor or another temporal-credit method is warranted. The V10 DEV gate and unchanged final goal criteria remain unmet; keep the goal **ACTIVE** and do not open TEST from this diagnostic.
