# V9 selected-policy DEV trace: reciprocal failure is technician assignment

This mechanism probe was prelocked in `v9_action_failure_diagnostic_plan_vi.md` after the audited V9 DEV no-go and before code/outcomes. It reuses already-opened V9 development physical seeds only. It is not TEST, a new method, fresh confirmation or proof of episode-level causality.

## Provenance and verification

Runner/protocol/tests were committed and pushed as `adabba3` after 7 focused tests and fresh Ubuntu smoke. Full trace `research_result/v9_action_failure_dev_20261020T065810Z/` completed all **640 selected V9 episodes, 30,720 H decisions, 240 hash-stratified bounded-root states and 84,672 forecast transitions**. It replayed each episode objective, operational metrics and request conservation against the immutable V9 DEV artifact. Independent audit `research_result/v9_action_failure_audit_20261020T070011Z/independent_audit.json` **PASS**: all source/input/checkpoint/output hashes, complete episode/action replay, exact hash sample, every 16-scenario common-bank six-tick root score, choices, taxonomy and summary. The separately run smoke completed 40 episodes and 120 root states. No model, simulator, cost or objective was changed.

## Which dispatch decision fails?

Contrary to the prelocked H1 timing conjecture, **neither V9 arm ever chose WAIT when a service matching was legal in either reciprocal case**. On its own visited states, each arm also chose the **same service cardinality as risk matching** throughout both reciprocal cases. The lower closed-loop service-start counts reported in V9 therefore are not direct WAIT/low-cardinality decisions on those visited states; they accompany different machine/technician choices and the resulting future resource occupancy.

The standout difference is assignment of technicians to the **same machine set**. Across the two reciprocal cases:

| Selected V9 policy | Training seeds | Decisions | Technician-assignment disagreements vs same-state risk | Extra known duration of each disagreement | WAIT with legal service |
|---|---|---:|---:|---|---:|
| All-action | strong 00/01/04 | 4,608 | **0** | — | 0 |
| All-action | weak 02/03 | 3,072 | **310** | **all 310 longer** | 0 |
| Sampled-action | all five | 7,680 | 599 | all 599 longer | 0 |

For weak all-action seed 02, reciprocal scale2 had 87 assignment disagreements, each longer than risk by a mean **7.68 ticks**; seed 03 had 93, mean **7.44**. On reciprocal scale3, each weak seed had 65 disagreements, each longer by a mean **11.45 ticks**. Strong all-action seeds had zero technician-assignment disagreements there, although they had some different machine subsets. In the weak all-action trajectories, a nonempty matching was legal at 649/3,072 reciprocal ticks, versus 2,012/4,608 strong-seed ticks; their same-state service cardinality still matched risk whenever legal. Long assignments occupying technicians are a concrete mechanism consistent with fewer later service opportunities and large corrective queues. This is a same-state/action and temporal-association finding, **not yet a closed-loop intervention proving that those assignment choices alone caused the entire cost difference**.

Case-level all-action taxonomy versus risk: interchangeable scale1 had 15 WAIT-with-legal-service ticks among 2,168 legal opportunities, reciprocal scale2 had zero among 1,531, reciprocal scale3 zero among 1,130, and specialist/generalist scale2 had 3 among 2,202. Reciprocal assignment disagreements were 180 (scale2) and 130 (scale3) for all-action; 359 and 240 for sampled-action. Thus adding a broad service/WAIT gate as V10 solely from V9 episode starts would target the wrong observed failure.

## Bounded current-information value check

All 240 sampled states had at least two legal roots. A fresh forecast stream (anchor 70360000) scored every legal root under 16 common scenarios, six ticks and risk continuation. On **all-action reciprocal-source states**, mean root regret relative to bounded planner was **2.69** across 36 strong-seed states versus **20.32** across 24 weak-seed states. Risk on those same states had regret 0.32 and 3.83 respectively. Weak all-action regret was ≥5 in **16/24** states and ≥10 in **15/24**; seven weak states differed from planner only by technician assignment, while six chose service when planner preferred WAIT. Nine of the 24 weak reciprocal states selected a noncanonical longer-duration assignment with a ≥5-unit gap to the best shortest-duration assignment for the same machine set. No strong all-action reciprocal sampled state selected a noncanonical assignment. This supports a material local assignment-credit problem and also warns that timing after assignment can matter.

A hard shortest-duration mask is **not** a universally valid feasibility rule. The bounded planner's best root was noncanonical in **3/240** sampled states, all in specialist/generalist sampled-action trajectories. V7 already showed only conditional benefit from such a mask and remained below strong rules. The next RL design should let assignment be learned and compared conditionally, with duration/opportunity information, rather than relabeling a heuristic restriction as physics.

## Decision and limits

The prelocked timing H1 is rejected for reciprocal V9 on-policy states. The evidence favors an **assignment-specific RL credit and policy factorization** question: can a learned controller explicitly compare technician assignments for a fixed serviced machine set, then use their values when choosing the set, while preserving legal non-shortest choices? A short, prelocked paired closed-loop intervention swapping only assignment is warranted to test how much of the weak-seed episode cost gap it can explain before a V10 architecture is fixed. If the intervention accounts for little, investigate machine-subset and long-horizon effects instead. Any V10 method must also improve strong replicas and challenge flat/risk/rollout; merely rescuing weak seeds will not satisfy the unchanged goal.

The bounded planner has finite-scenario six-tick values and risk continuation; its regrets are not true H48+tail economic regret or optimality proof. The 240 states are on different policies' own state distributions, and risk disagreements are not errors by definition. Full episode cause remains unverified until a paired intervention. No N/K, permutation, real-world or generalization claim follows from this trace.
