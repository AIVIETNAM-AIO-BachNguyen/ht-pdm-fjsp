# DEV-only V8 diagnostic: non-request machine context

Written 2026-10-20 after the prelocked plan `research_result/v8_context_diagnostic_plan_vi.md`, source commit `a941c8f` pushed before the DEV run. Raw run `research_result/v8_context_dev_20261020T051623Z/` is COMPLETED; independent audit `research_result/v8_context_dev_audit_20261020T051824Z/independent_audit.json` **PASS**. Smoke `research_result/v8_context_smoke_20261020T051522Z/` and its independent audit PASS. Ninety relevant simulator/framework tests PASS. This is exploratory DEV evidence; no new TEST, training or method superiority claim.

## Structural fact and test

The V6/V7 shared actor receives 56 features at the policy interface, but its edge scorer uses only clock, the target machine, that technician's first two features and their own eligibility/duration. Logits for a matching sum its selected edge scores. If an unrequested healthy machine is absent from all legal matchings, changing only its age cannot alter any legal matching logit. The critic's 56-D input does not remove this actor restriction. The flat MLP can, in principle, use the absent machine's age.

With fresh diagnostic physical seeds 69000000–31, the runner generated 768 trajectories over four fixed N4/K2 cases from risk matching and all five selected V7 `shared_canonical` checkpoints. For each source/case, it hash-selected 16 eligible states from ticks 4–41, giving **384 pairs**. Each pair changes only the lowest-index healthy, unrequested machine's age from 0/1 to 2; both versions pass state validation, have identical current requests, submissions, worker state and root legal actions. At the next tick the changed state creates a threshold PM request, while the original does not. The paired-state probe is valid but not a common-past physical trajectory claim.

The bounded current-information rollout used the same 16-scenario, six-tick forecast bank for both states of each pair. It changed root matching in **49/384** pairs; 31 flips had ≥1 cost unit advantage for both directions, only **4/384** had ≥5, and none had ≥10. Case counts at the prelocked ≥5 threshold: interchangeable 2/96, reciprocal2 1/96, reciprocal3 0/96, specialist 1/96. All four material flips were WAIT in the original state versus immediate service in the age-2 state, with two-sided advantages 5–8 units. The planner is a bounded rollout, not an optimal oracle; its six-tick margin may miss longer effects or have scenario error.

All five selected models from each of `shared_free`, `shared_canonical` and `flat_free` were probed on the 384 pairs (1.920 policy observations per arm). Shared legal-logit max error was **0** for both structured arms. `shared_canonical` changed **0** actions; `shared_free` changed **1** near-tie action despite equal legal logits and not on a material planner flip. Flat PPO changed **0** actions despite full observation. On the four material planner flips, all 15 models chose service in both states and incurred 5–7 forecast cost units in the original state versus the bounded planner's WAIT; they did not adjust to changed demand. Mean six-tick root-score regret on original/changed states: flat 2,375/2,623; shared canonical 3,244/3,068; shared free 6,798/6,589. These are common-state bounded forecast scores, **not** closed-loop economic cost or independent episode samples.

## Interpretation and next discriminating question

The shared actor has a proven representation invariant, and a current-information planner sometimes values the omitted context. On this fixed natural-state probe, material planner flips are scarce and the flat actor with full input also fails to react. Thus the data do **not** justify context aggregation as the next architecture by itself. They point to possible action-selection/learning issues, especially the tendency to start service regardless of a short-horizon WAIT opportunity. The much larger V7 performance deficit in reciprocal cases also remains unexplained by these four material context pairs. A targeted DEV diagnostic should compare policy root actions against planner root values on full current-request states, decomposing WAIT, machine-set, matching cardinality and technician assignment errors, including actions excluded by the V7 shortest-duration mask. Only after identifying a frequent, economically meaningful headroom class should a method be proposed. Historical assignment/headroom branches studied a different earlier simulator and are context, not confirmation for this request-lifecycle setting.

Independent verifier regenerated all 768 source trajectories and SHA-ranked selections, validated every original/perturbed state and root mask, recalculated all 384 paired planner banks, reloaded hashed checkpoints and reproduced all 5.760 policy rows/actions/logits/regrets. Source/artifact hashes PASS. No V1–V7 opened TEST seed was reused as confirmation.

## Reproduction

```bash
uv run pytest -q tests/test_v8_context_diagnostic.py tests/test_maintenance_framework_v*.py tests/test_maintenance_requests.py tests/test_maintenance_dispatch.py tests/test_maintenance_dispatch_policy.py tests/test_maintenance_request_experiment.py
uv run python research_result/diagnose_v8_context.py --profile dev --output-dir "research_result/v8_context_dev_$(date -u +%Y%m%dT%H%M%SZ)"
uv run python research_result/verify_v8_context.py --run-dir research_result/v8_context_dev_20261020T051623Z --output-dir "research_result/v8_context_dev_audit_$(date -u +%Y%m%dT%H%M%SZ)"
```

A replay of these diagnostic seeds is a technical replication; a new method needs a new locked DEV/TEST and independent replication design.
