# Research journal — RL maintenance dispatch continuation

## 2026-10-20: New goal intake and source audit

- Read the attached full objective. Completion bar is stronger than the prior V1–V7 goal: component ablation **and** competitive performance against the strongest prelocked comparator by criterion A (≥5% cost reduction) or B (≤1% cost gap with ≥10× decision speed and no simple-heuristic domination), plus ≥5 training seeds and independent replication. V7 does not meet it.
- Confirmed authoritative checkout `codex/maintenance-framework-v1-lab` at `5b1124f`; three pre-existing user changes preserved. V1–V7 proposals/protocols/findings and source are present. Historical lab handoff and literature note reviewed; many other branches exist and require targeted read-only comparison before claiming a new method.
- Current best structured candidate: V7 shared canonical, cost 961.27 on opened V7 TEST; flat 916.58, risk 906.53, rollout 861.02. A relative-to-V7-panel criterion A target is ≤817.97 and B cost bound ≤869.63, but a future campaign must lock comparator/threshold from its own DEV/reference protocol before fresh TEST.
- Structural source finding: V6/V7 shared actor's edge score has no other-machine context; logits are additive over edges. Its critic sees full 56-D input. This proves an invariance to non-request machine age changes in legal action scores, **not** that this matters to dispatch cost. Need a discriminating DEV diagnostic before architecture choice.
- Rejected prior hypotheses remain rejected: V4 temporal age2 opportunity and V5 post-decision residual wait did not improve their preregistered primary contrasts. V6 shared tying and V7 shortest-duration allocation had conditional gains but remain below strong baselines.
- Next: lock and run a paired-state context diagnostic with current-information planner and action-value margin, assess reachable-state frequency and under-service mechanism. If no material decision headroom exists, redirect to learning/credit/action structure rather than adding context aggregation by default.

## 2026-10-20: Prelocked context probe

Read V6/V7 actor, 56-D vector observation, request lifecycle, legal matching and bounded rollout source. `EdgeSharingActor` receives local machine/worker/edge features and clock per edge; additive matching logits cannot incorporate an absent-request machine's age into a current legal action. This is a structural proof but not a cost result. Registered `research_result/v8_context_diagnostic_plan_vi.md` before code: fixed new DEV seeds, six trajectory sources, valid age≤1→2 pair, stratified hash-selected states, common six-tick planner forecast, material action flip margin and all-five-seed policy probes. Negative diagnostic outcome will redirect method search.

## 2026-10-20: V8 context diagnostic complete, independent audit PASS

Prelocked runner commit `a941c8f` pushed after 90 relevant tests and smoke. Full DEV run `v8_context_dev_20261020T051623Z` completed on fixed fresh diagnostic seeds; independent verifier regenerated trajectories, hash sampling, planner values and all policy outputs (`v8_context_dev_audit_20261020T051824Z`, PASS). 384 pairs: 49 planner flips, 31 with both-sided margin≥1, 4 with margin≥5, none ≥10. Shared actor legal logits exact invariant; canonical shared and flat actors 0 action flips; free shared 1 near-tie flip. Material four were WAIT→service in the age-2 state. Flat's nonresponse weakens an explanation solely based on structured representation. Rejected immediate direction: add global context encoder merely because invariant exists. Next experiment should quantify full current-request decision headroom on policy trajectories and decompose timing/subset/assignment/heuristic-mask losses.

## 2026-10-20: Prelocked current-request headroom probe

Registered `research_result/v8_headroom_diagnostic_plan_vi.md` before code. All V7 flat/shared free/canonical training replicas will be replayed on new DEV physical seeds. Fixed hash/clock-bin sampling and common bounded-planner root scores will decompose available-action actor mistakes from V7 canonical-mask exclusions and classify WAIT, subset/cardinality and technician assignment differences. This is an exploratory mechanism gate, not a fresh method confirmation. V8 absent-request context evidence alone did not support context aggregation.

## 2026-10-20: Headroom engineering smoke

The first smoke completed source trajectories but failed while serializing a NumPy integer action index to JSON; its FAILED timestamped directory is retained. The pure index selection was converted to Python `int`, focused tests reran, and a **new** smoke directory `v8_headroom_smoke_20261020T052343Z` completed 120 episodes and 180 fixed sampled states. This is an engineering correction before any DEV result, not an outcome-driven method change.

## 2026-10-20: Headroom DEV and independent replay

Full prelocked diagnostic completed 1,920 episodes and 720 sampled states; independent audit replayed all episodes, selected states, root scores and decomposition, PASS. Canonical mean six-tick root regret 3.355 per selected state: 3.307 available-action actor selection and only 0.047 canonical-mask restriction; only 3/240 canonical-source states excluded the forecast-best matching. Weak structured seeds 02–04 had much larger actor gaps than strong 00–01; risk matching often beat them on the same states. See `v8_headroom_diagnostic_findings_vi.md` for cases, operational context, provenance and limits. Reject mask removal or absent-request context alone as the main next method. Next: prelock a resource-competition-aware learned matching method and discriminating ablations, with scratch RL and strong baseline gates.
