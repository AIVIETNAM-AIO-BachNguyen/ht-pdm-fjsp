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

## 2026-10-20: Prelocked representability probe and smoke

The audited V8 headroom gap does not identify representation versus learning. Locked `v8_representability_probe_plan_vi.md` before implementation: exact local-edge collision census and a small supervised capacity check using only the audited 720 DEV states, physical-seed split, five optimizer starts and fixed 100-epoch budget. This is diagnostic, not imitation-based method evidence. Focused tests passed. Fresh Ubuntu smoke `v8_representability_smoke_20261020T055000Z` completed 128 census states and the tiny train/selection/probe path, with fresh timestamped artifacts. The smoke numbers are only engineering checks. Commit/push verified source before the all-state DEV probe.

## 2026-10-20: Representability census correction before result interpretation

Initial all-state artifact `v8_representability_dev_20261020T055400Z` completed, but review found its collision implementation omitted the prelocked different-observed-context check. Opposite preferences with identical 56-D observations can arise from independent finite forecast scenario noise. Mark this run superseded for collision conclusions, preserve it, and apply the exact full-observation-byte inequality. Supervised fit protocol remains fixed. Fresh smoke and source push precede fresh full DEV.

## 2026-10-20: Corrected representability DEV, independent census audit PASS

After pushed correction `f3b39ca`, fresh 720-state DEV run and independent signed-edge/full-observation census audit passed. Exact two-sided ≥5 forecast-margin collisions cover only 8/720 states (3 keys), ≥10 only 2 states. On the physical-seed heldout subset, five fresh supervised edge scorers have mean root regret 1.479 versus risk 1.456, with flat/joint means ~1.634 despite lower score-MSE. This is not RL evidence or proof of full actor sufficiency, but weakens the proposed global-architecture-only direction. The initial unfiltered collision run remains explicitly superseded. Next method should address RL action credit/optimization across legal matchings, with a clear scratch-RL ablation and compute accounting.

## 2026-10-20: V9 prelocked scratch-RL method and Ubuntu smoke

V8 diagnostics showed legal-action actor choice/seed instability while exact material local-edge collisions were rare, so V9 targets RL action credit. `method_v9_proposal_vi.md` and `framework_v9_dev_protocol_vi.md` were written before source/outcomes. The method samples full remaining H48+tail16 current-actor returns for every feasible matching with paired common randomness, then updates the same V7 shared-edge actor by all-action expected policy gradient. Matched chosen-action ablation receives the same counterfactual bank and training compute, using only the sampled-action gradient. This is scratch RL, not planner labels; Expected Policy Gradients and maintenance graph RL are acknowledged prior work. Fixed 5 training seeds, 6 iterations, 16 source episodes/case/iteration, 256 selected states/iteration, 2 scenarios/action, DEV-only checkpoint selection and strong context baselines. No TEST use.

Initial smoke `framework_v9_smoke_20261020T062056Z` FAILED at a duplicate `selected_states` trace field after successfully exercising source trajectories and counterfactual rollouts; retained. Fixed the trace schema and reran a fresh smoke `framework_v9_smoke_20261020T062142Z`, COMPLETED. Added fixed physical-seed uniqueness/disjointness preflight and test; 46 focused simulator/policy/V9 tests PASS and fresh final smoke `framework_v9_smoke_20261020T062335Z` COMPLETED with reward/cost/request checks and full development/comparator grid. Smoke cost contrasts have only one train seed and two physical seeds and are not research evidence. Next: commit/push verified V9 source/protocol before the full prelocked DEV campaign.
