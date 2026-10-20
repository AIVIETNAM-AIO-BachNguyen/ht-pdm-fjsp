# Current research state — shared-resource maintenance RL

Updated: 2026-10-20 (Asia/Taipei). Goal: **ACTIVE** under the new V1–V7 continuation objective in the attached user request. This state file records what is established; it is not a completion claim. Full requirement text is in `/home/bachng/.codex/attachments/6230f6b6-fe5d-4e53-bd7d-8f728b748019/pasted-text-1.txt`.

## Authoritative checkout and preserved work

Branch `codex/maintenance-framework-v1-lab`, HEAD `5b1124ffd965b333579be85c3ae177ab8fdcd316` at the new goal start. Existing user changes are `research_result/framework_v3_findings_vi.md` (modified) and untracked `docs/codex_reservation_framework_goal_vi.md`, `docs/technician_reservation_framework_direction_vi.md`; preserve them. V1–V7 protocols, proposals, compact findings, source and verifiers are in this branch. Historical branches remain available for read-only comparison; do not assume branch recency implies better evidence.

## Established evidence and best candidate

V4 healthy-age anticipation and V5 post-decision residual-wait mechanisms failed preregistered contrasts. V6 shared edge head versus matched-budget unshared heads passed narrowly (cost −85.59, hierarchical CI95 [−171.24,−0.064]); V7 shared actor plus shortest-duration assignment mask beat free shared actor (−47.69 [−85.44,−13.73]) mainly by rescuing three weak training seeds. Neither establishes novelty or overall superiority. The V7 canonical shared policy is the current **best structured RL candidate**, not the best controller: case-balanced cost 961.27 versus flat PPO 916.58, risk matching 906.53 and bounded rollout 861.02 on V7 TEST. Rollout selection time 0.675 s/episode versus canonical shared 0.028 s. Strongest locked V7 comparator is rollout. On this panel, a 5% improvement target would require ≤817.97 cost; planner non-inferiority at +1% would require ≤869.63 while retaining ≥10× speed. These are planning distances from an opened panel, **not future confirmation thresholds computed from a new TEST**.

The new goal requires a method with defensible mechanism/prior-work distinction, fresh ablation accounting for ≥5 training seeds, competitive baseline criterion A or B, independent replication, per-case/operational/compute/failure reporting, and complete pushed provenance. None is yet proven. Do not mark complete based on V7 conditional positive result.

## Actor information bottleneck to test

`EdgeSharingActor` consumes all 56 observation coordinates as a tensor, but each edge score receives only the clock (2), its machine embedding (8→24), its technician embedding (first 2 of 3 worker coordinates→12) and its own eligibility/duration edge features (2). The third worker coordinate (assigned-machine identity) is excluded. Matching logits are sums of selected edge scores. Thus when a machine has no request and is absent from all currently legal matchings, changing only that machine's age/health cannot change **any legal actor logit**, though the critic sees the full observation. The flat actor can in principle respond. This is a structural invariant, not yet evidence that the omitted information changes an economically important current decision. Next diagnostic must test reachable, validated paired states and decision value with a current-information planner/forecast, then distinguish representation from optimization and action-structure failures. No V8 architecture is selected yet.

## Next gate

Write a prelocked, DEV-only context-sensitivity diagnostic plan (hypothesis, state construction, splits, metrics, stopping, artifact schema). Use new diagnostic seeds outside all registries; generate valid paired states that share current requests/workers but differ in age of a non-request machine. Verify mask equality, actor logit invariance, flat/planner behavior, and root-action value differences under common forecast tapes. Keep this exploratory and do not reuse V1–V7 opened TEST as confirmation. Update this file and `research_result/research_journal.md` after findings before proposing a new method.
