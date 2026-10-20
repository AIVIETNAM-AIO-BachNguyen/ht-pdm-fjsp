# V13 preventive-risk exposure on already opened DEV — independently audited

The V13 actor/rollout cost-component retrospective found that failure cost explained nearly all of V13's positive cost gap in interchangeable scale1 and reciprocal scale2, despite similar total H48 service starts. The first prelocked event-class plan, `v13_failure_event_protocol_vi.md`, was superseded transparently after its smoke replay revealed a structural identity: with the threshold requester, all observed H48 failures were existing PM requests upgraded to corrective, so splitting upgrades versus newly created CM requests was not informative. The 16-episode smoke artifact remains intact. `v13_failure_exposure_protocol_v2_vi.md` then locked an at-risk PM machine-tick and instantaneous feasibility analysis before the full replay; the V2 smoke and independent smoke audit passed.

## Provenance and locked panel

The full V2 replay used only the already opened V13 DEV: five selected `competition` training checkpoints 72500000–04 × 16 physical seeds 72510000–15 × four cases (320 actor episodes), plus rollout once per case/physical seed (64 episodes). The source artifact `framework_v13_dev_20261020T141536Z` and its original independent audit PASS were checked. New diagnostic `v13_failure_exposure_dev_20261020T191544Z` completed 384 episodes and 58,259 H48 event/exposure rows, with exact objective/failure/service/queue metric reconciliation to immutable `episodes.csv`. Independent `v13_failure_exposure_audit_20261020T191820Z/independent_audit.json` is **PASS** for source/output/checkpoint hashes, every deterministic actor/planner action and event, H64 episode metrics, all case summaries and 20,000-draw paired hierarchical bootstrap.

Two initial independent-audit directories are preserved as failed. The first exposed a verifier mismatch from comparing raw logits instead of reproducing V13's masked-softmax action tie behavior; the second preserved the precise mismatch. After matching the registered policy action rule, a later audit replayed every action/event but stopped on exact floating-point equality for an exposure fraction; that failed directory is also preserved. The final audit uses numerical tolerance for this fraction. No runner, source physical model, protocol, seed, checkpoint or outcome was changed after full replay began.

## Locked result

An exposed tick means an unstarted PM request is at failure risk at the current tick and the chosen matching does not start that machine. “Feasible” means some current legal matching could start it individually; this does not prove a different action would improve full-episode cost.

| Case | Exposed ticks Δ | Individually feasible Δ | Capacity constrained Δ | H48 failures Δ | H48 PM starts Δ | H48 total starts Δ | H64 cost Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Interchangeable scale1 | +5.95 | +1.64 | +4.31 | +2.43 | −2.36 | 0.00 | +56.66 |
| Reciprocal scale2 | +6.14 | +2.05 | +4.09 | +2.20 | −2.01 | +0.13 | +56.30 |
| Reciprocal scale3 | −6.62 | −4.16 | −2.46 | −2.26 | +5.26 | +3.80 | −96.23 |
| Specialist/generalist scale2 | +3.90 | +3.54 | +0.36 | +2.12 | −1.60 | +0.50 | +32.29 |

Candidate-minus-rollout exposure is positive in all five training seeds in both registered target cases. Interchangeable seed deltas are +5.69,+5.75,+6.31,+6.31,+5.69; mean +5.95, paired hierarchical CI95 **[+4.90,+7.03]**. Reciprocal2 seed deltas are +6.62,+5.81,+6.62,+5.81,+5.81; mean +6.14, CI95 **[+4.80,+7.41]**. Both meet the prelocked descriptive +3/4-of-5/CI-lower>0 gate. In these cases 4.31/5.95 and 4.09/6.14 of the mean excess exposure is capacity constrained at that instant; the remainder is individually feasible. Specialist/generalist shows the reverse pattern, 3.54/3.90 individually feasible, suggesting case-specific mechanisms. In reciprocal3 the actor reduces both exposure and failures and starts more PM service.

All 384 replayed episodes had zero new-corrective/residual H48 failures: H48 failures equaled PM-upgrade events under this fixed requester and physics. In the two target cases, the actor traded roughly 2.0–2.4 H48 PM starts for CM starts while leaving total starts almost unchanged, and its failure cost increased by +55.50/+54.38 of +56.66/+56.30 net cost. This supports a timing/resource-commitment question, not a simple throughput shortage. The trajectories are observational and actor checkpoints were selected on this opened DEV panel. Instantaneous infeasibility may reflect earlier actor commitments, request competition and service durations; it does **not** identify which earlier assignment caused the later failure or prove a reservation policy would work.

## Decision

The event-class split was discarded as non-discriminating; the V2 exposure gate passes as a **mechanism diagnostic**, not a method result. V14's capacity-matched identity ablation is still running and must be independently audited before promotion or rejection. If a further method is needed, test an earlier technician-commitment/at-risk-exposure intervention on shared states with proper causal continuation before writing an RL proposal. Do not infer from this retrospective that any specific architecture, heuristic restriction or planner consultation will meet the unchanged ≥5%/≤1% cost criteria. No new TEST or independent method replication was opened.
