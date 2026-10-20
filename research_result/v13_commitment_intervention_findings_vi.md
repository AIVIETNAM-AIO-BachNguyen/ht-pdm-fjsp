# V13 one-edge technician commitment intervention — audited negative diagnostic

## Question and provenance

The audited V13 exposure probe found more at-risk preventive machine-ticks without an immediately free technician in interchangeable scale1 and reciprocal scale2. The prospectively locked `v13_commitment_intervention_protocol_vi.md` tested a **hindsight oracle**: at the first such constrained tick, identify the compatible occupied technician with most service time left, rewind to that technician's most recent service start, delete only that start, and continue the original selected actor on the same physical tape through H48 plus the unchanged H16 risk tail. It is one edit per episode, and episodes without a trigger remain in the primary mean.

Protocol was pushed at `92e48fe`; runner, independent verifier and focused tests were pushed at `71c76fc` before full replay. Focused tests 31 PASS. Eight-episode Ubuntu smoke and independent smoke audit PASS, though the untrained smoke actor had no triggers. A narrow already opened DEV smoke independently replayed a real edited action. Full opened-DEV `v13_commitment_intervention_dev_20261020T193408Z` completed 320 actor episode pairs; independent `v13_commitment_intervention_audit_20261020T193438Z/independent_audit.json` **PASS** for source/checkpoint/output hashes, every trigger and legal action, 30,720 H48 action rows, H64 metrics and 20,000-draw paired hierarchical bootstrap. Early failed smoke audit directories are preserved: the verifier initially used a wrong metric label, then targeted an audit directory by a broad glob, then lost the original trigger when it moved to the edited trajectory. These were verifier-only fixes before full replay; no source experiment result was changed.

## Prelocked primary result

Edited minus original H64 economic cost; negative favors the edit. Intervals are paired hierarchical 95% CI over five training seeds and shared 16 physical indices per seed. The directional gate required **each** target mean ≤−10, ≥4/5 seed improvements and upper CI <0.

| Case | Triggered / 80 | Cost Δ | CI95 | H48 failure Δ | H48 exposed PM-tick Δ | H48 PM start Δ |
|---|---:|---:|---:|---:|---:|---:|
| Interchangeable scale1 | 35 | −1.55 | [−7.03, +2.81] | −0.04 | −0.03 | +0.03 |
| Reciprocal scale2 | 48 | +7.38 | [−14.51, +33.25] | +0.08 | +0.46 | −0.76 |
| Reciprocal scale3 | 75 | +93.83 | [+78.08, +107.89] | +4.89 | +8.73 | −4.89 |
| Specialist/generalist scale2 | 80 | +5.88 | [−7.91, +18.91] | +0.33 | +0.70 | −0.48 |

Both registered target gates fail; their equal-weight joint cost Δ is +2.91. Interchangeable per-seed cost deltas are −2.94,−3.88,+1.00,+1.00,−2.94; reciprocal2 are −17.81,+43.19,−17.81,+14.06,+15.25. Within triggered episodes the mean deltas are −3.54/+12.29 for the target cases. The actor eventually reissued the deleted machine service in every triggered episode (35/35 and 48/48 target cases; also 75/75 and 80/80 controls). The median trigger occurred at tick1 in interchangeable and tick2 elsewhere, so this intervention generally delays a service start rather than creating a stable technician reservation. In reciprocal3 it causes a marked failure and cost increase. Individual episode changes are heterogeneous, with both large helpful and harmful tails.

## Interpretation and next decision

The earlier observational capacity exposure is real, but this specific hindsight choice of occupied worker and one-edge deletion does **not** produce a consistent economic improvement on opened DEV. It does not prove technician reservation in general is useless: the edit ignores which service must be rescheduled, the actor's follow-up reissues it, and the actor was never trained for such interventions. It rules out using this simple earlier-commitment cancellation as a justified policy restriction or novelty claim. The V14 capacity-matched scratch-RL control remains the active method test; read its selected DEV outcomes only after its full independent audit, then apply its prelocked gates. No new TEST, replication or goal completion claim follows from this diagnostic.
