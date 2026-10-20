# V11 first-update Q-gradient diagnosis — audited DEV mechanism signal

Prelocked plan `v11_gradient_diagnostic_plan_vi.md`, code push `d570c4e`; fixed full scan `v11_gradient_dev_20261020T124737Z`; independent audit `v11_gradient_audit_20261020T124821Z/independent_audit.json` **PASS**. All 7,680 Q rows, all 60 old-actor checkpoint hashes, five identical initial-weight/source-bank pairs, every derived gradient and aggregation were checked. This only reuses opened V11 training states; it is not a learned-method result, new DEV evaluation, TEST, replication or causal proof.

At iteration 1, each training seed has the **same** initial actor, source states, legal actions and four-scenario Q bank in both credit arms. The quantity below is the first-order gradient of the V11 expected-improvement objective with respect to a hypothetical independent WAIT logit, `p_WAIT*(A_WAIT−ΣpA)`, where `A=(ΣpQ−Q)/10`; positive raises WAIT under gradient ascent. It is not the gradient of the actual shared network parameters.

| Train seed | Initial mean p(WAIT) | Mean self-Q WAIT gradient | Mean reference-Q WAIT gradient | Ref−self |
|---:|---:|---:|---:|---:|
| 70500000 | 0.301 | −0.1222 | +0.1025 | +0.2247 |
| 70500001 | 0.356 | −0.0738 | +0.1724 | +0.2462 |
| 70500002 | 0.289 | −0.0900 | +0.0773 | +0.1673 |
| 70500003 | 0.366 | −0.1427 | +0.2142 | +0.3569 |
| 70500004 | 0.350 | −0.1096 | +0.3022 | +0.4118 |

The prelocked directional gate passes in all **5/5** independent initial seeds: self-Q points away from WAIT in mean, reference-Q toward it. This is stronger evidence of a credit-direction difference than the earlier unweighted Q sign census, which failed its own continuation-bias gate. It does not isolate continuation inconsistency from estimator variance or large Q magnitude; no new method has passed.

The case decomposition shows where the signal enters the shared actor: iteration-1 mean reference/self WAIT gradients are +0.0128/−0.1906 interchangeable, +0.7480/−0.0731 reciprocal scale2, +0.0704/−0.0081 reciprocal scale3, and −0.1363/−0.1588 specialist/generalist. Mean reference-Q action spread in reciprocal scale2 is 148.8 cost units versus self-Q 44.4. Reference Q selected WAIT as the minimum-cost root in 146/640 first-iteration states versus self Q 158/640; root argmin frequency alone would miss the policy-gradient direction. The four-scenario Q estimates and one-coordinate derivative remain limits.

Decision: V10 and V11 self-credit already use H64 actor continuation, so that continuation alone is not a new method. Prelock a scratch RL design that **stabilizes self-consistent H64 counterfactual credit**, for example by using paired-scenario uncertainty to attenuate unreliable action gaps, against a compute-matched plain H64 self-credit ablation with the same actor, assignment objective, seeds and state selection. Separate extra-scenario compute from the proposed credit transformation where practical. Keep risk, flat PPO and bounded planner locked; test five fresh training seeds, do not use hard service shielding or select on opened TEST. The overall goal remains **ACTIVE**.
