# V8 DEV representability probe: rare exact collisions, large learning gap

The question and exact collision definition were prelocked in `v8_representability_probe_plan_vi.md`. This uses only the independently audited V8 headroom **DEV** states and forecast values. It is an exploratory diagnostic, not a new RL method, TEST confirmation or teacher-based training claim.

## Provenance and correction

Verified source was pushed at `0faf38b` before an initial all-state run. Review caught an implementation omission: the collision counter allowed identical full observations with differing finite-scenario forecast values, although the protocol required differing observed context. That artifact, `v8_representability_dev_20261020T055400Z`, is retained but **superseded for collision conclusions**. The correction, regression test and protocol note were pushed at `f3b39ca` after a fresh smoke. Corrected all-state run: `research_result/v8_representability_dev_20261020T060300Z/` (manifest actual UTC start is authoritative; the manually chosen directory suffix is an identifier, not a reliable clock reading). Independent audit `research_result/v8_representability_audit_20261020T060600Z/independent_audit.json` **PASS**: source/output hashes, independently reconstructed observations and signed-edge keys, distinct full-observation examples and exact actor-margin invariance, physical-seed split and finite fit-metric accounting. The supervised optimizer was not independently retrained by this audit.

## Exact local-edge representational conflicts

All 720 states yielded 4,027 non-tied legal action-pair records and 3,231 exact signed-edge-input keys. A conflict requires two states with different 56-D observations, identical input bytes for every edge in the candidate actions' symmetric difference, and opposite six-tick planner preference by the stated margin in **both** states. Any deterministic V7 additive local-edge actor must give the same action-score difference for such a pair, regardless of weights.

| Two-sided planner margin | Conflict keys | Opposite-preference record pairs | Unique states covered |
|---:|---:|---:|---:|
| ≥1 | 39 | 74 | 79 / 720 |
| ≥5 | 3 | 5 | **8 / 720** |
| ≥10 | 1 | 1 | **2 / 720** |

At ≥5, the eight states span interchangeable scale1 (1), reciprocal scale2 (3), specialist/generalist scale2 (4), and zero reciprocal scale3; they include both strong and weak V7 training replicas. A concrete ≥10 pair is states 110 and 591 for WAIT versus catalogue action 12, with opposite forecast margins +17.375 and −10.875 while exact involved-edge inputs agree. This proves a narrow structural impossibility for those finite-scenario planner preferences. It does **not** prove the true expected long-horizon ranking flips: the 16-scenario forecast is noisy, and the affected states are rare in this sample.

## Supervised capacity check

This check trains fresh score functions on already-opened planner root costs solely to probe representability. It is not imitation/pretraining used by the RL method. Physical seeds 69100000–19 supplied 449 train states, 20–23 supplied 101 checkpoint-selection states, and 24–31 supplied 170 final probe states. All legal action root scores were available; target was state-centered negative forecast cost. Five starts, Adam 0.001, 100 full-batch epochs, and validation-MSE checkpoint selection were fixed before fitting. Root regret below is mean cost above the six-tick forecast best action on the 170 states, averaged over five fitted starts.

| Scorer | Parameters | Probe root regret, mean [range] | Probe MSE | Probe regret ≥5, mean count |
|---|---:|---:|---:|---:|
| V7 local shared-edge | 1,597 | 1.479 [1.461, 1.506] | 81.90 | 20.4 / 170 |
| Flat full-observation | 9,173 | 1.634 [1.319, 1.903] | 57.56 | 21.6 / 170 |
| Global joint-action | 8,385 | 1.635 [1.586, 1.691] | 56.67 | 21.4 / 170 |

Risk matching regret on the same probe states was 1.456; original V7 source decisions averaged 3.465. Larger global models fit score magnitudes better (lower MSE) but did not consistently choose lower-regret actions. All five edge fits selected the 100-epoch cap, so their value-MSE capacity is not fully resolved. The models differ greatly in parameter count and fitting difficulty; this check does not isolate architecture in a controlled RL comparison. Nonetheless, a local-edge scorer trained on planner root values nearly matched risk decisions on these states, while original weak PPO choices often did not. Together with the rare exact conflicts and V8 headroom audit, this makes **RL credit/optimization stability** the better next hypothesis than a global context architecture alone.

## Decision and limits

Do not repeat V5's failed fixed residual-wait penalty, add a global encoder merely because an invariant exists, or declare the V7 actor expressive enough for the full stochastic objective. Next formulate a scratch-RL method that obtains more reliable action credit across all legal shared-resource matchings, with an ablation isolating that credit mechanism and a compute-matched control where practical. Keep flat PPO, V7 structured arms, risk and bounded rollout as comparators. Any candidate still needs ≥5 independent training seeds, fresh evaluation, independent replication and the unchanged competitive completion gates.

The supervised probe sees only 720 selected states and 16-scenario six-tick values; it may favor decisions that are poor over H48+tail16. The split is internal to opened DEV data and gives no fresh confirmatory evidence. Feasibility remains from the legal mask, not learned behavior. No generalization, permutation or N/K-scale claim is made.
