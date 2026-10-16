# Raw-state reservation v1 independent verification protocol

Registered before running verification, separate from frozen learning commit 16ea7cb4bd7748c15399cad2ec9a907e582e8ca6. This procedure does not select models, change endpoints, train, open confirmation seeds or overwrite the FULL artifact.

## Hypothesis and stopping rule
The completed artifact faithfully implements the registered raw-state reservation comparison. Fail immediately on any binding, seed, budget, primitive transition, policy support/greedy choice or endpoint discrepancy. Verification has a 15 minute hard stop; failed artifacts remain intact. A successful verification can validate a negative scientific verdict.

## Scope and metrics
Verify all committed source/protocol/registry and output SHA-256 bindings; all 24 fits, fresh random tensors, common initial private-critic tensors, global clipping configuration, 11,796,480 decisions, 245,760 training episodes, 184,320 Adam updates, paired physical tapes and 3,840 DEV checkpoint episodes/earliest selection. Reconstruct all 65,280 selected/frozen/tail physical transitions using pure dictionary dynamics and seeded Python random tapes, without importing the simulator. Check 52,224 policy decisions and all aggregate cost, policy-window failure, availability, backlog and broker metrics.

Use a separate NumPy MLP implementation for recorded greedy tokens, rebuilding per-technician prefix and reserved support. Frozen controls use candidate weights with the actual ablated policy conditioning/order. Float32 forward roundoff tolerance is 5e-6 + 1e-6 times chosen logit magnitude; exact support, canonical indexing and matching rules remain mandatory. Independently check baseline policies/decoders. Reuse previously tested primitive physics and baseline arithmetic; do not use the source policy or simulator for physical/NumPy reconstruction.

Recompute all 13 original DEV point gates, zero-invalid and the primary p95 from all 432 saved isolated timing samples. Bind saved observations, selected checkpoints, timing thread and worker metadata. No separate timing measurement may rescue the original verdict. No confidence intervals or scientific success from DEV.

## Seeds and artifacts
Use only the frozen DEV train seeds 270000000..2 and physical seeds 270100000..31. Derive no new learning or confirmation states. Write a new timestamped verification directory containing manifest.json, verification.json, verifier.py and this protocol, with SHA-256 bindings. Preserve source unchanged.

## Limits
Historical training actions and minibatch gradients were not retained; this verifies recorded training journals and fit replay audits, not independent recomputation of every training update. Zero conflicts alone and noninferiority of prefix conditioning do not establish a mechanism benefit. Generic autoregression is existing prior art. Larger N/K and scientific confirmation remain closed unless all registered gates pass.
