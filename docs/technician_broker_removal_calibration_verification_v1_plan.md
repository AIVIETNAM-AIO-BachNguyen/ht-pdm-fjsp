# Independent fresh calibration read-only audit

Verify frozen a2ce636 FULL after terminal completion; no new samples/learning.
Check source/model/code/output bindings, all15fixed nodes, disjoint402raw/417
derived streams, 128fit then256evaluation per node,5760episodes/276480decisions/
345600physical transitions/17280full11141dimgradient vectors and0updates.
Independently rebuild every primitive tape/state/observation/return and action
draw; all baselines from dense25proposal reference; every gradient from manual
F64 actor and densepreimage q_i with other-agent probability detached, as in
the previous independent gradient auditor. Tolerances unchanged1e-9+1e-10abs
for values,1e-9+1e-9abs for full gradients. No tolerance relaxation or window.

Recompute coefficient ONLY from centered fit vectors' covariance/dot products,
verify stored fit SHA/coefficient seal/phase, apply that exact fixed coefficient
to all heldout vectors. Recompute heldout covariance, direct mixed gradient
quadratic, all15gains/three means/9gates/original verdict. Step0 coefficient0 and
gaincanonical0 after invariance. No evaluation refit/selection. Source negative
cost and scientific CF/test/N>4 remain closed regardless audit/calibration PASS.

FreshUTC/refuseoverwrite/tqdm/oneCPU/30minutehardstop;tests/commit/push before
audit; preserve failures. Reports descriptive only, no CI/cost/goal claim.
