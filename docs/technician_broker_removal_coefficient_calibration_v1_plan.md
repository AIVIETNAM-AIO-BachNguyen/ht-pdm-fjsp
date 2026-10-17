# Fresh fit/heldout scalar control-variate calibration diagnostic v1

Register BEFORE implementation/new data. Preserve FULL costFAIL17/19 and frozen
gradientFAIL3/6. Post-hoc sample-optimal scalar ceilings over all5checkpoints were
5.7139%,31.5966%,6.2562% by source replica; evaluation-sample fitting is optimistic
and NOT held-out evidence. This motivates one bounded fresh calibration study,
not a coefficient sweep or new policy-learning claim. Classical control-variate
coefficients are prior art (Greensmith JMLR2004; optimal MARL baseline NeurIPS2021).
Contribution must still be broker-specific coordination and independently tested
scratch policy/cost benefit, not scalar fitting itself. No teacher/actor/critic
fine-tuning or optimizer, no scientific CF/test/N>4 opened.

Hypothesis: a fixed coefficient1 miscalibrates the conditional-removal control
variate relative to the actual shared actor score. A coefficient estimated once
from independent fit episodes generalizes to reduce actor-gradient variance on
fresh heldout episodes. Reject if this is merely fit-sample variance reduction.

ALL3candidate replicas and ALL5fixed checkpoints0/122880/245760/368640/491520.
Each node:128NEW fit episodes, then256DISJOINT new evaluation episodes. Same
physical panels across nodes, distinct fit/eval action streams per node. N4/K2,
H48+common12tail, unchanged physics/rewards, F64promotion of stored checkpoint
VALUES, independent-proposal behavior, no greedy filtering. Total5760episodes,
276480policy decisions,345600physical transitions,17280full gradient vectors,
0optimizer updates. Exactly3parallelCPUworkers,BLAS/torch1,60minutehardstop,
tqdm,freshUTC,no resume/overwrite/extra samples. Tests then commit/push before
FULL on this lab; no separate smoke.

Reuse the frozen three-estimator diagnostic with explicit injectable physical
namespace/environment only; defaults retain prior behavior and prior artifacts.
Every new state/return/baseline/draw/gradient independently replayed as before.
New physical hash role THISVERSION:N4:physical, new actions roles
THISVERSION:fit:actions and THISVERSION:evaluation:actions. Registry lists ALL
402raw seeds;417actual derived streams checked for collisions/history raw.
Source train seeds are intentional frozen references, not newly claimed fresh.

For each node, fit once on ONLY128fit gradient vectors: x=gV,h=gV-gB,
lambda=traceCov(x,h)/traceCov(h), or0 if denominator<=1e-20. No regularization,
grid, clipping, refitting, evaluation-selected coefficient or adding predictors.
Finite unconstrained coefficient may be negative or >1; that is a diagnostic
control-variate coefficient, not a deployed actor action. Freeze/hash coefficient
BEFORE collecting evaluation data. Evaluate gLambda=gV-lambda*(gV-gB) on all256
heldout episode vectors. Check exact covariance quadratic/direct-vector equality.
Step0 must have h≈0,lambda0,primarygain0. Report raw numerical invariance.

Primary per-node gain=1-trCov_eval(gLambda)/trCov_eval(gV). For EACHreplica,
unweighted mean ALL5gains>=5%, >=3positive checkpoints, and mean mixed gain
strictly greater than mean fixed1gain on SAMEevaluation trajectories. All9checks
required. Also finite nondegenerate covariance,all physics/gradient/binding/seeds
auditsPASS,exact budget,0updates. Point diagnostic only,no scientific CI/cost or
PPO finite-update guarantee. No choosing a favorable checkpoint/replica.

If any gateFAIL, reject THIS scalar-calibration route; do not change clip bounds,
sample counts, coefficient regularization or seed panel to rescue it. If ALLPASS,
only a separately registered scratch learning study with matched coefficient
estimation/critic/info/compute controls could test whether this helps PPO cost.
Do not promote a frozen checkpoint or override original cost failure regardless.

Artifacts: fresh manifests/protocol/registry/settings/code/source bindings;
nodefit and nodeevaluation full observations/actions/returns/baselines/physical
and policy journals/three full gradient matrices; fit coefficient JSON+SHA before
evaluation; raw trace covariance components/direct mixed variance; all15node
records/three means/9gates. Independent read-only verifier before interpreting
diagnostic outcome. No artifact transfer/SSH.
