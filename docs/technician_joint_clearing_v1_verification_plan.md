# Independent joint-clearing DEV verification procedure

Prepare while the registered FULL is running, before reading its final outcomes.
The learning tree remains frozen690be2d8241b9e5a2625c537e0c684f95674b4d0; this
branch edits verification/report code only. No new learning, physical rollout,
test observation, checkpoint selection, teacher labels or design search.

Source must be COMPLETED/auditPASS, testclosed, clean original Git identity and
all sealed source/protocol/config/registry/output SHA/size bindings unchanged.
Check copied source against committed Git blobs; tests and push verifier before
running it on the completed source. New timestamped report directory, no overwrite.

Verify24fits/11,796,480decisions/245,760trainEP/184,320Adamsteps,160update rows per
fit,5CPx32DEV grid perfit, every replica and original checkpoint selection. Check
raw initial tensors against independent Torch module construction using the
registered SHA-derived initialization seeds, equal core tensors and PPO/norm
raw initial tensors, architecture parameter counts and own final checkpoints.
Audit sampled-policy replay errors already recorded by the learner; no claim of
independently reconstructing all historical minibatch gradients/actions which
were not retained. Regenerate already-executed initial states/event-tape hashes
to check training pairing; metadata reconstruction is not another rollout.

For all1088selected/frozen/deadline episodes, independently implement primitive
maintenance dynamics and overdue-cost accounting from recorded state/actions/
events. Regenerate initial states and shocks using the specified physical seed,
check every state chain, transition cost and all48cutoff health/60tick components.
Recompute legal urgent pair support/current observation directly from state.
No environment, policy, simulator transition or experiment module is imported
for these checks. There are65,280physical transition records and52,224decisions.

For core utility decoders, use the RECORDED FP32 utility array, independently
round to integerq and verify support, arbitration and actual matching. AtK2,
prove max-weight partial matching value from the two per-column best bids: when
they contend, one keeps its best and the other takes its best remaining option
or WAIT. This top-two bound needs no matching catalogue or solver. Serial and
independent decoders reconstructed directly. Check the same optimal-value bound
for deadline scores; tie handling remains checked exactly by the primary audit.
PPO's existing21-token interface may be reconstructed for BASELINE verification
only, never supplied to the candidate. MAPPO arbitration checked directly.

Independent NumPy/float64 MLP and pooling reconstruct learned utility means on
all selected/frozen core decision states; compare recorded values with tolerance
5e-6+1e-6*abs(value). FP32 representation ties may differ at6decimalquantization,
so optimality checks use the exact recorded utilities, not recast NumPy means.
Also check baseline selected logits agree with their recorded greedy choice up
to the same floating tolerance, without replacing their actions. Do not infer
causal mechanism quality from utility-optimality invariants alone.

Recompute all13prelocked point endpoints/thresholds and every per-replica summary
from complete paired DEV arrays. Check source p95 latency/432samples procedure;
the primary stores aggregate timing rather than raw samples. Therefore perform
a SEPARATE read-only432sample timing of exact frozen actor+decoder on the same
stored48states x3rep x3repeats after training workers terminate,8warmup states per
replica. Save every timing sample; report primary and new measurement separately,
never rewrite or rescue the original source verdict. Only timing imports the
verified primary actor/decoder implementation. Torch/BLAS1, no other train jobs.

Output: verifier copy/hashes, verification.json, replay audit, independent gates,
per-replica/mean/frozen contrasts, raw timing samples and report. Report states
DEV not CI, original negative verdicts preserved; no confirmation/larger-N from
partial success. Any mismatch marks verificationFAILED and leaves the original
source intact. Verification hardstop15min with tqdm. Manual rsync -avhP --partial
for result copies; no SSH or transfer executed by agent.
