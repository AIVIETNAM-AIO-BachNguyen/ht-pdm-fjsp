# Frozen entropy alignment diagnostic v1

Register before new outcomes. Latest bounded learning-efficiency FULL is overall
FAIL despite14point gates: one replica AUCgain−1.960%, pooled+.948%. Preserve that
verdict, no new seed additions or optimizer/entropy sweep to rescue it. This is
ONE specific causal-candidate diagnostic, NOT a new learning experiment.

## Hypothesis and fixed models/data

The existing auxiliary regularizer uses mean proposal entropy Hp=H(U|s)/K while
the candidate PPO ratio uses physical matching likelihood q(S|s),S=T_s(U).
Hypothesis: on early stochastic-policy states, this regularizer has material
gradient opposition to physical matching entropy Hq=H(S|s)/K, and replacement
would have a favorable first-order projection on the reward gradient. This
could fail. Hp>=Hq from entropy chain rule is known mathematics, NOT evidence of
such opposition, a null-space gradient, lower cost, or a new RL theorem.

Use ONLY candidate broker_marginal_ppo, all3source replicas310000000..2, all8FIXED
checkpoints0/15360/30720/61440/92160/122880/184320/245760 from committed source
8a3c1f9, artifact technician_broker_learning_efficiency_v1_development_20261016T164552654335Z,
independently verified ...verification_20261016T165503354876Z.24frozen models,
no reselection. PRIMARY early set is15360/30720/61440/92160/122880 (five per
replica); initial and two late checkpoints descriptive only. No promotion of a
descriptive result if primary fails. Actor/critic/source artifacts unchanged.

Each model generates64stochastic episodes on fresh paired physical raw seeds
320100000..63, SOURCE physical namespace technician_broker_marginal_ppo_v1:N4:physical.
Action raw seeds320200000+i by replica index i, learner.role_seed with
technician_broker_entropy_alignment_diagnostic_v1:replica:i:actions. Reset that
private generator at each checkpoint; physical tapes paired across all24models.
Do not assert equal actions across different frozen policies.67raw/67derived
audit against historical registries/current/source learning-family streams.
Total1536episodes/73728policy decisions/92160physical ticks. No CF/test states.
Frozen policy labels itself only for this diagnostic; no targets consumed by any
learner, teacher, pretraining, warm-start or future label bank.

## Measurements / exact N4 reference

Record full physical and policy traces, observations/actions/behavior logs/values,
rewards and H48health/H60cost. Replay every transition with independent primitive
physics and same generated event tape. Sum rewards=-H60cost/20; original48+12
clock and tail; zero invalid; model digest and parameter grad fields unchanged.
Behavior logs/values replay within SOURCE tolerances (log2e-5;value2e-5*115.2+
1e-6|value|). Float64 MC returns match source float32 mc_returns within5e-4.
Normalize A=G−V ONCE over all3072states/model, population SD, as source PPO.

Compute exact Hq by explicit25latent proposal preimage sums into21physical
matching masses. For this N4DIAGNOSTIC only, dense enumeration is acceptable;
not a scalable entropy implementation, not permission to add a catalogue to a
future algorithm. Per state Hp=sum_j categoricalentropy_j/2,Hq=matchingentropy/2.
Verify normalization, legal mass and Hp>=Hq within floating roundoff.

Actor-only ascent gradients at frozen behavior parameters:
    gR = grad mean[log q(S|s) * stopgrad normalized_A];
    gP = grad mean Hp; gQ = grad mean Hq.
At ratio1 gR is the clipped-PPO surrogate derivative. It excludes entropy/value;
this is NOT an Adam step or post-global-clip/critic-coupled update. Do not claim
unbiased finite-batch advantages or infer training-gradient population variance.
Save full11141parameter vectors and four16episode-block gR vectors. Global A
normalization stays fixed; mean block gR must match full gR within numerical
tolerance. No optimizer, updates or scientific CI.

Per model report mean Hp,Hq, gap, relativealiasgap=(meanHp−meanHq)/max(meanHp,1e-20),
cos(gP,gQ), original beta=.01 replacement norm ratio
R=.01||gQ−gP||/max(||gR||,1e-20), and first-order projected replacement
P=.01*gR dot(gQ−gP)/max(||gR||²,1e-20). Include degenerate gradients explicitly;
for nonzero norms cosine uses ordinary Euclidean definition, else0 and flagged.
Positive P is only a local finite-batch projection, NOT a verified cost gain.
Different entropy gradients can both favor reward/exploration; Hp−Hq is not
automatically an irrelevant or harmful parameter direction.

## Locked engineering screen / rejection

For EACH of3replicas across exactly five PRIMARY early checkpoints require:
1. mean relativealiasgap>=.10;
2. at least3/5checkpoints have cos(gP,gQ)<=0 with both entropy gradients nonzero;
3. at least3/5checkpoints have R>=.10 with ||gR||²>1e-20;
4. mean P>=.01 with every gR nondegenerate.

ALL12replica checks required to motivate further design of this entropy remedy;
even PASS does not authorize an automatic learning trial. Need new scalable
entropy estimator, matched information/capacity/experience/compute controls,
random-init protocol/seeds and original health/quality safeguards. If any fail,
reject THIS material-opposition explanation; no automatic coefficient sweep,
physical-entropy learner, different window/subset or threshold change. Do not
infer causes of a specific AUC replica from correlations. No source verdict
change, CF/test opening, generalization or goal completion.

## Execution / tests / artifacts

Directlab FULL diagnostic, no separate smoke; torch/BLAS1, tqdm model/tick progress,
15minute hardstop. Necessary unit/integration tests then commit/push before run.
Tests: exact entropy/preimage chain identity/gradients vs production physical
logprob on tiny masks/ties/forcedWAIT, same coefficient/K units, MC and block
gradient identities, toy screen boundaries/degenerate norms, source metadata
and original primitive/scalar transition/behavior replay on ENG310700001 only.
No fresh320panel states or real learning outcomes in disposable tests.

Timestampdir/refuse overwrite/preserve FAILED artifacts. Source+verification
allbindings and code/checkpoint hashes before/after. Copy protocol/registry/code/
pure helpers;24checkpoint hashes; paired trace/episode journals; observation+
return/action arrays and full gradient vectors; permodel metrics, primary
replica means/checks and decision;manifest code/runtime/counts/testclosed/updates0.
Independent verification checks provenance, all dynamics, exact preimage entropy,
gradients on fixed registered contexts and screen math; no new training.
Standard entropy identities and action-marginalization are prior art; see
[existing boundary](technician_broker_method_prior_art_boundary_vi.md).
No new objective/novelty/scalability claim from diagnostic math or PASS audits.
