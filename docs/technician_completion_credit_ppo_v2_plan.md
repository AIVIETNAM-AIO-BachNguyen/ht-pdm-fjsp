# Completion-credit PPO v2: prospective protocol

Locked before new learning outcomes. V1 pilot is negative and cannot authorize
full development. All v1 runs, failed screens, replicas and unopened confirmation
remain preserved. No teacher, demonstrations, old learned weights, distillation
or baseline warm start. Every fit starts from random tensors.

## Research hypothesis and contribution boundary

Once value learning uses appropriate units, explicit completion/reservation
conditioning and prefix-conditioned credit may improve resource allocation over
matched graph policies without those mechanisms. The contribution must be that
domain allocation/learning mechanism and scalable implementation, not value
normalization. V1 frozen audit measured saturated critic heads, small prefix
differences and frozen output envelopes below observed greedy DEV returns.
This motivates a repair but does not prove causation. A new scratch run is needed.

Normalization is established, e.g. [van Hasselt et al.](https://arxiv.org/abs/1602.07714).
Here use fixed physical units, not adaptive PopArt; no novelty claim for this.
Generic autoregression, prefix values and component PPO are also prior art:
[MAT](https://arxiv.org/abs/2205.14953), [POAD](https://arxiv.org/abs/2405.15821),
[COMA](https://arxiv.org/abs/1705.08926). No BAD reproduction, new PPO theorem,
monotonicity/variance guarantee or broad novelty claim. A positive result must
survive calendar, actor-credit, joint-recipe and coordination controls.

## Architecture and units

Canonical fixed-index technician order: K machine-or-WAIT choices. Reserve each
chosen machine, WAIT permits later technicians to act, busy/no-support technician
has probability-one WAIT. All starts execute simultaneously, no physical time
between choices. Each physical matching has one trace (21 at N4/K2). No learned
policy matching catalogue. Machine permutation equivariance required; arbitrary
technician-order invariance not claimed. Shared current graph/broker assumed;
not local-only communication, network deployment or zero-shot scale transfer.

Raw inputs M6/T3/E4/G9, raw pair22, N4 state71. Separate actor/value base pair
MLPs; hidden27, all MLPs two hidden tanh layers and linear output. Calendar per
machine: active service, known remaining/12, known completion age/8 (floor of
age times one-minus-restoration); per-edge assignment flag. Current service and
own prefix only, no future events, stochastic forecasts, branching simulator or
teacher labels. Actor/value calendar4 MLPs separate; exact binary-flag0/1 cache.
Actor pair representation adds calendar to base, pools by machine/technician/
graph. Bid MLP takes four H contexts; WAIT takes two H and stage/K. Critic fuses
base+calendar by tanh before pooling, pools by technician, applies nonlinear
resource-context MLP with graph embedding and known role (index/K, index>=stage),
then means contexts; value head takes H+global9+stage/K. Values precede current
choice. All five graph arms allocate identical 18,444 parameters, same targets,
initial tensors, critic updates, optimizer/minibatch/action RNG roles and budgets.
Inference skips critic. Cached encodings, O(NK) choices plus O(NK²) pooled work;
separately trained sizes N4/8/16/32, K=N/2. No new learned weights from v1.

Physical value unit U(N)=unavailable_cost * N * 48 / 20 = 28.8*N, using known
cost, horizon and existing reward units. Graph value output is U(N)*head, loss
MSE(V/U, return/U). This REPLACES v1's N/4 multiplier/normalizer. Strong PPO,
local MAPPO and full MAPPO controls receive the identical U(4) treatment. Scalar
environment rewards remain -cost/20, gamma1, complete MC targets with common tail.
No reward clipping, changed objective/health/cost, tuned unit multiplier, extra
optimizer updates or environment sampling. Initial random value outputs are
rescaled consistently; actor tensors/probabilities are unaffected by unit choice.
Also retain literal original PPO with scale1/loss-normalizer1 at N4.

Five graph arms:

- completion_credit_ppo: prefix calendar actor/value, per-stage clipped ratios,
  A_j=MCreturn-V_old(state,prefix_j).
- state_credit_ppo: identical actor, all prefix critic targets/training/capacity;
  only actor baseline uses V_old(state,empty), broadcast to stages.
- joint_credit_ppo: same actor and all prefix critic targets; joint probability
  ratio and common empty-prefix actor advantage. Full-recipe contrast, not an
  isolated clipping ablation.
- independent_ppo: actor uses original calendar/masks, no previous reservations;
  proposals may collide, quality arbitration drops losers without retry. Critic
  still sees prefix intents, same targets. Joint ratio/common advantage.
- no_calendar_ppo: same canonical reservations/masks, prefix advantage/component
  ratios; actor and critic calendar embeddings use original physical calendar
  throughout internal stages. All calendar encoders remain allocated and trained.
  Removes own-prefix completion/occupation features, retains current raw inputs,
  fixed stage/roles and feasibility coordination. No external information change.

Normalize actor advantage once per rollout; shared mean/SD across batch/stages
for component modes, batch common advantage for joint mode. Average objective,
entropy and value supervision over K, forced WAIT log/entropy0. Shared MC return
on every prefix in every graph arm. Separate actor/value gradient clips .5.
At N4 baselines: original `ppo`, `normalized_ppo` (same catalogue/actor/critic
architecture and PPO surrogate, U(4) value units), normalized local `mappo`,
normalized full `full_mappo` (raw71+identity4). PPO18,966/local15,364/full18,052
parameters; disclose differences. Original/normalized PPO start with paired raw
tensors. Comparison method tags must not silently change policy action semantics.

## Fixed budgets, panels and stopping

Full fits491520 joint decisions, vector16, rollout3072, epochs4, mb256, lr3e-4,
clip.2, entropy.01, value.5, gradientclip.5. CP0/122880/245760/368640/491520.
Per-replica earliest minimum DEV mean cost selection; all budgets spent, all
replicas retained. Processes8, torchthreads1, tqdm. Fresh UTC artifact directories.
Global hard stop48h, prospectively increased from v1's24h for the extra controls
and measured CPU cost; no prior run extended, no metric-based early stopping or
extension. Treat a hard-stop/incomplete grid as technical incompleteness, not pass.

Smoke: N4/N32, 14fits, one replica384decisions, vector2,rollout192,epochs2,mb48,
CP0/192/384, DEV2/engineering-test2. Pilot: N4/N32, 28fits, two replicas61440
decisions, vector8,rollout768,epochs4,mb128,CP0/30720/61440,DEV16, no tests.
Full DEV: allsizes,72fits,3replicas/fullbudget/DEV32,no tests.
Confirmation: allsizes,120fits,5replicas/fullbudget/DEV64/test256.
Every size includes deadline reference on the same physical panels; only original/
normalized PPO and N4 deadline may use the small catalogue. Large deadline uses
polynomial min-cost flow. No catalogue in any graph arm at any size.

Registry groups80000000..81300000 fix train/DEV/test/analysis panels separately
for all profiles, 387 actual raw seeds mutually and historically disjoint.
Audit every derived role and all historical registries, including unopened v1
confirmation. Pair all methods' physical streams per size; graph initialization,
actions and minibatches share roles; original/normalized PPO share raw tensors
and action/minibatch roles. Sizes have independent physical namespaces.

Pilot -> fullDEV requires COMPLETE/auditPASS/testclosed, exact settings/bindings,
all nonpooled cost/health point screens PASS, at least one selected trained
candidate replica at BOTH N4/N32 (all replicas still included). All three pooled
pilot mechanism proxies must be reported; their N32-only failure may permit
exploratory fullDEV, not confirmation. FullDEV must pass EVERY point screen.
DEV -> confirmation requires full-budget COMPLETE/auditPASS/testclosed passing
DEV, identical source/PLAN/registry/core hyperparameters and sealed choices before
test opening. No teacher rescue, seed dropping, gate relaxation or test reuse.

## Scientific endpoints and unchanged system bounds

Round2 alpha=.05/6 remains unused by teacher-free confirmation; earlier teacher
round1 spent .025. No resetting alpha. Crossed bootstrap200000,41 Bonferroni
endpoints, two-sided confidence1-(.05/6)/41. Common train-replica resampling across
sizes, independent episode resampling per size, paired methods within size.
Pool arithmetic mean of large gains within each draw. Five replicas give
approximate coverage; disclose per-replica and leave-one-out means.

N4 eight gains vs originalPPO/normalizedPPO/localMAPPO/fullMAPPO/state_credit/
independent/joint_credit/no_calendar plus3health=11. Each large five gains vs
state_credit/independent/joint_credit/no_calendar/deadline plus3health=8*3=24.
Three large/N4 cost-per-machine ratios and pooled large credit, full-recipe and
calendar gains =6. Total41. Keep every prior v1 contrast and add calendar and
strong normalized PPO comparisons; do not drop failed contrasts.

Required N4 lower gains>=-1% except localMAPPO lower>0. Large state/joint/calendar
lower>=-1%, independent lower>0, deadline lower>=-2%, cost/machine ratio upper<=1.10.
ALL THREE primary pooled credit/full-recipe/calendar lower>=0.5%. Optional PPO
superiority requires lower>=1% vs BOTH original and normalized PPO, with new
test seeds; optional N4 credit lower>0. Internal legacy `pooled_clipping_gain`
means full-recipe vs joint-credit, never isolated clipping.

Allsize health reference original PPO at N4, deadline large: availability lower
diff>=-.01; pending/machine upper diff<=+.025; failures/machine upper diff<=+.25.
Zero invalid assignments. Candidate actor-only select p95<=50ms, measured after
training workers shut down, excludes observation/physical stepping/diagnostics.
No network latency claim. Lower cost cannot compensate for failed system bounds.
DEV screens use same numerical point limits. Pilot/smoke never establish success.

## Engineering gate and artifacts

All existing matching support/replay/calendar/causality/permutation/physics tests
must remain valid. Add exact unit conversion/loss/old-PPO preservation tests;
same graph initialization/capacity and critic targets for calendar ablation;
actor and critic use original calendar only in no-calendar arm; fixed roles and
unique feasible traces unchanged; role pairing/strict checkpoint algorithm tags;
41 complete/null/missing-grid endpoints and bootstrap pool calculations;
prerequisite/config/source corruption rejection. Test return ranges and behavior
replay in physical units, gradients/finite updates, independent 60-tick replay.

Fresh Linux tests/smoke before verified code commit/push, then show/run exact
pilot command locally under user override. No SSH/automatic transfers. Source,
PLAN, registry, historical registries, physics and resolved settings sealed;
checkpoints/digests, full training/DEV/test journals, all selections/replicas,
parameter counts/from-scratch provenance, independent replay, cost/health/CIs/
latencies, actor/critic gradients and actual update norms. Log prefix spread,
value prediction/return scales and saturation descriptively, not as variance
proof. Timestamped copied-script/hash analysis reports and visually checked
plots. Provide manual rsync -avhP --partial when results exist. Preserve negatives.

Engineering clarification before pilot: behavior log replay keeps absolute
tolerance2e-5. Critic replay uses absolute tolerance U(N)*2e-5 in physical units
(2e-5 for literal original PPO), plus relative1e-6. This is the same dimensionless
floating-point criterion under the fixed unit conversion, not a performance-gate
relaxation. Training return/prediction statistics are logged through an optional
read-only diagnostic callback; the old PPO update path remains unchanged.
Checkpoint metadata additionally records a deterministic first-DEV-observation
critic probe: output envelope, prediction in physical/normalized units and last
hidden saturation. This is descriptive, outside inference timing and training.
