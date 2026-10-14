# Completion-credit PPO v1: prospective protocol

Locked before any new learning outcome. Motivation is preserved negative v3 full
development and frozen N4 timing diagnostics; these do not establish causation.
No teacher, demonstrations, old weights, distillation or PPO warm start. All fits
initialize randomly. Existing v3 confirmation remains prohibited and unopened.

## Hypothesis and algorithm

Hypothesis: assigning credit to a technician choice conditional on already
reserved service occupation/restoration improves coupled maintenance allocation
over a shared state-only advantage with exactly the same policy and critic.
Also test the complete decomposed-learning recipe against joint-action PPO.
That contrast changes both credit and surrogate and cannot isolate clipping.
Main claimed
contribution is a domain allocation/learning mechanism, not generic autoregression,
prefix critics, PPO or a new theorem. See technician_completion_credit_prior_art_note.md,
particularly POAD, MAT and COMA. No monotonicity or guaranteed variance claim.

Canonical trace: K machine-or-WAIT choices, one for each technician in fixed index
order. A selected machine is reserved; WAIT leaves later technicians free to
choose. Busy technician/empty legal support forces probability-one WAIT. Every
physical matching has exactly one trace (21 at N4/K2); no order duplicates/global
STOP. Execute all starts simultaneously, no physical time between choices.
Fixed technician order is a disclosed assumption; machine permutation equivariance
is tested, not invariance to arbitrary changes in technician decoding order.

Raw inputs remain M6/T3/E4/G9, exactly the centralized N4 state71. Two separate
base pair MLP encoders consume raw22, actor and value. Hidden27. Maintain a
deterministic reservation calendar: per-machine service active, remaining time/12,
known completion age/8 = floor(age*(1-restoration))/8, plus per-edge assignment
flag. Initialize from current services; update only from selected current-state
edges. These are transformations of existing input and own prefix, not future
failures, simulator rollouts or another policy's labels. Calendar does not claim
to predict stochastic future health. Actor and value each have a calendar4 MLP,
added to their corresponding base edge embedding. Binary assignment flags permit
evaluating flag0/1 machine embeddings once then selecting per edge, exactly the
same function without NK repeated calendar MLP calls.

At each stage pool augmented edge embeddings over machines, technicians and
the graph. Bid MLP takes four H vectors for each candidate edge of the current
technician; WAIT takes technician/global embeddings plus stage/K. Critic applies
pointwise tanh to the sum of base and calendar value embeddings, binding current
job status to its reservation before any pooling. It then pools by technician,
applies a shared nonlinear resource-context MLP to each technician embedding
concatenated with the value graph embedding and two decoder-role scalars
(technician index/K, index>=current stage), then averages these contexts.
Roles are internal fixed-order information, not added external observations.
Value MLP takes that resource-context pool, global9
and stage/K. This preserves contention distributions instead of collapsing all
pair embeddings before the first nonlinear interaction. Separate actor/value
encoders; all MLPs have two hidden tanh layers. Expected18444 parameters for every graph
arm and size. Values output in units scaled N/4 and value losses normalize N/4.
Shared current graph/reservation broker assumed, not local-only communication.
Inference select evaluates only actor encoders/heads; prefix critic runs during
training/replay, and does not enter actor inference latency.
Cached base encodings, O(NK) candidate decisions plus O(NK²) pooled graph work;
no matching catalogue in learned graph policies. Sizes trained separately.

Four matched graph arms, SAME inputs, tensors, architecture, physical tapes,
action/minibatch RNG roles, optimizer-step counts and critic supervision:

- completion_credit_ppo: prefix-conditioned actor, exact conditional choice
  likelihoods, per-stage ratio clipping, A_j=MCreturn-V_old(state,prefix_j).
- state_credit_ppo: SAME actor and ALL prefix critic targets/training; change only
  A_j=MCreturn-V_old(state,empty), broadcast to stages. This isolates actor credit,
  without extra information/capacity/critic updates in the candidate.
- joint_credit_ppo: SAME actor and ALL prefix critic targets/training; joint ratio
  exp(sum_j(newlog_j-oldlog_j)), common empty-prefix advantage. Original macro-action
  PPO surrogate control; entropy is the same mean conditional entropy.
- independent_ppo: same allocated architecture; actor uses original calendar and
  original masks at each stage, no reservations/previous assignments. Technician
  proposals may collide on a machine; quality arbitration drops losers without
  retry. Critic still trains all prefix intent values against the same MC return;
  actor uses joint ratio/common empty-prefix advantage. Raw sensing is matched;
  reservation messages are the removed mechanism, not external privileged data.

Each stage value is computed BEFORE that stage's choice. MC return includes the
48-step episode and common tail; internal stage discount1, physical gamma1.
No bootstrapped new target/extra environment sampling. Normalize actor advantages
once per rollout, common over batch/stages; per-stage objective averages over K,
value MSE averages over K so increasing K does not increase optimizer steps.
Joint control normalizes its batch common advantage normally. Its contrast with
the candidate is a complete-learning-recipe comparison, not a clipping-only
ablation. No claimed global
trust-region theorem for component clipping. Forced WAIT contributes log/entropy0;
critic targets remain on every stage in every arm. Exact behavior replay required.

## Physics, comparisons, budgets, panels and stopping

Unchanged scalar balanced maintenance N4/8/16/32,K=N/2,H48+deadline tail12,
reward=-cost/20. Fresh original PPO/localMAPPO/full-informationMAPPO at N4;
fullMAPPO receives raw71+machine identity4. PPO18966/local15364/full18052 parameters,
graph18444: disclose capacity differences. Deadline reference uses min-cost-flow
large and catalogue only small. All graph arms exclude allmatching enumeration.

Full fits491520 joint decisions, vector16,rollout3072,epochs4,minibatch256,
lr3e-4,clip.2,entropy.01,value.5,actor/value gradient clip.5. Graph hidden27,
baseline64. CP0/122880/245760/368640/491520; select earliest minimum mean DEV cost,
whole budgets spent, every replica retained. Process8,torchthreads1,tqdm.
Hard24h per run; no extension or metric early stop. Fresh UTC artifacts.

Smoke11fits,N4/N32,1train,384steps,vector2/rollout192/epochs2/mb48,dev2 and
engineeringtest2. Pilot22fits,N4/N32,2train,61440steps,vector8/rollout768/epochs4/
mb128,CP0/30720/61440,dev16,NO scientific tests. FullDEV57fits,all sizes,
3fresh train/full budgets/dev32,NO tests. Confirmation95fits,all sizes,5fresh
train/full budgets/dev64/test256. Registry fixes independent raw/derived panels
70000000..71300000 groups; exclude all historical planned seeds, including unused
v3 confirmation. Pair graph initial/action/minibatch roles and all method physical
tapes by size. Sizes have independent physical namespaces.

After pilot: proceed to fullDEV only if every pilot cost/health point screen passes
and at least one trained candidate checkpoint is selected at both sizes. Report
both mechanism point screens even if negative; their N32-only pilot proxy cannot
measure the registered N8/N16/N32 pool. This exception is prospective and permits
exploratory fullDEV only. FullDEV must pass EVERY point screen before confirmation.
Confirmation additionally requires completed audited test-closed fullDEV, identical
source/protocol/registry/core hyperparameters and sealed selections before tests.
No test reuse, best-seed exclusion, teacher fallback or relaxed health/cost gate.

## Statistical family and physical quality gates

Scientific round2 retains alpha.05/6 (earlier teacher-assisted round1 spent.025;
no teacher-free confirmation has opened). Bootstrap200000;35 Bonferroni endpoints,
two-sided confidence1-(.05/6)/35. Common train-replica resampling weights across
sizes, independent episode resampling per size, methods paired within size.
All per-replica and leave-one-out values disclosed;5rep approximate coverage.

N4 six gains vs PPO/localMAPPO/fullMAPPO/state_credit/independent/joint_credit
plus3 health differences =9. Each large4 gains vs state_credit/independent/
joint_credit/deadline plus3health =21. Three large/N4 cost/machine ratios and
two pooled large gains (credit vs state_credit, full recipe vs joint_credit) =35.
Pool arithmetic mean of N8/N16/N32 gains within EACH bootstrap draw.

Required N4 lower gains>=-1%, except localMAPPO lower>0. Large state_credit and
joint_credit lower>=-1%, independent lower>0, deadline lower>=-2%, cost/machine
ratio upper<=1.10. BOTH main pooled credit and full-recipe lower>=0.5%. Optional
PPO superiority lower>=1%, positive N4 credit lower>0. DEV screens same POINT
thresholds. The internal artifact keys `pooled_clipping_gain` and
`clipping_mechanism` denote the full-recipe contrast; these names do not authorize
an isolated clipping claim. This interpretation is clarified before any pilot
outcome, with no numerical threshold or learning-source change. Do not remove
failed learning contrasts to redefine success.

All sizes health vs PPOsmall/deadlinelarge: availability difference lower>=-.01,
pending/machine difference upper<=.025, failure/machine difference upper<=.25.
Zero invalid; neural select p95<=50ms in isolated evaluation after training pool
shutdown, excluding observation/physical stepping/diagnostic probes. No network
latency or zero-shot scaling claim. Lower cost never offsets failed health bounds.

## Validation and artifact contract

Prospective engineering amendment, before pilot/development/scientific tests:
an initial global-edge-mean critic (hidden30,18723 parameters) exactly aliased
N4 ages(3,3,0,0) and(0,3,3,0), both at empty prefix and after reserving machine1
with technician0. Yet their maximum one-tick urgent matchings have sizes1 and2.
The amended critic retains nonlinear per-technician contexts, fuses job status
with reservation identity, and retains remaining decoder roles. Hidden27 keeps
total18444 parameters close to PPO18966. All graph controls receive the same amendment.
No outcome-based threshold, seed, budget or learning-mode change. Prior1104-test
PASS and two successful smoke runs apply to the pre-amendment architecture only;
fresh tests/smoke are required before authorizing pilot.

Tests: all21 unique small trace matchings and exact probability mass; canonical
busy WAIT and WAIT-then-assignment; illegal/duplicate action replay rejection;
stage values independent of current/future actions; distinguish the documented
resource-contention alias, reservation identity and known remaining decoder roles;
calendar completion formula
for existing and new services; cache flag0/1 identity to direct MLP; deterministic
sample/replay and gradients; machine permutation; no future inputs/no large
catalogue; matched initial tensors/counts; exact surrogate/advantage controls,
critic supervision unchanged, forced choices; old PPO path unchanged; checkpoint
identity, serial/spawn scratch determinism;60-tick independent replay; complete
35-endpoint/null/missing-grid statistics; prerequisite corruption rejection.

Run tests and fresh Linux smoke before code commit/push/pilot. Source/PLAN/registry/
physics snapshots, hashes, zero/learned CPs, parameter/init counts, full training/
DEV/test journals, sealed choices, independent replay, cost/health/bootstrap/latency
reports. Log actor/critic gradients and actual parameter steps, per-stage advantage
spread and value error descriptively. Analysis reports in timestamped directories
with copied scripts and hashes; visually verify plots. Preserve all negatives.
Direct lab runs authorized by AGENTS override; no SSH/automatic transfer. Show
exact compound run commands after push and manual rsync -avhP --partial at delivery.
