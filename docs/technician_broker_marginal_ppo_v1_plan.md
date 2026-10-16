# Broker-marginal PPO v1: prospective teacher-free N4/K2 comparison

Register before implementing or seeing any new learning outcomes. The raw-state
reservation FULL12/13 and its matched double-budget FULL11/13 remain failed.
Additional budget produced0% candidate selected-cost gain and negative gain
interaction; do not repeat the budget remedy or reinterpret those verdicts.

## One falsifiable learning mechanism

Fixed winner-only arbitration maps several independent technician proposals to
the same executed matching. Ordinary proposal PPO is valid, but learns/clips on
latent distinctions the physical system discards. Hypothesis: replacing ONLY its
likelihood in the joint PPO ratio with the exact broker-induced matching
likelihood removes irrelevant proposal distinctions and improves allocation
learning at matched sensing/capacity/data/update budget and the SAME broker.
This is NOT a new encoder/LR/attention sweep or a claim PPO was implemented wrong.

Motivation is the prospectively screened frozen diagnostic, independently
verified: on a fixed greedy-DEV state panel at122880, three independent actors
had39.23/63.06/75.97% of unit-weight actor-score Fisher removed by conditioning
on the physical matching; entropy and conflict thresholds passed in all three.
These are conditional score metrics, NOT return-gradient/clipped-PPO variance or
cost evidence. Source diagnostic:
`technician_broker_aliasing_diagnostic_v1_20261016T140144414038620Z`, independent
verification `technician_broker_aliasing_verification_20261016T141123400054613Z`.
No source weights, states or seeds become learning data in this study.

## Unchanged actor/critic and exact probability

Each technician independently proposes WAIT0 or machine+1 using the same shared
raw-state MLP as the previous independent control:103->64->64->5,tanh atN4/K2.
Input is full physical state71, own tech3, own edges16,12constant-zero legacy
prefix/reservation channels, j/K. The zero channels preserve the comparison's
parameterization; they carry no information. General input11N+3K+5NK+13;
outputN+1. No current teammate proposal conditions any other proposal. K samples
use explicit independent categorical draws; no physical time inside choices.
Full current physical state is already available to centralized PPO/full-info
MAPPO. Redundant own views add no sensing, future events, teacher labels or
learned predictor. Raw ordered inputs are not permutation equivariant.

Broker keeps the highest restoration/duration requester for each machine, then
lower technician ID on ties. Losers WAIT, with NO retry. Priorities/masks depend
only on current physical state, not actor parameters or other bids. Canonical
proposal vector remains stored by technician ID; the environment actually
receives those proposals and records their conflicts/rejections/physical starts.

For physical matchingS with winners(m,w_m) and idle techniciansI,

 q(S|s)=prod_(m,j)inS p_j(m|s)
        * prod_(j in I)[p_j(WAIT|s)
             +sum_(m,w)inS,w outranksj at m p_j(m|s)].

The winner's request is fixed; an idle technician may WAIT or request a machine
where it loses to the prescribed winner. This Cartesian preimage gives the
product of sums. Evaluate it with logsumexp, no proposal/matching enumeration.
It applies ONLY to independent proposals and a fixed winner-only/no-retry map.
No use for serial reservation, learned priority, global bid solver or fallback.
Quality ordering must match the canonical decoder exactly: use the same rounded
integer service durations and float64 comparison of stored float32 restoration
values, then ID tie-breaking, to avoid spurious float32 priority ties.

Store proposalU and old selected-mode log probability. For marginal arms the old
log is log q_old(T_s(U)); replay computes log q_new(T_s(U)) on the SAME recorded
state/proposal. For proposal arms both logs are log p(U). Never mix denominator
proposal likelihood with numerator physical likelihood. All aliases of a physical
matching must have identical q and gradients. Joint scalar PPOclip.2/MC advantage,
no per-component ratios/clipping, prefix credit or objective replacement.

State-only critic71->64->64->1, output multiplied byU4=115.2, MSE(V/U4,G/U4).
No critic sees proposals/physical actions. Core19974parameters (actor11141,
critic8833), all four EXACT same initial tensors/private objects and budgets.
All eight methods get freshly seeded identical initial critic tensors in private
objects, with private on-policy updates. No shared data/trained critic. PPO/norm
actor initial tensors paired. PPO18966/localMAPPO15364/fullMAPPO18052 parameters,
disclosed capacity difference; mechanism comparison matches exactly. GLOBAL
normclip.5 all methods, same as source studies; no grouped clipping intervention.

## Matched factorial controls and entropy treatment

Four core arms, same actor/critic/info/initial tensors/RNG roles/sample/Adam budget:
1. broker_marginal_ppo: quality broker, exact physical joint ratio (candidate).
2. broker_proposal_ppo: SAME quality broker/execution, proposal joint ratio.
   This is the PRIMARY removal of broker-aware likelihood learning.
3. id_marginal_ppo: fixed lower-ID broker, exact physical joint ratio.
4. id_proposal_ppo: fixed lower-ID broker, proposal joint ratio.

ID controls retain the same sensed duration/restoration features; only broker
priority changes. Initial proposal distributions match for all four; physical
initial distributions need not match across broker priorities. They are
robustness controls, not a claim quality priority is itself novel/superior.
All methods resolve invalid/urgent/busy support identically before arbitration.

IMPORTANT: retain beta=.01 times MEAN proposal categorical entropy acrossK for
ALL four arms. This is an explicitly shared AUXILIARY proposal regularizer;
it is NOT exact physical entropyH(q). No sampled -logq autodiff entropy shortcut.
Consequently the only learning contrast within each broker is the PPO ratio.
No reward shaping or conflict penalty. Match baselines' original entropy conventions
and disclose their different action alphabets. Keep original PPOunit1 and
normalized PPO/local/fullMAPPO U4 with common fresh critic initialization.

Greedy deployment is argmax per independent proposal then canonical broker,
NOT global mode of q. It is identical between likelihood modes at fixed weights.
Frozen candidate weights with proposal-quality mode MUST give identical physical
outcomes to candidate; changing a training likelihood cannot change fixed greedy
execution. Frozen ID-marginal/ID-proposal outcomes likewise match each other.
Retain these as correctness/descriptive checks, never claim frozen likelihood
removal degrades cost. Actual learning benefit requires separately retrained arms.

## Prior-art boundary and scaling scope

[PPO](https://arxiv.org/abs/1707.06347) provides the unchanged clipped objective.
[CAPG](https://proceedings.mlr.press/v80/fujita18a.html) and
[Marginal Policy Gradients](https://arxiv.org/html/1806.05134#S4) establish generic
marginalization/variance reduction. Do not claim a new generic MPG theorem.
The allocation-specific tractable likelihood and its controlled empirical
coordination evidence are the proposed contribution. This focused review is not
an exhaustive novelty proof. Whole physical ratio differs from separate-subaction
clipping in [Song et al.](https://arxiv.org/abs/2301.10919).

No guarantee clipped multi-epoch gradients or maintenance cost improve follows
from the score theorem. Both must be evaluated honestly. Output distribution
can express any deterministic feasible matching through unambiguous requests;
that representational observation is not a learning guarantee.

Actor neural work remains polynomial O(NK^2H) for fixed width, parameters grow
withNK; broker and observed-matching likelihood are polynomial, no catalogue.
Core inference only needs logits+broker; omit learning-only likelihood/Jacobian
from deployment timing. Original PPO baseline/commonN4deadline may enumerate;
unit/diagnostic reference enumeration is never on a core action/density path.
Only synthetic larger-shape correctness tests now; physical N/K scaling and
zero-shot/generalization claims remain closed until confirmed N4 success.
Full current-state broadcast is assumed, not communication-free local execution.

## Fixed FULL budgets / seeds / artifacts

N4/K2 balanced_a2_b4_u12,48policy+12common-tail ticks, originalreward=-cost/20.
8methods(four core+PPO/norm/localMAPPO/fullMAPPO)x3fresh replicas. Each983040
decisions, totals23592960 decisions/491520 trainEP/368640 Adam updates. 16envs,
rollout3072,epochs4,minibatch256,Adam3e-4/PPOclip.2/entropy.01/value.5/gamma1,
GLOBALclip.5. NineCP0..983040 every122880,each32DEV=6912conditions. Earliest
minimum meanH+Lcost selects each replica; spend entire budget regardless of choice.
No conditional extra training, resume, seed replacement, LR/temp/entropy/architecture
sweep or switch to a successful control after outcomes. Eight CPUworkers,
torch/BLAS1,tqdm,12h global hardstop; preserve technical/negative artifacts.
Same sample/update/checkpoint budgets, not an assertion of equal FLOPs; disclose
likelihood-computation/training overhead as well as deployment timing.

DEVtrain290000000..2/physical290100000..31/analysis290200000;
sealed CFtrain290300000..4/DEV290400000..63/TEST290500000..255/analysis290600000;
engineering290700000..3. Audit historical raw registries across worktrees and
known derived streams including all raw-state and budget studies/diagnostics.
CF hash derivation does not construct scientific states. DEV-only runner CLI.
Fresh timestamp directory, clean verified commit pushed before FULL. No teacher,
distillation, pretraining, warmstart, imported state bank or extra simulator steps.

Retain source/Git/protocol/config/physics/seed seals; all24 initial/final/CP tensors;
common critic hashes/private-storage tests; full training tapes/progress/behavior
replay and true actor/critic updates; completeDEVgrid/choices; selected768,
frozen288,deadline32 episodes=1088/65280physicaltransitions/52224decisions;
proposal+physical matching/priority/conflicts/rejections and cost components;
H48 health cutoff;432raw isolated NN+broker timing samples, choice/obs/threads/
workers bindings, audit and output hashes. No diagnostic actor Jacobian enters
learning. No short-budget remedy conditions in this NEW mechanism study.

## Locked DEV gates / confirmation and rejection

13mandatory point DEV endpoints:
- candidate cost gain vs broker_proposal_ppo >=.005 (main learning mechanism);
- gain vs each id_marginal_ppo/id_proposal_ppo >=-.01 (NI controls, no superiority);
- gain vs PPO and normalized PPO >=-.01 each;
- gain vs local and full-info MAPPO >0 each;
- vs BOTH PPOreferences: availabilitydelta>=-.01, pending/m delta<=.025,
  failures/m over policy48 delta<=.25 (six health).
Also0invalid and primary isolated NN+broker p95<=50ms. Zero conflicts or lower
Fisher alone never substitutes for cost+health. Every gate is required.

If FULL fails, preserve its verdict and do not auto-repeat with another entropy,
LR/decoder/priorities/budget/architecture configuration. Diagnose specific causes
before proposing any separately registered bounded follow-up. Do not rescue from
frozen outcomes, later checkpoint cherry-picking, cost-only or favorable seeds.

Only completed independently audited ALL-PASS DEV opens registration/testing of
CF harness at SAMEbudget,5fresh trainrep/64DEV/256sealedTEST. Seal checkpoint
choices before first test state. Paired crossed bootstrap200000 over train and
physical axes,13 simultaneous bounds, scientific familyr3 alpha=.05/12
(sequence alpha_r=.05/[r(r+1)]);confidence1-alpha/13. Prior source tests stayed
closed; this is smaller than their unused .05/6 and spends no old test evidence.
Bounds satisfy the same13thresholds. Claim superior PPO ONLY if its costgain
CI lower>=.01, the prelocked practical threshold. NI alone is near-PPO, not
superiority. Future scientific families must preserve the alpha-spending rule.
Physical largerN/K requires a new protocol after confirmed N4. Goal not complete
from DEV or an algebraic polynomial-work argument.

## Required correctness / workflow

Test exact physical density/gradient normalization against tiny exhaustive
references, alias-invariant likelihood/ratios/gradients, sample/replay behavior,
proposal-mode compatibility, forcedWAIT/illegal support, canonical rank/float64
near-tie/ID priority and decoder, exact frozen-mode equality/restoration,
critic no-action leak/private common initial tensors, capacities, checkpoint
roundtrip, all8real PPO update/physical-tail fixtures, independent primitive
cutoff/broker audits, serial-vs-process determinism, polynomial larger shapes/no
core enumeration, fresh seed namespaces/gate boundaries. Verify source and
NumPy actor/primitive trajectories independently after FULL. Retain actual
proposal arbitration diagnostics while learning on q(T(U)); never silently use
post-broker matching as if it were the sampled proposal.

Direct Ubuntu lab FULL; no separate smoke under active user override. Necessary
unit/integration fixtures, commit/push, tqdm and timestamped artifacts remain.
No SSH or initiated artifact transfer; provide manual rsync -avhP --partial when
results exist. All historical reports/verdicts remain available unchanged.
