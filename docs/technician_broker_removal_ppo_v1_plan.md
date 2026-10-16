# Broker-removal credit PPO-derived v1 — prospective FULL N4/K2

Register before implementation/new learning outcomes. Preserve all old negative
verdicts/unopened scientific panels. Random initialization only; no teacher,
distillation, old checkpoints, warm-start, simulator counterfactual outcomes,
Q-oracle targets, encoder/LR/entropy sweep. All states remain N4/K2 until gates
and separate scientific confirmation pass. No separate smoke; tests+commit/push
before FULL directly on this Ubuntu lab.

## Specific mechanism and theoretical boundary

Hypothesis: co-workers' sampled commitments create reward variation that a
state-only baseline misses. A removal-conditioned critic can assign more useful
credit, and exact broker-conditional averaging can remove latent-proposal noise
without discarding that information. This must improve trained matched controls,
not merely match an entropy or score identity.

Use the [removal-credit derivation](technician_broker_removal_credit_design.md).
Independent proposals U, fixed quality/ID-tie winner-only/no-retry broker T,
physical S=T(U). M_minus_i=T(U with i WAIT) depends only on other proposals.
Critic b_i(s,M_minus_i) predicts ACTUAL own-policy MC return conditional on that
input, not return of a no-technician world. No counterfactual simulator label.
B_i(s,S)=E[b_i(s,M_minus_i)|S] uses the exact polynomial fallback-winner law.
Idle-i removal keeps S; winning-i removal may promote a lower-priority idle
requester. Up to K(K+1)critic queries, no enumeration of all matchings/proposals
in learning/sample/select. Unit references may enumerate tiny fixtures.

The conditional score identity has prior-art foundations
[COMA](https://arxiv.org/abs/1705.08926),[MPG](https://arxiv.org/html/1806.05134).
No new generic baseline/variance theorem, first-use claim, monotonicity or hard
KL guarantee. Provisional contribution: this broker-aware removal-credit
construction and evidence. Earlier prefix/calendar credit failure is preserved;
this critic conditions on simultaneous pre-arbitration co-worker removals.

## Six exactly matched core arms

All core share identical fresh actor/state/removal-critic initial tensors and
private objects/updates, all current raw full-state information, same trajectories
RNG roles/tapes/budgets/entropy/loss weights. Actual future paths may diverge.
Quality broker uses restoration/duration then lower technician ID; no new learned
priority, retry, clock, health/reward/mask change.

1. removal_rb_ppo: candidate, physical log factors log f_i(S), B_i baseline.
2. factor_state_ppo: PRIMARY matched control, identical factors and BOTH critics
   trained on same kinds of real contexts/MC targets, actor uses old state V.
3. blind_removal_ppo: physical factors, removal critic uses zero matching channels
   in BOTH training and actor baseline query, retaining state/removed-index. Same
   allocated parameters,512context weights dormant by ablation. Tests whether
   co-action conditioning adds value beyond another state-value predictor.
4. proposal_removal_ppo: per-technician proposal logs, actual b_i(s,M_minus_i),
   before broker, not the conditional average. Same both critics/all queries.
5. proposal_state_ppo: per-technician proposal logs, old state V actor baseline.
6. joint_state_ppo: joint physical log q=sum_i log f_i, common old V advantage;
   objective dividedK to match behavior-gradient scale of factor_state_ppo.

Core actor is unchanged raw103->64->64->5 shared over technicians (11141params),
state critic71->64->64->1(8833),removal critic80->64->64->1(9409),total29383.
Removal input = rawstate71 + technician-major matching onehot8 + removed-ID/2.
All core/capacity critics initialize from a newly random common state V: copy its
first71input columns/hidden/output tensors into removal critic; extra9columns0.
Thus initial b_i(s,M)=V(s) for allM/i, not a pretrained/teacher initialization.
State/removal critics have private storage; subsequent updates independent.
All matched arms allocate and train the removal critic, including shadow critics
in state-baseline controls. Blind retains the same allocation but blocks matching
inputs explicitly. Actor input never gets other sampled proposals or new sensors.

## PPO-derived surrogate / cache / units

This factor-clipped surrogate is PPO-DERIVED, **not unchanged scalar-joint PPO**.
f_i are broker preimage factors, not standalone normalized action policies.
At frozen behavior parameters its summed score has the removal-credit identity;
that identity does not guarantee finite-update/clipped/batch-normalized returns.

Before ANY optimizer update in each3072transition rollout, cache with behavior
weights: original per-agent/joint logs, V_old, actual removal contexts,
b_old_i(s,M_minus_i), conditional weights and B_old_i. Cache all quantities for
every core arm even if unused, matching compute within core. Never refit critic
then recompute actor baselines on the same action/return batch. Detach complete
B/b, including actor-dependent conditional weights. Critic only trains on actual
M_minus_i, not conditional alternatives or any oracle label.

Let a0=G−V_old,mu=mean(a0),sd=populationSD(a0)+1e-8 over batch. ALL core variants
use this SAME normalization recipe, never independently center/scale removal
advantages. A_i=(G−baseline_i−mu)/sd, old baselines fixed for all4epochs.
Factor/proposal actor objective = mean_(batch,i) min(r_i*A_i,clip(r_i,.8,1.2)*A_i).
Joint control uses mean_batch min(r_q*A0,clip(r_q)*A0) /K, normalizedA0 asabove.
No claim that different policies have equal numerical rollout normalizers.

Loss = -actorobjective -.01mean proposal categorical entropy/K
       +.5critic_loss.
Core/capacity critic_loss = [MSE(V/U,G/U)+sum_i MSE(b_i/U,G/U)]/(K+1),U4=115.2.
Original non-aux baselines retain their original state-value-only loss; normalized
ones U4, literal PPO U1. All own-policy48step rewards include identical12tail,
gamma1,MCreturn,−cost/20,Adam3e-4,4epochs,mb256,globalgradclip.5,16envs,rollout3072.
Entropy remains the explicitly shared proposal regularizer; previous entropy
diagnostic negative is not reinterpreted. No coefficient/warmup/LR search.

## Strong external controls / budgets

Five external arms: literal ppo,normalized_ppo,capacity_ppo,mappo,full_mappo.
Capacity PPO has SAME actor/catalogue/units as normalized PPO plus the same
9409parameter removal shadow critic and averaged critic loss, actor still uses V;
its contexts derive from canonical technician tokens of its selected matching.
No independent-proposal theorem is claimed for this joint-action arm. Actor
never uses its shadow b. Initial ppo/norm/cap actor and raw state critic tensors
match; all11methods get freshly random common state critics/private storage.
Cap total28375;PPO/norm18966;localMAPPO15364;full-infoMAPPO18052. Actor/info
differences disclosed; exact29383capacity/info match holds for main core contrast.
More critic parameters cannot by themselves establish a coordination contribution.

33fits=11arms×3replicas, fixed491520decisions/model, all spent. CP0/122880/245760/
368640/491520,32DEVseeds/node =5280DEVcheckpoint episodes. Primary uses fixedFINAL
491520 for every method, no best-checkpoint substitution; ancillary min costs
descriptive only. Total16220160decisions,337920training episodes,253440Adamsteps.
8parallel workers,torch/BLAS1,12hour hardstop,technical abort sibling workers.
No metric-based stop, extra seeds or full resume; preserveFAILED and use newUTC
directory on a technical restart. tqdm for train/DEV/multi-fit progress.

## Nineteen locked point gates plus engineering/stability

Fixed-final cost gain positive=1-candidate/control:
- vs factor_state_ppo >=.005 (main trained credit mechanism);
- vs blind_removal_ppo >0 (co-action context mechanism);
- vs proposal_removal/proposal_state/joint_state/ppo/norm/cap >=-.01 each;
- vs local/fullMAPPO >0 each.
Ten cost endpoints. Nine original-form health bounds vs PPO,norm,cap each:
availability delta>=-.01,pending/machine<=+.025,failure/machine<=+.25.
ALL19 required. Also candidate gain vs factor-state strictly>0 in EACH3replicas,
zero invalid training/eval,reward/clock/budget/behavior/caching/source auditsPASS,
isolated432sample raw actorNN+broker p95<=50ms after workers close. No Ψ/V inselect.

Log actual baseline sensitivity/residuals/fallback weights during training,
both critic updates/cache digests/detachedadvantages; these are mechanism
descriptors, not causal performance claims. Primary evidence is same-info/capacity
trained control and context-blind trained control, not fixed-weight critic swap.
If any required gateFAIL: preserve/reject THIS configuration; no automatic
critic-weight/LR/normalization/entropy/budget/seed or clipping sweep. Diagnose a
specific cause before any separately registered follow-up. Do not promote a
control or choose an earlier checkpoint. No CF/test/largerN opening from algebra.

## Splits and sealed scientific contract

Registry new rawtrain330000000..2,DEV330100000..31,analysis330200000;
CFtrain330300000..4,DEV330400000..63,TEST330500000..255,analysis330600000;
ENG330700000..2. Actual init/action/minibatch/replay/physical/bootstrap roles use
THIS version. Audit all raw and actual derived streams/history before run. CF/test
seeds only listed/hashed for collision check; no states constructed in DEV.

Only independently verified ALLPASS DEV allows separate CF implementation:
5NEWscratch replicas, same491520budget/all11arms/fixed endpoint/64DEV/256sealedTEST.
Hashfreeze models before scientific test. Crossed paired bootstrap200000 draws
over train replicas/physicalseeds, shared draws across19endpoints. Sequential
family r5 alpha=.05/[5*6], each one-sided bound alpha/19. Require lower main
gain>=.005,blindcontext/local/full lower>0, allNI lower>=-.01 and9health bounds.
Optional PPO superiority ONLY same finalPPO gain lowerbound>=.01; never fromDEV
mean, extra info or favorable normalization. Near-PPO/scalability may be main
contribution, but original full goal still needs mechanism+CI/health and future
polynomial scaling after confirmed N4. No claim of decentralization or zero-shot.

## Tests and artifact contract

Retain8exhaustive removal-law/score baseline checks, add actual neural baseline
equivalence/privateinitialstorage, cachedbehavior detach beforeψupdate, factor
vsjointgradient scaling, per-factor clipping/math/advnormalization, shadowcritic
training incontrols, blindzero inputs, removalcontext includesfallback, checkpoint
roundtrip/RNGpairing/primitive physics/completebudget/seed/gate boundaries and
tiny engineering integration tests. No separate smoke run, no freshDEV outcome
in disposable tests. Commit/push verified code before FULL.

New timestampdir/refuse overwrite;manifest cleancommit/device/runtime/testclosed,
protocol/registry/physics/config/source seals, raw/derived seed audit, initial
actor/V/Ψhashes and capacities. Per-fit progress/episodes/cache diagnostics,
all checkpoints+DEVnodes, fixed-finalbindings, final actor/proposal/matching/
primitive traces and episodes, allcostcomponents/H48health, raw isolatedtimings,
19points/perreplicas/verdict. Independent read-only verification required for
any promotion; auditPASS does not mean sciencePASS. Provide manualrsync; no SSH
or agent transfers. No prior learned artifact is an input to this experiment.
