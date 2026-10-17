# Typed-capacity Gibbs PPO v1 — prospective scratch N4/K2 FULL

Registered AFTER engineering algebra audit b311bc3 PASS12tests, artifact
technician_typed_gibbs_algebra_v1_20261017T115444999991Z, BEFORE new learner or
physical experiment. Prior capacity-order FAILED12/16 and all3 primary replicas;
its fixed-order control is not promoted. All old negative findings remain.
One falsifiable mechanism; no encoder/LR/head/order/entropy/budget sweep. User
permits FULL here without separate smoke; tests, commit/push, fresh artifacts.

## Question and mathematical boundary

Does normalizing a JOINT physical matching with learned future typed-capacity
continuation utilities improve scratch PPO over a matched policy whose future
continuations count only feasible assignments? Candidate uses
pi(M|s)=exp(sum_(m,j in M)u(s,m,type(j)))/Z. Its exact log-space count DP is
specified in docs/technician_typed_gibbs_design.md. WAIT energy0. Multiplicities
count labeled workers. Sample a type group then uniform available worker; joint
log density is physical, not latent proposal density. No detach on learned
suffix partitions/logZ in actor loss. Standard scalar joint PPO, no new objective.
No teacher/oracle future service labels/distillation/old learned tensors.

Current benchmark has2 repeated technician service profiles. Enforce same
compatibility/duration/restoration within a type. Eligibility can vary by machine;
free labeled members of a type must share it. Capacity0/1 counts retain actual
free worker identities and busy-worker state. DP O(B*N*(K0+1)*(K1+1)), polynomial
for fixed2types ONLY; arbitrary distinct types have product-state complexity.
No catalogue enumeration in core policy/sampling/MAP/likelihood/entropy/decoder.
N>4 physical/CF/test remain CLOSED. Synthetic algebra tests are not scaling proof.
Gibbs matching/count DP/autoregression/RL matching are prior art; see design's
primary sources. Potential contribution is this maintenance policy mechanism
and evidence, not invention of a generic partition function or new PPO loss.

## Three matched policies and four fresh external baselines

1 typed_gibbs_ppo: weighted learned future messages, machineID traversal.
2 typed_myopic_ppo PRIMARY: same neural utilities, inputs, parameters, reservations,
  exact physical entropy/MAP and PPO; replace future suffix utilities by0 so
  future messages count feasible continuations. Current utilities stay learned;
  normalize conditionals correctly, differentiate actual physical policy.
3 reverse_typed_myopic_ppo: same myopic policy, reverse machineID traversal.

All3 share private identical fresh actor/critic tensors and raw observation.
Both density families start uniform over ALL feasible labeled matchings via
zero final utility layer, including21 at complete N4/K2. Both have EXACT physical
MAP via count-state Viterbi, using concrete-worker density not aggregate group
mass. Thus candidate has no privileged decoder. Lowest available label pertype
breaks physical MAP ties; group ties WAIT then type0 then type1, consistently.
Gibbs order invariance at fixed utilities is algebra, not scientific performance.
Myopic reverse is an order-sensitivity control, not a second tuning candidate.

Shared raw actor state71+ownmachine6+typeprofile6+machineID4+freecounts2=89,
64tanh/64tanh/output2. Profile is original edge[compatibility,duration,restoration]
for eachtype; ignores assignment flag here since fullstate contains it. Freecounts
scaled by total typecapacity. Utilities computed in one batched neural pass for
ALLmachines BEFORE any sample; no prefix or hidden physical tick. Actor10050,
statecritic71->64->64->1=8833,total18883. Rawstatebroadcast means not local-only,
communication-free, zero-shot or permutation-equivariant. Full state does not
contain any future random shock. Agents send utility vectors to count-DP broker.

External literalPPO18966/normalizedPPO18966/localMAPPO15364/full-informationMAPPO
18052: fresh actors and same private initial raw state-critic tensors. PPO/norm
actors paired. Original PPO valueunit1; allothers115.2. FullMAPPO has fullstate
plus machineID; localMAPPO lessinfo. Causal claim is within EXACT matched core;
external comparisons do not isolate architecture, entropy convention or capacity.

## Frozen learning budget / panels / stopping

Gamma1, MC common state advantage centered/scaled perrollout; physical H48 plus
12commondeadline tail, reward-cost/20; Adam3e-4,clip.2,entropy.01,value.5,global
clip.5,16env,rollout3072,4epochs,mb256. Core regularizer EXACT physical matching
entropy divided byN (same for all3); no conditional/latent entropy approximation.
Other baselines keep original entropy treatment, no cross-family entropy claim.
115.2 value loss unit except literalPPO1. State critic never used in select.

21fits=7arms*3replicas,491520decisions each,160rollouts,7680Adam/10240EP perfit.
Total10321920decisions/215040trainEP/161280Adam. Checkpoints0/122880/245760/
368640/491520,32DEV each=3360nodes. PRIMARY fixedfinal491520, no early best model
promotion. Ancillary earliest minimum descriptive only. FULL budgets ALLfits,
no stop on plateau/winning,16CPUprocesses/torchBLAS1/tqdm/12hhardstop. Technical
failures abort siblings, preserve timestamps; no silent restart/overwrite/resume.

DEV train380000000..2;physical380100000..31;analysis380200000.
CFtrain380300000..4;CFDEV380400000..63;sealedTEST380500000..255;analysis380600000.
ENGtrain380700000;physical380700001;analysis380700002 only tests. Raw registries
and actual hashed initialization/action/minibatch/physical streams audited before
FULL. Histories include prior capacity-order derived streams, not just raw seeds.
No CF/test environment is constructed. New namespace technician_typed_gibbs_ppo_v1.

## Twelve point endpoints + stability/correctness, all locked

Primary main cost gain vs myopic>=.005 (0.5% practical). Gain vsreverse/PPO/norm
>=-.01; vslocal/fullMAPPO>0 (6cost endpoints). Availability candidate-minus-BOTH
PPO/norm>=-.01; pending/machine<=+.025; failures/machine<=+.25 (6health). ALL12.
Extra EVERY3replica primary gain>0;0invalid train/eval; allscratch/budget/density/
source/behavior replay/physical replay gatesPASS. Isolated432samples actor+DP+
decoder p95<=50ms, postworker shutdown; no scaling inference from N4 latency.

Only independently verified ALLPASS allows separately registered implementation
of confirmation5NEWreplicas,64DEV,256TEST, all7/fixedfinal/samebudget; freeze model
hashes before TEST. Paired crossed train-replica/physical-seed bootstrap200000,
sequential familyround7 alpha=.05/(7*8), one-sided alpha/12; same practical/N.I./
health bounds. PPO superiority ONLY if gainlower>=.01 vsBOTH PPO/norm. Test not
needed now and cannot open for a near miss. AnyFAIL rejects THIS implementation;
retain original verdict and diagnose specifically before another registered study.
No rescue by selecting reverse/myopic or ancillary checkpoints/replicas as winner.

## Mechanism / metrics / tests / artifacts

Primary matched retraining tests learnedfuture coupling. Health failure/availability/
backlog remain joint gates. Report maintenance/unavailability/failure/waiting cost,
starts, utilization, contested-demand (#active>#free) and rejection/conflicts.
Core has hardmatching so zero conflicts is expected, not proof of learned benefit.
Log canonical utilities/chosen physical logp/exact entropy on EVERY fixedfinal
core decision, enabling independent manual NN+enumerated distribution verification.
Report same-weights future-message knockout MAP disagreement and conditional
message spread on frozen final candidate states as DESCRIPTIVE associations,
not causal evidence replacing retraining. No new training/selection from knockout.

Necessary tests: exact helper suite; sampling/replay logratios/finite PPO gradients;
all21cold probabilities/core parity; free/busy/masked/type-profile contract; exact
MAP/entropy vs independent tiny enumeration, reverse canonical mapping; nocritic/
nocatalogue select; common physics/reward/tail; private initial critic/coretensors;
all7 completeEP optimizerupdates/roundtrip;12gate boundaries; freshroles; synthetic
fixed2type shapes only. Tests first, codecommit/push before FULL, no separatesmoke.

Fresh timestamped manifest/Git/protocol/registry/code/physics seals; parametercounts/
initialhashes; EVERY progress/episode/checkpoint/DEV grid; fixedfinal choices,
candidate probe observations; selected_episodes/coordination/physical journals,
latency raws/12endpoints; source training audit. Independent manual MLP and joint
labeled matching probabilities + primitive physics replay before verdict. No
historical optimizer-gradient replay claim where not retained. Results descriptive
DEV, no CI/scaling/goalpass. Preserve failed artifacts. Manualrsync -avhP --partial.
