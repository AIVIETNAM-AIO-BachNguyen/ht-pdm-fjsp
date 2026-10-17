# Capacity-priority machine-agent PPO v1: prospective N4/K2 FULL

Register before implementation/new learning data. Preserve all previous negative
RSC/prefix/scarcity/broker/removal-credit/scalar-calibration verdicts. No teacher,
distillation, old learned weights, labels, warm start, optimizer/encoder/LR/entropy
sweep. New random initialization and raw/derived seed namespaces. FULL directly
on this lab, no separate smoke; unit/integration tests then commit/push first.

## Evidence, hypothesis and contribution boundary

Removal critic v1 passed independent correctness but failed cost vs matched state
and blind controls. Its heldout scalar calibration failed4/9gates. Stop those
routes. Earlier technician-first raw reservation failed vs independent; pooled
RSC/prefix models remained far from strong raw-state PPO. No finding establishes
that another generic encoder, critic or longer budget fixes the issue.

Hypothesis: allowing machines with fewer fast service options to claim a worker
before flexible machines helps preserve scarce worker opportunities; conditioning
subsequent machine decisions on actual reservations turns that priority into a
learned matching policy. Measure cost vs fixed-index order and equally informed
independent machine proposals. Do not claim the deterministic order is optimal:
WAIT stays legal and later machines can still act. No fixed short-job/long-job
ban, oracle service label or future simulator input.

Machine-agent serial allocation was ALREADY used in repository solver-comparison
v1 with rotating order/different physics. Raw technician-first and capacity
messages were also tested. The new contrast is capacity-priority MACHINE order
under current frozen physics/raw MLP, with unique matching density and matched
initial physical distributions across coordinated order/message controls. It is
not first autoregression, first reservation, or first machine agent.

Prior art: [autoregressive constrained allocation, NeurIPS2024]
(https://arxiv.org/abs/2409.18735) already studies sequential allocation and initial
sampling bias; [PARCO, NeurIPS2025](https://arxiv.org/abs/2409.03811) uses parallel
multi-agent decoding/conflict handlers. [MRV/variable ordering]
(https://aima.cs.berkeley.edu/4th-ed/pdfs/newchap05.pdf) is classical. No generic
novelty claim for ordering, feasibility masks or combinatorial counts. A possible
contribution is this maintenance coordination representation with same-info/budget
ablations, near-PPO quality/health and later polynomial scaling; unproven now.

## Policy and exact density

At physical state s, original urgent/legal pair mask remains unchanged. Compute
capacity c_m=sum_(currently legal free j)1/D_mj. Process active machines in
ascending positive c_m, tie lower physical machine ID; inactive machines last,
by ID. This order is fixed for this physical tick from CURRENT state, never
learned and never chosen from future outcomes. Reservations update within the
tick; no simulation/reward/time advances between agents. A machine chooses
WAIT0 or free technician j+1; chosen technician is reserved, WAIT lets later
machines act. Each machine visited exactly once, actions stored by machine ID.
Every feasible partial matching has one canonical trace, even across different
orders. Joint scalar PPO ratio uses SUM of conditional token logs.

Shared raw actor input: full state71 + own machine6 + own edges8 + current
plan8 + reserved-technician mask2 + processed-machine mask4 + machine onehot4
+position/N1 =104 atN4/K2. Hidden64,tanh,tanh,outputK+1=3. Actor11075,
state-only critic71->64->64->1=8833,total19908. Actor has full current state
broadcast; not local-only/communication-free/permutation-equivariant/zero-shot.
No critic during select. No matching catalogue in any core actor or decoder.

To avoid an order-dependent cold physical policy, zero the final actor linear
weights/bias in ALLcore arms. Use a fixed log feasibility-count prior. Current
benchmark supports complete bipartite graphs between urgent unassigned machines
and idle technicians. Let C(n,k) count partial matchings, C(0,k)=C(n,0)=1,
C(n,k)=C(n-1,k)+k*C(n-1,k-1), computed by log DP in O(NK), no enumeration.
For n unprocessed active machines and k free workers, token prior weights are
C(n-1,k) for WAIT and C(n-1,k-1) for EACH legal technician. Add their log weights
to learned residual logits then masked softmax. Initial probability of EVERY
feasible physical matching is1/C(n,k), independent of coordinated order/explicit
prefix channels. AtN4,K2 full support C=21. Assert complete-bipartite eligibility;
do not claim this count works for arbitrary skill masks. Larger-scale physical
tests prohibited now. Synthetic shape/correctness tests may cover larger arrays.

This elementary initialization is not teacher initialization or a new generic
de-biasing algorithm. A prior weight is not a return/action quality label. All
core receive it. No claim of isolated performance benefit without a trained
prior-removal study, which is NOT included here.

## Six exactly matched core arms / five external controls

1 capacity_order_ppo candidate: capacity order, full reservations/prefix inputs.
2 fixed_order_ppo PRIMARY: machine ID order, identical reservation/input/prior.
3 reverse_capacity_ppo: descending capacity, same coordinated policy otherwise.
4 mask_only_order_ppo: candidate order/hard reservation output support, but plan,
  reserved and processed INPUT channels zero; retains original state/ID/position.
5 independent_machine_q_ppo: no previous proposal/reservation INPUT or support
  conditioning; all machine requests independent given s. Fixed capacity order
  determines RNG slots/position only. Each requests original idle worker or WAIT.
  Fixed broker keeps largest R/D request per TECHNICIAN, tie lower machine ID;
  losers WAIT, no retry. Joint likelihood is exact polynomial physical matching
  mass (transpose the verified fixed-broker factor law), not latent proposal mass.
6 independent_machine_p_ppo: same independent actor/broker/prior, scalar JOINT
  latent proposal likelihood. Alias-density control, not the primary mechanism.

Independent priors use original n,k single-machine marginals: WAIT C(n-1,k)/C(n,k),
each worker C(n-1,k-1)/C(n,k), same for every active machine; no fake reservations.
Its initial physical distribution differs from coordinated uniform21 by intended
coordination removal. Do not attribute this contrast purely to learned messages.
Core share allocated19908parameters, private identical initial tensors, same
current sensing, critic units, physical/action/minibatch roles/budgets. Actual
trajectory paths diverge. Exact cold physical-policy parity holds for arms1..4.

External freshly trained controls: raw_tech_ppo (previous raw technician-first
reservation actor,19974parameters, new random actor); literal ppo18966;
normalized_ppo18966;localmappo15364;full-informationmappo18052. Common fresh raw
state-critic tensors/private objects across all11; PPO/norm actors paired. Original
PPO uses valueunit1, allothers115.2. Information/capacity/architecture differences
disclosed; causal contribution is tested within the matched machine-agent core.

## Unchanged PPO / experiment budget

Standard scalar JOINT clipped PPO, common state-value normalized MC advantage,
gamma1,48policy+12commondeadline tail,-cost/20,Adam3e-4,clip.2,entropy.01,value.5,
globalgradientclip.5,16vectorenvs,rollout3072,4epochs,minibatch256. Allcore entropy
is mean conditional entropy over Nmachine stages; for independent arms it is
latent proposal entropy, not physical entropy. Both independent q/p reported;
main order control shares exact entropy/likelihood semantics. External policies
keep original entropy treatment; no entropy-superiority claim from cross-family
comparison. No auxiliary/removal critic or new RL objective.

33fits11arms*3replicas,491520decisions/model,160rollouts/7680Adamupdates/10240EP.
Total16220160decisions,337920trainEP,253440Adamupdates. ALLfits spend fullbudget.
Checkpoints0/122880/245760/368640/491520,32DEV/node=5280DEVepisodes. PRIMARY uses
fixed final491520, ancillary earliest minimum descriptive only. Allreplicas stay.
16CPUprocesses,torch/BLAS1,tqdm,12hourhardstop/technical sibling abort, no resume,
newUTC/refuseoverwrite. No LR/budget/head/prior/order rule retuning from outcomes.

## Sixteen point gates and sealed scientific contract

Candidate cost gains: vsfixed_order>=.005;vsindependent_machine_q>=.005;
vsreverse/mask-only/independent_p/rawtech/ppo/norm>=-.01 each;
vslocal/fullMAPPO>0 each.10cost endpoints.6original health bounds vsBOTHppo/norm:
availability delta>=-.01,pending/machine<=+.025,failures/machine<=+.25. ALL16.
Extra EACH3replica main gain>0,0invalid training/evaluation, exact budgets/
behavior replay/source/initialization/density auditsPASS, isolated432raw actor+
decoder p95<=50ms after workers closed. N4timing not scaling proof. Zero
conflicts alone not mechanism validation. Same-weight knockout descriptive only
if laterregistered, not a replacement for independently trained controls.

Only independently verified ALLPASS DEV permits separately implemented/registered
confirmation:5NEWreplicas,64DEV,256sealedTEST,samebudget/all11/fixedfinal; hashfreeze
models before tests; paired crossed bootstrap200000, sequential familyround6
alpha=.05/(6*7),one-sidedalpha/16, same practical/health bounds. PPO superiority
ONLY if same-test gainlower>=.01 against BOTHppo/norm; not required for near-PPO
contribution. No CI/scientific/scaling claim from DEV. AnyFAILreject THIS design,
preserve artifacts and diagnose before another distinct prospective hypothesis.

Registry DEVtrain360000000..2/physical360100000..31/analysis360200000;
CFtrain360300000..4/DEV360400000..63/TEST360500000..255/analysis360600000;
ENGtrain360700000/physical360700001/analysis360700002. Fresh complete raw/actual
derived role/history audit. CF/test only seed metadata/hashes, no states in DEV.

## Tests / artifact schema

Exhaustive tiny matching support/normalization, all21uniform cold probabilities,
order/mask initial physical parity, forcedWAIT, non-bipartite mask rejection,
recorded trace sample/replay, independent q vs all81latent proposals, transposed
arbitration/ties, finite actor gradients and logratio, no critic inselect,
unchanged physics/reward/tail, same core capacities/private initial critic,
checkpoint roundtrip, actual full-episode PPOupdates forALL11arms,16gate math,
seed splits, synthetic polynomial shapes. No separately requested smoke.

Fresh artifact manifest/protocol/registry/config/physics/code seals, allinitial
hashes/counts/checkpoints, complete trainingprogress/episode/DEV grids, fixedfinal
choices/observations/selectedepisode/policy/physical journals, isolated rawtimings,
16gates/costcomponents/health/order/reservation/conflict audit. Independent manual
actor/primitive replay verification before scientific interpretation. Do not
SSH/transfer; provide manualrsync -avhP --partial after results.
