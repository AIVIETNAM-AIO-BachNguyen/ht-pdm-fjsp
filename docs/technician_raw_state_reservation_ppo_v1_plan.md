# Raw-state reservation PPO v1: prospective N4/K2 full study

Register before new learning. Preserve every negative prior result, including
joint-clearing FULL7/13 and frozen execution diagnostic: sampling increased
joint cost1.78% in all3replicas, so that proposed execution fix was rejected.
Previous goal turn made progress(FULL+independent verification/report); current
turn completes a new falsifying diagnostic. No scientific alpha/test panel opened.
Direct lab FULL, no separate smoke; necessary correctness/integration tests,
commit/push before execution. No teacher, pretrained weights, old checkpoint,
demonstration bank, future-event input, shaped reward or new PPO objective.

## Hypothesis and scope

Test whether preserving PPO's direct raw-state MLP representation while replacing
its matching catalogue with conditional technician tokens retains small-setting
quality, and whether the reservation/prefix service improves learning/outcomes
over equally informed independent proposals. Prefix conditions later choices on
the actual within-tick commitments, so they can account for resources already
promised; it is not a fixed preference for short service or WAIT.

Normalized raw-state PPO already learns near262/machine under this frozen physics;
multiple edge-pooled and Gaussian designs remain above it. That motivates a
different controlled design, NOT proof that pooling caused past failures. Input
representation and action distribution differ from past graph studies; do not
attribute cross-round changes to one encoder intervention. The causal mechanism
contrast here is full reservation/prefix versus its matched removal controls.

## Defined policy

State s is the SAME physical vector as centralized PPO:6N+3K+4NK+9 components
(71atN4/K2), no pooling of raw current observations before the actor. A shared
3-layer tanh MLP,H64, takes s plus current technician3features, its own4N edge
features, flattened provisional plan N*K, reserved-machine mask N, and public
technician index j/K. All extra physical descriptors are deterministic views
already present in s; no extra sensing. Plan/reservations are own team's current
internal choices, not other-policy/future labels. Actor input11N+3K+5NK+13
(103atN4), outputN+1logits: WAIT0 or machine+1.

Process Ktechnicians in fixed forward order. Mask current nonurgent/busy/illegal
pairs and already reserved machines; always permit WAIT. Sample one categorical
token per technician. Busy/no-legal technician forcedWAIT (log/entropy0). Reserve
selected machines and broadcast exact plan before the next choice. All starts
execute together; no physical time/reward advances inside the Kchoices. Canonical
action vector stores tokens by technician ID, shapeK, no latent order/dummy token.
Evaluate recorded likelihood by replaying the same conditional plan/masks.
Coordinated vectors uniquely determine a physical matching; exact joint density
is product of Kconditionals. Joint scalar PPO ratio on their log SUM; entropy
mean across K, state-only MC advantage, not token-wise clipping/prefix credit.
Greedy deployment takes conditional token argmax, no global-MAP claim.

Four core arms, EXACT same parameters/random tensors/current physical sensing,
actor and scalar critic capacity, learner RNG roles and sample/Adam budgets:
- raw_reservation_ppo: actor sees plan+reserved mask; hard reservations enforced.
- mask_only_ppo: actor plan+reserved INPUTS zero, hard reservation output mask
  still enforced. Isolates explicit prefix conditioning from collision exclusion.
- reverse_reservation_ppo: full prefix/reservations, technicians reverse order.
  Canonical stored vectors remain by technician ID. This is an order alternative.
- independent_raw_ppo: actor prefix inputs zero, each uses original support;
  duplicate machine requests retain highest restoration/duration then lower
  technician ID, losersWAIT with no retry. Exact proposal-vector likelihood,
  not marginalized post-arbitration matching probability (many-to-one mapping).

This removal tests the whole reservation/prefix service including collisions.
No claim that explicit plan alone is useful unless its contrast supports that.
Initial local parameters match; masks/conditioned traces can make physical initial
policies differ. Same RNG role does not guarantee pathwise-identical actions.
Frozen candidate weights also evaluated with mask-only/reverse/independent
POLICY conditioning AND matching rules changed together, then original mode
restored. These are descriptive DEV interventions, not new training replicas.

Separate critic is ordinary PPO MLP(s,H64,1), output multiplied byU(N)=28.8N,
MSE(V/U,G/U). AtN4same8833critic parameters as PPO; actor11141, total19974.
All methods receive the SAME freshly seeded initial critic tensors in separate
objects; no sharing on-policy training data/updates after initialization. A
dedicated critic RNG stream is audited, independent actor initialization roles.
PPO/norm actor initial tensors/RNG paired. Baseline parameter counts PPO/norm18966,
localMAPPO15364/full-infoMAPPO18052; disclose differences. All8methods use original
GLOBAL norm clipping.5 (different from earlier grouped graph clipping, common
here), standard Adam3e-4/PPOclip.2/entropy.01/value.5/gamma1/MC. Literal original
PPO retains unit1; normalized PPO/local/full useU4. Core and normalized PPO thus
share raw-state critic architecture/initial tensors/units/global-clipping regime.

Full current graph broadcast is assumed; not network-local or communication-free
execution. Fixed ordered raw inputs are not permutation-equivariant. Parameters
grow polynomially with N*K, not constant-size; sizes would train separately from
random weights. No zero-shot/generalization claim. Actor neural work O(N*K^2*H)
for fixed width; outputs N+1 per stage and never a full matching catalogue.
Only synthetic larger-shape correctness tests before N4passes; no larger-size
physical experiment now. Common tail uses existing exactN4deadline rule.

## Prior-art boundary

[PPO](https://arxiv.org/abs/1707.06347) supplies the unchanged objective;
[MAT](https://arxiv.org/abs/2205.14953) establishes autoregressive multi-agent
coordination (ours is an MLP with exact reservation mask, not its Transformer /
advantage-decomposition theorem); [POAD](https://arxiv.org/abs/2405.15821) already
studies action decomposition/credit (ours retains whole-action MC PPO, not BAD).
No first autoregressive policy, new PPO theorem or generic novelty claimed.
Provisional contribution is the teacher-free maintenance policy representation /
coordination with controlled quality/health and eventual polynomial scaling.
This focused review does not imply superiority over unimplemented published methods.

## Full budgets, fresh panels, stopping and artifacts

N4/K2 fixed balanced_a2_b4_u12,48policy+12common-tail ticks, reward=-cost/20.
24fits8methods×3replicas,491520decisions/model (11,796,480total),16vector envs,
rollout3072,4epochs/mb256=7680Adamsteps/model,184320total. Checkpoints0/122880/
245760/368640/491520,each32DEVepisodes (3840total); earliest minimum mean H+Lcost
chooses each replica checkpoint. Spend entire budget.8CPU workers,torch/BLAS1,
tqdm training/multi-seed evaluation,12h hardstop. No metric-based early stopping,
extra budget, best-replica selection, temperature/LR/encoder sweep or warm start.
After training,768selectedDEVreplays+288frozen-policy/decoderconditions+32deadline.

Registry DEVtrain270000000..2/DEV270100000..31/analysis270200000;CFtrain270300000..4,
DEV270400000..63/TEST270500000..255/analysis270600000;engineering270700000..3.
366raw/82438derived expected (includes9dedicated critic-init streams). Audit all
historical raw registries across worktrees and known learning/diagnostic derived
streams. Hash derivation never constructs scientific-test states. DEVrunner must
not instantiate CFpanel. Every run gets fresh UTCdirectory; never overwrite.

Artifacts: protocol/config/registry/source/Git hashes; physics; parameter counts /
initial tensors; streamed training progress/episodes/behavior replay; checkpoints;
complete DEV choices/grid; selected and frozen trajectories/proposals/reservations /
conflicts/rejections/cost decomposition/48cutoff health; isolated actor+canonical
decoder raw timing samples; audit and output SHA/size bindings. Tests: exact
conditional density normalization and gradients, recorded trace/sample replay,
prefix versus mask-only/independent semantics, reverse canonical indexing,
forcedWAIT/invalid support/arbitration, common critic values but separate storage,
checkpoint/temporary mode restoration, actual PPOupdates/physical tail, independent
primitive selected trajectory/cutoff/broker audits, serial-vs-process equality,
parameter growth/no core enumeration, fresh seeds and gate math. Commit/push before
FULL; no separate smoke. Keep every original failed artifact/verdict unchanged.

## Locked gates

Same13mandatory DEVpoint endpoints as joint-clearing, adapted method names:
independent gain>=.005; mask-only/reverse gain>=-.01 each; PPO/norm gain>=-.01
each; local/fullMAPPO gain>0 each; versus BOTH PPOreferences availabilitydelta
>=-.01, pending/m<=+.025, failures/mper48<=+.25 (six health). Also0invalid and
isolated candidate NN+token decoderp95<=50ms. No superiority claim from serial /
mask-only noninferiority; zero collisions alone does not prove mechanism quality.

Only independently audited PASSING FULLDEV unlocks fresh5replicas/64DEV/256sealed
TEST at the same budget. Freeze all checkpoint choices before tests. Crossed
paired bootstrap200000 resamples training and physical axes; family13,
round2alpha=.05/6unused, confidence1-alpha/13. Simultaneous bounds satisfy same
thresholds. Claim superiorPPOcost only if gainCIlower>=.01 (practical threshold
locked), otherwise near-PPO plus future scaling may be contribution. Confirmation
harness must be registered/tested before any scientific state is opened. Larger
N/K/generalization require separate prospective protocol after confirmed N4.
Manual rsync -avhP --partial only, no SSH or agent-initiated artifact transfer.
