# Technician Exchange PPO v2: prospective research protocol

Registered before any new learning outcome. Preserve negative scarcity v1/v2/v3,
completion-credit v1/v2 and exchange v1; none authorizes development or confirmation. No
teacher, demonstration, old learned tensors, distillation, PPO warm start, extra
simulator branch or future-event input. All fits initialize randomly.

## Hypothesis and prior-art boundary

Exchange v1 fails6/25 pilot screens. Its frozen diagnostic finds N4 exchange
activity with unchanged final matching in17/48 decisions and a cost/availability
gap with large replica differences. This does not prove cancellation causes
bad learning. Test whether committing repaired technician assignments reduces
redundant internal revisions and improves allocation under the same information,
architecture, objective and budget. All new policies learn from random weights.

Learned local improvement/rewriting, graph-based matching and sequential/PPO
action decomposition are established; see the v1 prior-art matrix. In particular,
[Wu et al., section IV](https://arxiv.org/pdf/1912.05784) masks the previously
selected node pair to prevent reversing a local operation. Generic anti-cycling,
masking and commitment are not claimed novel. Include an immediate-inverse
mask control, in addition to unrestricted exchange. The domain-specific ordered
commitment mechanism and its empirical cost/health/scale results are the
provisional contribution, not a new policy-gradient or trust-region theorem.
The focused review does not establish broad novelty or superiority over MAT,
POAD, NeuRewriter or published matching policies not implemented here.

## Exact internal action process

Each physical decision uses2K discrete internal slots, fixed technician order.
First K slots construct a provisional matching as in canonical technician
reservation. Token0=WAIT; tokens1..N request the corresponding legal urgent
machine; tokenN+1(DROP) is forbidden during construction. Reserve chosen machines.
Busy technician has only token0. No physical time, reward or restoration advances.

Next K slots revisit the technicians in the same order. Token0=KEEP, tokenN+1
releases that technician's provisional assignment (only if nonempty). Choosing
its current machine is forbidden as duplicate KEEP. Choosing an unclaimed legal
machine replaces its current assignment. In the bilateral arms, choosing a machine
held by another technician swaps their provisional jobs; if the requesting
technician has no job, the former holder becomes unassigned. A swap is allowed
only when the reciprocal job is legal for its recipient. All coordinated intermediate plans
are valid matchings; already active physical services are never moved or canceled.
Apply final starts simultaneously after all2K slots. No search-until-convergence,
objective oracle, best-of-samples or extra physical rollouts.

The candidate exchange_ppo permits a bilateral target only if its current
holder's index is strictly greater than the repairing technician's index.
After each repair slot, that technician's current assignment is committed and
cannot be changed by later technicians. Unclaimed replacements, KEEP and DROP
remain available subject to the same legality rules. No earlier committed
assignment is silently released. All masks are computed from observed state
and recorded internal choices, with no future event or learned oracle.

unrestricted_exchange_ppo permits exchanges with any other holder, exactly the
v1 mechanism (fresh weights). inverse_mask_ppo instead forbids only an immediate
inverse of a bilateral operation in the preceding internal slot: if slotj-1
exchanged with holderj, slotj cannot request the machine now held byj-1. This
also covers reversing a transfer. A preceding KEEP, DROP, unclaimed replacement
or construction slot does not install that restriction. The previous-slot
holder is reconstructed from the recorded trace when replaying likelihoods;
it adds no actor/critic features or parameters. All core actors compute the
same logits before arm-specific masking.

Ordered commitment prevents repair cancellation: at the earliest non-KEEP
repair, the current technician changes from its construction assignment and
is then immutable to later slots, so the final matching differs from the
construction matching. This mechanical invariant does not imply improvement.
All feasible physical matchings still have construction+all-KEEP traces.
Multiple different construction/repair traces can still yield the same final
matching, and equivalent technician identities can yield equivalent service
outcomes; do not claim unique physical traces or a quotient-policy solution.

Different valid traces may produce the same final matching. Optimize the sampled
**latent trace** with exact conditional log probabilities; never treat one trace's
probability as the marginalized physical matching probability. All feasible
physical matchings remain supported through construction followed by KEEP.
There are21 at N4/K2 with a dense legal pair mask; enumeration is allowed in small correctness tests only.

Core arms, identical architecture/tensors/current physical information and
provisional messages (except the independent control),2K neural calls,
optimizer counts and physical training budgets:

- exchange_ppo: ordered commitment, bilateral targets only unrepaired holders.
- unrestricted_exchange_ppo: exact unrestricted v1 exchange permissions. Main
  commitment ablation; identical before-mask actor/critic inputs and parameters.
- inverse_mask_ppo: only the previous-slot inverse mask above. A control for
  the established anti-reversal idea; all other unrestricted permissions remain.
- self_repair_ppo: own KEEP/DROP/unclaimed revisions, no change to another
  technician's assignment. Controls for having an actionable second pass.
- construct_only_ppo: all2K NN calls/logits, repairs masked to KEEP only.
- independent_ppo:2K individual own-intent choices and full current physical
  sensing, no other intentions or reservation masks; final quality arbitration
  drops collisions without retry. Same physical-state critic and raw sensing.

Also train a fresh completion_reference_ppo: the exact v2 completion-policy
architecture, prefix-credit/component clipping, fixed units, no old weights;
K choices,18444 parameters. It is an external algorithm baseline, not an
equal-internal-compute ablation. Disclose that difference. Retire unsupported
v2 calendar/prefix claims rather than relabeling them as success.

## Network and learning

Raw inputs unchanged M6/T3/E4/G9 (N4 state71). Shared current graph/broker assumed;
not network-local decentralized execution. Machine permutation equivariance;
fixed technician order/indices disclosed, not arbitrary-order invariance.
Per-edge raw22 actor encoder, hidden27; own provisional calendar active,
duration/12, known completion age/8, and edge assignment flag through calendar4
encoder. Rebuild descriptors from the current provisional matching each slot;
released jobs revert to their original physical descriptors. No new stochastic
forecast or inaccessible state. Binary flag caching allowed if exactly equivalent.

Move scorer inputs5H+2: candidate edge, current own job edge, target holder's
edge, reciprocal job edge (zero when absent), mean graph edge context, phase
and technician-index/K. Control head2H+2 scores WAIT/KEEP and DROP. Edge encoders
and scorer shared by all slots/technicians. Unused or illegal moves remain masked.
Independent actor constructs descriptors from its own intent only.

Separate physical-state critic, evaluated ONCE per physical decision: raw22/value
calendar4 encoders with pointwise tanh fusion; pool by technician; nonlinear
context2H+2 (technician vector, global vector, index/K, physical free flag); mean
contexts; value headH+global9. It does not consume proposed assignments, repair
actions, current/future choices or decoded prefixes. Expected core19255parameters
for every size/arm; verify count before pilot. Repeating scalar value across
2K for generic learner compatibility does not add critic calls or supervision.

Exact joint trace ratio exp(sum_j newlog_j-oldlog_j), ordinary clipped PPO with
one state-only MC advantage, not component clipping or a new policy-gradient
theorem. Same objective in every core arm. Entropy is mean over2K conditional
slots; forced tokens contribute0. Value MSE averaged over repeated slots equals
the scalar-state loss. MCgamma1,48 physical decisions plus common12deadline tail,
reward=-original cost/20, no intermediate shaping/rewards. Fixed value unit
U(N)=unavailable_cost*N*48/20=28.8*N; output U*head, loss MSE(V/U,return/U).
Actor and critic separately clipped at.5 for graph arms; no parameter sharing
between them. Original PPO remains literal scale1/loss1. Normalized PPO/local/
fullMAPPO useU4. Original/normalized PPO paired raw initial tensors and RNG roles.
PPO18966/local15364/full18052parameters; disclose capacity differences.

Only polynomial local candidate sets: N+2 per technician per slot, no listing
of full matchings. Raw pair encoding O(NK),2K slots each O(NK) pooling plus O(N)
move scores: O(NK²) neural work, bounded memory polynomial. Sizes trained separately;
no zero-shot scaling or optimality claim. Candidate inference skips the critic.

## Budgets, panels and prerequisites

Same scalar balanced maintenance N4/8/16/32,K=N/2,H48+tail12, all prices/failure
processes/eligibility and objective unchanged. Deadline large uses min-cost flow;
only N4 original/normalized PPO and N4 deadline may use their small catalogue.

Full fits491520 joint physical decisions,vector16,rollout3072,epochs4,mb256,
lr3e-4,clip.2,entropy.01,value.5,gradientclip.5. CP0/122880/245760/368640/491520.
Whole budgets spent, all replicas retained, earliest minimum DEV mean cost per
replica selected. Processes8,torchthreads1,tqdm, fresh UTC artifacts, global48h
hard stop, no extension or outcome-based early stop. Incomplete grids cannot pass.

Smoke: N4/N32,18fits,one replica384decisions,vector2/rollout192/epochs2/mb48,
CP0/192/384,DEV2/engineering-test2. Pilot: N4/N32,36fits,two replicas61440,
vector8/rollout768/epochs4/mb128,CP0/30720/61440,DEV16,NO scientific tests.
FullDEV96fits,all sizes,3rep/fullbudget/DEV32,no tests.
Confirmation160fits,all sizes,5rep/fullbudget/DEV64/test256.

Registry160000000..161300000 groups fixes all train/DEV/test/analysis panels,
387 actual raw seeds mutually/historically disjoint, including all unopened
previous confirmations. Audit all derived roles. Core arms paired initial/action/
minibatch roles; v2-reference has its own initialization/learning role; physical
tapes paired across every method per size. Sizes independently namespaced.

Pilot -> fullDEV only after audited COMPLETE/testclosed/exact settings and seals,
EVERY nonpooled cost/health point screen PASS, and at least one selected trained
candidate replica at BOTH sizes; all replicas remain in means. Report all four
N32-only pooled mechanism proxies even if negative: they may permit exploratory
fullDEV only because they do not measure the registered three-size pool.
FullDEV must pass EVERY point screen before confirmation, with identical frozen
source/PLAN/registry/core settings and selections sealed before opening test.
No test reuse, teacher rescue, best-seed exclusion, health relaxation or budget
extension. Existing failed protocols remain failed and cannot authorize this run.

## Scientific family, practical margins and system bounds

Round2 retains alpha=.05/6, no teacher-free confirmation has opened; previous
teacher round1 spent.025. Bootstrap200000,53Bonferroni endpoints, two-sided
confidence1-(.05/6)/53. Common replica resampling weights across sizes,
independent episode resampling by size, paired methods within size; pool large
gains inside every draw. Five replicas give approximate coverage; publish all
replicas and leave-one-out diagnostics.

N4 ten gains versus unrestricted/inverse-mask/self/construct/independent/
completion-reference/originalPPO/normalizedPPO/localMAPPO/fullMAPPO plus six
health differences (BOTH original and normalized PPO references)=16. Large
seven gains versus unrestricted/inverse-mask/self/construct/independent/reference/
deadline plus3health=10*3=30. Three large/N4 cost/machine ratios and FOUR pooled
large gains versus unrestricted, inverse-mask, self and construct=7. Total53.
Pilot/smoke N4/N32 have31 point endpoints; confidence still uses53 denominator.

N4 lower gains>=-1% except localMAPPO lower>0. Large lower gains versus
unrestricted/inverse-mask/self/construct/reference>=-1%, independent lower>0,
deadline lower>=-2%; large/N4 cost/machine upper ratio<=1.10. All FOUR PRIMARY
pooled large gains (commitment vs unrestricted, commitment vs inverse-mask,
bilateral vs self, repair vs construct) require lower>=.005. Optional PPO
superiority requires lower>=.01 versus BOTH original and normalized PPO on fresh
test seeds. No small-quality or health threshold is relaxed after the v1 result.

Allsize availability difference lower>=-.01, pending/machine upper difference
<=+.025, failures/machine upper difference<=+.25; references BOTH PPO variants
at N4, deadline large. Lower cost cannot compensate for failed health. Same
numeric point screens onDEV. Zero invalid assignments; candidate actor-select
p95<=50ms after training workers exit, excludes observation/physical stepping/
diagnostic probes, no network-latency claim. Failure/conflict/availability/backlog
all reported. Record training and evaluation bilateral exchanges, self revisions,
drops and proposal arbitration, and distinguish learning from runtime activity;
do not claim runtime exchange benefits if selected policies never exchange.

## Validation and artifacts

Before pilot: verify swap/transfer/release legality and immutable active services;
mask equivalence between independent Python broker and tensor policy; partial
trace support/mass and marginalized matching probabilities at N4; initial21
physical-support equality in all coordinated arms with duplicate latent traces
explicitly summed, forced tokens; no future inputs; both passes preserve machine
permutation; exact sample/replay/gradients/joint ratio; scalar critic independent
of every latent choice and called once; core initialization/counts equal; no
extra environment rewards/ticks/optimizer steps; independent arbitration;
v2-reference and all PPO action tags/routing; serial/spawn checkpoint identity;
53 complete/null/missing-cell/bootstrap-pool/health-reference tests; prerequisite
source/config/registry/budget corruption rejection; independent60-tick replay.

Separate log/value replay tolerances: logatol2e-5; physical valueatolU*2e-5 plus
rtol1e-6 (literal PPOatol2e-5). Same dimensionless float precision. Log raw/
normalized returns and values, actual actor/critic gradients and updates, value
envelopes/saturation descriptively; no implied calibration/variance guarantee.

Also test ordered prefix immutability, no net repair cancellation, full
construction+KEEP matching support, the immediate-inverse mask including
transfers and reset after nonbilateral slots, exact history-aware sample/replay
probabilities and scalar/tensor broker agreement for every reachable small
prefix. All six core parameter counts must remain19255. Log net provisional
plan changes and repair-with-unchanged-plan counts (matching for coordinated
arms, proposals for independent), alongside existing broker diagnostics.

Before training starts, jobs are scheduled by descending machine count, then
core arms before the separate completion reference and small baselines, with
registered arm order/train-seed order resolving ties. This only controls worker
scheduling; each fit retains independent explicit RNG tapes and the exact
budget. Processes remain8 and torchthreads1. Actor latency is isolated after
the entire training pool exits. The48h hard limit is not extended for overhead.

Run Linux tests and smoke first, commit/push verified code, then show and execute
exact fresh pilot command directly on this lab under user override. NoSSH/automatic
transfer. Seal source/PLAN/registry/history/physics/config; save all zero/learned
checkpoints, digests/counts/from-scratch audits, all training/DEV/test journals,
latent traces, final starts, proposal/exchange diagnostics, selections, cost/health/
CIs/latencies and independent replay. Timestamped copied-script/hash reports,
visual QA of scientific plots, manual rsync -avhP --partial at results delivery.
