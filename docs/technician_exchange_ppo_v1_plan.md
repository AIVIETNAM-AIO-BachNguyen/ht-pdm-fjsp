# Technician Exchange PPO v1: prospective research protocol

Registered before any new learning outcome. Preserve negative scarcity v1/v2/v3
and completion-credit v1/v2; none authorizes development or confirmation. No
teacher, demonstration, old learned tensors, distillation, PPO warm start, extra
simulator branch or future-event input. All fits initialize randomly.

## Hypothesis and prior-art boundary

V2 diagnosis found failures after resources were assigned elsewhere; shorter
service alone did not explain the strong normalized PPO's protection of at-risk
machines. Test whether bounded bilateral exchanges of provisional assignments
improve coupled allocation over equally budgeted self-only repairs. This is a
new domain allocation mechanism hypothesis, not a demonstrated causal diagnosis.

Learned local improvement is established: [NeuRewriter](https://arxiv.org/abs/1810.00337)
factorizes region/rule choices with actor-critic; [Wu et al.](https://arxiv.org/abs/1912.05784)
learn routing improvement policies. [Online bipartite matching RL](https://arxiv.org/abs/2109.10380)
and [two-sided matching with GNNs](https://proceedings.mlr.press/v222/begnardi24a.html)
already exist. [MAT](https://arxiv.org/abs/2205.14953) and
[POAD](https://arxiv.org/abs/2405.15821) precede sequential decisions and decomposed
PPO; normalization is also prior art. Do not claim generic exchanges, local
rewriting, graphs, PPO, prefix values, normalization, or a new trust-region theorem.
Our provisional contribution is bounded pre-service bilateral revision, exact
latent trace likelihood training, and empirical cost/health/scale evidence in
heterogeneous maintenance. Novelty remains provisional after this focused review.

## Exact internal action process

Each physical decision uses2K discrete internal slots, fixed technician order.
First K slots construct a provisional matching as in canonical technician
reservation. Token0=WAIT; tokens1..N request the corresponding legal urgent
machine; tokenN+1(DROP) is forbidden during construction. Reserve chosen machines.
Busy technician has only token0. No physical time, reward or restoration advances.

Next K slots revisit the technicians in the same order. Token0=KEEP, tokenN+1
releases that technician's provisional assignment (only if nonempty). Choosing
its current machine is forbidden as duplicate KEEP. Choosing an unclaimed legal
machine replaces its current assignment. In the candidate, choosing a machine
held by another technician swaps their provisional jobs; if the requesting
technician has no job, the former holder becomes unassigned. A swap is allowed
only when the reciprocal job is legal for its recipient. All intermediate plans
are valid matchings; already active physical services are never moved or canceled.
Apply final starts simultaneously after all2K slots. No search-until-convergence,
objective oracle, best-of-samples or extra physical rollouts.

Different valid traces may produce the same final matching. Optimize the sampled
**latent trace** with exact conditional log probabilities; never treat one trace's
probability as the marginalized physical matching probability. All feasible
physical matchings remain supported through construction followed by KEEP.
There are21 at N4/K2; enumeration is allowed in small correctness tests only.

Core arms, identical architecture/tensors/current information/2K slots/optimizer
counts/physical training budgets:

- exchange_ppo: construction with reservations, then self moves and bilateral
  swaps/transfers, shared provisional assignment messages.
- self_repair_ppo: same2K computations, inputs and self moves/KEEP/DROP, but
  cannot change another technician's assignment. Main exchange ablation; controls
  for a second pass and extra opportunity to revise a decision.
- construct_only_ppo: same NN calls in2K slots; repair support restricted to
  KEEP. All repair logits are computed before masking. This contrast includes
  the effect of having an actionable second pass, not only bilateral exchange.
- independent_ppo:2K individual choices, no others' intention/reservation
  messages or cross moves. Each stage may use its own current proposal and full
  current physical sensing. Other agents' proposals do not affect its masks or
  actor features. Final machine conflicts are resolved by existing quality
  arbitration, losers dropped without retry. Critic still sees the same physical
  graph as all core arms.

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

Smoke: N4/N32,14fits,one replica384decisions,vector2/rollout192/epochs2/mb48,
CP0/192/384,DEV2/engineering-test2. Pilot: N4/N32,28fits,two replicas61440,
vector8/rollout768/epochs4/mb128,CP0/30720/61440,DEV16,NO scientific tests.
FullDEV72fits,all sizes,3rep/fullbudget/DEV32,no tests.
Confirmation120fits,all sizes,5rep/fullbudget/DEV64/test256.

Registry120000000..121300000 groups fixes all train/DEV/test/analysis panels,
387 actual raw seeds mutually/historically disjoint, including all unopened
previous confirmations. Audit all derived roles. Core arms paired initial/action/
minibatch roles; v2-reference has its own initialization/learning role; physical
tapes paired across every method per size. Sizes independently namespaced.

Pilot -> fullDEV only after audited COMPLETE/testclosed/exact settings and seals,
EVERY nonpooled cost/health point screen PASS, and at least one selected trained
candidate replica at BOTH sizes; all replicas remain in means. Report both
N32-only pooled mechanism proxies even if negative: they may permit exploratory
fullDEV only because they do not measure the registered three-size pool.
FullDEV must pass EVERY point screen before confirmation, with identical frozen
source/PLAN/registry/core settings and selections sealed before opening test.
No test reuse, teacher rescue, best-seed exclusion, health relaxation or budget
extension. Existing failed protocols remain failed and cannot authorize this run.

## Scientific family, practical margins and system bounds

Round2 retains alpha=.05/6, no teacher-free confirmation has opened; previous
teacher round1 spent.025. Bootstrap200000,43Bonferroni endpoints, two-sided
confidence1-(.05/6)/43. Common replica resampling weights across sizes,
independent episode resampling by size, paired methods within size; pool large
gains inside every draw. Five replicas give approximate coverage; publish all
replicas and leave-one-out diagnostics.

N4 eight gains vs self/construct/independent/v2-reference/originalPPO/normalizedPPO/
localMAPPO/fullMAPPO plus six health differences (BOTH original and normalized
PPO references)=14. Large five gains vs self/construct/independent/v2-reference/
deadline plus3health=8*3=24. Three large/N4 cost/machine ratios and two pooled
large gains vs self-repair and construction-only=5. Total43. Two health references
at N4 strengthen the previous bound; neither is removed after outcomes.

N4 lower gains>=-1% except localMAPPO lower>0. Large lower gains vs self/construct/
v2-reference>=-1%, independent lower>0, deadline lower>=-2%; large/N4 cost per
machine upper ratio<=1.10. BOTH PRIMARY pooled large gains vs self-repair and
construction-only lower>=.005. Optional PPO superiority requires lower>=.01 vs
BOTH original and normalized PPO on fresh test seeds. V2-reference noninferiority
plus health gates is required, not an unsupported prefix/calendar claim.

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
43 complete/null/missing-cell/bootstrap-pool/health-reference tests; prerequisite
source/config/registry/budget corruption rejection; independent60-tick replay.

Separate log/value replay tolerances: logatol2e-5; physical valueatolU*2e-5 plus
rtol1e-6 (literal PPOatol2e-5). Same dimensionless float precision. Log raw/
normalized returns and values, actual actor/critic gradients and updates, value
envelopes/saturation descriptively; no implied calibration/variance guarantee.

Run Linux tests and smoke first, commit/push verified code, then show and execute
exact fresh pilot command directly on this lab under user override. NoSSH/automatic
transfer. Seal source/PLAN/registry/history/physics/config; save all zero/learned
checkpoints, digests/counts/from-scratch audits, all training/DEV/test journals,
latent traces, final starts, proposal/exchange diagnostics, selections, cost/health/
CIs/latencies and independent replay. Timestamped copied-script/hash reports,
visual QA of scientific plots, manual rsync -avhP --partial at results delivery.
