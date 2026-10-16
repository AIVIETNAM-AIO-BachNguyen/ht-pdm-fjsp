# Physical-marginal matching PPO v1: prospective N4/K2 protocol

Registered before implementation and any learning outcome. Follow the user's
[small-first sequence](technician_small_first_research_strategy.md): all learning,
DEV and confirmation here use four machines and two technicians only. No larger
training or generalization run is authorized by this protocol. Preserve every
negative scarcity/completion/exchange result and their unopened scientific panels.
All new policies initialize randomly; no teacher, old learned tensors, labels,
imitation, distillation, warm start, simulator fork or future-event input.

Active user override on 2026-10-16, before any learning run: execute the FULL
N4/K2 experiment directly on this lab and omit a separate smoke run. This
supersedes the smoke-before-full execution instructions below for this round.
Keep the unit/integration correctness checks, prospective budgets/panels/gates,
verified commit/push, shown command and fresh artifacts. The smoke profile can
remain implemented but is not a launch prerequisite. FULL here means the
registered full-budget development grid; confirmation still requires passing
development and sealed checkpoint choices before opening scientific tests.

## Hypothesis, contribution and prior-art boundary

Exchange v2's cancellation invariant holds but its N4 cost/health screens fail.
The diagnostic finds different repair activity with similarly poor returns and
does not establish that cancellation or service duration causes the gap. Test a
different learning representation: two canonical reservation orders, with PPO
ratios on the probability of the final physical matching rather than the sampled
latent order. Prefix completion conditioning remains a separate tested mechanism.
This is a hypothesis about learning useful joint allocations from original
long-horizon returns, not a diagnosis proved by the earlier trajectories.

Marginal policy gradients and variance reduction under deterministic action maps
are established: [Eisenach et al.](https://arxiv.org/abs/1806.05134). Sequential
allocation and learning order also have prior art; see the
[ordering matrix](technician_coordination_ordering_prior_art.md). In particular,
[MACSIM](https://arxiv.org/pdf/2510.12273) uses a permutation-invariant set loss
with pseudo-expert self-improvement, while [DPN](https://arxiv.org/abs/2405.17272)
uses agent-order symmetries in a reinforcement-learning baseline. Our proposed
implementation uses ordinary on-policy PPO from scratch with an exact bounded
order mixture. Do not claim a new Rao-Blackwell theorem, generic permutation
invariance, first use of set outcomes, monotonic improvement, or superiority to
unimplemented papers. The provisional contribution is the maintenance matching
policy/likelihood procedure and evidence from its matched controls.

## Canonical construction and exact physical probability

At each physical state s, define two orders: ascending technician index and its
reverse. Choose a latent order o uniformly (probability 1/2). Visit each technician
once in that order and sample machine-or-WAIT: token0 is WAIT; tokens1..N select
legal urgent machines. Reserve selected machines for remaining slots. Busy or
unsupported technicians have forced WAIT. WAIT does not stop later technicians.
There is no repair, DROP, internal physical transition or intermediate reward.
Apply all final starts simultaneously; existing services are never changed.

Represent the physical matching by a vector a indexed by physical technician,
not by construction position. For each fixed order, every feasible a has exactly
one canonical trace. Let p_o(a|s) be its product of legal conditionals. The mixed
physical probability is therefore exactly

    q(a|s) = (p_0(a|s) + p_1(a|s))/2.

Use logsumexp(log p_0, log p_1)-log2. No physical matching catalogue or sum over
all feasible matchings is used by the candidate. Busy and all-WAIT cases remain
normalized. At K2 these are all technician orders; this fact must not be extended
to arbitrary K, where the same fixed two-order mixture is only a mixture of two
orders. No larger-size experiment occurs here.

For candidate PPO, store the matching, sampled order and old log q. Reconstruct
both canonical prefixes during replay and use r=q_new(a|s)/q_old(a|s). The primary
control instead stores/replays log(0.5*p_o(a|s)) for the sampled order and clips
that joint latent ratio. Same random order distribution, physical policy class,
initial parameters, current information, critic and deployment in both arms.
For fixed s and a, grad log q equals the posterior-over-orders average of
grad log p_o. This elementary identity does not guarantee lower variance or
better returns for finite-batch normalized/clipped PPO. Clipping is not a hard
KL constraint and offers no new monotonicity guarantee.

## Five matched core arms

- `marginal_matching_ppo`: two-order mixture, physical q ratio, prefix calendar.
- `latent_order_ppo`: identical two-order sampling/policy, sampled-order joint
  ratio; isolates the learning likelihood and clipping unit.
- `forward_order_ppo`: forced ascending sampling order and physical probability
  p_0. Still compute both recorded order paths and the common regularizer.
- `reverse_order_ppo`: forced reverse sampling order, physical probability p_1;
  same otherwise. The fixed-order controls disclose any favorable ordering.
- `no_prefix_ppo`: two-order physical q ratio; actor calendar/assignment features
  use the original physical state throughout construction. Retain reservations
  in legal masks and the common physical-state critic. This removes completion
  messages from prior provisional starts, not feasibility enforcement or raw
  physical sensing. It is not an entirely uncoordinated policy.

All five share architecture, parameter count, initialization tensors, action and
minibatch RNG roles, physical tapes, optimizer settings, number of neural calls
and physical budgets. Fixed-order arms draw and discard the same latent-order
random token before forcing their registered order. No best-seed exclusion.
The local/full MAPPO baselines supply external uncoordinated comparisons;
winning against local MAPPO alone is insufficient when information differs.

## Network, optimizer and deployment

Use unchanged raw M6/T3/E4/G9 observations and raw pair22, N4 physical state71.
Shared graph/broker assumed; no network-local execution claim. Hidden27; every
MLP has two tanh hidden layers and a linear output. Actor pair encoder22->H and
calendar4->H; calendar is active service, duration/12, known completion age/8,
and assigned-edge flag, from current services or previously proposed starts.
Bid head4H->1 consumes pair, machine mean, technician mean and graph mean.
WAIT head(2H+1)->1 consumes technician mean, graph mean and position/K in the
chosen order. There is no technician-index embedding or repair head.

Separate state-only critic: pair22/calendar4 encoders, tanh fusion, technician
and graph means; context(2H+1)->H uses these means and physical free flag; mean
contexts; value head(H+9)->1. No proposed assignment, order, stage or sampled
action enters the critic. Evaluate it once per physical decision. Expected
18,390 parameters per core arm (9,803 actor, 8,587 critic); verify before learning.
The stochastic mixed policy is technician-swap equivariant at K2; deterministic
argmax tie-breaking is not claimed equivariant. Machine permutation likelihood
equivariance and physical critic invariance must be tested.

Ordinary scalar clipped PPO, gamma1, original reward=-cost/20, complete MC return
with the unchanged common 12-tick deadline tail after 48 physical decisions.
Advantage is MCreturn minus old physical-state value, standardized once over the
rollout batch. Value output U*head and MSE(V/U,G/U), with U=12*4*48/20=115.2.
Separate actor/critic clipping at0.5. No reward shaping, duration penalty,
auxiliary teacher, value target branch or extra physical sample.

Common core regularizer: mean conditional entropy over both recorded canonical
paths (2K conditionals), evaluated at the old matching's prefixes. Forced-token
entropy is zero. This is a recorded-prefix regularizer, not exact physical q
entropy or its exact gradient. Every core arm uses it with coefficient0.01;
the primary likelihood comparison changes no regularizer. PPO clip0.2, value
weight0.5, Adam learning rate3e-4, four epochs and minibatch256 at full budget.

Sampling computes K autoregressive choices, then replays both order paths for
the recorded matching: 3K actor slot evaluations, same for every core arm.
Optimizer replay computes2K. No forced slot bypasses its neural evaluation.
For deterministic deployment, construct one greedy plan per order, score each
under both order conditionals, then choose the plan of highest registered
physical probability (q, p_0 or p_1 depending on arm). Ties select the first
proposal. This is a fixed two-proposal neural decoder, not a globally optimal
MAP matching, cost oracle or simulator lookahead. All core arms spend the same
6K actor slot evaluations; the critic is skipped. Candidate and latent-order
control have identical deployment for identical tensors.

Core work is polynomial, conservatively O(NK²) with a fixed number of order
passes; no complete-matching enumeration. Extra internal compute compared to
PPO/MAPPO is disclosed. Do not infer scalability from N4 or label independent
larger-size training as zero-shot transfer.

External N4 baselines: literal original `ppo`, `normalized_ppo` with U4 critic
units, normalized local `mappo`, normalized `full_mappo` (raw71+machine identity4).
Keep their original architectures/surrogates and entropy0.01: 18,966 PPO,
15,364 local MAPPO, 18,052 full MAPPO parameters. Original/normalized PPO pair raw
initial tensors and RNG roles. No old checkpoint is loaded. Baseline compute
and architecture differences are not a matched mechanism ablation.

## Budgets, fresh panels and stopping

All full fits491520 joint physical decisions, vector16, rollout3072, epochs4,
minibatch256; checkpoints0/122880/245760/368640/491520. Train every fit to its
whole budget, then choose earliest minimum mean DEV-cost checkpoint separately
per replica. All replicas remain included, including a selected step0.
Processes8, torch threads1, tqdm during training and multiseed evaluation.
Hard12h for the whole grid, no extension or outcome stopping. First technical
failure aborts siblings and preserves the incomplete artifact as a failure.

There is no short quality pilot: small-setting full-budget DEV follows engineering
tests/smoke, avoiding a cheap-training result being treated as final quality.

- Smoke:9 methods, one seed,384 decisions, vector2/rollout192/epochs2/minibatch48,
  checkpoints0/192/384, DEV2, engineering-test2. Exactly72 training episodes,
  54 DEV plus2 deadline episodes,20 engineering-test episodes/1200 physical ticks.
  Classification ENGINEERING_ONLY; never scientific success or test alpha spend.
- Development:27 fits (9 methods x3 seeds), full budgets, DEV32, no tests.
  13,271,040 decisions;276,480 training episodes;4,320 DEV plus32 deadline episodes.
- Confirmation:45 fresh full-budget fits, DEV64, scientific test256. Select and
  seal all checkpoint choices before opening the scientific panel. No teacher
  rescue or extra environment/optimizer steps. Development alone is not success.
  Exactly22,118,400 training decisions/460,800 training episodes;14,400 DEV plus64
  deadline episodes;11,520 learned-method test plus256 deadline test episodes.

Registry `configs/technician_marginal_matching_ppo_v1_seed_registry.json` fixes:
smoke train200000000/dev200100000..1/engineering-test200200000..1/analysis200300000;
DEV train200400000..2/dev200500000..31/analysis200600000;
confirmation train200700000..4/dev200800000..63/test200900000..255/analysis201000000.
All368 raw seeds must be mutually and historically disjoint, including unopened
old test panels. Audit unique new derived initialization/action/minibatch/physical
roles and exclude all raw historical/new seeds. Separate VERSION namespaces;
do not claim exhaustive historical-derived collision checks unless performed.
Pair physical training/evaluation tapes across methods and pair core tensor/RNG
roles; all other comparisons disclose their architecture differences.

## Fourteen endpoints and N4 success gates

Preserve unused scientific round2 alpha=.05/6; no scientific test has been opened
in previous teacher-free rounds. Family14: eight candidate cost gains against
the other methods, plus three health differences against each PPO baseline.
For gain use1-candidate_mean_cost/reference_mean_cost; health differences are
candidate-reference, with pending/failures divided by4. Simultaneous confidence
level1-(.05/6)/14; paired hierarchical bootstrap200000 draws resamples training
replicas and episode seeds, maintaining method pairing. Disclose the approximate
coverage with five training replicas and provide every replica and leave-one-out
sensitivities. No per-seed winner selection, new test tuning or unspent-alpha reset.

Required gain lower bounds: >=0.005 versus BOTH latent_order_ppo and no_prefix_ppo;
>=-0.01 versus forward, reverse, original PPO and normalized PPO; >0 versus BOTH
local MAPPO and full-information MAPPO. Optional PPO superiority requires >=+0.01
against BOTH original and normalized PPO using those same registered intervals.
Optional bidirectional-order improvement requires >=+0.005 versus both fixed
orders; do not claim it merely from noninferiority.

Required health bounds against BOTH PPO baselines: availability lower>=-0.01,
pending/machine upper<=+0.025, failures/machine upper<=+0.25. Zero invalid starts;
candidate actor p95<=50ms measured isolated after the training pool, excluding
observation/physics/verification; no network-latency claim. Cost gains cannot
override health failures.

For isolated timing without extra physical trajectories, retain the 48 observed
states from the first DEV seed at every candidate checkpoint. After the whole
pool closes and checkpoint choices are fixed, load only the selected states,
warm up once on each state and time three select calls on each. This supplies
144/432/720 measured calls for smoke/DEV/confirmation. Store the observation
arrays and selected-state identities with output hashes. No timing result selects
a checkpoint or enters learning, and repeated actor calls do not advance physics.

Development uses the corresponding14 point gates. Confirmation is authorized
only by completed independently audited, test-closed DEV with identical frozen
source/PLAN/registry/full learning settings, all14 point gates and latency passed,
and at least one selected trained candidate replica; every replica stays included.
Confirmation must pass all confidence and health/mechanism/latency gates to count
as N4 success. Only then register larger-size or generalization research separately.

## Engineering and artifacts

Before full DEV: verify scalar/tensor legality and physical feasibility; normalization
and support on all21 dense N4 matchings in unit tests only; exact two-component
marginal likelihood; sample frequencies; sample/replay log and scalar-value
agreement; posterior-gradient identity; primary controls' matching initial physical
distributions/deployment; forced slots; no-prefix calendar removal; permutation
properties with tie limitations; action order ignored by physical transitions;
parameter counts; no critic action/order input; reward/budget/tape/optimizer
accounting; exact14 endpoints and prerequisite rejection of changed/incomplete
artifacts. Use numerical tolerances log atol2e-5/rtol0 and normalized value
atol2e-5/rtol1e-6. Do not silently relax a failed replay tolerance.

Run focused correctness/integration tests and relevant regression checks, then
commit/push verified code. Under the 2026-10-16 user override, omit a separate
smoke experiment and launch FULL development directly. Show the exact compound
command after push, then execute here. Independently verify output/source seals,
budget/tape accounting and available physical replay before accepting full results.
Never SSH or initiate artifact transfer. Fresh UTC output directory per run;
record git identity/dirty state, resolved config, registration/source snapshots,
raw/derived seed audit, parameter counts/initial digests, journals of training
progress/episodes and DEV/test/decisions/coordination, all checkpoints and sealed
choices, audit, summary and final input/output SHA/size bindings. Preserve failed
and interrupted artifacts. Provide the manual partial-preserving/progress rsync
command after a full run. No success claim from tests, smoke or DEV alone.
