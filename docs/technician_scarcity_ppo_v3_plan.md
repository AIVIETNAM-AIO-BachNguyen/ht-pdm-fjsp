# Cardinality-normalized opportunity PPO v3: prospective protocol

Locked before any v3 smoke/pilot/development outcome. V2 pilot and diagnostics
are preserved negative development evidence, not scientific tests. Retain round2
alpha0.05/6. Direct Ubuntu lab execution authorized; tests/smoke/commit/push before
pilot. No teacher, distillation, demonstrations, old checkpoint or warm start.

## Falsifiable hypothesis and main algorithmic mechanism

V2 learned flat edge/STOP policies deteriorated on small settings; all small
priced replicas selected random step0. An already opened DEV diagnostic found
greedy STOP with aggregate GO mass above one half. This is an association, not
a causal explanation. Test a structural change to conditional action probabilities
and execution, holding the opportunity messages and other learning settings fixed.

At each broker slot compute a GO/STOP decision independently of the number of
legal machine-technician pairs, then choose an edge conditional on GO:

    s = learned_stop(global, prefix, remaining_resources) + mean(WAIT) - 2
    p(STOP) = sigmoid(s)
    p(edge mj) = sigmoid(-s) * softmax_legal(bid_mj - price_mj)

The fixed -2 STOP offset initializes a mild GO prior (about0.88), not a pretrained
policy; it is COMMON to both hierarchical and flat broker controls. No bias from
learned old weights, teacher labels or future events. No resources or a previous
STOP forces a probability-one STOP suffix. Train exact joint trace likelihood
with joint PPO ratios and full conditional entropy/K. Greedy execution follows
the hierarchy: STOP if s>=0, otherwise best legal edge. It is the composition of
conditional greedy decisions, not a claim of globally optimal/MAP matching.

Retain v2 bounded duration-weighted exclude-own opportunity messages, tau0.2,
max0.5*sigmoid(gate), gate random-initialized0; no price oracle or optimization
solver in learned decisions. Prefix/suffix exclusive partitions O(NK) per slot,
O(NK²) per whole decision, constant19781 learned parameters for all sizes.
Raw22 inputs and H32 graph unchanged, no added identity or information. Shared
full current graph and broker, not local-only/network deployment.

Four arms, identical random tensors, parameter counts, physical tapes, learner
RNG roles, budgets and separate actor/value clipping at0.5:

- scarcity_ppo: hierarchical GO/STOP, opportunity messages, reservations.
- reservation_ppo: identical hierarchy/broker, messages disabled.
- independent_ppo: same v2 independent machine/WAIT proposals and arbitration,
  no reservations/messages. No broker STOP gate; allocated unused parameters kept.
- flat_stop_ppo: SAME messages/broker/STOP offset and clipping, but flat softmax
  over STOP and legal edges plus original flat greedy execution. Isolates the
  probability/execution factorization. No global-clipping arm in v3; v2 clipping
  findings preserved, no clipping improvement claim made.

Same-weight unpriced probes preserve each arm's factorization and identity,
change only price computation, restore actual diagnostics, occur outside timing,
and never expose another physical trajectory to training.

Generic hierarchical policies, autoregressive MARL/PPO and learned matching are
prior art (MAT https://arxiv.org/abs/2205.14953;
HAPPO https://arxiv.org/abs/2109.11251;
matching https://arxiv.org/abs/2109.10380).
Novelty remains provisional: coordinated maintenance allocation combining
cardinality-normalized conditional execution and duration-aware opportunity
messages with controlled scaling/health evidence. No new PPO theorem, optimal
price, monotonic improvement or incentive compatibility claim.

## Physics, budgets, splits, stopping and confirmation prerequisites

Unchanged balanced maintenance N4/8/16/32,K=N/2,H48+common deadline tail12,
reward=-cost/20,gamma1. N4 fresh original PPO/localMAPPO/full-informationMAPPO;
PPO18966/local15364/full18052 parameters, graph19781. Baseline capacity
differences disclosed. No matching catalogue for large learned policies or
min-cost-flow deadline. Sizes train separately from scratch, no zero-shot claim.

All full fits491520 joint decisions,vector16,rollout3072,epochs4,minibatch256,
lr3e-4,clip0.2,entropy0.01,value0.5,gradclip0.5; value output/loss normalization
N/4. Checkpoints0/122880/245760/368640/491520. Whole budgets spent, mean DEV
cost chooses earliest tie, every replica retained. No metric early stopping.
Training processes8, torch threads1, tqdm. Per-run hard stop24h; do not extend or
count incomplete runs as success. Fresh timestamped artifacts, no overwrite.

Smoke11fits,N4/N32,1seed384steps,vector2/rollout192/epochs2/mb48,
dev2/engineeringtest2. Pilot22fits,N4/N32,2seeds61440steps,
vector8/rollout768/epochs4/mb128, checkpoints0/30720/61440,dev16, NOtest.
DEV57fits,all sizes,3seeds/full budgets/dev32, NOtest.
Confirmation95fits,all sizes,5seeds/full budgets/dev64/test256.

Fresh panels in configs/technician_scarcity_ppo_v3_seed_registry.json:
smoke68000000/dev68100000..1/engineeringtest68200000..1/analysis68300000;
pilot68400000..1/dev68500000..15/analysis68600000;
DEV68700000..2/dev68800000..31/analysis68900000;
confirmation69000000..4/dev69100000..63/test69200000..255/analysis69300000.
Audit raw and derived roles against history, using separate physical/learner
hash functions. Pair graph initialization/RNG and every method's physical tapes
by size, independent size namespaces. No reuse of a trained policy.

Confirmation requires completed independently audited, passing, test-closed
full-budget DEV with identical source/protocol/registry/core settings; sealed
checkpoint choices before opening tests. Pilot alone cannot authorize confirmation.

## Thirty-five endpoints and unchanged physical quality limits

Round2 alpha0.008333333333333333, bootstrap200000. Common replica weights
across sizes, independently resampled episode axes by size; methods paired
within size. Bonferroni35, individual two-sided confidence1-alpha/35. Five
replicas: approximate coverage only, report every replica and leave-one-out.

N4 gains vs PPO/localMAPPO/fullMAPPO/reservation/independent/flatSTOP(6)
and three health differences vs PPO(3):9. Each large gains vs reservation/
independent/flatSTOP/deadline(4) plus three health differences(3):21.
Three large/N4 cost-per-machine ratios; pooled price gain vs reservation and
pooled cardinality gain vs flatSTOP, arithmetic means of three large gains
computed within every bootstrap draw(2): total35.

REQUIRED N4 lower gains>=-1%, except localMAPPO lower>0. At each large size,
reservation/flatSTOP lower>=-1%, independent lower>0, deadline lower>=-2%,
cost-ratio upper<=1.10. REQUIRED MAIN pooled price lower>=0.5% AND pooled
cardinality lower>=0.5%. Positive independent comparison alone is insufficient.
Optional PPO superiority lower>=1%, positive N4 price effect lower>0.
No relaxed cost or health bound relative to prior versions; second main mechanism
endpoint is stronger than v2's optional clipping improvement.

All sizes availability difference lower>=-0.01, pending/machine upper<=0.025,
failure/machine upper<=0.25 vs PPOsmall/deadlinelarge. Zero invalid assignments;
neural select p95<=50ms after training pool shuts down. Timing excludes feature
construction, physical stepping and descriptive probes; no communication claim.
DEV screen uses the same point thresholds, including both pooled gains>=0.5%.
No threshold changes, best-seed selection, opened-test reuse or teacher fallback.

## Validation and artifact contract

Test hierarchical probability mass and exact replay, flat control unchanged
given fixed STOP offset, GO/STOP cardinality invariance to duplicated alternatives,
greedy conditional threshold vs flat mode discrepancy, canonical STOP/empty masks,
all21 physical small matchings through33 legal traces, finite price/gate gradients,
permutation equivariance, identical initialization/capacity, no future inputs or
large catalogue, checkpoint provenance, serial/spawn determinism, same-weight
probe isolation, full35-endpoint/null/missing-grid statistics and prerequisite
corruption rejection. Full suite plus fresh Linux smoke before commit/push/pilot.

Manifest/config/source/registry/physics snapshots and input/output hashes;
zero/learned checkpoints, initialization/parameter counts, exact train/dev/test
journals, sealed selections, independent physical60-tick/coordination replay,
health/cost/latency/bootstrap summaries. Log raw gradients/actual update norms,
price activity and same-weight unpriced matching changes descriptively. Reports
in fresh directories with copied scripts and hashes; visually verify plots.
Preserve all negative versions. No SSH or automatic transfer; provide rsync -avhP
--partial for the local Mac when results are delivered.
