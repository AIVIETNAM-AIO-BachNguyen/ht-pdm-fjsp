# C1 diverse maintenance panel: prospective protocol

Locked before any C1 outcome. B2 relational-prefix development and A1
teacher-transfer development may still be running; neither result is used to
choose these C1 configurations. C1 tests the fixed B2 graph/coordinator,
including a separately trained mask-only control. A negative B2 result will
remain negative even if C1 later behaves differently. V4 physics and all old
seed panels remain historical.

## Falsifiable hypothesis and families

The candidate-conditioned selected-edge message improves assignments beyond
mask-only when skill contention and persistent maintenance demand vary, while
retaining cost and health near a DEV-selected strong rule. Use the unmodified
`WaitingEnv` transition and cost contract of the audited shared-technician
factorial benchmark: H=48 policy ticks, 12 common cleanup ticks,
failure_age=4, failure_probability=.45, max_age=8, preventive/corrective
costs 1/2, unavailable/failure costs 6/15, waiting_limit=4 and waiting_price=12.
The only configuration changes are registered skill edges, K and duration:

| Family | N/K configurations | Skill and service matrix |
|---|---|---|
| `skill_bottleneck` | 8/3, 16/6 | Technician `j` serves machine `m` iff `m%3` equals `j%3` or `(j+1)%3`; every compatible duration is 1 tick, restoration .75. |
| `long_service` | 8/4, 16/8 | Technician `j` serves machine `m` iff `j%4` is `m%4` or `(m+1)%4`; duration is 2 if `j%4 == m%4`, otherwise 4; restoration .75. |

All four configurations have compatible edges for every machine and worker.
The first family changes K/N and skill topology from V4. The second has
multi-tick nonpreemptive occupation and a persistent request pool. Both use
the **same** request and physics semantics so family differences are only
the registered matrices/capacity. To avoid ambiguous unrequested preventive
jobs, request creation is fixed at failed or age >=3; a request remains until
assigned, and an unassigned new failure enters the pool after the physical
step. The teacher and all learned/heuristic arms face identical requests,
shock tape, tail FCFS policy and terminal liability. No policy controls
request timing. This tests allocation, not learned request timing.

## Fixed observation adapter and arms

`WaitingEnv.observation` supplies the B2 24-input edge contract. For a
machine with a pending request, replace the machine feature at index 2 by
`(tick - request_since + 1)/12`; otherwise set it to zero. This encodes
healthy pending work in the existing slot, without changing tensor shapes or
weights. Pair feasibility is `eligible_edges AND machine in pending_pool`.
The current machine/technician/edge/global features, IDs and full graph are
otherwise unchanged. No future shock, request arrival or private oracle is
exposed. Before any outcome, tests must show each arm sees the same
observation and legal set, no duplicate starts in full/mask-only, request
conservation, busy-resource exclusion, service completion, per-tick cost and
tail/liability reconciliation. Report the adapter's zero-shot distribution
shift explicitly.

Four graph arms per configuration: relational full and mask-only both warm
from the **same** N4 central teacher bank and trained with equal 491,520 PPO
decisions/updates; no-reservation/no-prefix receives the same warm teacher
and budget; full scratch starts from matched random tensors and receives the
same PPO budget without teacher bank or pretraining. Warm N4 policy before
target PPO is evaluated separately as zero-shot. No-reservation collision
arbitration is deterministic and counted; full/mask-only use the same legal
set and order. The teacher is a source of extra information and compute.
Report foundation PPO, bank generation, 32 pretrain epochs, target PPO,
decisions/updates, wall time, memory and inference separately. The N4 bank
is shared across configurations for a student bundle; count reuse once and
also report cost per deployed configuration if amortized.

Two deterministic heuristics, skill-feasible FCFS and risk priority with
shortest-duration worker choice, are exact implementations of the audited
shared-technician definitions. Run both on all DEV configurations/seeds.
Select **one** global reference by lowest equal-configuration mean DEV
cost/machine; earliest tie FCFS. Freeze the choice before TEST. Report both
on TEST. This reference receives the same fixed request timing and physics.

## Panels, estimands and gates

Fresh disjoint raw seeds and role-derived streams are in the companion
registry. Engineering smoke: one bundle, N8 in both families, 192 N4 bank
states, two pretrain epochs, 384 PPO decisions, two DEV and four engineering
TEST seeds. Scientific development: three bundles, four configurations,
32 new DEV physical seeds per configuration and candidate checkpoint.
Eligible checkpoints 0, 122,880, 245,760, 368,640, 491,520; select lowest
mean DEV extended cost/machine within each arm/configuration/replica,
earliest tie; train the full budget regardless. DEV gate: pooled full vs
mask-only gain >=0.5%, neither family point gain <-1%, full vs globally
DEV-selected FCFS/risk point gain >=-1%, zero invalid assignments and full
health point guards against matched mask-only: availability >=-0.01,
pending/machine <=+0.025, failures/machine <=+0.25 per family. TEST stays
sealed if any guard fails.

If DEV passes, confirmation uses five new training bundles, 64 new DEV
physical seeds for checkpoint selection and 256 sealed TEST seeds per
configuration. No architecture, cost, physics, rule, checkpoint, analysis or
reference change after the gate. Global hardstop 72 h; timeout is INCOMPLETE.

For each arm/configuration, mean extended cost/machine first over paired
TEST seeds and training bundles. Pool by **equal weight across the four
configurations**; family means equally average their two configurations.
`gain(X,Y) = 1 - pooled_mean_cost(X)/pooled_mean_cost(Y)`. Primary full vs
mask-only gain must be >=1% with one-sided multiplicity-adjusted lower CI >0.
Each family full/mask-only point gain must be >=-1%. Full versus the
DEV-selected heuristic must have pooled gain lower CI >=-1%; do not select
FCFS/risk on TEST. Health reference is matched mask-only on the same tape:
family-averaged full-minus-mask availability lower CI >=-0.01,
terminal pending/machine upper CI <=+0.025 and policy failures/machine upper
CI <=+0.25. Zero invalid assignments. Record per-configuration and every
training replica, fixed-final cost, zero-shot warm cost, scratch train-separate
cost, selected checkpoint, learning AUC and variance. p95 `model.select`
<=50 ms at N16, with end-to-end observation/coordinator/physical latency
reported separately; no network latency claim.

The inferential family contains 10 endpoints: primary pooled full/mask cost,
two family full/mask costs, pooled full/heuristic cost and six family health
differences. Use 40,000 crossed bootstrap draws over paired training bundles
and paired physical seeds, independently by configuration, with one-sided
Bonferroni alpha `.025/10` per endpoint. Report all endpoints and CI even if
one gate fails. Five training bundles limit generality across teacher
foundations; leave-one-bundle-out sensitivity is mandatory. The 256-seed
panel is fixed; if precision is insufficient, report the inconclusive result
and prospectively design a new larger panel, never replace CI by a point
estimate or add TEST seeds post hoc.

Fresh timestamped artifacts for smoke, DEV and confirmation must contain
exact invocation and source/git/protocol/registry/teacher hashes, physical
config, adapter contract, source snapshot, warm bank/checkpoint and learner
logs, train/DEV/TEST episode and decision rows, frozen reference/checkpoint
selection, cost components, health, conflict/invalid and latency, CI,
compute, and output bindings. An independent read-only verifier must replay
requests, assignments and scalar physical transitions and recompute
estimands/CI before any C claim. `tqdm` covers bank, PPO and multi-seed
evaluation. Correctness tests and lab smoke, then commit/push, precede FULL.
