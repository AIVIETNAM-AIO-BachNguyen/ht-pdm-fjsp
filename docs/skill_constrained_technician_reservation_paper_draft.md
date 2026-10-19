# Skill-Constrained Technician Reservation: working paper draft

**Draft status (18 October 2026): incomplete research manuscript.** The V4
balanced-benchmark result is historical and confirmed under its own protocol.
Claims that teacher assistance beats scratch, selected-edge prefix beats
mask-only, and the method transfers to new maintenance families are **open**.
Do not submit this text as a completed positive framework paper until the new
A/B/C gates, independent audit and compute accounting pass.

## Abstract (current evidence only)

We study dynamic maintenance dispatch in which heterogeneous technicians share
machines with skill constraints. A centralized N4 PPO teacher chooses a complete
matching. A graph student instead selects a machine or WAIT for each technician
in public order, reserving machines and exposing a learned summary of earlier
selected edges. On the registered balanced benchmark, historical Reservation
PPO V4 achieves cost close to centralized PPO at N4 under a 1% noninferiority
margin and improves over a matched graph policy that removes the entire
coordination service at N8–N32. Its N4 cost superiority over centralized PPO is
unproven; its N4 coordination advantage is also unproven. The historical
comparison does not isolate prefix information from reservation feasibility,
and all large policies receive scale-specific PPO fine-tuning. Two
prospective prefix DEV screens have failed. Teacher transfer and
heterogeneous workload panels remain in progress. The supported positive
conclusion is bounded to the historical balanced simulation family.

## Problem and method

At each tick a plant presents a current bipartite machine–technician graph,
machine condition and outstanding urgent work. An edge is feasible only when
the technician has the requisite skill and is available. The student encodes
machine–technician edges, attends to competing machine embeddings, then chooses
one machine or WAIT per technician. The coordinator masks previously reserved
machines and broadcasts an embedding summary of earlier selected edges. The
resulting matching executes simultaneously in the physical simulator. The
student has fixed parameter count across N but its attention computation and
decision count grow with graph size. The N4 teacher uses a matching catalogue;
the large actor, critic and environment do not enumerate one.

Teacher assistance consists of previously trained centralized PPO, a bank of
N4 states labeled with complete teacher matchings, 32 epochs of graph student
pretraining, and 491,520 PPO joint decisions for **each target scale**. This
is adaptation with fine-tuning, not zero-shot transfer or evidence of total
compute efficiency. A zero-shot N4 policy and scratch students require separate
measurements.

## Prospective evaluation design (methods locked; outcomes pending)

The new experiments use disjoint registered training, DEV and confirmation
seed streams. Learned arms in a matched comparison share the physical tape,
public technician order, eligible checkpoint budgets and target-scale PPO
schedule. Every fit spends 491,520 joint decisions, including fits whose DEV
choice is an earlier checkpoint. Checkpoints at 0, 122,880, 245,760,
368,640 and 491,520 decisions are selected by minimum mean DEV extended
cost, with the earliest checkpoint breaking a tie. Selected-checkpoint cost,
fixed-final cost, learning curves and zero-shot warm-policy cost are distinct
measurements. Confirmation TEST is opened only after the corresponding
registered DEV screen and independent DEV audit pass. No post-TEST
checkpoint or heuristic reselection is allowed; 48-hour A1/B2 and 72-hour
C1 hard stops classify timeout as INCOMPLETE.

| Study | Primary matched contrast | DEV screen | Conditional confirmation |
|---|---|---|---|
| A1 teacher transfer | N4 teacher-distilled reservation graph versus the same graph initialized from matched random tensors and trained from scratch | Three student bundles; N8/N16/N32; 32 DEV episodes per checkpoint; pooled selected-cost gain at least 0.5%, no scale below −1%, positive at two scales, zero invalid assignments and health point guards | Five new student bundles; 64 new DEV episodes for selection and 256 sealed TEST episodes at each scale |
| B2 prefix mechanism | Relational selected-edge message versus a separately trained equal-parameter mask-only actor with identical reservation feasibility | Three bundles; N4/N8; 32 DEV episodes per checkpoint; pooled full/mask gain at least 0.5%, neither scale below −1%, zero invalid assignments and health point guards | Five new bundles; N4/N8/N16/N32; 64 new DEV and 256 sealed TEST episodes per scale |
| C1 maintenance diversity | The fixed B2 full architecture versus matched mask-only, plus a single FCFS/risk heuristic chosen globally on DEV | Three bundles; four registered configurations; 32 DEV episodes per configuration/checkpoint; full/mask gain at least 0.5%, neither family below −1%, full within 1% of the selected heuristic, zero invalid assignments and health point guards | Five new bundles; 64 new DEV and 256 sealed TEST episodes per configuration; heuristic choice frozen before TEST |

A1 measures teacher assistance at N8/N16/N32 on the balanced V4 physics.
It attributes any advantage to teacher information **plus its additional
teacher, bank and distillation compute**. B2 isolates the selected-edge
message: full and mask-only have the same legal sets and parameter count;
the no-reservation arm measures a broader coordination contrast and cannot
establish a prefix benefit. B2 replaces B1's shared average message with
candidate-query attention over earlier selected edges. A WAIT adds no
selected edge, and the message is zero for an empty prefix.

C1 keeps one request and cost contract across two non-V4 families. Its
`skill_bottleneck` configurations have N/K=8/3 and 16/6, one-tick service,
and compatibility when `m%3` equals `j%3` or `(j+1)%3`. Its `long_service`
configurations have N/K=8/4 and 16/8, compatibility when `j%4` equals
`m%4` or `(m+1)%4`, and two- or four-tick nonpreemptive service. Requests
are created at failure or age at least three and persist until assignment;
the policy chooses assignment, not request creation. The full and mask-only
actors see the same pending-request legal set. Both FCFS and risk rules run
on DEV in every configuration; one global reference is selected by equal
configuration mean cost, earliest-tie FCFS, before TEST. The paper must
report both rules on TEST, the globally selected reference, warm zero-shot,
full scratch, and the no-reservation arm.

Confirmation uses a fixed 40,000-draw crossed bootstrap over paired student
bundles and paired physical episodes, independently by scale or
configuration. The one-sided Bonferroni level is 0.025 divided by 13, 17
and 10 endpoints for A1, B2 and C1 respectively. Each primary selected-cost
gain must reach 1% and have an adjusted lower bound above zero. Registered
scale/family cost guards, availability, terminal pending and policy-failure
bounds, zero invalid assignments, and p95 `model.select` at most 50 ms
at N32 (A1/B2) or N16 (C1) are separate requirements. C1 also requires the adjusted lower bound for full versus its
DEV-selected heuristic to be at least −1%. A positive point estimate alone
cannot establish any claim. Independent verifiers must recompute the
estimands and replay physical trajectories; the C1 stage-level legal-mask
replay is prospectively limited to the first eight TEST seeds per
configuration, while its physical transition replay covers the full panel.

## Registered experiments and result table

| Contrast | Historical result / current status | Valid interpretation |
|---|---|---|
| V4 full vs centralized PPO, N4 | 0.0778% gain, interval [−0.2960%, 0.6168%] | Passes registered 1% noninferiority; does not show superiority |
| V4 full vs matched independent graph, N4 | 1.6430% gain, interval [−0.5997%, 5.7302%] | Positive coordination effect unconfirmed at N4 |
| V4 full vs matched independent graph, N8/N16/N32 | 20.7877% / 43.7460% / 56.8626% gains; registered lower bounds positive | Entire reservation/message service benefits balanced larger settings |
| B1 full vs mask-only vs no-reservation | New N4/N8 DEV gate failed: pooled +0.2205%; N4 +0.6746%, N8 −0.2437%; TEST remains sealed | V4 coordination benefit does not establish an added prefix benefit; no positive prefix claim |
| A teacher-assisted vs scratch on N8/N16/N32 | A1 DEV running on fresh seeds; no verdict or TEST opened | Must match PPO/optimizer/physics budgets and disclose extra teacher compute |
| B2 candidate-conditioned relational prefix vs mask-only | DEV gate failed: pooled −0.1644%, N4 −0.0168%, N8 −0.3119%; independent DEV audit passed; TEST sealed | The relational message has no demonstrated gain over a separately trained mask-only control on this DEV panel |
| C non-V4 skill/capacity and persistent-request families | C1 DEV and independent verifier PASS on two fixed N8/N16 families; confirmation running, no TEST verdict yet | Must beat matched mask-only and match DEV-selected strong FCFS/risk within registered bounds |

V4 used 48 policy ticks and 12 common deadline-tail ticks, K=N/2, repeated
balanced skill blocks. Its N32 failures per machine were higher than the
deadline reference despite passing the predeclared +0.25 health margin. Its
`model.select` p95 was 1.527–9.460 ms over N4–N32. End-to-end latency and
network communication were not included. Test decisions had zero invalid
assignments.

## Compute accounting (known and pending)

| Component | Historical V4 accounting | Interpretation / outstanding measurement |
|---|---:|---|
| Foundation PPO teachers | 5 × 491,520 = 2,457,600 PPO decisions | Obtained from an earlier 20-model benchmark suite lasting 6,078.89 s overall; isolated teacher-only wall time unavailable |
| Teacher-bank generation | 5 × 122,880 = 614,400 N4 state exposures | Shared between V4 graph arms; physical tape separate from PPO training |
| Distillation | 5 × 2 arms × 32 × 122,880 = 39,321,600 state presentations | Fixed epochs; optimizer steps and per-bundle logs are preserved in source artifact |
| Scale-specific PPO | 5 × 2 arms × 4 scales × 491,520 = 19,660,800 joint decisions | 40 graph fits; additional to teacher training and bank |
| V4 confirmation wall time | 76,390.11 s, about 21 h 13 min | Eight parallel CPU workers; includes training, DEV selection, TEST and replay; cannot attribute all wall time to one phase |
| V4 inference | `model.select` p95 1.527–9.460 ms by N | Observation, environment stepping and communication excluded |
| B1 DEV graph PPO | 18 fits × 491,520 = 8,847,360 joint decisions; 184,320 training episodes | Three full/mask-only/independent bundles at N4/N8; six N4 centralized reference evaluations were frozen, not additional learned fits |
| B1 DEV teacher bank and distillation | 3 × 122,880 = 368,640 N4 bank states; 3 × 3 × 32 × 122,880 = 35,389,440 graph state presentations | Foundation PPO teachers were reused; these are additional to B1 target PPO |
| B1 DEV wall time | 4,490.05 s, about 74 min 50 s | Includes bank, pretraining, training and DEV; no scientific TEST opened |
| B2 DEV graph PPO | 18 fits × 491,520 = 8,847,360 joint decisions; 184,320 training episodes | Three full/mask-only/independent bundles at N4/N8; six N4 centralized references were frozen evaluations, not new fits |
| B2 DEV teacher bank and distillation | 3 × 122,880 = 368,640 N4 bank states; 3 × 3 × 32 × 122,880 = 35,389,440 graph state presentations | Foundation PPO teachers reused; bank shared across three graph arms and scales |
| B2 DEV wall time | 12,432.63 s, about 3 h 27 min | Shared 16-core lab with simultaneous A1/C1 jobs; wall time is not isolated CPU cost; no scientific TEST opened |
| New A/C1/B3 and any confirmation | Populate from completed sealed artifacts | Report per-phase decisions, optimizer updates, presentations, wall time and p95 end-to-end latency separately |

## Prior art and novelty boundary

[MAT (Wen et al., NeurIPS 2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/69413f87e5a34897cd010ca698097d0a-Abstract-Conference.html)
already treats cooperative multi-agent joint decisions as a sequential
decision process. [PARCO (Berto et al., NeurIPS 2025)](https://proceedings.nips.cc/paper_files/paper/2025/hash/f404c0259def290306159047b0b52584-Abstract-Conference.html)
uses agent communication and priority-based conflict handling for multi-agent
combinatorial optimization. [Woo et al. (AAAI 2022)](https://ojs.aaai.org/index.php/AAAI/article/view/20845)
distill a high-performance combinatorial policy into a simpler solver. Thus
autoregressive action construction, conflict handling, graph encoding and
distillation are not novel in general. A defensible contribution would be a
carefully specified skill-constrained maintenance interface and independently
verified **domain-specific** teacher transfer, prefix-information benefit and
cost–health–scaling behavior against matched controls. Those additional
claims are currently hypotheses.

The domain boundary is also narrower than a claim to invent graph-based
maintenance learning. [Wang and Gombolay (ICAPS 2022)](https://ojs.aaai.org/index.php/ICAPS/article/view/19839)
already learned heterogeneous-graph policies for stochastic predictive
maintenance resource scheduling. [Bhatta and Chang (Robotics and
Computer-Integrated Manufacturing, 2023)](https://www.sciencedirect.com/science/article/pii/S0736584523000704)
combined heterogeneous graphs and MARL for robot assignment, tool change and
preventive maintenance in flexible manufacturing. [Chen and Zhou
(2026)](https://www.sciencedirect.com/science/article/abs/pii/S0951832025012864)
also combine multi-hierarchical graph representation and PPO for maintenance
optimization with inter-machine and resource dependencies. This further
limits any novelty claim based on combining a graph encoder with PPO. A nearby
but distinct field-service problem is [Kiesmüller, Wang and Kolisch (European Journal of
Operational Research, online 2026)](https://www.sciencedirect.com/science/article/pii/S0377221726004625),
which uses graph-based value approximation for heterogeneous technician
allocation/routing with uncertain repair outcomes. These studies establish
that graph-based maintenance scheduling and heterogeneous technician
allocation have prior art. Our potential distinction is the **registered
mechanism separation and transfer/health evidence for this simulator**, if
the new controlled tests support it. No head-to-head result against these
papers has been measured, so we must not claim superiority over their methods.

## Limitations and failure cases to retain

- V4 has one balanced physics family, K=N/2, and a common finite horizon.
- C1 adds K/N=3/8 in the skill-bottleneck family but retains K/N=1/2 in
  the long-service family. A result on these four configurations cannot
  establish robustness over staffing ratios, skill coverage, and offered
  load generally. The appropriate technician-to-machine ratio is a domain
  parameter, not a fixed law; no plant staffing data validate either ratio.
  Any broader claim requires a prospectively registered, disjoint panel
  varying these axes and distinguishing zero-shot from fine-tuning and
  training anew. Worker-flexibility benchmarks also vary worker counts and
  eligibility, but their production workers have a different resource role
  from shared maintenance technicians; see [Hutter et al.
  (2026)](https://arxiv.org/html/2501.16159v3).
- The graph student receives the complete current graph and a shared
  reservation service. Local-sensor decentralization and network performance
  were not tested.
- Checkpoint selection on DEV matters; V4 N4 selected warm checkpoint for two
  bundles, and some large final checkpoints degraded sharply.
- Five foundation teachers are a fixed historical collection, so uncertainty
  over entirely new teacher training is incompletely represented.
- Strong FCFS/risk policies on persistent-request, multi-tick technician
  workloads are credible competitors; a coupled-dispatch follow-up failed
  against them. The new framework must be compared directly, without changing
  physics or reference selection after TEST.
- Report negative B1/B2 and any failed A/C/B3 rounds, per-replica instability, health tradeoffs,
  decision latency and all extra teacher costs whether or not a gate passes.
- B2 development also failed its preregistered prefix gate on new N4/N8
  seeds: pooled full/mask-only gain −0.1644% versus the +0.5% DEV gate,
  with negative point gains at both scales. Its independent auditor passed
  source/output bindings and the recomputed cost/health contrasts. All health
  point guards passed, but B2 TEST remains sealed and no B2 positive claim
  is supported.
- The B1 development experiment failed its registered prefix gate despite
  valid training/physical audits. Its N8 full-versus-mask-only gain was
  −0.2437% on average and −1.6699% for one replica. It must remain visible
  even if a later follow-up succeeds on a new panel.
- A B2 DEV-only same-weight intervention found that zeroing the relational
  message changed 21/4,608 N4 and 549/4,608 N8 policy-tick decisions.
  The fitted full weights cost 1.3135% and 1.2121% less than their
  message-off copies on those opened DEV panels, with one adverse N8
  replica. This indicates use of the message within the fitted policy;
  the separately trained mask-only actor still won the registered B2
  contrast. The intervention cannot overturn B2's DEV failure.
- A B1 DEV-only same-weight intervention found that removing prefix changed
  621/4,608 N8 decisions and increased cost 2.624% on that reused panel.
  This diagnoses active use of prefix; it does not establish benefit over a
  separately trained mask-only control, which remained the failed main test.

## Planned manuscript sections

1. Maintenance dispatch problem, physics and observable state.
2. Graph policy, reservation coordinator, teacher adapter and feasibility proof.
3. Prospective A/B/C design, independent seed panels, inference family and audit.
4. V4 historical result and new teacher/scratch, full/mask-only/no-reservation ablations.
5. Non-V4 skill/capacity and persistent-request panels, strong heuristic controls.
6. Compute and latency accounting, scope, adverse cases and field limitations.


## Registered B3 follow-up (DEV failed; TEST sealed)

After the independently verified B2 DEV failure, one new mechanism test was
registered before observing any B3 scientific outcome. B3 keeps the B2
relational message and balanced maintenance physics, and adds two summaries
of earlier selected technician–machine pairs: service occupancy and
duration-weighted weak restoration, each normalized by `4*K`. Candidate
edge logits receive these summaries multiplied by that candidate's service
duration divided by four; WAIT receives the summaries directly. The matched
mask-only arm shares the entire current graph, reservation legal set, order,
resource pool, teacher bank, initialization, architecture size and PPO budget,
while both selected-pair channels are zeroed. The separate no-reservation arm
remains a service-level comparator. The prospective protocol and fresh
disjoint panels are in `docs/technician_prefix_capacity_v3_plan.md` and
`configs/technician_prefix_capacity_v3_seed_registry.json` at commit
`0d23db8`; smoke-verified code is commit `a99ab1c`. B3 DEV completed and failed its registered cost screen; its confirmation TEST remains sealed. The engineering smoke and its
independent replay establish implementation integrity only. B1 and B2
negative developmental verdicts are retained whatever B3 shows.


A separate confirmation-only branch (`079001a`) now binds any B3 TEST
opening to the exact DEV source/protocol/registry and an independent DEV
`DEV_GATE_PASS` report with the source manifest and verifier hashes. Its
independent replay checks both the full and mask-only policy trajectories;
on engineering smoke it covered 768 states, 2,304 decoder stages, 2,400
physical ticks and zero invalid assignments. These checks remain engineering
evidence only; B3 DEV subsequently failed its registered gate.


The A1 and C1 confirmation runners now also require independent DEV PASS
reports bound to the exact DEV manifest and saved verifier hash (confirmation
only commits `1b68a49` and `d1a8418`). Their scientific TEST panels remain
sealed while DEV training proceeds. These guards support provenance and do
not supply evidence for either claim.


B3's independently audited DEV result was pooled full/mask-only cost gain
+0.3646%, below its +0.5% DEV screen; N4 +1.1819%, N8 −0.4576%.
All point health guards passed and there were zero invalid assignments.
The source/output seals, training tape and checkpoint selection were
independently recomputed; the B3 confirmation TEST was not opened. Two of
three N8 training replicas were adverse. This is a third developmental
failure for the registered prefix-information contrast, despite a positive
N4 mean. It must remain visible in any eventual manuscript and cannot be
recast as confirmation or rescued by full-versus-no-reservation comparisons.

A subsequent, prospectively recorded DEV-only diagnosis held each selected
full checkpoint's fitted weights fixed. Zeroing only the new capacity inputs
changed 0 of 4,608 full-policy ticks at N4 and 2 of 4,608 at N8; the
corresponding mean cost per machine was unchanged at N4 and rose from
265.5482 to 265.5794 at N8. Zeroing both the capacity inputs and the
relational message changed 31 of 4,608 ticks at N4 and 470 of 4,608 at N8;
mean costs per machine rose from 262.8047 to 269.2005 and from 265.5482
to 270.4714, respectively. The 192-episode artifact's hashes, DEV panel,
frozen full-policy rows and aggregates passed an independent saved-artifact
audit. This is a same-weight mechanism diagnosis, not the separately trained
matched-control contrast. It leaves the B3 failure and sealed TEST intact.

A fourth prefix mechanism is prospectively registered after the B3 negative
result and same-weight diagnosis. It adds a parameter-free direct logit
residual between the current candidate and selected-edge message, using
the graph encoder's existing query/key maps. Its own fresh DEV and
conditional confirmation panels are disjoint from B1–B3. Eight focused
tests and an independently replayed engineering smoke validated the
implementation; that smoke is not scientific evidence. B4 DEV is pending,
and the B1–B3 failures remain part of the record.

The fourth preregistered prefix follow-up also failed its DEV screen.
Its independently audited full-versus-separately-trained mask-only gain
was −2.0386% pooled, −0.1643% at N4 and −3.9177% at N8. N8 full
availability was lower by 0.01074, beyond the registered 0.01 point
guard; all recorded invalid-assignment counts were zero. The DEV run
used three training bundles and 32 physical seeds per scale, with
184,320 training episodes across 24 models and 3,072 DEV episodes.
The scientific B4 TEST panel remains sealed. A same-weight intervention,
if performed, can diagnose this failure but cannot reverse it. B1–B4
are four distinct negative developmental rounds for the registered
prefix-information claim.

A frozen-weight B4 DEV diagnosis then zeroed only the direct selected-edge
logit residual. This changed 6/4,608 N4 and 1,490/4,608 N8 decisions
on full-policy states. Mean full costs per machine were 263.5260 versus
264.3464 with the residual off at N4, and 272.7109 versus 273.4805 at
N8, although one N8 replica was adverse. The 192-episode DEV-only
artifact passed an independent saved-output/panel/arithmetic audit.
This shows use of the fitted residual; it cannot overturn the
separately trained mask-only contrast, the N8 health failure or the
sealed B4 TEST. Four rounds now suggest that optimization and stability
of the augmented actor require investigation before another claim.

A fifth developmental follow-up was registered only after the negative B4
verdict and its same-weight diagnosis. It tests training stability by giving
full and mask-only policies an identical mask-only warm state and first
245,760 PPO decisions, then activating the B4 selected-prefix path only in
the full arm for the remaining 245,760 decisions while retaining Adam state.
Both arms use the same total budget and select among the same three
post-branch DEV checkpoints. Fresh DEV and conditional TEST panels and the
unchanged B4 screen were locked before implementation. Focused tests and
an independently replayed engineering smoke passed; the B5 scientific DEV
run subsequently failed its registered screen, as reported below.

B5 completed and failed its preregistered DEV screen. The independently
audited pooled full-versus-mask-only selected-cost gain was −12.8136%
(N4 −8.3221%, N8 −17.2770%), with multiple availability, pending and
failure health guards failing despite zero invalid assignments. The exact
common mask-only precursor and branch checkpoint were verified. On the
opened DEV panel, activating the full prefix at the shared 245,760-step
checkpoint produced cost-per-machine gaps of 297.7943 versus 263.0130
at N4 and 383.2148 versus 265.9141 at N8. Further PPO training narrowed
but did not eliminate the gaps. This descriptive checkpoint trajectory
does not identify which prefix component caused the damage. B5 TEST
remains sealed, and B1–B5 are negative developmental results for the
matched prefix claim. No efficacy or confirmation claim follows from B5.

A preregistered, exploratory B5 branch-point intervention used the same
245,760-step weights on the already opened DEV panel. Independent artifact,
panel, and arithmetic audit passed for 192 episodes. Removing only the
direct prefix logit residual, while preserving graph encoding, relational
prefix, capacity features, legality and base logits, reduced cost per
machine from 297.7943 to 263.0130 at N4 (mask-only 263.0130) and from
383.2148 to 266.4740 at N8 (mask-only 265.9141). These comparisons
attribute most *immediate* activation damage to the direct residual at
shared weights. They do not establish the effect after training or change
the failed B5 matched trained-control verdict. B5 TEST is sealed. A
separate residual-free, delayed relational/capacity follow-up was
registered on fresh seeds and panels before its efficacy run.

The prospectively specified B6 follow-up tests a separately trained,
residual-free relational/capacity prefix after the same delayed mask-only
precursor, on fresh training, DEV and conditional TEST panels. The B5 gates
and equal training budget remain fixed. This was the preregistered design;
the completed B6 DEV result is reported below.

C1 completed its developmental screen. An independent audit of 48 fitted
models, 7,680 DEV episodes, 491,520 training episodes and paired physical
tapes passed. The selected pooled full-versus-mask-only cost gain was
12.9747%, with 6.9969% in long service and 20.2024% in skill bottleneck;
all registered family health point guards passed. The DEV-selected risk
heuristic was worse than full by 9.4295% pooled. A fresh confirmation run
is underway with its runner bound to the exact independent DEV PASS.
These are DEV findings; C1 has no confirmatory TEST verdict yet.

B6 residual-free delayed prefix code passed focused tests and independently
replayed engineering smoke with zero direct-residual stages and active
selected-prefix messages. Its scientific DEV run completed with internal
audit PASS, 24 trained models and 1,920 DEV episodes. The independent
verifier passed 280 source seals, 453 output bindings, and paired precursor
records, and reproduced **DEV_GATE_FAIL**: selected pooled full-versus-
mask-only cost gain was only +0.0933% against the registered +0.5% DEV
screen; N4 was +0.2566% and N8 was −0.0729%. Health point guards and
zero-invalid checks passed. Its conditional TEST panel remains sealed.
B5's earlier same-weight intervention isolated its immediate direct-residual
damage; B6 has not demonstrated a separately trained prefix benefit. B1–B6 remain
negative developmental results for that claim.
