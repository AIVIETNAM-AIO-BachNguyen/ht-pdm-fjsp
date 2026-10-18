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
and all large policies receive scale-specific PPO fine-tuning. New matched
ablations and heterogeneous workload panels are in progress. Until they pass,
the supported conclusion is bounded to the balanced simulation family.

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

## Registered experiments and result table

| Contrast | Historical result / current status | Valid interpretation |
|---|---|---|
| V4 full vs centralized PPO, N4 | 0.0778% gain, interval [−0.2960%, 0.6168%] | Passes registered 1% noninferiority; does not show superiority |
| V4 full vs matched independent graph, N4 | 1.6430% gain, interval [−0.5997%, 5.7302%] | Positive coordination effect unconfirmed at N4 |
| V4 full vs matched independent graph, N8/N16/N32 | 20.7877% / 43.7460% / 56.8626% gains; registered lower bounds positive | Entire reservation/message service benefits balanced larger settings |
| B1 full vs mask-only vs no-reservation | New N4/N8 DEV gate failed: pooled +0.2205%; N4 +0.6746%, N8 −0.2437%; TEST remains sealed | V4 coordination benefit does not establish an added prefix benefit; no positive prefix claim |
| A teacher-assisted vs scratch on N8/N16/N32 | A1 DEV running on fresh seeds; no verdict or TEST opened | Must match PPO/optimizer/physics budgets and disclose extra teacher compute |
| B2 candidate-conditioned relational prefix vs mask-only | B2 DEV running on fresh seeds; no verdict or TEST opened | B1 negative result remains historical; B2 must pass its own matched contrast |
| C non-V4 skill/capacity and persistent-request families | C1 DEV running on two prospectively fixed N8/N16 families; no verdict or TEST opened | Must beat matched mask-only and match DEV-selected strong FCFS/risk within registered bounds |

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
| New B1 and future A/C | Populate from sealed artifacts after runs | Report per-phase decisions, updates, presentations, wall time and p95 end-to-end latency separately |

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
preventive maintenance in flexible manufacturing. A nearby but distinct
field-service problem is [Kiesmüller, Wang and Kolisch (European Journal of
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
- Report negative B1/A/C rounds, per-replica instability, health tradeoffs,
  decision latency and all extra teacher costs whether or not a gate passes.
- The B1 development experiment failed its registered prefix gate despite
  valid training/physical audits. Its N8 full-versus-mask-only gain was
  −0.2437% on average and −1.6699% for one replica. It must remain visible
  even if a later follow-up succeeds on a new panel.
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
