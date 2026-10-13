# Technician reservation PPO V4: sealed confirmation findings

The confirmation passes every required prospective gate on the balanced
maintenance benchmark at N/K = 4/2, 8/4, 16/8 and 32/16. The supported result is
near-central-PPO quality at N4, scalable assignment without a large matching
catalogue, and a positive coordination effect at every larger setting.
**It does not establish cost superiority over centralized PPO, or a positive
coordination effect at N4.**

## Evidence and protocol

The source experiment is
`artifacts/technician_reservation_ppo_v4_confirmation_20261012T213632Z`,
frozen source commit `92b675a94fc5f390f12fd9a4ce44b4eefe0c0f4a`.
The run completed at `2026-10-13T18:49:43.802502+00:00`, after 76390.108 seconds
(21 h 13 min), within the prelocked 24-hour limit. It ran directly on this
Ubuntu lab, with eight independent CPU workers and one Torch thread per worker.
Manifest: COMPLETED. Independent physical audit: PASS.

The [prospective plan](technician_reservation_ppo_v4_plan.md) and
[engineering gate](technician_reservation_ppo_v4_confirmation_engineering_gate.md)
precede scientific test opening. Full suite: 992 passed; the four-to-eight-worker
smoke comparison is exact for model tensors, budgets and physical outcomes.
Development rounds and their negative outcomes remain preserved.

Five teacher/student bundles use new student seeds 57400000–57400004,
development seeds 57500000–57500063 and sealed test seeds 57600000–57600255.
All five bundles and all 256 episode seeds per scale are retained. Choices were
sealed at `2026-10-13T17:59:15.201897+00:00` before test generation. References
are the original development-selected PPO/MAPPO models; no test-based reselection
occurred. The final grid contains 13824 episodes and 829440 independently replayed
physical transitions, with zero invalid assignments.

Intervals below use the predeclared 40000-draw crossed bootstrap: paired methods
within each scale, shared bundle weights across scales, independent episode axes
across scales. Round 1 alpha is 0.025; the Bonferroni family has 24 endpoints and
individual two-sided confidence 99.895833%. With only five bundles, coverage is
approximate and conditional on the existing five-teacher collection. Per-bundle
and leave-one-out results accompany the intervals.

Cost is H+L over 48 policy decisions plus 12 common deadline-tail ticks. Health
endpoints use the policy cutoff. Positive gain means
`100 * (1 - candidate_cost / comparator_cost)`.

## 1. Decision quality

| N4 method | Total cost | Cost/machine |
|---|---:|---:|
| Reservation PPO | 1048.6914 | 262.1729 |
| Central PPO | 1049.5078 | 262.3770 |
| Local MAPPO | 1079.4961 | 269.8740 |
| Matched independent graph PPO | 1066.2094 | 266.5523 |

| N4 contrast | Gain (%) | Predeclared interval (%) | Conclusion |
|---|---:|---:|---|
| Versus central PPO | 0.0778 | [-0.2960, 0.6168] | Noninferior within the locked 1% margin |
| Versus MAPPO | 2.8536 | [0.5184, 5.8940] | Lower cost on the fresh panel |
| Versus matched independent PPO | 1.6430 | [-0.5997, 5.7302] | Required 1% noninferiority passes; positive effect unproven |

Practical PPO superiority requires a gain lower bound of at least 1%; this
optional gate fails. The PPO result should be described as near-central quality.
MAPPO has less initial information and a smaller policy than the graph method;
beating MAPPO alone is not evidence that coordination caused the improvement.
The new methods also receive teacher assistance and additional pretraining.

## 2. Scaling and decision time

| Machines/technicians | Reservation cost/machine | Independent cost/machine | Deadline cost/machine | Neural P95 (ms) |
|---|---:|---:|---:|---:|
| 4/2 | 262.1729 | 266.5523 | 283.8848 | 1.527 |
| 8/4 | 265.9545 | 335.7488 | 270.3623 | 2.534 |
| 16/8 | 266.5710 | 473.8705 | 273.2949 | 4.703 |
| 32/16 | 264.8450 | 613.9570 | 271.6785 | 9.460 |

All four scales use 82819 graph parameters. Every neural timing panel contains
61440 single-state decisions after the training pool shuts down. P95 stays
below the locked 50 ms limit. These timings measure `model.select`; observation
construction, physical stepping/arbitration and network transport are excluded.
They are inference measurements, not a claim of fast training.

Upper interval bounds on the large/N4 cost-per-machine ratios are 1.03477,
1.02934 and 1.01933, below the locked 1.10 limit. Gains over deadline are:
N8 1.6303% [-0.4475, 3.1747], N16 2.4603% [1.4766, 3.6162],
N32 2.5153% [1.7799, 3.2717]. Each passes the locked 2% noninferiority condition;
only N16/N32 show positive gain intervals against deadline.

The actor represents machine–technician edges and selects one machine or WAIT
per technician, with sequential reservations and simultaneous physical execution.
Large environments, actors, critics and tail control do not enumerate matchings.
The teacher and central reference retain the original 21-entry N4 catalogue only.
For context, a complete catalogue would contain 3393, 1174226049 and
30777323181433121541633 matchings at N8/N16/N32. These are structural counts,
not measured runtime speedups. Competition attention is still computationally
expensive; evidence covers these four sizes, not arbitrarily large plants.

## 3. Coordination intervention and health

The independent graph arm shares complete initial graph information, IDs/order,
attention, capacity, initialization, teacher bank and permutation augmentation,
physical training tapes, checkpoint-selection rule and training budgets.
The intervention removes machine reservation masks and the selected-edge prefix;
colliding proposals are arbitrated without a retry. This compares the whole
coordination service, not attention or distillation in isolation.

| Scale | Cost gain over independent (%) | Predeclared interval (%) |
|---|---:|---:|
| N8 | 20.7877 | [0.8414, 35.9730] |
| N16 | 43.7460 | [28.7488, 53.9615] |
| N32 | 56.8626 | [46.1729, 64.4576] |

Every required larger-setting coordination interval is positive. Removing
coordination is associated with higher conflicts/backlog/failures and lower
availability on those settings; these diagnostics add no new hypothesis tests.

| N | Reservation availability (%) | Pending/machine | Failures/machine | Independent conflicts/episode |
|---|---:|---:|---:|---:|
| 4 | 66.8319 | 0.000000 | 0.142773 | 2.546 |
| 8 | 66.8184 | 0.001367 | 0.244629 | 69.134 |
| 16 | 66.9636 | 0.000732 | 0.270703 | 102.185 |
| 32 | 67.0397 | 0.000122 | 0.195264 | 150.800 |

Reservation has zero conflicts and rejected proposals at every scale. All health
gates pass: availability difference lower bound >= -0.01, pending/machine
difference upper bound <= 0.025, failures/machine difference upper bound <= 0.25,
versus PPO at N4 and deadline at larger N.

Passing margins does not mean uniformly better health. Failure differences versus
deadline are +0.078320 at N16, interval [-0.116992, 0.241073], and +0.111890 at
N32, interval [0.011788, 0.241614]. Both upper bounds are below 0.25, but near
that limit; N32 failures are higher than deadline with a positive interval.
At N32 availability improves by 2.0566 percentage points, with interval
[1.9058, 2.1968]. These tradeoffs must remain visible in any paper or presentation.

## Training budget, stability and scope

Each graph arm/seed/scale spends 491520 new PPO joint decisions: 40 fresh fits,
19660800 decisions, 409600 training episodes and 204800 paired physical tapes.
Each bundle adds 122880 shared N4 teacher states; each arm uses 32 fixed
pretraining epochs, or 3932160 presentations. The five shared banks total
614400 states. Original PPO/MAPPO foundation training and teacher generation are
additional costs. No optimizer-only or sample-efficiency superiority is claimed.

Checkpoint selection is necessary to the demonstrated result. All five replicas
spend their full budgets even when a warm or earlier checkpoint is selected.
Reservation selected steps, in seed order 57400000–57400004, are:

| N | Selected PPO fine-tuning steps |
|---|---|
| 4 | 491520, 0, 122880, 491520, 0 |
| 8 | 368640, 491520, 491520, 491520, 491520 |
| 16 | 245760, 245760, 122880, 245760, 245760 |
| 32 | 122880, 491520, 368640, 368640, 491520 |

Fine-tuning is not uniformly stable. At the final N16 checkpoint, development
cost/machine for seeds 57400000 and 57400001 is 854.5391 and 471.1104; the
predeclared development selection chooses earlier checkpoints. The final result
concerns that selected policy procedure, not monotonic learning or final-step
checkpoint quality. N4 includes two selected warm checkpoints, so improvement
from PPO fine-tuning is not established for every seed/scale.

Leave-one-bundle-out point-estimate ranges remain positive against MAPPO
(2.1287–3.4013%) and for N8/N16/N32 coordination
(15.8268–24.8076%, 39.9838–46.5752%, 54.3679–58.9643%). Against PPO the range
is -0.0160–0.1115%, consistent with the near-PPO conclusion. This is descriptive
sensitivity, not another inferential test or a reselection procedure.

This is a PPO-derived graph policy with shared full graph input and a shared
reservation service. It does not implement agents restricted to local sensors
or validate communication over a network. The benchmark repeats the balanced
skill profiles with technicians=N/2 and a fixed 48+12 horizon. General FJSP
production workloads, arbitrary skill/technician ratios and field deployment
remain outside this evidence. The contribution is the measured coordination and
scaling procedure on this benchmark, not invention of PPO, autoregressive
decoding or policy distillation.

## Reports and completion review

Raw artifacts are preserved; reports use fresh timestamped directories:

- `artifacts/technician_reservation_ppo_v4_analysis_20261013T185421Z`: scientific
  quality/CI/latency, development curves, teacher diagnostics and representation
  plots in PNG/PDF. Earlier render attempts are retained. The teacher legend was
  moved outside the bars after visual QA; experiment source is unchanged.
- `artifacts/technician_reservation_ppo_v4_evidence_report_20261013T185041Z`:
  complete endpoint/guard tables, all replica/checkpoint results, system health
  diagnostics, budgets and scope, with sealed input hashes and copied reporter.
- `artifacts/technician_reservation_ppo_v4_completion_review_20261013T185159Z`:
  independent verification of 666 output bindings, 248 source seals, exact test
  grid, raw mean reconciliation, budgets and every locked required guard. PASS.

The read-only report commands are:

```bash
uv run --no-project --with 'matplotlib>=3.9,<4' python scripts/technician_research_report.py /home/bachng/ht-pdm-fjsp/artifacts/technician_reservation_ppo_v4_confirmation_20261012T213632Z --output-root /home/bachng/ht-pdm-fjsp/artifacts
uv run --no-project --with 'matplotlib>=3.9,<4' python scripts/technician_reservation_evaluation_report.py /home/bachng/ht-pdm-fjsp/artifacts/technician_reservation_ppo_v4_confirmation_20261012T213632Z --output-root /home/bachng/ht-pdm-fjsp/artifacts --torch-python /home/bachng/ht-pdm-fjsp/.venv/bin/python
```

No further experiment is needed for the locked objective on this benchmark.
The remaining limitations above are explicit limits of that result.

To retrieve artifacts, run the canonical transfer on the Mac; the assistant
does not initiate it:

```bash
mkdir -p "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results" && \
rsync -avhP --partial \
  bachng@100.111.83.52:~/ht-pdm-fjsp/artifacts/ \
  "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results/"
```
