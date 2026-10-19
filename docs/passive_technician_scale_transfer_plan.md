# Passive technician zero-shot scale transfer

## Research question and falsifiable hypothesis

Does an entity-size agnostic machine--technician value decomposition preserve
maintenance performance when evaluated on larger systems and compatibility
graphs that were not present during training? The primary hypothesis is that
the relational edge encoder has lower mean objective cost than a pooled-set
control on each prespecified out-of-distribution size family at the final
20,000-episode checkpoint. A family supports the hypothesis only when the
paired training-seed mean difference (relational minus pooled-set) is below
zero; uncertainty intervals are reported and the four primary contrasts are
Holm adjusted.

## Environment and methods

All policies use the passive-technician FIFO simulator and the same objective.
Training uses the existing three-machine/two-technician in-distribution cell.
For evaluation, the source-size cell is retained as an in-distribution anchor.
Out-of-distribution cells contain 5 machines/3 technicians and 8 machines/4
technicians, each under a complete compatibility graph and a sparse graph
where each machine has one or two eligible technicians. Service times are
sampled deterministically from each evaluation seed and exposed in the
observation. The sparse generator guarantees every machine and technician has
at least one feasible edge.

Comparisons are Skill-aware FIFO, source-size standard QMIX, pooled Set-QMIX,
and Relational Set-QMIX. The two scalable learners share the same permutation-
invariant monotonic mixer, training budget, replay settings, and local
parameter-sharing; they differ in whether technician-edge values are composed
from separately encoded machine and technician sets or from an explicit
machine--technician pair encoder. The standard fixed-shape QMIX checkpoint is
an in-distribution reference and is not evaluated on incompatible tensor
shapes.

## Locked seeds, metrics, and stopping rule

Train seeds are 31--35. The common development panel is 501--550. The sealed
panel is 601--700 and remains unevaluated. Learned methods train for exactly
20,000 episodes and save checkpoints at 5,000, 10,000, and 20,000 episodes.
There is no adaptive stopping or checkpoint selection using development
results.

The primary metric is mean total objective cost per training seed, averaged
over the common development panel. Secondary metrics are failures, service
jobs, queue waiting, collisions, invalid requests, busy-technician requests,
and action diversity. The independent inferential unit is the training seed;
evaluation episodes are paired measurements, not independent replicates.

## Smoke and full-run gates

The Mac smoke uses one train seed, three evaluation seeds, and 8/16/32 episode
checkpoints. It must save and reload each checkpoint, evaluate all four
size/topology shifts plus source-size controls, write every expected row,
report zero invalid requests, show `tqdm` during training and evaluation, and
leave seeds 601--700 closed. Full completion requires every planned seed,
checkpoint, evaluation row, and audit to be present. A smoke pass is engineering
evidence only.

## Artifact schema

Each timestamped run contains `manifest.json`, `benchmark_config.json`,
`training_progress.csv`, `episodes.partial.csv`, `episodes.csv`,
`coordination.csv`, `budget_summary.csv`, `summary.json`, and learned model
checkpoints below `models/<method>/train_seed_<seed>/budget_<episodes>/`.
The summary includes per-seed means and paired relational-minus-set contrasts
for each test family.

## Interpretation limits

The experiment tests zero-shot scale and compatibility transfer on synthetic
systems. It does not establish dynamic technician availability, safety
guarantees, real-world validity, or universal superiority. The source-size
QMIX result is an anchor, not a size-transfer comparator because its mixer
input shape is tied to the training system size.
