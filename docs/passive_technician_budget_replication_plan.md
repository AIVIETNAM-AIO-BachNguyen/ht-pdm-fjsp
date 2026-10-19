# Passive technician budget replication

## Research question and hypotheses

Does the lower objective cost observed at 20,000 training episodes replicate
across fresh training seeds? The confirmatory hypothesis is that centralized
PPO has lower mean objective cost at 20,000 than at 5,000 episodes. The result
supports this hypothesis only when the two-sided 95% Student-t confidence
interval for the mean per-training-seed difference (20,000 minus 5,000) lies
below zero. Training seeds, not evaluation episodes, are the inferential units.

Independent Q-learning and independent PPO use the same budget checkpoints as
secondary replications. Their contrasts are descriptive and do not replace
the single primary centralized-PPO contrast.

## Locked protocol

The experiment reuses the three-machine/two-technician FIFO stress cell and
the fixed settings from the passive technician budget screen. It trains fresh
seeds 14--23, each for exactly 20,000 episodes, and saves checkpoints at
5,000, 10,000, and 20,000 episodes. Training seeds 11--13 belong to the
preceding screen and are not included in this replication's confirmatory
estimate.

All checkpoints are evaluated on the common development panel 101--200. This
panel was used by the preceding screen, so it is not a sealed generalization
panel. Seeds 201--300 remain closed and are reserved for a later locked
evaluation. Random-feasible and skill-aware FIFO policies are evaluated on
the same development panel as fixed references.

The primary metric is mean total objective cost. Secondary metrics are
failures, maintenance jobs, proposal collisions, waiting, and the across-
training-seed variation in objective. Training stops at the fixed episode
budget; no adaptive stopping, hyperparameter selection, or checkpoint
selection is performed.

The primary contrast is calculated per training seed after averaging that
checkpoint's objective over evaluation seeds 101--200. The reported 95%
Student-t interval uses the ten fresh training-seed contrasts. A lower
interval bound is not sufficient: the upper bound must be below zero to
support the hypothesis. If it crosses zero, the replication is inconclusive;
if the interval lies above zero, the hypothesis is contradicted for this
stress cell and seed panel.

## Smoke and full-run gates

The Mac smoke run uses the existing smoke profile, one training seed, three
evaluation seeds, and budgets 8, 16, and 32. It must complete all expected
rows, save and reload every checkpoint, produce the locked artifact schema,
show progress bars, and leave the sealed panel unopened. The GPU replication
is complete only when the manifest is `COMPLETED`, all expected rows and 90
checkpoints exist, the paired contrasts are present, and all checkpoint files
load.

## Artifact schema

Each run uses a new timestamped directory and contains `manifest.json`,
`benchmark_config.json`, `training_progress.csv`, `episodes.partial.csv`,
`episodes.csv`, `coordination.csv`, `budget_summary.csv`, and `summary.json`.
Model checkpoints follow
`<algorithm>/train_seed_<seed>/budget_<episodes>/model.(json|pt)`. The manifest
records fresh train seeds, development seeds, budgets, device, Git revision,
completion state, expected row/checkpoint counts, and that seeds 201--300
were not evaluated. The summary reports paired 20k-minus-5k objective
contrasts with training-seed count, per-seed deltas, and 95% confidence
intervals.

## Interpretation limits

This is a replication of the training-budget diagnostic, not a final
algorithm comparison. Results apply to this fixed stress cell and development
panel; no inference is made about sealed seeds or other production
configurations. A replicated budget effect does not explain whether the cause
is optimization, policy representation, or credit assignment. The fixed
heuristics remain reference policies, not causal controls.
