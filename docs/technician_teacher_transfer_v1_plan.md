# Teacher-to-graph transfer v1: prospective protocol

Locked before any A1 development outcome. B1 prefix/mask-only development
is a separate hypothesis and panel; A1 does not use its result to choose
architecture, learning rate or physical configuration. Historical V4 TEST
remains historical evidence and is excluded from A1 confirmation.

## Falsifiable hypothesis and matched arms

The N4 centralized PPO teacher supplies useful assignment information to a
graph policy with sequential reservation at larger scales. Compare:

- `teacher_reservation_ppo`: the V4 graph/reservation architecture initialized
  by N4 PPO matching distillation, then PPO at each target scale.
- `scratch_reservation_ppo`: the **same architecture, parameter count, current
  full graph, prefix and reservation service**, initialized with exactly the
  same random tensors as the teacher arm before distillation, then PPO directly
  at each target scale. It receives no teacher labels or pretraining.

Both use 491,520 new PPO joint decisions per scale, identical rollout,
optimizer schedule/updates, action RNG role, physical training tape, DEV/test
physics, and checkpoint-selection rule. Teacher assistance has extra teacher
training, N4 bank and pretraining compute. Interpret a gain as assistance from
additional information/compute, **not** total compute efficiency. The five
frozen source PPO teachers are mapped cyclically by bundle index for the
five fresh confirmation students; source teachers are not newly fitted.
Report this reuse and leave-one-teacher/student-bundle-out sensitivity. No
large matching catalogue is used by student.

## Physics, splits and budgets

Use frozen V4 balanced physical/cost configuration, N/K=8/4, 16/8, 32/16,
48 policy plus 12 common tail ticks. Both arms use identical stochastic tape
within scale/seed. Distillation bank: 122,880 N4 state exposures per teacher
arm/bundle, 32 epochs, minibatch 512, Adam 3e-4, same V4 permutation
augmentation. PPO: 16 environments, rollout 3,072, 4 epochs, minibatch 256,
Adam 3e-4, clip .2, entropy .01, value .5, gamma 1, gradient clip .5. Eligible
checkpoints: initialization/warm at 0, then 122,880, 245,760, 368,640 and
491,520. Select minimum mean extended cost on fresh DEV episodes, earliest
tie; spend full budget even when an early checkpoint is selected. Record the
unselected **fixed-final** checkpoint and its health at 491,520. Learning AUC
is trapezoidal mean DEV cost/machine over these five fixed decision budgets,
divided by 491,520; lower is better. Also report per-checkpoint cost and
across-replica variance. A zero-shot condition evaluates the N4 distilled
policy at each larger scale **before PPO**. It is separate from fine-tuning.
Scratch initialization is reported at step 0; it is not called zero-shot
transfer. `train separate` here means scratch PPO at the target scale.

Fresh disjoint raw and role-derived seeds are in the companion registry.
Engineering smoke uses one bundle, N8, 192 bank states, two pretraining
epochs, 384 PPO decisions, 2 DEV and 4 engineering test episodes. Development
FULL uses three bundles, all three scales and 32 DEV episodes per checkpoint.
The DEV gate is selected-checkpoint pooled gain teacher vs scratch >=0.5%,
no scale selected-checkpoint point gain below -1%, positive gain in at least
two of three scales, zero invalid assignments, and no teacher-arm health
point difference worse than V4 bounds versus scratch (availability -.01,
pending/machine +.025, failures/machine +.25). If DEV fails, preserve its
verdict and do not open A1 TEST. Diagnose one bounded mechanism follow-up on
fresh seeds if justified. Confirmation, only after DEV gate, uses five new
student bundles, 64 new DEV episodes and 256 sealed TEST episodes. A1 has a
48-hour hard stop; timeout yields INCOMPLETE.

## Confirmatory estimands and gates

For each scale, `gain = 1 - mean(teacher selected cost/machine) /
mean(scratch selected cost/machine)`. Primary pooled gain uses equal scale
weight via the sum of the three cost-per-machine means. Require pooled point
gain >=1% and a one-sided multiplicity-adjusted lower CI >0. Also require no
scale point gain < -1% and strictly positive point gain in at least 2/3
scales. The registered inferential family has 13 endpoints: pooled selected
cost gain, three scale selected cost gains and three health differences at
each scale. Health reference is matched scratch on the same test tape; require
availability lower bound >=-.01, pending/machine upper bound <=.025 and
failures/machine upper bound <=.25 at each scale. Zero invalid assignments;
report p95 `model.select` and end-to-end latency, with p95 select <=50 ms at
N32. One-sided Bonferroni alpha .025/13 per endpoint, 40,000 crossed
bootstrap draws over paired teacher/student bundles and paired episodes
(episode axes independent across scales), fixed before test. Leave-one-bundle
point sensitivity addresses the small fixed teacher collection. Report every
student replica and selected checkpoint, plus fixed-final cost, learning
AUC/variance, zero-shot, health and compute for each scale. They are required
measurements and cannot be silently omitted if the selected-cost gate passes.

Artifacts: fresh timestamped directory with manifest/command, git/source/
protocol/registry/teacher hashes, per-phase wall time, source teacher costs,
bank and augmentation hashes, checkpoints, PPO decisions/updates, complete
training/DEV/TEST episodes, choices sealed before TEST, latency, scalar
physical replay, source and output bindings, independent verifier report.
`tqdm` in teacher bank, distillation, PPO and multi-seed evaluation. Tests and
small lab smoke, commit and push precede any FULL. No SSH or artifact transfer.
