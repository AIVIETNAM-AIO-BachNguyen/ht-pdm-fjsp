# Prefix mechanism v1: prospective protocol

Locked before any B1 development outcome (2026-10-18 UTC). This is a bounded
mechanism experiment, not confirmation of all framework claims.

## Hypothesis and intervention

On balanced V4 physics, selected-edge prefix helps later heterogeneous
technicians allocate machines beyond duplicate prevention. Train three graph
arms from identical initial weights: `reservation_ppo` (full reservation and
selected-edge prefix), `mask_only_ppo` (identical reservation mask, feasible
resource summary, reserved fraction, fixed technician order, but prefix tensor
always zero), and `independent_ppo` (no reservation and no prefix). Each arm has
the same 82,819 parameters, current full graph, IDs, critic, teacher bank,
distillation presentations and PPO budget. No-reservation losing to full alone
cannot establish the prefix claim. The full/mask-only action set, at every
teacher-forced trajectory step with the same observation and valid prefix,
must be bit-identical; audit legal masks and physical validity.

Teacher is the five frozen N4 PPO checkpoints from the source benchmark, mapped
in index order. Count foundation PPO 491,520 decisions per teacher, N4 bank
122,880 states per bundle, 32 epochs of pretraining per arm, 491,520 PPO
decisions and optimizer updates per arm/scale, wall time and inference latency.
Same bank and augmentation RNG across the three arms. Same physical training
tapes and evaluation episodes within scale. Teacher is additional information
and compute, not free sample efficiency. Fresh seed roles in the companion
registry are disjoint from all historical registries.

## Stages and decisions

Engineering smoke: one seed, N4/N8, bank 192 states, two pretraining epochs,
384 PPO decisions, 2 DEV and 4 engineering test episodes. Full tests and smoke
must pass before code commit/push. Development FULL: three new bundles,
N4/N8, 32 DEV episodes per candidate checkpoint. Each arm spends 491,520 PPO
decisions/scale after the shared bank and 32 pretraining epochs. Eligible
checkpoints are warm/0, 122,880, 245,760, 368,640, 491,520; choose minimum
mean DEV extended cost, earliest tie. DEV gate: pooled gain full vs mask-only
over N4/N8 at least 0.5%, neither scale point gain below -1%, zero invalid
assignments, full availability no worse than mask-only by .01, pending/machine
no greater by .025, failures/machine no greater by .25. This gate only opens
confirmation, not a scientific claim. If failed, preserve the verdict and do
one mechanism diagnosis using train/DEV before proposing a fresh follow-up.

Confirmation, only after DEV gate and freezing the entire design/choices:
five disjoint training bundles, 64 new DEV episodes for checkpoint selection,
256 sealed TEST episodes, N4/N8/N16/N32. The test panel stays unopened until
the gate. No post-TEST reselection. Global hard stop 48h; timeout is INCOMPLETE,
never a positive result. Run directly on this Ubuntu lab after commit/push.

Primary gain uses mean extended cost per machine over the four scales and
bundles: `1 - mean(full cost/machine) / mean(mask-only cost/machine)`;
positive favors full. Requirement: pooled point gain >=.01 and multiplicity
adjusted one-sided lower confidence bound >0. Secondary per-scale gains must
be reported, with no scale point gain < -1%. Report every training replica.
Health reference is matched mask-only on the same test tapes: full minus
mask-only availability lower bound >=-.01; pending/machine upper <=.025;
failures/machine upper <=.25 per scale. Zero invalid assignments. Also report
full vs independent as a separate conflict-prevention contrast, plus central
PPO/MAPPO and deadline descriptively where available. For B, the family is
17 inferential endpoints: pooled cost, four scale costs, and three health
endpoints at each of four scales. Use crossed bootstrap over training bundles
and paired physical episodes, independent episode resampling across scales,
40,000 draws, one-sided Bonferroni alpha .025/17 for this first new mechanism
confirmation; interval lower/upper quantiles at alpha .025/17. Five bundles
limit coverage precision; do not substitute point estimates for CI.

Artifacts: fresh timestamped directory with manifest, git/source/protocol/
seed/teacher hashes, all banks/pretraining logs, checkpoints and choices,
training and DEV episodes, sealed choices before TEST, TEST decision/coordination
rows, summary/CI, per-replica and wall-time accounting, source and output
bindings, scalar replay and action-mask audits. Preserve negative and failed
runs. `tqdm` during bank, pretraining, PPO, DEV and TEST. No SSH or transfers.
