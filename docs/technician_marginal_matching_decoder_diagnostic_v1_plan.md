# Frozen N4 physical-matching decoder diagnostic v1

Registered on 2026-10-16 before any diagnostic episode. The parent full N4
marginal-matching development grid is still running, source f436ab6. Completed
candidate and latent-order DEV replicas currently select costs/machine
277.5078125/276.9140625/276.9140625 and
277.640625/276.9140625/276.9140625, respectively. The likelihood intervention's
point gain is ~0.016%, below the registered 0.5%. No scientific test is opened.
Do not stop or alter the parent grid. This diagnostic is conditional on the
entire parent development run completing with audit PASS and failing its
registered development screen. It cannot repair that round or authorize tests.

Question: does the candidate's two-greedy-plan decoder omit high-probability
physical matchings, or does the trained physical policy itself need redesign?
A second descriptive question concerns duration/restoration, contention,
availability, pending and failure patterns relative to fresh PPO/MAPPO baselines.
No causal attribution to duration alone is assumed.

## Fixed intervention, information and sample budget

Freeze ALL selected parent checkpoints for all nine methods and three train
replicas; retain every replica. No gradients, optimizer, teacher labels, actor
fine-tuning, checkpoint reselection or new scientific quality claims.

Evaluate the nine unchanged standard deployment policies. Also evaluate all
five core policies with an exact highest-physical-probability matching decoder.
At N4 ONLY, enumerate the 21 dense partial matchings, keep current legal ones,
score p0/p1 via recorded canonical traces and q=(p0+p1)/2 (p0 for fixed forward,
p1 for fixed reverse), and select maximum physical policy probability. For the
latent-order arm use q, not its training latent likelihood. Ties use catalogue
order, including WAIT; disclose order/label dependence. The auxiliary order bit
uses the larger component, or the fixed prior, and never changes physics.

This extra catalogue is a diagnostic privilege, not a scalable algorithm or
matching-cost oracle. It never uses realized future events or simulator
lookahead to select an action. The registered main decoder and training results
remain unchanged. At each standard/exact core state, record both decoder
proposals, true physical probabilities, matching disagreement, feasible support,
normalization residual and exact entropy of that small diagnostic distribution.
Inference uses float32 component/physical logs; score replay independently
recomputes the mixture in float64 with log/probability tolerance1e-6, and requires
the chosen index to match the recorded float32 argmax with catalogue tie order.
The full support must sum to one within2e-5. These are pre-episode engineering
precision checks, not quality margins.
Standard action selection never uses these measurements.

For each start record known duration/restoration, preventive/corrective type,
current resource availability and urgent support. Record final original cost,
availability, pending, failures, conflicts/rejections and full physical traces.
These diagnostics are descriptive; distribution differences do not prove that
service duration or restoration causes a quality gap.

Use exactly 64 NEW paired physical seeds 210000000..210000063, analysis metadata
seed210100000, unit/integration fixture seeds210200000..210200001. Raw new panels
must be disjoint internally and from all historical registries including parent
unopened confirmation. Keep the parent environment's physical-seed namespace
with these fresh raw seeds to reuse unchanged physics; pair physical population
hashes across every model/decoder. Source train seeds are frozen-model identities,
not new diagnostic training panels. Seed audit records this distinction.

Full fixed grid: 9 standard +5 exact decoder arms, x3 replicas x64 episodes,
plus64 deadline episodes =2752 episodes,132096 policy decisions,165120 physical
ticks including common12-tick tails. There are27 unique trained checkpoints,
42 model/decoder jobs plus1 deadline job. No early stopping by measured quality.
Eight CPU workers, one torch thread each. Global hard stop1800s, technical failures
abort siblings and preserve partial artifacts; no extension or partial-result
scientific interpretation. All timing under concurrency is descriptive only.
No standalone smoke launch per latest lab user override; correctness/integration
checks and commit/push precede the full diagnostic. Do not launch until the
parent pool and all parent validation have completed.

## Evidence and decision rule

Report paired point differences standard vs exact decoder for every core arm,
per-replica means, all raw samples, exact support/entropy/probability gaps,
aggregate service and health diagnostics. No confidence intervals or alpha spend;
scientific confirmation stays closed. Never declare the research goal achieved.
The result selects the subject of a separately preregistered next design:
substantial decoder disagreement with better exact-policy outcomes motivates
bounded decoding research; close agreement with similar bad outcomes points
toward policy representation/learning rather than an inference-only fix. This is
interpretation guidance, not a performance success gate or threshold search.

## Binding, independent audit and artifacts

Before episodes, require parent COMPLETED/development/audit PASS/testclosed,
exact registered full config and train/choice grids, failed development screen,
all consumed source/checkpoint/meta hashes and original selected parameter hashes.
Snapshot parent protocol/config/registry/summary/audits/choices/checkpoints plus
this diagnostic plan/registry/source. Match imported learning/env/policy files to
the parent source seals. Save initial and final parameter digests and require
unchanged tensors. Store source hashes and source manifest SHA256.

Fresh timestamped output directory only. Save manifest/resolved config/seed audit,
source snapshots, per-job episodes/decisions/coordination/decoder choices/starts,
merged logs, summary and independent audit with exact episode grids, population
pairing, zero invalids, reward-cost agreement, unchanged models, action decoding
and scalar replay of all60 physical transitions per episode. Progress via tqdm
for jobs/episode evaluation. Seal all inputs and final output hashes/sizes. A
FAILED diagnostic cannot be silently overwritten or treated as confirmation.
