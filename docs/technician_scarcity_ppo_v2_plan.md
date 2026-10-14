# Opportunity-message PPO v2: prospective teacher-free protocol

Locked before v2 smoke/pilot/development outcomes. V1 pilot negative result is
preserved in docs/technician_scarcity_ppo_v1_pilot_findings.md and artifact
technician_scarcity_ppo_v1_pilot_20261014T023754Z. No v1 scientific test opened;
retain round2 alpha0.05/6. Do not use any old opened scientific seed panel.
Direct lab execution is explicitly authorized; tests/smoke/commit/push precede
research runs. Random initialization only: no teacher, demonstrations,
distillation, old policy checkpoint or cross-scale warm transfer.

## Hypothesis, mechanism and prior-art boundary

V1 unit-capacity demand prices were nearly zero and failed N32 health/quality.
Hypothesis: a receiver-specific loss of other machines' alternatives, weighted
by the service occupation time, supplies useful coordination information that
does not require demand to exceed an arbitrary one-start capacity threshold.
Matched intervention must outperform reservation-only beyond a prelocked0.5%
pooled large-setting cost margin. Independent coordination gains alone do not
establish this contribution.

Raw graph remains machine6/technician3/edge4/global9, H32, no added identities,
shared22-feature pair encoder and pooled machine/technician/global context.
Base learned edge bids b_mj and WAIT bids w_m are computed once per physical
decision. At each of K broker slots, exclude already reserved resources and
calculate a soft alternative value at temperature tau0.2:

    R_mj = tau * [logsumexp(b_m/legal crews,w_m)/tau
                 - logsumexp(b_m/legal crews except j,w_m)/tau]
    O_mj = sum_{u != m} R_uj
    price_mj = 0.5 * sigmoid(gate) * (service_time_mj/4) * O_mj/(1+O_mj)

The formula's logsumexp shorthand means logsumexp of scaled logits. Inclusive
and exclusive partitions use prefix/suffix logcumsumexp scans, O(N*K) per slot;
no K copies, matching enumeration, counterfactual simulator rollouts or future
events. Multiply by legal masks. WAIT always exists so values remain finite.
Messages are bounded [0,0.5]; learned scalar gate starts0 (strength0.25) in all
graph arms, adding one allocated parameter. Use price-adjusted bids to sample
an edge or STOP, reserve both resources, canonical STOP suffix, all actions
execute simultaneously. Exact joint latent trace likelihood and joint PPO ratio
are retained. Entropy/K, separate value encoder, N/4 output/value-loss calibration
remain as v1. Service duration is already available to every method, not a label.

Raw gradient diagnosis on closed v1 DEV showed critic norm dominance, but Adam
normalization means that is not proof of weaker parameter updates. Adopt a
controlled actor/value clipping intervention and log actual parameter-step norms.
Use one Adam step per minibatch, unchanged epochs/budget; clip actor and critic
groups separately at0.5 for candidate/reservation/independent. Record mean raw
actor/critic grad norms and actual actor/critic update norms each rollout.

Four graph arms have identical19781 allocated parameters, random initial tensors,
raw information, physical tapes, RNG roles, budgets and selection rules:

- scarcity_ppo: opportunity messages, reservations, separate gradient clipping.
- reservation_ppo: same broker/clipping, price messages disabled.
- independent_ppo: no messages/reservation, independent machine proposals,
  same separate clipping; conflict arbitration/drop identical to v1.
- global_clip_ppo: SAME opportunity messages/broker, original joint global
  gradient clipping. This isolates the adopted learning intervention.

Separate gradient clipping is a standard numerical choice, not claimed novel.
Autoregressive MARL/PPO, learned matching and auctions have prior art:
https://arxiv.org/abs/2205.14953 ; https://arxiv.org/abs/2109.11251 ;
https://arxiv.org/abs/2109.10380 ; https://arxiv.org/abs/2106.07877 .
No theorem of optimal prices, convergence, incentive compatibility or monotonic
PPO improvement is claimed. Novelty is provisional: the domain-specific bounded,
duration-weighted exclude-own opportunity mechanism and its controlled evidence.
Full current graph and shared broker are assumed; no local-only/network claim.

## Unchanged physics, budgets, splits and prerequisites

Same balanced scalar maintenance, N4/N8/N16/N32,K=N/2, H48+common deadline12,
reward=-cost/20,gamma1. Large deadline uses min-cost flow, no catalogue. N4 freshly
trains original PPO/local MAPPO/full-information MAPPO, none using old models.
PPO18966/local15364/full18052 parameters; graph19781. Capacity differences to
baselines disclosed; exactly matched between graph interventions.

All full fits491520joint decisions, vector16,rollout3072,epochs4,minibatch256,
lr3e-4,clip0.2,entropy0.01,value0.5,gradclip0.5. Checkpoints0/122880/245760/
368640/491520; mean DEV cost chooses earliest tie; whole budget spent and all
replicas kept. Small graphs/baselines7arms and each large4arms: dev57fits,
confirmation95fits. Fresh UTC artifacts, no overwrite, tqdm and exact budget
and zero-checkpoint regeneration audit. Per-run hard stop24h, never silently
extend or treat incomplete as success. No metrics-based early stopping.

Smoke N4/N32, one replica384decisions, vector2/rollout192/epochs2/mb48,
two dev/two engineering tests:11fits. Pilot N4/N32, two replicas61440,
vector8/rollout768/epochs4/mb128, checkpoints0/30720/61440,dev16, NOtests:22fits.
Development all four sizes/three replicas/full budgets/dev32, NOtests.
Confirmation all four sizes/five replicas/full budgets/dev64/test256.

Raw panels in configs/technician_scarcity_ppo_v2_seed_registry.json:
smoke64000000/dev64100000..1/engineeringtest64200000..1/analysis64300000;
pilot64400000..1/dev64500000..15/analysis64600000;
development64700000..2/dev64800000..31/analysis64900000;
confirmation65000000..4/dev65100000..63/test65200000..255/analysis65300000.
Audit actual physical hash function separately from learner RNG; all roles and
historical registries checked. Initialization shared across graph arms and sizes,
but never reuse a trained checkpoint. Training/test tapes paired by size across
methods, independent namespaces across sizes. All choices sealed before tests.
Confirmation requires audited, full-budget, passing, test-closed DEV artifact
with identical source/protocol/registry/core settings. Pilot alone insufficient.

## Thirty-five locked scientific endpoints

Round2 alpha0.008333333333333333; bootstrap200000, common replica weights
across sizes, independently resampled episode axes by size, methods paired
within size. Bonferroni35, individual two-sided confidence1-alpha/35. Five
replicas only: approximate coverage, all replica results/leave-one-out disclosed.

N4 gains vs PPO/localMAPPO/fullMAPPO/reservation/independent/global_clip(6)
plus three health differences vs PPO(3):9. Each large gains vs reservation/
independent/global_clip/deadline(4) plus three health differences(3):21.
Three large/N4 cost-per-machine ratios; pooled price gain vs reservation and
pooled clipping gain vs global_clip (each arithmetic mean of three large gains,
computed inside each bootstrap draw):2. Total35, no endpoint removed later.

Required N4 gains lower>=-1% except localMAPPO lower>0. Each large reservation/
global_clip gain lower>=-1%, independent gain lower>0, deadline gain lower>=-2%,
cost ratio upper<=1.10. REQUIRED MAIN price pooled gain lower>=0.5%.
Pooled clipping improvement lower>0 is optional; do not claim it without passing.
Optional PPO superiority requires gain lower>=1%; positive N4 price optional.
All margins from v1 retained; clipping noninferiority adds a stronger control.

At every size availability difference lower>=-0.01, pending/machine difference
upper<=0.025, failure/machine difference upper<=0.25 (PPOsmall/deadlinelarge).
Zero invalid assignments, neural p95<=50ms after train pool shuts down. Report
health tradeoffs within margins. No network timing claim; model.select timing
excludes graph construction, physical stepping and separate descriptive probes.
DEV screen same point thresholds, pooled clipping gain>=-1% additionally; all
required screen conditions before confirmation. No opened-panel reuse, best-seed
selection, retrospective threshold changes or teacher fallback.

## Validation, artifacts and reporting

Exact latent probability normalization/21 small physical matching support,
canonical STOP/no repeated resources, stable exclusive partitions and finite
gradients including singleton/all-busy, duration/own-exclusion response, gate
bound, permutation equivariance, matched counts/init, no future input/teacher,
no large catalogue, scalar60-tick replay, checkpoint provenance, seed freshness,
serial/spawn reproducibility, grouped clipping independence and logging-only
tensor equivalence,35-endpoint/null/missing-grid tests, prerequisite/corruption
rejection. Full suite and Linux smoke before commit/push and pilot.

Artifacts follow v1 manifest/config/source/registry/physical provenance, exact
train/dev/test journals, parameter/init records, zero/learned checkpoints,
sealed choices, scalar physical replay, bootstrap+health+latency summary and
output bindings. Record price magnitude/activity, same-weight unpriced matching
changes and actual gradient/update norms descriptively, no extra hypothesis tests.
Reports/plots fresh directories and source hashes; preserve every negative run.
No SSH or automatic transfers. Provide canonical rsync -avhP --partial for Mac.
