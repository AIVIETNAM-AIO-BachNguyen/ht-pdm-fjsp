# Teacher-free opportunity PPO v3: negative full-development findings

Run `technician_scarcity_ppo_v3_development_20261014T072525Z` completed on the
Ubuntu lab at 2026-10-14 15:05:17 UTC, audit PASS, in 27590.74 seconds
(7 h 39 min 51 s). Frozen run commit
`4a1928fbffa5a7b2ce886800e5a8fcf1289e1dd7`; learning source commit `c4f2eb2`.
Classification is DEVELOPMENT_ONLY. No scientific tests or alpha spent.
**Development screen FAIL. Confirmation is prohibited under the locked plan.**
Keep this run and all earlier negative versions intact; do not loosen any gate
or use a teacher to rescue the result.

## Full budgets and provenance

All 57 fits completed from random initialization, each with 491520 joint
decisions, 7680 optimizer steps, 10240 training episodes and 614400 physical
ticks. Total 28016640 joint decisions, 583680 training episodes, 122880 paired
training tapes and 9120 checkpoint-development episodes. All three replicas and
all five checkpoints, including step0, are retained. Selection uses mean DEV
cost only, with earliest ties. No pretraining, teacher, old checkpoint or invalid
assignment. Audit independently checks initialization, the full training and
DEV grids, physical seed/tape pairing and budget/replay contracts. Scientific
physical test replay count is zero because this run opens no test panel.

The read-only mechanism report rehashed all 716 manifest-bound files, checking
their sizes too. Current 195 source files and 48 protocol/registry bindings match
the frozen run. Report-only commit `78276b4` changes wording and stage guidance
outside the frozen learning source, plan and registries.

## Selected development cost and health

Each learned method below averages all three replicas and 32 DEV episodes per
replica at its selected checkpoint. These are point estimates, not confidence
intervals or scientific performance claims.

| Machines | Method | Cost/machine | Availability | Pending/machine | Failures/machine |
|---|---|---:|---:|---:|---:|
| 4 | proposed hierarchical opportunity PPO | 282.4401 | 0.670085 | 0.005208 | 1.132813 |
| 4 | hierarchical reservation, no prices | 282.6745 | 0.671061 | 0.005208 | 1.153646 |
| 4 | independent PPO | 284.4635 | 0.672743 | 0.007813 | 1.359375 |
| 4 | flat STOP, same prices and broker | 272.3906 | 0.669596 | 0.002604 | 0.606771 |
| 4 | original centralized PPO | 262.2839 | 0.669325 | 0 | 0.156250 |
| 4 | local MAPPO | 280.1042 | 0.666016 | 0.020833 | 1.164063 |
| 4 | full-information MAPPO | 272.0078 | 0.665310 | 0.002604 | 0.658854 |
| 8 | proposed | 269.6341 | 0.669325 | 0.002604 | 0.425781 |
| 8 | no prices | 269.8164 | 0.670763 | 0.006510 | 0.462240 |
| 8 | flat STOP | 268.5130 | 0.669651 | 0.002604 | 0.388021 |
| 8 | deadline | 267.3828 | 0.663656 | 0 | 0.339844 |
| 16 | proposed | 267.4004 | 0.671102 | 0 | 0.341797 |
| 16 | no prices | 267.4004 | 0.671102 | 0 | 0.341797 |
| 16 | flat STOP | 267.2233 | 0.671075 | 0 | 0.328776 |
| 16 | deadline | 273.6523 | 0.650838 | 0.003906 | 0.208984 |
| 32 | proposed | 263.8193 | 0.671285 | 0 | 0.148438 |
| 32 | no prices | 264.0146 | 0.671048 | 0 | 0.151693 |
| 32 | flat STOP | 263.8193 | 0.671285 | 0 | 0.148438 |
| 32 | deadline | 271.4854 | 0.650289 | 0 | 0.077148 |

Seven of the 35 registered point screens fail:

| Endpoint | Estimate | Locked DEV requirement |
|---|---:|---:|
| N4 gain vs PPO | -7.6849% | >= -1% |
| N4 gain vs local MAPPO | -0.8340% | > 0 |
| N4 gain vs full MAPPO | -3.8353% | >= -1% |
| N4 gain vs flat STOP | -3.6894% | >= -1% |
| N4 failures/machine difference vs PPO | +0.976563 | <= +0.25 |
| pooled large price gain vs no-price reservation | +0.047180% | >= +0.5% |
| pooled large cardinality gain vs flat STOP | -0.161262% | >= +0.5% |

N4 availability difference is +0.000760 and pending/machine difference
+0.005208, both within their locked bounds; this does not offset the failed
failure and cost bounds. The promising short pilot did not predict full-budget
small-setting success. The candidate selects trained steps 368640, 122880 and
245760, so this negative result is not merely random checkpoint selection.

All large-size cost and health point screens pass. Candidate gains vs independent
PPO are 23.69%, 31.55% and 59.08% for N8/N16/N32. Gains vs deadline are -0.842%,
+2.285% and +2.824%. Candidate failure differences vs deadline are +0.085938,
+0.132813 and +0.071289 per machine: within bounds but increases, not improvements.
Large/N4 cost-per-machine ratios are 0.95466, 0.94675 and 0.93407.

## Mechanism and scaling interpretation

The four graph arms have identical inputs, 19781 allocated parameters, paired
initial tensors/physical tapes/RNG roles and training budgets. PPO/local/full
MAPPO parameter counts differ as disclosed in the locked plan. All policies use
the same available current state; the candidate has no future or teacher input.

The reservation broker eliminates conflicts by construction. Selected independent
controls average 57.41/80.98/97.73 conflicts per episode at N8/N16/N32, with
81.81/148.14/882.54 rejected proposals. All three broker arms have zero conflicts
and rejected proposals. Beating independent agents therefore supports the value
of reservation coordination in this development setting, but does not establish
a novel advantage of the proposed prices or hierarchy.

Price ablation gains by large size are 0.06756%, 0% and 0.07398%; hierarchy gains
vs flat STOP are -0.41752%, -0.06627% and 0%. Equal selected aggregate costs at
N16 (candidate/no-price) and N32 (candidate/flat) are retained, not rounded into
a positive mechanism claim. Same-weight unpriced matching changes remain
descriptive: they can change assignment order without a useful physical effect.
Actual actor update norms are about 0.0079–0.0085 for the candidate; raw critic
norm dominance under Adam is not evidence of suppressed actor parameter steps.

Learned decisions avoid matching enumeration, with O(NK²) work and fixed parameter
count; each scale is separately trained. Full shared current graph and broker are
assumed. No local-only communication or zero-shot scaling claim is supported.
Recorded selected DEV mean decision times are approximately 2.67/5.16/9.60/17.96 ms,
under training-pool load; they are not isolated p95 latency verification. The
registered confirmation p95 gate remains untested because confirmation is closed.

## Preserved reports and next diagnostic

`technician_scarcity_ppo_v3_development_report_20261014T150616Z` contains bound
tables and learning-curves PNG/PDF. PNG visually inspected: four readable panels,
all-replica min–max shading explicitly labeled as not CI.
`technician_scarcity_ppo_v3_mechanism_report_20261014T150619Z` rechecks all bound
files and retains gradients, actual parameter steps, gates and every checkpoint.

Before designing another prospective version, diagnose selected N4 policies on
already opened DEV episodes: preventive timing, STOP/idle opportunities and
technician assignment, compared with original PPO and the matched controls.
Do not train or open unused test seeds for this diagnosis. A revision needs a
new prelocked hypothesis, panels and validation; the full research goal remains
unachieved by v3.
