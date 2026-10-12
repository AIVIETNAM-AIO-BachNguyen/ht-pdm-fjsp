# V3 completed development findings

This is development evidence only. No scientific confirmation test has opened.
The frozen run completed its entire budget and audit passed; meeting the
point-estimate screen is eligibility for confirmation, not a scientific claim.

Run: `artifacts/technician_reservation_ppo_v3_development_20261012T171225Z`.
Source commit: `2cc0249`. Read `summary.json`, `audit.json`,
`development_episodes.csv` and `development_reference.csv` in that directory.

| Setting | Reservation cost | Independent cost | Frozen PPO | Frozen MAPPO | Deadline cost |
|---|---:|---:|---:|---:|---:|
| N4/K2 | 1061.1354 | 1097.8125 | 1057.5521 | 1087.7188 | 1136.5938 |
| N16/K8 | 4469.6458 | 9678.5729 | — | — | 4383.2500 |

Small reservation cost is 0.339% higher than PPO and 2.444% lower than MAPPO.
Large reservation cost is 1.971% higher than deadline, just within the locked
2% development screen. All development screen flags are true. These figures
have no test confidence intervals and are not confirmation results.

Selected reservation replicas expose substantial variation:

| N | Student seed | Fine-tuning checkpoint | Cost/machine | Availability | Pending/machine | Failures/machine |
|---|---:|---:|---:|---:|---:|---:|
| 4 | 55100000 | 245760 | 270.1172 | .6700846 | 0 | .6406250 |
| 4 | 55100001 | 122880 | 263.3828 | .6674805 | 0 | .2500000 |
| 4 | 55100002 | 491520 | 262.3516 | .6669922 | 0 | .1953125 |
| 16 | 55100000 | 0 | 270.1582 | .6672770 | .0019531 | .3789062 |
| 16 | 55100001 | 245760 | 272.1113 | .6634521 | .0039062 | .4101562 |
| 16 | 55100002 | 0 | 295.7891 | .6195068 | .0097656 | .4570312 |

N16 deadline availability is .6499023, pending/machine .0019531 and
failures/machine .2207031. Reservation averages are .6500787, .0052083 and
.4153646 respectively. Availability of the third replica is weak despite the
average screen passing. Reservation has zero conflicts and rejected proposals
in all selected development episodes. Fine-tuning generally degrades the
large-graph average; two N16 replicas select their warm start even after the
entire PPO budget was spent. This supports investigating transfer robustness
and critic calibration, not claiming successful reinforcement improvement.

Audit: 18 models, 122880 training episodes, 61440 paired training tapes,
2112 development episodes, zero invalid assignments, zero scientific test
physical ticks. Three banks total 368640 teacher state exposures. Per graph
arm/student bundle: shared bank 122880 states, 3932160 pretraining
presentations, 491520 fresh PPO decisions per scale. The foundation PPO also
had 491520 original decisions. The added teacher information and compute
preclude a fair sample-efficiency claim over the frozen PPO/MAPPO references.
The independent ablation matches the additional information and budgets.

V4 was prospectively locked and pushed before its outcomes, using partial V3
DEVELOPMENT feedback. It continues its fixed budget on a fresh seed registry.
Keep V3 tests closed while comparing the completed development evidence. Prefer
only a frozen, audited design with stable quality and health for the first
scientific confirmation round; all original margins and alpha rules apply.

Plots are generated read-only by `scripts/technician_research_report.py` into
fresh UTC analysis directories. Shading is replica min–max, not a confidence
interval. The report helper labels teacher diagnostics with actual student
seeds in sorted order and records its own SHA256. Those are presentation-only
changes after this run completed; no learner, protocol or raw run is changed.
