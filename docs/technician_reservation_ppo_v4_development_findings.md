# V4 completed development findings

DEVELOPMENT_ONLY. No scientific confirmation test has opened. The original
four-worker run completed its entire fixed budget and audit PASS. Its source
is the pushed learning design `a0e0493`; the raw artifact is
`artifacts/technician_reservation_ppo_v4_development_20261012T185104Z`.

| N/K | Reservation cost | Independent cost | Frozen PPO | Frozen MAPPO | Deadline cost |
|---|---:|---:|---:|---:|---:|
| 4/2 | 1055.3229 | 1063.4688 | 1054.7708 | 1084.1458 | 1170.5312 |
| 16/8 | 4220.3542 | 7887.6562 | — | — | 4387.1562 |

The entire locked development screen passes. Small cost is 0.052% higher
than PPO and 2.659% lower than MAPPO; this is no PPO-superiority claim.
At N16 cost/machine is 263.7721 versus deadline274.1973, a 3.802% point
reduction. Availability is .6692437 versus .6504720, pending/machine .0006510
versus0, failures/machine .1184896 versus .2304688. No cost or health confidence
interval is claimed from development.

| N | Student seed | Selected fine-tuning step | Cost/machine | Availability | Pending/machine | Failures/machine |
|---|---:|---:|---:|---:|---:|---:|
| 4 | 57100000 | 0 | 263.9062 | 0.6694336 | 0.0000000 | 0.1015625 |
| 4 | 57100001 | 0 | 263.9062 | 0.6694336 | 0.0000000 | 0.1015625 |
| 4 | 57100002 | 368640 | 263.6797 | 0.6684570 | 0.0000000 | 0.1015625 |
| 16 | 57100000 | 491520 | 265.6934 | 0.6696370 | 0.0000000 | 0.2285156 |
| 16 | 57100001 | 491520 | 263.1113 | 0.6693929 | 0.0000000 | 0.0820312 |
| 16 | 57100002 | 491520 | 262.5117 | 0.6687012 | 0.0019531 | 0.0449219 |

All selected reservation episodes have zero conflicts and rejected proposals.
All three N16 replicas select the final491520-step checkpoint, with cost/machine
262.51–265.69; each improves over its own warm start. N4 selects two warm
checkpoints and one368640-step checkpoint despite spending the full PPO budget.
Thus the development evidence supports scale adaptation and deployment transfer;
it does not establish beneficial fine-tuning at every setting.

The independent ablation has identical raw graph information, public identifiers,
remaining-worker order, capacity, initial weights, bank, augmentation and training
budgets. Removing reservation/prefix exchange causes substantial contention and
backlog in the completed N16 development comparison. This is evidence about the
whole coordination service, not an isolated attribution to attention, augmentation
or value scaling. Complete charts and replica min–max envelopes are in
`technician_reservation_ppo_v4_analysis_20261012T212644Z`; envelopes are not CIs.

Audit: 18 models,122880 training episodes,61440 paired training tapes,2112
development episodes,368640 teacher states,zero invalid assignments and zero
scientific test ticks. Every graph arm/bundle receives122880 teacher states,
3932160 pretraining presentations and491520 fresh PPO decisions per scale,
in addition to the existing foundation teacher with491520 original decisions.
Do not claim better sample efficiency than PPO/MAPPO from this experiment.

Select V4 using these completed development results and improved N16 replica
stability relative to V3. Preserve V1 negative development, V2 technical failure
and V3 passed-but-variable development. The timestamped design-selection record
binds all four manifests before scientific confirmation. Confirmation remains
round1/alpha.025 with the original fresh five bundles, four scales,256 test
episodes per model and all predeclared quality/mechanism/health/latency gates.
No scientific margin, endpoint or panel is changed by the execution amendment.
