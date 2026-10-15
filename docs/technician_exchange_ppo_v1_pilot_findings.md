# Exchange PPO v1 negative pilot findings

Frozen learning commit `3a96ecb`. Pilot
`artifacts/technician_exchange_ppo_v1_pilot_20261015T130233Z` completed in8995.40s
(2h29m55s), audit PASS:28 random-initialized fits,1720320 decisions,35840 training
episodes,5120 paired training tapes and1344 checkpoint DEV episodes. Zero
teacher/pretraining/invalid assignments; scientific test remains closed.

Independent report
`artifacts/technician_exchange_ppo_v1_pilot_report_20261015T173812655741Z`
rehashes all441 pilot output bindings and256 current input bindings, reconstructs
all checkpoint selections/means/25 point screens, and retains both replicas.
All five report output bindings and the report script identity were rechecked.
Learning-curves PNG visual QA PASS. These are DEV comparisons, with no
scientific confidence interval or alpha spent.

## The registered gate fails

Six of25 point screens fail. The actual `validate_pilot` call rejects this
artifact with `pilot cost/health screen fails`. No Exchange v1 full development
or confirmation is authorized by this pilot.

| Failed endpoint | Observed | Registered requirement |
|---|---:|---:|
| N4 gain vs self-repair | -4.2524% | >=-1% |
| N4 gain vs normalized PPO | -8.5354% | >=-1% |
| N4 gain vs local MAPPO | -4.8681% | >0 |
| N4 gain vs full-state MAPPO | -1.5671% | >=-1% |
| N4 availability difference vs original PPO | -0.020508 | >=-0.01 |
| N4 availability difference vs normalized PPO | -0.037272 | >=-0.01 |

N4 candidate cost/machine292.6641 versus normalized PPO269.6484,
self-repair280.7266 and local MAPPO279.0781. Candidate availability0.631510
versus normalized PPO0.668783. The descriptive4.6041% cost gain over literal
original PPO306.7891 cannot support success: health fails and stronger normalized
PPO remains better. All methods received the equally short registered pilot
budget; these are not comparisons against full-budget fitted policies.

Both candidate N4 replicas select61440 decisions. Their costs269.6250 and
315.7031 and availabilities0.661458 and0.601563 reveal large between-replica
differences. Keep both; selecting the favorable replica would not repair the
failed gate.

## Large-setting signal is encouraging but exploratory

At N32, candidate cost/machine263.6592 versus deadline270.7852 (2.6316% gain),
self-repair265.0039 (0.5074% gain) and construct-only265.1563 (0.5646% gain).
All N32 point cost/health screens pass. Availability0.671427 versus deadline
0.651489; pending/machine0 for both; failures/machine0.155273 versus0.080078.
The failure difference+0.075195 passes the locked+0.25 ceiling, but is not a
failure reduction relative to deadline.

Both N32-only mechanism proxies exceed0.5% at the point-estimate level. They do
not measure the registered N8/N16/N32 pool, carry no scientific interval, and
cannot override the small-setting failures. Large-scale latency and scientific
confirmation requirements are still unproven.

## Next diagnostic question

The selected policies actively use exchanges:39.25 exchange events/episode on
average at N4 and506.9375 at N32. The counter includes transfers as a subset.
Activity is not proof that the final matching changes or that a change helps.
Inspect construction versus final allocations, service duration/restoration,
unused legal capacity, at-risk machines and availability decomposition using
the frozen selected models and already opened DEV panel. This is descriptive
diagnosis of a negative result, not a new performance test or training rescue.

Preserve the pilot, report, thresholds and unopened scientific panels. Any new
learning mechanism requires a separate prospective protocol and fresh panels.
