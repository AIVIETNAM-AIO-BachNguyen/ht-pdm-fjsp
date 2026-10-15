# Completion-Credit v2 negative pilot

The complete pilot **does not authorize full development or confirmation**.
Eight of23 point screens fail (six nonpooled cost/health, two pooled mechanism
proxies). No scientific test has opened. Preserve every replica and checkpoint.
The actual validate_pilot prerequisite was called and correctly rejected this
artifact: `pilot cost/health screen fails`.

Run: `artifacts/technician_completion_credit_ppo_v2_pilot_20261015T071610Z`.
Learning commit0deba80a99438a880d05e0b436e83c76bfc1a98c, clean/pushed before start.
COMPLETE/auditPASS in7616.96s (126.95min),28 random fits61440decisions each,
1,720,320 decisions total,35,840 training episodes,5120 paired tapes,1344DEV
checkpoint episodes,0 scientific test episodes. Zero invalid assignments,
teacher or pretraining; exact budgets and optimizer counts, all replicas kept.

Independent report: `artifacts/technician_completion_credit_ppo_v2_pilot_report_20261015T121226860697Z`.
Verified all437 output SHA/size bindings and252 current source/PLAN/registry/history
bindings, reconstructed all selections/means, deadline panels, all23 point
estimates and screens. Plot visually checked: N4/N32 panels readable, all replicas,
shading min–max (not CI), all controls including normalized PPO and no-calendar.
Report code copied to `scripts/report_completion_credit_v2.py` outside learning SRC.

| N4 method | Cost/machine | Availability | Pending/machine | Failures/machine |
|---|---:|---:|---:|---:|
| Candidate |307.8750|0.637044|0.015625|1.726563|
| Original PPO |341.0000|0.576660|0.085938|1.703125|
| Normalized PPO |263.8281|0.666992|0|0.265625|
| Normalized local MAPPO |298.1875|0.670736|0|2.250000|
| Normalized full MAPPO |295.6719|0.673177|0|1.906250|
| State-credit graph control |288.0938|0.668294|0|1.476563|
| Independent graph control |286.8359|0.669271|0|1.468750|

Candidate beats equally short literal PPO by9.714% but loses to equally normalized
PPO by16.695%, local MAPPO3.249%, full MAPPO4.127%, state-credit6.866% and
independent7.335%. These fail their locked margins. Do not attribute improvement
against raw PPO to the coordination mechanism or claim scientific superiority.
Two candidate replicas cost294.0625 and321.6875; normalized PPO261.8594 and265.7969.
N4 system-health bounds vs literal PPO pass; that does not offset cost failures.

| N32 method | Cost/machine | Availability | Failures/machine |
|---|---:|---:|---:|
| Candidate |270.6650|0.671855|0.554688|
| Deadline |273.3203|0.649536|0.144531|
| State-credit |273.2637|0.674418|0.643555|
| Joint-credit |269.7002|0.670776|0.432617|
| No-calendar |270.3359|0.671163|0.502930|
| Independent |806.2793|0.248271|3.748047|

N32 cost gain vs deadline+0.9715%, availability difference+0.022319 and backlog
limit pass. Failure difference+.410156/machine violates+.25. Both candidate
replicas are stable on cost (271.0469 and270.2832) but both failure differences
exceed the limit; no bad-replica exclusion can fix the scientific requirement.
Independent gain66.43% cannot replace the absolute health/mechanism requirements.

N32-only mechanism proxies: credit vs state+.9510% passes its point proxy;
full recipe vs joint-.3577% and calendar vs no-calendar-.1217% fail+.5% targets.
These are exploratory, not the registered N8/N16/N32 pool or CIs. Candidate and
no-calendar performance is very close; no domain-calendar benefit established.
At N4 prefix credit is worse than the matched state-credit arm. Do not hide these
negative contrasts or claim isolated clipping improvements.

Selected first-DEV-observation critic probes show zero last-hidden saturation
for every graph/normalized baseline; output predictions now have appropriate
magnitude. Graph training value/return spread is nonzero and comparable in fixed
units. This supports that the measured v1 saturation symptom is absent in these
probes, not a general calibration or variance-reduction theorem. Different seeds
and physical namespaces prohibit treating v1/v2 mean cost differences as a paired
causal normalization experiment. In v2 original/normalized PPO are paired at
initialization/action/minibatch roles with identical architecture and physical
tapes, but pilot remains exploratory with two training replicas and16DEV seeds.

Next: frozen checkpoint trajectory analysis on every opened v2 DEV seed and
replica, including original/normalized PPO and calendar/credit controls. Diagnose
residual capacity, service duration/restoration and unserved risk/failure timing
under its separately locked plan. No optimization, new seeds or scientific tests.
Any revised algorithm requires a new prospective design, panels and validations.

Diagnostic validation before full analysis: syntax/diff checks PASS. Smoke
`artifacts/technician_completion_credit_ppo_v2_allocation_diagnostic_smoke_20261015T121440980117Z`
COMPLETE:16episodes,960 independently replayed physical ticks, all original
metrics reconciled, frozen parameters PASS,4.03s. Learning SRC/PLAN/registries
remain unchanged from the1158-test/14-fit smoke verified learning commit.
