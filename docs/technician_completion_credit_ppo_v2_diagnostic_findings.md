# Frozen v2 allocation diagnostic findings

Artifact: `artifacts/technician_completion_credit_ppo_v2_allocation_diagnostic_diagnostic_20261015T121558337448Z`.
Analysis code582e5b2, immutable learning code0deba80. COMPLETE129.67s:480episodes
on every openedDEV seed/replica/method plus deadline. Original episode metrics
reconciled within1e-8; parameters unchanged; independently replayed28800 physical
policy+tail transitions. All seven diagnostic output SHA/size bindings rechecked;
complete480 unique episode grid verified. No optimizer, new seed, teacher or
scientific test. V2 full development and confirmation remain prohibited.

Metrics below refer to48 policy decisions unless H+L cost is explicit. Start
means average episode means; failures exclude the common12-tick deadline tail.
Residual legal means a currently feasible urgent edge remains after starts; it
is not proof that starting that edge would improve a counterfactual trajectory.

| N4 method | H+L cost/m | Mean start duration | Mean restoration | Unserved at-risk/episode | Failures/episode | Failures with residual legal edge |
|---|---:|---:|---:|---:|---:|---:|
| Candidate |307.8750|1.3756|0.5952|15.9063|6.9063|0.5938|
| Normalized PPO |263.8281|1.0337|0.5168|2.4063|1.0625|0.0313|
| State-credit |288.0938|1.0376|0.5122|13.9688|5.9063|1.5000|
| Independent |286.8359|1.0721|0.5337|13.5625|5.8750|0.9063|
| Deadline |291.6562|1.0195|0.5077|15.3125|6.4375|0|

N4 candidate replicas: seed80400000 duration1.0142/failures7.3125/cost294.0625;
seed80400001 duration1.7369/failures6.5/cost321.6875. Shorter service alone does
not explain normalized PPO's low failures: both normalized PPO replicas use
roughly1.03 ticks/start but have0.625/1.5 failures and261.8594/265.7969 cost/m.
Most candidate failures occur with no residual free legal technician
(6.3125/episode), not while leaving available capacity idle. Do not attribute the
whole gap to WAIT or infer a proven preventable count from different trajectories.

| N32 method | Cost/m | Mean duration | Mean restoration | Unserved at-risk/episode | Failures/episode | Failures with residual legal edge |
|---|---:|---:|---:|---:|---:|---:|
| Candidate |270.6650|1.0004|0.5002|39.9063|17.7500|3.0938|
| Deadline |273.3203|1.2108|0.6043|10.9375|4.6250|0|
| Joint-credit |269.7002|1.0087|0.5043|29.8125|13.8438|0|
| No-calendar |270.3359|1.0061|0.5030|34.8125|16.0938|2.1250|
| State-credit |273.2637|1.0049|0.5025|44.8750|20.5938|4.7813|

N32 candidate already chooses almost exclusively one-tick jobs:503.5938 short vs
0.2188 longer starts/episode. Deadline uses353.125 short and93 longer starts.
The candidate's higher availability/lower cost coexists with its failed health
bound; reducing duration further is not supported as a standalone solution.
Candidate has14.65625 failures/episode without residual free legal technicians,
versus3.09375 with a residual legal edge. Both replicas fail the failure bound:
18.625/16.875 failures versus deadline4.625, or+.4375/+.3828125 per machine.
A possible next mechanism must address coupling between which machine is
protected, service restoration and future technician availability. This is an
inference motivating research, not causal proof of why a revised algorithm works.

Independent N32 has96 conflicts and671.75 rejected proposals/episode, with
119.9375 failures, all with a residual legal edge. Reservation removes mechanical
collisions in all coordinated arms, but candidate does not outperform the joint/
calendar-removal controls or meet health requirements. Collision elimination
alone is insufficient evidence for the proposed completion/prefix mechanism.

## Research decision

Retire v2 as a failed pilot; no fullDEV, confirmation or longer continuation.
Do not promote the best replica, weaken the failure bound, claim a benefit of
prefix credit/calendar from the near-ties, or rescue with a PPO teacher.
Preserve original and normalized PPO comparisons: normalized PPO remains a
strong small-setting teacher-free reference, not a source of labels/weights.

The saturation engineering repair is supported descriptively, but the domain
mechanisms are not validated. Next research must investigate a different way to
coordinate coupled assignments and health risk, rather than only changing critic
scale, WAIT bias, or a duration penalty. Any alternative requires focused prior-
art review, a prospective mechanism hypothesis and matched controls, fresh
panels, tests/smoke/commit/push and a new scratch pilot. Scientific goal remains
unachieved and active. No new algorithm experiment has been registered here.
