# Frozen Completion-Credit v1 allocation diagnostic

Exploratory opened-DEV analysis, no training, scientific tests, new learning seeds,
parameter updates or threshold changes. This does not authorize full development.

Artifact: `artifacts/technician_completion_credit_ppo_v1_allocation_diagnostic_diagnostic_20261015T063721938720Z`.
Analysis commit d796c58; immutable learning source remains 27d3e7f.
COMPLETE in 99.07s: every selected learned checkpoint, both replicas, both sizes,
all 16 opened DEV seeds, plus deadline references: 384 episodes. All original
metrics match within 1e-8; checkpoints unchanged. Independently replayed every
physical transition (23,040 ticks); all seven diagnostic output hashes/sizes
independently rechecked. No optimizer or teacher. Original pilot stays unchanged.

## Descriptive findings

All numbers refer to the 48 policy decisions unless explicitly H+L cost. Start
means average episode means. Policy failures exclude the common deadline tail.
Residual legal edges mean a feasible urgent edge remains after executed starts;
this is not proof that starting it would improve the overall objective.

| N4 policy | H+L cost/m | Availability | Mean start duration | Mean restoration | Residual-edge ticks/episode |
|---|---:|---:|---:|---:|---:|
| Candidate | 357.7578 | 0.557943 | 2.5296 | 0.8874 | 0.0000 |
| PPO | 309.4531 | 0.650716 | 1.2857 | 0.6303 | 7.0313 |
| Deadline | 290.9844 | 0.669271 | 1.0242 | 0.5099 | 0.0000 |

At N4 the candidate always exhausts currently feasible urgent assignments.
Its poor cost/availability therefore cannot be attributed to leaving a currently
legal edge idle on these selected trajectories. It makes 24.5625 starts with
service duration >1 versus 9.40625 one-tick starts per episode; PPO makes 13.0
long and 37.59375 one-tick starts. Candidate restoration is higher, yet its
availability is lower: greater restoration alone does not establish better
maintenance choices. Candidate failures (8.15625/episode) all occur on unserved
at-risk machines with no residual free legal technician; count is lower than
PPO's 8.9375 but does not compensate for cost/availability failure.

| N32 candidate replica | Cost/m | Availability | Mean duration | Mean restoration | Residual-edge ticks/episode | Failures with residual legal edge/episode |
|---|---:|---:|---:|---:|---:|---:|
| 70400000 | 271.9023 | 0.653442 | 1.1040 | 0.5184 | 0.0 | 0.0 |
| 70400001 | 613.2285 | 0.357381 | 2.8452 | 0.9762 | 20.0 | 32.0 |
| Deadline reference | 271.5527 | 0.649943 | 1.2080 | 0.6039 | 0.0 | 0.0 |

The weaker N32 replica combines longer service occupation, near-full restoration
and unused legal assignments. The better replica nearly matches deadline on
these opened DEV seeds, but excluding the weak replica is prohibited. Their
pooled availability/failure/cost screens fail. Neither association alone proves
why learning diverges or that a duration penalty/WAIT offset would solve it.

Independent N32 control averages 607.125 rejected proposals and 85.03125 machine
conflicts per episode. Candidate has zero conflicts/rejections due to exact
reservation. This establishes mechanical conflict prevention on the trajectories;
it does not establish the requested quality, health, or credit-learning benefit.
All three coordinated graph arms also have zero conflicts. Improvements against
the collision-prone independent control cannot replace the state/joint-credit
comparisons or the absolute health requirements.

## Next research decision

Do not run full development for this version. Preserve the negative pilot and
both replicas. Investigate learning of immediate service occupation versus future
restoration and the very small learned prefix-baseline differences before a new
prospective design. Any proposed mechanism needs a matched-information/capacity/
budget control, explicit prior-art boundary and new disjoint panels. Neither
blindly extending this budget nor switching to a teacher is authorized by this
negative result. The scientific goal remains unachieved and active.
