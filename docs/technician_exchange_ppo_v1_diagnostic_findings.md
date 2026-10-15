# Frozen Exchange v1 allocation diagnostic findings

Artifact: `artifacts/technician_exchange_ppo_v1_allocation_diagnostic_diagnostic_20261015T175057179326Z`.
Prospective plan/report commit60d0662, verified diagnostic codefe2dcfc. Full
diagnostic COMPLETE in260.22s:480 episodes,23040 policy decisions,28800 physical
policy+tail transitions,92406 starts. Every original selected-DEV metric
reconciles, all policy tensors remain unchanged, no teacher/training/test occurs.

Independent post-check PASS: all7 output SHA/size bindings, script/plan identity,
exact method/size/replica/seed grid,32 paired initial physical states, complete
60-tick state chains, episode costs, availability decomposition, starts and
every aggregate. Trace fields are null for noncomparable arms, and all trace
totals reconcile with decision rows. This is exploratory opened-DEV analysis;
the negative pilot still rejects full development and confirmation.
Verifier source and results are preserved in
`artifacts/technician_exchange_ppo_v1_diagnostic_verification_20261015T180119944959Z`.

## Availability accounting and replica instability

| N4 mean per episode | Candidate | Self-repair | Normalized PPO |
|---|---:|---:|---:|
| Policy service machine-ticks | 68.6250 | 63.4375 | 63.5000 |
| Policy failed-waiting machine-ticks | 2.1250 | 0.1250 | 0.09375 |
| Availability | 0.631510 | 0.668945 | 0.668783 |
| Mean of episode mean start durations | 1.390794 | 1.045849 | 1.048276 |
| Unserved at-risk machine observations | 5.78125 | 9.84375 | 5.68750 |
| Subsequent failures of those machines | 2.75000 | 4.43750 | 2.53125 |

The candidate's additional5.125 service ticks and2.03125 failed-waiting ticks
sum to7.15625 unavailable ticks versus normalized PPO; dividing by4*48 explains
the observed availability loss0.0372721 exactly. This accounts for the loss;
it does not explain causally why the learned policy chooses those services.

Replica120400000 has mean duration1.072759 and availability0.661458;
replica120400001 has duration1.708829 and availability0.601563. Both are retained.
Repair changes the summed proposed service durations by-12.5625 ticks/episode
in the former and+8.75 in the latter, relative to each decision's construction
prefix at the same physical state. These are static allocation contrasts, not
counterfactual environment returns. A generic penalty on duration would be an
unsupported change of objective; the original reward and health gates remain.

No candidate N4 failed-at-risk machine retains a residual free legal technician
after the final allocation. This describes contention under that allocation,
not whether some alternative assignment could have prevented its failure.

## Trace activity versus net allocation

The candidate N4 policy changes its construction matching in only8.125/48
decisions on average. In17/48 decisions, at least one exchange occurs but the
final matching equals the construction matching. These counts are11.25 and
18.5625 respectively for replica0, and5.0 and15.4375 for replica1. Thus frequent
exchange events cannot be treated as evidence of useful physical changes.

At N32, the final matching differs from construction in47.90625/48 decisions;
exchange-with-no-net-change occurs in0.09375. Summed proposed service durations
fall by55.03125 ticks/episode after repair, with no mean change in assigned
cardinality or at-risk coverage at those same physical states. Mean final start
duration is1.006931. Matching changes can also exchange equivalent technician
identities; counts alone do not measure outcome improvement or causal mediation.

N32 candidate failures/episode4.96875 versus self-repair8.21875 and deadline
2.5625 remain consistent with the pilot's health/cost results. All candidate
failed-at-risk machines lack residual free legal technicians; unused legal
capacity is not the entire resource-allocation problem.

## Consequence for the next research hypothesis

The current mechanism is not validated: small-setting quality/health gates
fail despite the N32 point signal. Repeated revisions and cancellation provide
a testable representation hypothesis, not proof of the cause of those failures.
A next iteration can test whether committing already repaired technician
assignments reduces redundant repair traces while retaining all feasible final
matching support through construction. It must include the unrestricted
exchange policy as an equally informed, equally sized, equally budgeted control,
use new panels, and preserve the cost/health requirements. No current checkpoint
may initialize that new experiment.
