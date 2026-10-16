# Exchange PPO v2: frozen allocation diagnostic findings

The registered diagnostic completed all 608 episodes in 518.48 seconds, without
parameter updates, new seeds or scientific tests. It does not rescue the failed
pilot or authorize development. Its observations motivate hypotheses, not causal
claims about alternative assignments or preventable failures.

Source diagnostic:
`artifacts/technician_exchange_ppo_v2_allocation_diagnostic_diagnostic_20261016T012828951377Z`.
Frozen analysis commit: `d3512e1b97d5e118f7fd2d9dfac8b40114054d6b`.
Independent verification:
`artifacts/technician_exchange_ppo_v2_diagnostic_verification_20261016T013826578250Z`.

All original episode metrics reconcile, selected checkpoint parameters remain
unchanged, and every one of 36,480 physical transitions is independently
replayed. The separate verifier passes all seven output bindings, the exact
608-episode/29,184-decision grid, state chains, cost and availability accounting,
127,828 starts, trace applicability/aggregation and all aggregate values. There
are 32 paired initial-state panels. The diagnostic also verifies all 493 pilot
output bindings and frozen learning source/protocol/registries before and after
execution. Relevant correctness tests: 24 pass. Smoke: 20 episodes/1,200 ticks,
also independently verified.

## Where the N4 availability gap occurs

Values below are means over all 32 selected-checkpoint episodes per method.
Duration and restoration are means of episode-wise mean start descriptors;
they are not pooled-start averages or causal service effects.

| N4 method | Policy service ticks | Failed-waiting ticks | Availability | Policy failures/episode | Mean start duration |
|---|---:|---:|---:|---:|---:|
| Ordered commitment | 70.6250 | 1.84375 | 0.622559 | 6.71875 | 1.47659 |
| Normalized PPO | 64.0000 | 0.06250 | 0.666341 | 0.56250 | 1.02522 |
| Unrestricted exchange | 64.7500 | 0.40625 | 0.660645 | 3.62500 | 1.06390 |
| Immediate-inverse mask | 67.6250 | 1.75000 | 0.638672 | 4.53125 | 1.24251 |
| Self repair | 72.2500 | 2.75000 | 0.609375 | 5.06250 | 1.68370 |

Candidate unavailable ticks exceed normalized PPO by 6.625 service ticks plus
1.78125 failed-waiting ticks. Their sum, 8.40625/(4*48) = 0.0437825521,
exactly accounts for the availability difference. Candidate services also have
greater mean restoration (0.62561 versus 0.51261), so these observations do not
identify a universally dominated long-service choice or justify penalizing
duration alone.

The candidate leaves 14.40625 machine-decisions at risk per episode, versus
0.875 for normalized PPO. Its 6.71875 next-tick failures in those unserved
machines include only 0.0625 with a residual free legal technician. Lack of
residual capacity describes contention after the chosen allocation; it does
not prove that earlier alternative scheduling could not have prevented failure.

Both candidate replicas are retained. Their mean start durations are 1.59718
and 1.35601, with availability 0.617188 and 0.627930. One changes its construction
matching in only 1/48 decisions; the other changes it in 23.25/48. In the latter,
repairs reduce the static summed duration by 94.0625 per episode relative to
that episode's construction plans. This is a same-state broker comparison,
not the return of deploying those construction plans without repair.

## Cancellation removal does not explain quality

Candidate complete repair cancellation is zero, as designed. Yet unrestricted
exchange has only 0.15625 exchange-with-unchanged-matching decisions per N4
episode on this panel and achieves lower cost and better availability. This
does not establish a stable cancellation mechanism across rounds or show that
forcing every repair to persist is beneficial. The pilot's matched-control
tests reject the registered practical quality hypothesis.

## N32 needs timing and coverage evidence too

| N32 method | Policy failures/episode | Mean start duration | Policy service ticks | Unserved at-risk machine-decisions |
|---|---:|---:|---:|---:|
| Ordered commitment | 5.96875 | 1.00984 | 505.5625 | 13.65625 |
| Unrestricted exchange | 3.09375 | 1.00267 | 505.0625 | 6.87500 |
| Deadline | 1.75000 | 1.21449 | 538.3750 | 3.75000 |

Candidate failed-waiting ticks are zero at N32. Almost-unit service durations
and low occupancy therefore coexist with more failures than unrestricted
exchange or deadline. Duration alone cannot describe this gap. Of its 5.96875
failures, 0.65625 occur with a residual free legal technician. Such opportunities
are descriptive; they do not establish that serving a particular machine would
have prevented its failure or lowered total cost.

The two N32 candidate replicas change construction in 7.375/48 versus 48/48
decisions, yet their extended costs are nearly equal (8467.5625 and 8467.75).
An activity count is not a measure of useful coordination. No post-hoc best
replica, repair-activity threshold, duration penalty or relaxed health bound is
adopted. The next algorithm requires a new prospective hypothesis, seed panel
and matched mechanism tests; the v2 scientific panel remains unopened.
