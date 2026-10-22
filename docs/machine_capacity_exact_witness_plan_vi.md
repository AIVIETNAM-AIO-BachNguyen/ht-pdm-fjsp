# Exact small-state capacity witness: prospective manual diagnostic

Date: 2026-10-22. This is a manually constructed mechanism check, not an
episode-distribution estimate, training run, or evidence of requester value.
The source kernel and economic costs are unchanged. Results must be reported
whether or not the proposed witness discriminates.

## Hypothesis and estimand

At the same target-machine health, a different current competing CM queue may
change the cost-minimizing *coordinator service start*. The estimand is the
exact expected cost-to-go of each legal first matching under optimal matching
continuation for a six-tick synthetic N2/K1 problem. Failure events are
enumerated under the existing Bernoulli law; no realized future tape is given
to a policy. This tests current resource competition, not whether delaying a
free PM request is useful.

## Frozen fixture

Use the existing `DispatchConfig` and `maintenance_requests.transition` with
N2/K1, horizon 6, failure age 4, failure probability 0.45, max age 8,
service durations machine0=3 and machine1=1 for the sole compatible worker,
restoration 0.75 on both edges, and default PM/CM/unavailability/failure costs.
At t=0 both states have idle worker, target machine0 operating at age 3 with
a queued PM request created at 0. State A has machine1 operating at age 0 with
a queued PM request created at 0. State B has machine1 failed at age 0 with
an automatic queued CM request created at 0. Both have pending_wait=(0,0),
remaining=(0,), assigned=(-1,), generations=(1,1). Each policy receives all
current requests; whenever a service completion leaves an operating machine
request-free, that machine submits PM at the next boundary. Matching choices
include WAIT. Request timestamps have no cost or tie preference.

At each state/action/tick, enumerate all four two-machine Bernoulli event
vectors, weighting by p=0.45 independently. Apply the source transition and
recurse over legal matchings to horizon 6 with zero terminal value. Memoize
states only for exact computation. Tie-break equal expected costs by the
lexicographic legal matching order. No alternate states, horizons, costs,
durations, or seeds will be tried after seeing the result. The scenario is
fully synthetic and has no training/evaluation seed role; the four events are
integrated exactly rather than sampled.

Pass the *witness* hypothesis only if the selected first matching contains
machine0 in one state but not the other, and the strict Q gap from the
runner-up exceeds 1e-9 in each state. Also report all first-root Q values,
physical states, branch count, memoized-state count and runtime. If this
gate fails, report the negative result and do not tune the fixture into a
positive anecdote. The structural always-request result is unaffected.

## Execution and artifact contract

Run with the existing source and a bounded 120-second CPU timeout. Write to a
new timestamped directory only: `manifest.json` with RUNNING/COMPLETED/FAILED,
commit and source/protocol SHA256, fixture and terminal status;
`first_roots.json` with exact Q values/actions and selected roots; and
`summary.json` with the locked witness verdict and counts. Preserve a
`protocol_snapshot.md` matching the manifest hash if this plan is later
amended. An independent table/hash check can confirm artifact integrity but
does not replace an independent implementation of the physics or establish
population performance. No full DEV, replication or sealed TEST opens here.
