# Frozen first-technician intervention: registered diagnostic procedure

POSTHOC DEVELOPMENT DIAGNOSTIC. Register before computing intervention returns.
The factorization FULL remains negative (2/16); no confirmation, new training,
teacher labels, larger sizes or new scientific test observations are authorized.

Source: technician_residual_capacity_factorization_v1_development_20261016T064203318666531Z,
learning a10e049e3f8b4e242efbc879b2d3dc3c13925bf4, COMPLETED/audit PASS.
Prior frozen-state diagnostic found nine candidate T1 starts lasting four ticks
and four for matched flat on the same 144 states; the same-prefix T2 GO explanation
failed its eligibility gate. Duration alone does not establish a harmful action.

Question: under the frozen candidate continuation, do the candidate's long T1
commitments cause higher remaining episode cost than deferral or a short option?
This is a controlled simulator action intervention, conditional on this policy,
these three DEV trajectories and their realized event tapes; not population or
optimal-policy evidence. Future events are available ONLY to the evaluator.

## Fixed corpus and interventions

Exactly the selected candidate checkpoints for train replicas 230000000..2,
physical DEV seed 230100000, all 48 saved states per replica. Verify all 548
source bindings and seals before/after; no weights or source artifacts change.
Replay each original episode with its candidate and compare every observation
bit-exactly against the saved arrays and final cost/health against the selected
source DEV row (absolute tolerance 1e-8). Abort on a mismatch, never repair data.

At every state copy the entire simulator, including accumulated metrics and the
same exogenous event tape. Evaluate exactly four branches, keeping all states:

1. Original candidate T1 token (no-op control).
2. WAIT for T1 only.
3. Shortest currently legal T1 service; tie by lower machine index; if no option,
   WAIT. This is an untrained diagnostic rule, not an optimal-action label.
4. Matched flat policy's T1 token on the same physical state.

Recompute T2 using the candidate's OWN model and the actual intervened T1
reservation/calendar, then resume the same candidate greedily for every later
decision. Change exactly one first-technician token at the chosen decision;
never continue with flat or repeat an intervention later. Preserve the existing
12-tick common deadline tail and episode cutoff. Interventions must respect the
original legal support. The no-op must equal the untouched greedy action, every
remaining source observation and all final source metrics.

This creates 576 replay branches, 144 no-op and 432 interventions, plus three
source-prefix replays, with no fresh physical seeds, learning updates or tests.
Report counterfactual branches separately from independent physical episodes.
Duplicate/no-op alternative tokens remain recorded, not dropped to inflate gain.
Do not construct training banks, oracle imitation targets or warm starts.

## Metrics, interpretation and stopping

Record every full intervention action, original/alternative T1 duration, final
extended cost, availability, pending and failures. For each branch compute paired
candidate-minus-alternative cost /4 (positive means intervention helps), and
alternative-minus-candidate health changes. Report all-state, original-T1-four-
tick, other-positive-T1 and original-WAIT strata, pooled descriptively and per
replica. States within episodes are dependent: no CI, p value, independent-sample
claim or extra alpha allocation. Never select only positive intervention states.
Also retain losses and zero effects. This does not refute or confirm broad capacity
information usefulness and cannot unlock the failed FULL's confirmation gate.

A specific deferral explanation is descriptively supported on this corpus only
if at least six original-four-tick contexts spanning all three replicas exist,
WAIT reduces mean cost by at least 1 cost/machine in that stratum, every replica
has positive mean gain, and pooled availability decreases by at most .01,
pending increases by at most .025/machine and failures by at most .25/machine.
This prelocked engineering signal can motivate a separately registered mechanism
test; it does not authorize a duration penalty or another train automatically.
Other alternatives are descriptive, without a selection gate.

Hard stop 10 minutes, torch/BLAS one thread; tqdm over replicas/states. New UTC
timestamp directory, no overwrite. Tests cover forced-prefix legality and exact
restoration, no-op equivalence, genuine T2 reservation recomputation, simulator
copy isolation, full tail, aggregate signs/strata/thresholds and missing grids.
Commit/push protocol and tested code before execution. No separate smoke.

Artifacts: manifest with hashes and input/code/protocol bindings; protocol/code
snapshots; 576 branch rows; 144 replay controls; summary with every replica and
fixed engineering verdict; report. Any technical mismatch marks FAILED. No SSH
or automatic transfer. Manual result copy uses rsync -avhP --partial.
