# Temporal allocation opportunity audit v1 — fixed N4 diagnostic

Register before new finite-horizon expectation outcomes. Existing broker-marginal
FULL and frozen matching-mode diagnostic are negative; preserve their verdicts.
No teacher, distillation, training, checkpoint reselection, mask/physics change,
test-state opening, larger N or automatic decoder sweep.

## Question and fixed reference calculation

Hypothesis: evaluating a technician's assignment conditional on a co-worker's
accepted commitment reveals material short-horizon opportunity which isolated
per-technician scoring misses on the currently visited N4 states. Current conflict
counts alone did not establish that opportunity: only2/904candidate rejected
proposals had a feasible alternative in the opened frozen diagnostic.

Source: completed independently verified frozen mode diagnostic5391313,
`technician_broker_mode_diagnostic_v1_20261016T161807576637Z`; verification33a33ce,
`technician_broker_mode_verification_v1_20261016T162157684784Z`.
Use EXACTLY original greedy states from candidate and PPO, all3source replicas,
physical raw seeds300100000..300100003, ticks0/12/24/36:96root records,48per origin.
Keep duplicates with their original frequency. Context selection is fixed before
Q outcomes but source trajectories/performance are already opened; diagnosticDEV
only, conditional on these selected states. No new raw/derived RNG, seed panel,
forecast tapes, or new policy trajectories. Current scientific test remains sealed.

At each root compute exact expected H4 physical cost with zero terminal value:

    Q_h(s,a) = E_e[c(s,a,e) + V_(h-1)(s')],
    V_h(s) = min_(legal matching a) Q_h(s,a); V_0=0.

Use actual balanced_a2_b4_u12 physics, urgent legal mask, objective including
failure, service and overdue waiting. Integrate independent Bernoulli(.45)
failures exactly; marginalize irrelevant events. Enumerating21N4 matchings is
permitted ONLY in this diagnostic oracle. This is NOT an online/scalable policy,
not a proposed RL objective, not a new mathematical contribution, and not a
source of labels, pretraining or warm-start for future learned methods.

Compare four fixed root choices under SAME continuation V3:
- joint: exact min Q4 over all legal matchings;
- isolated: each technician chooses WAIT/machine minimizing Q4 for its SINGLE
  assignment with the other technician WAIT; resolve duplicates with original
  quality/lower-ID winner-only broker;
- conditional: T0 chooses its isolated token, then T1 chooses WAIT/an unoccupied
  legal machine minimizing Q4 with T0 fixed. T0 WAIT is a real prefix too.
  Fixed T0->T1 order, no outcome-based order selection or retry;
- source: original recorded greedy root matching.

Canonical token and matching order resolve exact ties. Share memoized Bellman
values, no Monte Carlo selection bias. Record isolated tokens, conditional tokens,
all four physical roots/Q4, joint argmin and state hashes. Save scalar expected
values, not an action-label training bank. No Q table is consumed by a learner.

## Locked screen

Primary population: all48candidate-origin roots, including unchanged/zero-gain
contexts. Aggregate as ratio of MEAN Q costs (not mean ratios):
1. joint gain vs isolated >=.005;
2. conditional gain vs isolated >=.005;
3. source gap to joint >=.005;
4. conditional gain strictly>0 in EACH of3source replicas;
5. at least6candidate roots strictly improve with conditional vs isolated
   (absolute Q difference >1e-8).

Require finite expectations, feasibility and independent Bellman checks. ALL five
must hold to motivate further design work on this conditional-Q mechanism;
not automatic authorization of a learning trial, not statistical confirmation.
If any fail, reject THIS short-horizon opportunity explanation; do not extend H,
change context subset, order, mask, cost or tolerance to rescue this screen.
PPO-origin48roots are descriptive replication only, not another chance to pass.
No CI/p-value. Dependencies, the narrow sample and finite horizon prevent
population or full-episode performance claims. H4zero-terminal optimization can
prefer decisions harmful after4ticks; health/fullH60benefit remains untested.

## Budget, checks and artifacts

Unit/integration tests on synthetic states and source ENG290700001 only. Sparse
failure integration must equal explicit16event scalar transitions, including
failed/busy/overdue/completion/restoration cases. H1/H2Bellman values compared to
independent exhaustive trees. Closed-form/direct cost components reconcile;
isolation/conditional matching/ties/forcedWAIT and gate boundaries checked.
Commit/push before full audit on lab; no separate smoke; tqdm across96roots.
One CPU thread,10minute hardstop,250000memoized Bellman evaluations hard cap.
On cap/error preserve FAILED/partial artifacts and report INCOMPLETE, never a
negative mechanism verdict. No automatic larger cache/horizon/budget rerun.

New timestamp directory; refuse overwrite. manifest includes committed code,
clean checkout, source+verification hashes, root specification, cache/runtime
counts, status/no-learning/no-test fields, source bindings before/after.
Store protocol/code/helper snapshots, selected recorded states, partial/final
root results, summary with all replicas and five screen checks, file hashes.
Independent post-run recomputation verifies schema, provenance, arithmetic and
screen; the exact expectation/tree correctness is checked before run by tests.
No source artifact mutation, SSH or automatic artifact transfer.
