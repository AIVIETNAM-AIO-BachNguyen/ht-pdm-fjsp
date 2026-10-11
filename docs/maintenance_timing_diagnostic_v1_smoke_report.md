# Maintenance timing diagnostic v1 — engineering gate

Material Passport: Origin Skill academic-research-suite / experiment-agent;
Origin Mode inline plan/run; Date 2026-10-11; Version
maintenance_timing_diagnostic_v1; Verification Status PASS (ENGINEERING ONLY).

No training or checkpoint reselection. No new full-panel outcomes opened on
Mac. The earlier joint-RL full result is an explicitly reused, previously
opened source dependency. This report makes no scientific claim about RL's
preventive timing or performance.

## Tests and actual-checkpoint smoke

- Complete repository: **625 passed**, one existing tqdm/rich warning,
  215.40 seconds. Focused diagnostic suite: 10 passed in 6.73 seconds.
- Smoke run: `maintenance_timing_diagnostic_v1_smoke_20261011T071239Z`,
  COMPLETED / audit PASS, 3.831 seconds.
- Expected/actual: first numeric frozen model, four trajectories, 16 physical
  trajectory intervals, eight pre-action states; 77 candidate roots and
  1,848 forecast transitions (four scenarios x six intervals per root).
- Every candidate/scenario branch on all eight actual smoke states matched
  the original scalar simulator. Feasibility and four physical-cost component
  reconciliation passed at every simulated interval. Carrying existing busy
  repairs into branches, keeping CM assignments when dropping PM, matched
  forecast tapes, and privacy from the physical trajectory's future were tested.
- All 169 source artifacts and all ten frozen source checkpoints verified.
  Source manifest SHA256:
  `d57fef493b559362b2d305af6c0acae1ed84f60246b05f7adfd69d8de362329b`.
- Regenerated 51,200 source training configurations; 52,145 unique prior
  physical signatures excluded. Own full/smoke physical metadata are disjoint;
  full metadata construction did not execute full states/outcomes.
- Save/load copied frozen models reproduced weights, root action, log
  probability and critic value. Parameters remained unchanged throughout.
- Independent post-run validation verified every output hash/size, raw NPZ
  shape (8,21,4), every CSV scenario-cost array, component sums, controlled
  PM removal, paired deltas, all counts and unopened-full flag.
- Synthetic-source integration tests reproduce complete semantic artifacts
  across repeat runs and retain partial states plus a FAILED manifest on
  interruption. Synthetic checkpoints are engineering fixtures, not results.

Smoke output contains 2,702,740 bytes of hashed artifacts, mostly dependency
snapshots. The console log and whole-test log are engineering records outside
the hashed run itself. Source snapshots match the implementation to be committed.

## Compute estimate from engineering states

A separate timing probe used ONLY the eight saved smoke states, with the full
fork budget (32 scenarios x24 intervals). It did not generate fresh full-panel
states. Cold per-state times were 5.43–19.22ms; warm median times 5.70–17.66ms,
for 1–21 feasible roots. Worst measured cold time x2,560 states is 49.2 seconds
for forks alone on this Mac. This is an extrapolation, not an upper bound:
lab CPU, rule caches, 30,720 trajectory ticks, logging and input/output add time.
Expect minutes rather than another training run; measure actual lab runtime.

Full fixed budget: all ten models, 640 trajectories, 2,560 states, at most
41,287,680 forecast transitions. Padded raw NPZ payload is at most 13.8MB before
compression, plus CSV records, copied models and snapshots. No GPU required.

## Interpretation and handoff

The controlled contrast removes current healthy-machine starts and preserves
all corrective pairs and their worker assignments. It measures root PM
deferral under a shared deadline rule continuation, not a permanent policy
change. The rule-root contrast also changes machine/worker allocation.
Finite-window values are not Q under the learned policy; the sample-best
root is optimistic/noisy. Both state origins, forced WAIT, exposures and paired
Monte Carlo SE remain in the artifacts. Four exploratory crossed-bootstrap
98.75% intervals are conditional on the fixed forecast/physical shock tapes.

Next gate: commit/push verified code, then user runs the new full diagnostic in
the Ubuntu terminal with the existing full source run. No assistant SSH,
full launch or rsync. Results must return to the Mac before scientific analysis.
