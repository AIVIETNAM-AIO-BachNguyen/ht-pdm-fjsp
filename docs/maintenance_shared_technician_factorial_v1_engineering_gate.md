# Shared technician factorial v1 — engineering gate

2026-10-18 (Asia/Taipei), before full test panel. Protocol:
`docs/maintenance_shared_technician_factorial_v1_protocol.md`.

- Full repository regression: 998 passed, 1 pre-existing SB3 rich/tqdm warning,
  712.73 seconds. Targeted tests after the final trace fix: 6 passed.
- First 72-episode smoke completed, but traced only test episodes. A second
  smoke with trace replay failed because `pending_before` was written after a
  new failure event. The failed artifact is preserved at
  `artifacts/maintenance_shared_technician_factorial_v1_smoke_20261017T172515Z`.
  This was a trace ordering bug; episode dynamics were not changed.
- Corrected smoke: `artifacts/maintenance_shared_technician_factorial_v1_smoke_20261017T172539Z`.
  Manifest COMPLETED, integrity PASS, 72 unique episodes, 4,320 replayed ticks,
  72 cost/component/request reconciliations, no infeasible assignments.
  Duration 1.018 s on this lab CPU. The smoke is engineering evidence only.
- Locked full scope remains 2,880 episodes and 172,800 physical ticks, with
  2,304 test episode traces. Linear smoke extrapolation is about 41 seconds;
  allow I/O and analysis overhead. The prospective 7,200-second stopping limit
  is unchanged. No policy, condition, seed, metric or success gate was selected
  using smoke costs.
- No model/checkpoint exists because this is a deterministic, untrained
  factorial. Training and checkpoint save/load gates are not applicable.
