# Coupled dispatch follow-up v1 — engineering gate

2026-10-18 UTC, before any new full development/test panel. Protocol and
matrix/seed registry were written before this smoke. This is a follow-up
motivated by known v1/v2 outcomes, not retroactive confirmation.

- Historical audit artifact:
  `artifacts/maintenance_shared_technician_history_audit_20261018T055849Z/`.
  PASS: v1 2,304 traced test episodes/138,240 ticks; v2 1,536/92,160.
  Git blobs, frozen source/protocol/seed hashes, all 60 arm-condition rows,
  physical/request/action/cleanup replay and cost reconciliation verified.
  No correction to old performance data required. Prior isolated lookahead
  often waits despite failed pending demand and idle eligible technicians;
  its six-tick zero-terminal-value DP has horizon-sensitive gain.
- New targeted tests: 47 passed. They include exact simple-arm equivalence to
  v1 mechanics, forecast-vs-physical replay with the same supplied tape,
  ongoing committed service, seed bank separation, candidate legality,
  request conservation and development-only baseline selection.
- Smoke:
  `artifacts/maintenance_coupled_dispatch_followup_v1_smoke_20261018T060347Z/`.
  COMPLETED, integrity PASS, 20/20 episodes, 1,200/1,200 trace ticks replayed,
  no infeasible assignment or lost request. 1.124 seconds CPU. Smoke selected
  FCFS for its *two* development configurations only; this choice does not
  select the full baseline. Both smoke configurations had proposal differences,
  resource contention and coupled action differences. Smoke costs were not
  used to change policy, matrix, seed, metric or success gates.
- Fixed full budget remains 480 development +960 test =1,440 episodes, four
  decision scenarios, 16 independent diagnostic scenarios at one test state
  per coupled episode, and a 7,200-second hard limit. Simple linear smoke
  scaling is ~81 seconds before the full scenario count, diagnostic and I/O;
  the hard limit has ample margin. No RL/checkpoint gate applies.
