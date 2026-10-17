# Shared technician exploratory v2 — engineering gate

2026-10-18 (Asia/Taipei), before opening v2 test. V2 protocol is in
`docs/maintenance_shared_technician_exploratory_v2_protocol.md`.

- Relevant tests: 21 passed, including v1 compatibility, v2 service matrices,
  seed separation and physical request/dispatch semantics. Compilation and
  Git whitespace checks pass. The repository-wide regression passed 998 tests
  before the v1 run; the v2 change is a parameterized runner plus one new
  condition module, and targeted regression covers the touched path.
- Smoke artifact: `artifacts/maintenance_shared_technician_exploratory_v2_smoke_20261017T172903Z`.
  Manifest COMPLETED, integrity PASS; 48/48 unique episodes and 2,880/2,880
  decision ticks independently replayed. No checkpoint is applicable.
- Smoke duration 0.672 s for 48 episodes; the prospective full scope remains
  1,920 episodes, 115,200 ticks, with a 7,200-second hard limit. Linear
  extrapolation is about 27 s before I/O/analysis. No smoke cost was used to
  change conditions, policies, seeds or hypotheses. Results remain exploratory.
