# Maintenance coordination RL research

All new goal outputs live here. Each smoke, full campaign, audit, and action
diagnostic uses a new UTC timestamped subdirectory. Raw episodes, checkpoints,
and logs are ignored by Git; compact protocols, verifiers, and reports are
committed.

- [Goal conclusion](goal_conclusion_vi.md): setting, framework, evidence, limits,
  and exact reproduction commands.
- [V1 protocol](../docs/maintenance_framework_v1_protocol_vi.md),
  [findings](framework_v1_findings_vi.md),
  [independent verifier](verify_framework_v1.py).
- [V2 protocol](framework_v2_protocol_vi.md),
  [findings](framework_v2_findings_vi.md),
  [independent verifier](verify_framework_v2.py).
- [Action diagnostic plans](framework_v1_diagnostic_plan.md) and
  [v2 amendment](framework_v2_diagnostic_plan.md), with
  [trace runner](diagnose_framework_v1.py).

V1 full run: `framework_v1_full_20261019T153014Z/`;
V2 full run: `framework_v2_full_20261019T154912Z/`.
Both are `COMPLETED` and have separate independent `PASS` audit directories.
The raw folders must remain available on the lab for exact v2 reproduction,
which pins the selected v1 checkpoint hash.
