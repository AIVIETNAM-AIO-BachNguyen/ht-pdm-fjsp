# Maintenance coordination RL research

All new goal outputs live here. Each smoke, full campaign, audit, and action
diagnostic uses a new UTC timestamped subdirectory. Raw episodes, checkpoints,
and logs are ignored by Git; compact protocols, verifiers, and reports are
committed.

- [Goal conclusion](goal_conclusion_vi.md): setting, framework, evidence,
  limits, and exact reproduction commands.
- [V1 protocol](../docs/maintenance_framework_v1_protocol_vi.md),
  [findings](framework_v1_findings_vi.md),
  [independent verifier](verify_framework_v1.py).
- [V2 protocol](framework_v2_protocol_vi.md),
  [findings](framework_v2_findings_vi.md),
  [independent verifier](verify_framework_v2.py).
- [V3 protocol](framework_v3_protocol_vi.md),
  [findings](framework_v3_findings_vi.md),
  [independent verifier](verify_framework_v3.py).
- [Action diagnostic plans](framework_v1_diagnostic_plan.md),
  [v2 amendment](framework_v2_diagnostic_plan.md), and
  [v3 plan](framework_v3_diagnostic_plan.md), with
  [trace runner](diagnose_framework_v1.py).

Full runs:

- V1: framework_v1_full_20261019T153014Z/
- V2: framework_v2_full_20261019T154912Z/
- V3: framework_v3_full_20261019T162924Z/

All three are COMPLETED with separate independent PASS audit directories. The
raw folders must remain on the lab for checkpoint-based reproduction; v2 pins
the selected original v1 checkpoint hash. The v3 teacher banks and models are
also retained in its raw directory.
