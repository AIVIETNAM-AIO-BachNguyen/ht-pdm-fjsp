# A1/B2/C1 development execution recovery — 18 October 2026

At approximately 13:41 UTC, all three original development parent PIDs and
their workers disappeared within one second. Their manifests remained
`RUNNING` in `training_pool`; their console logs ended mid-progress, without
terminal exception or `abort.json`. The inspected kernel journal contained no
OOM event, and subsequent RAM/disk checks had ample free capacity. The exact
termination signal and cause are unknown. The immutable read-only incident
record is
`artifacts/reservation_framework_execution_interruption_20261018T134321Z/incident.json`.

Classify the following artifacts as **TECHNICAL_INCOMPLETE**, not scientific
DEV failures or passes. Do not read their partial development quality rows or
use their checkpoints for selection. All scientific TEST panels remained
sealed:

| Study | Interrupted artifact | Completed model jobs in stale manifest |
|---|---|---:|
| A1 | `/home/bachng/ht-pdm-fjsp-teacher-transfer-v1/artifacts/technician_teacher_transfer_v1_development_20261018T120344Z` | 6 |
| B2 | `/home/bachng/ht-pdm-fjsp-relational-v2/artifacts/technician_relational_prefix_v2_development_20261018T121324Z` | 15 |
| C1 | `/home/bachng/ht-pdm-fjsp-diverse-v1/artifacts/technician_diverse_maintenance_v1_development_20261018T122731Z` | 0 |

The runners have no tested resume contract. After a detached `systemd --user`
probe survived its launching terminal and a separate stdout/stderr probe
passed, the same registered DEV profiles were launched again in fresh UTC
artifact directories. This is a technical rerun of the same locked research
round: no old DEV summary or scientific TEST outcome was opened, and no learner,
physics, seed split, optimizer budget, checkpoint rule, endpoint or gate changed.
The environment and concurrent worker topology remain three pools of four
single-thread workers. Each adjacent `.launch.json` preserves the exact
`systemd-run` and Python argv, working directory, source commit and predecessor.

| Study | Fresh artifact | Detached unit | Source commit |
|---|---|---|---|
| A1 | `/home/bachng/ht-pdm-fjsp-teacher-transfer-v1/artifacts/technician_teacher_transfer_v1_development_20261018T134420Z` | `codex-reservation-a1-dev-20261018T134420Z` | `a60f716` |
| B2 | `/home/bachng/ht-pdm-fjsp-relational-v2/artifacts/technician_relational_prefix_v2_development_20261018T134357Z` | `codex-reservation-b2-dev-20261018T134357Z` | `b2a271f` |
| C1 | `/home/bachng/ht-pdm-fjsp-diverse-v1/artifacts/technician_diverse_maintenance_v1_development_20261018T134420Z` | `codex-reservation-c1-dev-20261018T134420Z` | `396ecb5` |

At launch, all three units were active and had fresh `RUNNING` manifests with
`test_opened=false`. Treat unit state plus manifest and final independent audit
as authoritative; a stale `RUNNING` manifest alone is not proof of a live run.
Confirmation remains gated by each completed DEV artifact and its independent
verifier. Preserve every interrupted and recovered artifact.

A bytewise recovery check after launch found **zero changed package files**
between old and new source snapshots (A1 197 files; B2 197; C1 199).
`protocol.md`, `seed_registry.json`, `physics.json` and
`resolved_config.json` were byte-identical within each old/new pair. The
source commit changed for A1 and B2 only because read-only verifier/document
work had been pushed; the frozen experiment package did not change.
