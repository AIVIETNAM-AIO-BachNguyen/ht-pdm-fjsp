# Reservation PPO v2 lab engineering gate

User explicitly authorizes running research on this lab. V2 is isolated in
/home/bachng/ht-pdm-fjsp-worktrees/reservation-v2 while v1 finishes unchanged.
All research artifacts are written to the original repository's artifacts/
folder. No scientific v1/v2 test outcome has been opened.

Frozen v2: separate actor/value edge encoders, selected-assignment prefix
messages, and public normalized machine/technician IDs supplied to BOTH graph
arms. Parameters70531 at every scale; matched ablation has exactly the same
initial weights/parameter count and optimizer/sample budgets. Small mechanism
positive-gain flag is always reported, but primary coordination evidence must
pass at ALL three scaling settings; small candidate must be noninferior within
1% to both PPO and independent and beat MAPPO. Amendments A/B/C explain all
prospective design/refinement choices and are frozen before v2 development.

Validation:
- Final entire suite: 982passed, one existing tqdm/rich warning,748.91s.
- Latest architecture targeted tests:39passed; includes categorical sequence
  normalization, legality, scalar physics/tail, checkpoint roundtrip, public
  identity control, critic-to-actor gradient isolation, serial/spawn exact
  initial/final parameter hash equivalence, and statistical gate regressions.
- Final smoke: technician_reservation_ppo_v2_smoke_20261012T154642Z COMPLETED,
  audit PASS;7models/56train episodes/42dev episodes/36engineering test episodes,
  all2160physical ticks independently replayed; no invalid assignment.
- Earlier smoke variants remain preserved under source hashes; no scientific
  claims use their outcomes.
- One earlier suite failed its serial/spawn test because source was edited while
  its parent process was alive: serial encoder64x22 versus newly imported child
  encoder64x24. Freeze and full rerun resolved this; no scientific run used the
  mixed-source state. The failed suite is not described as a scientific failure.

Research launch after this verified commit is pushed:
```
cd /home/bachng/ht-pdm-fjsp-worktrees/reservation-v2 && RUN_ID="technician_reservation_ppo_v2_development_$(date -u +%Y%m%dT%H%M%SZ)" && PYTHONPATH="$PWD/src" /home/bachng/ht-pdm-fjsp/.venv/bin/python -m ht_pdm_fjsp.technician_reservation_experiment --profile development --output-dir "/home/bachng/ht-pdm-fjsp/artifacts/$RUN_ID" 2>&1 | tee "/home/bachng/ht-pdm-fjsp/artifacts/$RUN_ID.log"
```
Set `set -o pipefail` before the pipeline. Four deterministic independent CPU
jobs, Torch threads1 each. Per-job journals/checkpoints stream locally; parent
merges journals in fixed model order after all21 jobs. Development491520 steps
matches the future confirmation budget. No metric-triggered early stopping,
resume, retries, test opening or final claim based on development alone.
