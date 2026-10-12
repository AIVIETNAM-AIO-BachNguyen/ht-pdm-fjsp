# Lab engineering gate

2026-10-12 user-authorized direct lab execution; branch
codex/technician-reservation-ppo-v1. Baseline observed before protocol: prior
balanced benchmark PPO cost1050.0789 and MAPPO1086.9328; these are motivation,
not reused confirmation observations or checkpoint selection.

Smoke artifacts (preserved, ignored by Git):
- technician_reservation_ppo_v1_smoke_20261012T150144Z: completed first smoke.
- technician_reservation_ppo_v1_smoke_20261012T150446Z: completed stronger replay
  gate,7models,56trainingepisodes,42developmentepisodes,36engineeringtestepisodes,
  2160physicalsteps independently replayed, no invalid assignments; sampled and
  recomputed likelihoods match, checkpoint roundtrips pass. These are engineering
  outcomes; their cost/health comparisons do not support scientific conclusions.

14newtests and22relevant baseline/training/checkpoint tests passed (36total).
Entire repository suite: 979 passed, one existing tqdm/rich warning, 737.78s on this lab. This gate passed before research launch. Final smoke
following prospective error-budget amendment: technician_reservation_ppo_v1_smoke_20261012T150803Z, COMPLETED with all audits PASS.

Research launch after verified commit/push:
```
cd /home/bachng/ht-pdm-fjsp && RUN_ID="technician_reservation_ppo_v1_development_$(date -u +%Y%m%dT%H%M%SZ)" && uv run python -m ht_pdm_fjsp.technician_reservation_experiment --profile development --output-dir "artifacts/$RUN_ID" 2>&1 | tee "artifacts/$RUN_ID.log"
```
Use `set -o pipefail` before the pipeline. Execution is by the assistant on this
lab under the current explicit user override; no SSH or handoff required.
