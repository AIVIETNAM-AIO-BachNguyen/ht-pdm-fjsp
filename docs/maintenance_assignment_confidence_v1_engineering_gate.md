## Material Passport

Origin Skill: ARS-Codex experiment-agent, inline authorized implementation.
Date: 2026-10-12. Version: maintenance_assignment_confidence_v1.
Verification Status: VERIFIED ENGINEERING ONLY; full scientific result unverified.

## Gate

- Repository suite:744 passed,3 existing dependency warnings,280.34 seconds.
- Latest focused confidence/value contracts:28 passed,10.15 seconds.
- Final Mac smoke:maintenance_assignment_confidence_v1_smoke_20261011T164818Z,
  COMPLETED/audit PASS,73.13 seconds,288 hashed artifacts,7.50MB.
- Six heads changed weights;24 optimizer steps.64 train/dev/calibration states,
  16 held-out root states,10 complete H48+12 episodes,600 logged evaluation ticks.
- All1,294 forecast futures/51,456 transitions scalar-replayed, including
  base/tail boundaries and components. Scalar/batch decisions match.
- Same PM machine sets and exact CM preserved; no WAIT/timing overrides.
  Empirical gate margin/disagreement/sign, disabled insufficient calibration,
  ties, condition detection, missing episode grids and overwrite failure tested.
- Datasets and selected checkpoints roundtrip; all frozen PPO weights unchanged.
  Every final-smoke source Python snapshot matches final implementation.
- Broad draw journals have exactly half probe labels; natural has none. Late
  probe labels available at6/18/30/42 in engineering data. Generated probe
  states are reported separately from naturally visited source states.
- 67,925 canonical historical configurations excluded;218 new signatures across
  full/smoke/train/development/calibration/test are mutually disjoint. RNG audit
  checks raw roles and derived roles, including independent forecast phases and
  member/epoch optimization streams. Probe anchors paired across sources.
- Calibration and checkpoint selections locked before smoke test. Full test
  outcome flag remains false. Training/evaluation tqdm present in console log.
- Hash/byte checks pass for all final-smoke artifacts. Existing full artifacts
  untouched. No SSH, lab full-launch or rsync performed by assistant.

## Full budget and estimate

Three frozen PPO replicas;two data arms xthree heads each =18 learned heads.
3072 train,384 development,384 calibration states;384 held-out root states.
Five controllers,2880 complete episodes;11,520 matched optimizer steps.
25,344 natural prefix transitions and4224 generated probe-prefix transitions;
138,240 test learned-window and34,560 tail transitions. Root/label forecast
upper bound63,258,624 transitions; singleton states skip needless forecasts.
All forecasts are simulator-label compute, not PPO training environment steps.

Reserved-smoke vectorized profile:21 roots from tick0,16 futures/20,160
transitions took0.358s;32 futures/40,320 transitions took0.658s on this Mac.
At those rates the conservative forecast upper bound alone extrapolates to
about17–19 minutes, before training/evaluation/I/O and state-specific overhead.
This is an extrapolation, not a Mac or lab full run. Budget roughly30–90 minutes
initially on lab CPU and use actual tqdm progress; lab hardware was not measured.
Reserve150–400MB for artifacts; actual roots, requests/log size and compression
vary. No GPU speedup, global safety bound or algorithm superiority claim.

## Handoff dependencies

Lab source directory must remain available in
artifacts/maintenance_assignment_scoring_v1_full_cpu_20261011T131859Z.
The runner verifies the source's entire artifact ledger and pinned3 model
checkpoints before generating data. Branch:codex/maintenance-assignment-confidence-v1.
CLI:uv run python -m ht_pdm_fjsp.maintenance_assignment_confidence.
Follow docs/experiment_workflow.md: user executes full lab command and partial-
progress rsync command; keep lab terminal open. Resume/overwrite unsupported.
