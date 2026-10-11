## Material Passport

- Origin Skill: ars-codex:academic-research-suite / experiment-agent.
- Origin Mode: implementation and local verification, inline.
- Origin Date: 2026-10-11.
- Verification Status: VERIFIED ENGINEERING SMOKE; FULL PILOT NOT EXECUTED.
- Version Label: maintenance_assignment_scoring_v1_mac_gate.
- Overall Confidence: engineering contracts verified, scientific hypotheses untested.

## Gates

Full repository suite:716 passed, one existing SB3 rich/TqdmExperimentalWarning,
232.16seconds. Final focused suite after strengthening full-width/nontrivial
root replay and adding diagnostic budget guards:22 passed in9.50seconds.
Original scientific source modules are unchanged; this branch adds an isolated
policy, benchmark, PPO runner, factorial evaluator, statistics and diagnostics.

Final CLI smoke:
artifacts/maintenance_assignment_scoring_v1_smoke_cpu_20261011T131551Z.
COMPLETED/PASS in6.67seconds. 2models,128 learned decisions,192 physical training
steps,32 training episodes,16 optimizer steps,26 development and20 test episodes,
120 test decisions,8 common diagnostic states/16 model-state records. H4+2 smoke
never opens new full outcomes. Both scorers have5218 parameters and exactly
matched initial weights, update weights, and pass sampled/reevaluated PPO
likelihood and checkpoint save/load checks. Full models have19650 parameters.

Smoke trajectories select no PM on diagnostic landmarks:16 unique roots and288
forecast transitions. This is a valid no-choice/exposure branch and not evidence
that either model learned timing or assignment. Nontrivial branches are covered
by focused tests: same-machine/fixed-CM conditional decoding, both pair orders,
normalization, stochastic likelihood gradients, and seven roots/four timing/two
assignment options at0/42/47 with H48+12 and hidden32. Every tested candidate's
scalar/vector action trace, costs/components and cutoff/terminal states agree.
The independent-future/tie tests do not tune or filter outcomes.

Fresh metadata excludes59967 historical canonical signatures, including all
previous pilot training and the latest diagnostic full/smoke pools. All fresh
full/smoke/train/dev/test physical signatures and raw/RNG roles are audited;
common forecast phases are paired across scorers but distinct from each other
and from physical trajectory tapes. All diagnostic parameters stay immutable.

225 artifacts excluding manifest,5716673bytes; every recorded hash/size verified.
All copied Python snapshots and plan match prepared source exactly. Dirty=true
is expected because the smoke precedes commit. The final lab run requires clean
committed source and OMP/OPENBLAS/MKL threads1. Full budget:6 models,737280 learned
steps,921600 physical training steps,15360 training episodes,11520 optimizer steps,
888 development episodes,2496 test episodes and149760 physical test decisions.
Separate diagnostic:360 common states/720 scorer-state records, at most19740672
forecast transitions. Cost CIs and timing/assignment mechanism outputs remain
separate; no scientific improvement has yet been established.

## Handoff

Branch codex/maintenance-assignment-scoring-v1. Command entry point:
uv run python -m ht_pdm_fjsp.maintenance_assignment_scoring.
No source checkpoint dependency. Full lab run and rsync are user-run only per
docs/experiment_workflow.md. Estimated lab runtime15–25minutes, provisional from
previous runs; actual runtime and artifact sizes recorded by manifest. Full
artifact footprint expected tens to low hundreds of MB, dominated by decision
logs and paired forecasts. Do not close the lab terminal during the run.
