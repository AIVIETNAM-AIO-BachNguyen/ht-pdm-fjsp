# V4 confirmation engineering gate

The selected learning design is the original pushed V4 source `a0e0493`.
The confirmation execution amendment changes one runner setting: eight
single-thread workers for confirmation/smoke, four for development. Every
actor, critic, distillation, PPO update and physical implementation is unchanged.
The amended protocol was locked before any scientific confirmation test.

Validation on this lab:

- Full suite: 992 passed in 760.20 seconds; one pre-existing tqdm/rich warning.
- Eight-worker smoke: `artifacts/technician_reservation_ppo_v4_smoke_20261012T212702Z`, COMPLETED, audit PASS.
- Exact comparison with four-worker smoke `technician_reservation_ppo_v4_smoke_20261012T184359Z`: PASS for all 14 checkpoint tensors/digests, both warm checkpoints, the compressed teacher bank, every chosen checkpoint, complete budget/episode audit and all 32 physical/quality episode outputs. Timing fields vary with execution.
- The verifier checks that every source snapshot file is byte-identical except the single concurrency setting, and that resolved configurations differ only in `training_processes` (4 to 8).
- Reproduce the read-only comparison with `scripts/verify_reservation_execution.py BASE NEW`; preserve `execution_equivalence.json` and its copied verifier script in the new smoke artifact.
- The strengthened verifier also validates every sealed output binding in both inputs; `execution_binding_equivalence.json` passes the same exact comparisons. Existing reports are preserved through `--output` with a fresh filename.

The additional engineering preflight
`technician_reservation_ppo_v4_engineering_preflight_20261012T200351Z`
completed 48 episodes, 2304 neural decisions and 2880 physical ticks, using
three fixed warm models on four engineering seeds at each N4/N8/N16/N32.
Neural p95 was 2.05/3.30/6.35/12.24 ms under lab load, parameter count 82819
at every scale, zero invalid assignments and conflicts. It has no quality
endpoint and opens no scientific test. Its prospective protocol, probe and
results are preserved; raw engineering seeds58300000..3 are registered as
history. Final inference latency still must pass on the sealed scientific grid,
after the training pool has shut down. Network transport is outside these timings.

Full tests and the eight-worker smoke ran with frozen source; this gate and the
read-only execution verifier were added after completion. The verifier itself
was run successfully on the real saved artifacts. No learning or protocol
changes follow verification. Commit and push before
scientific confirmation.
