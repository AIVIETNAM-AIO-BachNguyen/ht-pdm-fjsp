# Pre-full verification, 2026-10-16

The latest user instruction removes the separate smoke experiment. No standalone
smoke/pilot or scientific confirmation was launched for this protocol. Necessary
unit/integration tests used temporary pytest directories, not research artifacts.

Verified on the Ubuntu lab with the repository Python 3.12 virtual environment,
explicit worktree PYTHONPATH and OMP/MKL/OpenBLAS threads=1:

- 144 tests passed in 39.53 seconds across the new policy/runner tests and the
  unchanged shared PPO learner, completion-v2, reservation and physical benchmark
  environment regression tests.
- One additional final-audit integration test passed in 10.88 seconds: nine-arm
  temporary logs, exact whole-episode/optimizer budgets, paired physical tapes,
  checkpoint selection replay and independent replay of all 600 physical ticks
  from ten temporary evaluation episodes. Corrupting checkpoint selection fails.
- New raw seed audit: 368 unique integers, disjoint panels/history. New derived
  role audit: 82,431 unique derived integers, disjoint from raw new/history seeds.
  This does not claim exhaustive reconstruction of every historical derived role.
- Core arms have identical initial tensors and 18,390 parameters; original PPO
  is preserved at 18,966, local MAPPO at 15,364, full-information MAPPO at 18,052.
- Real optimizer/checkpoint plumbing passed for all nine methods; serial and
  spawned candidate fitting produced identical initial/final parameter digests
  and selected checkpoint SHA256. Actor replay precision gates remain unchanged.
- The full development configuration is N4/K2 only: 27 fits, 491,520 decisions per
  fit, 13,271,040 decisions total, 4,320 DEV episodes plus 32 deadline references,
  eight CPU workers, one torch thread each and a 12-hour whole-grid hard stop.
  It has no scientific test seeds. Confirmation requires passing the registered
  development cost/health/mechanism and isolated-latency gates first.

This is engineering evidence. Research quality and goal success remain unproven.
