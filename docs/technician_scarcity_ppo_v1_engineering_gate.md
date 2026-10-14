# Scarcity-Message PPO v1 engineering gate

User authorizes smoke and research runs directly on this Ubuntu lab. No teacher,
distillation, baseline checkpoint input or cross-scale warm start is used.

- Initial core full regression suite: 1011 passed in737.43s, one existing
  tqdm/rich warning. Log preserved with the engineering review.
- Final targeted suite after engineering amendments: 20 passed in9.45s.
  Covers likelihood normalization, all21 physical small matchings via33 canonical
  latent traces, STOP/busy masks, finite message gradients, symmetry, exact
  initialization/capacity, no future input/catalogue at large sizes, scalar
  physical replay, scratch checkpoint provenance, serial/spawn determinism,
  corruption rejection, complete30-endpoint family and prerequisite enforcement.
- First smoke `technician_scarcity_ppo_v1_smoke_20261014T020923Z` failed the
  independent seed audit: learner RNG hash was incorrectly used to validate
  physical RNG seeds. Corrected to maintenance_dispatch.role_seed, including
  actual physical roles in seed auditing. The failed artifact remains preserved.
- Corrected smoke `technician_scarcity_ppo_v1_smoke_20261014T021329Z` completed,
  audit PASS. Nine random-initialized fits,72 training episodes,16 paired
  physical tapes,54 dev episodes,22 engineering test episodes/1320 independently
  replayed physical ticks; zero invalid assignments and zero pretraining.
- Final smoke `technician_scarcity_ppo_v1_smoke_20261014T022933Z` completed,
  audit PASS on the same engineering grid. The observer/Monte-Carlo amendments
  are recorded prospectively in the plan before research-quality outcomes.
  Compare all27 checkpoint tensor sets and every original physical episode field
  exactly to the corrected smoke, excluding measured timing. All match.

Graph controls have19780 parameters at all scales and identical seeded initial
tensors. N4 PPO/local MAPPO/full-information MAPPO have18966/15364/18052
parameters. All are trained from scratch. Final physical audit regenerates the
random initialization and checks the step0 checkpoint against it; no teacher path.

Run the read-only comparison with:

```bash
/home/bachng/ht-pdm-fjsp/.venv/bin/python scripts/verify_scarcity_engineering.py /home/bachng/ht-pdm-fjsp/artifacts/technician_scarcity_ppo_v1_smoke_20261014T021329Z /home/bachng/ht-pdm-fjsp/artifacts/technician_scarcity_ppo_v1_smoke_20261014T022933Z --output-root /home/bachng/ht-pdm-fjsp/artifacts
```

Commit and push this verified design before pilot. Pilot has no scientific test
panel. Its N32 full training throughput is the next engineering measurement;
384-step smoke timing cannot by itself predict a491520-step full fit.
