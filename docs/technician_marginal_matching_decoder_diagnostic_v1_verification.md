# Pre-episode verification of the frozen N4 decoder diagnostic

The user authorizes FULL directly on the lab, omitting a separate smoke run.
Necessary correctness tests used temporary pytest directories and no optimizer.

- 46 checks passed in13.05s:18 diagnostic checks and28 unchanged physical-policy
  checks. All nine standard and five exact core decoder episode paths replayed
  complete physical episodes, with fixed parameter hashes. Serial/spawned frozen
  evaluation produced identical physical outcomes. Duplicate decoder records and
  unfinished/passing-screen parent prerequisites are rejected.
- Exact diagnostic support sums to one, selects the recorded physical-probability
  argmax, retains standard deployment action separately, and supports fixed-order
  priors. The latent-order diagnostic scores physical q, not latent training log.
- Registration audit passed67 raw and66 derived physical seeds, disjoint panels
  and raw historical registries. No diagnostic quality episode has run yet.
- The actual parent run completed2011.8749s,27 models, full13,271,040 decisions,
  auditPASS, scientific testclosed and failed development screen6/14. A separate
  read-only verifier recomputed budgets/grids/means/checkpoint choices/latency.
- `validate_source` passed on that actual parent artifact: full registered config,
  selected27 checkpoints, all parent output/source bindings and unchanged imported
  learning files. This allows this diagnostic; it does not authorize scientific
  confirmation or larger scales.

Next full diagnostic has fixed2752 fresh episodes,165120 physical ticks, eight
CPU workers and1800s global hard stop. No training, teacher, or source reselection.
