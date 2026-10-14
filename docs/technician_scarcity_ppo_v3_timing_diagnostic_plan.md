# Frozen v3 development timing diagnostic

Exploratory follow-up to the completed negative full-development run, locked
before executing this diagnostic. No training, test panel, new environment or
policy intervention. Do not infer causation from different policy trajectories.

Hypotheses: the small-setting failure gap may associate with (a) leaving a legal
urgent machine-technician pair unused, including first-slot STOP, or (b) selecting
slower/weaker service edges under contention. Diagnose both, without treating
unserved risk as a proven preventable failure.

Use only N4 selected checkpoints from all seven already trained arms, all three
training replicas and all 32 already opened DEV seeds (672 episodes). Smoke uses
the first existing replica and first existing DEV seed for all seven arms
(7 episodes). No best-replica choice. Pair exogenous populations by DEV seed;
policy trajectories may differ. Original 48 decision ticks and 12 deadline-tail
ticks unchanged. No optimizer, parameter updates, teacher or future-event inputs.

Metrics: first STOP with legal edges; residual legal edges after executed starts;
nonfailed at-risk machines not starting/servicing; newly failed at-risk machines;
start age, preventive/corrective status, service duration and restoration;
conflicts/rejections and original cost/availability/backlog/failure metrics.
Per-episode and per-method descriptive aggregates only. No scientific CI,
threshold changes, alpha spending or confirmation authorization.

Hard stop 600 seconds per diagnostic, no extension or performance-based stop.
Require complete episode grid, exact matching of original episode metrics,
unchanged checkpoint tensors/algorithm identity, paired initial population hashes,
and independent scalar replay of every policy and tail transition. This diagnostic
does not replace any scientific gate.

Fresh timestamped artifacts: copied script/plan, manifest with source/input/output
hashes, decisions.csv, starts.csv, physical.csv, episodes.csv, aggregate.json.
Use tqdm for episode progress. Run Python syntax/diff checks and smoke first,
commit/push analysis code, then show and execute full diagnostic locally under
the persistent lab override. Preserve original full run unchanged; no SSH or
automatic artifact transfer. Learning source/protocol/registries remain frozen.
