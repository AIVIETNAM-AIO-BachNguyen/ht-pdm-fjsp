# V3 engineering evidence (lab override)

The user explicitly directed execution on this machine, the Ubuntu lab. The
repository AGENTS.md override therefore replaces Mac smoke/manual lab handoff
for this research goal. No SSH or artifact transfer was launched.

Protocol and seed panels were recorded before v3 development. V1/v2 development
provided design feedback only; their raw outputs remain intact. No scientific
confirmation panel has been opened by any of these designs yet.

Frozen pipeline smoke:
`artifacts/technician_reservation_ppo_v3_smoke_20261012T165516Z`
completed with audit PASS. Four graph training jobs and two frozen baseline
reference jobs,32 PPO training episodes,28 development evaluations,32 engineering
test episodes,1920 independently replayed physical ticks,zero invalid assignments.
One shared teacher bank192 states/four episodes240 physical ticks; both graph
arms pretrain two epochs384 presentations/eight updates before PPO384 decisions
per targetscale. N4/N8 exercised. Graph parameter count82819 at both scales;
unit tests exercise constant counts/no catalogue/valid dispatch throughN32.
Scientific performance is not inferred from this smoke.

Meaningful checks cover exact original N4 observations/masks and physics/tail,
legality and collision arbitration, sequence probability normalization/replay,
finite policy gradients and value/actor gradient separation, immutable teacher
hashes/metadata, catalogue-to-token labels, exact bank samples/returns and
pretraining budgets, separate teacher/fine-tuning physical streams, constant-shape
warm transfer to all scales, original reference evaluation without refitting or
checkpoint reselection, serial/spawn equality, seed/family/health/mechanism gates,
checkpoint serialization and refusal to overwrite prior artifacts. Full suite:
987 passed, one existing tqdm/rich warning,945.17 seconds. Core source remained
frozen throughout; CPU contention with the ongoing v2 development run increased
elapsed time relative to earlier suites.

No claim of fair sample-efficiency superiority over the source PPO/MAPPO:
students receive extra teacher data/pretraining and an already-trained teacher.
Coordination evidence must come from the information/data/budget-matched graph
ablation. Inference measurements cover neural decoding on CPU and separately
environment work, not distributed network transport.
