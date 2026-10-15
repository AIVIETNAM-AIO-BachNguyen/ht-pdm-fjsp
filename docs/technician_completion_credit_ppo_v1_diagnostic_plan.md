# Frozen Completion-Credit v1 pilot diagnostic

Locked after the negative completed pilot, before this diagnostic runs. This is
an exploratory analysis of opened development trajectories, not training or a
scientific confirmation. The pilot fails nonpooled cost/health screens and does
not authorize full development. Preserve every failed screen and every replica.

Hypotheses: poor availability/cost may associate with (a) leaving an urgent legal
edge unused, or (b) selecting long services and allocating scarce technician types
to edges while urgent machines remain unserved. Compare service duration,
restoration, start age, preventive/corrective mix, residual feasible edges,
unserved at-risk machines and their subsequent failures, both with and without
residual free technicians. These are associations, not preventability or causal
proof. Record conflicts and rejections, including the independent control.

Use selected checkpoints from every pilot arm, both training replicas and all
16 opened DEV seeds at N4 and N32, plus the existing deadline references once per
size/seed: 384 episodes (224 N4 learned + 128 N32 learned + 32 references).
Smoke uses the first existing training replica and DEV seed for every method and
both sizes, including deadline: 13 episodes. Never open the scientific test panel,
sample a new seed, update a parameter, or select a better replica. Exogenous
populations remain paired by size and DEV seed; trajectories need not coincide.

Frozen 48 decisions plus common 12-tick tail. Hard stop 1200 seconds per
diagnostic; no extension or metric-based stopping. Complete grid required, every
original episode metric reconciled exactly (numerics tolerance 1e-8), parameters
unchanged, source/PLAN/registry identity verified, every bound input hash/size
verified, independent replay of every physical transition. No optimizer or extra
environment samples enter any learning process.

Fresh UTC artifacts contain copied script/plan, manifest with input/output hashes,
decisions.csv, starts.csv, physical.csv, episodes.csv and aggregate.json. Use tqdm.
Syntax/diff checks and smoke first, commit/push verified analysis code, then show
and execute the exact full diagnostic command on this lab under the user override.
Keep all learning source, original protocol and seed registries frozen. The
diagnostic does not relax gates, authorize full development, spend scientific
alpha or establish the main research contribution.
