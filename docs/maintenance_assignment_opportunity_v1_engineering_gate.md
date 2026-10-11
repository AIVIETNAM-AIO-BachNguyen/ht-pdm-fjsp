## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-12
- Verification Status: UNVERIFIED
- Version Label: exp_result_v1

# Mac engineering gate: PASS

Scientific full experiment is pending user execution on the Ubuntu lab. Smoke
is engineering evidence only. No full test outcomes have been opened.

- Branch: `codex/maintenance-assignment-opportunity-v1`, based on `5c4deb1`.
- Repository tests: **756 passed**,1 existing tqdm/rich warning,240.15 seconds.
- New focused tests: **12 passed**,5.54 seconds; covered source PM/CM preservation,
  real natural state chains at every clock, nested paired targets, state-table
  mapping, balanced and paired draws, calibration count/profile/late criteria,
  true near-tie fallback, scalar/batch actions, seed/config isolation, statistical
  contrasts, and non-overwriting/error contracts.
- Smoke run: `maintenance_assignment_opportunity_v1_smoke_20261011T173536Z`.
- Exit0, `COMPLETED`, audit `PASS`,36.877 seconds on macOS CPU.
- 9 learned heads,36 optimizer steps,14 closed-loop test episodes (672 base +
  168 tail transitions);840 logged physical decisions.
- 2,016 source transitions;22 train,7 development,16 calibration opportunities;
  one held-out natural root opportunity. Sparse smoke opportunities are not a
  scientific result; full calibration minimums remain40 labels/16 profiles/10
  late opportunities instead of the smoke1/1/0.
- All368 forecast future paths /18,088 transitions replayed scalarly, with cost
  components and boundary states exactly reconciled. All learned smoke actions
  checked against scalar inference; every matching feasible.
- Independent artifact check: all304 SHA256/size entries match,9.366 MB; nested
  low/high targets match stored costs, low/high state tables and draw indices
  match, frozen PPO hashes unchanged, all selected heads save/load exactly.
- 3 pinned source checkpoints and957 inherited source artifacts verified; smoke
  uses one source.174,848 new reserved derived RNG roles audited against89,389
  historical roles.538 fresh physical signatures across full/smoke splits,
  excluding68,143 historical signatures; configuration construction uses no
  outcomes. No generated probes.

The smoke manifest records the pre-commit parent and dirty implementation;
only the tested new files plus this gate are committed. Full requires clean
committed code and records its exact commit. Prior experiment code stays intact.

## User handoff

Run in the Ubuntu terminal at `bachng@100.111.83.52`, keeping it open:

```bash
cd ~/ht-pdm-fjsp && \
export PATH="$HOME/.local/bin:$PATH" && \
if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; fi && \
git fetch origin && \
(git switch codex/maintenance-assignment-opportunity-v1 || git switch --track origin/codex/maintenance-assignment-opportunity-v1) && \
git pull --ff-only && \
uv sync --frozen && \
SOURCE_RUN="artifacts/maintenance_assignment_scoring_v1_full_cpu_20261011T131859Z" && \
test -f "$SOURCE_RUN/manifest.json" && \
test -f "$SOURCE_RUN/additive_ppo/seed_18100000/model.pt" && \
test -f "$SOURCE_RUN/additive_ppo/seed_18100001/model.pt" && \
test -f "$SOURCE_RUN/additive_ppo/seed_18100002/model.pt" && \
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 && \
set -o pipefail && \
RUN_ID="maintenance_assignment_opportunity_v1_full_cpu_$(date -u +%Y%m%dT%H%M%SZ)" && \
mkdir -p "artifacts/$RUN_ID" && \
uv run python -m ht_pdm_fjsp.maintenance_assignment_opportunity \
  --profile full --device cpu --source-dir "$SOURCE_RUN" \
  --output-dir "artifacts/$RUN_ID" \
  2>&1 | tee "artifacts/$RUN_ID.log"
```

Full:27 learned heads,17,280 optimizer steps,4,032 episodes. Artifact size is
expected to be several hundred MB, depending on observed opportunity counts;
fixed collection budget, no outcome-based extension. Partial runs cannot resume;
if interrupted, preserve the directory and start a new timestamped run.

After completion, run on the local Mac:

```bash
mkdir -p "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results" && \
rsync -avhP --partial \
  bachng@100.111.83.52:~/ht-pdm-fjsp/artifacts/ \
  "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results/"
```
