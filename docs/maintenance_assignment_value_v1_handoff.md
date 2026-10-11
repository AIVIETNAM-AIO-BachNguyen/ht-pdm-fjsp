# Assignment relative-value pilot handoff

This pilot freezes the three additive PPO models from the completed
`maintenance_assignment_scoring_v1_full_cpu_20261011T131859Z` run. It learns a
matching-value head from fresh common-state simulator continuation returns.
At inference PPO selects the PM machines and exact CM edges, while the head
selects technicians among the same PM machine set. This is one policy
improvement step, not joint end-to-end PPO retraining. Timing is evaluated
separately using each controller's own continuation at the original H48 clock
and common12-tick deadline tail. The plan and seed/source contracts are frozen
in `docs/maintenance_assignment_value_v1_plan.md` and `configs/`.

Full budget:1536 training states,192 development states,288 held-out assignment
states,120 common timing states,1728 closed-loop episodes. True-label and
within-condition shuffled-label heads have identical initialization, updates,
minibatches and architecture. Checkpoint selection uses true development pair
MSE. Four primary episode contrasts use98.75% crossed paired source-replica/
profile bootstrap intervals and a1% material margin. Assignment/timing
mechanism checks are exploratory. Models do not forecast at deployment.

All three source checkpoints and957 source artifacts are verified before new
compute. Full runs require clean committed code and one CPU thread. New UTC
artifact directories are mandatory; overwrite/resume is refused. Full metadata
can be resolved locally; new full outcomes must only be run by the user on lab.
`tqdm` shows labeling, source trajectories, training, and multi-seed evaluation.

Run the following compound command **in the Ubuntu lab terminal**, keep it
open, and let it finish. Do not run it over SSH from the assistant.

```bash
cd ~/ht-pdm-fjsp && \
export PATH="$HOME/.local/bin:$PATH" && \
if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; fi && \
git fetch origin && \
(git switch codex/maintenance-assignment-value-v1 || git switch --track origin/codex/maintenance-assignment-value-v1) && \
git pull --ff-only && \
uv sync --frozen && \
SOURCE_RUN="artifacts/maintenance_assignment_scoring_v1_full_cpu_20261011T131859Z" && \
test -f "$SOURCE_RUN/manifest.json" && \
test -f "$SOURCE_RUN/additive_ppo/seed_18100000/model.pt" && \
test -f "$SOURCE_RUN/additive_ppo/seed_18100001/model.pt" && \
test -f "$SOURCE_RUN/additive_ppo/seed_18100002/model.pt" && \
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 && \
set -o pipefail && \
RUN_ID="maintenance_assignment_value_v1_full_cpu_$(date -u +%Y%m%dT%H%M%SZ)" && \
mkdir -p "artifacts/$RUN_ID" && \
uv run python -m ht_pdm_fjsp.maintenance_assignment_value \
  --profile full --device cpu --source-dir "$SOURCE_RUN" \
  --output-dir "artifacts/$RUN_ID" \
  2>&1 | tee "artifacts/$RUN_ID.log"
```

After completion, run this **on the local Mac**, preserving partial transfers
and displaying progress. The assistant must not start this transfer.

```bash
mkdir -p "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results" && \
rsync -avhP --partial \
  bachng@100.111.83.52:~/ht-pdm-fjsp/artifacts/ \
  "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results/"
```

Artifacts include manifest/count/hash audits, frozen plan/source/seed/config
snapshots, source trajectory decisions, paired labels and MC SEs, raw NPZ tapes/
costs/components/boundaries, tensor datasets and shuffle permutations, training
progress/checkpoint selection, test episodes/decisions/PM choices/request audits,
held-out assignment predictions written before forecasts, own-policy timing
records, clustered summaries and profile cells. Analysis begins only on the
completed transferred run with PASS audits. A small Mac smoke is engineering
evidence; its outcomes do not establish learning quality or algorithm benefit.
