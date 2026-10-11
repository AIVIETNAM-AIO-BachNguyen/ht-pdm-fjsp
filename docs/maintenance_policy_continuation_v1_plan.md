# Maintenance policy continuation v1 — locked paired follow-up

Material Passport: academic-research-suite / experiment-agent, inline plan/run;
2026-10-11; maintenance_policy_continuation_v1; UNVERIFIED pending tests/smoke.
Authorized question: do now versus WAIT when the frozen RL policy itself
continues, preserving the training horizon. No training/reselection/new reward.

## Dependency and prospective boundary

Reuse ALL 2,560 fixed pre-action states from completed timing diagnostic
maintenance_timing_diagnostic_v1_full_cpu_20261011T072307Z, commit e02b504,
manifest SHA256 2d9cf8caca189635e62239521a340a08860b7c8387e75c85394985a8581f0a27.
All source outcomes were already analyzed. This is an exploratory paired
mechanism follow-up on the same occupancy panel, NOT an independent physical
test/generalization replication. Freeze all ten copied joint models and prior
state/config/policy bytes by hashes; no selecting states by observed gaps.
Preflight verifies all source artifact hashes, clean source completion/settings,
all source dependency snapshots, ten model payloads/weights, full state grid,
physical configurations, recorded original root actions/probabilities/values,
and role seeds before opening new full forecast outcomes. Original joint run
is a pinned provenance dependency; its files need not be present again because
the timing source contains its frozen models and source contract.

Fresh forecast full seed14100000, analysis14200000; smoke14300000,
analysis14300010. Audit inherited raw registries and new derived forecast roles.
Model/cohort/physical shock identities intentionally reused from source, not
new split seeds. Forecast tapes independent from actual source physical tapes,
paired across root actions AND continuation arms. Full forecast outcomes remain
sealed on Mac. Smoke intentionally uses four historical source states also
present in full input, with separate engineering forecast tapes; no claim of
fresh smoke/full physical separation for this state-reuse experiment.

## Exact objective and interventions

Source training: undiscounted complete returns, learned horizonH48, rule tailL12,
no critic bootstrap, full physical stop at60. Policy observations always use
48-global_time. At a sampled tick t, charge ALL physical intervals t..59,
including current root, with repairs carried into/beyond48. Record base/tail
costs and cutoff/terminal busy/failure/wait states, including unfinished work.
Do not reset at48, cancel existing repair, shorten the tail, or infer past60.

Evaluate only these four root labels, deduplicating identical matchings:
RL root; WAIT(empty matching); RL root without PM(preserve all CM pairs and
worker assignments); inherited development-selected reference root. Every
root must be feasible. No all-action sample oracle/search or action relabeling.
Two paired continuations:
1. rl: deterministic argmax of SAME frozen policy on each branch's own state
   for global times<t48; deadline_matching for exactly48..59.
2. deadline: deadline_matching from the next interval through59, with its
   remaining horizon60-global_time. This controls continuation while retaining
   the SAME corrected total horizon. Historical rule24 is not a matched arm.
Training sampled stochastic actions; this diagnostic uses deployed deterministic
inference as in original full evaluation, not the stochastic training behavior
policy. No action RNG introduced. Batched inference features must match scalar
feature_batch; near-tied logits fall back to original scalar select to preserve
its tie/rounding behavior. Validate physical feasibility at every interval.

H1 per condition: mean[C(RL root)-C(root without PM)] under rl continuation,
conditional on original RL root containing PM. Positive favors deferring these
current PM starts, preserving all corrective starts; continuation decides freely
when/what to do later. H2 per condition: mean[C(RL root)-C(WAIT)] under rl
continuation conditional on original root NONEMPTY. Positive favors deferring
all current starts, including CM. H2 changes both PM/CM starts and cannot be
called a PM-only effect. Neither establishes improvement from forcing WAIT at
every policy decision. Negative favors the original root. Raw total simulator
cost is PRIMARY, matching complete-return objective; per-machine-remaining-tick
cost (divide by4*(60-t)) is descriptive only. Root costs begin at different
landmarks, so landmark strata are retained and no denominator change in primary.

Secondary: same contrasts under deadline continuation, paired change in delta
between continuations, reference-root comparison on structural choice states,
RL/reference WAIT frequencies, PM-risk/worker-busy/landmark/origin strata,
pairwise MCSE, future RL action histograms, base/tail cost components, terminal
unfinished work. No posthoc subgroup significance or optimistic root labels.

## Panel, budget and uncertainty

Full: source all10 model seeds x2 conditions x8 cohorts x2 physical shocks x2
origins xlandmarks{6,18,30,42} =2560 states, same sampling/duplicate occupancies
as source. Thirty-two private MC future tapes per state. Remaining physical
windows54/42/30/18; every branch ends at60. Max4 unique roots/state x2 arms:
23,592,960 forecast transition samples. Max7,536,640 batched RL continuation
decision rows (root itself is an intervention, not sampled by continuation).
Actual deduplicated roots, transitions and base/tail counts recorded.

Smoke: first numeric model/cohort/shock, both conditions/origins, landmark42
only =4 historical states. Four engineering scenarios; exact18 physical ticks
through60, with6 learned-window intervals and12 tail intervals. Max2304 total
forecast transition samples across2 arms. Scalar replay every smoke branch/
scenario, action sequence, cost component and cutoff/final state. Tests also
use engineering tick6 and47 fixtures to check long windows and t47->48 handoff.

Four PRIMARY exploratory contrasts (H1/H2 xconditions), shared20,000 crossed
bootstrap draws resample10 models and8 physical profiles independently; keep
shocks/origins/ticks paired. Ratios of summed conditional numerators/exposures,
zero numerator/exposure on ineligible cases. Two-sided98.75% percentile CIs;
>5% empty draws ->INSUFFICIENT, CI>0 HIGHER_ROOT_COST, CI<0 LOWER_ROOT_COST,
elseUNRESOLVED. Conditional on fixedMC tapes; pairedMCSE retained. Smoke only
ENGINEERING_ONLY, never scientific pass/fail. Secondary arms/groups descriptive.

Fixed full budget, no early-success stop/retries/resume/additional profiles.
No new physical trajectories or training steps. CPU/thread1, clean full checkout.
tqdm during model preflight and multi-state/root evaluation. New timestamp dir;
interruption/failure retains partial CSVs and FAILED manifest plus hashes.

## Artifacts and gates

manifest/config/plan/source contract/source audit/seed registry; frozen model
copies; states.partial.csv/states.csv; unique root_values.csv (paired scenario
cost arrays, base/tail components, terminal metrics); rollout_audits.csv;
raw_forecast_costs.npz(total/base/tail costs indexed state xcontinuation x4
root labels xscenario; duplicate root labels intentionally repeat identical
values); raw_boundary_states.npz(cutoff48 and terminal60 physical arrays);
summary.json/profile_cells.csv; behavior/landmark/continuation-difference tables;
inference counters/timings/dependency snapshots. All outputs hashed. NO teacher
labels, optimization, critic bootstrapping or model weight changes.

Tests: hash/contract rejection before new outcomes, complete source grid and
root reconstruction, own fresh role splits, batched/scalar feature/mask/action
equivalence, near-tie fallback, root deduplication and keep-CM intervention,
CRN pairing/no live-future access, exact48/60 clock and cost splits, ongoing
repair across cutoff, all smoke scalar branches, checkpoint roundtrip, forced
WAIT, conditional bootstrap/grid/empty-exposure, nonempty-output rejection,
clean/thread guards and graceful interruption. Whole repository tests and Mac
actual-checkpoint smoke gate, then commit/push; user lab execution and Mac
rsync commands. No assistant SSH/full launch/transfer. Compute estimate only
from engineering states; smoke never executes fresh full forecast seeds.
