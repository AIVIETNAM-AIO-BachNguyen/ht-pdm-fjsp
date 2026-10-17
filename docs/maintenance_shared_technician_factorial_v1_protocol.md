# Shared technician maintenance factorial v1 — prospective protocol

Locked 2026-10-18 before implementing or running this study. This is a new
synthetic maintenance-only experiment. No production scheduling or learning is
included. Prior artifacts are background only; no sealed panel is opened or
used to choose a setting.

## Question and falsifiable hypotheses

When do persistent maintenance requests and allocation of finite heterogeneous
technicians improve long-run total system cost over separate decisions?

Run a 3 × 4 factorial on each of three fixed conditions. Request policies:
`reactive` requests only after failure; `threshold` requests at age >= failure
age - 1; `capacity_aware` requests at age >= failure age - 1 and one tick
earlier when every compatible technician is occupied and the earliest release
is at least one tick away. Failure always creates corrective demand. Each
request, once made, persists until assignment; failed PM demand becomes CM
demand while retaining its original request timestamp. A committed job cannot
be canceled. A healthy pending machine keeps operating and degrading; a failed
pending machine is unavailable. The proposal enters the pending pool before
dispatch in the same tick. Dispatcher actions start service immediately;
restoration occurs only at completion.

Dispatchers: skill-feasible FCFS (request time, machine ID; shortest duration
then technician ID), risk priority greedy (failed, failed wait, age, hazard;
shortest duration), myopic maximum-weight partial matching (current expected
one-tick loss avoidance minus PM/CM and service opportunity cost), and
`lookahead` maximum-weight partial matching using the repository's six-tick
isolated-machine expected marginal value DP. The last is a strong computable
reference, but assumes technician availability after the current tick and is
not a joint oracle. All see the same current state, costs and known failure
probability, never the future shock tape. Matching may leave requests waiting.

Conditions use N4,H48 + fixed 12-tick cleanup tail: `scarce_overlap` K2, shared
skill edges, service durations 1–5; `partitioned` K2 with two machines per
technician; `ample` K4 all compatible, durations 1–3. All have failure age 4,
probability .45 per eligible tick, max age 8, restoration .75, PM 1, CM 2,
unavailability 6/machine-tick, failure 15, failed-wait overdue limit 4 and
price 12/tick. The physical failure tape is independent of policy eligibility.

Primary metric: total undiscounted system cost during H+tail plus terminal
liability. Secondary: availability during H, failures, failed waiting ticks,
overdue ticks, request wait (mean/p95), pending at H and at end, technician
utilization during H, PM/CM starts/completions, feasibility, unfinished work,
and computational wall time. The fixed cleanup policy is FCFS for every arm;
it makes no new preventive proposals, but failures still create corrective
demand. At the end of H+12, each pending job is charged its PM/CM cost plus
6 times minimum compatible service duration; failed pending jobs also incur
12 times that duration. Each committed unfinished job is charged 6 times
remaining service ticks. This finite-horizon liability is an approximation,
reported separately; both H-only and H+tail components are saved.

H1 (scarce overlap): `capacity_aware/lookahead` reduces mean total cost by at
least 5% versus both `capacity_aware/FCFS` (allocation contrast) and
`threshold/lookahead` (request contrast), with paired 98.75% bootstrap upper
bounds on both cost differences below zero. Both must hold. H2: the interaction
contrast [joint − capacity_aware/FCFS − threshold/lookahead + threshold/FCFS]
is at most −2% of threshold/FCFS mean cost, with 98.75% bootstrap upper bound
below zero. H3 is descriptive: compare joint-versus-threshold/FCFS gain across
scarce, partitioned and ample conditions, without treating these three fixed
configurations as a random population of plants. Negative or small effects are
valid findings. A hypothesis is supported only when all its gates pass; not
supported when the complete panel is valid and gates fail; inconclusive if
integrity fails or the interval spans a practically important benefit and
harm. The exact point/interval and gate results are always reported.

## Panels, budget and stopping

No training: train seeds and checkpoints are empty/not applicable. Smoke uses
raw seeds 91000000–91000001; development uses 91000100–91000115; sealed test
uses 91000200–91000263; bootstrap seed 91000400. All are disjoint from readable
historical seed registries at protocol lock. Same condition/seed has the same
initial ages and pre-generated shock tape across the 12 arms. Smoke is for
engineering only, development for runtime/integrity checks only, never tuning
policy thresholds or selecting a winning arm. Full evaluates all 16 development
and 64 test seeds × 3 conditions × 12 arms = 2,880 complete episodes, 172,800
physical ticks. The runner stops after this fixed panel or 7,200 wall seconds,
whichever comes first; a time-limited run is FAILED, not analyzed as complete.
Smoke runs 2 seeds × 3 × 12 =72 episodes. Before full, smoke duration is used
only to verify the 7,200-second feasibility, without changing profiles or rules.
No later seed, metric, stopping or configuration change is confirmatory; any
follow-up gets a separately labeled exploratory protocol.

Paired seed differences are the independent units here because all policies
are deterministic and there is no trained model. For each contrast, bootstrap
the 64 aligned test seeds jointly 10,000 times with a role-separated RNG;
report mean difference, relative difference, percentile 98.75% CI, paired
standardized effect, and win fraction. This interval is conditional on these
fixed three synthetic configurations. Development/test outputs stay separate.

## Artifact and gate contract

Each run gets a fresh UTC timestamped directory and refuses existing content.
`manifest.json` records protocol/source hash, Git commit, settings, seed splits,
expected and actual counts, runtime and RUNNING/COMPLETED/FAILED. Snapshots of
protocol and seed registry, `resolved_config.json`, streaming
`episodes.partial.csv`, final `episodes.csv`, `decisions.csv` for test episodes,
`integrity.json`, `summary.json`, and a progress log are required. Full results
must pass complete unique keys, no split overlap, matching/skill/capacity
checks, request conservation, physical and component-cost reconciliation,
H/tail/terminal reconciliation, and no erased demand before any performance
summary is computed. All 12 arms share the same configuration and shock tape
per condition/seed. Tests cover service duration, degradation while pending,
failure conversion, committed service, terminal liability and artifact gates.
Use tqdm for multi-seed evaluation. Run tests and lab smoke, then commit/push
verified code before full; full runs here under the user's active override.
