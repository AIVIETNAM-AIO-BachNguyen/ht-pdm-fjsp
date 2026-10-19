# Shared maintenance request lifecycle v1

## Material Passport
Synthetic simulator engineering and prospective diagnostic, authorized by the
user on 2026-10-19. Base commit 2e190435bde46e99ac608ea19a3413a9bcc3a08c.
No learning, human participants or claim of a novel algorithm. Mac smoke is
engineering evidence only; full diagnostic is handed off for the user to run.

## Hypotheses and scope
H1 (engineering): an unserved PM request persists while its machine operates
and degrades, upgrades to CM upon failure without losing its creation time,
and completes only when its assigned nonpreemptive service finishes. Common
pending requests can use any eligible free technician; no duplicate requests,
double assignments or premature restoration are allowed.
H2 (diagnostic, not assumed): changing request and dispatch rules changes
paired extended economic cost, waiting exposure and backlog in some physical
profiles. Report all cells and differences, including losses and zeros. A
zero/negative rollout gain is an outcome, not an engineering failure.

Reuse the pure completion-time physical transition in maintenance_dispatch.py
without changing historical environments. New request layer has no cancellation,
one active request per machine, automatic CM after failure, PM-to-CM upgrade,
and no technician binding until service starts. PM machines run while queued;
failed machines stop; machines in service stop and do not age. New PM requests
and service starts occur at a tick boundary, physical failures and completions
at the next boundary. Requests created by a new failure are available next tick.
Creation timestamps survive PM-to-CM upgrade. Request waiting ends at service
start; service time and waiting are reported separately.

IMPORTANT IDENTIFIABILITY LIMIT: requesting earlier has no intrinsic monetary
cost and does not stop production. An unconstrained centralized dispatcher can
ignore an early request, so these assumptions cannot establish a fundamental
advantage of delayed request submission over submitting all requests early.
Requester comparisons evaluate the specified operational rules only. Do not
invent request penalties to manufacture a timing tradeoff. Future work must
justify any communication/commitment constraint before stronger timing claims.

## Locked diagnostic
Family: N4/K2, hazard after age4 p=.45, max age8, costs PM1/CM2,
unavailable6/failure15. Three service scales 1,2,3 crossed with three skill
profiles: interchangeable, reciprocal specialization, specialist/generalist.
Repair restores .75 on every eligible edge, so duration/eligibility are isolated.
Smoke uses the three profiles at scale2; full uses all nine. H12+tail8 smoke;
H48+tail16 full. Both phases continue without reset. Tail requester is threshold
and tail dispatcher is risk matching for every method. Remaining work is
reported at both cutoffs; finite tail does not prove absence of endpoint bias.

Factorial: threshold requester (age>=3) versus resource-aware requester
(age>=max(1,3-minimum remaining time of eligible workers)); each crossed with
FCFS, risk matching and rollout dispatch. FCFS uses a shared ordered queue and
chooses the fastest free eligible worker per oldest request. Risk matching
maximizes positive urgency/restoration-per-duration scores over all legal
partial matchings including WAIT. These are disclosed heuristics, not exact
economic optimizers. Rollout enumerates all currently legal matchings on the
same submitted pool; future simulated decisions use the matching requester and
risk dispatcher. Horizon3/scenarios2 smoke; horizon6/scenarios16 full. It uses
known simulator dynamics, independent hypothetical shocks and CRN across roots,
never the actual future tape. Include WAIT; risk action wins numerical ties.
Root evaluations account for the fixed tail policy when crossing H.

No training/development/test split is needed for this fixed nonlearning
diagnostic. Smoke seeds61100000..61100001, full evaluation61000000..61000063;
forecast rawseed61200000, bootstrap rawseed61300000. Domain-separated streams;
same physical seed across all cells and methods is intentional pairing.
Audit against historical registries, including start/count ranges. These full
seeds are prospective diagnostic seeds, not an unopened confirmatory panel for
subsequent algorithm selection. No tuning, seed extension or outcome stopping.

Primary: mean H+tail economic cost (maintenance+unavailability+failures).
Secondary: H availability, failures, queued PM/CM ticks, all-request wait
exposure, maximum observed queued age, completed-request wait, pending and
in-service counts at H and endpoint, tail cost, controller time. Do not report
completed-only waiting as the system waiting statistic. No waiting penalty.
Paired rollout-minus-risk and resource-minus-threshold differences are
descriptive; optional 95% paired seed-bootstrap intervals are exploratory,
not a familywise claim or an algorithm superiority gate.

## Stopping, audits and artifacts
Fixed grid; hard timeout600s smoke/14400s full. No automatic resume. Failures
preserve FAILED manifests; each invocation requires a new empty directory.
Progress via tqdm over every multiseed evaluation; no training occurs.
Before push: lifecycle unit tests, deterministic scalar fixtures, legacy
regression, CLI smoke, source hash/cost/grid checks and replay of all smoke
physical transitions. Full run requires clean tracked source and no benchmark
parameter overrides. Do not SSH, execute full locally or transfer results.

Artifacts: manifest.json (status, commit, dirty, runtime, profile, source hashes,
seed panels, counts), protocol.md, seed_registry.json, configs.json,
episodes.partial.csv, episodes.csv, decisions.jsonl (full state/action/events/
following/cost/lifecycle), summary.json, audit.json and source_snapshot/.
No model/checkpoint applies. Replay validates all transitions, request invariants,
cost decomposition and episode metric sums. Raw artifacts never overwritten.
