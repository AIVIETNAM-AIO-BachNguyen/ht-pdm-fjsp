# Balanced 2–2 standalone confirmation — locked before execution

Material Passport: synthetic maintenance simulator; one fixed author-requested
profile; reuse of previously verified deterministic experiment engine; no RL
training, external datasets, or new literature claims. CPU Mac execution is
explicitly authorized by the human in this conversation, superseding lab handoff.

## Question and hypothesis
Does reciprocal expertise in a balanced 2–2 machine mix retain a measurable
allocation benefit and joint planning benefit without unacceptable health loss?
One candidate, `cross_d3_q050_balanced`, frozen from the earlier 24-seed screen.
That candidate failed the previous screen's assignment availability guard;
this is an explicitly requested fresh standalone validation, not an original
screen winner or a fallback within the previous protocol. Report PASS or FAIL.

## Fixed setting and methods
N=4, K=2; 48 policy ticks and 12 common deadline-tail ticks, without reset.
Service times [[1,3],[1,3],[3,1],[3,1]], restoration
[[.5,1],[.5,1],[1,.5],[1,.5]]. Every technician can work on every machine.
Failure age 4, conditional probability .45, max age 8; PM/CM cost 1/2,
unavailability 6 per machine-tick, failure cost 15, overdue wait limit 4 and
price 12. Unchanged urgent partial matching/WAIT legality.
Evaluate deadline_matching, isolated_h6, isolated_endpoint, assignment_rollout,
joint_rollout. Freeze isolated_endpoint as baseline from old development16;
do not choose a baseline using new confirmation outcomes. Original development
means, commit and baseline-file SHA256 are in the prospective registry.
Forecast 128 CRN scenarios/root, baseline continuation to tick48, deadline tail
to tick60; initial root replanned at each tick. Allocation-only intervention
fixes current state, tick and chosen subset; future schedules may diverge.

## Seeds and stopping
No training, new development, or new screening. Full physical raw seeds
43000000..43000255, bootstrap43100000, forecast43200000. Smoke physical raw
43300000..43300001, bootstrap43400000, forecast43500000. Fresh disjoint raw
roles audited against every prior registry. Reuse engine namespace
maintenance_cross_skill_mac_v1 for physical, selection, holdout and bootstrap
SHA256 derivations; fresh raw panels make streams fresh. Intentionally paired
physical shocks across the five methods; paired selection/holdout CRN across
roots, independent selection and holdout roles. No optional stopping or retries
with other panels. 7200-second full hard limit; 600 smoke; 20,000 seed-cluster
bootstrap samples full (200 smoke); max registered forecast transition cap.
Failures retain artifacts. Smoke never supplies scientific evidence.

## Metrics and eighteen locked gates
Primary: H+L total-cost ratio-of-means reductions, assignment vs frozen baseline
and joint vs each of three cheap baselines. All four lower CI bounds >=1%.
Local assay at baseline states ticks0,8,16,24,32,40,47: fix state/tick/subset,
select one root on selection bank, evaluate it on separate holdout bank vs
baseline, duration/restoration quality heuristic, and isolated endpoint DP.
All three local gain lower bounds >=.5 cost per common landmark, keeping zero
choice/WAIT landmarks and averaging seven landmarks within each raw seed.
For each planner vs baseline: availability difference lower bound >=−.01,
pending difference upper bound <=.10; absolute availability lower bound >=.4,
absolute terminal pending upper bound <=1. Assignment choice fraction lower
bound >=.10; each planner's H-only cost gain lower bound >0. Eighteen fixed
percentile seed-cluster bootstrap CIs at 1−.05/18 confidence (~99.722%). All
18 must pass for CROSS_SKILL_SETTING_CONFIRMED; otherwise NOT_CONFIRMED.
No guard relaxation or promotion of a subset of passing results.
These are approximate per-protocol simultaneous intervals, not family-wide
coverage over all earlier research searches or future trials.

## Validation, artifacts and interpretation
Run regression/unit/integration tests and two-seed CPU smoke first, commit/push
verified source, then full on this Mac. tqdm evaluation progress required.
New UTC timestamp artifact directory only. Store manifest/config/candidate,
prospective protocol/registry/source and historical-registry SHA256 snapshots,
frozen baseline_choices and author selection, decisions.csv for every tick,
episodes.partial.csv and episodes.csv, landmarks.csv and landmark_roots.csv,
summary.json and audit.json. Full grid1280episodes/76800ticks/1792landmarks.
Replay all physical transitions/costs/root grids; regenerate planner and local
forecast banks for first two raw seeds. Verify frozen source/choices, counts,
zero invalid assignments, fresh streams and cost aggregation. Independently
recompute scalar physics and indexed seed-cluster bootstrap after completion.
No claim that rollout is an optimal controller, learned policy, MARL result,
field validation, or that 2–2 is superior to3–1 on unpaired seed panels.

## Eleven interpretation checks
1. No test-set tuning. 2. No optional stopping. 3. No silent fallback.
4. No seed/landmark pseudoreplication. 5. No CRN selection-bank leakage into
holdout. 6. No zero-choice exclusion. 7. No horizon-end deferral without tail.
8. No cost-only victory ignoring health. 9. No future-schedule isolation claim.
10. No optimality/RL claim from rollout. 11. No unlimited-search coverage claim.
