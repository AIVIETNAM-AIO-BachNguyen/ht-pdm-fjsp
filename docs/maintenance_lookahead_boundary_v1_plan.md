## Material Passport

- Origin Skill: ars-codex:academic-research-suite / experiment-agent, inline.
- Origin Mode: prospective experiment authorized by user; Mac engineering smoke only.
- Origin Date: 2026-10-12.
- Verification Status: UNVERIFIED until tests/smoke; full outcomes unopened.
- Version Label: maintenance_lookahead_boundary_v1.

# Longer forecasts and baseline terminal continuation on one fixed setting

## Motivation, hypotheses and scope

The previous fresh-panel 12-tick experiment found assignment-only lookahead
worse than myopic; a recorded state [1,3,3,2] repeatedly returned after 12 ticks.
A root saving one forecast unit at tick 12 caused larger later realized costs.
This motivates a prospective boundary follow-up, not proof that every loss has
the same cause. Keep overlap3__fast1_slow325 as an exploratory candidate, with
no setting sweep, RL training, teacher labels or learned value function.

Primary H1: assignment-only rollout with baseline continuation to the episode
endpoint reduces mean H48+L12 realized cost by at least 1% versus myopic, with
availability/backlog guards. H2 (secondary): endpoint continuation improves
assignment-only realized cost versus the old 12-tick rule. Joint rollout is a
parallel scope diagnostic; 24 ticks tests a simple longer-window change. Never
select a winner or tune horizons from this panel. Failure of H1 does not prove
no assignment headroom or myopic optimality.

## Fixed physical environment and actions

Exactly the previous v1 environment: N4/K2; H48 measured plus common L12 tail,
without reset. Conditional failure at age >=4 with probability .45; max age8;
PM start1, CM start2, unavailability6 per machine tick, failure15, waiting
limit4 and overdue12. Restoration .75. Duration rows M0..M3 / T0,T1:
[(1,0),(1,3),(1,2),(1,5)]. Zero denotes ineligible; skill mask10,11,11,11.
Enumerate legal partial matchings including WAIT on free workers and unoccupied
failed or age>=3 machines. No early preventive action beyond v3's root space.
Same initial state and eligibility-independent physical shocks across all seven
methods on each episode seed. All 60 physical ticks run for every method.

## Seven locked methods and forecast contract

- myopic_matching: unchanged v3 deadline_matching urgency+restoration/duration;
  a current-state score, not an economic one-step optimizer.
- joint_h12, assignment_h12: previous 12-tick zero-terminal-value rule.
- joint_h24, assignment_h24: zero-terminal rule over min(24,60-current_tick).
- joint_endpoint, assignment_endpoint: 12-tick forecast plus Monte Carlo
  baseline terminal value, consisting of myopic cost from the forecast-end
  state through the fixed real episode endpoint tick60.

Joint enumerates all roots; assignment restricts roots to the current myopic
machine subset. Only the root is applied, then replan each real tick0..47.
Ticks48..59 always use myopic, for all methods. Forecast continuation after the
root always uses myopic, including the terminal extension. Endpoint scoring is
mathematically a remaining-episode-horizon myopic rollout: no claim that a
separate learned value head was introduced or that it reduces compute. The
terminal part predicts baseline continuation, not future replanning. Endpoint
knowledge is the fixed benchmark time limit, never actual future shocks.

Full128 scenarios/root, smoke8. Share domain-separated selection bank across
roots and all six planning methods at each episode/tick, with identical random
tape prefixes across horizons. Real shocks are separate. Myopic root first,
deterministic remaining order; myopic wins absolute1e-10 ties. Single-root
choices and all tail decisions skip forecasting. Scores, 12-tick prefix and
terminal continuation components are logged and must reconcile.

## Fresh seed splits, no tuning, stopping

Registry configs/maintenance_lookahead_boundary_v1_seed_registry.json.
Full evaluation39000000..39000127; bootstrap39100000; forecast39200000.
Smoke39300000..39300003; bootstrap39400000; forecast39500000.
Train/development panels empty. No smoke-performance or full-panel tuning.
Physical role VERSION:physical derived from episode; selection and independent
holdout VERSION:<phase>:<episode>:<tick> from forecast raw seed; bootstrap role
VERSION:bootstrap. Raw/derived uniqueness audited against all historical raw
registries, v1-v3 screen derived roles and both previous myopic-lookahead panels.
Intentional physical/CRN sharing across methods is not seed leakage.

Fixed full896 episodes/53,760 physical ticks, 896 common landmarks; smoke28
episodes/1,680 ticks, 28 landmarks. Maximum17 roots from this mask asserted;
conservative compute cap uses21 roots. Forecast cap:
seeds*21*scenarios*[2*sum(t=0..47)(12+min(24,60-t)+(60-t))
 +2*sum(t in landmarks)(60-t)]. The landmark calculation shares a single
max-horizon bank to obtain all prefix values, so it is counted once/bank.
Cap excludes separately logged replay compute. Exactly128 full seeds, no
outcome stop, adaptive scenarios, fallback, resume or setting replacement.
Timeout600s smoke/7200s full, checked inside forecasts; failures write FAILED
and require a fresh timestamp directory after diagnosing engineering cause.
Old-state fixture is allowed for engineering regression only, not evaluation.

## Metrics, statistical family and decision rule

Primary: 1-mean(assignment_endpoint extended_cost)/mean(myopic extended_cost).
Extended cost includes all H+L economic/waiting components. Bootstrap20,000
paired seed-cluster draws (smoke200), ratio of mean costs, retain all methods.
Twelve fixed two-sided percentile intervals have confidence1-.05/12
(Bonferroni nominal family>=95%; bootstrap approximate, not finite-sample
guarantee): six planning-method cost reductions versus myopic; endpoint versus
h12 cost reduction within each of two scopes; H availability difference and
H terminal-pending difference endpoint-minus-myopic within each scope.

ASSIGNMENT_HEADROOM_CONFIRMED iff primary CI lower>=.01, assignment availability
lower>=-.01 and pending upper<=.10; otherwise NOT_CONFIRMED. Secondary joint
headroom flag uses the same three margins, never replaces primary. Secondary
boundary-repair flag requires assignment endpoint-vs-h12 cost CI lower>0; it
does not by itself establish advantage versus myopic. Smoke always
ENGINEERING_ONLY regardless of arithmetic gates. Report all methods/losses.

Descriptive: H and tail costs, components, failures/overdue ticks, availability,
utilization, H/tail pending and failed machines; v3 contention rates on each
policy's own states; disagreements, same-subset/subset/cardinality/WAIT changes,
forecast transitions and latency. Denominator H48. These are policy-dependent,
not renewed setting-selection gates.

## Common-state mechanism diagnostics

Myopic states at ticks[0,8,16,24,32,40,47] for every seed, no selected-state
sampling. Evaluate every root with one selection bank to remaining endpoint,
retain cost samples at12, clipped24 and endpoint. Independently evaluate all
roots with a separate holdout bank on the same three checkpoints. Select each
of six rules using its selection-window mean only. Log heldout paired gain at
its own scoring horizon AND at endpoint, including zero for unchanged roots;
endpoint terminal component is endpoint minus12 prefix. Root CSV contains each
checkpoint mean/SE and selected flags. Diagnostic 95% seed-cluster CIs on mean
endpoint root gain per seed are exploratory, never a gate or training data.
These forecast values are not realized episode savings. Common-bank prefixes
isolate boundary/scoring differences without new Monte Carlo draws per method.

## Engineering gate and artifact schema

Tests plus a real macOS smoke before commit/push: scalar/vector equivalence,
prefix CRN equality, score/terminal decomposition, known prior-state horizon
regression, baseline/eligibility/ties, assignment restriction, no actual future
events, independent holdout not used for selection, paired tapes, seed audit,
bootstrap family/gate arithmetic, replay tamper checks, overwrite refusal,
timeout, clean committed full source, complete artifact schema and hashes.
No checkpoint required: no learning. Scientific outcomes open only on clean
committed full lab run; smoke is engineering only.

Fresh timestamp directory: manifest.json, protocol.md, seed_registry.json,
resolved_config.json, episodes.csv, streaming episodes.partial.csv, decisions.csv
(all physical states/actions/events/following/info and score identity/components),
landmarks.csv, landmark_roots.csv, summary.json, audit.json, source_snapshot/
(new runner, previous runner, screens, scalar/vector environment/helpers and
four inherited registries). SHA256/bytes for every file except manifest;
manifest records Git/runtime/seeds/full-open flag, status, expected/actual grids,
forecast and replay budgets. Live sources/protocol/registries sealed throughout.

Replay every physical action with scalar transition, costs/metrics and paired
tapes; regenerate all six planners and all landmark root banks for first two
full seeds/all smoke seeds. All other logged choices must satisfy feasible
root restrictions and finite scores<=baseline; this partial forecast audit is
explicit and can be expanded during subsequent analysis. Root/landmark grids
must be complete; no invalid assignments, NaN/Inf or hidden partial episodes.

## Interpretation and handoff

Finite-window bias is a hypothesis; remaining-horizon rollout still predicts
myopic continuation and may fail under closed-loop replanning. Any result is
known-model planning evidence in one exploratory setting, not RL learnability,
necessity of MARL or confirmation of v3's five setting gates. No oracle shocks.
After Mac gate, commit/push and stop at user handoff with exact lab compound
command and local rsync -avhP --partial. Do not SSH, run full or transfer.
