# Balanced 2–2 search, round1 — prospective protocol

## Material Passport
Synthetic CPU maintenance simulator, finite setting search with fresh heldout
confirmation; no participant data, external dataset or RL training. User has
explicitly authorized autonomous planning/implementation/full experiments on
this Mac until a balanced setting passes. This supersedes lab handoff and the
ARS code-runner's command-only/confirmation restrictions. Preserve all failures.

## Hypothesis and setting family
Previous balanced d3/q.50 confirmation passed17/18guards: allocation availability
noninferiority failed, despite cost improvement. Hypothesis: increasing fast
repair restoration improves future maintenance/occupancy and availability,
while reciprocal speed expertise preserves enough assignment value.
Select exactly six profiles d in{2,3,4}, q in{.625,.750}. Every profile has
service [[1,d],[1,d],[d,1],[d,1]], restoration
[[q,1],[q,1],[1,q],[1,q]]. Both technicians eligible on all four machines; each
is faster on exactly two. Nonselectable controls: old balanced d3/q.50 and
exchangeable duration1/restoration.75. Controls retained even if better.
Full8profiles; smoke2selectable anchors d2/q.625 and d4/q.75 plusbothcontrols.

Restoration is floor(age*(1-q)) at completion: at ordinary urgent PMage3,
q.625 leavesage1, q.75 leavesage0 just like q1. Thus q.75 principally tests
speed/occupancy coordination at ordinary PM, with quality contrasts at some
failed ages. Do not claim uniform quality differences from matrix entries.
All other physics unchanged: N4/K2, H48+common deadline tailL12 without reset,
failure age4, conditional failure probability.45, max age8; PM1/CM2,
unavailable6/machine-tick, failure15, waitlimit4/overdue12. Only urgent idle
machines (failed orage>=3) and freeworkers can start; WAIT and legal partial
matchings remain available.

## Methods, development and screening
Three cheap methods: deadline_matching, isolated_h6, isolated_endpoint. Choose
one baseline PER profile on development24 by meanH+Lcost, methodID ties. Freeze
this before screening. Assignment rollout changes only the worker allocation
of the cheap subset at the current state/tick; joint rollout can also change
subset/timing. Both forecast128selectionCRNscenarios per root to endpoint60,
cheap continuation through48 and common deadline tail. Replan eachtick.
Use the previously verified engine unchanged, including tie-breaking.

Evaluate all eight profiles on32screen seeds with five methods, local assays
on selected baseline states at ticks0,8,16,24,32,40,47. A candidate is eligible
only under the original strict screen point gates: assignment and all3joint
H+Lcostgains>=.015; all3localheldoutgains>=.75; choicefraction>=.15; BOTH
planners avdiff>=−.005, pendingdiff<=.05, absav>=.4, abspending<=.5, Hgain>0.
Among eligible selectable candidates choose max health stability score:
min over assignment/joint of(mean paired availabilitydifference minus
2*sampleSD(ddof1)/sqrt(32)). Ties choose greatest minimum of the four H+Lcost
gains, then lexicalcandidateID. This score is a ranking heuristic, NOT a
confidence interval or an extra scientific claim. Health-first ranking responds
to the previous sole failed guard. Exactly ONE frozen winner proceeds to256
confirmation seeds. No eligible candidate: NO_SCREEN_CANDIDATE, no confirmation
opened. Failed confirmation: keep NOT_CONFIRMED, no runner-up fallback.
Smoke may choose a selectable anchor for engineering coverage, never science.

## Seeds, spending, stopping and effort
Full dev44000000..44000023, screen44100000..44100031,
confirm44200000..44200255, bootstrap44300000,forecast44400000.
Smoke dev44500000..1, screen44600000..1,confirm44700000..1,
bootstrap44800000,forecast44900000. No training.
All rawroles disjoint across profiles/splits/history, start/count registries
expanded. Reuse namespace maintenance_cross_skill_mac_v1 for SHA256-derived
physical/selection/holdout/bootstraproles with newrawseeds. Intentional physical
pairing across methods/profiles, CRN pairing across roots; distinct selection
and holdoutbanks. No outcomes from unopened confirmation may affect selection.

This begins a NEW prospectively budgeted search series. Roundr familyalpha is
.05/2**r; unused alpha is not recycled. Firstalpha.025,18CIconfidence
1−.025/18=99.861111%;80,000seedcluster bootstrap draws. If a future round is
needed, register a new design and new disjoint panel before execution; doubles
bootstrap draws eachround so tail-count resolution stays comparable. Approximate
per-protocol bootstrap/spending interpretation does not retroactively adjust
previous adaptive searches or prove exact global coverage. Existing physical,
cost and health THRESHOLDS stay unchanged. This confidence is stricter than
previous99.722%. Do not rerun same design on favorable newseeds to mask a fail.
Full hardlimit7200s, smoke600s. Full fixed8*24*3development +8*32*5screening
and atmost256*5confirmation episodes. Forecast cap counts every legal root,
all horizons/scenarios. No optional stopping or seed extension. Save failures.

## Eighteen confirmation gates — thresholds unchanged
H+L ratio-of-means cost gain lowerCI>=.01 for assignment vs frozencheapbase
and jointvs EACH3cheapmethod (4gates). Local assay fixes initialstate/tick/
subset, choosesONEroot on selectionbank and evaluateson independentholdout
vsbaseline/quality(duration/restoration)/isolatedendpointDP. All3local mean
heldoutgain lowerCI>=.5cost/commonlandmark. Keep WAIT/nochoicezeros; average
sevenlandmarks withinseed before bootstrap.
For BOTHplanners: avdiff lowerCI>=−.01, pendingdiffupper<=.1, absavlower>=.4,
abspendingupper<=1 (8gates). Assignmentchoicefractionlower>=.1 (1gate);
H-onlygainlower>0 forBOTH (2gates). All18requiredfor
CROSS_SKILL_SETTING_CONFIRMED; never promote a passing subset.

## Engineering and artifact contract
Unit/regression tests and small local CLI CPU smoke, commit/push code before
fullonMac, tqdm evaluation. New UTC timestamped run directory; no overwrite.
Schema: manifest.json git/runtime/stages/hashes/counts; resolved_config;
protocol/registry/source/historical-registry SHA256snapshots; candidates;
development_episodes; frozenbaseline_choices; screening_episodes/landmarks/
roots; sealed selection.json; decisions.csv everyphysicaltick;
episodes.partial.csv/episodes.csv; landmarks/landmark_roots; summary/audit.
Full ifselected:576dev+1280screen+1280confirm=3136episodes/188160ticks,
3584landmarks. Replay everyphysicalstate/event/cost androotgrid; regenerate
plannerforecasts/heldoutbanks for first2rawseeds per phase/profile; assert
zero invalid assignments, exactgrids, source/choiceprovenance, freshseedroles,
nullcontrol, cap/time budgets. Independent scalar replay/indexedbootstrap
verifier aftercompletion; disclose sharedcontext and actual independence.

## Interpretation and eleven fallacy checks
Local root assignment effect isolates only currentstate/tick/subset; subsequent
continuation and fulltrajectories may diverge. Rollout is not optimal, learned
or MARL evidence; synthetic setting is not fieldvalidation. Report quality
floor equivalences and choice/diversity coverage. No direct comparison to
older3–1 or2–2 runs on differentpanels; new pairedcontrol comparisons remain
exploratory, not extra confirmation claims.
1.Testtuning absent. 2.Optionalstopping absent. 3.Fallback absent.
4.No landmarkpseudoreplication. 5.No selection/holdoutleakage.
6.Zerochoicerows retained. 7.Common tailincluded. 8.Healthguard required.
9.No future-schedule isolation. 10.No optimality/RL claims.
11.Alphaspending prospective fornewseries only; approximatebootstrap limits.
