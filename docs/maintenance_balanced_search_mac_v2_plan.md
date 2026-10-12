# Balanced 2–2 search, round2 — prospective protocol

## Material Passport
Synthetic CPU Mac maintenance simulator; a new finite design after round1
NO_SCREEN_CANDIDATE. All previous failed artifacts remain intact; no previous
confirmation panel is reused. User explicitly authorizes continued autonomous
experiments on this Mac until balanced2–2 passes. No RL, external datasets,
human subjects or lab handoff. Native bounded scientific review in shared
context supports this design; independent verification is not blind/cross-model.

## Hypothesis, variables and interpretation
Increasing fast repair restoration in round1 either collapsed local allocation
value or retained availability loss. Return to q.50 fast/q1 slow, with a real
routinePMage3 restoration contrast: floor age1 vsage0. New hypotheses:
(1) raising downtime cost directly changes rollout's objective toward uptime;
(2) mirrored mismatch-time heterogeneity gives occupied-worker opportunity
costs beyond isolated per-machine DP, while retaining symmetric2–2 expertise.

Exactly eight selectable profiles: wrong-worker time pairs(a,b) in
{(2,2),(2,4),(3,3),(3,5)} crossed with unavailable_costu in{9,12}.
Service [[1,a],[1,b],[a,1],[b,1]], restoration
[[.5,1],[.5,1],[1,.5],[1,.5]]. Bothtechnicians eligible on everymachine;
eachfaster onexactlytwo; mirroredclasses preserve technician symmetry.
Uniformpairs versus heterogeneouspairs give a simple mechanistic contrast.
Candidate IDs balanced_a{a}_b{b}_u{u:02d}. Controls nonselectable: oldbalanced
(a,b)=(3,3),down6/q.50; exchangeable duration1/rest.75/down6. No fallbackto
controls. Full10profiles; smoke selectableanchors(2,4,u9)/(3,5,u12)+2controls.
All otherphysics/reward components are unchanged: N4/K2,H48+deadline tailL12,
noreset, failureage4,p.45,maxage8,PM1/CM2,failure15,waitlimit4/overdue12,
urgent partialmatchings+WAIT. Only unavailablecostchanges; not a rescaling of
all costs. Higher local monetary gain alone cannot establish stronger physical
benefit: report unavailableticks/failures/overdueticks and components separately.
Future quality/capacity effects remain adaptive after an isolated currentroot.

## Methods, development and screening
Three cheap methods: deadline_matching, isolated_h6, isolated_endpoint. Choose
one baseline PER profile on development24 by meanH+Lcost, methodID ties. Freeze
this before screening. Assignment rollout changes only the worker allocation
of the cheap subset at the current state/tick; joint rollout can also change
subset/timing. Both forecast128selectionCRNscenarios per root to endpoint60,
cheap continuation through48 and common deadline tail. Replan eachtick.
Use the previously verified engine unchanged, including tie-breaking.

Evaluate all ten profiles on32screen seeds with five methods, local assays
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
Full dev45000000..45000023, screen45100000..45100031,
confirm45200000..45200255, bootstrap45300000,forecast45400000.
Smoke dev45500000..1, screen45600000..1,confirm45700000..1,
bootstrap45800000,forecast45900000. No training.
All rawroles disjoint across profiles/splits/history, start/count registries
expanded. Reuse namespace maintenance_cross_skill_mac_v1 for SHA256-derived
physical/selection/holdout/bootstraproles with newrawseeds. Intentional physical
pairing across methods/profiles, CRN pairing across roots; distinct selection
and holdoutbanks. No outcomes from unopened confirmation may affect selection.

The new series began prospectively at round1. Round2 familyalpha=.0125,
confidence1−.0125/18=99.930556%,160,000seedclusterbootstrapdraws; round1alpha.025
is not recycled. Future roundr familyalpha=.05/2**r and bootstrap80,000*2**(r−1),
registered newdesign/newpanel beforeexecution. Approximate bootstrap/spending
interpretation concerns this newseries only, not earlieradaptive research.
All18physical/cost/health thresholds remain EXACTLYunchanged, with stricter
confidence. No retry of unchanged failed designs on favorable newseeds.

Fullhardlimit7200s,smoke600s. Fixed10*24*3development +10*32*5screening and
atmost256*5confirmationepisodes; nooptionalstopping/seedextension. Forecastcap
counts everylegalroot/horizon/scenario. Save failures and no-winner status.

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
Full ifselected:720dev+1600screen+1280confirm=3600episodes/216000ticks,
4032landmarks. Replay everyphysicalstate/event/cost androotgrid; regenerate
plannerforecasts/heldoutbanks for first2rawseeds per phase/profile; assert
zero invalid assignments, exactgrids, source/choiceprovenance, freshseedroles,
nullcontrol, cap/time budgets. Independent scalar replay/indexedbootstrap
verifier aftercompletion; disclose sharedcontext and actual independence.

## Interpretation and eleven fallacy checks
Local root assignment effect isolates only currentstate/tick/subset; subsequent
continuation and fulltrajectories may diverge. Rollout is not optimal, learned
or MARL evidence; synthetic setting is not fieldvalidation. Report physical outcome/costcomponents and choice/diversity coverage. No direct comparison to
older3–1 or2–2 runs on differentpanels; new pairedcontrol comparisons remain
exploratory, not extra confirmation claims.
1.Testtuning absent. 2.Optionalstopping absent. 3.Fallback absent.
4.No landmarkpseudoreplication. 5.No selection/holdoutleakage.
6.Zerochoicerows retained. 7.Common tailincluded. 8.Healthguard required.
9.No future-schedule isolation. 10.No optimality/RL claims.
11.Alphaspending prospective fornewseries only; approximatebootstrap limits.
