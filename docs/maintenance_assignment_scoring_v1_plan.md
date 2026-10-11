## Material Passport

- Origin Skill: ars-codex:academic-research-suite / experiment-agent, inline.
- Origin Mode: authorized implementation and local smoke, user-run full lab.
- Origin Date: 2026-10-11.
- Verification Status: PROTOCOL LOCKED BEFORE NEW OUTCOMES.
- Version Label: maintenance_assignment_scoring_v1.
- Overall Confidence: small exploratory algorithm pilot; not novelty confirmation.

## Question and intervention

Can explicit candidate–resource interaction improve RL assignment scores, and
can exact conditional matching-mass decoding improve technician selection?
Train two fresh PPO arms: additive (original autoregressive context scorer),
interaction (same weights/shapes, add pair * mean(global, remaining-machine,
remaining-technician context) to candidate-machine actor feature). No added
parameters; STOP candidate pair/machine features remain zero. Critic, masks,
reward, optimizer, sampled sequence likelihood and complete returns unchanged.
This is an architectural hypothesis, not an established cause of prior weakness.

Evaluate each trained arm with greedy token decoding and assignment_mass:
first obtain its greedy matching, then maximize summed ordered-sequence mass
only among matchings with EXACTLY the same selected machine set and EXACTLY
its CM edges. This changes only PM technicians at the current state. No rollout
oracle, future event or heuristic labels enter training or this decoder. At
subsequent states trajectories and timing can diverge; episode gains measure
closed-loop consequences, not a timing-controlled entire episode.

All deterministic token decoders choose the lowest valid token within1e-5
logit units of the maximum. Matching-mass ties within1e-6 prefer the original
greedy matching, then canonical order. This common inference rule addresses
previous Mac/lab near ties, but is not a guarantee against every floating-point
boundary case. Stochastic training probabilities remain unrounded.

## Frozen budget and splits

N4/K2, dense and skill_mask, same generator family with randomly permuted roles.
H48 learned decisions followed by12 deadline_matching ticks; reward=-cost/20,
complete undiscounted return including tail. Two arms x3 replicas, each122880
learned steps,32 vector environments, rollout3072, minibatch256,4 PPO epochs,
hidden32, lr3e-4, clip.2, entropy.01, value.5, gradient clip.5.
Checkpoint updates0/10/20/30/40; select trained minimum mean development H+tail
cost under greedy decoding, earliest tie, BEFORE test. Both decoders share that
checkpoint; no decoder-specific tuning. Select strongest of seven rules on dev.
All three seeds retained. Fixed completion, no favorable-seed selection or
outcome-driven early stop. Fresh timestamps, no automatic resume/overwrite.

Full: train18100000..2, dev18200000..5 x18300000/1,
test18400000..23 x18500000..3, analysis18600000,
diagnostic selection18800000, independent evaluation18900000.
Smoke18700000-series (separate raw roles), H4+2, hidden16, two updates64steps/model.
Companion registry is authoritative. All previous raw registries checked;
canonical physical signatures exclude previous history, pilot train/dev/test,
latest diagnostic full/smoke, and all new train/dev/test/full/smoke metadata.
Metadata reconstruction may resolve full configs but never execute full outcomes
on Mac. Training configs, physical tapes, action/minibatch streams and initial
weights paired across arms. No checkpoint dependency or previous training data.

## Metrics and falsification

Four primary contrasts: interaction vs additive under greedy and under
assignment_mass, separately dense/skill_mask. Positive relative gain =
(baseline cost - method cost)/baseline mean cost. Report total H+tail objective,
base/tail and four cost components, per-replica and leave-one-replica-out gains,
20000 crossed replica/profile bootstrap draws with98.75% two-sided percentile
intervals (approximate Bonferroni .05/4), fixed physical shocks. Material margin1%:
lower bound>1% = pilot signal, upper<1% = below margin, otherwise unresolved.
Three replicas limit calibration/power. Secondary95% decoder effects within
each scorer, interaction-vs-rule, and timing/service behavior, explicitly
exploratory. No claim of learned optimal assignment from episode cost alone.

## Separate timing and assignment diagnostic

After selection/test, use a fixed panel of common states from the ADDITIVE
GREEDY trajectories: first6 sealed test profiles x first2 shocks x both skill
conditions x each paired replica x landmarks0/6/18/30/42 =360 states. Evaluate
both frozen scorers greedily on every common state (720 model-state records).
Smoke uses first profile and two shocks with landmarks0/3 at H4+2.
For each model root preserve selected CM exactly. timing menu keeps original
PM workers and omits a subset of originally selected PM machines; assignment
menu keeps all selected PM machines and varies workers. Do not conflate menus.
32 selection futures/root choose minimum mean within each menu, original wins
ties;64 independent evaluation futures/root estimate paired correction gain.
Both scorers share phase tapes on each common state. Root at t, its own greedy
RL at t+1..H-1, shared deadline tail H..H+11; never restart clock. Save all roots,
phase costs/tapes, boundaries, components and matching masses. Include no-choice
states with zero exposure; report denominator and coverage by time. Conditional
correction gaps are descriptive, not primary architecture contrasts and cannot
be compared across differing exposures as if on identical populations. Direct
conditional mass decoder root gain also saved with independent futures.
Monte Carlo labels never become training targets. Smoke2 futures/phase and
scalar replay all candidates/futures. Full diagnostic budget upper bound
720 x7 x96 x40.8 =19,740,672 forecast transitions, actual counts audited.

## Artifacts and gates

manifest/settings/plan/source/config snapshots; training_progress,
training_episodes, development_episodes, checkpoint_selection/coverage,
checkpoints/model.pt, episodes.partial/episodes, decisions, coordination,
PM choices, requests/carryover/time profiles, summary/profile differences.
Separate diagnostic_states, diagnostic_roots, diagnostic_audits, per-state NPZ
with phase tapes and costs/cutoff/terminal/components; diagnostic_summary with
separate timing and assignment exposure/gain strata. Immutable artifact hashes.

Tests: score/likelihood normalization and gradients; matched parameter/init
counts; STOP and live masks; exact sum over both pair orders; fixed machine/CM
assignment decode; near-tie policy; split disjointness; horizon/scalar/vector
replay and independent selection; checkpoint save/load; expected complete grid.
Full existing suite and Mac CLI smoke must pass before commit/push. Smoke is
engineering evidence only; full requires clean commit and CPU thread env1.
No full run or rsync by assistant; exact commands handed off per workflow.
