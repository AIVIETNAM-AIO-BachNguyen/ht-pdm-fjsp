# Next bounded design: physical entropy aggregation, not a new RL objective

Design note only, NOT a learning protocol or permission to launch an unspecified
trial. Current confirmation remains FAILED4/12CI and one primary replica sign.
Frozen optimization audit60feac5 rejected its locked 2x-budget entry rule:
meanlateTV=.001773<.01, despite4/5positiveCFDEVcosttrends. Do NOT run a budget rescue,
change its threshold or claim this proves global convergence. Its240states came
from one exogenous CFDEVseed; limitation remains explicit.

## Specific next question

Original typed-Gibbs policy uses scalar JOINT physical PPO ratio/state advantage,
but entropy regularizer is H(physicalmatching)/N. Original catalogue PPO uses
H(physicalmatching), with the same nominal beta=.01. AtN4, these differ by a known
factor4 in reward-relative regularization. This is an intentional normalization
choice, NOT an algebra/likelihood bug or proof of harmful entropy. All original
core ablations shared H/N, so their failed scientific verdict remains valid.

Hypothesis to test once: using whole-matching entropy at beta=.01 (no divisionN)
can delay probability saturation enough to make learned-future coordination
reliable. A normalization choice is engineering tuning, not the algorithmic
contribution. Main contribution would STILL need weighted future matching
messages versus a SAME-entropy equally informed/capacity/budget myopic control,
near-PPO quality/health and later measured scaling. No teacher/labels/distillation/
warmstart, encoder/LR/attention/critic/objective changes or longer budget.

Current source CFDEVfixed-state audit shows near-deterministic policies and tiny
late movement; that motivates a controlled test, not a causal conclusion about
why the fifth candidate or specific myopic replicas failed. Prior broker entropy
alias-opposition diagnostic was NEGATIVE and stays negative: that compared Hp
versus Hq at same /K units, while this asks about magnitude H versus H/N for an
already exact physical distribution. Do not relabel or reopen that old remedy.

## Necessary factorial comparison / scope

Use3density families Gibbs/myopic/reverse-myopic ×2entropy aggregations jointH
versus originalH/N. SAME NN/inputs/18883parameters/fresh private initial tensors/
physical statecritic/standard joint PPO/globalclip. All six start uniform feasible
matching density; entropy scaling must NEVER change a frozen policy's likelihood,
MAP, sampler, information or physical decoder. Only regularizer aggregation in
PPO evaluate changes. Four external PPO/norm/localMAPPO/fullMAPPO remain their
ORIGINAL implementations/hyperparameters/entropy treatment; no weakened baseline.

If myopic catches Gibbs after entropy alignment, reject the explanation that
weighted continuation is necessary for reliable quality. Do not announce success
from raising entropy alone, or select an old-entropy/reverse control as new winner.
A failed unit choice gets no additional beta sweep, entropy schedule, budget or
component stack from this note. Current N4/scientific/health restrictions remain.
Fixed beta.01 with H versus H/4 is ONE prescribed unit contrast, not a search.

A prospective full protocol must lock fresh panels, fixed491520budget EVERYmethod,
15quality/health endpoints for10arms (mainGjointvsMjoint>=.5%; NI−1%vsreversejoint/
all3original-unit cores/PPO/norm; strictsuperiorityvslocal/fullMAPPO; sixoriginal
healthbounds),all3primaryreplicaspositive,zero invalid/isolatedp95<=50ms; record
entropy/point/confidence interpretation and independent verification requirements.
Future confirmation only after independentALLPASS DEV, new5replicas/256TEST,
sequentialfamilyround8 and multiplicity15, no oldTESTreuse. Main practical gain
must survive calibrated SAME-unit retraining; no CI/scaling claim from DEV.

Before learning, register entropy readout on a fixed common CFDEV observation
panel across all6arms/checkpoints so comparison is not merely different visited
states. Observations are diagnostic only, never training labels. Specify the
readout and its interpretation before outcomes; do not tune it from currentTEST.

## Implementation cautions for the later protocol

Retain a separate base-density identifier from experiment algorithm labels:
original TypedGibbsPolicy.tables branches on self.algorithm, so changing names
without explicit mapping would silently make Gibbs become myopic. Tests must
prove same weights/observations yield identical full physical densities, entropy
factor4, exact gradients/MAP and sampling for each entropy-unit pair, as well as
current masks/type contracts/cold uniformity. Preserve base likelihood gradients
through learned logZ; no detach/teacher/catalogue added to production.

Prospective protocol, seedhistory audit, implementation/tests/commit/push before
FULL on lab. No separate smoke per user override. No code/learning/seeds allocated
by this design note yet. Goal ACTIVE/unachieved; N>4 closed.
