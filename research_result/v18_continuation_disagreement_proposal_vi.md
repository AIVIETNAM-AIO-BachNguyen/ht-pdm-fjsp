# V18 conditional method proposal — bounded continuation-disagreement credit

Status: prospective proposal after independently audited V17/V11 continuation diagnostics; no V18 code, seed tape, RL run or TEST yet. Original completion criteria stay unchanged.

## Problem and evidence

V17 same-capacity full-matching rank projection improves its own-Q control by 8.76 cost across five seeds, yet cost898.69 is 4.44% above rollout860.45 and 19.55 above frozen V14 joint on the same DEV panel. In reciprocal2, the selected V17 actor's first root has own-continuation Q 11.33 **lower** than rollout's root, even though its closed-loop episode is 93.45 **higher**; risk-continuation Q reverses the sign (+23.71). Forty-eight of 80 first divergences there are V17 immediate service versus rollout WAIT. In interchangeable, own-Q nearly ties while risk-Q favors rollout, and 43/80 first differences concern assignments. V11 trained wholesale on risk-continuation Q and suffered a WAIT/CM-queue collapse in two seeds; V12 scenario-variance weighting also failed. Thus the next question is whether a *bounded relative correction for continuation disagreement* can expose long-term resource commitments without letting a weak reference continuation take over policy learning.

## Core formulation

Use the V17 9,025-parameter full-matching actor with current 56-D state, matching incidence, served flags and pre-decision technician releases. From each on-policy state, estimate both own-policy H64 root costs `Q_o(s,a)` and deterministic risk-continuation H64 root costs `Q_r(s,a)` for **every legal matching** on the same four stochastic failure scenarios; own continuation also uses common action uniforms. Center each bank within the legal set. Candidate credit is

`Q_b(s,a) = Q_o(s,a) + 0.5 * clip([(Q_r(s,a)-mean_legal Q_r) - (Q_o(s,a)-mean_legal Q_o)], -40, 40)`.

The clip limits any relative reference correction to 20 cost units/action. Fit a masked full-action rank target `softmax(-Q_b/10)` with V17's KL penalty, hard KL≤0.02 and λ=0.5 within-machine-set assignment auxiliary computed from **Q_o for both arms**. Both arms add the same differentiable batch service-flow protection `10*ReLU(E_old[|a|]-E_new[|a|]-0.05)^2` to limit wholesale WAIT collapse while allowing a small reduction in expected occupied technicians. This is a soft training objective, not a service action mask or imposed feasibility rule. Deployed policy sees no forecast, future tape or planner; all legal matchings including WAIT remain available.

## Controlled ablation and prior work

Control has identical actor, initial weights, states-per-cell, legal support, two Q banks and forecast compute, optimizer, assignment auxiliary, service-flow term and checkpoint selection, but ranks `Q_o` only. Later on-policy states may diverge; record actual roots/transitions. Frozen V14 joint/count, V7 flat/shared, FCFS, risk and bounded rollout remain external comparators. V11 full risk-Q and V17 own-Q findings are historical context, not substitute same-network controls.

Softmax Q projection, KL policy improvement, conservative policy iteration and policy constraints are established (for example [Hybrid MPO](https://proceedings.mlr.press/v100/neunert20a/neunert20a.pdf), [Constrained Policy Optimization](https://proceedings.mlr.press/v70/achiam17a.html)); [Learning to Dispatch](https://proceedings.neurips.cc/paper_files/paper/2020/file/11958dfee29b6709f48a9ba0387a2431-Paper.pdf) and [AirME](https://ojs.aaai.org/index.php/ICAPS/article/view/19839) already cover structured RL dispatch and stochastic maintenance. A possible narrow contribution is the **compatible-matching, continuation-disagreement-limited credit mechanism** if a fair ablation and competitive replicated evaluation support it. Do not claim invented robust RL, policy constraints, shared-resource maintenance RL or guaranteed improvement.

## Falsifiability and next step

A useful candidate must improve its same-capacity own-Q ablation across strong/weak seeds, avoid V11-style WAIT/service collapse, outperform frozen V14 on fresh DEV and approach rollout with ≥10× inference speed. The old-policy four-scenario banks can be noisy; clipped disagreement may still encourage too much WAIT or may be too weak to change relevant matchings. Prelock five-seed DEV protocol/gate, source hashes, artifacts and independent audit before implementation. Tests, Ubuntu smoke and source push precede full run. Only a supported and competitive DEV candidate proceeds to disjoint five-seed replication and one fresh frozen TEST. An ablation gain without competition remains a no-go under the original user goal.
