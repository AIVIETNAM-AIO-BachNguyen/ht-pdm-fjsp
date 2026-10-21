# V19 temporal matching deficit — five-seed DEV, audited no-go

**Status:** full scratch-RL DEV complete; prelocked DEV promotion gate **FALSE**. Proposal, protocol and registry were pushed at `82bd7ef`; feature tests/discrimination source at `1bdff17`; campaign, 14 focused tests and independently audited Ubuntu smoke at `a8bf4e2` before the full run. Full run `framework_v19_dev_20261021T075021Z` completed in 3,971.05 s under the four-hour hard timeout. Independent audit `framework_v19_audit_20261021T085652Z/independent_audit.json` PASS: source/artifact/comparator hashes; equal 10,177 active parameters and initial weights for all five paired seeds; 240 stratified source/full-Q replays with independent future-matching-feature enumeration; all 640 selected V19 and 640 each frozen V14/V18 final episode replays; selection, arithmetic, bootstrap and gate. Its 15,360 logged Q rows and 3,072 final episodes remain in timestamped artifact directories outside Git. No V19 independent replication or fresh TEST was opened.

## Primary matched ablation and competition

Cost is the unchanged equal-four-case H48+16 economic objective, lower better. Candidate `temporal_deficit` and control `snapshot_deficit` use the same 56 current observations, full matching legal set including WAIT, network parameter count, future-demand profile, training source budget, four-scenario H64 own-policy Q ranks and optimizer. The only structured-input ablation is the action-dependent nine-offset maximum-compatible-matching deficit versus its τ1 snapshot repeated nine times. On 5 training seeds ×16 disjoint final DEV physical seeds:

| Controller | Equal-case mean cost | Selection seconds/episode |
|---|---:|---:|
| V19 temporal deficit | **869.72** | 0.02268 |
| V19 snapshot deficit | 877.63 | 0.02323 |
| Frozen V14 joint identity | 858.23 | 0.02315 |
| Frozen V14 count control | 857.84 | 0.02466 |
| Frozen V18 bounded | 870.87 | 0.01552 |
| V7 flat PPO | 900.83 | 0.02093 |
| FCFS / risk | 901.72 / 881.97 | 0.00120 / 0.00178 |
| Bounded rollout | **843.38** | 0.64204 |

Primary paired temporal−snapshot mean **−7.91 cost**, hierarchical 20,000-draw CI95 **[−16.11,−1.55]**, negative mean in **5/5** training seeds and **3/3** strongest control seeds. Per-training-seed means: −8.72, −2.81, −5.30, −18.89, −3.83. Thus the temporal deficit is a credible component signal on fresh DEV, beyond one weak seed; it **misses the locked material ≤−10 gate**. Full episode-pair counts: 44 wins, 6 ties, 30 losses out of 80 paired equal-case cells. On-policy paths and hence actual forecast transitions diverged; temporal 5,176,028 versus control 5,244,368, from 31,169 versus 31,570 legal roots. Both arms had 600 accepted optimizer minibatch updates over six scheduled iterations and 92,160 H48 source steps each. Training Q wall time from traces was ~1,789 versus ~1,849 s; identical scenario/root budget, with realized root-count difference due to the different visited states. Checkpoint selection used eight selection-only physical seeds and picked updates temporal 3/3/4/6/2, control 3/6/6/6/5 for the five seeds. No inference simulation or future event tape is exposed to deployed V19.

Candidate is **3.12% above** rollout cost (869.72 versus 843.38) while **28.31× faster** at action selection. It is 11.49 above frozen V14 joint. Original completion route B needs ≤851.81 on this panel (1% rollout margin), leaving **17.91 cost**. Route A needs ≤801.21 to beat the strongest locked comparator rollout by 5%, leaving **68.51 cost**, with uncertainty support still required. V19 also fails its prelocked ≤1.02× rollout and ≤V14 joint DEV gates. FCFS and risk are faster but more costly, so no simple-controller joint cost/runtime dominance. This does not rescue competition. The gate remains FALSE regardless of the positive ablation CI.

## Case and operational evidence

| Case | Temporal | Snapshot | Rollout | V14 joint | Temporal−snapshot | Temporal−rollout |
|---|---:|---:|---:|---:|---:|---:|
| interchangeable | 742.80 | 743.80 | 679.69 | 739.83 | −1.00 | +63.11 |
| reciprocal2 | 766.21 | 768.74 | 702.50 | 773.38 | −2.52 | +63.71 |
| reciprocal3 | 989.84 | 1011.00 | 1021.81 | 936.45 | −21.16 | −31.98 |
| specialist/generalist | 980.01 | 986.96 | 969.50 | 983.26 | −6.95 | +10.51 |

Temporal gains are most visible in reciprocal3 and specialist/generalist; interchangeable/reciprocal2, the historical planner-gap cases, gain only 1.00/2.52 cost. In interchangeable V19 has 3.78 H failures and 11.23 queued PM ticks versus rollout 0.88 and 2.31, despite both averaging 51.44 service starts. Reciprocal2 V19 has 4.43 H failures/13.25 PM queue versus rollout 1.81/5.56. This is a *timing/assignment* deficit, not simply too few starts. Reciprocal3 V19 improves over its snapshot by 21.16 and has 39.03 starts, 9.80 queued CM ticks, versus snapshot 37.40 and 13.38; nevertheless V14 joint cost is 53.39 lower there. Specialist/generalist temporal cost is 6.95 below snapshot with 7.74 H failures and 27.54 queued PM ticks, but still 10.51 above rollout. Full maintenance/unavailability/failure/tail and availability measures are in `summary.json`/`episodes.csv`. No permutation, N/K scalability or generalization claim was tested.

## Interpretation and next step

V19 shows that an action-specific, time-resolved compatible-capacity signal can improve scratch-RL relative to an equal-size snapshot control under the prelocked comparison. The improvement is too small and concentrated away from the two largest rollout-gap cases, while frozen V14 remains better overall. The current deterministic exposure approximation also does not account for stochastic failures, later learned dispatches or repeated requests after service; the opened V18 first-divergence audit already showed a continuation-dependent Q mismatch. These are hypotheses for the next *discriminating diagnostic*, not established causal explanations for V19 failure. Keep V19 no-go, retain all negative/positive artifacts and original goal gates. Do **not** proceed to V19 replication/TEST or claim novelty; `v19_primary_prior_art_boundary_vi.md` limits related-work claims. Next inspect first divergence, assignment/timing/continuation values for V19 versus rollout in interchangeable/reciprocal2 and versus frozen V14 in reciprocal3 on this already-opened DEV panel, then propose a distinct scratch-RL method only if the evidence supports one. Goal **ACTIVE**.
