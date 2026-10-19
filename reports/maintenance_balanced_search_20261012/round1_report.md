# Balanced 2–2 search round1: no eligible screen candidate

## Material Passport
Synthetic CPU Mac simulator;6selectable profiles+2controls, fresh development/
screening panels. No confirmation outcomes were opened. Native verifier reused
independently implemented scalar physics/indexed bootstrap in shared context;
not blind or cross-model. No RL training or external data.

**NO_SCREEN_CANDIDATE.** All8profiles were preserved. q.625 leaves routinePMage1
and its assignment advantage was effectively zero. q.75 at routinePMage3 restores
age0, same realized restoration asq1; d3 andd4 retained local assignment value,
but failed the strict assignment availability screen. Oldbalancedq.5 control
had useful cost/local gains but failed pendingdifference and was nonselectable.
No runner-up/fallback or repeated confirmation seeds were opened.

Code `a23d429e3c6261536a62621fec8d44eff0ccde24`, branch `codex/maintenance-balanced-search-mac-v1`,
pushed before full. Tests27PASS28.47s andCLI smoke10.051sPASS; committed smoke
independent auditPASS. Full283.703s with576development,
1280screening,0confirmationepisodes,111360physicalticks,1792landmarks,
3138rootrows and341377408forecasttransitions. Independent auditPASS:
64hashbindings,53committedsourceblobs, allphysicalstates/shocks/costs,
frozenbaselines and health-first ranking, rootassays, expandedseedregistryaudit.

| Profile | Assignment H+L gain | Local vs endpoint DP | Assignment av difference | Failed screen metrics |
|---|---:|---:|---:|---|
| balanced_d2_q625 | -0.034% | 0.0089 | -0.016pp | assignment_cost_gain, joint_cost_vs_deadline_matching, joint_cost_vs_isolated_h6, joint_cost_vs_isolated_endpoint, local_baseline, local_quality, local_endpoint_dp, assignment_interior_gain |
| balanced_d2_q750 | 3.294% | 0.0530 | -0.212pp | local_baseline, local_quality, local_endpoint_dp |
| balanced_d3_q625 | 0.155% | -0.0098 | 0.033pp | assignment_cost_gain, joint_cost_vs_deadline_matching, joint_cost_vs_isolated_h6, joint_cost_vs_isolated_endpoint, local_baseline, local_quality, local_endpoint_dp |
| balanced_d3_q750 | 1.549% | 1.3602 | -0.553pp | assignment_availability_difference |
| balanced_d4_q625 | 0.000% | 0.0000 | 0.000pp | assignment_cost_gain, joint_cost_vs_isolated_h6, joint_cost_vs_isolated_endpoint, local_baseline, local_quality, local_endpoint_dp, assignment_interior_gain |
| balanced_d4_q750 | 3.881% | 6.5839 | -1.270pp | assignment_availability_difference |
| old_balanced_d3_q050 | 8.585% | 8.6857 | 0.195pp | assignment_pending_difference |
| exchangeable | 0.000% | 0.0000 | 0.000pp | assignment_cost_gain, local_baseline, local_quality, local_endpoint_dp, assignment_interior_gain |

Screen values are point estimates, not confirmation evidence. Thresholds were
unchanged. The new-series familyalpha.025 was not recycled after this failure.
Future round needs a new frozen design/newrawseeds and alpha.0125, retaining
all18guards with stricter confidence1−.0125/18.

Artifact: `/Users/bachng/Coding/Reinforcement Learning/marl/artifacts/maintenance_balanced_search_mac_v1_full_cpu_20261012T091339Z`

Eleven interpretation checks:1no testtuning;2no optionalstopping;3no fallback;
4seedclusters rather thanlandmarkpseudoreplication;5selection/holdoutseparate;
6zerochoice retained;7tail included;8healthguard enforced;9onlycurrentroot
isolated;10no optimality/RL claims;11new-seriesalpha only, approximatecoverage.
