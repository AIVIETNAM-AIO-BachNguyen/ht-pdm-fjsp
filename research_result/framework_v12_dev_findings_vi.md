# V12 reliability-calibrated action credit — audited DEV no-go

Prelock and source push: `0904078` (`method_v12_proposal_vi.md`, `framework_v12_dev_protocol_vi.md`). Full fresh-seed run: `framework_v12_dev_20261020T125625Z`, **COMPLETED** in 40.3 min on Ubuntu. Independent verifier `verify_framework_v12.py`: `framework_v12_audit_20261020T134139Z/independent_audit.json`, **PASS**. It checked source/artifact/comparator hashes, full Q-row and update-weight arithmetic, paired first-iteration actors/banks, 240 independently replayed full-H64 four-scenario Q banks (164,064 transitions), all 640 selected RL episodes, 4,480 checkpoint-selection rows, 1,792 comparator rows, all case-level cost/operational/compute aggregates, and hierarchical bootstrap. Raw run/audit directories are immutable local artifacts ignored by Git; tracked source/report are reproducible.

## Method and matched ablation

Both arms start scratch from identical 1,597-parameter shared-edge free-matching actors for each of five fresh seeds 71500000–04. Each uses six iterations, 16 on-policy source episodes/case/iteration, 64 hash-selected states/case/iteration, four common stochastic scenarios per legal root through H48 under the old actor plus fixed risk tail to H64, and V10's λ=0.5 equal-machine-set assignment objective. `plain_credit` uses the mean paired-scenario action cost advantage. `reliability_credit` multiplies each action advantage by `min(1, |mean paired gap|/(2 standard errors + 1e−6))`, with legal actions and actor architecture unchanged. Both arms receive the same *forecast budget*; their later on-policy states diverge by design. Mean candidate action weight is 0.522, with 51.9% below 0.5. Forecast transitions were 4,701,268 candidate versus 4,697,704 control (0.076% difference). All selected checkpoint indices were chosen on the fixed 16-seed DEV grid, with no TEST access.

## Locked DEV result

| Controller | Four-case mean cost | Selection s/episode | Role |
|---|---:|---:|---|
| Reliability credit | 904.08 | 0.01226 | proposed RL |
| Plain credit | 897.63 | 0.01266 | same-compute ablation |
| Risk matching | 892.25 | 0.00141 | simple heuristic |
| Flat PPO | 905.46 | 0.02052 | locked RL baseline |
| Bounded rollout | **849.56** | 0.67711 | strongest locked planner |

Candidate minus ablation cost is **+6.45** (hierarchical 95% bootstrap **[−6.60,+28.89]** over five train seeds and 16 DEV tapes), so attenuation does not show an average gain. Seed differences are −0.64, **+43.94**, −6.28, −0.25, −4.53. Four of five seeds have small gains, but one large loss dominates. The candidate is 6.42% above rollout, beyond both the prelocked 2% DEV promotion distance and the goal's ≤1% planner-cost route; risk matching is cheaper and faster. Candidate selection is 55.2× faster than rollout, insufficient without planner-level cost. The V12 DEV promotion gate is **FALSE**; no V12 TEST or replication campaign is justified from this result.

| Case | Reliability | Plain | Risk | Rollout | Candidate−plain |
|---|---:|---:|---:|---:|---:|
| Interchangeable scale1 | 738.19 | 737.99 | 738.19 | 677.62 | +0.20 |
| Reciprocal scale2 | 851.46 | 824.25 | 812.69 | 710.38 | **+27.21** |
| Reciprocal scale3 | 1027.75 | 1028.45 | 1015.12 | 1018.94 | −0.70 |
| Specialist/generalist scale2 | 998.90 | 999.83 | 1003.00 | 991.31 | −0.93 |

Across four cases, candidate versus plain mean H availability is 0.524 vs 0.529; H failures 5.884 vs 5.803; queued CM ticks 7.709 vs 7.175; service starts 44.006 vs 44.431. Candidate cost parts are maintenance 52.08, unavailability 727.07 and failure 124.92, versus plain 52.45, 720.86 and 124.31. The major failure is seed 71500001 on reciprocal scale2: candidate 1027.69 vs plain 850.13 (+177.56), availability 0.443 vs 0.554, H failures 7.81 vs 5.75, CM queue 20.44 vs 6.69 ticks, and service starts 37.0 vs 47.19. This is an observed failure mode, not proof that the weight rule alone caused a specific action error. Candidate selected iteration6 for four seeds and iteration2 for seed00; plain selected iterations3/6/6/4/6. No untrained iteration0 checkpoint was selected.

Training used 184,320 actual environment steps across arms, about 9.40 million forecast transitions combined, plus 4,480 DEV and 1,792 comparator evaluations. Mean full episode wall time was 0.021 s candidate, 0.022 s plain, 0.010 s risk and 0.686 s rollout; selection times above exclude the rest of episode evaluation. The run completed inside the four-hour hard stop with distinct timestamped artifacts and no overwrite.

## Interpretation and prior work

The fixed reliability rule suppressed many action updates but did not close the temporal cost gap on the first two cases, where rollout remained far better. V12 must remain a negative result. Its action integration is related to [Expected Policy Gradients](https://www.jmlr.org/papers/v21/18-012.html); paired simulation comparisons using common random numbers are established in [Policy Search using Paired Comparisons](https://jmlr2020.csail.mit.edu/papers/v3/strens02a.html). The resource-constrained maintenance setting has meaningful prior work, including [HetGPO/AirME](https://ojs.aaai.org/index.php/ICAPS/article/view/19839), and learned job-shop dispatch is established by [Learning to Dispatch](https://papers.nips.cc/paper_files/paper/2020/hash/11958dfee29b6709f48a9ba0387a2431-Abstract.html). Thus this experiment does not establish a general new RL principle or a competitive shared-technician dispatcher. The user goal remains **ACTIVE**. A next method must address rollout's first-case/reciprocal2 gap and include a fresh ablation showing learned improvement beyond strong simple controllers, with fresh confirmation and independent replication before completion.

Reproduce the independent audit from repo root with `uv run python research_result/verify_framework_v12.py --run-dir research_result/framework_v12_dev_20261020T125625Z --output-dir research_result/framework_v12_audit_$(date -u +%Y%m%dT%H%M%SZ)`; it refuses an existing audit output directory.
