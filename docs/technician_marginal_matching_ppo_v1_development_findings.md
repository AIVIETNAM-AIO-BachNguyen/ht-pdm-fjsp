# Physical-marginal matching PPO v1: negative full N4 development

DEVELOPMENT_ONLY. The completed full-budget N4/K2 round does not meet the
registered research criteria. No scientific confirmation test has opened, and
no larger-scale run is authorized by these results. Preserve this negative round.

Source: `artifacts/technician_marginal_matching_ppo_v1_development_20261016T021444872382396Z`.
Frozen learning commit `f436ab6e3c6ba01e7377e6762773e05c4df66df1`.
Elapsed2011.8749s (33min32s), COMPLETED/auditPASS. All27 random-initialized fits
spent491520 joint decisions,7680 optimizer updates and10240 train episodes each:
13,271,040 decisions,276,480 train episodes,4320 DEV episodes plus32 deadline
reference episodes. All selected replicas are trained; no teacher/pretraining.

| Method | DEV cost/machine | Availability | Pending/machine | Failures/machine |
|---|---:|---:|---:|---:|
| Physical-marginal matching PPO | 277.111979 | .671820747 | 0 | .864583333 |
| Latent-order PPO | 277.156250 | .671820747 | 0 | .867187500 |
| Forward-order PPO | 276.536458 | .670301649 | .002604167 | .854166667 |
| Reverse-order PPO | 274.963542 | .670410156 | 0 | .750000000 |
| No-prefix PPO | 276.216146 | .671766493 | 0 | .828125000 |
| Original PPO | 260.820313 | .669108073 | 0 | .117187500 |
| Normalized PPO | 260.820313 | .669108073 | 0 | .117187500 |
| Local MAPPO | 265.877604 | .669542101 | 0 | .317708333 |
| Full-information MAPPO | 266.119792 | .667860243 | 0 | .414062500 |

Only6/14 mandatory point endpoints pass. The candidate costs6.2463% more than
both PPO variants,4.2254% more than localMAPPO and4.1305% more than fullMAPPO.
Gain versus matched latent-order control is only+.015973%, below the required
+.5%. Gain versus no-prefix is−.324323%, also failing the+.5% mechanism gate.
The two fixed-order noninferiority gates pass within1%; no positive order benefit
is established. Availability and pending bounds pass against bothPPO references,
but failures/machine differ by+.747396, beyond the+.25 limit. Zero invalids and
isolated actor p95=8.648619ms pass. Good latency cannot compensate for failed
quality/health/mechanism gates.

These are paired DEV point estimates, not confirmation confidence claims.
Candidate selected steps in seed order200400000..2 are122880,368640,368640.
Every replica spent the full fixed budget, including later checkpoints that were
not selected. Matching initialization/capacity/information/budgets were retained
for the five core arms. Scalar replay precision remained within the original
registered tolerances.

Independent read-only verification and plots:
`artifacts/technician_marginal_matching_ppo_v1_development_report_20261016T024846521063565Z`.
The verifier checks all output/input bindings, exact full training/DEV grids,
paired population hashes, budgets/reward/order accounting, all checkpoint
selection and14 point gates, isolated-latency contexts/samples and closed tests.
The descriptive figure's whiskers are min–max over three replicas, not CIs.

Next action is the prospectively registered frozen N4 decoder diagnostic on
new paired seeds, separating two-greedy-plan selection from exact highest-policy-
probability matching at N4 only. Its exhaustive diagnostic catalogue is not a
scalable proposed algorithm, teacher or cost oracle. No new learning or checkpoint
reselection occurs. The result will guide a separately registered next design;
it cannot rescue this failed round or reopen its scientific panel.
