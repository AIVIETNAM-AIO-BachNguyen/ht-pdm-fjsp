# V20 activation-margin scratch RL — audited five-seed DEV: no-go

**Status.** DEV only, 2026-10-21. The prelocked V20 activation-margin hypothesis failed. Do not replicate this V20 method or open TEST. The broader research goal remains ACTIVE.

## Protocol and provenance

- Prospective proposal: `v20_activation_margin_proposal_vi.md`; frozen protocol: `framework_v20_dev_protocol_vi.md`; registry: `configs/maintenance_framework_v20_seed_registry.json`.
- Five independent scratch-training seeds 73700000–04, eight selection-only physical seeds, 16 fresh final DEV physical seeds, four equal-weight cases. Eight updates per seed and arm; 3,219 active actor parameters in both arms. The activation-margin and generic-Q ablation use the same observation, factorized legal-action map, Q construction, forecast seed roles, four-scenario budget and checkpoint selection rule; each collects its own on-policy states. There is no imitation or pretraining.
- Source was tested, Ubuntu-smoked, independently audited and pushed at `6760346` before the full run. Full artifact: `framework_v20_dev_20261021T125049Z`, `manifest.json` status `COMPLETED`, 2026-10-21 12:50:57–15:25:46 UTC. The manifest records source/config hashes, source commit, seed registry and checkpoint hashes. Raw and checkpoints remain outside Git.
- Primary independent audit: `framework_v20_audit_20261021T152700Z/independent_audit.json`, **PASS**. It checked 20,480 counterfactual rows, replayed 320 complete sampled source/Q banks (314,964 forecast transitions), all 640 selected V20 episodes and 640 frozen V14 episodes, selection/evaluation disjointness, source hashes and bootstrap/gates.
- Baseline replay audit: `framework_v20_baseline_audit_20261021T152701Z/baseline_replay_audit.json`, **PASS** on all 1,152 V7/rule/planner episodes with zero metric error. Selection wall times are campaign measurements, not replayed.
- Detailed case, seed, operation and compute data: `framework_v20_analysis_20261021T153100Z/descriptive_analysis.json`; this derives from the audited final episodes. Its per-case intervals are exploratory, not new decision gates.

## Prelocked primary result

Four-case equal-weight economic cost (lower is better):

| Controller | DEV cost | Selection seconds / episode |
|---|---:|---:|
| V20 activation margin | 908.51 | 0.03190 |
| Matched generic-Q ablation | 896.52 | 0.03194 |
| Frozen V14 count-calibrated | 880.79 | 0.02467 |
| Frozen V14 competition | 881.69 | 0.02341 |
| Bounded rollout/planner | **865.88** | 0.74171 |
| Risk matching | 906.44 | **0.00188** |
| V7 flat PPO | 926.71 | 0.02093 |

Candidate minus matched ablation = **+12.00 cost**, five-seed hierarchical CI95 **[−0.85,+34.68]**; lower is better, so the point estimate is a regression and the interval gives no evidence of benefit. Training-seed differences were −1.75, +1.14, +8.08, +52.27, +0.25. Only one of five seeds improves and **zero of the three strongest control seeds** improves. The prelocked material H1 gate (≤−10, CI upper<0 and seed consistency) fails.

Candidate is +27.73 cost (+3.15%) versus V14 count and +42.64 (+4.92%) versus rollout. It is 23.25× faster than rollout at selection, but exceeds the original route-B 1% cost ceiling 874.53 by 33.98; risk matching is both cheaper and faster than V20. Original route-A ceiling is 822.58 (5% below strongest comparator); V20 misses by 85.93. The prelocked H2 and original A/B performance gates all fail. `summary.json` records `dev_go_gate=false`.

## Case and operational diagnosis

| Case | V20 | Generic Q | V14 count | Rollout | V20 − generic Q | V20 − V14 count |
|---|---:|---:|---:|---:|---:|---:|
| Interchangeable scale 1 | 777.11 | 776.06 | 772.90 | 693.31 | +1.05 | +4.21 |
| Reciprocal scale 2 | 802.00 | 793.54 | 797.34 | 707.12 | +8.46 | +4.66 |
| Reciprocal scale 3 | 1029.49 | 1004.95 | 961.54 | 1086.81 | +24.54 | +67.95 |
| Specialist/generalist scale 2 | 1025.45 | 1011.51 | 991.38 | 976.25 | +13.94 | +34.08 |

The activation-margin gate did not improve either case where V14 had the clearest rollout gap: it almost never waited when service was legal in interchangeable (0%) or reciprocal scale 2 (0.56%). In reciprocal scale 3 it waited on 13.4% of legal service ticks versus 0.4% for generic Q and 0.2% for V14 count. There, mean corrective queue ticks rose to 21.95 versus 6.65 generic Q and 0.99 V14 count; service starts fell to 38.53 versus 40.49 generic Q and 43.00 V14 count; H48 availability fell to 0.448 versus 0.48/0.502. The maintenance cost was lower only because fewer jobs were serviced; economic cost was higher.

Seed 73700003 is an acute failure: reciprocal scale 3 cost 1158.31, 20.31 legal-service WAIT ticks, 79.88 corrective queue ticks, 29.62 service starts and H48 availability 0.31, versus its matched generic-Q cost 1006.12 with no legal-service WAIT. Specialist/generalist cost for the same seed is 1069.81 versus generic Q 1015.94. The other four seeds also do not establish a useful core-component gain: their aggregate candidate-minus-control mean is +1.93. This is a deployed-policy outcome, not proof that any single source-Q target was wrong; final-state Q/calibration must be checked separately before assigning mechanism.

Per-case V20−V14 count descriptive CI95 is [+18.81,+141.78] in reciprocal scale 3 and [+14.39,+64.57] in specialist/generalist. These are exploratory intervals with a fixed frozen V14 ensemble; they do not replace the prelocked aggregate gate. The candidate's H48 failure counts and queues on interchangeable/reciprocal2 remain much worse than rollout despite similar total starts, consistent with the earlier V14 direct-gap diagnostic but not a causal explanation by itself.

## Compute and interpretation

Across five seeds per arm, each arm used 40 updates, 2,560 source episodes, 122,880 source steps and 10,240 selected counterfactual states. Activation margin used 9,771,412 forecast transitions and 4,443.51 seconds of counterfactual-plus-gradient wall time; generic Q used 9,640,740 and 4,377.86 seconds. These measured training times exclude source collection and final evaluation. Checkpoint selection was prelocked; selected candidate iterations were 8,8,8,7,6 and control iterations 6,8,7,5,6.

The result rejects this **particular** whole-matching activation-margin training mechanism. It does not show that future-resource competition is irrelevant: the prior V14 direct diagnostic shows costly root service choices, while this V20 update failed to correct them on the two target cases and introduced a starvation failure elsewhere. The current observation's action-specific unserved-machine identity omission is real in source, but V14's previous identity ablation did not show material benefit. No novelty or performance claim is advanced from V20.

## Next decision

Keep TEST closed and preserve this negative full run. Before another major method, run a prelocked final-state diagnostic on the already-opened V20 DEV panel: compare deployed V20/V14/rollout first divergent decisions and evaluate common-shock root WAIT-versus-best-service Q under each policy's own continuation, with explicit coverage of the seed-73700003 starvation states and the two weak rollout-gap cases. Distinguish (i) inaccurate counterfactual labels or continuation, (ii) failure to fit/calibrate the gate, and (iii) conditional matching/identity effects. Only then formulate and prelock the next scratch-RL mechanism with a matched ablation. Independent replication and fresh frozen TEST remain outstanding for any eventual successful candidate.
