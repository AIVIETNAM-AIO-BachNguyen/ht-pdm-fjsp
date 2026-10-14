# Opportunity-message PPO v2 pilot: technical pass, development negative

Frozen learning commit a12a00d7651e2d9a78ec6a22dfb2595a6a058f47.
Run `technician_scarcity_ppo_v2_pilot_20261014T040109Z`: COMPLETED,
independent audit PASS, elapsed 1894.58s (31.58min). No scientific test opened.
22 random-initialized models, 61440 decisions/1920 optimizer steps each;
1351680 decisions, 28160 training episodes, 5120 paired training tapes,
1056 development episodes. No teacher, pretraining or invalid assignment.
Results below are short-budget development point estimates, not confidence
intervals or evidence of superiority over full-budget PPO.

| N | Method | Selected cost/machine | Availability | Pending/machine | Failures/machine |
|---|---|---:|---:|---:|---:|
| 4 | opportunity PPO | 302.8281 | 0.633301 | 0.031250 | 1.265625 |
| 4 | reservation | 356.6797 | 0.558268 | 0.062500 | 2.109375 |
| 4 | independent | 362.4844 | 0.556803 | 0.109375 | 2.273438 |
| 4 | globally clipped priced control | 302.8281 | 0.633301 | 0.031250 | 1.265625 |
| 4 | original PPO | 290.5469 | 0.667806 | 0.031250 | 1.648438 |
| 4 | local MAPPO | 367.8359 | 0.553548 | 0.078125 | 2.523438 |
| 4 | full-information MAPPO | 364.0313 | 0.558919 | 0.085938 | 2.492188 |
| 4 | deadline | 279.5000 | 0.666667 | 0 | 0.984375 |
| 32 | opportunity PPO | 264.7549 | 0.670349 | 0 | 0.166016 |
| 32 | reservation | 347.4229 | 0.557495 | 0.018555 | 1.473633 |
| 32 | independent | 547.2852 | 0.399394 | 0.517578 | 4.177734 |
| 32 | globally clipped priced control | 265.3740 | 0.670817 | 0 | 0.204102 |
| 32 | deadline | 271.6230 | 0.649211 | 0 | 0.068359 |

Screen FAIL: N4 cost is 4.23% worse than equally short-budget PPO, outside the
1% noninferiority screen; availability difference -0.034505, below -0.01.
N32 registered point health limits pass; cost gains 23.79% vs reservation,
51.62% vs independent, 2.53% vs deadline. These promising descriptive values
do not override the small-setting failures or establish an out-of-sample effect.

Both N4 priced replicas select step0. N32 candidate selects step30720/step0;
selected controls are equally allowed step0. Candidate N4 per-replica costs:
318.109→493.688→394.031 and 287.547→289.375→400.234 across 0/30720/61440.
N32: 294.383→264.520→301.355 and 264.990→265.484→306.490.
This is unstable learning, not a successful trained policy. Separately clipped
vs globally clipped candidate selected gain is 0% N4/0.233% N32, with no
confidence interval. Raw norm dominance did not translate to a large difference
in mean actor parameter-step norm; do not claim clipping fixed learning.

N32 fit wall times including development under shared eight-worker CPU load:
candidate1789–1790s, global control1755–1771s, reservation1124–1137s,
independent800–821s. No isolated inference or network-latency claim from these.

## Preserved diagnostics

- `technician_scarcity_ppo_v2_development_report_20261014T043259Z`: bound inputs,
  learning curve PNG/PDF, all methods and all-replica min–max (not CI).
  PNG visually inspected and readable.
- `technician_scarcity_ppo_v2_mechanism_report_20261014T043259Z`: every graph
  replica/checkpoint, gates, intervention activity and actual update norms;
  all manifest-bound files independently verified. Unselected checkpoints kept.
- `technician_scarcity_ppo_v2_stop_diagnostic_20261014T043639Z`: all12 candidate
  checkpoints on one ALREADY OPENED DEV seed64500000, no optimizer updates,
  no parameter changes, no new panel. Capture returns the exact original
  log_softmax value and restores the hook. At N4 intermediate/final checkpoints,
  greedy STOP with legal edges occurs16/8 and2/12 ticks;15/8 and2/11 of these
  have aggregate GO mass above one half. N32 final checkpoints show2/6 such
  ticks, both entirely with majority GO mass. Step0 shows none on this seed.
  This identifies a decode/probability discrepancy, not a proven cause or cure.

Do not advance this failed design to confirmation or reinterpret thresholds.
Preserve the negative pilot. Next prospective design should test an explicit
GO/STOP gate separate from conditional edge selection against a flat-STOP
control, with fresh random initialization/panels and unchanged physical limits.
Generic hierarchical policies are prior art; no general PPO theorem claimed.
