# Teacher-free scarcity v1 pilot: negative development result

Source `technician_scarcity_ppo_v1_pilot_20261014T023754Z`, frozen9247b3c,
COMPLETED/auditPASS in1205.53s (~20min). All18models trained from random
initialization,61440decisions/1920optimizersteps each,1105920total decisions,
23040training episodes,5120paired physical tapes,864development episodes.
No teacher/pretraining, no scientific test opened, no scientific alpha spent.

The required development screen fails. Do not promote v1 to confirmation.

| N | Scarcity cost/machine | Reservation cost/machine | Independent cost/machine | Deadline cost/machine |
|---|---:|---:|---:|---:|
| 4 | 285.34375 | 295.40625 | 311.546875 | 278.46875 |
| 32 | 304.689453 | 281.546875 | 558.741211 | 270.810547 |

At N32 prices increase selected cost by8.22% versus reservation and12.51%
versus deadline. Availability is0.609904 versus deadline0.650675 (difference
-0.040771), and failures/machine0.731445 versus0.054688 (difference+0.676758).
Both availability and failure limits fail. Independent coordination is much
worse, but that alone does not establish the proposed price-message contribution.

At N4 candidate cost is7.09% lower than PPO at the equally short61440budget;
this is exploratory, two-replica development only, not a confidence-backed
claim against full-budget PPO. One candidate replica selects random checkpoint0.

Both N32 candidate replicas end worse than their selected earlier checkpoints.
Final mean candidate cost/machine360.5195; final reservation409.0703.
Price messages are weak: final N32 mean magnitude0.00076374, active-slot
fraction0.0850 and same-weight unpriced matching changes1.125/48policy ticks.
No significant mechanism effect or training-stability claim is supported.

N32 fit times including development: scarcity1138–1140s, reservation930–948s,
independent702–706s. These measure real learning under this pilot's vector8,
minibatch128 configuration. Full vector16/minibatch256 timing remains uncertain;
do not extrapolate from inference or guarantee a completion time.

The read-only gradient diagnostic uses one already-opened DEV seed60500000
at all six scarcity checkpoints per size, makes no optimizer update and opens
no test. Raw critic gradients dominate actor gradients; global clip factors
range about0.0028–0.0092. This is not proof of proportional parameter-step
suppression, because Adam normalizes gradients. A separate actor/critic clipping
intervention needs an actual matched comparison and update-size logs.

Bound reports (all original artifacts retained):

- `technician_scarcity_ppo_v1_development_report_20261014T031842Z`: read-only
  tables/learning curves, source hashes, PNG/PDF; visually inspected.
- `technician_scarcity_ppo_v1_gradient_diagnostic_20261014T031847Z`: validated
  checkpoint/source/config hashes, gradient CSV, zero-update diagnostic and
  copied source. Earlier diagnostic030707 is retained too.

Next design must address action-specific alternative-use loss and occupation
time, not silently strengthen/tune the v1 scalar price after outcomes. It must
use a new prospective version/seed panel, preserve existing margins and round2
alpha (no v1 scientific test was opened), and repeat tests/smoke/push before run.
