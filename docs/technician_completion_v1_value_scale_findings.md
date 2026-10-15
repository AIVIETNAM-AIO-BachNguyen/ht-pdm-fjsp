# Frozen value-scale audit findings

Artifact: `artifacts/technician_completion_credit_ppo_v1_value_scale_audit_20261015T064606545809Z`.
COMPLETE: all 66 saved checkpoints and 1,056 initial predictions; 1.68 seconds.
All bound pilot/diagnostic inputs verified, all actual predictions within exact
frozen-head envelopes, parameters unchanged. Output hashes/sizes independently
rechecked. No simulator or extra trajectories, optimizer, new seeds or tests.
Smoke: 11 selected method/size checkpoints, syntax/diff checks PASS before pushed
analysis commit ad334b2 and the full offline audit. Learning sources remain frozen.

At final 61,440-step checkpoints, both candidate N4 replicas have all last-hidden
tanh activations above abs .99. Initial empty-prefix predictions have zero FP32
spread across the sixteen opened DEV states. Their frozen-head lower bounds are
-18.8845 and -19.6128; predicted means -18.8749 and -19.6006; observed greedy DEV
initial return means -72.8563 and -73.6000. Every observed initial return is below
the envelope. Selected 30,720-step candidate heads have still smaller magnitudes
(lower bounds -10.7617 and -11.4780). Final checkpoints are reported as final,
not substituted for the protocol-selected checkpoints.

At N32 the candidate final lower bounds are -151.0322 and -157.0595, predictions
-150.9567 and -156.9603, observed initial return means -435.0438 and -981.1656.
Again all observed initial returns fall below the frozen envelope, saturation
fraction is 1, and initial predictions have negligible FP32 state dependence.
Matched graph controls show the same saturation pattern. Original PPO at N4 also
shows a scale problem: final lower bounds -41.5194/-40.7151, predictions
-41.3390/-40.5665 versus observed means -63.7063/-60.0750.

The envelope is exact for a frozen tanh-hidden/linear-output head: bias +/- L1
norm of weights, including the graph N/4 output multiplier. It is not an inherent
capacity bound of the architecture with different weights or longer training.
Greedy DEV returns differ from the stochastic-policy quantities used to train
the critic; this does not establish unbiased value calibration error or prove
that saturation caused the policy failures. It does establish the measured
saturation and absence of useful initial value discrimination on this panel.

Next iteration should put critic prediction/targets into physically meaningful
units and apply that treatment equally to graph controls and strong baselines.
Normalization is established prior art, e.g.
[van Hasselt et al.](https://arxiv.org/abs/1602.07714); a fixed unit conversion is
not PopArt and is not a novel contribution. The research claim must still come
from completion/reservation conditioning and prefix credit over normalized,
matched-information/capacity/budget controls. Add an explicit calendar-removal
ablation rather than attributing a generic engineering repair to coordination.
Keep the literal original PPO baseline as an additional comparison. All original
cost/health gates and failed results remain preserved. Fresh protocol and disjoint
panels are required; no scientific confirmation has opened.
