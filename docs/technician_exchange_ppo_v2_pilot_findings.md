# Exchange PPO v2: completed negative pilot

The pilot does not authorize development or confirmation. The actual
`validate_pilot` function rejects it with `pilot cost/health screen fails`.
Ordered commitment eliminated complete repair cancellation, but the registered
comparisons do not support a cost or health benefit from that mechanism.

Frozen learning commit: `d42072cbf29afe9b56feaabd0acbb12e653d5315`.
Source artifact:
`artifacts/technician_exchange_ppo_v2_pilot_20261015T184948Z`.
Independent report:
`artifacts/technician_exchange_ppo_v2_pilot_report_20261016T012334468284Z`.

All 36 randomly initialized fits completed 61,440 decisions each: 2,211,840
decisions, 46,080 training episodes, 5,120 paired physical tapes and 1,728 DEV
episodes, plus 32 deadline reference episodes. Elapsed time was 6,833.60 seconds.
No teacher, pretraining, warm start or scientific test was used. The training
audit passes. The independent report verifies all 493 output bindings, 260
current source/protocol/registry bindings, the episode grid, checkpoint choices,
means and all 31 point screens. Its five output hashes and script identity were
checked separately. The learning-curve PNG was visually inspected.

## Small-setting result

| N4 method | Cost/machine | Availability | Failures/machine |
|---|---:|---:|---:|
| Candidate: ordered commitment | 314.4141 | 0.622559 | 1.679688 |
| Unrestricted exchange | 281.4063 | 0.660645 | 0.906250 |
| Immediate-inverse mask | 294.9141 | 0.638672 | 1.132813 |
| Self repair | 313.4219 | 0.609375 | 1.265625 |
| Construct only | 297.3203 | 0.651367 | 1.531250 |
| Independent proposals | 286.5391 | 0.667155 | 1.414063 |
| Original PPO | 343.2734 | 0.569987 | 1.859375 |
| Normalized PPO | 263.7188 | 0.666341 | 0.140625 |
| Local MAPPO | 288.1719 | 0.670573 | 1.578125 |
| Full-state MAPPO | 286.5781 | 0.670410 | 1.570313 |

Candidate cost is 19.223% above normalized PPO and 11.730% above the matched
unrestricted-exchange control. Its availability difference against normalized
PPO is -0.043783, below the -0.01 bound; its failures/machine difference is
+1.539063, above the +0.25 bound. Candidate pending/machine is 0.015625 versus
zero for normalized PPO, within the +0.025 bound. Lower cost than original PPO
does not overcome the normalized-PPO cost/health failures or establish a
scientific improvement. Both candidate replicas select their final checkpoint;
neither is discarded.

Nine mandatory N4 screens fail: seven cost comparisons (unrestricted, inverse
mask, construct only, independent, normalized PPO, local MAPPO, full MAPPO) and
two health comparisons against normalized PPO (availability and failures).

## Large-setting result

N32 candidate cost/machine is 264.6143. Unrestricted exchange, inverse mask,
self repair and construct only achieve 262.8281, 263.4961, 263.4727 and 263.4355.
All four candidate gains against those controls are negative, approximately
-0.680%, -0.424%, -0.433% and -0.447%. These are N32-only exploratory proxies,
not a full N8/N16/N32 pool or confidence intervals. Their four +0.5% mechanism
screens fail. Including the nine mandatory failures gives 13/31 failed screens.

The N32 candidate passes the pilot's nonpooled cost/health bounds. Its cost is
2.530% below deadline's 271.4824; availability is 0.670858 versus 0.649495;
pending/machine is zero for both; failures/machine are 0.186523 versus 0.054688
(difference +0.131836, within +0.25). This is not evidence of fewer failures.
Independent proposals cost 757.5615/machine with much worse health, but the
other coordinated controls already achieve comparable or lower cost than the
candidate. Beating independent proposals does not isolate ordered commitment.

## Consequence and next evidence

All six core arms share 19,255 parameters, initial tensors, physical information,
the registered objective, 2K slots and training budget. Ordered commitment's
zero-cancellation invariant holds on every recorded candidate training and DEV
episode. That mechanical success did not translate into the registered quality
benefit. The pilot is exploratory, with two replicas and shorter-than-final
training budgets; no scientific confidence interval or confirmation alpha is
spent. There is no v2 development/confirmation run or promotion of the root
teacher-based method as a teacher-free success.

The next registered step is a frozen-trajectory diagnostic on this already
opened DEV panel. It will reconcile service occupancy, failed waiting,
availability, failure timing and provisional construction/repair choices.
Those observations can inform a new hypothesis; they cannot establish causal
preventability or justify a post-hoc duration penalty or relaxed health bound.

The registered diagnostic is now complete; see the
[frozen diagnostic findings](technician_exchange_ppo_v2_diagnostic_findings.md).
