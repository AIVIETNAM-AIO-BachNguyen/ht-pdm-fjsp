# V11 WAIT/Q diagnostic — audited DEV, no method gate

Run: `v11_wait_bias_dev_20261020T123512Z`; prelock: `v11_wait_bias_diagnostic_plan_vi.md`; source push `1e84bfb`. Independent replay: `v11_wait_bias_audit_20261020T124045Z/independent_audit.json`, **PASS** for all 7,680 stored Q-state calculations, 1,280 episode slots, 61,440 H decisions, original V11 development/selected metrics, checkpoint weights, provenance hashes and summary aggregation. This reuses opened V11 DEV data. It is mechanism diagnosis, not new learning, fresh confirmation, TEST or independent method replication.

## Observed policy behavior

The initial actors are parameter-identical across arms within each training seed; selected checkpoints separate. Across four cases and 16 physical DEV seeds each, reference-credit selected policy WAITed on 7,338/11,915 legal-service opportunities (61.6%), compared with 0/8,633 for self-credit. Mean H service starts were 16.35 versus 30.94; mean queued corrective-maintenance ticks 92.07 versus 11.62; mean cost 1138.76 versus 946.45. The reference-credit collapse is heterogeneous: selected seeds 70500001 and 70500003 each WAITed on all 3,052 legal-service opportunities and were selected at untrained iteration 0; seed 70500004 WAITed 1,234 times; seeds 70500000 and 70500002 WAITed zero times. Self-credit selected seeds WAITed zero times throughout. Selection and the full V11 DEV comparison were already audited separately.

| Case | Reference selected cost | Self selected cost | Reference legal WAITs | Self legal WAITs |
|---|---:|---:|---:|---:|
| Interchangeable scale1 | 1017.65 | 738.15 | 1896 | 0 |
| Reciprocal scale2 | 1103.66 | 927.38 | 1807 | 0 |
| Reciprocal scale3 | 1249.22 | 1128.74 | 1755 | 0 |
| Specialist/generalist scale2 | 1184.49 | 991.54 | 1880 | 0 |

## Stored paired-Q evidence

For each arm's 3,840 service-legal training states, Q scores for WAIT and every service matching came from the **same four-scenario bank**. Negative `Q(WAIT) − min Q(service)` favors WAIT. In reference-credit training states, Q_self favored WAIT in 851 states and Q_ref in 827; in self-credit training states the counts were 916 and 868. Thus the prelocked H1 claim that Q_ref *more often* favors WAIT fails. At the ≥5-unit margin, Q_ref favored WAIT in 687 versus Q_self 630 reference-credit states, and 732 versus 698 self-credit states. This occurred in four of five seeds in each arm, with seed 70500004 opposite. Signed Q preferences disagreed in 1,174/3,840 reference-credit states and 1,216/3,840 self-credit states (30.6% and 31.7%); a four-scenario estimate must not be treated as a true action value. The preference-change direction is mixed, so the larger WAIT margins alone cannot establish that assumed risk continuation caused the selected actor collapse.

The prelocked joint mechanism gate is **not met**: H1 fails on WAIT frequency, and H2's selected-policy difference is concentrated in three of five paired seeds, although its aggregate magnitude is large. The initial seeds 01/03 were severe WAIT policies in both arms; self-credit training recovered, while reference-credit selected the initial checkpoint. This supports examining credit estimator variance and actor/checkpoint optimization. It does not isolate a causal continuation mismatch or justify a hard service shield. V11 remains a DEV no-go, with no V11 TEST.

## Reproduction and next gate

Run `uv run python research_result/verify_v11_wait_bias.py research_result/v11_wait_bias_dev_20261020T123512Z --output-dir research_result/v11_wait_bias_audit_$(date -u +%Y%m%dT%H%M%SZ)` from the repo root. The verifier refuses an existing audit directory and checks raw data independently. The prior V11 training trace shows reference seed 01/03 selected iteration 0 while matched self-credit seeds selected iterations 4/3. Next prelock a trace/checkpoint diagnostic that measures whether root Q signs, advantage spread, and gradient/action changes are stable across iterations and scenarios before choosing another scratch-RL method. The overall user goal remains **ACTIVE**: no candidate meets the locked planner threshold or replication requirements.
