# Cardinality-normalized opportunity PPO v3 pilot

Learning commit c4f2eb20f1a2b39dae7235ebf1db329a1d651ed9.
`technician_scarcity_ppo_v3_pilot_20261014T050040Z` COMPLETED/auditPASS,
1890.87s (31.51min), 22 fresh fits,1351680 joint decisions,28160 training
episodes,5120 paired tapes,1056 development episodes. No scientific test
opened, teacher, pretraining or invalid assignments. Every fit spends61440
decisions/1920 optimizer steps; both replicas retained.

| N | Method | Selected cost/machine | Availability | Pending/machine | Failures/machine |
|---|---|---:|---:|---:|---:|
| 4 | hierarchical opportunity PPO | 273.0391 | 0.667806 | 0.007813 | 0.671875 |
| 4 | hierarchical reservation | 278.5391 | 0.661784 | 0 | 0.773438 |
| 4 | independent | 322.6250 | 0.614095 | 0.023438 | 1.937500 |
| 4 | flat STOP with messages | 309.3047 | 0.621908 | 0.023438 | 1.320313 |
| 4 | original PPO | 296.8281 | 0.670247 | 0.031250 | 2.179688 |
| 4 | local MAPPO | 351.3516 | 0.580241 | 0.078125 | 2.398438 |
| 4 | full-information MAPPO | 358.5000 | 0.573242 | 0.054688 | 2.398438 |
| 4 | deadline | 274.8906 | 0.667969 | 0.015625 | 0.796875 |
| 32 | hierarchical opportunity PPO | 264.5186 | 0.670553 | 0 | 0.186523 |
| 32 | hierarchical reservation | 264.7412 | 0.670085 | 0 | 0.186523 |
| 32 | independent | 537.4844 | 0.405762 | 0.495117 | 3.803711 |
| 32 | flat STOP with messages | 264.5684 | 0.670634 | 0 | 0.200195 |
| 32 | deadline | 271.8750 | 0.649170 | 0 | 0.080078 |

DEVELOPMENT_ONLY point estimates, no CI or scientific efficacy claim. N4 gain
8.01% vs equally short-budget PPO,22.29% vs local MAPPO,23.84% vs full MAPPO,
1.97% vs reservation,11.72% vs flat STOP. N4 health differences vs PPO:
availability-0.002441, pending/machine-0.023438, failures/machine-1.507813;
all registered point limits pass. This does not compare to full-budget PPO.

N32 gains:50.79% vs independent,2.71% vs deadline,0.0841% vs reservation,
0.0188% vs flat STOP. Health differences vs deadline availability+0.021383,
pending0, failures/machine+0.106445; within locked limits, but disclose increased
failures. All cost/health screening entries pass EXCEPT both mechanism endpoints.
Pilot screen is therefore FALSE; never call it a passed research goal.

Both N4 candidate replicas select trained step61440: per-replica cost paths
276.219→362.875→274.109 and342.391→284.875→271.969. N32 selects30720/61440:
264.838→264.738→324.398 and275.539→305.486→264.299. Learning instability
remains, particularly one final N32 checkpoint. No best-replica exclusion.
N32 fits including DEV under8-worker load: candidate1774–1799s,
flat1714–1739s, reservation1147–1152s, independent818–820s.

## Why run full-budget development while this pilot screen is negative

The registered scientific mechanism endpoints pool N8/N16/N32. The pilot only
contains N4/N32, so its automatic pooled proxy is N32 alone; it cannot measure
the registered three-size pool or N8/N16 behavior. Its promising trained small
policies and passing physical point limits justify measuring those missing cells
and longer-budget learning. This is an exploratory DEVELOPMENT run, not
confirmation authorization or a claim that the failed pilot passed.

The already locked plan allows full DEV and requires full-budget audited DEV
PASS before confirmation; pilot PASS is not a stated prerequisite. Use its
existing fresh DEV panel and491520-step budgets at ALL four sizes, keep every
endpoint/margin/source unchanged, and open NO scientific tests. If full DEV
fails either mechanism endpoint or any cost/health screen, confirmation is
prohibited. No post-hoc relaxation or teacher fallback.

## Information, capacity and preserved reports

N4 centralized state71 exactly flattens machine6×4, technician3×2, edge4×4×2
and global9 inputs used by the graph (technician_reservation_env.graph_observation).
Full MAPPO receives this state plus4-dimensional machine identity. Graph
controls share19781 parameters, input, initial tensors and budgets; PPO18966,
local MAPPO15364, full MAPPO18052 are disclosed capacity differences. No extra
future information or teacher labels are provided to the candidate.

`technician_scarcity_ppo_v3_development_report_20261014T072214Z` contains bound
tables and learning-curves PNG/PDF with all-replica min–max, not CI. PNG inspected
and readable. `technician_scarcity_ppo_v3_mechanism_report_20261014T072214Z`
verifies all manifest-bound inputs and records all graph checkpoints/gates and
actual gradient/update norms. Keep prior v1/v2 negative pilots unchanged.
