# Literature comparison for the research proposal

## Research focus

The proposed study investigates **decentralized maintenance control with passive shared heterogeneous technicians**. Each machine is a learning agent that chooses whether to defer maintenance or request a technician. Technicians do not learn; they have fixed capabilities and service times. Requests compete for a shared technician pool and are resolved by an explicit resource-allocation rule.

The table includes the papers that are most useful for positioning this problem. It deliberately omits papers whose contribution is primarily diagnosis, degradation prediction, digital twins, or unrelated fleet-level planning.

## Comparison table

| Study | Agents / decision-maker | Maintenance timing | Technician selection | Shared resource model | Technician heterogeneity | Method and main comparator | Missing element relative to the proposed study |
|---|---|---:|---:|---|---|---|---|
| Rodríguez et al. (2022), *Multi-agent deep reinforcement learning based Predictive Maintenance on parallel machines* [DOI](https://doi.org/10.1016/j.rcim.2022.102406) | One independent RL agent per machine | Yes | Yes | Shared technician pool; duplicate requests are invalid and restart the episode | Technician-component repair-time matrix | Independent PPO; corrective, preventive, and random policies | Closest precedent, but no persistent FIFO queue or exact small-instance validation |
| Pérez-Hernández et al. (2024), *Multi-Agent Learning of Asset Maintenance Plans through Localised Subnetworks* [DOI](https://doi.org/10.1016/j.engappai.2023.107362) | MARL agents in localized asset subnetworks | Yes | No | No explicit technician pool | Asset and network heterogeneity | Localized MARL versus network-wide and individual-asset approaches | No shared technician contention or technician choice |
| Chen and Liu (2025), *Sequential multi-objective multi-agent reinforcement learning approach for system predictive maintenance of turbofan engine* [DOI](https://doi.org/10.1016/j.aei.2025.103553) | MARL agents for system/component maintenance | Yes | No | No explicit technician resource | RUL and component uncertainty | Sequential multi-objective PPO with probabilistic RUL prediction | No shared human-resource model |
| Ruiz-Hernández, Pinar-Pérez and Delgado-Gómez (2020), *Multi-machine preventive maintenance scheduling with imperfect interventions: A restless bandit approach* [DOI](https://doi.org/10.1016/j.cor.2020.104927) | Central controller | Yes | No | Shared collection of repairmen; sequential allocation | Repairmen are not skill-differentiated | Restless-bandit index policy | No decentralized machine agents or technician identity selection |
| Demirci, Arts and van Houtum (2024), *A restless bandit approach for capacitated condition based maintenance scheduling* [DOI](https://doi.org/10.1007/s10696-024-09544-y) | Central controller | Yes | No | Worker-count capacity limits simultaneous maintenance | Machine degradation differs; workers are interchangeable | Whittle index, failure-based and threshold policies, LP lower bound | No skill-aware technician assignment or queueing semantics |
| An et al. (2021), *Flexible job-shop scheduling and heterogeneous repairman assignment with maintenance time window and employee timetable constraints* [DOI](https://doi.org/10.1016/j.eswa.2021.115693) | Central scheduler | Yes | Yes | Shared repairmen with timetable and non-overlap constraints | Competence levels, task requirements, efficiency, cost, and availability | HMOBBO with tabu search; six algorithmic comparators and hierarchical variants | No online decentralized machine-agent coordination |
| Yu et al. (2023), *The Field Technician Scheduling Problem with Experience-Dependent Service Times* [DOI](https://doi.org/10.3390/math11214562) | Central scheduler | Limited; task schedule is planned centrally | Yes | Limited technicians share tasks and routes | Skill domains, proficiency levels, and experience-dependent service times | MILP and Modified Iterated Local Search | Static routing/scheduling setting; no stochastic machine-agent requests |
| Sanhueza, Delgado and Klapp (2026), *On planning cost-efficient and flexible aircraft maintenance operations: Technician shift scheduling and task assignment over multiple bases* [DOI](https://doi.org/10.1016/j.cie.2026.111801) | Central planner | Yes, within aircraft ground-time windows | Yes | Shared multi-base workforce with shifts, relocation, and outsourcing | Uni-skilled, bi-skilled, and multi-skilled technicians | MILP and Price-and-Branch heuristic | Offline workforce planning; no decentralized contention learning |
| **Proposed study** | **One learning agent per machine** | **Yes: defer or request now** | **Yes: select one technician** | **Shared passive pool with FIFO queues** | **Fixed machine-technician feasibility and service-time matrix** | **Independent PPO, centralized PPO, skill-aware FIFO dispatcher, Rodríguez collision ablation, and exact finite-horizon DP oracle** | **Target combination of decentralized timing, technician selection, heterogeneous passive resources, and explicit contention semantics** |

## Positioning statement

The closest prior work is Rodríguez et al. because it already uses one maintenance agent per machine and allows an agent to select a technician with technician-dependent repair times. The proposed study narrows and strengthens that setting by treating technicians as an explicit passive resource layer with persistent FIFO queues, measuring waiting and contention directly, and validating learned policies against an exact dynamic-programming oracle on a small instance.

The shared-resource optimization papers show that maintenance capacity is an established problem, but they use a centralized controller and do not study decentralized technician selection. The workforce-scheduling papers provide established models for competence, skill matrices, and service-time heterogeneity, but they also use centralized offline optimization. The proposed contribution lies at the intersection of these strands: machine-level learning decisions coupled through shared, heterogeneous, non-learning technicians.

## Scope of the proposed contribution

The proposal should claim a focused problem formulation and controlled experimental comparison, rather than a general solution to all workforce-planning problems. The core research question is:

> Can decentralized machine agents learn maintenance timing and technician selection that reduce expected system cost when technicians are shared passive resources with heterogeneous fixed service capabilities?

The primary comparison should use the same stochastic failure process, horizon, costs, and evaluation seeds for every method. Total expected cost is the primary outcome; failures, downtime, queue waiting, proposal conflicts, invalid proposals, service counts, and technician utilization are secondary diagnostics.
