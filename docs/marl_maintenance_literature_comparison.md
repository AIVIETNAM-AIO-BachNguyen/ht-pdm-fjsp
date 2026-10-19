# Literature comparison: MARL maintenance with shared heterogeneous technicians

## Scope

This comparison focuses on the proposed problem:

> Each machine is a learning agent that decides when to request maintenance and which technician to request; technicians are shared, passive resources with fixed heterogeneous service capabilities.

The papers below are a focused subset of the reviewed folder. They are grouped by the part of the problem they actually cover. “No technician model” means that technician availability, assignment, and service heterogeneity are not part of the reported decision model. “Shared capacity” means that the paper limits the number of simultaneous maintenance activities, but does not necessarily choose among identifiable technicians.

## Table 1. MARL and multi-agent learning for maintenance

| Study | System and decision problem | Decision-making agents | Maintenance decision | Shared technician / resource model | Heterogeneity model | Method and evaluation | Limitation relative to this study |
|---|---|---|---|---|---|---|---|
| Rodríguez et al. (2022), *Multi-agent deep reinforcement learning based Predictive Maintenance on parallel machines* [DOI](https://doi.org/10.1016/j.rcim.2022.102406) | Identical parallel machines with stochastic component failures; objective is to keep machines working and reduce breakdowns | One independent RL agent per machine | Defer, or select a component and an available technician; therefore timing and technician choice are joint actions | A finite technician pool is shared; simultaneous requests for the same technician are invalid, penalized, and restart the episode | Fixed technician-component repair-time matrix; e.g. specialized short/long repair times | Independent PPO; 3-machine/2-technician and 5-machine/3-technician scenarios; corrective, preventive, and random policies | Closest precedent, but conflict handling is invalid/restart rather than a FIFO queue; the study does not isolate queueing effects or compare against an exact small-instance oracle |
| Pérez-Hernández et al. (2024), *Multi-Agent Learning of Asset Maintenance Plans through Localised Subnetworks* [DOI](https://doi.org/10.1016/j.engappai.2023.107362) | Networked multi-asset maintenance planning with interdependencies and scalability concerns | MARL agents operate within fixed-size localized subnetworks | Produce local maintenance plans and aggregate them into a network plan | No explicit technician pool, assignment, or service capacity model | Asset/network heterogeneity and topology, rather than technician skills | Localized MARL compared with network-wide and individual-asset approaches on random networks and a nationwide infrastructure case | Useful for decentralization and scalability, but cannot test competition for a shared technician or technician selection |
| Chen & Liu (2025), *Sequential multi-objective multi-agent reinforcement learning approach for system predictive maintenance of turbofan engine* [DOI](https://doi.org/10.1016/j.aei.2025.103553) | Turbofan predictive maintenance with mutually constraining replacement and inspection objectives | MARL agents for system/component maintenance decisions | Component replacement timing and inspection interval | No explicit technician identity, availability, queue, or assignment | Component/RUL uncertainty; no technician-skill heterogeneity | GRU-based probabilistic RUL prediction plus sequential multi-objective PPO; evaluates RUL, inspection interval, unscheduled replacement, and maintenance cost | Demonstrates MARL for timing under uncertainty, but leaves shared human-resource contention outside the model |

**Interpretation.** Rodríguez is the direct anchor because it already combines machine agents, shared technicians, and technician-dependent repair times. The proposed simulator should therefore be positioned as a controlled extension that changes the resource semantics to an explicit passive queue and makes the coordination effect measurable. Pérez-Hernández supports the decentralized-agent motivation; Chen & Liu supports the maintenance-timing motivation, but neither provides the missing resource-conflict mechanism.

## Table 2. Multi-machine maintenance with shared repair capacity

| Study | System and decision problem | Decision-maker | Maintenance decision | Shared technician / resource model | Heterogeneity model | Method and evaluation | Limitation relative to this study |
|---|---|---|---|---|---|---|---|
| Ruiz-Hernández, Pinar-Pérez & Delgado-Gómez (2020), *Multi-machine preventive maintenance scheduling with imperfect interventions: A restless bandit approach* [DOI](https://doi.org/10.1016/j.cor.2020.104927) | Several deteriorating machines; interventions may return a machine to an uncertain earlier wear state | Central controller | Sequentially allocate maintenance effort to machines | A collection of shared repairmen limits which interventions can be carried out; repairmen are not differentiated by skill | No technician-specific skill or service-time matrix | Restless-bandit formulation with an index policy; numerical experiments against the underlying dynamic allocation problem | Strong shared-resource benchmark, but it is centralized, sequential, and does not model decentralized machine agents selecting technicians |
| Demirci, Arts & van Houtum (2024), *A restless bandit approach for capacitated condition based maintenance scheduling* [DOI](https://doi.org/10.1007/s10696-024-09544-y) | Non-identical degrading machines with state-dependent maintenance cost | Central controller | Choose which machines receive maintenance under a simultaneous-activity limit | Number of maintenance workers is a capacity constraint; workers are interchangeable in the model | Machine degradation differs; technician capabilities do not | Whittle-index heuristic with an LP lower bound; compared with failure-based and threshold policies | Provides a useful capacity and lower-bound comparator, but not identifiable technician choice, skill heterogeneity, queues, or decentralized coordination |

**Interpretation.** These papers establish that shared maintenance capacity is a well-studied optimization issue. The missing dimension is the *identity* of the selected resource and the resulting coordination problem: two machine agents can prefer different technicians even when the total worker count is unchanged.

## Table 3. Heterogeneous technician assignment and workforce scheduling

| Study | System and decision problem | Decision-maker | Maintenance decision | Shared technician / resource model | Heterogeneity model | Method and evaluation | Limitation relative to this study |
|---|---|---|---|---|---|---|---|
| An et al. (2021), *Flexible job-shop scheduling and heterogeneous repairman assignment with maintenance time window and employee timetable constraints* [DOI](https://doi.org/10.1016/j.eswa.2021.115693) | Flexible job-shop scheduling coupled with maintenance tasks, time windows, and employee timetables | Central scheduler | Schedule jobs and maintenance, then assign a feasible repairman | Repairmen are shared and cannot overlap tasks; availability follows employee timetables | Competence levels, task requirements, efficiency, cost, and unavailable intervals | Multi-objective HMOBBO with flexible decoding and tabu search; compared with six intelligent algorithms and hierarchical variants | Rich skill-aware assignment benchmark, but production scheduling and timetable constraints dominate; no online machine agents or stochastic request contention |
| Yu et al. (2023), *The Field Technician Scheduling Problem with Experience-Dependent Service Times* [DOI](https://doi.org/10.3390/math11214562) | Field tasks must be matched, sequenced, and serviced by a limited technician workforce | Central scheduler | Select tasks, assign technicians, and sequence technician routes | Technicians are shared across tasks and travel from a depot | Explicit skill domains, proficiency levels, and experience-dependent service times | MILP plus Modified Iterated Local Search on generated benchmark instances | Good formal precedent for a service-time matrix, but it is a static routing/scheduling problem rather than decentralized online maintenance control |
| Dahite et al. (2022), *Multi-Objective Model and Variable Neighborhood Search Algorithms for the Joint Maintenance Scheduling and Workforce Routing Problem* [DOI](https://doi.org/10.3390/math10111807) | Geographically distributed machines with random failures; maintenance timing is coupled with technician routing | Central scheduler | Set preventive/corrective maintenance plan and technician visit sequence | Technicians are shared mobile resources; route and travel time determine availability | Technician/resource differences are mainly represented through routing and workload, not a fixed machine-technician skill matrix | Bi-objective mathematical model; Pareto VND/GVNS compared with a commercial solver and existing heuristics | Relevant for centralized scheduling, but routing introduces a separate spatial decision layer absent from the proposed simulator |
| Sanhueza, Delgado & Klapp (2026), *On planning cost-efficient and flexible aircraft maintenance operations: Technician shift scheduling and task assignment over multiple bases* [DOI](https://doi.org/10.1016/j.cie.2026.111801) | Aircraft line-maintenance planning across bases, shifts, ground-time windows, and external resources | Central planner | Decide where, when, and by whom each maintenance job is performed | Shared workforce with shifts, relocation, outsourcing, and multi-base capacity | Uni-skilled, bi-skilled, and multi-skilled technicians; temporal and spatial flexibility | MILP solved with Price-and-Branch; real provider data; multi-skilled workforce reduced cost by 48.5% versus the base case | Strong modern workforce benchmark, but offline planning with shifts, routing, and outsourcing; it does not study machine-agent coordination under stochastic contention |

**Interpretation.** The assignment literature supplies the right language for capability matrices, feasibility, service duration, and workforce constraints. Its decision-maker is almost always a central scheduler. That difference is essential: the proposed contribution is not merely another skill-aware assignment model, but a decentralized learning problem in which each machine proposes a technician request and the passive resource layer resolves contention.

## Synthesis gap matrix

Legend: **Yes** = explicit in the paper; **Partial** = represented only as an aggregate or indirect constraint; **No** = not modeled.

| Study | Machine agents | Joint timing + technician selection | Shared technicians | Fixed technician heterogeneity | Passive technicians | Decentralized contention | Exact small-instance validation |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rodríguez et al. (2022) | Yes | Yes | Yes | Yes | Yes | Partial: invalid duplicate request/restart | No |
| Pérez-Hernández et al. (2024) | Yes | Partial | No | No | No | No | No |
| Chen & Liu (2025) | Yes | Partial | No | No | No | No | No |
| Ruiz-Hernández et al. (2020) | No | No | Yes | No | Yes | No | Dynamic-programming formulation, but not the proposed Dec-POMDP comparison |
| Demirci et al. (2024) | No | No | Partial: worker-count capacity | No | Yes | No | LP lower bound, not joint-action enumeration |
| An et al. (2021) | No | No | Yes | Yes | Yes | No | Solver/algorithm comparisons, not an RL oracle |
| Yu et al. (2023) | No | No | Yes | Yes | Yes | No | MILP benchmark, not online stochastic control |
| Dahite et al. (2022) | No | Partial | Yes | Partial | Yes | No | Commercial-solver comparison |
| Sanhueza et al. (2026) | No | No | Yes | Yes | Yes | No | MILP/heuristic planning comparison |
| **Proposed simulator** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes: FIFO queue; Rodríguez collision as ablation** | **Yes: exact finite-horizon DP on a small cell** |

## What the comparison supports

The literature supports a narrow, defensible research definition:

> **Decentralized maintenance control with passive shared heterogeneous technicians.** In a finite-horizon multi-machine system, each machine is a learning agent. At each decision epoch it chooses to defer maintenance or request one technician. Technicians do not learn: each has fixed feasibility and service-time characteristics. Requests compete for a shared pool and are resolved by an explicit queue/dispatch rule. The objective is to minimize expected downtime, failure, maintenance, and waiting costs.

This definition separates three contributions that are often conflated:

1. **Maintenance timing:** whether a machine should request service now or defer.
2. **Technician selection:** which fixed-capability technician the machine should request.
3. **Coordination semantics:** what happens when multiple machines request the same scarce technician.

The third item is the clearest experimental gap. Rodríguez et al. already show the relevance of technician-aware machine agents, but their invalid-action/restart rule does not expose queue delay or a persistent resource-allocation process. The new simulator can therefore make a focused claim about semantics and coordination while retaining Rodríguez as a reproducibility anchor.

## Recommended comparison protocol for the research paper

Use the same failure process, horizon, cost function, and evaluation seeds for every method. Compare:

- independent machine-agent PPO;
- a skill-aware FIFO dispatcher;
- a centralized PPO policy;
- the Rodríguez invalid-collision ablation;
- an exact finite-horizon dynamic-programming oracle on the 2-machine/2-technician cell.

Report total cost as the primary metric, followed by failures, downtime, service jobs, queue waiting, proposal conflicts, invalid proposals, technician utilization, and machine-level service counts. This makes it possible to distinguish a policy that improves maintenance timing from one that merely avoids technician collisions.

## Selected references

- Rodríguez, M. L. Ruiz et al. (2022). *Multi-agent deep reinforcement learning based Predictive Maintenance on parallel machines*. Robotics and Computer-Integrated Manufacturing, 78, 102406.
- Pérez-Hernández, M., Puchkova, A., & Parlikad, A. K. (2024). *Multi-Agent Learning of Asset Maintenance Plans through Localised Subnetworks*. Engineering Applications of Artificial Intelligence, 127, 107362.
- Chen & Liu (2025). *Sequential multi-objective multi-agent reinforcement learning approach for system predictive maintenance of turbofan engine*. Advanced Engineering Informatics.
- Ruiz-Hernández, D., Pinar-Pérez, J. M., & Delgado-Gómez, D. (2020). *Multi-machine preventive maintenance scheduling with imperfect interventions: A restless bandit approach*. Computers & Operations Research, 119, 104927.
- Demirci, E. Z., Arts, J., & van Houtum, G.-J. (2024). *A restless bandit approach for capacitated condition based maintenance scheduling*. Flexible Services and Manufacturing Journal, 37, 179–207.
- An, Y., Chen, X., Li, Y., Zhang, J., & Jiang, J. (2021). *Flexible job-shop scheduling and heterogeneous repairman assignment with maintenance time window and employee timetable constraints*. Expert Systems with Applications, 186, 115693.
- Yu, X. et al. (2023). *The Field Technician Scheduling Problem with Experience-Dependent Service Times*. Mathematics, 11, 4562.
- Dahite, L. et al. (2022). *Multi-Objective Model and Variable Neighborhood Search Algorithms for the Joint Maintenance Scheduling and Workforce Routing Problem*. Mathematics, 10, 1807.
- Sanhueza, J. I., Delgado, F., & Klapp, M. A. (2026). *On planning cost-efficient and flexible aircraft maintenance operations: Technician shift scheduling and task assignment over multiple bases*. Computers & Industrial Engineering, 214, 111801.
