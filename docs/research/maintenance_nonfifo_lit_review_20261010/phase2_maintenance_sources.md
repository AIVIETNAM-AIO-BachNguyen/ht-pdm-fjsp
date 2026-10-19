# Phase 2: maintenance source verification notes

Search date: 2026-10-10. Narrative/integrative review. This file records primary-source observations, not a claim of exhaustive coverage or downstream synthesis. Sources used: publisher pages, university repositories, author manuscripts/arXiv. Search engines occasionally exposed indexed PDF sections even when direct fetch failed; access status distinguishes those cases. No paper was downloaded to the workspace.

Search clusters: exact DOI/title verification; maintenance + scarce crews + technician skills; dynamic maintenance + RL + simheuristic/rollout; 2024–2026 graph maintenance/generalization. Searches for public full text used title/DOI + PDF/arXiv/university/author. Abstract-only unknowns must not become claims of absence.

## M01 Rodríguez et al. (2022)

- **Title:** Multi-agent deep reinforcement learning based Predictive Maintenance on parallel machines.
- **Authors:** Marcelo Luis Ruiz Rodríguez; Sylvain Kubler; Andrea de Giorgio; Maxime Cordy; Jérémy Robert; Yves Le Traon.
- **Publication:** Robotics and Computer-Integrated Manufacturing 78, 102406 (2022). DOI: 10.1016/j.rcim.2022.102406.
- [Publisher](https://www.sciencedirect.com/science/article/pii/S0736584522000928); [university PDF](https://orbilu.uni.lu/bitstream/10993/54345/1/1-s2.0-S0736584522000928-main.pdf); [repository record](https://orbilu.uni.lu/handle/10993/54345).
- **Access/read:** Publisher states CC BY open access. University PDF indexed; direct fetch failed. Abstract and indexed Methods 3.2/4.3, Table 5, evaluation 4.4 checked.
- **Observed:** Partially observing machine agents choose component–available-technician or delay. PPO via RLlib, not QMIX. Heterogeneous component-specific repair durations: (8,2), (2,8), third (8,8); cases 3 machines/2 technicians and 5/3. PM, CM, random-action baselines. Conflicting requests are invalid and penalized with environment restart. FIFO queue semantics and zero-shot cross-size transfer were not verified. Do not identify this paper's environment with the repository's later FIFO/QMIX variant.

## M02 An et al. (2021)

- **Title:** Flexible job-shop scheduling and heterogeneous repairman assignment with maintenance time window and employee timetable constraints.
- **Authors:** Youjun An; Xiaohui Chen; Yinghe Li; Ji Zhang; Junwei Jiang.
- **Publication:** Expert Systems with Applications 186, 115693 (2021). DOI: 10.1016/j.eswa.2021.115693.
- [Publisher](https://www.sciencedirect.com/science/article/pii/S0957417421010770).
- **Access/read:** Subscription preview; no verified public PDF found. Abstract/introduction checked, mathematical methods unavailable.
- **Observed:** Integrated FJSP, maintenance windows, repairman competence and employee timetable; hybrid multi-objective biogeography optimization with tabu search. Reported comparisons include integrated versus hierarchical optimization. This establishes prior work on heterogeneous repairman assignment, not evidence for recurrent stochastic non-FIFO PM/CM or zero-shot generalization. Those details remain unknown.

## M03 Ruiz-Hernández et al. (2020)

- **Title:** Multi-machine preventive maintenance scheduling with imperfect interventions: A restless bandit approach.
- **Authors:** Diego Ruiz-Hernández; Jesús María Pinar-Pérez; David Delgado-Gómez.
- **Publication:** Computers & Operations Research 119, 104927 (2020). DOI: 10.1016/j.cor.2020.104927. arXiv deposit is 2024, not publication year.
- [University record](https://researchportal.uc3m.es/display/act529211); [arXiv](https://arxiv.org/abs/2401.14055); [PDF](https://arxiv.org/pdf/2401.14055); [full HTML](https://arxiv.org/html/2401.14055v1).
- **Access/read:** Public author manuscript; PDF opened successfully. Formulation and numerical sections checked.
- **Observed:** Limited repairmen allocated among deteriorating machines; interventions may return machines to an uncertain earlier wear state. Restless-bandit index policy. Experiments include 25 machines/2 repairmen and 50/3, comparing myopic, threshold and alternative indices. Technician-specific skill matching is not represented by the inspected machine-level active/passive formulation. This is an analytic scale baseline, not learned size generalization.

## M04 Demirci, Arts, van Houtum (2024 online / 2025 issue)

- **Title:** A restless bandit approach for capacitated condition based maintenance scheduling.
- **Authors:** Ece Zeliha Demirci; Joachim Arts; Geert-Jan van Houtum.
- **Publication:** Flexible Services and Manufacturing Journal 37, 179–207 (2025); online 13 June 2024. DOI: 10.1007/s10696-024-09544-y.
- [Publisher/full HTML](https://link.springer.com/article/10.1007/s10696-024-09544-y); [PDF](https://link.springer.com/content/pdf/10.1007/s10696-024-09544-y.pdf).
- **Access/read:** CC BY; PDF opened successfully. Model, MDP/RBP formulation and numerical setup checked.
- **Observed:** Independent non-identical machines; degradation-dependent revenue loss; limited cross-trained repairmen, negligible switching time; machine-specific exponential maintenance rates. Machine-selection binary vector with total capacity R, no worker identity/skill edge. Whittle policy, indexability result and LP lower bound; up to 160 machines/16 repairmen. Failure-based and threshold baselines use FCFS. Reported small gaps in larger/high-workload systems are numerical evidence, not unrestricted optimality of Whittle or evidence against RL under different assumptions.

## M05 Ertem et al. (2022)

- **Title:** Workers-constrained shutdown maintenance scheduling with skills flexibility: Models and solution algorithms.
- **Authors:** Mehmet Ertem; Rami As'ad; Mahmoud Awad; AbdulRahman Al-Bar.
- **Publication:** Computers & Industrial Engineering 172, 108575 (2022). DOI: 10.1016/j.cie.2022.108575.
- [Publisher](https://www.sciencedirect.com/science/article/abs/pii/S0360835222005769).
- **Access/read:** Publisher abstract/introduction inspected; full manuscript and public PDF not verified.
- **Observed:** Predetermined shutdown window, scarce multi-skilled workers; higher-skilled workers can perform lower-skill tasks, possibly multiple skills/workers per activity. Two time-indexed MILPs and constructive heuristics, industrial cement case and synthetic instances. Abstract reports 16.52% average shutdown-duration reduction from flexible worker utilization. Skill-level assignment rather than worker-specific identity is described in introduction. This concerns shutdown project scheduling; recurrent stochastic online PM/CM details remain unknown.

## M06 Pinciroli et al. (2021)

- **Title:** Deep Reinforcement Learning Based on Proximal Policy Optimization for the Maintenance of a Wind Farm with Multiple Crews.
- **Authors:** Luca Pinciroli; Piero Baraldi; Guido Ballabio; Michele Compare; Enrico Zio.
- **Publication:** Energies 14(20), 6743 (2021). DOI: 10.3390/en14206743.
- [Publisher](https://www.mdpi.com/1996-1073/14/20/6743); [publisher PDF](https://www.mdpi.com/1996-1073/14/20/6743/pdf).
- **Access/read:** Open access; indexed full HTML Methods 3.1/3.2 and Results 5 checked; direct fetch failed.
- **Observed:** One learning agent outputs destinations for multiple crews; PM and CM, estimated RUL, power forecasts, maintenance completion times; 50 turbines. PPO with imitation pretraining; corrective, scheduled and predictive policies, thresholds tuned by TPE. Changing crew count requires repeating training/test; this is not zero-shot size transfer. Explicit technician-specific skills were not checked. Costs of crew employment/management are neglected in reported results. Training time is reported separately from near-real-time policy use; no strong rollout comparison verified.

## M07 Pinciroli et al. (2022)

- **Title:** Optimization of the Operation and Maintenance of renewable energy systems by Deep Reinforcement Learning.
- **Authors:** Luca Pinciroli; Piero Baraldi; Guido Ballabio; Michele Compare; Enrico Zio.
- **Publication:** Renewable Energy 183, 752–763 (2022), online 2021. DOI: 10.1016/j.renene.2021.11.052.
- [Publisher](https://www.sciencedirect.com/science/article/pii/S0960148121016347); [university record](https://re.public.polimi.it/handle/11311/1195626).
- **Access/read:** Abstract and indexed introduction/conclusion checked. Repository record exists; direct access returned 403. No direct public PDF verified.
- **Observed:** Sequential O&M decision problem, PHM/RUL uncertainty, crew availability and varying demand/production; PPO plus imitation pretraining and environment model; 30-turbine case. Comparisons include maintenance rules and DRL alternatives. Shares authors/method family with M06; do not count as independent replication. Matching skills, exact planner baselines and train-small/test-large protocol are unknown from checked scope.

## M08 Ruiz-Rodríguez et al. (2024)

- **Title:** Dynamic maintenance scheduling approach under uncertainty: Comparison between reinforcement learning, genetic algorithm simheuristic, dispatching rules.
- **Authors:** Marcelo Luis Ruiz-Rodríguez; Sylvain Kubler; Jérémy Robert; Yves Le Traon.
- **Publication:** Expert Systems with Applications 248, 123404 (2024). DOI: 10.1016/j.eswa.2024.123404.
- [Publisher](https://www.sciencedirect.com/science/article/pii/S0957417424002690); [record](https://orbilu.uni.lu/handle/10993/62366); [PDF](https://orbilu.uni.lu/bitstream/10993/62366/1/1-s2.0-S0957417424002690-main.pdf).
- **Access/read:** CC BY university PDF indexed, direct fetch not confirmed. Abstract and indexed Methods 3.2/experimental setup checked.
- **Observed:** Masked PPO, GA-simheuristic and FIFO/FIBT rules. FIBT selects technician with shortest task repair duration. Uncertainty varied in failure distribution and repair time; 168-step horizon. GA-simheuristic achieves higher uptime at hourly reoptimization but deteriorates with less frequent rescheduling and incurs high computation. A real manufacturing case is included. This directly precedes quality–compute comparison; that comparison alone cannot be claimed novel. Cross-size zero-shot testing and exact worker action formulation need further full-text inspection.

## M09 Gao et al. (2025)

- **Title:** Two-stage graph attention networks and Q-learning based maintenance tasks scheduling.
- **Authors:** Xiaoyong Gao; Diao Peng; Yixu Yang; Fuyu Huang; Yu Yuan; Chaodong Tan; Feifei Li.
- **Publication:** Applied Intelligence 55(5), 331 (2025), online 16 January. DOI: 10.1007/s10489-025-06249-z.
- [Publisher](https://link.springer.com/article/10.1007/s10489-025-06249-z); [author university detailed news](https://www.cup.edu.cn/cupai/kxyj/kydt/3476628086fb435aaa43ef6248994cbf.htm).
- **Access/read:** Subscription preview explicitly states paid/institutional access. No public manuscript found after title/DOI/arXiv/author searches. Abstract and university methods summary checked.
- **Observed:** CMPNN embeddings, technician/context attention, probabilistic task assignment, Q-learning refinement and REINFORCE training. Publisher claims varying-task-count generalization and comparisons with CPLEX/OR-Tools. University illustrates 5 technicians/50 tasks and 10/100; these are example cases, not verified training/test sizes. Technician–task compatibility is represented. Exact skill dimensions, stochastic recurrent PM/CM, zero-shot technician-count transfer, compute accounting and solver configurations remain unknown. Do not convert these unknowns into claimed omissions.

## M10 Ruiz-Rodríguez et al. (2025)

- **Title:** Evolutionary multi-objective multi-agent deep reinforcement learning for sustainable maintenance scheduling.
- **Authors:** Marcelo Luis Ruiz-Rodríguez; Sylvain Kubler; Jérémy Robert; Alexandre Voisin; Yves Le Traon.
- **Publication:** Engineering Applications of Artificial Intelligence 156, 111126 (2025). DOI: 10.1016/j.engappai.2025.111126.
- [Publisher](https://www.sciencedirect.com/science/article/pii/S0952197625011273); [university PDF](https://orbilu.uni.lu/bitstream/10993/66631/1/1-s2.0-S0952197625011273-main.pdf).
- **Access/read:** CC BY publisher/university PDF indexed; direct fetch failed. Abstract and indexed Methods 4 state/actions checked.
- **Observed:** Economic, environmental/carbon and social/fatigue objectives; evolutionary preference-space agents. Each policy chooses among eight dispatch-rule combinations plus wait: highest degradation/cycle time × least fatigue/highest skill technician × perfect/imperfect repair. Thus “multi-agent” concerns exploration of preferences; inspected actions are not arbitrary joint machine–worker matchings. Baselines include CM/CBM and preference-trained DQN/PPO. Exact online stochastic semantics and cross-size generalization not verified. Shares research lineage with M01/M08.

## M11 Nucci and Papadia (2025)

- **Title:** Hybrid Genetic Algorithm and Deep Reinforcement Learning Framework for IoT-Enabled Healthcare Equipment Maintenance Scheduling.
- **Authors:** Francesco Nucci; Gabriele Papadia.
- **Publication:** Electronics 14(21), 4160 (2025). DOI: 10.3390/electronics14214160.
- [Publisher](https://www.mdpi.com/2079-9292/14/21/4160); [university PDF](https://iris.unisalento.it/bitstream/11587/564147/1/electronics-14-04160.pdf); [record](https://iris.unisalento.it/handle/11587/564147).
- **Access/read:** Open-access publisher and repository file indexed; direct fetch failed. Methods 3 and evaluation/discussion indexed excerpts checked.
- **Observed:** Multi-skill technician routing/time windows/shift availability and predicted failure intervals. GA elites seed behavior cloning, PPO refines feasible technician–task choices in receding-horizon setting. Four cases, 10–30 facilities; GA, DRL and OR-Tools baselines under stated equal 600s budgets, with GA/RL time allocation studied. Reported hybrid gains 0.93–2.86% over OR-Tools. This is a close skills/hybrid/compute neighbor; zero-shot cross-size transfer and recurrent degradation/reset PM+CM protocol were not verified. “Optimal” language in article is not an established global bound.

## M12 Tian and Shone (2026)

- **Title:** Dynamic repair and maintenance of heterogeneous machines dispersed on a network: A rollout method for online reinforcement learning.
- **Authors:** Dongnuan Tian; Rob Shone.
- **Publication:** Computers & Operations Research 193, 107517 (September 2026). DOI: 10.1016/j.cor.2026.107517.
- [Publisher](https://www.sciencedirect.com/science/article/abs/pii/S0305054826001358); [arXiv](https://arxiv.org/abs/2602.19277); [PDF](https://arxiv.org/pdf/2602.19277); [HTML](https://arxiv.org/html/2602.19277v1); [accepted manuscript](https://eprints.lancs.ac.uk/id/eprint/237332/1/Tian_Shone_COR_AuthorAccepted.pdf).
- **Access/read:** Public arXiv PDF opened successfully; formulation, index and rollout method and numerical sections checked.
- **Observed:** Single repairer, heterogeneous machines, stochastic degradation and repair/switching, interruptible repairs/routes. CTMDP long-run average cost; machines may degrade during repair. Index heuristic, offline value approximation, online rollout with reliability-based fallback to index. Experiments span 2–8 machines; optimal policies for 2–4. Explicit fast-event computation considered. This is learned-value-assisted planning, not a standalone neural dispatcher; heterogeneous workforce matching cannot be inferred from heterogeneous machines.

## Additional leads requiring limited-scope treatment

**M13 Hu et al. (2022):** Jueming Hu, Yuhao Wang, Yutian Pang, Yongming Liu. *Optimal maintenance scheduling under uncertainties using Linear Programming-enhanced Reinforcement Learning*. Engineering Applications of Artificial Intelligence109,104655. DOI10.1016/j.engappai.2021.104655 (DOI contains2021; issueyear2022). [Publisher](https://www.sciencedirect.com/science/article/pii/S0952197621004565), [author bibliography](https://jueming6.github.io/publications/). Abstract and section previews verified. Combines LP with rollout for stochastic constrained maintenance/infinite horizon, including pipeline-fatigue examples. Publisher lists an Open Manuscript link, but a direct downloadable manuscript has not been verified. Do not interpret “RL” in title as a learned graph actor or propagate the abstract's exact-optimization wording to arbitrary stochastic problems. Detailed methods/worker model not assessed from previews.

**MIRA (2026):** Md. Ashraful Babu, Ali AlArjani, Mohamed Lahby. *MIRA: Safety-Constrained Multi-Agent Reinforcement Learning for Joint Prescriptive Maintenance and Production Rescheduling in Industrial IoT*. Future Internet 18(8), 430. DOI10.3390/fi18080430. [Publisher](https://www.mdpi.com/1999-5903/18/8/430); [PDF](https://www.mdpi.com/1999-5903/18/8/430/pdf); [primary version record](https://www.mdpi.com/1999-5903/18/8/430/notes). Publisher metadata and abstract verified; full methods not inspected. Graph MARL integrates prescriptive maintenance, rescheduling, safety and communication. Keep as contextual neighbor; technician skill matching unknown.

**Unresolved IEEE lead, do not use for established methodological claims:** *Deep Reinforcement Learning for Dynamic Predictive Maintenance Task Assignment with Heterogeneous Workforce*, PHM-Xian 2025, DOI10.1109/PHM-Xian66756.2025.11427424. [IEEE landing](https://ieeexplore.ieee.org/document/11427424) gave no useful primary metadata/content. Secondary records identify Li/Zhang/Zhang and skill/proficiency dimensions, but primary authors/year/full text could not be checked. Especially close title merits manual university-library lookup before strong novelty claims.

## Verification caveats

- Successful PDF opens: Ruiz-Hernández2020 author manuscript; Demirci2024/2025 publisher PDF; Tian2026 arXiv.
- OA PDF URLs indexed but direct tool fetch failed: Rodríguez2022/2024/2025, Pinciroli2021, Nucci2025. Label as public links found, not local download success.
- No verified free full text: An2021, Ertem2022, Gao2025, Pinciroli2022.
- No unverified “first”, solver superiority, absence of FIFO/skills/planning, or exhaustive novelty conclusion follows from these notes.
- AI assistance was used for search, metadata checking and extraction; research interpretation and citation/full-text checks require author review before manuscript submission.
