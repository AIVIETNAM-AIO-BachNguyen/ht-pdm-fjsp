# Phase 2 — Generalization, constrained neural decisions, and contrary evidence
Search/access check: 2026-10-10. Targeted narrative-review source verification, not exhaustive systematic screening.
All 13 papers below have primary abstract/metadata and publicly accessible PDF bodies checked through web.open. “PDF public” means the tool fetched a PDF, not a guarantee about the user's network. OpenReview direct URLs sometimes return anti-bot; verified arXiv alternatives are supplied.
No repository code changed. No claim below establishes maintenance-domain transfer.

## G1 — Zhang et al. 2020 — L2D
**Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning.** Cong Zhang, Wen Song, Zhiguang Cao, Jie Zhang, Puay Siew Tan, Chi Xu. NeurIPS 2020.
Page: https://proceedings.neurips.cc/paper/2020/hash/11958dfee29b6709f48a9ba0387a2431-Abstract.html
PDF public: https://proceedings.neurips.cc/paper/2020/file/11958dfee29b6709f48a9ba0387a2431-Paper.pdf
Read method/experiments pp. 3–8: disjunctive-graph GIN embeddings, candidate-operation scoring, PPO learns a centralized dispatching rule. Train 20×20/30×20 policies; directly evaluate 50×20/100×20 (jobs×machines). Main comparison beats four priority rules; Table 2 still reports 22.1% gap to OR-Tools solutions at 50×20 for the 30×20 policy, and 9.4% at 100×20. Thus generalization is measured, but does not establish near-optimality or superiority to strong search. Static makespan JSSP, no stochastic degradation or technicians. Source: linked PDF §§4–5/Tables2–4.

## G2 — Wang et al. 2023 — DANIEL
**Flexible Job Shop Scheduling via Dual Attention Network-Based Reinforcement Learning.** Runqing Wang, Gang Wang, Jian Sun, Fang Deng, Jie Chen. IEEE TNNLS, online 2023. DOI 10.1109/TNNLS.2023.3306421.
Primary manuscript page: https://arxiv.org/abs/2305.05119
Publisher: https://doi.org/10.1109/TNNLS.2023.3306421
PDF public manuscript: https://arxiv.org/pdf/2305.05119
HTML read: https://arxiv.org/html/2305.05119v2
Read §§II–IV: distinct operation/message and machine/message attention blocks; machine competition relationships; centralized feasible operation–machine-pair decision, jointly sequencing and machine assignment. Generalization policies trained 10×5/20×10 tested 30×10/40×10. Comparisons include PDR, prior DRL, GA, and time-limited OR-Tools; some sampled-policy settings beat its bounded run. All jobs arrive at time zero; dynamic uncertainty listed as future work. Do not convert “beats time-limited OR-Tools on some tasks” into “outperforms exact optimum.” Source: primary HTML §IV and author repo https://github.com/ZongminYu/DANIEL .

## G3 — Kool et al. 2019 — Attention Model
**Attention, Learn to Solve Routing Problems!** Wouter Kool, Herke van Hoof, Max Welling. ICLR 2019.
Page: https://openreview.net/forum?id=ByxBFsRqYm
PDF public arXiv: https://arxiv.org/pdf/1803.08475
Direct publisher PDF (anti-bot on current open, search indexed): https://openreview.net/pdf?id=ByxBFsRqYm
Read §§3–5: self-attention encoder without positional encoding; autoregressive decoder conditions on partial solution and feasibility masks; REINFORCE using deterministic greedy rollout baseline. TSP, VRP variants, OP, stochastic PCTSP. Main models train/evaluate n=20/50/100; do not cite this as established large-scale zero-shot success. Neural decoder provides a reusable constrained construction method, not maintenance-specific proof. Source: arXiv PDF §§3–5.

## G4 — Kwon et al. 2020 — POMO
**POMO: Policy Optimization with Multiple Optima for Reinforcement Learning.** Yeong-Dae Kwon, Jinho Choo, Byoungjip Kim, Iljoo Yoon, Youngjune Gwon, Seungjai Min. NeurIPS 2020.
Page: https://arxiv.org/abs/2010.16011
PDF public proceedings: https://papers.neurips.cc/paper_files/paper/2020/file/f231f2107df69eab0a3862d50018a9b2-Paper.pdf
Read §§3–5: multiple starting positions exploit alternative equivalent solution representations; shared-baseline policy gradient and augmentation-based best-of-many inference. TSP/CVRP sizes20/50/100, knapsack50/100/200. TSP100 0.14% gap is with multiple rollouts and ×8 augmentation, not a single greedy decision. Paper explicitly evaluates single-trajectory mode separately and notes added trajectories; report inference budget. No larger-size zero-shot evidence extracted here. Source: PDF §5/Tables2–4.

## G5 — Li et al. 2021 — Learning to Delegate
**Learning to delegate for large-scale vehicle routing.** Sirui Li, Zhongxia Yan, Cathy Wu. NeurIPS 2021.
Page: https://papers.neurips.cc/paper_files/paper/2021/hash/dc9fa5f217a1e57b8a6adeb065560b38-Abstract.html
PDF public: https://papers.neurips.cc/paper_files/paper/2021/file/dc9fa5f217a1e57b8a6adeb065560b38-Paper.pdf
Read §§4–6: learned Transformer predicts useful subproblem selection; existing LKH/HGS subsolver improves selected region. Training mixes N=500/1000/2000 for each distribution; evaluation includes3000 without finetuning, so not train-only-tiny→arbitrary-large. Headline10–100× acceleration over whole solver includes decomposition benefit; learned selection's incremental benefit over random/heuristic selection is1.5–2×. Fixed-quality speedup and fixed-time improvement are both defined. Source: PDF §§5–6/Figures4–7. Code https://github.com/mit-wu-lab/learning-to-delegate .

## G6 — Luo et al. 2023 — LEHD
**Neural Combinatorial Optimization with Heavy Decoder: Toward Large Scale Generalization.** Fu Luo, Xi Lin, Fei Liu, Qingfu Zhang, Zhenkun Wang. NeurIPS 2023.
Page: https://proceedings.neurips.cc/paper_files/paper/2023/hash/1c10d0c087c14689628124bbc8fa69f6-Abstract-Conference.html
PDF public: https://proceedings.neurips.cc/paper_files/paper/2023/file/1c10d0c087c14689628124bbc8fa69f6-Paper-Conference.pdf
Read §§3–6: heavy decoder re-embeds remaining nodes at each partial-construction step; supervised partial-solution reconstruction, not RL. Train TSP100/CVRP100 with1m labeled instances each; test100/200/500/1000 and TSPLIB/CVRPLIB. Random Re-Construct optionally adds iterative inference improvement. Separate greedy construction from RRC compute. Teacher/reference uses Concorde for TSP and HGS for CVRP; HGS is heuristic, so do not call all CVRP labels certified optimal. Source: PDF §6 andTables1–2.

## G7 — Drakulic et al. 2023 — BQ-NCO
**BQ-NCO: Bisimulation Quotienting for Efficient Neural Combinatorial Optimization.** Darko Drakulic, Sofia Michel, Florian Mai, Arnaud Sors, Jean-Marc Andreoli. NeurIPS 2023.
Page: https://proceedings.neurips.cc/paper_files/paper/2023/hash/f445ba15f0f05c26e1d24f908ea78d60-Abstract.html
PDF public: https://papers.nips.cc/paper_files/paper/2023/file/f445ba15f0f05c26e1d24f908ea78d60-Paper-Conference.pdf
Alternate public PDF read: https://arxiv.org/pdf/2301.03313
Read method/§6: reduce constructive MDP by problem symmetries/bisimulation; attention policy imitates expert trajectories. Routing models train size100; test100/200/500/1000 plus realistic benchmarks. Most models limit test subgraphs to250 nearest neighbors. Theorem relates an *optimal policy in the reduced MDP* to solving COP; it does not prove the learned network is optimal. State reduction preserves a particular recursive COP's structure; cannot assume valid for maintenance dynamics. Source: arXiv PDF §§3–6.

## G8 — Ye et al. 2024 — GLOP
**GLOP: Learning Global Partition and Local Construction for Solving Large-Scale Routing Problems in Real-Time.** Haoran Ye, Jiarui Wang, Helan Liang, Zhiguang Cao, Yong Li, Fanzhang Li. AAAI2024,38(18),20284–20292. DOI10.1609/aaai.v38i18.30009.
Page: https://ojs.aaai.org/index.php/AAAI/article/view/30009
PDF public: https://ojs.aaai.org/index.php/AAAI/article/download/30009/31772
Read intro/method: hierarchical routing→TSP→shortest Hamiltonian path; nonautoregressive global partition + autoregressive local construction. TSP100K headline5.1% gap/174× speedup compares a specified1-run1-trial LKH-3 setting, not all possible LKH budgets. Evaluates TSP/ATSP/CVRP/PCTSP. Exact local training-size extraction not completed; do not invent. Shows scale can be addressed by decomposition rather than one global sequence decoder. Source: proceedings PDF,pp.1–4.

## G9 — Liu et al. 2023 — primary negative/benchmark evidence
**How Good Is Neural Combinatorial Optimization? A Systematic Evaluation on the Traveling Salesman Problem.** Shengcai Liu, Yu Zhang, Ke Tang, Xin Yao. IEEE Computational Intelligence Magazine18(3):14–28,2023. DOI10.1109/MCI.2023.3277768. arXiv initial2022.
Primary page: https://arxiv.org/abs/2209.10913
Publisher: https://doi.org/10.1109/MCI.2023.3277768
PDF public manuscript: https://arxiv.org/pdf/2209.10913
Read experiments: evaluate effectiveness, efficiency, stability, scale, distribution generalization of POMO/DACT/NeuroLKH versus tuned LKH/EAX. TableIV: uniformTSP50-trainedDACT→TSP100 gap27.54%; same-size trained0.66%. POMO transfer50→100 worsens0.1278%→0.6703%. Variable-length inputs alone therefore do not ensure good larger-size solutions. Findings apply to methods/distributions tested, not all subsequent NCO. Source: PDF §IV/TabIV. Author confirms2023journal metadata https://senshinel.github.io/ .

## G10 — Thyssens et al. 2023 — Routing Arena
**Routing Arena: A Benchmark Suite for Neural Routing Solvers.** Daniela Thyssens, Tim Dernedde, Jonas K. Falkner, Lars Schmidt-Thieme. arXiv2023; no peer-reviewed venue asserted.
Page: https://arxiv.org/abs/2310.04140
PDF public: https://arxiv.org/pdf/2310.04140
Read §§4/6: fixed-budget and anytime evaluation; WRAP/primal-integral metrics; hardware-normalized runtime scheme, with authors noting its imperfections. Tests neural methods and OR solvers including HGS/FILO/LKH. Table4 averagegap: HGS0.004%,FILO0.175%,LKH0.58%,NeuroLKH0.67% across evaluated sets at specified maximum budgets. Strong OR solves well early, challenging simplistic neural-vs-slow-search framing. Normalization scheme is not a universal theorem; same-hardware measurement can avoid its assumptions. Source: PDF §§4/6,Table4,Fig2.

## G11 — Chen et al. 2025 — TTPL
**Improving Generalization of Neural Combinatorial Optimization for Vehicle Routing Problems via Test-Time Projection Learning.** Yuanyao Chen, Rongsheng Chen, Fu Luo, Zhenkun Wang. NeurIPS2025.
Page: https://proceedings.neurips.cc/paper_files/paper/2025/hash/6edd46d69ef91f4555d67f7b321d6902-Abstract-Conference.html
PDF public: https://arxiv.org/pdf/2506.02392
Publisher PDF: https://papers.nips.cc/paper_files/paper/2025/file/6edd46d69ef91f4555d67f7b321d6902-Paper-Conference.pdf
Read abstract/intro: LLM-driven framework discovers distribution projection for use in inference without backbone retraining. Backbone trained100nodes tested routing scales up to100K, various distributions. Test-time pipeline includes local candidate subgraphs, projection, and multi-view decision fusion. It does not establish unchanged single-forward-pass scaling; however, do not equate its name with per-instance neural weight fine-tuning. Projection-design cost and inference budget require separate reporting. Detailed numerical table extraction not completed; no speedup number quoted. Source: PDF pp.1–2. Code https://github.com/CIAM-Group/TTPL .

## G12 — Berto et al. 2025 — PARCO
**PARCO: Parallel AutoRegressive Models for Multi-Agent Combinatorial Optimization.** Federico Berto, Chuanbo Hua, Laurin Luttmann, Jiwoo Son, Junyoung Park, Kyuree Ahn, Changhyun Kwon, Lin Xie, Jinkyoo Park. NeurIPS2025; initial arXiv2024 title was “Learning Parallel Autoregressive Policies for Efficient Multi-Agent Combinatorial Optimization.”
Page: https://arxiv.org/abs/2409.03811
PDF public: https://arxiv.org/pdf/2409.03811
Read §§4/5/AppendixB: Transformer communication, simultaneous multi-pointer action decoding, priority conflict handler, shared policy-gradient training. Tests heterogeneous capacitated routing, multi-depot pickup-delivery, flexible flow-shop. OMDCPDP training samplesN50–100 andM5–20; direct test up toN1000, agent counts outside training. FFSP separately trains per configurations; don't generalize the routing zero-shot claim to all tasks. Source: PDF §5/AppendixB.2–B.3. Multi-agent physical entities and shared central decoding do not by themselves establish a decentralized execution requirement. Code https://github.com/ai4co/parco .

## Optional G13 — Son et al. 2026 — realistic edge structure
**RRNCO: Towards Real-World Routing with Neural Combinatorial Optimization.** Jiwoo Son, Zhikai Zhao, Federico Berto, Chuanbo Hua, Zhiguang Cao, Changhyun Kwon, Jinkyoo Park. ICLR2026; arXivv1(2025) title/authors differ; currentv2PDFverified.
Page: https://arxiv.org/abs/2503.16159
PDF public: https://arxiv.org/pdf/2503.16159
Venue page anti-bot currently: https://openreview.net/forum?id=sKvo9ZZfpe
Read §§1/4/5: benchmark realistic asymmetric distance/duration matrices from100cities; node/edge data and neural adaptive bias, comparing neural and traditional solvers. Separates real topology/distribution from usual uniform Euclidean data. This is not a scale-generalization proof for maintenance; mechanism illustrates that edge compatibility/cost structure cannot be replaced automatically by homogeneous node assumptions. Source: currentPDF pp.1–8 and authorcode https://github.com/ai4co/real-routing-nco .

## Search log / limits
Queries: exact L2D title; flexible jobshop dual attention; LEHD heavy decoder; Kool attention routing; POMO; Learning to Delegate; BQ Drakulic; GLOP; NCO generalization2024/2025; How Good Is NCO; Routing Arena; PARCO; TTPL; RRNCO.
Selected primary publishers/proceedings/arXiv/authorcode. Secondary search hits used for discovery only, not final claims. Full PDFs fetched; selected relevant methods/results read, not every appendix exhausted. Sources support graph/set constrained decision methods and measurable failures; no blanket optimality/generalization guarantee. Updates after2026-10-10 not in scope.

