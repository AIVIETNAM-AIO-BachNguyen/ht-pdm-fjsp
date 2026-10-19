# Verified planning, imitation and evaluation sources

Date: 2026-10-10. Phase 2 notes. Primary-source verification; the synthesis is a separate artifact. Read scope is explicit. These records make no full-paper methodological assessment where only an abstract was read.

## P01 — Bertsekas & Castanon (1999)

**Rollout Algorithms for Stochastic Scheduling Problems.** *Journal of Heuristics, 5*, 89–108. DOI: 10.1023/A:1009634810396.

- [Publisher](https://link.springer.com/article/10.1023/A:1009634810396).
- [MIT technical-report version, April 1998](https://dspace.mit.edu/entities/publication/eb06d31c-812f-476e-b5c3-38870e357885); [repository download](https://dspace.mit.edu/bitstreams/2991307d-4523-4309-8562-0e0e3d67af28/download).
- Identity verified on publisher; publisher is subscription preview. MIT lists a public 1.82 MB PDF, but the browser tool returned HTTP405 for download. An author-hosted ASU URL found in search returned404 and is deliberately not offered as the download route.
- Read: publisher abstract and repository metadata. Abstract establishes rollout as heuristic-based approximation of stochastic dynamic programming, with improvement guarantees under stated circumstances. Those conditions were not audited here; no guarantee is assigned to the project's finite-scenario planner.
- Method weaknesses: not assessed (read scope: abstract_only).

## P02 — Ross, Gordon, & Bagnell (2011)

**A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning.** AISTATS, PMLR15, 627–635.

- [Proceedings](https://proceedings.mlr.press/v15/ross11a.html); [PDF](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf), opened as a nine-page PDF.
- Read: abstract, introduction and DAgger algorithm discussion (PDF pp.1–4). DAgger collects expert labels on states induced by the learner; the sequential distribution differs from an i.i.d. expert dataset. The theory has additional reduction assumptions and does not establish optimality for an approximate teacher.
- Transfer scope: general sequential imitation, not a maintenance evaluation. No maintenance performance claim is taken from this paper.

## P03 — Anthony, Tian, & Barber (2017)

**Thinking Fast and Slow with Deep Learning and Tree Search.** *Advances in Neural Information Processing Systems30*.

- [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2017/hash/d8e1344e27a5b08cdfd5d027d9b8d6de-Abstract.html); [PDF](https://proceedings.neurips.cc/paper/2017/file/d8e1344e27a5b08cdfd5d027d9b8d6de-Paper.pdf), opened as an eleven-page PDF.
- Read: abstract and method overview. Expert Iteration alternates planning/search with a network that generalizes the plans and guides later search. Empirical task is Hex. This establishes a learning–planning precedent, not evidence of technician dispatch performance.
- Detailed method weaknesses: not assessed beyond the recorded method overview.

## P04 — Mao, Schrotenboer, Wu, & van Jaarsveld (2025)

**Improving After-sales Service: Deep Reinforcement Learning for Dynamic Time Slot Assignment with Commitments and Customer Preferences.** arXiv:2509.17870v1, submitted2025-09-22. Treat as preprint: no journal publication was verified in the targeted title search.

- [Record](https://arxiv.org/abs/2509.17870); [PDF](https://arxiv.org/pdf/2509.17870), opened as a35-page PDF; [HTML](https://arxiv.org/html/2509.17870v1).
- Read: §§3–5, especially4.1–4.2 and5.1–5.4.
- Sequential commitments to customer slots precede technician/vehicle routing. ADRL-RE combines attention-based RL with online rollouts, using a neural routing heuristic for training/forecast trajectories and OR-Tools for actual end-of-day routing. Comparator includes tuned scenario-based planning.
- Tables2–3 report lower mean total costs for ADRL-RE than SBP across six systems, with decision times roughly0.7–1.0s versus~2s. These are paper-specific measurements, not our latency claims.
- §5.1.2 explicitly trains a separate ADRL model for each system. Therefore the reported cross-system table cannot be used as evidence for one checkpoint's zero-shot transfer across fleet sizes. This is a bounded reading of the stated protocol, not an abstract-derived absence claim.
- Transfer scope: appointment/routing commitments and costs differ from in-plant recurring degradation and idle-technician matching.

## P05 — Agarwal, Schwarzer, Castro, Courville, & Bellemare (2021)

**Deep Reinforcement Learning at the Edge of the Statistical Precipice.** *Advances in Neural Information Processing Systems34*.

- [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html); [PDF](https://proceedings.neurips.cc/paper_files/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf), opened as a17-page PDF; [author project](https://agarwl.github.io/rliable/).
- Read: introduction, Table1 and §§3–4. The paper shows that limited independent training runs make point-estimate comparisons unreliable, and recommends interval estimates, performance profiles and robust aggregate measures. Its empirical benchmarks are Atari/Procgen/control; it does not prescribe one universally correct confidence interval for this maintenance experiment.

## P06 — Ma, Pan, Li, & Yan (2025)

**ML4CO-Bench-101: Benchmark Machine Learning for Classic Combinatorial Problems on Graphs.** NeurIPS2025 Datasets and Benchmarks Track. DOI:10.52202/085713-3921.

- [Proceedings](https://papers.neurips.cc/paper_files/paper/2025/hash/aa9340420e8d78021c35ae984e1bda85-Abstract-Datasets_and_Benchmarks_Track.html); [PDF](https://papers.neurips.cc/paper_files/paper/2025/file/aa9340420e8d78021c35ae984e1bda85-Paper-Datasets_and_Benchmarks_Track.pdf), opened as35 pages; [author code](https://github.com/Thinklab-SJTU/ML4CO-Bench-101).
- Read: §1, §§2.1–2.3 and benchmark overview. Introduces a common framework and65 test datasets for seven CO tasks. Separates algorithm design, decoding and postprocessing; identifies inconsistent timing, random multi-sampling and selecting the best of training seeds as comparison hazards.
- Version note: the author README still says34 datasets; final proceedings says65. Use the final proceedings version, and avoid mixing numerical counts from the two versions.
- OpenReview record/PDF encounters browser verification in this tool. The proceedings PDF succeeds and is the preferred public link.
- Transfer scope: graph CO benchmarking, not direct empirical evidence for maintenance generalization.

## Verification route limitations

Semantic Scholar API request for arXiv2509.17870 failed DNS resolution in the sandbox (`curl` exit6). `[S2-API-UNAVAILABLE]`: use primary publisher/proceedings metadata and title/identifier deduplication; do not claim index triangulation or S2 verification. DOI metadata should be distinguished from full-text access. No unsupported paper identity is admitted merely because it appears in an inherited bibliography.
