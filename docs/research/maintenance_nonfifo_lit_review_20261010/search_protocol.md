# Search protocol — narrative/integrative review

Search date: 2026-10-10. User selected a narrative/integrative literature review in this conversation. This is not a registered systematic review or a PRISMA claim.

## Questions fixed before synthesis

1. What prior work addresses online preventive/corrective maintenance with scarce shared repair capacity and identifiable heterogeneous technicians?
2. What is already established about centralized learned dispatch, rollout/planning, and learning-assisted planning?
3. Which representations and evaluation protocols support variable-size decisions and credible zero-shot generalization in adjacent scheduling/combinatorial optimization?
4. Which proposed contributions remain hypotheses rather than demonstrated gaps or benefits?

## Scope and selection

- Direct maintenance/resource-allocation papers plus explicitly identified transfer-method papers from scheduling, neural combinatorial optimization, and planning/imitation learning.
- Emphasis 2020–2026, with earlier foundational methods where necessary. English-language primary papers; proceedings, publishers, institutional author manuscripts, and clearly labeled preprints.
- Include only papers whose identity can be checked on a primary source. Deduplicate by exact title/DOI/arXiv ID, merging preprint and published versions.
- Review inherited references in `docs/marl_maintenance_literature_comparison.md` as candidates, not as verified evidence.
- Distinguish abstract/metadata reading from method/experiment section reading. Do not infer paper weaknesses from an abstract. Report project-transfer limitations separately.
- Find public PDF links when available; a publisher record is not proof of free full-text access. Record crawler failures rather than equating them with paywalls.
- Seek counterevidence: strong capacity/index/planning baselines, limits to size transfer, effects of training/test distributions, and inference-time search/postprocessing.

## Retrieval routes

Web search followed by primary-source inspection (publisher pages, arXiv, PMLR, NeurIPS proceedings, OpenReview, university repositories). Targeted title searches and citation snowballing. No claim of exhaustive database coverage; no invented search-hit or PRISMA counts.

Initial queries:

```text
deep reinforcement learning maintenance planning resource constraints crews rollout policy multi component maintenance
reinforcement learning maintenance optimization limited repairmen heterogeneous technicians stochastic centralized 2024 2025 2026
neural combinatorial optimization benchmarking generalization training inference budget strong baselines 2024 2025 2026
online stochastic combinatorial optimization rollout policies approximate dynamic programming maintenance scheduling
"Neural Combinatorial Optimization" "benchmark" "2025" generalization evaluation
"Expert Iteration" Anthony Tian Barber 2017
"neural combinatorial optimization" "evaluation" "benchmark" "2024" classical
"ML4CO-Bench-101" paper
```

## Intended artifacts

- `literature_review_vi.md`: Vietnamese synthesis, project comparison, bounded gap statements, research implications, limitations, AI disclosure.
- Phase-2 source notes: verified identities, links, read scope, supported findings and access limitations.
- `paper_catalog.csv`: paper identifiers and links in a reusable table.
- `references.bib`: only verified selected references; preprints labeled as such.

The experiment workflow (tests/smoke/commit/push/lab handoff) applies when a code experiment is implemented. This review does not launch or change experiments.
