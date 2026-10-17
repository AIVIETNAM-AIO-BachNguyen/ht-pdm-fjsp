# Shared technician service-duration follow-up v2 — exploratory protocol

Prospectively written 2026-10-18 after v1 test results were opened. This is an
**exploratory** structural follow-up and cannot rescue or relabel v1 H1/H2.
The v1 `scarce_overlap` condition had a one-tick common technician; its
capacity-aware timing rule made zero distinct proposals in 64 test seeds.
V2 changes only fixed service matrices so busy compatible resources can span
decision epochs. No v2 outcomes were seen before this protocol.

## Question and fixed design

Does a persistent, earlier preventive request change actual assignment timing
or cost when technicians are busy for multiple ticks? Keep v1 semantics,
failure law, economic costs, H48+tail12, terminal liability, all 3 request
policies and 4 dispatchers, same component metrics and trace replay.

Two fixed N4/K2 matrices, restoration .75: `overlap_service2` =
`[[2,0],[2,3],[2,4],[2,5]]`; `balanced_service2` =
`[[2,0],[0,2],[2,4],[4,2]]`. Zero means incompatible. The first has a
common, faster shared technician; the second distributes dedicated work and
has cross-skilled machines. These are purpose-built structural probes, not a
random sample of plants. The original v1 matrices remain archived in the v1
artifact. No setting is selected by v2 development cost.

Primary metric: same total H+tail+terminal system cost. Secondary: H cost,
availability, failures, failed waiting, pending H/end, utilization,
request/assignment divergence from threshold under each dispatcher, and
decision replay feasibility. E1: capacity_aware/lookahead lowers cost versus
both capacity_aware/FCFS and threshold/lookahead; E2: the timing-allocation
interaction is negative; E3: capacity_aware creates proposals and changes
assignments before H in at least one condition. All are exploratory; report
paired differences, relative changes, 95% bootstrap CI and exact exposure
counts, with no pass label or multiplicity-adjusted causal confirmation.

No training/checkpoints. Smoke raw seeds 93000000–93000001, development
93000100–93000115, test 93000200–93000263, analysis seed 93000400. These are
disjoint from v1 and readable historical registries. Smoke =48 episodes;
full =1,920 episodes (16+64 seeds ×2 conditions ×12 arms), 115,200 physical
ticks. Stop after the fixed panel or 7,200 seconds. Development is only for
integrity/runtime. Test physical tapes are common across arms within each
condition/seed, and never used to change rules. Bootstrap resamples the 64
paired seed vectors 10,000 times. Distinguish zero effect, imprecise effect
and policy dominance; do not call a zero-invalid-assignment audit optimality.

Fresh timestamped output must include RUNNING/COMPLETED/FAILED manifest,
source/protocol/registry snapshots and hashes, configs, partial/final episodes,
test decisions, independent replay audit, summary and log. Refuse overwrite.
Tests and smoke must pass, then commit/push verified code before full on this
Ubuntu lab. Results remain explicitly exploratory because this redesign was
motivated by the opened v1 panel.
