# Framework v1 action diagnostic (exploratory)

Question: do the opened-test losses arise from waiting despite a legal service
opportunity, choosing longer service edges, or changing the PM/CM mix? This is
post hoc mechanism analysis, not a second confirmation test. No model, setting,
seed, cost, or checkpoint is changed. Use the single development-selected v1
checkpoint on all 64 already-opened sealed physical seeds in each of the four
registered cases. Stop after 4 × 64 × 48 policy decisions; do not tune from a
partial trace.

Per decision record case/seed/tick, legal matching count, queued PM/CM,
selected matching size, WAIT when a service action was legal, PM/CM starts,
selected mean duration, whether a chosen edge is slower than another currently
free eligible worker for the same machine, and agreement with risk matching on
that same state. Aggregate these with episode cost, H availability, H failures,
CM waiting and pending. Interpret slow edges only descriptively: a slower edge
may reserve a scarce fast worker for another machine. The action mask supplies
feasibility, not a learned feasibility claim.

Each analysis uses a fresh timestamped `research_result/framework_v1_diagnostic_*`
directory containing `trace.csv`, `summary.json`, and `manifest.json` with run,
checkpoint, source hash, counts, and exploratory label. Do not overwrite raw
campaign artifacts. No new statistical superiority claim is made from this
opened panel.
