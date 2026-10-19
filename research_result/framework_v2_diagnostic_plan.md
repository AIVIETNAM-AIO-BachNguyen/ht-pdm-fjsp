# V2 opened-panel action diagnostic (exploratory)

After the registered v2 TEST and independent audit, compare the frozen v1 PPO
and selected mixture PPO on the same already-opened 64 v2 physical seeds in all
four cases. No new model, environment, reward, seed panel, or checkpoint choice.
Use the fixed action trace fields from the v1 diagnostic: legal choices,
queued PM/CM, service starts, WAIT with a legal service opportunity, chosen
durations, slow eligible edges, and agreement with risk matching. Aggregate
H+tail cost and H health. Stop at 4 × 64 × 48 decisions per model. This is
mechanism description only; trajectory divergence prevents a causal claim from
per-tick action frequencies.

Write each model trace and summary into a new timestamped
`research_result/framework_v2_diagnostic_*` directory, recording the exact
checkpoint digest and opened-panel label. Keep both positive and negative
changes. The diagnostic cannot revise the registered v2 verdict.

## Exploratory amendment after first trace (2026-10-20)

The first trace shows mixture PPO removes empty WAIT with a legal opportunity,
yet starts fewer jobs on reciprocal cases. The original WAIT metric misses a
partial matching that leaves another compatible technician/request idle. Add
`extendable_action`: 1 when the selected matching is a strict subset of some
other legal matching on the same current state. Re-run both fixed policies on
the same opened panel in fresh directories; keep the first traces. This is a
post hoc descriptive extension and cannot revise the registered v2 result.
