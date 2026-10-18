# B1 prefix-use diagnostic (exploratory, DEV only)

Registered after B1 DEV gate failed and before running this diagnostic. This
is a bounded mechanism analysis, **not** a new confirmation or a replacement
for the failed B1 verdict. Scientific B1 TEST stays sealed.

Hypothesis: with the remaining-resource summary and reservation masks present,
the selected-edge prefix rarely changes the trained full policy's actions.
For each of the three registered B1 development replicas at N4/N8, use its
DEV-selected full checkpoint and the 32 opened B1 DEV physical seeds. On every
state visited by the full policy, compare the deterministic full action with
the action from the **same weights** under mask-only decoding. Record the
fraction of states and technician slots whose action changes, nonzero-prefix
stages, and exact legal-mask equality on the forced full action. Separately
run the same-weight prefix-disabled model closed-loop on the same physical
tape and report paired cost/machine and health differences. The already
trained mask-only DEV result remains the matched training contrast; the
same-weight intervention is only mechanistic diagnosis and can suffer
distribution shift.

Engineering smoke takes the first two opened DEV seeds per replica. Full
diagnostic uses all 32 opened DEV seeds, stops after 3,600 s, and stores a new
UTC artifact directory with input/source hashes, exact invocation, per-episode
CSV, per-replica and pooled descriptive results, action-change counts, legality
audit, manifest and script snapshot. `tqdm` covers episodes. No CI or positive
claim is assigned to this reused DEV panel. Preserve failure artifacts.
Tests and smoke, commit/push of diagnostic code precede full diagnostic.
