# V4 mechanism diagnostic plan (exploratory)

Written while the prelocked FULL was training, before DEV selection or sealed
TEST outcomes were inspected. This plan does not change the primary verdict
and does not add a new confirmation endpoint.

Question: Does the temporal age2 gate actually alter the selected policy's
choices, and on what request/worker states? After FULL and independent audit,
use the DEV-selected temporal checkpoint and the already opened physical
TEST panel. For each case/seed, replay the frozen policy trajectory. At each
current state, score all 21 actions twice with the same trained weights:
normal temporal gate and a read-only intervention setting its actor variant
to current_queue. Keep the legal mask exactly the same. Record the number of
legal-action argmax disagreements, age2 competitor count, queued PM/CM,
worker availability, opportunity scale, and the chosen action; save a CSV,
summary and checkpoint/source hashes in a new timestamped directory.
Stop at all 4 cases ×64 seeds ×48 decisions (12,288 common states).
No new training, checkpoint selection, or policy-cost claim from the
same-state intervention. If the primary is negative and the gate almost
never changes an action, that is a mechanism lead; if the gate changes many
actions but cost remains poor, analyze timing/credit or price formulation.
Neither case proves causality in full closed-loop trajectories.

Structural QA added before DEV/TEST opening: on the same 12,288 visited
states, also relabel machines by new-to-old order [2,0,3,1] and technicians
by [1,0]. Remap observation edge rows/columns, busy assigned-machine index,
matching catalogue and mask. Check all structured actor logits after inverse
action remapping within floating tolerance 1e-5, and deterministic legal
argmax equality except exact ties. This is a state-level equivariance check,
not a new performance or N/K generalization claim. Report maximum error and
mismatch count; failure requires diagnosing an implementation bug.
