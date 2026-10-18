# B3 selected-pair capacity DEV diagnosis (exploratory)

Registered after the independently audited B3 development gate failed, before
any B3 mechanism intervention. B3 TEST remains sealed. The B3 matched trained
full/mask-only verdict is unchanged: pooled +0.3646% below its +0.5% gate,
with N8 −0.4576% and two adverse N8 replicas.

## Fixed intervention and falsifiable question

For each DEV-selected full checkpoint, evaluate deterministic trajectories on
B3's already opened DEV physical seeds with exactly the same weights under:

1. Full selected-edge attention and selected-pair capacity scalars.
2. Capacity scalars zeroed at the two actor input slots and two WAIT input
   slots, retaining the fitted relational selected-edge message.
3. Both capacity scalars and relational selected-edge message zeroed, using
   the matched architecture's mask-only switch with the full tensors loaded.

The legal reservation mask, current graph, residual-resource pool, decision
order, and physics stay fixed. Compare decisions on the full-policy observed
states plus paired physical outcomes of separate trajectories. The diagnostic
asks whether the explicit selected-pair capacity channel alters decisions and
whether its own fitted weights use it beneficially or adversely on opened
DEV. It cannot establish benefit against a separately trained mask-only actor
or reverse B3's registered DEV failure.

## Panel, metrics, stop and artifact

Use N4/N8, all three B3 training bundles, all 32 opened DEV seeds per scale.
Engineering smoke uses the first two DEV seeds per bundle/scale. No TEST,
new outcome selection, alternate checkpoint selection, or parameter tuning.
Record per-episode extended cost, availability, pending/machine,
failures/machine, physical population hash and invalid assignments for all
three same-weight conditions. On the full states record action/slot changes
for each intervention, nonzero capacity and relational stages, and exact
legal-mask equality. Aggregate by scale and replica; disclose adverse cases.
A zero action-disagreement result rejects the narrow claim that this fitted
channel changes deployed choices; any cost differences remain exploratory.

Hard stop one hour; timeout is INCOMPLETE. Each run uses a fresh UTC timestamped
artifact directory with invocation, source manifest and independent-audit
hashes, script snapshot/hash, CSV/JSON outputs, elapsed time, bindings and
TEST-sealed flag. Use tqdm over DEV episodes. Run focused tests and a small
engineering smoke, then commit/push verified code before the full DEV-only
diagnostic. The B3 learner, physics, protocol, seed registry, and completed DEV
artifact remain frozen.
