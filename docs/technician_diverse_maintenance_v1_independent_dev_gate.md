# C1 independent DEV gate before scientific TEST

The C1 confirmation-only runner now requires an independent development audit
that reports PASS and `development_screen_pass=true`, confirms scientific TEST
is sealed, and binds its report to the exact DEV artifact path, manifest hash,
and saved verifier hash. The existing runner also checks the DEV internal gate,
prospective protocol, registry, source snapshot, and frozen DEV-selected
FCFS/risk rule. No learner, physics, seed, budget, or threshold changed.

Three focused confirmation tests passed after this guard was added. The live
C1 DEV run remains in progress; this guard is only a prospective condition
for a later confirmation run and is not a C1 scientific outcome.
