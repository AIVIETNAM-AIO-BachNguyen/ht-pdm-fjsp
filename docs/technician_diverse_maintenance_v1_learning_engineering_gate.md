# C1 learner engineering gate before scientific development

The C1 protocol, four physical configurations, fixed request adapter, seed
panels, reference selection and DEV/TEST gates were registered before any
learner outcome. The initial adapter-only smoke and its six tests passed at
commit `36b4d92`. The new learner reuses the B2 graph architecture and a
single N4 teacher bank per student bundle, then fits matched full,
mask-only, no-reservation and scratch-full arms on each registered
configuration. Scientific confirmation remains sealed.

Nine targeted tests now pass, including the request/physics contracts,
paired physical streams, matched full/mask stage masks on both N8 families,
fresh seed audit and the absent scientific TEST list in development.
`compileall` and `git diff --check` pass.

An initial learner smoke at
`artifacts/technician_diverse_maintenance_v1_smoke_20261018T122333Z`
trained all eight small models but failed while serializing a `numpy.bool_`
health guard. That artifact remains FAILED; the fault was the reporter, not
the physical or learning transition. The corrected smoke at
`artifacts/technician_diverse_maintenance_v1_smoke_20261018T122616Z`
COMPLETED with built-in audit PASS: eight 384-decision fits, 48 DEV episodes,
48 engineering TEST episodes, zero invalid assignments, source/output hashes
and frozen engineering checkpoint/reference choices before opening the smoke
TEST. Its DEV gate failed after only 384 PPO decisions and one bundle; this
engineering-scale result is not a scientific verdict. C1 development FULL
must use 491,520 decisions and fresh DEV seeds; its scientific TEST stays
sealed until the registered DEV gate passes.
