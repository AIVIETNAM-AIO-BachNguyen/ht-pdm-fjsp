# B2 independent DEV audit preparation

The read-only verifier `scripts/audit_relational_prefix_development.py` was
added while B2 development was running. It does not modify the experiment
source, protocol, registry or active artifact. It checks sealed TEST status,
source/output hashes, exact DEV grids, checkpoint choices, paired physical
populations, raw full/mask-only gains and health guards, matched initial
weights, PPO training tapes and budgets. It independently recomputes the
registered DEV verdict before any confirmation is permitted.

To exercise the verifier before B2 DEV finishes, it was run on the completed
historical B1 DEV artifact (same artifact schema, known negative gate). Fresh
output:
`artifacts/technician_relational_prefix_v2_verifier_historical_fixture_20261018T123249Z`.
It returned PASS and the original FAIL verdict from B1 raw rows, including
184,320 training episodes and their paired physical tapes. This fixture is
only a verifier regression check; it is not B2 outcome evidence. B2 DEV will
be audited when complete, and B2 scientific TEST remains sealed until its
prospective gate passes.
