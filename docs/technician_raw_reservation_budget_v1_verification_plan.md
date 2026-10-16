# Independent bounded-budget follow-up verification

Registered while the FULL run is live, before terminal outcomes or verification.
Frozen learning commit:4ade42381681e073efc8a2c7dc9efb43640a3809. No updates,
checkpoint reselection beyond the original earliest-minimum rule, new states,
confirmation seeds, timing rescue or changes to the learning source/artifact.

Extend the raw-state v1 independent verifier: pure dictionary dynamics/Python RNG
reconstruction, separate NumPy conditional MLP, baseline logits/decoders, frozen
candidate weights with actual execution conditioning, same float32 tolerance
5e-6+1e-6*abs(chosen logit), exact support and matching rules. Reuse its tested
physics arithmetic, not the simulator for primitive replay. All source and output
SHA-256 bindings and committed protocol/registry must match.

Check all24 models independently initialized on DEVtrain280000000..2, equal
fresh initial critic tensors, global clipping,983040 decisions/model,20480
training episodes/model,15360 Adam steps/model; totals23592960 decisions,
491520 episodes,368640 Adam steps and61440 paired physical tapes. All320 training
updates/model and6912 DEV rows across9 checkpoints and32 seeds. Long selection
uses all nine checkpoints; short selection uses only steps<=491520. Independently
bind and check both checkpoint choice grids/metadata. Independently recompute
short summary's means from its recorded DEV rows, both prelocked budget contrasts,
13 primary endpoints, invalid assignments and432 raw timing statistics/bindings.

Reconstruct all selected/frozen/deadline episodes:1088 episodes,65280 primitive
transitions,52224 decisions, full60tick costs and48tick health cutoff, brokers,
canonical greedy policies and fixed arbitration. Short summary is arithmetic from
recorded checkpoint rows, not an additional independent short physical replay.
Explicitly disclose that scope; do not assert all historical training gradients
were independently recomputed because their actions/gradients were not retained.

Primary pass =original13+engineering gates; overall eligibility additionally
requires interaction gain>=.005 and candidate long cost gain>=.005. Verify both
fields separately, never replace the old raw-state v1 negative verdict. A valid
negative study may receive verification PASS. No DEV CI/scientific conclusion.

Fresh timestamped verification directory, fail-fast on discrepancies,15minute
hardstop, preserve failed directory, source manifest unchanged. Retain
manifest.json,verification.json,verifier.py and this protocol with hashes.
Necessary unit tests, commit/push before execution. No scientific tests or
larger N/K unlocked by the verifier unless independently checked eligibility passes.
