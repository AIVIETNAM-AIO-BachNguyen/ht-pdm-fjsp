# B3 confirmation guard, prepared before DEV completion

This isolated branch extends only the confirmation entry gate and independent
verifier. The live B3 DEV learner, physics, protocol and seed registry remain
at `a99ab1c` in their original worktree. Confirmation now requires:

1. A completed B3 development manifest with internal audit PASS, unopened
   TEST, and the registered development screen PASS.
2. An independent development audit with verdict `DEV_GATE_PASS`, sealed TEST,
   a verifier hash and exact DEV manifest binding.
3. Identical prospective protocol, registry, and every development source
   snapshot module except the runner file that gains this confirmation gate.

The independent verifier checks feasible actions, resource summaries, the
zeroed mask-only selected-edge and capacity inputs, and the exact physical
selected-pair capacity inputs on both full and mask-only policy trajectories.
It recomputes cost, health and inference from raw rows and replays physical
transitions. It does not alter learning, budget, gate thresholds or physics.

Before commit: nine focused tests PASS. The confirmation source guard matches
the live B3 DEV snapshot without inspecting DEV quality. Replaying the
engineering smoke through the strengthened verifier PASSed: 384 full and 384
mask-only states, 2,304 decoder stages, 2,400 physical ticks, and zero invalid
assignments. The smoke does not establish B3 scientific performance. B3
scientific TEST remains sealed until both DEV gates pass.
