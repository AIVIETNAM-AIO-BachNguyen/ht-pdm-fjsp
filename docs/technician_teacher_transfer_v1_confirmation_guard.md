# A1 confirmation guard, prepared before DEV verdict

This isolated branch changes only the confirmation entry check. The running
A1 DEV learner, physics, protocol and panel remain at `a60f716` in their
original worktree. Scientific TEST may open only when the A1 DEV artifact is
COMPLETED, internally audited PASS, retains a sealed TEST panel, passes the
registered DEV screen, and has an independent DEV audit reporting PASS with
`development_screen_pass=true`. The independent report must bind to the exact
DEV path, manifest hash and saved verifier hash. The prospective protocol,
registry, and learner/physics source modules must equal the DEV snapshot.
The runner file differs only by this confirmation gate.

Eight focused tests passed. The protocol, registry and learner/physics source
matched the live DEV snapshot in a read-only check. No A1 DEV result or TEST
was inspected or changed by this preparation.
