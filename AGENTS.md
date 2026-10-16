# Experiment workflow

For every code experiment in this repository:

1. Plan the hypothesis, metrics, seed splits, stopping rule, and artifact schema.
2. Implement and run tests plus a small local macOS smoke experiment first.
3. Commit and push the verified code to GitHub.
4. After pushing, show the exact compound command for the user to run in the
   Ubuntu lab terminal at `bachng@100.111.83.52`; do not SSH to the lab or
   launch the full run on the user's behalf.
5. Show progress with `tqdm` during training and multi-seed evaluation.
6. Store each run in a new timestamped artifact directory; never overwrite a
   prior full run silently.
7. Show the exact `rsync` command for the user to run on the local Mac after
   the full run finishes; do not start the transfer on the user's behalf. The
   command must preserve partial transfers and display progress.

The canonical commands and handoff contract are documented in
`docs/experiment_workflow.md`.

## Active user override (2026-10-12)

The workspace is now the Ubuntu lab itself. The user explicitly authorizes
running smoke and full experiments directly here, inspecting results here,
and continuing research iterations. This supersedes the Mac smoke / manual
lab handoff restrictions above for this research task. Keep prospective
protocols, tests, commit/push gates, tqdm, and fresh timestamped artifacts.
Do not SSH or initiate an artifact transfer; provide rsync when results exist.

## Active user override (2026-10-16)

For the current technician-coordination research, run FULL experiments directly
on this Ubuntu lab without a separate smoke experiment. The user explicitly
removed the smoke-before-full prerequisite. Keep necessary unit/integration
correctness checks, prospective protocols, commit/push, tqdm and fresh artifacts.
Focus on N4/K2 until its registered quality, health and mechanism criteria pass;
larger-size/generalization experiments come afterward. Scientific confirmation
test seeds remain sealed until the registered development gate passes.

## Research discipline override (2026-10-16)

Keep each new research iteration tied to one explicit, falsifiable mechanism
hypothesis. Encoder changes, learning-rate tuning or added attention are not a
scientific explanation by themselves. Standard PPO from random initialization is
acceptable; contributions may be policy representation or coordination. Match
information/capacity/budgets, lock rejection criteria and mechanism measurements
before outcomes, and reject unsupported designs rather than stacking components
until a baseline is beaten. The residual-capacity proposal gets one locked full
comparison; failed gates do not authorize another capacity variant or test reuse.

## Active user tuning clarification (2026-10-16)

The user explicitly permits necessary tuning: a failed configuration may reflect
implementation or optimization limitations rather than a false broad hypothesis.
This supersedes the blanket ban on another capacity/configuration variant above.
Preserve every negative run and its original verdict. Investigate specific causes,
then register a bounded controlled follow-up before new learning outcomes; match
information/capacity/training budgets across methods and retain all health and
scientific-test safeguards. Tuning is permitted, not evidence of a contribution.
Do not launch an open-ended search or reinterpret an already failed protocol.
