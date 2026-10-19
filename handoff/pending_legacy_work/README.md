# Pending legacy work: preserved, not integrated

These files came from the older primary Mac working tree recorded in
`../import_provenance.json`. The handoff itself starts from newer verified
maintenance-request source at commit 8855c7e. Nothing here replaces active code.

- `src/` and `tests/`: uncommitted passive-technician scale-transfer module/test.
- `working_tree_changes.patch`: exact original tracked-file diff (CLI entry and
  train-on-in_distribution-only change for the long comparison runner).
- `alternate_docs/`: local variants of documents already present on the newer
  branch; kept separately instead of overwriting newer files.
- `provisional_research_proposal_revised.tex`: historical editable proposal.

These source changes were NOT tested or adopted in this handoff. They address
an older model, not the current persistent-request framework. Review their
protocol, current dependencies and applicability before cherry-picking any
idea. Do not apply the patch blindly to the handoff base. Local original files
were neither deleted nor reset. Compiled LaTeX files and editor/cache output
were omitted; editable .tex/.excalidraw sources were retained.
