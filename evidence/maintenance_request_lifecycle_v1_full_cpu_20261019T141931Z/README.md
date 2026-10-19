# Compact capsule of a completed run

Run: `maintenance_request_lifecycle_v1_full_cpu_20261019T141931Z`.
Source: `8855c7e3f79424b1d828b2e66ef6fcf941c3fcd9`.

Every copied raw file is byte-identical to the completed run. This capsule
includes 3,456 episode rows, configuration, summary, audit, manifest, protocol,
seed registry and source snapshots. It is not a complete run directory.

Intentionally omitted:

- `decisions.jsonl`: 401,302,792 bytes; needed for independent scalar replay.
- `episodes.partial.csv`: duplicate of the final CSV; 909,225 bytes.

The original manifest is unchanged and still describes the FULL run. Its
artifact hashes include these omitted files; do not interpret their absence
here as corrupt lab data. Hashes/sizes of omitted files are recorded in
`handoff/import_provenance.json`. The capsule's final hashes are recorded in
`handoff/package_manifest.json`. `scripts/verify_lab_handoff.py` checks the
included bytes against both package hashes and the original manifest.

Full raw location on lab:
`~/ht-pdm-fjsp/artifacts/maintenance_request_lifecycle_v1_full_cpu_20261019T141931Z/`.
The original Mac copy is under `lab_results/` with the same run name.
Never edit raw data to satisfy an audit, overwrite the completed run, or label
a summary-only capsule as a complete replay.
