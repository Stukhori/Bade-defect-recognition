# Phase 11B provenance

The pre-test apparatus is committed before any detector dependency or weight is
acquired. During Colab execution, `phase11b_weight_acquisition.json` is written
to the persistent Google Drive provenance directory before training can start.
It records the apparatus commit, official URL, license, filename, byte size, and
SHA-256 of the exact acquired weight.

After all three runs and validation-only selection, the CLI creates
`provenance/phase11b_selection_receipt.json` in the repository checkout. A human
must inspect, commit, and push that receipt without changing the apparatus
configuration. Only then can the separate `final-test` command pass its Git and
artifact-hash firewall.
