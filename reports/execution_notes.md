# Execution notes

The first export wrote all images and labels, then Windows returned `WinError 5` when renaming `data/processed.staging` to `data/processed`. A subsequent native PowerShell move succeeded after explicitly checking that both paths were inside this project and that the destination did not exist. No raw files were moved or changed.

The pipeline now retries transient directory rename errors. Its explicit `--resume-export` mode was used to regenerate artifacts and run full validation of the completed files against the same dry-run plan. Nothing was accepted solely because a staging directory existed. A regression test covers transient rename recovery.

Visual inspection of the generated cross-dataset examples then revealed rotated near-identical scenes with inconsistent annotation counts. The geometry-only export was superseded, not treated as the final dataset. Its generated images/quarantine were retained in `data/intermediate/geometry_only/`, with its manifest and JSON validation snapshots in `reports/history/geometry_only/`. The original source assets were not moved.

The final pipeline additionally investigates same-source variants under eight rotation/reflection alignments and quarantines confirmed near-identical image families with inconsistent counts or poorly matching boxes. A new dry run, export, full validation and report were required after that cleaning change. The final `data/processed/` and root reports describe this stricter dataset.

A further semantic review found cloud-region annotations in the Frederikshavn examples and suspect non-vessel regions in some 3B Visual examples. Class semantics could not be verified, so explicit conservative family exclusions were added with reproducible visual evidence. Subpixel annotations were also quarantined with a one-pixel minimum side and floating-point tolerance. Skagen control examples showed vessels and were not blanket-excluded. The orientation-only intermediate was preserved under `data/intermediate/orientation_only/` with its manifest/validation snapshots in `reports/history/orientation_only/`.

The final policy-frozen preparation was dry-run and then exported again. Only `data/processed/` is the final dataset. Intermediate directories are preserved audit artifacts and are excluded from Git with all other generated data.

The last repair overlay review also showed that clipping can leave boxes on incorrect image regions. The final approval gate therefore withholds all clipped images unless their alignment is explicitly approved; no approvals were available. The semantic-and-pixel-only candidate was preserved under `data/intermediate/semantic_and_pixel_only/`, with corresponding JSON/manifest snapshots under `reports/history/semantic_and_pixel_only/`. This conservative gate prevents known unresolved repair cases entering training while preserving all repair evidence.
