# Curation and provenance

The original Acaddrug workspace was preserved. `Final Submission` is a separate local repository; nothing has been uploaded or published.

## Selection

The presentation's completed scientific workflows determine inclusion. Numerical search results, selected-design tables, training/validation/test partitions, predictions, saved weights, configuration and audit outputs are retained. They are evidence and reproducibility inputs, not disposable logs. Scientific figures are retained as PNG previews and SVG vector exports where originally available.

The early and balanced PINN runs are intentional inclusions because the presentation discusses their failure modes. The abandoned `Failed_improvement/` external-data acquisition/inverse-affinity project is outside the final presentation's demonstrated work and excluded. Temporary notebook-construction scripts are excluded where the finished notebook already contains the actual implementation. V2/V4 notebooks are superseded by the retained baseline/V3/V5 sequence. Display-only ML/PINN notebooks duplicating existing plots are mostly excluded; one final PINN viewer remains to support its integrity verifier.

Logs, caches, environment binaries, ZIP bundles, generated PDF/Word/HTML reports, slide-writing drafts and duplicate code directories are excluded. Original paper/supplement and the submitted final presentation are retained as references. `selection_inventory.csv` lists original paths and dispositions; the virtual environment is represented as one excluded directory.

## Packaging edits

- Consolidated identical `ml_small_trial`/`ml_training` and `pinn_small_trial`/`pinn_training` code copies into the `*_training` directories.
- Moved numerical/plotting scripts from original output directories to `studies/`; changed their output-path resolution to continue using `output/`.
- Moved report builders to `report folder/`; restored their import/path context with a bootstrap.
- Extracted plot-only ML, final PINN and mechanism entry points. Kept full original mixed helpers in the report folder.
- Removed report cells from two modelling notebooks and cleared notebook outputs/execution counts. Preserved computational cells and equations. Updated the final PINN viewer to display structured results without requiring a generated report.
- Replaced ML fair-comparison log parsing with a structured `validation_wall_seconds` field. Future trial runs now persist that field; historic fair-comparison results already include it.
- Removed superseded V4 file-copy references from the supporting presentation plot helper.
- Preserved the main reusable ODE model byte-for-byte, so existing training-source hash checks remain valid.
- Added installation/run instructions, evidence caveats, code catalogue, inventories, checksums and local Git setup.

`source_manifest.json` records selected original source hashes and destinations at initial copy time. `checksums.sha256` covers final delivered files (excluding Git internals and the checksum file itself). Generated figures, tables and verification JSON may be refreshed by the validation steps; source and final hashes therefore need not match. The manifest is provenance, not a claim that every file was copied without edits. Newly extracted plot entry points and documentation are described above.

The repository is roughly 300 MB, mainly retained training arrays and predictions. Removing them would reduce size but prevent immediate checkpoint/data verification. Checkpoints and saved data are deliberately included.
