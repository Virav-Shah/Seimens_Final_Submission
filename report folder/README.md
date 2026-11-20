# Optional report folder

This folder isolates document and narrative-generation code from modelling, training and scientific plotting. It is not required for the main workflows.

- `repo/ml_training/report_trial.py`: original combined regression figures, Markdown report and result-viewer notebook builder.
- `repo/ml_training/export_data.py`: combined CSV export and narrative README builder.
- `repo/pinn_training/report.py`, `balanced_report.py`, `single_condition_report.py`: original combined PINN figures, prose and notebook builders.
- `repo/mechanism_comparison/report.py`: original figure/table exporter (a plot-only callable version is in the main package); `interpret.py` writes narrative conclusions.
- `replication/report_prototype.py`: prototype Markdown/LaTeX report builder.
- `presentation_revision/`: original mixed plotting/data-and-slide-text helpers. Main `studies/` copies omit the prose output.
- `notebook_cells/`: report-producing cells removed from baseline and boundary-review notebooks. These require the variables from their corresponding completed notebook and are not standalone programs. Execute them only within that same notebook kernel if the historical report is needed.

For a standalone helper, run from the submission root, for example:

```bash
python "report folder/repo/ml_training/report_trial.py"
python "report folder/repo/pinn_training/single_condition_report.py"
```

The relocated standalone helpers contain a short path bootstrap restoring their original imports. They can overwrite figures/result viewers and create reports in the legacy output directories. Run them only in a working copy if preserving the packaged evidence. Historical prose may refer to old duplicate folder names; the current authoritative execution instructions are `docs/REPRODUCING.md`. Report builders were syntax checked, not all executed; report generation is outside this submission's required scientific workflow.
