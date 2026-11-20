# Code catalogue

Every submitted Python file is listed below. Files in `report folder/` are optional reporting helpers, not part of the scientific execution path. Notebook cells are described separately.

## `replication/calibration_sensitivity.py`

Illustrative landmark-reading uncertainty; ranges are not statistical CIs.

Top-level functions/classes: `main`.

## `replication/model.py`

Mechanistic equations, parameters, state transitions and numerical integration.

Top-level functions/classes: `Parameters`, `densities`, `initial`, `rhs`, `simulate`, `endpoints`.

## `replication/prototype.py`

Calibrate shared effective physiology, optimise candidate rates and assess sensitivity.

Top-level functions/classes: `save_csv`, `evaluate`, `calibrate`, `scenario_params`, `trajectory`, `baseline`, `search`, `audit`, `plot`, `main`.

## `replication/run.py`

Run reconstructed paper workflow. Every output is explicitly non-exact.

Top-level functions/classes: `write_csv`, `trajectories`, `sweep`, `sweeps`, `optimize`, `main`.

## `replication/test_model.py`

Drug/receptor conservation, symmetry, limits, Emax weighting and solver checks.

Top-level functions/classes: `ModelTests`.

## `replication/validate_prototype.py`

Cross-solver and limiting-case validation of generated prototype candidates.

Executes sequentially as a script; inspect its input/output paths before running.

## `repo/mechanism_comparison/__init__.py`

Controlled comparisons of processes included in the V5 ADC model.

Executes sequentially as a script; inspect its input/output paths before running.

## `repo/mechanism_comparison/model.py`

Mechanistic equations, parameters, state transitions and numerical integration.

Top-level functions/classes: `Scenario`, `density`, `make_rhs`, `KillModel`, `check_inputs`, `integrate`, `assess`.

## `repo/mechanism_comparison/plot_results.py`

Recreate scientific figures and numerical comparison outputs from saved results.

Top-level functions/classes: `save`, `make_plots`.

## `repo/mechanism_comparison/study.py`

Seeded mechanism/policy optimisation and fixed-design comparisons.

Top-level functions/classes: `evaluate`, `margin`, `front`, `search`, `run_all`.

## `repo/ml_training/fair_comparison.py`

Compare equal total ODE budgets including regression model-selection samples.

Top-level functions/classes: `main`.

## `repo/ml_training/plot_results.py`

Recreate scientific figures and numerical comparison outputs from saved results.

Top-level functions/classes: `save`, `endpoints`, `box`, `arrow`.

## `repo/ml_training/run_trial.py`

Regression model selection, active sampling, direct NSGA-II and held-out evaluation.

Top-level functions/classes: `convert`, `labels`, `ev`, `evaluate`, `lhs`, `arrays`, `feasible`, `front`, `metrics`, `Regressor`, `choose`, `poolgen`, `verified`, `stats`, `main`.

## `repo/pinn_training/balanced_audit.py`

Independent checks of trajectory dynamics, beyond enforced accounting.

Top-level functions/classes: `main`.

## `repo/pinn_training/balanced_candidate_audit.py`

One-shot ODE audit of proposed candidates; not an optimisation benchmark.

Top-level functions/classes: `main`.

## `repo/pinn_training/balanced_model.py`

Nonnegative interval network with inventory accounting and exact initial conditions.

Top-level functions/classes: `BalancedNet`.

## `repo/pinn_training/balanced_paired_audit.py`

Compare both predictors on the same union of ODE-audited candidate designs.

Top-level functions/classes: `main`.

## `repo/pinn_training/balanced_trial.py`

Balanced physics-loss trial and matched supervised-network comparison.

Top-level functions/classes: `collocation`, `norms`, `fit`, `check`, `main`.

## `repo/pinn_training/check_artifacts.py`

Post-run reproducibility, shape, derivative and checkpoint integrity checks.

Top-level functions/classes: `main`.

## `repo/pinn_training/data.py`

ODE trajectory generation, partitioning, input configuration and equation checks.

Top-level functions/classes: `trajectory`, `one`, `coordinates`, `generate`, `preflight`.

## `repo/pinn_training/physics.py`

Differentiable PyTorch implementation of the ODE and inventory constraints.

Top-level functions/classes: `tensor`, `Physics`.

## `repo/pinn_training/plot_single_condition.py`

Recreate the final PINN/GP accuracy, trajectory and training figures without a report.

Top-level functions/classes: `save`, `main`.

## `repo/pinn_training/single_condition_predict.py`

Load the two saved checkpoints and predict cell killing from four log rates.

Top-level functions/classes: `predict`.

## `repo/pinn_training/single_condition_trial.py`

Final reference-condition PINN and Gaussian-process training/test benchmark.

Top-level functions/classes: `write`, `solve`, `dataset`, `prediction`, `fit_pinn`, `GP`, `main`.

## `repo/pinn_training/single_condition_verify.py`

Check checkpoint prediction agreement, partitions, provenance and result viewer.

Top-level functions/classes: `main`.

## `repo/pinn_training/train.py`

Initial trajectory PINN, residual loss, checkpoint selection and feasibility gate.

Top-level functions/classes: `endpoints`, `aggregate`, `accuracy`, `TrajectoryNet`, `sample_collocation`, `physical_metrics`, `fit`, `main`.

## `report folder/notebook_cells/Bispecific_ADC_Binding_Search_Limits_cell_11.py`

Optional report/notebook-generation helper. Study-specific calculation, plotting or validation helper; see named functions below.

Top-level functions/classes: `render_conclusions`.

## `report folder/notebook_cells/Bispecific_ADC_Modelling_cell_26.py`

Optional report/notebook-generation helper. Study-specific calculation, plotting or validation helper; see named functions below.

Top-level functions/classes: `num`, `sci`, `table_row`, `tex_escape`.

## `report folder/presentation_revision/clear_internalisation.py`

Optional report/notebook-generation helper. Study-specific calculation, plotting or validation helper; see named functions below.

Top-level functions/classes: `save`.

## `report folder/presentation_revision/prepare_numbers.py`

Optional report/notebook-generation helper. Study-specific calculation, plotting or validation helper; see named functions below.

Executes sequentially as a script; inspect its input/output paths before running.

## `report folder/replication/report_prototype.py`

Optional report/notebook-generation helper. Study-specific calculation, plotting or validation helper; see named functions below.

Top-level functions/classes: `main`.

## `report folder/repo/mechanism_comparison/interpret.py`

Optional report/notebook-generation helper. Study-specific calculation, plotting or validation helper; see named functions below.

Top-level functions/classes: `write_observations`.

## `report folder/repo/mechanism_comparison/report.py`

Optional report/notebook-generation helper. Study-specific calculation, plotting or validation helper; see named functions below.

Top-level functions/classes: `save`, `make_report`.

## `report folder/repo/ml_training/export_data.py`

Optional report/notebook-generation helper. Study-specific calculation, plotting or validation helper; see named functions below.

Executes sequentially as a script; inspect its input/output paths before running.

## `report folder/repo/ml_training/report_trial.py`

Optional report/notebook-generation helper. Study-specific calculation, plotting or validation helper; see named functions below.

Top-level functions/classes: `save`, `endpoints`, `box`, `arrow`.

## `report folder/repo/pinn_training/balanced_report.py`

Optional report/notebook-generation helper. Study-specific calculation, plotting or validation helper; see named functions below.

Top-level functions/classes: `save`, `main`.

## `report folder/repo/pinn_training/report.py`

Optional report/notebook-generation helper. Study-specific calculation, plotting or validation helper; see named functions below.

Top-level functions/classes: `save`, `main`.

## `report folder/repo/pinn_training/single_condition_report.py`

Optional report/notebook-generation helper. Study-specific calculation, plotting or validation helper; see named functions below.

Top-level functions/classes: `save`, `main`.

## `studies/adc_timecourses/model.py`

Mechanistic equations, parameters, state transitions and numerical integration.

Top-level functions/classes: `Scenario`, `density`, `make_rhs`, `KillModel`, `check_inputs`, `integrate`, `assess`.

## `studies/adc_timecourses/plot.py`

Study-specific calculation, plotting or validation helper; see named functions below.

Executes sequentially as a script; inspect its input/output paths before running.

## `studies/adc_timecourses/plot_low.py`

Study-specific calculation, plotting or validation helper; see named functions below.

Executes sequentially as a script; inspect its input/output paths before running.

## `studies/adc_timecourses/run.py`

Study-specific calculation, plotting or validation helper; see named functions below.

Top-level functions/classes: `run`.

## `studies/adc_timecourses/run_low.py`

Study-specific calculation, plotting or validation helper; see named functions below.

Top-level functions/classes: `run`.

## `studies/complex_internalisation/plot.py`

Study-specific calculation, plotting or validation helper; see named functions below.

Executes sequentially as a script; inspect its input/output paths before running.

## `studies/complex_internalisation/run.py`

Study-specific calculation, plotting or validation helper; see named functions below.

Top-level functions/classes: `run`.

## `studies/prepare_plots.py`

Presentation figures drawn from saved notebook results; no new model fitting.

Top-level functions/classes: `save`.

## `studies/presentation_revision/clear_internalisation.py`

Study-specific calculation, plotting or validation helper; see named functions below.

Top-level functions/classes: `save`.

## `studies/presentation_revision/internalisation_all_groups.py`

Study-specific calculation, plotting or validation helper; see named functions below.

Executes sequentially as a script; inspect its input/output paths before running.

## `studies/presentation_revision/make_plots.py`

Study-specific calculation, plotting or validation helper; see named functions below.

Top-level functions/classes: `save`, `compute`, `main`.

## `studies/presentation_revision/prepare_numbers.py`

Study-specific calculation, plotting or validation helper; see named functions below.

Executes sequentially as a script; inspect its input/output paths before running.

## Notebooks

| Notebook | Purpose |
|---|---|
| Bispecific_ADC_Modelling.ipynb | Calibrated baseline, density/binding sweeps and optimisation. Report cell removed. |
| Bispecific_ADC_Modelling_v3.ipynb | Intermediate biological extensions, validation, sensitivities and subgroup results. |
| Bispecific_ADC_Modelling_v5.ipynb | Final robust multiobjective optimisation and numerical output tables. |
| Bispecific_ADC_Binding_Search_Limits.ipynb | Search-domain sensitivity and convergence; report cell removed. |
| Bispecific_ADC_PINN_Single_Condition.ipynb | Display saved final experiment without retraining. |
