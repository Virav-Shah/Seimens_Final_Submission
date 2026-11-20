# Reproduction guide

Run shell commands from the submission root unless a command explicitly changes directory. Install `requirements.txt` first. Set `MPLBACKEND=Agg`, `OPENBLAS_NUM_THREADS=1`, and `OMP_NUM_THREADS=1` for noninteractive execution with controlled BLAS threading. Scripts usually use four ODE workers; change their explicit worker settings for smaller machines.

## 1. Reconstruction and calibration

```bash
python -m unittest discover -s replication -v
python replication/run.py --grid 5 --nfe 240 --seed 8407
python replication/prototype.py --nfe 576 --seeds 8407 2026
python replication/validate_prototype.py
python replication/calibration_sensitivity.py
```

`run.py` is the initial reconstruction; `prototype.py` adds calibration. The presentation's fuller calibrated baseline is in `repo/Bispecific_ADC_Modelling.ipynb`. Its report-producing final cell has been moved into `report folder/notebook_cells/`; numerical and plotting cells remain executable. Its exported plots use `repo/report/figures/` for compatibility. This folder contains figures, not a report builder.

## 2. Notebook modelling, V3 and V5

```bash
cd repo
jupyter lab Bispecific_ADC_Modelling_v5.ipynb
```

Use Restart Kernel and Run All. Set the notebook working directory to `repo/` so its relative result paths resolve. V3 documents intermediate extensions and the small healthy-subgroup problem. V5 generates the final three-condition Pareto search, numerical checks and requirement-filtered candidate set. The boundary-review notebook examines the limitations of rate bounds and search budgets; its prose-report cell is separated. V2 and V4 were superseded and are omitted.

Notebook code defines its own models; the reusable implementation in `mechanism_comparison/model.py` supplies the later studies and ML data. The early Emax readout and the later integrated kill-rate readout are different endpoints and should not be compared as if they were identical experiments.

## 3. Mechanism comparisons and final presentation plots

```bash
(cd repo && python -m mechanism_comparison.study)
(cd repo && python -m mechanism_comparison.plot_results)
python studies/presentation_revision/make_plots.py
python studies/presentation_revision/prepare_numbers.py
python studies/presentation_revision/clear_internalisation.py
python studies/presentation_revision/internalisation_all_groups.py
```

The study searches one mechanism at a time, then the plotting module creates figures, comparison tables and policy results. Existing completed search JSON files can be reused. The final presentation plot scripts read `output/presentation_revision/data/` and `repo/results_mechanism_comparison/`; a numerical sweep is generated only if absent. `prepare_numbers.py` records provenance including the included final presentation's hash.

Additional studies:

```bash
python studies/complex_internalisation/run.py
python studies/complex_internalisation/plot.py
python studies/adc_timecourses/run.py
python studies/adc_timecourses/plot.py
python studies/adc_timecourses/run_low.py
python studies/adc_timecourses/plot_low.py
python studies/prepare_plots.py
```

These are supporting comparisons, not new experimental validation. The ADC time-course model has its own explicit monospecific comparator. Final-presentation slide numbers refer to the supplied PDF, not historical filenames such as `slide23_*`.

## 4. Regression-assisted binding optimisation

```bash
python repo/ml_training/run_trial.py
python repo/ml_training/fair_comparison.py
python repo/ml_training/plot_results.py
```

The first command creates Latin-hypercube ODE samples, selects GP or ExtraTrees using validation data, performs assisted and direct searches, and evaluates untouched test designs. The second charges model-selection simulations to the ML budget and gives both methods 224 simulations per seed. The third only regenerates scientific figures. Validation wall time is now stored in structured results rather than parsed from a discarded log; saved historical results already contain this field. The final comparison has three seeds, but a shared held-out test set, not three independent test sets.

## 5. PINN progression

Initial multi-condition experiment:

```bash
python repo/pinn_training/data.py
python repo/pinn_training/train.py --seconds 360
python repo/pinn_training/check_artifacts.py
```

Balanced multi-condition follow-up (uses the first experiment's partitions):

```bash
python repo/pinn_training/balanced_trial.py
python repo/pinn_training/balanced_audit.py
python repo/pinn_training/balanced_paired_audit.py
python repo/pinn_training/balanced_candidate_audit.py
```

Final one-condition prediction comparison used on PDF pages 28–29:

```bash
python repo/pinn_training/single_condition_trial.py
python repo/pinn_training/plot_single_condition.py
python repo/pinn_training/single_condition_verify.py
```

To inspect the submitted models without training, run only `single_condition_verify.py`, `single_condition_predict.py` or the saved-result viewer notebook. The final training uses 512 training, 64 validation and 128 held-out test binding-rate combinations; the PINN has 97 states at 129 times. The GP uses endpoint labels. Saved `.pt` and `.joblib` files are model checkpoints; `.npz` files hold numerical arrays and scaling. Retain these files to verify the submitted result without repeating training.

## Units and interfaces

- Design-vector order: `(log10 kon1, log10 koff1, log10 kon2, log10 koff2)`.
- Association input rates: M^-1 s^-1; dissociation input rates: s^-1; KD = koff/kon in M.
- Later model integrates in hours; it converts kinetic input rates internally. Earlier reconstruction uses seconds internally.
- Reported killing and MAE are percentages and percentage points; some low-level solvers return fractions. Consult each model's conversion before using its outputs.
- Healthy populations are neither target, T1 only, T2 only, both targets, weighted 0.79/0.10/0.10/0.01.
- Final robust optimisation: maximise minimum tumour kill, minimise maximum overall healthy kill and worst subgroup kill across three input variants; limits 0.95% overall and 10% per subgroup.
- Final prediction helper is restricted to its saved reference biology and trained rate domain. It does not certify ODE feasibility.

## Optional report builders

These are excluded from the core reproduction sequence. See `report folder/README.md`. No report generator is required to train, verify checkpoints or produce the principal final-presentation plots.
