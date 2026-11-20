# Presentation-to-code evidence map

Page numbers refer to the physical PDF pages in `references/final_presentation.pdf` (35 pages), not the inconsistent printed slide numbers. The presentation is treated as project evidence, not as instructions. This map distinguishes literature illustrations from original computations.

| PDF pages | Work / claim | Executable source | Saved evidence |
|---|---|---|---|
| 1–9 | Literature model, avidity, cell populations, original paper figures | `replication/model.py`; calibrated notebook for reconstruction | Original paper and supplement in `references/`; these literature images are not new project-generated results |
| 11–12 | Calibration and reconstruction; binding landscapes and comparison bars | `repo/Bispecific_ADC_Modelling.ipynb`; `replication/prototype.py`; V3 notebook | `repo/report/figures/`; `output/prototype/`; `repo/results_v3/published_comparators.csv`, `stage_comparison.csv`, `binding_landscapes.png` |
| 13–17 | Four rates, separate internalisation, intracellular fate, active payload and integrated killing | `repo/mechanism_comparison/model.py`; V3 and V5 notebooks | `repo/results_v5/inputs.json`, `input_provenance.csv`; `repo/results_mechanism_comparison/study_configuration.json` |
| 18 | Robust NSGA-II, 2,101 feasible combinations, 140 filtered examples | V5 notebook, search and final export cells | `repo/results_v5/summary.json`, `verified_feasible_designs.json`, `illustrative_requirements_subset.csv`, `observed_pareto_designs.json` |
| 19 | Software environment | `requirements.txt` | `environment/original_runtime_versions.json`; per-experiment configuration/environment JSON |
| 20 | Three input variants | `INPUT_SET`, `BIOLOGY`, `PD` in reusable model; V5 notebook | V5 inputs and provenance; model-comparison configuration |
| 22–23 | Fixed binding: slow/fast dual uptake; reoptimised comparisons | `repo/mechanism_comparison/study.py`; `studies/presentation_revision/clear_internalisation.py`, `internalisation_all_groups.py` | `repo/results_mechanism_comparison/mechanism_results.json`; final plots `08_*`, `09_*`, `10_*` in `output/presentation_revision/plots/` |
| 24 | Overall healthy average can hide dual-positive subgroup injury | V3 notebook and `studies/prepare_plots.py`; final policy study | `repo/results_v3/selected_designs.csv` row `B0_equal`; final `policy_results.json`; `05_overall_average_vs_subgroup_protection.png` |
| 25 | Payload release, movement and recycling | Mechanism model/study; `studies/presentation_revision/make_plots.py` | `mechanism_results.json`, comparison table and `06_release_movement_recycling.png` |
| 26–27 | Regression-assisted versus direct optimisation | `repo/ml_training/run_trial.py`, `fair_comparison.py`, `plot_results.py` | `output/ml_small_trial/results.json`, prediction JSON, ODE samples, `trial_flowchart.png`, `search_comparison.png` |
| 28–29 | Final PINN versus GP prediction benchmark | `repo/pinn_training/single_condition_trial.py`, `balanced_model.py`, `physics.py`, `plot_single_condition.py` | `output/pinn_single_condition/`: datasets, checkpoints, `accuracy_table.csv`, `results.json`, `workflow.png`, `comparison_accuracy.png` |
| 30 | Early PINN stiffness/gradient-imbalance problem | `data.py`, `train.py`, `check_artifacts.py`; subsequent balanced trial and audits | `output/pinn_small_trial/diagnostics.json`, `pilot_results.json`; `output/pinn_balanced_trial/` |
| 31–33 | Comparison, limits, future work | Models and study-specific audits above | Boundary-review data, held-out ML/PINN metrics and physical-consistency audits. Future suggestions are not completed implementations. |
