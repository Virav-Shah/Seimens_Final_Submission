# Final Submission — Bispecific ADC modelling and optimisation

Virav Shah · IIT Mandi

This repository collects the executable work supporting the final presentation: calibrated reconstruction, extended mechanistic modelling, binding-rate optimisation, subgroup analysis, regression-assisted search, and physics-informed neural-network experiments. The original workspace has not been modified.

**Start here:** [presentation evidence map](docs/PRESENTATION_EVIDENCE.md), [reproduction instructions](docs/REPRODUCING.md), and [code catalogue](docs/CODE_CATALOG.md).

## Folder structure

```text
Final Submission/
├── README.md                     Project entry point
├── requirements.txt              Versions from the working environment
├── replication/                  Original ODE reconstruction, calibration and tests
├── repo/                         Main scientific implementation (legacy name retained)
│   ├── mechanism_comparison/     Extended ODE, mechanistic studies and plots
│   ├── ml_training/              GP/ExtraTrees-assisted optimisation and plots
│   ├── pinn_training/            PINN data, training, prediction, audits and plots
│   ├── *.ipynb                   Reconstruction, V3, V5, boundary study, saved-result viewer
│   ├── results_v3/               Earlier extension and subgroup evidence
│   ├── results_v5/               Final robust Pareto search and selected designs
│   ├── results_boundary_review/  Search-bound sensitivity and convergence evidence
│   ├── results_mechanism_comparison/  Fixed-rate versus reoptimised comparisons
│   └── report/figures/           Reconstruction figures only; no report builders
├── studies/                      Presentation plots and additional numerical studies
│   ├── presentation_revision/    Final internalisation and payload plots
│   ├── adc_timecourses/          Bispecific/monospecific time-course model and studies
│   └── complex_internalisation/ Common versus separate uptake-rate study
├── output/                       Saved numerical evidence and scientific figures
│   ├── ml_small_trial/           Regression search inputs, predictions and comparisons
│   ├── pinn_small_trial/         Initial PINN and gradient-imbalance evidence
│   ├── pinn_balanced_trial/      Revised PINN versus supervised-network experiment
│   ├── pinn_single_condition/    Final PINN/GP datasets, checkpoints and test results
│   └── ...                       Reconstruction and presentation-study results
├── report folder/                Optional report/notebook builders, outside core code
├── references/                   Submitted presentation and original paper/supplement
├── environment/                  Recorded original runtime metadata
└── docs/                         Execution, evidence, curation and provenance records
```

The `repo/` and result-directory names are retained because the original scientific code and recorded hashes use them. This avoids changing equations or silently invalidating provenance. Training folders are canonical copies: duplicate `ml_small_trial/` and `pinn_small_trial/` **code** directories have been removed; similarly named **output** directories contain the actual experiment evidence.

## Quick start

Use Python 3.14 to match the recorded project environment. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export MPLBACKEND=Agg
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python -m unittest discover -s replication -v
python repo/pinn_training/single_condition_verify.py
python repo/pinn_training/single_condition_predict.py --log-rates 4 -2 4 -2
```

Saved numerical results can be inspected immediately without retraining. Full training and optimisation overwrite their named outputs; use a copy of the repository when reproducing experiments. Stochastic optimisation, wall-time-limited neural training, library versions and hardware can change regenerated results. Checkpoint inference is the appropriate way to reproduce the submitted neural predictions exactly.

## Scope and interpretation

These are computational results under stated assumptions, not experimental or clinical validation. The reconstruction calibrates effective physiology using approximate figure landmarks; it is not exact replication of the authors' code. Simulated training labels come from the ODE solver. The final PINN figure compares endpoint prediction errors against a Gaussian process, not NSGA-II search quality. See the evidence map for further presentation/code differences.

Only selected scientific code, required numerical artifacts, checkpoints, reference documents and explanatory documentation are retained. Logs, virtual environments, caches, ZIP duplicates, generated reports, temporary notebook builders and abandoned data-download attempts are excluded. Failed PINN experiments are deliberately retained because they support the presentation's discussion of gradient imbalance.

See [validation performed](docs/VALIDATION.md) for what was checked during packaging and what was not rerun.
