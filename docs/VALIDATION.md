# Validation performed during packaging

Validation date: 20 November 2026. Commands used the existing Python environment from the original workspace, with the working directory inside `Final Submission`. No scientific import depended on the original workspace code.

| Check | Outcome |
|---|---|
| Python syntax and retained notebook schemas | Passed |
| Reconstruction unittest suite | 8/8 passed: mass balance, Emax weighting, symmetry, zero dose, zero targets, receptor accounting, detailed balance, tolerance convergence |
| Calibrated prototype: BDF versus Radau | All three scenarios passed; maximum endpoint discrepancy <9e-10 in fractional killing; mass errors <5e-13 |
| V5 notebook definitions and implementation-check cells (before search) | Passed |
| Final PINN/GP checkpoint prediction agreement | Passed at rtol=atol=1e-10 against saved predictions |
| Final training/validation/test partitions | Disjoint; 128 final test designs; saved predictions finite |
| Final reference-model source hash | Matches recorded training configuration |
| Final result-viewer notebook | Schema and in-process display-cell execution passed; no Jupyter kernel launch used |
| Initial PINN artifact audit | Saved checkpoint reproduces predictions; exact initial conditions/bridge continuity; finite residual gradients; 97 states, 129 times; partitions disjoint |
| Mechanism, regression and final PINN plot-only entry points | Completed successfully |
| Final presentation sweep/comparison plot helper | Completed using existing sweep data; its historical console text says “324 new ODE evaluations”, but the saved cache was reused during packaging |
| Internalisation replacement plots, supporting overview plots, numerical/provenance export | Completed successfully |
| Final feasible-candidate and filtered-table counts | Checked against presentation claims: 2,101 and 140 |
| Exclusion check | No virtual environment, log files, Python caches or generated HTML reports in delivery |

The full optimisation searches and neural training were **not rerun** during packaging. The saved results are evidence of the original runs, while the checks above verify portability, scientific implementation consistency, and checkpoint integrity. Wall-time-limited training cannot guarantee identical regenerated weights on different hardware.

An existing PyTorch audit emitted a Python 3.14 `torch.jit.script` deprecation/support warning; the audit nevertheless passed. No measured clinical or experimental accuracy is claimed.

The optional report builders were syntax checked, not regenerated. The dependencies were read from the original environment; a fresh dependency installation was not performed. Git was initialised locally with branch `main`; files have not been committed or published.
