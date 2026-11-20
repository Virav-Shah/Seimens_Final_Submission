from pathlib import Path
import sys
_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
__file__ = str(_SUBMISSION_ROOT / 'replication/report_prototype.py')
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(_SUBMISSION_ROOT / "repo"))
"""Render reproducible Markdown/LaTeX summaries from completed prototype outputs."""
import json,csv
from pathlib import Path
from prototype import OUT,RATES

def main():
    r=json.loads((OUT/'results.json').read_text());c=json.loads((OUT/'configuration.json').read_text())
    rows=[];rates=[];seeds=[];sens=[]
    for name,v in r.items():
        b=v['selected'];rows.append(f"| {name} | {b['tumor_kill_pct']:.3f}% | {b['healthy_kill_pct']:.3f}% |")
        rates.append(f"| {name} | "+' | '.join(f'{b[k]:.6g}' for k in RATES)+' |')
        seeds.append(f"- {name}: "+'; '.join(f"seed {q['seed']}: {q['best']['tumor_kill_pct']:.3f}% tumor / {q['best']['healthy_kill_pct']:.3f}% healthy" for q in v['runs']))
        worst=max(v['sensitivity'],key=lambda x:x['healthy_kill_pct']);sens.append(f"- {name}: largest tested healthy kill {worst['healthy_kill_pct']:.3f}% under `{worst['perturbation']}`.")
    md='''# Working optimisation prototype

We now have a functioning inference-led prototype that finds high-tumor/low-healthy-kill designs in all three scenarios. It uses the same mechanistic workflow as the paper, with explicit inferred physiology and a broader engineering search space. It does not claim the original source-code implementation or a global optimum.

## Selected results at 24 hours

| Scenario | Tumor kill | Healthy kill |
|---|---:|---:|
'''+ '\n'.join(rows)+'''

Healthy-cell kill is weighted over all healthy cells. Search feasibility requires <=0.95%, leaving 0.05 percentage-point nominal headroom below 1%. Selection maximises tumor kill among feasible evaluated candidates, after tighter-tolerance re-integration.

| Scenario | kon1 (M^-1 s^-1) | koff1 (s^-1) | kon2 (M^-1 s^-1) | koff2 (s^-1) |
|---|---:|---:|---:|---:|
'''+ '\n'.join(rates)+'''

## What we inferred and why

1. **Effective accessible cell burden**, rather than a guessed total-body cell count, controls the systemic binding sink. Fit value: approximately 4.73e12 reference cells.
2. **Effective intracellular volume** controls uptake-to-concentration conversion. Fit value: approximately 0.413 pL per cell, retaining the supplement's 500 µm² area. The paper never required a perfect spherical cell; real membrane area can exceed the area of a volume-equivalent sphere. This is an effective model parameter, not a measured anatomical value.
3. These two values were fitted only to approximate 24-hour tumor ADC concentrations read from Figure 4b (~56 µM) and Figure 6b (~280 µM). All other nuisance settings remained fixed. Matching those two anchors is calibration, not independent validation, and the fitted values are conditional on the remaining model assumptions.
4. Cell volume does not feed back into the current PK/binding equations. The ratio of the two uptake endpoints therefore constrains cell burden independently of the shared 1/volume concentration factor; their absolute scale then constrains volume. This gives a mechanistic reason for fitting these two parameters.
5. **Healthy uptake plotting:** the Figure 4 healthy curves appear to show the concentration averaged over target-positive healthy cells. Our predicted means are ~9.66 µM for the bispecific (all-healthy mean divided by 0.21) and ~16.90 µM for the monospecific (divided by 0.11), close to the figure's ~9 and ~17 µM. This is an inferred reporting convention. Toxicity still averages the separate Emax responses over ALL healthy cells; it is unchanged by the plotting convention.
6. **Search domain:** the figure heatmaps are not declared optimiser bounds. We explored kon=1e2–1e7 M^-1 s^-1 and koff=1e-5–1e2 s^-1. This permits weak initial capture with rapid second-arm rebinding. These are mathematical engineering candidates; experimental attainability of the selected kinetics is not established. Several selected on-rates lie below the paper's displayed sweep range.

## Baseline checks without further fitting

At the paper's fixed Figure 6 rates, the inferred model gives 98.59% tumor kill, 1.769% healthy kill, and 33.2% terminal occupancy, broadly consistent with its displayed behavior. The ~280 µM uptake is a calibration anchor, and its associated Emax kill is a direct consequence, not an independent validation point. The unoptimised bispecific gives 93.33% / 14.82% kill. The unoptimised monospecific gives 94.41% / 8.89% kill and 67.57 µM uptake, compared with roughly 74 µM in the figure. This recovers the expected unoptimised comparator ordering.

The unequal-density Figure 8 candidate remains a mismatch: predicted terminal T2 occupancy ~82.2% versus visually ~66%, uptake ~275 versus ~240 µM, and healthy kill ~4.59%. We did not fit this away. The new optimiser searches different binding rates under a strict healthy-kill constraint.

## Optimisation implementation

NSGA-II in Platypus; 48 candidates per generation; two independent seeds, 576 algorithm evaluations per scenario per seed; four log10-rate variables. Objectives are (-tumor kill, healthy kill); constraint is healthy fraction <=0.0095. Each evaluation integrates PK, binding, receptor turnover and ADC uptake for 24 hours. The selected best is retained from the entire evaluated archive, not just the last population. The selected states and full input parameters are saved in `results.json`.

'''+ '\n'.join(seeds)+'''

The repeated searches give similar useful designs but are not a convergence or global-optimality proof. The unequal-density candidate is near the lower kon2 bound, so further bound sensitivity is warranted before treating it as a final molecular design.

## Verification and sensitivity

- Eight existing model tests pass.
- Independent BDF and Radau integrations agree within 1e-6 in endpoint kill fraction for all selected candidates (actual differences <1e-9).
- Drug mass balance closes within 1e-6 relative error (actual selected errors <1e-12).
- All selected candidates remain <=0.95% healthy kill on tighter and independent integration.
- Twelve frozen-design checks per scenario explore cell burden, volume, turnover, target density and the alternative surface off-rate convention. No reoptimisation is used in these checks.
'''+ '\n'.join(sens)+'''

Thus the nominal designs meet the constraint, but they are not guaranteed to stay below 1% under every assumption change. `calibration_reading_sensitivity.csv` additionally refits the two nuisance parameters at four illustrative combinations of 50–62 µM and 250–310 µM plot anchors; those intervals are reading-sensitivity choices, not statistical confidence intervals.

## Run

```sh
.venv/bin/python replication/prototype.py --nfe 576 --seeds 8407 2026
.venv/bin/python replication/validate_prototype.py
.venv/bin/python replication/calibration_sensitivity.py
.venv/bin/python replication/report_prototype.py
```

The earlier reconstruction and results remain available under `output/simulation/`; these new results are under `output/prototype/`. `configuration.json` records calibration and bounds; `*_evaluations.csv` records all screened candidates; `*_selected_trajectory.csv` records full occupancy and uptake readouts; `*_sensitivity.csv` records frozen-design perturbations. No user input is required to run this prototype. Measured target densities, trafficking and feasible binding-rate ranges would make subsequent designs more specific.
'''
    (OUT/'PROTOTYPE_REPORT.md').write_text(md)
    table='\n'.join(name+r' & '+f"{v['selected']['tumor_kill_pct']:.3f} & {v['selected']['healthy_kill_pct']:.3f}"+r'\\' for name,v in r.items())
    rate_table='\n'.join(name+' & '+' & '.join(f"{v['selected'][k]:.4g}" for k in RATES)+r'\\' for name,v in r.items())
    tex=r'''\documentclass[11pt,a4paper]{article}
\usepackage[margin=23mm]{geometry}\usepackage{amsmath,booktabs,graphicx,hyperref}
\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}
\title{Working optimisation prototype\\Inference and results addendum}
\author{Stoddard et al. modelling project}\date{15 November 2026}
\begin{document}\maketitle
\textbf{Outcome.} A working prototype now finds high-tumor-kill candidates below 1\% healthy-cell kill for equal, unequal, and low target densities. This advances the earlier reconstruction by inferring two missing physiological inputs and exploring a broader binding-rate domain. It is not a claim to have recovered the authors' source code.

\section{Selected nominal results}
\begin{center}\begin{tabular}{lrr}\toprule
Scenario & Tumor kill (\%) & Healthy kill (\%)\\\midrule
@@TABLE@@
\bottomrule\end{tabular}\end{center}
All results are at 24 h. The low-density case uses $EC_{50}=10$ nM; the others use $4\,\mu$M. Cell kill is computed separately for each population before healthy-population weighting. The design constraint is $H\le0.0095$, leaving nominal headroom below the paper's 1\% threshold.

\begin{center}\begin{tabular}{lrrrr}\toprule
Scenario & $k_{on,1}$ & $k_{off,1}$ & $k_{on,2}$ & $k_{off,2}$\\\midrule
@@RATES@@
\bottomrule\end{tabular}\end{center}
On-rates are M$^{-1}$s$^{-1}$; off-rates are s$^{-1}$. These are candidate model parameters, not experimentally demonstrated antibody designs.

\section{What the paper's curves tell us}
The effective cell burden determines depletion of central free ADC, and intracellular volume converts accumulated uptake to concentration. The original guesses made the drug disappear too quickly and made concentration too small. We fitted only these two missing values to approximate Figure 4b and Figure 6b tumor uptake endpoints: 56 and 280 $\mu$M, respectively. These are manual visual estimates, not raw data.

The inferred values are
\[
 N_{\mathrm{ref}}\simeq4.73\times10^{12}\ \mathrm{cells},\qquad
 v_{\mathrm{cell}}\simeq0.413\ \mathrm{pL}.
\]
Area stays at $500\,\mu$m$^2$. A perfect sphere was our earlier assumption, not a requirement of the paper; effective membrane area and intracellular volume can be specified separately. These values remain conditional on the fixed PK, turnover and surface-binding conventions.

Because volume does not enter the PK/binding subsystem, write
\[
 A_s(24\mathrm{h})=\frac{a}{v}\int_0^{24\mathrm{h}}J_s(t;N_{\mathrm{ref}})\,dt.
\]
The ratio $A_6/A_4\simeq280/56=5$ constrains cell burden without the common $1/v$ factor. The absolute uptake scale then determines volume. This explains the fitting strategy; exact matching of two anchors with two parameters is calibration, not independent validation.

A further reporting clue is the healthy uptake mean. The reconstructed all-healthy means for the two Figure 4 scenarios are about 2.03 and 1.86 $\mu$M. Dividing by their target-positive fractions, 0.21 and 0.11, gives approximately 9.66 and 16.90 $\mu$M, close to the displayed healthy curves. Thus the plot may show a mean conditional on target expression. Both conventions are now exported. Healthy \emph{kill} still uses all-healthy weighting; no change to the efficacy equation is made.

\section{Checks against other displayed behaviour}
At the published Figure 6 rates, the inferred prototype gives 98.59\% tumor kill, 1.769\% healthy kill, and 33.2\% terminal occupancy. The first value follows directly from the calibrated uptake and Emax equation; the other readouts provide additional checks. The unoptimised monospecific predicts 67.57 $\mu$M uptake versus roughly 74 $\mu$M in the plot, and gives 94.41\% tumor / 8.89\% healthy kill. The unoptimised bispecific gives 93.33\% / 14.82\%, recovering the paper's comparator ordering.

The published unequal-density candidate remains imperfectly matched: terminal T2 occupancy is about 82.2\% versus visually 66\%, and uptake is about 275 versus 240 $\mu$M. We preserve that discrepancy. Its predicted healthy kill is 4.59\%; the new constrained optimisation therefore selects different rates.

\section{Optimisation that now works}
Freeze the inferred physiology, then optimise only four binding rates. In log coordinates,
\[
 \boldsymbol z=\log_{10}(k_{on,1},k_{off,1},k_{on,2},k_{off,2}),
 \qquad \min(-E,H),\quad H\le0.0095.
\]
We explore $10^2\le k_{on}\le10^7$ and $10^{-5}\le k_{off}\le10^2$. The paper's heatmap axes are not explicitly stated as optimiser bounds. Broadening the domain allows weak first-arm capture while retaining avidity through the rapid second-arm encounter. Several selected on-rates are below the paper's plotted sweep domain; experimental attainability is unverified. The unequal-density candidate approaches the lower T2 on-rate bound.

NSGA-II uses a population of 48 and 576 algorithm evaluations per seed, with seeds 8407 and 2026 for each scenario. Interpretable starting candidates are mixed with random candidates. Every candidate runs the complete ODE system. Selection retains the best feasible candidate from all evaluations and then reintegrates it at tighter tolerance. No global-optimum claim is made.

\newpage
\section{Search and selected time courses}
\includegraphics[width=.90\linewidth]{optimisation_dashboard.png}\par
The top row shows screened candidates; black stars mark selected designs and the red line marks 1\% healthy kill. The bottom row shows the selected trajectories. Two seeds returned similar high-selectivity designs, with final tumor-kill values differing by about 0.4 percentage points for equal density, 0.05 for unequal density and 0.95 for low density. This is useful repeat-search evidence, not proof of convergence.

\section{Verification and remaining sensitivity}
Eight existing model checks pass. Independent BDF and Radau integrations of selected candidates agree within $10^{-9}$ in endpoint kill fraction; relative mass-balance errors are below $10^{-12}$. All three remain below the stricter 0.95\% threshold under these numerical checks.

Frozen-design perturbations vary cell burden, cell volume, turnover, target density and surface off-rate convention. Doubling turnover raises healthy kill to approximately 1.125\%, 1.104\% and 1.525\% for equal, unequal and low density, respectively. The designs therefore meet the nominal constraint but do not meet it under every tested assumption change. A separate reading-sensitivity audit refits the calibration at four combinations of 50--62 and 250--310 $\mu$M anchors; these ranges are illustrative, not confidence intervals. Equal- and unequal-density designs stay below 1\% in all four reading-sensitivity cases; the low-density design exceeds it in two.

\section{Files and reproducibility}
\texttt{replication/prototype.py} performs calibration and screening. \texttt{validate\_prototype.py} checks independent integration. \texttt{calibration\_sensitivity.py} tests plot-reading uncertainty. Detailed inputs, all evaluated candidates, selected trajectories, both healthy concentration means, and sensitivity results are saved under \texttt{output/prototype/}. The original reconstruction is preserved as an earlier baseline.

The sources remain the supplied paper (DOI 10.20935/AcadDrug8407), its supplement, and the source models cited in the original analysis. This addendum records new inferences and computations; it does not attribute the inferred values or expanded search bounds to the authors.
\end{document}
'''.replace('@@TABLE@@',table).replace('@@RATES@@',rate_table)
    (OUT/'prototype_addendum.tex').write_text(tex)
if __name__=='__main__':main()
