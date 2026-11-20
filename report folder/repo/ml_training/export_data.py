from pathlib import Path
import sys
_SUBMISSION_ROOT = Path(__file__).resolve().parents[3]
__file__ = str(_SUBMISSION_ROOT / 'repo/ml_training/export_data.py')
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(_SUBMISSION_ROOT / "repo"))
from pathlib import Path
import csv,json,numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/ml_small_trial'
files=[OUT/'validation.json',OUT/'test.json']+list(OUT.glob('initial_*.json'))+list(OUT.glob('active_*.json'))+list(OUT.glob('direct_fair_[0-9]*.json'))
rows={}
for p in files:
 if 'verified' in p.name:continue
 for r in json.loads(p.read_text()):
  key=tuple(r['x'])
  if key not in rows:rows[key]={'r':r,'roles':[]}
  rows[key]['roles'].append(p.stem)
cols=['roles']+['log10_'+n for n in ['kon1','koff1','kon2','koff2']]+['kon1_M-1_s-1','koff1_s-1','kon2_M-1_s-1','koff2_s-1','KD1_M','KD2_M']+[f'{c}_{g}_kill_pct' for c in ['reference','A','B'] for g in ['tumour','H0','H1','H2','H12']]
with (OUT/'simulated_dataset.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(cols)
 for x,item in rows.items():
  rates=10.**np.array(x);w.writerow([';'.join(item['roles']),*x,*rates,rates[1]/rates[0],rates[3]/rates[2],*item['r']['y']])
print('Unique simulated combinations:',len(rows))
(OUT/'README.md').write_text('''# Completed small ML trial

Start with Trial_report.md and trial_flowchart.png. SVG figures are editable.

The flowchart covers simulated data generation, separate training/validation/test data, logarithmic rates, training-only standardisation, regression selection, objective selection, iterative training, prediction errors and the direct-search comparison.

Main finding: regression-assisted search covered slightly more trade-offs in all three seeds, but did not reduce mean runtime at this small budget. Predictions alone were insufficient to enforce healthy-cell limits. The study uses simulated biology and does not establish clinical accuracy.

Files:
- Trial_report.md: methodology, exact results and limitations.
- simulated_dataset.csv: unique inputs and ODE-generated labels, with dataset roles.
- biological_inputs.json: fixed biological assumptions for all three conditions.
- results.json: full accuracy and search metrics.
- PNG/SVG figures: flowchart, prediction error, search comparison and predicted-versus-simulated killing.
- Individual JSON files: complete simulation and acquisition records.

Reproduce from the project root with the existing Python environment:
1. python repo/ml_small_trial/run_trial.py
2. python repo/ml_small_trial/fair_comparison.py
3. python repo/ml_small_trial/report_trial.py
4. python repo/ml_small_trial/export_data.py

Set MPLBACKEND=Agg, OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1. These scripts import the unchanged model in repo/mechanism_comparison/model.py. Model-selection simulations count against the ML budget. The shared test simulations do not participate in either search. Final tighter checks are additional and their runtime is included.
''')
