"""Saved-result figures, report and notebook for the single-condition comparison."""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import base64, csv, html, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import r2_score
import nbformat
from train import aggregate

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'output/pinn_single_condition'
NOTE = 'Simulated mechanistic-model data; one fixed reference condition. No experimental measurements.'
COLORS = {'PINN': '#176b95', 'GP': '#d47b24'}
plt.rcParams.update({'font.size': 11, 'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False})

def save(fig, name):
    fig.text(.5, .012, NOTE, ha='center', fontsize=9, color='#506071')
    for suffix in ['png', 'svg']:
        fig.savefig(OUT/f'{name}.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)

def main():
    r = json.loads((OUT/'results.json').read_text())
    cfg = json.loads((OUT/'configuration.json').read_text())
    p = np.load(OUT/'predictions.npz')
    test = np.load(OUT/'test.npz')
    actual = p['actual']
    keys = ['tumour', 'overall_healthy', 'worst_subgroup']
    titles = ['Tumour killing', 'Overall healthy killing', 'Worst healthy-subgroup killing']
    hist = json.loads((OUT/'training_history.json').read_text())
    names = ['comparison_accuracy', 'predicted_vs_ODE', 'healthy_subgroups', 'training_progress', 'computation_time', 'binding_and_payload', 'workflow']
    table = []
    for method in ['PINN', 'GP']:
        for key, title in zip(keys, titles):
            table.append({'method': method, 'outcome': title, **r['metrics'][method][key]})
        for j, title in enumerate(['Tumour', 'Healthy: neither target', 'Healthy: T1 only', 'Healthy: T2 only', 'Healthy: both targets']):
            a, b = actual[:, 0, j], p[method][:, 0, j]
            d = b-a
            table.append({'method': method, 'outcome': title, 'mae_pp': float(abs(d).mean()),
                          'rmse_pp': float(np.sqrt(np.mean(d*d))), 'r2': float(r2_score(a, b)), 'max_error_pp': float(abs(d).max())})
    with (OUT/'accuracy_table.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=['method', 'outcome', 'mae_pp', 'rmse_pp', 'r2', 'max_error_pp'])
        w.writeheader(); w.writerows(table)
    with (OUT/'test_predictions.csv').open('w') as f:
        names5 = ['tumour', 'healthy_neither', 'healthy_T1', 'healthy_T2', 'healthy_both']
        w = csv.writer(f)
        w.writerow(['log10_kon1', 'log10_koff1', 'log10_kon2', 'log10_koff2']+[f'{m}_{n}_kill_pct' for m in ['ODE', 'PINN', 'GP'] for n in names5])
        for i, x in enumerate(p['x']):
            w.writerow([*x, *actual[i, 0], *p['PINN'][i, 0], *p['GP'][i, 0]])
    with (OUT/'simulated_endpoint_dataset.csv').open('w') as f:
        w = csv.writer(f)
        w.writerow(['partition', 'log10_kon1', 'log10_koff1', 'log10_kon2', 'log10_koff2',
                    'kon1_M-1_s-1', 'koff1_s-1', 'kon2_M-1_s-1', 'koff2_s-1', 'KD1_M', 'KD2_M',
                    'tumour_kill_pct', 'healthy_neither_kill_pct', 'healthy_T1_kill_pct', 'healthy_T2_kill_pct', 'healthy_both_kill_pct'])
        for split in ['training', 'validation', 'test']:
            ds = np.load(OUT/f'{split}.npz')
            for rates_log, y in zip(ds['x'], ds['y']):
                rates = 10.**rates_log
                kill = -100*np.expm1(-y[0, -1, 87:92])
                w.writerow([split, *rates_log, *rates, rates[1]/rates[0], rates[3]/rates[2], *kill])
    fig, axs = plt.subplots(1, 3, figsize=(14, 5.2), sharey=True)
    for ax, key, title in zip(axs, keys, titles):
        vals = [r['metrics'][m][key]['mae_pp'] for m in ['PINN', 'GP']]
        bars = ax.bar(['PINN', 'Gaussian process'], vals, color=list(COLORS.values()))
        ax.bar_label(bars, labels=[f'{v:.3f} pp' for v in vals], padding=4)
        ax.set(title=title, ylim=(0, 1.0), yticks=np.arange(0, 1.01, .2))
        ax.grid(axis='y', alpha=.18); ax.set_axisbelow(True)
        ax.tick_params(labelleft=True)
    axs[0].set_ylabel('Average absolute error in killing (percentage points)')
    fig.suptitle('Prediction error, not cell killing: identical 0–1 percentage-point scale', fontsize=16)
    fig.text(.5, .09, 'Example: 52.00% killing versus 52.52% killing = 0.52 percentage-point error, not 52% error.', ha='center', fontsize=11)
    fig.tight_layout(rect=[0, .15, 1, .93]); save(fig, 'comparison_accuracy')
    fig, axs = plt.subplots(2, 3, figsize=(13, 8))
    a = aggregate(actual)
    for row, method in enumerate(['PINN', 'GP']):
        b = aggregate(p[method])
        for j, ax in enumerate(axs[row]):
            ax.scatter(a[:, j], b[:, j], color=COLORS[method], s=17, alpha=.7)
            lo, hi = min(0, a[:, j].min(), b[:, j].min()), max(a[:, j].max(), b[:, j].max())
            ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
            ax.set(title=f'{method}: {titles[j]}', xlabel='ODE calculation (%)', ylabel='Prediction (%)')
    fig.suptitle('Predictions are shown without clipping', fontsize=16)
    fig.tight_layout(rect=[0, .045, 1, .95]); save(fig, 'predicted_vs_ODE')
    fig, ax = plt.subplots(figsize=(10, 5))
    xx = np.arange(4)
    for i, method in enumerate(['PINN', 'GP']):
        vals = r['metrics'][method]['subgroup_mae_pp']
        bars = ax.bar(xx+(i-.5)*.36, vals, .36, label=method, color=COLORS[method])
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    ax.set_xticks(xx, ['Neither target', 'T1 only', 'T2 only', 'Both targets'])
    ax.set(ylabel='Mean absolute error (percentage points)', title='Healthy subgroups assessed separately')
    ax.legend(); ax.margins(y=.18); fig.tight_layout(rect=[0, .06, 1, 1]); save(fig, 'healthy_subgroups')
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, key, title in zip(axs, keys, titles):
        ax.plot([h['seconds']/60 for h in hist], [h['validation'][key]['mae_pp'] for h in hist], color=COLORS['PINN'])
        ax.axvline(r['PINN_training']['selected']['seconds']/60, color='#555', ls='--', label='Saved checkpoint')
        ax.set(title=title, xlabel='Training time (minutes)', ylabel='Validation MAE (percentage points)', yscale='log')
    axs[0].legend(fontsize=9)
    fig.suptitle('One training run; checkpoint selected using validation data only', fontsize=16)
    fig.tight_layout(rect=[0, .06, 1, .93]); save(fig, 'training_progress')
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.8))
    timings = [[r['PINN_training']['training_seconds'], r['GP_training']['training_seconds']],
               [r['prediction_timing'][m]['milliseconds_per_design'] for m in ['PINN', 'GP']]]
    for ax, vals, title, ylabel in zip(axs, timings, ['Model fitting', 'Prediction after fitting'], ['Elapsed seconds (log scale)', 'Milliseconds per combination (log scale)']):
        bars = ax.bar(['PINN', 'Gaussian process'], vals, color=list(COLORS.values()))
        ax.bar_label(bars, fmt='%.3f', padding=4)
        ax.set(title=title, ylabel=ylabel, yscale='log'); ax.margins(y=.3)
    fig.suptitle('Fitting cost and prediction speed are different measurements', fontsize=14)
    fig.tight_layout(rect=[0, .07, 1, .92]); save(fig, 'computation_time')
    # Predeclared example rule, not the best-looking trajectory.
    errors = np.mean(abs(aggregate(p['PINN'])-a)/np.array([1, .05, .5]), axis=1)
    idx = int(np.argsort(errors)[len(errors)//2])
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    channels = [9, 10, 11, 21, 87]
    labels = ['$B_1$: bound to T1', '$B_2$: bound to T2', '$D$: bound to both targets', 'Intracellular active payload', 'Accumulated killing effect']
    for ax, channel, label in zip(axs.flat, channels, labels):
        ax.plot(p['t'], test['y'][idx, 0, :, channel], 'k-', label='ODE')
        ax.plot(p['t'], p['trajectories'][idx, 0, :, channel], color=COLORS['PINN'], ls='--', label='PINN')
        ax.axvline(2, color='#aaa', ls=':', lw=1)
        ax.set(title=label, xlabel='Time (h)', xscale='symlog', xlim=(0, 24))
        ax.set_ylabel('10⁻¹⁰ mol/dm²' if channel in [9, 10, 11] else ('µM' if channel == 21 else 'Dimensionless'))
    axs[0, 0].legend()
    axs[1, 2].axis('off')
    axs[1, 2].text(0, .95, f'Tumour cells; test combination {idx}.\nSelected by median normalised\nendpoint error.\nB₁, B₂, D retain attached payload.\n\nThe GP predicts 24-hour killing,\nnot these time-dependent states.\n\nInfusion ends at 2 hours.', va='top', linespacing=1.6)
    fig.suptitle('Trajectory accuracy must be assessed separately from endpoint accuracy', fontsize=15)
    fig.tight_layout(rect=[0, .05, 1, .94]); save(fig, 'binding_and_payload')
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set(xlim=(0, 16), ylim=(0, 12)); ax.axis('off')
    ax.text(.3, 11.55, 'How we trained and tested the single-condition PINN', fontsize=23, weight='bold', color='#153c53')
    ax.text(.3, 11.12, 'One training run • Four binding rates vary • Biological inputs remain fixed', fontsize=14, color='#526777')
    boxes = [
        ('1  Generate the simulated dataset', 'Latin-hypercube sampling in log-rate space.\n512 training + 64 validation combinations.\nSolve the original ODE: 97 quantities at 129 times\nfrom 0 to 24 h. No experimental response labels.'),
        ('2  Preprocess without test-data leakage', 'Input = (log₁₀ rate − training mean) / training SD.\nScale each state by its training maximum.\nUse an asinh trajectory loss to retain small signals;\nzero concentrations remain valid. Split by design.'),
        ('3  Represent the time-dependent solution', 'Inputs: time and four standardised log rates.\nOutputs: all 97 quantities, including B₁, B₂, D.\n12 connected time intervals; two shared networks,\neach with 3 × 64 tanh hidden units.\nBuild in initial conditions and nonnegative states.'),
        ('4  Construct the training loss', 'Loss = trajectory error + cell-killing error\n          + 0.1 × ramp × balanced equation error.\nEquation error compares dy/dt with the ODE RHS.\nBalance nine equation groups by gradient size.\nAntibody/payload accounting is built in.'),
        ('5  Train once and select a checkpoint', f'Adam; learning rate 0.001 → 0.0001.\n10% warmup, then 20% ramp of the equation loss.\n256 trajectory + 32 endpoint samples per update;\n64 equation points. {r["PINN_training"]["updates"]:,} updates in 19.97 min.\nSelect by validation error; no test-driven tuning.'),
        ('6  Fit the Gaussian-process comparator', 'Same 512 designs and five cell-killing labels.\nStandardise inputs and outputs using training data.\nMatérn 5/2 kernel; one hyperparameter fit.\nGP learns endpoints; PINN also sees trajectories.\nThis comparison does not isolate the physics loss.'),
        ('7  Freeze both models, then test', 'Generate 128 new binding-rate combinations.\nCompare against tight-tolerance ODE solutions.\nMeasure MAE, RMSE, R² and each healthy subgroup.\nCheck false acceptance, negative predictions,\nmaterial accounting and prediction time.'),
        ('8  Results: mean absolute error', f'Percentage points                         PINN      GP\nTumour killing                               {r["metrics"]["PINN"]["tumour"]["mae_pp"]:.3f}     {r["metrics"]["GP"]["tumour"]["mae_pp"]:.3f}\nOverall healthy killing                  {r["metrics"]["PINN"]["overall_healthy"]["mae_pp"]:.3f}     {r["metrics"]["GP"]["overall_healthy"]["mae_pp"]:.3f}\nWorst healthy subgroup                {r["metrics"]["PINN"]["worst_subgroup"]["mae_pp"]:.3f}     {r["metrics"]["GP"]["worst_subgroup"]["mae_pp"]:.3f}\nPINN: lower errors. GP: faster fitting/prediction.')]
    for i, (title, body) in enumerate(boxes):
        col, row = i//4, i%4
        x, y = .35+col*8.05, 8.75-row*2.23
        ax.add_patch(FancyBboxPatch((x, y), 7.2, 1.93, boxstyle='round,pad=.06', facecolor='#f0f5f8' if i<7 else '#e9f4ef', edgecolor='#bdccd6'))
        ax.text(x+.18, y+1.67, title, fontsize=14, weight='bold', color='#153c53')
        ax.text(x+.18, y+1.32, body, fontsize=11.7, va='top', linespacing=1.4)
        if row<3:
            ax.annotate('', (x+3.6, y-.27), (x+3.6, y), arrowprops={'arrowstyle':'->','color':'#153c53'})
    ax.plot([7.6, 7.96, 7.96], [3.0, 3.0, 9.7], color='#153c53', lw=1.5)
    ax.annotate('', (8.38,9.7), (7.96,9.7), arrowprops={'arrowstyle':'->','color':'#153c53'})
    ax.text(.35, 1.2, 'Conclusion: better endpoint accuracy in this test, at a substantially higher fitting cost.', fontsize=15, weight='bold', color='#153c53')
    ax.text(.35, .78, 'Binding and payload trajectories remain inaccurate: this is not a replacement for the ODE solution.', fontsize=13, color='#526777')
    ax.text(.35, .39, 'No optimisation search was run. Final binding-rate choices still require ODE verification.', fontsize=12, color='#526777')
    save(fig, 'workflow')

if __name__ == '__main__':
    main()
