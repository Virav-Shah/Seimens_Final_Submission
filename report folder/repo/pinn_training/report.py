from pathlib import Path
import sys
_SUBMISSION_ROOT = Path(__file__).resolve().parents[3]
__file__ = str(_SUBMISSION_ROOT / 'repo/pinn_training/report.py')
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(_SUBMISSION_ROOT / "repo"))
"""Generate figures and a notebook from actual completed feasibility results."""
import os
os.environ.setdefault('MPLBACKEND','Agg')
import json,html,base64,hashlib,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import nbformat
from data import OUT
from train import endpoints,aggregate
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
BLUE='#186b9b';ORANGE='#c85e24';GRAY='#667681'
NOTE='Simulated data from the existing mechanistic model; three fixed illustrative biological conditions.'

def save(fig,name):
    fig.text(.5,.012,NOTE,ha='center',fontsize=9,color=GRAY)
    fig.savefig(OUT/f'{name}.png',dpi=180,bbox_inches='tight');fig.savefig(OUT/f'{name}.svg',bbox_inches='tight');plt.close(fig)

def main():
    r=json.loads((OUT/'pilot_results.json').read_text());cfg=json.loads((OUT/'configuration.json').read_text());checks=json.loads((OUT/'equation_checks.json').read_text())
    d=np.load(OUT/'pilot_test_predictions.npz');actual=d['actual'];pred=d['prediction'];times=d['t'];a=aggregate(endpoints(actual[:,:,-1]));p=aggregate(endpoints(pred[:,:,-1]));test=r['test'];val=r['validation']
    fail=r['status']!='passed'
    fig,axs=plt.subplots(1,3,figsize=(13.8,4.5))
    titles=['Lowest tumour killing','Highest overall healthy killing','Highest healthy-subgroup killing']
    for j,(ax,title) in enumerate(zip(axs,titles)):
        ax.scatter(a[:,j],p[:,j],s=24,color=BLUE,alpha=.75)
        lo=min(0,float(a[:,j].min()),float(p[:,j].min()));hi=max(float(a[:,j].max()),float(p[:,j].max()),1)
        ax.plot([lo,hi],[lo,hi],color=GRAY,ls='--');ax.set(xlabel='ODE calculation (%)',ylabel='PINN prediction (%)',title=title)
        ax.text(.04,.95,f"MAE: {test[['tumour','overall_healthy','worst_subgroup'][j]]['mae_pp']:.2f} percentage points",transform=ax.transAxes,va='top',fontsize=9)
    fig.suptitle('Does the PINN reproduce cell killing for unseen binding rates?',fontsize=16);fig.tight_layout(rect=[0,.05,1,.93]);save(fig,'cell_killing_accuracy')
    fig,ax=plt.subplots(figsize=(9,4.8));errs=test['subgroup_mae_pp'];bars=ax.bar(['Neither target','T1 only','T2 only','Both targets'],errs,color=BLUE)
    ax.bar_label(bars,fmt='%.2f',padding=4);ax.set(ylabel='Mean absolute error (percentage points)',title='Prediction error in each healthy-cell subgroup');ax.set_ylim(0,max(errs)*1.25);fig.tight_layout(rect=[0,.05,1,1]);save(fig,'healthy_subgroup_errors')
    # Select median endpoint-error design transparently, not the best-looking one.
    err=np.mean(abs(p-a)/np.array([100,.95,10]),axis=1);idx=int(np.argsort(err)[len(err)//2]);condition=0
    cols=[9,10,11,21,87];names=['T1-bound ADC: B1','T2-bound ADC: B2','Doubly bound ADC: D','Intracellular active payload','Accumulated killing effect']
    units=['10⁻¹⁰ mol/dm²']*3+['µM','dimensionless']
    fig,axs=plt.subplots(2,3,figsize=(14,8));axs=axs.ravel()
    for j,ax in enumerate(axs[:5]):
        col=cols[j];ax.plot(times,actual[idx,condition,:,col],label='ODE',color=BLUE,lw=2)
        ax.plot(times,pred[idx,condition,:,col],label='PINN',color=ORANGE,ls='--',lw=2)
        ax.axvline(2,color=GRAY,ls=':',lw=1);ax.set(xlabel='Time (h)',ylabel=units[j],title=names[j]);ax.set_xscale('symlog',linthresh=.01);ax.set_xlim(0,24)
    axs[0].legend();axs[5].axis('off');axs[5].text(0,.85,'Example selection\n\nMedian scaled endpoint error among\n64 held-out designs.\n\nTumour cells, reference condition.\nDotted line: infusion ends at 2 h.\n\nNegative predictions are shown,\nnot removed.',va='top',linespacing=1.5)
    fig.suptitle('Binding and payload trajectories: where does the approximation differ?',fontsize=16);fig.tight_layout(rect=[0,.05,1,.95]);save(fig,'representative_trajectories')
    (OUT/'representative_design.json').write_text(json.dumps({'test_index':idx,'theta_log10':d['x'][idx].tolist(),'selection':'median mean normalised endpoint error','condition':'reference','cell_population':'tumour'},indent=2))
    hist=json.loads((OUT/'pilot_training.json').read_text());sec=np.array([z['seconds'] for z in hist]);scores=[z['validation_score'] for z in hist]
    fig,axs=plt.subplots(1,2,figsize=(12,4.5));axs[0].plot(sec,scores,color=BLUE,marker='o',ms=3);axs[0].set(xlabel='Training time (s)',ylabel='Normalised validation error',title='Endpoint prediction during training',yscale='log')
    for key,label,color in [('data_loss','Trajectory loss',BLUE),('ode_loss','Equation residual loss',ORANGE)]:axs[1].plot(sec,[z[key] if z[key]>0 else np.nan for z in hist],label=label,color=color)
    axs[1].set(xlabel='Training time (s)',ylabel='Scaled mean squared loss',yscale='log',title='Trajectory fit and equation agreement');axs[1].legend()
    selected=r['training']['selected_checkpoint']['seconds']
    axs[0].axvline(selected,color=ORANGE,ls='--',label='Selected full-physics checkpoint');axs[0].legend(fontsize=9)
    fig.tight_layout(rect=[0,.05,1,1]);save(fig,'training_history')
    # Explicitly display gate checks, including large failures without hiding scale.
    gate_names=['Tumour error','Overall healthy error','Worst subgroup error','Negative states','Material balance']
    observed=[val['tumour']['mae_pp'],val['overall_healthy']['mae_pp'],val['worst_subgroup']['mae_pp'],val['negative_scaled_max'],val['mass_fraction_max']]
    limits=np.array([5,.25,3,.001,.01]);ratios=np.array(observed)/limits
    fig,ax=plt.subplots(figsize=(10,5));bars=ax.barh(gate_names,ratios,color=[BLUE if z<1 else ORANGE for z in ratios]);ax.axvline(1,color=GRAY,ls='--');ax.set_xscale('log');ax.invert_yaxis();ax.set(xlabel='Observed error / permitted error (below 1 passes)',title='Predeclared feasibility gate')
    for b,z in zip(bars,ratios):ax.text(max(z,1e-8)*1.12,b.get_y()+b.get_height()/2,f'{z:.2g}×',va='center')
    ax.set_xlim(max(min(ratios)*.3,1e-6),max(ratios)*5);fig.tight_layout(rect=[0,.05,1,1]);save(fig,'feasibility_gate')
    # Editable vector flowchart, with executed and unexecuted stages distinguished.
    fig,ax=plt.subplots(figsize=(14,10));ax.set(xlim=(0,14),ylim=(0,10));ax.axis('off')
    boxes=[(0.3,7.6,6.2,1.35,'1  Preserve and check the equations',f'97 quantities; 63 sampled states\nMaximum derivative difference {checks["max_absolute_difference"]:.2e}',BLUE),
           (7.5,7.6,6.2,1.35,'2  Generate numerical training data','64 training + 32 validation + 64 test designs\n3 conditions × 129 times; tight ODE integration',BLUE),
           (7.5,5.4,6.2,1.35,'3  Scale inputs and trajectories','Log₁₀ of four binding rates; training-only standardisation\nLinear state scaling; two connected time intervals',BLUE),
           (0.3,5.4,6.2,1.35,'4  Fit the PINN','Two networks: 4 layers × 64 units; double precision\nTrajectory error + equations + balance + nonnegativity',BLUE),
           (0.3,3.2,6.2,1.35,'5  Evaluate the feasibility gate',f'Validation only selects checkpoint and gate\nResult: {"FAILED — stop search" if fail else "PASSED"}',ORANGE if fail else BLUE),
           (7.5,3.2,6.2,1.35,'6  Evaluate held-out prediction accuracy',f'Tumour / overall healthy / worst subgroup MAE:\n{test["tumour"]["mae_pp"]:.2f} / {test["overall_healthy"]["mae_pp"]:.2f} / {test["worst_subgroup"]["mae_pp"]:.2f} percentage points',BLUE),
           (0.3,.85,13.4,1.35,'Conditional next stage — NOT RUN' if fail else 'Conditional next stage','Four-method optimisation comparison runs only after the gate passes.\nNo new binding-rate recommendations or search-speed claims from a failed pilot.',GRAY)]
    for x,y,w,h,title,body,color in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.12',facecolor='#f4f7fa',edgecolor=color,lw=2));ax.text(x+.15,y+h-.2,title,va='top',fontsize=12,color=color,weight='bold');ax.text(x+.15,y+h-.64,body,va='top',fontsize=10,linespacing=1.55)
    for x1,y1,x2,y2 in [(6.65,8.25,7.3,8.25),(10.6,7.4,10.6,6.95),(7.3,6.1,6.65,6.1),(3.4,5.2,3.4,4.75),(6.65,3.9,7.3,3.9),(3.4,3,3.4,2.4)]:ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops={'arrowstyle':'->','color':GRAY,'lw':2})
    ax.text(7,9.6,'Small physics-informed neural-network trial',ha='center',fontsize=19,weight='bold');save(fig,'trial_flowchart')
    rows=[]
    for label,key,limit in [('Tumour killing','tumour',5),('Overall healthy killing','overall_healthy',.25),('Worst healthy subgroup','worst_subgroup',3)]:rows.append(f'{label}: validation MAE {val[key]["mae_pp"]:.4f} pp (gate < {limit}); test MAE {test[key]["mae_pp"]:.4f} pp; test RMSE {test[key]["rmse_pp"]:.4f} pp; test R² {test[key]["r2"]:.4f}.')
    generation=sum(json.loads((OUT/f'{n}_generation.json').read_text())['wall_seconds'] for n in ['initial_2026','validation','test'])
    text=f'''SMALL PINN TRIAL: METHOD AND RESULTS

Outcome: {r['status']}.
The predeclared feasibility rule was applied before optimisation. {'The four-method search comparison was not run. No conclusion about search speed or superior binding rates can be made.' if fail else 'The feasibility gate passed.'}

What was implemented
A separate differentiable copy of all 97 quantities in the existing ODE model, with no change to biological equations. Inputs were time, four log10 binding rates and a three-condition indicator. Two connected networks (four hidden layers of 64 tanh units each) represented 0–2 h and 2–24 h. Exact initial conditions and continuity were enforced by construction. CPU float64, Adam at 0.001, gradient clipping at norm 1, 256 supervised points and 128 equation points per update. The ODE and material-balance losses were introduced after the first fifth of the time budget and ramped to full weight over the next fifth. Half the equation points were replaced every 200 steps using the largest residuals among 1,024 fresh points. Collocation points cover unsupervised binding-rate combinations too.

Data
64 training, 32 validation and 64 untouched test designs; each has three conditions and 129 saved times. This is 480 numerical ODE trajectories and 61,920 saved state vectors. Every vector contains 97 quantities. Binding rates were sampled in log space using the previous trial's coordinates; the three biological conditions and limits are unchanged. Association-rate limits remain 10^2–10^7 M^-1 s^-1; dissociation-rate limits remain 10^-5–10^2 s^-1. These broad search limits are not experimentally established. Rates were standardised with training means and standard deviations; output scales are training maximum absolute values with a 1e-8 floor. Trajectories were generated by the numerical ODE solver, not invented by an LLM. Biological inputs remain illustrative assumptions, not clinical measurements. All partitions are disjoint by rate combination. Validation/test points were never training labels.

Equation verification
{checks['states_checked']} sampled states × 97 derivatives, before and after infusion. Largest absolute difference: {checks['max_absolute_difference']:.4g}. All entries passed rtol=1e-8, atol=1e-10. Initial conditions and automatic time derivatives were checked. The reference model hash is preserved in configuration.json.

Actual training
{r['training']['steps']} updates in {r['training']['training_seconds']:.1f} seconds. Best full-physics validation checkpoint: step {r['training']['selected_checkpoint']['step']}, {selected:.1f} seconds. Checkpoints before full equation weight were excluded from PINN selection; this avoids reporting a purely supervised warm-up as a PINN. No architecture expansion or extended training followed the gate decision.

Accuracy (pp = percentage points)
'''+ '\n'.join(rows)+f'''
Healthy subgroup test MAE, neither/T1-only/T2-only/both: {', '.join(f'{z:.4f}' for z in test['subgroup_mae_pp'])} pp.
Test actually feasible: {test['actual_feasible']}/64; predicted feasible: {test['predicted_feasible']}/64; falsely feasible: {test['false_feasible']}; falsely infeasible: {test['false_infeasible']}. With few feasible broad-domain test cases, these counts do not establish safety near constraints.

Physical errors at saved validation times
Maximum scaled negative-state magnitude: {val['negative_scaled_max']:.6g}; permitted <= 0.001.
Maximum dose-normalised material imbalance: {100*val['mass_fraction_max']:.6g}%; permitted < 1%.
These maxima are sampled-time checks, not proofs between time points. Negative predictions are retained in every figure and metric.

Interpretation
'''+('This particular data-assisted PINN did not meet the agreed basic accuracy/physical-consistency criteria within the six-minute fit. Correct differential equations in a loss function did not make the learned solution accurate enough. This does not prove that all PINNs fail for this problem, nor isolate stiffness as the sole cause. Limited training time, heterogeneous timescales, linear time-output construction near rapid transients, output scaling and unbalanced residual gradients are plausible contributors requiring separate experiments. The plot of training losses can show whether introducing the equation loss coincided with deterioration; it cannot alone establish causality.\n\nThe current ODE method and regression-assisted search remain the supported workflow. No new search-quality-versus-time plot is fabricated: no optimisation search was performed after failure. A larger neural network or longer training is not justified by these results alone. A later study could investigate scaling and shorter time intervals first, but those changes were outside this trial.\n' if fail else 'The feasibility gate passed; optimisation comparisons are required before drawing a conclusion about usefulness.\n')+f'''
Timing
Trajectory generation: {generation:.1f} wall seconds (four workers). Pilot including evaluation: {r['pilot_seconds']:.1f} seconds. Sum of these stages: {generation+r['pilot_seconds']:.1f} seconds, excluding imports, equation preflight, implementation and report generation. Generation of the held-out test set is evaluation cost, not a training requirement. Full logs and per-design ODE timings are retained. No speed comparison against previous experiments is inferred from these times.

Files
trial_flowchart.svg/png; cell_killing_accuracy.svg/png; healthy_subgroup_errors.svg/png; representative_trajectories.svg/png; feasibility_gate.svg/png; training_history.svg/png. Actual datasets, checkpoint and transformations are included. The notebook displays saved results without rerunning training. The conditional three-seed comparison remains unexecuted under the approved stopping rule.
'''
    if (OUT/'diagnostics.json').exists():
        diagnostic=json.loads((OUT/'diagnostics.json').read_text())
        gr=diagnostic['checkpoint_loss_diagnostics']
        ratio=gr['equations']['parameter_gradient_norm_before_clipping']/max(gr['trajectory']['parameter_gradient_norm_before_clipping'],1e-300)
        neg=diagnostic['worst_negative']
        text+=f"\nAdditional diagnosis at the selected checkpoint\nOn a fixed diagnostic batch, the equation-loss gradient norm was {ratio:.3g} times the trajectory-loss gradient norm before clipping. Equal loss weights therefore did not give equal influence. This is evidence of loss imbalance for this implementation, not proof that it is the only cause. The largest scaled negative prediction was in {neg['state']} at {neg['time_h']:.6g} h, condition {neg['condition']}, validation design {neg['validation_design']}. Its value was {neg['value']:.6g} in reference numerical units.\n"
    (OUT/'Report.txt').write_text(text)
    images=['trial_flowchart','cell_killing_accuracy','healthy_subgroup_errors','representative_trajectories','feasibility_gate','training_history']
    blocks=''.join(f'<h2>{n.replace("_"," ").capitalize()}</h2><img src="{n}.png" alt="{n}">' for n in images)
    (OUT/'Report.html').write_text('<!doctype html><html><meta charset="utf-8"><title>Small PINN trial</title><style>body{font:17px/1.6 system-ui;max-width:1150px;margin:40px auto;padding:0 20px;color:#193047}pre{white-space:pre-wrap;font:inherit}img{width:100%}h1,h2{line-height:1.2}</style><h1>Small PINN trial</h1><pre>'+html.escape(text)+'</pre>'+blocks+'</html>')
    nb=nbformat.v4.new_notebook();nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
    nb.cells=[nbformat.v4.new_markdown_cell('# Small PINN trial for bispecific ADC binding-rate optimisation\n\nThis notebook displays actual saved results. The numerical implementation is in `pinn_small_trial/`; rerunning this notebook does not retrain the network.\n\n**Outcome: '+r['status'].replace('_',' ')+'.** The agreed stopping rule determines whether optimisation proceeds.'),nbformat.v4.new_code_cell("from pathlib import Path\nimport json\nfrom IPython.display import display, Image\nROOT = Path.cwd()\nif not (ROOT / 'output/pinn_small_trial').exists():\n    ROOT = ROOT.parent\nOUT = ROOT / 'output/pinn_small_trial'\nprint((OUT / 'Report.txt').read_text())")]
    for n in images:nb.cells.append(nbformat.v4.new_code_cell(f"display(Image(filename=str(OUT / '{n}.png')))"))
    path=OUT.parents[1]/'repo/Bispecific_ADC_PINN_Small_Trial.ipynb';nbformat.write(nb,path)
    readme='''Reproduce from the Acaddrug project root with the existing .venv Python:
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python repo/pinn_small_trial/data.py
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python repo/pinn_small_trial/train.py
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python repo/pinn_small_trial/report.py

Inspect pilot_results.json. If status is stopped_at_feasibility_gate, do not run optimisation or claim a four-method comparison. This delivered experiment followed that conditional path. Raw trajectories are cached in NPZ files; remove only the trial datasets if intentionally regenerating. Keep the original model file unchanged. The checkpoint contains state_dict, selection record and timing; scaling is also saved in NPZ. CPU scheduling can change the step count within the wall-time cap, so bit-identical training is not promised.
'''
    (OUT/'README.txt').write_text(readme)
    print(text)
if __name__=='__main__':main()
