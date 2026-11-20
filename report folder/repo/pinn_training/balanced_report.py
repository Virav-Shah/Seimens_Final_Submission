from pathlib import Path
import sys
_SUBMISSION_ROOT = Path(__file__).resolve().parents[3]
__file__ = str(_SUBMISSION_ROOT / 'repo/pinn_training/balanced_report.py')
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(_SUBMISSION_ROOT / "repo"))
"""Report the revised PINN and matched data-only control without claiming search gains."""
import os
os.environ.setdefault('MPLBACKEND','Agg')
import json,html,base64,io,contextlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import nbformat
from data import OUT as OLD,TIMES
from balanced_trial import OUT
from train import aggregate,endpoints
plt.rcParams.update({'font.size':11,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
COL={'PINN':'#166c99','Supervised':'#df7b27','Earlier':'#89939c'}
NOTE='Simulated ODE data; fixed illustrative biological inputs. One training seed; no clinical measurements.'
def save(fig,name):
    fig.text(.5,.012,NOTE,ha='center',fontsize=9,color='#667681');fig.savefig(OUT/f'{name}.png',dpi=180,bbox_inches='tight');fig.savefig(OUT/f'{name}.svg',bbox_inches='tight');plt.close(fig)
def main():
    r=json.loads((OUT/'results.json').read_text());methods=r['methods'];old=json.loads((OLD/'pilot_results.json').read_text());cfg=json.loads((OUT/'configuration.json').read_text())
    keys=['tumour','overall_healthy','worst_subgroup'];titles=['Tumour killing','Overall healthy killing','Worst healthy-subgroup killing']
    fig,axs=plt.subplots(1,3,figsize=(13,4.7))
    for ax,key,title in zip(axs,keys,titles):
        vals=[methods[k]['fresh_test'][key]['mae_pp'] for k in ['Supervised','PINN']]
        bars=ax.bar(['Data only','With ODE loss'],vals,color=[COL['Supervised'],COL['PINN']]);ax.bar_label(bars,fmt='%.3f',padding=4);ax.set_ylim(0,max(vals)*1.3);ax.set(title=title,ylabel='Mean absolute error (percentage points)')
    fig.suptitle('Does the equation loss help? Same revised network and training-time allowance',fontsize=15);fig.tight_layout(rect=[0,.06,1,.92]);save(fig,'fresh_test_comparison')
    fig,axs=plt.subplots(1,3,figsize=(13.5,4.7))
    for ax,key,title in zip(axs,keys,titles):
        vals=[old['test'][key]['mae_pp'],methods['Supervised']['previous_test_paired'][key]['mae_pp'],methods['PINN']['previous_test_paired'][key]['mae_pp']]
        bars=ax.bar(['Earlier\nPINN','Revised\ndata only','Revised\nPINN'],vals,color=[COL['Earlier'],COL['Supervised'],COL['PINN']]);ax.bar_label(bars,fmt='%.3f',padding=4);ax.set_ylim(0,max(vals)*1.3);ax.set(title=title,ylabel='Mean absolute error (percentage points)')
    fig.suptitle('Paired comparison on the previous 64 test designs (retrospective)',fontsize=15);fig.tight_layout(rect=[0,.06,1,.92]);save(fig,'paired_previous_test')
    fig,axs=plt.subplots(1,3,figsize=(13.5,4.7))
    for kind in ['Supervised','PINN']:
        d=np.load(OUT/f'{kind}_fresh_test_predictions.npz');a=aggregate(endpoints(d['actual'][:,:,-1]));p=aggregate(endpoints(d['prediction'][:,:,-1]))
        for j,ax in enumerate(axs):ax.scatter(a[:,j],p[:,j],s=19,alpha=.65,color=COL[kind],label='Data only' if kind=='Supervised' else 'PINN')
    for j,ax in enumerate(axs):
        lim=max(ax.get_xlim()[1],ax.get_ylim()[1]);ax.plot([0,lim],[0,lim],ls='--',color=COL['Earlier']);ax.set(xlabel='ODE calculation (%)',ylabel='Neural prediction (%)',title=titles[j])
    axs[0].legend();fig.suptitle('Predictions on 64 newly generated test designs',fontsize=15);fig.tight_layout(rect=[0,.06,1,.92]);save(fig,'prediction_accuracy')
    fig,axs=plt.subplots(1,2,figsize=(12,4.8))
    for kind in ['Supervised','PINN']:
        h=json.loads((OUT/f'{kind}_training.json').read_text());axs[0].plot([v['seconds'] for v in h],[v['validation_score'] for v in h],color=COL[kind],label=kind)
        vals=methods[kind]['fresh_test']['subgroup_mae_pp'];ix=np.arange(4)+(-.18 if kind=='Supervised' else .18);bars=axs[1].bar(ix,vals,width=.35,color=COL[kind],label=kind);axs[1].bar_label(bars,fmt='%.2f',fontsize=8,padding=3)
    axs[0].set(xlabel='Training wall time (s)',ylabel='Normalised validation error',yscale='log',title='Training behaviour');axs[0].legend()
    axs[1].set(xticks=np.arange(4),xticklabels=['Neither','T1 only','T2 only','Both'],ylabel='Mean absolute error (percentage points)',title='Fresh-test healthy-subgroup errors');axs[1].legend();axs[1].set_ylim(0,axs[1].get_ylim()[1]*1.2)
    fig.tight_layout(rect=[0,.06,1,1]);save(fig,'training_and_subgroups')
    d=np.load(OUT/'PINN_fresh_test_predictions.npz');dn=np.load(OUT/'Supervised_fresh_test_predictions.npz');a=aggregate(endpoints(d['actual'][:,:,-1]));p=aggregate(endpoints(d['prediction'][:,:,-1]));idx=int(np.argsort(np.mean(abs(a-p)/np.array([100,.95,10]),axis=1))[32])
    fig,axs=plt.subplots(2,3,figsize=(13,8));cols=[9,10,11,21,87];names=['T1-bound ADC: B1','T2-bound ADC: B2','Doubly bound ADC: D','Intracellular payload','Accumulated killing effect'];units=['10⁻¹⁰ mol/dm²']*3+['µM','dimensionless']
    for j,ax in enumerate(axs.ravel()[:5]):
        ax.plot(TIMES,d['actual'][idx,0,:,cols[j]],color='black',label='ODE');ax.plot(TIMES,dn['prediction'][idx,0,:,cols[j]],color=COL['Supervised'],ls=':',label='Data only');ax.plot(TIMES,d['prediction'][idx,0,:,cols[j]],color=COL['PINN'],ls='--',label='PINN');ax.set_xscale('symlog',linthresh=.001);ax.set_xlim(0,24);ax.set(xlabel='Time (h)',ylabel=units[j],title=names[j]);ax.axvline(2,color=COL['Earlier'],ls=':',lw=1)
    axs[0,0].legend();axs[1,2].axis('off');axs[1,2].text(0,.9,f'Example test design {idx}\n\nSelected by median PINN endpoint error.\nTumour cells; reference condition.\nDotted vertical line: infusion ends.\n\nAccounting and nonnegativity are\nguaranteed by construction; trajectory\naccuracy must still be checked.',va='top',linespacing=1.6)
    fig.suptitle('Does the revised representation reproduce the trajectories?',fontsize=15);fig.tight_layout(rect=[0,.05,1,.94]);save(fig,'trajectories')
    fig,ax=plt.subplots(figsize=(13,8));ax.axis('off');ax.set(xlim=(0,13),ylim=(0,8))
    boxes=[(.2,5.7,5.8,1.2,'Same numerical training data','64 binding-rate designs × 3 biological conditions\n32 separate validation designs; log-scaled rate inputs'),(7,5.7,5.8,1.2,'New representation','12 continuous time intervals; nonnegative quantities\nExact antibody/payload accounting, including clearance'),(7,3.6,5.8,1.2,'Two matched training runs','Same network, seed, data and 6-minute allowance\nData only versus data + balanced equation losses'),(.2,3.6,5.8,1.2,'Balance nine equation groups','Update weights from gradient norms every 25 steps\nCompare binding, internalisation, payload and killing'),(.2,1.3,12.6,1.3,'Freeze models, then generate a fresh test set','64 unseen test designs, then ODE checks of selected binding-rate combinations\nReport prediction accuracy and equation agreement separately')]
    for x,y,w,h,head,body in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.1',facecolor='#f4f8fb',edgecolor=COL['PINN'],lw=2));ax.text(x+.14,y+h-.15,head,va='top',weight='bold',fontsize=12);ax.text(x+.14,y+h-.58,body,va='top',fontsize=10,linespacing=1.5)
    for x,y,xx,yy in [(6.2,6.3,6.8,6.3),(9.9,5.5,9.9,5.0),(6.8,4.2,6.2,4.2),(3,3.4,3,2.8)]:ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops={'arrowstyle':'->','lw':2,'color':COL['Earlier']})
    ax.text(6.5,7.6,'Revised PINN: changes tested and how they were assessed',ha='center',fontsize=17,weight='bold');save(fig,'flowchart')
    audit=json.loads((OUT/'dynamics_audit.json').read_text())
    candidate=json.loads((OUT/'candidate_audit.json').read_text())
    fig,ax=plt.subplots(figsize=(8,4.8));ks=['Supervised','PINN'];passed=[candidate['methods'][k]['metrics']['actual_feasible'] for k in ks];failed=[16-n for n in passed]
    ax.bar(['Data only','PINN'],passed,color='#238469',label='Meets limits in ODE calculation');ax.bar(['Data only','PINN'],failed,bottom=passed,color='#cc6540',label='Fails at least one ODE limit')
    for i,(yes,no) in enumerate(zip(passed,failed)):
        ax.text(i,yes/2,str(yes),ha='center',va='center',color='white',weight='bold');ax.text(i,yes+no/2,str(no),ha='center',va='center',color='white',weight='bold')
    ax.set(ylim=(0,20),ylabel='Number of proposed combinations',title='All 16 proposals per method were predicted to meet the limits');ax.legend(fontsize=9,loc='upper center');fig.tight_layout(rect=[0,.06,1,1]);save(fig,'candidate_verification')
    fig,axs=plt.subplots(1,3,figsize=(13,4.5))
    for ax,key,title in zip(axs,keys,titles):
        values=[candidate['paired_union']['methods'][k]['metrics'][key]['mae_pp'] for k in ks];bars=ax.bar(['Data only','PINN'],values,color=[COL[k] for k in ks]);ax.bar_label(bars,fmt='%.3f',padding=3);ax.set_ylim(0,max(values)*1.3);ax.set(title=title,ylabel='Mean absolute error (percentage points)')
    fig.suptitle('Both models evaluated on the same union of proposed candidates',fontsize=15);fig.tight_layout(rect=[0,.06,1,.92]);save(fig,'candidate_paired_accuracy')
    lines=['REVISED PINN TRIAL: RESULTS AND INTERPRETATION','',"Coarse validation gate passed; ODE candidate checks remain necessary. One training seed; matched six-minute fits.",'','Fresh test MAE (percentage points):']
    for k,title in zip(keys,titles):lines.append(f"{title}: data-only {methods['Supervised']['fresh_test'][k]['mae_pp']:.4f}; PINN {methods['PINN']['fresh_test'][k]['mae_pp']:.4f}.")
    lines+=['','What changed','The four association/dissociation rates remain log-scaled inputs. Biological equations, broad search limits, dose, three illustrative conditions, and healthy-cell limits are unchanged. The earlier two large time intervals were replaced by 12 connected intervals: '+str(cfg['knots_h'])+'. A shared knot network and local correction network ensure continuity. Both have three hidden layers of 64 tanh units.','',
    'Antibody is distributed among intact ADC, payload-free antibody and cleared antibody using nonnegative fractions. Payload still attached to intact ADC is counted in both inventories. The remaining payload is distributed among extracellular payload, intracellular payload and payload loss. Thus administered antibody equals antibody remaining plus antibody lost; administered payload equivalents equal attached plus free plus lost payload. This respects the existing model normalisation, not a new drug-to-antibody ratio. Receptors remain nonnegative; target-absent binding states are exactly zero. Cumulative release, killing effect and exposure are integrals of nonnegative piecewise-linear rates.','',
    'These guarantees are properties built into the representation, not evidence that its kinetics are correct. Cleared-antibody and cleared-payload inventories are nonnegative but their monotonicity is not hard-enforced; their ODE residuals still need to be satisfied. The instantaneous killing-rate upper bound is not separately hard-enforced.','',
    'Training and fair control','The revised PINN groups residuals into free antibody, free targets, surface complexes, internalised complexes, active payload, loss accounts, released payload, killing effect and exposure. Every 25 updates, each group weight is adjusted towards the supervised-loss gradient norm using an exponential moving average in log space. Ratios are bounded at 1e-30 and 1e6. Physics is introduced after 20% of the time budget and ramped over the next 20%. Training uses 256 trajectory points, endpoint labels from 16 designs, and 64 equation points per update. The supervised objective is scaled trajectory error plus cell-killing endpoint error. These are prediction-training errors, not changes to the downstream optimisation objectives.','',
    'The matched control has the same architecture, positivity, accounting, continuity, cumulative construction, seed, training labels and endpoint loss, but no equation residual. Thus the comparison separates the added equation-loss contribution from the architectural changes. Equal wall time permits different numbers of updates; this is a computational-budget comparison, not an equal-update comparison.','',
    'The original 64 training and 32 validation designs are reused. State scales and input statistics use only training data. Checkpoints are chosen by validation endpoint errors; PINN checkpoints must have reached full physics weighting. After both checkpoints are fixed, 64 new test designs (seed 66021) are generated by the unchanged tight-tolerance ODE solver. The previous 64 test designs are reported separately for retrospective paired comparison with the earlier PINN. They are not described as fresh data. No clinical data or LLM-invented labels are used.','',
    'Feasibility and physical checks']
    for kind in ['Supervised','PINN']:
        q=methods[kind];z=q['fresh_test'];lines += [f"{kind}: validation gate {q['passes']}; gate details {q['gate']}. Training {q['seconds']:.1f} s, {q['steps']} updates.",f"Fresh-test maximum negative scaled state {z['negative_scaled_max']:.3g}; maximum dose-normalised accounting error {z['mass_fraction_max']:.3g}. Feasible designs: actual {z['actual_feasible']}, predicted {z['predicted_feasible']}; falsely feasible {z['false_feasible']}, falsely infeasible {z['false_infeasible']}.",f"Fresh-test subgroup MAE (neither, T1 only, T2 only, both): {z['subgroup_mae_pp']}."]
    pin=[methods['PINN']['fresh_test'][k]['mae_pp'] for k in keys];sup=[methods['Supervised']['fresh_test'][k]['mae_pp'] for k in keys];wins=sum(a<b for a,b in zip(pin,sup))
    lines+=['','What can be concluded',f'The revised PINN has lower fresh-test endpoint MAE than its matched data-only control on {wins} of the three aggregate outcomes. This is a one-seed observation; it does not establish general superiority. The old-to-new comparison changes several components together, so it cannot isolate the benefit of any one architectural change. The matched data-only comparison is the relevant comparison for the added equation loss.',
    'Material accounting and negative-state problems from the earlier trial are removed by construction. Whether binding, internalisation and payload dynamics are predicted accurately must be judged separately using trajectory and endpoint errors. Zero falsely feasible classifications are not sufficient evidence of reliability when few truly feasible designs are in a broad-domain test set.']
    if not methods['PINN']['passes']:lines+=['The revised PINN still fails the original validation gate. No binding-rate optimisation is run with it. Further training or architecture expansion was not silently performed.']
    else:lines+=['The original coarse validation gate passes. This alone does not establish optimisation usefulness: fresh-test accuracy, matched-control performance and ODE-confirmed candidate results must be considered before any deployment.']
    lines+=['No search-speed or superior-affinity claim is made from prediction metrics. The original ODE solver remains the reference. A future search experiment would need independently verified candidate combinations and a fair end-to-end runtime comparison.',f"Total experiment wall time, including fresh-test generation and evaluation: {r['wall_seconds']:.1f} s. Cached training-trajectory generation cost is not included in this new-run time. Original model unchanged: {r['original_model_unchanged']}."]
    lines+=['','Targeted candidate check',f"A shared pool of 1,024 Sobol points and 256 local perturbations was used. Each model proposed 16 combinations: eight high-tumour predicted-feasible combinations and eight near a predicted constraint (with deterministic fallback if needed). These were recomputed using the tight ODE solver; {candidate['unique_ode_designs']} unique designs were evaluated. This is a one-shot audit, not a multi-round optimisation benchmark."]
    for kind in ['PINN','Supervised']:
        z=candidate['methods'][kind]['metrics'];paired=candidate['paired_union']['methods'][kind]['metrics'];ad=audit[kind]
        lines+=[f"{kind}: {z['predicted_feasible']}/16 predicted feasible, {z['actual_feasible']}/16 actually feasible, {z['false_feasible']} falsely feasible. On the shared candidate union, tumour/overall healthy/worst-subgroup MAE were {paired['tumour']['mae_pp']:.3f}/{paired['overall_healthy']['mae_pp']:.3f}/{paired['worst_subgroup']['mae_pp']:.3f} percentage points.",f"{kind}: recalculating killing from predicted payload on 16 fixed audit designs (1,025 time points) differed from the network's own killing predictions by {ad['payload_to_killing_consistency']['aggregate_mean_difference_pp']} percentage points on average, for tumour/overall healthy/worst subgroup. Lost-material inventories had a negative derivative in {ad['loss_accounts_negative_derivative_count']} of 512 sampled derivatives. These are unresolved dynamic-consistency errors despite exact total accounting."]
    lines+=['The PINN reduces sampled ODE-residual errors relative to the matched data-only network, especially in the surface-binding equations. However, some equation groups still have substantial errors; see dynamics_audit.json for the exact normalisation and all groups. The equation loss therefore contributes to equation agreement without establishing accurate full ODE solutions.','The representative trajectory plot also shows that the PINN can substantially underestimate bound complexes and intracellular payload while predicting accumulated killing more closely. Lower residuals and accurate total accounting do not exclude an incorrect distribution of material between states.', 'The revised PINN passes the coarse validation gate and fixes earlier accounting failures. It does not improve all prediction outcomes, and some candidate predictions fail ODE limits. Retain ODE verification and do not adopt it as an ODE replacement. These small results do not establish that the PINN beats our existing Gaussian-process-assisted search in time or optimisation quality.']
    text='\n\n'.join(lines);(OUT/'Report.txt').write_text(text)
    names=['flowchart','fresh_test_comparison','paired_previous_test','prediction_accuracy','training_and_subgroups','trajectories','candidate_verification','candidate_paired_accuracy']
    images=''.join(f'<h2>{n.replace("_"," ")}</h2><img src="{n}.png">' for n in names)
    (OUT/'Report.html').write_text('<!doctype html><meta charset="utf-8"><title>Revised PINN trial</title><style>body{font:17px/1.55 system-ui;max-width:1150px;margin:40px auto;padding:20px;color:#193047}pre{white-space:pre-wrap;font:inherit}img{width:100%}</style><h1>Revised PINN trial</h1><pre>'+html.escape(text)+'</pre>'+images)
    nb=nbformat.v4.new_notebook();nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
    nb.cells=[nbformat.v4.new_markdown_cell('# Revised PINN: balanced losses and physical accounting\n\nActual saved results from the completed scripts. This notebook displays results; it does not rerun training. Embedded outputs have unset execution counts because notebook-kernel socket binding is restricted in this environment.'),nbformat.v4.new_code_cell("from pathlib import Path\nfrom IPython.display import display, Image\nROOT=Path.cwd()\nif not (ROOT/'output/pinn_balanced_trial').exists(): ROOT=ROOT.parent\nOUT=ROOT/'output/pinn_balanced_trial'\nprint((OUT/'Report.txt').read_text())")]
    nb.cells[1].outputs=[nbformat.v4.new_output('stream',name='stdout',text=text)]
    for name in names:
        cell=nbformat.v4.new_code_cell(f"display(Image(filename=str(OUT/'{name}.png')))");cell.outputs=[nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode((OUT/f'{name}.png').read_bytes()).decode(),'text/plain':name})];nb.cells.append(cell)
    ns={}
    for cell in nb.cells:
        if cell.cell_type=='code':
            with contextlib.redirect_stdout(io.StringIO()):exec(compile(cell.source,'notebook-cell','exec'),ns)
    nbformat.validate(nb);nbformat.write(nb,OUT.parents[1]/'repo/Bispecific_ADC_PINN_Balanced_Trial.ipynb')
    print(text)
if __name__=='__main__':main()
