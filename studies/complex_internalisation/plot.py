from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parents[2]/'output/complex_internalisation'
r=json.loads((D/'results.json').read_text()); rows=r[1:]
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
colors=['#246b94','#d67525']; x=np.arange(3);w=.34
labels=['Reference','Condition A','Condition B']
fig,axs=plt.subplots(1,3,figsize=(17,6.5));fig.subplots_adjust(bottom=.28,top=.77,wspace=.3)
fig.suptitle('Does binding selected with one common internalisation rate still work?',fontsize=20,fontweight='bold')
for ax,metric,title in zip(axs,['tumour','healthy','subgroup'],['Tumour cells','All healthy cells: weighted average','H12: healthy cells with both targets']):
 for j,(key,label) in enumerate([('transferred','Binding selected assuming 2 / 2 / 2 per hour'),('reoptimised','Binding re-optimised for separate rates')]):
  vals=[v[key][metric] for v in rows];bars=ax.bar(x+(j-.5)*w,vals,w,color=colors[j],label=label)
  ax.bar_label(bars,labels=[f'{v:.2f}%' for v in vals],padding=4+12*j,fontsize=10)
 ax.set(xticks=x,xticklabels=labels,title=title,ylabel='Cells killed after 24 hours (%)')
 if metric!='tumour':
  cap=.95 if metric=='healthy' else 10
  ax.axhline(cap,ls='--',color='#b23535');ax.set_ylim(0,max(cap,max(v[k][metric] for v in rows for k in ['transferred','reoptimised']))*1.3)
 else:ax.set_ylim(0,100)
handles,leg=axs[0].get_legend_handles_labels();fig.legend(handles,leg,loc='upper center',bbox_to_anchor=(.5,.92),ncol=2,frameon=False)
fig.text(.5,.12,'T1–ADC / T2–ADC / T1T2–ADC (h⁻¹): Reference 0.2 / 4 / 2; A 0.16 / 3.2 / 1.6; B 0.24 / 4.8 / 2.4.',ha='center')
fig.text(.5,.07,'Only internalisation changes. Linker half-life = 96 h; payload exchange = 0.10 per hour; tumour / healthy C₅₀ = 4 / 4 µM.',ha='center',fontsize=11)
fig.text(.5,.02,'Illustrative inputs. Dashed lines: assumed limits of 0.95% overall healthy killing and 10% within each healthy subgroup.',ha='center',fontsize=11)
for ext in ('png','svg'):fig.savefig(D/f'01_common_vs_separate_internalisation.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(13,6));fig.subplots_adjust(bottom=.29,top=.82,wspace=.25)
fig.suptitle('Selected binding affinities when each complex has its own internalisation rate',fontsize=17,fontweight='bold')
for j,ax in enumerate(axs):
 vals=[10**(v['reoptimised']['theta'][2*j+1]-v['reoptimised']['theta'][2*j])*1e6 for v in r]
 bars=ax.bar(range(4),vals,color=['#246b94']+['#d67525']*3)
 ax.bar_label(bars,labels=[f'{v:,.1f}' for v in vals],padding=5,fontsize=11)
 ax.set(xticks=range(4),xticklabels=['Common\n2 / 2 / 2']+labels,title=f'Target {j+1}',ylabel='Dissociation constant Kᴅ (µM)',ylim=(0,max(vals)*1.25))
fig.text(.5,.16,'T1 / T2 / dual (h⁻¹): Common 2 / 2 / 2; Reference 0.2 / 4 / 2; A 0.16 / 3.2 / 1.6; B 0.24 / 4.8 / 2.4.',ha='center',fontsize=10)
fig.text(.5,.1,'Kᴅ = dissociation rate / association rate. Smaller Kᴅ means stronger binding.',ha='center')
fig.text(.5,.04,'Best-found highest-tumour-killing designs under the same healthy-cell limits; not a proven unique optimum.',ha='center',fontsize=11)
for ext in ('png','svg'):fig.savefig(D/f'02_selected_binding_affinities.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)
lines=['# Common versus complex-specific internalisation','', 'Only the internalisation row of the supplied table is varied. All other inputs stay at the reference values. The common-rate comparison assumes 2 h⁻¹ for all three complexes. This is an illustrative comparison within the extended model, not a rerun of the original paper.','','Optimise the four binding rates under the common-rate assumption, transfer that same design into each unequal-rate condition, and then optimise binding separately for each condition. Each search uses three objectives (increase tumour killing, reduce overall healthy killing, reduce the most affected healthy subgroup), with 0.95% overall and 10% subgroup limits. For plotting, select the highest-tumour-killing feasible design found.','','## Results','', '| Condition | Design | Tumour kill (%) | Overall healthy (%) | H0 (%) | H1 (%) | H2 (%) | H12 (%) |','|---|---|---:|---:|---:|---:|---:|---:|']
for v in r:
 for key in ['transferred','reoptimised']:
  a=v[key];lines.append('| '+v['label']+' | '+key+' | '+' | '.join(f'{n:.5f}' for n in [a['tumour'],a['healthy']]+a['groups'])+' |')
lines+=['','## Four rates selected together','','| Condition | k_on1 (M⁻¹ s⁻¹) | k_off1 (s⁻¹) | k_on2 (M⁻¹ s⁻¹) | k_off2 (s⁻¹) |','|---|---:|---:|---:|---:|']
for v in r:lines.append('| '+v['label']+' | '+' | '.join(f'{10**n:.8g}' for n in v['reoptimised']['theta'])+' |')
lines+=['','Two independent searches per condition, each with 512 evolutionary evaluations requested and local refinement, followed by tighter ODE evaluation. These results establish what the search found within its bounds; differences between selected rates alone do not prove that a unique optimum changed. Similar outcomes may be achievable with multiple rate combinations.','']
(D/'Explanation_and_results.md').write_text('\n'.join(lines))
fig,axs=plt.subplots(1,4,figsize=(18,6));fig.subplots_adjust(bottom=.26,top=.76,wspace=.32)
fig.suptitle('How each healthy subgroup responds before and after re-optimising binding',fontsize=18,fontweight='bold')
for g,(ax,name) in enumerate(zip(axs,['H0: neither target','H1: T1 only','H2: T2 only','H12: both targets'])):
 for j,(key,label) in enumerate([('transferred','Binding selected assuming equal internalisation'),('reoptimised','Binding re-optimised for separate rates')]):
  vals=[v[key]['groups'][g] for v in rows];bars=ax.bar(x+(j-.5)*w,vals,w,color=colors[j],label=label)
  ax.bar_label(bars,labels=[f'{v:.3f}' if g<3 else f'{v:.2f}' for v in vals],padding=4+12*j,fontsize=9)
 ax.set(xticks=x,xticklabels=['Reference','A','B'],title=name,ylabel='Cells killed after 24 hours (%)')
 top=max(v[k]['groups'][g] for v in rows for k in ['transferred','reoptimised'])
 ax.set_ylim(0,max(top,10 if g==3 else 0)*1.3)
 if g==3:ax.axhline(10,color='#b23535',ls='--')
handles,leg=axs[0].get_legend_handles_labels();fig.legend(handles,leg,loc='upper center',bbox_to_anchor=(.5,.91),ncol=2,frameon=False)
fig.text(.5,.13,'Internalisation T1–ADC / T2–ADC / T1T2–ADC (h⁻¹): Reference 0.2 / 4 / 2; A 0.16 / 3.2 / 1.6; B 0.24 / 4.8 / 2.4.',ha='center',fontsize=11)
fig.text(.5,.07,'Each panel has its own vertical scale. All other biological inputs remain fixed at the reference values.',ha='center',fontsize=11)
for ext in ('png','svg'):fig.savefig(D/f'03_all_healthy_subgroups.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)
