from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from repo.mechanism_comparison.study import *
ROOT=Path(__file__).resolve().parents[2]
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from joblib import Parallel,delayed,parallel_config
DEST=ROOT/'output/presentation_revision';PLOTS=DEST/'plots';DATA=DEST/'data'
records=json.loads((OUT/'mechanism_results.json').read_text());lookup={r['case']:r for r in records}
ref=lookup['reference']['fixed']['theta']
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
BLUE='#2375ad';ORANGE='#e67923';GREEN='#2f846b'
def save(fig,name):
 for ext in ['png','svg']:fig.savefig(PLOTS/f'{name}.{ext}',dpi=200,bbox_inches='tight')
 plt.close(fig)
def compute(task):
 rate,mult=task;s=replace(BIOLOGY,ki=(.2,4.,float(rate)));x=np.array(ref).copy();x[[1,3]]+=np.log10(mult)
 r=evaluate(x,s,True)
 return dict(internalisation_h=float(rate),dissociation_multiplier=float(mult),**r)
def main():
 path=DATA/'internalisation_sweep.json'
 rates=np.unique(np.r_[np.geomspace(.2,4,31),2.])
 gridrates=np.unique(np.r_[np.geomspace(.2,4,19),2.]);multipliers=np.geomspace(.25,4,15)
 if path.exists():data=json.loads(path.read_text())
 else:
  tasks=list(dict.fromkeys([(float(r),1.) for r in rates]+[(float(r),float(m)) for r in gridrates for m in multipliers]))
  with parallel_config(backend='loky',inner_max_num_threads=1):data=Parallel(n_jobs=4)(delayed(compute)(t) for t in tasks)
  path.write_text(json.dumps(data,indent=2))
 sweep=sorted([r for r in data if r['dissociation_multiplier']==1],key=lambda r:r['internalisation_h'])
 fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
 for ax,key,title,limit in zip(axes,['tumour','healthy','subgroup'],['Tumour cells','All healthy cells: weighted average','Most affected healthy subgroup'],[None,1,10]):
  ax.plot([r['internalisation_h'] for r in sweep],[r[key] for r in sweep],color=BLUE,lw=2,label='Same binding rates throughout')
  vals=[lookup[k]['reoptimised'][key] for k in ['dual_slow','reference','dual_fast']]
  ax.scatter([.2,2,4],vals,color=ORANGE,marker='D',s=60,zorder=4,label='Binding rates re-optimised\nat these three points only')
  ax.set(xscale='log',xlabel='Dually bound internalisation rate (h⁻¹)',ylabel='Cell kill at 24 h (%)',title=title)
  ax.set_xticks([.2,.5,1,2,4],['0.2','0.5','1','2','4']);ax.set_ylim(bottom=0)
  if limit:ax.axhline(limit,color='#b83232',ls='--',lw=1)
 axes[0].legend(fontsize=9);axes[1].set_ylim(0,1.1);axes[0].set_ylim(0,85)
 fig.supxlabel('Illustrative reference inputs. Singly bound internalisation stays at 0.2 and 4 h⁻¹. Re-optimisation limits: 0.95% overall and 10% per healthy subgroup.',fontsize=9)
 save(fig,'01_internalisation_sweep')
 # Heatmaps are a restricted slice, not a four-rate optimisation.
 grid={(r['internalisation_h'],r['dissociation_multiplier']):r for r in data}
 arrays={key:np.array([[grid[(float(r),float(m))][key] for r in gridrates] for m in multipliers]) for key in ['tumour','healthy','subgroup']}
 fig,axes=plt.subplots(1,2,figsize=(12,5.6),layout='constrained')
 for ax,key,title,cmap in zip(axes,['tumour','subgroup'],['Tumour cell kill (%)','Maximum healthy-subgroup kill (%)'],['viridis','magma']):
  im=ax.pcolormesh(gridrates,multipliers,arrays[key],shading='nearest',cmap=cmap,vmin=0,vmax=85)
  ax.contour(gridrates,multipliers,arrays['subgroup'],levels=[10],colors=['cyan'],linewidths=2)
  if arrays['healthy'].min()<.95<arrays['healthy'].max():ax.contour(gridrates,multipliers,arrays['healthy'],levels=[.95],colors=['white'],linestyles='--',linewidths=2)
  ax.plot(2,1,'wo',mec='black',ms=7)
  ax.set(xscale='log',yscale='log',xlabel='Dually bound internalisation rate (h⁻¹)',ylabel='Multiplier on both dissociation rates',title=title)
  ax.set_xticks([.2,.5,1,2,4],['0.2','0.5','1','2','4']);ax.set_yticks([.25,.5,1,2,4],['0.25','0.5','1','2','4'])
  fig.colorbar(im,ax=ax)
 fig.supxlabel('Below the cyan line: subgroup kill exceeds 10%. Dot: reference. All sampled points have overall healthy kill below 0.95%.\nAssociation rates remain fixed. Multipliers >1 mean faster dissociation from the first-bound complexes. This is a two-parameter slice, not an optimised range.',fontsize=9)
 save(fig,'02_internalisation_binding_sweep')
 # Four rates in unambiguous units: show dissociation constants without implying independence.
 sel=[lookup[k]['reoptimised'] for k in ['dual_slow','reference','dual_fast']]
 x=np.arange(3);fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
 for i,ax in enumerate(axes):
  kd=[1e6*10**(r['theta'][2*i+1]-r['theta'][2*i]) for r in sel]
  bars=ax.bar(x,kd,color=BLUE if i==0 else ORANGE,width=.55);ax.bar_label(bars,fmt='%.0f',padding=4)
  ax.set(xticks=x,xticklabels=['0.2','2 (reference)','4'],xlabel='Dually bound internalisation rate (h⁻¹)',ylabel='Dissociation constant Kᴅ (µM)',title=f'Target {i+1} binding',ylim=(0,max(kd)*1.2))
 fig.supxlabel('Lower Kᴅ means stronger binding. Both target values belong to the same four-rate design; they cannot be chosen independently. Synthetic inputs.',fontsize=9)
 save(fig,'03_internalisation_selected_affinities')
 # Explain the input parameters without confusing instantaneous rates with endpoints.
 fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
 t=np.linspace(0,24,200);p=np.linspace(0,16,200)
 for e,col in zip([.08,.10,.12],[GREEN,BLUE,ORANGE]):axes[0].plot(t,np.exp(-e*t),label=f'Exchange rate {e:.2f} h⁻¹',color=col,lw=2)
 axes[0].set(xlabel='Time (h)',ylabel='Intracellular free payload (µM)',title='What the exchange rate means');axes[0].legend(fontsize=9)
 for c,col in zip([3.2,4,4.8],[GREEN,BLUE,ORANGE]):axes[1].plot(p,8*p/(c+p),label=f'C₅₀ = {c:g} µM',color=col,lw=2)
 axes[1].axhline(4,color='gray',ls=':');axes[1].set(xlabel='Intracellular free payload (µM)',ylabel='Instantaneous kill rate (% of surviving cells/h)',title='What payload C₅₀ means',ylim=(0,8));axes[1].legend(fontsize=9)
 fig.supxlabel('Left: isolated exchange illustration, starting at 1 µM inside and zero outside, with outside held at zero and no other processes.\nRight: concentration-dependent kill rate, not the percentage killed at 24 hours. The maximum rate is 8% of surviving cells per hour.',fontsize=9)
 save(fig,'04_input_parameters_explained')
 # Simple display of the fair objective comparison.
 policy=json.loads((OUT/'policy_results.json').read_text());a,c=policy[0],policy[2]
 fig,axes=plt.subplots(1,3,figsize=(12,4.8),layout='constrained')
 for ax,key,title in zip(axes,['tumour','healthy','subgroup'],['Tumour kill','Overall healthy kill','Healthy cells expressing both targets']):
  vals=[r[key] for r in [a,c]];bars=ax.bar([0,1],vals,color=[BLUE,ORANGE],width=.6);ax.bar_label(bars,fmt='%.2f%%',padding=4)
  ax.set(xticks=[0,1],xticklabels=['Overall limit\nonly','Overall + subgroup\nlimits'],title=title,ylabel='Cell kill at 24 h (%)',ylim=(0,100 if key!='healthy' else 1.2))
  if key=='healthy':ax.axhline(1,color='#b83232',ls='--',lw=1)
  if key=='subgroup':ax.axhline(10,color='#b83232',ls='--',lw=1)
 fig.supxlabel('Same reference biological inputs and 0.95% overall limit. Added requirement: no healthy subgroup above 10%.\nBars show the highest-tumour candidate found under each set of requirements. Adding a subgroup objective alone does not force this choice.',fontsize=9)
 save(fig,'05_overall_average_vs_subgroup_protection')
 # One figure isolates nominal mechanisms without mixing cell-kill definitions.
 families=[('Premature payload release',['linker_stable','reference','linker_short'],['None','96 h half-life','48 h half-life']),('Free-payload movement',['movement_none','reference','movement_fast'],['0','0.1 h⁻¹','0.2 h⁻¹']),('Recycling',['recycling_none','reference','recycling_fast'],['0','0.6 h⁻¹','1.2 h⁻¹'])]
 fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
 for j,(title,keys,labels) in enumerate(families):
  for i,key in enumerate(['tumour','subgroup']):
   ax=axes[i,j]
   for offset,design,col,label in [(-.18,'fixed',BLUE,'Same reference binding rates'),(.18,'reoptimised',ORANGE,'Binding rates re-optimised')]:
    vals=[lookup[k][design][key] for k in keys];bars=ax.bar(np.arange(3)+offset,vals,.34,color=col,label=label);ax.bar_label(bars,fmt='%.1f',fontsize=9,padding=3)
   ax.set(xticks=np.arange(3),xticklabels=labels,ylabel='Tumour kill (%)' if i==0 else 'Maximum healthy-subgroup kill (%)',title=title if i==0 else '',ylim=(0,85 if i==0 else 22))
   if i==1:ax.axhline(10,color='#b83232',ls='--',lw=1)
 axes[0,0].legend(fontsize=8)
 fig.supxlabel('Each column changes only the named process. All other biological inputs remain fixed. Orange bars meet the same 0.95% overall and 10% subgroup limits. Synthetic inputs.',fontsize=9)
 save(fig,'06_release_movement_recycling')
 # Show three complete alternatives from the three-input calculation, without sample-count claims.
 pool=json.loads((ROOT/'repo/results_boundary_review/original_representative_front.json').read_text())
 choices=[]
 for label,cap in [('Lower subgroup injury',1),('Intermediate subgroup injury',5),('Higher tumour killing',10)]:
  candidates=[r for r in pool if r['healthy']<=.95 and r['subgroup']<=cap]
  r=max(candidates,key=lambda r:r['tumour']);rates=10.**np.array(r['theta']);choices.append(dict(label=label,subgroup_requirement=cap,**r,rates=rates.tolist(),KD_uM=[1e6*rates[1]/rates[0],1e6*rates[3]/rates[2]]))
 (DATA/'three_complete_designs.json').write_text(json.dumps(choices,indent=2))
 fig,axes=plt.subplots(1,3,figsize=(13,5),layout='constrained');labels=['Lower subgroup\ninjury','Intermediate\nsubgroup injury','Higher tumour\nkilling']
 for ax,key,title in zip(axes,['tumour','healthy','subgroup'],['Lowest tumour kill across inputs','Highest overall healthy kill','Highest healthy-subgroup kill']):
  vals=[r[key] for r in choices];bars=ax.bar(range(3),vals,color=[GREEN,BLUE,ORANGE],width=.6);ax.bar_label(bars,fmt='%.2f%%',padding=4)
  ax.set(xticks=range(3),xticklabels=labels,title=title,ylabel='Cell kill at 24 h (%)',ylim=(0,max(vals)*1.3))
  ax.tick_params(axis='x',labelsize=9)
 fig.supxlabel('One complete antibody design per colour, evaluated under all three input conditions. These are selected examples, not three independent affinity intervals.\nThey use the three-input criterion; the earlier mechanism figures use the reference input alone.',fontsize=9)
 save(fig,'07_three_complete_design_choices')
 # Preserve simple raw tables for the newly computed sweeps.
 pd.DataFrame([{k:r[k] for k in ['internalisation_h','dissociation_multiplier','tumour','healthy','subgroup','mass']} for r in data]).to_csv(DATA/'internalisation_sweep.csv',index=False)
 print('PLOTS COMPLETE',len(data),'new ODE evaluations',flush=True)
 print('CHOICES',[(r['label'],r['tumour'],r['healthy'],r['subgroup']) for r in choices],flush=True)
if __name__=='__main__':main()
