from .study import *
import hashlib, platform, importlib.metadata

FAMILIES=[
 ('internalisation','Internalisation of the dually bound complex',['dual_slow','reference','dual_fast'],['0.2 h⁻¹','2 h⁻¹\n(reference)','4 h⁻¹']),
 ('premature_release','Premature payload release outside cells',['linker_stable','reference','linker_short'],['No premature\nrelease','96 h half-life\n(reference)','48 h half-life']),
 ('payload_movement','Movement of free payload across cell membranes',['movement_none','reference','movement_fast'],['0 h⁻¹','0.1 h⁻¹\n(reference)','0.2 h⁻¹']),
 ('intracellular_processing','Intracellular processing and payload release',['processing_slow','reference','processing_fast'],['0.35 h⁻¹','0.7 h⁻¹\n(reference)','1.4 h⁻¹']),
 ('recycling','Recycling of internalised antibody–target complexes',['recycling_none','reference','recycling_fast'],['0 h⁻¹','0.6 h⁻¹\n(reference)','1.2 h⁻¹']),
 ('payload_metabolism','Intracellular payload metabolism',['metabolism_none','reference','metabolism_fast'],['0 h⁻¹','0.01 h⁻¹\n(reference)','0.02 h⁻¹'])]
BLUE='#1f77b4';ORANGE='#ff7f0e'

def save(fig,name):
 fig.savefig(OUT/f'{name}.png',dpi=200,bbox_inches='tight')
 fig.savefig(OUT/f'{name}.svg',bbox_inches='tight');plt.close(fig)

def make_plots(mechanisms_only=False):
 rows=json.loads((OUT/'mechanism_results.json').read_text());lookup={r['case']:r for r in rows}
 plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
 flat=[]
 for row in rows:
  for design in ('fixed','reoptimised'):
   r=row[design];rates=10.**np.array(r['theta'])
   flat.append(dict(case=row['case'],condition=row['label'],design=design,tumour_kill_percent=r['tumour'],overall_healthy_kill_percent=r['healthy'],maximum_healthy_subgroup_kill_percent=r['subgroup'],**dict(zip(['healthy_no_target_percent','healthy_T1_percent','healthy_T2_percent','healthy_T1T2_percent'],r['groups'])),**dict(zip(RATES,rates)),KD1_uM=1e6*rates[1]/rates[0],KD2_uM=1e6*rates[3]/rates[2],satisfies_limits=bool(r['healthy']<=.95 and r['subgroup']<=10)))
 pd.DataFrame(flat).to_csv(OUT/'comparisons_and_binding_rates.csv',index=False)
 for name,title,keys,labels in FAMILIES:
  fig,axes=plt.subplots(2,1,figsize=(9,7),layout='constrained');x=np.arange(3);w=.32
  for ax,metric,ylabel in zip(axes,['tumour','healthy'],['Tumour kill (%)','Overall healthy kill (%)']):
   for j,(design,color,label) in enumerate([('fixed',BLUE,'Reference binding rates transferred'),('reoptimised',ORANGE,'Binding rates re-optimised')]):
    vals=[lookup[k][design][metric] for k in keys]
    bars=ax.bar(x+(j-.5)*w,vals,w,color=color,label=label)
    ax.bar_label(bars,labels=[f'{v:.2f}' + ('†' if design=='fixed' and metric=='tumour' and (lookup[k][design]['healthy']>.95 or lookup[k][design]['subgroup']>10) else '') for k,v in zip(keys,vals)],fontsize=10,padding=3)
   ax.set(xticks=x,xticklabels=labels,ylabel=ylabel)
   ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
  axes[0].set_ylim(0,100);axes[0].legend(loc='upper left',fontsize=9)
  ymax=max(1.15,1.25*max(lookup[k][d]['healthy'] for k in keys for d in ('fixed','reoptimised')))
  axes[1].set_ylim(0,ymax)
  axes[1].axhline(1.,color='red',ls='--',lw=1.4,label='1% overall healthy kill')
  axes[1].axhline(.95,color='gray',ls=':',lw=1,label='0.95% search limit')
  axes[1].legend(handles=axes[1].lines,loc='upper right',fontsize=9)
  fig.suptitle(title,fontsize=14)
  fig.supxlabel('Illustrative inputs; all other biological parameters fixed.\nRe-optimisation limits: 0.95% overall healthy kill and 10% within each healthy subgroup.\n† Transferred rates exceed a healthy-cell limit.',fontsize=8)
  save(fig,name)
 # Rare-group injury must remain visible even where weighted averages are small.
 fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
 for ax,(name,title,keys,labels) in zip(axes.flat,FAMILIES):
  x=np.arange(3);w=.32
  for j,(d,col) in enumerate([('fixed',BLUE),('reoptimised',ORANGE)]):
   bars=ax.bar(x+(j-.5)*w,[lookup[k][d]['subgroup'] for k in keys],w,color=col)
   ax.bar_label(bars,fmt='%.2f',fontsize=8,padding=2)
  ax.axhline(10,color='red',ls='--',lw=1);ax.set(title=title,xticks=x,xticklabels=labels,ylabel='Maximum healthy subgroup kill (%)')
  ax.set_ylim(0,max(13,1.25*max(lookup[k][d]['subgroup'] for k in keys for d in ('fixed','reoptimised'))));ax.tick_params(axis='x',labelsize=9)
 fig.suptitle('Healthy-cell subgroups: transferred rates (blue) and re-optimised rates (orange)',fontsize=13);save(fig,'healthy_subgroups_by_process')
 # Kinetic changes are displayed explicitly, even when cell-kill changes are small.
 fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
 labels=[r['label'] for r in rows];x=np.arange(len(rows))
 for j,ax in enumerate(axes.flat):
  baseline=10.**np.array(rows[0]['fixed']['theta'])[j]
  ax.axhline(baseline,color=BLUE,label='Reference binding rate')
  ax.scatter(x,[10.**np.array(r['reoptimised']['theta'])[j] for r in rows],color=ORANGE,label='Re-optimised',zorder=3)
  ax.set(yscale='log',xticks=x,xticklabels=labels,ylabel=['Target 1 association (M⁻¹ s⁻¹)','Target 1 dissociation (s⁻¹)','Target 2 association (M⁻¹ s⁻¹)','Target 2 dissociation (s⁻¹)'][j])
  ax.tick_params(axis='x',rotation=70,labelsize=8);ax.grid(axis='y',alpha=.15)
  values=np.r_[baseline,[10.**np.array(r['reoptimised']['theta'])[j] for r in rows]]
  if np.ptp(np.log10(values))<1e-3:
   from matplotlib.ticker import NullLocator
   ax.set_ylim(baseline/2,baseline*2);ax.set_yticks([baseline/2,baseline,baseline*2]);ax.set_yticklabels([f'{baseline/2:g}',f'{baseline:g}',f'{baseline*2:g}']);ax.yaxis.set_minor_locator(NullLocator())
   ax.text(.02,.92,'All conditions ≈100 (lower search limit)',transform=ax.transAxes,fontsize=10)
  else:ax.set_ylim(values.min()*.8,values.max()*1.2)
 axes[0,0].legend(fontsize=9);fig.suptitle('Changes in the four binding rates; each set of four must be kept together');save(fig,'binding_rates')
 if mechanisms_only:return rows,None
 # Compare objective versus constraint, holding biology fixed.
 policy_rows=[];policy_fronts={}
 pooled=[]
 for policy in POLICIES:
  for seed in SEEDS:
   pooled.extend(json.loads((OUT/f'reference_{policy}_{seed}.json').read_text())['front'])
 for policy in POLICIES:
  runs=[json.loads((OUT/f'reference_{policy}_{seed}.json').read_text()) for seed in SEEDS]
  candidates=[v for v in pooled if margin(v,policy)>=0]
  policy_fronts[policy]=front(candidates,POLICIES[policy][0])
  b=max(candidates,key=lambda r:r['tumour']);policy_rows.append(dict(policy=policy,**b))
 (OUT/'policy_results.json').write_text(json.dumps(policy_rows,indent=2))
 labels=['Overall healthy\nkill objective','Add subgroup\nkill objective','Add subgroup objective\nand 10% limit']
 colors=[BLUE,'#2ca02c',ORANGE]
 fig,axes=plt.subplots(3,1,figsize=(10,10),layout='constrained')
 for ax,metric,title,limit in zip(axes,['tumour','healthy','subgroup'],['Tumour kill (%)','Overall healthy kill (%)','Maximum healthy subgroup kill (%)'],[None,1.,10.]):
  bars=ax.bar(np.arange(3),[r[metric] for r in policy_rows],color=colors,width=.6);ax.bar_label(bars,fmt='%.2f',padding=3)
  ax.set(xticks=np.arange(3),xticklabels=labels,ylabel=title,ylim=(0,100 if metric!='healthy' else max(1.2,1.2*max(r[metric] for r in policy_rows))))
  if limit is not None:ax.axhline(limit,color='red',ls='--',lw=1)
 fig.suptitle('Highest-tumour-kill combination allowed by each formulation\nSame illustrative biological inputs; 0.95% overall healthy-cell limit in all searches',fontsize=13)
 fig.supxlabel('Candidates pooled across searches, then assessed under each formulation. The subgroup limit is illustrative.',fontsize=8)
 save(fig,'optimisation_objectives')
 fig,ax=plt.subplots(figsize=(10,5),layout='constrained');x=np.arange(4);w=.24
 for j,r in enumerate(policy_rows):
  bars=ax.bar(x+(j-1)*w,r['groups'],w,color=colors[j],label=labels[j].replace('\n',' '));ax.bar_label(bars,fmt='%.2f',fontsize=8,padding=2)
 ax.axhline(10,color='red',ls='--');ax.set(xticks=x,xticklabels=['Neither target','T1 only','T2 only','T1 and T2'],ylabel='Kill within each healthy-cell subgroup (%)',ylim=(0,100));ax.legend(fontsize=9);save(fig,'optimisation_healthy_subgroups')
 fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
 for policy,col,label in zip(POLICIES,colors,labels):
  f=policy_fronts[policy]
  for ax,metric in zip(axes,['healthy','subgroup']):ax.scatter([r[metric] for r in f],[r['tumour'] for r in f],s=16,alpha=.65,c=col,label=label.replace('\n',' '))
 axes[0].set(xlabel='Overall healthy kill (%)',ylabel='Tumour kill (%)');axes[1].set(xlabel='Maximum healthy subgroup kill (%)',ylabel='Tumour kill (%)');axes[1].axvline(10,color='red',ls='--',lw=1);axes[0].legend(fontsize=8)
 fig.suptitle('Sampled non-dominated combinations (two projections of the objectives)');save(fig,'tradeoff_comparison')
 runs=[json.loads(p.read_text()) for p in OUT.glob('*_8407.json')]+[json.loads(p.read_text()) for p in OUT.glob('*_2026.json')]
 pd.DataFrame([dict(case=r['case'],policy=r['policy'],seed=r['seed'],evaluations=len(r['history']),local_success=r['local_success'],message=r['message'],best_tumour=r['best']['tumour'] if r['best'] else np.nan) for r in runs]).to_csv(OUT/'search_runs.csv',index=False)
 spec=dict(created='2026-11-19',biological_reference=asdict(BIOLOGY),kill_model=asdict(PD),cases={k:asdict(v[1]) for k,v in CASES.items()},bounds=BOUNDS,seeds=SEEDS,nominal_global_evaluations=BUDGET,population=48,local_max_iterations=50,model_sha256=hashlib.sha256((ROOT/'mechanism_comparison/model.py').read_bytes()).hexdigest(),python=platform.python_version(),versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','matplotlib','platypus-opt','joblib']})
 def json_safe(value):
  if isinstance(value,float) and not np.isfinite(value):return 'Infinity' if value>0 else '-Infinity'
  if isinstance(value,dict):return {k:json_safe(v) for k,v in value.items()}
  if isinstance(value,(tuple,list)):return [json_safe(v) for v in value]
  return value
 (OUT/'study_configuration.json').write_text(json.dumps(json_safe(spec),indent=2,allow_nan=False))
 return rows,policy_rows
if __name__=='__main__':make_plots()
