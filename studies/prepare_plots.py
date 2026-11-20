"""Presentation figures drawn from saved notebook results; no new model fitting."""
from pathlib import Path
import shutil
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/presentation_overview'
(OUT/'plots').mkdir(parents=True,exist_ok=True)
(OUT/'data').mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':13,'axes.titlesize':16,'axes.labelsize':13,
 'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
blue='#176B91'; orange='#C56B28'; purple='#774C9E'
v3=pd.read_csv(ROOT/'repo/results_v3/selected_designs.csv').set_index('case')
front=pd.read_csv(ROOT/'repo/results_v5/pareto_binding_rates.csv')
short=pd.read_csv(ROOT/'repo/results_v5/tradeoff_shortlist.csv').sort_values('minimum_tumour_kill_pct')
short['Presentation label']=list('ABCDEF')
short.to_csv(OUT/'data/shortlist_with_presentation_labels.csv',index=False)

def save(fig,name,caption):
    fig.text(.02,.015,textwrap.fill(caption, width=135),fontsize=9,color='#444444')
    fig.tight_layout(rect=(0,.09,1,1))
    for ext in ['png','svg']:fig.savefig(OUT/'plots'/f'{name}.{ext}',dpi=300,facecolor='white')
    plt.close(fig)

# Slide 18: retain a full 0-100 effect axis; show affinity changes separately.
ds=v3.loc[['B1_dual_0.2x','B1_dual_0.5x','B1_dual_1x','B1_dual_2x','B1_dual_5x']]
x=np.array([.2,.5,1,2,5])
fig,ax=plt.subplots(1,2,figsize=(12,4.9))
for col,label,c in [('KD1_uM','T1',blue),('KD2_uM','T2',orange)]:
 ax[0].plot(x,ds[col],'-o',label=label,color=c)
ax[0].set(xscale='log',xlabel='Dual-complex internalisation / baseline',ylabel=r'$K_D$ (µM)',title='Selected affinities after reoptimisation')
ax[0].set_xticks(x,[str(z) for z in x]);ax[0].legend()
for col,label,c in [('tumor_pct','Tumour',blue),('H12_pct','Healthy T1 + T2',orange)]:
 ax[1].plot(x,ds[col],'-o',label=label,color=c)
ax[1].set(xscale='log',xlabel='Dual-complex internalisation / baseline',ylabel='24 h Emax effect (%)',ylim=(0,105),title='Tumour and dual-expressing healthy cells')
ax[1].set_xticks(x,[str(z) for z in x]);ax[1].legend(loc='lower right')
save(fig,'slide18_internalisation','Exploratory synthetic comparisons; each point reoptimised. Overall healthy effect ≈ 0.95%. Emax is not the final cell-kill endpoint.')

# Slide 19: subgroup response and weighted contribution are distinct quantities.
r=v3.loc['B0_equal'];fig,ax=plt.subplots(1,2,figsize=(12,5))
vals=[0,r.H1_pct,r.H2_pct,r.H12_pct]
bars=ax[0].bar(['Neither','T1 only','T2 only','T1 + T2'],vals,color=[blue,blue,blue,orange])
ax[0].bar_label(bars,fmt='%.2f',padding=4);ax[0].set(ylim=(0,100),ylabel='Within-group Emax effect (%)',title='Response within each healthy population')
parts=np.array(vals)*np.array([.79,.1,.1,.01]);base=0
for val,label,c in zip(parts,['Neither','T1 only','T2 only','T1 + T2'],['#BBCBD4',blue,purple,orange]):
 ax[1].bar(['All healthy cells'],[val],bottom=base,label=label,color=c,width=.4);base+=val
ax[1].axhline(1,ls='--',c='#555555');ax[1].text(.98,1.025,'1% overall criterion',fontsize=11,ha='right',transform=ax[1].get_yaxis_transform())
ax[1].set(ylim=(0,1.4),ylabel='Contribution to overall effect (percentage points)',title=f'Weighted overall effect: {base:.3f}%')
ax[1].legend(loc='upper left',fontsize=10,ncol=2);ax[1].text(0,.55,f'T1 + T2 contributes\n{parts[3]:.3f} percentage points',ha='center',color='white',fontsize=12)
save(fig,'slide19_healthy_subgroups','Exploratory equal-density baseline; synthetic Emax result. Different y-axis scales distinguish within-group effects from weighted contributions.')

cases=['B0_equal','B3_shed_96h','B4_stable_exchange','B4_exchange_96h']
names=['Stable linker\nNo exchange','96 h linker\nNo exchange','Stable linker\nExchange','96 h linker\nExchange']
fig,ax=plt.subplots(figsize=(10,4.8));bars=ax.bar(names,v3.loc[cases,'tumor_pct'],color=[blue,blue,orange,orange],width=.6)
ax.bar_label(bars,fmt='%.2f%%',padding=4);ax.set(ylim=(0,108),ylabel='24 h tumour Emax effect (%)',title='Linker cleavage and payload exchange')
save(fig,'slide20_linker_and_exchange','Exploratory synthetic results after reoptimisation; overall healthy effect ≈ 0.95%. These are Emax effects, not the final cell-kill endpoint.')

fig,ax=plt.subplots(figsize=(10,5.7))
p=ax.scatter(front.maximum_overall_healthy_kill_pct,front.minimum_tumour_kill_pct,
 c=front.maximum_healthy_subgroup_kill_pct,cmap='viridis',s=23,alpha=.8)
for _,r in short.iterrows():
 ax.scatter(r.maximum_overall_healthy_kill_pct,r.minimum_tumour_kill_pct,facecolors='none',edgecolors='black',s=90)
 ax.annotate(r['Presentation label'],(r.maximum_overall_healthy_kill_pct,r.minimum_tumour_kill_pct),xytext=(6,7),textcoords='offset points',weight='bold')
ax.set(xlabel='Maximum overall healthy cell kill (%)',ylabel='Minimum tumour cell kill (%)',title='Observed tumour-selectivity trade-offs',ylim=(-2,49))
fig.colorbar(p,ax=ax,label='Maximum healthy-subgroup cell kill (%)')
save(fig,'slide22_pareto_selectivity','Final model: extrema across three synthetic input variants at 24 h. Each point is a joint four-rate combination; this is a 3-objective projection.')

fig,ax=plt.subplots(figsize=(10,5.7))
p=ax.scatter(front.KD1_uM,front.KD2_uM,c=front.minimum_tumour_kill_pct,cmap='plasma',s=23,alpha=.8)
for _,r in short.iterrows():
 ax.scatter(r.KD1_uM,r.KD2_uM,facecolors='none',edgecolors='black',s=90)
 ax.annotate(r['Presentation label'],(r.KD1_uM,r.KD2_uM),xytext=(7,6),textcoords='offset points',weight='bold')
ax.set(xscale='log',yscale='log',xlabel=r'T1 $K_D$ (µM)',ylabel=r'T2 $K_D$ (µM)',title='Joint affinities associated with the observed Pareto set')
fig.colorbar(p,ax=ax,label='Minimum tumour cell kill (%)')
save(fig,'slide23_joint_affinities','Final-model synthetic method demonstration. Retain complete on/off-rate combinations; the scatter does not define independent acceptable affinity intervals.')

(OUT/'original_notebook_plots').mkdir(parents=True,exist_ok=True)
# Faithful original exports supplied alongside enlarged slide figures.
for src in ['repo/results_v3/trafficking.png','repo/results_v3/stage_comparison.png',
 'repo/results_v5/selectivity_pareto.png']:
 p=ROOT/src;shutil.copy2(p,OUT/'original_notebook_plots'/f'{p.parent.name}_{p.name}')
for src in ['repo/results_v3/selected_designs.csv','repo/results_v3/stage_comparison.csv',
 'repo/results_v5/pareto_binding_rates.csv',
 'repo/results_v5/outcomes_by_input.csv','repo/results_v5/illustrative_requirements_subset.csv']:
 p=ROOT/src;shutil.copy2(p,OUT/'data'/f'{p.parent.name}_{p.name}')
print('Created five slide figures (PNG + SVG), copied four notebook figures and six source tables.')
