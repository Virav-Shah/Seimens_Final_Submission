from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
D=Path(__file__).resolve().parents[2]/'output/presentation_revision'
r=json.loads((D/'data/mechanism_results.json').read_text());r={x['case']:x for x in r}
blue='#246b94';orange='#d67525';red='#b23535';green='#34765b'
plt.rcParams.update({'font.size':13,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
def save(fig,name):
 for ext in ['png','svg']:fig.savefig(D/'plots'/f'{name}.{ext}',dpi=200,bbox_inches='tight')
 plt.close(fig)
# First isolate biology; the same four binding rates in both conditions.
fig,axes=plt.subplots(1,2,figsize=(12,6),layout='constrained')
fig.suptitle('1. Faster internalisation increases killing in BOTH cell types',fontsize=19,fontweight='bold')
keys=['dual_slow','dual_fast'];labels=['Slow internalisation\n0.2 per hour','Fast internalisation\n4 per hour']
for ax,metric,title,ylim in zip(axes,['tumour','subgroup'],['Tumour cells','Healthy cells expressing both targets'],[85,23]):
 vals=[r[k]['fixed'][metric] for k in keys]
 bars=ax.bar([0,1],vals,color=[blue,orange],width=.5)
 ax.bar_label(bars,labels=[f'{v:.1f}%' for v in vals],padding=6,fontsize=17,fontweight='bold')
 ax.set(xticks=[0,1],xticklabels=labels,ylabel='Cells killed after 24 hours (%)',title=title,ylim=(0,ylim))
 if metric=='subgroup':
  ax.axhline(10,color=red,ls='--',lw=1.5)
  ax.text(.02,12.0,'Assumed subgroup limit: 10%',color=red,transform=ax.get_yaxis_transform(),fontsize=11)
  ax.text(.98,.95,'Fast internalisation\nexceeds the subgroup limit',ha='right',va='top',transform=ax.transAxes,color=red,fontsize=11)
fig.supxlabel('Same antibody: all four binding rates unchanged. Only dually bound internalisation changes.\nOverall healthy-cell killing stays below 1% in both cases (0.09% and 0.26%), despite the subgroup injury.\nIllustrative reference inputs; the 10% subgroup limit is an assumed requirement.',fontsize=11)
save(fig,'08_internalisation_same_antibody')
# Then show what optimisation does separately for each biological condition.
fig,axes=plt.subplots(1,2,figsize=(13,6.5),layout='constrained')
fig.suptitle('2. Re-optimising binding restores the chosen subgroup protection',fontsize=18,fontweight='bold')
x=np.arange(2);w=.32
for ax,metric,title,ylim in zip(axes,['tumour','subgroup'],['Tumour cells','Healthy cells expressing both targets'],[85,23]):
 for offset,design,col,label in [(-w/2,'fixed',blue,'Reference binding rates'),(w/2,'reoptimised',orange,'Binding rates re-optimised')]:
  vals=[r[k][design][metric] for k in keys];bars=ax.bar(x+offset,vals,w,color=col,label=label)
  ax.bar_label(bars,labels=[f'{v:.1f}%' for v in vals],padding=5,fontsize=14,fontweight='bold')
 ax.set(xticks=x,xticklabels=labels,ylabel='Cells killed after 24 hours (%)',title=title,ylim=(0,ylim))
 if metric=='subgroup':
  ax.axhline(10,color=red,ls='--',lw=1.5)
  ax.text(.02,12.0,'Assumed subgroup limit: 10%',color=red,transform=ax.get_yaxis_transform(),fontsize=11)
axes[0].legend(loc='upper left',fontsize=10,frameon=False)
fig.supxlabel('At the slow rate: retuning binding raises tumour killing while remaining within the subgroup limit.\nAt the fast rate: retuning binding reduces subgroup injury, with a reduction in tumour killing.\nSame biology within each pair. Only the four binding rates are re-optimised. Both healthy-cell limits are enforced.',fontsize=11)
save(fig,'09_internalisation_binding_adjusted')
print('Created two replacement figures.')
