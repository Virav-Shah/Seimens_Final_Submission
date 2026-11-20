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
(D/'Internalisation_slide_replacement.md').write_text('''# Replace the three-panel graph with two slides

## Slide 1 — Faster internalisation increases killing in both tumour and healthy cells

Use `08_internalisation_same_antibody.png`.

**Put on the slide:**
- Keep the antibody’s four binding rates unchanged.
- Increase internalisation of the dually bound complex from 0.2 to 4 per hour.
- Tumour killing increases from **15.2% to 67.7%**.
- Killing of healthy cells expressing both targets increases from **1.3% to 17.7%**, exceeding the illustrative 10% subgroup limit.

**Say:** “Faster internalisation delivers more payload to tumour cells, but the same process also operates in healthy cells expressing both targets. It therefore increases both effects. Overall healthy killing remains below 1%, so the overall average alone misses this subgroup problem.”

## Slide 2 — Binding must be adjusted to meet the same healthy-cell requirements

Use `09_internalisation_binding_adjusted.png`.

**Put on the slide:**
- At slow internalisation, the reference design leaves room below the subgroup limit. Re-optimising binding raises tumour killing to **56.8%**.
- At fast internalisation, the reference design exceeds the subgroup limit. Re-optimising binding lowers subgroup killing to **10%**, while tumour killing falls to **57.6%**.
- The final tumour results are similar, but they come from different binding designs.

**Say:** “We now change the binding rates separately for each internalisation condition. We seek more tumour killing while keeping overall healthy killing below 0.95% and every healthy subgroup below 10%. At the slow rate, the redesign improves tumour activity. At the fast rate, the redesign sacrifices some tumour activity to meet the subgroup requirement. Faster internalisation cannot be judged from tumour killing alone.”

## What the original plot made difficult to see

It overlaid a fixed-antibody sensitivity curve with three re-optimised designs, used a logarithmic horizontal axis, and spread the explanation across three panels. The overall-average panel also obscured the central issue: substantial injury within a small healthy subgroup. These two figures separate the biological change from the redesign decision and label the actual values.

The values are calculated from illustrative inputs and are best-found designs under assumed rate limits. They do not establish a universal optimum. This replacement uses existing completed calculations; no additional optimisation was performed. The three-panel sweep remains suitable as a supplementary sensitivity figure.
''')
print('Created two replacement figures and slide text.')
