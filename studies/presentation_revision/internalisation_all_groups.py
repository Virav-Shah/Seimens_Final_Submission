from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
D=Path(__file__).resolve().parents[2]/'output/presentation_revision'
r={x['case']:x for x in json.loads((D/'data/mechanism_results.json').read_text())}
a,b=[r[k]['fixed'] for k in ('dual_slow','dual_fast')]
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,axs=plt.subplots(1,3,figsize=(16,7),gridspec_kw={'width_ratios':[1,1,1.55]})
fig.subplots_adjust(left=.055,right=.985,top=.78,bottom=.29,wspace=.32)
fig.suptitle('Healthy cells expressing BOTH targets (H12) show the large increase in killing',fontsize=19,fontweight='bold',y=.98)
fig.text(.5,.9,'Same antibody; only internalisation of the dually bound complex changes from 0.2 to 4 per hour.',ha='center',fontsize=13)
colors=['#246b94','#d67525']
for ax,vals,title,lim in [(axs[0],[a['tumour'],b['tumour']],'Tumour cells',85),(axs[1],[a['groups'][3],b['groups'][3]],'H12: both targets',23)]:
 bars=ax.bar([0,1],vals,color=colors,width=.58)
 ax.bar_label(bars,labels=[f'{v:.1f}%' for v in vals],padding=7,fontsize=18,fontweight='bold')
 ax.set(xticks=[0,1],xticklabels=['Slow','Fast'],ylim=(0,lim),title=title,ylabel='Cells killed after 24 hours (%)')
axs[1].axhline(10,color='#b23535',ls='--')
axs[1].text(-.35,11,'Assumed limit: 10%',color='#b23535',fontsize=11)
x=np.arange(3);w=.34
for offset,record,col,label in [(-w/2,a,colors[0],'Slow: 0.2 per hour'),(w/2,b,colors[1],'Fast: 4 per hour')]:
 vals=record['groups'][:3]
 bars=axs[2].bar(x+offset,vals,w,color=col,label=label)
 axs[2].bar_label(bars,labels=[f'{v:.3f}%' for v in vals],padding=5,fontsize=11)
axs[2].set(xticks=x,xticklabels=['H0\nNeither target','H1\nT1 only','H2\nT2 only'],ylim=(0,.16),title='Other healthy subgroups: small changes',ylabel='Cells killed after 24 hours (%)')
axs[2].legend(loc='upper left',fontsize=10,frameon=False)
axs[2].text(.5,-.19,'Enlarged vertical scale to show the small values',transform=axs[2].transAxes,ha='center',fontsize=10,color='#555555')
fig.text(.055,.16,f"H12: 1.25% → 17.66% killed — an increase of {b['groups'][3]-a['groups'][3]:.2f} percentage points, not a 17% increase.",fontsize=14,fontweight='bold')
fig.text(.055,.105,'H0, H1 and H2 each remain below 0.12% killed. Overall healthy killing rises only from 0.091% to 0.257%.',fontsize=13)
fig.text(.055,.055,'H12 represents 1% of healthy cells in these assumed inputs, so its injury is diluted in the overall average.',fontsize=13)
fig.text(.055,.012,'Illustrative model results; all binding rates and other inputs remain fixed. The 10% subgroup limit is an assumed requirement.',fontsize=10,color='#555555')
for ext in ('png','svg'):
 fig.savefig(D/'plots'/f'10_internalisation_all_healthy_subgroups.{ext}',dpi=200,bbox_inches='tight')
plt.close(fig)
