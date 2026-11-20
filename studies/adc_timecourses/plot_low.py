from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import integrate,BIOLOGY,replace
D=Path(__file__).resolve().parents[2]/'output/adc_timecourses'
common=json.loads((D/'common_best.json').read_text())['theta']
ref=json.loads((D/'reference_low_best.json').read_text())['theta']
start=np.log10([4.6e5,3.4e-4,4.6e5,3.4e-4])
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False})
records=[]
for name,title,optim,mono in [('01_unoptimised_bispecific','Unoptimised bispecific ADC',False,False),('02_monospecific','Monospecific ADC: bivalent binding to T1',False,True),('03_optimised_bispecific','Bispecific ADC: binding optimised for each internalisation condition',True,False)]:
 fig,ax=plt.subplots(figsize=(10,6.5));fig.subplots_adjust(top=.78,bottom=.28,right=.9)
 fig.suptitle(title,fontsize=17,fontweight='bold',y=.97)
 fig.text(.5,.89,'Common: 0.2 / 0.2 / 0.2 h⁻¹     |     Reference: 0.08 / 0.10 / 0.03 h⁻¹',ha='center',fontsize=12)
 fig.text(.5,.845,'Rates listed in order: T1–ADC / T2–ADC / T1T2–ADC',ha='center',fontsize=11)
 for i,(label,ki,theta) in enumerate([('Common',(.2,.2,.2),common if optim else start),('Reference',(.08,.10,.03),ref if optim else start)]):
  s=replace(BIOLOGY,ki=(ki[0],ki[0],ki[0]) if mono else ki,mode='monospecific' if mono else 'bispecific')
  v=integrate(theta,s,tight=True,points=241)
  for curve,col,cell in [(100*v['kill'][:,0],'#1976b5','Tumour'),(100*v['healthy'],'#e07820','Healthy')]:
   ax.plot(v['t'],curve,color=col,ls='-' if i==0 else '--',lw=2.5,label=f'{cell} — {label}')
  records.append(dict(adc=name,condition=label,theta=list(theta),tumour=float(v['kill'][-1,0]*100),healthy=float(v['healthy'][-1]*100),groups=(v['kill'][-1,1:]*100).tolist(),mass=float(v['mass']),time=v['t'].tolist(),tumour_curve=(100*v['kill'][:,0]).tolist(),healthy_curve=(100*v['healthy']).tolist()))
 ax.set(xlim=(0,24),ylim=(0,100),xlabel='Time (h)',ylabel='Cell kill (%)')
 ax.legend(loc='upper left',fontsize=10,frameon=False)
 a,b=records[-2:]
 fig.text(.5,.18,f"24 h — Common: tumour {a['tumour']:.2f}%, healthy {a['healthy']:.3f}%   |   Reference: tumour {b['tumour']:.2f}%, healthy {b['healthy']:.3f}%",ha='center',fontsize=11)
 note=('T1-bound states, including T1–ADC–T1: 0.2 h⁻¹ (Common), 0.08 h⁻¹ (Reference).\nT2 and T1T2 complexes are absent; their rates do not apply to this monospecific ADC.' if mono else ('Four binding rates selected separately; overall healthy ≤0.95%, each healthy subgroup ≤10%.\nBest-found designs within the searched bounds; not a proven unique optimum.' if optim else 'Same starting binding rates in both conditions: k_on,1 = k_on,2 = 4.6 × 10⁵ M⁻¹ s⁻¹;\nk_off,1 = k_off,2 = 3.4 × 10⁻⁴ s⁻¹.'))
 fig.text(.5,.09,note,ha='center',fontsize=10)
 fig.text(.5,.025,'Illustrative simulations using our extended model; all other inputs fixed. These are not the original paper’s curves.',ha='center',fontsize=9,color='#555555')
 fig.savefig(D/f'{name}_low_reference.png',dpi=200,bbox_inches='tight');plt.close(fig)
(D/'trajectories_low_reference.json').write_text(json.dumps(records,indent=2))
print([(v['adc'],v['condition'],v['tumour'],v['healthy'],v['mass']) for v in records])
