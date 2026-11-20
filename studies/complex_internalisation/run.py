from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'repo'))
from mechanism_comparison import study as st
from mechanism_comparison.model import replace,BIOLOGY
from joblib import Parallel,delayed,parallel_config
OUT=Path(__file__).resolve().parents[2]/'output/complex_internalisation'
CASES={'common':('Common rate: 2 / 2 / 2',replace(BIOLOGY,ki=(2.,2.,2.))), 'reference':('Reference: 0.2 / 4 / 2',BIOLOGY),'condition_a':('Condition A: 0.16 / 3.2 / 1.6',replace(BIOLOGY,ki=(.16,3.2,1.6))),'condition_b':('Condition B: 0.24 / 4.8 / 2.4',replace(BIOLOGY,ki=(.24,4.8,2.4)))}
def run(case,seed,anchor):
 st.OUT=OUT;st.CASES=CASES;st.BUDGET=512
 return st.search(case,'subgroup_limit',seed,anchor)
if __name__=='__main__':
 anchor=[4.518138471032275,1.2445341007518325,2.0000000000055165,.030172124658327883]
 with parallel_config(backend='loky',inner_max_num_threads=1):
  base=Parallel(n_jobs=2)(delayed(run)('common',s,anchor) for s in (8407,2026))
 common=max((v['best'] for v in base if v['best']),key=lambda x:x['tumour'])
 with parallel_config(backend='loky',inner_max_num_threads=1):
  runs=Parallel(n_jobs=4)(delayed(run)(c,s,common['theta']) for c in CASES if c!='common' for s in (8407,2026))
 allruns=base+runs
 rows=[]
 for c,(label,biology) in CASES.items():
  candidates=[st.evaluate(v['best']['theta'],biology,True) for v in allruns if v['best']]
  candidates.append(st.evaluate(anchor,biology,True))
  best=max((v for v in candidates if st.margin(v,'subgroup_limit')>=0),key=lambda x:x['tumour'])
  rows.append(dict(case=c,label=label,ki=biology.ki,transferred=st.evaluate(common['theta'],biology,True),reoptimised=best))
 (OUT/'results.json').write_text(json.dumps(rows,indent=2))
 print('COMPLETE',flush=True)
