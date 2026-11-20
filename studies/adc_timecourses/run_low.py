from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'repo'))
from mechanism_comparison import study as st
from mechanism_comparison.model import BIOLOGY,replace
from joblib import Parallel,delayed,parallel_config
D=Path(__file__).resolve().parents[2]/'output/adc_timecourses'
CASES={'reference_low':('Separate rates 0.08 / 0.10 / 0.03',replace(BIOLOGY,ki=(.08,.10,.03)))}
def run(seed):
 st.OUT=D;st.CASES=CASES;st.BUDGET=512
 return st.search('reference_low','subgroup_limit',seed,[4.518138471032275,1.2445341007518325,2.,.030172124658327883])
if __name__=='__main__':
 with parallel_config(backend='loky',inner_max_num_threads=1):runs=Parallel(n_jobs=2)(delayed(run)(seed) for seed in (8407,2026))
 best=max((r['best'] for r in runs if r['best']),key=lambda r:r['tumour'])
 (D/'reference_low_best.json').write_text(json.dumps(best,indent=2))
 print('COMPLETE',best['tumour'],flush=True)
