"""Illustrative landmark-reading uncertainty; ranges are not statistical CIs."""
import itertools,json,sys
import numpy as np
from scipy.optimize import least_squares
from dataclasses import replace
from prototype import OUT,save_csv,evaluate
from model import SCENARIOS,Parameters,simulate

def main():
    results=json.loads((OUT/'results.json').read_text());rows=[]
    for a4,a6 in itertools.product([50.,62.],[250.,310.]):
        def fun(z):
            n,v=10**np.array(z);out=[]
            for name,target in [('figure4_bispecific',a4),('figure6',a6)]:
                r=simulate(replace(SCENARIOS[name],total_cells=n,cell_volume_l=v),points=3)
                out.append(np.log(r['intracellular'][-1,0]*1e6/target))
            return out
        fit=least_squares(fun,[12.67,-12.38],bounds=([11,-13.3],[13.5,-11.5]))
        if not fit.success:raise RuntimeError(fit.message)
        n,v=10**fit.x
        for scenario,result in results.items():
            p=replace(Parameters(**result['selected']['parameters']),total_cells=n,cell_volume_l=v)
            e,h=evaluate(p,tight=True)
            rows.append(dict(figure4_anchor_uM=a4,figure6_anchor_uM=a6,effective_cells=n,cell_volume_l=v,scenario=scenario,tumor_kill_pct=100*e,healthy_kill_pct=100*h,below_one_percent=h<.01))
    save_csv(OUT/'calibration_reading_sensitivity.csv',rows)
    print('Landmark-reading sensitivity completed:',len(rows),'frozen-candidate evaluations')
if __name__=='__main__':main()
