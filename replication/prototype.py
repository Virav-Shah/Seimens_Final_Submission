"""Working inference-led optimisation prototype.
Calibrate two nuisance inputs to approximate figure landmarks; freeze them before
binding optimisation. Calibration is not independent validation. See report.
"""
import argparse, json, csv, random, itertools, os, time
from dataclasses import replace,asdict
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from model import Parameters, SCENARIOS, simulate, HEALTHY_WEIGHTS
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent.parent/'tmp'/'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from platypus import Problem,Real,Solution,InjectedPopulation,NSGAII
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'output'/'prototype'
RATES=('kon1','koff1','kon2','koff2')
# An engineering exploration domain, not experimentally demonstrated feasibility.
BOUNDS=[(2.,7.),(-5.,2.),(2.,7.),(-5.,2.)]
TARGET=.0095  # leave 0.05 percentage-point headroom below the paper's 1% limit

def save_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def evaluate(p,tight=False):
    r=simulate(p,points=3,rtol=1e-9 if tight else 1e-6,atol=1e-12 if tight else 1e-9)
    return float(r['tumor'][-1]),float(r['healthy'][-1])

def calibrate():
    # Manually read approximate plot landmarks, not raw/digitised measurements.
    anchors={'figure4_bispecific':56.,'figure6':280.}
    def residual(z):
        n,v=10**np.asarray(z)
        vals=[]
        for name,target in anchors.items():
            r=simulate(replace(SCENARIOS[name],total_cells=n,cell_volume_l=v),points=3)
            vals.append(np.log(r['intracellular'][-1,0]*1e6/target))
        return vals
    fit=least_squares(residual,[12.7,-12.4],bounds=([11,-13.3],[13.5,-11.5]),xtol=1e-7,ftol=1e-7,gtol=1e-7)
    if not fit.success:raise RuntimeError(fit.message)
    n,v=10**fit.x
    config=dict(total_cells=float(n),cell_volume_l=float(v))
    result=dict(status='INFERRED_PROTOTYPE',calibration_inputs=config,anchors_uM=anchors,
                anchor_provenance='Approximate visual readings of supplied Figures 4b and 6b at 24 h; NOT raw data',
                residual_log=fit.fun.tolist(),fixed_reference=asdict(Parameters()),rate_bounds_log10=BOUNDS,
                healthy_fraction_design_limit=TARGET,healthy_fraction_paper_limit=.01)
    (OUT/'configuration.json').write_text(json.dumps(result,indent=2))
    print('calibration',config,flush=True)
    return config

def scenario_params(config):
    return {'equal':replace(SCENARIOS['figure6'],**config),
            'unequal':replace(SCENARIOS['figure8'],**config),
            'low':replace(SCENARIOS['figure6'],rho1=1e-13,rho2=1e-13,ec50=1e-8,**config)}

def trajectory(name,p):
    r=simulate(p,points=241,rtol=1e-9,atol=1e-12);h=r['t']/3600
    rows=[]
    for i,t in enumerate(h):
        row=dict(time_h=float(t),tumor_kill_pct=float(100*r['tumor'][i]),healthy_kill_pct=float(100*r['healthy'][i]),central_uM=float(r['y'][i,0]),peripheral_uM=float(r['y'][i,1]))
        for j,label in enumerate(('tumor','healthy_T1','healthy_T2','healthy_T12')):
            row[label+'_uptake_uM']=float(r['intracellular'][i,j]*1e6)
            row[label+'_T1_occupancy_pct']=float(r['occupancy'][i,j,0]*100)
            row[label+'_T2_occupancy_pct']=float(r['occupancy'][i,j,1]*100)
        rows.append(row)
    for row in rows:
        mean=.1*row['healthy_T1_uptake_uM']+.1*row['healthy_T2_uptake_uM']+.01*row['healthy_T12_uptake_uM']
        row['healthy_all_mean_uM']=mean
        row['healthy_target_positive_mean_uM']=mean/(.11 if p.mode=='monospecific' else .21)
    save_csv(OUT/(name+'_trajectory.csv'),rows)
    return r

def baseline(config):
    rows=[]
    for name,p in SCENARIOS.items():
        p=replace(p,**config);r=trajectory(name,p)
        rows.append(dict(scenario=name,tumor_kill_pct=float(100*r['tumor'][-1]),healthy_kill_pct=float(100*r['healthy'][-1]),tumor_uptake_uM=float(r['intracellular'][-1,0]*1e6),tumor_T1_occupancy_24h=float(r['occupancy'][-1,0,0]),tumor_T2_occupancy_24h=float(r['occupancy'][-1,0,1])))
    save_csv(OUT/'calibrated_baselines.csv',rows)
    return rows

def search(name,base,seed,nfe):
    random.seed(seed);np.random.seed(seed)
    problem=Problem(4,2,1);problem.types[:]=[Real(*b) for b in BOUNDS]
    problem.directions[:]=[Problem.MINIMIZE]*2;problem.constraints[:]='<=0'
    rows=[];cache={};started=time.monotonic()
    def fun(z):
        key=tuple(float(x) for x in z)
        if key in cache:return cache[key]
        vals=10**np.asarray(z);p=replace(base,**dict(zip(RATES,vals)))
        e,h=evaluate(p)
        row={k:float(x) for k,x in zip(RATES,vals)}
        row.update(tumor_kill_pct=100*e,healthy_kill_pct=100*h,feasible_design=h<=TARGET,seed=seed)
        rows.append(row);ret=([-e,h],[h-TARGET]);cache[key]=ret
        if len(rows)%100==0:
            feasible=[x for x in rows if x['feasible_design']]
            print(name,seed,'evaluations',len(rows),'best tumor',max((x['tumor_kill_pct'] for x in feasible),default=None),'elapsed',round(time.monotonic()-started),flush=True)
            save_csv(OUT/f'{name}_seed{seed}_evaluations.csv',rows)
        return ret
    problem.function=fun
    # Interpretable symmetric starting grid plus asymmetric reported candidate.
    seeds=[]
    for on,off in itertools.product([1e2,1e3,1e4,1e5],[.01,.1,1.,10.,100.]):
        sol=Solution(problem);sol.variables[:]=np.log10([on,off,on,off]).tolist();seeds.append(sol)
    sol=Solution(problem);sol.variables[:]=np.log10([base.kon1,base.koff1,base.kon2,base.koff2]).tolist();seeds.append(sol)
    alg=NSGAII(problem,population_size=48,generator=InjectedPopulation(seeds));alg.run(nfe)
    save_csv(OUT/f'{name}_seed{seed}_evaluations.csv',rows)
    feasible=sorted((r for r in rows if r['feasible_design']),key=lambda r:r['tumor_kill_pct'],reverse=True)
    best=None
    for row in feasible:
        candidate=replace(base,**{k:row[k] for k in RATES});e,h=evaluate(candidate,tight=True)
        if h<=TARGET:
            best=dict(row,tumor_kill_pct=100*e,healthy_kill_pct=100*h,parameters=asdict(candidate));break
    status=dict(seed=seed,requested_nfe=nfe,algorithm_nfe=alg.nfe,evaluator_calls=len(rows),best=best)
    (OUT/f'{name}_seed{seed}_result.json').write_text(json.dumps(status,indent=2))
    print('finished',name,seed,'best',None if best is None else [best['tumor_kill_pct'],best['healthy_kill_pct']],flush=True)
    return status,rows

def audit(name,p):
    # Frozen candidates, no reoptimisation: reveal dependence on nuisance inputs.
    variants={'nominal':p,'cells_half':replace(p,total_cells=p.total_cells/2),'cells_double':replace(p,total_cells=p.total_cells*2),
              'volume_80pct':replace(p,cell_volume_l=.8*p.cell_volume_l),'volume_120pct':replace(p,cell_volume_l=1.2*p.cell_volume_l),
              'turnover_half':replace(p,turnover=p.turnover/2),'turnover_double':replace(p,turnover=p.turnover*2),
              'same_bond_off_rule':replace(p,surface_off_rule='same_bond'),
              'rho1_half':replace(p,rho1=p.rho1/2),'rho1_double':replace(p,rho1=p.rho1*2),
              'rho2_half':replace(p,rho2=p.rho2/2),'rho2_double':replace(p,rho2=p.rho2*2)}
    rows=[]
    for label,pp in variants.items():
        e,h=evaluate(pp,tight=True);rows.append(dict(scenario=name,perturbation=label,tumor_kill_pct=e*100,healthy_kill_pct=h*100,below_one_percent=h<.01))
    save_csv(OUT/(name+'_sensitivity.csv'),rows)
    return rows

def plot(summary,allrows,trajectories):
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for col,name in enumerate(('equal','unequal','low')):
        rows=allrows[name];ax=axes[0,col]
        ax.scatter([r['healthy_kill_pct'] for r in rows],[r['tumor_kill_pct'] for r in rows],s=7,alpha=.35)
        ax.axvline(1,color='red',ls='--');ax.set(xlabel='Healthy kill (%)',ylabel='Tumor kill (%)',xlim=(0,5),ylim=(0,100),title=name+' density')
        best=summary[name]['selected']
        if best:
            ax.scatter([best['healthy_kill_pct']],[best['tumor_kill_pct']],color='black',marker='*',s=120)
            r=trajectories[name];ax=axes[1,col]
            ax.plot(r['t']/3600,100*r['tumor'],label='Tumor')
            ax.plot(r['t']/3600,100*r['healthy'],label='Healthy')
            ax.set(xlabel='Time (h)',ylabel='Cell kill (%)',ylim=(0,100));ax.legend()
    fig.suptitle('Working prototype: inferred physiology, optimised binding rates\nCalibration is approximate; screening results are model predictions',fontsize=12)
    fig.savefig(OUT/'optimisation_dashboard.png',dpi=160);plt.close(fig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--nfe',type=int,default=576);ap.add_argument('--seeds',type=int,nargs='+',default=[8407,2026]);a=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    config=calibrate();baseline(config);summary={};allrows={};traces={}
    for name,p in scenario_params(config).items():
        runs=[];rows=[]
        for seed in a.seeds:
            status,rr=search(name,p,seed,a.nfe);runs.append(status);rows+=rr
        best=max((r['best'] for r in runs if r['best']),key=lambda r:r['tumor_kill_pct'],default=None)
        summary[name]=dict(runs=runs,selected=best)
        if best:
            selected=Parameters(**best['parameters']);traces[name]=trajectory(name+'_selected',selected)
            summary[name]['sensitivity']=audit(name,selected)
            assert max(abs(traces[name]['mass_error']))/selected.dose<1e-6
        allrows[name]=rows
        (OUT/'results.json').write_text(json.dumps(summary,indent=2))
    plot(summary,allrows,traces)
if __name__=='__main__':main()
