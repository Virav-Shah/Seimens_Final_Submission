"""Run reconstructed paper workflow. Every output is explicitly non-exact."""
import argparse,csv,json,random,itertools,os
from dataclasses import replace
from pathlib import Path
import numpy as np
os.environ.setdefault("MPLCONFIGDIR",str(Path(__file__).resolve().parent.parent/"tmp"/"matplotlib"))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import Parameters,SCENARIOS,simulate,endpoints,HEALTHY_WEIGHTS

OUT=Path(__file__).resolve().parent.parent/'output'/'simulation'

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def trajectories(scenarios=None, summary_name="scenario_summary.csv"):
    summary=[]
    for name,p in (SCENARIOS if scenarios is None else scenarios).items():
        r=simulate(p);h=r['t']/3600
        fig,ax=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
        for j,label in enumerate(('Tumor','Healthy T1','Healthy T2','Healthy T12')):
            target=1 if j==2 or name=='figure8' else 0
            ax[0].plot(h,100*r['occupancy'][:,j,target],label=label)
        for j,label in enumerate(('Tumor','Healthy mean')):
            a=r['intracellular'][:,0] if j==0 else r['intracellular'][:,1:]@HEALTHY_WEIGHTS
            ax[1].plot(h,a*1e6,label=label)
        ax[2].plot(h,100*r['tumor'],label='Tumor');ax[2].plot(h,100*r['healthy'],label='Healthy')
        for a,ylabel in zip(ax,('Target occupancy (%)','Accumulated ADC (uM)','Cell kill (%)')):
            a.set(xlabel='Time (h)',ylabel=ylabel);a.legend(fontsize=7)
        fig.suptitle(name+' | reconstruction; unpublished inputs assumed',fontsize=10)
        fig.savefig(OUT/(name+'.png'),dpi=150);plt.close(fig)
        write_csv(OUT/(name+'.csv'),[dict(time_h=float(h[i]),tumor_kill_pct=float(r['tumor'][i]*100),healthy_kill_pct=float(r['healthy'][i]*100),tumor_adc_uM=float(r['intracellular'][i,0]*1e6),mass_error_mol=float(r['mass_error'][i])) for i in range(len(h))])
        summary.append(dict(scenario=name,tumor_kill_pct=100*r['tumor'][-1],healthy_kill_pct=100*r['healthy'][-1],max_relative_mass_error=float(max(abs(r['mass_error']))/p.dose)))
    write_csv(OUT/summary_name,summary)

def sweep(name,xs,ys,fn):
    rows=[];Z=np.zeros((2,len(ys),len(xs)))
    for j,y in enumerate(ys):
        for i,x in enumerate(xs):
            a,b=endpoints(fn(x,y));Z[:,j,i]=[a*100,b*100]
            rows.append(dict(x=float(x),y=float(y),tumor_kill_pct=a*100,healthy_kill_pct=b*100,efficacy_toxicity_ratio=a/b if b>0 else None))
    write_csv(OUT/(name+'.csv'),rows)
    fig,ax=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
    ratio=np.divide(Z[0],Z[1],out=np.full_like(Z[0],np.nan),where=Z[1]>0)
    for a,z,title in zip(ax,[Z[0],Z[1],ratio],['Tumor kill (%)','Healthy kill (%)','Efficacy / toxicity']):
        m=a.pcolormesh(xs,ys,z,shading='nearest');fig.colorbar(m,ax=a);a.set(xscale='log',yscale='log',xlabel='x (see manifest)',ylabel='y (see manifest)',title=title)
    fig.suptitle(name+' | reconstruction; unpublished inputs assumed',fontsize=10)
    fig.savefig(OUT/(name+'.png'),dpi=150);plt.close(fig)
    print('completed',name,flush=True)

def sweeps(n):
    base=SCENARIOS['figure4_bispecific'];opt=SCENARIOS['figure6']
    on=np.logspace(4,7,n);off=np.logspace(-3,0,n)
    sweep('figure5ab_S1a',on,on,lambda x,y:replace(base,kon1=x,kon2=y))
    sweep('figure5cd_S1b',off,off,lambda x,y:replace(base,koff1=x,koff2=y))
    sweep('figure5ef_S1c',on,off,lambda x,y:replace(base,kon1=x,kon2=x,koff1=y,koff2=y))
    sweep('S2_monospecific',on,off,lambda x,y:replace(base,mode='monospecific',kon1=x,koff1=y))
    sweep('figure7',np.logspace(-12,-8,n),np.logspace(-12,-8,n),lambda x,y:replace(opt,rho1=x,rho2=y))
    sweep('S3_diffusion',np.logspace(11,15,n),np.logspace(11,15,n),lambda x,y:replace(opt,k2d1=x,k2d2=y))
    sweep('S4_density',np.logspace(-14,-9,n),np.logspace(-14,-9,n),lambda x,y:replace(opt,rho1=x,rho2=y))

def optimize(nfe,seed,low=False):
    from platypus import Problem,Real,NSGAII,nondominated,Solution,InjectedPopulation
    random.seed(seed);np.random.seed(seed)
    base=replace(SCENARIOS['figure8']) if not low else replace(SCENARIOS['figure6'],rho1=1e-13,rho2=1e-13,ec50=1e-8)
    problem=Problem(4,2,1)
    # Bounds inferred from Figure 5 axes, NOT disclosed optimizer bounds.
    problem.types[:]=[Real(4,7),Real(-3,0),Real(4,7),Real(-3,0)]
    problem.directions[:]=[Problem.MINIMIZE,Problem.MINIMIZE]
    problem.constraints[:]='<=0'
    evaluations=[]
    def evaluate(z):
        rates=10**np.asarray(z)
        a,b=endpoints(replace(base,kon1=rates[0],koff1=rates[1],kon2=rates[2],koff2=rates[3]))
        evaluations.append(dict(kon1=rates[0],koff1=rates[1],kon2=rates[2],koff2=rates[3],tumor_kill_pct=100*a,healthy_kill_pct=100*b,feasible=b<.01))
        return [-a,b],[b-(.01-1e-10)]
    problem.function=evaluate
    seeds=[]
    for values in itertools.product((4.,7.),(-3.,0.),(4.,7.),(-3.,0.)):
        solution=Solution(problem);solution.variables[:]=values;seeds.append(solution)
    for values in ([4.,0.,np.log10(5e4),0.],[np.log10(5e4),0.,np.log10(5e4),0.]):
        solution=Solution(problem);solution.variables[:]=values;seeds.append(solution)
    algorithm=NSGAII(problem,population_size=min(40,nfe),generator=InjectedPopulation(seeds));algorithm.run(nfe)
    prefix='S5_low_density' if low else 'figure8_optimization'
    write_csv(OUT/(prefix+'_evaluations.csv'),evaluations)
    feasible=[s for s in nondominated(algorithm.result) if s.feasible]
    rows=[dict(kon1=10**s.variables[0],koff1=10**s.variables[1],kon2=10**s.variables[2],koff2=10**s.variables[3],tumor_kill_pct=-100*s.objectives[0],healthy_kill_pct=100*s.objectives[1]) for s in feasible]
    if rows:write_csv(OUT/(prefix+'_pareto.csv'),rows)
    status=dict(requested_nfe=nfe,algorithm_nfe=algorithm.nfe,initialization='16 boundary corners + two specified seeds + random fill',evaluations=len(evaluations),feasible_pareto_count=len(rows),seed=seed,exact_replication=False)
    if rows:
        best=max(rows,key=lambda x:x['tumor_kill_pct']);status['best_feasible']=best
        selected=replace(base,**{k:best[k] for k in ('kon1','koff1','kon2','koff2')})
        trajectories({prefix+'_selected':selected},prefix+'_selected_summary.csv')
        from dataclasses import asdict
        status['selected_parameters']=asdict(selected)
    (OUT/(prefix+'_status.json')).write_text(json.dumps(status,indent=2))
    fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
    ax.scatter([v['healthy_kill_pct'] for v in evaluations],[v['tumor_kill_pct'] for v in evaluations],s=15)
    ax.axvline(1,color='red',ls='--');ax.set(xlabel='Healthy kill (%)',ylabel='Tumor kill (%)',title=prefix+' (reconstruction)')
    fig.savefig(OUT/(prefix+'.png'),dpi=150);plt.close(fig)
    print(prefix,status,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--grid',type=int,default=5);p.add_argument('--nfe',type=int,default=240);p.add_argument('--seed',type=int,default=8407);p.add_argument('--skip-opt',action='store_true');a=p.parse_args()
    if a.grid<2 or a.nfe<4:p.error('grid >=2, nfe >=4 required')
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'manifest.json').write_text(json.dumps(dict(status='RECONSTRUCTION_NOT_EXACT_REPLICATION',scenarios={k:simulate(v,points=3)['parameters'] for k,v in SCENARIOS.items()},grid=a.grid,nfe=a.nfe,seed=a.seed,sweep_axes={'figure5ab_S1a':['kon1','kon2'],'figure5cd_S1b':['koff1','koff2'],'figure5ef_S1c':['kon1=kon2','koff1=koff2'],'S2_monospecific':['kon','koff'],'figure7':['rho1','rho2'],'S3_diffusion':['k2d1','k2d2'],'S4_density':['rho1','rho2']}),indent=2))
    trajectories();sweeps(a.grid)
    if not a.skip_opt:optimize(a.nfe,a.seed);optimize(a.nfe,a.seed,low=True)
if __name__=='__main__':main()
