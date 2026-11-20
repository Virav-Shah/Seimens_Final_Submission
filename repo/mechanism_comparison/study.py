"""One-process-at-a-time comparisons using the unchanged V5 equations."""
from .model import *
from scipy.stats import qmc
from joblib import Parallel, delayed, parallel_config
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results_mechanism_comparison'
SEEDS=(8407,2026)
BUDGET=256
CASES={
 'reference':('Reference',BIOLOGY),
 'dual_slow':('Dually bound: 0.2 h⁻¹',replace(BIOLOGY,ki=(.2,4.,.2))),
 'dual_fast':('Dually bound: 4 h⁻¹',replace(BIOLOGY,ki=(.2,4.,4.))),
 'linker_stable':('No premature release',replace(BIOLOGY,half=np.inf)),
 'linker_short':('Linker half-life: 48 h',replace(BIOLOGY,half=48.)),
 'movement_none':('No free-payload movement',replace(BIOLOGY,permeability=0.)),
 'movement_fast':('Movement: 0.2 h⁻¹',replace(BIOLOGY,permeability=.2)),
 'processing_slow':('Processing: 0.35 h⁻¹',replace(BIOLOGY,processing_rates=(.35,)*3)),
 'processing_fast':('Processing: 1.4 h⁻¹',replace(BIOLOGY,processing_rates=(1.4,)*3)),
 'recycling_none':('No recycling',replace(BIOLOGY,recycling_rates=(0.,)*3)),
 'recycling_fast':('Recycling: 1.2 h⁻¹',replace(BIOLOGY,recycling_rates=(1.2,)*3)),
 'metabolism_none':('No payload metabolism',replace(BIOLOGY,metabolism=0.)),
 'metabolism_fast':('Payload metabolism: 0.02 h⁻¹',replace(BIOLOGY,metabolism=.02)),
}
POLICIES={'overall':(2,False),'subgroup_objective':(3,False),'subgroup_limit':(3,True)}
def evaluate(x,s,tight=False):
 r=integrate(x,s,PD,tight=tight); k=100*r['kill'][-1]
 return dict(theta=list(map(float,x)),tumour=float(k[0]),healthy=float(100*r['healthy'][-1]),subgroup=float(max(k[1:])),groups=k[1:].tolist(),mass=float(r['mass']))
def margin(r,policy):
 return min(.95-r['healthy'],10-r['subgroup']) if POLICIES[policy][1] else .95-r['healthy']
def front(rows,nobj=3):
 if not rows:return []
 a=np.array([[-r['tumour'],r['healthy'],r['subgroup']][:nobj] for r in rows])
 return [r for i,r in enumerate(rows) if not (np.all(a<=a[i]+1e-9,axis=1)&np.any(a<a[i]-1e-9,axis=1)).any()]
def search(case,policy,seed,anchor):
 path=OUT/f'{case}_{policy}_{seed}.json'
 if path.exists():
  saved=json.loads(path.read_text())
  if saved.get('complete'):return saved
 s=CASES[case][1]; random.seed(seed); history=[];cache={};started=time.time()
 def ev(x):
  key=tuple(map(float,x))
  if key not in cache:
   try:r=evaluate(x,s)
   except (RuntimeError,ValueError,FloatingPointError) as e:r=dict(theta=list(key),tumour=0.,healthy=100.,subgroup=100.,groups=[100.]*4,error=str(e))
   cache[key]=r;history.append(r)
  return cache[key]
 nobj=POLICIES[policy][0]
 p=Problem(4,nobj,1);p.types[:]=[Real(*b) for b in BOUNDS];p.constraints[:]='<=0'
 def fn(x):
  r=ev(x);return [-r['tumour'],r['healthy'],r['subgroup']][:nobj],[-margin(r,policy)]
 p.function=fn
 bounds=np.array(BOUNDS);grid=qmc.scale(qmc.LatinHypercube(4,seed=seed).random(48),bounds[:,0],bounds[:,1]);grid[0]=anchor
 starts=[]
 for x in grid:
  sol=Solution(p);sol.variables[:]=x;starts.append(sol)
 alg=NSGAII(p,population_size=48,generator=InjectedPopulation(starts));alg.run(BUDGET)
 feasible=[r for r in history if margin(r,policy)>=0]
 best=max(feasible,key=lambda r:r['tumour']) if feasible else min(history,key=lambda r:-margin(r,policy))
 if margin(ev(anchor),policy)>=0 and ev(anchor)['tumour']>best['tumour']:best=ev(anchor)
 def cons(x):
  r=ev(x)
  return np.r_[.95-r['healthy'],10-np.array(r['groups'])]-2e-4 if POLICIES[policy][1] else np.array([.95-r['healthy']-2e-4])
 fit=minimize(lambda x:-ev(x)['tumour'],best['theta'],method='SLSQP',bounds=BOUNDS,constraints=[dict(type='ineq',fun=cons)],options=dict(maxiter=50,ftol=1e-7,eps=1e-3));ev(fit.x)
 feasible=[r for r in history if margin(r,policy)>=0]
 selected=front(feasible,nobj)+sorted(feasible,key=lambda r:r['tumour'],reverse=True)[:8]
 unique={tuple(np.round(r['theta'],10)):r for r in selected};verified=[]
 for r in unique.values():
  try:v=evaluate(r['theta'],s,True)
  except RuntimeError:continue
  if margin(v,policy)>=0:verified.append(v)
 result=dict(complete=True,case=case,policy=policy,seed=seed,history=history,front=front(verified,nobj),best=max(verified,key=lambda r:r['tumour']) if verified else None,local_success=bool(fit.success),message=str(fit.message),seconds=time.time()-started)
 path.write_text(json.dumps(result));print('DONE',case,policy,seed,'best',result['best']['tumour'] if result['best'] else None,flush=True)
 return result

def run_all(workers=4):
 OUT.mkdir(exist_ok=True)
 # Historical original-domain endpoint from results_boundary_review/summary.json.
 # This is a starting point, not a fixed biological parameter or claimed optimum.
 old=[4.352579495321639,1.1486484597387134,2.0000000000001843,0.09538099516189319]
 with parallel_config(backend='loky',inner_max_num_threads=1):base=Parallel(n_jobs=2)(delayed(search)('reference','subgroup_limit',seed,old) for seed in SEEDS)
 anchor=max((r['best'] for r in base if r['best']),key=lambda r:r['tumour'])['theta']
 tasks=[(case,'subgroup_limit',seed,anchor) for case in CASES if case!='reference' for seed in SEEDS]
 tasks += [('reference',policy,seed,anchor) for policy in ('overall','subgroup_objective') for seed in SEEDS]
 with parallel_config(backend='loky',inner_max_num_threads=1):runs=base+Parallel(n_jobs=workers)(delayed(search)(*task) for task in tasks)
 rows=[]
 for case,(label,s) in CASES.items():
  best=max((r['best'] for r in runs if r['case']==case and r['policy']=='subgroup_limit' and r['best']),key=lambda r:r['tumour'])
  fixed=evaluate(anchor,s,True);rows.append(dict(case=case,label=label,fixed=fixed,reoptimised=best))
 (OUT/'mechanism_results.json').write_text(json.dumps(rows,indent=2));print('ALL SEARCHES COMPLETE',flush=True)
 return rows,runs
if __name__=='__main__':run_all()
