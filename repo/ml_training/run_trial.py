"""Small, reproducible surrogate-assisted search benchmark. No experimental claims."""
import os
os.environ.setdefault('MPLBACKEND','Agg')
import sys,json,time,random,warnings,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import qmc
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern,ConstantKernel,WhiteKernel
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error,r2_score
from joblib import Parallel,delayed
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mechanism_comparison.model import assess,BOUNDS
from platypus import Problem,Real,Solution,InjectedPopulation,NSGAII,Evaluator
OUT=Path(__file__).resolve().parents[2]/'output/ml_small_trial'; OUT.mkdir(exist_ok=True)
B=np.array(BOUNDS); SEEDS=[2026,8407,11939]; N0=64; BATCH=24; ROUNDS=4; NFINAL=32
F=np.array([.79,.10,.10,.01]); CAP=100*(-np.expm1(-.08*24))

def convert(a):
 a=np.asarray(a).reshape(-1,3,5)
 h=a[:,:,1:]@F
 return np.column_stack([a[:,:,0].min(1),h.max(1),a[:,:,1:].max((1,2))])
def labels(r): return np.asarray(r['outcomes'])[:,[0,2,3,4,5]].ravel()
def ev(x,tight=False):
 t=time.perf_counter()
 try:
  r=assess(x,tight=tight);y=labels(r)
  return dict(x=list(x),y=y.tolist(),seconds=time.perf_counter()-t,error=None)
 except Exception as e: return dict(x=list(x),error=repr(e),seconds=time.perf_counter()-t)
def evaluate(xs,tag):
 t=time.perf_counter(); rows=Parallel(n_jobs=4,backend='loky')(delayed(ev)(x) for x in xs)
 (OUT/f'{tag}.json').write_text(json.dumps(rows)); print(tag,'n',len(rows),'seconds',round(time.perf_counter()-t,2),'failures',sum(r['error'] is not None for r in rows),flush=True)
 if any(r['error'] for r in rows):raise RuntimeError('ODE failure retained in file; investigate before fitting')
 return rows,time.perf_counter()-t

def lhs(n,seed):return qmc.scale(qmc.LatinHypercube(4,seed=seed).random(n),B[:,0],B[:,1])
def arrays(rows):return np.array([r['x'] for r in rows]),np.array([r['y'] for r in rows])
def feasible(o):return (o[:,1]<=.95)&(o[:,2]<=10)
def front(o):
 f=np.column_stack([-o[:,0],o[:,1],o[:,2]])
 return np.array([i for i in range(len(f)) if not np.any(np.all(f<=f[i]+1e-10,axis=1)&np.any(f<f[i]-1e-10,axis=1))],int)
def metrics(y,p):
 a,b=convert(y),convert(p);d={}
 for j,n in enumerate(['tumour','overall_healthy','worst_subgroup']):
  d[n]={'mae_pp':float(np.abs(a[:,j]-b[:,j]).mean()),'rmse_pp':float(np.sqrt(np.mean((a[:,j]-b[:,j])**2))),'r2':float(r2_score(a[:,j],b[:,j]))}
 d['subgroup_mae_pp']=np.abs(y.reshape(-1,3,5)[:,:,1:]-p.reshape(-1,3,5)[:,:,1:]).mean((0,1)).tolist()
 pred=feasible(b);actual=feasible(a)
 d.update(n=len(y),actual_feasible=int(actual.sum()),predicted_feasible=int(pred.sum()),false_feasible=int((pred&~actual).sum()),false_infeasible=int((~pred&actual).sum()))
 return d
class Regressor:
 def __init__(self,kind,seed):self.kind=kind;self.seed=seed
 def fit(self,x,y):
  self.xs=StandardScaler().fit(x);self.ys=StandardScaler().fit(y)
  if self.kind=='GP':
   kernel=ConstantKernel(1.,(.01,100))*Matern(length_scale=np.ones(4),length_scale_bounds=(.05,20),nu=2.5)+WhiteKernel(.001,(1e-6,.1))
   self.m=GaussianProcessRegressor(kernel=kernel,alpha=1e-8,normalize_y=False,n_restarts_optimizer=0,random_state=self.seed)
  else:self.m=ExtraTreesRegressor(n_estimators=100,min_samples_leaf=2,max_features=1.,random_state=self.seed,n_jobs=1)
  with warnings.catch_warnings(record=True) as ww:
   warnings.simplefilter('always'); self.m.fit(self.xs.transform(x),self.ys.transform(y)); self.warnings=[str(w.message) for w in ww]
  return self
 def predict(self,x,std=False):
  z=self.xs.transform(x)
  if self.kind=='GP':
   if std:mu,sd=self.m.predict(z,return_std=True)
   else:mu=self.m.predict(z)
  else:
   mu=self.m.predict(z)
   if std:sd=np.std([a.predict(z) for a in self.m.estimators_],axis=0)
  mu=np.clip(self.ys.inverse_transform(mu),0,CAP)
  return (mu,sd*self.ys.scale_) if std else mu

def choose(pool,mu,sd,history,n,explore=True):
 o=convert(mu);allowed=feasible(o); ids=np.where(allowed)[0]
 candidates=list(ids[front(o[ids])]) if len(ids) else []
 candidates.sort(key=lambda i:-o[i,0])
 picked=[];ntrade=n//2 if explore else n
 # Greedy objective-space diversity, including maximum tumour endpoint.
 if candidates:
  picked=[candidates.pop(0)]
  while candidates and len(picked)<ntrade:
   scale=np.array([100,.95,10]); ds=np.min(np.linalg.norm((o[candidates,None,:]-o[picked][None,:,:])/scale,axis=2),axis=1)
   idx=int(np.argmax(ds));picked.append(candidates.pop(idx))
 # Remaining candidates ranked by feasibility violation and tumour response.
 violation=np.maximum(o[:,1]/.95-1,0)+np.maximum(o[:,2]/10-1,0)
 ranks=np.lexsort((-o[:,0],violation))
 for i in ranks:
  if len(picked)>=ntrade:break
  if i not in picked:picked.append(int(i))
 if explore:
  # Disagreement near constraints is a heuristic, not calibrated confidence.
  uncertainty=sd.mean(1)/(1+np.minimum(abs(o[:,1]-.95)/.95,abs(o[:,2]-10)/10))
  for i in np.argsort(-uncertainty):
   if len(picked)>=3*n//4:break
   if i not in picked:picked.append(int(i))
  hx=(history-B[:,0])/(B[:,1]-B[:,0]);px=(pool-B[:,0])/(B[:,1]-B[:,0])
  dist=np.min(np.linalg.norm(px[:,None,:]-hx[None,:,:],axis=2),axis=1)
  for i in np.argsort(-dist):
   if len(picked)>=n:break
   if i not in picked:picked.append(int(i))
 return np.array(picked,int)

def poolgen(seed,rows):
 rng=np.random.default_rng(seed); x,y=arrays(rows);o=convert(y);ids=np.where(feasible(o))[0]
 good=x[ids[front(o[ids])]] if len(ids) else x[np.argsort(o[:,2])[:8]]
 globalx=qmc.scale(qmc.Sobol(4,scramble=True,seed=seed).random_base2(11),B[:,0],B[:,1])
 local=np.clip(good[rng.integers(len(good),size=1024)]+rng.normal(0,.15,size=(1024,4))*(B[:,1]-B[:,0]),B[:,0],B[:,1])
 p=np.unique(np.round(np.vstack([globalx,local]),10),axis=0);old={tuple(np.round(z,10)) for z in x};return np.array([z for z in p if tuple(z) not in old])
def verified(rows,tag):
 x,y=arrays(rows);o=convert(y);ids=np.where(feasible(o))[0]
 if not len(ids):return [],0.
 ids=ids[front(o[ids])];ids=ids[np.argsort(-o[ids,0])];ids=ids[:12]
 t=time.perf_counter();v=Parallel(n_jobs=4)(delayed(ev)(x[i],True) for i in ids)
 (OUT/f'{tag}_verified.json').write_text(json.dumps(v));return v,time.perf_counter()-t

def stats(rows):
 if not rows:return {'feasible':0,'best_tumour':None,'hypervolume':0.}
 _,y=arrays(rows);o=convert(y);a=o[feasible(o)]; idx=front(a) if len(a) else []
 # Exact dominated volume for utility (T/100,1-H/.95,1-G/10), origin reference.
 u=np.column_stack([a[:,0]/100,1-a[:,1]/.95,1-a[:,2]/10]) if len(a) else np.zeros((0,3))
 volume=0.;lastx=0.
 for xx in np.unique(u[:,0]):
  yz=u[u[:,0]>=xx,1:];lasty=0.;area=0.
  for yy in np.unique(yz[:,0]):area+=(yy-lasty)*yz[yz[:,0]>=yy,1].max();lasty=yy
  volume+=(xx-lastx)*area;lastx=xx
 return dict(feasible=len(a),nondominated=len(idx),best_tumour=float(a[:,0].max()) if len(a) else None,hypervolume=float(volume))

def main():
 start=time.perf_counter();cfg={'seeds':SEEDS,'initial':N0,'rounds':ROUNDS,'batch':BATCH,'final':NFINAL,'budget_per_arm':N0+ROUNDS*BATCH+NFINAL,'bounds_log10':BOUNDS,'validation':32,'test':64,'contexts':'three fixed existing illustrative conditions','model_sha256':hashlib.sha256((Path(__file__).parents[1]/'mechanism_comparison/model.py').read_bytes()).hexdigest()};(OUT/'configuration.json').write_text(json.dumps(cfg,indent=2))
 val,tval=evaluate(lhs(32,5511),'validation');test,ttest=evaluate(lhs(64,9911),'test');xv,yv=arrays(val);xt,yt=arrays(test);runs=[]
 for seed in SEEDS:
  initial,tinit=evaluate(lhs(N0,seed),f'initial_{seed}');x,y=arrays(initial);selection=[];t0=time.perf_counter()
  for name in ['GP','ExtraTrees']:
   m=Regressor(name,seed).fit(x,y);pred=m.predict(xv);met=metrics(yv,pred); score=np.mean([met[k]['mae_pp']/s for k,s in zip(['tumour','overall_healthy','worst_subgroup'],[100,.95,10])]);selection.append({'model':name,'score':float(score),'metrics':met,'warnings':m.warnings})
  kind=min(selection,key=lambda z:z['score'])['model'];model_time=time.perf_counter()-t0;rows=list(initial);acq=[];pred_rows=[];ode_wall=0.; fit_seconds=0.
  for rnd in range(ROUNDS+1):
   t=time.perf_counter();x,y=arrays(rows);m=Regressor(kind,seed).fit(x,y);pool=poolgen(seed+100*(rnd+1),rows);mu,sd=m.predict(pool,True);ids=choose(pool,mu,sd,x,BATCH if rnd<ROUNDS else NFINAL,explore=rnd<ROUNDS);fit_seconds+=time.perf_counter()-t
   fresh,elapsed=evaluate(pool[ids],f'active_{seed}_{rnd}');ode_wall+=elapsed
   for r,p in zip(fresh,mu[ids]):pred_rows.append({'x':r['x'],'prediction':p.tolist(),'actual':r['y'],'round':rnd})
   rows+=fresh;acq.append({'round':rnd,'n':len(rows),**stats(rows)})
  t=time.perf_counter();x,y=arrays(rows);m=Regressor(kind,seed).fit(x,y);pt=m.predict(xt);fit_seconds+=time.perf_counter()-t
  vt,tv=verified(rows,f'assisted_{seed}');(OUT/f'assisted_{seed}.json').write_text(json.dumps(rows));(OUT/f'predictions_{seed}.json').write_text(json.dumps({'test_actual':yt.tolist(),'test_prediction':pt.tolist(),'selected_candidates':pred_rows}))
  # Equal initial design and same number of unique ODE evaluations for direct NSGA-II.
  random.seed(seed);history=list(initial);cache={tuple(r['x']):r for r in initial};budget=cfg['budget_per_arm'];p=Problem(4,3,2);p.types[:]=[Real(*b) for b in BOUNDS];p.constraints[:]='<=0'; exhausted=False
  def fn(z):
   key=tuple(z)
   if key not in cache:
    if len(history)>=budget:raise StopIteration('budget')
    r=ev(z)
    if r['error']:raise RuntimeError(r['error'])
    cache[key]=r;history.append(r)
   o=convert([cache[key]['y']])[0];return [-o[0],o[1],o[2]],[o[1]-.95,o[2]-10]
  p.function=fn;starts=[]
  for row in initial:
   sol=Solution(p);sol.variables[:]=row['x'];starts.append(sol)
  class BatchEvaluator(Evaluator):
   def evaluate_all(self,jobs,**kwargs):
    jobs=list(jobs);unique={tuple(j.solution.variables):list(j.solution.variables) for j in jobs if tuple(j.solution.variables) not in cache}
    selected=list(unique.values())[:max(0,budget-len(history))]
    fresh=Parallel(n_jobs=4)(delayed(ev)(z) for z in selected)
    for r in fresh:
     if r['error']:raise RuntimeError(r['error'])
     cache[tuple(r['x'])]=r;history.append(r)
    for job in jobs:
     if tuple(job.solution.variables) not in cache:raise StopIteration('budget')
     job.run()
    return jobs
  alg=NSGAII(p,population_size=N0,generator=InjectedPopulation(starts),evaluator=BatchEvaluator());td=time.perf_counter()
  try:
   while len(history)<budget:alg.step()
  except StopIteration:pass
  direct_seconds=time.perf_counter()-td;vd,tvd=verified(history,f'direct_{seed}');(OUT/f'direct_{seed}.json').write_text(json.dumps(history))
  # Compare serial-equivalent time because assisted batches use 4 workers, direct is sequential.
  assisted_compute=sum(r['seconds'] for r in rows)+model_time+fit_seconds+sum(r['seconds'] for r in vt)
  direct_compute=sum(r['seconds'] for r in history)+sum(r['seconds'] for r in vd)
  record={'seed':seed,'selected_model':kind,'model_selection':selection,'initial':stats(initial),'assisted':stats(rows),'direct':stats(history),'assisted_tight_subset':stats([r for r in vt if not r['error']]),'direct_tight_subset':stats([r for r in vd if not r['error']]),'test_metrics':metrics(yt,pt),'candidate_metrics':metrics(np.array([r['actual'] for r in pred_rows]),np.array([r['prediction'] for r in pred_rows])),'progress':acq,'timing':{'initial_wall':tinit,'assisted_new_ode_wall':ode_wall,'model_selection_seconds':model_time,'training_search_seconds':fit_seconds,'direct_new_ode_wall':direct_seconds,'assisted_total_wall':tinit+ode_wall+model_time+fit_seconds+tv,'direct_total_wall':tinit+direct_seconds+tvd,'assisted_serial_equivalent_seconds':assisted_compute,'direct_serial_equivalent_seconds':direct_compute,'assisted_tight_checks':len(vt),'direct_tight_checks':len(vd)}}
  runs.append(record);(OUT/'results.json').write_text(json.dumps({'configuration':cfg,'runs':runs,'validation_wall_seconds':tval,'shared_validation_seconds':sum(r['seconds'] for r in val),'shared_test_seconds':sum(r['seconds'] for r in test),'wall_seconds':time.perf_counter()-start},indent=2)); print('SEED COMPLETE',seed,kind,'assisted',record['assisted'],'direct',record['direct'],flush=True)
 print('COMPLETE',time.perf_counter()-start,flush=True)
if __name__=='__main__':main()
