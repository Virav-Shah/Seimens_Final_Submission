"""One-shot ODE audit of proposed candidates; not an optimisation benchmark."""
import json,time
import numpy as np
import torch
from scipy.stats import qmc
from joblib import Parallel,delayed
from balanced_trial import OUT,OLD
from balanced_model import BalancedNet
from train import tensor,endpoints,aggregate,accuracy
from data import one

def main():
    results=json.loads((OUT/'results.json').read_text())
    if not results['methods']['PINN']['passes']:
        (OUT/'candidate_audit.json').write_text(json.dumps({'status':'skipped: original validation gate failed'}));return
    d=np.load(OLD/'initial_2026.npz');b=np.array([[2,7],[-5,2],[2,7],[-5,2]])
    pool=qmc.scale(qmc.Sobol(4,scramble=True,seed=7392).random_base2(10),b[:,0],b[:,1])
    oo=aggregate(endpoints(d['y'][:,:,-1]));ids=np.where((oo[:,1]<=.95)&(oo[:,2]<=10))[0]
    if not len(ids):ids=np.argsort(oo[:,1]/.95+oo[:,2]/10)[:8]
    rng=np.random.default_rng(7392);local=np.clip(d['x'][rng.choice(ids,256)]+rng.normal(0,.1,(256,4))*(b[:,1]-b[:,0]),b[:,0],b[:,1]);pool=np.unique(np.vstack([pool,local]),axis=0)
    proposals={};unique={};started=time.perf_counter()
    for kind in ['PINN','Supervised']:
        model=BalancedNet(d['x'],d['y']);model.load_state_dict(torch.load(OUT/f'{kind}_checkpoint.pt',weights_only=False)['state_dict']);pred=[]
        with torch.no_grad():
            for st in range(0,len(pool),128):
                xx=tensor(pool[st:st+128]).repeat_interleave(3,0);c=torch.arange(3).repeat(len(xx)//3)
                yy=model(torch.full((len(xx),),24.),xx,c).numpy().reshape(-1,3,97);pred.extend(endpoints(yy))
        pred=np.array(pred);o=aggregate(pred);feas=(o[:,1]<=.95)&(o[:,2]<=10);ids=np.where(feas)[0];chosen=[]
        # Eight highest-tumour predicted-feasible, eight closest to a limit.
        for i in ids[np.argsort(-o[ids,0])][:8]:chosen.append(int(i))
        boundary=np.minimum(abs(o[:,1]-.95)/.95,abs(o[:,2]-10)/10)
        for i in ids[np.argsort(boundary[ids])]:
            if len(chosen)>=16:break
            if i not in chosen:chosen.append(int(i))
        violation=np.maximum(o[:,1]/.95-1,0)+np.maximum(o[:,2]/10-1,0)
        for i in np.lexsort((-o[:,0],violation)):
            if len(chosen)>=16:break
            if i not in chosen:chosen.append(int(i))
        proposals[kind]=[{'x':pool[i].tolist(),'prediction':pred[i].tolist(),'predicted_feasible':bool(feas[i])} for i in chosen]
        for r in proposals[kind]:unique[tuple(r['x'])]=r['x']
    xs=list(unique.values());rows=Parallel(n_jobs=4)(delayed(one)(x) for x in xs)
    errs=[r for r in rows if r['error']]
    if errs:
        (OUT/'candidate_audit_errors.json').write_text(json.dumps(errs));raise RuntimeError('ODE candidate failure')
    cache={tuple(x):endpoints(z['y'][:,-1]) for x,z in zip(xs,rows)};out={'status':'completed','description':'16 one-shot proposals per model from the same 1280-point global/local pool; no retraining or multi-round optimisation','unique_ode_designs':len(xs),'methods':{}}
    for kind,rr in proposals.items():
        for z in rr:
            a=cache[tuple(z['x'])];z['actual']=a.tolist();o=aggregate(a[None,:])[0];z['actual_feasible']=bool(o[1]<=.95 and o[2]<=10)
        out['methods'][kind]={'metrics':accuracy(np.array([z['actual'] for z in rr]),np.array([z['prediction'] for z in rr])),'candidates':rr}
    out['wall_seconds']=time.perf_counter()-started
    (OUT/'candidate_audit.json').write_text(json.dumps(out,indent=2));print({k:v['metrics'] for k,v in out['methods'].items()},flush=True)
if __name__=='__main__':main()
