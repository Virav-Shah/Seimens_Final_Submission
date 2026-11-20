"""Second, bounded PINN experiment; matched supervised control and fresh test data."""
import os
os.environ.setdefault('MPLBACKEND','Agg');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import json,time,copy,hashlib,sys
from pathlib import Path
import numpy as np
import torch
from joblib import Parallel,delayed
from data import OUT as OLD,TIMES,coordinates,one
from train import accuracy,endpoints,aggregate,physical_metrics,tensor
from balanced_model import BalancedNet,KNOTS
OUT=OLD.parent/'pinn_balanced_trial';OUT.mkdir(exist_ok=True)
torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
GROUPS={'free_antibody':[0,1,2,3],'free_targets':[],'surface_complexes':[],'internalised_complexes':[],'active_payload':[4],'loss_accounts':[5,6],'released_payload':[],'killing_effect':list(range(87,92)),'payload_exposure':list(range(92,97))}
for j in range(5):
    z=7+16*j;GROUPS['free_targets']+=list(range(z,z+2));GROUPS['surface_complexes']+=list(range(z+2,z+8));GROUPS['internalised_complexes']+=list(range(z+8,z+14));GROUPS['active_payload'].append(z+14);GROUPS['released_payload'].append(z+15)
assert sorted(sum(GROUPS.values(),[]))==list(range(97))

def collocation(n,rng):
    b=np.array([[2,7],[-5,2],[2,7],[-5,2]])
    seg=rng.integers(0,len(KNOTS)-1,n);t=KNOTS[seg]+rng.uniform(.001,.999,n)*(KNOTS[seg+1]-KNOTS[seg])
    return tensor(t),tensor(rng.uniform(b[:,0],b[:,1],(n,4))),torch.as_tensor(rng.integers(0,3,n))
def norms(loss,params):
    gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    return float(torch.sqrt(sum(g.detach().square().sum() for g in gs if g is not None)))
def fit(kind,x,y,xv,yv,seconds=360):
    torch.manual_seed(2026);rng=np.random.default_rng(2026);m=BalancedNet(x,y);opt=torch.optim.Adam(m.parameters(),lr=1e-3)
    xt=tensor(x);yt=tensor(y);tt=tensor(TIMES);params=tuple(m.parameters());logweights=None
    start=time.perf_counter();last=start;hist=[];best=float('inf');selected=None;state=None;step=0
    while time.perf_counter()-start<seconds:
        elapsed=time.perf_counter()-start;alpha=float(np.clip((elapsed/seconds-.2)/.2,0,1)) if kind=='PINN' else 0.
        ix=torch.as_tensor(rng.integers(0,len(x),256));c=torch.as_tensor(rng.integers(0,3,256));ti=torch.as_tensor(rng.integers(0,len(TIMES),256))
        yp=m(tt[ti],xt[ix],c);trajectory=((yp-yt[ix,c,ti])/m.scale).square().mean()
        # Fixed 16 designs / all conditions, to include the small weighted average.
        ie=torch.as_tensor(rng.integers(0,len(x),16));ce=torch.arange(3).repeat(16);xe=xt[ie].repeat_interleave(3,0)
        pe=m(torch.full((48,),24.),xe,ce).reshape(16,3,97);ae=yt[ie,:,-1]
        pk=-100*torch.expm1(-pe[:,:,87:92]);ak=-100*torch.expm1(-ae[:,:,87:92]);scale=tensor([100,10,10,10,10])
        e=((pk-ak)/scale).square().mean()+(((pk[:,:,1:]-ak[:,:,1:])@tensor([.79,.1,.1,.01]))/.95).square().mean()
        supervised=trajectory+e;loss=supervised;ploss={};gradnorms={}
        if alpha:
            ct,cx,cc=collocation(64,rng);_,res=m.residual(ct,cx,cc)
            ploss={name:res[:,ids].square().mean() for name,ids in GROUPS.items()}
            if logweights is None or step%25==0:
                ref=max(norms(supervised,params),1e-30);gradnorms={'supervised':ref}
                new={}
                for name,l in ploss.items():
                    g=max(norms(l,params),1e-30);gradnorms[name]=g;new[name]=float(np.log(np.clip(ref/g,1e-30,1e6)))
                logweights=new if logweights is None else {n:.9*logweights[n]+.1*new[n] for n in new}
            loss=supervised+alpha*sum(np.exp(logweights[n])*l for n,l in ploss.items())/len(ploss)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);opt.step();step+=1
        if time.perf_counter()-last>20 or step==1:
            vp=m.prediction(xv,np.array([24.]));met=accuracy(endpoints(yv[:,:,-1]),endpoints(vp[:,:,0]))
            score=float(np.mean([met[n]['mae_pp']/s for n,s in zip(['tumour','overall_healthy','worst_subgroup'],[100,.95,10])]))
            rec={'step':step,'seconds':time.perf_counter()-start,'physics_fraction':alpha,'trajectory_loss':float(trajectory.detach()),'endpoint_loss':float(e.detach()),'physics_losses':{n:float(l.detach()) for n,l in ploss.items()},'physics_weights':{n:float(np.exp(v)) for n,v in (logweights or {}).items()},'gradient_norms':gradnorms,'validation_score':score,'validation':met}
            hist.append(rec)
            if (kind!='PINN' or alpha>=1) and score<best:best=score;state=copy.deepcopy(m.state_dict());selected=rec
            (OUT/f'{kind}_training.json').write_text(json.dumps(hist,indent=2));last=time.perf_counter()
            print(kind,step,round(rec['seconds']),round(score,5),{n:round(met[n]['mae_pp'],3) for n in ['tumour','overall_healthy','worst_subgroup']},flush=True)
    if state is None:raise RuntimeError('No eligible checkpoint')
    m.load_state_dict(state);record={'steps':step,'seconds':time.perf_counter()-start,'selected':selected}
    torch.save({'state_dict':state,'training':record},OUT/f'{kind}_checkpoint.pt')
    np.savez(OUT/f'{kind}_scaling.npz',mean=m.mean.numpy(),std=m.std.numpy(),scale=m.scale.numpy())
    met,pred=physical_metrics(m,xv,yv)
    np.savez_compressed(OUT/f'{kind}_validation.npz',x=xv,actual=yv,prediction=pred,t=TIMES)
    gate={'finite':met['finite'],'tumour':met['tumour']['mae_pp']<5,'healthy':met['overall_healthy']['mae_pp']<.25,'subgroup':met['worst_subgroup']['mae_pp']<3,'nonnegative':met['negative_scaled_max']<=.001,'balance':met['mass_fraction_max']<.01}
    record.update(validation=met,gate=gate,passes=all(gate.values()))
    (OUT/f'{kind}_results.json').write_text(json.dumps(record,indent=2))
    return m,record

def check(m,x):
    rng=np.random.default_rng(321);t,theta,c=collocation(96,rng)
    y,dy=torch.func.jvp(lambda z:m(z,theta,c),(t,),(torch.ones_like(t),))
    balance=m.physics.balance(y,t,c).abs().max().item()
    assert y.min()>=0 and balance<1e-12
    torch.testing.assert_close(m(torch.zeros_like(t),theta,c),m.physics.initial(c),atol=0,rtol=0)
    # Nondecreasing cumulative release, hazards and exposure by construction.
    assert dy[:,m.cumul].min()>=-1e-10
    # Left/right limits approach the same decoded shared knot state.
    maxjump=0.
    for k in KNOTS[1:-1]:
        eps=min(k,24-k)*1e-8
        a=m(torch.full((3,),k-eps),theta[:3],c[:3]);b=m(torch.full((3,),k+eps),theta[:3],c[:3])
        jump=float(((b-a)/m.scale).abs().max());maxjump=max(maxjump,jump)
        assert jump<1e-4
    # Directional derivative correctness away from knots.
    t=tensor([.007,.3,3.,20.]);th=theta[:4];cc=c[:4];eps=1e-8
    _,d=torch.func.jvp(lambda z:m(z,th,cc),(t,),(torch.ones_like(t),))
    fd=(m(t+eps,th,cc)-m(t-eps,th,cc))/(2*eps)
    torch.testing.assert_close(d,fd,rtol=2e-4,atol=1e-5)
    return {'balance_max_fraction':balance,'nonnegative':True,'initial_conditions_exact':True,'cumulative_quantities_monotone':True,'maximum_scaled_knot_jump':maxjump,'time_derivatives_checked':True}

def main():
    start=time.perf_counter();d=np.load(OLD/'initial_2026.npz');v=np.load(OLD/'validation.npz');x,y,xv,yv=d['x'],d['y'],v['x'],v['y']
    torch.manual_seed(2026);pre=check(BalancedNet(x,y),x);print('Representation checks',pre,flush=True)
    cfg={'knots_h':KNOTS.tolist(),'seed':2026,'training_seconds_per_method':360,'training_designs':64,'validation_designs':32,'fresh_test_designs':64,'fresh_test_seed':66021,'physics_groups':GROUPS,'bounds_log10':[[2,7],[-5,2],[2,7],[-5,2]],'reference_sha256':hashlib.sha256((Path(__file__).parents[1]/'mechanism_comparison/model.py').read_bytes()).hexdigest(),'nonnegativity':'hard representation','conservation':'two shared inventories including cleared material','loss_weights':'gradient norms updated every 25 steps; log EMA 0.9; clipped to [1e-30,1e6]','reference_objectives_unchanged':True,'architecture':'shared knot MLP plus local interval correction MLP, each 3x64 tanh; 97 raw coordinates','training_loss':'scaled trajectory + endpoint error; optional balanced grouped ODE residual','preflight':pre,'comparison':'same seed/data/architecture/wall-time budget; different number of updates expected','data_source':'unchanged mechanistic ODE; no clinical labels'}
    (OUT/'configuration.json').write_text(json.dumps(cfg,indent=2))
    results={};models={}
    for kind in ['PINN','Supervised']:
        models[kind],results[kind]=fit(kind,x,y,xv,yv)
        results[kind]['representation_checks']=check(models[kind],x)
    # Fresh test only generated after both checkpoints have been frozen.
    tx=coordinates(64,66021)
    for a in [x,xv,np.load(OLD/'test.npz')['x']]:assert not set(map(tuple,tx))&set(map(tuple,a))
    st=time.perf_counter();rows=Parallel(n_jobs=4)(delayed(one)(a) for a in tx)
    errors=[r for r in rows if r['error']]
    (OUT/'fresh_test_generation.json').write_text(json.dumps({'wall_seconds':time.perf_counter()-st,'errors':errors},indent=2))
    if errors:raise RuntimeError('Failed test trajectory; errors retained')
    ty=np.stack([r['y'] for r in rows]);np.savez_compressed(OUT/'fresh_test_data.npz',x=tx,y=ty,t=TIMES)
    for kind in models:
        met,pred=physical_metrics(models[kind],tx,ty);results[kind]['fresh_test']=met
        np.savez_compressed(OUT/f'{kind}_fresh_test_predictions.npz',x=tx,actual=ty,prediction=pred,t=TIMES)
        # Old held-out set is a paired retrospective comparison, not fresh validation.
        od=np.load(OLD/'test.npz');om,op=physical_metrics(models[kind],od['x'],od['y']);results[kind]['previous_test_paired']=om
        (OUT/f'{kind}_results.json').write_text(json.dumps(results[kind],indent=2))
    final={'methods':results,'wall_seconds':time.perf_counter()-start,'status':'eligible_for_search' if results['PINN']['passes'] else 'stopped_before_search','original_model_unchanged':cfg['reference_sha256']==hashlib.sha256((Path(__file__).parents[1]/'mechanism_comparison/model.py').read_bytes()).hexdigest()}
    (OUT/'results.json').write_text(json.dumps(final,indent=2));print('COMPLETE',final['status'],flush=True)
if __name__=='__main__':main()
